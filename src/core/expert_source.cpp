// expert_source.cpp - expert blob sources and the hit/miss dispatcher (see core_expert_source.hpp).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "baloon/core_expert_source.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace baloon {
namespace core {

static bool geometry_ok(int64_t nl, int64_t ne, int64_t blob) { return nl > 0 && ne > 0 && blob > 0; }

// ---------------------------------------------------------------- FileExpertSource
FileExpertSource::~FileExpertSource() { close(); }

void FileExpertSource::close() {
#ifndef _WIN32
    if (base_ && map_bytes_) munmap(const_cast<uint8_t*>(base_), map_bytes_);
    if (fd_ >= 0) ::close(fd_);
#endif
    base_ = nullptr; map_bytes_ = 0; fd_ = -1;
    fallback_.clear(); fallback_.shrink_to_fit();
    n_layers_ = n_expert_ = blob_ = 0;
}

bool FileExpertSource::open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob) {
    close();
    if (!geometry_ok(n_layers, n_expert, blob)) return false;
    const uint64_t want = (uint64_t)n_layers * (uint64_t)n_expert * (uint64_t)blob;
#ifndef _WIN32
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;
    struct stat st;
    if (fstat(fd_, &st) != 0 || (uint64_t)st.st_size != want) { close(); return false; }   // exact size or nothing
    void* m = mmap(nullptr, (size_t)want, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { close(); return false; }
    base_ = static_cast<const uint8_t*>(m);
    map_bytes_ = (size_t)want;
#else
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    fallback_.resize((size_t)want);
    const size_t got = std::fread(fallback_.data(), 1, (size_t)want, f);
    std::fseek(f, 0, SEEK_END);
    const long end = std::ftell(f);
    std::fclose(f);
    if (got != (size_t)want || (uint64_t)end != want) { fallback_.clear(); return false; }
    base_ = fallback_.data();
#endif
    n_layers_ = n_layers; n_expert_ = n_expert; blob_ = blob;
    return true;
}

const uint8_t* FileExpertSource::blob(int64_t layer, int64_t expert) {
    if (!base_ || !in_range(layer, expert)) return nullptr;   // expert is range-checked on its own axis
    ++reads_;
    return base_ + (uint64_t)(layer * n_expert_ + expert) * (uint64_t)blob_;
}

// ---------------------------------------------------------------- ArenaExpertSource
ArenaExpertSource::~ArenaExpertSource() { close(); }

void ArenaExpertSource::close() {
#ifndef _WIN32
    if (pinned_ && !arena_.empty()) munlock(arena_.data(), arena_.size());
#endif
    pinned_ = false;
    arena_.clear(); arena_.shrink_to_fit();
    n_layers_ = n_expert_ = blob_ = 0;
}

bool ArenaExpertSource::open_bytes(const uint8_t* data, size_t bytes, int64_t n_layers, int64_t n_expert, int64_t blob,
                                   int64_t max_slot_blob) {
    close();
    if (!geometry_ok(n_layers, n_expert, blob) || !data) return false;
    const uint64_t want = (uint64_t)n_layers * (uint64_t)n_expert * (uint64_t)blob;
    if ((uint64_t)bytes != want) return false;
    const int64_t mb = std::max<int64_t>(blob, max_slot_blob);
    arena_.assign((size_t)(want + (uint64_t)mb), 0);       // arenaSlack = want + max blob
    std::memcpy(arena_.data(), data, (size_t)want);
    n_layers_ = n_layers; n_expert_ = n_expert; blob_ = blob;
#ifndef _WIN32
    pinned_ = (mlock(arena_.data(), arena_.size()) == 0);   // best effort
#endif
    return true;
}

bool ArenaExpertSource::open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob, int64_t max_slot_blob) {
    close();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> buf;
    uint8_t tmp[1 << 16];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    return open_bytes(buf.data(), buf.size(), n_layers, n_expert, blob, max_slot_blob);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (arena_.empty() || !in_range(layer, expert)) return nullptr;
    ++reads_;
    return arena_.data() + (uint64_t)(layer * n_expert_ + expert) * (uint64_t)blob_;
}

// ---------------------------------------------------------------- hit / miss job building
bool expert_hit_run(const std::vector<bool>& is_hit, int pcie_num, ExpertDispatch& out, const int32_t* experts) {
    out = ExpertDispatch{};
    if (pcie_num < 0 || pcie_num > 256) return false;
    out.k = (int32_t)is_hit.size();
    uint64_t nmiss = 0;
    for (bool h : is_hit) nmiss += h ? 0 : 1;
    const uint64_t m = pcie_share_of(nmiss, (uint64_t)pcie_num);
    const uint64_t first_pcie = nmiss - m;               // ranks first_pcie .. nmiss-1 go over PCIe
    uint64_t rank = 0;
    for (size_t i = 0; i < is_hit.size(); ++i) {
        ExpertJob j;
        j.out_index = (int32_t)i;
        j.expert = experts ? experts[i] : 0;
        if (is_hit[i]) { out.gpu_hits.push_back((int32_t)i); continue; }
        if (rank++ >= first_pcie) out.gpu_pcie.push_back(j);
        else out.jobs.push_back(j);
    }
    return true;
}

bool expert_pool_dispatch(ExpertSource& src, GpuPlanSink* sink, int64_t layer, const int32_t* experts,
                          const std::vector<bool>& is_hit, int pcie_num, ExpertDispatch& out) {
    if (!experts && !is_hit.empty()) { out = ExpertDispatch{}; out.failed = true; return false; }
    if (!expert_hit_run(is_hit, pcie_num, out, experts)) { out.failed = true; return false; }
    for (auto& j : out.jobs) {
        j.blob = src.blob(layer, j.expert);
        if (!j.blob) { out.failed = true; return false; }
    }
    for (auto& j : out.gpu_pcie) {
        j.blob = src.blob(layer, j.expert);
        if (!j.blob) { out.failed = true; return false; }
    }
    if (sink) sink->plan(layer, out);
    return true;
}

bool expert_pool_dispatch_multi(ExpertSource& src, GpuPlanSink* sink, int64_t layer,
                                const std::vector<std::vector<int32_t>>& experts,
                                const std::vector<std::vector<bool>>& is_hit, int pcie_num,
                                std::vector<ExpertDispatch>& out) {
    out.clear();
    if (experts.size() != is_hit.size()) return false;
    bool ok = true;
    for (size_t t = 0; t < experts.size(); ++t) {
        if (experts[t].size() != is_hit[t].size()) { ok = false; break; }
        ExpertDispatch d;
        const bool r = expert_pool_dispatch(src, sink, layer, experts[t].data(), is_hit[t], pcie_num, d);
        out.push_back(std::move(d));
        if (!r) { ok = false; break; }     // a failure latches: later tokens are not dispatched
    }
    return ok;
}

// ---------------------------------------------------------------- GGUF expert loading
bool load_experts_gguf(const uint8_t* gate, const uint8_t* up, const uint8_t* down, int64_t n_expert, int64_t up_off,
                       int64_t down_off, int64_t blob, uint8_t* dst, int64_t* n_chunks) {
    if (n_chunks) *n_chunks = 0;
    if (!gate || !up || !down || !dst || n_expert <= 0 || up_off <= 0 || blob <= 0) return false;
    if (down_off != 2 * up_off || 2 * up_off > blob) return false;
    const BlobRegions r = blob_regions(up_off, down_off, blob);
    const uint8_t* srcs[3] = {gate, up, down};
    int64_t chunks = 0;
    for (int64_t e0 = 0; e0 < n_expert; e0 += kGgufExpertsPerChunk) {
        const int64_t n = std::min<int64_t>(kGgufExpertsPerChunk, n_expert - e0);
        for (int64_t e = e0; e < e0 + n; ++e)
            for (int t = 0; t < 3; ++t)
                std::memcpy(dst + e * blob + r.at[t], srcs[t] + e * r.per[t], (size_t)r.per[t]);
        ++chunks;
    }
    if (n_chunks) *n_chunks = chunks;
    return true;
}

}  // namespace core
}  // namespace baloon
