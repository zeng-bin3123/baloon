// expert_cache.cpp - host model of the resident expert cache (see core_expert_cache.hpp).
#include "baloon/core_expert_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace baloon {
namespace core {

static int32_t rd_i32(const uint8_t* p) {
    uint32_t u = 0;
    for (int i = 0; i < 4; ++i) u |= (uint32_t)p[i] << (8 * i);
    return (int32_t)u;
}
static void wr_i32(std::vector<uint8_t>& b, int32_t v) {
    const uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; ++i) b.push_back((uint8_t)(u >> (8 * i)));
}

bool parse_expert_profile(const uint8_t* data, size_t bytes, int64_t n_layers, int64_t n_expert, int64_t want,
                          std::vector<std::pair<int32_t, int32_t>>& out) {
    out.clear();
    if (bytes < 12) return false;
    const int64_t nl = rd_i32(data), ne = rd_i32(data + 4), nr = rd_i32(data + 8);
    if (!profile_header_ok(nl, ne, nr, want, n_layers, n_expert)) return false;
    if ((uint64_t)bytes != 12ull + 8ull * (uint64_t)nr) return false;
    std::vector<std::pair<int32_t, int32_t>> v;
    v.reserve((size_t)nr);
    for (int64_t i = 0; i < nr; ++i) {
        const int32_t l = rd_i32(data + 12 + 8 * i), e = rd_i32(data + 16 + 8 * i);
        if (l < 0 || e < 0 || l >= n_layers || e >= n_expert) return false;   // every ranked pair must be in range
        v.emplace_back(l, e);
    }
    out.swap(v);
    return true;
}

bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t want,
                         std::vector<std::pair<int32_t, int32_t>>& out) {
    out.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<uint8_t> buf;
    uint8_t tmp[4096];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + n);
    std::fclose(f);
    return parse_expert_profile(buf.data(), buf.size(), n_layers, n_expert, want, out);
}

std::vector<uint8_t> encode_expert_profile(int64_t n_layers, int64_t n_expert,
                                           const std::vector<std::pair<int32_t, int32_t>>& pairs) {
    std::vector<uint8_t> b;
    wr_i32(b, (int32_t)n_layers); wr_i32(b, (int32_t)n_expert); wr_i32(b, (int32_t)pairs.size());
    for (auto& p : pairs) { wr_i32(b, p.first); wr_i32(b, p.second); }
    return b;
}

ExpertCache::~ExpertCache() { close(); }

void ExpertCache::close() {
    table_.clear(); next_.clear(); off_.clear(); arena_.clear();
    open_ = false; n_layers_ = n_expert_ = slots_ = blob_ = resident_ = pending_ = next_free_ = 0;
}

static bool dims_ok(int64_t nl, int64_t ne) { return nl > 0 && ne > 0 && nl <= (1 << 20) && ne <= (1 << 20); }

bool ExpertCache::open(int64_t n_layers, int64_t n_expert, int64_t slots, int64_t blob, bool per_layer) {
    close();
    if (!dims_ok(n_layers, n_expert) || slots <= 0 || blob <= 0) return false;
    // Uniform slots are packed back to back (slots * blob), not re-aligned.
    n_layers_ = n_layers; n_expert_ = n_expert; slots_ = slots; blob_ = blob; per_layer_ = per_layer;
    off_.resize((size_t)slots + 1);
    for (int64_t i = 0; i <= slots; ++i) off_[(size_t)i] = i * blob;
    arena_.assign((size_t)(slots * blob), 0);
    table_.assign((size_t)(n_layers * n_expert), kNotResident);
    next_.resize((size_t)n_layers);
    for (int64_t l = 0; l < n_layers; ++l) next_[(size_t)l] = layer_slot_range(l).first;
    open_ = true;
    return true;
}

bool ExpertCache::open_sized(int64_t n_layers, int64_t n_expert, const std::vector<int64_t>& blobs, bool per_layer) {
    close();
    if (!dims_ok(n_layers, n_expert) || blobs.empty()) return false;
    for (int64_t b : blobs) if (b <= 0) return false;
    n_layers_ = n_layers; n_expert_ = n_expert; slots_ = (int64_t)blobs.size(); per_layer_ = per_layer;
    blob_ = *std::max_element(blobs.begin(), blobs.end());
    off_.assign(1, 0);
    for (int64_t b : blobs) off_.push_back(off_.back() + cache_align256(b));   // each slot advances by its aligned size
    arena_.assign((size_t)off_.back(), 0);
    table_.assign((size_t)(n_layers * n_expert), kNotResident);
    next_.resize((size_t)n_layers);
    for (int64_t l = 0; l < n_layers; ++l) next_[(size_t)l] = layer_slot_range(l).first;
    sized_ = blobs;
    open_ = true;
    return true;
}

int32_t ExpertCache::slot_of(int64_t layer, int64_t expert) const {
    if (!open_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    return table_[(size_t)(layer * n_expert_ + expert)];
}

std::pair<int64_t, int64_t> ExpertCache::layer_slot_range(int64_t layer) const {
    if (n_layers_ <= 0 || layer < 0 || layer >= n_layers_) return {0, 0};
    const int64_t q = slots_ / n_layers_;
    const int64_t lo = layer * q;
    const int64_t hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
    return {lo, hi};
}

int32_t ExpertCache::admit(int64_t layer, int64_t expert) {
    if (!open_ || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return kNotResident;
    int32_t& cell = table_[(size_t)(layer * n_expert_ + expert)];
    if (cell != kNotResident) return cell;                      // already resident
    int64_t slot;
    if (per_layer_) {
        const int64_t hi = layer_slot_range(layer).second;
        int64_t& nx = next_[(size_t)layer];
        if (!(nx < hi)) return kNotResident;                    // layer full: nothing is evicted
        slot = nx++;
    } else {
        if (!(next_free_ < slots_)) return kNotResident;
        slot = next_free_++;
    }
    cell = (int32_t)slot;
    ++resident_;
    return cell;
}

uint8_t* ExpertCache::device_slot(int64_t slot) {
    if (!open_ || slot < 0 || slot >= slots_) return nullptr;
    return arena_.data() + off_[(size_t)slot];
}

int64_t ExpertCache::slot_bytes(int64_t slot) const {
    if (!open_ || slot < 0 || slot >= slots_) return 0;
    return sized_.empty() ? blob_ : sized_[(size_t)slot];
}

bool ExpertCache::fill_slot(int64_t slot, const void* src, int64_t bytes) {
    uint8_t* dst = device_slot(slot);
    if (!dst || !src) return false;
    const int64_t n = fill_len(bytes, slot_bytes(slot));
    std::memcpy(dst, src, (size_t)n);
    ++pending_;
    return true;
}

bool ExpertCache::fill_slot_blocking(int64_t slot, const void* src, int64_t bytes) {
    if (!fill_slot(slot, src, bytes)) return false;
    sync();
    return true;
}

bool ExpertCache::verify_slot(int64_t slot, const void* src, int64_t bytes) const {
    if (!open_ || slot < 0 || slot >= slots_ || !src) return false;
    const int64_t n = fill_len(bytes, slot_bytes(slot));
    return std::memcmp(arena_.data() + off_[(size_t)slot], src, (size_t)n) == 0;
}

}  // namespace core
}  // namespace baloon
