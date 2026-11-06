// pinned.cu - implementation of core_pinned_hw.hpp (real page-locked arena + parallel checksummed expert load).
#include "baloon/core_pinned_hw.hpp"

#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)   // MAP_HUGE_SHIFT(26) | log2(2 MiB)
#endif

// Plain, unmangled symbol name `fnv1a_chunks_kernel` (global scope, C linkage) so tools can recognise the launch.
extern "C" __global__ void fnv1a_chunks_kernel(const uint8_t* d, uint64_t chunk, uint64_t nchunks, uint64_t seed, uint64_t* out) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nchunks) return;
    const uint8_t* p = d + i * chunk;
    uint64_t h = seed;
    for (uint64_t j = 0; j < chunk; ++j) { h ^= p[j]; h *= 1099511628211ull; }
    out[i] = h;
}

namespace baloon {
namespace hw {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= kFnvPrime; }
    return h;
}

std::vector<uint64_t> uniform_bounds(uint64_t total, uint64_t parts) {
    std::vector<uint64_t> b;
    if (parts == 0) return b;
    b.reserve((size_t)parts + 1);
    const uint64_t q = total / parts, r = total % parts;
    uint64_t at = 0; b.push_back(0);
    for (uint64_t i = 0; i < parts; ++i) { at += q + (i < r ? 1 : 0); b.push_back(at); }
    return b;
}

int clear_error() {
    int n = 0;
    while (cudaGetLastError() != cudaSuccess && n < 16) ++n;
    return n;
}

PinnedArena::PinnedArena() {}

bool PinnedArena::reserve(uint64_t bytes, uint64_t slice) {
    release();
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX / 2) return false;
    const uint64_t huge = 2ull << 20, page = 4096;
    void* m = MAP_FAILED; uint64_t mapped = 0;
    if (bytes >= huge) {
        const uint64_t hb = (bytes + huge - 1) / huge * huge;
        m = mmap(nullptr, (size_t)hb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (m != MAP_FAILED) { mapped = hb; backing_ = Backing::HugePinned; }
    }
    if (m == MAP_FAILED) {
        m = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { backing_ = Backing::None; return false; }
        mapped = bytes; backing_ = Backing::Pageable;
    }
    p_ = (uint8_t*)m; bytes_ = bytes; map_bytes_ = mapped;
    if (slice == 0 || slice >= bytes) slice = bytes;
    else { slice = slice / page * page; if (slice < page) slice = page; }
    slice_ = slice;
    const uint64_t nslices = bytes / slice;          // whole slices only; the remainder (< slice) is the tail
    registered_ = 0;
    for (uint64_t i = 0; i < nslices; ++i) {
        if (cudaHostRegister(p_ + i * slice, (size_t)slice, cudaHostRegisterDefault) != cudaSuccess) { clear_error(); break; }
        registered_ = (i + 1) * slice;
    }
    // Tail (or everything, if registration failed at once): locked resident instead of registered.
    locked_ok_ = true;
    if (registered_ < bytes) locked_ok_ = (mlock(p_ + registered_, (size_t)(bytes - registered_)) == 0);
    if (registered_ == bytes && backing_ == Backing::Pageable) backing_ = Backing::Pinned;
    else if (registered_ != bytes && backing_ == Backing::HugePinned) backing_ = Backing::Pageable;
    return true;
}

void PinnedArena::release() {
    if (p_) {
        const uint64_t n = slice_ ? registered_ / slice_ : 0;
        for (uint64_t i = 0; i < n; ++i) cudaHostUnregister(p_ + i * slice_);
        clear_error();
        if (registered_ < bytes_ && locked_ok_) munlock(p_ + registered_, (size_t)(bytes_ - registered_));
        munmap(p_, (size_t)map_bytes_);
    }
    p_ = nullptr; bytes_ = map_bytes_ = registered_ = slice_ = 0; backing_ = Backing::None; locked_ok_ = true;
}

PinnedArena::~PinnedArena() { release(); }

// Reads [off, off+n) into dst_off in `chunk` steps, folding the hash chunk by chunk (running value is the next seed).
static bool read_span(PinnedArena& a, const Reader& rd, uint64_t off, uint64_t n, uint64_t dst_off, uint64_t chunk,
                      uint64_t& h, uint64_t& bytes, uint64_t& reads) {
    if (chunk == 0 || dst_off > a.bytes() || n > a.bytes() - dst_off) return false;
    for (uint64_t pos = 0; pos < n;) {
        const uint64_t m = n - pos < chunk ? n - pos : chunk;
        uint8_t* d = a.data() + dst_off + pos;
        if (!rd(off + pos, m, d)) return false;
        h = fnv1a64(d, m, h); bytes += m; ++reads; pos += m;
    }
    return true;
}

LoadResult load_experts(PinnedArena& a, const Reader& rd, uint64_t layers, uint64_t bpl, uint64_t blob, uint64_t chunk,
                        unsigned workers) {
    LoadResult res;
    if (bpl != 0 && blob > UINT64_MAX / bpl) { res.ok = false; return res; }
    const uint64_t lb = bpl * blob;
    if (lb != 0 && layers > UINT64_MAX / lb) { res.ok = false; return res; }
    res.hashes.assign((size_t)layers, kRefFnvSeed);
    res.claimed_by.assign((size_t)layers, 0xFFFFFFFFu);
    if (workers == 0) workers = 1;
    std::atomic<uint64_t> next_layer{0};
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> tot_bytes{0}, tot_reads{0};
    auto work = [&](unsigned w) {
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);           // each layer is claimed exactly once
            if (L >= layers || failed.load()) return;
            uint64_t h = kRefFnvSeed, b = 0, r = 0;
            if (!read_span(a, rd, L * lb, lb, L * lb, chunk, h, b, r)) { failed.store(true); return; }
            res.hashes[(size_t)L] = h; res.claimed_by[(size_t)L] = w;
            tot_bytes += b; tot_reads += r;
        }
    };
    std::vector<std::thread> ts;
    for (unsigned w = 1; w < workers; ++w) ts.emplace_back(work, w);
    work(0);
    for (auto& t : ts) t.join();
    res.ok = !failed.load();
    res.bytes = tot_bytes.load(); res.reads = tot_reads.load();
    return res;
}

LoadResult load_experts_ranges(PinnedArena& a, const Reader& rd, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                               uint64_t chunk) {
    LoadResult res; uint64_t dst = 0;
    for (auto& r : ranges) {
        uint64_t h = kRefFnvSeed;
        if (!read_span(a, rd, r.first, r.second, dst, chunk, h, res.bytes, res.reads)) { res.ok = false; break; }
        res.hashes.push_back(h); dst += r.second;
    }
    return res;
}

Bandwidth stream_bandwidth(PinnedArena& a, void* dst, uint64_t chunk, uint64_t passes, cudaStream_t stream) {
    Bandwidth s;
    if (chunk == 0 || !dst || !a.data()) { s.ok = false; return s; }
    const uint64_t moved = a.bytes() / chunk * chunk;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t p = 0; p < passes && s.ok; ++p) {
        for (uint64_t pos = 0; pos < moved; pos += chunk)
            if (cudaMemcpyAsync((uint8_t*)dst + pos, a.data() + pos, (size_t)chunk, cudaMemcpyHostToDevice, stream) != cudaSuccess) { s.ok = false; break; }
        if (s.ok) { s.bytes += moved; ++s.passes; }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { clear_error(); s.ok = false; }
    s.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    s.gbps = s.ms > 0 ? (double)s.bytes / (s.ms * 1e6) : 0.0;
    return s;
}

bool device_chunk_hashes(const void* dev, uint64_t bytes, uint64_t chunk, std::vector<uint64_t>& out, uint64_t seed,
                         cudaStream_t stream) {
    out.clear();
    if (!dev || chunk == 0) return false;
    const uint64_t n = bytes / chunk;
    if (n == 0) return true;
    uint64_t* d_out = nullptr;
    if (cudaMalloc((void**)&d_out, (size_t)(n * sizeof(uint64_t))) != cudaSuccess) { clear_error(); return false; }
    const unsigned threads = 64, blocks = (unsigned)((n + threads - 1) / threads);
    fnv1a_chunks_kernel<<<blocks, threads, 0, stream>>>((const uint8_t*)dev, chunk, n, seed, d_out);
    out.assign((size_t)n, 0);
    bool ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess &&
              cudaMemcpy(out.data(), d_out, (size_t)(n * sizeof(uint64_t)), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (!ok) { clear_error(); out.clear(); }
    cudaFree(d_out);
    return ok;
}

}  // namespace hw
}  // namespace baloon
