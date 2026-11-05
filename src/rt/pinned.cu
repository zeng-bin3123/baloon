// pinned.cu - implementation of rt_pinned.hpp.
#include "baloon/rt_pinned.hpp"

#include <sys/mman.h>

#include <chrono>
#include <cstring>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#define PIN_HUGE_2MB (21 << 26)   // MAP_HUGE_SHIFT(26) | log2(2 MiB)

namespace baloon {
namespace rt {

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) { h ^= p[i]; h *= kFnvPrime64; }
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

PinArena::PinArena() {}

bool PinArena::reserve(uint64_t bytes, uint64_t slices) {
    release();
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX / 2) return false;
    const uint64_t huge = 2ull << 20;
    void* m = MAP_FAILED; uint64_t mapped = 0;
    if (bytes >= huge) {
        const uint64_t hb = (bytes + huge - 1) / huge * huge;
        m = mmap(nullptr, (size_t)hb, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | PIN_HUGE_2MB, -1, 0);
        if (m != MAP_FAILED) { mapped = hb; backing_ = Backing::HugePinned; }
    }
    if (m == MAP_FAILED) {
        m = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { backing_ = Backing::None; return false; }
        mapped = bytes; backing_ = Backing::Pageable;
    }
    p_ = (uint8_t*)m; bytes_ = bytes; map_bytes_ = mapped;
    // Slice boundaries sit on page boundaries so neighbouring registrations never share a page.
    const uint64_t page = 4096, pages = (bytes + page - 1) / page;
    if (slices == 0) slices = 1;
    if (slices > pages) slices = pages;
    bounds_ = uniform_bounds(pages, slices);
    for (auto& b : bounds_) b = b * page;
    bounds_.back() = bytes;
    pinned_.assign((size_t)slices, false);
    registered_ = 0;
    for (uint64_t i = 0; i < slices; ++i) {
        const uint64_t lo = bounds_[i], hi = bounds_[i + 1];
        if (cudaHostRegister(p_ + lo, (size_t)(hi - lo), cudaHostRegisterDefault) == cudaSuccess) {
            pinned_[i] = true; registered_ += hi - lo;
        } else clear_error();
    }
    if (registered_ != bytes) backing_ = Backing::Pageable;       // partial pinning is reported as pageable
    else if (backing_ == Backing::Pageable) backing_ = Backing::Pinned;
    return true;
}

void PinArena::release() {
    if (p_) {
        for (size_t i = 0; i < pinned_.size(); ++i) if (pinned_[i]) { cudaHostUnregister(p_ + bounds_[i]); }
        clear_error();
        munmap(p_, (size_t)map_bytes_);
    }
    p_ = nullptr; bytes_ = map_bytes_ = registered_ = 0; backing_ = Backing::None;
    bounds_.clear(); pinned_.clear();
}

PinArena::~PinArena() { release(); }

static bool load_span(PinArena& a, const Reader& rd, uint64_t off, uint64_t n, uint64_t dst_off, uint64_t chunk, LoadReport& st) {
    if (chunk == 0 || dst_off > a.bytes() || n > a.bytes() - dst_off) return st.ok = false;
    for (uint64_t pos = 0; pos < n;) {
        const uint64_t m = n - pos < chunk ? n - pos : chunk;
        uint8_t* d = a.data() + dst_off + pos;
        if (!rd(off + pos, m, d)) return st.ok = false;
        st.checksum = fnv1a64(d, m, st.checksum); st.bytes += m; ++st.reads; pos += m;
    }
    return true;
}

LoadReport load_experts(PinArena& a, const Reader& rd, uint64_t layers, uint64_t blobs, uint64_t blob_bytes, uint64_t chunk) {
    LoadReport st;
    if (blobs != 0 && blob_bytes > UINT64_MAX / blobs) { st.ok = false; return st; }
    const uint64_t lb = blobs * blob_bytes;
    if (lb != 0 && layers > UINT64_MAX / lb) { st.ok = false; return st; }
    for (uint64_t L = 0; L < layers && st.ok; ++L) load_span(a, rd, L * lb, lb, L * lb, chunk, st);
    return st;
}

LoadReport load_experts_ranges(PinArena& a, const Reader& rd, const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t chunk) {
    LoadReport st; uint64_t dst = 0;
    for (auto& r : ranges) {
        if (!st.ok) break;
        if (!load_span(a, rd, r.first, r.second, dst, chunk, st)) break;
        dst += r.second;
    }
    return st;
}

BandwidthReport stream_bandwidth(PinArena& a, void* dst, uint64_t chunk, uint64_t passes, cudaStream_t stream) {
    BandwidthReport s;
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

}  // namespace rt
}  // namespace baloon
