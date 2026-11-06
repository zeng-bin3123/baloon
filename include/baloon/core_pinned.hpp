// core_pinned.hpp - pinned host arena (host model): sliced registration bookkeeping, checksummed per-layer expert loads,
// range loads and the streaming bandwidth loop. fnv1a64 comes from pinned_arena.hpp (parena::fnv1a64), re-exported here.
#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "baloon/pinned_arena.hpp"

namespace baloon {
namespace core {

enum class PageBacking { Pageable, Pinned, Huge };

inline uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed = parena::kFnvOffset) { return parena::fnv1a64(p, n, seed); }

struct LoadStats { uint64_t bytes = 0, reads = 0, checksum = parena::kFnvOffset; bool ok = true; };
struct StreamStats { uint64_t bytes = 0, passes = 0; int64_t ns = 0; double gbps = 0; };

using ExpertReader = std::function<bool(uint64_t off, uint64_t n, uint8_t* dst)>;

class PinnedArena {
public:
    bool init(uint64_t bytes, uint64_t slice, PageBacking b = PageBacking::Pinned) {
        if (bytes == 0) return false;
        data_.assign((size_t)bytes, 0); slice_ = slice; backing_ = b; return true;
    }
    uint8_t* data() { return data_.data(); }
    uint64_t bytes() const { return data_.size(); }
    PageBacking backing() const { return backing_; }
    uint64_t slices() const { return parena::slice_count(bytes(), slice_); }
    uint64_t covered() const { return parena::slice_covered(bytes(), slice_); }
    uint64_t tail() const { return parena::slice_tail(bytes(), slice_); }   // covered() + tail() == bytes()
private:
    std::vector<uint8_t> data_;
    uint64_t slice_ = 0;
    PageBacking backing_ = PageBacking::Pinned;
};

// Read `n` bytes at `off` into the arena at `dst_off` in `chunk`-byte steps, updating stats (checksum is a left fold).
inline bool load_span(PinnedArena& a, const ExpertReader& rd, uint64_t off, uint64_t n, uint64_t dst_off, uint64_t chunk, LoadStats& st) {
    if (chunk == 0 || dst_off > a.bytes() || n > a.bytes() - dst_off) return st.ok = false;
    for (uint64_t pos = 0; pos < n;) {
        const uint64_t m = parena::read_step(n - pos, chunk);
        uint8_t* d = a.data() + dst_off + pos;
        if (!rd(off + pos, m, d)) return st.ok = false;
        st.checksum = fnv1a64(d, m, st.checksum); st.bytes += m; ++st.reads; pos += m;
    }
    return true;
}

// All layers in order: layer L occupies [layer_off, layer_off + layer_bytes) in both source and arena.
inline LoadStats load_experts(PinnedArena& a, const ExpertReader& rd, uint64_t n_layers, uint64_t blobs_per_layer, uint64_t blob_bytes, uint64_t chunk) {
    LoadStats st;
    for (uint64_t L = 0; L < n_layers && st.ok; ++L) {
        const uint64_t off = parena::layer_off(blobs_per_layer, blob_bytes, L);
        load_span(a, rd, off, parena::layer_bytes(blobs_per_layer, blob_bytes), off, chunk, st);
    }
    return st;
}

// Arbitrary (offset, length) ranges, packed back to back into the arena.
inline LoadStats load_experts_ranges(PinnedArena& a, const ExpertReader& rd, const std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t chunk) {
    LoadStats st; uint64_t dst = 0;
    for (auto& r : ranges) { if (!st.ok) break; load_span(a, rd, r.first, r.second, dst, chunk, st); dst += r.second; }
    return st;
}

// `passes` sweeps of the arena in chunk steps; `copy` models the transfer, `clock_ns` is injectable.
inline StreamStats stream_bandwidth(PinnedArena& a, uint64_t chunk, uint64_t passes, const std::function<void(const uint8_t*, uint64_t)>& copy,
                                    const std::function<int64_t()>& clock_ns) {
    StreamStats s;
    if (chunk == 0) return s;
    const int64_t t0 = clock_ns();
    for (uint64_t p = 0; p < passes; ++p) {
        const uint64_t moved = parena::moved_per_pass(a.bytes(), chunk);
        for (uint64_t pos = 0; pos < moved; pos += chunk) copy(a.data() + pos, chunk);
        s.bytes += moved; ++s.passes;
    }
    s.ns = clock_ns() - t0;
    s.gbps = s.ns > 0 ? (double)s.bytes / (double)s.ns : 0.0;   // bytes/ns == GB/s
    return s;
}

}  // namespace core
}  // namespace baloon
