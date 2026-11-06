// prefill_chunk_driver.hpp - host-side orchestration of prefill: the device allocator (counting, bump and owned modes),
// the chunk loop, grouping of (token, expert) pairs by expert (counting sort) and the staging/dequant ring discipline.
// bytes_needed must replay exactly the allocation sequence of init; the chunk loop visits every token exactly once in
// pieces of at most T; expert slots are disjoint per expert; ring slots reused inside the lookahead window are distinct.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>

namespace baloon {
namespace prefill_driver {

// ---- allocator -------------------------------------------------------------------------------------------------
// Each allocation is rounded up to 256 bytes and always leaves 256 bytes of slack.
inline uint64_t alloc_bytes(uint64_t b) { return (b + 511) / 256 * 256; }

// One bump allocation from a borrowed region; returns false (and leaves used) when it does not fit.
inline bool bump(uint64_t& used, uint64_t cap, uint64_t bytes) {
    if (used + bytes <= cap) { used += bytes; return true; }
    return false;
}
// Whole allocation sequence from a region: returns the final `used`, or false.
inline bool take_all(uint64_t cap, uint64_t used, const std::vector<uint64_t>& seq, uint64_t& out) {
    for (uint64_t b : seq) if (!bump(used, cap, b)) return false;
    out = used;
    return true;
}
// Counting mode: sum of the rounded sizes of the sequence (bytes_needed).
inline uint64_t bytes_needed(const std::vector<uint64_t>& seq) {
    uint64_t s = 0;
    for (uint64_t b : seq) s += alloc_bytes(b);
    return s;
}

// Allocator with the three modes of the driver. Counting mode only totals; bump mode carves a borrowed region; owned
// mode behaves like bump over a region sized by a preceding count.
struct Alloc {
    enum Mode { Count, Bump, Owned } mode = Count;
    uint64_t cap = 0, used = 0, counted = 0;
    bool failed = false;
    static Alloc counting() { return Alloc{}; }
    static Alloc bump_region(uint64_t cap) { Alloc a; a.mode = Bump; a.cap = cap; return a; }
    static Alloc owned(uint64_t cap) { Alloc a; a.mode = Owned; a.cap = cap; return a; }
    // returns the offset of the allocation (0 in counting mode)
    uint64_t take(uint64_t bytes) {
        const uint64_t r = alloc_bytes(bytes);
        counted += r;
        if (mode == Count) return 0;
        const uint64_t off = used;
        if (!bump(used, cap, r)) { failed = true; return off; }
        return off;
    }
};

// ---- chunk loop ------------------------------------------------------------------------------------------------
// Lengths of the chunks `for c0 in 0..n step m` visits (T = min(m, n - c0)); fuel bounds the iterations.
inline std::vector<uint64_t> chunk_lens(uint64_t m, uint64_t fuel, uint64_t n) {
    std::vector<uint64_t> out;
    for (uint64_t f = 0; f < fuel && n != 0; ++f) { const uint64_t t = std::min(m, n); out.push_back(t); n -= t; }
    return out;
}

// ---- expert grouping (counting sort by expert) -------------------------------------------------------------------
// m.off: prefix sum over m.cnt.
inline std::vector<uint64_t> prefix_offsets(const std::vector<uint64_t>& cnt) {
    std::vector<uint64_t> off(cnt.size() + 1, 0);
    for (size_t e = 0; e < cnt.size(); ++e) off[e + 1] = off[e] + cnt[e];
    return off;
}
struct Grouping {
    std::vector<uint64_t> cnt, off;      // per-expert counts, offsets (size n_experts + 1)
    std::vector<uint64_t> slot;          // slot of each input pair in the sorted order
    std::vector<uint32_t> order;         // order[slot] = input pair index
};
// experts[i] = expert id of pair i (all < n_experts). Stable within an expert.
inline Grouping group_by_expert(const std::vector<uint32_t>& experts, uint32_t n_experts) {
    Grouping g;
    g.cnt.assign(n_experts, 0);
    for (uint32_t e : experts) g.cnt[e]++;
    g.off = prefix_offsets(g.cnt);
    std::vector<uint64_t> next(g.off.begin(), g.off.end() - 1);
    g.slot.assign(experts.size(), 0);
    g.order.assign(experts.size(), 0);
    for (size_t i = 0; i < experts.size(); ++i) {
        const uint64_t s = next[experts[i]]++;
        g.slot[i] = s;
        g.order[s] = (uint32_t)i;
    }
    return g;
}

// ---- rings -----------------------------------------------------------------------------------------------------
constexpr uint32_t kStageRing = 8;    // STAGE: staging ring slots
constexpr uint32_t kDqRing = 2;       // DQ: dequant ring buffers
constexpr uint32_t kLookahead = 8;    // lookahead window
inline uint32_t stage_slot(uint64_t j) { return (uint32_t)(j % kStageRing); }
inline uint32_t ring_next(uint32_t s) { return (s + 1) % kStageRing; }
inline uint32_t dq_slot(uint64_t j) { return (uint32_t)(j % kDqRing); }

// ---- sizes derived from geometry -------------------------------------------------------------------------------
constexpr uint64_t kGdnFloatsPerLayer = 128ull * 48 * 128 + 10240ull * 3;
constexpr uint64_t kGdnBytesPerLayer = kGdnFloatsPerLayer * 4;
static_assert(kGdnFloatsPerLayer == 817152, "gdn floats");
static_assert(kGdnBytesPerLayer == 3268608, "gdn bytes");
constexpr int kLayers = 48;
inline bool is_full_attention_layer(int l) { return l % 4 == 3; }
inline int count_full_attention(int layers) { int c = 0; for (int l = 0; l < layers; ++l) c += is_full_attention_layer(l); return c; }

}  // namespace prefill_driver
}  // namespace baloon
