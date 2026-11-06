// mtp_draft_layer.hpp - MTP draft layer bookkeeping: 256-byte bump arena (count pass and carve pass agree), attention
// window capacity (multiple of 64), step-row staging for the catch-up / chain / full-layer graphs, draft vocab size.
// Rows: catch-up [0,T), chain row j at max_t+j-1, staged full-layer row 2*max_t-1; all disjoint and below 2*max_t.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace baloon {
namespace mtp {

// ---- bump allocator --------------------------------------------------------------------------------------------
inline uint64_t align256(uint64_t b) { return (b + 255) / 256 * 256; }

// Cursor after a sequence of takes starting at `acc`.
inline uint64_t bump_end(uint64_t acc, const std::vector<uint64_t>& sizes) {
    for (uint64_t b : sizes) acc += align256(b);
    return acc;
}

// Arena with a count pass (cap == 0 and counting) and a carve pass over a real capacity; both use the same sizes.
struct Bump {
    uint64_t used = 0, cap = 0;
    bool counting = true, overflow = false;
    static Bump count() { return Bump{}; }
    static Bump carve(uint64_t capacity) { Bump b; b.cap = capacity; b.counting = false; return b; }
    // returns the byte offset of the block
    uint64_t take(uint64_t bytes) {
        const uint64_t off = used;
        used += align256(bytes);
        if (!counting && used > cap) overflow = true;
        return off;
    }
    template <class T> uint64_t take_n(uint64_t n) { return take(n * sizeof(T)); }
};

// ---- attention window ------------------------------------------------------------------------------------------
inline uint64_t window_of(uint64_t window, uint64_t max_cells) {
    return (0 < window && window < max_cells) ? window : max_cells;
}
inline uint64_t cap_of(uint64_t window, uint64_t max_cells) { return (window_of(window, max_cells) + 63) / 64 * 64; }

// ---- step rows -------------------------------------------------------------------------------------------------
inline uint64_t staged_row(uint64_t max_t) { return 2 * max_t - 1; }
inline uint64_t chain_row(uint64_t max_t, uint64_t j) { return max_t + j - 1; }
// Valid for max_t >= 2 and chain steps 1 <= j <= max_t-2.
inline bool rows_disjoint(uint64_t max_t, uint64_t j) {
    return max_t >= 2 && j >= 1 && j + 2 <= max_t &&
           max_t - 1 < chain_row(max_t, j) && chain_row(max_t, j) < staged_row(max_t) && staged_row(max_t) < 2 * max_t;
}

// ---- draft vocabulary ------------------------------------------------------------------------------------------
inline uint64_t n_dvocab(uint64_t file_bytes) { return file_bytes / 4; }

}  // namespace mtp
}  // namespace baloon
