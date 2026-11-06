// mtp_drafter.hpp - MTP drafter arithmetic: 256-byte bump arena totals, window capacity, staging rows, chunk loops,
// draft count bounds.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace baloon {
namespace mtp_drafter {

constexpr uint64_t align256(uint64_t n) { return (n + 255) / 256 * 256; }
inline uint64_t bump_total(const std::vector<uint64_t>& l) {
    uint64_t t = 0;
    for (uint64_t b : l) t += align256(b);
    return t;
}

constexpr uint64_t eff_window(uint64_t window, uint64_t max_cells) {
    return (0 < window && window < max_cells) ? window : max_cells;
}
constexpr uint64_t cap_of(uint64_t window, uint64_t max_cells) { return (eff_window(window, max_cells) + 63) / 64 * 64; }
constexpr uint64_t ident_index(uint64_t t, uint64_t i, uint64_t cap) { return t * cap + i; }

constexpr uint64_t chain_row(uint64_t max_t, uint64_t j) { return max_t + j - 1; }
constexpr uint64_t full_layer_row(uint64_t max_t) { return 2 * max_t - 1; }

// Eight-column chunk starting at c0 < total.
constexpr uint64_t mmvq_chunk(uint64_t total, uint64_t c0) { return std::min<uint64_t>(8, total - c0); }
// Prefill chunk starting at c < n.
constexpr uint64_t prefill_chunk(uint64_t n, uint64_t c, uint64_t max_t) { return std::min<uint64_t>(max_t, n - c); }
// Draft chain: n = 1 + k extra steps (k <= max_t-2).
constexpr uint64_t draft_count(uint64_t k) { return 1 + k; }
constexpr bool draft_arg_row_ok(uint64_t T, uint64_t max_t, uint64_t a) { return 1 <= T && T <= max_t && a < T; }

}  // namespace mtp_drafter
}  // namespace baloon
