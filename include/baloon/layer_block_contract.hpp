// layer_block_contract.hpp - structural contracts of one decoder layer: MoE split at the router, block_layer_pre stage
// halves, 16-byte alignment, QSA q|gate split, pooled-key row bound, dump buffer tiling.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace layer_block {

// route -> shared -> combine; pre = route, post = combine o shared. State-threading, callable-generic.
template <class S, class R, class Sh, class C> S moe_layer(R route, Sh shared, C combine, S s) {
    return combine(shared(route(s)));
}
template <class S, class R> S pre(R route, S s) { return route(s); }
template <class S, class Sh, class C> S post(Sh shared, C combine, S s) { return combine(shared(s)); }

// Stages run by `half`: 1 = {0,1,2}, 2 = {3,4}, otherwise all five.
inline std::vector<int> stages_of(int half) {
    if (half == 1) return {0, 1, 2};
    if (half == 2) return {3, 4};
    return {0, 1, 2, 3, 4};
}

constexpr uint64_t align16(uint64_t n) { return (n + 15) / 16 * 16; }

constexpr uint64_t dump_len(uint64_t ne, uint64_t hc, uint64_t nh, uint64_t hd) { return 2 * ne + 2 * hc + nh * hd; }

constexpr uint64_t q_idx(uint64_t hd, uint64_t h, uint64_t i) { return h * (2 * hd) + i; }
constexpr uint64_t g_idx(uint64_t hd, uint64_t h, uint64_t i) { return h * (2 * hd) + hd + i; }
constexpr bool q_full_in_range(uint64_t nh, uint64_t hd, uint64_t h, uint64_t i) { return h < nh && i < hd; }
constexpr uint64_t q_full_len(uint64_t nh, uint64_t hd) { return nh * 2 * hd; }

// Pooled-key row of a cell; the state allocates max_cells/b + 2 rows.
constexpr uint64_t pooled_row(uint64_t pos, uint64_t b) { return pos / b; }
constexpr uint64_t pooled_rows(uint64_t max_cells, uint64_t b) { return max_cells / b + 2; }

}  // namespace layer_block
}  // namespace baloon
