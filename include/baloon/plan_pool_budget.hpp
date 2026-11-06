// plan_pool_budget.hpp - arithmetic facts of the pool budget beyond the planner's refusal logic.
#pragma once
#include <cstdint>
#include <optional>
#include "baloon/memory_planner.hpp"

namespace baloon {
namespace budget {

// Natural-number (truncating) subtraction.
inline uint64_t sat_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

// The cached indexer key is ONE key head x width per qsa_block-token block (32 bytes/token/layer for 1x128/4).
inline uint64_t indexer_bytes_per_token_layer(uint64_t key_heads, uint64_t key_width, uint64_t block) {
    return key_heads * key_width / block;
}

// KV scale bytes: one fp16 scale per group; exact only when elems is a multiple of the group.
inline std::optional<uint64_t> kv_scale_bytes(uint64_t elems, uint64_t group) {
    if (group == 0 || elems % group != 0) return std::nullopt;
    return elems / group * 2;
}

// Per-layer bytes = INT8 K+V + scales + indexer.
inline uint64_t kv_layer_bytes(const planner::Geometry& g) {
    const uint64_t elems = 2 * g.n_kv_heads * g.head_dim;
    return elems + (elems / g.kv_group) * 2 + g.idx_key_heads * g.idx_key_dim / g.qsa_block;
}

// Whole expert-cache slots left after kv and fixed costs.
inline uint64_t slots_for(uint64_t pool, uint64_t kv, uint64_t fixed, uint64_t blob) {
    return blob == 0 ? 0 : sat_sub(sat_sub(pool, kv), fixed) / blob;
}
inline uint64_t leftover(uint64_t pool, uint64_t kv, uint64_t fixed, uint64_t blob) {
    return blob == 0 ? 0 : sat_sub(sat_sub(pool, kv), fixed) % blob;
}
// Reported use: pool minus the sub-slot remainder (= kv + fixed + slots * blob).
inline uint64_t used_bytes(uint64_t pool, uint64_t kv, uint64_t fixed, uint64_t blob) {
    return pool - leftover(pool, kv, fixed, blob);
}
// How many whole expert-cache slots the GDN state is worth.
inline uint64_t state_in_slots(const planner::Geometry& g, uint64_t blob) {
    return blob == 0 ? 0 : planner::state_bytes(g) / blob;
}

}  // namespace budget
}  // namespace baloon
