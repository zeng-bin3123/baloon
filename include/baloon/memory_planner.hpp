// memory_planner.hpp - VRAM planner: KV / indexer-key / GDN-state byte costs from geometry; either an
// expert-cache plan that fits the pool or a refusal.
#pragma once
#include <cstdint>
#include <optional>

namespace baloon {
namespace planner {

struct Geometry {
    uint64_t n_layers = 48;
    uint64_t n_qsa = 12;
    uint64_t n_kv_heads = 2;
    uint64_t head_dim = 256;
    uint64_t kv_group = 64;
    uint64_t qsa_block = 4;
    uint64_t idx_key_heads = 1;
    uint64_t idx_key_dim = 128;
    uint64_t ssm_state = 128;
    uint64_t ssm_v_heads = 48;
    uint64_t ssm_conv = 4;
    uint64_t ssm_channels = 10240;

    uint64_t n_gdn() const { return n_layers - n_qsa; }
};

// Fixed, non-evictable GDN recurrence + conv state.
inline uint64_t state_bytes(const Geometry& g) {
    return g.n_gdn() * (g.ssm_state * g.ssm_v_heads * g.ssm_state * 4 + (g.ssm_conv - 1) * g.ssm_channels * 4);
}

// Per-token cost over QSA layers only: int8 K+V, one fp16 scale per group, one cached indexer key
// (idx_key_dim wide) per qsa_block-token block.
inline uint64_t kv_bytes_per_token(const Geometry& g) {
    const uint64_t kv_elems = 2 * g.n_kv_heads * g.head_dim;
    const uint64_t kv_scales = (kv_elems / g.kv_group) * 2;
    const uint64_t per_layer = kv_elems + kv_scales;
    const uint64_t idx_layer = g.idx_key_heads * g.idx_key_dim / g.qsa_block;
    return g.n_qsa * (per_layer + idx_layer);
}

// Expert blob (codes + scales) in bytes, for the default expert shape.
constexpr uint64_t kExpertBlobBytes = 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2;

struct Plan {
    uint64_t kv = 0;
    uint64_t state = 0;
    uint64_t cache = 0;
    uint64_t slots = 0;
    uint64_t used = 0;
};

// std::nullopt = DoesNotClose: fixed costs exceed the pool, or no whole expert slot remains.
// Never overflows: an unrepresentable product cannot fit any pool, so it refuses.
inline std::optional<Plan> make_plan(uint64_t pool, uint64_t kv_t, uint64_t ctx, uint64_t fixed, uint64_t blob) {
    if (blob == 0) return std::nullopt;
    unsigned __int128 kv128 = (unsigned __int128)kv_t * ctx;
    if (kv128 + fixed > (unsigned __int128)pool) return std::nullopt;
    const uint64_t kv = (uint64_t)kv128;
    const uint64_t cache = pool - kv - fixed;
    const uint64_t slots = cache / blob;
    if (slots == 0) return std::nullopt;
    Plan p;
    p.kv = kv;
    p.state = 0;
    p.cache = cache;
    p.slots = slots;
    p.used = pool - cache % blob;
    return p;
}

}  // namespace planner
}  // namespace baloon
