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

    // Truncating (natural-number) subtraction: never wraps.
    uint64_t n_gdn() const { return n_layers > n_qsa ? n_layers - n_qsa : 0; }
};

// Fixed, non-evictable GDN recurrence + conv state.
inline uint64_t state_bytes(const Geometry& g) {
    const uint64_t conv_hist = g.ssm_conv > 0 ? g.ssm_conv - 1 : 0;
    return g.n_gdn() * (g.ssm_state * g.ssm_v_heads * g.ssm_state * 4 + conv_hist * g.ssm_channels * 4);
}

// Per-token cost over QSA layers only: int8 K+V, one fp16 scale per group, one cached indexer key
// (idx_key_dim wide) per qsa_block-token block.
inline uint64_t kv_bytes_per_token(const Geometry& g) {
    const uint64_t kv_elems = 2 * g.n_kv_heads * g.head_dim;
    const uint64_t kv_scales = g.kv_group ? (kv_elems / g.kv_group) * 2 : 0;  // x/0 = 0, as in the spec
    const uint64_t per_layer = kv_elems + kv_scales;
    const uint64_t idx_layer = g.qsa_block ? g.idx_key_heads * g.idx_key_dim / g.qsa_block : 0;
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

// ---- Costs-based form: fixed per-sequence costs, plan carries the full accounting ----------------------

// Byte costs fixed for the life of a sequence.
struct Costs {
    uint64_t expert_blob = kExpertBlobBytes;
    uint64_t dense = 0;
    uint64_t embd = 0;
    uint64_t mtp = 0;
    uint64_t workspace = 0;
    uint64_t state = 0;
};

// Sum of the fixed (non-KV, non-cache) costs, in 128 bits so it cannot wrap.
inline unsigned __int128 fixed_cost_wide(const Costs& c) {
    return (unsigned __int128)c.dense + c.embd + c.mtp + c.workspace + c.state;
}

struct FullPlan {
    uint64_t max_context = 0;
    uint64_t kv_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t cache_bytes = 0;
    uint64_t cache_slots = 0;
    uint64_t vram_budget = 0;
    uint64_t vram_used = 0;
};

// std::nullopt = DoesNotClose. Sound: a returned plan has kv + fixed + slots*blob <= pool, slots >= 1,
// vram_used <= pool. Complete: if kv + fixed + one blob fit in the pool (blob > 0), a plan is returned.
inline std::optional<FullPlan> make_plan(uint64_t ctx, const Geometry& g, const Costs& c, uint64_t pool) {
    if (c.expert_blob == 0) return std::nullopt;
    const unsigned __int128 kv = (unsigned __int128)kv_bytes_per_token(g) * ctx;
    const unsigned __int128 need = kv + fixed_cost_wide(c);
    if ((unsigned __int128)pool < need) return std::nullopt;
    const uint64_t cache = (uint64_t)((unsigned __int128)pool - need);
    const uint64_t slots = cache / c.expert_blob;
    if (slots == 0) return std::nullopt;
    FullPlan p;
    p.max_context = ctx;
    p.kv_bytes = (uint64_t)kv;
    p.state_bytes = c.state;
    p.cache_bytes = cache;
    p.cache_slots = slots;
    p.vram_budget = pool;
    p.vram_used = pool - cache % c.expert_blob;
    return p;
}

}  // namespace planner
}  // namespace baloon
