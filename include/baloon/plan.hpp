// plan.hpp - VRAM plan front end: geometry/cost types of the planner, a plan that carries the full accounting, an
// explicit refusal type (DoesNotClose) with the reason, the pool derived from device memory, and printable forms.
//
// Invariants (checked by tests/k22_plan):
//  * make_plan is sound:    kv_bytes + fixed + cache_slots * expert_blob <= vram_used <= vram_budget, cache_slots >= 1;
//  * make_plan is complete: a refusal happens only if blob == 0, kv + fixed > pool, or no whole expert slot is left;
//  * 128-bit intermediate sums: an unrepresentable cost refuses instead of wrapping;
//  * vram_pool_bytes never exceeds the device memory and never wraps (truncating subtraction).
#pragma once
#include <cstdint>
#include <string>
#include <variant>
#include "baloon/memory_planner.hpp"

namespace baloon {
namespace plan {

// Own types (not aliases of the planner's): keeps overload resolution unambiguous under argument-dependent lookup.
struct Geometry {
    uint64_t n_layers = 48, n_qsa = 12, n_kv_heads = 2, head_dim = 256, kv_group = 64, qsa_block = 4;
    uint64_t idx_key_heads = 1, idx_key_dim = 128, ssm_state = 128, ssm_v_heads = 48, ssm_conv = 4, ssm_channels = 10240;
    uint64_t n_gdn() const { return n_layers > n_qsa ? n_layers - n_qsa : 0; }
};

struct Costs {
    uint64_t expert_blob = planner::kExpertBlobBytes;
    uint64_t dense = 0, embd = 0, mtp = 0, workspace = 0, state = 0;
};

// Full accounting of a plan that closes.
struct Plan {
    uint64_t max_context = 0, kv_bytes = 0, state_bytes = 0, cache_bytes = 0, cache_slots = 0, vram_budget = 0, vram_used = 0;
};

namespace detail {
inline planner::Geometry lower(const Geometry& g) {
    planner::Geometry o;
    o.n_layers = g.n_layers; o.n_qsa = g.n_qsa; o.n_kv_heads = g.n_kv_heads; o.head_dim = g.head_dim;
    o.kv_group = g.kv_group; o.qsa_block = g.qsa_block; o.idx_key_heads = g.idx_key_heads; o.idx_key_dim = g.idx_key_dim;
    o.ssm_state = g.ssm_state; o.ssm_v_heads = g.ssm_v_heads; o.ssm_conv = g.ssm_conv; o.ssm_channels = g.ssm_channels;
    return o;
}
inline planner::Costs lower(const Costs& c) {
    planner::Costs o;
    o.expert_blob = c.expert_blob; o.dense = c.dense; o.embd = c.embd; o.mtp = c.mtp; o.workspace = c.workspace; o.state = c.state;
    return o;
}
}  // namespace detail

// Fixed, non-evictable GDN recurrence + conv state.
inline uint64_t state_bytes(const Geometry& g) { return planner::state_bytes(detail::lower(g)); }
// Per-token KV cost over the QSA layers (int8 K+V, fp16 group scales, indexer key per block).
inline uint64_t kv_bytes_per_token(const Geometry& g) { return planner::kv_bytes_per_token(detail::lower(g)); }

// Why no plan exists.
enum class Reason : uint8_t { ZeroExpertBlob = 0, FixedExceedsPool = 1, NoWholeSlot = 2 };

// Refusal: the fixed costs do not fit the pool, or not even one whole expert slot remains.
struct DoesNotClose {
    Reason reason = Reason::FixedExceedsPool;
    uint64_t pool = 0;      // bytes available
    uint64_t kv_bytes = 0;  // saturated at UINT64_MAX when unrepresentable
    uint64_t fixed = 0;     // saturated at UINT64_MAX when unrepresentable
    uint64_t blob = 0;      // bytes of one expert slot
};

using Result = std::variant<Plan, DoesNotClose>;

namespace detail {
inline uint64_t sat64(unsigned __int128 v) { return v > (unsigned __int128)UINT64_MAX ? UINT64_MAX : (uint64_t)v; }
}  // namespace detail

// Pool the planner may spend: device memory minus the reserve kept for the runtime. Truncating subtraction.
inline uint64_t vram_pool_bytes(uint64_t device_bytes, uint64_t reserve_bytes) {
    return device_bytes > reserve_bytes ? device_bytes - reserve_bytes : 0;
}

// Plan for `ctx` tokens of context in `pool` bytes, or the refusal with its reason.
inline Result make_plan(uint64_t ctx, const Geometry& g, const Costs& c, uint64_t pool) {
    const unsigned __int128 kv = (unsigned __int128)kv_bytes_per_token(g) * ctx;
    const unsigned __int128 fixed = planner::fixed_cost_wide(detail::lower(c));
    DoesNotClose d;
    d.pool = pool; d.kv_bytes = detail::sat64(kv); d.fixed = detail::sat64(fixed); d.blob = c.expert_blob;
    if (c.expert_blob == 0) { d.reason = Reason::ZeroExpertBlob; return d; }
    if ((unsigned __int128)pool < kv + fixed) { d.reason = Reason::FixedExceedsPool; return d; }
    const auto p = planner::make_plan(ctx, detail::lower(g), detail::lower(c), pool);
    if (!p) { d.reason = Reason::NoWholeSlot; return d; }
    Plan o;
    o.max_context = p->max_context; o.kv_bytes = p->kv_bytes; o.state_bytes = p->state_bytes; o.cache_bytes = p->cache_bytes;
    o.cache_slots = p->cache_slots; o.vram_budget = p->vram_budget; o.vram_used = p->vram_used;
    return o;
}

inline const char* to_string(Reason r) {
    switch (r) {
        case Reason::ZeroExpertBlob: return "expert blob is zero bytes";
        case Reason::FixedExceedsPool: return "kv and fixed costs exceed the pool";
        case Reason::NoWholeSlot: return "no whole expert slot fits in the remaining pool";
    }
    return "unknown";
}

inline std::string to_string(const DoesNotClose& d) {
    return std::string("plan does not close: ") + to_string(d.reason) + " (pool=" + std::to_string(d.pool) +
           " kv=" + std::to_string(d.kv_bytes) + " fixed=" + std::to_string(d.fixed) +
           " blob=" + std::to_string(d.blob) + ")";
}

inline std::string to_string(const Plan& p) {
    return "plan: context=" + std::to_string(p.max_context) + " kv=" + std::to_string(p.kv_bytes) +
           " state=" + std::to_string(p.state_bytes) + " cache=" + std::to_string(p.cache_bytes) +
           " slots=" + std::to_string(p.cache_slots) + " used=" + std::to_string(p.vram_used) + "/" +
           std::to_string(p.vram_budget);
}

inline std::string to_string(const Result& r) {
    return std::holds_alternative<Plan>(r) ? to_string(std::get<Plan>(r)) : to_string(std::get<DoesNotClose>(r));
}

}  // namespace plan
}  // namespace baloon
