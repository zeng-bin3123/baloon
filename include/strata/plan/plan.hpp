#pragma once
// strata/plan/plan.hpp — VRAM budget planner: geometry, state/KV sizing, cache slot allocation
// Target: Iluvatar BI-V100 (ivcore10)
//
// Geometry: model dimensions for a 48-layer hybrid GDN/QSA architecture.
//   48 total layers: 36 GDN (recurrent) + 12 QSA (attention).
//
// State sizing:
//   GDN recurrence: S × h_v × S × sizeof(float) per layer
//   GDN conv history: (d_conv - 1) × C × sizeof(float) per layer
//   Total fixed cost: 36 × (3,145,728 + 122,880) = 117,669,888 bytes ≈ 112 MiB
//
// KV sizing per token:
//   KV elements:  2 × n_kv_heads × head_dim = 1024
//   KV scales:    kv_elems / kv_group × 2 = 32 (fp16 scale per group)
//   Per QSA layer: kv_elems × 1 + kv_scales = 1056 bytes (INT8)
//   Indexer key:   key_heads × key_dim / qsa_block = 32 bytes
//   Total: 12 × (1056 + 32) = 13,056 bytes/token
//
// Plan: partitions VRAM pool into fixed costs + KV + expert cache.
//   DoesNotClose thrown if KV + fixed > pool, or cache_slots == 0.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata {

// ─── PlanGeometry ────────────────────────────────────────────────────
// Model dimensions for the hybrid GDN/QSA architecture.
//
// 48 total layers, split:
//   36 GDN layers (state-space / recurrent)
//   12 QSA layers (sparse attention with indexer)
//
// Indexer key sizing (corrected twice during design):
//   Final: 1 × 128 / 4 = 32 bytes/token/layer
//   (NOT 1×256/4=64, NOT 4×128/4=128)
struct PlanGeometry {
    int n_layers         = 48;
    int n_qsa_layers     = 12;
    int n_kv_heads       = 2;
    int head_dim         = 256;
    int kv_group         = 64;
    int qsa_block        = 4;
    int indexer_q_heads   = 4;
    int indexer_key_heads = 1;
    int indexer_key_dim   = 128;
    int ssm_state_size   = 128;
    int ssm_v_heads      = 48;
    int ssm_d_conv       = 4;
    int ssm_conv_channels = 10240;

    // Number of GDN (recurrent) layers = n_layers - n_qsa_layers.
    int n_gdn_layers() const { return n_layers - n_qsa_layers; }
};

// ─── Costs ───────────────────────────────────────────────────────────
// Pre-computed byte costs for non-cache, non-KV VRAM residents.
// Populated externally (by weight loader or config parser).
struct PlanCosts {
    uint64_t dense_bytes     = 0;   // dense (non-expert) weight arena
    uint64_t embd_bytes      = 0;   // embedding table
    uint64_t mtp_bytes       = 0;   // MTP drafter VRAM
    uint64_t workspace_bytes = 0;   // scratch / activation workspace
    uint64_t state_bytes     = 0;   // GDN recurrence state (fixed, non-evictable)
    uint64_t expert_blob     = 0;   // bytes per expert cache slot (1,382,400)
};

// ─── state_bytes ─────────────────────────────────────────────────────
// Total bytes for GDN recurrence state across all GDN layers.
//
// Per GDN layer:
//   recurrence = S × h_v × S × 4  = 128 × 48 × 128 × 4 = 3,145,728 B
//   conv_hist  = (d_conv-1) × C × 4 = 3 × 10240 × 4     =   122,880 B
//   per_layer  = 3,268,608 B
//
// Total = 36 × 3,268,608 = 117,669,888 B ≈ 112.2 MiB
//
// This is a FIXED cost, NOT evictable.
inline uint64_t state_bytes(const PlanGeometry& g) {
    const uint64_t per_layer =
        static_cast<uint64_t>(g.ssm_state_size) *
        static_cast<uint64_t>(g.ssm_v_heads) *
        static_cast<uint64_t>(g.ssm_state_size) * sizeof(float)
        +
        static_cast<uint64_t>(g.ssm_d_conv - 1) *
        static_cast<uint64_t>(g.ssm_conv_channels) * sizeof(float);
    return static_cast<uint64_t>(g.n_gdn_layers()) * per_layer;
}

// ─── kv_bytes_per_token ──────────────────────────────────────────────
// Bytes consumed in VRAM per token of KV cache across all QSA layers.
//
//   kv_elems   = 2 × n_kv_heads × head_dim = 1024
//   kv_scales  = (kv_elems / kv_group) × 2  =   32  (fp16 scale per group)
//   per_layer  = kv_elems × 1 + kv_scales   = 1056  (INT8: 1 byte/elem)
//   idx_layer  = key_heads × key_dim / qsa_block = 32
//   total      = n_qsa × (per_layer + idx_layer) = 12 × 1088 = 13,056 B/token
inline uint64_t kv_bytes_per_token(const PlanGeometry& g) {
    const uint64_t kv_elems =
        2ull * static_cast<uint64_t>(g.n_kv_heads) *
        static_cast<uint64_t>(g.head_dim);
    const uint64_t kv_scales =
        (kv_elems / static_cast<uint64_t>(g.kv_group)) * 2;
    const uint64_t per_layer = kv_elems * 1 + kv_scales;
    const uint64_t idx_layer =
        static_cast<uint64_t>(g.indexer_key_heads) *
        static_cast<uint64_t>(g.indexer_key_dim) /
        static_cast<uint64_t>(g.qsa_block);
    return static_cast<uint64_t>(g.n_qsa_layers) * (per_layer + idx_layer);
}

// ─── Plan ────────────────────────────────────────────────────────────
// Result of VRAM budget allocation.
struct Plan {
    uint64_t max_context  = 0;   // max tokens in KV cache
    uint64_t kv_bytes     = 0;   // total KV bytes = kv_bytes_per_token × max_context
    uint64_t state_bytes  = 0;   // GDN recurrence state (fixed)
    uint64_t cache_bytes  = 0;   // remaining VRAM for expert cache
    uint64_t cache_slots  = 0;   // floor(cache_bytes / expert_blob)
    uint64_t vram_budget  = 0;   // total pool size
    uint64_t vram_used    = 0;   // actual bytes used (budget minus wasted remainder)
    uint64_t dram_experts = 0;   // bytes of expert weights kept in DRAM
    std::vector<std::string> notes;
};

// ─── DoesNotClose ────────────────────────────────────────────────────
// Thrown when the VRAM budget cannot fit even the minimum requirements.
// Fail at startup, NOT at token 4000.
struct DoesNotClose : std::runtime_error {
    explicit DoesNotClose(const std::string& reason)
        : std::runtime_error(reason) {}
};

// ─── vram_pool_bytes ─────────────────────────────────────────────────
// Default VRAM pool for a single Iluvatar BI-V100.
// 5,943,000,000 B ≈ 5.53 GiB of usable pool after driver/firmware overhead.
inline uint64_t vram_pool_bytes() { return 5943000000ull; }

// ─── make_plan ───────────────────────────────────────────────────────
// Partition the VRAM pool into fixed costs, KV cache, and expert cache.
//
// Layout:
//   pool = dense + embd + mtp + workspace + state + KV + cache
//
// If KV + fixed > pool → DoesNotClose (startup failure)
// If cache_slots == 0  → DoesNotClose (no experts can be cached)
//
// Returns a fully-populated Plan on success.
inline Plan make_plan(uint64_t max_context,
                      const PlanGeometry& g,
                      const PlanCosts& c,
                      uint64_t pool = vram_pool_bytes()) {
    Plan p;
    p.max_context = max_context;
    p.vram_budget = pool;
    p.kv_bytes    = kv_bytes_per_token(g) * max_context;
    p.state_bytes = c.state_bytes;
    p.dram_experts = c.dense_bytes + c.embd_bytes;

    const uint64_t fixed = c.dense_bytes + c.embd_bytes +
                           c.mtp_bytes + c.workspace_bytes +
                           c.state_bytes;

    if (p.kv_bytes + fixed > pool) {
        throw DoesNotClose(
            "VRAM budget exceeded: KV(" + std::to_string(p.kv_bytes) +
            ") + fixed(" + std::to_string(fixed) +
            ") = " + std::to_string(p.kv_bytes + fixed) +
            " > pool(" + std::to_string(pool) + ")");
    }

    p.cache_bytes = pool - p.kv_bytes - fixed;
    p.cache_slots = p.cache_bytes / c.expert_blob;
    p.vram_used   = pool - (p.cache_bytes % c.expert_blob);

    if (p.cache_slots == 0) {
        throw DoesNotClose(
            "Zero expert cache slots: cache_bytes=" +
            std::to_string(p.cache_bytes) +
            " < expert_blob=" + std::to_string(c.expert_blob));
    }

    return p;
}

}  // namespace strata

