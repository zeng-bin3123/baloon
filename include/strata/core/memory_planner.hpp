#pragma once
// strata/core/memory_planner.hpp — VRAM budget arithmetic
//
// Given GPU memory capacity, model parameters, and target context length,
// computes how to divide VRAM among:
//   1. Dense weights (always resident)
//   2. KV cache (INT8, grows with context length)
//   3. Expert cache slots (VRAM-tier, remainder goes here)
//   4. Scratch buffers (fixed overhead for activation intermediates)
//
// The planner does not allocate — it only computes sizes.
// Allocation is done by DeviceArena.

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata {

// ─── Model parameters relevant to VRAM planning ─────────────────────
struct ModelSpec {
    int64_t n_layers          = 48;     // number of transformer layers
    int64_t n_embd            = 2560;   // embedding dimension
    int64_t n_expert          = 128;    // experts per layer
    int64_t n_expert_active   = 8;      // active experts per token
    int64_t expert_blob_bytes = 5181440; // bytes per expert blob (Q2_0)
    int64_t dense_bytes       = 0;      // total dense weight bytes (computed or set)
    int64_t head_bytes        = 0;      // output head weight bytes
    int64_t kv_heads          = 4;      // number of KV heads (GQA)
    int64_t head_dim          = 128;    // dimension per head
    int64_t hc                = 4;      // hyper-connection multiplier
};

// ─── VRAM plan ─────────────────────────────────────────────────────
struct VRAMPlan {
    // Inputs
    uint64_t vram_total        = 0;  // total VRAM available
    uint64_t vram_reserved     = 0;  // CUDA runtime overhead (safety margin)
    uint64_t vram_usable       = 0;  // vram_total - vram_reserved

    // Allocations
    uint64_t dense_bytes       = 0;  // dense weights (always-resident)
    uint64_t head_bytes        = 0;  // output head weights
    uint64_t kv_cache_bytes    = 0;  // KV cache for target context
    uint64_t scratch_bytes     = 0;  // activation scratch buffers
    uint64_t expert_vram_bytes = 0;  // VRAM expert cache (remainder)

    // Derived
    int64_t  expert_slots      = 0;  // how many expert blobs fit in expert_vram_bytes
    int64_t  context_length    = 0;  // target context length used for planning

    // Diagnostics
    double   expert_vram_frac  = 0.0; // fraction of VRAM for experts
    double   kv_frac           = 0.0; // fraction of VRAM for KV cache

    // Is the plan feasible? (non-expert allocations fit in VRAM)
    bool feasible() const {
        return dense_bytes + head_bytes + kv_cache_bytes + scratch_bytes <= vram_usable;
    }

    // Summary string for logging.
    std::string summary() const;
};

// ─── plan_vram ────────────────────────────────────────────────────────
// Compute a VRAM allocation plan.
//
// `vram_total`:      total GPU memory in bytes
// `spec`:            model parameters
// `context_length`:  target context length in tokens
// `reserved_bytes`:  CUDA runtime overhead / safety margin (default 512 MiB)
// `scratch_per_token`: scratch bytes per token (default 0 = auto-compute)
//
// The expert cache gets whatever VRAM remains after dense weights,
// head, KV cache, and scratch are subtracted.
VRAMPlan plan_vram(uint64_t vram_total,
                   const ModelSpec& spec,
                   int64_t context_length,
                   uint64_t reserved_bytes = 512ull * 1024 * 1024,
                   uint64_t scratch_per_token = 0);

// ─── kv_cache_bytes ──────────────────────────────────────────────────
// Compute KV cache size in bytes for a given context length.
// Layout: INT8, per layer, per KV head, per position, key + value.
// Formula: n_layers * kv_heads * 2 * head_dim * context_length
uint64_t kv_cache_bytes(const ModelSpec& spec, int64_t context_length);

// ─── scratch_bytes_auto ──────────────────────────────────────────────
// Auto-compute scratch buffer requirements.
// Covers: activation buffer, router logits, expert selection indices,
// temporary GEMV outputs, hyper-connection residual stack.
uint64_t scratch_bytes_auto(const ModelSpec& spec);

}  // namespace strata

