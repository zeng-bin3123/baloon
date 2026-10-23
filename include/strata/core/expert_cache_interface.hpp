#pragma once
// strata/core/expert_cache_interface.hpp — VRAM expert cache API contract
// Target: Iluvatar BI-V100 (ivcore10)
//
// Defines the interface for batch residency lookup and hit/miss partitioning
// used by the MoE layer compositor.  The ExpertCache implements this contract;
// the hit_miss_hook uses it to split routed experts into GPU-resident hits
// and CPU-pool misses.

#include <cstdint>

namespace strata {

// Result of a batch residency lookup for one layer's routed experts.
struct ExpertLookupResult {
    int32_t n_hits  = 0;          // experts found in VRAM
    int32_t n_miss  = 0;          // experts not in VRAM (need CPU pool)
    int32_t n_total = 0;          // n_hits + n_miss == k (top-k routed)
};

// Batch lookup: given k routed expert ids for a layer, fill slot_out with
// the VRAM slot index for each hit or kNotResident (-1) for each miss.
// Returns the ExpertLookupResult summary.
//
// Contract:
//   - ids[0..k) are valid expert indices in [0, n_expert)
//   - slot_out[0..k) is caller-owned, written by the lookup
//   - The residency table is read-only during lookup (no admission)
//
// This is the "read path" — no CUDA calls, no allocation, pure table lookup.
// The layer compositor calls this once per token per MoE layer.

// Hit/miss partition: given k routed expert ids and their slot lookups,
// partition into hits (VRAM-resident) and misses (CPU-pool).
// hit_ids/hit_slots/hit_weights: dense arrays of the resident experts
// miss_ids/miss_weights: dense arrays of the non-resident experts
//
// The partition preserves the original routing weight for weighted combination.
struct HitMissPartition {
    // Hit side (GPU-resident experts)
    int32_t* hit_ids    = nullptr;   // [n_hits] expert ids
    int32_t* hit_slots  = nullptr;   // [n_hits] VRAM slot indices
    float*   hit_weights = nullptr;  // [n_hits] routing weights

    // Miss side (CPU-pool experts)
    int32_t* miss_ids    = nullptr;  // [n_miss] expert ids
    float*   miss_weights = nullptr; // [n_miss] routing weights

    int32_t n_hits = 0;
    int32_t n_miss = 0;
};

}  // namespace strata

