#pragma once
// strata/core/hit_miss_hook.hpp — VRAM expert tier seam
// Target: Iluvatar BI-V100 (ivcore10)
//
// The hit/miss hook sits between the MoE router output and the two expert
// compute tiers (GPU grouped kernel for hits, CPU pool for misses).
// It queries the ExpertCache residency table and partitions the routed
// expert set into VRAM-resident hits and CPU-pool misses.
//
// This is the seam that reduced CPU pool drain from 19.1 → 10.3 ms/token
// at 4096 per-layer slots (54.4% hit rate).

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_cache_interface.hpp"
#include <cstdint>
#include <vector>

namespace strata {

// HitMissHook: stateful adapter between router output and expert compute tiers.
// Owns scratch buffers for the hit/miss partition arrays.
class HitMissHook {
public:
    HitMissHook() = default;

    // Initialize for a given top-k (max routed experts per token).
    // Allocates partition scratch buffers.
    void init(int32_t max_k);

    // Partition routed experts into hits and misses.
    //
    // Input:
    //   layer       — current layer index
    //   ids[0..k)   — routed expert ids from the router
    //   weights[0..k) — routing weights (renormalized softmax)
    //   k           — number of routed experts
    //   cache       — the VRAM expert cache (read-only lookup)
    //
    // Output:
    //   partition   — filled with hit/miss arrays pointing into scratch_
    //   result      — hit/miss counts
    //
    // No CUDA calls.  Pure CPU table lookup + array partition.
    ExpertLookupResult partition(int64_t layer,
                                const int32_t* ids,
                                const float* weights,
                                int32_t k,
                                const ExpertCache& cache,
                                HitMissPartition& partition);

    // Update routing frequency counters for adaptive swap.
    // Called after partition() with the same ids.
    void update_route_counts(int64_t layer, const int32_t* ids, int32_t k);

    // Access route counts (size = n_layers * n_expert).
    const std::vector<int32_t>& route_counts() const { return route_counts_; }
    void reset_route_counts(int64_t n_layers, int64_t n_expert);

    // Decay all route counts by right-shifting (for adaptive swap).
    void decay_route_counts();

    int32_t max_k() const { return max_k_; }

private:
    int32_t max_k_ = 0;

    // Scratch buffers for partition output (owned, reused per call)
    std::vector<int32_t> hit_ids_;
    std::vector<int32_t> hit_slots_;
    std::vector<float>   hit_weights_;
    std::vector<int32_t> miss_ids_;
    std::vector<float>   miss_weights_;

    // Routing frequency counters for adaptive swap
    std::vector<int32_t> route_counts_;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
};

}  // namespace strata

