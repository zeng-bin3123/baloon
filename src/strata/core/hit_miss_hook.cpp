// strata/core/hit_miss_hook.cpp — VRAM expert tier seam implementation
#include "strata/core/hit_miss_hook.hpp"

namespace strata {

void HitMissHook::init(int32_t max_k) {
    max_k_ = max_k;
    hit_ids_.resize(static_cast<size_t>(max_k));
    hit_slots_.resize(static_cast<size_t>(max_k));
    hit_weights_.resize(static_cast<size_t>(max_k));
    miss_ids_.resize(static_cast<size_t>(max_k));
    miss_weights_.resize(static_cast<size_t>(max_k));
}

ExpertLookupResult HitMissHook::partition(int64_t layer,
                                           const int32_t* ids,
                                           const float* weights,
                                           int32_t k,
                                           const ExpertCache& cache,
                                           HitMissPartition& part) {
    ExpertLookupResult result;
    result.n_total = k;

    int32_t nh = 0, nm = 0;
    for (int32_t i = 0; i < k; ++i) {
        int32_t slot = cache.slot_of(layer, ids[i]);
        if (slot != kNotResident) {
            hit_ids_[nh]     = ids[i];
            hit_slots_[nh]   = slot;
            hit_weights_[nh] = weights[i];
            ++nh;
        } else {
            miss_ids_[nm]     = ids[i];
            miss_weights_[nm] = weights[i];
            ++nm;
        }
    }

    result.n_hits = nh;
    result.n_miss = nm;

    part.hit_ids     = hit_ids_.data();
    part.hit_slots   = hit_slots_.data();
    part.hit_weights = hit_weights_.data();
    part.miss_ids    = miss_ids_.data();
    part.miss_weights = miss_weights_.data();
    part.n_hits = nh;
    part.n_miss = nm;

    return result;
}

void HitMissHook::update_route_counts(int64_t layer, const int32_t* ids, int32_t k) {
    if (route_counts_.empty()) return;
    for (int32_t i = 0; i < k; ++i) {
        int32_t e = ids[i];
        if (e >= 0 && e < n_expert_ && layer >= 0 && layer < n_layers_) {
            size_t idx = static_cast<size_t>(layer * n_expert_ + e);
            if (route_counts_[idx] < INT32_MAX) {
                ++route_counts_[idx];
            }
        }
    }
}

void HitMissHook::reset_route_counts(int64_t n_layers, int64_t n_expert) {
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    route_counts_.assign(static_cast<size_t>(n_layers * n_expert), 0);
}

void HitMissHook::decay_route_counts() {
    for (auto& c : route_counts_) {
        c >>= 1;
    }
}

}  // namespace strata

