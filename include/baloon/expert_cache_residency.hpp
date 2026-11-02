// VRAM-resident expert tier: slot arena bookkeeping plus a (layer, expert) -> slot residency table,
// with never-evicting admission (global arrival order or per-layer slot ranges).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace baloon::expert_cache {

inline constexpr uint32_t kNotResident = UINT32_MAX;

// ---- per-layer slot ranges ---------------------------------------------------

// q = slots / n_layers (0 when there are no layers).
inline uint64_t layer_quota(uint64_t slots, uint64_t n) { return n ? slots / n : 0; }

// Layer l owns [layer_lo, layer_hi); the last layer takes the remainder.
inline uint64_t layer_lo(uint64_t slots, uint64_t n, uint64_t l) { return l * layer_quota(slots, n); }
inline uint64_t layer_hi(uint64_t slots, uint64_t n, uint64_t l) {
    return (l + 1 == n) ? slots : (l + 1) * layer_quota(slots, n);
}

// ---- admission step (bump allocator with a ceiling) --------------------------

struct AdmitStep {
    std::optional<uint64_t> slot;
    uint64_t next;
};
inline AdmitStep admit_step(uint64_t next, uint64_t hi) {
    if (next >= hi) return {std::nullopt, next};
    return {next, next + 1};
}

// ---- flat residency index ----------------------------------------------------

// layer * n_expert + expert, only for an in-range layer AND expert.
inline std::optional<uint64_t> flat_index(uint64_t n_layers, uint64_t n_expert, uint64_t l, uint64_t e) {
    if (l >= n_layers || e >= n_expert) return std::nullopt;
    return l * n_expert + e;
}

// ---- slot alignment, sizes and fills -----------------------------------------

inline uint64_t align256(uint64_t b) { return (b + 255) / 256 * 256; }

// Sum of aligned slot sizes (the final offset).
inline uint64_t total(const std::vector<uint64_t>& sizes) {
    uint64_t t = 0;
    for (uint64_t b : sizes) t += align256(b);
    return t;
}

// Offsets (size + 1 entries, starting at 0) of back-to-back slots whose sizes are 256-aligned.
inline std::vector<uint64_t> aligned_offsets(const std::vector<uint64_t>& raw_sizes) {
    std::vector<uint64_t> offs{0};
    for (uint64_t b : raw_sizes) offs.push_back(offs.back() + align256(b));
    return offs;
}

// Prefix sums of the sizes exactly as given.
inline std::vector<uint64_t> slot_offsets(const std::vector<uint64_t>& sizes) {
    std::vector<uint64_t> offs{0};
    for (uint64_t b : sizes) offs.push_back(offs.back() + b);
    return offs;
}

// Arena bytes: the last offset for a sized arena, else slots * blob (uniform).
inline uint64_t arena_bytes(const std::vector<uint64_t>& offs, uint64_t slots, uint64_t blob) {
    return offs.empty() ? slots * blob : offs.back();
}

// A fill copies `bytes` when 0 < bytes <= blob, else the whole blob.
inline uint64_t fill_len(uint64_t bytes, uint64_t blob) { return (bytes > 0 && bytes <= blob) ? bytes : blob; }

// Resident count: per-layer admitted total when per-layer admission is on, else the global counter.
inline uint64_t resident_count(bool per_layer, uint64_t admitted, uint64_t next_free) {
    return per_layer ? admitted : next_free;
}

// ---- profile validity --------------------------------------------------------

inline bool valid_profile(const std::vector<std::pair<uint64_t, uint64_t>>& ranked, uint64_t n_layers,
                          uint64_t n_expert) {
    for (const auto& p : ranked)
        if (!(p.first < n_layers && p.second < n_expert)) return false;
    return true;
}

// ---- the cache ---------------------------------------------------------------

class ExpertCache {
public:
    ExpertCache(uint32_t slots, uint32_t n_layers, uint32_t n_expert)
        : slots_(slots), n_layers_(n_layers), n_expert_(n_expert),
          res_(static_cast<size_t>(n_layers) * n_expert, kNotResident), layer_next_(n_layers) {
        for (uint32_t l = 0; l < n_layers; ++l) layer_next_[l] = static_cast<uint32_t>(layer_lo(slots, n_layers, l));
    }

    uint32_t slots() const { return slots_; }
    uint32_t next_free() const { return next_free_; }
    uint32_t admitted_per_layer_total() const { return per_layer_admitted_; }

    // Bounds-checked residency lookup: kNotResident for out-of-range layer/expert or non-resident.
    uint32_t slot_of(uint64_t l, uint64_t e) const {
        auto i = flat_index(n_layers_, n_expert_, l, e);
        return i ? res_[*i] : kNotResident;
    }

    // Global admission: next free slot from one counter; refuses when full; never evicts.
    std::optional<uint32_t> admit(uint64_t l, uint64_t e) {
        auto i = flat_index(n_layers_, n_expert_, l, e);
        if (!i || next_free_ >= slots_) return std::nullopt;
        uint32_t s = next_free_++;
        res_[*i] = s;
        return s;
    }

    // Per-layer admission: layer l hands out slots only from [layer_lo, layer_hi); never evicts.
    std::optional<uint32_t> admit_per_layer(uint64_t l, uint64_t e) {
        auto i = flat_index(n_layers_, n_expert_, l, e);
        if (!i) return std::nullopt;
        AdmitStep st = admit_step(layer_next_[l], layer_hi(slots_, n_layers_, l));
        if (!st.slot) return std::nullopt;
        layer_next_[l] = static_cast<uint32_t>(st.next);
        res_[*i] = static_cast<uint32_t>(*st.slot);
        ++per_layer_admitted_;
        return static_cast<uint32_t>(*st.slot);
    }

private:
    uint32_t slots_, n_layers_, n_expert_;
    uint32_t next_free_ = 0;
    uint32_t per_layer_admitted_ = 0;
    std::vector<uint32_t> res_;
    std::vector<uint32_t> layer_next_;
};

}  // namespace baloon::expert_cache

