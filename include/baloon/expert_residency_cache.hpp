// expert_residency_cache.hpp - VRAM expert tier residency table: bounds-checked (layer,expert)->slot lookup,
// never-evicting admission (global or per-layer ranges), 256-byte aligned variable slots, fill clamp, profile reader.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace baloon {
namespace residency {

constexpr int64_t kNone = -1;

// Slot range [lo,hi) layer `layer` may admit into. Invalid geometry or layer: empty [0,0).
inline std::pair<uint64_t, uint64_t> layer_range(uint64_t slots, uint64_t nl, uint64_t layer) {
    if (nl == 0 || slots == 0 || nl <= layer) return {0, 0};
    const uint64_t q = slots / nl;
    return {layer * q, layer == nl - 1 ? slots : (layer + 1) * q};
}

// Admission step: hands out `next` while below the ceiling `hi`; never decreases the counter.
inline bool admit_slot(uint64_t hi, uint64_t next, uint64_t& slot) {
    if (hi <= next) return false;
    slot = next;
    return true;
}
inline uint64_t admit_next(uint64_t hi, uint64_t next) { return hi <= next ? next : next + 1; }

// (layer,expert) -> slot, checked against both axes. `res` is the flat l*ne+e table (kNone = not resident).
inline int64_t slot_of(const std::vector<int64_t>& res, uint64_t nl, uint64_t ne, uint64_t l, uint64_t e) {
    if (!(l < nl && e < ne)) return kNone;
    const uint64_t i = l * ne + e;
    return i < res.size() ? res[i] : kNone;
}

constexpr uint64_t align256(uint64_t n) { return (n + 255) / 256 * 256; }
inline uint64_t off_at(const std::vector<uint64_t>& sizes, uint64_t i) {
    uint64_t t = 0;
    for (uint64_t k = 0; k < i; ++k) t += align256(k < sizes.size() ? sizes[k] : 0);
    return t;
}
// Bytes a fill copies: `bytes` if in (0, blob], else the whole slot.
constexpr uint64_t fill_bytes(uint64_t blob, uint64_t bytes) { return (0 < bytes && bytes <= blob) ? bytes : blob; }

// read_expert_profile acceptance rule.
inline bool read_profile(uint64_t nl, uint64_t ne, uint64_t hdr_nl, uint64_t hdr_ne, uint64_t want,
                         uint64_t n_ranked, const std::vector<std::pair<uint64_t, uint64_t>>& pairs,
                         std::vector<std::pair<uint64_t, uint64_t>>& out) {
    if (hdr_nl != nl || hdr_ne != ne || n_ranked > want || pairs.size() != n_ranked) return false;
    for (const auto& p : pairs)
        if (!(p.first < nl && p.second < ne)) return false;
    out = pairs;
    return true;
}

// Residency table with no-eviction admission.
class Table {
public:
    Table(uint64_t nl, uint64_t ne, uint64_t slots, bool per_layer)
        : nl_(nl), ne_(ne), slots_(slots), per_layer_(per_layer), res_(nl * ne, kNone), layer_next_(nl, 0) {
        for (uint64_t l = 0; l < nl; ++l) layer_next_[l] = layer_range(slots, nl, l).first;
    }
    int64_t slot(uint64_t l, uint64_t e) const { return slot_of(res_, nl_, ne_, l, e); }
    // Admit (l,e): returns its slot (existing or new) or kNone if invalid/full. Never evicts.
    int64_t admit(uint64_t l, uint64_t e) {
        if (!(l < nl_ && e < ne_)) return kNone;
        const int64_t have = slot(l, e);
        if (have != kNone) return have;
        uint64_t s = 0;
        if (per_layer_) {
            const uint64_t hi = layer_range(slots_, nl_, l).second;
            if (!admit_slot(hi, layer_next_[l], s)) return kNone;
            layer_next_[l] = admit_next(hi, layer_next_[l]);
        } else {
            if (!admit_slot(slots_, next_free_, s)) return kNone;
            next_free_ = admit_next(slots_, next_free_);
        }
        res_[l * ne_ + e] = (int64_t)s;
        return (int64_t)s;
    }
    uint64_t next_free() const { return next_free_; }
    uint64_t layer_next(uint64_t l) const { return layer_next_[l]; }
    uint64_t slots() const { return slots_; }
private:
    uint64_t nl_, ne_, slots_;
    bool per_layer_;
    std::vector<int64_t> res_;
    std::vector<uint64_t> layer_next_;
    uint64_t next_free_ = 0;
};

}  // namespace residency
}  // namespace baloon
