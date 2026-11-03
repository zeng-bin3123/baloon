// expert_cache_residency.hpp - VRAM expert-cache slot arithmetic: per-layer slot ranges, no-eviction admission,
// slot addressing, 256-byte aligned variable slots, VRAM check and the fill-size clamp.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace expert_cache {

constexpr uint64_t align256(uint64_t b) { return (b + 255) / 256 * 256; }

// Sum of aligned slot sizes (the single allocation of a variable-slot cache).
inline uint64_t total(const std::vector<uint64_t>& sizes) {
    uint64_t t = 0;
    for (uint64_t b : sizes) t += align256(b);
    return t;
}

// Offset of slot i: sum of the aligned sizes before it (beyond the list: the total).
inline uint64_t off(const std::vector<uint64_t>& sizes, uint64_t i) {
    uint64_t t = 0;
    for (uint64_t k = 0; k < i && k < sizes.size(); ++k) t += align256(sizes[k]);
    return t;
}

// Half-open slot range of layer l; the last layer takes the remainder. Out-of-range layer: empty [0,0).
constexpr uint64_t slot_lo(uint64_t nl, uint64_t s, uint64_t l) {
    return (nl == 0 || s == 0 || nl <= l) ? 0 : l * (s / nl);
}
constexpr uint64_t slot_hi(uint64_t nl, uint64_t s, uint64_t l) {
    return (nl == 0 || s == 0 || nl <= l) ? 0 : (l == nl - 1 ? s : (l + 1) * (s / nl));
}

// No-eviction admission step. Returns true and fills slot/next_out when a slot is granted.
constexpr bool admit_step(uint64_t next, uint64_t hi, uint64_t& slot, uint64_t& next_out) {
    if (hi <= next) return false;
    slot = next;
    next_out = next + 1;
    return true;
}

// Residency table index layer*ne + expert (valid coordinates only).
constexpr uint64_t residency_idx(uint64_t ne, uint64_t layer, uint64_t expert) { return layer * ne + expert; }

// Byte offset of a device slot; defined only for slot < slots.
constexpr bool device_slot_offset(uint64_t slots, uint64_t blob, uint64_t slot, uint64_t& out) {
    if (slot >= slots) return false;
    out = slot * blob;
    return true;
}

// VRAM check: a request larger than free is refused.
constexpr bool vram_ok(uint64_t free_bytes, uint64_t want) { return !(free_bytes < want); }

// Copy size of fill_slot: the request if in (0, blob], otherwise the whole blob.
constexpr int64_t fill_bytes(int64_t bytes, int64_t blob) { return (0 < bytes && bytes <= blob) ? bytes : blob; }

}  // namespace expert_cache
}  // namespace baloon
