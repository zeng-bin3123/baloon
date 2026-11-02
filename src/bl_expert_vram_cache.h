// bl_expert_vram_cache.h - VRAM-resident expert tier: slot arena, residency table,
// per-layer slot quotas, 256-byte aligned sized slots, bounded slot fill.
#pragma once
#include <cstdint>
#include <optional>
#include <vector>
#include "bl_nat.h"

namespace baloon::expert_vram_cache {

// Per-layer ranges [layerLo, layerHi): quota q = slots / nl each; the last layer takes the remainder.
constexpr nat layerLo(nat slots, nat nl, nat l) { return l * ndiv(slots, nl); }
constexpr nat layerHi(nat slots, nat nl, nat l) {
  return (nl > 0 && l == nl - 1) ? slots : (l + 1) * ndiv(slots, nl);
}

// Admission never evicts: a counter hands out slots below its ceiling and refuses (without moving) at it.
constexpr std::optional<nat> admitSlot(nat next, nat hi) {
  if (next >= hi) return std::nullopt;
  return next;
}
constexpr nat admitNext(nat next, nat hi) { return next >= hi ? next : next + 1; }

constexpr nat align256(nat n) { return (n + 255) / 256 * 256; }

// End offset of a run of sized slots laid out back to back from `start`, each rounded to 256 bytes.
inline nat offsetsFrom(nat start, const std::vector<nat>& blobs) {
  for (nat b : blobs) start += align256(b);
  return start;
}
// Start offset of every slot (same layout as offsetsFrom).
inline std::vector<nat> slotOffsets(nat start, const std::vector<nat>& blobs) {
  std::vector<nat> out;
  out.reserve(blobs.size());
  for (nat b : blobs) { out.push_back(start); start += align256(b); }
  return out;
}

// A fill copies the requested byte count when it is positive and fits one slot, else one whole slot.
constexpr nat fillBytes(nat bytes, nat blob) { return (0 < bytes && bytes <= blob) ? bytes : blob; }

// Concrete cache: residency table (layer, expert) -> slot, per-layer or global slot counters.
class ExpertVramCache {
 public:
  static constexpr std::int64_t kNone = -1;
  ExpertVramCache(nat slots, nat nLayers, nat nExpert, bool perLayer)
      : slots_(slots), nl_(nLayers), ne_(nExpert), perLayer_(perLayer),
        res_(nLayers * nExpert, kNone), next_(perLayer ? nLayers : 1, 0) {
    if (perLayer_) for (nat l = 0; l < nl_; ++l) next_[l] = layerLo(slots_, nl_, l);
  }
  // Returns the slot of (l, e) or kNone. Out-of-range keys are never resident.
  std::int64_t lookup(nat l, nat e) const {
    if (l >= nl_ || e >= ne_) return kNone;
    return res_[l * ne_ + e];
  }
  // Admit (l, e): an already-resident key keeps its slot; a full range refuses; out-of-range refuses.
  std::int64_t admit(nat l, nat e) {
    if (l >= nl_ || e >= ne_) return kNone;
    std::int64_t& r = res_[l * ne_ + e];
    if (r != kNone) return r;
    nat& nx = perLayer_ ? next_[l] : next_[0];
    nat hi = perLayer_ ? layerHi(slots_, nl_, l) : slots_;
    auto s = admitSlot(nx, hi);
    if (!s) return kNone;
    nx = admitNext(nx, hi);
    r = static_cast<std::int64_t>(*s);
    return r;
  }
  nat slots() const { return slots_; }
 private:
  nat slots_, nl_, ne_;
  bool perLayer_;
  std::vector<std::int64_t> res_;
  std::vector<nat> next_;
};

}  // namespace baloon::expert_vram_cache
