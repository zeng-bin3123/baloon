// bl_expert_dispatch_split.h - hit/miss split of one layer's routed experts between GPU and CPU pool.
#pragma once
#include <vector>
#include "bl_nat.h"

namespace baloon::expert_dispatch_split {

struct Split {
  std::vector<nat> hitRows;   // output rows computed on GPU (CPU row zeroed)
  std::vector<nat> missRows;  // output rows that became CPU jobs; job j serves output row missRows[j]
  nat missCount() const { return missRows.size(); }
};

// Every routed index lands in exactly one of hitRows / missRows, both in routed order.
template <class IsHit>
Split splitRouted(const std::vector<nat>& ids, IsHit hit) {
  Split s;
  for (nat i = 0; i < ids.size(); ++i) (hit(ids[i]) ? s.hitRows : s.missRows).push_back(i);
  return s;
}

// PCIe share of the missed experts: (nmiss * pcieNum) >> 8 (never more than nmiss for pcieNum <= 256).
constexpr nat pcieShare(nat nmiss, nat pcieNum) { return nmiss * pcieNum / 256; }
// Miss ranks handed to PCIe are [nmiss - m, nmiss).
constexpr nat pcieRankFirst(nat nmiss, nat m) { return sat_sub(nmiss, m); }

}  // namespace baloon::expert_dispatch_split
