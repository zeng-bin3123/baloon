// bl_qsa_cache_indexer_attention.h - QSA step state, exact top-k by threshold search, pooled-block completion,
// GQA head mapping and paged-cache position decomposition.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::qsa {

// width = min(n_kv, top_k + r - 1)
constexpr nat selWidth(nat nKv, nat topK, nat r) { return nmin(nKv, topK + r - 1); }
struct Step { nat nKv, nBid, width; };
// Step state derived from one position: n_kv = pos+1, n_bid = n_kv / r, width = selWidth.
constexpr Step stepOf(nat pos, nat topK, nat r) { return {pos + 1, ndiv(pos + 1, r), selWidth(pos + 1, topK, r)}; }
constexpr bool poolCompletes(nat pos, nat r) { return r > 0 && pos % r == r - 1; }
constexpr nat kvHead(nat h, nat nHead, nat nHeadKv) { return ndiv(h, ndiv(nHead, nHeadKv)); }
constexpr nat pageOf(nat pos, nat ps) { return ndiv(pos, ps); }
constexpr nat slotOf(nat pos, nat ps) { return nmod(pos, ps); }
constexpr bool gatherAligned(nat hd) { return hd % 4 == 0; }

// Exact top-k by threshold: find the key (score) of rank `width` by search, take every cell strictly above it plus
// the first (width - c_gt) equal cells by ascending index. Returns the selected cell indices in ascending order.
inline std::vector<nat> thresholdSelect(const std::vector<float>& s, nat width) {
  std::vector<nat> out;
  if (width == 0 || s.empty()) return out;
  width = nmin(width, s.size());
  std::vector<float> sorted(s);
  std::sort(sorted.begin(), sorted.end(), [](float a, float b) { return a > b; });
  float thr = sorted[width - 1];
  nat cGt = 0;
  for (float v : s) cGt += v > thr ? 1 : 0;
  nat budget = width - cGt;
  for (nat i = 0; i < s.size(); ++i) {
    if (s[i] > thr) out.push_back(i);
    else if (s[i] == thr && budget > 0) { out.push_back(i); --budget; }
  }
  return out;
}

}  // namespace baloon::qsa
