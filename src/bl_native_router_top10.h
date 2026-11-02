// bl_native_router_top10.h - single-row 512-expert softmax / top-10 / renormalise router. One 32-lane warp holds
// 16 experts per lane: expert e lives in lane e % 32, register e / 32. Ties break by lowest index.
#pragma once
#include <cmath>
#include <cstdint>
#include "bl_nat.h"

namespace baloon::native_router {

constexpr nat EXPERTS = 512, LANES = 32, REGS = 16, TOPK = 10;
constexpr float SUM_FLOOR = 6.103515625e-05f;  // 2^-14

constexpr bool overlap(nat a, nat an, nat b, nat bn) { return a < b + bn && b < a + an; }
constexpr nat expertOf(nat lane, nat i) { return lane + i * LANES; }
constexpr nat laneOf(nat e) { return e % LANES; }
constexpr nat slotOf(nat e) { return e / LANES; }
// Strict preference: larger value, then smaller index.
constexpr bool better(float v, nat i, float v2, nat i2) { return v > v2 || (v == v2 && i < i2); }
inline float clampSum(float s) { return s > SUM_FLOOR ? s : SUM_FLOOR; }

// Warp-structured reference: softmax over the row, then TOPK rounds of (per-lane best -> xor-shuffle tree),
// the winning lane masks its expert; the selected sum (clamped below by 2^-14) renormalises; output rank r is
// read from lane r.
inline void routerTop10(const float* logits, nat* ids, float* weights) {
  float p[EXPERTS];
  float m = logits[0];
  for (nat e = 1; e < EXPERTS; ++e) m = logits[e] > m ? logits[e] : m;
  float sum = 0;
  for (nat e = 0; e < EXPERTS; ++e) { p[e] = std::exp(logits[e] - m); sum += p[e]; }
  for (nat e = 0; e < EXPERTS; ++e) p[e] /= sum;
  bool taken[EXPERTS] = {};
  float selSum = 0;
  float laneWeight[LANES] = {};
  nat laneId[LANES] = {};
  for (nat r = 0; r < TOPK; ++r) {
    float bv[LANES];
    nat bi[LANES];
    for (nat l = 0; l < LANES; ++l) {
      bv[l] = -1.f; bi[l] = EXPERTS;
      for (nat i = 0; i < REGS; ++i) {
        nat e = expertOf(l, i);
        if (!taken[e] && better(p[e], e, bv[l], bi[l])) { bv[l] = p[e]; bi[l] = e; }
      }
    }
    for (nat off = LANES / 2; off > 0; off /= 2) {
      float nv[LANES];
      nat ni[LANES];
      for (nat l = 0; l < LANES; ++l) {
        nat o = l ^ off;
        bool takeO = better(bv[o], bi[o], bv[l], bi[l]);
        nv[l] = takeO ? bv[o] : bv[l];
        ni[l] = takeO ? bi[o] : bi[l];
      }
      for (nat l = 0; l < LANES; ++l) { bv[l] = nv[l]; bi[l] = ni[l]; }
    }
    taken[bi[0]] = true;
    selSum += bv[0];
    laneWeight[r] = bv[0];
    laneId[r] = bi[0];
  }
  float denom = clampSum(selSum);
  for (nat r = 0; r < TOPK; ++r) { ids[r] = laneId[r]; weights[r] = laneWeight[r] / denom; }
}

}  // namespace baloon::native_router
