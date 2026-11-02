// bl_native_gdn_recurrence.h - thread mapping and span contracts of the warp-per-(head, column) gated delta net
// step: lane l owns rows r*32+l (r<4); columns col = z*4 + y; q/k head is head % h_k.
#pragma once
#include <cmath>
#include <cstdint>
#include "bl_nat.h"

namespace baloon::native_gdn {

constexpr nat S = 128, LANES = 32, REGS = 4, COLS_PER_BLOCK = 4, MAX_GRID_X = 65535;

constexpr nat column(nat z, nat y) { return z * COLS_PER_BLOCK + y; }
constexpr nat laneRow(nat r, nat l) { return r * LANES + l; }
constexpr nat stateElem(nat i, nat head, nat col, nat hv) { return (i * hv + head) * S + col; }
constexpr nat qkHead(nat head, nat hk) { return nmod(head, hk); }
constexpr nat qkElem(nat head, nat i, nat hk) { return qkHead(head, hk) * S + i; }
constexpr nat voElem(nat head, nat col) { return head * S + col; }

// Shape validity (checked before spans): S = 128, h_k > 0, 0 < h_v <= 65535, h_v % h_k = 0.
constexpr bool validShape(nat s, nat hk, nat hv) { return s == 128 && hk > 0 && hv > 0 && hv <= MAX_GRID_X && hv % hk == 0; }
constexpr nat stateBytes(nat hv) { return 128 * 128 * hv * 4; }
constexpr nat qkBytes(nat hk) { return 128 * hk * 4; }
constexpr nat voBytes(nat hv) { return 128 * hv * 4; }
constexpr nat headBytes(nat hv) { return hv * 4; }
// Half-open span overlap.
constexpr bool overlaps(nat a, nat an, nat b, nat bn) { return a < b + bn && b < a + an; }

// Reference step of one (head, column): state column s[0..127] (decayed by exp(g)), delta rule, output scaled
// by 1/sqrt(S). The K^T S and Q^T S contractions are reduced as 4 per-lane partials then a 32-lane xor tree,
// exactly the warp's data flow.
inline float warpReduce(const float* parts32) {
  float v[LANES];
  for (nat l = 0; l < LANES; ++l) v[l] = parts32[l];
  for (nat off = LANES / 2; off > 0; off /= 2)
    for (nat l = 0; l < LANES; ++l) { if ((l ^ off) > l) { float t = v[l] + v[l ^ off]; v[l] = t; v[l ^ off] = t; } }
  return v[0];
}
inline float stepColumn(float* s, const float* q, const float* k, float v, float g, float beta) {
  float decay = std::exp(g);
  float kp[LANES], qp[LANES];
  for (nat l = 0; l < LANES; ++l) {
    kp[l] = 0; qp[l] = 0;
    for (nat r = 0; r < REGS; ++r) { s[laneRow(r, l)] *= decay; kp[l] += k[laneRow(r, l)] * s[laneRow(r, l)]; }
  }
  float kv = warpReduce(kp);
  float delta = (v - kv) * beta;
  for (nat l = 0; l < LANES; ++l)
    for (nat r = 0; r < REGS; ++r) {
      s[laneRow(r, l)] += k[laneRow(r, l)] * delta;
      qp[l] += q[laneRow(r, l)] * s[laneRow(r, l)];
    }
  return warpReduce(qp) / std::sqrt(float(S));
}

}  // namespace baloon::native_gdn
