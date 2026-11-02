// bl_iq_quant_mat_vec.h - layout and index contracts of the IQ-family (and Q2_0/Q3_K) quantised mat-vec kernels:
// block traits, row-dot call enumeration, grouped expert blob layout, q8_1 quantiser, dequantiser thread maps.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::iq_quant {

constexpr nat blockBytes(nat t) {
  switch (t) {
    case 16: return 66; case 17: return 74; case 18: return 98; case 20: return 18; case 21: return 110;
    case 22: return 82; case 29: return 56; case 23: return 136; case 11: return 110; case 42: return 18;
    default: return 0;
  }
}
constexpr nat blockValues(nat t) {
  switch (t) {
    case 20: return 32; case 42: return 64;
    case 16: case 17: case 18: case 21: case 22: case 29: case 23: case 11: return 256;
    default: return 0;
  }
}
constexpr nat kIqTypes[] = {16, 17, 18, 20, 21, 22, 23, 29, 42, 11};
constexpr nat kMmvqTypes[] = {16, 17, 18, 20, 21, 22, 29, 42};
constexpr nat kDownTypes[] = {20, 42};
constexpr nat rowBytes(nat t, nat n) { return ndiv(n, blockValues(t)) * blockBytes(t); }

// Row dot call k -> (block, part).
constexpr nat callBlock(nat k, nat ipb) { return ndiv(k, ipb); }
constexpr nat callPart(nat k, nat ipb) { return nmod(k, ipb); }
constexpr nat q8Start(nat kbx, nat qk) { return kbx * (qk / 32); }

constexpr nat mmvqGridBlocks(nat n_out) { return (n_out + 3) / 4; }
constexpr nat mmvqOut(nat c, nat r, nat n_out) { return c * n_out + r; }

// Grouped expert blob: gate | up | down.
struct Layout {
  nat nEmbd, nFf, guRow, dRow;
  constexpr nat upOff() const { return nFf * guRow; }
  constexpr nat downOff() const { return 2 * upOff(); }
  constexpr nat bytes() const { return downOff() + nEmbd * dRow; }
};
constexpr nat guGridBlocks(nat n_ff) { return (2 * n_ff + 7) / 8; }
constexpr nat downGridBlocks(nat n_embd) { return (n_embd + 7) / 8; }
constexpr nat entryOut(nat e, nat r, nat n_ff) { return e * n_ff + r; }

constexpr nat alignUp256(nat n) { return (n + 255) / 256 * 256; }

// q8_1 quantiser: block of 32, d = amax/127, code 0 when amax = 0.
struct Q81 { float d; std::int8_t q[32]; };
inline Q81 quantize_q8_1(const float* x) {
  float amax = 0.f;
  for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
  Q81 b{amax / 127.f, {}};
  for (int i = 0; i < 32; ++i) b.q[i] = amax == 0.f ? 0 : std::int8_t(std::lrintf(x[i] / b.d));
  return b;
}
constexpr nat q8Grid(nat n) { return (n + 255) / 256; }

// Dequantiser thread maps.
constexpr nat dq8Offset(nat tid, nat j) { return 32 * (tid % 8) + 8 * (tid / 8) + j; }
constexpr nat dq4Offset(nat ib, nat il, nat j, nat h) { return 32 * ib + 4 * il + j + 16 * h; }
constexpr nat dqQ2Offset(nat b, nat p, nat j) { return b * 64 + (p * 8 + j); }
constexpr nat embedOut(nat t, nat b, nat o, nat n_embd) { return t * n_embd + (b * 256 + o); }
constexpr nat guInterleave(nat r, nat parity, nat c, nat per_row) { return (2 * r + parity) * per_row + c; }

}  // namespace baloon::iq_quant
