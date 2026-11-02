// bl_ple_block.h - GPU half of the PLE block: depthwise causal dilated conv + SiLU over [history | new],
// row-fastest history, signed-sqrt sigmoid gate, history shift, scratch layout and overlap guard.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::ple {

constexpr nat round16(nat n) { return (n + 15) / 16 * 16; }
// Row of [history | new] read by tap k.
constexpr std::int64_t convRow(nat nhist, nat kern, nat dil, nat k) {
  return std::int64_t(nhist) - (std::int64_t(kern) - 1 - std::int64_t(k)) * std::int64_t(dil);
}
// Row-fastest flat index of the conv state.
constexpr nat idx(nat nhist, nat row, nat c) { return row + nhist * c; }
// Scratch regions back to back, each rounded to 16 bytes.
constexpr nat regionOffset(const nat* sizes, nat k) {
  nat off = 0;
  for (nat i = 0; i < k; ++i) off += round16(sizes[i]);
  return off;
}
// Pointer-range overlap test.
constexpr bool ovl(nat aa, nat an, nat bb, nat bn) { return aa < bb ? (bb - aa < an) : (aa - bb < bn); }
constexpr int sgn(double s) { return s > 0 ? 1 : (s < 0 ? -1 : 0); }
inline float gate(float s) {
  float v = float(sgn(s)) * std::sqrt(std::fabs(s));
  return 1.f / (1.f + std::exp(-v));
}
inline float silu(float v) { return v / (1.f + std::exp(-v)); }

// One conv step for channel c. `hist` is the whole state (row-fastest, nhist rows per channel); `w` holds `kern`
// taps per channel. Output = SiLU(sum_k w[c][k] * row(convRow(k))), then the history shifts left and appends x.
inline float convStep(std::vector<float>& hist, const std::vector<float>& w, nat nhist, nat kern, nat dil, nat c, float x) {
  float acc = 0;
  for (nat k = 0; k < kern; ++k) {
    std::int64_t r = convRow(nhist, kern, dil, k);
    float v = r == std::int64_t(nhist) ? x : hist[idx(nhist, nat(r), c)];
    acc += w[c * kern + k] * v;
  }
  for (nat r = 0; r + 1 < nhist; ++r) hist[idx(nhist, r, c)] = hist[idx(nhist, r + 1, c)];
  hist[idx(nhist, nhist - 1, c)] = x;
  return silu(acc);
}

}  // namespace baloon::ple
