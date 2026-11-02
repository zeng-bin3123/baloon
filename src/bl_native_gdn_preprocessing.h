// bl_native_gdn_preprocessing.h - 4-tap causal conv + SiLU with history shift, L2 norm over width 128,
// beta sigmoid, gate softplus, and their launch contracts.
#pragma once
#include <cmath>
#include <cstdint>
#include "bl_nat.h"

namespace baloon::native_gdn_pre {

constexpr nat S = 128, TAPS = 4, HIST = 3, NORM_THREADS = 256, MAX_COUNT = 65535;

// Window [h0,h1,h2,x]; the new history is the last three entries [h1,h2,x].
inline void window(const float* h, float x, float w[4]) { w[0] = h[0]; w[1] = h[1]; w[2] = h[2]; w[3] = x; }
inline void nextHistory(float* h, float x) { h[0] = h[1]; h[1] = h[2]; h[2] = x; }
inline float dot4(const float* taps, const float* h, float x) {
  return h[0] * taps[0] + h[1] * taps[1] + h[2] * taps[2] + x * taps[3];
}
inline float silu(float v) { return v / (1.f + std::exp(-v)); }
// One conv step for a channel: returns SiLU output, writes the raw value, shifts the history.
inline float convStep(float* h, const float* taps, float x, float* raw) {
  float r = dot4(taps, h, x);
  *raw = r;
  nextHistory(h, x);
  return silu(r);
}
constexpr nat histAddr(nat c, nat tap) { return c * HIST + tap; }
constexpr nat tapAddr(nat c, nat tap) { return c * TAPS + tap; }
constexpr nat gridBlocks256(nat n) { return ndiv(n + 255, 256); }

// L2 row norm of width 128: scale = rsqrt(sum/128 + eps/128), then 1/sqrt(128).
inline void l2NormRow(const float* x, float eps, float* out) {
  float sum = 0;
  for (nat i = 0; i < S; ++i) sum += x[i] * x[i];
  float scale = 1.f / std::sqrt(sum / float(S) + eps / float(S));
  float f = 1.f / std::sqrt(float(S));
  for (nat i = 0; i < S; ++i) out[i] = x[i] * scale * f;
}
constexpr nat normRowAddr(nat row, nat col) { return row * S + col; }
constexpr bool threadActive(nat t) { return t < S; }  // threads 128..255 idle

inline float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }
// softplus(v) = v for v > 20.
constexpr bool softplusBranch(double v) { return v > 20.0; }
inline float softplus(float v) { return softplusBranch(v) ? v : std::log1p(std::exp(v)); }
inline float gate(float alpha, float dt, float ssm_a) { return softplus(alpha + dt) * ssm_a; }

constexpr bool countOk(std::int64_t c) { return 1 <= c && c <= std::int64_t(MAX_COUNT); }
constexpr bool normGeometryOk(std::int64_t rows, std::int64_t cols) { return 1 <= rows && rows <= std::int64_t(MAX_COUNT) && cols == 128; }
constexpr nat convSpanBytes(nat channels) { return 3 * (channels * 4) + 4 * (channels * 4); }

}  // namespace baloon::native_gdn_pre
