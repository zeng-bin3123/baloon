// bl_native_short_flash_attention.h - geometry and step validity of the single-token FlashAttention-vec kernel
// (24 query heads x 256 dims, 2 KV heads, contexts 1..256, 128 threads per head, tiles of 128 cells).
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::native_short_flash {

constexpr nat N_HEAD = 24, N_KV = 2, HEAD_DIM = 256, HEADS_PER_KV = 12, TILE = 128, PAD = 256;
constexpr nat WARPS = 4, LANES = 32, ROWS = 8, REDUCE_FLOATS = 4 * 4 * 256;

// Step buffer validity: 1 <= width <= maxc, nkv == width, pos == width-1, nbid == width/4.
constexpr bool validStep(std::int64_t width, std::int64_t nkv, std::int64_t pos, std::int64_t nbid, std::int64_t maxc) {
  return 1 <= width && width <= maxc && nkv == width && pos == width - 1 && nbid == width / 4;
}
constexpr nat kvHead(nat head) { return head / HEADS_PER_KV; }
// Tile-local cell of (warp, lane, row): warp*32 + (lane & ~7) + row.
constexpr nat tileCell(nat warp, nat lane, nat row) { return warp * 32 + (lane - lane % 8) + row; }
constexpr nat kvIndex(nat cell, nat kv, nat d) { return (cell * N_KV + kv) * HEAD_DIM + d; }
constexpr nat dimPair(nat m, nat l, nat j) { return 2 * (32 * m + 4 * l + j); }
constexpr nat reduceIndex(nat w, nat group, nat i0, nat tid) { return w * 1024 + group * 256 + i0 + tid; }
constexpr nat outIndex(nat head, nat e) { return head * HEAD_DIM + e; }
constexpr bool supportedShape(std::int64_t nh, std::int64_t nkv, std::int64_t hd, std::int64_t cap, std::int64_t maxc) {
  return nh == 24 && nkv == 2 && hd == 256 && 256 <= cap && 1 <= maxc && maxc <= 256;
}

// Reference attention of one head: naive softmax(q.K * scale) V over `width` cells (K, V in (cell,kv,d) layout).
inline void refAttention(const float* q, const float* K, const float* V, nat width, nat kv, float scale, float* out) {
  std::vector<float> s(width);
  float m = -INFINITY;
  for (nat c = 0; c < width; ++c) {
    float a = 0;
    for (nat d = 0; d < HEAD_DIM; ++d) a += q[d] * K[kvIndex(c, kv, d)];
    s[c] = a * scale;
    m = s[c] > m ? s[c] : m;
  }
  float sum = 0;
  for (nat c = 0; c < width; ++c) { s[c] = std::exp(s[c] - m); sum += s[c]; }
  for (nat d = 0; d < HEAD_DIM; ++d) {
    float a = 0;
    for (nat c = 0; c < width; ++c) a += s[c] * V[kvIndex(c, kv, d)];
    out[d] = a / sum;
  }
}

// Same result computed in the kernel's tile order with online (running max / rescale) softmax; padded cells
// (cell >= width) are masked, so they never contribute.
inline void tiledAttention(const float* q, const float* K, const float* V, nat width, nat kv, float scale, float* out) {
  float m = -INFINITY, sum = 0;
  std::vector<float> acc(HEAD_DIM, 0.f);
  for (nat base = 0; base < PAD; base += TILE) {
    for (nat local = 0; local < TILE; ++local) {
      nat cell = base + local;
      if (cell >= width) continue;  // mask
      float a = 0;
      for (nat d = 0; d < HEAD_DIM; ++d) a += q[d] * K[kvIndex(cell, kv, d)];
      a *= scale;
      float nm = a > m ? a : m;
      float corr = std::exp(m - nm), p = std::exp(a - nm);
      sum = sum * corr + p;
      for (nat d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * corr + p * V[kvIndex(cell, kv, d)];
      m = nm;
    }
  }
  for (nat d = 0; d < HEAD_DIM; ++d) out[d] = acc[d] / sum;
}

}  // namespace baloon::native_short_flash
