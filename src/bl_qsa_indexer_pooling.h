// bl_qsa_indexer_pooling.h - tail ring, block pooling (mean, RMS-norm * gamma, RoPE on the first 64 dims) and
// launch contracts of the QSA indexer append kernel. D = 128, R = 4, ROT = 64.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::qsa_indexer {

constexpr nat D = 128, R = 4, ROT = 64, TAIL_SLOTS = R - 1;

constexpr nat slotOf(nat pos) { return pos % R; }
constexpr nat blockOf(nat pos) { return pos / R; }
constexpr bool poolTriggered(nat pos) { return pos == 0 || pos % R == R - 1; }
constexpr nat tailIndex(nat slot, nat d) { return slot * D + d; }
constexpr nat pooledRows(nat maxCells) { return maxCells / R + 1; }
constexpr nat seedRow(nat pos) { return pos / R + 1; }  // seeded with the "dead" copy at block-closing pos != 0
constexpr nat pooledElem(nat row, nat d) { return row * D + d; }
constexpr nat pooledBytes(nat maxCells) { return pooledRows(maxCells) * D * 4; }
constexpr nat ropePos(nat posBase, nat b) { return posBase + R * b; }
constexpr nat pairOf(nat d) { return d % 32; }
constexpr bool appendArgsValid(std::int64_t idxDim, std::int64_t idxBlock, std::int64_t nRot, std::int64_t maxCells,
                               std::int64_t posBase) {
  return idxDim == 128 && idxBlock == 4 && nRot == 64 && 1 <= maxCells && maxCells <= 2147483647 && 0 <= posBase &&
         posBase % 4 == 0 && posBase + maxCells <= 2147483647;
}

// Reference pooling of block b: mean over `count` keys (4, or 1 at position 0), RMS-norm with gamma, then
// RoPE on dims [0,64): dim pair (p, p+32), p<32, angle = ropePos * base^(-p/32).
inline void poolBlock(const float* keys, nat count, const float* gamma, float eps, float base, nat ropePosition, float* out) {
  float mean[D];
  for (nat d = 0; d < D; ++d) {
    float s = 0;
    for (nat t = 0; t < count; ++t) s += keys[t * D + d];
    mean[d] = s / float(count);
  }
  float ss = 0;
  for (nat d = 0; d < D; ++d) ss += mean[d] * mean[d];
  float inv = 1.f / std::sqrt(ss / float(D) + eps);
  for (nat d = 0; d < D; ++d) out[d] = mean[d] * inv * gamma[d];
  for (nat p = 0; p < ROT / 2; ++p) {
    float ang = float(ropePosition) * std::pow(base, -float(p) / 32.f);
    float c = std::cos(ang), s = std::sin(ang);
    float a = out[p], b = out[p + 32];
    out[p] = a * c - b * s;
    out[p + 32] = a * s + b * c;
  }
}

// Reference of the streaming kernel: append one key per position, keeping the 3-slot tail ring.
struct Indexer {
  std::vector<float> tail, pooled;
  nat maxCells, posBase;
  float eps, base;
  std::vector<float> gamma;
  Indexer(nat mc, nat pb, float e, float bs, std::vector<float> g)
      : tail(TAIL_SLOTS * D, 0.f), pooled(pooledRows(mc) * D, 0.f), maxCells(mc), posBase(pb), eps(e), base(bs), gamma(std::move(g)) {}
  void append(nat pos, const float* key) {
    nat slot = slotOf(pos);
    if (slot < TAIL_SLOTS)
      for (nat d = 0; d < D; ++d) tail[tailIndex(slot, d)] = key[d];
    if (!poolTriggered(pos)) return;
    float keys[R * D];
    nat count = pos == 0 ? 1 : R;
    if (pos == 0) {
      for (nat d = 0; d < D; ++d) keys[d] = key[d];
    } else {
      for (nat s = 0; s < TAIL_SLOTS; ++s)
        for (nat d = 0; d < D; ++d) keys[s * D + d] = tail[tailIndex(s, d)];
      for (nat d = 0; d < D; ++d) keys[TAIL_SLOTS * D + d] = key[d];
    }
    poolBlock(keys, count, gamma.data(), eps, base, pos == 0 ? posBase : ropePos(posBase, blockOf(pos)), &pooled[pooledElem(blockOf(pos), 0)]);
  }
};

}  // namespace baloon::qsa_indexer
