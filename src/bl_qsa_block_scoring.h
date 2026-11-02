// bl_qsa_block_scoring.h - tile geometry, shared layout, step validity and cell export of the QSA score kernel:
// per pooled block row, 4 indexer heads scored over K = 128, ReLU'd, summed, biased, masked, replicated to 4 cells.
#pragma once
#include <cstdint>
#include <vector>
#include "bl_nat.h"

namespace baloon::qsa_score {

constexpr nat ROWS = 32, WARPS = 2, STRIDE = 36, COMBINE = 68, R = 4, K = 128, HEADS = 8, MAX_WIDTH = 2051;
constexpr float PARTIAL_PENALTY = 1e9f;

constexpr bool validStep(std::int64_t n, std::int64_t maxc, std::int64_t pos, std::int64_t full, std::int64_t width) {
  return 1 <= n && n <= maxc && pos == n - 1 && full == n / 4 && width == (n < 2051 ? n : 2051);
}
constexpr nat gridBlocks(nat maxBlocks) { return ndiv(maxBlocks + ROWS - 1, ROWS); }
constexpr nat rowOf(nat block, nat lane) { return block * ROWS + lane; }
constexpr nat kCol(nat w, nat l, nat it) { return w * 32 + l + 64 * it; }
constexpr nat sharedTileFloats() { return WARPS * 16 * STRIDE; }
constexpr nat loadAIndex(nat tid) { return (tid % 16) * STRIDE + (tid / 16) * 4; }  // + 0..3
constexpr nat loadBIndex(nat tid) { return (tid % 8) * STRIDE + ((tid / 8) * 4) % 8; }  // + 0..3
constexpr nat combineRow(nat w, nat ia, nat l, nat lane) { return w * 32 + ia * 16 + (l / 2) * 8 + lane / 4; }
constexpr nat combineHead(nat l, nat lane) { return (lane % 4) * 2 + l % 2; }
constexpr nat combineAddr(nat h, nat i) { return h * COMBINE + i; }
constexpr nat maxBlocksFor(nat maxCells) { return maxCells / R + 1; }
constexpr bool capacitiesOk(nat maxCells, nat maxBlocks) { return 1 <= maxCells && maxBlocks == maxCells / R + 1; }
constexpr bool rowScored(nat row, nat n) { return row <= n / R; }
constexpr bool rowPartial(nat row, nat n) { return row == n / R && n % R != 0; }
constexpr nat cellLo(nat row) { return row * R; }
constexpr nat cellHi(nat row, nat n) { return nmin(n, (row + 1) * R); }

// Reference score of one pooled row: sum_h relu(q_h . pooled_row) + bias, minus 1e9 for the partial last block.
inline float scoreRow(const float* pooledRow, const float* q /*4 x K*/, nat nHeads, float bias, bool partial) {
  float total = 0;
  for (nat h = 0; h < nHeads; ++h) {
    float a = 0;
    for (nat k = 0; k < K; ++k) a += q[h * K + k] * pooledRow[k];
    total += a > 0.f ? a : 0.f;
  }
  total += bias;
  return partial ? total - PARTIAL_PENALTY : total;
}
// Export of rows 0..n/4 into the cell array: each cell of the row gets the row score.
inline void exportCells(const std::vector<float>& rowScores, nat n, std::vector<float>& cells) {
  for (nat row = 0; row < rowScores.size() && rowScored(row, n); ++row)
    for (nat i = cellLo(row); i < cellHi(row, n); ++i) cells[i] = rowScores[row];
}

}  // namespace baloon::qsa_score
