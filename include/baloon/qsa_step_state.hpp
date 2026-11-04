// qsa_step_state.hpp - per-token device step record, selection width, paged KV row mapping, indexer pooling schedule
// and the cell -> pooled-block map of the QSA decode layer. Host/device inline so the kernels and the tests share one definition.
#pragma once
#include <cstdint>
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace qsa_step {

constexpr int kTopK = 2048;
constexpr int kR = 4;                       // cells per pooled block
constexpr int kMaxWidth = kTopK + kR - 1;   // 2051
constexpr int kTopkMaxCells = 32768;        // largest cell count the one-block top-k handles
constexpr int kHeadDim = 256;
constexpr int kGroup = 12;                  // query heads per kv head
constexpr int kIdxDim = 128;
constexpr int kIdxHeads = 4;
constexpr float kTailBias = 1.0e9f;

// Four int32 = 16 bytes, uploaded as one record.
struct Step { int32_t pos, n_kv, n_bid, width; };
static_assert(sizeof(Step) == 16, "step record is four int32");

BALOON_BITS_HD int selection_width(int n_kv, int top_k, int r) {
    const int cap = top_k + r - 1;
    return n_kv < cap ? n_kv : cap;
}
BALOON_BITS_HD Step step_fill(int pos, int top_k, int r) {
    Step s;
    s.pos = pos; s.n_kv = pos + 1; s.n_bid = (pos + 1) / r; s.width = selection_width(pos + 1, top_k, r);
    return s;
}
// Shapes of one QSA layer (the real model: 24 heads, 2 kv heads, head_dim 256, 4 indexer heads of 128).
struct Shapes {
    int n_head, n_head_kv, head_dim, idx_block, top_k, page_size, max_cells;
};
BALOON_BITS_HD Shapes real_shapes(int page_size, int max_cells) {
    Shapes s;
    s.n_head = 24; s.n_head_kv = 2; s.head_dim = kHeadDim; s.idx_block = kR; s.top_k = kTopK;
    s.page_size = page_size; s.max_cells = max_cells;
    return s;
}
// Physical row of cell t, kv head h: (page * n_head_kv + h) * page_size + t % page_size, page = table[t / page_size].
BALOON_BITS_HD int64_t kv_row(int64_t page, int n_head_kv, int h, int page_size, int64_t t) {
    return (page * n_head_kv + h) * (int64_t)page_size + (t % page_size);
}
// A block completes exactly when pos % r == r - 1.
BALOON_BITS_HD bool completes_block(int pos, int r) { return pos % r == r - 1; }
// Pooled row of cell j: min(j / r, n_bid). The spare row n_bid holds the incomplete tail.
BALOON_BITS_HD int cell_block(int j, int n_kv, int r) {
    const int b = j / r, nb = n_kv / r;
    return b < nb ? b : nb;
}
BALOON_BITS_HD bool tail_cell(int j, int n_kv, int r) { return j >= (n_kv / r) * r; }
BALOON_BITS_HD int block_weight(int b, int n_kv, int r) { const int nb = n_kv / r; return b < nb ? r : n_kv - nb * r; }
// Monotone key: float order == unsigned integer order. -0.0 is folded onto +0.0 first (s + 0.0f).
BALOON_BITS_HD uint32_t order_key_bits(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
BALOON_BITS_HD uint32_t order_key(float f) { return order_key_bits(f16bits::bits_of(f + 0.0f)); }

// Bytes of one device step record.
BALOON_BITS_HD int step_bytes() { return (int)sizeof(Step); }
// RMS-norm epsilon of the QSA layer (same value as the layer stack).
BALOON_BITS_HD float rms_eps() { return 1.0e-6f; }
// RoPE frequency base used when the model metadata does not provide one (callers pass the metadata value when present).
BALOON_BITS_HD float freq_base() { return 1.0e6f; }

// Device buffers of the indexer of one layer (all f32): raw tail rows, pooled block keys, per-block and per-cell scores.
struct IndexerBuffers {
    float* tail;         // [r - 1][128]
    float* pooled;       // [max_cells / r + 1][128]
    float* block_score;  // [max_cells / r + 1]
    float* cell_score;   // [max_cells]
};
BALOON_BITS_HD int64_t pooled_rows(int max_cells, int r) { return (int64_t)max_cells / r + 1; }

}  // namespace qsa_step
}  // namespace baloon
