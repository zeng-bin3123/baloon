// qsa_cache_step.hpp - geometry, step state and index arithmetic of the QSA decode layer (unit u316bd6b9ec).
// Real-model constants (shapes, rope base, rms eps, top-k capacity), the per-token step record, selection width,
// paged KV row mapping, indexer pooling schedule, tail-bias cell mapping, the monotone f32 order key and the
// threshold/tie selection rule. Pure C++ so it compiles standalone; the device kernels live in src/ilu_qsa_*.cu and
// src/ilu_qsa_geometry.cu, which evaluate the same formulas on the GPU for cross-checking.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "baloon/qsa_step_kernels.hpp"

namespace baloon {
namespace qsa_cs {

// ---- the device step record {pos, n_kv, n_bid, width} (16 bytes) ----------------------------------------------------
using QsaStep = qsa_step::Step;
constexpr size_t kStepBytes = 4 * sizeof(int32_t);
inline size_t qsa_step_bytes() { return kStepBytes; }

// Capacity of the one-block top-k: widest selection it may ever write is top_k + r - 1 = 2051 cells out of at most
// kTopkMaxCells cached cells.
constexpr int kTopkMaxCells = 32768;

// ---- real shapes -----------------------------------------------------------------------------------------------------
struct QsaShapes {
    int n_head, n_head_kv, head_dim;       // attention: 12 query heads per kv head, 256-wide rows
    int idx_heads, idx_dim, idx_block;     // indexer: 4 heads, 128-dim pooled keys, r = 4 cells per pooled block
    int top_k;                             // 2048
    int page_size;                         // cells per KV page
    int group() const { return n_kv_ok() ? n_head / n_head_kv : 0; }
    bool n_kv_ok() const { return n_head_kv > 0 && n_head > 0 && n_head % n_head_kv == 0; }
    int sel_cap() const { return top_k + idx_block - 1; }   // 2051
};
inline QsaShapes qsa_real_shapes() { return QsaShapes{24, 2, 256, 4, 128, 4, 2048, 64}; }
inline float qsa_freq_base() { return 10000000.0f; }          // rope base of the indexer key rotation
inline float qsa_rms_eps() { return 1.0e-6f; }                 // RMS-norm epsilon of the pooled indexer key
// freq_step handed to the pooled-key rotation: base^(-2/64) per pair (32 pairs in 64 dims).
inline float qsa_freq_step(float base = qsa_freq_base()) { return std::pow(base, -2.0f / 64.0f); }

// ---- selection width and step state ---------------------------------------------------------------------------------
inline int qsa_selection_width(int n_kv, int top_k = 2048, int r = 4) {
    const int cap = top_k + r - 1;
    return n_kv < cap ? n_kv : cap;
}
inline QsaStep step_fill_host(int pos, int top_k = 2048, int r = 4) {
    return QsaStep{pos, pos + 1, (pos + 1) / r, qsa_selection_width(pos + 1, top_k, r)};
}
inline bool step_consistent(const QsaStep& s, int max_cells, int top_k = 2048, int r = 4) {
    return s.n_kv >= 1 && s.n_kv <= max_cells && s.pos == s.n_kv - 1 && s.n_bid == s.n_kv / r &&
           s.width == qsa_selection_width(s.n_kv, top_k, r);
}

// ---- indexer buffers: raw tail (r-1 rows), pooled rows (max_cells/r + 1) and the spare "dead" row, 16-byte aligned ----
struct QsaIndexerBuffers {
    size_t tail_off, pooled_off, dead_off, total;   // byte offsets
    int rows;                                       // pooled rows = max_cells/r + 1
};
inline size_t align16(size_t n) { return (n + 15) / 16 * 16; }
inline QsaIndexerBuffers indexer_buffers(int max_cells, int idx_dim = 128, int r = 4) {
    QsaIndexerBuffers b{};
    b.rows = max_cells / r + 1;
    b.tail_off = 0;
    b.pooled_off = align16((size_t)(r - 1) * idx_dim * sizeof(float));
    b.dead_off = b.pooled_off + align16((size_t)b.rows * idx_dim * sizeof(float));
    b.total = b.dead_off + align16((size_t)idx_dim * sizeof(float));
    return b;
}

// ---- paged KV mapping ------------------------------------------------------------------------------------------------
inline uint64_t kv_row(uint64_t page, uint64_t n_head_kv, uint64_t h, uint64_t page_size, uint64_t slot) {
    return (page * n_head_kv + h) * page_size + slot;
}
inline uint64_t kv_row_of_cell(const int32_t* table, uint64_t cell, uint64_t n_head_kv, uint64_t h, uint64_t page_size) {
    return kv_row((uint64_t)table[cell / page_size], n_head_kv, h, page_size, cell % page_size);
}

// ---- pooling schedule and cell -> block mapping ---------------------------------------------------------------------
inline bool block_completes(int pos, int r = 4) { return pos % r == r - 1; }
inline int cell_block(int j, int n_kv, int r = 4) { int b = j / r, s = n_kv / r; return b < s ? b : s; }
inline bool cell_in_tail(int j, int n_kv, int r = 4) { return (n_kv / r) * r <= j; }
inline bool has_tail(int n_kv, int r = 4) { return n_kv % r != 0; }
inline int block_weight(int b, int n_bid, int n_kv, int r = 4) { return b < n_bid ? r : n_kv - n_bid * r; }

// ---- order key and threshold selection ------------------------------------------------------------------------------
inline uint32_t order_key_bits(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
inline uint32_t neg_zero_fold(uint32_t b) { return b == 0x80000000u ? 0u : b; }
inline uint32_t order_key_f32(float f) { uint32_t b; std::memcpy(&b, &f, 4); return order_key_bits(neg_zero_fold(b)); }
inline uint32_t selected_count(uint32_t gt, uint32_t eq, uint32_t width) { return gt + (eq < width - gt ? eq : width - gt); }

}  // namespace qsa_cs
}  // namespace baloon
