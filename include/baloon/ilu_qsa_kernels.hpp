// ilu_qsa_kernels.hpp - host API of the BI-V100 QSA kernels: split-K decode attention (chunk_partial + chunk_merge),
// block scoring (block_score), radix-select top-k (block_select), tile block scoring (mma_block_score) and the pooled
// indexer append (pool_append). Pure C++ (no CUDA headers) so it compiles standalone; implementations live in
// src/ilu_qsa_*.cu. Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
// All shuffles use 32-lane segments, so results do not depend on the hardware wave width (64 on ivcore10).
#pragma once
#include <cstddef>
#include <cstdint>
#include "baloon/native_qsa_indexer_append.hpp"
#include "baloon/native_qsa_score.hpp"
#include "baloon/qsa_block_select.hpp"
#include "baloon/qsa_decode_attention.hpp"

namespace baloon {
namespace ilu_qsa {

constexpr int kOk = 0;
constexpr int kInvalid = -1;
constexpr float kTailBias = 1.0e9f;
constexpr int kTopKWidth = 2051;  // top_k (2048) + r (4) - 1

// Order-preserving map float -> uint32 (larger float, larger key; -0 < +0 is irrelevant for scores).
inline uint32_t order_key(float f) {
    uint32_t u;
    static_assert(sizeof(u) == sizeof(f), "float is 32-bit");
    __builtin_memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One span of device memory touched by a launch; `write` spans are outputs.
struct Region { uint64_t addr; uint64_t bytes; bool write; };
inline Region region(const void* p, uint64_t bytes, bool write) { return Region{(uint64_t)(uintptr_t)p, bytes, write}; }
// Non-null, no address wrap, and no write region overlapping any other region. Zero-byte regions are not exempt.
inline bool regions_ok(const Region* r, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (r[i].addr == 0 || !native_qsa_indexer::span_ok(r[i].addr, r[i].bytes)) return false;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if ((r[i].write || r[j].write) && native_qsa_indexer::overlap(r[i].addr, r[i].bytes, r[j].addr, r[j].bytes))
                return false;
    return true;
}

// ------------------------------------------------------------------------------------------------------------
// Split-K decode attention. One KV pool of `n_cells` cells per kv head; the row of cell c, kv head h is
//   row = (page * n_kv + h) * page_size + slot,  page = c / page_size, slot = c % page_size   (qsa_decode::pool_row)
// K/V pools hold rows of 256 values: fp16 (q8 == false) or int8 codes + float scales[row*4 + d/64] (q8 == true).
//   q:      [nq][n_kv*12][256] floats (multiplied by `scale` inside the kernel)
//   ids:    [nq][cap] selected cell ids (ascending is not required); n_ids[q] <= cap are valid; ids >= n_cells are masked
//   scratch:[nq][n_chunks*n_head*258] floats, n_chunks = ceil(cap/64)  (qsa_decode::query_stride)
//   out:    [nq][n_kv*12][256] floats; heads with nothing attended get zeros.
struct DecodeArgs {
    const float* q;
    const void* k;
    const void* v;
    const float* k_scale;  // q8 only
    const float* v_scale;  // q8 only
    const int32_t* ids;
    const int32_t* n_ids;
    float* scratch;
    float* out;
    int nq, n_kv, cap, page_size, n_cells;
    bool q8;
    float scale;
};
int decode_attention(const DecodeArgs& a, void* stream);

// ------------------------------------------------------------------------------------------------------------
// Block scoring + top-k selection over pooled indexer blocks (128-dim keys, 4 indexer heads, R = 4 cells per block).
//   score(b) = sum_h hw[h] * relu(q_h . key_b) (+ bias[b] if bias) (+ 1e9 for the tail block when n_kv % 4 != 0)
//   blocks considered: nb = n_kv/4 + (n_kv % 4 != 0).
//   pooled: [rows][128] floats, rows >= nb;  iq: [4][128];  hw: [4];  scores out: [nb].
int block_score(const float* pooled, const float* iq, const float* hw, const float* bias, float* scores,
                int n_kv, int rows, void* stream);
// Radix-select top-k of cells from block scores: selects `width` cells (all cells ascending when n_kv <= width).
//   out_ids: [width] ascending cell ids padded with -1; n_out: single int written with the number selected.
int block_select(const float* scores, int n_kv, int width, int32_t* out_ids, int32_t* n_out, void* stream);

// Tile scoring: 32 pooled rows per block, two 32-lane warps splitting the 128-dim key. Same score as block_score;
// also broadcasts each block score to its cells (cell_scores: [n_kv]). block_scores may be null.
// Accepted only for pos == n_kv - 1, n_bid == n_kv/4 and width == min(n_kv, 2051).
int mma_block_score(const float* pooled, const float* iq, const float* hw, const float* bias, float* block_scores,
                    float* cell_scores, int pos, int n_kv, int n_bid, int width, int max_cells, int rows, void* stream);

// ------------------------------------------------------------------------------------------------------------
// Append one raw indexer key at `pos` (single block of 256 threads, 128 active dims).
//   tail: [3][128] raw keys; pooled: [max_cells/4 + 1][128]; dead: [128] spare row.
//   pos == 0 initialises pooled row 0 and dead; slot 3 mean-pools tail[0..2] + key, RMS-normalises (eps), rotates dims
//   0..63 in NeoX pairs (d, d+32) with angle rope_pos * freq_step^(d%32), writes row pos/4 and copies dead to row pos/4+1.
struct AppendArgs {
    const float* key;
    float* tail;
    float* pooled;
    float* dead;
    int pos, max_cells, rows;
    long long pos_base;
    float freq_step, eps;
};
int pool_append(const AppendArgs& a, void* stream);

}  // namespace ilu_qsa
}  // namespace baloon

