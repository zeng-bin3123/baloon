#pragma once
// strata/kernels/qsa.hpp — QSA (Query-Score-Attend) full-attention kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// QsaShapes: geometry of one QSA layer.
//   n_head=24, n_head_kv=2, head_dim=256, n_rot=64.
//   idx_n_head=4, idx_dim=128, idx_block=4, idx_top_k=2048.
//   page_size=512 (KV page granule in cells).
//   q→kv head map: q / (n_head / n_head_kv), NOT q % n_head_kv.
//   head_dim must be a multiple of 4.
//
// qsa_selection_width: min(n_kv, idx_top_k + idx_block - 1).
//   The (r-1 = 3) is the incomplete tail, not slack.
//   n_kv <= 2051 ⟹ selection is IDENTITY.
//
// QsaStep: per-token device-resident step state for CUDA graph captureability.
//   step[0]=pos, step[1]=pos+1=n_kv, step[2]=n_kv/idx_block=n_bid,
//   step[3]=qsa_selection_width(n_kv, s).
//   Purpose: without this, a CUDA graph would bake first-token values.
//
// kv_append: write K/V to paged KV pool.
//   Physical row: page = table[pos / page_size],
//   row = (page * n_head_kv + h) * page_size + (pos % page_size).
//   FP16 conversion via f16_from_f32 (not __float2half).
//   k_cur MUST be already normed and rotated; V is never rotated.
//
// kv_gather: gather KV from paged pool by ascending unique cell ids.
//   Unselected cells have weight 0 in attention (equiv to -inf mask).
//
// indexer_key_append: indexer key management for block-pooled selection.
//   tail[pos % r] ← raw key (never normed/rotated).
//   On block completion (pos % r == r-1): pooled[b] = rope(rms_norm(mean(raw[block])), pos_base).
//   Rotation position is block's FIRST cell, not last.
//   Spare slot: pooled[n_bid] = rope(rms_norm(raw[0]), 0) always maintained.
//   pos_dev is DEVICE pointer for captureability.
//
// QsaIndexerBuffers: per-QSA-layer, per-sequence indexer state (all DEVICE).
//
// qsa_index: block-level scoring without softmax.
//   score[j] = bias[cb] + sum_h ReLU(<pooled[cb], q_idx[h]>)
//   CRITICAL: ReLU is PER HEAD, sum over heads AFTERWARDS.
//   Tail cells get +1e9 bias.  No softmax, no 1/sqrt(d).
//
// topk_512: deterministic single-block top-k selection.
//   Ties broken by ascending index.  n_kv ≤ kTopkMaxCells = 32768.
//   cap ≥ width (refuses rather than truncating).  No -1 padding.
//
// qsa_attend: GQA attention over gathered KV scratch.
//   scores[h,j] = <K_j[h_kv], q[h]> / sqrt(head_dim).
//   Empty ids ⟹ attn = 0.
//
// qsa_gate_apply / qsa_gate_apply_f32: gated attention output.
//   gate = SECOND half of q_full (not interleaved).
//   out = attn * sigmoid(gate).  SIGMOID, not SiLU.
//   Single rounding after multiply, not two.
//   _f32 variant: no fp16 round at end (for Q8_K quantization).
//
// _step variants: identical math but read counts from device step buffer
//   instead of host scalars.  Grid/shared memory sized at constant capacity.
//   CRITICAL: non-step wrappers are WRONG inside a graph.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace strata {
namespace kernels {

// ─── QsaShapes ──────────────────────────────────────────────────
// Geometry of one QSA (full-attention) layer.
//
// n_head=24, n_head_kv=2, head_dim=256, n_rot=64.
// idx_n_head=4, idx_dim=128, idx_block=4, idx_top_k=2048.
// page_size=512 (KV page granule in cells).
//
// q→kv head map: q / (n_head / n_head_kv), NOT q % n_head_kv.
// head_dim must be a multiple of 4.
struct QsaShapes {
    int64_t n_head     = 0;
    int64_t n_head_kv  = 0;
    int64_t head_dim   = 0;
    int64_t n_rot      = 0;
    int64_t idx_n_head = 0;
    int64_t idx_dim    = 0;
    int64_t idx_block  = 0;
    int64_t idx_top_k  = 0;
    int64_t page_size  = 0;

    bool valid() const {
        return n_head > 0 && n_head_kv > 0 && head_dim > 0
               && n_rot > 0 && idx_n_head > 0 && idx_dim > 0
               && idx_block > 0 && idx_top_k > 0 && page_size > 0
               && (head_dim % 4 == 0)
               && (n_head % n_head_kv == 0);
    }

    // Map query head index to KV head index.
    // CRITICAL: division, not modulo.
    int64_t kv_head(int64_t q_head) const {
        return q_head / (n_head / n_head_kv);
    }
};

// ─── qsa_selection_width ────────────────────────────────────────
// min(n_kv, idx_top_k + idx_block - 1)
// The (r-1 = 3) is the incomplete tail, not slack.
// n_kv ≤ 2051 ⟹ selection is IDENTITY.
inline int64_t qsa_selection_width(int64_t n_kv, const QsaShapes& shapes) {
    const int64_t w = shapes.idx_top_k + shapes.idx_block - 1;
    return n_kv < w ? n_kv : w;
}

// ─── QsaStep ────────────────────────────────────────────────────
// Per-token device-resident step state for CUDA graph captureability.
//
// step[0] = pos
// step[1] = pos + 1 = n_kv
// step[2] = n_kv / idx_block = n_bid
// step[3] = qsa_selection_width(n_kv, s)
//
// Without this, a CUDA graph would bake first-token values.
enum QsaStep : int {
    kStepPos   = 0,
    kStepNKv   = 1,
    kStepNBid  = 2,
    kStepWidth = 3,
    kStepCount = 4,
};

inline uint64_t qsa_step_bytes() {
    return sizeof(int32_t) * static_cast<uint64_t>(kStepCount);
}

// Fill host step buffer from position and shapes.
void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& shapes);

// ─── kv_append ──────────────────────────────────────────────────
// Write K/V to paged KV pool.
//
// Physical row: page = table[pos / page_size],
//   row = (page * n_head_kv + h) * page_size + (pos % page_size).
// FP16 conversion via f16_from_f32 (not __float2half).
// k_cur MUST be already normed and rotated; V is never rotated.
void kv_append(uint16_t* k_pool, uint16_t* v_pool,
               const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur,
               const QsaShapes& shapes, cudaStream_t stream);

// ─── kv_gather ──────────────────────────────────────────────────
// Gather KV from pool by ascending unique cell ids from topk.
//
// ids: ascending, unique logical cells.
// scratch[j] = pool[ids[j]] (through page table).
// Unselected cells have weight 0 in attention (equiv to -inf mask).
void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool,
               const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& shapes,
               uint16_t* k_scratch, uint16_t* v_scratch,
               cudaStream_t stream);

// ─── QsaIndexerBuffers ─────────────────────────────────────────
// Per-QSA-layer, per-sequence indexer state (all DEVICE).
//
// tail:      float[(r-1) × idx_dim]   — raw keys of incomplete block
// dead:      float[idx_dim]           — spare slot key = rms_norm(raw[0]), constant
// pooled:    float[(max_cells/r + 1) × idx_dim]  — block b at row b, spare at n_bid
// block_pos: int32[1]                 — scratch for rotation position
//
// Spare slot MOVES (at row n_bid), but its VALUE is constant.
struct QsaIndexerBuffers {
    float*   tail      = nullptr;
    float*   dead      = nullptr;
    float*   pooled    = nullptr;
    int32_t* block_pos = nullptr;
};

// ─── indexer_key_append ─────────────────────────────────────────
// Manage indexer key state for block-pooled selection.
//
// tail[pos % r] ← raw (raw key, NEVER normed/rotated).
// On block completion (pos % r == r-1):
//   pooled[b] = rope_neox(rms_norm(mean(raw[block]), w_k_norm), pos_base)
//   Rotation position is block's FIRST cell, not last.
// Spare slot (always):
//   pooled[n_bid] = rope_neox(rms_norm(raw[0], w_k_norm), 0)
//
// pos_dev is DEVICE pointer ⟹ capturable (no host scalar baked in).
// pos_base: sequence's first cell position (constant for a sequence).
void indexer_key_append(const float* raw, const int32_t* pos_dev,
                        int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& bufs,
                        const QsaShapes& shapes, const float* cos_tab,
                        const float* sin_tab, cudaStream_t stream);

// ─── qsa_index ──────────────────────────────────────────────────
// Block-level scoring for cell selection.
//
// cb(j) = min(j / r, n_bid)
// score[j] = bias[cb] + sum_h ReLU(<pooled[cb], q_idx[h]>)
//
// CRITICAL: ReLU is PER HEAD, sum over heads AFTERWARDS.
//   relu(sum_h dot_h) ≠ sum_h relu(dot_h)
// j ≥ (n_kv / r) * r ⟹ score[j] += 1e9  (tail bias)
// No softmax, no 1/sqrt(d).
void qsa_index(const float* pooled, int64_t n_bid,
               const float* q_idx, const float* bias,
               const QsaShapes& shapes, int64_t n_kv,
               float* cell_scores, cudaStream_t stream);

// ─── topk_512 ───────────────────────────────────────────────────
// Deterministic single-block top-k selection.
//
// |ids| = qsa_selection_width(n_kv, s).
// Ties broken by ASCENDING index.
// Single block, deterministic ⟹ n_kv ≤ kTopkMaxCells = 32768.
// cap ≥ width (refuses rather than truncating).
// No -1 padding.
inline constexpr int64_t kTopkMaxCells = 32768;

void topk_512(const float* cell_scores, int64_t n_kv,
              const QsaShapes& shapes, int64_t cap,
              int32_t* ids, cudaStream_t stream);

// ─── qsa_attend ─────────────────────────────────────────────────
// GQA attention over gathered KV scratch.
//
// scores[h,j] = <K_j[h * n_head_kv / n_head], q[h]> / sqrt(head_dim)
// w[h] = softmax_j(scores[h,:])
// attn[h] = sum_j w[h,j] * V_j[h * n_head_kv / n_head]
// Empty ids ⟹ attn = 0.
void qsa_attend(const float* q, const uint16_t* k_scratch,
                const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& shapes, float* attn,
                float* weights, cudaStream_t stream);

// ─── qsa_gate_apply ─────────────────────────────────────────────
// Gated attention output (fp16).
//
// gate[h,d] = q_full[h, head_dim + d]  (SECOND half, not interleaved)
// out[h,d] = f16(attn[h,d] * sigmoid(gate[h,d]))
// SIGMOID, not SiLU.  Rounding AFTER multiply (one rounding, not two).
void qsa_gate_apply(const float* attn, const float* q_full,
                    const QsaShapes& shapes, uint16_t* out,
                    cudaStream_t stream);

// ─── qsa_gate_apply_f32 ────────────────────────────────────────
// Same arithmetic as qsa_gate_apply but no fp16 round at end.
// For Q8_K quantization (vec_dot_type of K-quant attn_output).
// Measured cost of wrong choice: 0.66–1.41% per GEMV.
void qsa_gate_apply_f32(const float* attn, const float* q_full,
                        const QsaShapes& shapes, float* out,
                        cudaStream_t stream);

// ─── Capturable step variants ───────────────────────────────────
// Identical to non-step versions except:
//   Per-token counts read from step (device buffer) instead of host scalars.
//   Grid/shared memory sized at CONSTANT capacity, not per-token count.
//   Kernels guard surplus: return early past actual count.
//
// CRITICAL: non-step wrappers are WRONG inside a graph.
//   A captured qsa_index leaves other pooled rows unscored.
//   A captured qsa_attend reads past shared buffer as sequence grows.

void qsa_index_step(const float* pooled, const float* q_idx,
                    const float* bias, const QsaShapes& shapes,
                    const int32_t* step, int64_t max_blocks,
                    float* cell_scores, cudaStream_t stream);

void topk_512_step(const float* cell_scores, const QsaShapes& shapes,
                   int64_t cap, const int32_t* step,
                   int32_t* ids, cudaStream_t stream);

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool,
                    const int32_t* page_table, const int32_t* ids,
                    const int32_t* step, int64_t max_ids,
                    const QsaShapes& shapes,
                    uint16_t* k_scratch, uint16_t* v_scratch,
                    cudaStream_t stream);

void qsa_attend_step(const float* q, const uint16_t* k_scratch,
                     const uint16_t* v_scratch, const int32_t* step,
                     int64_t max_ids, const QsaShapes& shapes,
                     float* attn, float* weights, cudaStream_t stream);

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool,
                    const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur,
                    const QsaShapes& shapes, cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

