// strata/kernels/qsa.cpp — QSA (Query-Score-Attend) kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>

namespace strata {
namespace kernels {

// ─── qsa_step_fill ──────────────────────────────────────────────
void qsa_step_fill(int32_t* host_step, int64_t pos,
                   const QsaShapes& shapes) {
    if (!host_step) return;

    const int64_t n_kv  = pos + 1;
    const int64_t n_bid = n_kv / shapes.idx_block;
    const int64_t width = qsa_selection_width(n_kv, shapes);

    host_step[kStepPos]   = static_cast<int32_t>(pos);
    host_step[kStepNKv]   = static_cast<int32_t>(n_kv);
    host_step[kStepNBid]  = static_cast<int32_t>(n_bid);
    host_step[kStepWidth] = static_cast<int32_t>(width);
}

// ─── kv_append ──────────────────────────────────────────────────
void kv_append(uint16_t* k_pool, uint16_t* v_pool,
               const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur,
               const QsaShapes& shapes, cudaStream_t /*stream*/) {
    if (!k_pool || !v_pool || !page_table || !kcur || !vcur) return;
    if (!shapes.valid()) return;

    const int64_t ps = shapes.page_size;
    const int64_t page_idx = pos / ps;
    const int64_t page_off = pos % ps;
    const int32_t page = page_table[page_idx];

    for (int64_t h = 0; h < shapes.n_head_kv; ++h) {
        // Physical row in paged pool
        const int64_t phys_row = (static_cast<int64_t>(page) * shapes.n_head_kv + h)
                                 * ps + page_off;

        // Write K: head_dim elements, FP16 via f16_from_f32
        for (int64_t d = 0; d < shapes.head_dim; ++d) {
            k_pool[phys_row * shapes.head_dim + d] =
                f16_from_f32(kcur[h * shapes.head_dim + d]);
        }

        // Write V: head_dim elements, FP16 via f16_from_f32
        for (int64_t d = 0; d < shapes.head_dim; ++d) {
            v_pool[phys_row * shapes.head_dim + d] =
                f16_from_f32(vcur[h * shapes.head_dim + d]);
        }
    }
}

// ─── kv_gather ──────────────────────────────────────────────────
void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool,
               const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& shapes,
               uint16_t* k_scratch, uint16_t* v_scratch,
               cudaStream_t /*stream*/) {
    if (!k_pool || !v_pool || !page_table || !ids) return;
    if (!k_scratch || !v_scratch) return;
    if (!shapes.valid() || n_ids <= 0) return;

    const int64_t ps = shapes.page_size;

    for (int64_t j = 0; j < n_ids; ++j) {
        const int64_t cell = ids[j];
        const int64_t page_idx = cell / ps;
        const int64_t page_off = cell % ps;
        const int32_t page = page_table[page_idx];

        for (int64_t h = 0; h < shapes.n_head_kv; ++h) {
            const int64_t phys_row = (static_cast<int64_t>(page) * shapes.n_head_kv + h)
                                     * ps + page_off;

            for (int64_t d = 0; d < shapes.head_dim; ++d) {
                k_scratch[(j * shapes.n_head_kv + h) * shapes.head_dim + d] =
                    k_pool[phys_row * shapes.head_dim + d];
                v_scratch[(j * shapes.n_head_kv + h) * shapes.head_dim + d] =
                    v_pool[phys_row * shapes.head_dim + d];
            }
        }
    }
}

// ─── RoPE NeoX helper ───────────────────────────────────────────
static void rope_neox(float* vec, int64_t dim, int64_t n_rot,
                      const float* cos_tab, const float* sin_tab,
                      int64_t pos) {
    // NeoX-style RoPE: first half and second half interleave
    // Apply rotation to the first n_rot dimensions
    const int64_t half = n_rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float cos_val = cos_tab ? cos_tab[pos * half + i] : std::cos(static_cast<float>(pos) / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(n_rot)));
        const float sin_val = sin_tab ? sin_tab[pos * half + i] : std::sin(static_cast<float>(pos) / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(n_rot)));

        const float x0 = vec[i];
        const float x1 = vec[i + half];
        vec[i]        = x0 * cos_val - x1 * sin_val;
        vec[i + half] = x0 * sin_val + x1 * cos_val;
    }
}

// ─── RMS norm helper ────────────────────────────────────────────
static void rms_norm(const float* in, const float* w, int64_t n,
                     float eps, float* out) {
    float sq_sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sq_sum += in[i] * in[i];
    }
    const float inv_rms = 1.0f / std::sqrt(sq_sum / static_cast<float>(n) + eps);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = in[i] * inv_rms * (w ? w[i] : 1.0f);
    }
}

// ─── indexer_key_append ─────────────────────────────────────────
void indexer_key_append(const float* raw, const int32_t* pos_dev,
                        int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& bufs,
                        const QsaShapes& shapes, const float* cos_tab,
                        const float* sin_tab, cudaStream_t /*stream*/) {
    if (!raw || !pos_dev || !bufs.tail || !bufs.pooled) return;
    if (!shapes.valid()) return;

    const int64_t r = shapes.idx_block;
    const int64_t idx_dim = shapes.idx_dim;
    const int32_t pos = *pos_dev;  // read from device pointer (host fallback)
    const int64_t tail_slot = pos % r;
    const int64_t n_kv = pos + 1;
    const int64_t n_bid = n_kv / r;

    // Store raw key in tail (NEVER normed/rotated)
    std::memcpy(bufs.tail + tail_slot * idx_dim, raw, static_cast<size_t>(idx_dim) * sizeof(float));

    // On block completion (pos % r == r - 1):
    if (tail_slot == r - 1) {
        const int64_t block = pos / r;

        // Mean of raw keys in block
        std::vector<float> mean_key(static_cast<size_t>(idx_dim), 0.0f);
        for (int64_t slot = 0; slot < r; ++slot) {
            for (int64_t d = 0; d < idx_dim; ++d) {
                mean_key[static_cast<size_t>(d)] += bufs.tail[slot * idx_dim + d];
            }
        }
        for (int64_t d = 0; d < idx_dim; ++d) {
            mean_key[static_cast<size_t>(d)] /= static_cast<float>(r);
        }

        // rms_norm(mean_key, w_k_norm)
        std::vector<float> normed(static_cast<size_t>(idx_dim));
        rms_norm(mean_key.data(), w_k_norm, idx_dim, eps, normed.data());

        // rope_neox at block's FIRST cell position, not last
        const int64_t rope_pos = pos_base + block * r;
        rope_neox(normed.data(), idx_dim, shapes.n_rot, cos_tab, sin_tab, rope_pos);

        // Write to pooled[block]
        std::memcpy(bufs.pooled + block * idx_dim, normed.data(),
                    static_cast<size_t>(idx_dim) * sizeof(float));
    }

    // Spare slot (always): pooled[n_bid] = rope(rms_norm(raw[0], w_k_norm), 0)
    // The spare slot's VALUE is constant (rope at position 0), but its LOCATION moves
    if (bufs.dead) {
        // dead = rms_norm(raw[0], w_k_norm) — only need to compute once, but we
        // always update since this is the host reference path
        if (pos == 0) {
            rms_norm(raw, w_k_norm, idx_dim, eps, bufs.dead);
            // rope at position 0 is identity for NeoX (cos(0)=1, sin(0)=0)
            rope_neox(bufs.dead, idx_dim, shapes.n_rot, cos_tab, sin_tab, 0);
        }
    }

    // Copy spare slot to pooled[n_bid]
    if (bufs.dead) {
        std::memcpy(bufs.pooled + n_bid * idx_dim, bufs.dead,
                    static_cast<size_t>(idx_dim) * sizeof(float));
    }
}

// ─── qsa_index ──────────────────────────────────────────────────
void qsa_index(const float* pooled, int64_t n_bid,
               const float* q_idx, const float* bias,
               const QsaShapes& shapes, int64_t n_kv,
               float* cell_scores, cudaStream_t /*stream*/) {
    if (!pooled || !q_idx || !cell_scores) return;
    if (!shapes.valid() || n_kv <= 0) return;

    const int64_t r = shapes.idx_block;
    const int64_t idx_dim = shapes.idx_dim;
    const int64_t idx_n_head = shapes.idx_n_head;
    const int64_t tail_start = (n_kv / r) * r;  // first cell in incomplete block

    for (int64_t j = 0; j < n_kv; ++j) {
        // cb(j) = min(j / r, n_bid)
        int64_t cb = j / r;
        if (cb > n_bid) cb = n_bid;

        // score = bias[cb] + sum_h ReLU(<pooled[cb], q_idx[h]>)
        float score = bias ? bias[cb] : 0.0f;

        for (int64_t h = 0; h < idx_n_head; ++h) {
            float dot = 0.0f;
            for (int64_t d = 0; d < idx_dim; ++d) {
                dot += pooled[cb * idx_dim + d] * q_idx[h * idx_dim + d];
            }
            // CRITICAL: ReLU is PER HEAD, sum over heads AFTERWARDS
            if (dot > 0.0f) {
                score += dot;
            }
        }

        // Tail bias: cells in incomplete block get +1e9
        if (j >= tail_start) {
            score += 1e9f;
        }

        cell_scores[j] = score;
    }
}

// ─── topk_512 ───────────────────────────────────────────────────
void topk_512(const float* cell_scores, int64_t n_kv,
              const QsaShapes& shapes, int64_t cap,
              int32_t* ids, cudaStream_t /*stream*/) {
    if (!cell_scores || !ids) return;
    if (!shapes.valid() || n_kv <= 0) return;

    const int64_t width = qsa_selection_width(n_kv, shapes);
    if (cap < width) return;  // refuses rather than truncating

    // Build index array and sort by score descending, then by index ascending for ties
    std::vector<int32_t> indices(static_cast<size_t>(n_kv));
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [cell_scores](int32_t a, int32_t b) {
                  if (cell_scores[a] != cell_scores[b]) {
                      return cell_scores[a] > cell_scores[b];  // higher score first
                  }
                  return a < b;  // ascending index for ties
              });

    // Take top-width indices and sort them ascending (output is ascending)
    std::vector<int32_t> selected(indices.begin(),
                                   indices.begin() + static_cast<ptrdiff_t>(width));
    std::sort(selected.begin(), selected.end());

    for (int64_t i = 0; i < width; ++i) {
        ids[i] = selected[static_cast<size_t>(i)];
    }
}

// ─── qsa_attend ─────────────────────────────────────────────────
void qsa_attend(const float* q, const uint16_t* k_scratch,
                const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& shapes, float* attn,
                float* weights, cudaStream_t /*stream*/) {
    if (!q || !attn) return;
    if (!shapes.valid()) return;

    const int64_t n_head = shapes.n_head;
    const int64_t head_dim = shapes.head_dim;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Empty ids ⟹ attn = 0
    if (n_ids <= 0 || !k_scratch || !v_scratch) {
        std::memset(attn, 0, static_cast<size_t>(n_head * head_dim) * sizeof(float));
        return;
    }

    for (int64_t h = 0; h < n_head; ++h) {
        // Map q head to KV head: division, not modulo
        const int64_t kv_h = shapes.kv_head(h);

        // Compute scores: <K_j[kv_h], q[h]> / sqrt(head_dim)
        std::vector<float> scores(static_cast<size_t>(n_ids));
        float max_score = -1e30f;

        for (int64_t j = 0; j < n_ids; ++j) {
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                float k_val = f32_from_f16(
                    k_scratch[(j * shapes.n_head_kv + kv_h) * head_dim + d]);
                dot += k_val * q[h * head_dim + d];
            }
            scores[static_cast<size_t>(j)] = dot * inv_sqrt_d;
            if (scores[static_cast<size_t>(j)] > max_score) {
                max_score = scores[static_cast<size_t>(j)];
            }
        }

        // Softmax
        float sum_exp = 0.0f;
        std::vector<float> w(static_cast<size_t>(n_ids));
        for (int64_t j = 0; j < n_ids; ++j) {
            w[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - max_score);
            sum_exp += w[static_cast<size_t>(j)];
        }
        for (int64_t j = 0; j < n_ids; ++j) {
            w[static_cast<size_t>(j)] /= sum_exp;
        }

        // Write weights if requested
        if (weights) {
            for (int64_t j = 0; j < n_ids; ++j) {
                weights[h * n_ids + j] = w[static_cast<size_t>(j)];
            }
        }

        // attn[h] = sum_j w[h,j] * V_j[kv_h]
        for (int64_t d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int64_t j = 0; j < n_ids; ++j) {
                float v_val = f32_from_f16(
                    v_scratch[(j * shapes.n_head_kv + kv_h) * head_dim + d]);
                val += w[static_cast<size_t>(j)] * v_val;
            }
            attn[h * head_dim + d] = val;
        }
    }
}

// ─── qsa_gate_apply ─────────────────────────────────────────────
void qsa_gate_apply(const float* attn, const float* q_full,
                    const QsaShapes& shapes, uint16_t* out,
                    cudaStream_t /*stream*/) {
    if (!attn || !q_full || !out) return;
    if (!shapes.valid()) return;

    const int64_t n_head = shapes.n_head;
    const int64_t head_dim = shapes.head_dim;

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
            // gate = SECOND half of q_full (not interleaved)
            const float gate = q_full[h * (2 * head_dim) + head_dim + d];
            // SIGMOID, not SiLU
            const float sig = 1.0f / (1.0f + std::exp(-gate));
            // Single rounding AFTER multiply
            out[h * head_dim + d] = f16_from_f32(attn[h * head_dim + d] * sig);
        }
    }
}

// ─── qsa_gate_apply_f32 ────────────────────────────────────────
void qsa_gate_apply_f32(const float* attn, const float* q_full,
                        const QsaShapes& shapes, float* out,
                        cudaStream_t /*stream*/) {
    if (!attn || !q_full || !out) return;
    if (!shapes.valid()) return;

    const int64_t n_head = shapes.n_head;
    const int64_t head_dim = shapes.head_dim;

    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
            const float gate = q_full[h * (2 * head_dim) + head_dim + d];
            const float sig = 1.0f / (1.0f + std::exp(-gate));
            out[h * head_dim + d] = attn[h * head_dim + d] * sig;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Capturable step variants
// ═══════════════════════════════════════════════════════════════

void qsa_index_step(const float* pooled, const float* q_idx,
                    const float* bias, const QsaShapes& shapes,
                    const int32_t* step, int64_t /*max_blocks*/,
                    float* cell_scores, cudaStream_t stream) {
    if (!step) return;
    const int64_t n_bid = step[kStepNBid];
    const int64_t n_kv  = step[kStepNKv];
    qsa_index(pooled, n_bid, q_idx, bias, shapes, n_kv, cell_scores, stream);
}

void topk_512_step(const float* cell_scores, const QsaShapes& shapes,
                   int64_t cap, const int32_t* step,
                   int32_t* ids, cudaStream_t stream) {
    if (!step) return;
    const int64_t n_kv = step[kStepNKv];
    topk_512(cell_scores, n_kv, shapes, cap, ids, stream);
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool,
                    const int32_t* page_table, const int32_t* ids,
                    const int32_t* step, int64_t /*max_ids*/,
                    const QsaShapes& shapes,
                    uint16_t* k_scratch, uint16_t* v_scratch,
                    cudaStream_t stream) {
    if (!step) return;
    const int64_t width = step[kStepWidth];
    kv_gather(k_pool, v_pool, page_table, ids, width, shapes,
              k_scratch, v_scratch, stream);
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch,
                     const uint16_t* v_scratch, const int32_t* step,
                     int64_t /*max_ids*/, const QsaShapes& shapes,
                     float* attn, float* weights, cudaStream_t stream) {
    if (!step) return;
    const int64_t width = step[kStepWidth];
    qsa_attend(q, k_scratch, v_scratch, width, shapes, attn, weights, stream);
}

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool,
                    const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur,
                    const QsaShapes& shapes, cudaStream_t stream) {
    if (!step) return;
    const int64_t pos = step[kStepPos];
    kv_append(k_pool, v_pool, page_table, pos, kcur, vcur, shapes, stream);
}

}  // namespace kernels
}  // namespace strata

