// qsa_kernels.hpp - host API of the step-driven QSA kernels (paged fp16 KV cache, pooled indexer keys, cell scores,
// exact top-k, selected-cell attention, output gate) plus block scoring/top-k, split-K decode attention, the
// short-context flash kernel, the tensor-tile scorer and the int8 KV cache. Pure C++ (no CUDA headers).
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
// Every kernel that depends on the token position reads it from a device-resident Step record (pos, n_kv, n_bid, width),
// so a captured graph replays with a new position by rewriting those 16 bytes only.
#pragma once
#include <cstddef>
#include <cstdint>
#include "baloon/qsa_step_state.hpp"

namespace baloon {
namespace qsa_dev {

using qsa_step::Shapes;
using qsa_step::Step;

constexpr int kOk = 0;
constexpr int kInvalid = -1;

// ---- step record ------------------------------------------------------------------------------------------------------
// Device scratch step record of `slot` (lazily allocated, one per slot).
Step* step_scratch(int slot);
// Upload a whole record / the raw 16 bytes / only the width field (asynchronously on `stream`).
int step_upload(const Step& s, Step* dev, void* stream);
int step_upload_raw(const void* host16, Step* dev, void* stream);
int step_upload_width(int width, Step* dev, void* stream);
// Fill the record for position `pos` (n_kv = pos + 1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1)) and upload it.
int qsa_step_fill(int pos, const Shapes& sh, Step* dev, void* stream);

// ---- paged fp16 KV cache: pools of uint16 half bits, row = (page * n_head_kv + h) * page_size + t % page_size ------------
//   table[t / page_size] is the physical page of logical page t / page_size.
int kv_append(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, const int32_t* table, int pos,
              const Shapes& sh, void* stream);
int kv_append_step(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, const int32_t* table,
                   const Step* step, const Shapes& sh, void* stream);
// out[(i * n_head_kv + h) * head_dim + d] = pool row of cell ids[i] (i < n) widened to f32.
int kv_gather(const uint16_t* pool, const int32_t* table, const int32_t* ids, int n, const Shapes& sh, float* out,
              void* stream);
int kv_gather_step(const uint16_t* pool, const int32_t* table, const int32_t* ids, const Step* step, const Shapes& sh,
                   float* out, void* stream);

// ---- indexer: pooled keys of 128 floats; raw tail of r - 1 rows ---------------------------------------------------------
//   pos % r != r-1 : tail[pos % r] = key;   pos % r == r-1 : pooled[pos / r] = mean(tail[0..r-2], key).
int indexer_key_append(const float* key, float* tail, float* pooled, int pos, int r, void* stream);

// Per-cell scores from per-block scores: cell_score[j] = block_score[min(j / r, n_bid)]  (j < n_kv).
int qsa_index(const float* block_score, float* cell_score, int n_kv, int r, void* stream);
int qsa_index_step(const float* block_score, float* cell_score, const Step* step, int cap, int r, void* stream);

// Exact top-`width` of n scores (one block of 512 threads, 4 radix passes). Output ids ascending, ties by ascending index;
// n <= width gives the identity. n_out receives the count written. n <= 32768.
int topk_512(const float* score, int n, int width, int32_t* ids, int32_t* n_out, void* stream);
int topk_512_step(const float* score, const Step* step, int32_t* ids, int32_t* n_out, void* stream);

// Attention of one token over the selected cells: out[h] = softmax(scale * q[h] . K[ids]) V[ids] for each of n_head heads,
// kv head = h / (n_head / n_head_kv). ids: `width` cells (width from the step for the _step form).
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, const int32_t* ids,
               int n_ids, float scale, const Shapes& sh, float* out, void* stream);
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table,
                    const int32_t* ids, const Step* step, float scale, const Shapes& sh, float* out, void* stream);

// out[i] *= sigmoid(gate[i]); gate as half bits / as f32.
int qsa_gate_apply(float* out, const uint16_t* gate, int n, void* stream);
int qsa_gate_apply_f32(float* out, const float* gate, int n, void* stream);

// ---- block scoring + block-weighted top-k (qsa_select.cu) ---------------------------------------------------------------
//   score(b) = sum_h hw[h] * relu(q_h . pooled[b]) for rows b <= n_bid; the spare row n_bid gets +1e9 iff n_kv % r != 0.
int qsa_block_scores(const float* pooled, const float* iq, const float* hw, float* scores, int n_kv, int r,
                     int rows, void* stream);
// Select `width` cells from block scores (block b weighs r cells, the spare row n_kv - n_bid * r). Cells ascending.
int qsa_block_topk(const float* scores, int n_kv, int r, int width, int32_t* ids, int32_t* n_out, void* stream);

}  // namespace qsa_dev
}  // namespace baloon
