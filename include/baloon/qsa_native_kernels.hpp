// qsa_native_kernels.hpp - host API of the split-K decode attention, the short-context flash kernel, the tensor-tile
// indexer scorer and the int8 KV cache. Pure C++ (no CUDA headers). Return codes as in qsa_kernels.hpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include "baloon/qsa_kernels.hpp"

namespace baloon {
namespace qsa_dev {

// ---- split-K decode attention (GQA: 12 query heads per kv head, head_dim 256, chunks of 64 selected cells) --------------
// One KV pool per tensor, rows of 256 values: fp16 half bits (q8 == false) or int8 codes with float scales[row * 4 + d / 64].
// Row of cell c, kv head h: (page * n_kv_heads + h) * page_size + c % page_size, page = table ? table[c / page_size] : c / page_size.
struct QsaAttnPools {
    const void* k;
    const void* v;
    const float* k_scale;  // q8 only
    const float* v_scale;  // q8 only
    const int32_t* table;  // may be null (identity page map)
    int page_size;
    int n_kv_heads;
    bool q8;
};
// Floats of scratch one query needs: chunks * n_head * (256 + 2) + 64, chunks = ceil(cap / 64), n_head = 12 * n_kv_heads.
size_t qsa_decode_attn_scratch_floats(int cap, int n_head);
// Host-side definition shared with the tests (qsa_decode_attn_scratch_floats forwards to it).
inline size_t scratch_floats_model(int cap, int n_head) { return (size_t)((cap + 63) / 64) * (size_t)n_head * 258 + 64; }
//   q: [n_head][256] (scaled by `scale` inside), ids: [cap], out: [n_head][256]; the number of valid ids is step->width.
int qsa_decode_attn_step(const QsaAttnPools& pools, const float* q, const int32_t* ids, const Step* step, int cap, float scale,
                         float* scratch, float* out, void* stream);
//   nq queries (grid z): q [nq][n_head][256], ids [nq][cap], n_ids[nq], scratch nq * scratch_floats, out [nq][n_head][256].
int qsa_decode_attn_batch(const QsaAttnPools& pools, const float* q, const int32_t* ids, const int32_t* n_ids, int nq, int cap,
                          float scale, float* scratch, float* out, void* stream);

// ---- short-context flash attention (Q 24 x 256, KV 2 x 256 fp16, <= 256 cells, one 128-thread block per head) ------------
// K/V pools: [max_context][2][256] half bits. Valid only if 1 <= width <= max_context, n_kv == width, pos == width - 1,
// n_bid == width / 4 (read from the device step); otherwise the output is filled with NaN and *status is set to 1.
struct Span { uint64_t addr, bytes; };
bool validate_spans(const Span* spans, int n, bool* writes);
// Host-side definition shared with the tests (validate_spans forwards to it): non-null, 4-aligned sizes, no address wrap and
// no write span overlapping any other span (read/read overlap is fine).
inline bool spans_ok_model(const Span* spans, int n, const bool* writes) {
    for (int i = 0; i < n; ++i) {
        if (spans[i].addr == 0 || spans[i].addr % 4 != 0 || spans[i].bytes % 4 != 0) return false;
        if (spans[i].addr + spans[i].bytes < spans[i].addr) return false;
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (!(writes[i] || writes[j])) continue;
            if (spans[i].addr < spans[j].addr + spans[j].bytes && spans[j].addr < spans[i].addr + spans[i].bytes) return false;
        }
    return true;
}
int native_flash_attn_short_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const Step* step, int max_context,
                                 float* out, int32_t* status, void* stream);

// ---- tensor-tile scorer (m16n8k8 tf32 tiles) ------------------------------------------------------------------------------
// Per-cell scores: cell j of block row j/4 gets sum_h hw[h] * relu(q_h . pooled[row]) (spare row +1e9 iff n % 4 != 0).
struct ScoreArgs {
    const float* pooled;   // [rows][128]
    const float* iq;       // [4][128]
    const float* hw;       // [4]
    float* cell_score;     // [max_cells]
    const Step* step;      // device step record
    int rows, max_cells;
};
bool native_qsa_score_enabled();
void native_qsa_score_set_enabled(bool on);
// Returns kInvalid when disabled, misaligned, out of bounds or overlapping spans; nothing is launched then.
int native_qsa_score(const ScoreArgs& a, void* stream);

// ---- int8 KV cache (per 64-value group scale) ----------------------------------------------------------------------------
// Pools: int8 [rows][256], scales float [rows][4]; same row mapping as the fp16 cache.
int kv_append_q8_step(const float* k, const float* v, int8_t* kpool, int8_t* vpool, float* kscale, float* vscale,
                      const int32_t* table, const Step* step, const Shapes& sh, void* stream);
int kv_gather_q8_step(const int8_t* pool, const float* scale, const int32_t* table, const int32_t* ids, const Step* step,
                      const Shapes& sh, float* out, void* stream);

}  // namespace qsa_dev
}  // namespace baloon
