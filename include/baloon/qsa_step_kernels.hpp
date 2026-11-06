// qsa_step_kernels.hpp - device step state, fp16 paged KV append/gather, indexer pooling + scoring, one-block top-k,
// selected-cell decode attention and output gating (src/ilu_qsa_step.cu). Pure C++ declarations.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace qsa_step {

constexpr int kTopK = 2048, kBlockR = 4, kMaxWidth = kTopK + kBlockR - 1;  // 2051
constexpr int kKeyDim = 128, kIdxHeads = 4;

// Step buffer = four int32 {pos, n_kv, n_bid, width}; everything derives from pos.
struct Step { int32_t pos, n_kv, n_bid, width; };
inline Step make_step(int pos, int top_k = kTopK, int r = kBlockR) {
    const int n = pos + 1, cap = top_k + r - 1;
    return Step{pos, n, n / r, n < cap ? n : cap};
}

// Persistent device scratch of at least `bytes` (grown on demand, never shrunk). nullptr on failure.
void* step_scratch(size_t bytes);
// Upload the position, then let the device derive the other three entries.
int step_upload(int32_t* dev_step, int pos, int top_k, int r, void* stream);
int step_upload_raw(int32_t* dev_step, const Step& s, void* stream);       // four entries verbatim
int step_upload_width(int32_t* dev_step, int width, void* stream);          // overwrite the width entry only
int qsa_step_fill(int32_t* dev_step, int top_k, int r, void* stream);      // derive entries 1..3 from entry 0

// fp16 paged pool row (page*kv_heads + h)*page_size + pos%page_size, page = table[pos/page_size].
int kv_append(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len, int pos, int kv_heads,
              int head_dim, int page_size, int64_t pool_rows, void* stream);
int kv_append_step(const float* src, uint16_t* pool, const int32_t* table, int64_t table_len, int pos, int kv_heads,
                   int head_dim, int page_size, int64_t pool_rows, void* stream);
// out[n_ids][kv_heads][head_dim]; ids < 0 or out of the table give zeros.
int kv_gather(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids, int n_ids, int kv_heads,
              int head_dim, int page_size, int64_t pool_rows, float* out, void* stream);
int kv_gather_step(const uint16_t* pool, const int32_t* table, int64_t table_len, const int32_t* ids, int n_ids,
                   int kv_heads, int head_dim, int page_size, int64_t pool_rows, float* out, void* stream);

// Indexer: slots 0..r-2 go to tail[3][128]; slot r-1 mean-pools tail + key, RMS-normalises (eps) into pooled[pos/4].
int indexer_key_append(const float* key, float* tail, float* pooled, int pos, int max_cells, float eps, void* stream);
// cell_scores[j] = score(block min(j/4, n_bid)); score = sum_h hw[h]*relu(q_h . pooled_row) (+1e9 on the incomplete tail).
// pooled needs rows >= n_kv/4 + (n_kv%4 != 0); q: [4][128]; hw: [4].
int qsa_index(const float* pooled, const float* q, const float* hw, int n_kv, float* cell_scores, void* stream);
int qsa_index_step(const float* pooled, const float* q, const float* hw, int n_kv, int rows, float* cell_scores, void* stream);
// One block of 512 threads: ids of the `width` largest scores, ascending, ties to the lowest index, padded with -1;
// n_out receives the count (n if n <= width).
int topk_512(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream);
int topk_512_step(const float* scores, int n, int width, int32_t* out_ids, int32_t* n_out, void* stream);

// Decode attention over ids[0..n_sel) (positions; <0 masked): out[h] = softmax(scale * q_h . K) V, kv head = h/(n_head/n_kv).
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, int64_t table_len,
               const int32_t* ids, int n_sel, int n_head, int n_kv_heads, int head_dim, int page_size, int64_t pool_rows,
               float scale, float* out, void* stream);
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* table, int64_t table_len,
                    const int32_t* ids, int n_sel, int n_head, int n_kv_heads, int head_dim, int page_size,
                    int64_t pool_rows, float scale, float* out, void* stream);
// out[i] *= sigmoid(gate[i]); gate as fp16 bits or as float.
int qsa_gate_apply(float* out, const uint16_t* gate, int64_t n, void* stream);
int qsa_gate_apply_f32(float* out, const float* gate, int64_t n, void* stream);

}  // namespace qsa_step
}  // namespace baloon
