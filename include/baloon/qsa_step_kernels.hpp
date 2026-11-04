// qsa_step_kernels.hpp - host API of the single-token QSA decode step (src/qsa.cu): paged fp16 KV append/gather, indexer
// key append, per-cell index scoring, top-512 selection, attention over the selected cells and the output gate.
// Pure C++ (no CUDA headers). Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t.
// fp16 values are passed as raw uint16_t bits. All pointers are device pointers unless noted.
//
// Step state (device int32[8], owned by step_scratch()): [0] pos, [1] n_kv = pos + 1, [2] width = min(n_kv, 512),
// [3] n_sel (written by topk_512_step). The `_step` variants read their position / counts from it so the launches can be
// captured into a graph and replayed after a small upload.
//
// Paged cache row of cell c, kv head h: ((table[c / page_size] * kv_heads) + h) * page_size + c % page_size, rows of
// head_dim fp16 values. Cells whose page index is outside [0, n_pages) are skipped (append) / read as zeros (gather).
#pragma once
#include <cstdint>

namespace baloon {
namespace qsa_step {

constexpr int kTopK = 512;       // selected cells per step
constexpr int kStepInts = 8;
constexpr int kIndexDim = 128;   // indexer key width
constexpr int kIndexHeads = 4;
constexpr int kMaxHeadDim = 256;

// Order-preserving float -> uint32 map (larger float, larger key).
inline uint32_t order_key_host(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Debug: when on, every host wrapper synchronises its stream and reports asynchronous errors.
void set_debug_sync(bool on);
const char* last_error();   // message of the last -1 return on this thread

// ---- step state ----------------------------------------------------------------------------------------------------
int32_t* step_scratch();                                              // lazily allocated device int32[kStepInts]; null on failure
int step_upload_raw(const int32_t* host_vals, int n, void* stream);   // writes step[0..n) (n <= kStepInts)
int qsa_step_fill(void* stream);                                      // derives step[1], step[2] from step[0]
int step_upload(int32_t pos, void* stream);                           // step_upload_raw({pos}) + qsa_step_fill
int step_upload_width(int32_t width, void* stream);                   // overrides step[2], 1 <= width <= kTopK

// ---- paged fp16 KV ---------------------------------------------------------------------------------------------------
// k, v: [kv_heads][head_dim] floats -> fp16 caches (round to nearest even).
int kv_append(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, const int32_t* table, int64_t pos,
              int kv_heads, int head_dim, int page_size, int n_table, int n_pages, void* stream);
int kv_append_step(const float* k, const float* v, uint16_t* k_cache, uint16_t* v_cache, const int32_t* table, int kv_heads,
                   int head_dim, int page_size, int n_table, int n_pages, void* stream);   // pos = step[0]
// out: [n_ids][kv_heads][head_dim] floats; ids < 0 or >= n_table * page_size give zeros.
int kv_gather(const uint16_t* cache, const int32_t* table, const int32_t* ids, int n_ids, int kv_heads, int head_dim,
              int page_size, int n_table, int n_pages, float* out, void* stream);
int kv_gather_step(const uint16_t* cache, const int32_t* table, const int32_t* ids, int max_ids, int kv_heads, int head_dim,
                   int page_size, int n_table, int n_pages, float* out, void* stream);   // n_ids = step[3] (<= max_ids)

// ---- indexer ---------------------------------------------------------------------------------------------------------
// keys: [max_cells][128] floats. Stores the raw key of cell `pos`.
int indexer_key_append(const float* key, float* keys, int64_t pos, int max_cells, void* stream);
int indexer_key_append_step(const float* key, float* keys, int max_cells, void* stream);   // pos = step[0]
// score(c) = sum_h hw[h] * relu(q_h . key_c), q: [4][128], hw: [4], c in [0, n). scores: [n].
int qsa_index(const float* q, const float* hw, const float* keys, float* scores, int n, void* stream);
int qsa_index_step(const float* q, const float* hw, const float* keys, float* scores, int max_n, void* stream);  // n = step[1]
// Top-512 cells by score: out_ids = [512] ascending cell ids padded with -1 (all cells when n <= 512); ties keep the
// lowest indices. *out_n (device int, may be null) = number selected.
int topk_512(const float* scores, int n, int32_t* out_ids, int32_t* out_n, void* stream);
int topk_512_step(const float* scores, int32_t* out_ids, int32_t* out_n, void* stream);   // n = step[1]; also writes step[3]

// ---- attention + gate ------------------------------------------------------------------------------------------------
// q, out: [kv_heads * group][head_dim] floats (head_dim <= 256); query head i uses kv head i / group. Softmax over the
// valid selected cells (ids in [0, n_table*page_size)) of scale * q.k; heads with nothing valid output zeros.
int qsa_attend(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const int32_t* table, const int32_t* ids,
               const int32_t* n_sel, int kv_heads, int group, int head_dim, int page_size, int n_table, int n_pages,
               float scale, float* out, void* stream);
int qsa_attend_step(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const int32_t* table, const int32_t* ids,
                    int kv_heads, int group, int head_dim, int page_size, int n_table, int n_pages, float scale, float* out,
                    void* stream);   // n_sel = step[3]
// out[i] *= sigmoid(gate[i]); gate as fp16 bits / float.
int qsa_gate_apply(float* out, const uint16_t* gate, int64_t n, void* stream);
int qsa_gate_apply_f32(float* out, const float* gate, int64_t n, void* stream);

}  // namespace qsa_step
}  // namespace baloon
