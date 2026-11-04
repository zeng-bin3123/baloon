// device_kv_qsa_api.hpp - host API of the KV-Q8 cache kernels, the quantized-sparse-attention (QSA) step kernels and the
// activation quantizers. All launchers return 0 on success, -1 on invalid arguments, -2 on a CUDA error.
// "step" variants read the per-step scalars (pos, n_kv, n_sel) from a device scratch so a recorded launch sequence can be
// replayed after only a small upload (qsa_step_fill).
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace dk {

// ---------------- KV Q8 cache (group of 64 values: fp16 scale amax/127 + int8 codes in [-127,127]) ----------------
struct KvQ8Shape { int kv_heads; int head_dim; int page_size; int n_pages_table; };
// Append one token at position *d_pos (device scalar): src [kv_heads][head_dim] f32.
int kv_append_q8_step(const KvQ8Shape& s, const float* d_src, int8_t* d_codes, uint16_t* d_scales,
                      const int32_t* d_table, const int32_t* d_pos, void* st);
// Gather n_ids positions (device ids) into dst f32 [n_ids][kv_heads][head_dim]. n_ids_dev (optional) caps the count.
int kv_gather_q8_step(const KvQ8Shape& s, int n_ids_max, const int32_t* d_ids, const int32_t* d_n_ids,
                      const int8_t* d_codes, const uint16_t* d_scales, const int32_t* d_table, float* d_dst, void* st);

// ---------------- QSA (single kv head, half-precision caches stored as raw uint16 bits) ----------------
struct QsaShape { int n_heads; int head_dim; int index_heads; int index_dim; };
constexpr int kTopK = 512;

struct StepScalars { int32_t pos; int32_t n_kv; int32_t n_sel; int32_t pad; };
StepScalars* step_scratch();                                        // lazily allocated device scratch (per process)
int step_upload_raw(void* d_dst, const void* h_src, size_t bytes, void* st);
int step_upload(int32_t* d_dst, int32_t value, void* st);
int step_upload_width(int32_t* d_dst, const int32_t* h_src, int width, void* st);
int qsa_step_fill(int32_t pos, int32_t n_kv, int32_t n_sel, void* st);

// Append one K/V row (f32 [row_dim]) at position pos as fp16 bits.
int kv_append(const float* d_src, uint16_t* d_cache, int row_dim, int pos, void* st);
int kv_append_step(const float* d_src, uint16_t* d_cache, int row_dim, void* st);
int indexer_key_append(const float* d_src, uint16_t* d_keys, int index_dim, int pos, void* st);
// Gather rows d_ids[0..n) of cache [*, row_dim] (fp16 bits) into f32 dst [n][row_dim].
int kv_gather(const uint16_t* d_cache, const int32_t* d_ids, int n, int row_dim, float* d_dst, void* st);
int kv_gather_step(const uint16_t* d_cache, const int32_t* d_ids, int n_max, int row_dim, float* d_dst, void* st);
// Indexer score of every key j < n_kv: sum_h hw[h] * relu(dot(iq[h], key_j)).   iq [index_heads][index_dim].
int qsa_index(const QsaShape& s, const float* d_iq, const float* d_hw, const uint16_t* d_keys, int n_kv, float* d_scores,
              void* st);
int qsa_index_step(const QsaShape& s, const float* d_iq, const float* d_hw, const uint16_t* d_keys, int n_kv_max,
                   float* d_scores, void* st);
// Top-512 of scores[0..n): ids ascending, n_out = min(512, n). Ties at the threshold take the smallest indices.
int topk_512(const float* d_scores, int n, int32_t* d_ids, int32_t* d_n_out, void* st);
int topk_512_step(const float* d_scores, int32_t* d_ids, int32_t* d_n_out, void* st);
// Attention over the selected rows: out [n_heads][head_dim] = softmax(scale * q.K[id]) V[id].
int qsa_attend(const QsaShape& s, const float* d_q, const uint16_t* d_k, const uint16_t* d_v, const int32_t* d_ids,
               int n_sel, float scale, float* d_out, void* st);
int qsa_attend_step(const QsaShape& s, const float* d_q, const uint16_t* d_k, const uint16_t* d_v, const int32_t* d_ids,
                    float scale, float* d_out, void* st);
// out[i] = attn[i] * sigmoid(gate[i]) (gate as fp16 bits, or f32).
int qsa_gate_apply(const float* d_attn, const uint16_t* d_gate, float* d_out, int n, void* st);
int qsa_gate_apply_f32(const float* d_attn, const float* d_gate, float* d_out, int n, void* st);

// ---------------- activation quantizers ----------------
constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;  // Q8_K: f32 d + int8[256] + int16 bsums[16]
constexpr int Q8_0_BYTES = 34;                                       // fp16 d + int8[32]
int quantize_q8_0(const float* d_x, uint8_t* d_out, int n, void* st);                 // n % 32 == 0
int quantize_q8_0_scaled(const float* d_x, float scale, uint8_t* d_out, int n, void* st);
int quantize_q8_K(const float* d_x, uint8_t* d_out, int n, void* st);                 // n % 256 == 0
int dequant_q8_0(const uint8_t* d_in, float* d_x, int n, void* st);
int dequant_q8_K(const uint8_t* d_in, float* d_x, int n, void* st);

}  // namespace dk
}  // namespace baloon
