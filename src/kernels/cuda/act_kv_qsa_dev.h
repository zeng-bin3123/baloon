// act_kv_qsa_dev.h - host entry points of the activation-quantisation, Q8 KV-cache and fp16 QSA step kernels.
// Pure declarations (no CUDA headers) so tests and drivers can include it. Return codes: 0 ok, -1 rejected by host
// validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {

// ---- Q8 KV cache (int8 codes + one fp16 scale per 64-value group; layout of baloon/kv_q8_quantization.hpp) ----
namespace kvq8dev {
int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos,
                      int kv_heads, int head_dim, int page_size, int table_len, int64_t n_pages);
int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, const int32_t* ids,
                      int n_ids, int kv_heads, int head_dim, int page_size, int table_len, int64_t n_pages,
                      float* out);
}  // namespace kvq8dev

// ---- activation quantisation: Q8_0 (32-value blocks, fp16 scale, 34 bytes) and Q8_K (256-value, 292 bytes) ----
namespace actq {
constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;  // float d; int8 qs[256]; int16 bsums[16]
constexpr int Q8_0_BYTES = 34;  // fp16 d; int8 qs[32]
int quantize_q8_0(const float* x, uint8_t* out, int64_t n_blocks);
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* out, int64_t n_blocks);  // quantises x[i]*scale
int quantize_q8_K(const float* x, uint8_t* out, int64_t n_blocks);
int dequant_q8_0(const uint8_t* in, float* y, int64_t n_blocks);
int dequant_q8_K(const uint8_t* in, float* y, int64_t n_blocks);
}  // namespace actq

// ---- fp16 KV + indexer + top-512 + single-query attention, with a device-resident step record ----
namespace qsak {
constexpr int kTopK = 512;
constexpr int kMaxIds = 512;
constexpr int kKeyDim = 128;
constexpr int kIdxHeads = 4;
// Per-step scalars kept on the device so a captured graph can be replayed without re-recording.
struct StepArgs { int32_t pos; int32_t n_kv; int32_t width; int32_t n_ids; };
void qsa_step_fill(StepArgs& s, int pos, int n_kv, int width, int n_ids);
StepArgs* step_scratch();  // device pointer, allocated on first use (nullptr on failure)
int step_upload_raw(const void* src, size_t bytes, size_t offset);
int step_upload(int pos, int n_kv);
int step_upload_width(int width, int n_ids);

int kv_append(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int pos, int dim, int n_cells);
int kv_append_step(const float* k, const float* v, uint16_t* kc, uint16_t* vc, int dim, int n_cells);
int kv_gather(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_ids, int dim, int n_cells,
              float* ko, float* vo);
int kv_gather_step(const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int max_ids, int dim, int n_cells,
                   float* ko, float* vo);
int indexer_key_append(const float* key, float* keys, int pos, int max_cells);
// score(c) = sum_h hw[h] * relu(q_h . keys[c]) for c < n_kv, -inf for c in [n_kv, cap)
int qsa_index(const float* q, const float* hw, const float* keys, float* scores, int n_kv, int cap);
int qsa_index_step(const float* q, const float* hw, const float* keys, float* scores, int cap);
// k = min(512, n) highest scores (ties: lower index first); ids ascending, padded with -1 up to 512; n_out = k
int topk_512(const float* scores, int n, int32_t* out_ids, int32_t* n_out);
int topk_512_step(const float* scores, int32_t* out_ids, int32_t* n_out);
int qsa_attend(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_ids, int n_cells,
               int dim, float scale, float* out);
int qsa_attend_step(const float* q, const uint16_t* kc, const uint16_t* vc, const int32_t* ids, int n_cells, int dim,
                    float scale, float* out);
int qsa_gate_apply(float* out, const uint16_t* gate_f16, int n);   // out *= sigmoid(gate)
int qsa_gate_apply_f32(float* out, const float* gate, int n);
}  // namespace qsak
}  // namespace baloon
