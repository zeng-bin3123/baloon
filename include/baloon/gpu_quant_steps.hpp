// gpu_quant_steps.hpp - host API of three BI-V100 kernel files: activation quantisation (Q8_0 / Q8_K),
// the paged int8 KV cache (append / gather) and the single-token QSA decode step (fp16 KV pool, indexer score,
// top-512 selection, attention over the selected cells, output gate). Pure C++ (no CUDA headers).
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
// `stream` is a cudaStream_t passed as void* (nullptr = default stream).
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {

// ---- activation quantisation ---------------------------------------------------------------------------------
//   Q8_0 block (34 B): fp16 d = amax/127, then 32 int8 codes, code = nearest_even(x / d) clamped to +-127.
//   Q8_K block (292 B): float d, 256 int8 codes, int16 bsums[16]; d = 1/iscale, iscale = -127/(signed value of
//   largest magnitude), code = min(127, nearest(iscale*x)), bsums = sums of 16 consecutive codes.
namespace actq {
constexpr int kQ80Elems = 32, kQ80Bytes = 34, kQ8KElems = 256, kQ8KBytes = 292;
int quantize_q8_0(const float* x, uint8_t* y, int64_t nblocks, void* stream);
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* y, int64_t nblocks, void* stream);  // codes of x*scale
int quantize_q8_K(const float* x, uint8_t* y, int64_t nblocks, void* stream);
int dequant_q8_0(const uint8_t* y, float* out, int64_t nblocks, void* stream);
int dequant_q8_K(const uint8_t* y, float* out, int64_t nblocks, void* stream);
}  // namespace actq

// ---- paged int8 KV cache -------------------------------------------------------------------------------------
// Layout and arithmetic follow kv_q8_quantization.hpp (kvq8::append_address, quantize_group, gather_decompose).
namespace kvq8gpu {
struct Geom { int kv_heads; int head_dim; int page_size; };  // head_dim is a multiple of 64
// Appends one token ([kv_heads][head_dim] floats on device) at the position read from *pos_dev. Positions outside
// the table or pages outside [0, n_pages) are ignored on the device (nothing written).
int kv_append_q8_step(const float* src, const Geom& g, int8_t* codes, uint16_t* scales, const int32_t* table,
                      int32_t table_len, int64_t n_pages, const int32_t* pos_dev, void* stream);
// out[n_ids][kv_heads][head_dim] floats; ids outside [0, table_len*page_size) produce zeros.
int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const Geom& g, const int32_t* table,
                      int32_t table_len, int64_t n_pages, const int32_t* ids, int32_t n_ids, float* out, void* stream);
}  // namespace kvq8gpu

// ---- QSA decode step -----------------------------------------------------------------------------------------
namespace qsa_step {
constexpr int kDim = 256;   // head dimension of the KV pool
constexpr int kTopK = 512;  // selected cells per step
struct StepParams { int32_t pos; int32_t n_cells; int32_t width; int32_t reserved; };
// Host-side fill: n_cells = clamp(pos + 1, 0, cap).
void qsa_step_fill(StepParams& p, int pos, int cap, int width);
const StepParams* step_scratch();  // per-device scratch (lazily allocated); nullptr on failure
int step_upload_raw(const void* host, size_t offset, size_t bytes, void* stream);
int step_upload(int pos, int cap, void* stream);   // writes pos and n_cells
int step_upload_width(int width, void* stream);    // writes width (<= kTopK)

// fp16 KV pools: [cap][n_kv][kDim]. Append converts f32 -> f16 (round-to-nearest-even) into cell *pos.
int kv_append_step(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, int n_kv, int cap, void* stream);
int kv_append(const float* k, const float* v, uint16_t* kpool, uint16_t* vpool, int n_kv, int cap, int pos, void* stream);
// Gather selected cells (ids on device, -1 = none) into out[kTopK][n_kv][kDim] floats (zeros for invalid ids).
int kv_gather_step(const uint16_t* pool, const int32_t* ids, int n_kv, int cap, float* out, void* stream);
int kv_gather(const uint16_t* pool, const int32_t* ids, int width, int n_kv, int cap, float* out, void* stream);
// Indexer: ikpool [cap][d_idx] f32; key appended at `pos` (host value).
int indexer_key_append(const float* key, float* ikpool, int d_idx, int cap, int pos, void* stream);
// scores[c] = sum_h w[h] * relu(dot(iq[h], ikpool[c])) for c < n_cells, -inf for n_cells <= c < cap.
int qsa_index_step(const float* iq, const float* w, const float* ikpool, int n_idx_heads, int d_idx, int cap,
                   float* scores, void* stream);
int qsa_index(const float* iq, const float* w, const float* ikpool, int n_idx_heads, int d_idx, int cap, int n_cells,
              float* scores, void* stream);
// ids[kTopK]: the cells holding the kTopK largest scores in ascending cell order (ties -> lowest cell), or all cells
// 0..n-1 followed by -1 when n <= kTopK.
int topk_512_step(const float* scores, int cap, int32_t* ids, void* stream);
int topk_512(const float* scores, int cap, int n_cells, int32_t* ids, void* stream);
// out[n_head][kDim]: softmax attention of q*scale over the first `width` ids (invalid ids skipped).
int qsa_attend_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* ids, int n_head,
                    int n_kv, int cap, float scale, float* out, void* stream);
int qsa_attend(const float* q, const uint16_t* kpool, const uint16_t* vpool, const int32_t* ids, int width, int n_head,
               int n_kv, int cap, float scale, float* out, void* stream);
// out = attn * sigmoid(gate), as f16 bits / as f32.
int qsa_gate_apply(const float* attn, const float* gate, uint16_t* out, int64_t n, void* stream);
int qsa_gate_apply_f32(const float* attn, const float* gate, float* out, int64_t n, void* stream);
}  // namespace qsa_step
}  // namespace baloon
