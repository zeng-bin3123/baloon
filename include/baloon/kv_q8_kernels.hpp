// kv_q8_kernels.hpp - host API of the Q8 KV-cache append/gather kernels (src/kv_q8.cu). Contract = kv_q8_quantization.hpp:
// 64-value groups, fp16 scale = amax/127, int8 codes in [-127,127]; row = (page*kv_heads + head)*page_size + pos%page_size
// with page = table[pos/page_size]. Return codes: 0 ok, -1 rejected by host validation, >0 a cudaError_t.
// Device pointers: src/out/codes/scales/table/ids. Positions whose page index is outside [0, n_pages) are skipped (append)
// or gathered as zeros (gather); ids < 0 gather as zeros.
#pragma once
#include <cstdint>

namespace baloon {
namespace kvq8dev {

// One token: src = [kv_heads][head_dim] floats; head_dim a multiple of 64 (<= 1024).
// codes: [rows][head_dim] int8, scales: [rows][head_dim/64] fp16 bits.
int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t pos, int kv_heads,
                      int head_dim, int page_size, int n_table, int n_pages, void* stream);
// out = [n_ids][kv_heads][head_dim] floats, ids[i] = cell position of output slot i.
int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, const int32_t* ids, int n_ids,
                      int kv_heads, int head_dim, int page_size, int n_table, int n_pages, float* out, void* stream);

}  // namespace kvq8dev
}  // namespace baloon
