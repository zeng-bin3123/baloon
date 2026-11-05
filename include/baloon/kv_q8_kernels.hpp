// kv_q8_kernels.hpp - device append / gather of the Q8 paged KV cache (src/ilu_kv_q8.cu). Pure C++ declarations.
// Row of (pos, head): (table[pos/page_size]*kv_heads + head)*page_size + pos%page_size (see kv_q8_quantization.hpp).
// codes: int8 [pool_rows][head_dim]; scales: fp16 bits [pool_rows][head_dim/64]. head_dim must be a multiple of 64.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstdint>

namespace baloon {
namespace kv_q8_dev {
// Quantize one token's [kv_heads][head_dim] floats (device) into row(pos, h) for every head. `table_len` bounds pos/page_size.
int kv_append_q8_step(const float* src, int8_t* codes, uint16_t* scales, const int32_t* table, int64_t table_len,
                      int64_t pos, int kv_heads, int head_dim, int page_size, int64_t pool_rows, void* stream);
// Dequantize the cells ids[0..n_ids) (positions) into out[n_ids][kv_heads][head_dim]; ids < 0 give zeros.
int kv_gather_q8_step(const int8_t* codes, const uint16_t* scales, const int32_t* table, int64_t table_len,
                      const int32_t* ids, int n_ids, int kv_heads, int head_dim, int page_size, int64_t pool_rows,
                      float* out, void* stream);
}  // namespace kv_q8_dev
}  // namespace baloon
