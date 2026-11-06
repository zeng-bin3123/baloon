// q2_avx2_rows.hpp - Q2_0 rows in the GGUF block layout (18 bytes per 64 weights: fp16 scale, then 16 code bytes)
// and the activation quantizer, for CPUs without AVX-512. Implemented in src/cpu/q2_avx2_rows.cpp, which must be
// built with -mavx2 -mf16c -mfma. This header only declares, so it compiles with any flags.
//
// Layout facts (Layer 1):
//  * value i of a 64-value block is in code byte i/4 at bit 2*(i%4); values 0..31 come from bytes 0..7, 32..63 from 8..15
//  * block b of a row starts at 18*b (scale at +0..1, codes at +2..17); value i of block b uses activation chunk 2*b + i/32
//  * token groups hold at most 4 tokens
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace q2rows {

constexpr uint32_t BLOCK_BYTES = 18;
constexpr uint32_t BLOCK_VALUES = 64;
constexpr uint32_t CHUNK = 32;
constexpr uint32_t MAX_GROUP = 4;

constexpr uint32_t code_byte(uint32_t i) { return i / 4; }
constexpr uint32_t code_shift(uint32_t i) { return 2 * (i % 4); }
constexpr uint64_t block_start(uint64_t b) { return (uint64_t)BLOCK_BYTES * b; }
constexpr uint32_t chunk_of(uint32_t b, uint32_t i) { return 2 * b + i / CHUNK; }
constexpr uint32_t group_size(uint32_t nt, uint32_t t0) { return nt - t0 < MAX_GROUP ? nt - t0 : MAX_GROUP; }

// True when the translation unit was built with AVX2 + F16C + FMA (the kernels below require them).
bool built_with_required_isa();

// Quantize n floats (n % 32 == 0): per 32-value chunk d = amax/127, q = clamp(rne(x/d), -127, 127), hx = sum of q.
void activation_quantizer(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);

// Expand 16 code bytes to the 64 codes (0..3) in value order.
void unpack_codes_64(const uint8_t* codes16, uint8_t out[64]);

// Dot of one row (nb blocks) against n_tok (1..4) quantized activations; out[t] for t < n_tok.
void multi_token_row_dot(const uint8_t* row, uint32_t nb, uint32_t n_tok, const int8_t* q, size_t q_stride,
                         const float* xd, const int32_t* hx, size_t ch_stride, float* out);

// Single token: out[r - row0] for rows [row0, row1) of a row-major matrix with nb blocks per row.
void row_range(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, const int8_t* q, const float* xd,
               const int32_t* hx, float* out);

// Rows [row0, row1) against nt tokens (in groups of at most 4); out[t * out_stride + r].
void gguf_rows_multi(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                     size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride);


// Reference-named entry points (same kernels, flat C++ names).
// act_quant_q8_1_avx2: activation quantizer; q2_0_gguf_rows_multi_avx2: rows [row0,row1) x nt tokens in groups of <= 4.
void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                               size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out,
                               size_t out_stride);

}  // namespace q2rows
}  // namespace baloon
