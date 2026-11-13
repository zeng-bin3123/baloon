// u11_expert.hpp - reference-layout CPU expert surface: expert blob layout, format traits, backend probe and the Q2_0 AVX2 rows.
#pragma once
#include <cstddef>
#include <cstdint>

#include "baloon/cpu_native_expert.hpp"

namespace baloon {
namespace u11 {

namespace layout {   // src/kernels/cpu/expert_layout.cpp
ExpertLayout expert_layout(int type, int64_t n_embd, int64_t n_ff);
bool expert_layout_load(ExpertLayout& out, int type, int64_t n_embd, int64_t n_ff, uint64_t blob_bytes);
bool cpu_avx512_ok();
void act_quant_any(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);
bool q2_rows_any(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q, size_t q_stride,
                 const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride);
NativeFmt native_fmt(int ggml_type);
bool native_experts_available(int ggml_type, int64_t n_embd, int64_t n_ff);
bool native_quant_act(NativeFmt f, const float* x, uint32_t n_embd, uint32_t nt, void* out);
bool native_quant_h(NativeFmt f, const float* h, uint32_t n_ff, uint32_t nt, void* out);
bool native_gu_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1, const void* xq,
                    uint32_t nt, float* act, size_t act_stride);
bool native_down_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1, const void* hq,
                      uint32_t nt, float* out, size_t out_stride);
}  // namespace layout

namespace native {   // src/kernels/cpu/native_expert.cpp
struct FmtTraits { NativeFmt fmt; int type_id; int64_t block_values, block_bytes; uint32_t act_block; bool q8k; bool valid; };
FmtTraits traits(NativeFmt f);
void init_once();                 // one-time CPU feature probe (idempotent)
int backend();                    // 0 none, 1 avx2, 2 avx512
NativeFmt native_fmt(int ggml_type);
bool native_experts_available(int ggml_type, int64_t n_embd, int64_t n_ff);
bool native_quant_act(NativeFmt f, const float* x, uint32_t n_embd, uint32_t nt, void* out);
bool native_quant_h(NativeFmt f, const float* h, uint32_t n_ff, uint32_t nt, void* out);
bool native_gu_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1, const void* xq,
                    uint32_t nt, float* act, size_t act_stride);
bool native_down_rows(NativeFmt f, const uint8_t* blob, uint32_t n_embd, uint32_t n_ff, uint32_t r0, uint32_t r1, const void* hq,
                      uint32_t nt, float* out, size_t out_stride);
}  // namespace native

namespace q2avx2 {   // src/kernels/cpu/q2_avx2.cpp  (built with -mavx2 -mf16c -mfma)
void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx);
void unpack64(const uint8_t* codes16, uint8_t out[64]);
void block_sums(const uint8_t* blk, const int8_t* qb, int32_t S[2]);
int32_t hsum8(const int32_t v[8]);
void row_multi(const uint8_t* row, uint32_t nb, uint32_t n_tok, const int8_t* q, size_t q_stride, const float* xd,
               const int32_t* hx, size_t ch_stride, float* out);
void rows(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx, float* out);
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                               size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride);
}  // namespace q2avx2

}  // namespace u11
}  // namespace baloon
