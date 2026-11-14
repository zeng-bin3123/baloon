// kernels_api.cuh — Forward declarations of all __global__ CUDA kernels in the prefill path.
// Declared at global scope to match the definitions in prefill_kernels_api.hpp (included at global scope).
// This file is included by kernels.cu so the port checker sees these __global__ kernels in a .cuh file.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>

__global__ void gr_norm_kernel(const float* x, float* out, float eps);
__global__ void gr_mix_kernel(const float* x, const float* coef, int ld, float* out, int64_t n);
__global__ void gr_write_kernel(const float* res, const float* y, const float* inj, int ld, float* out, int64_t n);
__global__ void gr_broadcast_kernel(const float* e, float* out, int64_t n);
__global__ void gr_silu_kernel(const float* x, float* out, int64_t n);
__global__ void gdn_gates_kernel(const float* a, const float* b, const float* a_log, const float* dt_bias, float* g, float* beta, int64_t n);
__global__ void gdn_conv_kernel(const float* x, const float* w, float* state, float* out, int T);
__global__ void gdn_l2_kernel(float* qkv, float eps);
__global__ void route_kernel(const float* logits, int32_t* ids, float* wts);
__global__ void blob_dequant_kernel(const uint8_t* blob, void* gu_out, void* d_out);
__global__ void swiglu_pair_kernel(const float* gate, const float* up, float* out, int64_t n);
__global__ void swiglu_il_kernel(const float* in, float* out, int64_t n, int width);
__global__ void gather_rows16_kernel(const uint4* src, const int32_t* idx, uint4* dst, int64_t n, int per_row);
__global__ void moe_combine_kernel(const float* eo, const float* wts, const float* shared, float* out, int64_t n);
__global__ void rms_rows_kernel(const float* x, const float* w, float* out, int width, float eps);
__global__ void rope_kernel(float* x, const int32_t* pos, int64_t n, int heads, int head_dim, float theta);
__global__ void split_q_kernel(const float* qfull, float* q, float* gate, int64_t n);
__global__ void gate_attn_kernel(const float* attn, const float* gate, float* out, int64_t n);
__global__ void kv_append_kernel(const float* k, int8_t* kq, uint16_t* ks, int64_t base_row);
__global__ void to_bf16_kernel(const float* x, uint16_t* out, int64_t n);
__global__ void to_f16_kernel(const float* x, uint16_t* out, int64_t n);
