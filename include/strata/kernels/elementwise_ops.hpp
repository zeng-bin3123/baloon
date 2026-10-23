#pragma once
// strata/kernels/elementwise_ops.hpp — GPU-accelerated elementwise dispatch
// Target: Iluvatar BI-V100 (ivcore10)
//
// This header provides a dispatch layer that routes elementwise operations
// to GPU kernels when available, or falls back to CPU implementations.
//
// The dispatch layer sits between the session/layer code and the actual
// kernel implementations. For the GPU path, it calls the ix_ew_* kernels
// from ix_elementwise.cu. For CPU path, it calls the functions in
// elementwise.hpp (host fallback).
//
// Usage:
//   strata::kernels::ops::silu_inplace(x, n, stream, use_gpu);
//
// When use_gpu == true and stream != nullptr, dispatches to GPU kernel.
// Otherwise falls back to CPU implementation.

#include <cuda_runtime.h>
#include <cstdint>

namespace strata {
namespace kernels {
namespace ops {

// ─── GPU kernel availability flag ──────────────────────────────────
// Set by native_cuda_ops initialization. When false, all dispatchers
// use CPU fallback regardless of the use_gpu parameter.
extern bool g_gpu_kernels_loaded;

// ─── Dispatch functions ────────────────────────────────────────────
// Each function routes to GPU or CPU based on g_gpu_kernels_loaded and
// the use_gpu parameter.

void embedding_gather(const uint8_t* codes, const float* scales,
                      const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out,
                      cudaStream_t stream, bool use_gpu = true);

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
              float* gate, int64_t n_tokens, int64_t h_v,
              cudaStream_t stream, bool use_gpu = true);

void scale_inplace(float* x, int64_t n, float s,
                   cudaStream_t stream, bool use_gpu = true);

void add_inplace(float* dst, const float* src, int64_t n,
                 cudaStream_t stream, bool use_gpu = true);

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n,
                     cudaStream_t stream, bool use_gpu = true);

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n,
                      cudaStream_t stream, bool use_gpu = true);

void silu_inplace(float* x, int64_t n,
                  cudaStream_t stream, bool use_gpu = true);

void rms_norm_weighted(float* x, const float* w, int64_t rows,
                       int64_t cols, float eps,
                       cudaStream_t stream, bool use_gpu = true);

void doorbell_ring(uint32_t* d_seq,
                   cudaStream_t stream, bool use_gpu = true);

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq,
                   cudaStream_t stream, bool use_gpu = true);

void copy_from_mapped(float* dst, const float* src, int64_t n,
                      cudaStream_t stream, bool use_gpu = true);

void doorbell_publish(const float* x, const int32_t* ids,
                      const float* weights, int64_t n, int64_t k,
                      float* x_out, int32_t* ids_out, float* weights_out,
                      uint32_t* d_seq,
                      cudaStream_t stream, bool use_gpu = true);

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n,
                          cudaStream_t stream, bool use_gpu = true);

}  // namespace ops
}  // namespace kernels
}  // namespace strata

