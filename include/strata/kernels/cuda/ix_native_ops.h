// ix_native_ops.h — Public C API for unit 8 native CUDA ops (BI-V100)
//
// Extends ix_kernels.h with:
//   - F16/BF16/FP32 type conversion kernels
//   - Activation quantization/dequantization (Q8_0, Q8_1)
//   - Elementwise GPU kernels (embedding gather, silu, scale, doorbell, etc.)
//
// All kernels are compiled with clang++ -x cuda for ivcore10.
// Call from C++ host code or Python ctypes.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════
// F16/BF16/FP32 Conversion Kernels
// ═══════════════════════════════════════════════════════════════════

// FP32 → FP16 (round-to-nearest-even, vectorized for large n)
void ix_cvt_f32_to_f16(const float* src, uint16_t* dst, int64_t n,
                        cudaStream_t stream);

// FP16 → FP32 (exact)
void ix_cvt_f16_to_f32(const uint16_t* src, float* dst, int64_t n,
                        cudaStream_t stream);

// BF16 → FP32 (exact)
void ix_cvt_bf16_to_f32(const uint16_t* src, float* dst, int64_t n,
                         cudaStream_t stream);

// FP16 → BF16 (via FP32, round-to-nearest-even)
void ix_cvt_f16_to_bf16(const uint16_t* src, uint16_t* dst, int64_t n,
                         cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// Activation Quantization / Dequantization
// ═══════════════════════════════════════════════════════════════════

// FP32 → Q8_1 quantization (per-32-element blocks, scale + sum)
void ix_activation_quantize_q8_1(const float* input, void* output,
                                  int64_t n, cudaStream_t stream);

// Q8_0 → FP32 dequantization
void ix_activation_dequant_q8_0(const void* input, float* output,
                                 int64_t n, cudaStream_t stream);

// Q8_1 → FP32 dequantization
void ix_activation_dequant_q8_1(const void* input, float* output,
                                 int64_t n, cudaStream_t stream);

// Fused SiLU + Q8_1 quantize (for MLP gate path)
void ix_silu_quantize_q8_1(const float* gate, const float* up,
                            void* output, int64_t n, cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════════
// Elementwise GPU Kernels
// ═══════════════════════════════════════════════════════════════════

// Embedding gather: dequantize packed codes
void ix_ew_embedding_gather(const uint8_t* codes, const float* scales,
                             const float* offsets, int64_t n, int code_bits,
                             int code_bias, int group_elems, float* out,
                             cudaStream_t stream);

// SiLU in-place: x[i] = x[i] / (1 + exp(-x[i]))
void ix_ew_silu_inplace(float* x, int64_t n, cudaStream_t stream);

// Scale in-place: x[i] *= s
void ix_ew_scale_inplace(float* x, int64_t n, float s, cudaStream_t stream);

// Add in-place: dst[i] += src[i]
void ix_ew_add_inplace(float* dst, const float* src, int64_t n,
                        cudaStream_t stream);

// RMSNorm weighted: y[r][c] = x[r][c] / sqrt(mean(x²) + eps) * w[c]
void ix_ew_rms_norm_weighted(float* x, const float* w, int64_t rows,
                              int64_t cols, float eps, cudaStream_t stream);

// Doorbell ring: atomic increment on mapped pinned memory
void ix_ew_doorbell_ring(uint32_t* d_seq, cudaStream_t stream);

// Doorbell wait: spin until *d_flag == *d_seq
void ix_ew_doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq,
                          cudaStream_t stream);

// Copy from mapped: kernel-based memcpy (not copy-engine)
void ix_ew_copy_from_mapped(float* dst, const float* src, int64_t n,
                             cudaStream_t stream);

// Copy int32 from mapped
void ix_ew_copy_i32_from_mapped(int32_t* dst, const int32_t* src,
                                 int64_t n, cudaStream_t stream);

// Doorbell publish: fused copy + fence + increment
void ix_ew_doorbell_publish(const float* x, const int32_t* ids,
                             const float* weights, int64_t n, int64_t k,
                             float* x_out, int32_t* ids_out,
                             float* weights_out, uint32_t* d_seq,
                             cudaStream_t stream);

// Bulk F32 → FP16
void ix_ew_f32_to_f16_bulk(const float* src, uint16_t* dst, int64_t n,
                             cudaStream_t stream);

// Bulk F32 → BF16
void ix_ew_f32_to_bf16_bulk(const float* src, uint16_t* dst, int64_t n,
                              cudaStream_t stream);

// GDN gate: softplus(alpha + dt) * ssm_a
void ix_ew_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t n_tokens, int64_t h_v,
                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

