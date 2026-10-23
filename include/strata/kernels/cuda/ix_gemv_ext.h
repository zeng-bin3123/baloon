// ix_gemv_ext.h — Public C API for unit 2 GEMV CUDA kernels (BI-V100)
//
// Extends ix_kernels.h with:
//   - S-family GEMV: GPU-side structured GEMV (s_gemv, s_gemv_split, s_gemv_q8k)
//   - S2 × Q8_0 GEMV: direct quantized activation path
//   - S2 GEMV quads/fast: amortized decode + LUT optimization
//   - BF16 GEMV extensions: split, q8k, fused add, batch
//   - GGUF MMVQ: Q5_K, Q6_K, IQ4_XS quantization formats
//
// All kernels target ivcore10 (warp64).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════
// ix_s_family_gemv — Structured GEMV family (GPU)
// ═══════════════════════════════════════════════════════════════

// Generic structured GEMV: y[o] = Σ_i fp16(x[i]) * W[o,i]
// codebook_id: 0=Affine (code+bias), 1=Iq4Nl (non-linear table)
void ix_s_gemv(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    cudaStream_t stream);

// Split GEMV: same as above with multiple warps per row
void ix_s_gemv_split(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    int threads_per_row,
    cudaStream_t stream);

// Q8_K activation GEMV
void ix_s_gemv_q8k(
    const uint8_t* x_q8k, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════
// ix_s2_gemv_q8 — S2 (2-bit) × Q8_0 activation GEMV
// ═══════════════════════════════════════════════════════════════

// S2 × Q8_0 GEMV: one warp per row
void ix_s2_gemv_q8(
    const uint8_t* x_q8, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    cudaStream_t stream);

// S2 × Q8_0 GEMV: split across warps
void ix_s2_gemv_q8_split(
    const uint8_t* x_q8, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════
// ix_s2_gemv_quads — S2 optimized quads and fast GEMV
// ═══════════════════════════════════════════════════════════════

// S2 quads: amortized 4-at-a-time code + activation loading
void ix_s2_gemv_quads(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row,
    cudaStream_t stream);

// S2 fast: quads + shared-memory staging + constant LUT
// stage_x: 1 = stage activation in shared memory, 0 = read from global
void ix_s2_gemv_fast(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row, int stage_x,
    cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════
// ix_bf16_gemv_ext — Extended BF16 GEMV variants
// ═══════════════════════════════════════════════════════════════

// BF16 GEMV split across multiple warps per row
void ix_bf16_gemv_split(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    int threads_per_row,
    cudaStream_t stream);

// BF16 weights × Q8_K activations GEMV
void ix_bf16_gemv_q8k(
    const uint8_t* x_q8k, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    cudaStream_t stream);

// Fused BF16 GEMV + residual add
void ix_bf16_gemv_fused_add(
    const uint16_t* x_bf16, const uint16_t* w_bf16,
    const float* residual, float* y,
    int n_in, int n_out,
    cudaStream_t stream);

// Batch BF16 GEMV (multiple activations × same weights)
void ix_bf16_gemv_batch(
    const uint16_t** x_ptrs, const uint16_t* w_bf16,
    float* y, int n_in, int n_out,
    int batch, cudaStream_t stream);

// ═══════════════════════════════════════════════════════════════
// ix_gguf_mmvq — GGUF quantization format MMVQ
// ═══════════════════════════════════════════════════════════════

// Q5_K × Q8_1 GEMV (5-bit K-quant)
void ix_mmvq_q5k(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream);

// Q6_K × Q8_1 GEMV (6-bit K-quant)
void ix_mmvq_q6k(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream);

// IQ4_XS × Q8_1 GEMV (importance quantization, non-linear codebook)
void ix_mmvq_iq4xs(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

