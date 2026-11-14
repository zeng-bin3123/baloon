// kern_mmvq_types.hpp - explicit per-type MMVQ kernel declarations (macro-expanded from kern_native_mmvq.hpp).
// Provides literal-name declarations for every quantization type so the port map can reference them directly.
// All functions live in baloon::kern. Signatures match the BALOON_DECL_MMVQ expansion in kern_native_mmvq.hpp.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace baloon {
namespace kern {

// Q2_0: 2-bit symmetric, block 32
cudaError_t native_q2_0_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q2_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q4_0: 4-bit symmetric, block 32
cudaError_t native_q4_0_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q4_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q5_0: 5-bit symmetric, block 32
cudaError_t native_q5_0_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q5_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q8_0: 8-bit symmetric, block 32
cudaError_t native_q8_0_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q8_0_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// IQ4_NL: 4-bit non-linear codebook, block 32
cudaError_t native_iq4_nl_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_iq4_nl_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q3_K: 3-bit grouped K-quant
cudaError_t native_q3_k_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q3_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q4_K: 4-bit grouped K-quant
cudaError_t native_q4_k_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q4_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q5_K: 5-bit grouped K-quant
cudaError_t native_q5_k_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q5_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// Q6_K: 6-bit grouped K-quant
cudaError_t native_q6_k_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_q6_k_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

// IQ4_XS: 4-bit extra-small non-linear, block 256
cudaError_t native_iq4_xs_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);
cudaError_t native_iq4_xs_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols, void* scratch_q8_1, cudaStream_t s);

}  // namespace kern
}  // namespace baloon

