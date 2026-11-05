// kern_native_mmvq.hpp - native quantized matrix-vector kernels (weights in ggml block formats x Q8_1 activations).
// y[col * n_out + row] = sum_i w[row][i] * x[col][i], columns (1..8) quantized as one vector: column c starts at
// activation block c * (n_in / 32). Weight rows are contiguous: row r starts at r * (n_in / block_elems) * block_bytes.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace kern {

// Types: 2 Q4_0, 6 Q5_0, 8 Q8_0, 20 IQ4_NL, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 23 IQ4_XS, 42 Q2_0.
bool native_mmvq_supported(int type);
// Bytes of an n_out x n_in weight matrix (0 when the type or shape is unknown).
size_t native_mmvq_weight_bytes(int type, int n_in, int n_out);
// Bytes of the Q8_1 image of ncols columns of n_in values: ncols * (n_in / 32) * 36.
size_t native_q8_1_bytes(int ncols, int n_in);
// Quantize n values (n % 32 == 0) to Q8_1 blocks (one thread per element, 256 threads per CUDA block).
cudaError_t native_quantize_q8_1(const float* x, void* out_q8_1, int64_t n, cudaStream_t stream);

// Same column result whatever the number of columns that ride along (default true). When false, 5..8 columns use the
// upstream 2-warp layout (a different summation order).
void native_mmvq_set_multi_exact(bool on);
bool native_mmvq_multi_exact();

// Dispatcher over the weight type; x_q8_1 is already quantized.
cudaError_t native_mmvq(int type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);

#define BALOON_DECL_MMVQ(name)                                                                                          \
    cudaError_t native_##name##_mmvq(const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,       \
                                     cudaStream_t s);                                                                   \
    /* float activations: quantized into scratch (native_q8_1_bytes(ncols, n_in) bytes) first */                         \
    cudaError_t native_##name##_f32(const void* w, const float* x, float* y, int n_in, int n_out, int ncols,            \
                                    void* scratch_q8_1, cudaStream_t s);
BALOON_DECL_MMVQ(q2_0)
BALOON_DECL_MMVQ(q4_0)
BALOON_DECL_MMVQ(q5_0)
BALOON_DECL_MMVQ(q8_0)
BALOON_DECL_MMVQ(iq4_nl)
BALOON_DECL_MMVQ(q3_k)
BALOON_DECL_MMVQ(q4_k)
BALOON_DECL_MMVQ(q5_k)
BALOON_DECL_MMVQ(q6_k)
BALOON_DECL_MMVQ(iq4_xs)
#undef BALOON_DECL_MMVQ

}  // namespace kern
}  // namespace baloon
