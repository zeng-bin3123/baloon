// mmvq_oracle.hpp - a deliberately plain device-side reference for the block-format x Q8_1 matrix-vector product.
// One thread computes one output y[col * n_out + row]; no tiling, no warp reductions, no vector loads: the value is the
// block-ordered sum  sum_b f16(d_w) * f16(d_x) * (sum_i qw_i * qx_i)  accumulated in float. It exists to cross-check the tuned
// kernels (native_mmvq) on the device, independent of their summation order. Types: 8 = Q8_0 (34 B / 32 values),
// 2 = Q4_0 (18 B / 32 values, low nibbles are values 0..15, high nibbles 16..31, both minus 8). Activations are Q8_1 blocks
// of 36 bytes (f16 d, f16 d*sum, int8[32]); column c starts at block c * (n_in / 32).
#pragma once
#include <cuda_runtime.h>

#include <cstdint>

namespace baloon {
namespace kg {

bool mmvq_oracle_supported(int type);
// Asynchronous on `s`. Returns cudaErrorInvalidValue for an unsupported type, null pointers, ncols < 1, n_in <= 0,
// n_out <= 0 or n_in % 32 != 0.
cudaError_t mmvq_oracle(int type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s);

}  // namespace kg
}  // namespace baloon
