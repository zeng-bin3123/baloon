// s_gemv_block_row.cu - block-per-row S-family GEMV, FP16 activations. Thread t takes quads (four consecutive elements)
// q = t, t + tpr, ...; a quad lies inside one scale group and one activation block because the group size is a multiple
// of 16. Per quad the four products are added as ((p0 + p1) + (p2 + p3)) and multiplied by the group scale.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/f16_bits.hpp"
#include "baloon/s_gemv_block_row.hpp"

namespace baloon {
namespace kern {

namespace {

__device__ __constant__ int8_t kBlockRowIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

__device__ __forceinline__ float decode_code(int bits, uint32_t code) {
    return bits == 2 ? (float)((int)code - 1) : bits == 4 ? (float)kBlockRowIq4Nl[code & 15] : (float)(int)(int8_t)code;
}

__global__ void s_gemv_block_row_kernel(int bits, const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                        const uint16_t* __restrict__ x, float* __restrict__ y, int n_in, int n_out,
                                        int group_shift, int tpr) {
    extern __shared__ float partial[];
    const int row = blockIdx.x;
    if (row >= n_out) return;
    const int t = threadIdx.x;
    const uint8_t* crow = codes + (size_t)row * n_in * bits / 8;
    const uint16_t* srow = scales + (size_t)row * (n_in >> group_shift);
    const uint32_t mask = (1u << bits) - 1u;
    float acc = 0.f;
    for (int q = t; q < n_in / 4; q += tpr) {
        float p[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const long long i = 4ll * q + k, bit = i * bits;
            const uint32_t code = ((uint32_t)crow[bit >> 3] >> (bit & 7)) & mask;
            p[k] = decode_code(bits, code) * f16bits::f32_from_f16(x[i]);
        }
        acc += f16bits::f32_from_f16(srow[(4 * q) >> group_shift]) * ((p[0] + p[1]) + (p[2] + p[3]));
    }
    partial[t] = acc;
    __syncthreads();
    for (int step = tpr >> 1; step > 0; step >>= 1) {
        if (t < step) partial[t] += partial[t + step];
        __syncthreads();
    }
    if (t == 0) y[row] = partial[0];
}

}  // namespace

cudaError_t s_gemv_block_row(int bits, const uint8_t* codes, const uint16_t* scales, const uint16_t* x_f16, float* y,
                             int n_in, int n_out, int group_shift, int threads_per_row, cudaStream_t stream) {
    if (!codes || !scales || !x_f16 || !y) return cudaErrorInvalidValue;
    if (bits != 2 && bits != 4 && bits != 8) return cudaErrorInvalidValue;
    if (group_shift < 4 || group_shift > 6) return cudaErrorInvalidValue;
    if (n_in <= 0 || n_out <= 0 || n_in % 16 != 0 || n_in % (1 << group_shift) != 0) return cudaErrorInvalidValue;
    if (threads_per_row < 1 || threads_per_row > 1024 || (threads_per_row & (threads_per_row - 1)) != 0) return cudaErrorInvalidValue;
    if (((uintptr_t)codes & 3) != 0) return cudaErrorInvalidValue;
    s_gemv_block_row_kernel<<<n_out, threads_per_row, (size_t)threads_per_row * sizeof(float), stream>>>(
        bits, codes, scales, x_f16, y, n_in, n_out, group_shift, threads_per_row);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
