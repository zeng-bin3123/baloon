// s2_gemv_quads.cu - Q2_0 GEMV that works on whole code bytes ("quads" of four 2-bit codes): lane l of a wave owns byte l,
// l+64, ... of the row's flat code stream; every block contributes 16 bytes, so lanes >= 16 of a 64-lane stride land in later blocks.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int CODES_PER_BYTE_S2 = 4;
constexpr int QK_S2 = 64;
constexpr int BYTES_PER_BLOCK = QK_S2 / CODES_PER_BYTE_S2;     // 16

__global__ void s2_gemv_quads_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 63, nb = n_in / QK_S2, nbytes = nb * BYTES_PER_BLOCK;
    const uint8_t* rp = w + (size_t)row * nb * 18;
    float acc = 0.f;
    for (int u = lane; u < nbytes; u += 64) {
        const int b = u / BYTES_PER_BLOCK, k = u % BYTES_PER_BLOCK;
        const uint8_t* p = rp + (size_t)b * 18;
        const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
        const uint8_t byte = p[2 + k];
        const float* xv = x + (size_t)b * QK_S2 + k * CODES_PER_BYTE_S2;
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < CODES_PER_BYTE_S2; ++j) a += (float)(((byte >> (2 * j)) & 3) - 1) * xv[j];
        acc += d * a;
    }
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[row] = acc;
}
}  // namespace

cudaError_t s2_gemv_quads(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (!w || !x || !y || n_in <= 0 || n_in % QK_S2 || n_out <= 0) return cudaErrorInvalidValue;
    s2_gemv_quads_kernel<<<(n_out + 3) / 4, 256, 0, s>>>(w, x, y, n_in, n_out);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
