// s2_gemv_q8.cu - Q2_0 weights x Q8_0 activations. Activation block = fp16 d + 32 int8 (34 bytes / 32 elements). The integer
// dot sum((code-1) * q) is exact; the float result is sum over weight blocks of d_w * (d_x0 * dot0 + d_x1 * dot1).
// One wave per row; lane l handles element l of every 64-wide weight block (= element l of the first and second Q8 block).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;

__global__ void s2_gemv_q8_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ xq, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 63, nb = n_in / QK_S2;
    const uint8_t* rp = w + (size_t)row * nb * 18;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = rp + (size_t)b * 18;
        const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
        const int code = (p[2 + lane / 4] >> (2 * (lane % 4))) & 3;
        const uint8_t* xb = xq + (size_t)(2 * b + lane / QK8_0) * 34;
        const float dx = kgc::f16_to_f((uint16_t)(xb[0] | (xb[1] << 8)));
        const int q = (int)(int8_t)xb[2 + lane % QK8_0];
        acc += d * dx * (float)((code - 1) * q);
    }
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[row] = acc;
}
}  // namespace

cudaError_t s2_gemv_q8(const uint8_t* w, const uint8_t* xq8, float* y, int n_in, int n_out, cudaStream_t s) {
    if (!w || !xq8 || !y || n_in <= 0 || n_in % QK_S2 || n_out <= 0) return cudaErrorInvalidValue;
    s2_gemv_q8_kernel<<<(n_out + 3) / 4, 256, 0, s>>>(w, xq8, y, n_in, n_out);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
