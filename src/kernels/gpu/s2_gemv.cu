// s2_gemv.cu - y[row] = sum_i w[row][i] * x[i] with w in Q2_0 blocks (64 elements / 18 bytes: fp16 d + 16 code bytes,
// value d*(code-1)) and f32 x. One 64-lane wave per row (4 rows per block), lane l owns element l of every block.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int QK = 64;
constexpr int BB = 18;

__global__ void s2_gemv_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 63, nb = n_in / QK;
    const uint8_t* rp = w + (size_t)row * nb * BB;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = rp + (size_t)b * BB;
        const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
        const int code = (p[2 + lane / 4] >> (2 * (lane % 4))) & 3;
        acc += d * (float)(code - 1) * x[b * QK + lane];
    }
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[row] = acc;
}
}  // namespace

cudaError_t s2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (!w || !x || !y || n_in <= 0 || n_in % QK || n_out <= 0) return cudaErrorInvalidValue;
    s2_gemv_kernel<<<(n_out + 3) / 4, 256, 0, s>>>(w, x, y, n_in, n_out);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
