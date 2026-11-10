// s2_gemv_fast.cu - Q2_0 GEMV with a byte->4-values lookup table in constant memory and the activation staged in shared
// memory as f16. NOTE the activation operand is rounded to f16 (that is the price of fitting n_in values in shared memory):
// the result equals s2_gemv applied to f16-rounded x. One wave per row; lane l handles code bytes l, l+64, ... of each block.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int QK_S2 = 64;
constexpr int MAX_SHARED_HALVES = 4096;           // theory: maxSharedHalves = 4096 (8 KiB of f16; + 4 KiB LUT stays far below 48 KiB)
bool g_lut_ready = false;                // the byte->values table lives in shared memory and is rebuilt per block

cudaError_t ensure_lut() { g_lut_ready = true; return cudaSuccess; }

__global__ void s2_gemv_fast_kernel(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int n_in, int n_out) {
    extern __shared__ uint16_t xs[];
    __shared__ float lut[256 * 4];
    for (int u = threadIdx.x; u < 256 * 4; u += blockDim.x) { const int b = u >> 2, j = u & 3; lut[u] = (float)(((b >> (2 * j)) & 3) - 1); }
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) xs[i] = kgc::f_to_f16(x[i]);
    __syncthreads();
    const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (row >= n_out) return;
    const int lane = threadIdx.x & 63, nb = n_in / QK_S2;
    const uint8_t* rp = w + (size_t)row * nb * 18;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = rp + (size_t)b * 18;
        const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
        if (lane < 16) {                                   // 16 code bytes, 4 values each
            const float* l = &lut[p[2 + lane] * 4];
            const uint16_t* xv = &xs[b * QK_S2 + lane * 4];
            float a = 0.f;
            a += l[0] * kgc::f16_to_f(xv[0]); a += l[1] * kgc::f16_to_f(xv[1]);
            a += l[2] * kgc::f16_to_f(xv[2]); a += l[3] * kgc::f16_to_f(xv[3]);
            acc += d * a;
        }
    }
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[row] = acc;
}
}  // namespace

cudaError_t s2_gemv_fast(const uint8_t* w, const float* x, float* y, int n_in, int n_out, cudaStream_t s) {
    if (!w || !x || !y || n_in <= 0 || n_in % QK_S2 || n_out <= 0 || n_in > MAX_SHARED_HALVES) return cudaErrorInvalidValue;
    cudaError_t e = ensure_lut();
    if (e != cudaSuccess) return e;
    s2_gemv_fast_kernel<<<(n_out + 3) / 4, 256, (size_t)n_in * sizeof(uint16_t), s>>>(w, x, y, n_in, n_out);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
