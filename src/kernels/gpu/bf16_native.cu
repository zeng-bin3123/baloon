// bf16_native.cu - "native mmvf" BF16 matrix-vector product: y[row] = sum_c bf16(w[row][c]) * x[c] with f32 activations and
// f32 accumulation; one block per row, block size picked from the column count (64..256 threads, a multiple of the wave).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
__device__ __forceinline__ float mmvf_warp_sum(float v) { return kgc::wave_sum(v); }

int mmvf_block_size(long long cols) {
    int bs = 64;
    while (bs < 256 && (long long)bs * 8 < cols) bs <<= 1;     // ~8 columns per thread, capped at 256
    return bs;
}

template <int BS>
__global__ void bf16_f32_mmvf_kernel(const uint16_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, long long cols) {
    __shared__ float sh[BS / 64];
    const uint16_t* row = w + (size_t)blockIdx.x * cols;
    float acc = 0.f;
    for (long long c = threadIdx.x; c < cols; c += BS) acc += kgc::bf16_to_f(row[c]) * x[c];
    acc = mmvf_warp_sum(acc);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < BS / 64; ++k) t += sh[k];
        y[blockIdx.x] = t;
    }
}
}  // namespace

#define REF_MMVF_CASE(BS) case BS: bf16_f32_mmvf_kernel<BS><<<(unsigned)rows, BS, 0, s>>>(w, x, y, cols); break;

cudaError_t bf16_gemv_fp32_mmvf(const uint16_t* w, const float* x, float* y, long long rows, long long cols, cudaStream_t s) {
    if (!w || !x || !y || rows <= 0 || cols <= 0 || rows > 2147483647ll) return cudaErrorInvalidValue;
    switch (mmvf_block_size(cols)) {
        REF_MMVF_CASE(64)
        REF_MMVF_CASE(128)
        REF_MMVF_CASE(256)
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
#undef REF_MMVF_CASE
}  // namespace kg
}  // namespace baloon
