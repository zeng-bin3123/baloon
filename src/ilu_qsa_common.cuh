// ilu_qsa_common.cuh - 32-lane segmented shuffles and block helpers shared by the QSA kernels (wave64-safe).
#pragma once
#include <cuda_runtime.h>

namespace baloon {
namespace ilu_qsa {

__device__ __forceinline__ float shfl_xor32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
// Sum / max over each aligned group of 32 lanes; every lane of the group receives the result.
__device__ __forceinline__ float lane_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += shfl_xor32(v, off);
    return v;
}
__device__ __forceinline__ float lane_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, shfl_xor32(v, off));
    return v;
}
// Sum over a 256-thread block through shared memory (>= 8 floats); every thread receives the total.
__device__ __forceinline__ float block_sum256(float v, float* sums) {
    v = lane_sum(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) r += sums[w];
    __syncthreads();
    return r;
}

}  // namespace ilu_qsa
}  // namespace baloon
