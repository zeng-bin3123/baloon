// ilu_gdn_shfl.cuh - 32-lane segmented reductions that behave the same on wave64 (ivcore10) and warp32 hardware.
#pragma once
#include <cuda_runtime.h>

namespace baloon {
namespace ilu_gdn {

__device__ __forceinline__ float shfl_xor32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}

// Sum over each aligned group of 32 lanes; every lane of the group receives the total.
__device__ __forceinline__ float lane_reduce(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += shfl_xor32(v, off);
    return v;
}

// Sum over a 256-thread block: per-32-lane partials land in sums[t/32] (8 used of 32), all threads read all 8.
__device__ __forceinline__ float block_sum256(float v, float* sums) {
    v = lane_reduce(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) r += sums[w];
    __syncthreads();
    return r;
}

}  // namespace ilu_gdn
}  // namespace baloon
