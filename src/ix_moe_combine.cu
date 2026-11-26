// ix_moe_combine.cu - MoE block final combination: y[j] = sum_e w[e]*parts[e][j] (+ shared[j], added plain).
// Port of Ref src/kernels/cuda/shared_expert.cu moe_combine_kernel.
// Float accumulator; BI-V100 device double arithmetic gives wrong results (measured).
// STATUS: direct BI-V100 kernel — no baloon::kg delegation; validates args, then launches.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace ix_mc {

// Single token: parts [k, n_embd] row-major, weights [k], shared [n_embd] (skipped if has_shared==0).
__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared,
                                   float* __restrict__ y, int n_embd, int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += weights[e] * parts[(size_t)e * n_embd + j];
    if (has_shared) acc += shared[j];
    y[j] = acc;
}

} // namespace ix_mc

// Single token: parts [k, n_embd], weights [k], shared [n_embd] (ignored if has_shared==0), y [n_embd].
// Validates args and launches ix_mc::moe_combine_kernel.
inline cudaError_t ix_moe_combine(const float* parts, const float* weights, const float* shared, float* y,
                                  int n_embd, int k, int has_shared, cudaStream_t s = 0) {
    if (!parts || !weights || !y || n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    ix_mc::moe_combine_kernel<<<(n_embd + 255) / 256, 256, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
