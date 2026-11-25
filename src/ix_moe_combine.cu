// ix_moe_combine.cu - MoE block final combination: y[j] = sum_e w[e]*parts[e][j] (+ shared[j], added plain).
// Port of Ref src/kernels/cuda/shared_expert.cu moe_combine_kernel (HEAD 46c152a1, L146, matches v9).
// No warp/wave ops, no ivcore10 change needed: kernel body is verbatim. Float accumulator, ascending e.
// Float throughout: BI-V100 device double arithmetic gives wrong results and hangs the GPU (measured;
// same fix as ix_router_top10_wave.cu and ix_shared_expert.cu). k<=8 so float is exact enough.
// STATUS: compiled and run on BI-V100 (float path, matches baloon::kg::moe_combine reference).
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared, float* __restrict__ y, int n_embd,
                                   int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    // Accumulate in float, ascending e (matches reference order). k<=8 so f32 is exact enough.
    // Device double arithmetic gives wrong results and hangs the BI-V100 (measured).
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += weights[e] * parts[(size_t) e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += shared[j];
    y[j] = acc;
}

// Single token: parts [k, n_embd], weights [k], shared [n_embd] (ignored if has_shared==0), y [n_embd].
inline cudaError_t ix_moe_combine(const float* parts, const float* weights, const float* shared, float* y,
                                  int n_embd, int k, int has_shared, cudaStream_t s = 0) {
    if (!parts || !weights || !y || n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    const int TH = 256;
    moe_combine_kernel<<<(n_embd + TH - 1) / TH, TH, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
