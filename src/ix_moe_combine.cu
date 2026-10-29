// ix_moe_combine.cu - MoE block final combination: y[j] = sum_e w[e]*parts[e][j] (+ shared[j], added plain).
// Port of Strata src/kernels/cuda/shared_expert.cu moe_combine_kernel (HEAD 46c152a1, L146, matches v9).
// No warp/wave ops, no ivcore10 change needed: kernel body is verbatim. Double accumulator, ascending e (matches reference).
// STATUS: NOT yet compiled or run on BI-V100.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

__global__ void moe_combine_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                   const float* __restrict__ shared, float* __restrict__ y, int n_embd,
                                   int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_embd) return;
    // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
    // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
    double acc = 0.0;
    for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
    // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

// Single token: parts [k, n_embd], weights [k], shared [n_embd] (ignored if has_shared==0), y [n_embd].
inline cudaError_t ix_moe_combine(const float* parts, const float* weights, const float* shared, float* y,
                                  int n_embd, int k, int has_shared, cudaStream_t s = 0) {
    if (n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    const int TH = 256;
    moe_combine_kernel<<<(n_embd + TH - 1) / TH, TH, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
