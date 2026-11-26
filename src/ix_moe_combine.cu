// ix_moe_combine.cu - MoE block final combination: y[j] = sum_e w[e]*parts[e][j] (+ shared[j], added plain).
// Delegates to baloon::kg::moe_combine (src/kernels/gpu/shared_expert_k.cu, pcd027bc02f).
// The __global__ moe_combine_kernel lives only in shared_expert_k.cu (anonymous namespace);
// this wrapper keeps the ix_moe_combine() call site and validates args before forwarding.
// Float accumulator, ascending e; BI-V100 device double arithmetic gives wrong results (measured).
// STATUS: delegates to baloon::kg::moe_combine — same kernel, same float path, bit-exact.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include "baloon/kg_kernels.hpp"

// Single token: parts [k, n_embd], weights [k], shared [n_embd] (ignored if has_shared==0), y [n_embd].
// Validates args and forwards to baloon::kg::moe_combine (shared_expert_k.cu).
inline cudaError_t ix_moe_combine(const float* parts, const float* weights, const float* shared, float* y,
                                  int n_embd, int k, int has_shared, cudaStream_t s = 0) {
    if (!parts || !weights || !y || n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    return baloon::kg::moe_combine(parts, weights, shared, y, n_embd, k, has_shared, s);
}
