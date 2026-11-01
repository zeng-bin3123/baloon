// ix_shared_expert.cu - Phase 1 #4: shared-expert helpers for ivcore10 (wave64).
// Port of Strata src/kernels/cuda/shared_expert.cu (HEAD 46c152a1): swiglu_kernel, scalar_gate_kernel, scale_kernel.
// ivcore10 changes (v9 Part H): warp_sum_d 32->64 lanes (ix_warp_sum_d), lane&63, warp>>6, scratch[4] for <<<1,256>>>.
// Semantics kept verbatim: SiLU goes on the GATE tensor; the gate is ONE scalar per token = sigmoid(dot(x, w_shexp)),
// both operands BF16, double accumulator; the shared output is added plain by ix_moe_combine (not router-weighted).
// STATUS: NOT yet compiled or run on BI-V100.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "ix_compat.h"

namespace ix_se {
constexpr int GATE_THREADS = 256;                       // 4 wavefronts of 64
constexpr int GATE_WAVES   = GATE_THREADS / IX_WARP_SIZE;

// silu(g) * up, silu evaluated in DOUBLE then cast (reference is float64), multiply in float - Strata L55-61.
__global__ void swiglu_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                              float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float) (x / (1.0 + exp(-x))) * up[i];
}

// sigmoid(dot(x_bf16, w_bf16)): launch <<<1, GATE_THREADS>>> only (scratch is sized for it).
__global__ void scalar_gate_kernel(const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ w_bf16,
                                   float* __restrict__ out, int n_embd) {
    __shared__ double scratch[GATE_WAVES];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x)
        acc += (double) ix_bf16_to_f32(x_bf16[i]) * (double) ix_bf16_to_f32(w_bf16[i]);
    const int lane = ix_lane_id(), wave = ix_warp_id();
    acc = ix_warp_sum_d(acc);                            // every lane of the wave now holds the wave sum
    if (lane == 0) scratch[wave] = acc;
    __syncthreads();
    if (wave == 0) {
        acc = (threadIdx.x < GATE_WAVES) ? scratch[threadIdx.x] : 0.0;
        acc = ix_warp_sum_d(acc);                        // lanes >= GATE_WAVES contribute 0
        if (threadIdx.x == 0) out[0] = (float) (1.0 / (1.0 + exp(-acc)));
    }
}

__global__ void scale_kernel(float* __restrict__ out, const float* __restrict__ g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= g[0];
}

inline cudaError_t ix_swiglu(const float* gate, const float* up, float* out, int n, cudaStream_t s = 0) {
    if (n <= 0) return cudaErrorInvalidValue;
    swiglu_kernel<<<(n + 127) / 128, 128, 0, s>>>(gate, up, out, n);
    return cudaGetLastError();
}
inline cudaError_t ix_scalar_gate(const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd,
                                  cudaStream_t s = 0) {
    if (n_embd <= 0) return cudaErrorInvalidValue;
    scalar_gate_kernel<<<1, GATE_THREADS, 0, s>>>(x_bf16, w_bf16, out, n_embd);
    return cudaGetLastError();
}
inline cudaError_t ix_scale(float* out, const float* g, int n, cudaStream_t s = 0) {
    if (n <= 0) return cudaErrorInvalidValue;
    scale_kernel<<<(n + 127) / 128, 128, 0, s>>>(out, g, n);
    return cudaGetLastError();
}
}  // namespace ix_se
