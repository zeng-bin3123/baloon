// ix_router_top10_wave.cu - MoE router (softmax / top-k / renormalise), one wavefront (64 lanes) per token.
// Port of Strata src/kernels/cuda/router_top10.cu router_top10_wave_kernel (HEAD 46c152a1, L190-263).
// ivcore10 change: guard STRATA_HIP_GFX906 -> __ILUVATAR__. Kernel body is verbatim (already wave64).
// STATUS: NOT yet compiled or run on BI-V100 (no corex toolchain on the dev box).
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

#if defined(__ILUVATAR__)
// AMD (wave64): the same routing in ONE wavefront per token.  The block-wide kernel above spends its time in
// barriers - 10 selection passes with two __syncthreads each over 8 logical warps - not in arithmetic (43 us per
// token on gfx906).  Here every lane holds up to RW_PER probabilities in registers and each pass is a 64-lane
// butterfly argmax: no LDS round trip, no block barrier.  Bit-identical to the kernel above: fmaxf trees are
// exact, the exponentials and the (float) products are the same expressions, the sum stays one serial ascending
// scan, and the argmax keeps the lowest index on a tie (a total order, so the butterfly's association is free).
constexpr int RW_PER = 8;   // n_expert <= 64 * 8
__global__ void __launch_bounds__(64) router_top10_wave_kernel(const float* __restrict__ logits, int n_tokens,
                                                               int n_expert, int k, int* __restrict__ ids,
                                                               float* __restrict__ weights) {
    __shared__ double s_ex[64 * RW_PER];
    __shared__ double s_sum;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t) t * n_expert;
    float lv[RW_PER];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        mx = fmaxf(mx, lv[j]);
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        if (e < n_expert) s_ex[e] = exp((double) lv[j] - (double) mx);
    }
    __syncthreads();
    if (lane == 0) {
        double sum = 0.0;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = (float) (1.0 / s_sum);
    float p[RW_PER];
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        p[j] = e < n_expert ? (float) (s_ex[e] * inv) : -INFINITY;
    }
    unsigned taken = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
#pragma unroll
        for (int j = 0; j < RW_PER; ++j) {
            const int e = lane + 64 * j;
            if (e >= n_expert || (taken >> j & 1u)) continue;
            if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
        }
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 64);
            const int oi = __shfl_xor(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi >= n_expert) break;
        if ((bi & 63) == lane) taken |= 1u << (bi >> 6);
        if (lane == 0) { ids[(size_t) t * k + i] = bi; weights[(size_t) t * k + i] = bv; }
    }
    if (lane == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
        const double sc = fmax(s, 6.103515625e-05);       // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
}

// Host launcher: logits [n_tokens, n_expert] f32 -> ids/weights [n_tokens, k]. Requires n_expert <= 64*RW_PER (512).
inline cudaError_t ix_router_top10_wave(const float* logits, int n_tokens, int n_expert, int k,
                                        int* ids, float* weights, cudaStream_t s = 0) {
    if (n_expert > 64 * RW_PER || n_expert <= 0 || k <= 0 || k > n_expert) return cudaErrorInvalidValue;
    router_top10_wave_kernel<<<n_tokens, 64, 0, s>>>(logits, n_tokens, n_expert, k, ids, weights);
    return cudaGetLastError();
}
#endif  // __ILUVATAR__
