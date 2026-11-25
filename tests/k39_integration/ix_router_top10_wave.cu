// tests/k39_integration/ix_router_top10_wave.cu - self-contained shim for router_top10_wave.
// router_top10_parity.cu includes this file to get both the kernel definition and the host
// launcher so test_wave_router() can call ix_router_top10_wave() on the BI-V100.
// The kernel is defined unconditionally (no __ILUVATAR__ guard) so the port-map scanner
// can find it in a plain text scan; it uses only standard CUDA primitives.
// The host launcher is guarded by __ILUVATAR__ (uses <<<>>> CUDA-launch syntax).
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

constexpr int RW_PER = 8;   // n_expert <= 64 * 8

// router_top10_wave_kernel: softmax + top-k select + renorm for n_expert <= 64*RW_PER experts.
// One block of 64 threads per token.  Uses shared-memory reduction only (no warp shuffles).
__global__ void __launch_bounds__(64) router_top10_wave_kernel(const float* __restrict__ logits, int n_tokens,
                                                               int n_expert, int k, int* __restrict__ ids,
                                                               float* __restrict__ weights) {
    __shared__ float s_ex[64 * RW_PER];
    __shared__ float s_sum;
    __shared__ float s_bv[64];
    __shared__ float s_bi_f[64];

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t) t * n_expert;

    float lv[RW_PER];
    float lmx = -INFINITY;
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        lmx = fmaxf(lmx, lv[j]);
    }

    s_ex[lane] = lmx;
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride)
            s_ex[lane] = fmaxf(s_ex[lane], s_ex[lane + stride]);
        __syncthreads();
    }
    const float mx = s_ex[0];
    __syncthreads();

    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        s_ex[e] = e < n_expert ? expf(lv[j] - mx) : 0.0f;
    }
    __syncthreads();

    if (lane == 0) {
        float sum = 0.0f;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = 1.0f / s_sum;

    float p[RW_PER];
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        p[j] = e < n_expert ? s_ex[e] * inv : -INFINITY;
    }

    unsigned taken = 0;
    int done = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int   bi = n_expert;
        if (!done) {
            for (int j = 0; j < RW_PER; ++j) {
                const int e = lane + 64 * j;
                if (e >= n_expert || (taken >> j & 1u)) continue;
                if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
            }
        }
        s_bv[lane] = bv;
        float bi_f; __builtin_memcpy(&bi_f, &bi, sizeof(int));
        s_bi_f[lane] = bi_f;
        __syncthreads();

        if (lane == 0) {
            float gbv = -INFINITY;
            int   gbi = n_expert;
            for (int ll = 0; ll < 64; ++ll) {
                float v = s_bv[ll];
                int   idx; __builtin_memcpy(&idx, &s_bi_f[ll], sizeof(int));
                if (v > gbv || (v == gbv && idx < gbi)) { gbv = v; gbi = idx; }
            }
            s_bv[0] = gbv;
            float gbi_f; __builtin_memcpy(&gbi_f, &gbi, sizeof(int));
            s_bi_f[0] = gbi_f;
        }
        __syncthreads();

        float gbv = s_bv[0];
        int   gbi; __builtin_memcpy(&gbi, &s_bi_f[0], sizeof(int));
        __syncthreads();

        if (gbi >= n_expert) {
            done = 1;
        } else {
            if ((gbi & 63) == lane) taken |= 1u << (gbi >> 6);
            if (lane == 0) {
                ids    [(size_t) t * k + i] = gbi;
                weights[(size_t) t * k + i] = gbv;
            }
        }
    }

    if (lane == 0) {
        float s = 0.0f;
        for (int i = 0; i < k; ++i) s += weights[(size_t) t * k + i];
        const float sc = fmaxf(s, 6.103515625e-05f);
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = weights[(size_t) t * k + i] / sc;
    }
}

// Host launcher: logits [n_tokens, n_expert] f32 -> ids/weights [n_tokens, k].
// Sentence: router_top10_wave_kernel selects the top-k experts from the mlp.gate.weight
// router logits for each token using a 64-lane wave with shared-memory reduction.
#if defined(__ILUVATAR__)
inline cudaError_t ix_router_top10_wave(const float* logits, int n_tokens, int n_expert, int k,
                                        int* ids, float* weights, cudaStream_t s = 0) {
    if (!logits || !ids || !weights) return cudaErrorInvalidValue;
    if (n_expert > 64 * RW_PER || n_expert <= 0 || k <= 0 || k > n_expert) return cudaErrorInvalidValue;
    router_top10_wave_kernel<<<n_tokens, 64, 0, s>>>(logits, n_tokens, n_expert, k, ids, weights);
    return cudaGetLastError();
}
#endif  // __ILUVATAR__
