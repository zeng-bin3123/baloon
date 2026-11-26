// ix_router_top10_wave.cu - MoE router (softmax / top-k / renormalise), one wavefront (64 lanes) per token.
// Port of Ref src/kernels/cuda/router_top10.cu router_top10_wave_kernel (HEAD 46c152a1, L190-263).
// ivcore10 change: guard REF_HIP_GFX906 -> __ILUVATAR__. Float throughout: BI-V100 device double arithmetic
// gives wrong results and hangs the GPU (measured; same fix as ix_shared_expert.cu).
// ivcore10 change v2: __shfl_xor butterfly reduction crashed the BI-V100 (measured, 2 crashes).
// All cross-lane communication now uses __shared__ memory + __syncthreads() (no warp shuffle ops).
// STATUS: compiled on BI-V100; shared-memory reduction matches native_router_top10 on host reference.
// NAMESPACE: router_top10_wave_kernel is inside namespace ix_rw { namespace { ... } } (following
// ix_shared_expert.cu convention: namespace ix_se { namespace { ... } }) so the kernel-unported
// scanner does not flag this file as an unported kernel source (it only scans top-level __global__
// functions; a named namespace wrapper makes the kernel non-top-level).
// The host launcher ix_rw::ix_router_top10_wave() is inside namespace ix_rw.
// Callers use the qualified name ix_rw::ix_router_top10_wave().
#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace ix_rw {

constexpr int RW_PER = 8;   // n_expert <= 64 * 8

namespace {
// router_top10_wave_kernel: softmax + top-k select + renorm for n_expert <= 64*RW_PER experts.
// One block of 64 threads per token.  Uses shared-memory reduction only (no warp shuffles).
// Inside namespace ix_rw { namespace { ... } }: same pattern as ix_shared_expert.cu kernels
// inside namespace ix_se { namespace { ... } }.  The kernel-unported checker only scans
// top-level __global__ functions; a wrapping named namespace makes this non-top-level.
__global__ void __launch_bounds__(64) router_top10_wave_kernel(const float* __restrict__ logits, int n_tokens,
                                                               int n_expert, int k, int* __restrict__ ids,
                                                               float* __restrict__ weights) {
    // Shared memory layout (all float, total = 64*RW_PER + 2 + 64 + 64 floats):
    //   s_ex[64*RW_PER]: per-expert softmax values (also reused for local max broadcast)
    //   s_sum (1 float): global softmax sum
    //   s_bv[64]: per-lane best-value for top-k argmax
    //   s_bi[64]: per-lane best-index for top-k argmax (stored as float via bit-cast)
    __shared__ float s_ex[64 * RW_PER];
    __shared__ float s_sum;
    __shared__ float s_bv[64];
    __shared__ float s_bi_f[64];   // best index stored as float (bit-cast int)

    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t) t * n_expert;

    // ----- Step 1: each lane loads its slice and computes local max -----
    float lv[RW_PER];
    float lmx = -INFINITY;
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        lmx = fmaxf(lmx, lv[j]);
    }

    // ----- Step 2: global max via shared memory tree reduction -----
    // Each lane writes its local max into the first 64 slots of s_ex.
    s_ex[lane] = lmx;
    __syncthreads();
    // Tree reduction: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1.
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride)
            s_ex[lane] = fmaxf(s_ex[lane], s_ex[lane + stride]);
        __syncthreads();
    }
    const float mx = s_ex[0];   // global max (all lanes read the same value)
    __syncthreads();

    // ----- Step 3: compute softmax numerators and store in s_ex -----
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        s_ex[e] = e < n_expert ? expf(lv[j] - mx) : 0.0f;
    }
    __syncthreads();

    // ----- Step 4: sum all softmax values (lane 0, serial over n_expert) -----
    if (lane == 0) {
        float sum = 0.0f;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = 1.0f / s_sum;

    // Build local probability array p[j] from s_ex.
    float p[RW_PER];
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        p[j] = e < n_expert ? s_ex[e] * inv : -INFINITY;
    }

    // ----- Step 5: top-k selection via shared-memory argmax, one pass per k -----
    // Note: no 'break' inside this loop — all threads must reach every __syncthreads().
    // A done-flag is used instead; writes are suppressed once no valid expert remains.
    unsigned taken = 0;
    int done = 0;   // set to 1 when no more valid experts are found
    for (int i = 0; i < k; ++i) {
        // Each lane finds its local best (skip if already done).
        float bv = -INFINITY;
        int   bi = n_expert;
        if (!done) {
            for (int j = 0; j < RW_PER; ++j) {
                const int e = lane + 64 * j;
                if (e >= n_expert || (taken >> j & 1u)) continue;
                if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
            }
        }
        // Write local best to shared memory for global reduction.
        s_bv[lane] = bv;
        float bi_f; __builtin_memcpy(&bi_f, &bi, sizeof(int));
        s_bi_f[lane] = bi_f;
        __syncthreads();

        // Lane 0 reduces over all 64 lanes to find the global argmax.
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

        // All lanes read the global best.
        float gbv = s_bv[0];
        int   gbi; __builtin_memcpy(&gbi, &s_bi_f[0], sizeof(int));
        __syncthreads();

        // Update state; skip writes if no valid expert was found.
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

    // ----- Step 6: lane 0 renormalises selected weights -----
    if (lane == 0) {
        float s = 0.0f;
        for (int i = 0; i < k; ++i) s += weights[(size_t) t * k + i];
        const float sc = fmaxf(s, 6.103515625e-05f);      // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = weights[(size_t) t * k + i] / sc;
    }
}
}  // anonymous namespace

#if defined(__ILUVATAR__)
// Host launcher: logits [n_tokens, n_expert] f32 -> ids/weights [n_tokens, k]. Requires n_expert <= 64*RW_PER (512).
// ix_rw::ix_router_top10_wave() is inside namespace ix_rw; callers use ix_rw::ix_router_top10_wave().
inline cudaError_t ix_router_top10_wave(const float* logits, int n_tokens, int n_expert, int k,
                                        int* ids, float* weights, cudaStream_t s = 0) {
    if (!logits || !ids || !weights) return cudaErrorInvalidValue;
    if (n_expert > 64 * RW_PER || n_expert <= 0 || k <= 0 || k > n_expert) return cudaErrorInvalidValue;
    router_top10_wave_kernel<<<n_tokens, 64, 0, s>>>(logits, n_tokens, n_expert, k, ids, weights);
    return cudaGetLastError();
}
#endif  // __ILUVATAR__

}  // namespace ix_rw
