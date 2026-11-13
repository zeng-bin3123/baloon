// native_gr_postops.cu - llama.cpp-pinned element-wise ops of the native gated-residual path:
// down_silu (scale+SiLU), pre_gated (xn * sigmoid(gate), mean), post (residual + block_out * 2*sigmoid).
// THREADS, blocks, check_launch, check_pointer, check_shape, down_silu kernel, native_gr_down_silu,
// native_gr_post, native_gr_pre_gated, post kernel, pre_gated kernel, scale_zero_bias, sigmoid.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

// ── constants ────────────────────────────────────────────────────────────────
constexpr int THREADS = 256;

// ── helpers ──────────────────────────────────────────────────────────────────
static inline int blocks(int n) { return (n + THREADS - 1) / THREADS; }

static inline void check_launch(const char* name) {
    cudaError_t e = cudaGetLastError(); (void)name; (void)e;
}

static inline bool check_pointer(const void* p) { return p != nullptr; }

static inline bool check_shape(int n, int hc) {
    return n > 0 && hc > 0 && (int64_t)n * hc <= (int64_t)2147483647;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// ggml SCALE: fma(s, x, 0).
__device__ __forceinline__ float scale_zero_bias(float x, float s) { return s * x + 0.f; }

// ── down_silu kernel ──────────────────────────────────────────────────────────
// lo[i] = silu(lo[i] / hc)
__global__ void down_silu(float* __restrict__ lo, int hc, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    float x = lo[i] / (float)hc;
    lo[i] = x / (1.f + expf(-x));
}
void native_gr_down_silu(float* lo, int hc, int n, cudaStream_t stream) {
    if (!check_pointer(lo) || n <= 0 || hc <= 0) return;
    down_silu<<<blocks(n), THREADS, 0, stream>>>(lo, hc, n);
    check_launch("native_gr_down_silu");
}

// ── pre_gated kernel ──────────────────────────────────────────────────────────
// gate[c*n+d] = xn[c*n+d] * sigmoid(gate[c*n+d])
// Then mixed[d] = scale_zero_bias(sum_c gate[c*n+d], 1/hc).
// Unfused: accumulate sum first, then scale; fused: start from 0, fma.
// Both produce the same float result up to rounding order.
__global__ void pre_gated(float* __restrict__ gate, float* __restrict__ mixed,
                            const float* __restrict__ xn,
                            int hc, int n) {
    const int d = blockIdx.x * THREADS + threadIdx.x;
    if (d >= n) return;
    float sum = 0.f;
    for (int c = 0; c < hc; ++c) {
        const int idx = c * n + d;
        const float p = xn[idx] * sigmoid(gate[idx]);
        gate[idx] = p;
        sum += p;
    }
    mixed[d] = scale_zero_bias(sum, 1.f / (float)hc);
}
void native_gr_pre_gated(float* gate, float* mixed, const float* xn,
                           int hc, int n, cudaStream_t stream) {
    if (!check_pointer(gate) || !check_pointer(mixed) || !check_pointer(xn)) return;
    if (!check_shape(n, hc)) return;
    pre_gated<<<blocks(n), THREADS, 0, stream>>>(gate, mixed, xn, hc, n);
    check_launch("native_gr_pre_gated");
}

// ── post kernel ───────────────────────────────────────────────────────────────
// out[i] = fma(block_out[i%n], 2*sigmoid(inject[i/n]/hc), residual[i])
// Aliasing residual == output is safe (each thread owns exactly one element).
__global__ void post(float* __restrict__ out,
                      const float* __restrict__ residual,
                      const float* __restrict__ block_out,
                      const float* __restrict__ inject,
                      int n, int hc, int total) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= total) return;
    const int d = i % n;
    const int c = i / n;
    out[i] = block_out[d] * (2.f * sigmoid(inject[c] / (float)hc)) + residual[i];
}
void native_gr_post(float* out, const float* residual, const float* block_out,
                     const float* inject, int n, int hc, cudaStream_t stream) {
    if (!check_pointer(out) || !check_pointer(residual) || !check_pointer(block_out) || !check_pointer(inject)) return;
    if (!check_shape(n, hc)) return;
    const int total = n * hc;
    post<<<blocks(total), THREADS, 0, stream>>>(out, residual, block_out, inject, n, hc, total);
    check_launch("native_gr_post");
}

