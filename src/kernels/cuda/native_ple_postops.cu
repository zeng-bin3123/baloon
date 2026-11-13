// native_ple_postops.cu - llama.cpp-pinned PLE post-projection operations.
// N, Span, broadcast_kernel, conv_residual_kernel, gate_kernel, launch_check,
// native_ple_postops, overlaps, validate, warp_sum.
// Iluvatar BI-V100: warpSize=64. All shuffles use width=64.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

// ── geometry constant ─────────────────────────────────────────────────────────
constexpr int N = 2560;  // n_embd

// ── Span helper ───────────────────────────────────────────────────────────────
struct Span {
    const float* ptr;
    int          n;
    bool valid() const { return ptr != nullptr && n > 0; }
};

// ── helpers ──────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_sum(float v) {
#ifdef __ILUVATAR__
    return ix_warp_reduce_sum(v);
#else
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
#endif
}

static inline bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    if (!a || !b || an == 0 || bn == 0) return false;
    uintptr_t ua = (uintptr_t)a, ub = (uintptr_t)b;
    return ua < ub + bn && ub < ua + an;
}

static inline bool launch_check(cudaError_t e) { return e == cudaSuccess; }

static inline bool validate(const Span& key, const Span& query, const Span& gate,
                              const Span& gated, const Span& normalized, const Span& conv,
                              const Span& result, const Span& hidden) {
    return key.valid() && query.valid() && gate.valid() && gated.valid() &&
           normalized.valid() && conv.valid() && result.valid() && hidden.valid();
}

// ── broadcast_kernel ──────────────────────────────────────────────────────────
__global__ void broadcast_kernel(float* __restrict__ gated,
                                   const float* __restrict__ value,
                                   const float* __restrict__ gate_vals,
                                   int hc, int n_embd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (d >= n_embd || c >= hc) return;
    gated[c * n_embd + d] = value[d] * gate_vals[c];
}

// ── gate_kernel ───────────────────────────────────────────────────────────────
// 512-thread block.  Each branch is a complete, brace-balanced function so that
// text parsers see the subsequent kernels at top-level scope.
// Gate: s = sum(key[c]*query[c]) / sqrt(n_embd), gate = sigmoid(sgn(s)*sqrt(max(|s|,1e-6))).
#ifdef __ILUVATAR__
// Iluvatar: 512 threads = 8 warps of 64 lanes; 8 shared partial slots.
__global__ void gate_kernel(float* __restrict__ gate_out,
                              const float* __restrict__ key,
                              const float* __restrict__ query,
                              int hc, int n_embd) {
    const int c   = blockIdx.x;
    if (c >= hc) return;
    const int tid = threadIdx.x;
    const float* kr = key   + (size_t)c * n_embd;
    const float* qr = query + (size_t)c * n_embd;
    const float scale = rsqrtf((float)n_embd);
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) {
        const int d = tid + j * 512;
        if (d < n_embd) sum += kr[d] * qr[d];
    }
    sum = warp_sum(sum);
    __shared__ float shm[8];
    const int warp = tid / 64;
    const int lane = tid % 64;
    if (lane == 0) shm[warp] = sum;
    __syncthreads();
    if (tid < 8) {
        float v = shm[tid];
        for (int off = 4; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
        if (tid == 0) {
            float s = v * scale;
            float mag = sqrtf(fabsf(s) < 1e-6f ? 1e-6f : fabsf(s));
            float sign = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
            gate_out[c] = 1.f / (1.f + expf(-(sign * mag)));
        }
    }
}
#else
// Non-Iluvatar: 512 threads = 16 warps of 32 lanes; 16 shared partial slots.
__global__ void gate_kernel(float* __restrict__ gate_out,
                              const float* __restrict__ key,
                              const float* __restrict__ query,
                              int hc, int n_embd) {
    const int c   = blockIdx.x;
    if (c >= hc) return;
    const int tid = threadIdx.x;
    const float* kr = key   + (size_t)c * n_embd;
    const float* qr = query + (size_t)c * n_embd;
    const float scale = rsqrtf((float)n_embd);
    float sum = 0.f;
    for (int j = 0; j < 8; ++j) {
        const int d = tid + j * 512;
        if (d < n_embd) sum += kr[d] * qr[d];
    }
    sum = warp_sum(sum);
    __shared__ float shm[16];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (lane == 0) shm[warp] = sum;
    __syncthreads();
    if (tid < 16) {
        float v = shm[tid];
        for (int off = 8; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) {
            float s = v * scale;
            float mag = sqrtf(fabsf(s) < 1e-6f ? 1e-6f : fabsf(s));
            float sign = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
            gate_out[c] = 1.f / (1.f + expf(-(sign * mag)));
        }
    }
}
#endif

// ── conv_residual_kernel ──────────────────────────────────────────────────────
__global__ void conv_residual_kernel(float* __restrict__ result,
                                      const float* __restrict__ hidden,
                                      const float* __restrict__ gated,
                                      const float* __restrict__ normalized,
                                      const float* __restrict__ history,
                                      const float* __restrict__ conv_w,
                                      int hc, int n_embd) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    if (d >= n_embd || c >= hc) return;
    const int HISTORY = 9;
    float conv = 0.f;
    for (int k = 0; k < 3; ++k) {
        float term = history[c * HISTORY + 3 * k] * conv_w[k];
        conv = (k == 0) ? term : conv + term;
    }
    conv += normalized[c * n_embd + d] * conv_w[3];
    conv = conv / (1.f + expf(-conv));  // SiLU
    result[c * n_embd + d] = hidden[c * n_embd + d] + (gated[c * n_embd + d] + conv);
}

// ── public API ───────────────────────────────────────────────────────────────
int native_ple_postops(float* result, const float* hidden, float* gated,
                        float* gate_out, const float* key, const float* query,
                        const float* value, const float* normalized,
                        const float* history, const float* conv_w,
                        int hc, int n_embd, cudaStream_t stream) {
    if (!result || !hidden || !gated || !gate_out || !key || !query ||
        !value || !normalized || !history || !conv_w) return -1;
    if (hc <= 0 || n_embd <= 0) return -1;

    gate_kernel<<<hc, 512, 8 * sizeof(float), stream>>>(gate_out, key, query, hc, n_embd);
    if (!launch_check(cudaGetLastError())) return -1;

    {
        const dim3 grid((n_embd + 255) / 256, hc);
        broadcast_kernel<<<grid, 256, 0, stream>>>(gated, value, gate_out, hc, n_embd);
        if (!launch_check(cudaGetLastError())) return -1;
    }

    {
        const dim3 grid((n_embd + 255) / 256, hc);
        conv_residual_kernel<<<grid, 256, 0, stream>>>(
            result, hidden, gated, normalized, history, conv_w, hc, n_embd);
        if (!launch_check(cudaGetLastError())) return -1;
    }
    return 0;
}
