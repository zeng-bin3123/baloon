// native_gr_norm.cu - llama.cpp-pinned weighted F32 RMS norm for the native gated-residual path.
// Block size 256 (n_cols < 1024) or 1024 (n_cols >= 1024); thread tid visits tid + m*B.
// Iluvatar BI-V100: warpSize=64. 256 threads = 4 warps; 1024 threads = 16 warps.
// norm_warp_sum, check_pointer, weighted_rms_norm kernel, native_gr_rms_norm_weighted.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

// ── helpers ──────────────────────────────────────────────────────────────────

static inline bool check_pointer(const void* p) { return p != nullptr; }

// Warp-level sum. On Iluvatar, one warp = 64 lanes.
__device__ __forceinline__ float norm_warp_sum(float v) {
#ifdef __ILUVATAR__
    return ix_warp_reduce_sum(v);
#else
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
#endif
}

// ── kernel ───────────────────────────────────────────────────────────────────
// Block size B = 256 or 1024.  Thread tid accumulates squared values over columns.
// Iluvatar: warp = tid/64, lane = tid%64. nwarps = B/64 (4 for B=256, 16 for B=1024).
// Shared sums[nwarps]; lanes < nwarps do the final summation.
template <int B>
__global__ void weighted_rms_norm(const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   float* __restrict__ output,
                                   int n_cols, float eps) {
    extern __shared__ float sums[];
#ifdef __ILUVATAR__
    const int warp = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    constexpr int NWARPS = B / 64;
#else
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    constexpr int NWARPS = B / 32;
#endif
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;

    const float* xr  = input  + (size_t)row * n_cols;
    const float* wr  = gamma  + (size_t)row * n_cols;
    float*       or_ = output + (size_t)row * n_cols;

    float ss = 0.f;
    for (int c = tid; c < n_cols; c += B) ss += xr[c] * xr[c];
    float ws = norm_warp_sum(ss);
    if (lane == 0) sums[warp] = ws;
    __syncthreads();

    // Block-level reduction over nwarps values using first warp lanes.
    if (tid < NWARPS) {
        float v = sums[tid];
#ifdef __ILUVATAR__
        for (int off = NWARPS / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
#else
        for (int off = NWARPS / 2; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
#endif
        if (tid == 0) sums[0] = v;
    }
    __syncthreads();

    const float scale = rsqrtf(sums[0] / (float)n_cols + eps);
    for (int c = tid; c < n_cols; c += B) or_[c] = scale * xr[c] * wr[c];
}

// ── public API ───────────────────────────────────────────────────────────────
void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                   int n_rows, int n_cols, float eps, cudaStream_t stream) {
    if (!check_pointer(input) || !check_pointer(gamma) || !check_pointer(output)) return;
    if (n_rows <= 0 || n_cols <= 0 || eps < 0.f) return;

    if (n_cols < 1024) {
        const int B = 256;
#ifdef __ILUVATAR__
        const int nwarps = B / 64;
#else
        const int nwarps = B / 32;
#endif
        const int shared = nwarps * sizeof(float);
        weighted_rms_norm<256><<<n_rows, B, shared, stream>>>(input, gamma, output, n_cols, eps);
    } else {
        const int B = 1024;
#ifdef __ILUVATAR__
        const int nwarps = B / 64;
#else
        const int nwarps = B / 32;
#endif
        const int shared = nwarps * sizeof(float);
        weighted_rms_norm<1024><<<n_rows, B, shared, stream>>>(input, gamma, output, n_cols, eps);
    }
}
