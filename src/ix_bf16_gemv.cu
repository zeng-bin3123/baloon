// ix_bf16_gemv.cu - Phase 1 #8: BF16 GEMV for ivcore10 (wave64).
// Port of Strata src/kernels/cuda/bf16_gemv.cu (HEAD 46c152a1): naive / warp-per-row / split / pair kernels.
// ivcore10 changes (v9 Part H): ALL warp32 patterns -> wave64: lane&63, tid>>6, stride 64, shfl_down off=32..1 width 64,
// 4 waves per 256-thread block. Weights/activations stay BF16 *bits* (ix_bf16_to_f32 is an exact <<16, no FP16 round trip);
// accumulation is float, as in Strata.
// NOT bitwise equal to Strata's warp32 result: lane stride and tree shape change the float summation order. Accept = within
// float tolerance of a double reference. What IS preserved: pair kernel == standalone warp kernel, bitwise (same loop/order).
// threads_per_row==64 selects warp-per-row (Strata used 32). tpr=32 goes through the shared-memory tree and is correct but slow.
// STATUS: NOT yet compiled or run on BI-V100.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "ix_compat.h"

namespace ix_bf {
constexpr int THREADS = 256;
constexpr int WAVES = THREADS / IX_WARP_SIZE;   // 4

__device__ __forceinline__ float wave_sum_f(float v) {
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
    return v;                                   // valid in lane 0 only (as in Strata)
}

__global__ void bf16_gemv_naive_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                       float* __restrict__ y, long long n_in, long long n_out) {
    const long long o = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = 0; i < n_in; ++i) acc += ix_bf16_to_f32(x[i]) * ix_bf16_to_f32(row[i]);
    y[o] = acc;
}

__global__ void bf16_gemv_warp_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                      float* __restrict__ y, long long n_in, long long n_out) {
    const long long o = (long long) blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    if (o >= n_out) return;                       // whole wave exits together (o is wave-uniform)
    const int lane = threadIdx.x & 63;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = lane; i < n_in; i += 64) acc += ix_bf16_to_f32(x[i]) * ix_bf16_to_f32(row[i]);
    acc = wave_sum_f(acc);
    if (lane == 0) y[o] = acc;
}

// tpr threads per row, shared-memory tree; tpr must be a power of two (<= 1024). Unchanged from Strata (no warp ops).
__global__ void bf16_gemv_split_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
                                       float* __restrict__ y, long long n_in, long long n_out, int tpr) {
    extern __shared__ float scratch[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int t = threadIdx.x;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
    for (long long i = t; i < n_in; i += tpr) acc += ix_bf16_to_f32(x[i]) * ix_bf16_to_f32(row[i]);
    scratch[t] = acc;
    __syncthreads();
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        __syncthreads();
    }
    if (t == 0) y[o] = scratch[0];
}

__global__ void bf16_gemv_pair_warp_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w1,
                                           float* __restrict__ y1, long long n_out1,
                                           const uint16_t* __restrict__ w2, float* __restrict__ y2,
                                           long long n_out2, long long n_in) {
    const long long o = (long long) blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
    const int lane = threadIdx.x & 63;
    const long long n_out = n_out1 > n_out2 ? n_out1 : n_out2;
    if (o >= n_out) return;
    const int run1 = o < n_out1, run2 = o < n_out2;
    const uint16_t* r1 = w1 + o * n_in;
    const uint16_t* r2 = w2 + o * n_in;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (long long i = lane; i < n_in; i += 64) {
        const float xv = ix_bf16_to_f32(x[i]);
        if (run1) acc1 += xv * ix_bf16_to_f32(r1[i]);
        if (run2) acc2 += xv * ix_bf16_to_f32(r2[i]);
    }
    // run1/run2 are wave-uniform (depend only on o), so the shuffles below are never divergent.
    if (run1) { const float a = wave_sum_f(acc1); if (lane == 0) y1[o] = a; }
    if (run2) { const float a = wave_sum_f(acc2); if (lane == 0) y2[o] = a; }
}
}  // namespace ix_bf

// Host launchers return cudaError_t (Strata's exit(1) on error is not appropriate for a library).
inline cudaError_t ix_bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out,
                                cudaStream_t s = 0) {
    if (n_in <= 0 || n_out <= 0) return cudaErrorInvalidValue;
    if (n_out >= 64) {   // threshold carried over from Strata (48 SMs there); retune once BI-V100 SM count is known
        ix_bf::bf16_gemv_warp_kernel<<<(unsigned) ((n_out + ix_bf::WAVES - 1) / ix_bf::WAVES), ix_bf::THREADS, 0, s>>>(
            x, w, y, n_in, n_out);
    } else {
        ix_bf::bf16_gemv_naive_kernel<<<(unsigned) ((n_out + ix_bf::THREADS - 1) / ix_bf::THREADS), ix_bf::THREADS, 0, s>>>(
            x, w, y, n_in, n_out);
    }
    return cudaGetLastError();
}

inline cudaError_t ix_bf16_gemv_pair(const uint16_t* x, const uint16_t* w1, float* y1, long long n_out1,
                                     const uint16_t* w2, float* y2, long long n_out2, long long n_in,
                                     cudaStream_t s = 0) {
    if (n_in <= 0 || (n_out1 <= 0 && n_out2 <= 0)) return cudaErrorInvalidValue;
    const long long n_out = n_out1 > n_out2 ? n_out1 : n_out2;
    ix_bf::bf16_gemv_pair_warp_kernel<<<(unsigned) ((n_out + ix_bf::WAVES - 1) / ix_bf::WAVES), ix_bf::THREADS, 0, s>>>(
        x, w1, y1, n_out1, w2, y2, n_out2, n_in);
    return cudaGetLastError();
}

inline cudaError_t ix_bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out,
                                      int threads_per_row, cudaStream_t s = 0) {
    if (n_in <= 0 || n_out <= 0) return cudaErrorInvalidValue;
    if (threads_per_row == 64) {   // warp-per-row path (Strata: 32)
        ix_bf::bf16_gemv_warp_kernel<<<(unsigned) ((n_out + ix_bf::WAVES - 1) / ix_bf::WAVES), ix_bf::THREADS, 0, s>>>(
            x, w, y, n_in, n_out);
        return cudaGetLastError();
    }
    if (threads_per_row <= 0 || threads_per_row > 1024 || (threads_per_row & (threads_per_row - 1)) != 0)
        return cudaErrorInvalidValue;
    ix_bf::bf16_gemv_split_kernel<<<(unsigned) n_out, threads_per_row, (size_t) threads_per_row * sizeof(float), s>>>(
        x, w, y, n_in, n_out, threads_per_row);
    return cudaGetLastError();
}
