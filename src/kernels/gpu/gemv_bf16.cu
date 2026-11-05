// gemv_bf16.cu - y[o] = sum_i bf16(x[i]) * bf16(w[o][i]) in float, three schedules: thread-per-row (naive), wave-per-row
// (warp) and block-per-row with a shared-memory tree (split). Weights/activations are bf16 BITS; the widening is an exact <<16.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int THREADS = 256;                 // 4 waves
constexpr int WAVES = THREADS / 64;

__global__ void bf16_gemv_naive_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y,
                                       long long n_in, long long n_out) {
    const long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.f;
    for (long long i = 0; i < n_in; ++i) acc += kgc::bf16_to_f(x[i]) * kgc::bf16_to_f(row[i]);
    y[o] = acc;
}

__global__ void bf16_gemv_warp_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y,
                                      long long n_in, long long n_out) {
    const long long o = (long long)blockIdx.x * WAVES + (threadIdx.x >> 6);
    if (o >= n_out) return;                  // o is wave-uniform: the whole wave leaves together
    const int lane = threadIdx.x & 63;
    const uint16_t* row = w + o * n_in;
    float acc = 0.f;
    for (long long i = lane; i < n_in; i += 64) acc += kgc::bf16_to_f(x[i]) * kgc::bf16_to_f(row[i]);
    acc = kgc::wave_sum(acc);
    if (lane == 0) y[o] = acc;
}

// tpr threads per row (power of two <= 1024), one block per row.
__global__ void bf16_gemv_split_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y,
                                       long long n_in, long long n_out, int tpr) {
    extern __shared__ float scratch[];
    const long long o = blockIdx.x;
    if (o >= n_out) return;
    const int t = threadIdx.x;
    const uint16_t* row = w + o * n_in;
    float acc = 0.f;
    for (long long i = t; i < n_in; i += tpr) acc += kgc::bf16_to_f(x[i]) * kgc::bf16_to_f(row[i]);
    scratch[t] = acc;
    __syncthreads();
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        __syncthreads();
    }
    if (t == 0) y[o] = scratch[0];
}

cudaError_t finish(cudaStream_t) { return cudaGetLastError(); }
}  // namespace

cudaError_t bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out, cudaStream_t s) {
    if (!x || !w || !y || n_in <= 0 || n_out <= 0) return cudaErrorInvalidValue;
    if (n_out >= 64)
        bf16_gemv_warp_kernel<<<(unsigned)((n_out + WAVES - 1) / WAVES), THREADS, 0, s>>>(x, w, y, n_in, n_out);
    else
        bf16_gemv_naive_kernel<<<(unsigned)((n_out + THREADS - 1) / THREADS), THREADS, 0, s>>>(x, w, y, n_in, n_out);
    return finish(s);
}

cudaError_t bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, long long n_in, long long n_out, int tpr,
                            cudaStream_t s) {
    if (!x || !w || !y || n_in <= 0 || n_out <= 0) return cudaErrorInvalidValue;
    if (tpr <= 0 || tpr > 1024 || (tpr & (tpr - 1)) != 0) return cudaErrorInvalidValue;
    bf16_gemv_split_kernel<<<(unsigned)n_out, tpr, (size_t)tpr * sizeof(float), s>>>(x, w, y, n_in, n_out, tpr);
    return finish(s);
}
}  // namespace kg
}  // namespace baloon
