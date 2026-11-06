// ew_ops.cu - elementwise kernels and the host<->device doorbell primitives.
//   add / scale / silu in place, f32 -> bf16/f16 bulk conversion, embedding gather, GDN gate, weighted RMS norm,
//   copies out of host-mapped memory and a bounded-spin doorbell (ring / publish / wait).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int THREADS = 256;

__device__ __forceinline__ float softplus_dev(float x) { return kgc::softplus(x); }

__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void scale_kernel(float* __restrict__ y, float a, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] *= a;
}
__global__ void silu_kernel(float* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = kgc::silu(y[i]);
}
__global__ void to_bf16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = kgc::f_to_bf16(x[i]);
}
__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = kgc::f_to_f16(x[i]);
}
__global__ void copy_from_mapped_kernel(const float* __restrict__ src, float* __restrict__ dst, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void copy_i32_from_mapped_kernel(const int32_t* __restrict__ src, int32_t* __restrict__ dst, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
// out-of-range token ids give a zero row.
__global__ void embedding_gather_kernel(const float* __restrict__ table, long long vocab, long long n,
                                        const int32_t* __restrict__ tokens, float* __restrict__ out) {
    const long long row = blockIdx.x;
    const int32_t t = tokens[row];
    const bool ok = t >= 0 && (long long)t < vocab;
    for (long long c = threadIdx.x; c < n; c += blockDim.x) out[row * n + c] = ok ? table[(long long)t * n + c] : 0.f;
}
// g[h] = -exp(a_log[h]) * softplus(alpha[h] + dt_bias[h])
__global__ void gdn_gate_kernel(const float* __restrict__ alpha, const float* __restrict__ a_log,
                                const float* __restrict__ dt_bias, float* __restrict__ g, int n) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n) g[h] = -expf(a_log[h]) * softplus_dev(alpha[h] + dt_bias[h]);
}
// one block per row: y = x * rsqrt(mean(x^2) + eps) * w   (double accumulation)
__global__ void rms_norm_weighted_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                         int cols, float eps) {
    __shared__ float part[THREADS / 64];
    const float* px = x + (size_t)blockIdx.x * cols;
    float acc = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) acc += px[i] * px[i];
    acc = kgc::wave_sum(acc);
    if ((threadIdx.x & 63) == 0) part[threadIdx.x >> 6] = acc;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < (int)(blockDim.x >> 6); ++k) tot += part[k];
    const float inv = 1.0f / sqrtf(tot / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[(size_t)blockIdx.x * cols + i] = px[i] * inv * w[i];
}

// Doorbell word protocol (device-visible, host-mapped uint32): ring = counter++ ; publish = counter := value ;
// wait = spin until counter >= target or max_spins reads elapsed; status[0] = 1 on success, 0 on timeout.
__global__ void doorbell_ring_kernel(volatile uint32_t* bell) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { __threadfence(); atomicAdd((unsigned int*)bell, 1u); __threadfence(); }
}
__global__ void doorbell_publish_kernel(volatile uint32_t* bell, uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { __threadfence(); *bell = value; __threadfence(); }
}
__global__ void doorbell_wait_kernel(volatile uint32_t* bell, uint32_t target, uint64_t max_spins, uint32_t* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t n = 0;
    while (*bell < target && n < max_spins) { __threadfence(); ++n; }
    status[0] = (*bell >= target) ? 1u : 0u;
}
}  // namespace

unsigned grid_for(long long n, int threads) { return kgc::grid_for_n(n, threads); }
cudaError_t check_launch(const char*) { return cudaGetLastError(); }
cudaError_t sync_if_needed(cudaStream_t s, bool sync) { return sync ? cudaStreamSynchronize(s) : cudaSuccess; }

cudaError_t add_inplace(float* y, const float* x, long long n, cudaStream_t s) {
    if (n < 0 || (n && (!x || !y))) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    add_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(y, x, n);
    return check_launch("add");
}
cudaError_t scale_inplace(float* y, float a, long long n, cudaStream_t s) {
    if (n < 0 || (n && !y)) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    scale_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(y, a, n);
    return check_launch("scale");
}
cudaError_t silu_inplace(float* y, long long n, cudaStream_t s) {
    if (n < 0 || (n && !y)) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    silu_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(y, n);
    return check_launch("silu");
}
cudaError_t f32_to_bf16_bulk(const float* x, uint16_t* y, long long n, cudaStream_t s) {
    if (n < 0 || (n && (!x || !y))) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    to_bf16_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(x, y, n);
    return check_launch("to_bf16");
}
cudaError_t f32_to_f16_bulk(const float* x, uint16_t* y, long long n, cudaStream_t s) {
    if (n < 0 || (n && (!x || !y))) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    to_f16_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(x, y, n);
    return check_launch("to_f16");
}
cudaError_t copy_from_mapped(const float* src, float* dst, long long n, cudaStream_t s) {
    if (n < 0 || (n && (!src || !dst))) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    copy_from_mapped_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(src, dst, n);
    return check_launch("copy_f32");
}
cudaError_t copy_i32_from_mapped(const int32_t* src, int32_t* dst, long long n, cudaStream_t s) {
    if (n < 0 || (n && (!src || !dst))) return cudaErrorInvalidValue;
    if (!n) return cudaSuccess;
    copy_i32_from_mapped_kernel<<<grid_for(n, THREADS), THREADS, 0, s>>>(src, dst, n);
    return check_launch("copy_i32");
}
cudaError_t embedding_gather(const float* table, long long vocab, long long n, const int32_t* tokens, long long count,
                             float* out, cudaStream_t s) {
    if (vocab <= 0 || n <= 0 || count < 0 || count > 2147483647ll) return cudaErrorInvalidValue;
    if (!count) return cudaSuccess;
    if (!table || !tokens || !out) return cudaErrorInvalidValue;
    embedding_gather_kernel<<<(unsigned)count, THREADS, 0, s>>>(table, vocab, n, tokens, out);
    return check_launch("embedding_gather");
}
cudaError_t gdn_gate(const float* alpha, const float* a_log, const float* dt_bias, float* g, int n, cudaStream_t s) {
    if (n <= 0 || !alpha || !a_log || !dt_bias || !g) return cudaErrorInvalidValue;
    gdn_gate_kernel<<<grid_for(n, 128), 128, 0, s>>>(alpha, a_log, dt_bias, g, n);
    return check_launch("gdn_gate");
}
cudaError_t rms_norm_weighted(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s) {
    if (rows <= 0 || cols <= 0 || !x || !w || !y) return cudaErrorInvalidValue;
    rms_norm_weighted_kernel<<<rows, THREADS, 0, s>>>(x, w, y, cols, eps);
    return check_launch("rms_norm");
}
cudaError_t doorbell_ring(uint32_t* bell, cudaStream_t s) {
    if (!bell) return cudaErrorInvalidValue;
    doorbell_ring_kernel<<<1, 1, 0, s>>>(bell);
    return check_launch("doorbell_ring");
}
cudaError_t doorbell_publish(uint32_t* bell, uint32_t value, cudaStream_t s) {
    if (!bell) return cudaErrorInvalidValue;
    doorbell_publish_kernel<<<1, 1, 0, s>>>(bell, value);
    return check_launch("doorbell_publish");
}
cudaError_t doorbell_wait(uint32_t* bell, uint32_t target, uint64_t max_spins, uint32_t* status, cudaStream_t s) {
    if (!bell || !status) return cudaErrorInvalidValue;
    doorbell_wait_kernel<<<1, 1, 0, s>>>(bell, target, max_spins, status);
    return check_launch("doorbell_wait");
}
}  // namespace kg
}  // namespace baloon
