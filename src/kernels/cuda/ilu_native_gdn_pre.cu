// ilu_native_gdn_pre.cu - GDN preprocessing kernels for BI-V100: conv+SiLU with a 3-tap history slide, per-row L2 norm
// with scale, beta sigmoid, gate softplus, and output RMS-norm with sigmoid gating. Norm kernels use 256 threads per
// row of width 128 (8 partials per block sum); counts must lie in [1, 65535]; grids are ceil(count/256).
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "baloon/ilu_native_gdn.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace ilu_nat {
namespace pre_impl {
// One span of device memory touched by a launch.
struct Span { uint64_t addr; uint64_t bytes; };
bool valid(const Span& s);
bool disjoint(const Span& a, const Span& b);
}  // namespace pre_impl

namespace {

constexpr int S = 128;
constexpr int kThreads = 256;
constexpr float kSoftplusMax = 20.f;

__device__ __forceinline__ float warp_sum(float v) { return ilu_gdn::lane_reduce(v); }

// Block sum over 256 threads: per-32-lane partials in sums[0..7], every thread reads all 8.
__device__ __forceinline__ float norm_sum(float v, float* sums) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) r += sums[w];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void conv_silu(float* __restrict__ hist, const float* __restrict__ x, const float* __restrict__ w,
                          float* __restrict__ out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float h0 = hist[c * 3 + 0], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2], xv = x[c];
    float acc = h0 * w[c * 4 + 0] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + xv * w[c * 4 + 3];
    acc = acc + 0.0f;
    out[c] = acc * sigmoid(acc);
    hist[c * 3 + 0] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = xv;
}

__global__ void l2_norm(const float* __restrict__ in, float* __restrict__ out, float eps, float scale) {
    __shared__ float sums[32];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * S + col;
    const float v = col < S ? in[off] : 0.f;
    const float sum = norm_sum(v * v, sums);
    if (col < S) out[off] = v * rsqrtf(sum + eps) * scale;
}

__global__ void beta_sigmoid(const float* __restrict__ raw, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sigmoid(raw[i]);
}

__global__ void gate_softplus(const float* __restrict__ alpha, const float* __restrict__ bias,
                              const float* __restrict__ a_neg, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = alpha[i] + bias[i];
    const float sp = v > kSoftplusMax ? v : log1pf(expf(v));
    out[i] = sp * a_neg[i];
}

__global__ void out_norm(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ gamma,
                         float* __restrict__ y, float eps) {
    __shared__ float sums[32];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * S + col;
    const float v = col < S ? o[off] : 0.f;
    const float sum = norm_sum(v * v, sums);
    const float inv = rsqrtf(sum / (float)S + eps);
    if (col < S) y[off] = v * inv * gamma[col] * sigmoid(z[off]);
}

inline unsigned blocks_for(int n) { return (unsigned)((n + kThreads - 1) / kThreads); }

int check_launch() { return (int)cudaGetLastError(); }

using pre_impl::Span;
Span mk(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }

// Every span valid; writable spans (the first nw) pairwise disjoint and disjoint from every read span.
bool layout_ok(const Span* sp, int n, int nw) {
    for (int i = 0; i < n; ++i)
        if (!pre_impl::valid(sp[i])) return false;
    for (int i = 0; i < nw; ++i)
        for (int j = i + 1; j < n; ++j)
            if (!pre_impl::disjoint(sp[i], sp[j])) return false;
    return true;
}

}  // namespace

namespace pre_impl {
bool valid(const Span& s) { return s.addr != 0 && s.addr % 4 == 0 && s.bytes <= UINT64_MAX - s.addr; }
bool disjoint(const Span& a, const Span& b) { return !(a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes); }
bool valid(uint64_t addr, uint64_t bytes) { return valid(Span{addr, bytes}); }
bool disjoint(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return disjoint(Span{a, an}, Span{b, bn}); }
bool count_and_stream(int count, void* stream) { return stream != nullptr && count >= 1 && count <= 65535; }
bool norm_geometry(int rows, int width, int threads, unsigned* grid) {
    if (width != S || threads != kThreads || rows < 1 || rows > 65535) return false;
    if (grid) *grid = (unsigned)rows;
    return true;
}
}  // namespace pre_impl

int native_gdn_conv_silu(float* hist, const float* x, const float* w, float* out, int channels, void* stream) {
    if (!pre_impl::count_and_stream(channels, stream)) return kReject;
    const uint64_t c = (uint64_t)channels;
    const Span sp[] = {mk(hist, c * 12), mk(out, c * 4), mk(x, c * 4), mk(w, c * 16)};
    if (!layout_ok(sp, 4, 2)) return kReject;
    conv_silu<<<blocks_for(channels), kThreads, 0, (cudaStream_t)stream>>>(hist, x, w, out, channels);
    return check_launch();
}

int native_gdn_l2_norm(const float* in, float* out, int rows, float eps, float scale, void* stream) {
    unsigned grid = 0;
    if (!pre_impl::count_and_stream(rows, stream) || !pre_impl::norm_geometry(rows, S, kThreads, &grid)) return kReject;
    const uint64_t b = (uint64_t)rows * S * 4;
    const Span sp[] = {mk(out, b), mk(in, b)};
    if (!layout_ok(sp, 2, 1)) return kReject;
    l2_norm<<<grid, kThreads, 0, (cudaStream_t)stream>>>(in, out, eps, scale);
    return check_launch();
}

int native_gdn_beta_gate(const float* raw, float* out, int n, void* stream) {
    if (!pre_impl::count_and_stream(n, stream)) return kReject;
    const uint64_t b = (uint64_t)n * 4;
    const Span sp[] = {mk(out, b), mk(raw, b)};
    if (!layout_ok(sp, 2, 1)) return kReject;
    beta_sigmoid<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(raw, out, n);
    return check_launch();
}

int native_gdn_gate(const float* alpha, const float* bias, const float* a_neg, float* out, int n, void* stream) {
    if (!pre_impl::count_and_stream(n, stream)) return kReject;
    const uint64_t b = (uint64_t)n * 4;
    const Span sp[] = {mk(out, b), mk(alpha, b), mk(bias, b), mk(a_neg, b)};
    if (!layout_ok(sp, 4, 1)) return kReject;
    gate_softplus<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(alpha, bias, a_neg, out, n);
    return check_launch();
}

int native_gdn_out_norm(const float* o, const float* z, const float* gamma, float* y, int rows, float eps, void* stream) {
    unsigned grid = 0;
    if (!pre_impl::count_and_stream(rows, stream) || !pre_impl::norm_geometry(rows, S, kThreads, &grid)) return kReject;
    const uint64_t b = (uint64_t)rows * S * 4;
    const Span sp[] = {mk(y, b), mk(o, b), mk(z, b), mk(gamma, (uint64_t)S * 4)};
    if (!layout_ok(sp, 4, 1)) return kReject;
    out_norm<<<grid, kThreads, 0, (cudaStream_t)stream>>>(o, z, gamma, y, eps);
    return check_launch();
}

}  // namespace ilu_nat
}  // namespace baloon
