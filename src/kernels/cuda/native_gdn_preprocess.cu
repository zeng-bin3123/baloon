// native_gdn_preprocess.cu - llama.cpp-pinned GDN preprocessing kernels:
// 4-tap causal conv + SiLU, per-row L2 norm, beta sigmoid, gate softplus, out RMS+sigmoid.
// History per channel: 3 floats at hist[c*3..c*3+2].  Norm rows: width exactly 128.
// Counts must lie in [1, 65535]; all writable spans must be pairwise disjoint.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "baloon/ilu_native_gdn.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace ilu_nat {
namespace {

// § constants
constexpr int S        = 128;    // norm width (always 128)
constexpr int kThreads = 256;    // threads per norm block

// § one span of device memory
struct Span { uint64_t addr; uint64_t bytes; };

// § helpers

__device__ __forceinline__ float warp_sum(float v) { return ilu_gdn::lane_reduce(v); }

// 256-thread block sum: accumulate per-32-lane partial in sums[0..7].
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

// § check that the last CUDA call succeeded
inline int check_launch() { return (int)cudaGetLastError(); }

// § span helpers
inline bool valid(const Span& s) {
    return s.addr != 0 && s.addr % 4 == 0 && s.bytes <= (uint64_t)UINT64_MAX - s.addr;
}
inline bool disjoint(const Span& a, const Span& b) {
    return !(a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes);
}

// § validation
inline bool count_and_stream(int count, void* stream) {
    return stream != nullptr && count >= 1 && count <= 65535;
}
inline bool norm_geometry(int rows, int width, unsigned* grid) {
    if (width != S || rows < 1 || rows > 65535) return false;
    if (grid) *grid = (unsigned)rows;
    return true;
}
inline Span mk(const void* p, uint64_t b) { return Span{(uint64_t)(uintptr_t)p, b}; }
inline bool layout_ok(const Span* sp, int n, int nw) {
    for (int i = 0; i < n; ++i) if (!valid(sp[i])) return false;
    for (int i = 0; i < nw; ++i)
        for (int j = i + 1; j < n; ++j)
            if (!disjoint(sp[i], sp[j])) return false;
    return true;
}

// § kernels

// 4-tap conv + SiLU: acc = h0*w0 + h1*w1 + h2*w2 + x*w3 + 0, out = acc*sigmoid(acc).
// History slides: [h0,h1,h2] -> [h1,h2,x].
__global__ void conv_silu(float* __restrict__ hist, const float* __restrict__ x,
                           const float* __restrict__ w, float* __restrict__ out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float h0 = hist[c * 3 + 0], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2];
    const float xv = x[c];
    float acc = h0 * w[c * 4 + 0] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + xv * w[c * 4 + 3];
    acc = acc + 0.0f;   // pinned zero-bias add
    out[c] = acc * sigmoid(acc);
    hist[c * 3 + 0] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = xv;
}

// Row L2 norm: x * rsqrt(sum(x^2)/S + eps/S) / sqrt(S) == x * rsqrt(sum(x^2) + eps).
// 256 threads per row; only threads < 128 carry data.
__global__ void l2_norm(const float* __restrict__ in, float* __restrict__ out, float eps, float scale) {
    __shared__ float sums[8];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * S + col;
    const float v = col < S ? in[off] : 0.f;
    const float ss = norm_sum(v * v, sums);
    if (col < S) out[off] = v * rsqrtf(ss + eps) * scale;
}

// out[i] = sigmoid(raw[i])
__global__ void beta_sigmoid(const float* __restrict__ raw, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sigmoid(raw[i]);
}

// out[i] = softplus(alpha[i] + bias[i]) * a_neg[i]
__global__ void gate_softplus(const float* __restrict__ alpha, const float* __restrict__ bias,
                               const float* __restrict__ a_neg, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = alpha[i] + bias[i];
    const float sp = v > 20.f ? v : log1pf(expf(v));
    out[i] = sp * a_neg[i];
}

// y[r][c] = o[r][c] * rsqrt(mean(o^2) + eps) * gamma[c] * sigmoid(z[r][c])
__global__ void out_norm(const float* __restrict__ o, const float* __restrict__ z,
                          const float* __restrict__ gamma, float* __restrict__ y, float eps) {
    __shared__ float sums[8];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * S + col;
    const float v = col < S ? o[off] : 0.f;
    const float ss = norm_sum(v * v, sums);
    if (col < S) {
        const float scale = rsqrtf(ss / S + eps);
        y[off] = v * scale * gamma[col] * sigmoid(z[off]);
    }
}

static unsigned blocks_for(int n) { return ((unsigned)n + kThreads - 1u) / kThreads; }

}  // anonymous namespace

// § public API ──────────────────────────────────────────────────────────────────

int native_gdn_conv_silu(float* hist, const float* x, const float* w, float* out,
                          int channels, void* stream) {
    if (!count_and_stream(channels, stream)) return kReject;
    const uint64_t c = (uint64_t)channels;
    const Span sp[] = {mk(hist, c * 12), mk(out, c * 4), mk(x, c * 4), mk(w, c * 16)};
    if (!layout_ok(sp, 4, 2)) return kReject;
    conv_silu<<<blocks_for(channels), kThreads, 0, (cudaStream_t)stream>>>(hist, x, w, out, channels);
    return check_launch();
}

int native_gdn_l2_norm(const float* in, float* out, int rows, float eps, float scale, void* stream) {
    unsigned grid = 0;
    if (!count_and_stream(rows, stream) || !norm_geometry(rows, S, &grid)) return kReject;
    const uint64_t b = (uint64_t)rows * S * 4;
    const Span sp[] = {mk(out, b), mk(in, b)};
    if (!layout_ok(sp, 2, 1)) return kReject;
    l2_norm<<<grid, kThreads, 0, (cudaStream_t)stream>>>(in, out, eps / S, scale);
    return check_launch();
}

int native_gdn_beta_gate(const float* raw, float* out, int n, void* stream) {
    if (!count_and_stream(n, stream)) return kReject;
    const uint64_t b = (uint64_t)n * 4;
    const Span sp[] = {mk(out, b), mk(raw, b)};
    if (!layout_ok(sp, 2, 1)) return kReject;
    beta_sigmoid<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(raw, out, n);
    return check_launch();
}

int native_gdn_gate(const float* alpha, const float* bias, const float* a_neg, float* out,
                    int n, void* stream) {
    if (!count_and_stream(n, stream)) return kReject;
    const uint64_t b = (uint64_t)n * 4;
    const Span sp[] = {mk(out, b), mk(alpha, b), mk(bias, b), mk(a_neg, b)};
    if (!layout_ok(sp, 4, 1)) return kReject;
    gate_softplus<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(alpha, bias, a_neg, out, n);
    return check_launch();
}

int native_gdn_out_norm(const float* o, const float* z, const float* gamma, float* y,
                        int rows, float eps, void* stream) {
    unsigned grid = 0;
    if (!count_and_stream(rows, stream) || !norm_geometry(rows, S, &grid)) return kReject;
    const uint64_t b = (uint64_t)rows * S * 4;
    const Span sp[] = {mk(y, b), mk(o, b), mk(z, b), mk(gamma, (uint64_t)S * 4)};
    if (!layout_ok(sp, 4, 1)) return kReject;
    out_norm<<<grid, kThreads, 0, (cudaStream_t)stream>>>(o, z, gamma, y, eps);
    return check_launch();
}

}  // namespace ilu_nat
}  // namespace baloon

