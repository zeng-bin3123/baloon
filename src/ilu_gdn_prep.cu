// ilu_gdn_prep.cu - GDN preprocessing kernels for BI-V100: conv+SiLU with 3-deep history, 128-wide row norms,
// beta sigmoid, gate softplus and output RMSNorm with sigmoid gate. Norm kernels use 256 threads per row;
// threads 128..255 contribute zero to the block sum.
#include <cuda_runtime.h>
#include "baloon/ilu_gdn_kernels.hpp"
#include "ilu_gdn_shfl.cuh"

namespace baloon {
namespace ilu_gdn {
namespace {

constexpr int kThreads = native_gdn_pre::kThreads;  // 256
constexpr int kRow = native_gdn_pre::kS;            // 128

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

__global__ void conv_silu_kernel(float* __restrict__ hist, const float* __restrict__ x, const float* __restrict__ w,
                                 float* __restrict__ out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float h0 = hist[c * 3 + 0], h1 = hist[c * 3 + 1], h2 = hist[c * 3 + 2], xv = x[c];
    const float acc = h0 * w[c * 4 + 0] + h1 * w[c * 4 + 1] + h2 * w[c * 4 + 2] + xv * w[c * 4 + 3];
    out[c] = acc / (1.f + expf(-acc));
    hist[c * 3 + 0] = h1;
    hist[c * 3 + 1] = h2;
    hist[c * 3 + 2] = xv;
}

__global__ void row_norm_kernel(float* __restrict__ x, float eps, int rms) {
    __shared__ float sums[32];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * kRow + col;
    const float v = col < kRow ? x[off] : 0.f;
    const float sum = block_sum256(v * v, sums);
    const float inv = rms ? rsqrtf(sum / (float)kRow + eps) : rsqrtf(sum + eps);
    if (col < kRow) x[off] = v * inv;
}

__global__ void beta_sigmoid_kernel(const float* __restrict__ raw, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = sigmoidf_(raw[i]);
}

__global__ void gate_kernel(const float* __restrict__ alpha, const float* __restrict__ bias,
                            const float* __restrict__ a_neg, float* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = alpha[i] + bias[i];
    const float sp = v > kSoftplusThreshold ? v : log1pf(expf(v));
    out[i] = sp * a_neg[i];
}

__global__ void out_norm_kernel(const float* __restrict__ o, const float* __restrict__ z, const float* __restrict__ w,
                                float* __restrict__ y, float eps) {
    __shared__ float sums[32];
    const int col = threadIdx.x;
    const size_t off = (size_t)blockIdx.x * kRow + col;
    const float v = col < kRow ? o[off] : 0.f;
    const float sum = block_sum256(v * v, sums);
    const float inv = rsqrtf(sum / (float)kRow + eps);
    if (col < kRow) y[off] = v * inv * w[col] * sigmoidf_(z[off]);
}

inline int blocks_for(int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

int conv_silu(float* hist, const float* x, const float* w, float* out, int channels, void* stream) {
    if (!native_gdn_pre::count_ok(channels)) return kInvalid;
    const uint64_t c = (uint64_t)channels;
    const Region rg[] = {region(hist, c * 12, true), region(out, c * 4, true), region(x, c * 4, false),
                         region(w, c * 16, false)};
    if (!regions_ok(rg, 4)) return kInvalid;
    conv_silu_kernel<<<blocks_for(channels), kThreads, 0, (cudaStream_t)stream>>>(hist, x, w, out, channels);
    return (int)cudaGetLastError();
}

int row_norm(float* x, int rows, float eps, bool rms, void* stream) {
    if (!native_gdn_pre::count_ok(rows)) return kInvalid;
    const Region rg[] = {region(x, (uint64_t)rows * kRow * 4, true)};
    if (!regions_ok(rg, 1)) return kInvalid;
    row_norm_kernel<<<rows, kThreads, 0, (cudaStream_t)stream>>>(x, eps, rms ? 1 : 0);
    return (int)cudaGetLastError();
}

int beta_sigmoid(const float* raw, float* out, int n, void* stream) {
    if (!native_gdn_pre::count_ok(n)) return kInvalid;
    const Region rg[] = {region(out, (uint64_t)n * 4, true), region(raw, (uint64_t)n * 4, false)};
    if (!regions_ok(rg, 2)) return kInvalid;
    beta_sigmoid_kernel<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(raw, out, n);
    return (int)cudaGetLastError();
}

int gate_from_alpha(const float* alpha, const float* bias, const float* a_neg, float* out, int n, void* stream) {
    if (!native_gdn_pre::count_ok(n)) return kInvalid;
    const uint64_t b = (uint64_t)n * 4;
    const Region rg[] = {region(out, b, true), region(alpha, b, false), region(bias, b, false), region(a_neg, b, false)};
    if (!regions_ok(rg, 4)) return kInvalid;
    gate_kernel<<<blocks_for(n), kThreads, 0, (cudaStream_t)stream>>>(alpha, bias, a_neg, out, n);
    return (int)cudaGetLastError();
}

int out_rms_gate(const float* o, const float* z, const float* w, float* y, int rows, float eps, void* stream) {
    if (!native_gdn_pre::count_ok(rows)) return kInvalid;
    const uint64_t rb = (uint64_t)rows * kRow * 4;
    const Region rg[] = {region(y, rb, true), region(o, rb, false), region(z, rb, false),
                         region(w, (uint64_t)kRow * 4, false)};
    if (!regions_ok(rg, 4)) return kInvalid;
    out_norm_kernel<<<rows, kThreads, 0, (cudaStream_t)stream>>>(o, z, w, y, eps);
    return (int)cudaGetLastError();
}

}  // namespace ilu_gdn
}  // namespace baloon
