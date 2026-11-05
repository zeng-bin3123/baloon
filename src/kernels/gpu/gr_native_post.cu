// gr_native_post.cu - element-wise post-operations of the gated-residual path with shape/pointer validation:
//   pre_gated : gated[i] = xn[i] * sigmoid(logit[i])
//   down_silu : h[i] = silu(h[i]) in place
//   post      : out[i] = R[i] + block_out[i % n] * (w[c] * scale + bias[c]),  c = i / n  (bias optional = zero bias)
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int THREADS = 256;
long long blocks(long long n) { return (n + THREADS - 1) / THREADS; }
cudaError_t check_launch() { return cudaGetLastError(); }
bool check_pointer(const void* p) { return p != nullptr; }
bool check_shape(long long n, long long hc = 1) { return n > 0 && hc > 0 && n <= (1ll << 40) / hc; }
__device__ __forceinline__ float sigmoid(float x) { return kgc::sigmoid(x); }
__device__ __forceinline__ float scale_zero_bias(float v, float scale, const float* bias, int c) { return v * scale + (bias ? bias[c] : 0.f); }

__global__ void pre_gated(const float* __restrict__ xn, const float* __restrict__ logit, float* __restrict__ gated, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gated[i] = xn[i] * sigmoid(logit[i]);
}
__global__ void down_silu(float* __restrict__ h, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = kgc::silu(h[i]);
}
__global__ void post(const float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ w, const float* __restrict__ bias,
                     float scale, float* __restrict__ out, int n, long long total) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i / n);
    out[i] = R[i] + bo[i % n] * scale_zero_bias(w[c], scale, bias, c);
}
}  // namespace

cudaError_t native_gr_pre_gated(const float* xn, const float* logit, float* gated, long long n, cudaStream_t s) {
    if (!check_shape(n) || !check_pointer(xn) || !check_pointer(logit) || !check_pointer(gated)) return cudaErrorInvalidValue;
    pre_gated<<<(unsigned)blocks(n), THREADS, 0, s>>>(xn, logit, gated, n);
    return check_launch();
}
cudaError_t native_gr_down_silu(float* h, long long n, cudaStream_t s) {
    if (!check_shape(n) || !check_pointer(h)) return cudaErrorInvalidValue;
    down_silu<<<(unsigned)blocks(n), THREADS, 0, s>>>(h, n);
    return check_launch();
}
cudaError_t native_gr_post(const float* R, const float* block_out, const float* w, const float* bias, float scale, float* out, int n, int hc,
                           cudaStream_t s) {
    if (!check_shape(n, hc) || !check_pointer(R) || !check_pointer(block_out) || !check_pointer(w) || !check_pointer(out)) return cudaErrorInvalidValue;
    const long long total = (long long)n * hc;
    post<<<(unsigned)blocks(total), THREADS, 0, s>>>(R, block_out, w, bias, scale, out, n, total);
    return check_launch();
}
}  // namespace kg
}  // namespace baloon
