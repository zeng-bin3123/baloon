// ple_native_post.cu - post-operations of the per-layer-embedding (PLE) branch, one block of N threads per row (strided over any width):
//   gate_kernel         : t[i] = gelu_tanh(gate[i]) * ple[i]
//   conv_residual_kernel: out[i] = resid[i] + rmsnorm(proj)[i] * norm_w[i]       (float accumulation, eps inside the root)
//   broadcast_kernel    : out[i] *= scalar[0]                                    (one layer scalar for the whole tensor)
// native_ple_postops runs the three stages; all spans are validated and `out` must not overlap proj / resid / norm_w.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int N = 256;
struct Span { const void* p; size_t bytes; };
bool validate(const Span& s) { return s.p != nullptr && s.bytes > 0; }
bool overlaps(const Span& a, const Span& b) {
    const char* pa = (const char*)a.p; const char* pb = (const char*)b.p;
    return pa < pb + b.bytes && pb < pa + a.bytes;
}
cudaError_t launch_check() { return cudaGetLastError(); }
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }

__global__ void gate_kernel(const float* __restrict__ gate, const float* __restrict__ ple, float* __restrict__ t, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float u = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    t[i] = 0.5f * g * (1.0f + tanhf(u)) * ple[i];
}
__global__ void conv_residual_kernel(const float* __restrict__ proj, const float* __restrict__ nw, const float* __restrict__ resid,
                                     float* __restrict__ out, int cols, float eps) {
    __shared__ float sh[N / 64];
    const float* p = proj + (size_t)blockIdx.x * cols;
    float a = 0.f;
    for (int i = threadIdx.x; i < cols; i += N) a += p[i] * p[i];
    a = warp_sum(a);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = a;
    __syncthreads();
    float tot = 0.f;
    for (int k = 0; k < N / 64; ++k) tot += sh[k];
    const float inv = 1.0f / sqrtf(tot / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += N) {
        const size_t j = (size_t)blockIdx.x * cols + i;
        out[j] = resid[j] + p[i] * inv * nw[i];
    }
}
__global__ void broadcast_kernel(float* __restrict__ out, const float* __restrict__ scalar, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= scalar[0];
}
}  // namespace

cudaError_t native_ple_postops(const float* gate, const float* ple, float* t_out, const float* proj, const float* norm_w, const float* resid,
                               const float* layer_scalar, float* out, int rows, int cols, float eps, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const size_t nb = (size_t)rows * cols * 4;
    const Span g{gate, nb}, pl{ple, nb}, t{t_out, nb}, pr{proj, nb}, nw{norm_w, (size_t)cols * 4}, rs{resid, nb}, ls{layer_scalar, 4}, o{out, nb};
    const Span all[8] = {g, pl, t, pr, nw, rs, ls, o};
    for (int i = 0; i < 8; ++i) if (!validate(all[i])) return cudaErrorInvalidValue;
    if (overlaps(o, pr) || overlaps(o, rs) || overlaps(o, nw) || overlaps(o, ls)) return cudaErrorInvalidValue;
    if (overlaps(t, g) && t_out != gate) return cudaErrorInvalidValue;
    if (overlaps(t, pl)) return cudaErrorInvalidValue;
    const long long total = (long long)rows * cols;
    gate_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(gate, ple, t_out, total);
    conv_residual_kernel<<<rows, N, 0, s>>>(proj, norm_w, resid, out, cols, eps);
    broadcast_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(out, layer_scalar, total);
    return launch_check();
}
}  // namespace kg
}  // namespace baloon
