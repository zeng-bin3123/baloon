// gr_native_norm.cu - weighted RMS norm for the gated-residual path: y = x * (1 / sqrt(mean(x^2) + eps)) * w, float accumulation,
// one block per row. y may alias x (each element is read before it is written by the same thread).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
bool check_pointer(const void* p, size_t bytes) { return p != nullptr && bytes > 0; }
__device__ __forceinline__ float norm_warp_sum(float v) { return kgc::wave_sum(v); }

__global__ void weighted_rms_norm(const float* __restrict__ x, const float* __restrict__ w, float* y, int cols, float eps) {
    __shared__ float sh[4];
    const float* px = x + (size_t)blockIdx.x * cols;
    float a = 0.f;
    for (int i = threadIdx.x; i < cols; i += 256) a += px[i] * px[i];
    a = norm_warp_sum(a);
    if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = a;
    __syncthreads();
    const float tot = ((sh[0] + sh[1]) + sh[2]) + sh[3];
    const float inv = 1.0f / sqrtf(tot / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += 256) y[(size_t)blockIdx.x * cols + i] = px[i] * inv * w[i];
}
}  // namespace

cudaError_t native_gr_rms_norm_weighted(const float* x, const float* w, float* y, int rows, int cols, float eps, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    if (!check_pointer(x, (size_t)rows * cols * 4) || !check_pointer(w, (size_t)cols * 4) || !check_pointer(y, (size_t)rows * cols * 4)) return cudaErrorInvalidValue;
    if (y != x) {   // partial overlap is never allowed, full aliasing is
        const char* a = (const char*)x; const char* b = (const char*)y; const size_t n = (size_t)rows * cols * 4;
        if (a < b + n && b < a + n) return cudaErrorInvalidValue;
    }
    weighted_rms_norm<<<rows, 256, 0, s>>>(x, w, y, cols, eps);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
