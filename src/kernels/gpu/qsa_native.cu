// qsa_native.cu - attention-side element operators of the quantized-sparse-attention path:
//   native_qsa_gate_apply          : out[i] *= sigmoid(gate[i])                      (attention output gate)
//   native_qsa_rms_norm_weighted   : per head row, x = x * rsqrt(mean(x^2)+eps) * w  (in place, one wave per row, w shared by rows)
// Buffers are validated (non-null, element counts consistent, no partial overlap between gate and out).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
bool g_enabled = true;
bool valid(const void* p, size_t bytes) { return p != nullptr && bytes > 0; }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}
bool elements(long long rows, long long cols) { return rows > 0 && cols > 0 && rows <= (1ll << 31) && cols <= (1 << 20); }
bool buffers(const void* a, const void* b, size_t bytes) { return valid(a, bytes) && valid(b, bytes) && !overlap(a, bytes, b, bytes); }
cudaError_t check_launch() { return cudaGetLastError(); }
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }
__device__ __forceinline__ float gate_value(float g) { return kgc::sigmoid(g); }

__global__ void gate(float* __restrict__ out, const float* __restrict__ g, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= gate_value(g[i]);
}
// 4 rows per block (one wave each)
__global__ void norm(float* __restrict__ x, const float* __restrict__ w, int rows, int cols, float eps) {
    const int r = blockIdx.x * 4 + (threadIdx.x >> 6);
    if (r >= rows) return;
    const int lane = threadIdx.x & 63;
    float* p = x + (size_t)r * cols;
    float a = 0.f;
    for (int i = lane; i < cols; i += 64) a += p[i] * p[i];
    a = warp_sum(a);
    const float inv = 1.0f / sqrtf(a / (float)cols + eps);
    for (int i = lane; i < cols; i += 64) p[i] = p[i] * inv * w[i];
}
}  // namespace

bool native_qsa_enabled() { return g_enabled; }
void native_qsa_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_gate_apply(float* out, const float* gate_v, long long n, cudaStream_t s) {
    if (n <= 0) return cudaErrorInvalidValue;
    if (!buffers(out, gate_v, (size_t)n * 4)) return cudaErrorInvalidValue;
    gate<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(out, gate_v, n);
    return check_launch();
}
cudaError_t native_qsa_rms_norm_weighted(float* x, const float* w, int rows, int cols, float eps, cudaStream_t s) {
    if (!elements(rows, cols) || !valid(x, (size_t)rows * cols * 4) || !valid(w, (size_t)cols * 4)) return cudaErrorInvalidValue;
    if (overlap(x, (size_t)rows * cols * 4, w, (size_t)cols * 4)) return cudaErrorInvalidValue;
    norm<<<(rows + 3) / 4, 256, 0, s>>>(x, w, rows, cols, eps);
    return check_launch();
}
}  // namespace kg
}  // namespace baloon
