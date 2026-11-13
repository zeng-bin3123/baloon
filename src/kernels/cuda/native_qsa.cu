// native_qsa.cu - Native (llama.cpp-pinned) QSA helpers for BI-V100:
//   native_qsa_gate_apply     : out[i] *= sigmoid(gate[i])      (attention output gate)
//   native_qsa_rms_norm_weighted : per-head RMS norm with learned weights (in-place)
// Return codes: cudaError_t.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace {

static bool g_enabled = true;

// ---- span / buffer checks -------------------------------------------------------
struct Span { uint64_t addr; uint64_t bytes; };
static bool valid(const Span& s) {
    return s.addr != 0 && s.bytes != 0 && s.addr % 4 == 0 && s.addr + s.bytes > s.addr;
}
static bool overlap(const Span& a, const Span& b) {
    return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes;
}
static Span span(const void* p, uint64_t n) { return Span{(uint64_t)(uintptr_t)p, n}; }

static int check_launch(cudaError_t e) { return e == cudaSuccess ? 0 : (int)e; }

static bool elements(int cols, int rows) {
    return cols > 0 && rows > 0 && (int64_t)cols * rows <= 2147483647;
}

static bool buffers(const Span& inp, const Span& out) {
    return inp.addr == out.addr || !overlap(inp, out);
}

// ---- warp reduction -------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---- gate kernel: out[i] *= sigmoid(gate[i]) ------------------------------------
__global__ void gate(float* __restrict__ out, const float* __restrict__ gate_v, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] *= 1.f / (1.f + expf(-gate_v[i]));
}

// ---- norm kernel: in-place per-head RMS norm ------------------------------------
__global__ void norm(float* __restrict__ x, const float* __restrict__ w,
                     int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float* p = x + (int64_t)row * cols;
    // Compute mean-square
    float ss = 0.f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) ss += p[j] * p[j];
    ss = warp_sum(ss);
    // Reduce across warps via shared memory
    __shared__ float sms;
    if (threadIdx.x == 0) sms = 0.f;
    __syncthreads();
    if (threadIdx.x < 32) atomicAdd(&sms, ss);
    __syncthreads();
    const float scale = rsqrtf(sms / cols + eps);
    // Apply scale * w
    for (int j = threadIdx.x; j < cols; j += blockDim.x) p[j] = p[j] * scale * w[j];
}

} // anonymous namespace

bool native_qsa_enabled() { return g_enabled; }
void native_qsa_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_gate_apply(float* out, const float* gate_v, long long n, cudaStream_t s) {
    if (!out || !gate_v || n <= 0) return cudaErrorInvalidValue;
    if (!g_enabled) return cudaSuccess;
    const int block = 256;
    const int grid = (int)((n + block - 1) / block);
    gate<<<grid, block, 0, s>>>(out, gate_v, n);
    return cudaGetLastError();
}

cudaError_t native_qsa_rms_norm_weighted(float* x, const float* w, int rows, int cols,
                                          float eps, cudaStream_t s) {
    if (!x || !w || !elements(cols, rows)) return cudaErrorInvalidValue;
    if (!g_enabled) return cudaSuccess;
    const int block = (cols < 1024) ? 256 : 1024;
    norm<<<rows, block, 0, s>>>(x, w, rows, cols, eps);
    return cudaGetLastError();
}

} // namespace baloon
