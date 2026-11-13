// elementwise.cu - glue kernels: embedding gather, GDN gate (softplus), scale/add/silu,
// f16/bf16 bulk conversion, warp-per-row weighted RMS norm, mapped-memory copies,
// host<->device doorbell (ring / wait / publish).
// Iluvatar BI-V100: warpSize=64; use ix_compat.h helpers; __threadfence_system NOT supported.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

// ── constants ────────────────────────────────────────────────────────────────
constexpr int THREADS = 256;

// ── helpers ──────────────────────────────────────────────────────────────────
static inline int grid_for(int n) { return (n + THREADS - 1) / THREADS; }

static inline void check_launch(const char* name) {
    cudaError_t e = cudaGetLastError(); (void)name; (void)e;
}

static inline bool sync_if_needed(cudaStream_t s) {
    if (!s) return cudaDeviceSynchronize() == cudaSuccess;
    return true;
}

__device__ __forceinline__ float softplus_dev(float x) {
    return x > 20.f ? x : log1pf(expf(x));
}

// Warp-level sum: full warp (64 lanes on Iluvatar).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#ifdef __ILUVATAR__
    return ix_warp_reduce_sum(v);
#else
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
#endif
}

// ── scale kernel ─────────────────────────────────────────────────────────────
__global__ void scale_kernel(float* __restrict__ x, float s, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i < n) x[i] *= s;
}
void scale_inplace(float* x, float s, int n, cudaStream_t stream) {
    if (n <= 0) return;
    scale_kernel<<<grid_for(n), THREADS, 0, stream>>>(x, s, n);
    check_launch("scale_inplace");
}

// ── add kernel ───────────────────────────────────────────────────────────────
__global__ void add_kernel(float* __restrict__ dst, const float* __restrict__ src, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i < n) dst[i] += src[i];
}
void add_inplace(float* dst, const float* src, int n, cudaStream_t stream) {
    if (n <= 0) return;
    add_kernel<<<grid_for(n), THREADS, 0, stream>>>(dst, src, n);
    check_launch("add_inplace");
}

// ── silu kernel ───────────────────────────────────────────────────────────────
__global__ void silu_kernel(float* __restrict__ x, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i < n) { float v = x[i]; x[i] = v / (1.f + expf(-v)); }
}
void silu_inplace(float* x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    silu_kernel<<<grid_for(n), THREADS, 0, stream>>>(x, n);
    check_launch("silu_inplace");
}

// ── GDN gate kernel (softplus) ────────────────────────────────────────────────
__global__ void gdn_gate_kernel(float* __restrict__ gate, const float* __restrict__ alpha,
                                 const float* __restrict__ dt,  const float* __restrict__ ssm_a,
                                 int h_v, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i < n) {
        const int h = i % h_v;
        gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    }
}
void gdn_gate(float* gate, const float* alpha, const float* dt, const float* ssm_a,
               int h_v, int n, cudaStream_t stream) {
    if (n <= 0) return;
    gdn_gate_kernel<<<grid_for(n), THREADS, 0, stream>>>(gate, alpha, dt, ssm_a, h_v, n);
    check_launch("gdn_gate");
}

// ── RMS norm (one warp per row, Iluvatar warp=64) ────────────────────────────
// grid = rows; block = 64 threads (1 full Iluvatar warp); each block handles 1 row.
#ifdef __ILUVATAR__
  #define WARP_SIZE_EL 64
#else
  #define WARP_SIZE_EL 32
#endif

__global__ void rms_norm_weighted_kernel(const float* __restrict__ x,
                                          const float* __restrict__ w,
                                          float* __restrict__ out,
                                          int rows, int cols, float eps) {
    const int row  = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x;    // 0..WARP_SIZE_EL-1
    const float* xr = x + (size_t)row * cols;
    const float* wr = w + (size_t)row * cols;
    float*       or_ = out + (size_t)row * cols;
    float ss = 0.f;
    for (int c = lane; c < cols; c += WARP_SIZE_EL) ss += xr[c] * xr[c];
    ss = warp_reduce_sum(ss);
    // lane-0 of the warp has the total; broadcast via shuffle.
#ifdef __ILUVATAR__
    ss = ix_shfl(ss, 0, 64);
#else
    ss = __shfl_sync(0xffffffff, ss, 0);
#endif
    const float scale = rsqrtf(ss / (float)cols + eps);
    for (int c = lane; c < cols; c += WARP_SIZE_EL) or_[c] = scale * xr[c] * wr[c];
}
void rms_norm_weighted(const float* x, const float* w, float* out,
                        int rows, int cols, float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    // 1 warp (64 on Iluvatar, 32 on NVIDIA) per row.
    rms_norm_weighted_kernel<<<rows, WARP_SIZE_EL, 0, stream>>>(x, w, out, rows, cols, eps);
    check_launch("rms_norm_weighted");
}

// ── embedding gather ──────────────────────────────────────────────────────────
__global__ void embedding_gather_kernel(const uint8_t* __restrict__ codes,
                                         const float*   __restrict__ scales,
                                         const float*   __restrict__ offsets,
                                         float* __restrict__ out,
                                         int n, int bits, int bias, int group_size,
                                         bool has_offset) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    const int per_byte = 8 / bits;
    const int byte_idx = i / per_byte;
    const int slot     = i % per_byte;
    const int mask     = (1 << bits) - 1;
    const int code     = (codes[byte_idx] >> (slot * bits)) & mask;
    const int g        = i / group_size;
    float v = (float)(code + bias) * scales[g];
    if (has_offset) v += offsets[g];
    out[i] = v;
}
void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                       float* out, int n, int bits, int bias, int group_size,
                       bool has_offset, cudaStream_t stream) {
    if (n <= 0) return;
    embedding_gather_kernel<<<grid_for(n), THREADS, 0, stream>>>(
        codes, scales, offsets, out, n, bits, bias, group_size, has_offset);
    check_launch("embedding_gather");
}

// ── f32 -> bf16 / f16 bulk ────────────────────────────────────────────────────
__global__ void to_bf16_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    uint32_t u; __builtin_memcpy(&u, &src[i], 4);
    uint32_t rnd = u + ((u >> 16) & 1) + 0x7FFFu;
    dst[i] = (uint16_t)(rnd >> 16);
}
void f32_to_bf16_bulk(const float* src, uint16_t* dst, int n, cudaStream_t stream) {
    if (n <= 0) return;
    to_bf16_kernel<<<grid_for(n), THREADS, 0, stream>>>(src, dst, n);
    check_launch("f32_to_bf16_bulk");
}

__global__ void to_f16_kernel(const float* __restrict__ src, uint16_t* __restrict__ dst, int n) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    if (i >= n) return;
    __half h = __float2half(src[i]);
    uint16_t u; __builtin_memcpy(&u, &h, 2);
    dst[i] = u;
}
void f32_to_f16_bulk(const float* src, uint16_t* dst, int n, cudaStream_t stream) {
    if (n <= 0) return;
    to_f16_kernel<<<grid_for(n), THREADS, 0, stream>>>(src, dst, n);
    check_launch("f32_to_f16_bulk");
}

// ── mapped-memory copies ──────────────────────────────────────────────────────
__global__ void copy_from_mapped_kernel(const float* __restrict__ src,
                                         float* __restrict__ dst, int n4) {
    const float4* s = reinterpret_cast<const float4*>(src);
    float4*       d = reinterpret_cast<float4*>(dst);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
        d[i] = s[i];
}
void copy_from_mapped(const float* src, float* dst, int n, cudaStream_t stream) {
    if (n <= 0 || n % 4 != 0) return;
    const int n4     = n / 4;
    const int blocks = (n4 + THREADS - 1) / THREADS < 64 ? (n4 + THREADS - 1) / THREADS : 64;
    copy_from_mapped_kernel<<<blocks, THREADS, 0, stream>>>(src, dst, n4);
    check_launch("copy_from_mapped");
}

__global__ void copy_i32_from_mapped_kernel(const int32_t* __restrict__ src,
                                              int32_t* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}
void copy_i32_from_mapped(const int32_t* src, int32_t* dst, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int blocks = (n + THREADS - 1) / THREADS < 64 ? (n + THREADS - 1) / THREADS : 64;
    copy_i32_from_mapped_kernel<<<blocks, THREADS, 0, stream>>>(src, dst, n);
    check_launch("copy_i32_from_mapped");
}

// ── doorbell ──────────────────────────────────────────────────────────────────
// ring: increment *seq in memory after a device fence (__threadfence_system NOT on Iluvatar).
__global__ void doorbell_ring_kernel(volatile uint32_t* __restrict__ seq) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef __ILUVATAR__
        ix_threadfence();
#else
        __threadfence_system();
#endif
        atomicAdd((uint32_t*)seq, 1u);
    }
}
void doorbell_ring(volatile uint32_t* seq, cudaStream_t stream) {
    doorbell_ring_kernel<<<1, 1, 0, stream>>>(seq);
    check_launch("doorbell_ring");
}

__global__ void doorbell_wait_kernel(volatile uint32_t* __restrict__ flag, uint32_t seq) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        while (*flag != seq) {}
}
void doorbell_wait(volatile uint32_t* flag, uint32_t seq, cudaStream_t stream) {
    doorbell_wait_kernel<<<1, 1, 0, stream>>>(flag, seq);
    check_launch("doorbell_wait");
}

__global__ void doorbell_publish_kernel(volatile uint32_t* __restrict__ dst, uint32_t val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef __ILUVATAR__
        ix_threadfence();
#else
        __threadfence_system();
#endif
        *dst = val;
    }
}
void doorbell_publish(volatile uint32_t* dst, uint32_t val, cudaStream_t stream) {
    doorbell_publish_kernel<<<1, 1, 0, stream>>>(dst, val);
    check_launch("doorbell_publish");
}
