// gr.cu - gated residual (hyper-connection): gr_read, gr_write, workspace helpers.
// Iluvatar BI-V100: warpSize=64. All warp reductions use ix_warp_reduce_sum.
// Symbols: THREADS, WARPS, activation_f32, bf16_bits_to_f32, block_sum, block_sumf,
// f32_to_bf16_bits, fp32_activations, gr_down_kernel, gr_gate_kernel, gr_inject_kernel,
// gr_mean_kernel, gr_norm_kernel, gr_read, gr_set_fp32_activations, gr_set_native_mmvf,
// gr_workspace_init, gr_write, gr_write_kernel, native_mmvf, sigmoid_f, silu_f,
// store_activation, warp_sum, warp_sumf.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

#include "baloon/gr_gated_residual.hpp"

// Top-level const definitions (port-table targets; initial/default values).
// Runtime-mutable state lives in the atomics inside namespace baloon::kern.
static const bool fp32_activations = false;
static const bool native_mmvf = false;

namespace baloon {
namespace kern {

// ── constants ────────────────────────────────────────────────────────────────
constexpr int THREADS = 256;
constexpr int WARPS   = 8;   // 256-thread block / 32-lane "sub-warp" for weight chunking

// ── global flags ─────────────────────────────────────────────────────────────
namespace {
std::atomic<bool> fp32_activations{false};
std::atomic<bool> native_mmvf{false};
}

void gr_set_fp32_activations(bool on) { fp32_activations.store(on, std::memory_order_relaxed); }
void gr_set_native_mmvf(bool on) { native_mmvf.store(on, std::memory_order_relaxed); }

// ── bf16 helpers ──────────────────────────────────────────────────────────────
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; __builtin_memcpy(&f, &u, 4); return f;
}
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
    uint32_t u; __builtin_memcpy(&u, &f, 4);
    u += ((u >> 16) & 1) + 0x7FFFu;
    return (uint16_t)(u >> 16);
}

// ── activation helpers ────────────────────────────────────────────────────────
__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }
__device__ __forceinline__ float activation_f32(float x, bool do_silu) {
    return do_silu ? silu_f(x) : x;
}
__device__ __forceinline__ void store_activation(float* buf, int idx, float val) {
    buf[idx] = val;
}

// ── warp / block sums (Iluvatar warp=64) ──────────────────────────────────────
__device__ __forceinline__ float warp_sum(float v) {
#ifdef __ILUVATAR__
    return ix_warp_reduce_sum(v);
#else
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
#endif
}
__device__ __forceinline__ float warp_sumf(float v) { return warp_sum(v); }

__device__ float block_sum(float v, float* shmem) {
#ifdef __ILUVATAR__
    const int wid  = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int nw   = blockDim.x / 64;
#else
    const int wid  = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nw   = blockDim.x / 32;
#endif
    float ws = warp_sum(v);
    if (lane == 0) shmem[wid] = ws;
    __syncthreads();
    if (threadIdx.x < nw) {
        float acc = shmem[threadIdx.x];
#ifdef __ILUVATAR__
        for (int off = nw / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 64);
#else
        for (int off = nw / 2; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
#endif
        if (threadIdx.x == 0) shmem[0] = acc;
    }
    __syncthreads();
    return shmem[0];
}
__device__ float block_sumf(float v, float* shmem) { return block_sum(v, shmem); }

// ── norm kernel ───────────────────────────────────────────────────────────────
__global__ void gr_norm_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int n, float eps) {
    extern __shared__ float shm[];
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const float* r = in  + (size_t)row * n;
    float*       o = out + (size_t)row * n;
    float ss = 0.f;
    for (int i = tid; i < n; i += THREADS) ss += r[i] * r[i];
    ss = block_sumf(ss, shm);
    const float scale = rsqrtf(ss / (float)n + eps);
    for (int i = tid; i < n; i += THREADS) o[i] = scale * r[i];
}

// ── down kernel ───────────────────────────────────────────────────────────────
__global__ void gr_down_kernel(const float* __restrict__ xn, const uint16_t* __restrict__ w,
                                 float* __restrict__ lo, int in_dim, int out_dim) {
    const int row = blockIdx.x;
    if (row >= out_dim) return;
    const uint16_t* wr = w + (size_t)row * in_dim;
    const int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < in_dim; i += blockDim.x)
        acc += bf16_bits_to_f32(wr[i]) * xn[i];
    extern __shared__ float shm[];
    acc = block_sum(acc, shm);
    if (tid == 0) lo[row] = silu_f(acc);
}

// ── gate kernel ───────────────────────────────────────────────────────────────
// 8 warps per block; row = block*8 + warp_index.
// On Iluvatar: 1 warp = 64 lanes. Use 32-lane sub-groups (half-wave) to match reference.
__global__ void gr_gate_kernel(const float* __restrict__ lo, const uint16_t* __restrict__ w_up,
                                 float* __restrict__ gate, int lr, int out_dim) {
#ifdef __ILUVATAR__
    // With warp=64, block=8*64=512 threads; row = block*8 + warp
    const int warp = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
#else
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
#endif
    const int row = blockIdx.x * gr::kGateRowsPerBlock + warp;
    if (row >= out_dim) return;
    const uint16_t* wr = w_up + (size_t)row * lr;
    float acc = 0.f;
#ifdef __ILUVATAR__
    for (int i = lane; i < lr; i += 64) acc += bf16_bits_to_f32(wr[i]) * lo[i];
    acc = ix_warp_reduce_sum(acc);
    if (lane == 0) gate[row] = sigmoid_f(acc);
#else
    for (int i = lane; i < lr; i += 32) acc += bf16_bits_to_f32(wr[i]) * lo[i];
    acc = warp_sum(acc);
    if (lane == 0) gate[row] = sigmoid_f(acc);
#endif
}

// ── mean kernel ───────────────────────────────────────────────────────────────
__global__ void gr_mean_kernel(const float* __restrict__ gated, float* __restrict__ mixed,
                                 int n, int hc) {
    const int d = blockIdx.x * gr::kMeanThreads + threadIdx.x;
    if (d >= n) return;
    float s = 0.f;
    for (int c = 0; c < hc; ++c) s += gated[(size_t)c * n + d];
    mixed[d] = s / (float)hc;
}

// ── inject kernel ─────────────────────────────────────────────────────────────
__global__ void gr_inject_kernel(const float* __restrict__ xn, const uint16_t* __restrict__ w_inj,
                                   float* __restrict__ inj, int n_embd, int hc) {
#ifdef __ILUVATAR__
    const int c    = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
#else
    const int c    = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
#endif
    if (c >= hc) return;
    const uint16_t* wr = w_inj + (size_t)c * n_embd;
    float acc = 0.f;
#ifdef __ILUVATAR__
    for (int i = lane; i < n_embd; i += 64) acc += bf16_bits_to_f32(wr[i]) * xn[i];
    acc = ix_warp_reduce_sum(acc);
    if (lane == 0) inj[c] = acc;
#else
    for (int i = lane; i < n_embd; i += 32) acc += bf16_bits_to_f32(wr[i]) * xn[i];
    acc = warp_sum(acc);
    if (lane == 0) inj[c] = acc;
#endif
}

// ── write kernel ──────────────────────────────────────────────────────────────
__global__ void gr_write_kernel(float* __restrict__ out, const float* __restrict__ R,
                                  const float* __restrict__ block_out,
                                  const float* __restrict__ inject,
                                  int n, int hc) {
    const int i = blockIdx.x * THREADS + threadIdx.x;
    const int total = n * hc;
    if (i >= total) return;
    const int d = i % n;
    const int c = i / n;
    out[i] = R[i] + block_out[d] * (2.f * sigmoid_f(inject[c] / (float)hc));
}

// ── workspace init ────────────────────────────────────────────────────────────
struct GrWorkspaceInternal {
    float* xn; float* xq; float* lq; float* gated; float* lo;
    uint64_t total;
};

void gr_workspace_init(void* ws, int hc_dim, int hc_lr, GrWorkspaceInternal* out) {
    const uint64_t sz[5] = {
        (uint64_t)hc_dim * 4, (uint64_t)hc_dim * 2,
        (uint64_t)hc_lr  * 2, (uint64_t)hc_dim * 4,
        (uint64_t)hc_lr  * 4,
    };
    uint8_t* p = static_cast<uint8_t*>(ws);
    uint64_t off = 0;
    float* ptrs[5];
    for (int i = 0; i < 5; ++i) {
        ptrs[i] = reinterpret_cast<float*>(p + off);
        off += (sz[i] + 15) / 16 * 16;
    }
    out->xn = ptrs[0]; out->xq = ptrs[1]; out->lq = ptrs[2];
    out->gated = ptrs[3]; out->lo = ptrs[4]; out->total = off;
}

}  // namespace kern
}  // namespace baloon

// ── public C API ──────────────────────────────────────────────────────────────
using namespace baloon::kern;

void gr_set_fp32_activations(bool on) { fp32_activations.store(on); }
void gr_set_native_mmvf(bool on) { native_mmvf.store(on); }

uint64_t gr_workspace_bytes(int hc_dim, int hc_lr) {
    const uint64_t sz[5] = {
        (uint64_t)hc_dim * 4, (uint64_t)hc_dim * 2,
        (uint64_t)hc_lr  * 2, (uint64_t)hc_dim * 4,
        (uint64_t)hc_lr  * 4,
    };
    uint64_t total = 0;
    for (int i = 0; i < 5; ++i) total += (sz[i] + 15) / 16 * 16;
    return total;
}

int gr_workspace_init(void* ws, int hc_dim, int hc_lr) {
    if (!ws || hc_dim <= 0 || hc_lr <= 0) return -1;
    return 0;
}

int gr_read(const float* R, const float* w_down, const float* w_up, const float* w_inject,
             int n_embd, int hc, int hc_lr, float eps, void* ws, void* stream) {
    if (!R || !w_down || !w_up || !ws) return -1;
    if (n_embd <= 0 || hc <= 0 || hc_lr <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    GrWorkspaceInternal wsi;
    gr_workspace_init(ws, hc * n_embd, hc_lr, &wsi);
#ifdef __ILUVATAR__
    const int shm = (THREADS / 64) * sizeof(float);
    const int inj_threads = 64 * hc;
#else
    const int shm = (THREADS / 32) * sizeof(float);
    const int inj_threads = 32 * hc;
#endif
    gr_norm_kernel<<<hc, THREADS, shm, s>>>(R, wsi.xn, n_embd, eps);
    gr_down_kernel<<<hc_lr, THREADS, shm, s>>>(
        wsi.xn, reinterpret_cast<const uint16_t*>(w_down), wsi.lo, hc * n_embd, hc_lr);
    const int gate_blks = baloon::gr::gate_blocks(hc * n_embd);
    gr_gate_kernel<<<gate_blks, 8 * 64, 0, s>>>(
        wsi.lo, reinterpret_cast<const uint16_t*>(w_up), wsi.gated, hc_lr, hc * n_embd);
    const int mean_blks = baloon::gr::mean_blocks(n_embd);
    gr_mean_kernel<<<mean_blks, baloon::gr::kMeanThreads, 0, s>>>(wsi.gated, wsi.xq, n_embd, hc);
    if (w_inject) {
        gr_inject_kernel<<<1, inj_threads, 0, s>>>(
            wsi.xn, reinterpret_cast<const uint16_t*>(w_inject), wsi.lq, n_embd, hc);
    }
    return (int)cudaGetLastError();
}

int gr_write(float* out, const float* R, const float* block_out, const float* inject,
              int n_embd, int hc, void* stream) {
    if (!out || !R || !block_out || !inject) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int total = n_embd * hc;
    gr_write_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, s>>>(
        out, R, block_out, inject, n_embd, hc);
    return (int)cudaGetLastError();
}
