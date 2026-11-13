// fused_gr.cu - single-launch fused gated-residual read at the fixed geometry
// n_embd=2560, hc=4, hc_lr=320. Iluvatar BI-V100: warpSize=64.
// Symbols: D, DOWN_BLOCKS, GrMulti, HC, LR, N, THREADS, TILE, TQ, UPM_BLOCKS, UPM_COLS,
// UP_BLOCKS, UP_COLS, WARPS, __launch_bounds__, dot8, fused_gr_read, fused_gr_read_multi,
// fused_gr_supported, sigmoidf_, warp_sum.
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __ILUVATAR__
#include "ix_compat.h"
#endif

#include "baloon/fused_gr_kernels.hpp"

namespace baloon {
namespace kern {
namespace {

// ── geometry ─────────────────────────────────────────────────────────────────
constexpr int N    = 2560;
constexpr int HC   = 4;
constexpr int D    = N * HC;    // 10240
constexpr int LR   = 320;
constexpr int THREADS = 256;
constexpr int WARPS   = 8;      // 32-lane "sub-warps" per 256-thread block (Ref semantics)
constexpr int TILE    = N;
constexpr int TQ      = 32;     // kFusedGrMaxT

constexpr int DOWN_BLOCKS = LR / WARPS;    // 40
constexpr int UP_BLOCKS   = N / 32;        // 80
constexpr int UP_COLS     = 32;
constexpr int UPM_BLOCKS  = N / 16;        // 160
constexpr int UPM_COLS    = 16;

// ── helpers ───────────────────────────────────────────────────────────────────
__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// Full-warp sum on Iluvatar (64 lanes); half-wave (32 lanes) on NVIDIA.
__device__ __forceinline__ float warp_sum(float v) {
#ifdef __ILUVATAR__
    return ix_warp_reduce_sum(v);
#else
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
#endif
}

// dot8: 8 bf16 weights × 8 fp32 activations.
__device__ __forceinline__ float dot8(const uint16_t* w, const float* x) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t u = (uint32_t)w[i] << 16;
        float    f; __builtin_memcpy(&f, &u, 4);
        acc += f * x[i];
    }
    return acc;
}

// Multi-token argument bundle.
struct GrMulti {
    const float*    R;
    const uint16_t* w_down;
    const uint16_t* w_up;
    const uint16_t* w_inject;
    const float*    inj_prev;
    const float*    bo_prev;
    float*          inject_out;
    float*          R_out;
    bool            apply;
    int             n_tok;
};

// ── fused down kernel ─────────────────────────────────────────────────────────
// Block b (b < DOWN_BLOCKS): 256-thread block, 8 "sub-warps" of 32 lanes each (Ref layout).
// On Iluvatar warp=64, so we use sub-warp lanes (tid & 31) and sub-warp id (tid >> 5).
__global__ __launch_bounds__(THREADS, 2)
void fused_gr_down_kernel(const float* __restrict__ R, const uint16_t* __restrict__ w_down,
                            float* __restrict__ lo, float* __restrict__ inj_out,
                            const uint16_t* __restrict__ w_inj,
                            const float* __restrict__ inj_prev,
                            const float* __restrict__ bo_prev,
                            bool apply) {
    const int bk     = blockIdx.x;
    const int tid    = threadIdx.x;
    const int sub    = tid >> 5;    // sub-warp index (0..7)
    const int lane   = tid & 31;   // sub-warp lane (0..31)

    if (bk < DOWN_BLOCKS) {
        const int row = bk * WARPS + sub;
        const uint16_t* wr = w_down + (size_t)row * (HC * N);
        float acc = 0.f;
        for (int d = lane; d < HC * N; d += 32) {
            uint32_t u = (uint32_t)wr[d] << 16;
            float    f; __builtin_memcpy(&f, &u, 4);
            float    x = R[d];
            if (apply && inj_prev && bo_prev) {
                x += bo_prev[d % N] * (2.f * sigmoidf_(inj_prev[d / N] / (float)HC));
            }
            acc += f * x;
        }
        // Half-wave (32-lane) reduction using width=32.
#ifdef __ILUVATAR__
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
#else
        acc = warp_sum(acc);
#endif
        if (lane == 0) lo[row] = acc / (1.f + expf(-acc));  // SiLU
    } else {
        if (sub >= HC || !w_inj || !inj_out) return;
        const int c = sub;
        const uint16_t* wr = w_inj + (size_t)c * (HC * N);
        float acc = 0.f;
        for (int d = lane; d < HC * N; d += 32) {
            uint32_t u = (uint32_t)wr[d] << 16;
            float    f; __builtin_memcpy(&f, &u, 4);
            acc += f * R[d];
        }
#ifdef __ILUVATAR__
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
#else
        acc = warp_sum(acc);
#endif
        if (lane == 0) inj_out[c] = acc;
    }
}

// ── fused up kernel ───────────────────────────────────────────────────────────
__global__ __launch_bounds__(THREADS, 2)
void fused_gr_up_kernel(const float* __restrict__ lo, const uint16_t* __restrict__ w_up,
                          const float* __restrict__ R, float* __restrict__ R_out,
                          bool apply, const float* __restrict__ inj_prev,
                          const float* __restrict__ bo_prev) {
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    const int c   = tid / UP_COLS;
    const int dd  = tid % UP_COLS;
    if (c >= HC) return;
    const int d  = b * UP_COLS + dd;
    const int ri = c * N + d;

    const uint16_t* wr = w_up + (size_t)(c * N + d) * LR;
    float gate = 0.f;
#pragma unroll 8
    for (int k = 0; k < LR; ++k) {
        uint32_t u = (uint32_t)wr[k] << 16;
        float    f; __builtin_memcpy(&f, &u, 4);
        gate += f * lo[k];
    }
    gate = sigmoidf_(gate);

    float R_base = R[ri];
    if (apply && inj_prev && bo_prev)
        R_base += bo_prev[d] * (2.f * sigmoidf_(inj_prev[c] / (float)HC));
    R_out[ri] = R_base * gate / (float)HC;
}

}  // anonymous
}  // kern

// ── public API ───────────────────────────────────────────────────────────────
namespace ref_fused_gr {

bool fused_gr_supported(int n_embd, int hc, int hc_lr) {
    return n_embd == kern::N && hc == kern::HC && hc_lr == kern::LR;
}

int fused_gr_read(const void* R, const void* w_down, const void* w_up, const void* w_inject,
                   const void* inj_prev, const void* bo_prev,
                   void* inject_out, void* R_out,
                   bool apply, cudaStream_t stream) {
    if (!R || !w_down || !w_up || !R_out) return -1;
    using namespace kern;
    float* lo = nullptr;
    if (cudaMalloc(&lo, LR * sizeof(float)) != cudaSuccess) return -2;
    kern::fused_gr_down_kernel<<<DOWN_BLOCKS + 1, THREADS, 0, stream>>>(
        static_cast<const float*>(R), static_cast<const uint16_t*>(w_down),
        lo, static_cast<float*>(inject_out),
        static_cast<const uint16_t*>(w_inject),
        static_cast<const float*>(inj_prev), static_cast<const float*>(bo_prev), apply);
    kern::fused_gr_up_kernel<<<UP_BLOCKS, THREADS, 0, stream>>>(
        lo, static_cast<const uint16_t*>(w_up),
        static_cast<const float*>(R), static_cast<float*>(R_out),
        apply, static_cast<const float*>(inj_prev), static_cast<const float*>(bo_prev));
    cudaFree(lo);
    return (int)cudaGetLastError();
}

int fused_gr_read_multi(const void* R, const void* w_down, const void* w_up, const void* w_inject,
                          const void* inj_prev, const void* bo_prev,
                          void* inject_out, void* R_out,
                          bool apply, int n_tok, cudaStream_t stream) {
    if (n_tok < 1 || n_tok > kern::TQ) return -1;
    for (int t = 0; t < n_tok; ++t) {
        const float* Rt = static_cast<const float*>(R) + (size_t)t * kern::D;
        float*       ot = static_cast<float*>(R_out)   + (size_t)t * kern::D;
        int rc = fused_gr_read(Rt, w_down, w_up, w_inject, inj_prev, bo_prev,
                                inject_out, ot, apply, stream);
        if (rc != 0) return rc;
    }
    return 0;
}

}  // ref_fused_gr
}  // baloon
