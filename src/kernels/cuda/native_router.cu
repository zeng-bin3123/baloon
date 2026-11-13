// native_router.cu - pinned ggml topk-moe router: softmax / top-10 / renormalise.
// Fixed shape: 512 experts, top-10; one warp (32 lanes) holds 16 logits per lane.
// Exposes native_router_enabled / native_router_set_enabled / native_router_top10.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "ix_compat.h"

namespace baloon {
namespace kern {

namespace {

std::atomic<bool> g_router_enabled{true};

// § span / overlap guards
static inline bool valid(uint64_t addr, uint64_t bytes) {
    return addr != 0 && addr % 4 == 0 && bytes <= (uint64_t)UINT64_MAX - addr;
}
static inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b + bn && b < a + an;
}

// § warp helpers
__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// § kernel: 32 lanes × 16 slots = 512 experts; block dim (256) → 8 warps; only row y==0 is active.
__global__ void __launch_bounds__(256, 1)
native_router_kernel(const float* __restrict__ logits, int* __restrict__ ids,
                     float* __restrict__ weights, int n_tokens) {
    const int tok  = blockIdx.x * blockDim.y + threadIdx.y;
    if (tok >= n_tokens || threadIdx.y != 0) return;  // only the first warp's row is active
    const int lane = threadIdx.x;                     // 0..31

    // load 16 logit values into registers
    float lv[16];
    const float* lg = logits + (size_t)tok * 512;
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) { lv[i] = lg[lane + i * 32]; mx = fmaxf(mx, lv[i]); }
    mx = warp_max(mx);

    // softmax denominator (sum of exp)
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) { lv[i] = expf(lv[i] - mx); s += lv[i]; }
    s = warp_sum(s);
    const float inv = 1.f / s;
#pragma unroll
    for (int i = 0; i < 16; ++i) lv[i] *= inv;

    // 10 rounds: each round = lane-local scan + XOR butterfly argmax + mask winner
    for (int r = 0; r < 10; ++r) {
        // lane-local best
        float bv = -INFINITY; int bi = 512;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int e = lane + i * 32;
            if (lv[i] > bv || (lv[i] == bv && e < bi)) { bv = lv[i]; bi = e; }
        }
        // XOR butterfly: 5 steps (2^0..2^4)
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        // lane r writes this round's winner
        if (lane == r) { ids[(size_t)tok * 10 + r] = bi; weights[(size_t)tok * 10 + r] = bv; }
        // mask winner's slot with −∞
        if ((bi & 31) == lane) lv[bi >> 5] = -INFINITY;
    }

    // renormalise selected weights (lane 0 only)
    if (lane == 0) {
        float sel = 0.f;
        for (int i = 0; i < 10; ++i) sel += weights[(size_t)tok * 10 + i];
        const float sc = fmaxf(sel, 6.103515625e-5f);  // 2^-14
        for (int i = 0; i < 10; ++i) weights[(size_t)tok * 10 + i] /= sc;
    }
}

}  // namespace

// § public API
bool native_router_enabled() { return g_router_enabled.load(std::memory_order_relaxed); }
void native_router_set_enabled(bool on) { g_router_enabled.store(on, std::memory_order_relaxed); }

cudaError_t native_router_top10(const float* logits, int* ids, float* weights,
                                 int n_tokens, void* stream) {
    if (!stream || n_tokens <= 0) return cudaErrorInvalidValue;
    const uint64_t lg_bytes = (uint64_t)n_tokens * 512u * 4u;
    const uint64_t id_bytes = (uint64_t)n_tokens * 10u * 4u;
    if (!valid((uint64_t)(uintptr_t)logits,  lg_bytes)) return cudaErrorInvalidValue;
    if (!valid((uint64_t)(uintptr_t)ids,     id_bytes)) return cudaErrorInvalidValue;
    if (!valid((uint64_t)(uintptr_t)weights, id_bytes)) return cudaErrorInvalidValue;
    if (overlap((uint64_t)(uintptr_t)logits, lg_bytes, (uint64_t)(uintptr_t)ids,     id_bytes) ||
        overlap((uint64_t)(uintptr_t)logits, lg_bytes, (uint64_t)(uintptr_t)weights, id_bytes) ||
        overlap((uint64_t)(uintptr_t)ids,    id_bytes, (uint64_t)(uintptr_t)weights, id_bytes))
        return cudaErrorInvalidValue;
    // block: (256) = 8 warps × 32 lanes; only warp 0 (y=0) works → dim3(32,8)
    native_router_kernel<<<(unsigned)n_tokens, dim3(32, 8), 0, (cudaStream_t)stream>>>(
        logits, ids, weights, n_tokens);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
