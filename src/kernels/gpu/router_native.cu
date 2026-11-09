// router_native.cu - one-wave-per-token MoE router (n_expert <= 512): every lane keeps up to 8 probabilities in registers, each of the k
// selection passes is a 64-lane butterfly argmax (lowest index wins ties). Softmax and renormalisation are done in double exactly like
// router_top10, so ids and weights are bit-identical to it.
#include <cstdint>
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int RW_PER = 8;
bool g_enabled = true;
// span_valid: non-null, 4-byte aligned, and the span cannot wrap past the top of the address space.
bool valid(const void* p, size_t bytes) {
    const uintptr_t a = (uintptr_t)p;
    return p != nullptr && bytes > 0 && (a & 3u) == 0 && bytes <= (uintptr_t)UINTPTR_MAX - a;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}
__device__ __forceinline__ float warp_max(float v) { return kgc::wave_max(v); }
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }

__global__ void __launch_bounds__(64) native_router_kernel(const float* __restrict__ logits, int n_expert, int k, int32_t* __restrict__ ids,
                                                           float* __restrict__ weights) {
    __shared__ float s_ex[64 * RW_PER];
    __shared__ float s_sum;
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* l = logits + (size_t)t * n_expert;
    float lv[RW_PER];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) { const int e = lane + 64 * j; lv[j] = e < n_expert ? l[e] : -INFINITY; mx = fmaxf(mx, lv[j]); }
    mx = warp_max(mx);
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) { const int e = lane + 64 * j; if (e < n_expert) s_ex[e] = expf(lv[j] - mx); }
    __syncthreads();
    if (lane == 0) { float sum = 0.f; for (int e = 0; e < n_expert; ++e) sum += s_ex[e]; s_sum = sum; }
    __syncthreads();
    const float inv = 1.0f / s_sum;
    float p[RW_PER];
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) { const int e = lane + 64 * j; p[j] = e < n_expert ? s_ex[e] * inv : -INFINITY; }
    unsigned taken = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY; int bi = n_expert;
#pragma unroll
        for (int j = 0; j < RW_PER; ++j) {
            const int e = lane + 64 * j;
            if (e >= n_expert || ((taken >> j) & 1u)) continue;
            if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
        }
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 64); const int oi = __shfl_xor(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi >= n_expert) break;
        if ((bi & 63) == lane) taken |= 1u << (bi >> 6);
        if (lane == 0) { ids[(size_t)t * k + i] = bi; weights[(size_t)t * k + i] = bv; }
    }
    if (lane == 0) {
        float s = 0.f;
        for (int i = 0; i < k; ++i) s += weights[(size_t)t * k + i];
        const float sc = s > 6.103515625e-05f ? s : 6.103515625e-05f;
        for (int i = 0; i < k; ++i) weights[(size_t)t * k + i] = weights[(size_t)t * k + i] / sc;
    }
}
}  // namespace

bool native_router_enabled() { return g_enabled; }
void native_router_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_router_top10(const float* logits, int n_tokens, int n_expert, int k, int32_t* ids, float* weights, cudaStream_t s) {
    if (n_tokens <= 0 || n_expert <= 0 || n_expert > 64 * RW_PER || k <= 0 || k > n_expert) return cudaErrorInvalidValue;
    const size_t lb = (size_t)n_tokens * n_expert * 4, ob = (size_t)n_tokens * k * 4;
    if (!valid(logits, lb) || !valid(ids, ob) || !valid(weights, ob)) return cudaErrorInvalidValue;
    if (overlap(ids, ob, weights, ob) || overlap(ids, ob, logits, lb) || overlap(weights, ob, logits, lb)) return cudaErrorInvalidValue;
    native_router_kernel<<<n_tokens, 64, 0, s>>>(logits, n_expert, k, ids, weights);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
