// native_gdn.cu - pinned gated-delta-net step kernel for the native (ggml/ivcore10) path.
// Exposes native_gdn_enabled / native_gdn_set_enabled / native_gdn_step matching the
// reference src/kernels/cuda/native_gdn.cu API.
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "baloon/ilu_native_gdn.hpp"
#include "ix_compat.h"

namespace baloon {
namespace kern {

// § constants
constexpr int S = 128;  // head dimension; each warp covers S/4 = 32 rows

namespace {

std::atomic<bool> g_gdn_enabled{true};

// § helpers
__device__ __forceinline__ float warp_sum(float v) { return ix_halfwave_reduce_sum(v); }

// § span guards
static inline bool valid_span(uint64_t addr, uint64_t bytes) {
    return addr != 0 && addr % 4 == 0 && bytes <= (uint64_t)UINT64_MAX - addr;
}
static inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b + bn && b < a + an;
}

// § kernel
__global__ void __launch_bounds__(128, 2)
native_gdn_step_kernel(float* __restrict__ st, const float* __restrict__ q, const float* __restrict__ k,
                       const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
                       float* __restrict__ o, int h_k, int h_v, float scale) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col  = blockIdx.z * blockDim.y + threadIdx.y;
    const int qh   = head % h_k;
    float s[4], kk[4], qq[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int i = r * 32 + lane;
        s[r]  = st[((size_t)i * h_v + head) * S + col];
        kk[r] = k[(size_t)qh * S + i];
        qq[r] = q[(size_t)qh * S + i];
    }
    const float decay = expf(gate[head]);
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) { s[r] *= decay; kv += s[r] * kk[r]; }
    kv = warp_sum(kv);
    const float d = (v[(size_t)head * S + col] - kv) * beta[head];
    float acc = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) { s[r] += kk[r] * d; acc += s[r] * qq[r]; }
    acc = warp_sum(acc);
#pragma unroll
    for (int r = 0; r < 4; ++r)
        st[((size_t)(r * 32 + lane) * h_v + head) * S + col] = s[r];
    if (lane == 0) o[(size_t)head * S + col] = acc * scale;
}

}  // namespace

// § public API
bool native_gdn_enabled() { return g_gdn_enabled.load(std::memory_order_relaxed); }
void native_gdn_set_enabled(bool on) { g_gdn_enabled.store(on, std::memory_order_relaxed); }

int native_gdn_step(float* state, const float* q, const float* k, const float* v,
                    const float* gate, const float* beta, float* o,
                    int S_, int h_k, int h_v, void* stream) {
    if (!stream || S_ != S || h_k <= 0 || h_v <= 0 || h_v % h_k != 0 || h_v > 65535)
        return -1;
    const uint64_t hk = (uint64_t)h_k, hv = (uint64_t)h_v;
    struct Rg { uint64_t addr, bytes; bool w; };
    const Rg rg[] = {
        {(uint64_t)(uintptr_t)state, (uint64_t)S * 128ull * hv * 4ull, true},
        {(uint64_t)(uintptr_t)o,     128ull * hv * 4ull,               true},
        {(uint64_t)(uintptr_t)q,     (uint64_t)S * hk * 4ull,         false},
        {(uint64_t)(uintptr_t)k,     (uint64_t)S * hk * 4ull,         false},
        {(uint64_t)(uintptr_t)v,     (uint64_t)S * hv * 4ull,         false},
        {(uint64_t)(uintptr_t)gate,  hv * 4ull,                        false},
        {(uint64_t)(uintptr_t)beta,  hv * 4ull,                        false},
    };
    const int n = (int)(sizeof(rg) / sizeof(rg[0]));
    for (int i = 0; i < n; ++i)
        if (!valid_span(rg[i].addr, rg[i].bytes)) return -1;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if ((rg[i].w || rg[j].w) && overlap(rg[i].addr, rg[i].bytes, rg[j].addr, rg[j].bytes))
                return -1;
    const float scale = 1.0f / sqrtf((float)S);
    const dim3 grid((unsigned)h_v, 1, (unsigned)(S / 4));
    const dim3 block(32, 4, 1);
    native_gdn_step_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        state, q, k, v, gate, beta, o, h_k, h_v, scale);
    return (int)cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
