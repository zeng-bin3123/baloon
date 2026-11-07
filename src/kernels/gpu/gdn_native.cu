// gdn_native.cu - native GDN recurrent step: one 64-lane wave per value head, two state columns per lane (S = 128). The per-column
// arithmetic (decay, contraction with k, rank-1 update, read-out with the UPDATED state, ascending row order) is the same
// sequence of float operations as gdn_step, so the two agree bit for bit; only the work packing differs.
// Every pointer span is validated and the state must not overlap any other operand.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int S = 128;
bool g_enabled = true;

struct Rng { const char* b; const char* e; };
// span validity: non-null, float-aligned, and the end address cannot wrap the address space
bool valid_span(const void* p, size_t bytes) {
    const uint64_t a = (uint64_t)(uintptr_t)p;
    return p != nullptr && bytes > 0 && a % 4 == 0 && (uint64_t)bytes <= UINT64_MAX - a;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }

__global__ void __launch_bounds__(64) native_gdn_step_kernel(float* __restrict__ state, const float* __restrict__ q, const float* __restrict__ k,
                                                             const float* __restrict__ v, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float* __restrict__ o, int h_k, int h_v) {
    const int h = blockIdx.x, lane = threadIdx.x, src = h % h_k;
    const float dec = expf(gate[h]), b = beta[h];
    const float* kk = k + (size_t)src * S;
    const float* qq = q + (size_t)src * S;
    const size_t stride = (size_t)h_v * S;
    float* base = state + (size_t)h * S;
    float sk0 = 0.f, sk1 = 0.f;
    for (int i = 0; i < S; ++i) {
        float* p = base + (size_t)i * stride + lane;
        const float s0 = p[0] * dec, s1 = p[64] * dec;
        p[0] = s0; p[64] = s1;
        sk0 += s0 * kk[i]; sk1 += s1 * kk[i];
    }
    const float d0 = (v[(size_t)h * S + lane] - sk0) * b, d1 = (v[(size_t)h * S + lane + 64] - sk1) * b;
    float o0 = 0.f, o1 = 0.f;
    for (int i = 0; i < S; ++i) {
        float* p = base + (size_t)i * stride + lane;
        const float s0 = p[0] + kk[i] * d0, s1 = p[64] + kk[i] * d1;
        p[0] = s0; p[64] = s1;
        o0 += s0 * qq[i]; o1 += s1 * qq[i];
    }
    o[(size_t)h * S + lane] = o0;
    o[(size_t)h * S + lane + 64] = o1;
}
}  // namespace

bool native_gdn_enabled() { return g_enabled; }
void native_gdn_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* o,
                            int S_, int h_k, int h_v, cudaStream_t s) {
    if (S_ != S || h_k <= 0 || h_v <= 0) return cudaErrorInvalidValue;
    if (h_v % h_k != 0 || h_v > 65535) return cudaErrorInvalidValue;     // modulo head pairing needs h_k | h_v; grid.x limit
    const size_t st = (size_t)S * h_v * S * 4, hq = (size_t)h_k * S * 4, hv = (size_t)h_v * S * 4, hs = (size_t)h_v * 4;
    if (!valid_span(state, st) || !valid_span(q, hq) || !valid_span(k, hq) || !valid_span(v, hv) || !valid_span(gate, hs) ||
        !valid_span(beta, hs) || !valid_span(o, hv)) return cudaErrorInvalidValue;
    const void* ps[6] = {q, k, v, gate, beta, o};
    const size_t ns[6] = {hq, hq, hv, hs, hs, hv};
    for (int i = 0; i < 6; ++i) if (overlap(state, st, ps[i], ns[i])) return cudaErrorInvalidValue;
    for (int i = 0; i < 5; ++i) if (overlap(o, hv, ps[i], ns[i])) return cudaErrorInvalidValue;   // output disjoint from q, k, v, gate, beta
    native_gdn_step_kernel<<<h_v, 64, 0, s>>>(state, q, k, v, gate, beta, o, h_k, h_v);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
