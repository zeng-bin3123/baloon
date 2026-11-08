// ple_postops_native.cu - native PLE post-projection operators at N = 2560, H = 4 streams, D = N*H = 10240.
//   gate_kernel          : one block of 512 threads per stream; 8 partial lanes d = tid + j*512 (masked to d < N) of
//                          sum_d normed[s*N+d] * gate_w[s*N+d]; 32-lane warp sums, then the 16 warp partials; the stream gate is
//                          sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))).
//   broadcast_kernel     : gated[i] = value[i % N] * gate[i / N].
//   conv_residual_kernel : per channel c, taps read history rows 0, 3, 6 and the fresh normalised value for tap 3 (history is
//                          row-fastest, c*9 + row; weights are fp16, c*4 + k); resid = hidden + (gated + silu(conv)).
// native_ple_postops validates every span: non-null, aligned, no address wrap, outputs disjoint from every input and from each
// other, with exactly two aliasing exceptions (resid == hidden, resid == gated; never both at once).
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/qsa_native_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace ple_native {
namespace {

constexpr int N = kN;
constexpr int H = kH;
constexpr int D = kD;
static_assert(N * H == D && D % 256 == 0, "geometry");
static_assert(kGateThreads / 32 == 16 && kGateSlots * kGateThreads >= N, "gate tiling");

struct Span { uint64_t addr; uint64_t bytes; };

bool validate(const Span& s, uint64_t align) {
    if (s.addr == 0 || s.bytes == 0) return false;
    if (s.addr % align != 0) return false;
    return s.addr + s.bytes > s.addr;  // no wrap
}
// Half-open byte-range intersection.
bool overlaps(const Span& a, const Span& b) { return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes; }
bool same(const Span& a, const Span& b) { return a.addr == b.addr && a.bytes == b.bytes; }
Span span(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }

cudaError_t launch_check() { return cudaGetLastError(); }

__device__ __forceinline__ float warp_sum(float v) { return ilu_qsa::lane_sum(v); }

__device__ __forceinline__ float half_bits_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {
            e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = s | (e << 23) | (m << 13);
        }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    return __uint_as_float(bits);
}

__global__ void gate_kernel(const float* __restrict__ normed, const float* __restrict__ gate_w, float* __restrict__ gate) {
    __shared__ float partials[kGateThreads / 32];
    const int s = blockIdx.x, tid = threadIdx.x;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < kGateSlots; ++j) {
        const int d = tid + j * kGateThreads;
        if (d < N) acc += normed[(size_t)s * N + d] * gate_w[(size_t)s * N + d];
    }
    acc = warp_sum(acc);
    if ((tid & 31) == 0) partials[tid >> 5] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.f;
        for (int w = 0; w < kGateThreads / 32; ++w) tot += partials[w];
        const float m = fabsf(tot) > kGateFloor ? fabsf(tot) : kGateFloor;
        const float sg = tot > 0.f ? 1.f : (tot < 0.f ? -1.f : 0.f);
        gate[s] = 1.0f / (1.0f + expf(-(sg * sqrtf(m))));
    }
}

__global__ void broadcast_kernel(const float* __restrict__ value, const float* __restrict__ gate, float* __restrict__ gated) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D) gated[i] = value[i % N] * gate[i / N];
}

// hidden / gated / resid may be the same buffers as allowed by the span check: each element is read before it is written.
__global__ void conv_residual_kernel(const float* __restrict__ normed, const uint16_t* __restrict__ conv_w,
                                     const float* __restrict__ hist, const float* hidden, const float* gated, float* resid) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D) return;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kTaps; ++k) {
        const float tap = k < kTaps - 1 ? hist[(size_t)c * kHist + kDilation * k] : normed[c];
        acc += half_bits_to_float(conv_w[(size_t)c * kTaps + k]) * tap;
    }
    const float act = acc / (1.0f + expf(-acc));
    const float h = hidden[c], g = gated[c];
    resid[c] = h + (g + act);
}

}  // namespace

cudaError_t native_ple_postops(const Args& a, cudaStream_t s) {
    if (!a.normed || !a.gate_w || !a.value || !a.conv_w || !a.hist || !a.hidden || !a.gate || !a.gated || !a.resid)
        return cudaErrorInvalidValue;
    const Span in[6] = {span(a.normed, (uint64_t)D * 4), span(a.gate_w, (uint64_t)D * 4), span(a.value, (uint64_t)N * 4),
                        span(a.conv_w, (uint64_t)D * kTaps * 2), span(a.hist, (uint64_t)D * kHist * 4), span(a.hidden, (uint64_t)D * 4)};
    const uint64_t in_align[6] = {4, 4, 4, 2, 4, 4};
    const Span out[3] = {span(a.gate, (uint64_t)H * 4), span(a.gated, (uint64_t)D * 4), span(a.resid, (uint64_t)D * 4)};
    for (int i = 0; i < 6; ++i) if (!validate(in[i], in_align[i])) return cudaErrorInvalidValue;
    for (int i = 0; i < 3; ++i) if (!validate(out[i], 4)) return cudaErrorInvalidValue;

    const bool resid_is_hidden = same(out[2], in[5]);
    const bool resid_is_gated = same(out[2], out[1]);
    // resid == hidden == gated at once is rejected by the output/input rule below: gated (an output) overlaps hidden (an input)
    // outputs against inputs
    for (int o = 0; o < 3; ++o)
        for (int i = 0; i < 6; ++i)
            if (overlaps(out[o], in[i]) && !(o == 2 && i == 5 && resid_is_hidden)) return cudaErrorInvalidValue;
    // outputs against each other
    for (int o = 0; o < 3; ++o)
        for (int p = o + 1; p < 3; ++p)
            if (overlaps(out[o], out[p]) && !(o == 1 && p == 2 && resid_is_gated)) return cudaErrorInvalidValue;

    gate_kernel<<<H, kGateThreads, 0, s>>>(a.normed, a.gate_w, a.gate);
    broadcast_kernel<<<D / 256, 256, 0, s>>>(a.value, a.gate, a.gated);
    conv_residual_kernel<<<D / 256, 256, 0, s>>>(a.normed, a.conv_w, a.hist, a.hidden, a.gated, a.resid);
    return launch_check();
}

}  // namespace ple_native
}  // namespace baloon
