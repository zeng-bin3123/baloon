// native_flash_attn.cu - Pinned llama.cpp fattn-vec specialisation for BI-V100:
//   D = 256, Q 24 heads / KV 2 heads (GQA factor 12), one column per thread block.
//   Accepts only the dense short-context step: n_kv == width, pos == width-1,
//   n_bid == width/4; refuses anything else with kNativeFlashAttnUnsupportedStep.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace baloon {
namespace native_flash {
namespace {

struct Span { uint64_t addr; uint64_t bytes; };

static bool validate_spans(const Span* spans, int n) {
    for (int i = 0; i < n; ++i) {
        if (spans[i].addr == 0 || spans[i].bytes == 0) return false;
        for (int j = 0; j < i; ++j) {
            if (spans[i].addr < spans[j].addr + spans[j].bytes &&
                spans[j].addr < spans[i].addr + spans[i].bytes) return false;
        }
    }
    return true;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// Dense short-context flash attention: one block per query head.
// q: [n_head][HD], k/v: [n_kv*2][HD] (interleaved k at kv*0, v at kv*1),
// out: [n_head][HD]. HD=256.
__global__ void __launch_bounds__(256) native_flash_attn_kernel(
        const float* __restrict__ q, const __half* __restrict__ kv,
        float* __restrict__ out, int n_kv, int n_head, float scale) {
    const int h = blockIdx.x;         // query head
    const int kvh = h / 12;           // kv head (GQA: 12 query per kv)
    const int d = threadIdx.x;        // dimension (0..255)
    if (h >= n_head || d >= 256) return;
    const float* qh = q + h * 256 + d;
    const float qd = qh[0];
    // Softmax-free flash attention over n_kv cells
    float m = -FLT_MAX, l = 0.f;
    __shared__ float acc[256];
    acc[d] = 0.f;
    __syncthreads();
    for (int c = 0; c < n_kv; ++c) {
        // Key at row (c*2 + 0)*256 + d (kv head kvh)
        const int64_t kr = ((int64_t)(c * 2 + 0) + (int64_t)kvh * 2 * n_kv) * 256;
        const int64_t vr = ((int64_t)(c * 2 + 1) + (int64_t)kvh * 2 * n_kv) * 256;
        const float kd = __half2float(kv[kr + d]);
        float dot = qd * kd * scale;
        dot = warp_sum(dot);
        const float mn = fmaxf(m, dot);
        const float alpha = expf(m - mn);
        m = mn;
        l = l * alpha + expf(dot - m);
        const float vd = __half2float(kv[vr + d]);
        acc[d] = acc[d] * alpha + expf(dot - m) * vd;
    }
    out[h * 256 + d] = acc[d] / l;
}

} // anonymous namespace

// Host-callable step: validates the step fields and launches the kernel.
int native_flash_attn_short_step(
        const float* q, const __half* kv, float* out,
        int pos, int n_kv, int n_bid, int width, int max_ctx,
        int n_head, float scale, void* stream) {
    // Step validation
    if (width < 1 || width > max_ctx) return -1;
    if (n_kv != width || pos != width - 1 || n_bid != width / 4) return -1;
    if (!q || !kv || !out) return -1;
    // Span disjointness check
    Span spans[3] = {
        {(uint64_t)(uintptr_t)q,   (uint64_t)n_head * 256 * 4},
        {(uint64_t)(uintptr_t)kv,  (uint64_t)n_kv * 2 * 256 * 2},
        {(uint64_t)(uintptr_t)out, (uint64_t)n_head * 256 * 4}
    };
    if (!validate_spans(spans, 3)) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    native_flash_attn_kernel<<<n_head, 256, 0, s>>>(q, kv, out, n_kv, n_head, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace native_flash
} // namespace baloon
