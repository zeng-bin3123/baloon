// native_flash_attn.cu - short-context (<= 256 cells) flash-attention decode kernel, Q 24x256 / KV 2x256 fp16,
// one 128-thread block per query head, online softmax over two passes of 128 cells (padded length 256).
// Valid only for a consistent step (1 <= width <= max_context, n_kv == width, pos == width - 1, n_bid == width / 4);
// otherwise every output element is NaN and *status = 1.
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "baloon/qsa_native_kernels.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int HD = 256;
constexpr int NT = 128;
constexpr int PAD = 256;
constexpr int NHEAD = 24;
constexpr int NKV = 2;
constexpr int GROUP = 12;

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += shfl32(v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, shfl32(v, o));
    return v;
}

__global__ void __launch_bounds__(NT) flash_short_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kp,
                                                         const uint16_t* __restrict__ vp, const Step* __restrict__ step,
                                                         int max_context, float* __restrict__ out, int32_t* __restrict__ status) {
    __shared__ float qs[HD];
    __shared__ float p[PAD];
    __shared__ float red[8];
    const int head = blockIdx.x, t = threadIdx.x;
    const int width = step->width, nkv = step->n_kv, pos = step->pos, nbid = step->n_bid;
    const bool valid = width >= 1 && width <= max_context && width <= PAD && nkv == width && pos + 1 == width && nbid == width / 4;
    if (!valid) {
        for (int d = t; d < HD; d += NT) out[head * HD + d] = __int_as_float(0x7FC00000);
        if (t == 0) *status = 1;
        return;
    }
    const int kvh = head / GROUP;
    for (int d = t; d < HD; d += NT) qs[d] = q[head * HD + d] * 0.0625f;  // 1 / sqrt(256)
    __syncthreads();
    float m = -FLT_MAX, l = 0.f;
    float acc0 = 0.f, acc1 = 0.f;  // output dims t and t + 128
    for (int pass = 0; pass < PAD / NT; ++pass) {
        const int cell = pass * NT + t;
        float s = -FLT_MAX;  // masked cells carry -inf-like scores
        if (cell < width) {
            const uint16_t* kr = kp + ((size_t)cell * NKV + kvh) * HD;
            float a = 0.f;
            for (int d = 0; d < HD; ++d) a += qs[d] * f16bits::f32_from_f16(kr[d]);
            s = a;
        }
        // block max of this pass
        float pm = warp_max(s);
        if ((t & 31) == 0) red[t >> 5] = pm;
        __syncthreads();
        pm = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        __syncthreads();
        const float next = fmaxf(m, pm);
        const float rescale = expf(m - next);  // m - next <= 0
        const float e = cell < width ? expf(s - next) : 0.f;
        p[pass * NT + t] = e;
        float ps = warp_sum(e);
        if ((t & 31) == 0) red[4 + (t >> 5)] = ps;
        __syncthreads();
        const float psum = red[4] + red[5] + red[6] + red[7];
        l = l * rescale + psum;
        acc0 *= rescale; acc1 *= rescale;
        const int base = pass * NT;
        const int hi = min(NT, width - base);
        for (int c = 0; c < hi; ++c) {
            const uint16_t* vr = vp + ((size_t)(base + c) * NKV + kvh) * HD;
            const float w = p[base + c];
            acc0 += w * f16bits::f32_from_f16(vr[t]);
            acc1 += w * f16bits::f32_from_f16(vr[t + NT]);
        }
        m = next;
        __syncthreads();
    }
    out[head * HD + t] = acc0 / l;
    out[head * HD + t + NT] = acc1 / l;
}

}  // namespace

bool validate_spans(const Span* spans, int n, bool* writes) { return spans_ok_model(spans, n, writes); }

int native_flash_attn_short_step(const float* q, const uint16_t* kpool, const uint16_t* vpool, const Step* step, int max_context,
                                 float* out, int32_t* status, void* stream) {
    if (!q || !kpool || !vpool || !step || !out || !status || max_context < 1 || max_context > PAD) return kInvalid;
    const Span sp[] = {{(uint64_t)(uintptr_t)q, (uint64_t)NHEAD * HD * 4},
                       {(uint64_t)(uintptr_t)kpool, (uint64_t)max_context * NKV * HD * 2},
                       {(uint64_t)(uintptr_t)vpool, (uint64_t)max_context * NKV * HD * 2},
                       {(uint64_t)(uintptr_t)out, (uint64_t)NHEAD * HD * 4},
                       {(uint64_t)(uintptr_t)status, 4}};
    bool wr[] = {false, false, false, true, true};
    if (!validate_spans(sp, 5, wr)) return kInvalid;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    flash_short_kernel<<<NHEAD, NT, 0, st>>>(q, kpool, vpool, step, max_context, out, status);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace qsa_dev
}  // namespace baloon
