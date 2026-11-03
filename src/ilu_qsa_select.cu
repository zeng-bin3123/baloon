// ilu_qsa_select.cu - block scoring and radix-select top-k over pooled indexer blocks (R = 4) for BI-V100.
//  block_score : one 32-lane warp per pooled block, 8 warps per 256-thread block; each lane loads a float4 of the key.
//  block_select: one 256-thread block. The threshold key is found by 4 radix passes of 8 bits over weighted
//    histograms (block weight R, tail weight n_kv - 4*n_bid); cells above the threshold are taken, plus at most
//    width - above cells equal to it, lowest index first, each thread clamped to its share of the budget.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/ilu_qsa_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace ilu_qsa {
namespace {

constexpr int NT = 256;
static_assert(NT == (int)qsa_select::kTopKThreads && (int)qsa_select::kR == 4, "geometry");

__device__ __forceinline__ uint32_t order_key_d(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(NT) block_score_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                         const float* __restrict__ hw, const float* __restrict__ bias,
                                                         float* __restrict__ scores, int n_kv) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int b = blockIdx.x * 8 + warp;
    if (b >= nb) return;
    const float4 key = reinterpret_cast<const float4*>(pooled + (size_t)b * 128)[lane];
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 qv = reinterpret_cast<const float4*>(iq + h * 128)[lane];
        const float dot = lane_sum(key.x * qv.x + key.y * qv.y + key.z * qv.z + key.w * qv.w);
        s += hw[h] * fmaxf(dot, 0.f);
    }
    if (lane == 0) {
        if (bias) s += bias[b];
        if (b == n_bid && tail > 0) s += kTailBias;
        scores[b] = s;
    }
}

__global__ void __launch_bounds__(NT) block_select_kernel(const float* __restrict__ scores, int n_kv, int width,
                                                          int32_t* __restrict__ out_ids, int32_t* __restrict__ n_out) {
    __shared__ uint32_t hist[256];
    __shared__ int32_t gtc[NT], eqc[NT], offs[NT], share[NT];
    __shared__ uint32_t s_prefix, s_above;
    const int t = threadIdx.x;

    if (n_kv <= width) {  // identity branch: everything, ascending
        for (int i = t; i < width; i += NT) out_ids[i] = i < n_kv ? i : -1;
        if (t == 0) *n_out = n_kv;
        return;
    }
    const int n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int per = (nb + NT - 1) / NT;
    const int b_lo = min(nb, t * per), b_hi = min(nb, (t + 1) * per);
    if (t == 0) { s_prefix = 0; s_above = 0; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        for (int i = t; i < 256; i += NT) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        for (int b = b_lo; b < b_hi; ++b) {
            const uint32_t k = order_key_d(scores[b]);
            const bool match = pass == 0 || (k >> (32 - 8 * pass)) == prefix;
            if (match) atomicAdd(&hist[(k >> (24 - 8 * pass)) & 0xFFu], (uint32_t)(b < n_bid ? 4 : tail));
        }
        __syncthreads();
        if (t == 0) {
            uint32_t cum = s_above;
            int chosen = 0;
            for (int d = 255; d >= 0; --d) {
                if (cum + hist[d] >= (uint32_t)width) { chosen = d; break; }
                cum += hist[d];
            }
            s_above = cum;
            s_prefix = pass == 0 ? (uint32_t)chosen : ((prefix << 8) | (uint32_t)chosen);
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;
    int gt = 0, eq = 0;
    for (int b = b_lo; b < b_hi; ++b) {
        const uint32_t k = order_key_d(scores[b]);
        const int w = b < n_bid ? 4 : tail;
        if (k > thr) gt += w; else if (k == thr) eq += w;
    }
    gtc[t] = gt; eqc[t] = eq;
    __syncthreads();
    if (t == 0) {
        const int budget = width - (int)s_above;  // 0 < budget <= total ties
        int before_eq = 0, running = 0;
        for (int i = 0; i < NT; ++i) {
            const int sh = min(eqc[i], max(0, budget - before_eq));  // clampEq
            share[i] = sh;
            offs[i] = running;
            running += gtc[i] + sh;
            before_eq += eqc[i];
        }
        *n_out = running;
    }
    __syncthreads();
    int left = share[t], pos = offs[t];
    for (int b = b_lo; b < b_hi; ++b) {
        const uint32_t k = order_key_d(scores[b]);
        const int w = b < n_bid ? 4 : tail;
        for (int c = 0; c < w; ++c) {
            if (k > thr) out_ids[pos++] = b * 4 + c;
            else if (k == thr && left > 0) { out_ids[pos++] = b * 4 + c; --left; }
        }
    }
}

}  // namespace

int block_score(const float* pooled, const float* iq, const float* hw, const float* bias, float* scores, int n_kv,
                int rows, void* stream) {
    if (!pooled || !iq || !hw || !scores || n_kv < 1) return kInvalid;
    const int nb = n_kv / 4 + (n_kv % 4 != 0 ? 1 : 0);
    if (rows < nb) return kInvalid;
    if ((((uintptr_t)pooled | (uintptr_t)iq) & 15u) != 0) return kInvalid;  // float4 loads
    const Region r[] = {region(pooled, (uint64_t)rows * 512, false), region(iq, 4 * 512, false), region(hw, 16, false),
                        region(scores, (uint64_t)nb * 4, true), region(bias ? (const void*)bias : (const void*)hw, bias ? (uint64_t)nb * 4 : 16, false)};
    if (!regions_ok(r, sizeof(r) / sizeof(r[0]))) return kInvalid;
    block_score_kernel<<<qsa_select::score_grid((uint32_t)nb), NT, 0, static_cast<cudaStream_t>(stream)>>>(pooled, iq, hw, bias, scores, n_kv);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

int block_select(const float* scores, int n_kv, int width, int32_t* out_ids, int32_t* n_out, void* stream) {
    if (!scores || !out_ids || !n_out || n_kv < 1 || width < 1) return kInvalid;
    const int nb = n_kv / 4 + (n_kv % 4 != 0 ? 1 : 0);
    const Region r[] = {region(scores, (uint64_t)nb * 4, false), region(out_ids, (uint64_t)width * 4, true), region(n_out, 4, true)};
    if (!regions_ok(r, sizeof(r) / sizeof(r[0]))) return kInvalid;
    block_select_kernel<<<1, NT, 0, static_cast<cudaStream_t>(stream)>>>(scores, n_kv, width, out_ids, n_out);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace ilu_qsa
}  // namespace baloon
