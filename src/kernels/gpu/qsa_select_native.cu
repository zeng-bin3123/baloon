// qsa_select_native.cu - block scoring and exact radix-select top-k of the QSA indexer for BI-V100.
//   qsa_block_scores : one 32-lane warp per pooled block, 8 warps per block (SCORE_WARPS); lane l handles dims l + 32*j (j < 4), so
//                      no alignment beyond 4 bytes is needed. score = sum_h hw[h]*relu(iq_h . key) (+ bias) (+1e9 on the tail block).
//   qsa_block_topk   : one block of TOPK_T = 256 threads. The 32-bit threshold key is fixed 8 bits at a time with weighted
//                      histograms (a block weighs 4 cells, the tail block n_kv - 4*n_bid); the selection is every cell strictly above
//                      the threshold plus the first (lowest index) width - above threshold-equal cells, emitted in ascending order.
// order_key is monotone on float bit patterns (negative patterns reverse); NaN maps to the reserved key 0, below every finite score.
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/qsa_native_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace qsa_native {
namespace {

constexpr int IDX_DIM = kD;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;
static_assert(IDX_DIM == 128 && SCORE_WARPS * 32 == 256 && TOPK_T == 256, "geometry");

struct Span { uint64_t addr; uint64_t bytes; };
bool validate(const Span& s) { return s.addr != 0 && s.bytes != 0 && s.addr % 4 == 0 && s.addr + s.bytes > s.addr; }
bool overlaps(const Span& a, const Span& b) { return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes; }
Span span(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }

__host__ __device__ __forceinline__ uint32_t order_key(float f) {
#ifdef __CUDA_ARCH__
    const uint32_t u = __float_as_uint(f);
#else
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
#endif
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) return kNanKey;  // NaN
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                                        const float* __restrict__ hw, const float* __restrict__ bias,
                                                                        float* __restrict__ scores, int n_kv) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_bid = n_kv / kR, tail = n_kv - n_bid * kR;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int b = blockIdx.x * SCORE_WARPS + warp;
    if (b >= nb) return;
    const float* key = pooled + (size_t)b * IDX_DIM;
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float part = 0.f;
#pragma unroll
        for (int j = 0; j < IDX_DIM / 32; ++j) part += key[lane + 32 * j] * iq[h * IDX_DIM + lane + 32 * j];
        const float dot = ilu_qsa::lane_sum(part);
        s += hw[h] * fmaxf(dot, 0.f);
    }
    if (lane == 0) {
        if (bias) s += bias[b];
        if (b == n_bid && tail > 0) s += kTailBias;
        scores[b] = s;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores, int n_kv, int width,
                                                            int32_t* __restrict__ out_ids, int32_t* __restrict__ n_out) {
    __shared__ uint32_t hist[256];
    __shared__ int32_t gtc[TOPK_T], eqc[TOPK_T], offs[TOPK_T], share[TOPK_T];
    __shared__ uint32_t s_prefix, s_above;
    const int t = threadIdx.x;

    if (n_kv <= width) {  // identity: every cell, ascending, padded with -1
        for (int i = t; i < width; i += TOPK_T) out_ids[i] = i < n_kv ? i : -1;
        if (t == 0) *n_out = n_kv;
        return;
    }
    const int n_bid = n_kv / kR, tail = n_kv - n_bid * kR;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int per = (nb + TOPK_T - 1) / TOPK_T;
    const int b_lo = min(nb, t * per), b_hi = min(nb, (t + 1) * per);
    if (t == 0) { s_prefix = 0; s_above = 0; }
    __syncthreads();

    for (int pass = 0; pass < 4; ++pass) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix;
        for (int b = b_lo; b < b_hi; ++b) {
            const uint32_t k = order_key(scores[b]);
            const bool match = pass == 0 || (k >> (32 - 8 * pass)) == prefix;
            if (match) atomicAdd(&hist[(k >> (24 - 8 * pass)) & 0xFFu], (uint32_t)(b < n_bid ? kR : tail));
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
        const uint32_t k = order_key(scores[b]);
        const int w = b < n_bid ? kR : tail;
        if (k > thr) gt += w; else if (k == thr) eq += w;
    }
    gtc[t] = gt;
    eqc[t] = eq;
    __syncthreads();
    if (t == 0) {
        const int budget = width - (int)s_above;  // threshold-equal cells that fit
        int before_eq = 0, running = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int sh = min(eqc[i], max(0, budget - before_eq));
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
        const uint32_t k = order_key(scores[b]);
        const int w = b < n_bid ? kR : tail;
        for (int c = 0; c < w; ++c) {
            if (k > thr) out_ids[pos++] = b * kR + c;
            else if (k == thr && left > 0) { out_ids[pos++] = b * kR + c; --left; }
        }
    }
}

}  // namespace

cudaError_t qsa_block_scores(const float* pooled, const float* iq, const float* hw, const float* bias, float* scores, int n_kv,
                             int rows, cudaStream_t s) {
    if (!pooled || !iq || !hw || !scores || n_kv < 1) return cudaErrorInvalidValue;
    const int nb = n_kv / kR + (n_kv % kR != 0 ? 1 : 0);
    if (rows < nb) return cudaErrorInvalidValue;
    const Span in[4] = {span(pooled, (uint64_t)rows * IDX_DIM * 4), span(iq, (uint64_t)4 * IDX_DIM * 4), span(hw, 16),
                        span(bias ? (const void*)bias : (const void*)hw, bias ? (uint64_t)nb * 4 : 16)};
    const Span out = span(scores, (uint64_t)nb * 4);
    for (int i = 0; i < 4; ++i) if (!validate(in[i]) || overlaps(out, in[i])) return cudaErrorInvalidValue;
    if (!validate(out)) return cudaErrorInvalidValue;
    block_scores_kernel<<<(unsigned)((nb + SCORE_WARPS - 1) / SCORE_WARPS), SCORE_WARPS * 32, 0, s>>>(pooled, iq, hw, bias, scores, n_kv);
    return cudaGetLastError();
}

cudaError_t qsa_block_topk(const float* scores, int n_kv, int width, int32_t* out_ids, int32_t* n_out, cudaStream_t s) {
    if (!scores || !out_ids || !n_out || n_kv < 1 || width < 1) return cudaErrorInvalidValue;
    const int nb = n_kv / kR + (n_kv % kR != 0 ? 1 : 0);
    const Span in = span(scores, (uint64_t)nb * 4), ids = span(out_ids, (uint64_t)width * 4), no = span(n_out, 4);
    if (!validate(in) || !validate(ids) || !validate(no)) return cudaErrorInvalidValue;
    if (overlaps(in, ids) || overlaps(in, no) || overlaps(ids, no)) return cudaErrorInvalidValue;
    block_topk_kernel<<<1, TOPK_T, 0, s>>>(scores, n_kv, width, out_ids, n_out);
    return cudaGetLastError();
}

}  // namespace qsa_native
}  // namespace baloon
