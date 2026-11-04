// qsa_select.cu - block scoring and block-weighted exact top-k over pooled indexer blocks (r = 4) for BI-V100.
//  qsa_block_scores: one 32-lane warp per pooled row, 8 warps per 256-thread block; every lane owns 4 of the 128 key dims.
//  qsa_block_topk  : one 256-thread block; 4 radix passes over weighted histograms (completed block weighs r cells, the
//                    spare row n_kv - n_bid * r); cells above the threshold are taken, plus the first width - above
//                    cells equal to it (lowest index first); output cells ascending.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/qsa_kernels.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int IDX_DIM = 128;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;
constexpr int kMaxR = 16;

__device__ __forceinline__ float shfl32(float v, int off) {
#if defined(__ILUVATAR__)
    return __shfl_xor(v, off, 32);
#else
    return __shfl_xor_sync(0xffffffffu, v, off, 32);
#endif
}
__device__ __forceinline__ uint32_t order_key(float f) {
    const uint32_t u = __float_as_uint(f + 0.0f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) qsa_block_scores_kernel(const float* __restrict__ pooled,
                                                                           const float* __restrict__ iq,
                                                                           const float* __restrict__ hw,
                                                                           float* __restrict__ scores, int n_kv, int r) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_bid = n_kv / r;
    const int b = blockIdx.x * SCORE_WARPS + warp;
    if (b > n_bid) return;  // rows 0..n_bid (the spare row is n_bid)
    float key[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) key[i] = pooled[(size_t)b * IDX_DIM + lane * 4 + i];
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < qsa_step::kIdxHeads; ++h) {
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) dot += key[i] * iq[h * IDX_DIM + lane * 4 + i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += shfl32(dot, o);
        s += hw[h] * fmaxf(dot, 0.f);
    }
    if (lane == 0) {
        if (b == n_bid && (n_kv % r) != 0) s += qsa_step::kTailBias;
        scores[b] = s;
    }
}

__global__ void __launch_bounds__(TOPK_T) qsa_block_topk_kernel(const float* __restrict__ scores, int n_kv, int r, int width,
                                                               int32_t* __restrict__ out_ids, int32_t* __restrict__ n_out) {
    __shared__ uint32_t hist[256];
    __shared__ int32_t gtc[TOPK_T], eqc[TOPK_T], offs[TOPK_T], share[TOPK_T];
    __shared__ uint32_t s_prefix, s_above;
    const int t = threadIdx.x;
    if (n_kv <= width) {
        for (int i = t; i < n_kv; i += TOPK_T) out_ids[i] = i;
        if (t == 0) *n_out = n_kv;
        return;
    }
    const int n_bid = n_kv / r, tail = n_kv - n_bid * r;
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
            if (pass == 0 || (k >> (32 - 8 * pass)) == prefix)
                atomicAdd(&hist[(k >> (24 - 8 * pass)) & 0xFFu], (uint32_t)(b < n_bid ? r : tail));
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
        const int w = b < n_bid ? r : tail;
        if (k > thr) gt += w; else if (k == thr) eq += w;
    }
    gtc[t] = gt; eqc[t] = eq;
    __syncthreads();
    if (t == 0) {
        const int budget = width - (int)s_above;
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
        const int w = b < n_bid ? r : tail;
        for (int c = 0; c < w; ++c) {
            if (k > thr) out_ids[pos++] = b * r + c;
            else if (k == thr && left > 0) { out_ids[pos++] = b * r + c; --left; }
        }
    }
}

}  // namespace

int qsa_block_scores(const float* pooled, const float* iq, const float* hw, float* scores, int n_kv, int r, int rows,
                     void* stream) {
    if (!pooled || !iq || !hw || !scores || n_kv < 1 || r < 1 || r > kMaxR) return kInvalid;
    if (rows < n_kv / r + 1) return kInvalid;  // pooled must hold rows 0..n_bid
    const int n_rows = n_kv / r + 1;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    qsa_block_scores_kernel<<<(n_rows + SCORE_WARPS - 1) / SCORE_WARPS, SCORE_WARPS * 32, 0, st>>>(pooled, iq, hw, scores, n_kv, r);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

int qsa_block_topk(const float* scores, int n_kv, int r, int width, int32_t* ids, int32_t* n_out, void* stream) {
    if (!scores || !ids || !n_out || n_kv < 1 || r < 1 || r > kMaxR || width < 1) return kInvalid;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    qsa_block_topk_kernel<<<1, TOPK_T, 0, st>>>(scores, n_kv, r, width, ids, n_out);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace qsa_dev
}  // namespace baloon
