// router_top10_k.cu - MoE router: softmax over all experts, k passes of a BLOCK-WIDE argmax (ties -> the lowest expert index),
// gather of the selected probabilities, renormalisation by max(sum, 2^-14). One block per token.
//   threads  = round_up_32(min(n_expert, RT_MAX_THREADS))          (router_top_k_selection.hpp: launch_threads)
//   dynamic shared bytes = taken mask padded to 16 B | n 8-byte pairs | n floats      (router_top_k_selection.hpp: smem_bytes)
//     taken     : bitmask in 32-bit words (bit e set once expert e is selected); word accesses only, no sub-word shared stores
//     cand[n]   : {float probability, int expert}, the (value, index) pair every argmax pass compares (no fp64 on the device)
//     prob[n]   : float, the softmax probability, never modified: the gather reads it
// A candidate (ov, oi) replaces the best (bv, bi) iff ov > bv or (ov == bv and oi < bi): irreflexive, asymmetric, transitive and
// total on distinct indices, so the reduction tree shape does not change the winner.
// The softmax recipe (float exp, sequential float sum in expert order, multiply by the float reciprocal) is shared with
// native_router_top10 (router_native.cu), which must stay bit-identical.
#include <climits>

#include "baloon/kg_kernels.hpp"
#include "baloon/router_top_k_selection.hpp"
#include "kg_common.cuh"

namespace baloon {
namespace kg {
namespace {
constexpr int RT_MAX_THREADS = 512;
constexpr float kRenormClamp = 6.103515625e-05f;   // 2^-14
struct Cand { float v; int i; };

__device__ __forceinline__ bool beats(float ov, int oi, float bv, int bi) { return ov > bv || (ov == bv && oi < bi); }

__global__ void router_top10_kernel(const float* __restrict__ logits, int n_expert, int k, int32_t* __restrict__ ids, float* __restrict__ weights) {
    extern __shared__ __align__(16) unsigned char rt_smem[];
    __shared__ float red_v[RT_MAX_THREADS];
    __shared__ int red_i[RT_MAX_THREADS];
    __shared__ float sum_sh;
    const int t = blockIdx.x, tid = threadIdx.x, nthr = blockDim.x;
    const size_t taken_bytes = ((size_t)n_expert + 15) / 16 * 16;
    unsigned* taken = (unsigned*)rt_smem;
    Cand* cand = (Cand*)(rt_smem + taken_bytes);
    float* prob = (float*)(cand + n_expert);
    const float* l = logits + (size_t)t * n_expert;

    // ---- softmax: max, exp, sum, scale
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += nthr) mx = fmaxf(mx, l[e]);
    red_v[tid] = mx;                       // block max by a tree (independent of the wave size and of partial waves)
    __syncthreads();
    int span0 = 1;
    while (span0 < nthr) span0 <<= 1;
    for (int s = span0 >> 1; s > 0; s >>= 1) {
        if (tid < s && tid + s < nthr && red_v[tid + s] > red_v[tid]) red_v[tid] = red_v[tid + s];
        __syncthreads();
    }
    mx = red_v[0];
    __syncthreads();
    for (int e = tid; e < n_expert; e += nthr) prob[e] = expf(l[e] - mx);
    for (int w = tid; w < (int)(taken_bytes >> 2); w += nthr) taken[w] = 0u;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int e = 0; e < n_expert; ++e) sum += prob[e];
        sum_sh = sum;
    }
    __syncthreads();
    const float inv = 1.0f / sum_sh;
    for (int e = tid; e < n_expert; e += nthr) { const float p = prob[e] * inv; prob[e] = p; cand[e].v = p; cand[e].i = e; }
    __syncthreads();

    // ---- k block-wide argmax passes
    const int kk = k < 64 ? k : 64;
    for (int pass = 0; pass < kk; ++pass) {
        float bv = -INFINITY; int bi = INT_MAX;
        for (int e = tid; e < n_expert; e += nthr)            // ascending scan: strict > keeps the lowest index of a tie
            if (!((taken[e >> 5] >> (e & 31)) & 1u) && beats(cand[e].v, cand[e].i, bv, bi)) { bv = cand[e].v; bi = cand[e].i; }
        red_v[tid] = bv; red_i[tid] = bi;
        __syncthreads();
        int span = 1;
        while (span < nthr) span <<= 1;
        for (int s = span >> 1; s > 0; s >>= 1) {
            if (tid < s && tid + s < nthr && beats(red_v[tid + s], red_i[tid + s], red_v[tid], red_i[tid])) { red_v[tid] = red_v[tid + s]; red_i[tid] = red_i[tid + s]; }
            __syncthreads();
        }
        if (tid == 0) {
            const int w = red_i[0];
            if (w == INT_MAX) { ids[(size_t)t * k + pass] = 0; weights[(size_t)t * k + pass] = 0.f; }   // nothing selectable (NaN row)
            else { ids[(size_t)t * k + pass] = w; taken[w >> 5] |= 1u << (w & 31); }
        }
        __syncthreads();
    }

    // ---- gather + renormalise
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < kk; ++i) s += prob[ids[(size_t)t * k + i]];
        const float sc = s > kRenormClamp ? s : kRenormClamp;
        for (int i = 0; i < kk; ++i) {
            const int e = ids[(size_t)t * k + i];
            weights[(size_t)t * k + i] = prob[e] / sc;
        }
    }
}
}  // namespace

cudaError_t router_top10(const float* logits, int n_tokens, int n_expert, int k, int32_t* ids, float* weights, cudaStream_t s) {
    if (!logits || !ids || !weights || n_tokens <= 0 || n_expert <= 0 || k <= 0 || k > n_expert || k > 64) return cudaErrorInvalidValue;
    const size_t smem = (size_t)router_topk::smem_bytes((uint64_t)n_expert);
    if (smem > 48 * 1024) return cudaErrorInvalidValue;
    const int threads = (int)router_topk::launch_threads((uint64_t)n_expert);
    router_top10_kernel<<<n_tokens, threads, smem, s>>>(logits, n_expert, k, ids, weights);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
