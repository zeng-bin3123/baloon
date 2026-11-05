// router_top10_k.cu - MoE router: softmax over experts (double), top-k by descending probability with the LOWEST index winning
// ties, then renormalisation by max(sum of selected, 2^-14). One block per token; the k selection passes run on thread 0's wave.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int RT_MAX_THREADS = 256;

__global__ void router_top10_kernel(const float* __restrict__ logits, int n_expert, int k, int32_t* __restrict__ ids, float* __restrict__ weights) {
    extern __shared__ float sm[];                   // probabilities [n_expert]
    __shared__ float sv[RT_MAX_THREADS / 64];
    __shared__ float sum_sh[RT_MAX_THREADS / 64];
    __shared__ int taken_cnt;
    const int t = blockIdx.x, tid = threadIdx.x, nw = blockDim.x >> 6;
    const float* l = logits + (size_t)t * n_expert;
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += blockDim.x) mx = fmaxf(mx, l[e]);
    mx = kgc::wave_max(mx);
    if ((tid & 63) == 0) sv[tid >> 6] = mx;
    __syncthreads();
    mx = sv[0];
    for (int w = 1; w < nw; ++w) mx = fmaxf(mx, sv[w]);
    for (int e = tid; e < n_expert; e += blockDim.x) sm[e] = expf(l[e] - mx);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.f;
        for (int e = 0; e < n_expert; ++e) sum += sm[e];
        sum_sh[0] = sum;
        taken_cnt = 0;
    }
    __syncthreads();
    const float inv = 1.0f / sum_sh[0];
    for (int e = tid; e < n_expert; e += blockDim.x) sm[e] = sm[e] * inv;
    __syncthreads();
    if (tid == 0) {
        float chosen[64];
        const int kk = k < 64 ? k : 64;
        for (int i = 0; i < kk; ++i) {
            float bv = -INFINITY; int bi = -1;
            for (int e = 0; e < n_expert; ++e) {
                if (sm[e] < 0.f) continue;          // already taken
                const float pv = (float)sm[e];
                if (bi < 0 || pv > bv) { bv = pv; bi = e; }   // strict >: lowest index kept on ties
            }
            if (bi < 0) break;
            ids[(size_t)t * k + i] = bi; chosen[i] = bv; sm[bi] = -1.f;
        }
        float s = 0.f;
        for (int i = 0; i < kk; ++i) s += chosen[i];
        const float sc = s > 6.103515625e-05f ? s : 6.103515625e-05f;
        for (int i = 0; i < kk; ++i) weights[(size_t)t * k + i] = chosen[i] / sc;
    }
}
}  // namespace

cudaError_t router_top10(const float* logits, int n_tokens, int n_expert, int k, int32_t* ids, float* weights, cudaStream_t s) {
    if (!logits || !ids || !weights || n_tokens <= 0 || n_expert <= 0 || k <= 0 || k > n_expert || k > 64) return cudaErrorInvalidValue;
    router_top10_kernel<<<n_tokens, RT_MAX_THREADS, (size_t)n_expert * sizeof(float), s>>>(logits, n_expert, k, ids, weights);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
