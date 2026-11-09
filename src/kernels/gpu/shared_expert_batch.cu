// shared_expert_batch.cu - token-batched moe_combine: one thread per (token, column); the shared output is added plain
// (not router-weighted), routed terms accumulate in ascending expert order. Guards: 1 <= k <= 64, n_tok >= 1.
#include "kg_common.cuh"
#include "baloon/shared_expert_batch.hpp"
#include "baloon/shared_expert_layout.hpp"

namespace baloon {
namespace kg {
// External linkage on purpose: the kernel symbol must be visible by name to launch tracing.
__global__ void moe_combine_rows_kernel(const float* __restrict__ parts, const float* __restrict__ weights,
                                        const float* __restrict__ shared, float* __restrict__ y,
                                        int n_embd, int k, int has_shared, long long total) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = (int)(idx / n_embd), j = (int)(idx % n_embd);
    const float* wt = weights + (size_t)t * k;
    const float* pt = parts + (size_t)t * k * n_embd + j;
    float acc = has_shared ? shared[idx] : 0.f;          // shared output enters as the starting accumulator (plain)
    for (int e = 0; e < k; ++e) acc += wt[e] * pt[(size_t)e * n_embd];
    y[idx] = acc;
}

cudaError_t moe_combine_rows(const float* parts, const float* weights, const float* shared, float* y,
                             int n_tok, int n_embd, int k, int has_shared, cudaStream_t s) {
    if (!parts || !weights || !y || n_tok < 1 || n_embd <= 0 || (has_shared && !shared)) return cudaErrorInvalidValue;
    if (!shared_expert::valid_combine_k((uint64_t)(k > 0 ? k : 0))) return cudaErrorInvalidValue;
    const long long total = (long long)n_tok * n_embd;
    moe_combine_rows_kernel<<<(unsigned)((total + 127) / 128), 128, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared, total);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
