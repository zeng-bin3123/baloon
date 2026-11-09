// shared_expert_batch.hpp - token-batched routed-plus-shared combination (theory: SharedExpertBlock, `combine`).
//   y[t][j] = shared[t][j] (plain, optional) + sum_{e<k} w[t][e] * parts[t][e][j], ascending e, k <= 64.
#pragma once
#include <cuda_runtime.h>
namespace baloon {
namespace kg {
// parts [n_tok][k][n_embd], weights [n_tok][k], shared [n_tok][n_embd] (ignored when has_shared == 0), y [n_tok][n_embd].
cudaError_t moe_combine_rows(const float* parts, const float* weights, const float* shared, float* y,
                             int n_tok, int n_embd, int k, int has_shared, cudaStream_t s = 0);
}  // namespace kg
}  // namespace baloon
