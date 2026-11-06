// moe_native.cu - float-accumulating MoE combine: y[j] = sum_e w[e] * parts[e][j] (+ shared[j] plain), ascending e, f32 accumulator
// (the double-precision form is moe_combine). Spans are validated; y must not overlap an input.
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
bool g_enabled = true;
bool valid_span(const void* p, size_t bytes) { return p != nullptr && bytes > 0; }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const char* pa = (const char*)a; const char* pb = (const char*)b;
    return pa < pb + bn && pb < pa + an;
}
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ w, const float* __restrict__ shared, float* __restrict__ y,
                        int n, int k, int has_shared) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.f;
    for (int e = 0; e < k; ++e) acc += w[e] * parts[(size_t)e * n + j];
    if (has_shared) acc += shared[j];
    y[j] = acc;
}
}  // namespace

bool native_moe_combine_enabled() { return g_enabled; }
void native_moe_combine_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_moe_combine(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k, int has_shared, cudaStream_t s) {
    if (n_embd <= 0 || k <= 0) return cudaErrorInvalidValue;
    const size_t pb = (size_t)n_embd * k * 4, nb = (size_t)n_embd * 4;
    if (!valid_span(parts, pb) || !valid_span(weights, (size_t)k * 4) || !valid_span(y, nb) || (has_shared && !valid_span(shared, nb))) return cudaErrorInvalidValue;
    if (overlap(y, nb, parts, pb) || overlap(y, nb, weights, (size_t)k * 4) || (has_shared && overlap(y, nb, shared, nb))) return cudaErrorInvalidValue;
    combine<<<(n_embd + 255) / 256, 256, 0, s>>>(parts, weights, shared, y, n_embd, k, has_shared);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
