// native_moe.cu - pinned ggml moe-weighted-reduction combine kernel.
// Accepts 1..15 routed parts plus an optional shared output.
// Exposes native_moe_combine_enabled / native_moe_combine_set_enabled / native_moe_combine.
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include "ix_compat.h"

namespace baloon {
namespace kern {

namespace {

std::atomic<bool> g_moe_enabled{true};

// § span guards
static inline bool valid_span(uint64_t addr, uint64_t bytes) {
    return addr != 0 && addr % 4 == 0 && bytes <= (uint64_t)UINT64_MAX - addr;
}
static inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return a < b + bn && b < a + an;
}

// § kernel: p0*w0 + p1*w1 + ... + shared (no leading 0+, part*weight order)
__global__ void combine(const float* __restrict__ parts, const float* __restrict__ weights,
                        const float* __restrict__ shared, float* __restrict__ out,
                        int n_embd, int k, int has_shared) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) return;
    float acc = parts[col] * weights[0];
    for (int e = 1; e < k; ++e)
        acc += parts[(size_t)e * n_embd + col] * weights[e];
    if (has_shared) acc += shared[col];
    out[col] = acc;
}

}  // namespace

// § public API
bool native_moe_combine_enabled() { return g_moe_enabled.load(std::memory_order_relaxed); }
void native_moe_combine_set_enabled(bool on) { g_moe_enabled.store(on, std::memory_order_relaxed); }

cudaError_t native_moe_combine(const float* parts, const float* weights, const float* shared,
                                float* out, int n_embd, int k, int has_shared, cudaStream_t st) {
    if (!st || n_embd <= 0 || n_embd > (int)(0x7fffffff) || k < 1 || k > 15)
        return cudaErrorInvalidValue;
    const uint64_t row = (uint64_t)n_embd * 4u;
    if (!valid_span((uint64_t)(uintptr_t)parts,   row * (uint64_t)k)) return cudaErrorInvalidValue;
    if (!valid_span((uint64_t)(uintptr_t)weights,  (uint64_t)k * 4u)) return cudaErrorInvalidValue;
    if (!valid_span((uint64_t)(uintptr_t)out,       row))             return cudaErrorInvalidValue;
    // output must not overlap parts or weights
    if (overlap((uint64_t)(uintptr_t)out, row, (uint64_t)(uintptr_t)parts,   row * (uint64_t)k))
        return cudaErrorInvalidValue;
    if (overlap((uint64_t)(uintptr_t)out, row, (uint64_t)(uintptr_t)weights, (uint64_t)k * 4u))
        return cudaErrorInvalidValue;
    if (has_shared) {
        if (!valid_span((uint64_t)(uintptr_t)shared, row)) return cudaErrorInvalidValue;
        if (overlap((uint64_t)(uintptr_t)out, row, (uint64_t)(uintptr_t)shared, row))
            return cudaErrorInvalidValue;
    }
    const int TH = 256;
    combine<<<((unsigned)n_embd + TH - 1u) / TH, TH, 0, st>>>(
        parts, weights, shared, out, n_embd, k, has_shared);
    return cudaGetLastError();
}

}  // namespace kern
}  // namespace baloon
