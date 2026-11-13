// native_qsa_indexer.cu - Native (single-block) indexer key append for BI-V100:
//   append kernel: tail-append (slot < r-1) or block-pool + rotate + spare-row write (slot == r-1).
//   Host API: native_qsa_indexer_append (wraps the CUDA kernel).
// Return codes: cudaError_t.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace baloon {
namespace qsa_indexer_native {
namespace {

constexpr int D = 128;    // indexer key dimension

static bool g_enabled = true;

struct Span { uint64_t addr; uint64_t bytes; };
static bool validate(const Span& s) {
    return s.addr != 0 && s.bytes != 0 && s.addr % 4 == 0;
}
static bool overlaps(const Span& a, const Span& b) {
    return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// append kernel: tail store or block pool + RMS norm + write to pooled[b] and spare[b+1]
__global__ void append(const float* __restrict__ key, float* __restrict__ tail,
                       float* __restrict__ pooled, int pos, int max_cells, float eps, int r) {
    const int d = threadIdx.x;
    if (d >= D) return;
    const int slot = pos % r;
    if (slot < r - 1) {
        // Write to tail[slot][d]
        tail[slot * D + d] = key[d];
        return;
    }
    // Completing block: pool tail + current key → pooled[b]
    float sum = key[d];
    for (int s = 0; s < r - 1; ++s) sum += tail[s * D + d];
    sum /= (float)r;
    // RMS norm: compute mean square
    float sq = sum * sum;
    __shared__ float sss;
    if (d == 0) sss = 0.f;
    __syncthreads();
    atomicAdd(&sss, sq);
    __syncthreads();
    const float rms = rsqrtf(sss / D + eps);
    const float normed = sum * rms;
    const int b = pos / r;
    if (b < max_cells / r + 1) {
        pooled[(int64_t)b * D + d] = normed;
        // Spare row at b+1 (always written to maintain the dead-row invariant)
        pooled[(int64_t)(b + 1) * D + d] = normed;
    }
}

} // anonymous namespace

bool native_qsa_indexer_enabled() { return g_enabled; }
void native_qsa_indexer_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_indexer_append(const float* key, float* tail, float* pooled,
                                       int pos, int max_cells, float eps, cudaStream_t s) {
    if (!key || !tail || !pooled || pos < 0 || max_cells <= 0) return cudaErrorInvalidValue;
    const Span sk = {(uint64_t)(uintptr_t)key,    (uint64_t)D * 4};
    const Span st = {(uint64_t)(uintptr_t)tail,   (uint64_t)3 * D * 4};
    const Span sp = {(uint64_t)(uintptr_t)pooled, (uint64_t)(max_cells / 4 + 2) * D * 4};
    if (!validate(sk) || !validate(st) || !validate(sp)) return cudaErrorInvalidValue;
    if (overlaps(sk, st) || overlaps(sk, sp) || overlaps(st, sp)) return cudaErrorInvalidValue;
    if (!g_enabled) return cudaSuccess;
    append<<<1, D, 0, s>>>(key, tail, pooled, pos, max_cells, eps, 4);
    return cudaGetLastError();
}

} // namespace qsa_indexer_native
} // namespace baloon
