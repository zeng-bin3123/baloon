// qsa_native_idx.cu - indexer key append of the sparse-attention path (D = 128 floats per key). The new key is stored at row `pos` of the
// index cache L2-normalised (unit length), and its original length is recorded in norms[pos]. One 64-lane wave, two values per lane.
// A zero key is stored as zeros with norm 0 (no NaN).
#include "kg_common.cuh"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int D = 128;
bool g_enabled = true;
struct Span { const void* p; size_t bytes; };
bool validate(const Span& s) { return s.p != nullptr && s.bytes > 0; }
bool overlaps(const Span& a, const Span& b) {
    const char* pa = (const char*)a.p; const char* pb = (const char*)b.p;
    return pa < pb + b.bytes && pb < pa + a.bytes;
}
__device__ __forceinline__ float warp_sum(float v) { return kgc::wave_sum(v); }

__global__ void append(float* __restrict__ cache, float* __restrict__ norms, const float* __restrict__ key, int pos) {
    const int lane = threadIdx.x;
    const float a = key[lane], b = key[lane + 64];
    const float ss = warp_sum(a * a + b * b);
    const float nrm = sqrtf(ss);
    const float inv = nrm > 0.f ? 1.0f / nrm : 0.f;
    cache[(size_t)pos * D + lane] = a * inv;
    cache[(size_t)pos * D + lane + 64] = b * inv;
    if (lane == 0) norms[pos] = nrm;
}
}  // namespace

bool native_qsa_indexer_enabled() { return g_enabled; }
void native_qsa_indexer_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_indexer_append(float* cache, float* norms, const float* key, int pos, int capacity, cudaStream_t s) {
    if (pos < 0 || capacity <= 0 || pos >= capacity) return cudaErrorInvalidValue;
    const Span c{cache, (size_t)capacity * D * 4}, n{norms, (size_t)capacity * 4}, k{key, (size_t)D * 4};
    if (!validate(c) || !validate(n) || !validate(k)) return cudaErrorInvalidValue;
    if (overlaps(c, n) || overlaps(c, k) || overlaps(n, k)) return cudaErrorInvalidValue;
    append<<<1, 64, 0, s>>>(cache, norms, key, pos);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
