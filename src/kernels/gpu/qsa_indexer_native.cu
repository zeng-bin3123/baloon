// qsa_indexer_native.cu - single-block indexer key append with block pooling (D = 128 active dims, 256 threads).
//   slot = pos % 4. Slots 0..2 store the raw key in the 3-row tail. Slot 3 completes block b = pos/4: mean of tail rows 0..2 and
//   the incoming key (0.25 scale), RMS normalisation (eps), NEOX rotation of dims 0..63 (pairs d, d+32; angle = rope_pos *
//   freq_step^(d%32)) at rope_pos = pos_base + 4*b (the block's FIRST cell), store to pooled row b, move the spare row to b+1.
//   pos == 0 initialises pooled row 0 and the spare row: the gathered indices all name cell zero (four copies of the key) and the
//   rotation position is 0.
// All four buffers must be pairwise non-overlapping; pos_base must be a multiple of 4 and pos_base + max_cells fit int32.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "baloon/qsa_native_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace qsa_native {
namespace {

constexpr int D = kD;
constexpr int NT = 256;
bool g_enabled = true;

struct Span { uint64_t addr; uint64_t bytes; };
bool validate(const Span& s) { return s.addr != 0 && s.bytes != 0 && s.addr % 4 == 0 && s.addr + s.bytes > s.addr; }
bool overlaps(const Span& a, const Span& b) { return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes; }
Span span(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }

__device__ __forceinline__ float warp_sum(float v) { return ilu_qsa::lane_sum(v); }

__global__ void append(const float* key, float* tail, float* pooled, float* dead, int pos, int max_cells, long long pos_base,
                       float freq_step, float eps) {
    __shared__ float wsum[NT / 32];
    __shared__ float y[D];
    const int d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos & 3;
    const bool act = d < D;
    const float incoming = act ? key[d] : 0.f;

    if (slot < 3 && act) tail[slot * D + d] = incoming;
    if (slot != 3 && pos != 0) return;  // nothing to pool

    float mean;
    if (pos == 0) mean = incoming;      // four copies of cell zero: (x + x + x + x) * 0.25 == x
    else mean = act ? (tail[d] + tail[D + d] + tail[2 * D + d] + incoming) * 0.25f : 0.f;

    // block sum of squares over the 128 active dims
    float ss = warp_sum(mean * mean);
    if ((d & 31) == 0) wsum[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w) tot += wsum[w];
    const float v = mean * rsqrtf(tot / (float)D + eps);
    if (act) y[d] = v;
    __syncthreads();

    float outv = v;
    if (act && d < 64) {  // NEOX pairing (d, d + 32)
        const long long rpos = pos == 0 ? 0 : pos_base + 4ll * (pos >> 2);
        const int pair = d & 31;
        const float ang = (float)rpos * powf(freq_step, (float)pair);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float lo = y[pair], hi = y[pair + 32];
        outv = d < 32 ? lo * cs - hi * sn : hi * cs + lo * sn;
    }
    if (!act) return;
    if (pos == 0) {
        pooled[d] = outv;
        dead[d] = outv;
    } else {
        const int b = pos >> 2;
        pooled[(size_t)b * D + d] = outv;
        pooled[(size_t)(b + 1) * D + d] = dead[d];
    }
}

}  // namespace

bool native_qsa_indexer_enabled() { return g_enabled; }
void native_qsa_indexer_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_indexer_append(const IndexerArgs& a, cudaStream_t s) {
    if (!a.key || !a.tail || !a.pooled || !a.dead) return cudaErrorInvalidValue;
    if (a.max_cells < 1 || a.pos < 0 || a.pos >= a.max_cells) return cudaErrorInvalidValue;
    if (a.pos_base < 0 || a.pos_base % kR != 0) return cudaErrorInvalidValue;
    if ((unsigned long long)a.pos_base + (unsigned long long)a.max_cells > 2147483647ull) return cudaErrorInvalidValue;
    if (!(a.eps >= 0.f) || !std::isfinite(a.eps) || !(a.freq_step > 0.f) || !std::isfinite(a.freq_step)) return cudaErrorInvalidValue;
    const uint64_t pooled_rows = (uint64_t)a.max_cells / kR + 1;
    const Span sp[4] = {span(a.key, (uint64_t)D * 4), span(a.tail, (uint64_t)(kR - 1) * D * 4), span(a.pooled, pooled_rows * D * 4),
                        span(a.dead, (uint64_t)D * 4)};
    for (int i = 0; i < 4; ++i) if (!validate(sp[i])) return cudaErrorInvalidValue;
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (overlaps(sp[i], sp[j])) return cudaErrorInvalidValue;
    append<<<1, NT, 0, s>>>(a.key, a.tail, a.pooled, a.dead, a.pos, a.max_cells, a.pos_base, a.freq_step, a.eps);
    return cudaGetLastError();
}

}  // namespace qsa_native
}  // namespace baloon
