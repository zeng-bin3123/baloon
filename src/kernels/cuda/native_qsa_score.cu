// qsa_score_native.cu - tile block scoring of the indexer for BI-V100 (no tf32 mma instruction on ivcore10, so the fragment layout
// of an m16n8k8-style tile is kept and every lane computes its own four accumulator slots in fp32).
//   * a CUDA block handles 32 pooled rows with two logical 32-lane warps splitting the 128-dim key: warp w owns key columns
//     w*32 + lane and w*32 + lane + 64;
//   * TileA is a 16 x 32 slice of pooled keys staged in shared memory (row stride 36), TileB a 32-wide slice of the 8 padded
//     query heads, TileC the lane's accumulator slots l = 0..3: row (l/2)*8 + lane/4 of the tile, head (lane%4)*2 + l%2;
//   * partials are exchanged through comb[head*68 + warp*32 + row]; the combine step sums the two warps, applies ReLU per head,
//     weights and sums the 4 real heads (heads 4..7 are padding), adds the optional bias and +1e9 for an incomplete tail row.
//   * row r writes cells [4r, min(4r+4, n)); rows 0..n_bid are scored (the tail row only when n % 4 != 0).
#include <cuda_runtime.h>

#include <cstdint>

#include "baloon/qsa_native_kernels.hpp"

namespace baloon {
namespace qsa_native {
namespace {

constexpr int D = kD;
constexpr int ROWS = kRows;
constexpr int STRIDE = 36;
constexpr int COMBINE = 68;
constexpr int WARPS = 2;
bool g_enabled = true;

struct Span { uint64_t addr; uint64_t bytes; };
bool validate(const Span& s) { return s.addr != 0 && s.bytes != 0 && s.addr % 4 == 0 && s.addr + s.bytes > s.addr; }
bool overlaps(const Span& a, const Span& b) { return a.addr < b.addr + b.bytes && b.addr < a.addr + a.bytes; }
Span span(const void* p, uint64_t bytes) { return Span{(uint64_t)(uintptr_t)p, bytes}; }

struct TileA { float* v; };        // 16 rows x 32 columns, row stride STRIDE (shared memory)
struct TileB { const float* q; };  // query heads, head h column j at q[h*D + j]
struct TileC { float v[4]; };      // accumulator slots

// Stage rows row0..row0+15 of columns col0 + lane (zero past the last real row `nb`).
__device__ __forceinline__ void load_a(TileA& a, const float* pooled, int row0, int nb, int col0, int lane) {
    for (int i = 0; i < 16; ++i) {
        const int row = row0 + i;
        a.v[i * STRIDE + lane] = row < nb ? pooled[(size_t)row * D + col0 + lane] : 0.f;
    }
}
__device__ __forceinline__ void load_b(TileB& b, const float* sq, int col0) { b.q = sq + col0; }

// c.v[l] += sum_{k<32} A[row(l, lane)][k] * B[head(l, lane)][k]; padded heads (>= 4) are skipped.
__device__ __forceinline__ void mma(TileC& c, const TileA& a, const TileB& b, int lane) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int head = (lane % 4) * 2 + l % 2;
        if (head < 4) {
            const int lrow = (l / 2) * 8 + lane / 4;
            float s = 0.f;
            for (int k = 0; k < 32; ++k) s += a.v[lrow * STRIDE + k] * b.q[head * D + k];
            c.v[l] += s;
        }
    }
}

__global__ void __launch_bounds__(64) score_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                   const float* __restrict__ hw, const float* __restrict__ bias,
                                                   float* __restrict__ block_scores, float* __restrict__ cell_scores, int n) {
    __shared__ float smem[WARPS * 16 * STRIDE];  // tiles, later the exchange array
    __shared__ float sq[4 * D];
    __shared__ float shw[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_bid = n / kR, tail = n - n_bid * kR;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int row0 = blockIdx.x * ROWS;

    for (int i = tid; i < 4 * D; i += 64) sq[i] = iq[i];
    if (tid < 4) shw[tid] = hw[tid];
    __syncthreads();

    TileC acc[2];
    for (int ia = 0; ia < 2; ++ia)
        for (int l = 0; l < 4; ++l) acc[ia].v[l] = 0.f;

    TileA ta{smem + warp * (16 * STRIDE)};
    TileB tb{nullptr};
    for (int ia = 0; ia < 2; ++ia) {
        for (int it = 0; it < 2; ++it) {
            const int col = warp * 32 + 64 * it;  // first column of this warp's slice
            load_a(ta, pooled, row0 + ia * 16, nb, col, lane);
            load_b(tb, sq, col);
            __syncthreads();
            mma(acc[ia], ta, tb, lane);
            __syncthreads();
        }
    }
    // exchange: writer cell head*68 + warp*32 + row (row < 32); the exchange array lives in the same allocation
    for (int ia = 0; ia < 2; ++ia)
        for (int l = 0; l < 4; ++l) {
            const int head = (lane % 4) * 2 + l % 2;
            if (head < 4) smem[head * COMBINE + warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4] = acc[ia].v[l];
        }
    __syncthreads();
    if (warp == 0) {  // reader: row = lane
        const int row = row0 + lane;
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 4; ++h) s += shw[h] * fmaxf(smem[h * COMBINE + lane] + smem[h * COMBINE + 32 + lane], 0.f);
        if (row < nb) {
            if (bias) s += bias[row];
            if (row == n_bid && tail > 0) s += kTailBias;
            if (block_scores) block_scores[row] = s;
            for (int i = 0; i < kR && row * kR + i < n; ++i) cell_scores[row * kR + i] = s;
        }
    }
}

}  // namespace

bool native_qsa_score_enabled() { return g_enabled; }
void native_qsa_score_set_enabled(bool on) { g_enabled = on; }

cudaError_t native_qsa_score(const ScoreArgs& a, cudaStream_t s) {
    if (!a.pooled || !a.iq || !a.hw || !a.cell_scores) return cudaErrorInvalidValue;
    if (a.n < 1 || a.max_cells < a.n || a.pos < 0 || a.n_bid < 0 || a.width < 0 || a.max_blocks < 1) return cudaErrorInvalidValue;
    if (!score_step_ok(a.n, a.max_cells, a.pos, a.n_bid, a.width)) return cudaErrorInvalidValue;
    if (a.max_blocks != a.max_cells / kR + 1) return cudaErrorInvalidValue;
    const int nb = a.n / kR + (a.n % kR != 0 ? 1 : 0);
    const Span in[5] = {span(a.pooled, (uint64_t)a.max_blocks * D * 4), span(a.iq, (uint64_t)4 * D * 4), span(a.hw, 16),
                        span(a.bias ? (const void*)a.bias : (const void*)a.hw, a.bias ? (uint64_t)nb * 4 : 16),
                        span(a.cell_scores, (uint64_t)a.n * 4)};
    for (int i = 0; i < 5; ++i) if (!validate(in[i])) return cudaErrorInvalidValue;
    // outputs: cell_scores (in[4]) and the optional block_scores
    const Span cs = in[4];
    for (int i = 0; i < 4; ++i) if (overlaps(cs, in[i])) return cudaErrorInvalidValue;
    if (a.block_scores) {
        const Span bs = span(a.block_scores, (uint64_t)nb * 4);
        if (!validate(bs) || overlaps(bs, cs)) return cudaErrorInvalidValue;
        for (int i = 0; i < 4; ++i) if (overlaps(bs, in[i])) return cudaErrorInvalidValue;
    }
    const unsigned blocks = (unsigned)((a.max_blocks + ROWS - 1) / ROWS);
    score_kernel<<<blocks, 64, 0, s>>>(a.pooled, a.iq, a.hw, a.bias, a.block_scores, a.cell_scores, a.n);
    return cudaGetLastError();
}

}  // namespace qsa_native
}  // namespace baloon
