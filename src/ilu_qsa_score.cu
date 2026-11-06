// ilu_qsa_score.cu - tile block scoring for BI-V100. BI-V100 (ivcore10) has no tf32 m16n8k8 instruction, so the
// accumulator fragment layout of that instruction is kept and each lane computes its own 4 fragment slots in fp32:
//   * 32 pooled rows per block, 2 logical warps (32 lanes) splitting the 128-dim key: warp w owns key columns
//     w*32 + lane and w*32 + lane + 64 (iteration it = 0, 1);
//   * accumulator slot l of A-tile ia (rows ia*16..ia*16+15) holds row ia*16 + (l/2)*8 + lane/4 and head
//     (lane%4)*2 + l%2; heads 4..7 are padding and skipped;
//   * K columns are staged through a 16 x 36 shared tile per warp; partials are exchanged through
//     comb[head*68 + warp*32 + row], which aliases the tile allocation (2*16*36 floats);
//   * the combine step sums the two warps' partials, applies the per-head ReLU, weights and sums the 4 heads,
//     adds the optional bias and the 1e9 tail bias, and broadcasts the block score to the block's cells.
#include <cuda_runtime.h>
#include <cstdint>
#include "baloon/ilu_qsa_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace ilu_qsa {
namespace {

constexpr int ROWS = 32, WARPS = 2, STRIDE = 36, COMBINE = 68, D = 128;
static_assert(ROWS == (int)native_qsa_score::kRows && STRIDE == (int)native_qsa_score::kStride &&
              COMBINE == (int)native_qsa_score::kCombine, "geometry");

__global__ void __launch_bounds__(64) mma_score_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                       const float* __restrict__ hw, const float* __restrict__ bias,
                                                       float* __restrict__ block_scores, float* __restrict__ cell_scores,
                                                       int n_kv) {
    __shared__ float smem[WARPS * 16 * STRIDE];  // tiles, later the exchange array
    __shared__ float sq[4 * D];
    __shared__ float shw[4];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int n_bid = n_kv / 4, tail = n_kv - n_bid * 4;
    const int nb = n_bid + (tail > 0 ? 1 : 0);
    const int row0 = blockIdx.x * ROWS;

    for (int i = tid; i < 4 * D; i += 64) sq[i] = iq[i];
    if (tid < 4) shw[tid] = hw[tid];
    __syncthreads();

    float acc[2][4];
#pragma unroll
    for (int ia = 0; ia < 2; ++ia)
#pragma unroll
        for (int l = 0; l < 4; ++l) acc[ia][l] = 0.f;

    float* tile = smem + warp * (16 * STRIDE);
    for (int ia = 0; ia < 2; ++ia) {
        for (int it = 0; it < 2; ++it) {
            const int col = warp * 32 + lane + 64 * it;
            for (int i = 0; i < 16; ++i) {  // tile[i*36 + lane] = K[row][col]
                const int row = row0 + ia * 16 + i;
                tile[i * STRIDE + lane] = row < nb ? pooled[(size_t)row * D + col] : 0.f;
            }
            __syncthreads();
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int head = (lane % 4) * 2 + l % 2;
                if (head < 4) {
                    const int lrow = (l / 2) * 8 + lane / 4;
                    float s = 0.f;
                    for (int c = 0; c < 32; ++c) s += tile[lrow * STRIDE + c] * sq[head * D + warp * 32 + c + 64 * it];
                    acc[ia][l] += s;
                }
            }
            __syncthreads();
        }
    }
    // exchange: writer cell head*68 + warp*32 + row (row < 32); the exchange array lives in the same allocation
#pragma unroll
    for (int ia = 0; ia < 2; ++ia)
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int head = (lane % 4) * 2 + l % 2;
            if (head < 4) smem[head * COMBINE + warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4] = acc[ia][l];
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
            for (int i = 0; i < 4; ++i) {  // broadcast to the cells of this block
                const int cell = row * 4 + i;
                if (cell < n_kv) cell_scores[cell] = s;
            }
        }
    }
}

}  // namespace

int mma_block_score(const float* pooled, const float* iq, const float* hw, const float* bias, float* block_scores,
                    float* cell_scores, int pos, int n_kv, int n_bid, int width, int max_cells, int rows, void* stream) {
    if (!pooled || !iq || !hw || !cell_scores || n_kv < 1 || max_cells < n_kv || pos < 0 || n_bid < 0 || width < 0) return kInvalid;
    if (!native_qsa_score::step_consistent((uint32_t)pos, (uint32_t)n_kv, (uint32_t)n_bid, (uint32_t)width)) return kInvalid;
    if (rows < 1 || !native_qsa_score::max_blocks_ok((uint32_t)rows, (uint32_t)max_cells)) return kInvalid;
    const int nb = n_kv / 4 + (n_kv % 4 != 0 ? 1 : 0);
    const Region r[] = {region(pooled, (uint64_t)rows * 512, false), region(iq, 4 * 512, false), region(hw, 16, false),
                        region(cell_scores, (uint64_t)n_kv * 4, true),
                        region(block_scores ? (const void*)block_scores : (const void*)hw, block_scores ? (uint64_t)nb * 4 : 16, block_scores != nullptr),
                        region(bias ? (const void*)bias : (const void*)hw, bias ? (uint64_t)nb * 4 : 16, false)};
    if (!regions_ok(r, sizeof(r) / sizeof(r[0]))) return kInvalid;
    mma_score_kernel<<<native_qsa_score::grid_blocks((uint32_t)nb), 64, 0, static_cast<cudaStream_t>(stream)>>>(
        pooled, iq, hw, bias, block_scores, cell_scores, n_kv);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace ilu_qsa
}  // namespace baloon
