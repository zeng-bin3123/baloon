// qsa_select.cu - Block-level score and top-k select for QSA decode:
//   qsa_block_scores  : score each block by its cell range (mean of cell_scores[lo..hi))
//   qsa_block_topk    : radix select the top-k weighted blocks; expand each to its cells
// Both kernels mirror the reference's block → cell expansion semantics.
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace baloon {
namespace qsa_block_select {
namespace {

constexpr int IDX_DIM   = 128;    // indexer key dimension (D)
constexpr int SCORE_WARPS = 8;    // warps per block in the score kernel
constexpr int TOPK_T    = 32;     // threads per top-k block

// Order key: makes f32 bit patterns sort by value (negatives inverted).
__host__ __device__ __forceinline__ uint32_t order_key(uint32_t b) {
    return (b & 0x80000000u) ? (0xFFFFFFFFu - b) : (b | 0x80000000u);
}

// Score one block: score = cell_scores[b*R] (each cell already carries the block score).
__global__ void __launch_bounds__(256) qsa_block_scores_kernel(
        const float* __restrict__ cell_scores, int n_kv, int r,
        float* __restrict__ block_scores, int n_blocks) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const int lo = b * r;
    if (lo >= n_kv) { block_scores[b] = -1e38f; return; }
    block_scores[b] = cell_scores[lo];
}

// Top-k over blocks weighted by cell count; expand each selected block into its cells.
__global__ void __launch_bounds__(32) qsa_block_topk_kernel(
        const float* __restrict__ block_scores, int n_bid, int r, int n_kv, int width,
        int32_t* __restrict__ out_ids, int32_t* __restrict__ n_out) {
    // Simple sorted scan (single warp for the small block counts in practice)
    __shared__ uint32_t skeys[2051];
    const int tid = threadIdx.x;
    const int nb = n_bid + 1;
    for (int b = tid; b < nb; b += TOPK_T) {
        uint32_t bits;
        const float s = b < nb ? block_scores[b] : -1e38f;
        __builtin_memcpy(&bits, &s, 4);
        skeys[b] = order_key(bits);
    }
    __syncthreads();
    // Single-thread scan for correctness (block count <= 2051 is small)
    if (tid == 0) {
        // Find threshold
        int emit = 0;
        uint32_t thr = 0;
        for (int bk = 31; bk >= 0; --bk) {
            uint32_t cand = thr | (1u << bk);
            int cnt = 0;
            for (int b = 0; b < nb; ++b) if (skeys[b] >= cand) cnt++;
            if (cnt >= 1) thr = cand;
        }
        for (int b = 0; b < nb && emit < width; ++b) {
            if (skeys[b] > thr || (skeys[b] == thr && emit < width)) {
                // Expand block b into its cells
                const int lo = b * r;
                const int hi = (b < n_bid) ? lo + r : n_kv;
                for (int c = lo; c < hi && emit < width; ++c)
                    out_ids[emit++] = c;
            }
        }
        *n_out = emit;
    }
}

} // anonymous namespace

int qsa_block_scores(const float* cell_scores, int n_kv, int r,
                     float* block_scores, void* stream) {
    if (!cell_scores || !block_scores || n_kv <= 0 || r <= 0) return -1;
    const int nb = n_kv / r + (n_kv % r != 0 ? 1 : 0);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    qsa_block_scores_kernel<<<(nb + block - 1) / block, block, 0, s>>>(
        cell_scores, n_kv, r, block_scores, nb);
    return cudaGetLastError() == cudaSuccess ? 0 : (int)cudaGetLastError();
}

int qsa_block_topk(const float* block_scores, int n_bid, int r, int n_kv, int width,
                   int32_t* out_ids, int32_t* n_out, void* stream) {
    if (!block_scores || !out_ids || !n_out || n_kv <= 0 || width <= 0) return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    qsa_block_topk_kernel<<<1, TOPK_T, 0, s>>>(block_scores, n_bid, r, n_kv, width, out_ids, n_out);
    return cudaGetLastError() == cudaSuccess ? 0 : (int)cudaGetLastError();
}

} // namespace qsa_block_select
} // namespace baloon
