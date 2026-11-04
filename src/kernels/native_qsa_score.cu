// native_qsa_score.cu - tile scorer of pooled indexer keys against the 4 indexer query heads (m16n8k8 tf32 tiles).
// The tile product is written as an explicit 16x8x8 fma loop over TileA (keys) x TileB (queries, 4 heads padded to 8) with
// inputs truncated to tf32, accumulating in f32 in ascending k order. A 64-thread block (2 warps) scores 32 pooled rows:
// the 128 key columns are split between the warps (64 each); the partial dots are combined in shared memory, then
// score(row) = sum_h hw[h] * relu(dot_h), written to cells [4 row, min(n, 4 row + 4)). The spare row gets +1e9 iff n % 4 != 0.
// The kernel runs only for a consistent step (1 <= n <= max_cells, pos = n - 1, full = n / 4, width = min(n, 2051)).
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include "baloon/qsa_native_kernels.hpp"

namespace baloon {
namespace qsa_dev {
namespace {

constexpr int D = 128;     // key width
constexpr int ROWS = 32;   // pooled rows per block
constexpr int WARPS = 2;
constexpr int R = 4;

struct TileA { float v[16][8]; };  // 16 key rows x 8 k
struct TileB { float v[8][8]; };   // 8 k x 8 query columns (4 heads + 4 zero pads)
struct TileC { float v[16][8]; };  // 16 rows x 8 columns

__device__ __forceinline__ float tf32(float x) { return __uint_as_float(__float_as_uint(x) & 0xFFFFE000u); }

// Cooperative loads by the 32 lanes of one warp: lane l fills 4 elements of A (128 per tile) / 2 of B (64 per tile).
__device__ __forceinline__ void load_a(TileA& a, const float* pooled, int row0, int rows_valid, int k0, int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = lane * 4 + i, r = e / 8, k = e % 8;
        a.v[r][k] = (row0 + r < rows_valid) ? tf32(pooled[(size_t)(row0 + r) * D + k0 + k]) : 0.f;
    }
}
__device__ __forceinline__ void load_b(TileB& b, const float* iq, int k0, int lane) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int e = lane * 2 + i, k = e / 8, c = e % 8;
        b.v[k][c] = c < qsa_step::kIdxHeads ? tf32(iq[c * D + k0 + k]) : 0.f;
    }
}
// c += a * b for the 4 outputs owned by `lane` (element e = lane * 4 + i -> row e / 8, column e % 8).
__device__ __forceinline__ void mma(TileC& c, const TileA& a, const TileB& b, int lane) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int e = lane * 4 + i, r = e / 8, col = e % 8;
        float acc = c.v[r][col];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc = fmaf(a.v[r][k], b.v[k][col], acc);
        c.v[r][col] = acc;
    }
}

__global__ void __launch_bounds__(WARPS * 32) score_kernel(const float* __restrict__ pooled, const float* __restrict__ iq,
                                                           const float* __restrict__ hw, float* __restrict__ cell_score,
                                                           const Step* __restrict__ step, int max_cells) {
    __shared__ TileA ta[WARPS];
    __shared__ TileB tb[WARPS];
    __shared__ TileC tc[WARPS];
    __shared__ float comb[16][8];
    const int n = step->n_kv, pos = step->pos, full = step->n_bid, width = step->width;
    const bool ok = n >= 1 && n <= max_cells && pos == n - 1 && full == n / R && width == (n < 2051 ? n : 2051);
    if (!ok) return;  // inconsistent step: write nothing (uniform across the block)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int blk_row0 = blockIdx.x * ROWS;
    if (blk_row0 > full) return;
    for (int tile = 0; tile < ROWS / 16; ++tile) {
        const int row0 = blk_row0 + tile * 16;
        const int rows_valid = full + 1;
#pragma unroll
        for (int i = 0; i < 4; ++i) tc[warp].v[(lane * 4 + i) / 8][(lane * 4 + i) % 8] = 0.f;
        for (int ks = 0; ks < D / WARPS / 8; ++ks) {   // 8 k-steps of 8 per warp (its own half of the key)
            const int k0 = warp * (D / WARPS) + ks * 8;
            load_a(ta[warp], pooled, row0, rows_valid, k0, lane);
            load_b(tb[warp], iq, k0, lane);
            __syncthreads();
            mma(tc[warp], ta[warp], tb[warp], lane);
            __syncthreads();
        }
        // combine the two K halves: warp 0 publishes, warp 1 adds
        if (warp == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) comb[(lane * 4 + i) / 8][(lane * 4 + i) % 8] = tc[0].v[(lane * 4 + i) / 8][(lane * 4 + i) % 8];
        }
        __syncthreads();
        if (warp == 1) {
#pragma unroll
            for (int i = 0; i < 4; ++i) comb[(lane * 4 + i) / 8][(lane * 4 + i) % 8] += tc[1].v[(lane * 4 + i) / 8][(lane * 4 + i) % 8];
        }
        __syncthreads();
        if (warp == 0 && lane < 16) {
            const int row = row0 + lane;
            if (row <= full) {
                float s = 0.f;
#pragma unroll
                for (int h = 0; h < qsa_step::kIdxHeads; ++h) s += hw[h] * fmaxf(comb[lane][h], 0.f);
                if (row == full && (n % R) != 0) s += qsa_step::kTailBias;
                for (int c = 0; c < R; ++c) {
                    const int cell = row * R + c;
                    if (cell < n) cell_score[cell] = s;
                }
            }
        }
        __syncthreads();
    }
}

std::atomic<bool> g_enabled{false};

bool overlaps(uint64_t x, uint64_t nx, uint64_t y, uint64_t ny) { return x < y + ny && y < x + nx; }

bool validate(const ScoreArgs& a) {
    if (!a.pooled || !a.iq || !a.hw || !a.cell_score || !a.step || a.rows < 1 || a.max_cells < 1) return false;
    if (a.rows < a.max_cells / R + 1) return false;  // capacity convention max_blocks = max_cells / 4 + 1
    struct S { uint64_t p, n; bool w; };
    const S sp[] = {{(uint64_t)(uintptr_t)a.pooled, (uint64_t)a.rows * D * 4, false},
                    {(uint64_t)(uintptr_t)a.iq, 4ull * D * 4, false},
                    {(uint64_t)(uintptr_t)a.hw, 16, false},
                    {(uint64_t)(uintptr_t)a.cell_score, (uint64_t)a.max_cells * 4, true},
                    {(uint64_t)(uintptr_t)a.step, sizeof(Step), false}};
    for (const S& s : sp)
        if (s.p % 4 != 0 || s.p + s.n < s.p) return false;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if ((sp[i].w || sp[j].w) && overlaps(sp[i].p, sp[i].n, sp[j].p, sp[j].n)) return false;
    return true;
}

}  // namespace

bool native_qsa_score_enabled() { return g_enabled.load(); }
void native_qsa_score_set_enabled(bool on) { g_enabled.store(on); }

int native_qsa_score(const ScoreArgs& a, void* stream) {
    if (!g_enabled.load() || !validate(a)) return kInvalid;
    const int max_blocks = a.max_cells / R + 1;
    cudaStream_t st = static_cast<cudaStream_t>(stream);
    score_kernel<<<(max_blocks + ROWS - 1) / ROWS, WARPS * 32, 0, st>>>(a.pooled, a.iq, a.hw, a.cell_score, a.step, a.max_cells);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace qsa_dev
}  // namespace baloon
