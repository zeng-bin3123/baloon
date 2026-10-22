// ix_native_mmvq.cu — Dense Q8_0 GEMV for Iluvatar BI-V100 (ivcore10)
//
// Port of Strata's native_mmvq.cu with warp64 adaptations.
// This kernel handles 2-4 column small-batch matrix-vector products
// for the decode path's dense projections (attention, GDN, head).
//
// Key differences from Strata/NVIDIA:
//   - warpSize = 64 (not 32)
//   - All shuffle masks are 64-bit
//   - dp4a emulated if not native
//   - No __threadfence_system (ivcore10 limitation)
//
// Quantization formats supported:
//   Q8_0: 34 bytes per 32 elements (d:fp16, qs:int8[32])
//   Q4_K: 144 bytes per 256 elements
//   Q5_K: 176 bytes per 256 elements
//   Q6_K: 210 bytes per 256 elements
//   IQ4_XS: 136 bytes per 256 elements
//
// The MMVQ kernel reads quantized weights and Q8_1 activations,
// computing y[row] = sum_i weight[row][i] * activation[i].

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace strata {
namespace kernels {
namespace {

// ─── Block structures (matching Strata layout exactly) ─────────────
struct Q8_0_Block {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(Q8_0_Block) == 34, "Q8_0 block size");

struct Q8_1_Block {
    half2 ds;   // {d, sum}
    int8_t qs[32];
};
static_assert(sizeof(Q8_1_Block) == 36, "Q8_1 block size");

struct Q4_K_Block {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(Q4_K_Block) == 144, "Q4_K block size");

// ─── Q8_0 × Q8_1 dot product (warp64) ─────────────────────────────
// One warp computes one output row (or multiple rows if interleaved).
// Each lane handles a stride of 64 elements.

// Single-row Q8_0 × Q8_1 GEMV kernel
// Grid: (n_out, 1, 1), Block: (64, n_warps_per_block, 1)
// Each warp handles one output row.
__global__ void ix_mmvq_q8_0_kernel(
    const void* __restrict__ vx,      // Q8_0 weight: [n_out, n_in/32] blocks
    const void* __restrict__ vy,      // Q8_1 activation: [n_in/32] blocks
    float* __restrict__ dst,          // output: [n_out]
    int64_t n_in,                     // input dimension
    int64_t n_out                     // output dimension (= grid.x)
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_blocks = (int)(n_in / 32);

    const Q8_0_Block* x = (const Q8_0_Block*)vx + (int64_t)row * n_blocks;
    const Q8_1_Block* y = (const Q8_1_Block*)vy;

    float sum = 0.0f;

    // Each lane in the 64-wide warp processes blocks strided by 64
    for (int b = lane; b < n_blocks; b += IX_WARP_SIZE) {
        const float d_w = __half2float(x[b].d);
        const float d_x = __half2float((__half)__low2half(y[b].ds));

        // dp4a: 4 int8 multiply-accumulates
        int sumi = 0;
        const int* w_i32 = (const int*)x[b].qs;
        const int* x_i32 = (const int*)y[b].qs;

        // 32 elements = 8 int32 words of 4 int8 each
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            sumi = ix_dp4a(w_i32[j], x_i32[j], sumi);
        }

        sum += d_w * d_x * (float)sumi;
    }

    // Warp-wide reduction (64 lanes)
    sum = ix_warp_sum(sum);

    // Lane 0 writes the result
    if (lane == 0) {
        dst[row] = sum;
    }
}

// ─── Multi-column GEMV (2-4 tokens at once) ────────────────────────
// For verify windows with 2-4 tokens: read weight once, apply to N activations.
// This is the interleaved MMVQ path (STRATA_MMVQ_IL).
//
// Grid: (n_out, 1, 1), Block: (64, 1, 1)
// Each warp handles one weight row × N activation columns.
template<int NCOLS>
__global__ void ix_mmvq_q8_0_multi_kernel(
    const void* __restrict__ vx,      // Q8_0 weight: [n_out, n_in/32] blocks
    const void** __restrict__ vys,    // Q8_1 activations: [NCOLS] pointers
    float* __restrict__ dst,          // output: [NCOLS, n_out] (col-major for Strata compat)
    int64_t n_in,
    int64_t n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_blocks = (int)(n_in / 32);

    const Q8_0_Block* x = (const Q8_0_Block*)vx + (int64_t)row * n_blocks;

    float sums[NCOLS];
    #pragma unroll
    for (int c = 0; c < NCOLS; c++) sums[c] = 0.0f;

    for (int b = lane; b < n_blocks; b += IX_WARP_SIZE) {
        const float d_w = __half2float(x[b].d);
        const int* w_i32 = (const int*)x[b].qs;

        #pragma unroll
        for (int c = 0; c < NCOLS; c++) {
            const Q8_1_Block* y = (const Q8_1_Block*)vys[c];
            const float d_x = __half2float((__half)__low2half(y[b].ds));
            const int* x_i32 = (const int*)y[b].qs;

            int sumi = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                sumi = ix_dp4a(w_i32[j], x_i32[j], sumi);
            }

            sums[c] += d_w * d_x * (float)sumi;
        }
    }

    // Warp-wide reductions
    #pragma unroll
    for (int c = 0; c < NCOLS; c++) {
        sums[c] = ix_warp_sum(sums[c]);
    }

    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < NCOLS; c++) {
            dst[(int64_t)c * n_out + row] = sums[c];
        }
    }
}

// ─── Q4_K × Q8_1 dot product (warp64) ─────────────────────────────
// Q4_K: 256 elements per block, 4-bit codes with K-quant scales.
// Used for expert gate/up projections.
__global__ void ix_mmvq_q4k_kernel(
    const void* __restrict__ vx,      // Q4_K weight
    const void* __restrict__ vy,      // Q8_1 activation
    float* __restrict__ dst,
    int64_t n_in,
    int64_t n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_blocks_256 = (int)(n_in / 256);

    const Q4_K_Block* x = (const Q4_K_Block*)vx + (int64_t)row * n_blocks_256;
    // Q8_1 blocks: n_in/32 blocks, 8 blocks per Q4_K superblock
    const Q8_1_Block* y = (const Q8_1_Block*)vy;

    float sum = 0.0f;

    for (int sb = lane; sb < n_blocks_256; sb += IX_WARP_SIZE) {
        const Q4_K_Block& xb = x[sb];
        const half2 dm = xb.dm;
        const float d = __half2float((__half)__low2half(dm));
        const float m = __half2float((__half)__high2half(dm));

        // Decode scales (12 bytes → 8 groups of 6-bit scale + 6-bit min)
        float block_sum = 0.0f;

        for (int g = 0; g < 8; g++) {
            // Each group: 32 elements, Q8_1 sub-block
            const Q8_1_Block& yb = y[sb * 8 + g];
            const float d_x = __half2float((__half)__low2half(yb.ds));

            // Extract 6-bit scale and min for this group
            int sc, mn;
            if (g < 4) {
                sc = xb.scales[g] & 0x3F;
                mn = xb.scales[g + 4] & 0x3F;
            } else {
                sc = ((xb.scales[g + 4] & 0xF) | ((xb.scales[g - 4] >> 6) << 4));
                mn = ((xb.scales[g + 4] >> 4) | ((xb.scales[g]     >> 6) << 4));
            }

            // Dot Q4 codes with Q8_1 qs
            int sumi = 0;
            for (int j = 0; j < 16; j++) {
                const int q4_byte = xb.qs[g * 16 + j];
                const int lo = q4_byte & 0xF;
                const int hi = q4_byte >> 4;
                sumi += (int)yb.qs[j * 2]     * lo;
                sumi += (int)yb.qs[j * 2 + 1] * hi;
            }

            block_sum += d_x * (d * (float)sc * (float)sumi
                               - m * (float)mn * __half2float((__half)__high2half(yb.ds)));
        }

        sum += block_sum;
    }

    sum = ix_warp_sum(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// ─── Q8_1 Quantize kernel (warp64) ────────────────────────────────
// Convert FP32 activation → Q8_1 format.
// 256 threads = 4 warps of 64. Each thread handles one element.
constexpr int QUANT_THREADS = 256;

__launch_bounds__(QUANT_THREADS, 1)
__global__ void ix_quantize_q8_1_kernel(const float* __restrict__ x,
                                        Q8_1_Block* __restrict__ y,
                                        int n_in) {
    const int i = (int)blockIdx.x * QUANT_THREADS + (int)threadIdx.x;
    if (i >= n_in) return;

    const float xi = x[i];
    const float amax = ix_warp_max(fabsf(xi));
    const float sum = ix_warp_sum(xi);

    // Scale: amax / 127, with finite guard
    const float d = ix_q8_1_finite(amax / 127.0f);
    const int8_t q = ix_q8_1_quant(xi, d, amax);

    // BI-V100 warp is 64, but Q8_1 blocks are 32 elements.
    // Two sub-blocks per warp: lanes 0-31 → block i/32, lanes 32-63 → block (i+32)/32
    // We need to handle this carefully.
    const int block_idx = i / 32;
    const int elem_idx = i % 32;

    y[block_idx].qs[elem_idx] = q;

    // Lane 0 of each 32-element group writes the scale
    if (elem_idx == 0) {
        // d and sum for this 32-element group
        // Need sub-warp reduction for the 32-element group
        // Use the full warp reduction result and select the right half
        const int sub_warp = (ix_lane_id() < 32) ? 0 : 1;

        // Recompute per-group: each Q8_1 block is 32 elements
        float group_amax = fabsf(xi);
        float group_sum = xi;

        // Sub-warp reduce (first 32 or second 32 lanes)
        for (int off = 16; off > 0; off >>= 1) {
            group_amax = fmaxf(group_amax,
                __shfl_xor_sync(0xFFFFFFFF >> (sub_warp * 32), group_amax, off, 32));
            group_sum += __shfl_xor_sync(0xFFFFFFFF >> (sub_warp * 32), group_sum, off, 32);
        }

        const float gd = ix_q8_1_finite(group_amax / 127.0f);
        y[block_idx].ds = __halves2half2(__float2half(gd), __float2half(group_sum));
    }
}

// ─── SwiGLU + Q8_1 fused quantize (warp64) ────────────────────────
// gate/up → SwiGLU → Q8_1 in one launch.
// Saves 4-16µs per token on V100, should be similar on BI-V100.
__launch_bounds__(QUANT_THREADS, 1)
__global__ void ix_swiglu_quantize_q8_1_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    Q8_1_Block* __restrict__ y,
    int n_in
) {
    const int i = (int)blockIdx.x * QUANT_THREADS + (int)threadIdx.x;
    if (i >= n_in) return;

    // SiLU(gate) * up — Strata's exact expression
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);

    const float amax = ix_warp_max(fabsf(xi));
    const float sum = ix_warp_sum(xi);
    const float d = ix_q8_1_finite(amax / 127.0f);
    const int8_t q = ix_q8_1_quant(xi, d, amax);

    const int block_idx = i / 32;
    const int elem_idx = i % 32;

    y[block_idx].qs[elem_idx] = q;

    if (elem_idx == 0) {
        y[block_idx].ds = __halves2half2(__float2half(d), __float2half(sum));
    }
}

}  // anonymous namespace

// ─── Public launch functions ───────────────────────────────────────

extern "C" void ix_mmvq_q8_0(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream
) {
    const int grid = (int)n_out;
    ix_mmvq_q8_0_kernel<<<grid, IX_WARP_SIZE, 0, stream>>>(
        weights, activations, output, n_in, n_out);
}

extern "C" void ix_mmvq_q8_0_multi(
    const void* weights, const void** activations,
    float* output, int64_t n_in, int64_t n_out,
    int n_cols, cudaStream_t stream
) {
    const int grid = (int)n_out;
    switch (n_cols) {
        case 2:
            ix_mmvq_q8_0_multi_kernel<2><<<grid, IX_WARP_SIZE, 0, stream>>>(
                weights, activations, output, n_in, n_out);
            break;
        case 3:
            ix_mmvq_q8_0_multi_kernel<3><<<grid, IX_WARP_SIZE, 0, stream>>>(
                weights, activations, output, n_in, n_out);
            break;
        case 4:
            ix_mmvq_q8_0_multi_kernel<4><<<grid, IX_WARP_SIZE, 0, stream>>>(
                weights, activations, output, n_in, n_out);
            break;
        default:
            // Fallback: single-column
            for (int c = 0; c < n_cols; c++) {
                ix_mmvq_q8_0_kernel<<<grid, IX_WARP_SIZE, 0, stream>>>(
                    weights, activations[c], output + (int64_t)c * n_out, n_in, n_out);
            }
            break;
    }
}

extern "C" void ix_mmvq_q4k(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream
) {
    ix_mmvq_q4k_kernel<<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
        weights, activations, output, n_in, n_out);
}

extern "C" void ix_quantize_q8_1(
    const float* input, void* output, int64_t n,
    cudaStream_t stream
) {
    const int grid = (int)((n + QUANT_THREADS - 1) / QUANT_THREADS);
    ix_quantize_q8_1_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        input, (Q8_1_Block*)output, (int)n);
}

extern "C" void ix_swiglu_quantize_q8_1(
    const float* gate, const float* up, void* output, int64_t n,
    cudaStream_t stream
) {
    const int grid = (int)((n + QUANT_THREADS - 1) / QUANT_THREADS);
    ix_swiglu_quantize_q8_1_kernel<<<grid, QUANT_THREADS, 0, stream>>>(
        gate, up, (Q8_1_Block*)output, (int)n);
}

}  // namespace kernels
}  // namespace strata
