// ix_s2_gemv_q8.cu — S2 GEMV with Q8_0 activation format for BI-V100
//
// Specialized 2-bit quantized weight × Q8_0 activation GEMV.
//
// Unlike the generic s_gemv which takes FP16 activations, this kernel
// operates directly on Q8_0 quantized activations, avoiding the
// dequant→fp16→gemv pipeline and saving bandwidth.
//
// Q8_0 block: 34 bytes per 32 elements
//   { d: fp16, qs: int8[32] }
//   x_dequant[i] = d * qs[i]
//
// Weight: 2-bit affine codes
//   w[i] = (code(i) + code_bias) * scale[i/g]
//   code(i): 2 bits, packed 4 per byte
//   No offset (S2 has none).
//
// Optimization: For 2-bit codes, each byte holds 4 codes.
// We process 4 elements at a time, extracting all 4 codes from one byte,
// and use dp4a-like accumulation in int32 before final float conversion.
//
// Target: warp64

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace strata {
namespace kernels {
namespace {

// Q8_0 block layout (same as in ix_native_mmvq.cu)
struct S2Q8Block {
    half d;
    int8_t qs[32];
};
static_assert(sizeof(S2Q8Block) == 34, "Q8_0 block size");

// ═══════════════════════════════════════════════════════════════
// S2 × Q8_0 GEMV kernel (warp64)
// ═══════════════════════════════════════════════════════════════
//
// Grid: (n_out,)  Block: (64,)
// One warp per output row.
//
// Inner loop processes 32 elements at a time (one Q8_0 block).
// Within each block, codes are packed: 8 bytes hold 32 2-bit codes.
// We accumulate integer dot products for numerical precision,
// then convert at the block boundary.

__global__ void ix_s2_gemv_q8_kernel(
    const uint8_t*  __restrict__ x_q8,    // Q8_0 activation: n_in/32 blocks
    const uint8_t*  __restrict__ codes,   // 2-bit weight codes: n_out × n_in/4 bytes
    const float*    __restrict__ scales,  // per-group scales
    float*          __restrict__ y,       // output: [n_out]
    int64_t n_in,
    int64_t n_out,
    int     code_bias,
    int     group_elems
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();

    // Layout
    const int64_t code_bytes_per_row = n_in / 4;  // 2 bits per element, 4 per byte
    const int64_t groups_per_row = n_in / group_elems;
    const int64_t n_q8_blocks = n_in / 32;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;
    const S2Q8Block* x_blocks = (const S2Q8Block*)x_q8;

    float dot = 0.0f;

    // Each lane processes Q8_0 blocks strided by warp size
    for (int64_t blk = lane; blk < n_q8_blocks; blk += IX_WARP_SIZE) {
        const S2Q8Block& xb = x_blocks[blk];
        const float d_act = __half2float(xb.d);

        // Base element index for this block
        const int64_t base = blk * 32;

        // Process 32 elements: 8 code-bytes × 4 codes each
        // Each code byte: bits [1:0] [3:2] [5:4] [7:6]
        float block_sum = 0.0f;

        // We accumulate in float per-group
        // code_byte_offset = base / 4 = blk * 8
        const uint8_t* blk_codes = row_codes + blk * 8;

        for (int j = 0; j < 8; j++) {
            const uint8_t cb = blk_codes[j];
            const int elem_base = j * 4;

            // Extract 4 codes and compute 4 products
            for (int k = 0; k < 4; k++) {
                const int code = (cb >> (k * 2)) & 0x3;
                const float w_raw = (float)(code + code_bias);
                const int64_t elem = base + elem_base + k;
                const int64_t g = elem / group_elems;
                const float w_val = w_raw * row_sc[g];
                const float x_val = d_act * (float)xb.qs[elem_base + k];
                block_sum += x_val * w_val;
            }
        }

        dot += block_sum;
    }

    dot = ix_warp_sum(dot);

    if (lane == 0) {
        y[row] = dot;
    }
}

// ═══════════════════════════════════════════════════════════════
// S2 × Q8_0 split GEMV (multiple warps per row)
// ═══════════════════════════════════════════════════════════════
//
// Same computation as above but with inter-warp reduction.
// Useful when n_in is large relative to n_out.

__global__ void ix_s2_gemv_q8_split_kernel(
    const uint8_t*  __restrict__ x_q8,
    const uint8_t*  __restrict__ codes,
    const float*    __restrict__ scales,
    float*          __restrict__ y,
    int64_t n_in,
    int64_t n_out,
    int     code_bias,
    int     group_elems,
    int     threads_per_row
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int tid = threadIdx.x;

    const int64_t code_bytes_per_row = n_in / 4;
    const int64_t groups_per_row = n_in / group_elems;
    const int64_t n_q8_blocks = n_in / 32;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;
    const S2Q8Block* x_blocks = (const S2Q8Block*)x_q8;

    float dot = 0.0f;

    // Stride by total threads in block
    for (int64_t blk = tid; blk < n_q8_blocks; blk += threads_per_row) {
        const S2Q8Block& xb = x_blocks[blk];
        const float d_act = __half2float(xb.d);
        const int64_t base = blk * 32;
        const uint8_t* blk_codes = row_codes + blk * 8;

        float block_sum = 0.0f;
        for (int j = 0; j < 8; j++) {
            const uint8_t cb = blk_codes[j];
            const int elem_base = j * 4;
            for (int k = 0; k < 4; k++) {
                const int code = (cb >> (k * 2)) & 0x3;
                const float w_raw = (float)(code + code_bias);
                const int64_t elem = base + elem_base + k;
                const int64_t g = elem / group_elems;
                const float w_val = w_raw * row_sc[g];
                block_sum += d_act * (float)xb.qs[elem_base + k] * w_val;
            }
        }
        dot += block_sum;
    }

    // Warp-level reduce
    dot = ix_warp_sum(dot);

    // Inter-warp reduce
    const int warp_id = tid / IX_WARP_SIZE;
    const int n_warps = (threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE;

    extern __shared__ float smem[];

    if (ix_lane_id() == 0) {
        smem[warp_id] = dot;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            total += smem[w];
        }
        y[row] = total;
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Public launch functions
// ═══════════════════════════════════════════════════════════════

extern "C" void ix_s2_gemv_q8(
    const uint8_t* x_q8, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 32 != 0) return;
    ix_s2_gemv_q8_kernel<<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
        x_q8, codes, scales, y, n_in, n_out, code_bias, group_elems);
}

extern "C" void ix_s2_gemv_q8_split(
    const uint8_t* x_q8, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 32 != 0 || threads_per_row <= 0) return;
    const int block = ((threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE) * IX_WARP_SIZE;
    const int n_warps = block / IX_WARP_SIZE;
    const int smem = n_warps * (int)sizeof(float);

    ix_s2_gemv_q8_split_kernel<<<(int)n_out, block, smem, stream>>>(
        x_q8, codes, scales, y, n_in, n_out, code_bias, group_elems, block);
}

}  // namespace kernels
}  // namespace strata

