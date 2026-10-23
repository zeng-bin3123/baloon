// ix_s2_gemv_quads.cu — S2 GEMV quads kernel for BI-V100 (ivcore10)
//
// Optimized 2-bit (S2) GEMV kernels:
//
// s2_gemv_quads:
//   Code load amortized over 4 consecutive elements (one byte = 4 2-bit codes).
//   Activation load amortized over 4 half-precision values (64-bit word read).
//   No offset parameter (S2 has no offset).
//
// s2_gemv_fast:
//   All of s2_gemv_quads plus:
//     1. Activation vector staged in shared memory (read once from global,
//        reused across all output rows in the same block — but we use one
//        warp per row, so this matters when multiple blocks tile the output)
//     2. 256-entry constant-memory lookup table for code unpacking:
//        One byte → float4 (four decoded weight values).
//
// Target: warp64

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace strata {
namespace kernels {
namespace {

// ─── 256-entry constant-memory LUT for 2-bit code → float4 ────
// Each byte value encodes 4 2-bit codes. LUT[byte] = {cb[c0], cb[c1], cb[c2], cb[c3]}
// with Affine codebook cb[c] = c + code_bias.
// Default code_bias = -1, so values are {-1, 0, 1, 2}.
__constant__ float4 d_s2_lut[256];

// Host-side LUT initialization flag
static bool s_lut_initialized = false;
static int  s_lut_bias = -999;  // last bias used

static void init_s2_lut(int code_bias) {
    if (s_lut_initialized && s_lut_bias == code_bias) return;

    float4 host_lut[256];
    for (int b = 0; b < 256; b++) {
        int c0 = (b >> 0) & 0x3;
        int c1 = (b >> 2) & 0x3;
        int c2 = (b >> 4) & 0x3;
        int c3 = (b >> 6) & 0x3;
        host_lut[b] = make_float4(
            (float)(c0 + code_bias),
            (float)(c1 + code_bias),
            (float)(c2 + code_bias),
            (float)(c3 + code_bias)
        );
    }
    cudaMemcpyToSymbol(d_s2_lut, host_lut, sizeof(host_lut));
    s_lut_initialized = true;
    s_lut_bias = code_bias;
}

// ─── FP16 helper ───────────────────────────────────────────────
__device__ __forceinline__ float fp16u_to_f32(uint16_t h) {
    half hv;
    memcpy(&hv, &h, sizeof(half));
    return __half2float(hv);
}

// ═══════════════════════════════════════════════════════════════
// s2_gemv_quads — Amortized code and activation loading
// ═══════════════════════════════════════════════════════════════
//
// Grid: (n_out,)  Block: (64 * warps_per_row,)
// Each group of warps_per_row warps handles one output row.
// Process 4 elements per code byte, reading activations as
// a 64-bit (4×fp16) word where aligned.

__global__ void ix_s2_gemv_quads_kernel(
    const uint16_t* __restrict__ x,
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

    // n_in must be divisible by 4 for quads processing
    const int64_t n_quads = n_in / 4;
    const int64_t code_bytes_per_row = n_in / 4;  // 1 byte per 4 elements
    const int64_t groups_per_row = n_in / group_elems;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;

    float dot = 0.0f;

    // Process 4 elements (one code byte) per iteration
    for (int64_t q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t cb = row_codes[q];
        const int64_t base = q * 4;

        // Extract 4 codes
        const int c0 = (cb >> 0) & 0x3;
        const int c1 = (cb >> 2) & 0x3;
        const int c2 = (cb >> 4) & 0x3;
        const int c3 = (cb >> 6) & 0x3;

        // Decode weights (Affine codebook)
        const float w0 = (float)(c0 + code_bias);
        const float w1 = (float)(c1 + code_bias);
        const float w2 = (float)(c2 + code_bias);
        const float w3 = (float)(c3 + code_bias);

        // Load 4 activations
        const float x0 = fp16u_to_f32(x[base + 0]);
        const float x1 = fp16u_to_f32(x[base + 1]);
        const float x2 = fp16u_to_f32(x[base + 2]);
        const float x3 = fp16u_to_f32(x[base + 3]);

        // Per-group scaling — all 4 elems might span group boundary
        const int64_t g0 = (base + 0) / group_elems;
        const int64_t g1 = (base + 1) / group_elems;
        const int64_t g2 = (base + 2) / group_elems;
        const int64_t g3 = (base + 3) / group_elems;

        dot += x0 * w0 * row_sc[g0];
        dot += x1 * w1 * row_sc[g1];
        dot += x2 * w2 * row_sc[g2];
        dot += x3 * w3 * row_sc[g3];
    }

    // Warp-level reduce
    dot = ix_warp_sum(dot);

    // Inter-warp reduce if needed
    const int warp_id = tid / IX_WARP_SIZE;
    const int n_warps = (threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE;

    if (n_warps > 1) {
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
    } else {
        if (ix_lane_id() == 0) {
            y[row] = dot;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// s2_gemv_fast — Shared-memory staging + constant LUT
// ═══════════════════════════════════════════════════════════════
//
// Grid: (n_out,)  Block: (64 * warps_per_row,)
// Activation vector x is staged in shared memory first, then
// each warp reads from shared instead of global memory.
// Code bytes are decoded through the constant-memory LUT.
//
// stage_x: if true, stage activation in shared memory (the A path).
//          if false, read from global (the B path, for measurement).

__global__ void ix_s2_gemv_fast_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t*  __restrict__ codes,
    const float*    __restrict__ scales,
    float*          __restrict__ y,
    int64_t n_in,
    int64_t n_out,
    int     group_elems,
    int     threads_per_row,
    int     stage_x_flag
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int tid = threadIdx.x;
    const int64_t n_quads = n_in / 4;
    const int64_t code_bytes_per_row = n_in / 4;
    const int64_t groups_per_row = n_in / group_elems;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;

    // Shared memory layout:
    //   [0 .. n_warps-1]: inter-warp reduction scratch
    //   [n_warps .. n_warps + n_in - 1]: staged activation (if stage_x)
    extern __shared__ float smem[];
    const int n_warps = (threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE;
    float* s_reduce = smem;
    float* s_x = smem + n_warps;

    // Stage activation into shared memory
    if (stage_x_flag) {
        for (int64_t i = tid; i < n_in; i += threads_per_row) {
            s_x[i] = fp16u_to_f32(x[i]);
        }
        __syncthreads();
    }

    float dot = 0.0f;

    for (int64_t q = tid; q < n_quads; q += threads_per_row) {
        // LUT decode: one constant-memory read gives all 4 weights
        const uint8_t cb = row_codes[q];
        const float4 w4 = d_s2_lut[cb];

        const int64_t base = q * 4;

        float x0, x1, x2, x3;
        if (stage_x_flag) {
            x0 = s_x[base + 0];
            x1 = s_x[base + 1];
            x2 = s_x[base + 2];
            x3 = s_x[base + 3];
        } else {
            x0 = fp16u_to_f32(x[base + 0]);
            x1 = fp16u_to_f32(x[base + 1]);
            x2 = fp16u_to_f32(x[base + 2]);
            x3 = fp16u_to_f32(x[base + 3]);
        }

        const int64_t g0 = (base + 0) / group_elems;
        const int64_t g1 = (base + 1) / group_elems;
        const int64_t g2 = (base + 2) / group_elems;
        const int64_t g3 = (base + 3) / group_elems;

        dot += x0 * w4.x * row_sc[g0];
        dot += x1 * w4.y * row_sc[g1];
        dot += x2 * w4.z * row_sc[g2];
        dot += x3 * w4.w * row_sc[g3];
    }

    // Warp-level reduce
    dot = ix_warp_sum(dot);

    // Inter-warp reduce
    if (n_warps > 1) {
        if (ix_lane_id() == 0) {
            s_reduce[ix_warp_id()] = dot;
        }
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < n_warps; w++) {
                total += s_reduce[w];
            }
            y[row] = total;
        }
    } else {
        if (ix_lane_id() == 0) {
            y[row] = dot;
        }
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Public launch functions
// ═══════════════════════════════════════════════════════════════

extern "C" void ix_s2_gemv_quads(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 4 != 0 || threads_per_row <= 0) return;

    const int block = ((threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE) * IX_WARP_SIZE;
    const int n_warps = block / IX_WARP_SIZE;
    const int smem = n_warps * (int)sizeof(float);

    ix_s2_gemv_quads_kernel<<<(int)n_out, block, smem, stream>>>(
        x, codes, scales, y, n_in, n_out, code_bias, group_elems, block);
}

extern "C" void ix_s2_gemv_fast(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, float* y,
    int64_t n_in, int64_t n_out,
    int code_bias, int group_elems,
    int threads_per_row, int stage_x,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 4 != 0 || threads_per_row <= 0) return;

    // Initialize the constant-memory LUT if needed
    init_s2_lut(code_bias);

    const int block = ((threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE) * IX_WARP_SIZE;
    const int n_warps = block / IX_WARP_SIZE;
    // Shared memory: reduction slots + (if staging) full activation vector
    const int smem = n_warps * (int)sizeof(float)
                   + (stage_x ? (int)(n_in * sizeof(float)) : 0);

    ix_s2_gemv_fast_kernel<<<(int)n_out, block, smem, stream>>>(
        x, codes, scales, y, n_in, n_out, group_elems, block, stage_x);
}

}  // namespace kernels
}  // namespace strata

