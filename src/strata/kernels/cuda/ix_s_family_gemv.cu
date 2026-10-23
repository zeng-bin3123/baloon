// ix_s_family_gemv.cu — Structured GEMV family kernels for BI-V100 (ivcore10)
//
// GPU implementations of the s_gemv kernel family:
//   - s_gemv: one thread (warp) per output row, generic code_bits {2,4,8}
//   - s_gemv_split: multiple warps per output row with shared-mem reduction
//   - s_gemv_q8k: Q8_K activation format variant
//   - s_gemv_split_async: async (stream-launched) version of split
//
// Weight decode:
//   value(i) = codebook[code(i)] * scale[i/g] + offset[i/g]
//   Codebook Affine: cb[c] = c + code_bias
//   Codebook Iq4Nl:  cb[c] = kvalues_iq4nl[c & 0xF]
//
// Target: warp64

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace strata {
namespace kernels {
namespace {

// ─── IQ4NL codebook in constant memory ─────────────────────────
__constant__ int8_t d_kIq4Nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

// ─── Code extraction helpers (device) ──────────────────────────

__device__ __forceinline__ int extract_code_2bit(const uint8_t* codes, int64_t idx) {
    return (codes[idx >> 2] >> ((idx & 3) << 1)) & 0x3;
}

__device__ __forceinline__ int extract_code_4bit(const uint8_t* codes, int64_t idx) {
    return (codes[idx >> 1] >> ((idx & 1) << 2)) & 0xF;
}

__device__ __forceinline__ int extract_code_8bit(const uint8_t* codes, int64_t idx) {
    return codes[idx];
}

// ─── Codebook decode (device) ──────────────────────────────────

// codebook_id: 0 = Affine, 1 = Iq4Nl
__device__ __forceinline__ float decode_cb(int code, int codebook_id, int code_bias) {
    if (codebook_id == 1) {
        return (float)d_kIq4Nl[code & 0xF];
    }
    return (float)(code + code_bias);
}

// ─── FP16 → FP32 (device, from raw uint16_t) ──────────────────
__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    half hv;
    memcpy(&hv, &h, sizeof(half));
    return __half2float(hv);
}

// ═══════════════════════════════════════════════════════════════
// s_gemv — one warp per output row
// ═══════════════════════════════════════════════════════════════
//
// Grid: (n_out,)  Block: (64,)
// Each warp computes y[row] = Σ_i fp16(x[i]) * W[row,i]
// where W[row,i] = cb[code(i)] * scale[i/g] + offset[i/g]

template<int CODE_BITS>
__global__ void ix_s_gemv_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t*  __restrict__ codes,
    const float*    __restrict__ scales,
    const float*    __restrict__ offset,
    float*          __restrict__ y,
    int64_t n_in,
    int64_t n_out,
    int     code_bias,
    int     codebook_id,  // 0=Affine, 1=Iq4Nl
    int     group_elems,
    int     has_offset
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int64_t code_bytes_per_row = (n_in * CODE_BITS + 7) / 8;
    const int64_t groups_per_row = n_in / group_elems;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;
    const float*   row_off   = (has_offset && offset) ? offset + (int64_t)row * groups_per_row : nullptr;

    float dot = 0.0f;

    for (int64_t i = lane; i < n_in; i += IX_WARP_SIZE) {
        int code;
        if (CODE_BITS == 2)      code = extract_code_2bit(row_codes, i);
        else if (CODE_BITS == 4) code = extract_code_4bit(row_codes, i);
        else                     code = extract_code_8bit(row_codes, i);

        const float cb_val = decode_cb(code, codebook_id, code_bias);
        const int64_t g = i / group_elems;
        float w_val = cb_val * row_sc[g];
        if (row_off) w_val += row_off[g];

        const float x_val = fp16_to_f32(x[i]);
        dot += x_val * w_val;
    }

    dot = ix_warp_sum(dot);

    if (lane == 0) {
        y[row] = dot;
    }
}

// ═══════════════════════════════════════════════════════════════
// s_gemv_split — multiple sub-warps per output row
// ═══════════════════════════════════════════════════════════════
//
// Grid: (n_out,)  Block: (64 * warps_per_row,) up to hardware limit
// Each group of warps_per_row warps handles one output row.
// Final reduction via shared memory.
//
// For warps_per_row == 1, this is identical to s_gemv (but slower
// due to shared-memory overhead).

template<int CODE_BITS>
__global__ void ix_s_gemv_split_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t*  __restrict__ codes,
    const float*    __restrict__ scales,
    const float*    __restrict__ offset,
    float*          __restrict__ y,
    int64_t n_in,
    int64_t n_out,
    int     code_bias,
    int     codebook_id,
    int     group_elems,
    int     has_offset,
    int     threads_per_row
) {
    // Thread's position within the row group
    const int tid_in_block = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int64_t code_bytes_per_row = (n_in * CODE_BITS + 7) / 8;
    const int64_t groups_per_row = n_in / group_elems;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;
    const float*   row_off   = (has_offset && offset) ? offset + (int64_t)row * groups_per_row : nullptr;

    float dot = 0.0f;

    // Each thread handles a stride of total threads_per_row
    for (int64_t i = tid_in_block; i < n_in; i += threads_per_row) {
        int code;
        if (CODE_BITS == 2)      code = extract_code_2bit(row_codes, i);
        else if (CODE_BITS == 4) code = extract_code_4bit(row_codes, i);
        else                     code = extract_code_8bit(row_codes, i);

        const float cb_val = decode_cb(code, codebook_id, code_bias);
        const int64_t g = i / group_elems;
        float w_val = cb_val * row_sc[g];
        if (row_off) w_val += row_off[g];

        const float x_val = fp16_to_f32(x[i]);
        dot += x_val * w_val;
    }

    // Warp-level reduce first
    dot = ix_warp_sum(dot);

    // Inter-warp reduce via shared memory
    const int warp_id = tid_in_block / IX_WARP_SIZE;
    const int n_warps = (threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE;

    extern __shared__ float smem[];

    if (ix_lane_id() == 0) {
        smem[warp_id] = dot;
    }
    __syncthreads();

    if (tid_in_block == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) {
            total += smem[w];
        }
        y[row] = total;
    }
}

// ═══════════════════════════════════════════════════════════════
// s_gemv_q8k — Q8_K activation format
// ═══════════════════════════════════════════════════════════════
//
// Q8_K block (292 bytes per 256 elements):
//   { d:f32, qs:int8[256], bsums:int16[16] }
//   x_dequant[i] = d * qs[i]
//
// Grid: (n_out,)  Block: (64,)

template<int CODE_BITS>
__global__ void ix_s_gemv_q8k_kernel(
    const uint8_t*  __restrict__ x_q8k,
    const uint8_t*  __restrict__ codes,
    const float*    __restrict__ scales,
    const float*    __restrict__ offset,
    float*          __restrict__ y,
    int64_t n_in,
    int64_t n_out,
    int     code_bias,
    int     codebook_id,
    int     group_elems,
    int     has_offset
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int64_t code_bytes_per_row = (n_in * CODE_BITS + 7) / 8;
    const int64_t groups_per_row = n_in / group_elems;

    const uint8_t* row_codes = codes + (int64_t)row * code_bytes_per_row;
    const float*   row_sc    = scales + (int64_t)row * groups_per_row;
    const float*   row_off   = (has_offset && offset) ? offset + (int64_t)row * groups_per_row : nullptr;

    // Q8_K constants
    constexpr int64_t Q8K_BLOCK = 256;
    constexpr int64_t Q8K_BYTES = 292;  // 4 (d:f32) + 256 (qs:int8) + 32 (bsums:int16)

    float dot = 0.0f;

    for (int64_t i = lane; i < n_in; i += IX_WARP_SIZE) {
        // Dequantize activation from Q8_K
        const int64_t blk = i / Q8K_BLOCK;
        const int64_t off_in_blk = i % Q8K_BLOCK;
        const uint8_t* block_ptr = x_q8k + blk * Q8K_BYTES;

        float d_act;
        memcpy(&d_act, block_ptr, 4);
        const int8_t qs = ((const int8_t*)(block_ptr + 4))[off_in_blk];
        const float x_val = d_act * (float)qs;

        // Decode weight
        int code;
        if (CODE_BITS == 2)      code = extract_code_2bit(row_codes, i);
        else if (CODE_BITS == 4) code = extract_code_4bit(row_codes, i);
        else                     code = extract_code_8bit(row_codes, i);

        const float cb_val = decode_cb(code, codebook_id, code_bias);
        const int64_t g = i / group_elems;
        float w_val = cb_val * row_sc[g];
        if (row_off) w_val += row_off[g];

        dot += x_val * w_val;
    }

    dot = ix_warp_sum(dot);

    if (lane == 0) {
        y[row] = dot;
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Public launch functions
// ═══════════════════════════════════════════════════════════════

extern "C" void ix_s_gemv(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0) return;
    switch (code_bits) {
        case 2:
            ix_s_gemv_kernel<2><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        case 4:
            ix_s_gemv_kernel<4><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        case 8:
            ix_s_gemv_kernel<8><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        default:
            break;
    }
}

extern "C" void ix_s_gemv_split(
    const uint16_t* x, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    int threads_per_row,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || threads_per_row <= 0) return;

    // Clamp threads_per_row to nearest multiple of warp size
    const int block_size = ((threads_per_row + IX_WARP_SIZE - 1) / IX_WARP_SIZE) * IX_WARP_SIZE;
    const int n_warps = block_size / IX_WARP_SIZE;
    const int smem = n_warps * (int)sizeof(float);

    switch (code_bits) {
        case 2:
            ix_s_gemv_split_kernel<2><<<(int)n_out, block_size, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset, block_size);
            break;
        case 4:
            ix_s_gemv_split_kernel<4><<<(int)n_out, block_size, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset, block_size);
            break;
        case 8:
            ix_s_gemv_split_kernel<8><<<(int)n_out, block_size, smem, stream>>>(
                x, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset, block_size);
            break;
        default:
            break;
    }
}

extern "C" void ix_s_gemv_q8k(
    const uint8_t* x_q8k, const uint8_t* codes,
    const float* scales, const float* offset,
    float* y, int64_t n_in, int64_t n_out,
    int code_bits, int code_bias, int codebook_id,
    int group_elems, int has_offset,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0) return;
    switch (code_bits) {
        case 2:
            ix_s_gemv_q8k_kernel<2><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        case 4:
            ix_s_gemv_q8k_kernel<4><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        case 8:
            ix_s_gemv_q8k_kernel<8><<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
                x_q8k, codes, scales, offset, y, n_in, n_out,
                code_bias, codebook_id, group_elems, has_offset);
            break;
        default:
            break;
    }
}

}  // namespace kernels
}  // namespace strata

