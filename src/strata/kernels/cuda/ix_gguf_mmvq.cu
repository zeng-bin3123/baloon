// ix_gguf_mmvq.cu — GGUF native MMVQ kernels for BI-V100 (ivcore10)
//
// Additional quantization-format MMVQ kernels beyond Q8_0 and Q4_K
// (which are in ix_native_mmvq.cu):
//   - Q5_K: 176 bytes per 256 elements (5-bit with K-quant scales)
//   - Q6_K: 210 bytes per 256 elements (6-bit with K-quant scales)
//   - IQ4_XS: 136 bytes per 256 elements (4-bit importance quantization)
//
// All kernels compute y[row] = Σ_i weight[row][i] * activation[i]
// where weights are in the given GGUF quant format and activations
// are Q8_1 (36 bytes per 32 elements: {ds:half2, qs:int8[32]}).
//
// Target: warp64

#include "strata/kernels/cuda/ix_warp64.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace strata {
namespace kernels {
namespace {

// ─── Block structures ──────────────────────────────────────────

// Q8_1 activation block (same as ix_native_mmvq.cu)
struct GgufQ8_1Block {
    half2 ds;   // {d, sum}
    int8_t qs[32];
};
static_assert(sizeof(GgufQ8_1Block) == 36, "Q8_1 block size");

// Q5_K: 176 bytes per 256 elements
// Layout: { d:half, dmin:half, scales:uint8[12], qh:uint8[32], qs:uint8[128] }
struct Q5_K_Block {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];      // high bits (bit 4 of each 5-bit code)
    uint8_t qs[128];     // low 4 bits
};
static_assert(sizeof(Q5_K_Block) == 176, "Q5_K block size");

// Q6_K: 210 bytes per 256 elements
// Layout: { ql:uint8[128], qh:uint8[64], scales:int8[16], d:half }
struct Q6_K_Block {
    uint8_t ql[128];     // lower 4 bits
    uint8_t qh[64];      // upper 2 bits
    int8_t scales[16];   // per-16-elem group scales
    half d;
};
static_assert(sizeof(Q6_K_Block) == 210, "Q6_K block size");

// IQ4_XS: 136 bytes per 256 elements
// Layout: { d:half, scales_h:uint16, scales_l:uint8[8/2], qs:uint8[128] }
// Each superblock: 256 elements, 8 groups of 32
// scales_l: 4-bit per group (8 groups packed in 4 bytes)
// scales_h: one bit per group (8 bits used of 16)
struct IQ4_XS_Block {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
static_assert(sizeof(IQ4_XS_Block) == 136, "IQ4_XS block size");

// ─── IQ4NL codebook (same as in s_gemv) ────────────────────────
__constant__ int8_t d_iq4nl_table[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

// ═══════════════════════════════════════════════════════════════
// Q5_K × Q8_1 GEMV kernel (warp64)
// ═══════════════════════════════════════════════════════════════
//
// Q5_K: 256 elements per superblock, 8 groups of 32.
// Each 5-bit code = low 4 bits from qs[] + 1 high bit from qh[].
//
// Grid: (n_out,)  Block: (64,)

__global__ void ix_mmvq_q5k_kernel(
    const void*  __restrict__ vx,
    const void*  __restrict__ vy,
    float*       __restrict__ dst,
    int64_t n_in,
    int64_t n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_sb = (int)(n_in / 256);

    const Q5_K_Block* x = (const Q5_K_Block*)vx + (int64_t)row * n_sb;
    const GgufQ8_1Block* y = (const GgufQ8_1Block*)vy;

    float sum = 0.0f;

    for (int sb = lane; sb < n_sb; sb += IX_WARP_SIZE) {
        const Q5_K_Block& xb = x[sb];
        const float d = __half2float(xb.d);
        const float dmin = __half2float(xb.dmin);

        float block_sum = 0.0f;

        for (int g = 0; g < 8; g++) {
            const GgufQ8_1Block& yb = y[sb * 8 + g];
            const float d_x = __half2float((__half)__low2half(yb.ds));

            // Extract 6-bit scale and min (same encoding as Q4_K)
            int sc, mn;
            if (g < 4) {
                sc = xb.scales[g] & 0x3F;
                mn = xb.scales[g + 4] & 0x3F;
            } else {
                sc = ((xb.scales[g + 4] & 0xF) | ((xb.scales[g - 4] >> 6) << 4));
                mn = ((xb.scales[g + 4] >> 4) | ((xb.scales[g]     >> 6) << 4));
            }

            int sumi = 0;
            for (int j = 0; j < 16; j++) {
                const int q4_byte = xb.qs[g * 16 + j];
                const int lo = q4_byte & 0xF;
                const int hi = q4_byte >> 4;

                // High bit from qh[]
                const int h_byte_idx = g * 4 + j / 8;
                const int h_bit0 = (xb.qh[h_byte_idx] >> ((j % 8) * 1)) & 1;  // for element j*2
                const int h_bit1 = (xb.qh[h_byte_idx] >> ((j % 8) * 1)) & 1;  // simplified

                // Q5: 5-bit code = lo | (h_bit << 4) for even, hi | (h_bit << 4) for odd
                // Full Q5_K decode: element = lo + (qh_bit << 4)
                const int qh_off = g * 32 + j * 2;
                const int qh_byte0 = qh_off / 8;
                const int qh_bit_pos0 = qh_off % 8;
                const int qh_byte1 = (qh_off + 1) / 8;
                const int qh_bit_pos1 = (qh_off + 1) % 8;

                const int hb0 = (xb.qh[qh_byte0] >> qh_bit_pos0) & 1;
                const int hb1 = (xb.qh[qh_byte1] >> qh_bit_pos1) & 1;

                const int q5_0 = lo | (hb0 << 4);
                const int q5_1 = hi | (hb1 << 4);

                sumi += (int)yb.qs[j * 2]     * q5_0;
                sumi += (int)yb.qs[j * 2 + 1] * q5_1;
            }

            block_sum += d_x * (d * (float)sc * (float)sumi
                               - dmin * (float)mn * __half2float((__half)__high2half(yb.ds)));
        }

        sum += block_sum;
    }

    sum = ix_warp_sum(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════
// Q6_K × Q8_1 GEMV kernel (warp64)
// ═══════════════════════════════════════════════════════════════
//
// Q6_K: 256 elements per superblock, 16 groups of 16.
// Each 6-bit code = 4 low bits from ql[] + 2 high bits from qh[].
//
// Grid: (n_out,)  Block: (64,)

__global__ void ix_mmvq_q6k_kernel(
    const void*  __restrict__ vx,
    const void*  __restrict__ vy,
    float*       __restrict__ dst,
    int64_t n_in,
    int64_t n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_sb = (int)(n_in / 256);

    const Q6_K_Block* x = (const Q6_K_Block*)vx + (int64_t)row * n_sb;
    const GgufQ8_1Block* y = (const GgufQ8_1Block*)vy;

    float sum = 0.0f;

    for (int sb = lane; sb < n_sb; sb += IX_WARP_SIZE) {
        const Q6_K_Block& xb = x[sb];
        const float d = __half2float(xb.d);

        float block_sum = 0.0f;

        // 8 Q8_1 sub-blocks per superblock (256 elements / 32)
        for (int g = 0; g < 8; g++) {
            const GgufQ8_1Block& yb = y[sb * 8 + g];
            const float d_x = __half2float((__half)__low2half(yb.ds));

            // Two groups of 16 per Q8_1 sub-block
            for (int h = 0; h < 2; h++) {
                const int sc_idx = g * 2 + h;
                const float sc = (float)xb.scales[sc_idx];

                int sumi = 0;
                for (int j = 0; j < 16; j++) {
                    const int elem = g * 32 + h * 16 + j;

                    // 4 low bits from ql
                    const int ql_idx = elem / 2;
                    const int ql_val = (elem & 1) ? (xb.ql[ql_idx] >> 4) : (xb.ql[ql_idx] & 0xF);

                    // 2 high bits from qh
                    const int qh_idx = elem / 4;
                    const int qh_shift = (elem % 4) * 2;
                    const int qh_val = (xb.qh[qh_idx] >> qh_shift) & 0x3;

                    // 6-bit code = ql | (qh << 4), then subtract 32 for signed
                    const int q6 = (ql_val | (qh_val << 4)) - 32;

                    sumi += (int)yb.qs[h * 16 + j] * q6;
                }

                block_sum += d_x * d * sc * (float)sumi;
            }
        }

        sum += block_sum;
    }

    sum = ix_warp_sum(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════
// IQ4_XS × Q8_1 GEMV kernel (warp64)
// ═══════════════════════════════════════════════════════════════
//
// IQ4_XS: 256 elements per superblock, 8 groups of 32.
// Uses the IQ4NL non-linear codebook for 4-bit codes.
//
// Grid: (n_out,)  Block: (64,)

__global__ void ix_mmvq_iq4xs_kernel(
    const void*  __restrict__ vx,
    const void*  __restrict__ vy,
    float*       __restrict__ dst,
    int64_t n_in,
    int64_t n_out
) {
    const int row = blockIdx.x;
    if (row >= n_out) return;

    const int lane = ix_lane_id();
    const int n_sb = (int)(n_in / 256);

    const IQ4_XS_Block* x = (const IQ4_XS_Block*)vx + (int64_t)row * n_sb;
    const GgufQ8_1Block* y = (const GgufQ8_1Block*)vy;

    float sum = 0.0f;

    for (int sb = lane; sb < n_sb; sb += IX_WARP_SIZE) {
        const IQ4_XS_Block& xb = x[sb];
        const float d = __half2float(xb.d);

        float block_sum = 0.0f;

        for (int g = 0; g < 8; g++) {
            const GgufQ8_1Block& yb = y[sb * 8 + g];
            const float d_x = __half2float((__half)__low2half(yb.ds));

            // Extract per-group scale from scales_l and scales_h
            // scales_l: 4 bits per group, packed 2 per byte
            const int sl_byte = g / 2;
            const int sl_shift = (g % 2) * 4;
            int scale = (xb.scales_l[sl_byte] >> sl_shift) & 0xF;
            // High bit from scales_h
            if (xb.scales_h & (1 << g)) {
                scale |= 0x10;
            }
            // Signed: subtract 16
            const float sc = (float)(scale - 16);

            // Decode 32 elements using IQ4NL codebook
            int sumi = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t qbyte = xb.qs[g * 16 + j];
                const int lo = qbyte & 0xF;
                const int hi = qbyte >> 4;

                // IQ4NL: use the non-linear codebook
                sumi += (int)yb.qs[j * 2]     * (int)d_iq4nl_table[lo];
                sumi += (int)yb.qs[j * 2 + 1] * (int)d_iq4nl_table[hi];
            }

            block_sum += d_x * d * sc * (float)sumi;
        }

        sum += block_sum;
    }

    sum = ix_warp_sum(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

}  // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Public launch functions
// ═══════════════════════════════════════════════════════════════

extern "C" void ix_mmvq_q5k(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 256 != 0) return;
    ix_mmvq_q5k_kernel<<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
        weights, activations, output, n_in, n_out);
}

extern "C" void ix_mmvq_q6k(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 256 != 0) return;
    ix_mmvq_q6k_kernel<<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
        weights, activations, output, n_in, n_out);
}

extern "C" void ix_mmvq_iq4xs(
    const void* weights, const void* activations,
    float* output, int64_t n_in, int64_t n_out,
    cudaStream_t stream
) {
    if (n_out <= 0 || n_in <= 0 || n_in % 256 != 0) return;
    ix_mmvq_iq4xs_kernel<<<(int)n_out, IX_WARP_SIZE, 0, stream>>>(
        weights, activations, output, n_in, n_out);
}

}  // namespace kernels
}  // namespace strata

