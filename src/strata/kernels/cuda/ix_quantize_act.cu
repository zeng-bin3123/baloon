// ix_quantize_act.cu -- Activation quantization (Q8_0, Q8_0_scaled) for BI-V100
// Q8_0: per-32-element block, d16=f16(amax/127), q=rint(x/d32) in double precision.
// Three subtleties: d32 != d16; division in float64+rint; dequant uses d16 not d32.
// Q8_0_scaled: same quant but with fp32 scales for hit path parity.

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

constexpr int Q8_0_BLOCK = 32;
// Q8_0 block layout: 2 bytes (fp16 scale) + 32 bytes (int8 quants) = 34 bytes
constexpr int Q8_0_BYTES = 34;

// ─── Q8_0 quantize kernel ───────────────────────────────────────
// One warp per Q8_0 block (32 elements, one per thread in warp64 first 32 lanes).
__global__ void ix_quantize_q8_0_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ out,       // packed Q8_0 blocks
    int64_t n_elements
) {
    const int64_t block_id = (int64_t)blockIdx.x * (blockDim.x / Q8_0_BLOCK) + threadIdx.x / Q8_0_BLOCK;
    const int lane = threadIdx.x % Q8_0_BLOCK;
    const int64_t base = block_id * Q8_0_BLOCK;

    if (base >= n_elements) return;

    float val = 0.0f;
    if (base + lane < n_elements) val = x[base + lane];

    // Find amax across 32 elements via warp shuffle (within sub-warp of 32)
    float amax = val < 0 ? -val : val;
    for (int o = 16; o > 0; o >>= 1) {
        float other = ix_shfl_xor(amax, o);
        if (other > amax) amax = other;
    }
    // Lane 0 has amax
    amax = ix_shfl(amax, 0);  // broadcast

    float d32 = amax / 127.0f;
    uint16_t d16 = ix_f32_to_f16_bits(d32);

    // Quantize: rint(double(val) / double(d32))
    int8_t q = 0;
    if (d32 > 0.0f) {
        double dval = (double)val;
        double dd32 = (double)d32;
        int qi = (int)rint(dval / dd32);
        qi = qi < -128 ? -128 : (qi > 127 ? 127 : qi);
        q = (int8_t)qi;
    }

    // Write block: [d16 (2 bytes), qs[32] (32 bytes)]
    uint8_t* block_out = out + block_id * Q8_0_BYTES;
    if (lane == 0) {
        block_out[0] = (uint8_t)(d16 & 0xFF);
        block_out[1] = (uint8_t)(d16 >> 8);
    }
    block_out[2 + lane] = (uint8_t)q;
}

// ─── Q8_0 dequantize kernel ─────────────────────────────────────
__global__ void ix_dequant_q8_0_kernel(
    const uint8_t* __restrict__ blocks,
    float* __restrict__ out,
    int64_t n_elements
) {
    const int64_t block_id = (int64_t)blockIdx.x * (blockDim.x / Q8_0_BLOCK) + threadIdx.x / Q8_0_BLOCK;
    const int lane = threadIdx.x % Q8_0_BLOCK;
    const int64_t base = block_id * Q8_0_BLOCK;

    if (base >= n_elements) return;

    const uint8_t* blk = blocks + block_id * Q8_0_BYTES;
    uint16_t d16 = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    float d = ix_f16_to_f32_bits(d16);
    int8_t q = (int8_t)blk[2 + lane];

    if (base + lane < n_elements) {
        out[base + lane] = (float)q * d;
    }
}

// ─── Q8_0_scaled: same quant but fp32 scale for hit path ─────────
// Layout: 4 bytes (fp32 scale) + 32 bytes (int8 quants) = 36 bytes
constexpr int Q8_0S_BYTES = 36;

__global__ void ix_quantize_q8_0_scaled_kernel(
    const float* __restrict__ x,
    uint8_t* __restrict__ out,
    float* __restrict__ scales,      // [n_blocks] fp32 scales
    int64_t n_elements
) {
    const int64_t block_id = (int64_t)blockIdx.x * (blockDim.x / Q8_0_BLOCK) + threadIdx.x / Q8_0_BLOCK;
    const int lane = threadIdx.x % Q8_0_BLOCK;
    const int64_t base = block_id * Q8_0_BLOCK;

    if (base >= n_elements) return;

    float val = 0.0f;
    if (base + lane < n_elements) val = x[base + lane];

    float amax = val < 0 ? -val : val;
    for (int o = 16; o > 0; o >>= 1) {
        float other = ix_shfl_xor(amax, o);
        if (other > amax) amax = other;
    }
    amax = ix_shfl(amax, 0);

    float d32 = amax / 127.0f;

    int8_t q = 0;
    if (d32 > 0.0f) {
        int qi = (int)roundf(val / d32);
        qi = qi < -128 ? -128 : (qi > 127 ? 127 : qi);
        q = (int8_t)qi;
    }

    uint8_t* block_out = out + block_id * Q8_0_BLOCK;
    block_out[lane] = (uint8_t)q;
    if (lane == 0 && scales) {
        scales[block_id] = d32;
    }
}

} // anon

extern "C" {
void ix_quantize_q8_0(const float* x, uint8_t* out, int64_t n,
                      cudaStream_t stream) {
    int64_t n_blocks = (n + Q8_0_BLOCK - 1) / Q8_0_BLOCK;
    // 2 blocks per warp (64 threads / 32 = 2)
    int threads = 256;
    int blocks_per_tb = threads / Q8_0_BLOCK;
    int grid = (int)((n_blocks + blocks_per_tb - 1) / blocks_per_tb);
    ix_quantize_q8_0_kernel<<<grid, threads, 0, stream>>>(x, out, n);
}

void ix_dequant_q8_0(const uint8_t* blocks, float* out, int64_t n,
                     cudaStream_t stream) {
    int64_t n_blocks = (n + Q8_0_BLOCK - 1) / Q8_0_BLOCK;
    int threads = 256;
    int blocks_per_tb = threads / Q8_0_BLOCK;
    int grid = (int)((n_blocks + blocks_per_tb - 1) / blocks_per_tb);
    ix_dequant_q8_0_kernel<<<grid, threads, 0, stream>>>(blocks, out, n);
}

void ix_quantize_q8_0_scaled(const float* x, uint8_t* out, float* scales,
                             int64_t n, cudaStream_t stream) {
    int64_t n_blocks = (n + Q8_0_BLOCK - 1) / Q8_0_BLOCK;
    int threads = 256;
    int blocks_per_tb = threads / Q8_0_BLOCK;
    int grid = (int)((n_blocks + blocks_per_tb - 1) / blocks_per_tb);
    ix_quantize_q8_0_scaled_kernel<<<grid, threads, 0, stream>>>(x, out, scales, n);
}
} // extern C
}} // ns
