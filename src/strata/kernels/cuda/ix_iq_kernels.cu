// ix_iq_kernels.cu -- IQ (importance quantization) kernels for BI-V100 (ivcore10)
// IQ4_NL dequant: codebook[nibble] * scale + offset. No -8 subtraction.
// Q8_K dot product: sum_g scale_a[g] * scale_b[g] * sum_i a[i]*b[i]

#include "strata/kernels/cuda/ix_warp64.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace strata { namespace kernels { namespace {

// IQ4_NL codebook (kvalues_iq4nl from ggml)
__device__ __constant__ float c_iq4nl[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f, 0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,  1.0f
};

// IQ4_NL dequantize: each byte holds two 4-bit codes
// block layout: [scale(fp16, 2B)] [codes(n/2 bytes)]
__global__ void ix_iq4nl_dequant_kernel(
    const uint8_t* __restrict__ data,
    float* __restrict__ out,
    int64_t n
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pair = i;  // each thread handles one byte = 2 values
    const int64_t out_idx = pair * 2;
    if (out_idx >= n) return;

    // Find which block this pair belongs to (32 elements per block)
    const int64_t block_elems = 32;
    const int64_t block_bytes = 2 + block_elems / 2;  // 2 (scale) + 16 (codes) = 18
    const int64_t block_id = out_idx / block_elems;
    const int64_t in_block = out_idx % block_elems;
    const int64_t byte_in_block = in_block / 2;

    const uint8_t* blk = data + block_id * block_bytes;
    uint16_t scale_bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    float scale = ix_f16_to_f32_bits(scale_bits);

    uint8_t byte = blk[2 + byte_in_block];
    int lo = byte & 0x0F;
    int hi = (byte >> 4) & 0x0F;

    if (out_idx < n) out[out_idx] = c_iq4nl[lo] * scale;
    if (out_idx + 1 < n) out[out_idx + 1] = c_iq4nl[hi] * scale;
}

// Q8_K dot product: dot(a, b) = sum_g scale_a[g] * scale_b[g] * sum_i a[i]*b[i]
// Q8_K block: 256 values, 8 groups of 32, fp32 scale per group
// Layout: [d(fp32, 4B)] [groups: 8 x {scale(fp32,4B), qs(int8,32B)}]
// Total: 4 + 8*(4+32) = 292 bytes per block
__global__ void ix_vec_dot_q8k_kernel(
    const uint8_t* __restrict__ a_blocks,
    const uint8_t* __restrict__ b_blocks,
    float* __restrict__ result,
    int64_t n_blocks
) {
    // One thread per block for correctness. Phase 3 parallelizes.
    const int64_t bid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;

    const int Q8K_BLOCK = 292;
    const uint8_t* a = a_blocks + bid * Q8K_BLOCK;
    const uint8_t* b = b_blocks + bid * Q8K_BLOCK;

    // Global scale (d)
    float d_a, d_b;
    memcpy(&d_a, a, 4);
    memcpy(&d_b, b, 4);

    float dot = 0.0f;
    for (int g = 0; g < 8; g++) {
        const uint8_t* ga = a + 4 + g * 36;
        const uint8_t* gb = b + 4 + g * 36;

        float sa, sb;
        memcpy(&sa, ga, 4);
        memcpy(&sb, gb, 4);

        int32_t isum = 0;
        for (int i = 0; i < 32; i++) {
            isum += (int32_t)(int8_t)ga[4 + i] * (int32_t)(int8_t)gb[4 + i];
        }
        dot += sa * sb * (float)isum;
    }
    result[bid] = dot * d_a * d_b;
}

} // anon

extern "C" {
void ix_iq4nl_dequant(const uint8_t* data, float* out, int64_t n,
                      cudaStream_t stream) {
    int64_t n_pairs = (n + 1) / 2;
    int threads = 256;
    int grid = (int)((n_pairs + threads - 1) / threads);
    ix_iq4nl_dequant_kernel<<<grid, threads, 0, stream>>>(data, out, n);
}

void ix_vec_dot_q8k(const uint8_t* a, const uint8_t* b, float* result,
                    int64_t n_blocks, cudaStream_t stream) {
    int threads = 64;
    int grid = (int)((n_blocks + threads - 1) / threads);
    ix_vec_dot_q8k_kernel<<<grid, threads, 0, stream>>>(a, b, result, n_blocks);
}
} // extern C
}} // ns
