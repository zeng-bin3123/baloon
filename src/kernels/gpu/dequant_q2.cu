// dequant_q2.cu - 2-bit dequantization to f32.
//
// dequant_s2 / dequant_s2_kernel: the CANONICAL S2 plane (S2CanonicalDequant). Codes are 2 bits, four per byte LSB-first
// (element i at byte i/4, bits (i%4)*2); one f32 scale per 64-element group. Thread b decodes block b with a per-block
// loop: code byte b*16 + j/4, shift (j%4)*2, output b*64 + j, j < 64. The bias -1 is applied to the CODE in the integer
// domain, then ONE multiply by the scale. Grid ceil(n_blocks/256) x 256; n_blocks <= 0 launches nothing.
//
// dequant_q2_0 / dequant_q2_0_kernel: the separate interleaved GGUF Q2_0 block form (fp16 scale d + 16 code bytes per
// 64 elements, 18 bytes/block), one thread per element, element i = d * (code_i - 1).
#include "kg_common.cuh"
#include "baloon/isa_selfprobe.hpp"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int CODES_PER_BYTE = 4;                         // 2-bit codes per byte
constexpr int QK = 64;                                    // elements per group / block
constexpr int CODE_BYTES = QK / CODES_PER_BYTE;           // 16 code bytes per block
constexpr int CODE_BIAS = -1;                             // integer-domain bias
constexpr int Q2_0_BLOCK_BYTES = 2 + CODE_BYTES;          // interleaved Q2_0: fp16 d + codes
constexpr int THREADS = 256;

// Canonical S2 plane kernel: one thread per 64-element block.
__global__ void dequant_s2_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                  float* __restrict__ out, long long n_blocks) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const float scale = scales[b];                        // f32 per-group scale
    const uint8_t* cb = codes + b * CODE_BYTES;           // byte b*16 + j/4
    float* o = out + b * QK;                              // output b*64 + j
#pragma unroll 4
    for (int j = 0; j < QK; ++j) {
        const int code = (cb[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 3;
        o[j] = (float)(code + CODE_BIAS) * scale;         // bias in the integer domain, then ONE multiply
    }
}

// Interleaved GGUF Q2_0 blocks (not the S2 plane): one thread per element.
__global__ void dequant_q2_0_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ out, long long n_blocks) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_blocks * QK) return;
    const long long b = e / QK;
    const int i = (int)(e % QK);
    const uint8_t* p = blocks + b * Q2_0_BLOCK_BYTES;
    const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
    const int code = (p[2 + i / CODES_PER_BYTE] >> (2 * (i % CODES_PER_BYTE))) & 3;
    out[e] = d * (float)(code + CODE_BIAS);
}
}  // namespace

cudaError_t dequant_s2(const uint8_t* codes, const float* scales, float* out, long long n_blocks, cudaStream_t s) {
    if (n_blocks <= 0) return cudaSuccess;                                   // nothing to launch
    if (!codes || !scales || !out) return cudaErrorInvalidValue;
    const unsigned grid = (unsigned)((n_blocks + THREADS - 1) / THREADS);    // ceil(n_blocks / 256) x 256
    dequant_s2_kernel<<<grid, THREADS, 0, s>>>(codes, scales, out, n_blocks);
    return cudaGetLastError();
}

cudaError_t dequant_q2_0(const uint8_t* blocks, float* out, long long n_blocks, cudaStream_t s) {
    if (n_blocks < 0 || (n_blocks > 0 && (!blocks || !out))) return cudaErrorInvalidValue;
    if (n_blocks == 0) return cudaSuccess;
    dequant_q2_0_kernel<<<kgc::grid_for_n(n_blocks * QK, THREADS), THREADS, 0, s>>>(blocks, out, n_blocks);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
