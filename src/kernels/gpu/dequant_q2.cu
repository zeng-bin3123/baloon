// dequant_q2.cu - Q2_0 ("S2") block dequantization to f32. Block = fp16 scale d + 16 bytes of 2-bit codes (64 elements,
// 18 bytes); element i = d * (code_i - 1), code_i = (byte[i/4] >> 2*(i%4)) & 3, so codes {0,1,2,3} -> {-d,0,d,2d}.
#include "kg_common.cuh"
#include "baloon/isa_selfprobe.hpp"
#include "baloon/kg_kernels.hpp"

namespace baloon {
namespace kg {
namespace {
constexpr int CODES_PER_BYTE = 4;
constexpr int QK = 64;
constexpr int BLOCK_BYTES = 2 + QK / CODES_PER_BYTE;

__global__ void dequant_s2_kernel(const uint8_t* __restrict__ blocks, float* __restrict__ out, long long n_blocks) {
    const long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_blocks * QK) return;
    const long long b = e / QK;
    const int i = (int)(e % QK);
    const uint8_t* p = blocks + b * BLOCK_BYTES;
    const float d = kgc::f16_to_f((uint16_t)(p[0] | (p[1] << 8)));
    const int code = (p[2 + i / CODES_PER_BYTE] >> (2 * (i % CODES_PER_BYTE))) & 3;
    out[e] = d * (float)(code - 1);
}
}  // namespace

cudaError_t dequant_s2(const uint8_t* blocks, float* out, long long n_blocks, cudaStream_t s) {
    if (n_blocks < 0 || (n_blocks > 0 && (!blocks || !out))) return cudaErrorInvalidValue;
    if (n_blocks == 0) return cudaSuccess;
    dequant_s2_kernel<<<kgc::grid_for_n(n_blocks * QK, 256), 256, 0, s>>>(blocks, out, n_blocks);
    return cudaGetLastError();
}
}  // namespace kg
}  // namespace baloon
