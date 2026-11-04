// act_quant_kernels.hpp - host API of the activation quantizers/dequantizers on the GPU (src/quantize_act.cu).
// Pure C++ (no CUDA headers). Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t.
//  Q8_0 block (34 B): fp16 d = amax/127 (little endian) + int8 q[32]; q = round-half-away(x / d), |q| <= 127.
//  Q8_K block (292 B): float d (= -max/127 for the signed max-magnitude element), int8 q[256] with
//     q = min(127, nearest_even(-127/max * x)), int16 bsums[16] = sums of each 16 consecutive codes.
// n must be a whole number of blocks (32 resp. 256 values). All pointers are device pointers.
#pragma once
#include <cstdint>

namespace baloon {
namespace actq {

constexpr int kBlockQ8_0Bytes = 34;
constexpr int kBlockQ8KBytes = 292;

int quantize_q8_0(const float* x, uint8_t* out, int64_t n, void* stream);
// Values are multiplied by `scale` (finite) before quantizing.
int quantize_q8_0_scaled(const float* x, float scale, uint8_t* out, int64_t n, void* stream);
int quantize_q8_K(const float* x, uint8_t* out, int64_t n, void* stream);
int dequant_q8_0(const uint8_t* in, float* out, int64_t n, void* stream);
int dequant_q8_K(const uint8_t* in, float* out, int64_t n, void* stream);

}  // namespace actq
}  // namespace baloon
