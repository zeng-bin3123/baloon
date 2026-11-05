// quantize_act_kernels.hpp - device activation quantizers (src/ilu_quantize_act.cu). Pure C++ declarations.
// Q8_0 block = fp16 d + 32 int8 (34 B); Q8_K block = f32 d + 256 int8 + 16 int16 bsums (292 B).
// Return codes: 0 ok, -1 rejected (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace quant_act {
constexpr int kQ8_0Elems = 32, kQ8_0Bytes = 34;
constexpr int kQ8KElems = 256, kQ8KBytes = 292, kQ8KBsums = 16;
// ggml rule: d = amax/127, q = round-half-away(x/d), clamped to [-128, 127]. n % 32 == 0.
int quantize_q8_0(const float* x, uint8_t* out, int64_t n, void* stream);
// CPU-parity rule: x is first multiplied by `scale`; d = amax/127 (IEEE division), q = nearest-even, clamp [-127, 127].
int quantize_q8_0_scaled(const float* x, uint8_t* out, int64_t n, float scale, void* stream);
// ggml Q8_K: iscale = -127/max (signed max of largest magnitude), q = min(127, nearest_int(iscale*x)), bsums per 16.
int quantize_q8_K(const float* x, uint8_t* out, int64_t n, void* stream);
int dequant_q8_0(const uint8_t* in, float* y, int64_t n, void* stream);
int dequant_q8_K(const uint8_t* in, float* y, int64_t n, void* stream);
}  // namespace quant_act
}  // namespace baloon
