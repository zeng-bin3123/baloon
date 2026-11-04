// act_quant_kernels.hpp - device activation quantizers / dequantizers for the GGML block formats the quantized GEMVs consume.
//   Q8_0: 32 values -> fp16 scale d = amax / 127 (2 bytes) + 32 int8 = 34 bytes, q = round(x / d) (ties away from zero).
//   Q8_K: 256 values -> float d (4 bytes) + 256 int8 + 16 int16 group sums (32 bytes) = 292 bytes. iscale = -127 / (signed
//         value of largest magnitude), q = min(127, nearest_int(iscale x)), d = 1 / iscale, bsums[j] = sum of the 16 codes of group j.
// Pure C++ declarations; the host reference implementations used by the tests are inline here.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"

namespace baloon {
namespace act_quant {

constexpr int QK8_0 = 32;
constexpr int QK_K = 256;
constexpr int Q8_0_BYTES = 34;
constexpr int Q8K_BYTES = 292;

inline int nearest_int(float f) {  // round half to even through the magic-number trick (|f| < 4194304)
    float val = f + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Host references (bit-for-bit what the kernels compute).
inline void ref_quantize_q8_0(const float* x, uint8_t* out, float scale = 1.f) {
    float amax = 0.f;
    for (int i = 0; i < QK8_0; ++i) amax = std::fmax(amax, std::fabs(x[i] * scale));
    const float d = amax / 127.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t dh = f16bits::f16_from_f32(d);
    std::memcpy(out, &dh, 2);
    for (int i = 0; i < QK8_0; ++i) out[2 + i] = (uint8_t)(int8_t)std::lround(x[i] * scale * id);
}
inline void ref_quantize_q8_K(const float* x, uint8_t* out) {
    float amax = 0.f, mx = 0.f;
    for (int i = 0; i < QK_K; ++i)
        if (std::fabs(x[i]) > amax) { amax = std::fabs(x[i]); mx = x[i]; }
    int8_t* qs = reinterpret_cast<int8_t*>(out + 4);
    int16_t bs[16] = {0};
    float d = 0.f;
    if (amax == 0.f) {
        std::memset(out, 0, Q8K_BYTES);
        return;
    }
    const float iscale = -127.f / mx;
    for (int i = 0; i < QK_K; ++i) {
        int v = nearest_int(iscale * x[i]);
        if (v > 127) v = 127;
        qs[i] = (int8_t)v;
        bs[i / 16] = (int16_t)(bs[i / 16] + v);
    }
    d = 1.f / iscale;
    std::memcpy(out, &d, 4);
    std::memcpy(out + 4 + QK_K, bs, 32);
}

}  // namespace act_quant

namespace act_dev {
constexpr int kOk = 0;
constexpr int kInvalid = -1;
// n values (multiple of 32 / 256) per row, `rows` rows; out holds rows * n / elems blocks.
int quantize_q8_0(const float* x, uint8_t* out, int rows, int n, void* stream);
int quantize_q8_0_scaled(const float* x, uint8_t* out, int rows, int n, float scale, void* stream);
int quantize_q8_K(const float* x, uint8_t* out, int rows, int n, void* stream);
int dequant_q8_0(const uint8_t* in, float* y, int rows, int n, void* stream);
int dequant_q8_K(const uint8_t* in, float* y, int rows, int n, void* stream);
}  // namespace act_dev
}  // namespace baloon
