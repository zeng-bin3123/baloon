// f16_bits_rounding.hpp - integer-level model of the fp16 <-> f32 and f32 -> bf16 conversions: round-to-nearest-even on
// the dropped remainder, the inf/NaN versus finite-overflow split, and the dense (non-block) dequantisers built on them.
// Host/device usable: integer arithmetic only. The scalar conversions themselves live in f16_conversion.hpp; this header
// exposes the theory's step functions so each can be tested against the LAYER 1 statements.
#pragma once
#include <cstdint>
#include "baloon/f16_conversion.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

// Macro kept for parity with the standalone dequant tool: when defined, the tool's main() is compiled out so the header
// set can be linked into a larger binary.
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_ENABLED 1
#endif

namespace baloon {
namespace f16bits {
using namespace ::baloon::f16;

constexpr uint32_t kHalfTie = 4096u;        // half of the 13 dropped mantissa bits
constexpr uint32_t kSignHalf = 0x8000u;
constexpr uint32_t kInfHalf = 0x7C00u;      // exponent field 31, mantissa 0
constexpr uint32_t kQuietBit = 0x200u;

// One rounding step: `rem` is the dropped 13-bit remainder, `h` the truncated half bits.
BALOON_HD constexpr uint32_t rne_step(uint32_t h, uint32_t rem) {
    return (rem > kHalfTie || (rem == kHalfTie && (h & 1u))) ? h + 1u : h;
}

// Encoder top level. Returns -1 for "normal or subnormal path", else the finished half bits.
// raw exponent 255 -> inf/NaN (NaN keeps a non-zero mantissa field); finite overflow -> signed inf (mantissa zero).
BALOON_HD constexpr int32_t encode_top(uint32_t sign, uint32_t rawexp, uint32_t man) {
    return rawexp == 255u ? (int32_t)(sign + kInfHalf + (man != 0u ? kQuietBit : 0u))
         : ((int32_t)rawexp - 127 + 15 >= 31) ? (int32_t)(sign + kInfHalf)
         : -1;
}

// bf16 rounding: (i + ((i >> 16) & 1) + 0x7FFF) >> 16 as 64-bit arithmetic so a carry out of bit 31 is not lost.
BALOON_HD constexpr uint32_t bf16_round(uint32_t i) {
    return (uint32_t)(((uint64_t)i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}

// Named entry points (theory form-template identifiers).
BALOON_HD inline uint16_t round_f32_to_f16_bits(float f) { return float_to_half(f); }
BALOON_HD inline float widen_f16_bits(uint16_t h) { return half_to_float(h); }
BALOON_HD inline uint16_t round_f32_to_bf16_bits(float f) {
    const uint32_t i = f32_bits(f);
    if (((i >> 23) & 0xFFu) == 255u && (i & 0x7FFFFFu) != 0u) return (uint16_t)((i >> 16) | 0x40u);  // NaN stays NaN
    return (uint16_t)bf16_round(i);
}

// ---- dense (non-block) dequantisers ----------------------------------------------------------------------------------
inline void dequantize_f32(const uint8_t* b, float* y, int n) {
    for (int i = 0; i < n; ++i) { uint32_t u = (uint32_t)b[4*i] | ((uint32_t)b[4*i+1] << 8) | ((uint32_t)b[4*i+2] << 16) | ((uint32_t)b[4*i+3] << 24); y[i] = bits_f32(u); }
}
inline void dequantize_f16_dense(const uint8_t* b, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = half_to_float(dequant::read_u16(b + 2 * i));
}
inline void dequantize_bf16_dense(const uint8_t* b, float* y, int n) {
    for (int i = 0; i < n; ++i) y[i] = bf16_to_float(dequant::read_u16(b + 2 * i));
}
// IQ4_NL block: d(2) + qs(16) -> 32 values, low nibbles then high nibbles through the non-linear table.
inline void dequantize_iq4_nl(const uint8_t* b, float* y) {
    const float d = dequant::read_f16(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = d * (float)dequant::kvalues_iq4nl[qs[j] & 0xf];
        y[j + 16] = d * (float)dequant::kvalues_iq4nl[qs[j] >> 4];
    }
}

}  // namespace f16bits
}  // namespace baloon
