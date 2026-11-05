// dequant_rows.hpp - whole-row widening of the unquantised tensor types (F32, F16, BF16) and the IQ4_NL block decoder.
// Complements ggml_dequant_layouts.hpp (K-quants, Q2_0/Q4_0/Q5_0, IQ4_XS) and canonical_form_roundtrip.hpp (Q8_0).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_bits.hpp"
#include "baloon/ggml_dequant_layouts.hpp"

// Nonzero when the dequantiser tool (src/artifact/dequant.cpp) must not provide its own main() (tests that include the
// tool as a library define it to 1 first).
#ifndef BALOON_DEQUANT_MAIN_DISABLED
#define BALOON_DEQUANT_MAIN_DISABLED 0
#endif

namespace baloon {
namespace dequant_rows {

inline uint16_t le_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// bf16 bits -> f32 is exact: the bits are the upper half of the f32 pattern.
inline float widen_bf16(uint16_t h) { return f16bits::as_f32((uint32_t)h << 16); }
inline float widen_fp16(uint16_t h) { return f16bits::widen_f16_bits(h); }

inline void dequantize_f32_row(const uint8_t* src, size_t n, float* y) { std::memcpy(y, src, n * 4); }
inline void dequantize_f16_row(const uint8_t* src, size_t n, float* y) {
    for (size_t i = 0; i < n; ++i) y[i] = widen_fp16(le_u16(src + 2 * i));
}
inline void dequantize_bf16_row(const uint8_t* src, size_t n, float* y) {
    for (size_t i = 0; i < n; ++i) y[i] = widen_bf16(le_u16(src + 2 * i));
}

// IQ4_NL block: fp16 d, 16 bytes of nibbles; y[j] = d*kv[lo(j)], y[j+16] = d*kv[hi(j)].
constexpr size_t kIq4NlBytes = 18, kIq4NlElems = 32;
inline void dequantize_iq4_nl_block(const uint8_t* b, float* y) {
    const float d = widen_fp16(le_u16(b));
    for (int j = 0; j < 16; ++j) {
        y[j] = d * (float)dequant::kvalues_iq4nl[b[2 + j] & 0xf];
        y[j + 16] = d * (float)dequant::kvalues_iq4nl[b[2 + j] >> 4];
    }
}

}  // namespace dequant_rows
}  // namespace baloon
