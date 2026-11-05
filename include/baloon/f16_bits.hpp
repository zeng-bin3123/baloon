// f16_bits.hpp - hand-written fp16 <-> f32 and f32 -> bf16 conversions (integer arithmetic only, host + device).
// Contracts (see the F16BitsRounding theory):
//  * f32 -> f16 rounds the 13 dropped mantissa bits to nearest, ties to even; a carry may bump the exponent.
//  * f32 inf/NaN (raw exponent 255) stays inf/NaN; a FINITE overflow saturates to inf, never NaN.
//  * f16 -> f32 normalises subnormals by shifting until bit 10 is set; the f32 exponent field is then 113 - s.
//  * f32 -> bf16 adds 0x7FFF + lsb and truncates: round-to-nearest-even on the upper 16 bits.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define BALOON_F16_HD __host__ __device__
#else
#define BALOON_F16_HD
#endif

namespace baloon {
namespace f16bits {

BALOON_F16_HD inline uint32_t as_u32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
BALOON_F16_HD inline float as_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// One rounding step on the dropped 13-bit remainder.
BALOON_F16_HD constexpr uint32_t rne_step(uint32_t h, uint32_t rem) {
    return (rem > 4096u || (rem == 4096u && (h & 1u))) ? h + 1u : h;
}

BALOON_F16_HD inline uint16_t round_f32_to_f16_bits(float f) {
    const uint32_t x = as_u32(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 255u) return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));  // inf/NaN
    const int32_t e = (int32_t)rawexp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);  // finite overflow -> inf (mantissa field zero)
    if (e >= 1) return (uint16_t)(sign + rne_step(((uint32_t)e << 10) | (man >> 13), man & 0x1FFFu));
    if (e < -10) return (uint16_t)sign;  // below half the smallest subnormal
    const uint32_t full = man | 0x800000u;
    const uint32_t sh = (uint32_t)(14 - e);
    const uint32_t q = full >> sh;
    const uint32_t rem = full & ((1u << sh) - 1u), half = 1u << (sh - 1u);
    return (uint16_t)(sign | ((rem > half || (rem == half && (q & 1u))) ? q + 1u : q));
}

BALOON_F16_HD inline float widen_f16_bits(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t ex = ((uint32_t)h >> 10) & 0x1Fu;
    const uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0) {
        if (m == 0) return as_f32(sign);
        uint32_t mm = m, s = 0;
        while (mm < 1024u) { mm <<= 1; ++s; }  // at most 10 iterations for m in [1, 1023]
        return as_f32(sign | ((113u - s) << 23) | ((mm & 0x3FFu) << 13));
    }
    if (ex == 31) return as_f32(sign | 0x7F800000u | (m << 13));
    return as_f32(sign | ((ex + 112u) << 23) | (m << 13));  // ex - 15 + 127 as a signed quantity
}

// NaN keeps a quiet payload (the add would otherwise carry NaNs into the infinity pattern).
BALOON_F16_HD inline uint16_t round_f32_to_bf16_bits(float f) {
    const uint32_t u = as_u32(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace f16bits
}  // namespace baloon
