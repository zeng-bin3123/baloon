// f16_bits.hpp - integer-level fp16 <-> fp32 conversions (round-to-nearest-even) usable on host and device.
// round_f32_to_f16_bits: ties-to-even on the 13 dropped mantissa bits; inf/NaN (raw exponent 255) stay inf/NaN,
// a FINITE overflow saturates to a signed infinity (never NaN); tiny values flush through the subnormal range.
// widen_f16_bits: exact; subnormals are normalised by left shifting until bit 10 is set, f32 exponent field 113 - s.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define BALOON_BITS_HD __host__ __device__
#else
#define BALOON_BITS_HD
#endif

namespace baloon {
namespace f16bits {

BALOON_BITS_HD inline uint32_t as_u32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
BALOON_BITS_HD inline float as_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// One rounding step on the dropped remainder (13 bits): bump when above half, or at the tie with an odd result.
BALOON_BITS_HD constexpr uint32_t rne_step(uint32_t h, uint32_t rem) {
    return (rem > 4096u || (rem == 4096u && (h & 1u))) ? h + 1u : h;
}

BALOON_BITS_HD inline uint16_t round_f32_to_f16_bits(float f) {
    const uint32_t x = as_u32(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 255u) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u) | (man >> 13));  // inf / NaN
    const int32_t e = (int32_t)rawexp - 127 + 15;                  // signed: never evaluated in unsigned arithmetic
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);                // finite overflow -> inf, mantissa field zero
    if (e >= 1) return (uint16_t)(sign + rne_step((uint32_t)e * 1024u + (man >> 13), man & 0x1FFFu));
    if (e < -10) return (uint16_t)sign;
    const uint32_t full = man | 0x800000u;
    const uint32_t sh = (uint32_t)(14 - e);
    const uint32_t q = full >> sh, rem = full & ((1u << sh) - 1u), half = 1u << (sh - 1u);
    return (uint16_t)(sign | (rem > half || (rem == half && (q & 1u)) ? q + 1u : q));
}

BALOON_BITS_HD inline float widen_f16_bits(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t ex = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0u) {
        if (m == 0u) return as_f32(sign);
        uint32_t s = 0;
        while (m < 1024u && s < 10u) { m <<= 1; ++s; }             // at most 10 shifts, then bit 10 is set
        return as_f32(sign | ((113u - s) << 23) | ((m & 0x3FFu) << 13));
    }
    if (ex == 31u) return as_f32(sign | 0x7F800000u | (m << 13));
    return as_f32(sign | ((uint32_t)((int32_t)ex - 15 + 127) << 23) | (m << 13));
}

// f32 -> bf16: add 0x7FFF + lsb and keep the upper 16 bits (round-to-nearest-even on the upper half).
BALOON_BITS_HD inline uint16_t round_f32_to_bf16_bits(float f) {
    const uint32_t i = as_u32(f);
    if ((i & 0x7F800000u) == 0x7F800000u && (i & 0x7FFFFFu)) return (uint16_t)((i >> 16) | 0x40u);  // keep NaN a NaN
    return (uint16_t)((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}

}  // namespace f16bits
}  // namespace baloon
