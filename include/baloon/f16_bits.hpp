// f16_bits.hpp - integer-level fp32 -> fp16 / bf16 rounding and fp16 -> fp32 widening (host + device).
//   * rne_round / rne_step: round-to-nearest, ties to even, on a dropped remainder (13 bits for fp16 normals).
//   * round_f32_to_f16_bits: inf/NaN (raw exponent 255) stay inf/NaN, FINITE overflow saturates to inf (never NaN),
//     subnormal results are rounded on the wider remainder, the carry may bump the exponent.
//   * widen_f16_bits: subnormals are normalised by left shifts until bit 10 is set, exponent field 113 - shifts;
//     the normal exponent re-bias ex - 15 + 127 is evaluated as a signed quantity.
//   * round_f32_to_bf16_bits: add 0x7FFF + lsb and truncate (ties to even on the upper 16 bits).
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define BALOON_BITS_HD __host__ __device__
#else
#define BALOON_BITS_HD
#endif
// REF_HD: the host+device qualifier of the reference layout (same meaning as BALOON_BITS_HD).
#ifndef REF_HD
#define REF_HD BALOON_BITS_HD
#endif

namespace baloon {
namespace f16bits {

// Generic RNE: q is the truncated value, rem the dropped remainder, half the tie point (1 << (dropped_bits - 1)).
BALOON_BITS_HD constexpr uint32_t rne_round(uint32_t q, uint32_t rem, uint32_t half) {
    return (rem > half || (rem == half && (q & 1u) == 1u)) ? q + 1u : q;
}
// One rounding step of the fp16 normal path: rem is the dropped 13-bit remainder, h the truncated half bits.
BALOON_BITS_HD constexpr uint32_t rne_step(uint32_t h, uint32_t rem) { return rne_round(h, rem, 4096u); }

BALOON_BITS_HD inline uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
BALOON_BITS_HD inline float float_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

BALOON_BITS_HD inline uint16_t f16_from_f32(float f) {
    const uint32_t x = bits_of(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 255u) return (uint16_t)(sign + 31744u + (man != 0u ? 512u : 0u));   // inf stays inf, NaN stays NaN
    const int e = (int)rawexp - 127 + 15;                                             // half exponent field (signed)
    if (e >= 31) return (uint16_t)(sign + 31744u);                                    // finite overflow -> inf
    if (e >= 1) return (uint16_t)rne_step(sign + (uint32_t)e * 1024u + (man >> 13), man & 8191u);
    if (e < -10) return (uint16_t)sign;                                               // below half the smallest subnormal
    const uint32_t full = man | 0x800000u;
    const uint32_t sh = (uint32_t)(14 - e);                                           // 14..24
    const uint32_t q = full >> sh;
    const uint32_t rem = full & ((1u << sh) - 1u);
    return (uint16_t)(sign + rne_round(q, rem, 1u << (sh - 1u)));
}

BALOON_BITS_HD inline float f32_from_f16(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const int ex = (int)((h >> 10) & 0x1Fu);
    uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0) {
        if (m == 0u) return float_of(sign);
        uint32_t s = 0;
        while (m < 1024u) { m *= 2u; ++s; }                                           // at most 10 shifts
        return float_of(sign | ((113u - s) << 23) | ((m & 0x3FFu) << 13));
    }
    if (ex == 31) return float_of(sign | 0x7F800000u | (m << 13));
    return float_of(sign | (uint32_t)((ex - 15 + 127) << 23) | (m << 13));           // signed exponent arithmetic
}

// (i + ((i >> 16) & 1) + 0x7FFF) >> 16, with NaN kept NaN (quiet bit forced) instead of carrying into the exponent.
BALOON_BITS_HD inline uint16_t round_f32_to_bf16_bits(float f) {
    const uint32_t i = bits_of(f);
    if ((i & 0x7F800000u) == 0x7F800000u && (i & 0x7FFFFFu) != 0u) return (uint16_t)((i >> 16) | 0x40u);
    return (uint16_t)((i + ((i >> 16) & 1u) + 0x7FFFu) >> 16);
}
BALOON_BITS_HD inline float bf16_bits_to_f32(uint16_t h) { return float_of((uint32_t)h << 16); }

}  // namespace f16bits
}  // namespace baloon
