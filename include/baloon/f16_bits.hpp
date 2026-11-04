// f16_bits.hpp - bit-exact fp16 / bf16 conversion usable from host and device code.
//  f32 -> f16 : round to nearest, ties to even on the 13 dropped mantissa bits (the carry may bump the exponent);
//               f32 inf/NaN stays inf/NaN (a NaN keeps a non-zero mantissa), a FINITE value too large for fp16
//               saturates to a signed infinity (never NaN); tiny values go through the subnormal path.
//  f16 -> f32 : subnormals are normalised by shifting left until bit 10 is set, the f32 exponent field is 113 - shifts.
//  bf16       : add 0x7FFF + lsb of the kept half and truncate (round-to-nearest-even on the upper 16 bits).
#pragma once
#include <cstdint>

#if defined(__CUDACC__) || defined(__CUDA__) || defined(__HIPCC__)
#define BALOON_BITS_HD __host__ __device__ inline
#else
#define BALOON_BITS_HD inline
#endif

namespace baloon {
namespace f16bits {

BALOON_BITS_HD uint32_t bits_of(float f) {
    union { float f; uint32_t u; } c;
    c.f = f;
    return c.u;
}
BALOON_BITS_HD float float_of(uint32_t u) {
    union { float f; uint32_t u; } c;
    c.u = u;
    return c.f;
}

// Round `h` up by one when the dropped remainder `rem` (of `half` * 2 possible values) is above the tie, or at the tie with odd h.
BALOON_BITS_HD uint32_t rne_step(uint32_t h, uint32_t rem, uint32_t half) {
    return (rem > half || (rem == half && (h & 1u))) ? h + 1u : h;
}

BALOON_BITS_HD uint16_t f16_from_f32(float f) {
    const uint32_t u = bits_of(f);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int rawexp = (int)((u >> 23) & 0xFFu);
    const uint32_t man = u & 0x7FFFFFu;
    if (rawexp == 255) {  // inf / NaN: classified on the RAW exponent, never on the re-biased one
        return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));
    }
    const int e = rawexp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);  // finite overflow saturates to inf
    if (e <= 0) {                                      // subnormal half (or zero)
        const int shift = 14 - e;                      // 13 dropped bits + (1 - e) denormalisation shifts
        if (shift > 25) return (uint16_t)sign;
        const uint32_t m = man | 0x800000u;            // implicit one (f32 zero/subnormal has rawexp 0 -> e very negative)
        if (rawexp == 0) return (uint16_t)sign;
        const uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        return (uint16_t)(sign | rne_step(h, rem, 1u << (shift - 1)));
    }
    const uint32_t h = ((uint32_t)e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    return (uint16_t)(sign | rne_step(h, rem, 0x1000u));
}

BALOON_BITS_HD float f32_from_f16(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t ex = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0) {
        if (m == 0) return float_of(sign);
        uint32_t s = 0;
        while (!(m & 0x400u)) { m <<= 1; ++s; }  // at most 10 iterations for a non-zero mantissa
        m &= 0x3FFu;
        return float_of(sign | ((113u - s) << 23) | (m << 13));
    }
    if (ex == 31) return float_of(sign | 0x7F800000u | (m << 13));
    return float_of(sign | ((ex + 112u) << 23) | (m << 13));  // unsigned: ex - 15 + 127 without Nat truncation
}

// bf16: upper 16 bits with round-to-nearest-even; NaN payloads are forced quiet so they cannot round into inf.
BALOON_BITS_HD uint16_t bf16_bits_from_f32(float f) {
    const uint32_t u = bits_of(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
BALOON_BITS_HD float f32_from_bf16(uint16_t b) { return float_of((uint32_t)b << 16); }

}  // namespace f16bits
}  // namespace baloon
