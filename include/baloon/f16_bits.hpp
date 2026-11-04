// f16_bits.hpp - integer-level fp32 <-> fp16 conversions and fp32 -> bf16 rounding.
//  * f16_from_f32: round-to-nearest-even on the 13 dropped mantissa bits (a carry may bump the exponent, up to the
//    infinity pattern); raw exponent 255 (inf/NaN) stays inf/NaN, a FINITE overflow saturates to inf (never NaN);
//    results below half the smallest subnormal become a signed zero.
//  * f32_from_f16: exact; subnormals are normalised by shifting left until bit 10 is set, f32 exponent field 113 - s.
//    The exponent re-bias `ex - 15 + 127` is evaluated as a SIGNED quantity.
//  * bf16_from_f32: adds 0x7FFF + lsb and truncates (ties to even on the upper 16 bits); NaN keeps a payload bit.
// Host/device usable: integer arithmetic and bit casts only.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define F16B_HD __host__ __device__
#else
#define F16B_HD
#endif

namespace baloon {
namespace f16bits {

F16B_HD inline uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
F16B_HD inline float float_of(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// One rounding step: `rem` is the dropped 13-bit remainder, `h` the truncated half bits.
F16B_HD constexpr uint32_t rne_step(uint32_t h, uint32_t rem) {
    return (rem > 4096u || (rem == 4096u && (h & 1u))) ? h + 1u : h;
}

F16B_HD inline uint16_t f16_from_f32(float f) {
    const uint32_t x = bits_of(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 255u) return (uint16_t)(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));  // inf / NaN
    const int32_t e = (int32_t)rawexp - 127 + 15;                                                   // signed re-bias
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);                                                  // finite overflow -> inf
    if (e >= 1) return (uint16_t)rne_step(sign + (uint32_t)e * 1024u + (man >> 13), man & 0x1FFFu);
    if (e < -10) return (uint16_t)sign;                                                              // below half the least subnormal
    const uint32_t full = man | 0x800000u;                                                           // subnormal regime
    const uint32_t sh = (uint32_t)(14 - e);                                                          // 14..24
    uint32_t q = full >> sh;
    const uint32_t rem = full & ((1u << sh) - 1u), half = 1u << (sh - 1u);
    if (rem > half || (rem == half && (q & 1u))) ++q;                                                // may carry to min normal
    return (uint16_t)(sign | q);
}

F16B_HD inline float f32_from_f16(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const int32_t ex = (int32_t)(((uint32_t)h >> 10) & 0x1Fu);
    uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0) {
        if (m == 0) return float_of(sign);
        int32_t s = 0;
        while (m < 1024u && s < 10) { m <<= 1; ++s; }                  // bit 10 set after at most 10 shifts
        return float_of(sign | ((uint32_t)(113 - s) << 23) | ((m & 0x3FFu) << 13));
    }
    if (ex == 31) return float_of(sign | 0x7F800000u | (m << 13));
    return float_of(sign | ((uint32_t)(ex - 15 + 127) << 23) | (m << 13));
}

F16B_HD inline uint16_t bf16_from_f32(float f) {
    const uint32_t x = bits_of(f);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((x >> 16) | 0x40u);   // NaN: no carry-out, stays NaN
    return (uint16_t)((x + ((x >> 16) & 1u) + 0x7FFFu) >> 16);
}
F16B_HD inline float f32_from_bf16(uint16_t h) { return float_of((uint32_t)h << 16); }

}  // namespace f16bits
}  // namespace baloon
