// f16_conversion.hpp - bit-level fp32 <-> fp16 conversions (round-to-nearest-even).
// Host/device usable: only integer arithmetic and bit casts.
#pragma once
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define BALOON_HD __host__ __device__
#else
#define BALOON_HD
#endif

namespace baloon {
namespace f16 {

// ---- pure integer helpers (mirror the contract) ------------------------------------------
// Packed sign/exponent/truncated-mantissa before rounding. s is 0 or 0x8000, e the half exponent
// field, m the 23-bit fp32 mantissa.
BALOON_HD constexpr uint32_t h0(uint32_t s, uint32_t e, uint32_t m) { return s + e * 1024u + m / 8192u; }

// Round-to-nearest-even on the 13 dropped mantissa bits. A carry may bump the exponent (up to the
// infinity pattern) but never reaches the sign bit.
BALOON_HD constexpr uint32_t round_normal(uint32_t s, uint32_t e, uint32_t m) {
    return ((m % 8192u) > 4096u || ((m % 8192u) == 4096u && (h0(s, e, m) % 2u) == 1u)) ? h0(s, e, m) + 1u
                                                                                       : h0(s, e, m);
}

// Subnormal normalisation loop: shift until bit 10 is set; `fuel` bounds the iterations.
struct NormResult { uint32_t m; uint32_t s; };
BALOON_HD constexpr NormResult norm(uint32_t m, uint32_t s, uint32_t fuel) {
    return (fuel == 0 || m >= 1024u) ? NormResult{m, s} : norm(2u * m, s + 1u, fuel - 1u);
}

// ---- conversions ---------------------------------------------------------------------------
BALOON_HD inline uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
BALOON_HD inline float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// fp32 -> fp16 bits, round-to-nearest-even. NaN stays NaN (quiet), inf stays inf, overflow -> inf,
// results below half the smallest subnormal flush to a signed zero.
BALOON_HD inline uint16_t float_to_half(float f) {
    const uint32_t x = f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t expf = (int32_t)((x >> 23) & 0xFFu);
    const uint32_t man = x & 0x7FFFFFu;
    if (expf == 255) {                                   // inf / nan
        if (man == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7C00u | 0x200u | (man >> 13));
    }
    const int32_t e = expf - 127 + 15;                   // half exponent field
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);      // overflow -> inf
    if (e >= 1) return (uint16_t)round_normal(sign, (uint32_t)e, man);  // normal regime
    if (e < -10) return (uint16_t)sign;                  // flush to signed zero
    // subnormal regime: e in [-10, 0], shift = 14 - e in [14, 24]
    const uint32_t full = man | 0x800000u;               // 24-bit significand (fp32 normal)
    const uint32_t sh = (uint32_t)(14 - e);
    uint32_t q = full >> sh;                             // < 1024
    const uint32_t rem = full & ((1u << sh) - 1u);
    const uint32_t half = 1u << (sh - 1u);
    if (rem > half || (rem == half && (q & 1u))) ++q;    // may carry into the min normal: correct
    return (uint16_t)(sign | q);
}

// fp16 bits -> fp32, exact.
BALOON_HD inline float half_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    const uint32_t ex = ((uint32_t)h >> 10) & 0x1Fu;
    uint32_t m = (uint32_t)h & 0x3FFu;
    if (ex == 0) {
        if (m == 0) return bits_f32(sign);
        const NormResult r = norm(m, 0u, 10u);           // at most 10 shifts, exponent field 113 - s
        return bits_f32(sign | ((113u - r.s) << 23) | ((r.m & 0x3FFu) << 13));
    }
    if (ex == 31) return bits_f32(sign | 0x7F800000u | (m << 13));
    return bits_f32(sign | ((ex - 15u + 127u) << 23) | (m << 13));
}

// bf16 -> fp32 is `h << 16`, exactly invertible by taking the high half.
BALOON_HD inline float bf16_to_float(uint16_t h) { return bits_f32((uint32_t)h << 16); }
BALOON_HD inline uint16_t float_high_half(float f) { return (uint16_t)(f32_bits(f) >> 16); }

}  // namespace f16
}  // namespace baloon
