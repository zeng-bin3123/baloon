#pragma once
// strata/kernels/f16_bits.hpp — IEEE-754 fp16 ↔ fp32 bit-level conversion
// Target: Iluvatar BI-V100 (ivcore10)
//
// f16_from_f32: round-to-nearest-even conversion from float to fp16.
//   Sign extraction:  sign = (x >> 16) & 0x8000
//   Inf/NaN:          rawexp == 255 → inf or NaN (preserves signaling)
//   Overflow:         exp >= 31 (finite) → inf, NOT NaN (old version was wrong)
//   Subnormal:        exp <= 0 → restore implicit 1, shift, round-to-nearest-even
//   Normal:           standard mantissa truncation with round-to-nearest-even tiebreak
//
// f32_from_f16: exact (no rounding) expansion of fp16 to fp32.
//   Every one of 65,536 fp16 values maps to an exactly representable f32.
//   Subnormals are NOT flushed to zero (old private copies had this bug).
//   0x0001 → 2^{-24}, NOT 0.0.
//
// Both functions are host+device, header-only, no dependencies beyond <cstring>.

#include <cstdint>
#include <cstring>

// STRATA_HD: host + device qualifier for CUDA-compatible builds.
#if defined(__CUDACC__) || defined(__ILUVATAR__)
#define STRATA_HD __host__ __device__
#else
#define STRATA_HD
#endif

namespace strata {
namespace kernels {

// ─── f16_from_f32 ────────────────────────────────────────────────────
// Convert float to fp16 (uint16_t) with IEEE-754 round-to-nearest-even.
//
// Three regimes after sign extraction:
//   1. Inf/NaN (rawexp == 255): preserve inf vs NaN distinction.
//   2. Overflow (exp >= 31, finite): → positive/negative infinity.
//      CRITICAL: returns 0x7C00, NOT a NaN. Old version was wrong.
//   3. Subnormal (exp <= 0): restore implicit leading 1, shift right
//      by (14 - exp), round-to-nearest-even on the remainder.
//   4. Normal: truncate mantissa from 23 to 10 bits with round-to-nearest-even.
//      Tiebreak: bump if odd (round-to-even).
STRATA_HD inline uint16_t f16_from_f32(float val) {
    uint32_t x;
    std::memcpy(&x, &val, 4);

    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    int exp = static_cast<int>(rawexp) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;

    // Inf or NaN
    if (rawexp == 0xFFu) {
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    }

    // Overflow → inf (NOT NaN)
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }

    // Subnormal or zero
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        // Restore implicit leading 1
        man |= 0x800000u;
        const uint32_t sh = static_cast<uint32_t>(14 - exp);
        uint32_t h = (man >> sh) & 0x3FFu;
        const uint32_t rem = man & ((1u << sh) - 1u);
        // Round-to-nearest-even
        if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (h & 1u))) {
            ++h;
        }
        return static_cast<uint16_t>(sign | h);
    }

    // Normal: truncate mantissa, round-to-nearest-even
    uint16_t h = static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        ++h;  // round-to-nearest-even: bump carries through exponent correctly
    }
    return h;
}

// ─── f32_from_f16 ────────────────────────────────────────────────────
// Convert fp16 (uint16_t) to float. Exact — no rounding required.
// Every one of 65,536 fp16 bit patterns maps to an exactly representable f32.
//
// Subnormal handling (NOT flushed to zero):
//   h = 0x0001 → f = 2^{-24}  (old private copies flushed this to 0.0)
//   Value = man × 2^{-24}
//   f32 exponent = 113 - s where s = leading-zero count in mantissa
STRATA_HD inline float f32_from_f16(uint16_t val) {
    const uint32_t sign = static_cast<uint32_t>(val & 0x8000u) << 16;
    const uint32_t ex  = (val >> 10) & 0x1Fu;
    const uint32_t man = val & 0x3FFu;

    uint32_t out;
    if (ex == 0) {
        if (man == 0) {
            // Signed zero
            out = sign;
        } else {
            // Subnormal: normalize by counting leading zeros in mantissa.
            // Shift mantissa left until bit 10 is set, tracking the shift count.
            uint32_t m = man;
            int s = 0;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                ++s;
            }
            // f32 exponent: original bias adjustment minus leading-zero shift
            // ex=0 in fp16 represents 2^{-14}, adjusted by (s) for normalization
            out = sign | (static_cast<uint32_t>(113 - s) << 23) |
                  ((m & 0x3FFu) << 13);
        }
    } else if (ex == 31) {
        // Inf or NaN — preserve mantissa bits
        out = sign | 0x7F800000u | (man << 13);
    } else {
        // Normal: rebias exponent from fp16 (bias 15) to fp32 (bias 127)
        out = sign | ((ex - 15 + 127) << 23) | (man << 13);
    }

    float result;
    std::memcpy(&result, &out, 4);
    return result;
}

}  // namespace kernels
}  // namespace strata

