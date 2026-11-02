// Round-to-nearest-even binary32 -> binary16 and exact binary16 -> binary32, on raw bit patterns.
#pragma once

#include <cstdint>

namespace baloon::fp16 {

inline constexpr uint32_t kNarrowInf = 0x7C00u;
inline constexpr uint32_t kNarrowQuietBit = 0x200u;
inline constexpr uint32_t kWideInf = 0x7F800000u;

// f32 bits -> f16 bits (round to nearest even).
//  * raw exponent 255: inf stays inf, NaN gets the quiet bit.
//  * finite overflow (raw exponent >= 143) saturates to signed inf, never NaN.
//  * raw exponent < 102 flushes to signed zero.
//  * 102..112 give subnormal results with RNE at the moved rounding point.
//  * otherwise normal: round from 13 dropped bits; a carry may bump the exponent up to inf.
inline uint16_t f16_from_f32(uint32_t x) {
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t rawexp = (x >> 23) & 0xFFu;
    const uint32_t man = x & 0x7FFFFFu;
    if (rawexp == 255) return static_cast<uint16_t>(sign | kNarrowInf | (man != 0 ? kNarrowQuietBit : 0u));
    if (rawexp >= 143) return static_cast<uint16_t>(sign | kNarrowInf);
    if (rawexp <= 112) {
        if (rawexp < 102) return static_cast<uint16_t>(sign);
        const uint32_t m = man | 0x800000u;
        const uint32_t sh = 126u - rawexp;  // 14..24
        uint32_t h = (m >> sh) & 0x3FFu;
        const uint32_t rem = m & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1u))) h += 1;
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = sign | ((rawexp - 112u) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1;
    return static_cast<uint16_t>(h);
}

// f16 bits -> f32 bits (exact: zero keeps its sign, subnormals are normalised, never flushed).
inline uint32_t f32_from_f16(uint32_t h) {
    h &= 0xFFFFu;
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t ex = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    if (ex == 0) {
        if (man == 0) return sign;
        uint32_t s = 0;
        while ((man & 0x400u) == 0) {
            man <<= 1;
            ++s;
        }
        return sign | ((113u - s) << 23) | ((man & 0x3FFu) << 13);
    }
    if (ex == 31) return sign | kWideInf | (man << 13);
    return sign | ((ex + 112u) << 23) | (man << 13);
}

inline bool is_nan16(uint32_t h) { return ((h >> 10) & 0x1Fu) == 31u && (h & 0x3FFu) != 0u; }

}  // namespace baloon::fp16

