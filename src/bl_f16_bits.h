// bl_f16_bits.h - hand-written fp16 <-> fp32 bit conversions (1 sign | 5 exponent | 10 mantissa).
#pragma once
#include <cstdint>
#include "bl_nat.h"

namespace baloon::f16_bits {

constexpr nat exF(nat h) { return h / 1024 % 32; }
constexpr nat manF(nat h) { return h % 1024; }
constexpr nat signF(nat h) { return h / 32768 % 2; }

// Decode: normals map to f32 exponent ex+112, subnormals to 113-s after s normalising shifts,
// inf/NaN (ex = 31) to 255. Exact for every input.
constexpr std::uint32_t f32BitsFromF16(std::uint16_t h) {
  std::uint32_t sign = (std::uint32_t(h) & 0x8000u) << 16;
  std::uint32_t ex = (h >> 10) & 0x1Fu;
  std::uint32_t man = h & 0x3FFu;
  if (ex == 0) {
    if (man == 0) return sign;
    std::uint32_t s = 0;
    while (!(man & 0x400u)) { man <<= 1; ++s; }
    man &= 0x3FFu;
    return sign | ((113u - s) << 23) | (man << 13);
  }
  if (ex == 31) return sign | 0x7F800000u | (man << 13);
  return sign | ((ex + 112u) << 23) | (man << 13);
}

// Encode: round to nearest even, finite overflow saturates to inf, a mantissa carry may bump the exponent
// (0x7BFF + 1 = 0x7C00 = inf), subnormal results use a right shift of 14..24.
constexpr std::uint16_t f16FromF32Bits(std::uint32_t f) {
  std::uint32_t sign = (f >> 16) & 0x8000u;
  std::uint32_t rawexp = (f >> 23) & 0xFFu;
  std::uint32_t man = f & 0x7FFFFFu;
  if (rawexp == 255) {
    if (man == 0) return std::uint16_t(sign | 0x7C00u);
    return std::uint16_t(sign | 0x7C00u | 0x200u | (man >> 13));  // quiet NaN, payload kept
  }
  int e = int(rawexp) - 127 + 15;
  if (e >= 31) return std::uint16_t(sign | 0x7C00u);
  if (e <= 0) {
    if (e < -10) return std::uint16_t(sign);
    std::uint32_t m = man | 0x800000u;  // 24-bit significand
    std::uint32_t sh = std::uint32_t(14 - e);
    std::uint32_t h = m >> sh;
    std::uint32_t rem = m & ((1u << sh) - 1u);
    std::uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u))) ++h;
    return std::uint16_t(sign | h);
  }
  std::uint32_t h = (std::uint32_t(e) << 10) | (man >> 13);
  std::uint32_t rem = man & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return std::uint16_t(sign | h);
}

}  // namespace baloon::f16_bits
