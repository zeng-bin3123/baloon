#pragma once
#include <cstdint>
typedef uint16_t half;
struct half2 { half x, y; };
inline half __float2half(float f) { (void)f; return 0; }
inline float __half2float(half h) { (void)h; return 0.0f; }
inline uint16_t __half_as_ushort(half h) { return h; }
inline half __ushort_as_half(uint16_t u) { return u; }
inline half2 __halves2half2(half a, half b) { return {a, b}; }
inline half __low2half(half2 h2) { return h2.x; }
inline half __high2half(half2 h2) { return h2.y; }
inline int __float2int_rn(float f) { return (int)(f + 0.5f); }
