// Shared host/device helpers for unit 4 (header-only, no ix_compat dependency so they can be unit-tested on host).
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__ILUVATAR__)
#define IX4_HD __host__ __device__ inline
#else
#define IX4_HD inline
#endif
IX4_HD int ix4_ceil_div(int a, int b) { return (a + b - 1) / b; }
// float -> IEEE half bits, round to nearest even
IX4_HD uint16_t ix4_f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0));
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u; int sh = 14 - e; uint32_t r = m >> sh; uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) r++;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t)e << 10) | (m >> 13); uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) r++;
    return (uint16_t)(sign | r);
}
IX4_HD float ix4_f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { float f = ldexpf((float)m, -24); if (s) f = -f; return f; } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
