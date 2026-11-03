// ggml_dequant_layouts.hpp - block layouts, code ranges and scalar dequantizers for
// Q2_0, Q4_0, Q5_0, Q6_K, Q4_K, Q5_K, Q3_K and IQ4_XS (little-endian hosts).
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace dequant {

// ---- block geometry (bytes / elements) --------------------------------------------------------
constexpr size_t kQ2_0Bytes = 18,  kQ2_0Elems = 64;   // d(2) + qs(16)
constexpr size_t kQ4_0Bytes = 18,  kQ4_0Elems = 32;   // d(2) + qs(16)
constexpr size_t kQ5_0Bytes = 22,  kQ5_0Elems = 32;   // d(2) + qh(4) + qs(16)
constexpr size_t kKElems = 256;                       // K-quant super-block
constexpr size_t kQ6KBytes = 210;                     // ql 0..127, qh 128..191, scales 192..207, d 208
constexpr size_t kQ4KBytes = 144;                     // d(2) dmin(2) scales(12) qs(128)
constexpr size_t kQ5KBytes = 176;                     // d dmin scales(12) qh(32) qs(128)
constexpr size_t kQ3KBytes = 110;                     // hmask(32) qs(64) scales(12) d(2)
constexpr size_t kIq4XsBytes = 136;                   // d(2) scales_h(2) scales_l(4) qs(128)

constexpr size_t kQ6KQlOff = 0, kQ6KQhOff = 128, kQ6KScOff = 192, kQ6KDOff = 208;
constexpr size_t kQ4KDOff = 0, kQ4KDminOff = 2, kQ4KScalesOff = 4, kQ4KQsOff = 16;
constexpr size_t kQ5KDOff = 0, kQ5KDminOff = 2, kQ5KScalesOff = 4, kQ5KQhOff = 16, kQ5KQsOff = 48;
constexpr size_t kQ3KHmaskOff = 0, kQ3KQsOff = 32, kQ3KScalesOff = 96, kQ3KDOff = 108;
constexpr size_t kIq4XsDOff = 0, kIq4XsScalesHOff = 2, kIq4XsScalesLOff = 4, kIq4XsQsOff = 8;

static_assert(2 + 16 == kQ2_0Bytes && 64 * 2 == 16 * 8, "q2_0 layout");
static_assert(2 + 4 + 16 == kQ5_0Bytes, "q5_0 layout");
static_assert(128 + 64 + 16 + 2 == kQ6KBytes && kQ6KScOff == 128 + 64 && kQ6KDOff == 192 + 16, "q6k layout");
static_assert(2 + 2 + 12 + 128 == kQ4KBytes, "q4k layout");
static_assert(16 + 32 + 128 == kQ5KBytes, "q5k layout");
static_assert(32 + 64 + 12 + 2 == kQ3KBytes && kQ3KScalesOff == 32 + 64 && kQ3KDOff == 96 + 12, "q3k layout");
static_assert(2 + 2 + 4 + 128 == kIq4XsBytes, "iq4xs layout");

// IQ4_NL / IQ4_XS 16-entry codebook.
constexpr int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ---- helpers ------------------------------------------------------------------------------------
inline uint16_t read_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline float read_f16(const uint8_t* p) { return f16::half_to_float(read_u16(p)); }

// 2-bit code at position r (0..3) inside byte b; symbol is code - 1 in {-1,0,1,2}.
constexpr int q2_code(unsigned b, unsigned r) { return (int)((b >> (2 * r)) & 3u); }
constexpr int q2_sym(int c) { return c - 1; }

// 6-bit scale/min unpack used by Q4_K and Q5_K (reads only bytes 0..11).
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---- dequantizers: each takes exactly one block and writes exactly `*Elems` floats ---------------
inline void dequantize_q2_0(const uint8_t* b, float* y) {
    const float d = read_f16(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 64; ++j) y[j] = d * (float)q2_sym(q2_code(qs[j / 4], (unsigned)(j % 4)));
}

inline void dequantize_q4_0(const uint8_t* b, float* y) {
    const float d = read_f16(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = (float)((int)(qs[j] & 0xF) - 8) * d;
        y[j + 16] = (float)((int)(qs[j] >> 4) - 8) * d;
    }
}

inline void dequantize_q5_0(const uint8_t* b, float* y) {
    const float d = read_f16(b);
    const uint32_t qh = read_u32(b + 2);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        const int x0 = (int)((qs[j] & 0xF) | (((qh >> j) << 4) & 0x10)) - 16;
        const int x1 = (int)((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
        y[j] = (float)x0 * d;
        y[j + 16] = (float)x1 * d;
    }
}

inline void dequantize_q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b + kQ6KQlOff;
    const uint8_t* qh = b + kQ6KQhOff;
    const int8_t* sc = (const int8_t*)(b + kQ6KScOff);
    const float d = read_f16(b + kQ6KDOff);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

inline void dequantize_q4_k(const uint8_t* b, float* y) {
    const float d = read_f16(b + kQ4KDOff);
    const float min = read_f16(b + kQ4KDminOff);
    const uint8_t* scales = b + kQ4KScalesOff;
    const uint8_t* q = b + kQ4KQsOff;
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32; is += 2;
    }
}

inline void dequantize_q5_k(const uint8_t* b, float* y) {
    const float d = read_f16(b + kQ5KDOff);
    const float min = read_f16(b + kQ5KDminOff);
    const uint8_t* scales = b + kQ5KScalesOff;
    const uint8_t* qh = b + kQ5KQhOff;
    const uint8_t* ql = b + kQ5KQsOff;
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; is += 2;
        u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
    }
}

// Unpack the 16 six-bit Q3_K scales (stored value, before the -32 bias) from the 12 packed bytes.
inline void q3k_unpack_scales(const uint8_t* packed12, int8_t out[16]) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    std::memcpy(aux, packed12, 12);
    aux[3] = 0;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(out, aux, 16);
}

inline void dequantize_q3_k(const uint8_t* b, float* y) {
    const float dall = read_f16(b + kQ3KDOff);
    const uint8_t* hm = b + kQ3KHmaskOff;
    const uint8_t* q = b + kQ3KQsOff;
    int8_t scales[16];
    q3k_unpack_scales(b + kQ3KScalesOff, scales);
    uint8_t m = 1;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = dall * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *y++ = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = dall * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}

inline void dequantize_iq4_xs(const uint8_t* b, float* y) {
    const float d = read_f16(b + kIq4XsDOff);
    const uint16_t scales_h = read_u16(b + kIq4XsScalesHOff);
    const uint8_t* scales_l = b + kIq4XsScalesLOff;
    const uint8_t* qs = b + kIq4XsQsOff;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j] = dl * (float)kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32; qs += 16;
    }
}

}  // namespace dequant
}  // namespace baloon
