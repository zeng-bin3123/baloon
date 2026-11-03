// scalar_dequantizers.hpp - scalar reference dequantizers: Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS.
// Block geometry, bit-field addressing and value ranges follow the ggml block layouts (little endian).
#pragma once
#include <cstdint>
#include <cstring>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace sdq {

constexpr int kQK = 32, kQK2_0 = 64, kQK_K = 256;
constexpr int kBlkQ2_0 = 18, kBlkQ4_0 = 18, kBlkQ5_0 = 22, kBlkQ8_0 = 34, kBlkIQ4NL = 18;
constexpr int kBlkQ3K = 110, kBlkQ4K = 144, kBlkQ5K = 176, kBlkQ6K = 210, kBlkIQ4XS = 136;

constexpr int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float rd_f16(const uint8_t* p) { return f16::half_to_float((uint16_t)(p[0] | (p[1] << 8))); }

// ---- Q2_0: 64 elements, d(2) + 16 payload bytes, code c in 0..3 -> (c-1)*d, four codes per byte LSB-first -----------
inline int q2_code(uint8_t b, int k) { return (b >> (k * 2)) & 3; }
inline uint8_t q2_pack(int c0, int c1, int c2, int c3) { return (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)); }
inline void deq_q2_0(const uint8_t* b, float* y) {
    const float d = rd_f16(b);
    for (int j = 0; j < 64; ++j) y[j] = d * (float)(q2_code(b[2 + j / 4], j % 4) - 1);
}

// ---- Q4_0 / IQ4_NL / Q5_0 / Q8_0 -------------------------------------------------------------------------------------
inline void deq_q4_0(const uint8_t* b, float* y) {
    const float d = rd_f16(b); const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) { y[j] = d * (float)((qs[j] & 0xF) - 8); y[j + 16] = d * (float)((qs[j] >> 4) - 8); }
}
inline void deq_iq4_nl(const uint8_t* b, float* y) {
    const float d = rd_f16(b); const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) { y[j] = d * (float)kIq4nl[qs[j] & 0xF]; y[j + 16] = d * (float)kIq4nl[qs[j] >> 4]; }
}
inline void deq_q5_0(const uint8_t* b, float* y) {
    const float d = rd_f16(b);
    uint32_t qh; std::memcpy(&qh, b + 2, 4);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        const int h0 = (int)((qh >> j) & 1u) << 4, h1 = (int)((qh >> (j + 16)) & 1u) << 4;
        y[j] = d * (float)(((qs[j] & 0xF) | h0) - 16);
        y[j + 16] = d * (float)(((qs[j] >> 4) | h1) - 16);
    }
}
inline void deq_q8_0(const uint8_t* b, float* y) {
    const float d = rd_f16(b);
    for (int j = 0; j < 32; ++j) y[j] = d * (float)(int8_t)b[2 + j];
}

// ---- K-quants: 6-bit scale/min of Q4_K / Q5_K ------------------------------------------------------------------------
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
inline void deq_q4_k(const uint8_t* b, float* y) {
    const float d = rd_f16(b), dmin = rd_f16(b + 2);
    const uint8_t* sc = b + 4; const uint8_t* q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(is, sc, &s1, &m1); scale_min_k4(is + 1, sc, &s2, &m2);
        const float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0xF) - mm1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - mm2;
        q += 32; is += 2;
    }
}
// Q5_K: the fifth bit is selected by a mask pair (u1,u2) that shifts left by 2 per 64-element group.
inline void deq_q5_k(const uint8_t* b, float* y) {
    const float d = rd_f16(b), dmin = rd_f16(b + 2);
    const uint8_t* sc = b + 4; const uint8_t* qh = b + 16; const uint8_t* ql = b + 48;
    int is = 0; uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(is, sc, &s1, &m1); scale_min_k4(is + 1, sc, &s2, &m2);
        const float d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - mm1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - mm2;
        ql += 32; is += 2; u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
    }
}
inline void deq_q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192);
    const float d = rd_f16(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}
// Q3_K: signed 6-bit scales with -32 bias; hmask bit (1 << (4n+j)) is hoisted across both halves (never reset).
inline void q3k_scales(const uint8_t* s12, int out[16]) {
    for (int i = 0; i < 16; ++i) {
        const int lo = i < 8 ? (s12[i] & 0xF) : (s12[i - 8] >> 4);
        const int hi = (s12[8 + (i % 4)] >> (2 * (i / 4))) & 3;
        out[i] = (lo | (hi << 4)) - 32;
    }
}
inline void deq_q3_k(const uint8_t* b, float* y) {
    const uint8_t* hm = b; const uint8_t* q = b + 32;
    int sc[16]; q3k_scales(b + 96, sc);
    const float d = rd_f16(b + 108);
    int is = 0; uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d * (float)sc[is++];
            for (int l = 0; l < 16; ++l) *y++ = dl * (float)(((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d * (float)sc[is++];
            for (int l = 0; l < 16; ++l) *y++ = dl * (float)(((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2; m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}
// IQ4_XS: d(2) scales_h(2) scales_l(4) qs(128); eight groups of 16 bytes, 6-bit group scale - 32.
inline void deq_iq4_xs(const uint8_t* b, float* y) {
    const float d = rd_f16(b);
    const uint16_t sh = (uint16_t)(b[2] | (b[3] << 8));
    const uint8_t* sl = b + 4; const uint8_t* qs = b + 8;
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (int j = 0; j < 16; ++j) { y[j] = dl * (float)kIq4nl[qs[j] & 0xF]; y[j + 16] = dl * (float)kIq4nl[qs[j] >> 4]; }
        y += 32; qs += 16;
    }
}

}  // namespace sdq
}  // namespace baloon
