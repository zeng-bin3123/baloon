// spec_models.hpp - unit u63af4bc057 oracles, written from the LAYER 1 theory (not from the code under test):
// the Nat/Int models of the theory (rneBump, roundMan13, normShift, bf16Encode, f2bf, scaleMinK4, q2Code, iq4xsScale,
// q6kPos, q3kMask, alignUp, groupsOf4, ...) and independent element-by-element spec decoders of the block formats.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace u63 {

// ------------------------------------------------------------------------------------------- HalfPrecisionBitConversion
inline uint32_t f16Sign(uint32_t h) { return h / 32768 % 2; }
inline uint32_t f16Exp(uint32_t h) { return h / 1024 % 32; }
inline uint32_t f16Man(uint32_t h) { return h % 1024; }
inline uint32_t rneBump(uint64_t q, uint64_t r, uint64_t half) { return (r > half || (r == half && q % 2 == 1)) ? 1 : 0; }
inline uint32_t roundMan13(uint32_t man) { return man / 8192 + rneBump(man / 8192, man % 8192, 4096); }
inline uint64_t roundDiv(uint64_t x, uint64_t half) { return x / (2 * half) + rneBump(x / (2 * half), x % (2 * half), half); }
inline uint32_t normShift(uint32_t fuel, uint32_t m) {
    uint32_t s = 0;
    while (fuel > 0 && m < 1024) { m *= 2; ++s; --fuel; }
    return s;
}
// encodeSpecial: returns true and sets out for the special branches (sign is 0 or 0x8000).
inline bool encodeSpecial(uint32_t sign, uint32_t rawexp, uint32_t man, int exp, uint32_t* out) {
    if (rawexp == 255) { *out = sign + 0x7C00 + (man != 0 ? 0x200 : 0); return true; }
    if (exp >= 31) { *out = sign + 0x7C00; return true; }
    return false;
}
inline uint32_t conflatedOverflow(uint32_t sign, uint32_t man) { return sign + 0x7C00 + (man != 0 ? 0x200 : 0); }
inline uint32_t bf16Encode(uint64_t i) { return (uint32_t)((i + i / 65536 % 2 + 32767) % 4294967296ull / 65536); }
inline uint32_t bf16EncodeBit15(uint64_t i) { return (uint32_t)((i + i / 32768 % 2 + 32767) % 4294967296ull / 65536); }
inline uint64_t bf16Decode(uint32_t h) { return (uint64_t)h * 65536; }

// Exact value of an fp16 word (non-NaN), computed with ldexp (independent of any bit manipulation in the code).
inline double f16_exact(uint16_t h) {
    const int s = f16Sign(h), e = (int)f16Exp(h), m = (int)f16Man(h);
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) v = INFINITY;
    else v = std::ldexp(1024.0 + m, e - 25);
    return s ? -v : v;
}

// --------------------------------------------------------------------------------------------------- GgmlBlockDequant
inline uint32_t f2bf(uint64_t u) { return (uint32_t)((u + 0x7fff + u / 65536 % 2) % 4294967296ull / 65536 % 65536); }
inline void scaleMinK4(int j, const uint8_t* q, int* sc, int* mn) {
    if (j < 4) { *sc = q[j] & 63; *mn = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4); *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
inline uint32_t groupsPerBlock(uint32_t e) { return e / 32; }
inline bool iqOnly(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }
inline const int* gbdSupported(int* n) { static const int t[] = {2, 6, 8, 20, 11, 12, 13, 14, 23, 42}; *n = 10; return t; }
// theory geometry table of GgmlBlockDequant / layoutBytes
inline bool gbdGeometry(int t, uint32_t* e, uint32_t* b) {
    switch (t) {
        case 2: *e = 32; *b = 2 + 16; return true;            case 6: *e = 32; *b = 2 + 4 + 16; return true;
        case 8: *e = 32; *b = 2 + 32; return true;            case 20: *e = 32; *b = 2 + 16; return true;
        case 11: *e = 256; *b = 32 + 64 + 12 + 2; return true; case 12: *e = 256; *b = 2 + 2 + 12 + 128; return true;
        case 13: *e = 256; *b = 2 + 2 + 12 + 32 + 128; return true; case 14: *e = 256; *b = 128 + 64 + 16 + 2; return true;
        case 23: *e = 256; *b = 2 + 2 + 4 + 128; return true; case 42: *e = 64; *b = 2 + 16; return true;
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------------ BlockDequantization
inline const int* iq4nlTable() { static const int t[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}; return t; }
inline uint32_t slotDiv(uint32_t k) { return 1u << (2 * k); }
inline uint32_t q2Code(const uint8_t* qs, uint32_t j) { return qs[j / 4] / slotDiv(j % 4) % 4; }
inline uint32_t packByte(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) { return c0 + 4 * c1 + 16 * c2 + 64 * c3; }
inline uint32_t q6kPos(uint32_t n, uint32_t l, uint32_t k) { return 128 * n + l + 32 * k; }
inline uint32_t q6kScale(uint32_t n, uint32_t l, uint32_t k) { return 8 * n + l / 16 + 2 * k; }
inline uint32_t q5kMask1(uint32_t g) { return 1u << (2 * g); }
inline uint32_t q5kMask2(uint32_t g) { return 2u << (2 * g); }
inline uint32_t q3kMask(uint32_t half, uint32_t j) { return 1u << (4 * half + j); }
inline uint32_t iq4xsScale(const uint8_t* sl, uint32_t sh, uint32_t ib) {
    return sl[ib / 2] / (ib % 2 == 0 ? 1 : 16) % 16 + 16 * (sh >> (2 * ib) & 3);
}


// Independent spec decoders, element by element, following the theory's index maps. Arithmetic is done in double so the
// oracle carries no rounding of its own beyond the final value.
inline void spec_q2_0(const uint8_t* b, double* y, double d) { for (uint32_t j = 0; j < 64; ++j) y[j] = ((int)q2Code(b + 2, j) - 1) * d; }
inline void spec_q4_0(const uint8_t* b, double* y, double d) {
    for (int p = 0; p < 32; ++p) { const uint8_t x = b[2 + p % 16]; y[p] = ((p < 16 ? x % 16 : x / 16) - 8) * d; }
}
inline void spec_q5_0(const uint8_t* b, double* y, double d) {
    uint32_t qh; std::memcpy(&qh, b + 2, 4);
    for (int p = 0; p < 32; ++p) { const uint8_t x = b[6 + p % 16]; const int nib = p < 16 ? x % 16 : x / 16;
        y[p] = (nib + 16 * (int)(qh >> p & 1) - 16) * d; }
}
inline void spec_q8_0(const uint8_t* b, double* y, double d) { for (int p = 0; p < 32; ++p) y[p] = (int8_t)b[2 + p] * d; }
inline void spec_iq4_nl(const uint8_t* b, double* y, double d) {
    for (int p = 0; p < 32; ++p) { const uint8_t x = b[2 + p % 16]; y[p] = iq4nlTable()[p < 16 ? x % 16 : x / 16] * d; }
}
inline void spec_q6_K(const uint8_t* b, double* y, double d) {
    const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192);
    for (uint32_t n = 0; n < 2; ++n) for (uint32_t l = 0; l < 32; ++l) for (uint32_t k = 0; k < 4; ++k) {
        const uint8_t lb = ql[64 * n + l + 32 * (k % 2)];
        const int lo = k < 2 ? lb % 16 : lb / 16;
        const int hi = qh[32 * n + l] >> (2 * k) & 3;
        y[q6kPos(n, l, k)] = d * sc[q6kScale(n, l, k)] * (lo + 16 * hi - 32);
    }
}
inline void spec_q4_K(const uint8_t* b, double* y, double d, double dmin) {
    for (uint32_t g = 0; g < 4; ++g) for (uint32_t h = 0; h < 2; ++h) for (uint32_t l = 0; l < 32; ++l) {
        int sc, mn; scaleMinK4((int)(2 * g + h), b + 4, &sc, &mn);
        const uint8_t x = b[16 + 32 * g + l];
        y[64 * g + 32 * h + l] = d * sc * (h ? x / 16 : x % 16) - dmin * mn;
    }
}
inline void spec_q5_K(const uint8_t* b, double* y, double d, double dmin) {
    const uint8_t* qh = b + 16; const uint8_t* qs = b + 48;
    for (uint32_t g = 0; g < 4; ++g) for (uint32_t h = 0; h < 2; ++h) for (uint32_t l = 0; l < 32; ++l) {
        int sc, mn; scaleMinK4((int)(2 * g + h), b + 4, &sc, &mn);
        const uint8_t x = qs[32 * g + l];
        const uint32_t mask = h ? q5kMask2(g) : q5kMask1(g);
        const int q = (h ? x / 16 : x % 16) + ((qh[l] & mask) ? 16 : 0);
        y[64 * g + 32 * h + l] = d * sc * q - dmin * mn;
    }
}
inline int q3kScaleByte(const uint8_t* s, int i) {   // the 16 repacked 6-bit scales (before the -32 bias)
    const int t = s[8 + i % 4];
    if (i < 4) return (s[i] & 15) | ((t >> 0 & 3) << 4);
    if (i < 8) return (s[i] & 15) | ((t >> 2 & 3) << 4);
    if (i < 12) return (s[i - 8] >> 4) | ((t >> 4 & 3) << 4);
    return (s[i - 8] >> 4) | ((t >> 6 & 3) << 4);
}
inline void spec_q3_K(const uint8_t* b, double* y, double d) {
    const uint8_t* hm = b; const uint8_t* qs = b + 32; const uint8_t* sc = b + 96;
    for (uint32_t half = 0; half < 2; ++half) for (uint32_t j = 0; j < 4; ++j) for (uint32_t s = 0; s < 2; ++s) for (uint32_t l = 0; l < 16; ++l) {
        const uint32_t p = 128 * half + 32 * j + 16 * s + l;
        const int lo = qs[32 * half + 16 * s + l] >> (2 * j) & 3;
        const bool hbit = (hm[16 * s + l] & q3kMask(half, j)) != 0;
        y[p] = d * (q3kScaleByte(sc, (int)(p / 16)) - 32) * (lo - (hbit ? 0 : 4));
    }
}
inline void spec_iq4_xs(const uint8_t* b, double* y, double d) {
    const uint32_t sh = (uint32_t)(b[2] | b[3] << 8); const uint8_t* sl = b + 4; const uint8_t* qs = b + 8;
    for (uint32_t ib = 0; ib < 8; ++ib) for (uint32_t p = 0; p < 32; ++p) {
        const uint8_t x = qs[16 * ib + p % 16];
        y[32 * ib + p] = d * ((int)iq4xsScale(sl, sh, ib) - 32) * iq4nlTable()[p < 16 ? x % 16 : x / 16];
    }
}

// ------------------------------------------------------------------------------------------------- GgufHeaderParsing
inline uint64_t alignUp(uint64_t pos, uint64_t a) { return (pos + a - 1) / a * a; }
inline bool gguf_geometry(uint32_t t, uint64_t* e, uint64_t* b) {
    switch (t) {
        case 2: *e = 32; *b = 18; return true;   case 6: *e = 32; *b = 22; return true;   case 8: *e = 32; *b = 34; return true;
        case 20: *e = 32; *b = 18; return true;  case 11: *e = 256; *b = 110; return true; case 12: *e = 256; *b = 144; return true;
        case 13: *e = 256; *b = 176; return true; case 14: *e = 256; *b = 210; return true; case 23: *e = 256; *b = 136; return true;
        case 16: *e = 256; *b = 66; return true; case 17: *e = 256; *b = 74; return true; case 18: *e = 256; *b = 98; return true;
        case 21: *e = 256; *b = 110; return true; case 22: *e = 256; *b = 82; return true; case 42: *e = 64; *b = 18; return true;
        default: return false;
    }
}

// --------------------------------------------------------------------------------------------------- IQuantRowDecode
inline bool iqSupported(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }
inline uint32_t oddScale(uint32_t s) { return 2 * (s % 16) + 1; }
inline uint32_t laneScaleSlot(uint32_t i) { return i / 8; }
inline std::vector<uint32_t> groupsOf4(uint32_t n) { std::vector<uint32_t> g; while (n > 0) { const uint32_t k = n < 4 ? n : 4; g.push_back(k); n -= k; } return g; }
inline uint8_t ksign(uint32_t i) { return (uint8_t)(i | ((__builtin_popcount(i) & 1u) << 7)); }

// Spec weights of one i-quant block as (integer magnitude*sign, odd scale) pairs; value v of the block contributes
// d * scale_v * w_v * K where K = 1/8 (IQ2_*), 1/4 (IQ3_XXS), 1 (IQ3_S).
struct IqGridsHost { const uint64_t* g2xxs; const uint64_t* g2xs; const uint64_t* g2s; const uint32_t* g3xxs; const uint32_t* g3s; };
inline uint8_t byte_of(uint64_t g, int j) { return (uint8_t)(g >> (8 * j)); }
inline void spec_iq_block(uint32_t ty, const uint8_t* blk, const IqGridsHost& G, int* w, int* sc, double* K) {
    for (uint32_t ib = 0; ib < 8; ++ib) {
        for (uint32_t l = 0; l < 4; ++l) for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t v = 32 * ib + 8 * l + j;
            int mag = 0; bool neg = false; uint32_t s = 0;
            if (ty == 16) {
                const uint8_t* p = blk + 2 + 8 * ib; uint32_t a; std::memcpy(&a, p + 4, 4);
                mag = byte_of(G.g2xxs[p[l]], (int)j); neg = ksign(a >> (7 * l) & 127) >> j & 1; s = a >> 28;
            } else if (ty == 17) {
                uint16_t q; std::memcpy(&q, blk + 2 + 8 * ib + 2 * l, 2);
                mag = byte_of(G.g2xs[q % 512], (int)j); neg = ksign(q / 512) >> j & 1;
                s = l < 2 ? blk[66 + ib] % 16 : blk[66 + ib] / 16;
            } else if (ty == 18) {
                const uint8_t* p = blk + 2 + 8 * ib; uint32_t a; std::memcpy(&a, blk + 66 + 4 * ib, 4);
                mag = byte_of(G.g3xxs[p[2 * l + j / 4]], (int)(j % 4)); neg = ksign(a >> (7 * l) & 127) >> j & 1; s = a >> 28;
            } else if (ty == 21) {
                const uint8_t* qs = blk + 2 + 8 * ib; const uint32_t qh = blk[66 + ib]; const uint8_t* sg = blk + 74 + 4 * ib;
                const uint32_t which = 2 * l + j / 4;                       // grid entry index within the group (0..7)
                const uint32_t idx = qs[which] + 256 * (qh >> which & 1);   // 9th bit from qh
                mag = byte_of(G.g3s[idx], (int)(j % 4)); neg = sg[l] >> j & 1;
                s = ib % 2 == 0 ? blk[106 + ib / 2] % 16 : blk[106 + ib / 2] / 16;
            } else if (ty == 22) {
                const uint8_t* qs = blk + 2 + 4 * ib; const uint8_t* sg = blk + 34 + 4 * ib; const uint32_t qh = blk[66 + ib];
                const uint32_t idx = qs[l] + 256 * (qh >> (2 * l) & 3);
                mag = byte_of(G.g2s[idx], (int)j); neg = sg[l] >> j & 1;
                s = l < 2 ? blk[74 + ib] % 16 : blk[74 + ib] / 16;
            }
            w[v] = neg ? -mag : mag; sc[v] = (int)oddScale(s);
        }
    }
    *K = ty == 18 ? 0.25 : ty == 21 ? 1.0 : 0.125;
}

}  // namespace u63

