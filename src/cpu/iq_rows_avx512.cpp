// iq_rows_avx512.cpp - AVX-512 multi-token i-quant row dots. Built with the AVX-512 ISA flags.
//
// Per 256-value block the dot is   d_w * d_act * (K/8) * sum_g scale_g * sum_{i in group g} sign_i * mag_i * q_i
// with K/8 = 1/8 (IQ2_XXS, IQ2_XS, IQ2_S), 1/4 (IQ3_XXS), 1 (IQ3_S). The integer part is evaluated per 64-value chunk:
// magnitudes (u8) x signed activations by maddubs (int16 pair sums never saturate: 43*127*2 < 32767), then madd with
// the per-lane scale vector (lane l of 32 takes scale l/8). The float accumulation is per block in block order.
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/kern_geometry.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace baloon {
namespace kern {
namespace {

// The format tables are registered by the caller; these two macros keep the declaration and the definition of the
// holder at one place.
#define GGML_COMMON_DECL_CPP struct TableHolder { IqTables t; };
#define GGML_COMMON_IMPL_CPP TableHolder g_tables;
GGML_COMMON_DECL_CPP
GGML_COMMON_IMPL_CPP

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
[[maybe_unused]] inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// 128-entry sign table: 7 sign bits plus an even-parity bit.
inline uint8_t ksign(uint32_t i) { return (uint8_t)(i | ((__builtin_popcount(i) & 1u) << 7)); }

// Grid lookups for the 4-value formats (G3 = one u32 entry -> 4 magnitude bytes).
#define G3(tbl, idx) ((tbl)[(idx)])

// Scale vector: int16 lane l (0..31) = s[l / 8].
inline __m512i scales4(const uint32_t s[4]) {
    alignas(64) int16_t v[32];
    for (uint32_t l = 0; l < 32; ++l) v[l] = (int16_t)s[iqgeo::scale_lane(l)];
    return _mm512_load_si512((const void*)v);
}

// One decoded 64-value chunk.
struct Chunk {
    alignas(64) uint8_t mag[64];
    uint64_t signs;
    uint32_t scale[4];
};

template <uint32_t TY> struct Fmt;

// IQ2_XXS: per 32-group 8 bytes at 2+16j+8k: four grid indices, then a u32 of 4x7 sign indices + scale (top 4 bits).
template <> struct Fmt<16> {
    static constexpr uint32_t bytes = 66, K = 1;
    static void decode(const uint8_t* blk, uint32_t j, Chunk& c) {
        const uint64_t* grid = g_tables.t.iq2xxs_grid;
        c.signs = 0;
        for (uint32_t k = 0; k < 2; ++k) {
            const uint8_t* p = blk + 2 + 16 * j + 8 * k;
            const uint32_t a1 = u32(p + 4);
            for (uint32_t l = 0; l < 4; ++l) {
                const uint64_t g = grid[p[l]];
                std::memcpy(c.mag + 32 * k + 8 * l, &g, 8);
                c.signs |= (uint64_t)ksign((a1 >> (7 * l)) & 127u) << (32 * k + 8 * l);
            }
            c.scale[2 * k] = c.scale[2 * k + 1] = 2 * (a1 >> 28) + 1;
        }
    }
};
// IQ2_XS: u16 words (9-bit grid index, 7-bit sign index); scale nibbles per 16 values.
template <> struct Fmt<17> {
    static constexpr uint32_t bytes = 74, K = 1;
    static void decode(const uint8_t* blk, uint32_t j, Chunk& c) {
        const uint64_t* grid = g_tables.t.iq2xs_grid;
        c.signs = 0;
        for (uint32_t k = 0; k < 2; ++k) {
            const uint8_t* p = blk + 2 + 16 * j + 8 * k;
            const uint8_t sc = blk[66 + 2 * j + k];
            for (uint32_t l = 0; l < 4; ++l) {
                const uint16_t w = u16(p + 2 * l);
                const uint64_t g = grid[w & 511u];
                std::memcpy(c.mag + 32 * k + 8 * l, &g, 8);
                c.signs |= (uint64_t)ksign(w >> 9) << (32 * k + 8 * l);
            }
            c.scale[2 * k] = 2 * (sc & 0xFu) + 1;
            c.scale[2 * k + 1] = 2 * (sc >> 4) + 1;
        }
    }
};
// IQ3_XXS: 8-bit grid indices (4 values each), u32 per 32-group with 4x7 sign indices + scale.
template <> struct Fmt<18> {
    static constexpr uint32_t bytes = 98, K = 2;
    static void decode(const uint8_t* blk, uint32_t j, Chunk& c) {
        const uint32_t* grid = g_tables.t.iq3xxs_grid;
        c.signs = 0;
        for (uint32_t k = 0; k < 2; ++k) {
            const uint8_t* p = blk + 2 + 16 * j + 8 * k;
            const uint32_t a = u32(blk + 66 + 8 * j + 4 * k);
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t g1 = G3(grid, p[2 * l]), g2 = G3(grid, p[2 * l + 1]);
                std::memcpy(c.mag + 32 * k + 8 * l, &g1, 4);
                std::memcpy(c.mag + 32 * k + 8 * l + 4, &g2, 4);
                c.signs |= (uint64_t)ksign((a >> (7 * l)) & 127u) << (32 * k + 8 * l);
            }
            c.scale[2 * k] = c.scale[2 * k + 1] = 2 * (a >> 28) + 1;
        }
    }
};
// IQ3_S: 8 low grid bits in qs, the 9th from qh; explicit sign bytes; one 4-bit scale per 32-group.
template <> struct Fmt<21> {
    static constexpr uint32_t bytes = 110, K = 8;
    static void decode(const uint8_t* blk, uint32_t j, Chunk& c) {
        const uint32_t* grid = g_tables.t.iq3s_grid;
        c.signs = 0;
        const uint8_t sc = blk[106 + j];
        for (uint32_t k = 0; k < 2; ++k) {
            const uint8_t* qs = blk + 2 + 16 * j + 8 * k;
            const uint32_t qh = blk[66 + 2 * j + k];
            const uint8_t* sg = blk + 74 + 8 * j + 4 * k;
            for (uint32_t l = 0; l < 4; ++l) {
                const uint32_t g1 = G3(grid, qs[2 * l] | ((qh << (8 - 2 * l)) & 256u));
                const uint32_t g2 = G3(grid, qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256u));
                std::memcpy(c.mag + 32 * k + 8 * l, &g1, 4);
                std::memcpy(c.mag + 32 * k + 8 * l + 4, &g2, 4);
                c.signs |= (uint64_t)sg[l] << (32 * k + 8 * l);
            }
            c.scale[2 * k] = c.scale[2 * k + 1] = 2 * ((k == 0 ? sc & 0xFu : sc >> 4)) + 1;
        }
    }
};
// IQ2_S: 8 low grid bits in qs (first half), explicit sign bytes (second half), 2 extra bits from qh, nibble scales.
template <> struct Fmt<22> {
    static constexpr uint32_t bytes = 82, K = 1;
    static void decode(const uint8_t* blk, uint32_t j, Chunk& c) {
        const uint64_t* grid = g_tables.t.iq2s_grid;
        c.signs = 0;
        for (uint32_t k = 0; k < 2; ++k) {
            const uint8_t* qs = blk + 2 + 8 * j + 4 * k;
            const uint8_t* sg = blk + 2 + 32 + 8 * j + 4 * k;
            const uint32_t qh = blk[66 + 2 * j + k];
            const uint8_t sc = blk[74 + 2 * j + k];
            for (uint32_t l = 0; l < 4; ++l) {
                const uint64_t g = grid[qs[l] | ((qh << (8 - 2 * l)) & 0x300u)];
                std::memcpy(c.mag + 32 * k + 8 * l, &g, 8);
                c.signs |= (uint64_t)sg[l] << (32 * k + 8 * l);
            }
            c.scale[2 * k] = 2 * (sc & 0xFu) + 1;
            c.scale[2 * k + 1] = 2 * (sc >> 4) + 1;
        }
    }
};

// Integer dot of one decoded chunk with 64 activations (int32 lanes, 16 of them).
inline __m512i chunk_dot(const Chunk& c, __m512i sv, const int8_t* q) {
    const __m512i y = _mm512_loadu_si512((const void*)q);
    const __m512i ys = _mm512_mask_sub_epi8(y, (__mmask64)c.signs, _mm512_setzero_si512(), y);   // masked negate
    const __m512i pr = _mm512_maddubs_epi16(_mm512_load_si512((const void*)c.mag), ys);
    return _mm512_madd_epi16(pr, sv);
}

// Dot of one row against NT activation vectors (the decode is shared by the NT tokens). out[t] for t < NT.
template <uint32_t TY, uint32_t NT>
inline void row_dot(const uint8_t* row, uint32_t nb, const Q8KBlock* acts, float* out) {
    float total[NT];
    for (uint32_t t = 0; t < NT; ++t) total[t] = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t)b * Fmt<TY>::bytes;
        const float d = h2f(u16(blk));
        __m512i acc[NT];
        for (uint32_t t = 0; t < NT; ++t) acc[t] = _mm512_setzero_si512();
        for (uint32_t j = 0; j < 4; ++j) {
            Chunk c;
            Fmt<TY>::decode(blk, j, c);
            const __m512i sv = scales4(c.scale);
            for (uint32_t t = 0; t < NT; ++t) acc[t] = _mm512_add_epi32(acc[t], chunk_dot(c, sv, acts[(size_t)t * nb + b].qs + 64 * j));
        }
        for (uint32_t t = 0; t < NT; ++t) total[t] += d * acts[(size_t)t * nb + b].d * (float)_mm512_reduce_add_epi32(acc[t]);
    }
    for (uint32_t t = 0; t < NT; ++t) out[t] = total[t] * ((float)Fmt<TY>::K / 8.f);
}

template <uint32_t TY, uint32_t NT>
void dot_rows_nt(const uint8_t* w, size_t row_bytes, uint32_t r0, uint32_t r1, uint32_t nb, const Q8KBlock* acts, float* out, size_t out_stride) {
    for (uint32_t r = r0; r < r1; ++r) {
        float o[NT];
        row_dot<TY, NT>(w + (size_t)r * row_bytes, nb, acts, o);
        for (uint32_t t = 0; t < NT; ++t) out[(size_t)t * out_stride + r] = o[t];
    }
}

template <uint32_t TY, uint32_t NT>
void gu_rows_nt(const uint8_t* w, size_t row_bytes, uint32_t n_ff, uint32_t r0, uint32_t r1, uint32_t nb, const Q8KBlock* acts, float* act, size_t act_stride) {
    for (uint32_t r = r0; r < r1; ++r) {
        float g[NT], u[NT];
        row_dot<TY, NT>(w + (size_t)r * row_bytes, nb, acts, g);
        row_dot<TY, NT>(w + (size_t)(n_ff + r) * row_bytes, nb, acts, u);
        for (uint32_t t = 0; t < NT; ++t) act[(size_t)t * act_stride + r] = g[t] / (1.f + std::exp(-g[t])) * u[t];
    }
}

template <uint32_t TY>
void dot_rows(const uint8_t* w, size_t row_bytes, uint32_t r0, uint32_t r1, uint32_t nb, const Q8KBlock* acts, uint32_t nt, float* out, size_t out_stride) {
    for (uint32_t t0 = 0; t0 < nt; t0 += iqgeo::group_size(nt, t0)) {
        const uint32_t n = iqgeo::group_size(nt, t0);
        const Q8KBlock* a = acts + (size_t)t0 * nb;
        float* o = out + (size_t)t0 * out_stride;
        switch (n) {
            case 1: dot_rows_nt<TY, 1>(w, row_bytes, r0, r1, nb, a, o, out_stride); break;
            case 2: dot_rows_nt<TY, 2>(w, row_bytes, r0, r1, nb, a, o, out_stride); break;
            case 3: dot_rows_nt<TY, 3>(w, row_bytes, r0, r1, nb, a, o, out_stride); break;
            default: dot_rows_nt<TY, 4>(w, row_bytes, r0, r1, nb, a, o, out_stride); break;
        }
    }
}

template <uint32_t TY>
void gu_rows(const uint8_t* w, size_t row_bytes, uint32_t n_ff, uint32_t r0, uint32_t r1, uint32_t nb, const Q8KBlock* acts, uint32_t nt, float* act, size_t act_stride) {
    for (uint32_t t0 = 0; t0 < nt; t0 += iqgeo::group_size(nt, t0)) {
        const uint32_t n = iqgeo::group_size(nt, t0);
        const Q8KBlock* a = acts + (size_t)t0 * nb;
        float* o = act + (size_t)t0 * act_stride;
        switch (n) {
            case 1: gu_rows_nt<TY, 1>(w, row_bytes, n_ff, r0, r1, nb, a, o, act_stride); break;
            case 2: gu_rows_nt<TY, 2>(w, row_bytes, n_ff, r0, r1, nb, a, o, act_stride); break;
            case 3: gu_rows_nt<TY, 3>(w, row_bytes, n_ff, r0, r1, nb, a, o, act_stride); break;
            default: gu_rows_nt<TY, 4>(w, row_bytes, n_ff, r0, r1, nb, a, o, act_stride); break;
        }
    }
}

}  // namespace

void iq512_set_tables(const IqTables& t) { g_tables.t = t; }

bool iq512_tables_ready(uint32_t type) {
    const IqTables& t = g_tables.t;
    switch (type) {
        case 16: return t.iq2xxs_grid != nullptr;
        case 17: return t.iq2xs_grid != nullptr;
        case 18: return t.iq3xxs_grid != nullptr;
        case 21: return t.iq3s_grid != nullptr;
        case 22: return t.iq2s_grid != nullptr;
        default: return false;
    }
}

bool iq512_supported(uint32_t type) { return iqgeo::supported(type); }

void iq512_rows(uint32_t type, const uint8_t* w, size_t row_bytes, uint32_t r0, uint32_t r1, uint32_t nb, const Q8KBlock* acts,
                uint32_t nt, float* out, size_t out_stride) {
    switch (type) {
        case 16: dot_rows<16>(w, row_bytes, r0, r1, nb, acts, nt, out, out_stride); break;
        case 17: dot_rows<17>(w, row_bytes, r0, r1, nb, acts, nt, out, out_stride); break;
        case 18: dot_rows<18>(w, row_bytes, r0, r1, nb, acts, nt, out, out_stride); break;
        case 21: dot_rows<21>(w, row_bytes, r0, r1, nb, acts, nt, out, out_stride); break;
        case 22: dot_rows<22>(w, row_bytes, r0, r1, nb, acts, nt, out, out_stride); break;
        default: break;
    }
}

void iq512_gu_rows(uint32_t type, const uint8_t* w, size_t row_bytes, uint32_t n_ff, uint32_t r0, uint32_t r1, uint32_t nb,
                   const Q8KBlock* acts, uint32_t nt, float* act, size_t act_stride) {
    switch (type) {
        case 16: gu_rows<16>(w, row_bytes, n_ff, r0, r1, nb, acts, nt, act, act_stride); break;
        case 17: gu_rows<17>(w, row_bytes, n_ff, r0, r1, nb, acts, nt, act, act_stride); break;
        case 18: gu_rows<18>(w, row_bytes, n_ff, r0, r1, nb, acts, nt, act, act_stride); break;
        case 21: gu_rows<21>(w, row_bytes, n_ff, r0, r1, nb, acts, nt, act, act_stride); break;
        case 22: gu_rows<22>(w, row_bytes, n_ff, r0, r1, nb, acts, nt, act, act_stride); break;
        default: break;
    }
}

}  // namespace kern
}  // namespace baloon
