// q2_avx2_kernel.hpp - AVX2-only Q2_0 row dot over GGUF-layout blocks (18 bytes: fp16 scale + 16 code bytes) and
// the scalar-rule activation quantiser, for CPUs without AVX-512.
//
// Portable scalar code is always available; when compiled with AVX2 enabled, `row_dot_avx2` provides the
// maddubs/madd implementation. Both compute identical integer sums.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "baloon/f16_conversion.hpp"
#include "baloon/q2_expert_vnni_kernel.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace baloon {
namespace q2_avx2 {

constexpr uint32_t BLOCK_BYTES = 18;
constexpr uint32_t BLOCK_VALUES = 64;
static_assert(2 + 16 == BLOCK_BYTES, "scale + codes");

constexpr uint64_t scale_offset(uint64_t b) { return BLOCK_BYTES * b; }
constexpr uint64_t codes_offset(uint64_t b) { return BLOCK_BYTES * b + 2; }
constexpr uint32_t low_chunk(uint32_t b) { return 2 * b; }        // values 64b .. 64b+31
constexpr uint32_t high_chunk(uint32_t b) { return 2 * b + 1; }   // values 64b+32 .. 64b+63

constexpr uint32_t n_chunks(uint32_t n) { return n / 32; }
constexpr uint32_t batch_size(uint32_t nt, uint32_t t0) { return nt - t0 < 4 ? nt - t0 : 4; }

// Value 4k+t of an unpacked block is code t of byte k.
constexpr uint32_t code_of(uint32_t byte, uint32_t t) { return (byte >> (2 * t)) & 3u; }

inline void unpack_block(const uint8_t* codes16, uint8_t out[64]) {
    for (uint32_t k = 0; k < 16; ++k)
        for (uint32_t t = 0; t < 4; ++t) out[4 * k + t] = (uint8_t)code_of(codes16[k], t);
}

// Activation quantiser: same scalar rule as the expert kernel (per-32 scale, int8 in [-127,127], chunk sums).
inline void act_quant(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    q2_expert::act_quant(x, n, q, d, hx);
}

// Integer sum of the 32 codes of one half against a 32-value activation chunk (scalar).
inline int32_t half_dot_scalar(const uint8_t* codes16, uint32_t half, const int8_t* qchunk) {
    uint8_t u[64];
    unpack_block(codes16, u);
    int32_t s = 0;
    for (uint32_t i = 0; i < 32; ++i) s += (int32_t)u[half * 32 + i] * (int32_t)qchunk[i];
    return s;
}

// Row dot over nb GGUF Q2_0 blocks (scalar). Equals sum_b d_b * sum_h xd * (sum c*q - hx).
inline float row_dot_scalar(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        uint16_t h; std::memcpy(&h, row + scale_offset(b), 2);
        const float wd = f16::half_to_float(h);
        const uint8_t* codes = row + codes_offset(b);
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t ch = 2 * b + half;
            const int32_t s = half_dot_scalar(codes, half, q + (size_t)ch * 32);
            acc += wd * xd[ch] * (float)(s - hx[ch]);
        }
    }
    return acc;
}

#if defined(__AVX2__)
// Unpack 16 code bytes into two 32-byte halves in value order.
inline void unpack64_avx2(const uint8_t* codes16, __m256i& lo, __m256i& hi) {
    const __m128i x = _mm_loadu_si128((const __m128i*)codes16);
    const __m128i m = _mm_set1_epi8(3);
    const __m128i v0 = _mm_and_si128(x, m);
    const __m128i v1 = _mm_and_si128(_mm_srli_epi16(x, 2), m);
    const __m128i v2 = _mm_and_si128(_mm_srli_epi16(x, 4), m);
    const __m128i v3 = _mm_and_si128(_mm_srli_epi16(x, 6), m);
    const __m128i ab_lo = _mm_unpacklo_epi8(v0, v1), ab_hi = _mm_unpackhi_epi8(v0, v1);
    const __m128i cd_lo = _mm_unpacklo_epi8(v2, v3), cd_hi = _mm_unpackhi_epi8(v2, v3);
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(ab_lo, cd_lo), _mm_unpacklo_epi16(ab_lo, cd_lo));
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(ab_hi, cd_hi), _mm_unpacklo_epi16(ab_hi, cd_hi));
}

inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int32_t dot32_avx2(__m256i codes, const int8_t* qchunk) {
    const __m256i qv = _mm256_loadu_si256((const __m256i*)qchunk);
    const __m256i p = _mm256_maddubs_epi16(codes, qv);               // exact: 2*3*127 <= 32767
    return hsum_epi32(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

inline float row_dot_avx2(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        uint16_t h; std::memcpy(&h, row + scale_offset(b), 2);
        const float wd = f16::half_to_float(h);
        __m256i lo, hi;
        unpack64_avx2(row + codes_offset(b), lo, hi);
        const uint32_t c0 = 2 * b, c1 = 2 * b + 1;
        acc += wd * xd[c0] * (float)(dot32_avx2(lo, q + (size_t)c0 * 32) - hx[c0]);
        acc += wd * xd[c1] * (float)(dot32_avx2(hi, q + (size_t)c1 * 32) - hx[c1]);
    }
    return acc;
}
#endif

// Best available row dot.
inline float row_dot(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
#if defined(__AVX2__)
    return row_dot_avx2(row, nb, q, xd, hx);
#else
    return row_dot_scalar(row, nb, q, xd, hx);
#endif
}

// Multi-token: tokens in batches of at most four. q stride per token q_stride values; xd/hx stride ch_stride.
inline void row_dot_multi(const uint8_t* row, uint32_t nb, uint32_t nt, const int8_t* q, size_t q_stride,
                          const float* xd, const int32_t* hx, size_t ch_stride, float* out) {
    for (uint32_t t0 = 0; t0 < nt; t0 += batch_size(nt, t0)) {
        const uint32_t bs = batch_size(nt, t0);
        for (uint32_t k = 0; k < bs; ++k) {
            const uint32_t t = t0 + k;
            out[t] = row_dot(row, nb, q + t * q_stride, xd + t * ch_stride, hx + t * ch_stride);
        }
    }
}

// Rows of a matrix (row_stride bytes apart) against nt tokens; out[row * nt + t].
inline void rows(const uint8_t* mat, size_t row_stride, uint32_t n_rows, uint32_t nb, uint32_t nt, const int8_t* q,
                 size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out) {
    for (uint32_t r = 0; r < n_rows; ++r)
        row_dot_multi(mat + r * row_stride, nb, nt, q, q_stride, xd, hx, ch_stride, out + (size_t)r * nt);
}

}  // namespace q2_avx2
}  // namespace baloon

