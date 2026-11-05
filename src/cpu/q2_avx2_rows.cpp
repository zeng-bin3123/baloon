// q2_avx2_rows.cpp - AVX2 Q2_0 GGUF rows + activation quantizer (no AVX-512). Build with -mavx2 -mf16c -mfma.
#include "baloon/q2_avx2_rows.hpp"

#include <immintrin.h>

#include <cstring>

namespace baloon {
namespace q2rows {
namespace {

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }

inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

// 16 code bytes -> lo (values 0..31) and hi (values 32..63), one code per byte, value order.
inline void unpack64(const uint8_t* codes16, __m256i& lo, __m256i& hi) {
    const __m128i x = _mm_loadu_si128((const __m128i*)codes16);
    const __m128i m = _mm_set1_epi8(3);
    const __m128i v0 = _mm_and_si128(x, m), v1 = _mm_and_si128(_mm_srli_epi16(x, 2), m);
    const __m128i v2 = _mm_and_si128(_mm_srli_epi16(x, 4), m), v3 = _mm_and_si128(_mm_srli_epi16(x, 6), m);
    const __m128i ab_lo = _mm_unpacklo_epi8(v0, v1), ab_hi = _mm_unpackhi_epi8(v0, v1);
    const __m128i cd_lo = _mm_unpacklo_epi8(v2, v3), cd_hi = _mm_unpackhi_epi8(v2, v3);
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(ab_lo, cd_lo), _mm_unpacklo_epi16(ab_lo, cd_lo));
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(ab_hi, cd_hi), _mm_unpacklo_epi16(ab_hi, cd_hi));
}

// Exact integer dot of 32 codes (0..3) with 32 int8 activations: 2*3*127 <= 32767, so maddubs cannot saturate.
inline int32_t dot32(__m256i codes, const int8_t* qchunk) {
    const __m256i p = _mm256_maddubs_epi16(codes, _mm256_loadu_si256((const __m256i*)qchunk));
    return hsum_i32(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

// n tokens (compile-time) against one row.
template <int N>
inline void row_n(const uint8_t* row, uint32_t nb, const int8_t* q, size_t q_stride, const float* xd, const int32_t* hx,
                  size_t ch_stride, float* out) {
    float acc[N];
    for (int t = 0; t < N; ++t) acc[t] = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + block_start(b);
        const float wd = h2f(load_u16(blk));
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < N; ++t) {
            const int8_t* qb = q + (size_t)t * q_stride + (size_t)b * BLOCK_VALUES;
            const size_t c0 = (size_t)t * ch_stride + chunk_of(b, 0), c1 = (size_t)t * ch_stride + chunk_of(b, 32);
            const int32_t s0 = dot32(lo, qb) - hx[c0];
            const int32_t s1 = dot32(hi, qb + CHUNK) - hx[c1];
            acc[t] = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(wd * xd[c0]), _mm_set_ss((float)s0), _mm_set_ss(acc[t])));
            acc[t] = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(wd * xd[c1]), _mm_set_ss((float)s1), _mm_set_ss(acc[t])));
        }
    }
    for (int t = 0; t < N; ++t) out[t] = acc[t];
}

}  // namespace

bool built_with_required_isa() {
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
    return true;
#else
    return false;
#endif
}

void activation_quantizer(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (uint32_t c = 0; c < n / CHUNK; ++c) {
        const float* xc = x + (size_t)c * CHUNK;
        __m256 v[4], m = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) { v[k] = _mm256_loadu_ps(xc + 8 * k); m = _mm256_max_ps(_mm256_and_ps(v[k], absmask), m); }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float amax = _mm_cvtss_f32(m4);
        const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i qi[4], sum = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            qi[k] = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v[k], vid)), lo), hi);  // RNE, clamp
            sum = _mm256_add_epi32(sum, qi[k]);
        }
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(qi[0], qi[1]), _mm256_packs_epi32(qi[2], qi[3]));
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q + (size_t)c * CHUNK), p);
        d[c] = dd;
        hx[c] = hsum_i32(sum);
    }
}

void unpack_codes_64(const uint8_t* codes16, uint8_t out[64]) {
    __m256i lo, hi;
    unpack64(codes16, lo, hi);
    _mm256_storeu_si256((__m256i*)out, lo);
    _mm256_storeu_si256((__m256i*)(out + 32), hi);
}

void multi_token_row_dot(const uint8_t* row, uint32_t nb, uint32_t n_tok, const int8_t* q, size_t q_stride,
                         const float* xd, const int32_t* hx, size_t ch_stride, float* out) {
    switch (n_tok) {
        case 1: row_n<1>(row, nb, q, q_stride, xd, hx, ch_stride, out); break;
        case 2: row_n<2>(row, nb, q, q_stride, xd, hx, ch_stride, out); break;
        case 3: row_n<3>(row, nb, q, q_stride, xd, hx, ch_stride, out); break;
        case 4: row_n<4>(row, nb, q, q_stride, xd, hx, ch_stride, out); break;
        default: break;  // callers pass 1..4 (group_size)
    }
}

void row_range(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, const int8_t* q, const float* xd,
               const int32_t* hx, float* out) {
    for (uint32_t r = row0; r < row1; ++r)
        row_n<1>(w + (size_t)r * nb * BLOCK_BYTES, nb, q, 0, xd, hx, 0, out + (r - row0));
}

void gguf_rows_multi(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                     size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride) {
    for (uint32_t r = row0; r < row1; ++r) {
        const uint8_t* row = w + (size_t)r * nb * BLOCK_BYTES;
        for (uint32_t t0 = 0; t0 < nt; t0 += MAX_GROUP) {
            const uint32_t n = group_size(nt, t0);
            float o[MAX_GROUP];
            multi_token_row_dot(row, nb, n, q + t0 * q_stride, q_stride, xd + t0 * ch_stride, hx + t0 * ch_stride, ch_stride, o);
            for (uint32_t t = 0; t < n; ++t) out[(size_t)(t0 + t) * out_stride + r] = o[t];
        }
    }
}

}  // namespace q2rows
}  // namespace baloon
