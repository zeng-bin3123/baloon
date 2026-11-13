// q2_avx2.cpp - AVX2 Q2_0 GGUF rows and the Q8_1-style activation quantiser (reference layout). Build with -mavx2 -mf16c -mfma.
#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "baloon/q2_avx2_fallback.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/u11_expert.hpp"

namespace baloon {
namespace u11 {
namespace q2avx2 {

using namespace baloon::q2rows;

namespace {
inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    static_assert(baloon::q2avx2_model::shuffle_constants(), "shuffle_constants");
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, baloon::q2avx2_model::kSwapHalves));  // lanes [2,3,0,1]
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, baloon::q2avx2_model::kSwapPairs));   // lanes [1,0,3,2]
    return _mm_cvtsi128_si32(s);
}
// Theory shift16_bytes_independent: shifting the 16-bit lanes right by 2k and masking every byte with 3 yields code k
// of EACH byte on its own - the bits the shift moves in from the neighbouring byte land above bit 1 and are masked off.
template <int K> inline __m128i code_plane(__m128i x) {
    static_assert(K >= 0 && K < 4, "a byte holds four 2-bit codes");
    static_assert(baloon::q2avx2_model::shift16_independent(), "shift16_bytes_independent");
    return _mm_and_si128(_mm_srli_epi16(x, 2 * K), _mm_set1_epi8((char)baloon::q2avx2_model::kCodeMask));
}
// 16 code bytes -> 64 two-bit codes, one per byte, in value order (lo: values 0..31, hi: 32..63).
// The epi8 then epi16 interleave puts value i < 16 at byte i/4, slot i%4 (q2avx2_model::unpack_pos_order).
inline void unpack64_v(const uint8_t* c16, __m256i& lo, __m256i& hi) {
    static_assert(baloon::q2avx2_model::unpack_pos_order(), "unpackPos_order");
    const __m128i x = _mm_loadu_si128((const __m128i*)c16);
    const __m128i v0 = code_plane<0>(x), v1 = code_plane<1>(x), v2 = code_plane<2>(x), v3 = code_plane<3>(x);
    const __m128i ab_lo = _mm_unpacklo_epi8(v0, v1), ab_hi = _mm_unpackhi_epi8(v0, v1);
    const __m128i cd_lo = _mm_unpacklo_epi8(v2, v3), cd_hi = _mm_unpackhi_epi8(v2, v3);
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(ab_lo, cd_lo), _mm_unpacklo_epi16(ab_lo, cd_lo));
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(ab_hi, cd_hi), _mm_unpacklo_epi16(ab_hi, cd_hi));
}
// Exact: codes 0..3 times int8 activations, pair sums <= 2*3*127, so maddubs cannot saturate.
inline int32_t dot32(__m256i codes, const int8_t* qc) {
    static_assert(baloon::q2avx2_model::kMaddubsPairMax <= 32767, "maddubs_codes_bound");
    const __m256i p = _mm256_maddubs_epi16(codes, _mm256_loadu_si256((const __m256i*)qc));
    return hsum_i32(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
}  // namespace

// All 64 values in dequantizer order: group g of the four epi16 unpacks holds values 16g..16g+15 (unpack_groups_cover).
void unpack64(const uint8_t* codes16, uint8_t out[64]) {
    static_assert(baloon::q2avx2_model::unpack_groups_cover(), "unpack_groups_cover");
    __m256i lo, hi;
    unpack64_v(codes16, lo, hi);
    _mm256_storeu_si256((__m256i*)out, lo);
    _mm256_storeu_si256((__m256i*)(out + 32), hi);
}

// Integer sums S0 (values 0..31) and S1 (32..63) of one GGUF block's codes against 64 int8 activations (exact).
void block_sums(const uint8_t* blk, const int8_t* qb, int32_t S[2]) {
    __m256i lo, hi;
    unpack64_v(blk + baloon::q2avx2_model::kCodesOff, lo, hi);
    S[0] = dot32(lo, qb);
    S[1] = dot32(hi, qb + CHUNK);
}

// Per block: d_w * (d_x0 (S0 - hx0) + d_x1 (S1 - hx1)) - the corrected form of d (s0 S0 + s1 S1) - d (hx0 + hx1).
void row_multi(const uint8_t* row, uint32_t nb, uint32_t n_tok, const int8_t* q, size_t q_stride, const float* xd,
               const int32_t* hx, size_t ch_stride, float* out) {
    for (uint32_t t = 0; t < n_tok; ++t) {
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; ++b) {
            static_assert(baloon::q2avx2_model::block_scale_off(1) == BLOCK_BYTES && baloon::q2avx2_model::gguf_block_stride(64), "gguf_block_stride");
            const uint8_t* blk = row + baloon::q2avx2_model::block_scale_off(b);  // scale at 18 b, codes at 18 b + 2
            const float wd = h2f(load_u16(blk));
            const int8_t* qb = q + (size_t)t * q_stride + (size_t)b * BLOCK_VALUES;
            const size_t c0 = (size_t)t * ch_stride + chunk_of(b, 0), c1 = (size_t)t * ch_stride + chunk_of(b, 32);
            static_assert(baloon::q2avx2_model::block_correction_ok(), "blockResult_correction");
            int32_t S[2];
            block_sums(blk, qb, S);
            const int32_t s0 = S[0] - hx[c0], s1 = S[1] - hx[c1];
            acc = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(wd * xd[c0]), _mm_set_ss((float)s0), _mm_set_ss(acc)));
            acc = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(wd * xd[c1]), _mm_set_ss((float)s1), _mm_set_ss(acc)));
        }
        out[t] = acc;
    }
}

void rows(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx, float* out) {
    for (uint32_t r = row0; r < row1; ++r) row_multi(w + (size_t)r * baloon::q2avx2_model::row_bytes(nb), nb, 1, q, 0, xd, hx, 0, out + (r - row0));
}

// Integer horizontal sum of 8 int32 lanes: fold 256 -> 128 bits, then the two shuffles (hsum4_exact) leave the sum in lane 0.
int32_t hsum8(const int32_t v[8]) {
    static_assert(baloon::q2avx2_model::hsum4(1, 2, 3, 4) == 10, "hsum4_exact");
    return hsum_i32(_mm256_loadu_si256((const __m256i*)v));
}

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                               size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride) {
    for (uint32_t r = row0; r < row1; ++r) {
        const uint8_t* row = w + (size_t)r * baloon::q2avx2_model::row_bytes(nb);
        static_assert(baloon::q2avx2_model::kMaxGroup == MAX_GROUP && baloon::q2avx2_model::groups_of4_cover(64), "groupsOf4_cover");
        for (uint32_t t0 = 0; t0 < nt; t0 += MAX_GROUP) {
            const uint32_t n = baloon::q2avx2_model::token_group(nt, t0);  // 1..4, the groups cover all nt tokens
            float o[MAX_GROUP];
            row_multi(row, nb, n, q + t0 * q_stride, q_stride, xd + t0 * ch_stride, hx + t0 * ch_stride, ch_stride, o);
            for (uint32_t t = 0; t < n; ++t) out[(size_t)(t0 + t) * out_stride + r] = o[t];
        }
    }
}

// Round half away from zero on 8 lanes (exact: the fractional part of t - trunc(t) is exactly representable).
static inline __m256i cvt_rha_epi32(__m256 t) {
    const __m256 sign = _mm256_set1_ps(-0.f);
    const __m256 tr = _mm256_round_ps(t, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 fr = _mm256_andnot_ps(sign, _mm256_sub_ps(t, tr));
    const __m256 ge = _mm256_cmp_ps(fr, _mm256_set1_ps(0.5f), _CMP_GE_OQ);
    const __m256 one = _mm256_or_ps(_mm256_set1_ps(1.f), _mm256_and_ps(t, sign));
    return _mm256_cvttps_epi32(_mm256_add_ps(tr, _mm256_and_ps(ge, one)));
}

// Per 32-value chunk: d = amax/127, q = clamp(round(x/d), -127, 127) (round half away from zero), hx = sum of the codes.
void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    static_assert(baloon::q2avx2_model::kChunk == CHUNK && baloon::q2avx2_model::kChunkSumMax == 4064, "chunk_sum_bound");
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (uint32_t c = 0; c < n / CHUNK; ++c) {
        const float* xc = x + (size_t)c * CHUNK;
        __m256 v[4], m = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) { v[k] = _mm256_loadu_ps(xc + 8 * k); m = _mm256_max_ps(_mm256_and_ps(v[k], absmask), m); }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float dd = _mm_cvtss_f32(m4) / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        const __m256 vid = _mm256_set1_ps(id);
        __m256i qi[4], sum = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            qi[k] = _mm256_min_epi32(_mm256_max_epi32(cvt_rha_epi32(_mm256_mul_ps(v[k], vid)), lo), hi);
            sum = _mm256_add_epi32(sum, qi[k]);
        }
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(qi[0], qi[1]), _mm256_packs_epi32(qi[2], qi[3]));
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q + (size_t)c * CHUNK), p);
        d[c] = dd;
        hx[c] = hsum_i32(sum);
    }
}

}  // namespace q2avx2
}  // namespace u11
}  // namespace baloon
