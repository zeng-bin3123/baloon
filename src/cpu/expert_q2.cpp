// expert_q2.cpp - CPU Q2_0 routed-expert kernels. Scalar reference, AVX2 helpers, AVX-512 VNNI paths.
// Built with -mavx512bw -mavx512dq -mavx512f -mavx512vbmi -mavx512vl -mavx512vnni -mf16c -mfma and -ffp-contract=off:
// the vector paths do their integer work in SIMD and the float accumulation in the same order as the scalar
// reference, so results are bit-identical.
#include "baloon/kern_cpu_expert.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace baloon {
namespace kern {

namespace {

constexpr uint32_t kZmm = 64;   // bytes of one 512-bit register (= codes of one Q2_0 block)
std::atomic<bool> oracle_flag{false};
inline bool expert_oracle_q8_0() { return oracle_flag.load(std::memory_order_relaxed); }

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
[[maybe_unused]] inline float hsum_ps(__m512 v) { return _mm512_reduce_add_ps(v); }
inline float silu_f(float v) { return v / (1.f + std::exp(-v)); }

// Eight fp16 scales (nb <= 8 valid, others zero) as floats. The load is masked: a tail never reads past the row.
inline __m256 scales8(const uint8_t* s, uint32_t nb) {
    const __mmask8 m = (__mmask8)(nb >= 8 ? 0xFFu : ((1u << nb) - 1u));
    return _mm256_cvtph_ps(_mm_maskz_loadu_epi16(m, s));
}

// Scalar unpack of one 16-byte block into 64 codes.
inline void unpack_q2_0(const uint8_t* p16, uint8_t out[64]) {
    for (uint32_t j = 0; j < 64; ++j) out[j] = (uint8_t)((p16[j / 4] >> (2 * (j % 4))) & 3u);
}

// 64 codes (one byte each) of a 16-byte block: qword Q of the source holds bytes 2Q, 2Q+1, and output byte p of
// qword Q takes the 8-bit window at bit 8*((p%8)/4) + 2*(p%4); the low two bits of that window are the code.
inline __m512i unpack64_q2_0(const uint8_t* p16) {
    // control byte of output byte p: 8*((p%8)/4) + 2*(p%4)  ->  0,2,4,6,8,10,12,14 repeated per qword
    const __m512i ctrl = _mm512_set1_epi64(0x0E0C0A0806040200LL);
    const __m512i src = _mm512_cvtepu16_epi64(_mm_loadu_si128((const __m128i*)p16));
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, src), _mm512_set1_epi8(3));
}

// Integer dot of one 64-weight block (codes 0..3, no offset) against 64 int8 activations: low and high 32.
inline void block_dot(const uint8_t* codes16, const int8_t* q64, int32_t& lo, int32_t& hi) {
    const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unpack64_q2_0(codes16), _mm512_loadu_si512((const void*)q64));
    lo = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_castsi512_si256(s), 0));
    hi = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_extracti64x4_epi64(s, 1), 0));
}

// ---- scalar reference (blob layout) ------------------------------------------------------------------------
inline float row_dot(const uint8_t* codes, const uint8_t* scales, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        uint8_t c[64];
        unpack_q2_0(codes + (size_t)b * 16, c);
        const float wd = h2f(load_u16(scales + 2 * (size_t)b));
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t ch = 2 * b + h;
            int32_t s = 0;
            for (uint32_t i = 0; i < 32; ++i) s += (int32_t)c[h * 32 + i] * (int32_t)q[(size_t)ch * 32 + i];
            acc += wd * xd[ch] * (float)(s - hx[ch]);
        }
    }
    return acc;
}

inline void row_dot_multi(const uint8_t* codes, const uint8_t* scales, uint32_t nb, uint32_t nt, const ActQ* a, float* out) {
    for (uint32_t t = 0; t < nt; ++t) out[t] = row_dot(codes, scales, nb, a[t].q, a[t].d, a[t].hx);
}

inline float row_dot_z(const uint8_t* codes, const uint8_t* scales, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b0 = 0; b0 < nb; b0 += 8) {
        const uint32_t n = nb - b0 < 8 ? nb - b0 : 8;
        alignas(32) float wd[8];
        _mm256_store_ps(wd, scales8(scales + 2 * (size_t)b0, n));
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t b = b0 + k;
            int32_t lo, hi;
            block_dot(codes + (size_t)b * 16, q + (size_t)b * kZmm, lo, hi);
            acc += wd[k] * xd[2 * b] * (float)(lo - hx[2 * b]);
            acc += wd[k] * xd[2 * b + 1] * (float)(hi - hx[2 * b + 1]);
        }
    }
    return acc;
}

// Multi-token: block codes are unpacked once and shared by all tokens; per-token float order equals row_dot_z.
inline void row_dot_multi_z(const uint8_t* codes, const uint8_t* scales, uint32_t nb, uint32_t nt, const ActQ* a, float* out) {
    float acc[MAXT] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t b0 = 0; b0 < nb; b0 += 8) {
        const uint32_t n = nb - b0 < 8 ? nb - b0 : 8;
        alignas(32) float wd[8];
        _mm256_store_ps(wd, scales8(scales + 2 * (size_t)b0, n));
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t b = b0 + k;
            const __m512i c = unpack64_q2_0(codes + (size_t)b * 16);
            for (uint32_t t = 0; t < nt; ++t) {
                const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c, _mm512_loadu_si512((const void*)(a[t].q + (size_t)b * kZmm)));
                const int32_t lo = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_castsi512_si256(s), 0));
                const int32_t hi = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_extracti64x4_epi64(s, 1), 0));
                acc[t] += wd[k] * a[t].d[2 * b] * (float)(lo - a[t].hx[2 * b]);
                acc[t] += wd[k] * a[t].d[2 * b + 1] * (float)(hi - a[t].hx[2 * b + 1]);
            }
        }
    }
    for (uint32_t t = 0; t < nt; ++t) out[t] = acc[t];
}

// ---- oracle contract (Q8_0 activations: fp16 scale, round half away from zero, clamp [-128, 127]) -------------
void quantize_oracle_q8_0(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    for (uint32_t c = 0; c < n / QKA; ++c) {
        const float* xc = x + (size_t)c * QKA;
        float amax = 0.f;
        for (uint32_t i = 0; i < QKA; ++i) amax = std::fmax(amax, std::fabs(xc[i]));
        const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        int32_t s = 0;
        for (uint32_t i = 0; i < QKA; ++i) {
            const int v = actq::clamp_ggml((int)std::round(xc[i] * id));
            q[(size_t)c * QKA + i] = (int8_t)v;
            s += v;
        }
        d[c] = h2f(_cvtss_sh(dd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        hx[c] = s;
    }
}

inline int32_t chunk_dot_oracle(const uint8_t* codes16, uint32_t h, const int8_t* qchunk) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 32; ++i) {
        const uint32_t j = h * 32 + i;
        s += (int32_t)((codes16[j / 4] >> (2 * (j % 4))) & 3u) * (int32_t)qchunk[i];
    }
    return s;
}

inline float row_dot_oracle(const uint8_t* codes, const uint8_t* scales, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const float wd = h2f(load_u16(scales + 2 * (size_t)b));
        for (uint32_t h = 0; h < 2; ++h) {
            const uint32_t ch = 2 * b + h;
            acc += wd * xd[ch] * (float)(chunk_dot_oracle(codes + (size_t)b * 16, h, q + (size_t)ch * 32) - hx[ch]);
        }
    }
    return acc;
}

inline bool oracle_on() { return expert_oracle_q8_0(); }

inline float rd(const uint8_t* codes, const uint8_t* scales, uint32_t nb, const ActQ& a) {
    return oracle_on() ? row_dot_oracle(codes, scales, nb, a.q, a.d, a.hx) : row_dot_z(codes, scales, nb, a.q, a.d, a.hx);
}
inline void rdm(const uint8_t* codes, const uint8_t* scales, uint32_t nb, uint32_t nt, const ActQ* a, float* out) {
    if (oracle_on()) row_dot_multi(codes, scales, nb, nt, a, out);   // oracle: per token, same order as row_dot_oracle
    else row_dot_multi_z(codes, scales, nb, nt, a, out);
}

inline const uint8_t* gu_codes(const uint8_t* blob, uint32_t slot) { return blob + O_GU_CODES + (size_t)slot * ROW_GU; }
inline const uint8_t* gu_scales(const uint8_t* blob, uint32_t slot) { return blob + O_GU_SCALES + (size_t)slot * SC_GU * 2; }
inline const uint8_t* dn_codes(const uint8_t* blob, uint32_t r) { return blob + O_D_CODES + (size_t)r * ROW_D; }
inline const uint8_t* dn_scales(const uint8_t* blob, uint32_t r) { return blob + O_D_SCALES + (size_t)r * SC_D * 2; }

// Multi-token pieces over row ranges.
void gu_rows_multi(const uint8_t* blob, const ActQ* xq, uint32_t nt, uint32_t r0, uint32_t r1, float (*act)[FF]) {
    for (uint32_t r = r0; r < r1; ++r) {
        float g[MAXT], u[MAXT];
        rdm(gu_codes(blob, 2 * r), gu_scales(blob, 2 * r), SC_GU, nt, xq, g);
        rdm(gu_codes(blob, 2 * r + 1), gu_scales(blob, 2 * r + 1), SC_GU, nt, xq, u);
        for (uint32_t t = 0; t < nt; ++t) act[t][r] = silu_f(g[t]) * u[t];
    }
}
void down_rows_multi(const uint8_t* blob, const ActQ* aq, uint32_t nt, uint32_t r0, uint32_t r1, float* out, size_t out_stride) {
    for (uint32_t r = r0; r < r1; ++r) {
        float o[MAXT];
        rdm(dn_codes(blob, r), dn_scales(blob, r), SC_D, nt, aq, o);
        for (uint32_t t = 0; t < nt; ++t) out[(size_t)t * out_stride + r] = o[t];
    }
}

void expert_multi(const uint8_t* blob, const float* x, uint32_t nt, float* out, ExpertScratchMulti& s) {
    for (uint32_t t = 0; t < nt; ++t) act_quant_q8_1(x + (size_t)t * H, H, s.x[t].q, s.x[t].d, s.x[t].hx);
    s2_expert_gu_rows_multi(blob, s.x, nt, 0, FF, s.act);
    for (uint32_t t = 0; t < nt; ++t) act_quant_q8_1(s.act[t], FF, s.a[t].q, s.a[t].d, s.a[t].hx);
    s2_expert_down_rows_multi(blob, s.a, nt, 0, H, out, H);
}

// ---- GGUF-layout rows --------------------------------------------------------------------------------------------
inline void q2g_rows(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx, float* out) {
    for (uint32_t r = row0; r < row1; ++r) {
        const uint8_t* row = w + (size_t)r * nb * BB;
        float acc = 0.f;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* blk = row + (size_t)b * BB;
            const float wd = h2f(load_u16(blk));
            int32_t lo, hi;
            block_dot(blk + 2, q + (size_t)b * kZmm, lo, hi);
            acc += wd * xd[2 * b] * (float)(lo - hx[2 * b]);
            acc += wd * xd[2 * b + 1] * (float)(hi - hx[2 * b + 1]);
        }
        out[r] = acc;
    }
}
inline void q2g_row_multi(const uint8_t* row, uint32_t nb, uint32_t nt, const int8_t* q, size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out) {
    float acc[MAXT] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t)b * BB;
        const float wd = h2f(load_u16(blk));
        const __m512i c = unpack64_q2_0(blk + 2);
        for (uint32_t t = 0; t < nt; ++t) {
            const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c, _mm512_loadu_si512((const void*)(q + t * q_stride + (size_t)b * kZmm)));
            const int32_t lo = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_castsi512_si256(s), 0));
            const int32_t hi = _mm512_reduce_add_epi32(_mm512_inserti64x4(_mm512_setzero_si512(), _mm512_extracti64x4_epi64(s, 1), 0));
            acc[t] += wd * xd[t * ch_stride + 2 * b] * (float)(lo - hx[t * ch_stride + 2 * b]);
            acc[t] += wd * xd[t * ch_stride + 2 * b + 1] * (float)(hi - hx[t * ch_stride + 2 * b + 1]);
        }
    }
    for (uint32_t t = 0; t < nt; ++t) out[t] = acc[t];
}

}  // namespace

// ============================================================================================ public: features
const char* CpuFeatures::reason() const {
    if (!avx512f) return "AVX-512F is missing";
    if (!avx512bw) return "AVX-512BW is missing";
    if (!avx512vl) return "AVX-512VL is missing";
    if (!avx512_vnni) return "AVX-512 VNNI is missing";
    if (!avx512_vbmi) return "AVX-512 VBMI is missing";
    return "";
}

CpuFeatures cpu_features() {
    CpuFeatures f{};
#if defined(__x86_64__) || defined(__i386__)
    const unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 7) return f;
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(1, a, b, c, d);
    const bool osxsave = (c >> 27) & 1u;
    bool zmm_ok = false, ymm_ok = false;
    if (osxsave) {
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        ymm_ok = (lo & 0x6u) == 0x6u;
        zmm_ok = (lo & 0xE6u) == 0xE6u;
    }
    __cpuid_count(7, 0, a, b, c, d);
    f.avx2 = ymm_ok && ((b >> 5) & 1u);
    f.avx512f = zmm_ok && ((b >> 16) & 1u);
    f.avx512bw = zmm_ok && ((b >> 30) & 1u);
    f.avx512vl = zmm_ok && ((b >> 31) & 1u);
    f.avx512_vnni = zmm_ok && ((c >> 11) & 1u);
    f.avx512_vbmi = zmm_ok && ((c >> 1) & 1u);
#endif
    return f;
}

void cpu_require_expert_support() {
    const CpuFeatures f = cpu_features();
    if (!f.usable()) {
        std::fprintf(stderr, "baloon: the CPU expert kernels need AVX-512 F/BW/VL/VNNI/VBMI: %s\n", f.reason());
        std::abort();
    }
}

void expert_set_oracle_q8_0(bool on) { oracle_flag.store(on, std::memory_order_relaxed); }
bool expert_oracle_q8_0_enabled() { return expert_oracle_q8_0(); }

// ============================================================================================ public: quantization
void act_quant_q8_1(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    if (oracle_on()) { quantize_oracle_q8_0(x, n, q, d, hx); return; }
    for (uint32_t c = 0; c < n / QKA; ++c) {
        const float* xc = x + (size_t)c * QKA;
        float amax = 0.f;
        for (uint32_t i = 0; i < QKA; ++i) amax = std::max(amax, std::fabs(xc[i]));
        const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        int32_t s = 0;
        for (uint32_t i = 0; i < QKA; ++i) {
            const int v = actq::clamp_cpu((int)std::nearbyint(xc[i] * id));
            q[(size_t)c * QKA + i] = (int8_t)v;
            s += v;
        }
        d[c] = dd;
        hx[c] = s;
    }
}

void act_quant_q8_1_avx2(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) {
    if (oracle_on()) { quantize_oracle_q8_0(x, n, q, d, hx); return; }
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (uint32_t c = 0; c < n / QKA; ++c) {
        const float* xc = x + (size_t)c * QKA;
        __m256 v[4], m = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) { v[k] = _mm256_loadu_ps(xc + 8 * k); m = _mm256_max_ps(_mm256_and_ps(v[k], absmask), m); }
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float amax = _mm_cvtss_f32(m4);
        const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
        const __m256 vid = _mm256_set1_ps(id);
        const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
        __m256i qi[4], sum = _mm256_setzero_si256();
        for (int k = 0; k < 4; ++k) {
            qi[k] = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(v[k], vid)), lo), hi);   // RNE
            sum = _mm256_add_epi32(sum, qi[k]);
        }
        __m256i p01 = _mm256_packs_epi32(qi[0], qi[1]), p23 = _mm256_packs_epi32(qi[2], qi[3]);
        __m256i p = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q + (size_t)c * QKA), p);
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        d[c] = dd;
        hx[c] = _mm_cvtsi128_si32(s4);
    }
}

// ============================================================================================ public: GGUF rows
void q2_0_gguf_rows_multi(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                          size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride) {
    if (nt == 1 && !oracle_on()) {
        // single token: write straight into out
        q2g_rows(w, row0, row1, nb, q, xd, hx, out);
        return;
    }
    for (uint32_t r = row0; r < row1; ++r) {
        float o[MAXT];
        for (uint32_t t0 = 0; t0 < nt; t0 += MAXT) {
            const uint32_t n = std::min<uint32_t>(MAXT, nt - t0);
            q2g_row_multi(w + (size_t)r * nb * BB, nb, n, q + t0 * q_stride, q_stride, xd + t0 * ch_stride, hx + t0 * ch_stride, ch_stride, o);
            for (uint32_t t = 0; t < n; ++t) out[(size_t)(t0 + t) * out_stride + r] = o[t];
        }
    }
}

// AVX2 form: 16 code bytes -> 64 codes via four shifted/masked vectors interleaved with unpack; maddubs + madd dots.
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q,
                               size_t q_stride, const float* xd, const int32_t* hx, size_t ch_stride, float* out,
                               size_t out_stride) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m128i m3 = _mm_set1_epi8(3);
    for (uint32_t r = row0; r < row1; ++r) {
        const uint8_t* row = w + (size_t)r * nb * BB;
        for (uint32_t t0 = 0; t0 < nt; t0 += MAXT) {
            const uint32_t n = std::min<uint32_t>(MAXT, nt - t0);
            float acc[MAXT] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t b = 0; b < nb; ++b) {
                const uint8_t* blk = row + (size_t)b * BB;
                const float wd = h2f(load_u16(blk));
                const __m128i v = _mm_loadu_si128((const __m128i*)(blk + 2));
                const __m128i t0v = _mm_and_si128(v, m3), t1v = _mm_and_si128(_mm_srli_epi16(v, 2), m3);
                const __m128i t2v = _mm_and_si128(_mm_srli_epi16(v, 4), m3), t3v = _mm_and_si128(_mm_srli_epi16(v, 6), m3);
                const __m128i u01l = _mm_unpacklo_epi8(t0v, t1v), u01h = _mm_unpackhi_epi8(t0v, t1v);
                const __m128i u23l = _mm_unpacklo_epi8(t2v, t3v), u23h = _mm_unpackhi_epi8(t2v, t3v);
                const __m256i clo = _mm256_set_m128i(_mm_unpackhi_epi16(u01l, u23l), _mm_unpacklo_epi16(u01l, u23l));
                const __m256i chi = _mm256_set_m128i(_mm_unpackhi_epi16(u01h, u23h), _mm_unpacklo_epi16(u01h, u23h));
                for (uint32_t t = 0; t < n; ++t) {
                    const int8_t* qb = q + (t0 + t) * q_stride + (size_t)b * 64;
                    const __m256i slo = _mm256_madd_epi16(_mm256_maddubs_epi16(clo, _mm256_loadu_si256((const __m256i*)qb)), ones);
                    const __m256i shi = _mm256_madd_epi16(_mm256_maddubs_epi16(chi, _mm256_loadu_si256((const __m256i*)(qb + 32))), ones);
                    auto hs = [](__m256i s) {
                        __m128i a = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));
                        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1));
                        return _mm_cvtsi128_si32(a);
                    };
                    const size_t ch = (size_t)(t0 + t) * ch_stride + 2 * b;
                    acc[t] += wd * xd[ch] * (float)(hs(slo) - hx[ch]);
                    acc[t] += wd * xd[ch + 1] * (float)(hs(shi) - hx[ch + 1]);
                }
            }
            for (uint32_t t = 0; t < n; ++t) out[(size_t)(t0 + t) * out_stride + r] = acc[t];
        }
    }
}

// ============================================================================================ public: expert paths
void s2_expert_scalar(const uint8_t* blob, const float* x, float* out) {
    ActQ xq, aq;
    float act[FF];
    act_quant_q8_1(x, H, xq.q, xq.d, xq.hx);
    for (uint32_t r = 0; r < FF; ++r) {
        const float g = oracle_on() ? row_dot_oracle(gu_codes(blob, 2 * r), gu_scales(blob, 2 * r), SC_GU, xq.q, xq.d, xq.hx)
                                    : row_dot(gu_codes(blob, 2 * r), gu_scales(blob, 2 * r), SC_GU, xq.q, xq.d, xq.hx);
        const float u = oracle_on() ? row_dot_oracle(gu_codes(blob, 2 * r + 1), gu_scales(blob, 2 * r + 1), SC_GU, xq.q, xq.d, xq.hx)
                                    : row_dot(gu_codes(blob, 2 * r + 1), gu_scales(blob, 2 * r + 1), SC_GU, xq.q, xq.d, xq.hx);
        act[r] = silu_f(g) * u;
    }
    act_quant_q8_1(act, FF, aq.q, aq.d, aq.hx);
    for (uint32_t r = 0; r < H; ++r)
        out[r] = oracle_on() ? row_dot_oracle(dn_codes(blob, r), dn_scales(blob, r), SC_D, aq.q, aq.d, aq.hx)
                             : row_dot(dn_codes(blob, r), dn_scales(blob, r), SC_D, aq.q, aq.d, aq.hx);
}

void s2_expert_gu_rows(const uint8_t* blob, const ActQ& xq, uint32_t r0, uint32_t r1, float* act) {
    for (uint32_t r = r0; r < r1; ++r) {
        const float g = rd(gu_codes(blob, 2 * r), gu_scales(blob, 2 * r), SC_GU, xq);
        const float u = rd(gu_codes(blob, 2 * r + 1), gu_scales(blob, 2 * r + 1), SC_GU, xq);
        act[r] = silu_f(g) * u;
    }
}
void s2_expert_down_rows(const uint8_t* blob, const ActQ& aq, uint32_t r0, uint32_t r1, float* out) {
    for (uint32_t r = r0; r < r1; ++r) out[r] = rd(dn_codes(blob, r), dn_scales(blob, r), SC_D, aq);
}
void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* xq, uint32_t nt, uint32_t r0, uint32_t r1, float (*act)[FF]) {
    gu_rows_multi(blob, xq, nt, r0, r1, act);
}
void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* aq, uint32_t nt, uint32_t r0, uint32_t r1, float* out, size_t out_stride) {
    down_rows_multi(blob, aq, nt, r0, r1, out, out_stride);
}

void s2_expert_vnni_q(const uint8_t* blob, const ActQ& xq, float* out, ExpertScratch& s) {
    s2_expert_gu_rows(blob, xq, 0, FF, s.act);
    act_quant_q8_1(s.act, FF, s.a.q, s.a.d, s.a.hx);
    s2_expert_down_rows(blob, s.a, 0, H, out);
}
void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& s) {
    act_quant_q8_1_avx2(x, H, s.x.q, s.x.d, s.x.hx);
    s2_expert_vnni_q(blob, s.x, out, s);
}
void s2_expert_vnni_multi(const uint8_t* blob, const float* x, uint32_t nt, float* out, ExpertScratchMulti& s) {
    if (nt == 0) return;
    if (nt > MAXT || oracle_on()) {   // outside the dispatch ladder / oracle contract: token by token
        for (uint32_t t = 0; t < nt; ++t) {
            ExpertScratch* one = new ExpertScratch;
            s2_expert_vnni(blob, x + (size_t)t * H, out + (size_t)t * H, *one);
            delete one;
        }
        return;
    }
    expert_multi(blob, x, nt, out, s);
}

}  // namespace kern
}  // namespace baloon

