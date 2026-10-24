// strata/kernels/cpu/expert_kernel.cpp — AVX-512 VNNI Q2_0 expert MLP kernel
// Target: Iluvatar BI-V100 host CPU
//
// Q2_0 × Q8_1 dot product via VPMULTISHIFTQB unpack + VPDPBUSD accumulate.
// Full SwiGLU expert MLP: gate/up → SiLU → requantize → down.

#include "strata/kernels/cpu/expert_kernel.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

// Conditionally include SIMD headers
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#else
// Stub for non-x86 builds (compile-only, not expected to run)
#endif

namespace strata { namespace kernels { namespace cpu {

std::atomic<bool> oracle_mode{false};

// ─── fp16 → float conversion ────────────────────────────────────
float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    // IEEE 754 half-precision decode
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1f;
    const uint32_t man  = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
            // subnormal
            float val = std::ldexp(static_cast<float>(man), -24);
            if (sign) val = -val;
            return val;
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000 | (man << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (man << 13);
    }
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

// ─── Activation quantization (Q8_1, half-away-from-zero) ────────
void act_quant_q8_1(const float* x, int n, ActQ& act) {
    if (oracle_mode.load(std::memory_order_relaxed)) {
        act_quant_q8_0_oracle(x, n, act);
        return;
    }

    act.nchunks = n / QKA;

#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512 halfv = _mm512_set1_ps(0.5f);
    const __m512 mhalfv = _mm512_set1_ps(-0.5f);
    const __m512 zerov = _mm512_setzero_ps();
    const __m512i lov = _mm512_set1_epi32(-127);
    const __m512i hiv = _mm512_set1_epi32(127);
    const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));

    for (int i = 0; i < act.nchunks; ++i) {
        const float* blk = x + i * QKA;
        const __m512 x0 = _mm512_loadu_ps(blk);
        const __m512 x1 = _mm512_loadu_ps(blk + 16);

        const float amax = _mm512_reduce_max_ps(
            _mm512_max_ps(_mm512_and_ps(x0, absmask),
                          _mm512_and_ps(x1, absmask)));

        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m512 vinv = _mm512_set1_ps(inv);

        const __m512 t0 = _mm512_mul_ps(x0, vinv);
        const __m512 t1 = _mm512_mul_ps(x1, vinv);

        // Half-away-from-zero: add ±0.5 based on sign, then truncate
        const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask(t0, zerov, _CMP_GE_OQ), mhalfv, halfv));
        const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask(t1, zerov, _CMP_GE_OQ), mhalfv, halfv));

        __m512i v0 = _mm512_cvttps_epi32(r0);
        __m512i v1 = _mm512_cvttps_epi32(r1);

        v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lov), hiv);
        v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lov), hiv);

        _mm_storeu_si128((__m128i*)(act.q + i * QKA),
                         _mm512_cvtepi32_epi8(v0));
        _mm_storeu_si128((__m128i*)(act.q + i * QKA + 16),
                         _mm512_cvtepi32_epi8(v1));

        const int32_t s32 = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
        act.scale[i] = s;
        act.sum[i] = s32;
        act.hx[i] = s * static_cast<float>(s32);
    }
#else
    // Scalar fallback
    for (int i = 0; i < act.nchunks; ++i) {
        const float* blk = x + i * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) {
            float a = std::fabs(blk[j]);
            if (a > amax) amax = a;
        }
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) {
            float t = blk[j] * inv;
            float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int32_t q = static_cast<int32_t>(r);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            act.q[i * QKA + j] = static_cast<int8_t>(q);
            sum += q;
        }
        act.scale[i] = s;
        act.sum[i] = sum;
        act.hx[i] = s * static_cast<float>(sum);
    }
#endif
}

// ─── Oracle quantizer (Q8_0 rounding, matches ggml exactly) ─────
void act_quant_q8_0_oracle(const float* x, int n, ActQ& act) {
    act.nchunks = n / QKA;
    for (int i = 0; i < act.nchunks; ++i) {
        const float* blk = x + i * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) {
            float a = std::fabs(blk[j]);
            if (a > amax) amax = a;
        }
        const float d = amax / 127.f;
        // Round-to-nearest-even (rint) with double division for Q8_0 compatibility
        const float inv = d > 0.f ? 1.f / d : 0.f;
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) {
            double dv = static_cast<double>(blk[j]) * static_cast<double>(inv);
            int32_t q = static_cast<int32_t>(std::rint(dv));
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            act.q[i * QKA + j] = static_cast<int8_t>(q);
            sum += q;
        }
        act.scale[i] = d;
        act.sum[i] = sum;
        act.hx[i] = d * static_cast<float>(sum);
    }
}

// ─── Q2_0 unpack: 8 bytes → 32 codes via VPMULTISHIFTQB ────────
#if defined(__AVX512VBMI__)
static inline __m256i unpack_q2(const uint8_t* codes) {
    const __m128i packed = _mm_loadl_epi64((const __m128i*)codes);
    const __m256i lanes = _mm256_cvtepu16_epi64(packed);
    const __m256i ctrl = _mm256_set1_epi64x(static_cast<long long>(0x0E0C0A0806040200ULL));
    return _mm256_and_si256(
        _mm256_multishift_epi64_epi8(ctrl, lanes),
        _mm256_set1_epi8(3));
}
#endif

// ─── Q2_0 × Q8_1 block dot via VNNI ────────────────────────────
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
static inline __m256i block_dot_q2(const uint8_t* codes, const int8_t* xq) {
    return _mm256_dpbusd_epi32(
        _mm256_setzero_si256(),
        unpack_q2(codes),
        _mm256_load_si256((const __m256i*)xq));
}
#endif

// ─── Horizontal sum of __m256 ────────────────────────────────────
#if defined(__AVX2__)
static inline float hsum_avx(__m256 v) {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 s = _mm_add_ps(lo, hi);
    const __m128 s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s2, _mm_movehdup_ps(s2)));
}
#endif

// ─── Row dot product (Q2_0 × Q8_1, full row) ────────────────────
float row_dot_q2(const uint8_t* codes, const uint8_t* scales,
                 const ActQ& act, int nblocks) {
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.f;

    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);

        // Each Q2_0 block has 64 values = 2 chunks of 32
        const __m256 slo = _mm256_cvtepi32_ps(
            block_dot_q2(codes + b * 16, act.q + b * QK));
        const __m256 shi = _mm256_cvtepi32_ps(
            block_dot_q2(codes + b * 16 + 8, act.q + b * QK + QKA));

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * act.scale[2 * b]), slo, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * act.scale[2 * b + 1]), shi, acc);
        corr += d * (act.hx[2 * b] + act.hx[2 * b + 1]);
    }

    return hsum_avx(acc) - corr;
#else
    // Scalar fallback
    float result = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        for (int j = 0; j < QK; ++j) {
            // Unpack 2-bit code
            int byte_idx = j / 4;
            int shift = (j % 4) * 2;
            int code = (codes[b * 16 + byte_idx] >> shift) & 3;
            int chunk = j / QKA;
            int elem = j % QKA;
            float x_val = act.scale[2 * b + chunk] *
                          static_cast<float>(act.q[b * QK + j]);
            result += d * static_cast<float>(code - 1) * x_val;
        }
    }
    return result;
#endif
}

// ─── Expert MLP kernel (single token) ───────────────────────────
void expert_kernel(const uint8_t* blob, const ActQ& act,
                   float* out, ExpertScratch& ws) {
    if (oracle_mode.load(std::memory_order_relaxed)) {
        expert_kernel_oracle(blob, act, out, ws);
        return;
    }

    // Derive layout from BLOB constant
    // The Q2_0 blob has interleaved gate/up then down
    // Block structure: 18 bytes per block of 64 values
    const int blocks_gu = static_cast<int>(BLOCKS_PER_ROW);
    const int blocks_d  = static_cast<int>(BLOCKS_PER_DOWN_ROW);

    // Phase 1: gate + up projections → SiLU gating
    for (int r = 0; r < FF; ++r) {
        // Gate row offset: row 2*r within gate/up section
        const size_t g_off = static_cast<size_t>(2 * r) * ROW_GU;
        const size_t u_off = static_cast<size_t>(2 * r + 1) * ROW_GU;

        const float gate = row_dot_q2(
            blob + g_off, blob + g_off, act, blocks_gu);
        const float up = row_dot_q2(
            blob + u_off, blob + u_off, act, blocks_gu);

        // SiLU(gate) * up
        ws.ff[r] = (gate / (1.f + std::exp(-gate))) * up;
    }

    // Requantize intermediate
    act_quant_q8_1(ws.ff, FF, ws.a2);

    // Phase 2: down projection
    const size_t d_base = DOWN_OFF;
    for (int r = 0; r < H; ++r) {
        const size_t off = d_base + static_cast<size_t>(r) * ROW_D;
        out[r] = row_dot_q2(blob + off, blob + off, ws.a2, blocks_d);
    }
}

// ─── Oracle expert kernel ────────────────────────────────────────
void expert_kernel_oracle(const uint8_t* blob, const ActQ& act,
                          float* out, ExpertScratch& ws) {
    // Same structure as expert_kernel but uses oracle quantizer
    const int blocks_gu = static_cast<int>(BLOCKS_PER_ROW);
    const int blocks_d  = static_cast<int>(BLOCKS_PER_DOWN_ROW);

    for (int r = 0; r < FF; ++r) {
        const size_t g_off = static_cast<size_t>(2 * r) * ROW_GU;
        const size_t u_off = static_cast<size_t>(2 * r + 1) * ROW_GU;
        const float gate = row_dot_q2(blob + g_off, blob + g_off, act, blocks_gu);
        const float up   = row_dot_q2(blob + u_off, blob + u_off, act, blocks_gu);
        ws.ff[r] = (gate / (1.f + std::exp(-gate))) * up;
    }
    act_quant_q8_0_oracle(ws.ff, FF, ws.a2);
    const size_t d_base = DOWN_OFF;
    for (int r = 0; r < H; ++r) {
        const size_t off = d_base + static_cast<size_t>(r) * ROW_D;
        out[r] = row_dot_q2(blob + off, blob + off, ws.a2, blocks_d);
    }
}

}}}  // namespace strata::kernels::cpu
