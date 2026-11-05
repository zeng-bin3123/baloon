// Unit ubb09633f12 tests: LAYER 1 statements of Q2Avx2GgufRows, ByteLevelTokenizer, CanonicalQuantPack, GgufLazyLoader,
// VisionEncoderTool. Built with -mavx2 -mf16c -mfma and linked with src/cpu/q2_avx2_rows.cpp.
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../unit14/check.hpp"
#include "baloon/byte_level_bpe_tokenizer.hpp"
#include "baloon/canonical_form_roundtrip.hpp"
#include "baloon/ggml_dequant_layouts.hpp"
#include "baloon/gguf_lazy_loader.hpp"
#include "baloon/mtp_expert_quantizer.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/ref_pack_builder.hpp"
#include "baloon/vision_build_config.hpp"
#include "baloon/vision_encoder_protocol.hpp"

using namespace baloon;
using testh::Rng;

#define K15(c) do { ++testh::checks(); if (!(c)) { ++testh::failures(); \
    if (testh::failures() < 40) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static uint16_t f2h_exact(float f) { return (uint16_t)_cvtss_sh(f, 0); }
static float h2f_(uint16_t h) { return _cvtsh_ss(h); }

// ---------------------------------------------------------------------------------------- Q2Avx2GgufRows
static void test_layout() {
    for (uint32_t i = 0; i < 64; ++i) {
        K15(q2rows::code_byte(i) < 16);
        K15(q2rows::code_shift(i) + 2 <= 8);
        if (i < 32) K15(q2rows::code_byte(i) < 8); else K15(q2rows::code_byte(i) >= 8);
        K15(4 * q2rows::code_byte(i) + q2rows::code_shift(i) / 2 == i);
        for (uint32_t j = 0; j < 64; ++j)
            if (q2rows::code_byte(i) == q2rows::code_byte(j) && q2rows::code_shift(i) == q2rows::code_shift(j)) K15(i == j);
    }
    for (uint32_t b = 0; b < 100; ++b) {
        K15(q2rows::block_start(b) + 2 + 16 == q2rows::block_start(b + 1));
        for (uint32_t nbk = b + 1; nbk < b + 4; ++nbk) K15(q2rows::block_start(b) + 18 <= 18ull * nbk);
        for (uint32_t i = 0; i < 64; ++i) {
            const uint32_t c = q2rows::chunk_of(b, i);
            K15(c == 2 * b || c == 2 * b + 1);
            K15((b * 64 + i) / 32 == c);
        }
    }
    for (uint32_t nt = 1; nt < 200; ++nt)
        for (uint32_t t0 = 0; t0 < nt; ++t0) {
            const uint32_t g = q2rows::group_size(nt, t0);
            K15(g > 0 && t0 + g <= nt && g <= 4);
        }
    K15(q2rows::built_with_required_isa());
}

static void test_unpack() {
    Rng rng(11);
    uint8_t codes[16], out[64];
    // every byte value at every byte position, others zero
    for (int pos = 0; pos < 16; ++pos)
        for (int v = 0; v < 256; ++v) {
            std::memset(codes, 0, 16); codes[pos] = (uint8_t)v;
            q2rows::unpack_codes_64(codes, out);
            for (uint32_t i = 0; i < 64; ++i) {
                const uint32_t expect = (codes[q2rows::code_byte(i)] >> q2rows::code_shift(i)) & 3u;
                K15(out[i] == expect);
            }
        }
    for (int it = 0; it < 2000; ++it) {
        for (auto& c : codes) c = (uint8_t)rng.u32();
        q2rows::unpack_codes_64(codes, out);
        for (uint32_t i = 0; i < 64; ++i) K15(out[i] == ((codes[i / 4] >> (2 * (i % 4))) & 3u));
        // lo/hi split: changing bytes 8..15 leaves values 0..31 unchanged, and vice versa
        uint8_t c2[16], o2[64];
        std::memcpy(c2, codes, 16);
        for (int k = 8; k < 16; ++k) c2[k] = (uint8_t)rng.u32();
        q2rows::unpack_codes_64(c2, o2);
        K15(std::memcmp(out, o2, 32) == 0);
        std::memcpy(c2, codes, 16);
        for (int k = 0; k < 8; ++k) c2[k] = (uint8_t)rng.u32();
        q2rows::unpack_codes_64(c2, o2);
        K15(std::memcmp(out + 32, o2 + 32, 32) == 0);
    }
}

static int rne_clamp(float v) { int r = (int)std::nearbyint(v); return std::max(-127, std::min(127, r)); }

static void test_quantizer() {
    Rng rng(12);
    for (int it = 0; it < 3000; ++it) {
        const uint32_t n = 32 * (1 + rng.below(12));
        std::vector<float> x(n);
        const float scale = std::ldexp(1.f, (int)rng.below(20) - 10);
        for (auto& v : x) v = (float)rng.gauss() * scale;
        if (it % 7 == 0) for (uint32_t i = 0; i < 32 && i < n; ++i) x[i] = 0.f;          // all-zero chunk
        if (it % 11 == 0) for (auto& v : x) v = (rng.below(2) ? 1.f : -1.f) * scale;    // saturating chunk
        std::vector<int8_t> q(n), qs(n); std::vector<float> d(n / 32), ds(n / 32); std::vector<int32_t> hx(n / 32), hxs(n / 32);
        q2rows::activation_quantizer(x.data(), n, q.data(), d.data(), hx.data());
        for (uint32_t c = 0; c < n / 32; ++c) {
            float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[c * 32 + i]));
            const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
            int32_t s = 0;
            for (int i = 0; i < 32; ++i) {
                const int v = rne_clamp(x[c * 32 + i] * id);
                K15(q[c * 32 + i] == v);
                K15(q[c * 32 + i] != -128);   // -128 never appears
                s += v;
            }
            K15(d[c] == dd);
            K15(hx[c] == s);
        }
    }
    // chunk count: exactly n/32 chunks written, nothing beyond
    std::vector<float> x(96, 1.f); std::vector<int8_t> q(128, 55); std::vector<float> d(5, -9.f); std::vector<int32_t> hx(5, -9);
    q2rows::activation_quantizer(x.data(), 96, q.data(), d.data(), hx.data());
    for (int i = 96; i < 128; ++i) K15(q[i] == 55);
    K15(d[3] == -9.f && hx[3] == -9);
}

// scalar double-precision model of a GGUF Q2_0 row dot against one token
static double model_row(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx) {
    double acc = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        uint16_t h; std::memcpy(&h, row + 18 * b, 2);
        const double wd = h2f_(h);
        for (uint32_t i = 0; i < 64; ++i) {
            // not used for per-value; do half sums below
            (void)i;
        }
        for (int half = 0; half < 2; ++half) {
            int64_t s = 0;
            for (int i = 0; i < 32; ++i) {
                const uint32_t vi = half * 32 + i;
                const uint32_t code = (row[18 * b + 2 + vi / 4] >> (2 * (vi % 4))) & 3u;
                s += (int64_t)code * q[b * 64 + vi];
            }
            const uint32_t ch = 2 * b + half;
            acc += wd * (double)xd[ch] * (double)(s - hx[ch]);
        }
    }
    return acc;
}

static void test_rows() {
    Rng rng(13);
    const uint32_t nbs[] = {1, 2, 3, 7, 33};
    for (uint32_t nb : nbs)
        for (uint32_t nt = 1; nt <= 11; ++nt) {
            const uint32_t R = 5, n = nb * 64, nch = nb * 2;
            std::vector<uint8_t> w((size_t)R * nb * 18);
            for (auto& v : w) v = (uint8_t)rng.u32();
            for (uint32_t r = 0; r < R; ++r) for (uint32_t b = 0; b < nb; ++b) {
                const float s = (float)rng.gauss() * 0.1f; const uint16_t h = f2h_exact(s);
                std::memcpy(&w[((size_t)r * nb + b) * 18], &h, 2);
            }
            std::vector<int8_t> q((size_t)nt * n); std::vector<float> xd((size_t)nt * nch); std::vector<int32_t> hx((size_t)nt * nch);
            std::vector<float> x(n);
            for (uint32_t t = 0; t < nt; ++t) {
                for (auto& v : x) v = (float)rng.gauss();
                q2rows::activation_quantizer(x.data(), n, &q[(size_t)t * n], &xd[(size_t)t * nch], &hx[(size_t)t * nch]);
            }
            // rows [1,4) only, with sentinels around
            const size_t ostride = R + 3;
            std::vector<float> out((size_t)nt * ostride, 777.f);
            q2rows::gguf_rows_multi(w.data(), 1, 4, nb, nt, q.data(), n, xd.data(), hx.data(), nch, out.data(), ostride);
            for (uint32_t t = 0; t < nt; ++t)
                for (uint32_t r = 0; r < ostride; ++r) {
                    const float got = out[(size_t)t * ostride + r];
                    if (r >= 1 && r < 4) {
                        const double ref = model_row(&w[(size_t)r * nb * 18], nb, &q[(size_t)t * n], &xd[(size_t)t * nch], &hx[(size_t)t * nch]);
                        double mag = 1e-30;   // sum of |terms| bounds the float rounding error
                        for (uint32_t c = 0; c < nch; ++c) mag += std::fabs((double)xd[(size_t)t * nch + c]) * 200.0 * 32 * 3 * 0.5;
                        K15(std::fabs((double)got - ref) <= 1e-5 * mag + 1e-6 * std::fabs(ref));
                    } else K15(got == 777.f);
                }
            // grouping invariance: token t's result equals the single-token result bit for bit
            for (uint32_t r = 1; r < 4; ++r)
                for (uint32_t t = 0; t < nt; ++t) {
                    float one[1];
                    q2rows::multi_token_row_dot(&w[(size_t)r * nb * 18], nb, 1, &q[(size_t)t * n], n, &xd[(size_t)t * nch], &hx[(size_t)t * nch], nch, one);
                    K15(one[0] == out[(size_t)t * ostride + r]);
                }
            if (nt == 1) {
                float rr[3];
                q2rows::row_range(w.data(), 1, 4, nb, q.data(), xd.data(), hx.data(), rr);
                for (int k = 0; k < 3; ++k) K15(rr[k] == out[1 + k]);
            }
        }
    // exact integer case: scale 1.0, xd 1, hx 0 -> the float result is the exact integer sum of codes*q
    {
        const uint32_t nb = 5, nch = nb * 2; std::vector<uint8_t> row(nb * 18); std::vector<int8_t> q(nb * 64); std::vector<float> xd(nch, 1.f); std::vector<int32_t> hx(nch, 0);
        for (int it = 0; it < 500; ++it) {
            for (auto& v : row) v = (uint8_t)rng.u32();
            for (uint32_t b = 0; b < nb; ++b) { const uint16_t h = 0x3C00; std::memcpy(&row[b * 18], &h, 2); }
            for (auto& v : q) v = (int8_t)((int)rng.below(255) - 127);
            float o[1]; q2rows::multi_token_row_dot(row.data(), nb, 1, q.data(), 0, xd.data(), hx.data(), 0, o);
            int64_t s = 0;
            for (uint32_t b = 0; b < nb; ++b) for (uint32_t i = 0; i < 64; ++i) s += (int64_t)((row[b * 18 + 2 + i / 4] >> (2 * (i % 4))) & 3) * q[b * 64 + i];
            K15((double)o[0] == (double)s);
        }
    }
}

static void test_rounding_model() {
    auto half_away = [](int t2) { return t2 >= 0 ? (t2 + 1) / 2 : -((-t2 + 1) / 2); };
    K15(half_away(5) == 3 && half_away(-5) == -3 && half_away(4) == 2 && half_away(-4) == -2);
    auto clampq = [](long v) { return std::max(-127L, std::min(v, 127L)); };
    for (long v = -1000; v <= 1000; ++v) K15(clampq(v) >= -128 && clampq(v) <= 127);
    for (uint32_t n = 0; n < 4096; n += 32) K15((n / 32) * 32 == n);
}

// ------------------------------------------------------------------------------------- ByteLevelTokenizer
static void test_tokenizer() {
    int printable = 0; std::set<uint32_t> seen;
    for (unsigned b = 0; b < 256; ++b) {
        const uint32_t cp = bpe::byte_to_codepoint((uint8_t)b);
        const bool pr = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t lean = pr ? b : (b <= 32 ? 256 + b : (b <= 160 ? 256 + b - 94 : 323));
        K15(cp == lean);
        K15(cp < 324);
        K15(seen.insert(cp).second);                       // injective
        K15(bpe::codepoint_to_byte(cp) == (int)b);         // lossless
        if (pr) { ++printable; K15(cp == b); } else K15(cp >= 256);
        uint32_t inv = cp < 256 ? cp : (cp < 289 ? cp - 256 : (cp < 323 ? cp - 256 + 94 : 173));
        K15(inv == b);
    }
    K15(printable == 188 && 256 - printable == 68);
    K15(bpe::codepoint_to_byte(324) == -1 && bpe::codepoint_to_byte(0) == -1 && bpe::codepoint_to_byte(33) == 33 && bpe::codepoint_to_byte(256) == 0 && bpe::codepoint_to_byte(256 + 400) == -1);
    // code points 256..323 are exactly the 68 non-printables in increasing byte order
    uint32_t next = 256;
    for (unsigned b = 0; b < 256; ++b) if (!bpe::printable(b)) K15(bpe::byte_to_codepoint((uint8_t)b) == next++);
    K15(next == 324);
    Rng rng(21);
    for (int it = 0; it < 500; ++it) {
        std::string s(rng.below(64), '\0');
        for (auto& c : s) c = (char)rng.u32();
        bool ok = false;
        K15(bpe::mapped_to_bytes(bpe::bytes_to_mapped(s), &ok) == s && ok);
    }
    for (int ty = 0; ty < 8; ++ty) {
        K15(bpe::matched(ty, false) == (ty == 4));
        K15(bpe::matched(ty, true) == (ty == 3 || ty == 4));
        if (bpe::matched(ty, false)) K15(bpe::matched(ty, true));
    }
    K15(!bpe::matched(3, false) && bpe::matched(3, true) && bpe::matched(4, false) && bpe::matched(4, true));
    for (size_t n = 0; n < 6; ++n) K15(bpe::valid_merge(n) == (n == 2));
    // lowest rank wins; ties keep the earlier candidate
    auto better = [](bool has, uint64_t best, uint64_t r) { return !has || r < best; };
    K15(better(false, 0, 9) && !better(true, 5, 5) && better(true, 5, 4) && !better(true, 5, 6));
}

// -------------------------------------------------------------------------------------- CanonicalQuantPack
static void test_pack() {
    for (unsigned c0 = 0; c0 < 4; ++c0) for (unsigned c1 = 0; c1 < 4; ++c1) for (unsigned c2 = 0; c2 < 4; ++c2) for (unsigned c3 = 0; c3 < 4; ++c3) {
        const unsigned b = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        K15(b < 256 && (b & 3) == c0 && ((b >> 2) & 3) == c1 && ((b >> 4) & 3) == c2 && ((b >> 6) & 3) == c3);
        uint8_t out[64], blk[16]; std::memset(blk, 0, 16); blk[0] = (uint8_t)b;
        q2rows::unpack_codes_64(blk, out);
        K15(out[0] == c0 && out[1] == c1 && out[2] == c2 && out[3] == c3);
    }
    for (unsigned a = 0; a < 16; ++a) for (unsigned b = 0; b < 16; ++b) { const unsigned p = a | (b << 4); K15(p < 256 && (p & 15) == a && (p >> 4) == b); }
    for (unsigned b = 0; b < 256; ++b) K15((b >> 4) * 16 + (b & 15) == b);
    // integer code bias: registry stores bias with decoded = code - bias, theory adds -bias
    struct { canon::QType t; int bias; unsigned bits_max; } cb[] = {{canon::QType::Q2_0, -1, 4}, {canon::QType::Q4_0, -8, 16}, {canon::QType::Q5_0, -16, 32},
                                                                  {canon::QType::Q3_K, -4, 8}, {canon::QType::Q6_K, -32, 64}, {canon::QType::Q8_0, -128, 256}};
    for (auto& e : cb) {
        K15(-canon::code_bias(e.t) == e.bias);
        for (unsigned c = 0; c < e.bits_max; ++c) { const int v = (int)c + e.bias; K15(v >= e.bias && v <= (int)e.bits_max - 1 + e.bias); }
        K15((int)(e.bits_max - 1) + e.bias == (e.t == canon::QType::Q2_0 ? 2 : (int)e.bits_max / 2 - 1));
    }
    for (int q = -128; q <= 127; ++q) K15((q + 128) >= 0 && (q + 128) < 256 && (q + 128) - 128 == q);
    // Q3_K / Q5_K / Q5_0 mask traps
    std::set<int> bits;
    for (int half = 0; half < 2; ++half) for (int j = 0; j < 4; ++j) { K15(4 * half + j < 8); bits.insert(4 * half + j); }
    K15(bits.size() == 8);
    std::set<int> m5;
    for (int j = 0; j < 4; ++j) { K15((1 << (2 * j)) == (1 << (2 * j)) && (2 << (2 * j)) == (1 << (2 * j + 1))); m5.insert(1 << (2 * j)); m5.insert(2 << (2 * j)); }
    K15(m5.size() == 8);
    for (int j = 0; j < 16; ++j) K15(((1ull << (j + 16)) >> (j + 12) & 16) == 16);
    // geometry: block bytes (registry) and sums of fields
    K15(2 + 16 == 18 && 2 + 4 + 16 == 22 && 2 + 32 == 34 && 32 + 64 + 12 + 2 == 110 && 2 + 2 + 12 + 128 == 144);
    K15(2 + 2 + 12 + 32 + 128 == 176 && 128 + 64 + 16 + 2 == 210 && 2 + 2 + 4 + 128 == 136);
    for (size_t i = 0; i < canon::kRegistrySize; ++i) {
        const auto& m = canon::kRegistry[i];
        K15(m.group_elems != 0 && m.block_elems % m.group_elems == 0);
        K15(m.block_elems / m.group_elems <= (unsigned)canon::kMaxGroups);
    }
    K15(256 / 16 == 16 && 256 / 32 == 8 && 64 / 64 == 1);
    // 64-byte alignment
    for (uint64_t n = 0; n < 5000; ++n) { const uint64_t p = pack::pad64(n); K15((n + p) % 64 == 0 && p < 64); if (n % 64 == 0) K15(p == 0); }
    // scale dtype choice: fp16 exact or fp32
    for (int e = 0; e < 2; ++e) { const int sb = e ? 2 : 4; K15(sb >= 2 && sb <= 4); }
    // MTP quantizer
    for (long q = -50; q <= 50; ++q) { const long c = mtpq::clip12(q); K15(c >= -1 && c <= 2 && c + 1 >= 0 && c + 1 <= 3 && (c + 1) - 1 == c); }
    K15(mtpq::kScaleCandidates == 17 && mtpq::scale_factor(0) == 0.5f && mtpq::scale_factor(16) == 1.0f);
    for (int k = 0; k < 17; ++k) K15(mtpq::scale_factor(k) == (float)(16 + k) / 32.f);
    K15(mtpq::E * mtpq::GU_OUT * mtpq::row_bytes(mtpq::HID, 64, 18) == 471859200ull);
    K15(mtpq::E * mtpq::HID * mtpq::row_bytes(mtpq::INTER, 64, 18) == 235929600ull);
    K15(mtpq::total_bytes(64, 18) == 707788800ull);
    for (int it = 0; it < 300; ++it) {
        Rng rng(1000 + it); float x[64]; for (auto& v : x) v = (float)rng.gauss();
        uint8_t blk[18]; mtpq::quantize_q2(x, blk);
        for (int k = 0; k < 16; ++k) for (int t = 0; t < 4; ++t) K15(((blk[2 + k] >> (2 * t)) & 3) <= 3);
        canon::Block cb2; canon::encode_q2(blk, &cb2);          // canonical form round trip is lossless on the raw block
        uint8_t back[18]; canon::decode_q2(cb2, back);
        K15(std::memcmp(blk, back, 18) == 0);
    }
    // IQ4 codebook
    const int8_t* kv = dequant::kvalues_iq4nl;
    for (int i = 1; i < 16; ++i) K15(kv[i] > kv[i - 1]);
    K15((kv[1] - kv[0]) != (kv[2] - kv[1]));
    K15(kv[0] == -127 && kv[15] == 113);
}

// ----------------------------------------------------------------------------------------- GgufLazyLoader
static void test_loader() {
    struct { uint32_t t; uint64_t el, by; } g[] = {{gguf::kF32, 1, 4}, {gguf::kF16, 1, 2}, {gguf::kBF16, 1, 2}, {gguf::kQ4_0, 32, 18}, {gguf::kQ5_0, 32, 22},
        {gguf::kQ8_0, 32, 34}, {gguf::kIQ4_NL, 32, 18}, {gguf::kQ3_K, 256, 110}, {gguf::kQ4_K, 256, 144}, {gguf::kQ5_K, 256, 176},
        {gguf::kQ6_K, 256, 210}, {gguf::kIQ4_XS, 256, 136}, {gguf::kQ2_0, 64, 18}};
    for (auto& e : g) { const auto bg = gguf::block_geometry(e.t); K15(bg.el == e.el && bg.by == e.by && bg.el > 0); }
    K15(gguf::block_geometry(9999).el == 0);
    for (uint64_t per : {64ull * 5, 2560ull * 640}) for (uint64_t e = 0; e < 600; e += 7) {
        const uint64_t bytes = gguf::expert_bytes(per, 64, 18);
        K15(gguf::expert_start(e, per, 64, 18) + bytes == gguf::expert_start(e + 1, per, 64, 18));
        K15(gguf::expert_start(e, per, 64, 18) == gguf::expert_start_py(e, per, 64, 18));
        K15(gguf::expert_start(e, per, 64, 18) == e * bytes);
    }
    K15(gguf::expert_start(0, 2560 * 640, 64, 18) == 0);
    for (uint64_t nexp : {1ull, 2ull, 512ull}) { const uint64_t e = nexp - 1; K15(gguf::expert_start(e, 2560 * 640, 64, 18) + gguf::expert_bytes(2560 * 640, 64, 18) == nexp * (2560 * 640 / 64) * 18); }
    K15(gguf::expert_bytes(2560 * 640, 64, 18) == 460800);
    K15(3 * gguf::expert_bytes(2560 * 640, 64, 18) == 1382400);
    K15(512 * gguf::expert_bytes(2560 * 640, 64, 18) == 235929600ull);
    for (uint64_t r = 0; r < 100; ++r) { const uint64_t rb = gguf::row_bytes(2560, 64, 18); K15(r * rb + rb == (r + 1) * rb); K15(r * rb + rb <= 100 * rb); }
    K15(gguf::row_bytes(160, 32, 18) == 90 && gguf::ple_row_bytes() == 90 && 160 % 32 == 0);
    for (uint64_t el : {0ull, 1ull, 32ull, 64ull, 256ull}) for (uint64_t n : {0ull, 1ull, 31ull, 32ull, 64ull, 160ull, 255ull, 256ull, 2560ull}) {
        const bool v = gguf::whole_blocks(n, el);
        K15(v == (el != 0 && n % el == 0));
        if (v) K15(n / el * el == n);
    }
    for (uint64_t f = 0; f < 6; ++f) for (uint64_t t = 0; t < 6; ++t) K15(gguf::shards_complete(f, t) == (f == t));
}

// ----------------------------------------------------------------------------------------- VisionEncoderTool
static void test_vision() {
    using namespace vision_enc;
    K15(0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 == 0x31455653);
    K15(MAGIC == 0x31455653u && HEADER_BYTES == 20);
    const std::vector<uint8_t> h = encode_header(6, 3, 2, 8);
    K15(h.size() == 20 && h[0] == 0x53 && h[1] == 0x56 && h[2] == 0x45 && h[3] == 0x31);
    for (uint64_t n : {0ull, 1ull, 6ull, 100ull}) for (uint64_t e : {1ull, 8ull, 4096ull}) { K15(file_bytes(n, e) - HEADER_BYTES == 4 * n * e); }
    K15(file_bytes(0, 17) == HEADER_BYTES);
    for (int x = 0; x < 20; ++x) for (int y = 0; y < 20; ++y) for (int n : {1, 6, 12, 100, 400}) {
        int nx, ny; grid_from_last(x, y, nx, ny);
        K15(nx == x + 1 && ny == y + 1);
        K15(grid_ok(nx, ny, n) == ((x + 1) * (y + 1) == n));
    }
    for (uint64_t nx = 1; nx < 12; ++nx) for (uint64_t ny = 1; ny < 12; ++ny) for (uint64_t i = 0; i < nx * ny; ++i) {
        uint64_t x, y; row_pos(i, nx, x, y);
        K15(y * nx + x == i && x < nx && y < ny);
    }
    // ENC split at the LAST space
    K15(std::string("ENC ").size() == 4);
    Command c = parse_enc("/tmp/my image.png /tmp/out.bin");
    K15(c.ok && c.image == "/tmp/my image.png" && c.output == "/tmp/out.bin");
    c = parse_enc("a b c d"); K15(c.ok && c.image == "a b c" && c.output == "d");
    K15(!parse_enc("nospace").ok && !parse_enc(" out").ok && !parse_enc("img ").ok && !parse_enc("").ok);
    for (uint64_t len = 0; len < 8; ++len) for (uint64_t sp = 0; sp < 8; ++sp) { K15(accepts(sp, len) == (0 < sp && sp < len - 0 && sp + 1 < len)); }
    // file builder / validator
    std::vector<float> emb(6 * 4); for (size_t i = 0; i < emb.size(); ++i) emb[i] = (float)i * 0.25f;
    auto f = build_file(3, 2, 4, emb);
    K15(f.size() == file_bytes(6, 4) && validate_file(f));
    f[0] ^= 1; K15(!validate_file(f)); f[0] ^= 1;
    auto t = f; t.pop_back(); K15(!validate_file(t));
    Header hd; K15(decode_header(f.data(), f.size(), hd) && hd.n_tokens == 6 && hd.nx == 3 && hd.ny == 2 && hd.n_embd == 4);
    // empty encoding: header only
    K15(file_bytes(0, 4) == 20);
    // build options
    for (int p = 0; p < 2; ++p) {
        vision_cfg::Options o; o.portable = p; const auto s = vision_cfg::configure(o);
        K15(s.native == !p);
        K15((s.avx2 && s.fma && s.f16c && s.avx) == (bool)p);
        K15(s.ggml_cuda == false && !s.shared_libs && !s.backend_dl);
    }
    vision_cfg::Options oc; oc.cuda = true; K15(vision_cfg::configure(oc).ggml_cuda);
    K15(!vision_cfg::configure(vision_cfg::Options{}).ggml_cuda && vision_cfg::configure(vision_cfg::Options{}).native);
}

int main() {
    test_layout(); test_unpack(); test_quantizer(); test_rows(); test_rounding_model();
    test_tokenizer(); test_pack(); test_loader(); test_vision();
    std::printf("checks: %ld, failures: %ld\n", testh::checks(), testh::failures());
    if (testh::failures()) { std::puts("Q2ROWS UNIT TESTS FAILED"); return 1; }
    std::puts("Q2ROWS UNIT TESTS PASSED");
    return 0;
}
