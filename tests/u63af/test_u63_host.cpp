// test_u63_host.cpp - unit u63af4bc057, host side. Checks the LAYER 1 statements of
//   HalfPrecisionBitConversion (include/baloon/f16_bits.hpp, include/baloon/port_scalar.hpp),
//   BlockDequantization       (include/baloon/artifact_dequant.hpp),
//   GgufHeaderParsing         (include/baloon/gguf_reader.hpp),
//   IQuantRowDecode           (src/cpu/iq_rows_avx512.cpp, built with the AVX-512 ISA flags)
// against the theory models / independent spec decoders in spec_models.hpp. Exits non-zero on any failed check.
#include "baloon/artifact_dequant.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/gguf_reader.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/port_scalar.hpp"
#include "spec_models.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (g_fail < 40) std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

using namespace baloon;
static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool is_f16_nan(uint32_t h) { return u63::f16Exp(h) == 31 && u63::f16Man(h) != 0; }

// =========================================================================================== HalfPrecisionBitConversion
static void test_half_bits() {
    using namespace f16bits;
    // f16_fields_recompose / f16_fields_bounded (all 2^16 words)
    for (uint32_t h = 0; h < 65536; ++h) {
        CHECK(h == u63::f16Sign(h) * 32768 + u63::f16Exp(h) * 1024 + u63::f16Man(h));
        CHECK(u63::f16Sign(h) < 2 && u63::f16Exp(h) < 32 && u63::f16Man(h) < 1024);
    }
    // roundMan13_error / tie_even / direction, and the code's rne_step equals the model on every 23-bit mantissa
    for (uint32_t man = 0; man < (1u << 23); man += (man < 65536 ? 1 : 97)) {
        const uint32_t r = u63::roundMan13(man);
        CHECK((uint64_t)r * 8192 <= (uint64_t)man + 4096 && man <= (uint64_t)r * 8192 + 4096);
        if (man % 8192 == 4096) CHECK(r % 2 == 0);
        if (man % 8192 < 4096) CHECK(r == man / 8192);
        if (man % 8192 > 4096) CHECK(r == man / 8192 + 1);
        CHECK(rne_step(man >> 13, man & 8191u) == r);
    }
    // roundDiv_error for every shift amount of the subnormal path
    std::mt19937_64 rng(1);
    for (int sh = 1; sh <= 30; ++sh) for (int k = 0; k < 2000; ++k) {
        const uint64_t half = 1ull << (sh - 1), x = rng() >> 20;
        const uint64_t r = u63::roundDiv(x, half);
        CHECK(r * 2 * half <= x + half && x <= r * 2 * half + half);
        CHECK(rne_round((uint32_t)(x / (2 * half)), (uint32_t)(x % (2 * half)), (uint32_t)half) == (uint32_t)r || x / (2 * half) > 0xFFFFFFFFull);
    }
    // carry_bumps_exponent
    for (uint32_t s = 0; s < 2; ++s) for (uint32_t e = 0; e < 31; ++e) {
        const uint32_t h = s * 32768 + e * 1024 + 1023;
        CHECK(u63::f16Man(h + 1) == 0 && u63::f16Exp(h + 1) == e + 1 && u63::f16Sign(h + 1) == s);
    }
    // f32_from_f16 is exact (every non-NaN word), subnormals normalise in <= 10 shifts to exponent field 113 - s
    for (uint32_t h = 0; h < 65536; ++h) {
        if (is_f16_nan(h)) { CHECK(std::isnan(f32_from_f16((uint16_t)h))); continue; }
        const float f = f32_from_f16((uint16_t)h);
        CHECK((double)f == u63::f16_exact((uint16_t)h));
        const uint32_t m = u63::f16Man(h);
        if (u63::f16Exp(h) == 0 && m != 0) {
            const uint32_t s = u63::normShift(10, m);
            CHECK(s <= 10 && 1024 <= (m << s) && (m << s) < 2048);
            const uint32_t fe = (fbits(f) >> 23) & 0xFF;
            CHECK(fe == 113 - s && fe >= 103 && fe <= 113);
            CHECK(127 - 15 + 1 == 113);
        }
        if (u63::f16Exp(h) >= 1 && u63::f16Exp(h) <= 30) {
            const uint32_t fe = (fbits(f) >> 23) & 0xFF;
            CHECK(fe == u63::f16Exp(h) - 15 + 127 && fe >= 113 && fe <= 142);
        }
        // decode then encode is the identity
        CHECK(f16_from_f32(f) == h);
    }
    // nearest/ties-to-even at every boundary between consecutive finite fp16 values (both signs):
    // the exact midpoint goes to the even word, one f32 ulp below to the lower, one above to the upper (this also
    // covers the mantissa carry into the exponent field at ...3FF -> ...400).
    for (uint32_t s = 0; s < 2; ++s) for (uint32_t h = 0; h < 0x7BFF; ++h) {
        const double lo = std::fabs(u63::f16_exact((uint16_t)h)), hi = std::fabs(u63::f16_exact((uint16_t)(h + 1)));
        const float mid = (float)((lo + hi) / 2);
        CHECK((double)mid == (lo + hi) / 2);   // representable in f32
        const float smid = s ? -mid : mid;
        const uint32_t even = (h % 2 == 0) ? h : h + 1;
        CHECK(f16_from_f32(smid) == (uint16_t)(s * 32768 + even));
        CHECK(f16_from_f32(std::nextafter(smid, 0.f)) == (uint16_t)(s * 32768 + h));
        CHECK(f16_from_f32(std::nextafter(smid, s ? -INFINITY : INFINITY)) == (uint16_t)(s * 32768 + h + 1));
    }
    // random f32 inputs: the result is the nearest fp16 (error <= half an fp16 ulp), never NaN for finite input
    for (int k = 0; k < 2000000; ++k) {
        const uint32_t u = (uint32_t)rng();
        const float f = bitsf(u);
        if (std::isnan(f)) {
            const uint16_t o = f16_from_f32(f);
            CHECK(u63::f16Exp(o) == 31 && u63::f16Man(o) != 0 && u63::f16Sign(o) == (u >> 31));   // nan_stays_nan
            continue;
        }
        const uint16_t o = f16_from_f32(f);
        CHECK(!is_f16_nan(o));
        const int rawexp = (int)((u >> 23) & 0xFF), exp = rawexp - 127 + 15;
        uint32_t sp;
        if (u63::encodeSpecial((u >> 16) & 0x8000u, (uint32_t)rawexp, u & 0x7FFFFF, exp, &sp)) {
            CHECK(o == sp);
            if (rawexp != 255) CHECK(u63::f16Man(o) == 0 && u63::f16Exp(o) == 31);   // finite_overflow_is_inf
            continue;
        }
        const double v = std::fabs((double)f), r = std::fabs(u63::f16_exact(o));
        if (std::isinf(r)) { CHECK(v >= 65520.0); continue; }
        const double ulp = u63::f16Exp(o) == 0 ? std::ldexp(1.0, -24) : std::ldexp(1.0, (int)u63::f16Exp(o) - 25);
        CHECK(std::fabs(v - r) <= ulp / 2 * (1 + 1e-12) + (u63::f16Man(o) == 0 && u63::f16Exp(o) > 1 ? ulp / 4 : 0));
    }
    // overflow: a FINITE value with rebiased exponent >= 31 saturates to exactly +-inf, never NaN (the conflated rule
    // the header replaced would give NaN for a non-zero mantissa)
    for (float f : {65536.f, 1e10f, 3.4e38f, 131071.f, -1e20f, 70000.5f}) {
        const uint16_t o = f16_from_f32(f);
        CHECK(u63::f16Man(o) == 0 && u63::f16Exp(o) == 31 && u63::f16Sign(o) == (f < 0 ? 1u : 0u));
        const uint32_t man = fbits(f) & 0x7FFFFF;
        if (man) CHECK(u63::f16Man(u63::conflatedOverflow(0, man)) != 0);
    }
    CHECK(f16_from_f32(INFINITY) == 0x7C00 && f16_from_f32(-INFINITY) == 0xFC00);

    // ---- bf16
    CHECK(u63::bf16Encode(0x3F808000u) == 0x3F80 && u63::bf16EncodeBit15(0x3F808000u) == 0x3F81);   // bit-15 variant is wrong
    CHECK(u63::bf16Encode(0x7F800001u) == 0x7F80);   // theory observation: the unguarded rule turns this NaN into inf
    CHECK(u63::bf16Encode(0xFFFFFFFFu) == 0);        // and wraps this NaN to +0
    for (uint32_t h = 0; h < 65536; ++h) {
        CHECK(u63::bf16Encode(u63::bf16Decode(h)) == h);                                           // model roundtrip
        CHECK(fbits(refk::f32_from_bf16((uint16_t)h)) == (uint32_t)u63::bf16Decode(h));             // decode is << 16
        CHECK(fbits(bf16_bits_to_f32((uint16_t)h)) == (uint32_t)u63::bf16Decode(h));
        const float f = bitsf((uint32_t)u63::bf16Decode(h));
        if (std::isnan(f)) {   // the production encoders keep NaN a NaN (guarded; see the unit summary)
            CHECK(std::isnan(refk::f32_from_bf16(refk::bf16_from_f32(f))));
            CHECK(std::isnan(bf16_bits_to_f32(round_f32_to_bf16_bits(f))));
            continue;
        }
        CHECK(refk::bf16_from_f32(f) == h && round_f32_to_bf16_bits(f) == h);                        // exact round trip
    }
    // direction and ties-to-even of the production encoders agree with the theory model on every non-NaN input
    for (int k = 0; k < 3000000; ++k) {
        uint32_t u = (uint32_t)rng();
        if (k % 4 == 0) u = (u & 0xFFFF0000u) | 0x8000u;   // exact ties
        const float f = bitsf(u);
        if (std::isnan(f)) continue;
        const uint32_t m = u63::bf16Encode(u);
        CHECK(refk::bf16_from_f32(f) == m && round_f32_to_bf16_bits(f) == m && u63::f2bf(u) == m);
        const uint32_t hh = u >> 16, r = u & 0xFFFF;
        if (hh < 65535) {
            if (r < 32768) CHECK(m == hh);
            if (r > 32768) CHECK(m == hh + 1);
            if (r == 32768) CHECK(m == hh + hh % 2 && m % 2 == 0);
        }
    }
}

// ================================================================================================ BlockDequantization
static std::mt19937 brng(7);
static uint8_t rb() { return (uint8_t)(brng() & 0xFF); }
static uint16_t rand_scale_f16() {   // a finite, moderate fp16 scale (exponent 5..20), random sign
    return (uint16_t)(((brng() & 1) << 15) | ((5 + brng() % 16) << 10) | (brng() & 0x3FF));
}
static void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static double df16(const uint8_t* p) { return u63::f16_exact((uint16_t)(p[0] | p[1] << 8)); }
static bool close(double a, double b, double scale) { return std::fabs(a - b) <= 1e-6 * scale + 1e-30; }

static void test_block_dequant() {
    using namespace artifact;
    // Q2_0 packing: unpack_pack, packByte_lt, index map
    for (uint32_t c = 0; c < 256; ++c) {
        const uint32_t c0 = c & 3, c1 = c >> 2 & 3, c2 = c >> 4 & 3, c3 = c >> 6 & 3;
        const uint32_t b = u63::packByte(c0, c1, c2, c3);
        CHECK(b < 256 && b == c);
        CHECK(b / u63::slotDiv(0) % 4 == c0 && b / u63::slotDiv(1) % 4 == c1 && b / u63::slotDiv(2) % 4 == c2 && b / u63::slotDiv(3) % 4 == c3);
    }
    // IQ4_NL codebook: identical in the code, 16 entries, strictly increasing, int8
    for (int i = 0; i < 16; ++i) {
        CHECK(kvalues_iq4nl[i] == u63::iq4nlTable()[i]);
        CHECK(-128 <= u63::iq4nlTable()[i] && u63::iq4nlTable()[i] <= 127);
        if (i) CHECK(u63::iq4nlTable()[i - 1] < u63::iq4nlTable()[i]);
    }
    // get_scale_min_k4 vs the theory scaleMinK4: random 12-byte fields, all j < 8, 6-bit outputs
    for (int k = 0; k < 20000; ++k) {
        uint8_t q[12]; for (auto& x : q) x = rb();
        for (int j = 0; j < 8; ++j) {
            uint8_t d, m; get_scale_min_k4(j, q, &d, &m);
            int sd, sm; u63::scaleMinK4(j, q, &sd, &sm);
            CHECK(d == sd && m == sm && d < 64 && m < 64);
        }
    }
    // block byte budgets and the artifact's table
    size_t nt; const BlockType* bt = block_types(&nt);
    CHECK(nt == 10);
    struct { const char* n; size_t by, el; } want[] = {{"q2_0", 18, 64}, {"q4_0", 18, 32}, {"q5_0", 22, 32}, {"q8_0", 34, 32},
        {"iq4_nl", 18, 32}, {"q3_K", 110, 256}, {"q4_K", 144, 256}, {"q5_K", 176, 256}, {"q6_K", 210, 256}, {"iq4_xs", 136, 256}};
    for (auto& w : want) { const BlockType* t = find_block_type(w.n); CHECK(t && t->bytes == w.by && t->elems == w.el); }

    // every dequantizer equals the independent spec decoder on random blocks
    for (size_t ti = 0; ti < nt; ++ti) {
        const BlockType& t = bt[ti];
        const std::string name = t.name;
        for (int it = 0; it < 3000; ++it) {
            std::vector<uint8_t> b(t.bytes); for (auto& x : b) x = rb();
            // finite scales at the fields each format reads them from
            if (name == "q6_K") put16(&b[208], rand_scale_f16());
            else if (name == "q3_K") put16(&b[108], rand_scale_f16());
            else { put16(&b[0], rand_scale_f16()); if (name == "q4_K" || name == "q5_K") put16(&b[2], rand_scale_f16()); }
            std::vector<float> got(t.elems); t.fn(b.data(), got.data());
            std::vector<double> ref(t.elems);
            double scale = 0;
            if (name == "q2_0") { const double d = df16(&b[0]); u63::spec_q2_0(b.data(), ref.data(), d); scale = std::fabs(d) * 2; }
            else if (name == "q4_0") { const double d = df16(&b[0]); u63::spec_q4_0(b.data(), ref.data(), d); scale = std::fabs(d) * 8; }
            else if (name == "q5_0") { const double d = df16(&b[0]); u63::spec_q5_0(b.data(), ref.data(), d); scale = std::fabs(d) * 16; }
            else if (name == "q8_0") { const double d = df16(&b[0]); u63::spec_q8_0(b.data(), ref.data(), d); scale = std::fabs(d) * 128; }
            else if (name == "iq4_nl") { const double d = df16(&b[0]); u63::spec_iq4_nl(b.data(), ref.data(), d); scale = std::fabs(d) * 127; }
            else if (name == "q6_K") { const double d = df16(&b[208]); u63::spec_q6_K(b.data(), ref.data(), d); scale = std::fabs(d) * 128 * 32; }
            else if (name == "q4_K") { const double d = df16(&b[0]), m = df16(&b[2]); u63::spec_q4_K(b.data(), ref.data(), d, m); scale = (std::fabs(d) * 15 + std::fabs(m)) * 63; }
            else if (name == "q5_K") { const double d = df16(&b[0]), m = df16(&b[2]); u63::spec_q5_K(b.data(), ref.data(), d, m); scale = (std::fabs(d) * 31 + std::fabs(m)) * 63; }
            else if (name == "q3_K") { const double d = df16(&b[108]); u63::spec_q3_K(b.data(), ref.data(), d); scale = std::fabs(d) * 32 * 4; }
            else if (name == "iq4_xs") { const double d = df16(&b[0]); u63::spec_iq4_xs(b.data(), ref.data(), d); scale = std::fabs(d) * 32 * 127; }
            bool ok = true;
            for (size_t p = 0; p < t.elems; ++p) ok = ok && close(got[p], ref[p], scale);
            if (!ok && g_fail < 40) std::printf("  mismatch in %s\n", t.name);
            CHECK(ok);
            if (name == "q2_0") {   // the --check structural invariant: every output is one of {-d, 0, d, 2d}, exactly
                const float d = fp16_to_fp32(read_u16(b.data()));
                for (float v : got) CHECK(v == -d || v == 0.f || v == d || v == 2 * d);
            }
        }
    }
    // Q6_K: the scale applied to output p is scale index p / 16 (all codes = 33 -> symbol 1, distinct scales)
    {
        std::vector<uint8_t> b(210, 0);
        for (int i = 0; i < 128; ++i) b[i] = 0x11;        // low nibbles 1
        for (int i = 0; i < 64; ++i) b[128 + i] = 0xAA;   // high pairs 2 -> code 1 + 32 = 33
        for (int i = 0; i < 16; ++i) b[192 + i] = (uint8_t)(i + 1);
        put16(&b[208], 0x3C00);                            // 1.0
        float y[256]; artifact::dequantize_q6_K(b.data(), y);
        for (int p = 0; p < 256; ++p) CHECK(y[p] == (float)(p / 16 + 1));
        for (uint32_t n = 0; n < 2; ++n) for (uint32_t l = 0; l < 32; ++l) for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t p = u63::q6kPos(n, l, k);
            CHECK(p < 256 && p / 128 == n && p % 128 / 32 == k && p % 32 == l && u63::q6kScale(n, l, k) == p / 16);
        }
    }
    // Q5_K: the 8 masks 4^g, 2*4^g are the 8 bits of a byte; setting only bit `bit` of every qh byte gives +16 to exactly
    // the 32 outputs of (group bit/2, nibble half bit%2)
    for (uint32_t g = 0; g < 4; ++g) CHECK(u63::q5kMask1(g) == (1u << (2 * g)) && u63::q5kMask2(g) == (2u << (2 * g)));
    for (int bit = 0; bit < 8; ++bit) {
        std::vector<uint8_t> b(176, 0);
        put16(&b[0], 0x3C00); put16(&b[2], 0);
        for (int i = 0; i < 4; ++i) b[4 + i] = 1;           // scales j<4 = 1, mins 0
        for (int i = 8; i < 12; ++i) b[4 + i] = 1;          // scales j>=4 low nibble 1, mins 0
        for (int i = 0; i < 32; ++i) b[16 + i] = (uint8_t)(1u << bit);
        float y[256]; artifact::dequantize_q5_K(b.data(), y);
        for (int p = 0; p < 256; ++p) CHECK(y[p] == ((p / 32) == bit ? 16.f : 0.f));
    }
    // Q3_K: the high-bit mask is NOT reset between the two 128-halves: step (half, j) uses bit 4*half + j. With only bit
    // `bit` set in hmask and zero low bits, exactly the 32 outputs of that step lose the -4 correction.
    for (int bit = 0; bit < 8; ++bit) {
        std::vector<uint8_t> b(110, 0);
        for (int i = 0; i < 32; ++i) b[i] = (uint8_t)(1u << bit);
        for (int i = 0; i < 8; ++i) b[96 + i] = 0x11;       // every scale low nibble 1
        for (int i = 8; i < 12; ++i) b[96 + i] = 0xAA;      // high pairs 2 -> scale 1 + 32 = 33 -> signed 1
        put16(&b[108], 0x3C00);
        float y[256]; artifact::dequantize_q3_K(b.data(), y);
        const uint32_t half = bit / 4, j = bit % 4;
        CHECK(u63::q3kMask(half, j) == (1u << bit));
        for (int p = 0; p < 256; ++p) CHECK(y[p] == (((uint32_t)p / 128 == half && (uint32_t)p % 128 / 32 == j) ? 0.f : -4.f));
    }
    // IQ4_XS scale: low nibble of scales_l[ib/2] by ib%2, two bits of scales_h at 2 ib; 6-bit; reads in range
    for (int k = 0; k < 5000; ++k) {
        uint8_t sl[4]; for (auto& x : sl) x = rb(); const uint32_t sh = brng() & 0xFFFF;
        for (uint32_t ib = 0; ib < 8; ++ib) {
            const uint32_t s = u63::iq4xsScale(sl, sh, ib);
            CHECK(s < 64 && ib / 2 < 4 && 2 * ib + 1 < 16);
            CHECK(s == (((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4)));
        }
    }
}

// ================================================================================================= GgufHeaderParsing
struct W {
    std::vector<uint8_t> b;
    void u(uint64_t v, int n) { for (int i = 0; i < n; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
    void s(const std::string& x) { u(x.size(), 8); b.insert(b.end(), x.begin(), x.end()); }
};
struct Kv { std::string k; int type; uint64_t v; std::string sv; };
static std::vector<uint8_t> gguf_file(const std::vector<Kv>& kvs, bool with_tensor, size_t* header_end) {
    W w; w.u(0x46554747u, 4); w.u(3, 4); w.u(with_tensor ? 1 : 0, 8); w.u(kvs.size(), 8);
    for (auto& kv : kvs) {
        w.s(kv.k); w.u((uint64_t)kv.type, 4);
        if (kv.type == 8) w.s(kv.sv); else w.u(kv.v, (int)gguf_reader::scalar_size((gguf_reader::MetaType)kv.type));
    }
    if (with_tensor) { w.s("t"); w.u(1, 4); w.u(64, 8); w.u(42, 4); w.u(0, 8); }   // one Q2_0 tensor of 64 elements
    *header_end = w.b.size();
    if (!with_tensor) w.b.resize((w.b.size() + 31) / 32 * 32, 0);   // the image reaches the aligned data start
    return w.b;
}
static std::vector<Kv> guard_kvs() {
    return {{"general.architecture", 8, 0, "qwen4exp"}, {"qwen4exp.block_count", 4, 48, ""},
            {"qwen4exp.embedding_length", 4, 2560, ""}, {"qwen4exp.expert_count", 4, 512, ""},
            {"qwen4exp.expert_used_count", 4, 10, ""}, {"qwen4exp.attention.head_count", 4, 24, ""},
            {"qwen4exp.attention.head_count_kv", 4, 2, ""}};
}

static void test_gguf() {
    using namespace gguf_reader;
    // Cursor: need(n) fails iff pos + n > size; successful reads advance by n and keep pos <= size
    std::vector<uint8_t> buf(100, 0xAB);
    for (uint64_t size = 0; size <= 40; ++size) for (uint64_t start = 0; start <= size; ++start) for (uint64_t n = 0; n <= 12; ++n) {
        Cursor c(buf.data(), size); c.pos = start;
        const bool ok = c.need(n);
        CHECK(ok == !(start + n > size));
        if (ok) { const uint64_t p0 = c.pos; if (n == 1 || n == 2 || n == 4 || n == 8) { c.uint_le((unsigned)n); CHECK(c.pos == p0 + n && c.pos <= size); } }
        else CHECK(!c.ok);
    }
    // readAll: a sequence of reads advances by the sum and stops at the first failure
    for (int k = 0; k < 2000; ++k) {
        const uint64_t size = brng() % 60; Cursor c(buf.data(), size);
        uint64_t sum = 0; bool alive = true;
        for (int r = 0; r < 6; ++r) {
            const unsigned n = 1u << (brng() % 4);
            c.uint_le(n);
            if (alive && sum + n <= size) sum += n; else alive = false;
            CHECK(c.ok == alive && c.pos <= size);
            if (alive) CHECK(c.pos == sum);
        }
    }
    // a string consumes 8 + len bytes
    for (uint64_t len : {0ull, 1ull, 5ull, 40ull}) {
        W w; w.s(std::string(len, 'x')); w.u(7, 1);
        Cursor c(w.b.data(), w.b.size()); const std::string s = c.str();
        CHECK(c.ok && s.size() == len && c.pos == 8 + len);
        Cursor t(w.b.data(), 8 + len - (len ? 1 : 0)); t.str(); CHECK(len == 0 ? t.ok : !t.ok);
    }
    // arrays: count > 2^24 refused; at most 64 items sampled; exactly 2^24 is not refused by the length guard
    {
        W w; w.u(0, 4); w.u((1ull << 24) + 1, 8);
        std::vector<uint8_t> big(w.b); big.resize(big.size() + (1u << 24) + 1, 0);
        Cursor c(big.data(), big.size()); (void)read_value(c, MetaType::Array); CHECK(!c.ok);
        W w2; w2.u(0, 4); w2.u(1ull << 24, 8);
        std::vector<uint8_t> ok(w2.b); ok.resize(ok.size() + (1u << 24), 1);
        Cursor c2(ok.data(), ok.size()); MetaValue v = read_value(c2, MetaType::Array);
        CHECK(c2.ok && v.sample_count() == 64);
    }
    for (uint64_t n : {0ull, 1ull, 63ull, 64ull, 65ull, 100000ull, 1ull << 24}) {
        const uint64_t kept = array_sample_count(n);
        CHECK(kept <= 64 && kept <= n && (n > 64 || kept == n) && kept == std::min<uint64_t>(n, 64));
    }
    // signed metadata survives uint64 storage (sign extension, two's complement mod 2^64)
    const struct { MetaType t; int nb; int64_t v; } sv[] = {
        {MetaType::I8, 1, -128}, {MetaType::I8, 1, 127}, {MetaType::I8, 1, -1}, {MetaType::I16, 2, -32768},
        {MetaType::I16, 2, 12345}, {MetaType::I32, 4, INT32_MIN}, {MetaType::I32, 4, -7}, {MetaType::I64, 8, INT64_MIN},
        {MetaType::I64, 8, INT64_MAX}, {MetaType::I64, 8, -2}};
    for (auto& x : sv) {
        W w; w.u((uint64_t)x.v, x.nb);
        Cursor c(w.b.data(), w.b.size()); MetaValue v = read_value(c, x.t);
        CHECK(c.ok && v.i == x.v && v.u == (uint64_t)x.v && (int64_t)v.u == x.v && c.pos == (uint64_t)x.nb);
    }
    // bool decodes to 0/1
    for (int b = 0; b < 2; ++b) { W w; w.u((uint64_t)b, 1); Cursor c(w.b.data(), 1); MetaValue v = read_value(c, MetaType::Bool); CHECK(c.ok && v.u == (uint64_t)b && v.b == (b == 1)); }

    // block geometry: the theory table (Q2_0 = (64, 18), the dequant layouts and the i-quant strides)
    for (uint32_t t = 0; t < 64; ++t) {
        uint64_t e, b;
        if (u63::gguf_geometry(t, &e, &b)) { const BlockGeometry g = block_geometry(t); CHECK(g.el == e && g.by == b); }
    }
    CHECK(2 + 64 * 2 / 8 == 18);
    { const BlockGeometry g = block_geometry(42); CHECK(2560 / g.el * g.by == 720); }
    CHECK(std::string(ggml_type_name(42)).size() > 0);

    // data section alignment: data_start = alignUp(header_end, align); default 32; aligned, >= end, < end + align
    for (uint64_t a = 1; a <= 256; ++a) for (uint64_t pos = 0; pos < 300; pos += 7) {
        const uint64_t s = u63::alignUp(pos, a);
        CHECK(s % a == 0 && pos <= s && s < pos + a);
    }
    for (uint32_t align : {0u, 32u, 64u, 128u}) for (int pad = 0; pad < 9; ++pad) {
        std::vector<Kv> kv = {{"general.name", 8, 0, std::string((size_t)pad * 5 + 1, 'n')}};
        if (align) kv.push_back({"general.alignment", 4, align, ""});
        size_t end; std::vector<uint8_t> img = gguf_file(kv, true, &end);
        const uint64_t a = align ? align : 32;
        const uint64_t ds = u63::alignUp(end, a);
        img.resize(ds + 18, 0);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value());
        if (f) {
            CHECK(f->alignment == a && f->data_start == ds && f->data_start % a == 0 && f->data_start >= end && f->data_start < end + a);
            CHECK(f->tensors.size() == 1 && f->tensors[0].nbytes() == 18);
        }
    }
    // a truncated header never parses (the cursor never runs past the mapping)
    {
        size_t end; std::vector<uint8_t> img = gguf_file(guard_kvs(), true, &end);
        for (size_t cut = 0; cut < end; cut += 3) { auto f = GgufFile::parse(img.data(), cut); CHECK(!f.has_value()); }
    }

    // architecture guard: "" iff qwen4exp and all six keys equal the compiled-in constants; else the FIRST failure
    {
        size_t end; std::vector<uint8_t> img = gguf_file(guard_kvs(), false, &end);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f).empty());
        if (f) CHECK(check_architecture(*f, "qwen4exp"));
        Qwen4ExpGuard g; if (f) CHECK(g.check(*f));
    }
    const std::vector<Kv> base = guard_kvs();
    for (size_t miss = 0; miss < base.size(); ++miss) for (int mode = 0; mode < 2; ++mode) {
        std::vector<Kv> kv;
        for (size_t i = 0; i < base.size(); ++i) {
            if (i != miss) { kv.push_back(base[i]); continue; }
            if (mode == 1) { Kv x = base[i]; if (x.type == 8) x.sv = "llama"; else x.v += 1; kv.push_back(x); }   // mismatch
        }
        size_t end; std::vector<uint8_t> img = gguf_file(kv, false, &end);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f.has_value());
        if (f) { const std::string why = check_architecture(*f); CHECK(why == base[miss].k); }
    }
    // two failures: the first in order is reported
    {
        std::vector<Kv> kv = guard_kvs(); kv[6].v = 4; kv[3].v = 256;   // head_count_kv and expert_count wrong
        size_t end; std::vector<uint8_t> img = gguf_file(kv, false, &end);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f) == "qwen4exp.expert_count");
    }
    for (size_t a = 1; a < 7; ++a) for (size_t b = a + 1; b < 7; ++b) {   // every pair of wrong keys: the earlier one wins
        std::vector<Kv> kv = guard_kvs(); kv[a].v += 3; kv[b].v += 5;
        size_t end; std::vector<uint8_t> img = gguf_file(kv, false, &end);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f) == guard_kvs()[a].k);
    }
    {   // wrong architecture string is reported before any key
        std::vector<Kv> kv = guard_kvs(); kv[0].sv = "qwen3moe"; kv[1].v = 1;
        size_t end; std::vector<uint8_t> img = gguf_file(kv, false, &end);
        auto f = GgufFile::parse(img.data(), img.size());
        CHECK(f && check_architecture(*f) == "general.architecture");
    }
}

// =================================================================================================== IQuantRowDecode
static void test_iq512() {
    using namespace kern;
    std::vector<uint32_t> sup;
    for (uint32_t t = 0; t < 64; ++t) if (iq512_supported(t)) sup.push_back(t);
    CHECK((sup == std::vector<uint32_t>{16, 17, 18, 21, 22}));
    for (uint32_t t = 0; t < 64; ++t) CHECK(iq512_supported(t) == u63::iqSupported(t));
    CHECK(!iq512_supported(29));
    // block sizes are the field sums and agree with the GGUF table
    const uint32_t tys[5] = {16, 17, 18, 21, 22}, byt[5] = {66, 74, 98, 110, 82};
    CHECK(2 + 64 == 66 && 2 + 64 + 8 == 74 && 2 + 32 + 32 + 8 + 8 == 82 && 2 + 64 + 32 == 98 && 2 + 64 + 8 + 32 + 4 == 110);
    for (int i = 0; i < 5; ++i) { const auto g = gguf_reader::block_geometry(tys[i]); CHECK(g.el == 256 && g.by == byt[i]); }
    for (uint32_t j = 0; j < 4; ++j) {
        CHECK(2 + 16 * j + 16 <= 66 && 66 + 2 * j + 1 < 74);
        CHECK(2 + 8 * j + 8 <= 34 && 34 + 8 * j + 8 <= 66 && 66 + 2 * j + 1 < 74 && 74 + 2 * j + 1 < 82);
        CHECK(2 + 16 * j + 16 <= 66 && 66 + 8 * j + 8 <= 98);
        CHECK(2 + 16 * j + 16 <= 66 && 66 + 2 * j + 1 < 74 && 74 + 8 * j + 8 <= 106 && 106 + j < 110);
        CHECK(64 * j + 64 <= 256);
    }
    for (uint32_t s = 0; s < 64; ++s) { const uint32_t o = u63::oddScale(s); CHECK(o % 2 == 1 && o >= 1 && o <= 31); }
    for (uint32_t v = 0; v < 64; ++v) CHECK((u63::laneScaleSlot(v / 2) < 2) == (v < 32));   // (sa, sa, sb, sb)
    for (int k = 0; k < 100000; ++k) {   // packed u32 word recomposes: four 7-bit sign indices + 4-bit scale
        const uint64_t a = (uint64_t)brng();
        CHECK(a == a % 128 + 128 * (a / 128 % 128) + 16384 * (a / 16384 % 128) + 2097152 * (a / 2097152 % 128) + 268435456 * (a / 268435456));
        CHECK(a / 268435456 < 16);
    }
    CHECK(62 * 128 * 2 <= 32767 && 4 * (2 * (2 * 62 * 128) * 31) < 16777216);
    for (uint32_t n = 0; n < 50; ++n) {
        const auto g = u63::groupsOf4(n); uint32_t sum = 0;
        for (uint32_t x : g) { CHECK(x >= 1 && x <= 4); sum += x; }
        CHECK(sum == n);
    }

    // grids: random magnitudes in [0, 62] (the format data is supplied by the caller)
    std::mt19937_64 r(99);
    auto mag8 = [&]() { uint64_t g = 0; for (int i = 0; i < 8; ++i) g |= (uint64_t)(r() % 63) << (8 * i); return g; };
    auto mag4 = [&]() { uint32_t g = 0; for (int i = 0; i < 4; ++i) g |= (uint32_t)(r() % 63) << (8 * i); return g; };
    std::vector<uint64_t> g2xxs(256), g2xs(512), g2s(1024); std::vector<uint32_t> g3xxs(256), g3s(512);
    for (auto& x : g2xxs) x = mag8();
    for (auto& x : g2xs) x = mag8();
    for (auto& x : g2s) x = mag8();
    for (auto& x : g3xxs) x = mag4();
    for (auto& x : g3s) x = mag4();
    g2xxs[0] = 0x3E3E3E3E3E3E3E3Eull;   // extreme magnitudes (62) to exercise the no-saturation bound
    IqTables tb; tb.iq2xxs_grid = g2xxs.data(); tb.iq2xs_grid = g2xs.data(); tb.iq2s_grid = g2s.data();
    tb.iq3xxs_grid = g3xxs.data(); tb.iq3s_grid = g3s.data();
    iq512_set_tables(tb);
    for (uint32_t t : tys) CHECK(iq512_tables_ready(t));
    const u63::IqGridsHost G{g2xxs.data(), g2xs.data(), g2s.data(), g3xxs.data(), g3s.data()};

    for (int ti = 0; ti < 5; ++ti) {
        const uint32_t ty = tys[ti], bb = byt[ti];
        const uint32_t nb = 3, rows = 7, ntmax = 9;
        const size_t row_bytes = (size_t)nb * bb;
        std::vector<uint8_t> w(rows * row_bytes);
        for (auto& x : w) x = (uint8_t)r();
        for (uint32_t rr = 0; rr < rows; ++rr) for (uint32_t b = 0; b < nb; ++b) put16(&w[rr * row_bytes + b * bb], rand_scale_f16());
        std::vector<Q8KBlock> acts(ntmax * nb);
        for (uint32_t t = 0; t < ntmax; ++t) for (uint32_t b = 0; b < nb; ++b) {
            Q8KBlock& a = acts[t * nb + b];
            a.d = 0.01f + 0.001f * (float)(r() % 100);
            for (int i = 0; i < 256; ++i) a.qs[i] = (int8_t)((int)(r() % 255) - 127);
            if (t == 0 && b == 0) for (int i = 0; i < 256; ++i) a.qs[i] = (i % 2) ? 127 : -127;   // extreme activations
            std::memset(a.bsums, 0, sizeof a.bsums);
        }
        // spec dot (double) for every (token, row)
        std::vector<double> ref(ntmax * rows), refmag(ntmax * rows);
        for (uint32_t t = 0; t < ntmax; ++t) for (uint32_t rr = 0; rr < rows; ++rr) {
            double tot = 0, mag = 0;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint8_t* blk = &w[rr * row_bytes + b * bb];
                int wv[256], sc[256]; double K;
                u63::spec_iq_block(ty, blk, G, wv, sc, &K);
                int64_t isum = 0, imag = 0;
                for (int v = 0; v < 256; ++v) {
                    const int64_t q = acts[t * nb + b].qs[v];
                    isum += (int64_t)sc[v] * wv[v] * q; imag += (int64_t)sc[v] * std::abs(wv[v]) * std::llabs(q);
                }
                const double d = df16(blk) * acts[t * nb + b].d;
                tot += d * (double)isum * K; mag += std::fabs(d) * (double)imag * K;
            }
            ref[t * rows + rr] = tot; refmag[t * rows + rr] = mag;
        }
        // every token count 1..9 (groups of at most four) gives the spec value, and identical floats per token
        std::vector<float> single(ntmax * rows);
        for (uint32_t t = 0; t < ntmax; ++t) {
            std::vector<float> o(rows, NAN);
            iq512_rows(ty, w.data(), row_bytes, 0, rows, nb, &acts[t * nb], 1, o.data(), rows);
            for (uint32_t rr = 0; rr < rows; ++rr) single[t * rows + rr] = o[rr];
        }
        bool ok = true;
        for (size_t i = 0; i < single.size(); ++i) ok = ok && std::fabs(single[i] - ref[i]) <= 1e-5 * refmag[i] + 1e-6;
        if (!ok && g_fail < 40) std::printf("  i-quant type %u differs from the spec decode\n", ty);
        CHECK(ok);
        for (uint32_t nt = 1; nt <= ntmax; ++nt) {
            std::vector<float> o(nt * rows + 3, NAN);
            iq512_rows(ty, w.data(), row_bytes, 2, rows, nb, acts.data(), nt, o.data(), rows);
            for (uint32_t t = 0; t < nt; ++t) for (uint32_t rr = 2; rr < rows; ++rr) CHECK(o[t * rows + rr] == single[t * rows + rr]);
            for (uint32_t t = 0; t < nt; ++t) for (uint32_t rr = 0; rr < 2; ++rr) CHECK(std::isnan(o[t * rows + rr]));   // rows outside [r0, r1) untouched
        }
        // gate/up pairs: act = silu(gate) * up with gate row r, up row n_ff + r
        {
            const uint32_t n_ff = 3, nt = 5;
            std::vector<float> a(nt * n_ff, NAN);
            iq512_gu_rows(ty, w.data(), row_bytes, n_ff, 0, n_ff, nb, acts.data(), nt, a.data(), n_ff);
            for (uint32_t t = 0; t < nt; ++t) for (uint32_t rr = 0; rr < n_ff; ++rr) {
                const double g = single[t * rows + rr], u = single[t * rows + n_ff + rr];
                const double want = g / (1 + std::exp(-g)) * u;
                CHECK(std::fabs(a[t * n_ff + rr] - want) <= 1e-5 * (std::fabs(want) + 1e-3));
            }
        }
    }
}

int main() {
    test_half_bits();     std::printf("[half_precision_bit_conversion] done, failures so far %d\n", g_fail);
    test_block_dequant(); std::printf("[block_dequantization] done, failures so far %d\n", g_fail);
    test_gguf();          std::printf("[gguf_header_parsing] done, failures so far %d\n", g_fail);
    test_iq512();         std::printf("[iquant_row_decode] done, failures so far %d\n", g_fail);
    std::printf("u63af host: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}

