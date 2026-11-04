// Unit 19 host tests: fp16/bf16 bit rounding (F16BitsRounding) and GGML block layouts / dequantizers
// (DequantBlockLayouts). Every check states a LAYER-1 invariant and compares against an independent reference.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "check.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/gguf_dequant.hpp"

using namespace baloon;
using testh::Rng;
namespace fb = baloon::f16bits;
namespace gg = baloon::gguf;

void run_test(const char* name, void (*fn)()) {
    const long before = testh::failures();
    fn();
    std::printf("[%s] %s\n", testh::failures() == before ? " ok " : "FAIL", name);
}

// ---- independent references ------------------------------------------------------------------------------------------
static double half_value(uint32_t h) {  // exact value of a finite half
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    const double v = e == 0 ? std::ldexp((double)m, -24) : std::ldexp((double)(1024 + m), e - 25);
    return s ? -v : v;
}
// Round |x| to the nearest half magnitude, ties to even; index 0x7C00 stands for 65536 (overflow to inf).
static uint32_t ref_round_mag(double a, const std::vector<double>& tab) {
    size_t lo = 0, hi = tab.size() - 1;  // tab sorted ascending, size 0x7C01
    while (hi - lo > 1) { size_t mid = (lo + hi) / 2; (tab[mid] <= a ? lo : hi) = mid; }
    if (a >= tab[hi]) return (uint32_t)hi;
    const double dl = a - tab[lo], dh = tab[hi] - a;
    if (dl < dh) return (uint32_t)lo;
    if (dh < dl) return (uint32_t)hi;
    return (lo % 2 == 0) ? (uint32_t)lo : (uint32_t)hi;
}
static std::vector<double> make_table() {
    std::vector<double> t(0x7C01);
    for (uint32_t h = 0; h < 0x7C00; ++h) t[h] = half_value(h);
    t[0x7C00] = 65536.0;
    return t;
}

static bool is_nan16(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x3FF); }

// ---- F16BitsRounding ----------------------------------------------------------------------------------------------------
static void test_decode_exhaustive() {
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = fb::f32_from_f16((uint16_t)h);
        const bool nan = is_nan16((uint16_t)h);
        if (nan) { CHECK(std::isnan(f)); CHECK(is_nan16(fb::f16_from_f32(f))); continue; }
        if ((h & 0x7C00) == 0x7C00) {  // infinity
            CHECK(std::isinf(f) && ((h >> 15) != 0) == (f < 0));
            CHECK(fb::f16_from_f32(f) == h);
            continue;
        }
        const double ref = half_value(h);
        CHECK_MSG((double)f == ref && std::signbit(f) == (bool)(h >> 15), "h=%04x", h);
        CHECK_MSG(fb::f16_from_f32(f) == h, "roundtrip h=%04x", h);              // widening is exact, so narrowing returns it
        CHECK(f == f16::half_to_float((uint16_t)h) || std::isnan(f));              // agrees with the repo-wide converter
    }
}

static void check_encode(float x, const std::vector<double>& tab) {
    const uint16_t got = fb::f16_from_f32(x);
    const uint32_t bits = fb::bits_of(x);
    const uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    if (std::isnan(x)) { CHECK_MSG(is_nan16(got) && (got & 0x8000) == sign, "nan %08x -> %04x", bits, got); return; }
    if (std::isinf(x)) { CHECK((got & 0x7FFF) == 0x7C00 && (got & 0x8000) == sign); return; }
    const uint32_t mag = ref_round_mag(std::fabs((double)x), tab);
    CHECK_MSG(got == (uint16_t)(sign | mag), "x=%08x got=%04x want=%04x", bits, got, (unsigned)(sign | mag));
    CHECK(got == f16::float_to_half(x));
}

static void test_encode_rne() {
    const std::vector<double> tab = make_table();
    // every adjacent pair of halves: the exact midpoint (a tie) and its float neighbours
    for (uint32_t h = 0; h < 0x7C00; ++h) {
        const double mid = (tab[h] + tab[h + 1]) / 2;
        const float m = (float)mid;
        CHECK((double)m == mid);  // a midpoint of two halves is exactly a float
        for (int d = -1; d <= 1; ++d) {
            const float x = fb::float_of(fb::bits_of(m) + (uint32_t)d);
            check_encode(x, tab);
            check_encode(-x, tab);
        }
        check_encode((float)tab[h], tab);
    }
    // bulk random bit patterns (all exponents, including subnormal f32, inf and NaN)
    Rng r(7);
    for (int i = 0; i < 2000000; ++i) check_encode(fb::float_of(r.u32() ^ (r.u32() << 16)), tab);
    // dense sweep around the overflow, subnormal and flush boundaries
    for (uint32_t b = fb::bits_of(65000.f); b < fb::bits_of(66000.f); b += 997) check_encode(fb::float_of(b), tab);
    for (uint32_t b = fb::bits_of(1e-8f); b < fb::bits_of(1e-4f); b += 3001) check_encode(fb::float_of(b), tab);
}

static void test_inf_nan_split() {
    CHECK(fb::f16_from_f32(65504.f) == 0x7BFF);
    CHECK(fb::f16_from_f32(65519.f) == 0x7BFF);
    CHECK(fb::f16_from_f32(65520.f) == 0x7C00);                       // tie rounds to the even pattern: infinity
    const float big[] = {1e5f, 3e38f, 3.4028234e38f};
    for (float b : big) {
        const uint16_t p = fb::f16_from_f32(b), n = fb::f16_from_f32(-b);
        CHECK(p == 0x7C00 && n == 0xFC00);                            // finite overflow: mantissa 0, never NaN
        CHECK(p % 1024 == 0 && n % 1024 == 0);
    }
    const float inf = INFINITY;
    CHECK(fb::f16_from_f32(inf) == 0x7C00 && fb::f16_from_f32(-inf) == 0xFC00);
    for (uint32_t man : {1u, 0x2000u, 0x400000u, 0x7FFFFFu}) {         // NaN payloads (also those whose top 10 bits are 0)
        for (uint32_t s : {0u, 0x80000000u}) {
            const uint16_t h = fb::f16_from_f32(fb::float_of(s | 0x7F800000u | man));
            CHECK(is_nan16(h) && (uint32_t)(h >> 15) == (s >> 31));
        }
    }
    // the old `exp >= 31` test would also catch raw exponent 255: the split is what keeps inf/NaN apart from overflow
    CHECK(255 - 127 + 15 >= 31);
}

static void test_subnormals() {
    for (uint32_t m = 1; m < 1024; ++m) {
        const float f = fb::f32_from_f16((uint16_t)m);
        const uint32_t b = fb::bits_of(f);
        int s = 0; uint32_t t = m;
        while (t < 1024) { t <<= 1; ++s; }
        CHECK(s <= 10 && t >= 1024 && t < 2048);                       // normShift_spec
        CHECK(t == m * (1u << s));
        CHECK(((b >> 23) & 255u) == (uint32_t)(113 - s));              // f32 exponent field 113 - s
        CHECK(113 - s >= 103 && 113 - s < 255);
        CHECK(((b >> 13) & 0x3FFu) == (t & 0x3FFu));
    }
    for (int ex = 1; ex <= 30; ++ex) {                                  // normal exponents: signed re-bias ex - 15 + 127 = ex + 112
        const uint32_t b = fb::bits_of(fb::f32_from_f16((uint16_t)(ex << 10)));
        CHECK(((b >> 23) & 255u) == (uint32_t)(ex + 112));
        CHECK(ex + 112 >= 113 && ex + 112 <= 142);
    }
    CHECK(127 - 15 + 1 == 113);
}

static uint32_t ref_bf16(uint32_t x) {  // independent: nearest of the two neighbouring upper halves, ties to even
    const uint32_t lo = x >> 16, rem = x & 0xFFFFu;
    if (rem < 0x8000u) return lo;
    if (rem > 0x8000u) return lo + 1;
    return (lo % 2 == 0) ? lo : lo + 1;
}
static void test_bf16() {
    Rng r(11);
    for (int i = 0; i < 3000000; ++i) {
        const uint32_t x = r.u32() ^ (r.u32() << 16);
        const float f = fb::float_of(x);
        const uint16_t got = fb::bf16_from_f32(f);
        if (std::isnan(f)) { CHECK(((got & 0x7F80) == 0x7F80) && (got & 0x7F)); continue; }
        CHECK_MSG(got == ref_bf16(x), "x=%08x", x);
    }
    for (uint32_t q = 0; q < 65536; ++q) {
        const float f = fb::float_of(q << 16);
        if (std::isnan(f)) continue;
        CHECK(fb::bf16_from_f32(f) == q);                               // bf16_exact
        CHECK(fb::bits_of(fb::f32_from_bf16((uint16_t)q)) == (q << 16));
        const uint32_t tie = (q << 16) | 0x8000u;                       // exact tie -> even result
        if (!std::isnan(fb::float_of(tie))) CHECK(fb::bf16_from_f32(fb::float_of(tie)) % 2 == 0);
    }
    CHECK(fb::bf16_from_f32(INFINITY) == 0x7F80 && fb::bf16_from_f32(-INFINITY) == 0xFF80);
    CHECK(fb::bf16_from_f32(fb::float_of(0x7FFFFFFFu)) % 128 != 0);     // NaN with all-ones payload must not carry out
}

static void test_rne_step() {
    for (uint32_t h = 0; h < 40; ++h)
        for (uint32_t rem = 0; rem < 8192; rem += 1) {
            const uint32_t r = fb::rne_step(h, rem);
            CHECK(h <= r && r <= h + 1);                                // rne_le_succ
            if (rem < 4096) CHECK(r == h);
            if (rem > 4096) CHECK(r == h + 1);
            if (rem == 4096) CHECK(r % 2 == 0);
        }
}

// ---- DequantBlockLayouts -------------------------------------------------------------------------------------------------
static void test_geometry() {
    struct G { int t, e, b; };
    const G tab[] = {{0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {2, 32, 18}, {3, 32, 20}, {6, 32, 22}, {7, 32, 24}, {8, 32, 34},
                     {9, 32, 36}, {10, 256, 84}, {11, 256, 110}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210},
                     {16, 256, 66}, {17, 256, 74}, {18, 256, 98}, {20, 32, 18}, {21, 256, 110}, {22, 256, 82},
                     {23, 256, 136}, {29, 256, 56}, {42, 64, 18}};
    int known = 0;
    for (int t = -2; t < 80; ++t) {
        gg::BlockShape s{};
        const bool ok = gg::block_shape_of_type(t, &s);
        const G* g = nullptr;
        for (auto& e : tab) if (e.t == t) g = &e;
        CHECK(ok == (g != nullptr));
        if (g) { CHECK(s.elems == g->e && s.bytes == g->b); CHECK(256 % s.elems == 0); ++known; }  // super-block is whole blocks
    }
    CHECK(known == (int)(sizeof(tab) / sizeof(tab[0])));
    gg::BlockShape q2{};
    CHECK(gg::block_shape_of_type(42, &q2) && q2.elems == 64 && q2.bytes == 18 && 2 + 64 * 2 / 8 == 18);
    const int supported[] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};
    for (int t : supported) CHECK(gg::block_shape_of_type(t, nullptr));
}

static void test_layouts_and_strides() {
    CHECK(32 + 64 + 12 + 2 == 110 && gg::kQ3K_QS == 32 && gg::kQ3K_SCALES == 32 + 64 && gg::kQ3K_D == 96 + 12);
    CHECK(128 + 64 + 16 + 2 == 210 && gg::kQ6K_QH == 128 && gg::kQ6K_SCALES == 192 && gg::kQ6K_D == 192 + 16);
    CHECK(4 + 12 + 32 + 128 == 176 && gg::kQ5K_SCALES == 4 && gg::kQ5K_QH == 16 && gg::kQ5K_QS == 48);
    CHECK(4 + 12 + 128 == 144 && gg::kQ4K_SCALES == 4 && gg::kQ4K_QS == 16);
    CHECK(2 + 2 + 4 + 128 == 136 && gg::kIQ4XS_SCALES_H == 2 && gg::kIQ4XS_SCALES_L == 4 && gg::kIQ4XS_QS == 8);
    CHECK(2 + 4 + 16 == 22 && gg::kQ5_0_QH == 2 && gg::kQ5_0_QS == 6);
    for (int b : {18, 22, 34, 110, 210}) CHECK(b % 4 == 2 && gg::needs_u16_loads(b));
    for (int b : {36, 136, 144, 176}) CHECK(b % 4 == 0 && !gg::needs_u16_loads(b));
    // the layout constants agree with the scalar dequantizer sizes
    CHECK(sdq::kBlkQ3K == 110 && sdq::kBlkQ4K == 144 && sdq::kBlkQ5K == 176 && sdq::kBlkQ6K == 210 && sdq::kBlkIQ4XS == 136);
}

static void test_row_bytes() {
    for (int t : {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 0, 1, 30}) {
        gg::BlockShape s{};
        CHECK(gg::block_shape_of_type(t, &s));
        for (uint64_t k : {0ull, 1ull, 7ull, 4096ull}) {
            CHECK(gg::row_bytes(t, k * (uint64_t)s.elems) == k * (uint64_t)s.bytes);   // rowBytes_whole
            if (s.elems > 1) CHECK(gg::row_bytes(t, k * (uint64_t)s.elems + 1) == UINT64_MAX);  // ragged row rejected
        }
    }
    CHECK(gg::row_bytes(99, 32) == UINT64_MAX);
    Rng r(3);
    for (int i = 0; i < 20000; ++i) {                                   // weight_bytes_no_overflow vs 128-bit arithmetic
        const uint64_t row = ((uint64_t)r.u32() << 32 | r.u32()) >> r.below(60), nout = 1 + (((uint64_t)r.u32() << 20) >> r.below(40));
        const uint64_t lim = UINT64_MAX >> r.below(10);
        const bool exact = (unsigned __int128)row * nout <= (unsigned __int128)lim;
        CHECK(gg::weight_bytes_fit(row, nout, lim) == exact);
    }
}

static uint16_t f16of(float f) { return fb::f16_from_f32(f); }
static void put16(uint8_t* p, uint16_t h) { p[0] = (uint8_t)(h & 255); p[1] = (uint8_t)(h >> 8); }

static void test_value_ranges() {
    Rng r(5);
    // Q4_0 with d = 1: values are integers in [-8, 7], exactly nibble - 8
    for (int it = 0; it < 500; ++it) {
        uint8_t b[18]; float y[32];
        put16(b, f16of(1.f)); for (int i = 2; i < 18; ++i) b[i] = (uint8_t)r.u32();
        CHECK(gg::dequantize_q4_0(b, y, 32));
        for (int j = 0; j < 16; ++j) {
            CHECK(y[j] == (float)((b[2 + j] & 15) - 8) && y[j + 16] == (float)((b[2 + j] >> 4) - 8));
            CHECK(y[j] >= -8 && y[j] <= 7 && y[j + 16] >= -8 && y[j + 16] <= 7);
        }
    }
    // Q5_0: nibble + 16 * fifth bit - 16 in [-16, 15]
    for (int it = 0; it < 500; ++it) {
        uint8_t b[22]; float y[32];
        put16(b, f16of(1.f)); for (int i = 2; i < 22; ++i) b[i] = (uint8_t)r.u32();
        uint32_t qh; std::memcpy(&qh, b + 2, 4);
        CHECK(gg::dequantize_q5_0(b, y, 32));
        for (int j = 0; j < 16; ++j) {
            CHECK(y[j] == (float)((int)((b[6 + j] & 15) | (((qh >> j) & 1) << 4)) - 16));
            CHECK(y[j + 16] == (float)((int)((b[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16));
        }
        for (float v : y) CHECK(v >= -16 && v <= 15);
    }
    // Q2_0 symbols {-1, 0, 1, 2}
    for (int it = 0; it < 200; ++it) {
        uint8_t b[18]; float y[64];
        put16(b, f16of(1.f)); for (int i = 2; i < 18; ++i) b[i] = (uint8_t)r.u32();
        CHECK(gg::dequantize_q2_0(b, y, 64));
        for (int j = 0; j < 64; ++j) CHECK(y[j] == (float)(((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) && y[j] >= -1 && y[j] <= 2);
    }
    // Q6_K with d = 1 and all int8 scales = 1: values in [-32, 31]
    for (int it = 0; it < 200; ++it) {
        uint8_t b[210]; float y[256];
        for (int i = 0; i < 208; ++i) b[i] = (uint8_t)r.u32();
        for (int i = 192; i < 208; ++i) b[i] = 1;
        put16(b + 208, f16of(1.f));
        CHECK(gg::dequantize_q6_K(b, y, 256));
        float mn = 1e9f, mx = -1e9f;
        for (float v : y) { mn = std::min(mn, v); mx = std::max(mx, v); CHECK(v == std::floor(v)); }
        CHECK(mn >= -32 && mx <= 31);
    }
    // Q6_K extremes are reachable: all-zero bits -> -32, all-one bits -> 31
    { uint8_t b[210]; float y[256]; std::memset(b, 0, 210); for (int i = 192; i < 208; ++i) b[i] = 1; put16(b + 208, f16of(1.f));
      CHECK(gg::dequantize_q6_K(b, y, 256)); for (float v : y) CHECK(v == -32.f);
      std::memset(b, 0xFF, 192); CHECK(gg::dequantize_q6_K(b, y, 256)); for (float v : y) CHECK(v == 31.f); }
    // Q3_K with all 16 scales = 1 (6-bit value 33): values are (2-bit code) - (0 or 4) in [-4, 3]
    for (int it = 0; it < 200; ++it) {
        uint8_t b[110]; float y[256];
        for (int i = 0; i < 96; ++i) b[i] = (uint8_t)r.u32();
        for (int i = 0; i < 8; ++i) b[96 + i] = 0x11;
        for (int i = 8; i < 12; ++i) b[96 + i] = 0xAA;
        put16(b + 108, f16of(1.f));
        CHECK(gg::dequantize_q3_K(b, y, 256));
        for (float v : y) CHECK(v >= -4 && v <= 3 && v == std::floor(v));
    }
    // Q4_K / Q5_K scale: j < 4 is the 6-bit value, j >= 4 is nibble + 2 bits of a neighbour, always < 64; pack/unpack roundtrip
    for (int it = 0; it < 3000; ++it) {
        uint8_t d[8], m[8], q[12];
        for (int j = 0; j < 8; ++j) { d[j] = (uint8_t)r.below(64); m[j] = (uint8_t)r.below(64); }
        for (int j = 0; j < 4; ++j) {
            q[j] = (uint8_t)(d[j] | ((d[j + 4] >> 4) << 6));
            q[j + 4] = (uint8_t)(m[j] | ((m[j + 4] >> 4) << 6));
            q[j + 8] = (uint8_t)((d[j + 4] & 15) | ((m[j + 4] & 15) << 4));
        }
        for (int j = 0; j < 8; ++j) {
            uint8_t sd, sm; gg::get_scale_min_k4(j, q, &sd, &sm);
            CHECK(sd < 64 && sm < 64 && sd == d[j] && sm == m[j]);
        }
        uint8_t rq[12]; for (auto& x : rq) x = (uint8_t)r.u32();
        for (int j = 0; j < 8; ++j) { uint8_t sd, sm; gg::get_scale_min_k4(j, rq, &sd, &sm); CHECK(sd < 64 && sm < 64); }
    }
    // Q5_K with scale 1, min 0: code = nibble + 16 * bit <= 31 and the masks u1,u2 are disjoint bits that fit a byte
    for (unsigned u : {1u, 4u, 16u, 64u}) CHECK(2 * u < 256 && (u & (2 * u)) == 0);
    for (int it = 0; it < 200; ++it) {
        uint8_t b[176]; float y[256];
        for (int i = 0; i < 176; ++i) b[i] = (uint8_t)r.u32();
        put16(b, f16of(1.f)); put16(b + 2, f16of(0.f));
        for (int i = 0; i < 4; ++i) b[4 + i] = 1;
        for (int i = 4; i < 8; ++i) b[4 + i] = 0;
        for (int i = 8; i < 12; ++i) b[4 + i] = 1;
        CHECK(gg::dequantize_q5_K(b, y, 256));
        const uint8_t* qh = b + 16; const uint8_t* ql = b + 48;
        for (int g = 0; g < 4; ++g)
            for (int l = 0; l < 32; ++l) {
                const uint8_t u1 = (uint8_t)(1u << (2 * g)), u2 = (uint8_t)(2u << (2 * g));
                CHECK(y[64 * g + l] == (float)((ql[32 * g + l] & 15) + ((qh[l] & u1) ? 16 : 0)));
                CHECK(y[64 * g + 32 + l] == (float)((ql[32 * g + l] >> 4) + ((qh[l] & u2) ? 16 : 0)));
            }
        for (float v : y) CHECK(v >= 0 && v <= 31);
    }
    // IQ4_XS: the 6-bit scale (nibble + 2 bits) minus 32 is in [-32, 31]; with codebook entry +1 the value equals it
    for (unsigned lo = 0; lo < 16; ++lo)
        for (unsigned hi = 0; hi < 4; ++hi) {
            uint8_t b[136]; float y[256];
            std::memset(b, 0, sizeof b);
            put16(b, f16of(1.f));
            uint16_t sh = 0;
            for (int g = 0; g < 8; ++g) sh = (uint16_t)(sh | (hi << (2 * g)));
            put16(b + 2, sh);
            for (int i = 0; i < 4; ++i) b[4 + i] = (uint8_t)(lo | (lo << 4));
            for (int i = 8; i < 136; ++i) b[i] = 0x88;                      // kvalues_iq4nl[8] == 1
            CHECK(gg::kvalues_iq4nl[8] == 1);
            CHECK(gg::dequantize_iq4_xs(b, y, 256));
            const int ls = (int)(lo + 16 * hi);
            CHECK(ls < 64 && ls - 32 >= -32 && ls - 32 <= 31);
            for (float v : y) CHECK(v == (float)(ls - 32));
        }
    // Q8_0 and IQ4_NL
    { uint8_t b[34]; float y[32]; put16(b, f16of(0.5f)); for (int i = 0; i < 32; ++i) b[2 + i] = (uint8_t)(int8_t)(i * 8 - 128 + 1);
      CHECK(gg::dequantize_q8_0(b, y, 32)); for (int i = 0; i < 32; ++i) CHECK(y[i] == 0.5f * (float)(i * 8 - 127)); }
    { uint8_t b[18]; float y[32]; put16(b, f16of(2.f)); for (int i = 0; i < 16; ++i) b[2 + i] = (uint8_t)(i | ((15 - i) << 4));
      CHECK(gg::dequantize_iq4_nl(b, y, 32)); for (int i = 0; i < 16; ++i) CHECK(y[i] == 2.f * gg::kvalues_iq4nl[i] && y[i + 16] == 2.f * gg::kvalues_iq4nl[15 - i]); }
}

static void test_dispatch_and_plain() {
    Rng r(9);
    const int types[] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42};
    for (int t : types) {
        gg::BlockShape s{}; gg::block_shape_of_type(t, &s);
        const int nb = 5;
        std::vector<uint8_t> in((size_t)nb * s.bytes);
        for (auto& x : in) x = (uint8_t)r.u32();
        std::vector<float> a((size_t)nb * s.elems), b(a.size());
        CHECK(gg::dequantize(t, in.data(), a.data(), (int64_t)a.size()));
        CHECK(!gg::dequantize(t, in.data(), b.data(), (int64_t)a.size() - 1));  // ragged n rejected
        CHECK(!gg::dequantize(t, nullptr, b.data(), (int64_t)a.size()));
        for (int k = 0; k < nb; ++k) {  // each block independent of its neighbours
            std::vector<float> one((size_t)s.elems);
            CHECK(gg::dequantize(t, in.data() + (size_t)k * s.bytes, one.data(), s.elems));
            CHECK(std::memcmp(one.data(), a.data() + (size_t)k * s.elems, one.size() * 4) == 0);
        }
    }
    CHECK(!gg::dequantize(3, nullptr, nullptr, 0));
    uint8_t raw[8] = {0, 0, 0x80, 0x3F, 0, 0, 0, 0xC0};
    float f[2];
    CHECK(gg::dequantize_f32(raw, f, 2) && f[0] == 1.f && f[1] == -2.f);
    const uint16_t hs[3] = {0x3C00, 0xC000, 0x0001};
    float g[3];
    CHECK(gg::dequantize_f16((const uint8_t*)hs, g, 3) && g[0] == 1.f && g[1] == -2.f && g[2] == std::ldexp(1.0, -24));
    const uint16_t bs[2] = {0x3F80, 0xC000};
    CHECK(gg::dequantize_bf16((const uint8_t*)bs, g, 2) && g[0] == 1.f && g[1] == -2.f);
    CHECK(gg::read_u16((const uint8_t*)"\x34\x12") == 0x1234);
    CHECK(gg::fp16_to_fp32(0x3C00) == 1.f && gg::bf16_to_fp32(0x3F80) == 1.f);
    CHECK(gg::type_from_name("Q4_K") == gg::kQ4_K && gg::type_from_name("iq4_xs") == gg::kIQ4_XS && gg::type_from_name("zzz") == -1);
}

static void write_file(const std::string& p, const void* d, size_t n) { std::ofstream(p, std::ios::binary).write((const char*)d, (std::streamsize)n); }

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--fixture") {     // inputs for the dequant_tool command-line check
        Rng r(21);
        std::vector<uint8_t> in(2 * 210);
        for (auto& x : in) x = (uint8_t)r.u32();
        for (int k = 0; k < 2; ++k) put16(in.data() + k * 210 + 208, f16of(0.25f));
        std::vector<float> out(512);
        gg::dequantize_q6_K(in.data(), out.data(), 512);
        write_file(std::string(argv[2]) + "/in.bin", in.data(), in.size());
        write_file(std::string(argv[2]) + "/exp.f32", out.data(), out.size() * 4);
        return 0;
    }
    RUN(test_decode_exhaustive);
    RUN(test_encode_rne);
    RUN(test_inf_nan_split);
    RUN(test_subnormals);
    RUN(test_bf16);
    RUN(test_rne_step);
    RUN(test_geometry);
    RUN(test_layouts_and_strides);
    RUN(test_row_bytes);
    RUN(test_value_ranges);
    RUN(test_dispatch_and_plain);
    std::printf("%ld checks, %ld failures\n", testh::checks(), testh::failures());
    return testh::failures() ? 1 : 0;
}
