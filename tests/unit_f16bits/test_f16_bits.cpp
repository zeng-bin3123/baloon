// test_f16_bits.cpp - host invariants of the F16BitsRounding theory (rneStep, rne_below_half, rne_above_half,
// rne_tie_even, rne_le_succ) and everything built on it: exhaustive rounding vs a double-precision nearest-even oracle,
// the inf/NaN versus finite-overflow split, subnormal widening, bf16 rounding, and the artifact dequantizers.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/artifact_dequant.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/f16_conversion.hpp"

using namespace baloon;
using f16bits::f16_from_f32;
using f16bits::f32_from_f16;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool half_is_nan(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0; }

// ---- theory: rneStep ----------------------------------------------------------------------------------------
static void test_rne_step() {
    for (uint32_t h = 0; h < 4100; ++h) {
        for (uint32_t rem = 0; rem < 8192; ++rem) {
            const uint32_t r = f16bits::rne_step(h, rem);
            if (rem < 4096) CHECK(r == h);                          // rne_below_half
            if (rem > 4096) CHECK(r == h + 1);                      // rne_above_half
            if (rem == 4096) CHECK(r % 2 == 0);                     // rne_tie_even
            CHECK(h <= r && r <= h + 1);                            // rne_le_succ
        }
    }
    // the definition itself: rem > 4096 or (rem = 4096 and h odd)
    CHECK(f16bits::rne_step(6, 4096) == 6 && f16bits::rne_step(7, 4096) == 8);
    CHECK(f16bits::rne_step(0, 4097) == 1 && f16bits::rne_step(0, 4095) == 0);
}

// ---- oracle: nearest-even half of a float, computed in double -------------------------------------------------
static uint16_t oracle_f16(float x) {
    const uint32_t u = bits(x);
    const uint16_t sign = (uint16_t)((u >> 16) & 0x8000u);
    const double a = std::fabs((double)x);
    if (a >= 65520.0) return (uint16_t)(sign | 0x7C00);              // rounds to the next (overflow) value
    // largest finite half <= a, by binary search on the monotone value of codes 0..0x7BFF
    int lo = 0, hi = 0x7BFF;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if ((double)f32_from_f16((uint16_t)mid) <= a) lo = mid; else hi = mid - 1;
    }
    const double vl = (double)f32_from_f16((uint16_t)lo);
    if (lo == 0x7BFF) {                                              // between max finite and the 65520 threshold
        return (uint16_t)(sign | (uint16_t)lo);
    }
    const double vh = (double)f32_from_f16((uint16_t)(lo + 1));
    const double dl = a - vl, dh = vh - a;
    int pick = dl < dh ? lo : (dh < dl ? lo + 1 : ((lo & 1) ? lo + 1 : lo));
    return (uint16_t)(sign | (uint16_t)pick);
}

static void test_widen_exhaustive() {
    int nan_count = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const float a = f32_from_f16((uint16_t)h);
        const float b = f16::half_to_float((uint16_t)h);              // independent earlier implementation
        CHECK(bits(a) == bits(b));
        if (half_is_nan((uint16_t)h)) { ++nan_count; CHECK(std::isnan(a)); continue; }
        // encode(decode(h)) == h for every non-NaN half
        CHECK(f16_from_f32(a) == (uint16_t)h);
    }
    CHECK(nan_count == 2 * 1023);
    // subnormal normalisation: value is m * 2^-24 exactly, exponent field 113 - s
    for (uint32_t m = 1; m < 1024; ++m) {
        const float v = f32_from_f16((uint16_t)m);
        CHECK((double)v == (double)m * std::ldexp(1.0, -24));
        uint32_t mm = m, s = 0;
        while (mm < 1024) { mm *= 2; ++s; }
        CHECK(s <= 10);
        CHECK(((bits(v) >> 23) & 0xFF) == 113 - s);
        CHECK(f32_from_f16((uint16_t)(0x8000 | m)) == -v);
    }
    // normal exponents: ex + 112 as a signed quantity
    for (uint32_t ex = 1; ex <= 30; ++ex) CHECK(((bits(f32_from_f16((uint16_t)(ex << 10))) >> 23) & 0xFF) == ex + 112);
    CHECK(f32_from_f16(0x7C00) == INFINITY && f32_from_f16(0xFC00) == -INFINITY);
    CHECK(bits(f32_from_f16(0x8000)) == 0x80000000u && bits(f32_from_f16(0)) == 0);
}

static void test_encode_oracle() {
    std::mt19937_64 rng(12345);
    long n = 0;
    auto one = [&](float x) {
        if (std::isnan(x)) return;
        ++n;
        CHECK(f16_from_f32(x) == oracle_f16(x));
        CHECK(f16_from_f32(x) == f16::float_to_half(x) || std::isnan(x));
    };
    for (int i = 0; i < 3000000; ++i) one(flt((uint32_t)rng()));                     // every exponent
    for (int i = 0; i < 1500000; ++i) {                                               // focus on the half range
        const double mag = std::ldexp(1.0 + (double)(rng() & 0x7FFFFF) / 8388608.0, (int)(rng() % 44) - 28);
        one((float)((rng() & 1) ? -mag : mag));
    }
    // all exact ties and their neighbours: midpoints of adjacent halves
    for (uint32_t h = 0; h < 0x7BFF; ++h) {
        const double mid = ((double)f32_from_f16((uint16_t)h) + (double)f32_from_f16((uint16_t)(h + 1))) / 2.0;
        const float m = (float)mid;
        if ((double)m == mid) {
            const uint16_t r = f16_from_f32(m);
            CHECK(r % 2 == 0);                                                         // ties to even
            CHECK(r == h || r == h + 1);
        }
        const uint32_t mb = bits((float)mid);
        for (int d = -2; d <= 2; ++d) { one(flt(mb + d)); one(-flt(mb + d)); }
    }
    CHECK(n > 4000000);
}

static void test_inf_nan_split() {
    for (uint32_t sign : {0u, 0x80000000u}) {
        const uint16_t s16 = sign ? 0x8000 : 0;
        CHECK(f16_from_f32(flt(sign | 0x7F800000u)) == (uint16_t)(s16 | 0x7C00));        // inf stays inf
        for (uint32_t man : {1u, 2u, 0x1000u, 0x400000u, 0x7FFFFFu, 0x2000u}) {
            const uint16_t r = f16_from_f32(flt(sign | 0x7F800000u | man));
            CHECK(half_is_nan(r) && (r & 0x8000) == s16);                                // NaN stays NaN (even payload 1)
        }
        for (uint32_t e = 143; e <= 254; ++e)                                            // finite overflow -> inf, never NaN
            for (uint32_t man : {0u, 1u, 0x7FFFFFu, 0x400000u}) {
                const uint16_t r = f16_from_f32(flt(sign | (e << 23) | man));
                CHECK(r == (uint16_t)(s16 | 0x7C00) && !half_is_nan(r));
            }
    }
    CHECK(f16_from_f32(65504.0f) == 0x7BFF && f16_from_f32(65519.0f) == 0x7BFF && f16_from_f32(65520.0f) == 0x7C00);
    CHECK(f16_from_f32(0.0f) == 0 && f16_from_f32(-0.0f) == 0x8000);
    CHECK((255 - 127 + 15) >= 31);                                                       // conflation hazard: raw exp 255 passes `>= 31`
}

// ---- bf16 ---------------------------------------------------------------------------------------------------
static void test_bf16() {
    std::mt19937_64 rng(777);
    for (uint32_t q = 0; q < 65536; ++q) {                                               // exact for already-bf16 values
        const float f = flt(q << 16);
        if (std::isnan(f)) { CHECK(std::isnan(f16bits::bf16_bits_to_f32(f16bits::round_f32_to_bf16_bits(f)))); continue; }
        CHECK(f16bits::round_f32_to_bf16_bits(f) == q);
        CHECK(bits(f16bits::bf16_bits_to_f32((uint16_t)q)) == (q << 16));
    }
    for (int i = 0; i < 2000000; ++i) {
        const uint32_t u = (uint32_t)rng();
        if ((u & 0x7F800000u) == 0x7F800000u) continue;                                  // finite only
        const uint32_t lo = u & 0xFFFF, qh = u >> 16;
        const uint16_t r = f16bits::round_f32_to_bf16_bits(flt(u));
        if (lo < 0x8000) CHECK(r == qh);                                                 // below half: truncate
        if (lo > 0x8000) CHECK(r == (uint16_t)(qh + 1));                                 // above half: round up
        if (lo == 0x8000) CHECK(r % 2 == 0);                                             // tie: even
    }
    for (uint32_t qh : {0u, 1u, 0x3F80u, 0x3F81u, 0x7F7Eu, 0x7F7Fu, 0x8001u}) {          // explicit ties
        const uint16_t r = f16bits::round_f32_to_bf16_bits(flt((qh << 16) | 0x8000u));
        CHECK(r % 2 == 0 && (r == qh || r == qh + 1));
    }
    CHECK(f16bits::round_f32_to_bf16_bits(flt(0x7F7FFFFFu)) == 0x7F80);                  // max finite rounds to inf
}

// ---- dequantizers -------------------------------------------------------------------------------------------
static void test_dequant() {
    using namespace artifact;
    CHECK(read_u16((const uint8_t*)"\x34\x12") == 0x1234);
    CHECK(fp16_to_fp32(0x3C00) == 1.0f && bf16_to_fp32(0x3F80) == 1.0f && fp16_to_fp32(0xC000) == -2.0f);
    size_t nt = 0;
    const BlockType* t = block_types(&nt);
    CHECK(nt == 10);
    for (size_t i = 0; i < nt; ++i) CHECK(find_block_type(t[i].name) == &t[i]);
    CHECK(find_block_type("nope") == nullptr);

    // q8_0: y = d * q
    uint8_t b8[34];
    const uint16_t d = f16_from_f32(0.5f);
    b8[0] = d & 0xFF; b8[1] = d >> 8;
    for (int j = 0; j < 32; ++j) b8[2 + j] = (uint8_t)(int8_t)(j - 16);
    float y[256];
    dequantize_q8_0(b8, y);
    for (int j = 0; j < 32; ++j) CHECK(y[j] == 0.5f * (float)(j - 16));
    // iq4_nl: low nibbles first half, high nibbles second half
    uint8_t bn[18];
    bn[0] = d & 0xFF; bn[1] = d >> 8;
    for (int j = 0; j < 16; ++j) bn[2 + j] = (uint8_t)(j | ((15 - j) << 4));
    dequantize_iq4_nl(bn, y);
    for (int j = 0; j < 16; ++j) {
        CHECK(y[j] == 0.5f * (float)kvalues_iq4nl[j]);
        CHECK(y[j + 16] == 0.5f * (float)kvalues_iq4nl[15 - j]);
    }
    CHECK(kvalues_iq4nl[0] == -127 && kvalues_iq4nl[15] == 113);
    for (int i = 1; i < 16; ++i) CHECK(kvalues_iq4nl[i] > kvalues_iq4nl[i - 1]);
    // q4_0: y = (nibble - 8) * d
    uint8_t b4[18];
    b4[0] = d & 0xFF; b4[1] = d >> 8;
    for (int j = 0; j < 16; ++j) b4[2 + j] = (uint8_t)(j | ((15 - j) << 4));
    dequantize_q4_0(b4, y);
    for (int j = 0; j < 16; ++j) { CHECK(y[j] == 0.5f * (float)(j - 8)); CHECK(y[j + 16] == 0.5f * (float)(15 - j - 8)); }
    // plain arrays
    const uint16_t hv[3] = {f16_from_f32(1.5f), f16_from_f32(-3.0f), 1};
    float o3[3];
    dequantize_f16((const uint8_t*)hv, o3, 3);
    CHECK(o3[0] == 1.5f && o3[1] == -3.0f && o3[2] == std::ldexp(1.0f, -24));
    const uint16_t bv[2] = {0x3FC0, 0xBF80};
    dequantize_bf16((const uint8_t*)bv, o3, 2);
    CHECK(o3[0] == 1.5f && o3[1] == -1.0f);
    const float fv[2] = {2.25f, -7.0f};
    dequantize_f32((const uint8_t*)fv, o3, 2);
    CHECK(o3[0] == 2.25f && o3[1] == -7.0f);
    // every block type produces finite output of the declared width on structured data, wrappers == layout module
    std::mt19937 rng(5);
    for (size_t i = 0; i < nt; ++i) {
        std::vector<uint8_t> blk(t[i].bytes);
        for (int rep = 0; rep < 200; ++rep) {
            for (auto& c : blk) c = (uint8_t)rng();
            const uint16_t sc = f16_from_f32(0.125f);                                   // keep the fp16 scales finite
            if (!std::strcmp(t[i].name, "q3_K")) { blk[108] = sc & 0xFF; blk[109] = sc >> 8; }
            else if (!std::strcmp(t[i].name, "q6_K")) { blk[208] = sc & 0xFF; blk[209] = sc >> 8; }
            else { blk[0] = sc & 0xFF; blk[1] = sc >> 8; if (blk.size() >= 144 && std::strcmp(t[i].name, "iq4_xs")) { blk[2] = sc & 0xFF; blk[3] = sc >> 8; } }
            std::vector<float> out(t[i].elems, 12345.f);
            t[i].fn(blk.data(), out.data());
            for (float v : out) CHECK(std::isfinite(v) && v != 12345.f);
        }
    }
}

int main() {
    test_rne_step();             std::printf("[ ok ] test_rne_step\n");
    test_widen_exhaustive();     std::printf("[ ok ] test_widen_exhaustive\n");
    test_encode_oracle();        std::printf("[ ok ] test_encode_oracle\n");
    test_inf_nan_split();        std::printf("[ ok ] test_inf_nan_split\n");
    test_bf16();                 std::printf("[ ok ] test_bf16\n");
    test_dequant();              std::printf("[ ok ] test_dequant\n");
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("F16BITS UNIT TESTS FAILED\n"); return 1; }
    std::printf("F16BITS UNIT TESTS PASSED\n");
    return 0;
}
