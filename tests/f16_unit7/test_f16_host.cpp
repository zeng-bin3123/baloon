// test_f16_host.cpp - host invariants of the fp16/bf16 bit conversions (LAYER 1 of F16BitsRounding): round-to-nearest-even
// against an independent oracle (nearest entry of the enumerated half table), the inf/NaN versus finite-overflow split,
// subnormal normalisation (bit 10 set, <= 10 shifts, exponent field 113 - s), the signed exponent re-bias, bf16 ties,
// plus the row dequantisers, the dequant tool and the weight-table aliases.
#define BALOON_DEQUANT_MAIN_DISABLED 1
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>
#include "baloon/core_weights.hpp"
#include "baloon/dequant_rows.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/f16_conversion.hpp"
#include "../../src/artifact/dequant.cpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static_assert(std::is_same<core::Conv, core::WConv>::value && std::is_same<core::Seg, core::WSeg>::value &&
              std::is_same<core::IndexRow, core::WIndexRow>::value && core::CHUNK == 8388608ull, "weight aliases");

// ---- independent oracles ------------------------------------------------------------------------------------------
static double decode_half(unsigned h) {  // direct definition, no shifts
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp(1024.0 + m, e - 25);
    return s ? -v : v;
}
static double g_tab[0x7C01];  // finite non-negative halves, strictly increasing; [0x7C00] = 2^16 (the value an overflow would carry to)
static unsigned oracle_f16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const unsigned sign = (u >> 16) & 0x8000u;
    if (std::isnan(f)) return 0x7E00u | sign;  // class only
    if (std::isinf(f)) return sign | 0x7C00u;
    const double a = std::fabs((double)f);
    if (a >= 65520.0) return sign | 0x7C00u;   // midpoint between 65504 and the next (2^16) rounds to even = inf
    int lo = 0, hi = 0x7BFF;
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (g_tab[mid] <= a) lo = mid; else hi = mid - 1; }
    unsigned r = (unsigned)lo;
    if (a != g_tab[lo]) {
        const double d1 = a - g_tab[lo], d2 = g_tab[lo + 1] - a;
        if (d2 < d1 || (d2 == d1 && (lo & 1))) r = (unsigned)lo + 1;
    }
    return sign | r;
}
static unsigned oracle_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if (std::isnan(f)) return ((u >> 16) | 0x40u) & 0xFFFFu;
    const uint32_t lo = u >> 16, rem = u & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && (lo & 1u))) return (lo + 1) & 0xFFFFu;
    return lo;
}
static float bits_to_f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool same_f16(float f) {
    const unsigned got = f16bits::round_f32_to_f16_bits(f), want = oracle_f16(f);
    if (std::isnan(f)) return (got & 0x7C00u) == 0x7C00u && (got & 0x3FFu) != 0 && (got & 0x8000u) == (want & 0x8000u);
    return got == want;
}

int main() {
    for (unsigned h = 0; h < 0x7C00; ++h) g_tab[h] = decode_half(h);
    g_tab[0x7C00] = 65536.0;

    // widen: every half pattern against the direct definition; round trip; agreement with the older implementation
    for (unsigned h = 0; h < 65536; ++h) {
        const float w = f16bits::widen_f16_bits((uint16_t)h);
        const double ref = decode_half(h);
        if (std::isnan(ref)) { CHECK(std::isnan(w)); CHECK(((f16bits::round_f32_to_f16_bits(w)) & 0x7C00u) == 0x7C00u && (f16bits::round_f32_to_f16_bits(w) & 0x3FFu) != 0); }
        else {
            CHECK((double)w == ref);
            CHECK(std::signbit(w) == (bool)(h & 0x8000u));
            CHECK(f16bits::round_f32_to_f16_bits(w) == h);  // exact round trip, -0 and subnormals included
        }
        CHECK(f16::half_to_float((uint16_t)h) == w || std::isnan(w));
    }

    // theory: rounding step lemmas
    for (uint32_t h = 0; h < 8; ++h) {
        for (uint32_t rem = 0; rem < 8192; ++rem) {
            const uint32_t r = f16bits::rne_step(h, rem);
            if (rem < 4096) CHECK(r == h);
            if (rem > 4096) CHECK(r == h + 1);
            if (rem == 4096) CHECK(r % 2 == 0);
            CHECK(r >= h && r <= h + 1);
        }
    }
    // theory: subnormal normalisation, 113 bias, signed re-bias
    for (unsigned m = 1; m < 1024; ++m) {
        unsigned mm = m, s = 0;
        while (mm < 1024) { mm <<= 1; ++s; }
        CHECK(mm >= 1024 && mm < 2048 && s <= 10 && mm == m * (1u << s));
        CHECK(113 - s >= 103 && 113 - s < 255);
        const uint32_t u = (uint32_t)f16bits::as_u32(f16bits::widen_f16_bits((uint16_t)m));
        CHECK(((u >> 23) & 0xFF) == 113 - s);
    }
    for (int ex = 1; ex <= 30; ++ex) {
        CHECK(ex - 15 + 127 == ex + 112 && ex + 112 >= 113 && ex + 112 <= 142);
        const uint32_t u = f16bits::as_u32(f16bits::widen_f16_bits((uint16_t)(ex << 10)));
        CHECK(((u >> 23) & 0xFF) == (unsigned)(ex + 112));
    }

    // inf / NaN versus finite overflow
    std::mt19937 rng(12345);
    for (int sgn = 0; sgn < 2; ++sgn) {
        const uint32_t sb = sgn ? 0x80000000u : 0;
        CHECK(f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x7F800000u)) == (sgn ? 0xFC00u : 0x7C00u));  // inf stays inf
        for (uint32_t man = 1; man < (1u << 23); man = man * 3 + 1)                                       // NaN stays NaN
            { const unsigned g = f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x7F800000u | man)); CHECK((g & 0x7C00u) == 0x7C00u && (g & 0x3FFu) != 0); }
        for (uint32_t man = 1; man < 4096; ++man)
            { const unsigned g = f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x7F800000u | man)); CHECK((g & 0x3FFu) != 0); }
        for (uint32_t re = 143; re <= 254; ++re)                                                          // finite overflow -> inf
            for (int k = 0; k < 64; ++k) {
                const unsigned g = f16bits::round_f32_to_f16_bits(bits_to_f(sb | (re << 23) | (rng() & 0x7FFFFFu)));
                CHECK(g == (sgn ? 0xFC00u : 0x7C00u));
            }
        CHECK(f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x477FE000u)) == (sgn ? 0xFBFFu : 0x7BFFu));  // 65504 exact
        CHECK(f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x477FEFFFu)) == (sgn ? 0xFBFFu : 0x7BFFu));  // just below 65520
        CHECK(f16bits::round_f32_to_f16_bits(bits_to_f(sb | 0x477FF000u)) == (sgn ? 0xFC00u : 0x7C00u));  // 65520 ties to even = inf
    }

    // rounding vs oracle: all ties and neighbours between consecutive halves, subnormal range, random bit patterns
    for (unsigned h = 0; h < 0x7BFF; ++h) {
        const double mid = (g_tab[h] + g_tab[h + 1]) / 2;
        const float fm = (float)mid;
        if ((double)fm != mid) continue;
        for (int d = -2; d <= 2; ++d) {
            uint32_t u = f16bits::as_u32(fm) + d;
            CHECK(same_f16(bits_to_f(u)));
            CHECK(same_f16(-bits_to_f(u)));
        }
    }
    for (long i = 0; i < 3000000; ++i) { const float f = bits_to_f((uint32_t)rng()); CHECK(same_f16(f)); CHECK(f16bits::round_f32_to_f16_bits(f) == f16::float_to_half(f) || std::isnan(f)); }
    for (long i = 0; i < 2000000; ++i) { const float f = bits_to_f(0x33000000u + (rng() % 0x0E000000u)); CHECK(same_f16(f)); }  // 2^-25 .. 2^17

    // bf16: ties to even, exact values unchanged, below/above half
    for (long i = 0; i < 3000000; ++i) { const float f = bits_to_f((uint32_t)rng()); CHECK(f16bits::round_f32_to_bf16_bits(f) == oracle_bf16(f)); }
    for (uint32_t q = 0; q < 65536; q += 7) {
        const uint32_t base = q << 16;
        if ((base & 0x7F800000u) == 0x7F800000u) continue;
        CHECK(f16bits::round_f32_to_bf16_bits(bits_to_f(base)) == q);
        CHECK(f16bits::round_f32_to_bf16_bits(bits_to_f(base | 0x7FFFu)) == q);
        CHECK(f16bits::round_f32_to_bf16_bits(bits_to_f(base | 0x8001u)) == ((q + 1) & 0xFFFFu));
        CHECK(f16bits::round_f32_to_bf16_bits(bits_to_f(base | 0x8000u)) % 2 == 0);
    }
    CHECK(f16bits::round_f32_to_bf16_bits(bits_to_f(0x7F800000u)) == 0x7F80u);                       // inf stays inf
    CHECK((f16bits::round_f32_to_bf16_bits(bits_to_f(0x7FFFFFFFu)) & 0x7F80u) == 0x7F80u && (f16bits::round_f32_to_bf16_bits(bits_to_f(0x7FFFFFFFu)) & 0x7Fu) != 0);

    // row dequantisers and the tool
    {
        std::vector<uint8_t> raw(2 * 64);
        for (auto& b : raw) b = (uint8_t)rng();
        std::vector<float> a(64), b(64);
        dequant_rows::dequantize_f16_row(raw.data(), 64, a.data());
        for (int i = 0; i < 64; ++i) { uint16_t h = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)); CHECK(std::memcmp(&a[i], &(b[i] = f16bits::widen_f16_bits(h)), 4) == 0); }
        dequant_rows::dequantize_bf16_row(raw.data(), 64, a.data());
        for (int i = 0; i < 64; ++i) { uint32_t u = (uint32_t)(raw[2 * i] | (raw[2 * i + 1] << 8)) << 16; CHECK(std::memcmp(&a[i], &u, 4) == 0); }
        std::vector<uint8_t> f32raw(16);
        const float v[4] = {1.5f, -2.25f, 0.f, 1e30f};
        std::memcpy(f32raw.data(), v, 16);
        dequant_rows::dequantize_f32_row(f32raw.data(), 4, a.data());
        for (int i = 0; i < 4; ++i) CHECK(a[i] == v[i]);
        uint8_t nl[18] = {0x00, 0x3C};  // d = 1.0
        for (int j = 0; j < 16; ++j) nl[2 + j] = (uint8_t)(j | ((15 - j) << 4));
        float y[32];
        dequant_rows::dequantize_iq4_nl_block(nl, y);
        for (int j = 0; j < 16; ++j) { CHECK(y[j] == (float)dequant::kvalues_iq4nl[j]); CHECK(y[j + 16] == (float)dequant::kvalues_iq4nl[15 - j]); }
        std::vector<float> out;
        CHECK(dequant_tool::dequant_buffer("iq4_nl", nl, 18, out) && out.size() == 32 && out[3] == y[3]);
        CHECK(!dequant_tool::dequant_buffer("iq4_nl", nl, 17, out));
        CHECK(!dequant_tool::dequant_buffer("nope", nl, 18, out));
        uint8_t q8[34] = {0x00, 0x40};  // d = 2.0
        for (int j = 0; j < 32; ++j) q8[2 + j] = (uint8_t)(int8_t)(j - 16);
        CHECK(dequant_tool::dequant_buffer("q8_0", q8, 34, out) && out.size() == 32 && out[0] == -32.f && out[31] == 30.f);
    }

    std::printf("f16_host checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("F16 HOST TESTS FAILED\n"); return 1; }
    std::printf("F16 HOST TESTS PASSED\n");
    return 0;
}
