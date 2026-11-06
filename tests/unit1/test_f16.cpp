#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "check.hpp"

using namespace baloon::f16;

namespace {
// Independent oracle: nearest finite half (or virtual 65536 = inf), ties to the even code.
struct Oracle {
    std::vector<double> val;  // val[code] for code in 0..0x7BFF, val[0x7C00] = 65536
    Oracle() {
        val.resize(0x7C01);
        for (int c = 0; c < 0x7C00; ++c) {
            const int ex = c >> 10, m = c & 1023;
            val[c] = ex == 0 ? std::ldexp((double)m, -24) : std::ldexp((double)(1024 + m), ex - 25);
        }
        val[0x7C00] = 65536.0;
    }
    uint16_t round(float f) const {
        const uint16_t sign = std::signbit(f) ? 0x8000 : 0;
        const double a = std::fabs((double)f);
        // largest code with val <= a
        int lo = (int)(std::upper_bound(val.begin(), val.end(), a) - val.begin()) - 1;
        if (lo < 0) lo = 0;
        if (lo >= 0x7C00) return (uint16_t)(sign | 0x7C00);
        const int hi = lo + 1;
        const double dl = a - val[lo], dh = val[hi] - a;
        int pick;
        if (dl < dh) pick = lo;
        else if (dh < dl) pick = hi;
        else pick = (lo % 2 == 0) ? lo : hi;
        return (uint16_t)(sign | pick);
    }
};
}  // namespace

void test_f16_decode_exhaustive() {
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t ex = (h >> 10) & 31, m = h & 1023;
        const bool neg = (h & 0x8000) != 0;
        const float got = half_to_float((uint16_t)h);
        if (ex == 31) {
            if (m == 0) CHECK_MSG(std::isinf(got) && std::signbit(got) == neg, "h=%04x", h);
            else CHECK_MSG(std::isnan(got) && (f32_bits(got) & 0x7FFFFFu) == (m << 13), "h=%04x", h);
            continue;
        }
        double ref = ex == 0 ? std::ldexp((double)m, -24) : std::ldexp((double)(1024 + m), (int)ex - 25);
        if (neg) ref = -ref;
        CHECK_MSG((double)got == ref && std::signbit(got) == neg, "h=%04x got=%g ref=%g", h, got, ref);
    }
}

void test_f16_roundtrip_all_halves() {
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = half_to_float((uint16_t)h);
        const uint16_t back = float_to_half(f);
        if (((h >> 10) & 31) == 31 && (h & 1023) != 0) CHECK_MSG((back & 0x7C00) == 0x7C00 && (back & 0x3FF) != 0, "nan h=%04x", h);
        else CHECK_MSG(back == h, "h=%04x back=%04x", h, back);
    }
}

void test_f16_rne_against_oracle() {
    static const Oracle O;
    auto check = [&](float f) {
        if (std::isnan(f)) return;
        const uint16_t want = O.round(f), got = float_to_half(f);
        CHECK_MSG(got == want, "f=%a bits=%08x got=%04x want=%04x", f, f32_bits(f), got, want);
    };
    // every exact midpoint between adjacent halves, plus one fp32 ulp on either side, both signs
    for (int c = 0; c < 0x7C00; ++c) {
        const double mid = 0.5 * (O.val[c] + O.val[c + 1]);
        const float fm = (float)mid;
        CHECK((double)fm == mid);  // midpoint is representable: the tie is a true tie
        for (float f : {fm, std::nextafterf(fm, 0.f), std::nextafterf(fm, 1e30f), (float)O.val[c],
                        std::nextafterf((float)O.val[c], 0.f), std::nextafterf((float)O.val[c], 1e30f)}) {
            check(f);
            check(-f);
        }
    }
    // overflow / extremes
    for (float f : {65504.f, 65519.f, 65519.996f, 65520.f, 65521.f, 1e10f, std::numeric_limits<float>::max(),
                    0.f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
                    std::ldexp(1.f, -24), std::ldexp(1.f, -25), std::ldexp(1.5f, -25), std::ldexp(1.f, -26),
                    std::ldexp(1.f, -14), 1.f, 0.1f, 3.14159265f}) {
        check(f);
        check(-f);
    }
    CHECK(float_to_half(std::numeric_limits<float>::infinity()) == 0x7C00);
    CHECK(float_to_half(-std::numeric_limits<float>::infinity()) == 0xFC00);
    CHECK(float_to_half(std::numeric_limits<float>::quiet_NaN()) != 0);
    CHECK((float_to_half(std::numeric_limits<float>::quiet_NaN()) & 0x7C00) == 0x7C00);
    CHECK((float_to_half(std::numeric_limits<float>::quiet_NaN()) & 0x3FF) != 0);
    // signed zeros
    CHECK(float_to_half(0.f) == 0x0000 && float_to_half(-0.f) == 0x8000);
    // random bit patterns (including fp32 subnormals) and dense random values around the half range
    std::mt19937_64 rng(12345);
    for (int i = 0; i < 3000000; ++i) check(bits_f32((uint32_t)rng()));
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (int i = 0; i < 1000000; ++i) {
        const double e = ud(rng) * 26.0;  // exponents -26..26
        check((float)(std::ldexp(1.0, (int)std::floor(e)) * (1.0 + 0.5 * (ud(rng) + 1.0))));
    }
}

void test_f16_round_normal_contract() {
    for (uint32_t s : {0u, 32768u})
        for (uint32_t e = 1; e <= 30; ++e)
            for (uint32_t k = 0; k < 1024; ++k)
                for (uint32_t r : {0u, 1u, 4095u, 4096u, 4097u, 8191u}) {
                    const uint32_t m = k * 8192u + r;
                    const uint32_t v = round_normal(s, e, m), base = h0(s, e, m);
                    CHECK(v <= s + 31744u);                              // never past inf, never into sign
                    CHECK((v & 0x8000u) == s);                           // sign preserved
                    if (r > 4096) CHECK(v == base + 1);                  // above half rounds up
                    if (r < 4096) CHECK(v == base);                      // below half rounds down
                    if (r == 4096 && base % 2 == 0) CHECK(v == base);    // tie, even: down
                    if (r == 4096 && base % 2 == 1) CHECK(v == base + 1);// tie, odd: up
                    if (r == 4096) CHECK(v % 2 == 0);                    // ties land on an even code
                }
}

void test_f16_subnormal_contract() {
    for (int e = -10; e <= 0; ++e) {
        const int sh = 14 - e;
        CHECK(sh >= 14 && sh <= 24);
        // shifted 24-bit mantissa fits in 10 bits
        CHECK(((1u << 24) - 1u) >> sh < 1024u);
    }
    for (int e = -30; e < -10; ++e) CHECK(14 - e > 24);
    // flush: fp32 exponent such that half exponent < -10 gives signed zero
    CHECK(float_to_half(std::ldexp(1.f, -26)) == 0x0000);
    CHECK(float_to_half(-std::ldexp(1.f, -26)) == 0x8000);
    CHECK(float_to_half(std::ldexp(1.f, -25)) == 0x0000);  // tie to even (0)
    CHECK(float_to_half(std::nextafterf(std::ldexp(1.f, -25), 1.f)) == 0x0001);
    CHECK(float_to_half(std::ldexp(1.f, -24)) == 0x0001);
    CHECK(float_to_half(std::ldexp(1.f, -14)) == 0x0400);   // min normal
    // carry out of the subnormal range lands exactly on the minimum normal
    CHECK(float_to_half(std::nextafterf(std::ldexp(1.f, -14), 0.f)) == 0x0400);
}

void test_f16_norm_contract() {
    for (uint32_t m = 1; m < 1024; ++m) {
        const NormResult r = norm(m, 0, 10);
        CHECK(r.m >= 1024 && r.m < 2048);                   // bit 10 set, below 2048
        CHECK(r.s <= 10);                                   // at most 10 shifts
        CHECK((m << r.s) == r.m);                           // value preserved by shifting
        CHECK(r.s >= 1 && 113 - r.s >= 1 && 113 - r.s <= 254);
        const NormResult r2 = norm(m, 5, 10);
        CHECK(r2.s <= 5 + 10 && r2.s == r.s + 5);           // shift count grows by <= 1 per iteration
    }
    CHECK(norm(5, 3, 0).m == 5 && norm(5, 3, 0).s == 3);    // no fuel: unchanged
    CHECK(norm(1500, 0, 10).s == 0);                        // already normalised
    for (uint32_t ex = 1; ex <= 30; ++ex) CHECK(ex - 15 + 127 >= 113 && ex - 15 + 127 <= 142);
}

void test_bf16() {
    for (uint32_t h = 0; h < 65536; ++h) {
        const float f = baloon::f16::bf16_to_float((uint16_t)h);
        CHECK(f32_bits(f) == h * 65536u);
        CHECK(float_high_half(f) == h);
    }
}
