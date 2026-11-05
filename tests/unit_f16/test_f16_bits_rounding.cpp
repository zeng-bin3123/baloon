// Host tests of the F16BitsRounding LAYER 1 statements (rneStep, encodeTop, normShift, re-bias, bf16Round) and of the
// dequantisers built on them. Exit code is non-zero on any failed check.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include "baloon/f16_bits_rounding.hpp"
#include "check.hpp"
using namespace baloon;
using namespace baloon::f16bits;
using namespace baloon::f16;

static void test_rne() {
    for (uint32_t h = 0; h < 70000; h += (h < 40 ? 1 : 97)) {
        for (uint32_t rem = 0; rem < 8192; ++rem) {
            const uint32_t r = rne_step(h, rem);
            if (rem < 4096) CHECK(r == h);
            if (rem > 4096) CHECK(r == h + 1);
            if (rem == 4096) CHECK(r % 2 == 0);
            CHECK(h <= r && r <= h + 1);
        }
    }
}

static void test_encode_top() {
    for (uint32_t sign : {0u, 0x8000u}) {
        for (uint32_t rawexp = 0; rawexp < 256; ++rawexp) {
            for (uint32_t man : {0u, 1u, 0x200u, 0x2000u, 0x400000u, 0x7FFFFFu}) {
                const int32_t v = encode_top(sign, rawexp, man);
                if (rawexp == 255) {
                    CHECK(v >= 0);
                    CHECK(((uint32_t)v & 0x7C00u) == 0x7C00u);
                    CHECK((((uint32_t)v % 1024u) != 0) == (man != 0));   // NaN stays NaN, inf stays inf
                    CHECK(((uint32_t)v & 0x8000u) == sign);
                } else if (rawexp >= 143) {                               // finite overflow -> inf, never NaN
                    CHECK(v >= 0);
                    CHECK((uint32_t)v % 1024u == 0 && (uint32_t)v == sign + 0x7C00u);
                } else {
                    CHECK(v == -1);
                }
                if (rawexp == 255) CHECK((int)rawexp - 127 + 15 >= 31);  // conflation hazard: 255 also satisfies exp >= 31
            }
        }
    }
    // Agreement with the full encoder on every raw exponent >= 143 and every sign.
    std::mt19937 rng(7);
    for (int i = 0; i < 2000000; ++i) {
        const uint32_t x = (uint32_t)rng();
        const uint32_t sign = (x >> 16) & 0x8000u, re = (x >> 23) & 0xFFu, man = x & 0x7FFFFFu;
        const int32_t t = encode_top(sign, re, man);
        const uint16_t h = float_to_half(f16::bits_f32(x));
        if (t >= 0) CHECK((uint16_t)t == h || (re == 255 && man != 0 && (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0));
        if (re == 255 && man != 0) CHECK((h & 0x3FFu) != 0);
        if (re != 255 && re >= 143) CHECK((h & 0x7FFFu) == 0x7C00u);
    }
}

// Normal regime of the encoder equals the rneStep model, exhaustively over exponent and mantissa.
static void test_normal_regime_matches_step() {
    for (uint32_t e = 1; e <= 30; ++e) {
        for (uint32_t man = 0; man < 0x800000u; ++man) {
            const uint32_t x = ((e - 15u + 127u) << 23) | man;
            const uint32_t trunc = e * 1024u + (man >> 13);
            const uint32_t want = rne_step(trunc, man & 8191u);
            if ((uint32_t)float_to_half(f16::bits_f32(x)) != want) { CHECK(false); return; }
        }
    }
    CHECK(true);
}

static void test_decoder() {
    for (uint32_t m = 1; m < 1024; ++m) {
        const f16::NormResult r = f16::norm(m, 0u, 10u);
        CHECK(r.m >= 1024u && r.m < 2048u && r.s <= 10u && r.m == m * (1u << r.s));
    }
    CHECK(127 - 15 + 1 == 113);
    for (uint32_t s = 0; s <= 10; ++s) CHECK(113 - s >= 103 && 113 - s < 255);
    for (int ex = 1; ex <= 30; ++ex) { const int f = ex - 15 + 127; CHECK(f == ex + 112 && f >= 113 && f <= 142); }
    CHECK((uint32_t)(1u - 15u + 127u) == 113u);   // unsigned wrap is fine; truncated naturals would give 0 + 127 - ... wrong
    // exhaustive decode vs exact double value
    for (uint32_t h = 0; h < 65536; ++h) {
        const uint32_t ex = (h >> 10) & 31u, m = h & 1023u;
        if (ex == 31) continue;
        const double mag = ex == 0 ? std::ldexp((double)m, -24) : std::ldexp((double)(1024 + m), (int)ex - 25);
        const double want = (h & 0x8000u) ? -mag : mag;
        const float got = widen_f16_bits((uint16_t)h);
        CHECK((double)got == want || (want == 0.0 && got == 0.0f));
        if (want == 0.0) CHECK(std::signbit(got) == ((h & 0x8000u) != 0));
    }
}

static void test_bf16() {
    std::mt19937 rng(11);
    for (uint32_t q = 0; q < 65536; q += 1) CHECK(bf16_round(q << 16) == q);                     // exact
    for (int i = 0; i < 3000000; ++i) {
        const uint32_t v = (uint32_t)rng() & 0x7FFFFFFFu, lo = v & 0xFFFFu, hi = v >> 16;
        const uint32_t r = bf16_round(v);
        if (lo < 32768) CHECK(r == hi);
        if (lo > 32768) CHECK(r == hi + 1);
        if (lo == 32768) CHECK(r % 2 == 0);
    }
    for (uint32_t hi = 0; hi < 32768; ++hi) CHECK(bf16_round((hi << 16) | 32768u) % 2 == 0);       // every tie
    CHECK(((round_f32_to_bf16_bits(f16::bits_f32(0x7FC00001u)) & 0x7F80u) == 0x7F80u) &&
          (round_f32_to_bf16_bits(f16::bits_f32(0x7FC00001u)) & 0x7Fu) != 0);                       // NaN stays NaN
    CHECK(round_f32_to_bf16_bits(f16::bits_f32(0x7F800000u)) == 0x7F80u);                           // inf stays inf
}

static void test_dequant_dense() {
    float f[3] = {1.5f, -2.0f, 0.1f};
    uint8_t raw[12]; std::memcpy(raw, f, 12);
    float y[3];
    f16bits::dequantize_f32(raw, y, 3);
    CHECK(y[0] == f[0] && y[1] == f[1] && y[2] == f[2]);
    uint8_t h16[6], b16[6];
    for (int i = 0; i < 3; ++i) {
        const uint16_t h = round_f32_to_f16_bits(f[i]), b = round_f32_to_bf16_bits(f[i]);
        h16[2*i] = (uint8_t)h; h16[2*i+1] = (uint8_t)(h >> 8); b16[2*i] = (uint8_t)b; b16[2*i+1] = (uint8_t)(b >> 8);
    }
    f16bits::dequantize_f16_dense(h16, y, 3);
    CHECK(y[0] == 1.5f && y[1] == -2.0f && std::fabs(y[2] - 0.1f) < 1e-4f);
    f16bits::dequantize_bf16_dense(b16, y, 3);
    CHECK(y[0] == 1.5f && y[1] == -2.0f && std::fabs(y[2] - 0.1f) < 1e-3f);
    uint8_t blk[18] = {0x00, 0x40};   // d = 2.0
    for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)(i | (15 - i) << 4);
    float q[32];
    f16bits::dequantize_iq4_nl(blk, q);
    for (int j = 0; j < 16; ++j) { CHECK(q[j] == 2.0f * dequant::kvalues_iq4nl[j]); CHECK(q[j + 16] == 2.0f * dequant::kvalues_iq4nl[15 - j]); }
}

int main() {
    test_rne(); test_encode_top(); test_normal_regime_matches_step(); test_decoder(); test_bf16(); test_dequant_dense();
    std::printf("f16 bits rounding host tests: %ld checks, %ld failures\n", testh::checks(), testh::failures());
    return testh::failures() == 0 ? 0 : 1;
}
