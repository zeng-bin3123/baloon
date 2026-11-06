// test_decode_bf16.cpp - host invariants of normShift, normShift_spec, subnormal_bias, subnormal_exp_field,
// normal_exp_field, nat_truncation_hazard, bf16Round, bf16_exact, bf16_below_half, bf16_above_half, bf16_tie_even.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <cmath>
#include "baloon/f16_decode_bf16.hpp"

using namespace baloon::f16bits;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 40) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

int main() {
    // normShift_spec: all non-zero subnormal mantissas
    for (uint32_t m = 1; m < 1024; ++m) {
        NormShift r = norm_shift(10, m, 0);
        CHECK(r.m >= 1024u && r.m < 2048u && r.s <= 10u && r.m == (m << r.s));
        // minimality: s is the least shift that sets bit 10
        CHECK((m << r.s) >= 1024u && (r.s == 0 || (m << (r.s - 1)) < 1024u));
        // subnormal_exp_field validity
        const int32_t f = subnormal_exp_field(r.s);
        CHECK(f >= 103 && f < 255);
    }
    // normShift base cases: no fuel keeps the input; bit 10 already set does not shift
    { NormShift r = norm_shift(0, 5, 3); CHECK(r.m == 5u && r.s == 3u); }
    { NormShift r = norm_shift(7, 1024, 0); CHECK(r.m == 1024u && r.s == 0u); }
    { NormShift r = norm_shift(3, 1, 0); CHECK(r.m == 8u && r.s == 3u); }   // fuel limits the loop
    // subnormal_bias
    CHECK(kSubnormalBias == 113u && 127 - 15 + 1 == 113);
    for (uint32_t s = 0; s <= 10; ++s) CHECK(103 <= (int)(113 - s) && (int)(113 - s) < 255);
    // normal_exp_field: signed, == ex + 112, in 113..142
    for (int ex = 1; ex <= 30; ++ex) {
        CHECK(normal_exp_field(ex) == ex + 112 && normal_exp_field(ex) >= 113 && normal_exp_field(ex) <= 142);
        // the real widener agrees on every mantissa
        for (uint32_t m : {0u, 1u, 0x155u, 0x3FFu}) {
            const uint32_t b = bits_of(f32_from_f16((uint16_t)((ex << 10) | m)));
            CHECK((int)((b >> 23) & 0xFF) == ex + 112);
        }
    }
    // nat_truncation_hazard: truncated naturals give (1 - 15) -> 0, so 0 + 127 != 1 + 112
    { uint32_t trunc = (1u > 15u ? 1u - 15u : 0u) + 127u; CHECK(trunc != 1u + 112u); CHECK(trunc == 127u); }
    // exhaustive real widener vs model exponent field over all 65536 halves
    for (uint32_t h = 0; h < 65536; ++h) {
        const int ex = (h >> 10) & 0x1F; const uint32_t m = h & 0x3FF;
        int field = ex == 0 ? (m ? subnormal_exp_field(norm_shift(10, m, 0).s) : 0) : ex == 31 ? 255 : normal_exp_field(ex);
        CHECK((int)((bits_of(f32_from_f16((uint16_t)h)) >> 23) & 0xFF) == field);
    }
    // bf16_exact
    for (uint64_t q = 0; q < 65536; ++q) CHECK(bf16_round(65536u * q) == q);
    CHECK(bf16_round(65536ull * 65535ull) == 65535ull);
    // below / above / tie over a dense sweep of low halves and every upper half for sampled lows
    std::mt19937_64 rng(12345);
    for (uint64_t hi = 0; hi < 65536; ++hi) {
        for (uint64_t lo : {0ull, 1ull, 32767ull, 32768ull, 32769ull, 65535ull, (unsigned long long)(rng() & 0xFFFF)}) {
            const uint64_t i = (hi << 16) | lo;
            const uint64_t r = bf16_round(i);
            if (lo < 32768) CHECK(r == hi);
            if (lo > 32768) CHECK(r == hi + 1);
            if (lo == 32768) CHECK(r % 2 == 0 && (r == hi || r == hi + 1) && (r == (hi % 2 ? hi + 1 : hi)));
        }
    }
    // bf16_round matches the real conversion on every non-NaN float pattern (stride sweep + specials)
    for (uint64_t x = 0; x < (1ull << 32); x += 8191) {
        uint32_t u = (uint32_t)x;
        if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x7FFFFFu)) continue;
        const uint64_t m = bf16_round(u);
        if (m < 65536ull) CHECK(round_f32_to_bf16_bits(float_of(u)) == (uint16_t)m);
    }
    // NaN stays NaN in the real conversion (the pure model would carry into the exponent)
    for (uint32_t pl = 1; pl < 4096; ++pl) { uint16_t b = round_f32_to_bf16_bits(float_of(0x7F800000u | pl)); CHECK((b & 0x7F80) == 0x7F80 && (b & 0x7F)); }
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("DECODE BF16 TESTS FAILED\n"); return 1; }
    std::printf("DECODE BF16 TESTS PASSED\n");
    return 0;
}
