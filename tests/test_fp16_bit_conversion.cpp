#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "baloon/fp16_bit_conversion.hpp"

using namespace baloon::fp16;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            if (g_fail < 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static double f32_bits_to_double(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return static_cast<double>(f);
}

// Independent value of a finite non-negative fp16 magnitude pattern (0..0x7BFF).
static double fp16_mag_value(uint32_t h) {
    uint32_t ex = (h >> 10) & 31, man = h & 1023;
    return ex == 0 ? std::ldexp(static_cast<double>(man), -24) : std::ldexp(static_cast<double>(1024 + man), static_cast<int>(ex) - 25);
}

static void test_vectors() {
    CHECK(f32_from_f16(0) == 0 && f32_from_f16(0x8000) == 0x80000000u);
    CHECK(f32_from_f16(1) == 0x33800000u);
    CHECK(f16_from_f32(0x3F800000u) == 0x3C00);
    CHECK(f16_from_f32(0x3F801000u) == 0x3C00 && f16_from_f32(0x3F803000u) == 0x3C02);
    CHECK(f16_from_f32(0x477FE000u) == 0x7BFF && f16_from_f32(0x477FF000u) == 0x7C00 &&
          f16_from_f32(0x47800000u) == 0x7C00 && f16_from_f32(0x7149F2CAu) == 0x7C00 &&
          f16_from_f32(0xF149F2CAu) == 0xFC00);
    CHECK(f16_from_f32(0x33800000u) == 0x0001 && f16_from_f32(0x33000000u) == 0x0000 &&
          f16_from_f32(0x33000001u) == 0x0001 && f16_from_f32(0x32800000u) == 0x0000 &&
          f16_from_f32(0x80000000u) == 0x8000);
    CHECK(f16_from_f32(0x7F800000u) == 0x7C00 && f16_from_f32(0xFF800000u) == 0xFC00 &&
          f16_from_f32(0x7FC00000u) == 0x7E00);
    // a NaN with only a low mantissa bit still becomes a NaN (quiet bit set)
    CHECK(is_nan16(f16_from_f32(0x7F800001u)));
    CHECK(is_nan16(f16_from_f32(0xFFFFFFFFu)));
}

static void test_widen_exhaustive() {
    for (uint32_t h = 0; h < 65536; ++h) {
        uint32_t w = f32_from_f16(h);
        // sign preserved; nonzero never flushed
        CHECK((w >> 31) == (h >> 15));
        if ((h & 0x7FFF) != 0) CHECK((w & 0x7FFFFFFFu) != 0);
        uint32_t ex = (h >> 10) & 31, man = h & 1023;
        if (ex == 31) {
            CHECK(((w >> 23) & 0xFF) == 255 && (w & 0x7FFFFF) == (man << 13));
        } else {
            // exact: matches the independent double value
            double expect = fp16_mag_value(h & 0x7FFF) * ((h & 0x8000) ? -1.0 : 1.0);
            double got = f32_bits_to_double(w);
            CHECK(got == expect);
            if (h == 0x8000) CHECK(std::signbit(got));
        }
        // every non-NaN value round-trips bit-exactly; NaN stays NaN
        uint16_t back = f16_from_f32(w);
        if (!is_nan16(h)) CHECK(back == h);
        else CHECK(is_nan16(back));
    }
}

// Independent RNE reference built on the sorted table of finite fp16 magnitudes.
static uint16_t reference_narrow(uint32_t x, const std::vector<double>& tab) {
    uint16_t sign = (x >> 16) & 0x8000;
    uint32_t e = (x >> 23) & 0xFF, m = x & 0x7FFFFF;
    if (e == 255) return sign | 0x7C00 | (m ? 0x200 : 0);
    double v = std::fabs(f32_bits_to_double(x));
    // largest index i with tab[i] <= v
    size_t lo = 0, hi = tab.size() - 1;
    while (lo < hi) {
        size_t mid = (lo + hi + 1) / 2;
        if (tab[mid] <= v) lo = mid; else hi = mid - 1;
    }
    size_t i = lo;
    if (i == tab.size() - 1) {  // at or above the max finite value
        return v >= 65520.0 ? (sign | 0x7C00) : (sign | static_cast<uint16_t>(i));
    }
    double d_lo = v - tab[i], d_hi = tab[i + 1] - v;
    size_t pick = d_lo < d_hi ? i : d_hi < d_lo ? i + 1 : ((i & 1) == 0 ? i : i + 1);
    return sign | static_cast<uint16_t>(pick);
}

static void test_narrow_vs_reference() {
    std::vector<double> tab(0x7C00);
    for (uint32_t h = 0; h < 0x7C00; ++h) tab[h] = fp16_mag_value(h);
    // 1) every exponent with dense mantissa patterns (ties, near-ties, carries), both signs
    for (uint32_t sgn = 0; sgn < 2; ++sgn)
        for (uint32_t e = 0; e < 255; ++e) {
            for (uint32_t k = 0; k < 8192; ++k) {
                // low 13 bits sweep with varied high bits
                uint32_t hi10 = (k * 2654435761u) & 0x3FF;
                uint32_t m = (hi10 << 13) | k;
                uint32_t x = (sgn << 31) | (e << 23) | (m & 0x7FFFFF);
                CHECK(f16_from_f32(x) == reference_narrow(x, tab));
            }
        }
    // 2) exact tie points and neighbours between every adjacent pair of fp16 magnitudes
    for (uint32_t h = 0; h + 1 < 0x7C00; ++h) {
        double mid = (tab[h] + tab[h + 1]) / 2.0;
        float f = static_cast<float>(mid);  // exact (needs <= 12 significant bits)
        uint32_t b;
        std::memcpy(&b, &f, 4);
        for (int d = -1; d <= 1; ++d) {
            uint32_t x = b + d;
            CHECK(f16_from_f32(x) == reference_narrow(x, tab));
            CHECK(f16_from_f32(x | 0x80000000u) == reference_narrow(x | 0x80000000u, tab));
        }
    }
    // 3) random bit patterns
    uint32_t s = 12345;
    for (int i = 0; i < 2000000; ++i) {
        s = s * 1664525u + 1013904223u;
        uint32_t x = s;
        CHECK(f16_from_f32(x) == reference_narrow(x, tab));
    }
}

static void test_monotonic_and_overflow() {
    // finite overflow never yields NaN
    for (uint32_t sgn = 0; sgn < 2; ++sgn)
        for (uint32_t e = 143; e < 255; ++e)
            for (uint32_t m : {0u, 1u, 0x400000u, 0x7FFFFFu}) {
                uint16_t r = f16_from_f32((sgn << 31) | (e << 23) | m);
                CHECK(r == ((sgn << 15) | 0x7C00));
                CHECK(!is_nan16(r));
            }
    // underflow region flushes to signed zero
    for (uint32_t sgn = 0; sgn < 2; ++sgn)
        for (uint32_t e = 0; e < 102; ++e) CHECK(f16_from_f32((sgn << 31) | (e << 23) | 0x7FFFFF) == (sgn << 15));
    // monotone over positive finite f32 (as integers)
    uint16_t prev = 0;
    for (uint32_t x = 0; x < 0x7F800000u; x += 997) {
        uint16_t r = f16_from_f32(x);
        CHECK(r >= prev);
        prev = r;
    }
}

int main() {
    test_vectors();
    test_widen_exhaustive();
    test_narrow_vs_reference();
    test_monotonic_and_overflow();
    if (g_fail) {
        std::fprintf(stderr, "fp16_bit_conversion: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("fp16_bit_conversion: all checks passed\n");
    return 0;
}

