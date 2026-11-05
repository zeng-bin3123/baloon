// test_encode_top.cpp - host invariants of the F16BitsRounding items encodeTop, finite_overflow_is_inf, nan_stays_nan,
// inf_stays_inf and conflation_hazard. `encode_top` is the integer model of the encoder's top-level split; the real
// converter f16_from_f32 must agree with it on every raw exponent / sign / mantissa class it classifies.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include "baloon/f16_bits.hpp"

using baloon::f16bits::f16_from_f32;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// encodeTop: nullopt means "normal or subnormal path".
static std::optional<uint32_t> encode_top(uint32_t sign, uint32_t rawexp, uint32_t man) {
    if (rawexp == 255) return sign + 31744u + (man != 0 ? 512u : 0u);
    if ((int64_t)rawexp - 127 + 15 >= 31) return sign + 31744u;
    return std::nullopt;
}

int main() {
    const uint32_t mans[] = {0u, 1u, 2u, 0x1FFFu, 0x2000u, 0x400000u, 0x7FFFFFu, 0x155555u};
    for (uint32_t sign : {0u, 32768u}) {
        // finite_overflow_is_inf: rawexp != 255 and 143 <= rawexp -> some v, v % 1024 == 0
        for (uint32_t e = 143; e <= 254; ++e)
            for (uint32_t m : mans) {
                auto r = encode_top(sign, e, m);
                CHECK(r.has_value() && *r % 1024 == 0 && *r == sign + 31744u);
                CHECK(f16_from_f32(flt((sign << 16) | (e << 23) | m)) == (uint16_t)*r);
            }
        // nan_stays_nan: any nonzero mantissa at 255 keeps a nonzero mantissa field
        for (uint32_t m : mans) {
            if (m == 0) continue;
            auto r = encode_top(sign, 255, m);
            CHECK(r.has_value() && *r % 1024 != 0);
            CHECK(f16_from_f32(flt((sign << 16) | (255u << 23) | m)) == (uint16_t)*r || (f16_from_f32(flt((sign << 16) | (255u << 23) | m)) & 0x3FF) != 0);
        }
        for (uint32_t m = 1; m < 4096; ++m) {   // exhaustive over low payload bits
            auto r = encode_top(sign, 255, m);
            CHECK(r.has_value() && *r % 1024 != 0);
            const uint16_t h = f16_from_f32(flt((sign << 16) | (255u << 23) | m));
            CHECK((h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0);
        }
        // inf_stays_inf
        {
            auto r = encode_top(sign, 255, 0);
            CHECK(r.has_value() && *r % 1024 == 0 && *r == sign + 31744u);
            CHECK(f16_from_f32(flt((sign << 16) | (255u << 23))) == (uint16_t)*r);
        }
        // below the overflow boundary the model defers to the normal/subnormal path
        for (uint32_t e = 0; e <= 142; ++e)
            for (uint32_t m : mans) CHECK(!encode_top(sign, e, m).has_value());
    }
    // the model is total on 0..255: every raw exponent classified exactly as the theory says
    for (uint32_t e = 0; e <= 255; ++e) {
        const bool top = encode_top(0, e, 0).has_value();
        CHECK(top == (e >= 143));
    }
    // conflation_hazard: the old test `exp >= 31` also fires for raw exponent 255, so 255 must be tested first
    CHECK((int64_t)255 - 127 + 15 >= 31);
    CHECK(encode_top(0, 255, 1).value() % 1024 != 0 && encode_top(0, 254, 1).value() % 1024 == 0);
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("ENCODE TOP TESTS FAILED\n"); return 1; }
    std::printf("ENCODE TOP TESTS PASSED\n");
    return 0;
}
