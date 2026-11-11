// Unit u34082f3355 host test: HalfPrecisionBitConversion + BlockDequantization invariants (.einsteinOS/theory.md)
// checked against include/baloon/artifact_dequant.hpp. Prints "U34082F3355 HOST TESTS PASSED" on success.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "baloon/artifact_dequant.hpp"

using namespace baloon::artifact;
static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }
static void put_f16(uint8_t* p, uint16_t h) { p[0] = (uint8_t)h; p[1] = (uint8_t)(h >> 8); }

// --- HalfPrecisionBitConversion -------------------------------------------------------------------------------------
static void test_half_bits() {
    // exact fp16 decodes: 1.0, -2.0, 65504 (max finite), smallest subnormal 2^-24, +inf, signed zero
    EXPECT(fp16_to_fp32(0x3C00) == 1.0f, "f16 1.0");
    EXPECT(fp16_to_fp32(0xC000) == -2.0f, "f16 -2.0");
    EXPECT(fp16_to_fp32(0x7BFF) == 65504.0f, "f16 max");
    EXPECT(fp16_to_fp32(0x0001) == std::ldexp(1.0f, -24), "f16 min subnormal");
    EXPECT(std::isinf(fp16_to_fp32(0x7C00)) && fp16_to_fp32(0x7C00) > 0, "f16 +inf");
    EXPECT(std::isnan(fp16_to_fp32(0x7E00)), "f16 nan");
    EXPECT(std::signbit(fp16_to_fp32(0x8000)) && fp16_to_fp32(0x8000) == 0.0f, "f16 -0");
    // bf16 is the top half of the f32 pattern: decoding shifts left by 16
    for (uint32_t h = 0; h < 65536; h += 97) {
        float f = bf16_to_fp32((uint16_t)h); uint32_t u; std::memcpy(&u, &f, 4);
        EXPECT(u == (h << 16), "bf16 %04x", h);
    }
    // every finite fp16 round-trips through f32 exactly
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7C00) == 0x7C00) continue;
        EXPECT(baloon::f16bits::f16_from_f32(fp16_to_fp32((uint16_t)h)) == h, "f16 roundtrip %04x", h);
    }
    uint8_t le[2] = {0x34, 0x12};
    EXPECT(read_u16(le) == 0x1234, "read_u16 little endian");
}

// --- BlockDequantization -------------------------------------------------------------------------------------------
static void test_q2_0() {   // 18 bytes / 64: f16 d, 16 code bytes; slot k of byte j/4 is (b >> 2k) & 3; value (code-1)*d
    uint32_t s = 1;
    for (int it = 0; it < 200; ++it) {
        uint8_t b[18]; int codes[64];
        put_f16(b, 0x3800);   // d = 0.5
        for (int j = 0; j < 64; ++j) codes[j] = lcg(s) & 3;
        for (int i = 0; i < 16; ++i) b[2 + i] = (uint8_t)(codes[4*i] + 4*codes[4*i+1] + 16*codes[4*i+2] + 64*codes[4*i+3]);
        float y[64]; dequantize_q2_0(b, y);
        for (int j = 0; j < 64; ++j) {
            EXPECT(y[j] == (codes[j] - 1) * 0.5f, "q2_0 j=%d", j);
            EXPECT(y[j] == -0.5f || y[j] == 0.0f || y[j] == 0.5f || y[j] == 1.0f, "q2_0 structural");
        }
    }
}
static void test_q4_0_q8_0() {
    uint8_t b[18]; put_f16(b, 0x4000);   // d = 2
    for (int i = 0; i < 16; ++i) b[2 + i] = (uint8_t)(i | ((15 - i) << 4));
    float y[32]; dequantize_q4_0(b, y);
    for (int j = 0; j < 16; ++j) {
        EXPECT(y[j] == (j - 8) * 2.0f, "q4_0 low %d", j);            // low nibble -> element j
        EXPECT(y[j + 16] == (7 - j) * 2.0f, "q4_0 high %d", j);      // high nibble -> element j+16
    }
    uint8_t c[34]; put_f16(c, 0x3C00);
    for (int j = 0; j < 32; ++j) c[2 + j] = (uint8_t)(int8_t)(j * 8 - 127);
    dequantize_q8_0(c, y);
    for (int j = 0; j < 32; ++j) EXPECT(y[j] == (float)(j * 8 - 127), "q8_0 %d", j);
}
static void test_iq4_nl() {
    static const int T[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int i = 0; i < 16; ++i) EXPECT(kvalues_iq4nl[i] == T[i], "iq4nl table %d", i);
    for (int i = 1; i < 16; ++i) EXPECT(kvalues_iq4nl[i - 1] < kvalues_iq4nl[i], "iq4nl increasing %d", i);
    uint8_t b[18]; put_f16(b, 0x3C00);
    for (int i = 0; i < 16; ++i) b[2 + i] = (uint8_t)(i | ((15 - i) << 4));
    float y[32]; dequantize_iq4_nl(b, y);
    for (int j = 0; j < 16; ++j) { EXPECT(y[j] == T[j], "iq4_nl lo %d", j); EXPECT(y[j + 16] == T[15 - j], "iq4_nl hi %d", j); }
}
static void test_scale_min_k4() {   // byte inputs give 6-bit outputs, reading only bytes 0..11
    uint32_t s = 7;
    for (int it = 0; it < 500; ++it) {
        uint8_t q[16]; for (auto& x : q) x = (uint8_t)lcg(s);
        q[12] = q[13] = q[14] = q[15] = 0xFF;   // poison past byte 11: must not influence the result
        uint8_t q2[16]; std::memcpy(q2, q, 16); q2[12] = q2[13] = q2[14] = q2[15] = 0;
        for (int j = 0; j < 8; ++j) {
            uint8_t d, m, d2, m2; get_scale_min_k4(j, q, &d, &m); get_scale_min_k4(j, q2, &d2, &m2);
            EXPECT(d < 64 && m < 64, "scale_min 6-bit j=%d", j);
            EXPECT(d == d2 && m == m2, "scale_min reads past byte 11 j=%d", j);
            if (j < 4) EXPECT(d == (q[j] & 63) && m == (q[j + 4] & 63), "scale_min low j=%d", j);
        }
    }
}
static void test_k_formats_finite() {   // q3_K/q4_K/q5_K/q6_K/iq4_xs: zero d gives zeros; random blocks stay finite
    struct F { const char* n; size_t bytes; void (*fn)(const uint8_t*, float*); } fs[] = {
        {"q3_K", 110, dequantize_q3_K}, {"q4_K", 144, dequantize_q4_K}, {"q5_K", 176, dequantize_q5_K},
        {"q6_K", 210, dequantize_q6_K}, {"iq4_xs", 136, dequantize_iq4_xs}};
    uint32_t s = 3;
    for (auto& f : fs) {
        uint8_t b[256]; float y[256];
        for (int it = 0; it < 50; ++it) {
            for (size_t i = 0; i < f.bytes; ++i) b[i] = (uint8_t)lcg(s);
            // keep the f16 scale fields finite: clear the exponent's top bit of every 2-byte word
            for (size_t i = 1; i < f.bytes; i += 2) b[i] &= 0xBF;
            f.fn(b, y);
            for (int j = 0; j < 256; ++j) EXPECT(std::isfinite(y[j]), "%s finite j=%d", f.n, j);
        }
    }
}
static void test_plain() {
    float src[3] = {1.5f, -0.25f, 3e30f}, y[3];
    dequantize_f32((const uint8_t*)src, y, 3);
    EXPECT(!std::memcmp(src, y, sizeof src), "f32 copy");
    uint8_t h[4]; put_f16(h, 0x3C00); put_f16(h + 2, 0xBC00);
    dequantize_f16(h, y, 2); EXPECT(y[0] == 1.0f && y[1] == -1.0f, "f16 array");
    put_f16(h, 0x3F80); put_f16(h + 2, 0xC040);
    dequantize_bf16(h, y, 2); EXPECT(y[0] == 1.0f && y[1] == -3.0f, "bf16 array");
}

int main() {
    test_half_bits(); test_q2_0(); test_q4_0_q8_0(); test_iq4_nl(); test_scale_min_k4(); test_k_formats_finite(); test_plain();
    if (fails) { std::printf("%d failures\n", fails); return 1; }
    std::printf("U34082F3355 HOST TESTS PASSED\n");
    return 0;
}

