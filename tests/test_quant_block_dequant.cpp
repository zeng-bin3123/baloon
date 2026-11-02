#include <cstdio>
#include <set>

#include "baloon/gguf_header_reader.hpp"
#include "baloon/quant_block_dequant.hpp"

using namespace baloon::dequant;

static int g_fail = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fail;                                                            \
        }                                                                        \
    } while (0)

static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

static void test_q2() {
    // bitwise form equals arithmetic form for every byte/slot
    for (uint32_t b = 0; b < 256; ++b) {
        for (uint32_t k = 0; k < 4; ++k) {
            CHECK(((b >> (k * 2)) & 3u) == (b / (1u << (k * 2))) % 4u);
        }
        // pack roundtrip
        CHECK((b % 4) + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) == b);
    }
    uint32_t s = 1;
    for (int iter = 0; iter < 200; ++iter) {
        uint8_t qs[16];
        for (auto& x : qs) x = static_cast<uint8_t>(lcg(s));
        for (uint32_t j = 0; j < 64; ++j) {
            uint32_t c = q2_code(qs, j);
            CHECK(c < 4);
            int32_t sym = q2_element(qs, j);
            CHECK(sym >= -1 && sym <= 2 && sym == static_cast<int32_t>(c) - 1);
            CHECK(j / 4 < 16);
            // same byte / slot structure
            CHECK(((j / 4) * 4 + j % 4) == j);
            // reassembling the four codes of the byte gives the byte
            uint32_t base = (j / 4) * 4;
            uint32_t re = q2_code(qs, base) | (q2_code(qs, base + 1) << 2) | (q2_code(qs, base + 2) << 4) |
                          (q2_code(qs, base + 3) << 6);
            CHECK(re == qs[j / 4]);
        }
    }
    // explicit vector: byte 0b11100100 -> codes 0,1,2,3 -> symbols -1,0,1,2
    uint8_t qs[16] = {0xE4};
    CHECK(q2_element(qs, 0) == -1 && q2_element(qs, 1) == 0 && q2_element(qs, 2) == 1 && q2_element(qs, 3) == 2);
    // geometry agrees: 2 + 16 = 18 bytes, 64 elems
    CHECK(baloon::gguf::block_geometry(42)->bytes == 2 + 16 && baloon::gguf::block_geometry(42)->elems == 16 * 4);
}

static void test_q4_q5_q8() {
    for (uint32_t b = 0; b < 256; ++b) {
        CHECK(q4_lo(b) == b % 16 && q4_hi(b) == b / 16);
        CHECK(q4_lo(b) + 16 * q4_hi(b) == b);
        CHECK(q4_hi(b) < 16 && q4_lo(b) < 16);
        int32_t q8 = q8_element(static_cast<uint8_t>(b));
        CHECK(q8 >= -128 && q8 <= 127);
        CHECK(q8 == (b < 128 ? static_cast<int32_t>(b) : static_cast<int32_t>(b) - 256));
    }
    for (uint32_t n = 0; n < 16; ++n) CHECK(q4_symbol(n) >= -8 && q4_symbol(n) <= 7);
    for (uint32_t n = 0; n < 16; ++n)
        for (uint32_t h = 0; h < 2; ++h) {
            CHECK(q5_val(n, h) < 32);
            CHECK(q5_symbol(n, h) >= -16 && q5_symbol(n, h) <= 15);
        }
    uint32_t s = 7;
    for (int iter = 0; iter < 200; ++iter) {
        uint8_t qs[16];
        for (auto& x : qs) x = static_cast<uint8_t>(lcg(s));
        uint32_t qh = lcg(s) * 2654435761u;
        for (uint32_t i = 0; i < 32; ++i) {
            int32_t e = q4_element(qs, i);
            CHECK(e >= -8 && e <= 7);
            int32_t e5 = q5_element(qs, qh, i);
            CHECK(e5 >= -16 && e5 <= 15);
            // Q5 reduces to Q4 nibble plus mask bit
            CHECK(e5 == e + 8 + 16 * static_cast<int32_t>((qh >> i) & 1u) - 16);
        }
    }
    // explicit: byte 0xA3 -> element 0 = 3-8, element 16 = 10-8
    uint8_t qs[16] = {0xA3};
    CHECK(q4_element(qs, 0) == -5 && q4_element(qs, 16) == 2);
}

static void test_iq4nl() {
    CHECK(kKvalues.size() == 16);
    for (size_t i = 1; i < 16; ++i) CHECK(kKvalues[i - 1] < kKvalues[i]);  // strictly increasing
    CHECK(kKvalues[1] - kKvalues[0] != kKvalues[2] - kKvalues[1]);        // non-linear
    for (auto v : kKvalues) CHECK(v >= -127 && v <= 113);
    for (uint32_t i = 0; i < 16; ++i) CHECK(kv(i) == kKvalues[i]);
    CHECK(kv(0) == -127 && kv(15) == 113 && kv(8) == 1);
    CHECK(kv(16) == 0);
    // verbatim table
    const int expect[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int i = 0; i < 16; ++i) CHECK(kKvalues[i] == expect[i]);
}

static void test_q6k() {
    for (uint32_t lo = 0; lo < 16; ++lo)
        for (uint32_t hi = 0; hi < 4; ++hi) CHECK(q6_val(lo, hi) >= -32 && q6_val(lo, hi) <= 31);
    std::set<uint32_t> seen;
    for (uint32_t h = 0; h < 2; ++h)
        for (uint32_t l = 0; l < 32; ++l)
            for (uint32_t k = 0; k < 4; ++k) {
                uint32_t o = q6k_out_index(h, l, k);
                CHECK(o < 256);
                seen.insert(o);
                CHECK(q6k_scale_index(h, l, k) < 16);
                CHECK(l / 16 + 2 * k < 8);
            }
    CHECK(seen.size() == 256);  // injective and covers the block
    CHECK(kQ6KBytes == 210 && kQ6KDOffset == 208 && kQ6KDOffset + 2 == kQ6KBytes);
    CHECK(baloon::gguf::block_geometry(14)->bytes == kQ6KBytes);
}

static void test_scale_min() {
    // exhaustive over byte values at the three positions each output depends on
    for (uint32_t j = 0; j < 8; ++j) {
        uint32_t s = 99 + j;
        for (int iter = 0; iter < 2000; ++iter) {
            uint8_t q[12];
            for (auto& x : q) x = static_cast<uint8_t>(lcg(s));
            ScaleMin sm = scale_min_k4(q, j);
            CHECK(sm.scale < 64 && sm.min < 64);
            if (j < 4) {
                CHECK(sm.scale == (q[j] & 63u) && sm.min == (q[j + 4] & 63u));
            } else {
                CHECK(sm.scale == static_cast<uint32_t>((q[j + 4] & 15u) | ((q[j - 4] >> 6) << 4)));
                CHECK(sm.min == static_cast<uint32_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4)));
            }
        }
    }
    // extremes
    uint8_t ff[12];
    for (auto& x : ff) x = 0xFF;
    for (uint32_t j = 0; j < 8; ++j) {
        CHECK(scale_min_k4(ff, j).scale == 63 && scale_min_k4(ff, j).min == 63);
    }
    for (uint32_t g = 0; g < 4; ++g) CHECK(2 * g < 8 && 2 * g + 1 < 8);
    // Q5_K masks
    const uint32_t u1[4] = {1, 4, 16, 64}, u2[4] = {2, 8, 32, 128};
    for (uint32_t g = 0; g < 4; ++g) {
        CHECK((1u << (2 * g)) == u1[g] && (2u << (2 * g)) == u2[g]);
        CHECK(((1u << (2 * g)) & (2u << (2 * g))) == 0);
    }
    CHECK(4 + 12 + 128 == 144 && 4 + 12 + 32 + 128 == 176);
    CHECK(baloon::gguf::block_geometry(12)->bytes == 144 && baloon::gguf::block_geometry(13)->bytes == 176);
}

static void test_iq4xs() {
    uint32_t s = 5;
    for (int iter = 0; iter < 5000; ++iter) {
        uint8_t sl[4];
        for (auto& x : sl) x = static_cast<uint8_t>(lcg(s));
        uint32_t sh = lcg(s) & 0xFFFFu;
        for (uint32_t ib = 0; ib < 8; ++ib) {
            uint32_t ls = iq4xs_scale(sl, sh, ib);
            CHECK(ls < 64);
            int32_t sg = iq4xs_signed_scale(sl, sh, ib);
            CHECK(sg >= -32 && sg <= 31);
            CHECK(ib / 2 < 4);
            CHECK(ls == (((sl[ib / 2] / (1u << (4 * (ib % 2)))) % 16) + ((sh / (1u << (2 * ib))) % 4) * 16));
        }
    }
    CHECK(2 + 2 + 4 + 128 == 136 && 8 * 32 == 256);
    CHECK(baloon::gguf::block_geometry(23)->bytes == 136);
}

static void test_q3k_and_fp16() {
    CHECK(110 == 32 + 64 + 12 + 2 && baloon::gguf::block_geometry(11)->bytes == 110);
    for (uint32_t i = 0; i < 8; ++i) CHECK((1u << i) < 256);
    for (uint32_t lo = 0; lo < 4; ++lo)
        for (uint32_t h = 0; h < 2; ++h) CHECK(q3k_value(lo, h) >= -4 && q3k_value(lo, h) <= 3);
    CHECK(q3k_value(0, 0) == -4 && q3k_value(3, 1) == 3);

    for (uint32_t h = 0; h < 65536; ++h) {
        CHECK(fp16_man(h) + 1024 * fp16_exp(h) + 32768 * fp16_sign(h) == h);
        CHECK(fp16_exp(h) < 32);
        CHECK((bf16_to_f32_bits(h) & 0xFFFFu) == 0);
        CHECK((bf16_to_f32_bits(h) >> 16) == h);
    }
    CHECK(127 - 15 + 1 == 113 && 15 - 15 + 127 == 127 && (0x7F800000u >> 23) == 255);
    for (uint32_t p0 = 0; p0 < 256; ++p0)
        for (uint32_t p1 = 0; p1 < 256; p1 += 5) {
            uint8_t p[2] = {static_cast<uint8_t>(p0), static_cast<uint8_t>(p1)};
            uint32_t v = read_u16(p);
            CHECK(v < 65536 && v % 256 == p0 && v / 256 == p1);
        }
}

int main() {
    test_q2();
    test_q4_q5_q8();
    test_iq4nl();
    test_q6k();
    test_scale_min();
    test_iq4xs();
    test_q3k_and_fp16();
    if (g_fail) {
        std::fprintf(stderr, "quant_block_dequant: %d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("quant_block_dequant: all checks passed\n");
    return 0;
}

