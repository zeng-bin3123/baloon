#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/ggml_dequant_layouts.hpp"
#include "check.hpp"

using namespace baloon::dequant;
using baloon::f16::half_to_float;

namespace {
std::mt19937_64 rng(777);
uint32_t R(uint32_t n) { return (uint32_t)(rng() % n); }
uint16_t rand_half() {  // finite, moderate magnitude
    for (;;) {
        const uint16_t h = (uint16_t)rng();
        const uint32_t ex = (h >> 10) & 31;
        if (ex >= 8 && ex <= 22) return h;
    }
}
void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)(v & 255); p[1] = (uint8_t)(v >> 8); }

// Pack 8 six-bit scales / mins into the 12-byte K-quant scale area (inverse of get_scale_min_k4).
void pack_k4(const int sc[8], const int mn[8], uint8_t* s) {
    std::memset(s, 0, 12);
    for (int j = 0; j < 4; ++j) { s[j] = (uint8_t)sc[j]; s[j + 4] = (uint8_t)mn[j]; }
    for (int j = 4; j < 8; ++j) {
        s[j + 4] = (uint8_t)((sc[j] & 15) | ((mn[j] & 15) << 4));
        s[j - 4] |= (uint8_t)((sc[j] >> 4) << 6);
        s[j] |= (uint8_t)((mn[j] >> 4) << 6);
    }
}

// Compare against expected within a tiny tolerance and verify nothing outside the output span was written.
template <size_t N, typename F>
void run_block(const std::vector<uint8_t>& blk, size_t block_bytes, F deq, const float* want) {
    // surround the block with different random bytes: dequant must not read outside its block
    std::vector<uint8_t> a(block_bytes + 64), b(block_bytes + 64);
    for (auto& x : a) x = (uint8_t)rng();
    for (auto& x : b) x = (uint8_t)rng();
    std::memcpy(a.data() + 32, blk.data(), block_bytes);
    std::memcpy(b.data() + 32, blk.data(), block_bytes);
    std::vector<float> ya(N + 16, 12345.f), yb(N + 16, 12345.f);
    deq(a.data() + 32, ya.data());
    deq(b.data() + 32, yb.data());
    for (size_t i = 0; i < N; ++i) {
        CHECK_MSG(ya[i] == yb[i], "reads outside block at %zu", i);
        CHECK_MSG(std::fabs(ya[i] - want[i]) <= 1e-6f * std::fabs(want[i]) + 1e-30f, "i=%zu got=%g want=%g", i, ya[i], want[i]);
    }
    for (size_t i = N; i < N + 16; ++i) CHECK_MSG(ya[i] == 12345.f && yb[i] == 12345.f, "wrote past N at %zu", i);
}
}  // namespace

void test_dequant_geometry() {
    CHECK(kQ2_0Bytes == 18 && kQ2_0Elems == 64 && kQ4_0Bytes == 18 && kQ5_0Bytes == 22);
    CHECK(kQ6KBytes == 210 && kQ4KBytes == 144 && kQ5KBytes == 176 && kQ3KBytes == 110 && kIq4XsBytes == 136);
    CHECK(kQ6KScOff == kQ6KQhOff + 64 && kQ6KDOff == kQ6KScOff + 16 && kQ6KDOff + 2 == kQ6KBytes);
    CHECK(kQ4KQsOff + 128 == kQ4KBytes && kQ5KQsOff + 128 == kQ5KBytes && kQ5KQhOff + 32 == kQ5KQsOff);
    CHECK(kQ3KScalesOff == kQ3KHmaskOff + 32 + 64 && kQ3KDOff == kQ3KScalesOff + 12 && kQ3KDOff + 2 == kQ3KBytes);
    CHECK(kIq4XsQsOff + 128 == kIq4XsBytes);
    // read_u16: p0 | p1 << 8, bytes recoverable
    for (int p0 = 0; p0 < 256; p0 += 5)
        for (int p1 = 0; p1 < 256; p1 += 7) {
            const uint8_t b[2] = {(uint8_t)p0, (uint8_t)p1};
            const uint16_t v = read_u16(b);
            CHECK(v == p0 + 256 * p1 && v % 256 == p0 && v / 256 == p1);
        }
    // code ranges
    for (unsigned b = 0; b < 256; ++b)
        for (unsigned r = 0; r < 4; ++r) { CHECK(q2_code(b, r) <= 3); const int s = q2_sym(q2_code(b, r)); CHECK(s >= -1 && s <= 2); }
    // codebook is strictly increasing
    for (int i = 1; i < 16; ++i) CHECK(kvalues_iq4nl[i] > kvalues_iq4nl[i - 1]);
    // q2 injectivity of (byte, position)
    bool seen[16][4] = {};
    for (int j = 0; j < 64; ++j) { CHECK(j / 4 < 16); CHECK(!seen[j / 4][j % 4]); seen[j / 4][j % 4] = true; }
}

void test_scale_min_k4() {
    for (int iter = 0; iter < 2000; ++iter) {
        int sc[8], mn[8];
        for (int j = 0; j < 8; ++j) { sc[j] = (int)R(64); mn[j] = (int)R(64); }
        uint8_t packed[16];
        for (auto& x : packed) x = (uint8_t)rng();
        pack_k4(sc, mn, packed);
        uint8_t tail_a[16], tail_b[16];
        std::memcpy(tail_a, packed, 16); std::memcpy(tail_b, packed, 16);
        for (int k = 12; k < 16; ++k) { tail_a[k] = 0x00; tail_b[k] = 0xFF; }
        for (int j = 0; j < 8; ++j) {
            uint8_t d, m, d2, m2;
            get_scale_min_k4(j, tail_a, &d, &m);
            get_scale_min_k4(j, tail_b, &d2, &m2);
            CHECK(d == sc[j] && m == mn[j]);        // round trips through the packed form
            CHECK(d == d2 && m == m2);              // reads only bytes 0..11
            CHECK(d < 64 && m < 64);
        }
    }
}

void test_dequant_q2_0() {
    for (int it = 0; it < 300; ++it) {
        std::vector<uint8_t> blk(kQ2_0Bytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data(), dh);
        float want[64];
        for (int j = 0; j < 64; ++j) {
            const int c = (int)R(4);
            blk[2 + j / 4] |= (uint8_t)(c << (2 * (j % 4)));
            want[j] = half_to_float(dh) * (float)(c - 1);
        }
        run_block<64>(blk, kQ2_0Bytes, dequantize_q2_0, want);
    }
}

void test_dequant_q4_0() {
    for (int it = 0; it < 300; ++it) {
        std::vector<uint8_t> blk(kQ4_0Bytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data(), dh);
        int nib[32];
        float want[32];
        for (int j = 0; j < 32; ++j) nib[j] = (int)R(16);
        for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t)(nib[j] | (nib[j + 16] << 4));
        for (int j = 0; j < 32; ++j) { want[j] = (float)(nib[j] - 8) * half_to_float(dh); CHECK(nib[j] - 8 >= -8 && nib[j] - 8 <= 7); }
        run_block<32>(blk, kQ4_0Bytes, dequantize_q4_0, want);
    }
}

void test_dequant_q5_0() {
    for (int it = 0; it < 300; ++it) {
        std::vector<uint8_t> blk(kQ5_0Bytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data(), dh);
        int v[32];
        float want[32];
        uint32_t qh = 0;
        for (int j = 0; j < 32; ++j) { v[j] = (int)R(32); qh |= (uint32_t)(v[j] >> 4) << j; }
        for (int j = 0; j < 4; ++j) blk[2 + j] = (uint8_t)(qh >> (8 * j));
        for (int j = 0; j < 16; ++j) blk[6 + j] = (uint8_t)((v[j] & 15) | ((v[j + 16] & 15) << 4));
        for (int j = 0; j < 32; ++j) { want[j] = (float)(v[j] - 16) * half_to_float(dh); CHECK(v[j] - 16 >= -16 && v[j] - 16 <= 15); }
        run_block<32>(blk, kQ5_0Bytes, dequantize_q5_0, want);
    }
}

void test_dequant_q6_k() {
    for (int it = 0; it < 200; ++it) {
        std::vector<uint8_t> blk(kQ6KBytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data() + kQ6KDOff, dh);
        int8_t sc[16];
        for (auto& s : sc) s = (int8_t)(int)R(256);
        std::memcpy(blk.data() + kQ6KScOff, sc, 16);
        int v[256];
        float want[256];
        for (int n = 0; n < 2; ++n)
            for (int l = 0; l < 32; ++l) {
                int val[4];
                for (int k = 0; k < 4; ++k) { val[k] = (int)R(64); v[n * 128 + l + 32 * k] = val[k]; }
                blk[n * 64 + l] = (uint8_t)((val[0] & 15) | ((val[2] & 15) << 4));
                blk[n * 64 + 32 + l] = (uint8_t)((val[1] & 15) | ((val[3] & 15) << 4));
                blk[kQ6KQhOff + n * 32 + l] = (uint8_t)((val[0] >> 4) | ((val[1] >> 4) << 2) | ((val[2] >> 4) << 4) | ((val[3] >> 4) << 6));
            }
        const float d = half_to_float(dh);
        for (int n = 0; n < 2; ++n)
            for (int l = 0; l < 32; ++l)
                for (int k = 0; k < 4; ++k) {
                    const int si = 8 * n + l / 16 + 2 * k;
                    CHECK(si < 16);
                    want[n * 128 + l + 32 * k] = d * (float)sc[si] * (float)(v[n * 128 + l + 32 * k] - 32);
                }
        run_block<256>(blk, kQ6KBytes, dequantize_q6_k, want);
    }
}

void test_dequant_q4_k_q5_k() {
    for (int five = 0; five < 2; ++five)
        for (int it = 0; it < 200; ++it) {
            const size_t bytes = five ? kQ5KBytes : kQ4KBytes;
            std::vector<uint8_t> blk(bytes, 0);
            const uint16_t dh = rand_half(), mh = rand_half();
            put16(blk.data(), dh);
            put16(blk.data() + 2, mh);
            int sc[8], mn[8];
            for (int j = 0; j < 8; ++j) { sc[j] = (int)R(64); mn[j] = (int)R(64); }
            pack_k4(sc, mn, blk.data() + 4);
            const size_t qsOff = five ? kQ5KQsOff : kQ4KQsOff;
            int v[256];
            float want[256];
            for (int g = 0; g < 4; ++g)
                for (int l = 0; l < 32; ++l) {
                    const int a = (int)R(five ? 32 : 16), b = (int)R(five ? 32 : 16);
                    v[g * 64 + l] = a; v[g * 64 + 32 + l] = b;
                    blk[qsOff + g * 32 + l] = (uint8_t)((a & 15) | ((b & 15) << 4));
                    if (five) blk[kQ5KQhOff + l] |= (uint8_t)(((a >> 4) << (2 * g)) | ((b >> 4) << (2 * g + 1)));
                }
            const float d = half_to_float(dh), dmin = half_to_float(mh);
            for (int g = 0; g < 4; ++g) {
                const float d1 = d * (float)sc[2 * g], m1 = dmin * (float)mn[2 * g];
                const float d2 = d * (float)sc[2 * g + 1], m2 = dmin * (float)mn[2 * g + 1];
                for (int l = 0; l < 32; ++l) {
                    want[g * 64 + l] = d1 * (float)v[g * 64 + l] - m1;
                    want[g * 64 + 32 + l] = d2 * (float)v[g * 64 + 32 + l] - m2;
                }
            }
            if (five) run_block<256>(blk, bytes, dequantize_q5_k, want);
            else run_block<256>(blk, bytes, dequantize_q4_k, want);
        }
    for (int g = 0; g < 4; ++g) { CHECK(2 * (1 << (2 * g)) < 256); CHECK(2 * g + 1 < 8); }  // q5k masks fit a byte
}

void test_dequant_q3_k() {
    for (int it = 0; it < 300; ++it) {
        std::vector<uint8_t> blk(kQ3KBytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data() + kQ3KDOff, dh);
        int s[16];
        for (auto& x : s) x = (int)R(64);
        uint8_t* sp = blk.data() + kQ3KScalesOff;
        for (int i = 0; i < 8; ++i) sp[i] = (uint8_t)((s[i] & 15) | ((s[i + 8] & 15) << 4));
        for (int i = 0; i < 16; ++i) sp[8 + i % 4] |= (uint8_t)((s[i] >> 4) << (2 * (i / 4)));
        // unpacked scales must round trip
        int8_t un[16];
        q3k_unpack_scales(sp, un);
        for (int i = 0; i < 16; ++i) CHECK(un[i] == s[i]);
        float want[256];
        const float d = half_to_float(dh);
        for (int n = 0; n < 2; ++n)
            for (int j = 0; j < 4; ++j)
                for (int h = 0; h < 2; ++h) {
                    const int is = n * 8 + j * 2 + h;
                    CHECK(is < 16);
                    const float dl = d * (float)(s[is] - 32);
                    for (int l = 0; l < 16; ++l) {
                        const int val = (int)R(8) - 4;  // [-4, 3]
                        const int low = (val + 4) & 3, hi = val >= 0 ? 1 : 0;
                        blk[kQ3KQsOff + n * 32 + h * 16 + l] |= (uint8_t)(low << (2 * j));
                        if (hi) blk[kQ3KHmaskOff + h * 16 + l] |= (uint8_t)(1u << (n * 4 + j));
                        want[n * 128 + j * 32 + h * 16 + l] = dl * (float)val;
                    }
                }
        run_block<256>(blk, kQ3KBytes, dequantize_q3_k, want);
    }
}

void test_dequant_iq4_xs() {
    for (int it = 0; it < 300; ++it) {
        std::vector<uint8_t> blk(kIq4XsBytes, 0);
        const uint16_t dh = rand_half();
        put16(blk.data(), dh);
        int ls[8];
        uint16_t sh = 0;
        for (int ib = 0; ib < 8; ++ib) {
            ls[ib] = (int)R(64);
            blk[kIq4XsScalesLOff + ib / 2] |= (uint8_t)((ls[ib] & 15) << (4 * (ib % 2)));
            sh |= (uint16_t)((ls[ib] >> 4) << (2 * ib));
            CHECK(ib / 2 < 4 && 2 * ib <= 14 && ls[ib] - 32 >= -32 && ls[ib] - 32 <= 31);
        }
        put16(blk.data() + kIq4XsScalesHOff, sh);
        float want[256];
        const float d = half_to_float(dh);
        for (int ib = 0; ib < 8; ++ib) {
            const float dl = d * (float)(ls[ib] - 32);
            for (int j = 0; j < 16; ++j) {
                const int a = (int)R(16), b = (int)R(16);
                blk[kIq4XsQsOff + ib * 16 + j] = (uint8_t)(a | (b << 4));
                want[ib * 32 + j] = dl * (float)kvalues_iq4nl[a];
                want[ib * 32 + 16 + j] = dl * (float)kvalues_iq4nl[b];
            }
        }
        run_block<256>(blk, kIq4XsBytes, dequantize_iq4_xs, want);
    }
}

void test_dequant_random_bytes_bounded() {
    // arbitrary bytes (finite d): outputs finite and bounded by the code ranges
    auto rb = [&](size_t n, size_t doff, size_t dminoff) {
        std::vector<uint8_t> b(n);
        for (auto& x : b) x = (uint8_t)rng();
        put16(b.data() + doff, rand_half());
        if (dminoff != (size_t)-1) put16(b.data() + dminoff, rand_half());
        return b;
    };
    for (int it = 0; it < 500; ++it) {
        float y[256];
        { auto b = rb(kQ2_0Bytes, 0, -1); dequantize_q2_0(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data()))); for (int i = 0; i < 64; ++i) CHECK(std::fabs(y[i]) <= 2 * d); }
        { auto b = rb(kQ4_0Bytes, 0, -1); dequantize_q4_0(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data()))); for (int i = 0; i < 32; ++i) CHECK(std::fabs(y[i]) <= 8 * d); }
        { auto b = rb(kQ5_0Bytes, 0, -1); dequantize_q5_0(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data()))); for (int i = 0; i < 32; ++i) CHECK(std::fabs(y[i]) <= 16 * d); }
        { auto b = rb(kQ6KBytes, kQ6KDOff, -1); dequantize_q6_k(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data() + kQ6KDOff))); for (int i = 0; i < 256; ++i) CHECK(std::fabs(y[i]) <= d * 128.f * 32.f * 1.0001f); }
        { auto b = rb(kQ3KBytes, kQ3KDOff, -1); dequantize_q3_k(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data() + kQ3KDOff))); for (int i = 0; i < 256; ++i) CHECK(std::fabs(y[i]) <= d * 32.f * 4.f * 1.0001f); }
        { auto b = rb(kIq4XsBytes, 0, -1); dequantize_iq4_xs(b.data(), y); const float d = std::fabs(half_to_float(read_u16(b.data()))); for (int i = 0; i < 256; ++i) CHECK(std::fabs(y[i]) <= d * 32.f * 127.f * 1.0001f); }
        { auto b = rb(kQ4KBytes, 0, 2); dequantize_q4_k(b.data(), y); for (int i = 0; i < 256; ++i) CHECK(std::isfinite(y[i])); }
        { auto b = rb(kQ5KBytes, 0, 2); dequantize_q5_k(b.data(), y); for (int i = 0; i < 256; ++i) CHECK(std::isfinite(y[i])); }
    }
}
