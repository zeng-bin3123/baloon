// test_u5c4.cpp - LAYER 1 invariants of the unit-5 modules that are checkable on the host, against PRODUCTION code:
//   ExpertRowDot / Q2Avx2Rows -> q2rows (src/cpu/q2_avx2_rows.cpp): unpack, activation quantizer, row dots, token groups
//   IqAvx512Rows              -> iquant512 layout helpers + kern::iq512_* (src/cpu/iq_rows_avx512.cpp)
//   S2ExpertGroupedHits       -> s2_blob geometry / gate-major indexing
//   SGemvSplitKernels         -> sgemv_idx indexing contracts
// Exits non-zero on any failed check.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/iquant_avx512_rows.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/s2_expert_grouped_blob.hpp"
#include "baloon/s_family_gemv_indexing.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

using namespace baloon;

static std::mt19937 rng(12345);

static float h2f_ref(uint16_t h) {  // exact fp16 -> float
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m + 1024), e - 25);
    return s ? -v : v;
}

// fp16 scale 0x3000 = 0.125, 0x3400 = 0.25, 0x3800 = 0.5 ...
static uint16_t rand_scale() { static const uint16_t t[] = {0x3000, 0x3400, 0x3800, 0x3c00, 0x2c00}; return t[rng() % 5]; }

// ---- chunksK / mask / dispatch models of the theory (used to cross-check production group sizes) -----------------------
static std::vector<uint32_t> groups(uint32_t k, uint32_t n) {
    std::vector<uint32_t> g;
    while (n) { const uint32_t t = n < k ? n : k; g.push_back(t); n -= t; }
    return g;
}

static void test_unpack_and_codes() {
    for (int it = 0; it < 200; ++it) {
        uint8_t b16[16];
        for (auto& b : b16) b = (uint8_t)rng();
        uint8_t out[64];
        q2rows::unpack_codes_64(b16, out);
        for (int v = 0; v < 64; ++v) {
            const int expect = (b16[v / 4] >> (2 * (v % 4))) & 3;      // theory: code byte v = (byte / 4^(v%4)) % 4
            CHECK(out[v] == expect);
            CHECK(out[v] < 4);
            CHECK(q2rows::code_byte(v) == (uint32_t)(v / 4) && q2rows::code_shift(v) == (uint32_t)(2 * (v % 4)));
        }
        // lossless: four codes recombine into the byte
        for (int i = 0; i < 16; ++i)
            CHECK(out[4 * i] + 4 * out[4 * i + 1] + 16 * out[4 * i + 2] + 64 * out[4 * i + 3] == b16[i]);
        for (int v = 0; v < 32; ++v) CHECK(q2rows::code_byte(v) < 8);                // lo vector: bytes 0..7
        for (int v = 32; v < 64; ++v) CHECK(q2rows::code_byte(v) >= 8 && q2rows::code_byte(v) < 16);
    }
    CHECK(q2rows::BLOCK_VALUES == 16 * 4 && q2rows::BLOCK_BYTES == 18);
    for (uint32_t b = 0; b < 40; ++b)
        for (uint32_t h = 0; h < 2; ++h) {
            CHECK(q2rows::chunk_of(b, 32 * h) / 2 == b);                              // chunk 2b+h belongs to block c/2
            CHECK(q2rows::chunk_of(b, 32 * h) * 32 == b * 64 + h * 32);               // chunk offset identity
        }
}

static void test_act_quant() {
    for (int it = 0; it < 200; ++it) {
        const uint32_t n = 32 * (1 + rng() % 6);
        std::vector<float> x(n);
        const float amp = (it % 7 == 0) ? 1e4f : (it % 5 == 0 ? 1e-3f : 3.f);
        std::uniform_real_distribution<float> u(-amp, amp);
        for (auto& v : x) v = u(rng);
        if (it % 11 == 0) std::fill(x.begin(), x.begin() + 32, 0.f);                 // all-zero chunk
        std::vector<int8_t> q(n);
        std::vector<float> d(n / 32);
        std::vector<int32_t> hx(n / 32);
        q2rows::act_quant_q8_1_avx2(x.data(), n, q.data(), d.data(), hx.data());
        for (uint32_t c = 0; c < n / 32; ++c) {
            int32_t s = 0;
            float amax = 0;
            for (int i = 0; i < 32; ++i) {
                const int v = q[c * 32 + i];
                CHECK(v >= -127 && v <= 127);                                        // clamp127_range
                s += v;
                amax = std::fmax(amax, std::fabs(x[c * 32 + i]));
            }
            CHECK(s == hx[c]);                                                       // hx is the exact chunk sum
            CHECK(std::abs(hx[c]) <= 32 * 127);                                      // sum bound
            CHECK(d[c] == amax / 127.f);
            if (d[c] > 0) {
                for (int i = 0; i < 32; ++i) {                                       // round-to-nearest within half a step
                    const double r = (double)x[c * 32 + i] / d[c];
                    CHECK(std::fabs(r - q[c * 32 + i]) <= 0.5 + 1e-3);
                }
            }
        }
        // the two entry points are the same kernel
        std::vector<int8_t> q2v(n);
        std::vector<float> d2(n / 32);
        std::vector<int32_t> hx2(n / 32);
        q2rows::activation_quantizer(x.data(), n, q2v.data(), d2.data(), hx2.data());
        CHECK(q2v == q && d2 == d && hx2 == hx);
    }
}

// scalar double reference of one Q2_0 row (nb blocks) against one quantized token
static double ref_row(const uint8_t* row, uint32_t nb, const int8_t* q, const float* xd, const int32_t* hx, int64_t* int_total) {
    double acc = 0;
    int64_t itot = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + 18 * b;
        const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
        const double wd = h2f_ref(dbits);
        for (int h = 0; h < 2; ++h) {
            int64_t centered = 0, raw = 0, qs = 0;
            for (int i = 0; i < 32; ++i) {
                const int v = h * 32 + i;
                const int c = (blk[2 + v / 4] >> (2 * (v % 4))) & 3;
                const int qq = q[b * 64 + v];
                centered += (int64_t)(c - 1) * qq;
                raw += (int64_t)c * qq;
                qs += qq;
            }
            CHECK(centered == raw - qs);                                            // offDot_eq
            CHECK(qs == hx[2 * b + h]);
            CHECK(std::llabs(raw) <= 3 * 127 * 32);                                 // exact VNNI/maddubs partial sums
            itot += centered;
            acc += wd * (double)xd[2 * b + h] * (double)centered;
        }
    }
    *int_total = itot;
    return acc;
}

static void test_rows() {
    const uint32_t nb = 7, n = nb * 64, rows = 9;
    for (uint32_t nt = 1; nt <= 11; ++nt) {
        std::vector<uint8_t> w((size_t)rows * nb * 18);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = &w[((size_t)r * nb + b) * 18];
                const uint16_t s = rand_scale();
                blk[0] = (uint8_t)(s & 255); blk[1] = (uint8_t)(s >> 8);
                for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)rng();
            }
        std::vector<float> x((size_t)nt * n);
        std::uniform_real_distribution<float> u(-2.f, 2.f);
        for (auto& v : x) v = u(rng);
        const uint32_t ch = n / 32;
        std::vector<int8_t> q((size_t)nt * n);
        std::vector<float> xd((size_t)nt * ch);
        std::vector<int32_t> hx((size_t)nt * ch);
        for (uint32_t t = 0; t < nt; ++t)
            q2rows::act_quant_q8_1_avx2(&x[t * n], n, &q[t * n], &xd[t * ch], &hx[t * ch]);

        std::vector<float> out((size_t)nt * rows, -777.f), out2((size_t)nt * rows, -777.f);
        q2rows::q2_0_gguf_rows_multi_avx2(w.data(), 0, rows, nb, nt, q.data(), n, xd.data(), hx.data(), ch, out.data(), rows);
        q2rows::gguf_rows_multi(w.data(), 0, rows, nb, nt, q.data(), n, xd.data(), hx.data(), ch, out2.data(), rows);
        CHECK(std::memcmp(out.data(), out2.data(), out.size() * 4) == 0);            // named entry == production entry
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t r = 0; r < rows; ++r) {
                int64_t it;
                const double ref = ref_row(&w[(size_t)r * nb * 18], nb, &q[t * n], &xd[t * ch], &hx[t * ch], &it);
                const double got = out[(size_t)t * rows + r];
                CHECK(std::fabs(got - ref) <= 1e-4 * (std::fabs(ref) + 1.0));        // every token of every group written once
            }
        // a sub-range must leave the other rows untouched
        std::vector<float> part((size_t)nt * rows, -777.f);
        q2rows::q2_0_gguf_rows_multi_avx2(w.data(), 2, 5, nb, nt, q.data(), n, xd.data(), hx.data(), ch, part.data(), rows);
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t r = 0; r < rows; ++r) {
                if (r >= 2 && r < 5) CHECK(part[(size_t)t * rows + r] == out[(size_t)t * rows + r]);
                else CHECK(part[(size_t)t * rows + r] == -777.f);
            }
        // a token's result does not depend on how many tokens ride along
        if (nt > 1) {
            std::vector<float> one(rows);
            q2rows::q2_0_gguf_rows_multi_avx2(w.data(), 0, rows, nb, 1, q.data(), n, xd.data(), hx.data(), ch, one.data(), rows);
            for (uint32_t r = 0; r < rows; ++r) CHECK(one[r] == out[r]);
        }
        // token groups: sizes in 1..4, covering all nt exactly once
        uint32_t tot = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += q2rows::MAX_GROUP) { const uint32_t g = q2rows::group_size(nt, t0); CHECK(g >= 1 && g <= 4); tot += g; }
        CHECK(tot == nt);
        CHECK(groups(4, nt).size() == (nt + 3) / 4);
    }
}

static void test_q2_extremes() {  // the maddubs / int32 sums are exact even at the extreme codes and activations
    const uint32_t nb = 3, n = nb * 64;
    std::vector<uint8_t> w(nb * 18, 0xFF);                                          // all codes 3
    for (uint32_t b = 0; b < nb; ++b) { w[b * 18] = 0x00; w[b * 18 + 1] = 0x3c; }     // scale 1.0
    std::vector<int8_t> q(n, 127);
    std::vector<float> xd(n / 32, 1.f);
    std::vector<int32_t> hx(n / 32, 32 * 127);
    float out = 0;
    q2rows::q2_0_gguf_rows_multi_avx2(w.data(), 0, 1, nb, 1, q.data(), n, xd.data(), hx.data(), 0, &out, 1);
    CHECK(out == (float)(n * 2 * 127));                                             // (3-1) * 127 per value
    std::fill(q.begin(), q.end(), -127);
    std::fill(hx.begin(), hx.end(), -32 * 127);
    q2rows::q2_0_gguf_rows_multi_avx2(w.data(), 0, 1, nb, 1, q.data(), n, xd.data(), hx.data(), 0, &out, 1);
    CHECK(out == -(float)(n * 2 * 127));
}

static void test_iq512_layout() {
    namespace iq = iquant512;
    for (uint32_t t = 0; t < 64; ++t) {
        const bool exp = (t == 16 || t == 17 || t == 18 || t == 21 || t == 22);
        CHECK(kern::iq512_supported(t) == exp);
        CHECK(iq::supported(t) == exp);
        CHECK((iq::bytes_of(t) != 0) == exp);
    }
    CHECK(!kern::iq512_supported(29) && !kern::iq512_supported(42));                // IQ1_M stays on ggml-cpu; Q2_0 is not an i-quant
    CHECK(iq::bytes_of(16) == 66 && iq::bytes_of(17) == 74 && iq::bytes_of(18) == 98 && iq::bytes_of(21) == 110 && iq::bytes_of(22) == 82);
    CHECK(iq::QK_K == 4 * 64);
    const uint32_t types[5] = {16, 17, 18, 21, 22};
    for (uint32_t t : types)
        for (uint32_t j = 0; j < 4; ++j) {
            const iq::ChunkFields c = iq::chunk_fields(t, j);
            std::vector<uint8_t> used(iq::bytes_of(t), 0);
            for (uint32_t k = 0; k < (uint32_t)iq::F_COUNT; ++k) {
                const iq::Field f = c.f[k];
                CHECK((uint64_t)f.offset + f.len <= iq::bytes_of(t));                // every decode read stays inside its block
                for (uint32_t i = 0; i < f.len && f.offset + i < used.size(); ++i) { CHECK(used[f.offset + i] == 0); used[f.offset + i] = 1; }
            }
        }
    // chunks of one block read disjoint ranges per field kind
    for (uint32_t t : types)
        for (uint32_t k = 0; k < (uint32_t)iq::F_COUNT; ++k)
            for (uint32_t a = 0; a < 4; ++a)
                for (uint32_t b = a + 1; b < 4; ++b) {
                    const iq::Field fa = iq::chunk_fields(t, a).f[k], fb = iq::chunk_fields(t, b).f[k];
                    if (fa.len && fb.len) CHECK(fa.offset + fa.len <= fb.offset || fb.offset + fb.len <= fa.offset);
                }
    for (uint32_t l = 0; l < 32; ++l) CHECK(iq::scale_of_lane(l) < 4 && iq::scale_of_lane(l) == l / 8);
    for (int y = -127; y <= 127; ++y) { CHECK(iq::apply_sign(true, y) >= -127 && iq::apply_sign(true, y) <= 127); CHECK(iq::apply_sign(true, iq::apply_sign(true, y)) == y); CHECK(iq::apply_sign(false, y) == y); }
    for (uint32_t nt = 1; nt <= 40; ++nt) {
        uint32_t tot = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += 4) { const uint32_t g = iq::batch_size(nt, t0); CHECK(g >= 1 && g <= 4); tot += g; }
        CHECK(tot == nt);
    }
}

static void test_s2_blob() {
    namespace sb = s2_blob;
    CHECK(sb::H == 2560 && sb::FF == 640);
    CHECK(sb::ROW_GU == 640 && sb::ROW_D == 160 && sb::SCALE_GU == 40 && sb::SCALE_D == 10);
    CHECK(sb::O_D_CODES == 819200 && sb::O_GU_SCALES == 1228800 && sb::O_D_SCALES == 1331200);
    CHECK(sb::BLOB_BYTES == 1382400 && sb::BLOB_BYTES == 1228800 + 153600);
    for (uint64_t i = 0; i < 1280; ++i) { CHECK(i * 640 + 640 <= sb::O_D_CODES); CHECK(sb::O_GU_SCALES + i * 80 + 80 <= sb::O_D_SCALES); }
    for (uint64_t r = 0; r < 2560; ++r) { CHECK(sb::O_D_CODES + r * 160 + 160 <= sb::O_GU_SCALES); CHECK(sb::O_D_SCALES + r * 20 + 20 <= sb::BLOB_BYTES); }
    for (uint64_t c = 0; c < 80; ++c) CHECK(c / 2 < 40);
    for (uint64_t c = 0; c < 20; ++c) CHECK(c / 2 < 10);
    // gate-major output: gate at h*FF+r, up at n*FF + h*FF + r; injective and disjoint
    for (uint64_t n = 1; n <= 8; ++n) {
        std::vector<char> seen(2 * n * sb::FF, 0);
        for (uint64_t h = 0; h < n; ++h)
            for (uint64_t i = 0; i < 2 * sb::FF; ++i) {
                const uint64_t o = sb::gu_out(n, sb::FF, h, i);
                CHECK(o < 2 * n * sb::FF);
                CHECK(!seen[o]);
                seen[o] = 1;
                if (i % 2 == 0) CHECK(o == h * sb::FF + i / 2 && o < n * sb::FF);
                else CHECK(o == n * sb::FF + h * sb::FF + i / 2 && o >= n * sb::FF);
                if (i % 2 == 0) CHECK(sb::gu_out(n, sb::FF, h, i + 1) == o + n * sb::FF);  // swiglu pairing
            }
    }
    CHECK(8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 == 23040 && 23040 <= 49152);
    CHECK(2 * 640 / 32 == 40 && 2560 / 64 == 40);
    for (uint64_t x = 0; x < 5000; ++x) { CHECK(sb::align16(x) >= x && sb::align16(x) % 16 == 0 && sb::align16(x) - x < 16); }
    // weight-bias identity on random data
    for (int it = 0; it < 100; ++it) {
        std::vector<int64_t> c(64), x(64);
        for (auto& v : c) v = rng() % 4;
        for (auto& v : x) v = (int64_t)(rng() % 255) - 127;
        CHECK(sb::dot_minus(c, x) == sb::dot(c, x) - sb::dot_ones(c, x));
    }
}

static void test_sgemv_idx() {
    namespace si = sgemv_idx;
    for (uint32_t b : {2u, 4u, 8u}) {
        CHECK(si::per_byte(b) * b == 8 && si::valid_width(b) && 32 % si::per_byte(b) == 0);
        CHECK(si::bytes_per_quad_word(b) == (b == 2 ? 1u : b == 4 ? 2u : 4u));
        for (int it = 0; it < 200; ++it) {
            uint8_t row[16];
            for (auto& v : row) v = (uint8_t)rng();
            for (uint32_t e = 0; e < 16 * si::per_byte(b); ++e) {
                CHECK(si::code_of_elem(row, e, b) < (1u << b));
                CHECK(si::byte_of_elem(e, b) == e / si::per_byte(b));
            }
        }
        for (uint32_t q = 0; q < 8; ++q)
            for (uint32_t k = 0; k < 4; ++k) CHECK(si::byte_of_elem(4 * q + k, b) >= (4 * q * b) / 8 && si::byte_of_elem(4 * q + k, b) < (4 * q * b) / 8 + si::bytes_per_quad_word(b));
    }
    CHECK(!si::valid_width(3) && !si::valid_width(16));
    for (uint32_t lane = 0; lane < 32; ++lane)
        for (uint32_t t = 0; t < 8; ++t) {
            const uint32_t i = lane * 16 + 512 * t;
            CHECK(i % 16 == 0 && i % 256 + 16 <= 256 && i % 32 + 16 <= 32);        // a lane iteration never straddles a block
        }
    CHECK(640 % 32 == 0 && 640 % 256 != 0);                                        // ffn_down_shexp: Q8_0 only
    CHECK(si::row_code_bytes(640, 2) == 160);
}

static void test_theory_lemmas() {
    // exhaustive small-range checks of the pure-integer statements of ExpertRowDot / Q2Avx2Rows
    for (int c = 0; c <= 3; ++c)
        for (int q = -127; q <= 127; ++q) CHECK(std::abs(c * q) <= 381);
    for (int c0 = 0; c0 <= 3; ++c0) for (int c1 = 0; c1 <= 3; ++c1)
        for (int q0 : {-127, -1, 0, 1, 127}) for (int q1 : {-127, -1, 0, 1, 127})
            CHECK(std::abs(c0 * q0 + c1 * q1) <= 762 && c0 * q0 + c1 * q1 >= -32768 && c0 * q0 + c1 * q1 <= 32767);
    for (uint32_t l = 0; l < 16; ++l) {
        for (uint32_t e = 0; e < 4; ++e) CHECK((4 * l + e) / 32 == l / 8);        // a VNNI lane's four values share one scale
        for (uint32_t i = 0; i < 8; ++i) CHECK(l / 8 + 2 * i < 16);               // permute index stays in the 16 lanes
    }
    for (uint32_t nb = 0; nb <= 8; ++nb) {                                        // mask16
        const uint32_t m = nb >= 8 ? 65535u : (1u << (2 * nb)) - 1u;
        CHECK(m < 65536u);
        if (nb < 8) CHECK(m + 1 == (1u << (2 * nb)));
        else CHECK(m == 65535u);
    }
    for (uint32_t n = 1; n <= 64; ++n) {                                          // blocks walk in groups of at most 8
        uint32_t tot = 0;
        for (uint32_t g : groups(8, n)) { CHECK(g >= 1 && g <= 8); tot += g; }
        CHECK(tot == n);
        const uint32_t nt = (n >= 1 && n <= 7) ? n : 8;                           // token-count dispatch
        CHECK(nt >= 1 && nt <= 8);
        if (n <= 7) CHECK(nt == n);
        if (n <= 8) CHECK(n <= nt);
    }
    for (int a = -3; a <= 3; ++a) for (int b = -3; b <= 3; ++b) for (int c = -3; c <= 3; ++c) for (int d = -3; d <= 3; ++d)
        for (int e = -3; e <= 3; e += 2) for (int f = -3; f <= 3; f += 3) for (int g = -3; g <= 3; g += 3) for (int h = -3; h <= 3; h += 3) {
            CHECK(((a + e) + (b + f)) + ((c + g) + (d + h)) == a + b + c + d + e + f + g + h);
            CHECK((((a + e) + (c + g)) + ((b + f) + (d + h))) == a + b + c + d + e + f + g + h);
        }
}

int main() {
    CHECK(q2rows::built_with_required_isa());
    test_unpack_and_codes();
    test_act_quant();
    test_rows();
    test_q2_extremes();
    test_iq512_layout();
    test_s2_blob();
    test_sgemv_idx();
    test_theory_lemmas();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("U5C4 HOST TESTS FAILED\n"); return 1; }
    std::printf("U5C4 HOST TESTS PASSED\n");
    return 0;
}

