#include <cstring>
#include <vector>
#include "baloon/q2_avx2_kernel.hpp"
#include "check.hpp"

using namespace baloon;
namespace f16 = baloon::f16;

namespace {
struct Row {
    std::vector<uint8_t> gguf;      // 18-byte blocks
    std::vector<uint8_t> codes;     // 16 bytes per block (expert-kernel form)
    std::vector<uint8_t> scales;    // fp16 per block
    uint32_t nb;
};
Row make_row(testh::Rng& rng, uint32_t nb) {
    Row r; r.nb = nb;
    r.gguf.resize(nb * 18); r.codes.resize(nb * 16); r.scales.resize(nb * 2);
    for (uint32_t b = 0; b < nb; ++b) {
        const uint16_t h = f16::float_to_half((float)(0.01 + 0.2 * rng.uni()));
        std::memcpy(&r.gguf[18 * b], &h, 2);
        std::memcpy(&r.scales[2 * b], &h, 2);
        for (uint32_t k = 0; k < 16; ++k) { r.gguf[18 * b + 2 + k] = r.codes[16 * b + k] = (uint8_t)rng.below(256); }
    }
    return r;
}
}  // namespace

void test_avx2_layout() {
    CHECK(q2_avx2::BLOCK_BYTES == 18 && 2 + 16 == q2_avx2::BLOCK_BYTES);
    for (uint64_t b = 0; b < 100; ++b) {
        CHECK(q2_avx2::codes_offset(b) + 16 == q2_avx2::scale_offset(b + 1));
        CHECK(q2_avx2::codes_offset(b) == q2_avx2::scale_offset(b) + 2);
    }
    for (uint32_t b = 0; b < 100; ++b) {
        CHECK(64 * b + 32 == 32 * q2_avx2::high_chunk(b));
        CHECK(64 * b == 32 * q2_avx2::low_chunk(b));
    }
    for (uint32_t n = 0; n < 700; ++n) for (uint32_t k = 0; k < q2_avx2::n_chunks(n); ++k) CHECK(32 * k + 32 <= n);
    for (uint32_t nt = 1; nt <= 30; ++nt) {
        uint32_t cov = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += q2_avx2::batch_size(nt, t0)) {
            const uint32_t b = q2_avx2::batch_size(nt, t0);
            CHECK(b >= 1 && b <= 4 && t0 + b <= nt);
            cov += b;
        }
        CHECK(cov == nt);
    }
}

void test_avx2_unpack() {
    // value 4k+t is code t of byte k; low half = bytes 0..7, high half = bytes 8..15
    for (uint32_t byte = 0; byte < 256; ++byte) {
        uint8_t blk[16], out[64];
        for (uint32_t k = 0; k < 16; ++k) blk[k] = (uint8_t)((byte + 37 * k) & 255);
        q2_avx2::unpack_block(blk, out);
        for (uint32_t i = 0; i < 64; ++i) {
            CHECK(out[i] < 4);
            CHECK(out[i] == ((blk[i / 4] >> (2 * (i % 4))) & 3));
            if (i < 32) CHECK(i / 4 < 8); else CHECK(i / 4 >= 8 && i / 4 < 16);
        }
        uint8_t other[64];
        q2_expert::unpack_block(blk, other);
        CHECK(std::memcmp(out, other, 64) == 0);
    }
#if defined(__AVX2__)
    testh::Rng rng(3);
    for (int it = 0; it < 4000; ++it) {
        uint8_t blk[16], ref[64], got[64];
        for (auto& v : blk) v = (uint8_t)(it < 256 ? it : rng.below(256));
        q2_avx2::unpack_block(blk, ref);
        __m256i lo, hi;
        q2_avx2::unpack64_avx2(blk, lo, hi);
        _mm256_storeu_si256((__m256i*)got, lo);
        _mm256_storeu_si256((__m256i*)(got + 32), hi);
        CHECK(std::memcmp(ref, got, 64) == 0);
    }
    std::printf("  (AVX2 intrinsic path compiled and checked)\n");
#else
    std::printf("  (AVX2 intrinsic path not compiled in this build)\n");
#endif
}

void test_avx2_exact() {
    // maddubs of two code*int8 products never saturates int16; half sums fit int32
    for (int c0 = 0; c0 < 4; ++c0) for (int c1 = 0; c1 < 4; ++c1)
        for (int q0 = -127; q0 <= 127; ++q0) for (int q1 = -127; q1 <= 127; q1 += 3) {
            const int v = c0 * q0 + c1 * q1;
            CHECK(v >= -32768 && v <= 32767);
        }
    CHECK(2 * 3 * 127 <= 32767);
    CHECK((int64_t)3 * 127 * 32 < ((int64_t)1 << 31));
    for (int c = 0; c < 4; ++c) CHECK(c - 1 >= -1 && c - 1 <= 2);
}

void test_avx2_rows() {
    testh::Rng rng(91);
    for (int it = 0; it < 30; ++it) {
        const uint32_t nb = 1 + rng.below(9);
        Row r = make_row(rng, nb);
        std::vector<float> x(nb * 64);
        for (auto& v : x) v = (float)rng.gauss();
        q2_expert::ActQ a = q2_expert::act_quant(x.data(), nb * 64);
        std::vector<int8_t> q(nb * 64); std::vector<float> d(nb * 2); std::vector<int32_t> hx(nb * 2);
        q2_avx2::act_quant(x.data(), nb * 64, q.data(), d.data(), hx.data());
        CHECK(q == a.q && d == a.d && hx == a.hx);
        for (auto v : q) CHECK(v >= -127 && v <= 127);

        const float s = q2_avx2::row_dot_scalar(r.gguf.data(), nb, q.data(), d.data(), hx.data());
        const float best = q2_avx2::row_dot(r.gguf.data(), nb, q.data(), d.data(), hx.data());
        const float expert = q2_expert::row_dot(r.codes.data(), r.scales.data(), nb, q.data(), d.data(), hx.data());
        CHECK(std::memcmp(&s, &expert, 4) == 0);   // GGUF layout and expert layout agree bit for bit
        CHECK(std::memcmp(&s, &best, 4) == 0);     // SIMD (when compiled) equals scalar bit for bit
    }
    // multi-token and matrix forms
    const uint32_t nb = 5, nt = 7, nrows = 6;
    std::vector<uint8_t> mat;
    for (uint32_t r = 0; r < nrows; ++r) { Row row = make_row(rng, nb); mat.insert(mat.end(), row.gguf.begin(), row.gguf.end()); }
    std::vector<int8_t> q((size_t)nt * nb * 64); std::vector<float> d((size_t)nt * nb * 2); std::vector<int32_t> hx((size_t)nt * nb * 2);
    for (uint32_t t = 0; t < nt; ++t) {
        std::vector<float> x(nb * 64);
        for (auto& v : x) v = (float)rng.gauss();
        q2_avx2::act_quant(x.data(), nb * 64, q.data() + (size_t)t * nb * 64, d.data() + (size_t)t * nb * 2, hx.data() + (size_t)t * nb * 2);
    }
    std::vector<float> out((size_t)nrows * nt, -1.f);
    q2_avx2::rows(mat.data(), nb * 18, nrows, nb, nt, q.data(), nb * 64, d.data(), hx.data(), nb * 2, out.data());
    for (uint32_t r = 0; r < nrows; ++r)
        for (uint32_t t = 0; t < nt; ++t) {
            const float one = q2_avx2::row_dot_scalar(mat.data() + (size_t)r * nb * 18, nb, q.data() + (size_t)t * nb * 64,
                                                      d.data() + (size_t)t * nb * 2, hx.data() + (size_t)t * nb * 2);
            CHECK(std::memcmp(&one, &out[(size_t)r * nt + t], 4) == 0);
        }
}
