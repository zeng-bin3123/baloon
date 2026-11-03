#include <cmath>
#include <cstring>
#include <vector>
#include "baloon/q2_expert_vnni_kernel.hpp"
#include "check.hpp"

using namespace baloon::q2_expert;
namespace f16 = baloon::f16;

namespace {
// Random blob: random code bytes, small positive fp16 scales.
std::vector<uint8_t> make_blob(uint64_t seed) {
    testh::Rng rng(seed);
    std::vector<uint8_t> b(BLOB);
    for (auto& v : b) v = (uint8_t)rng.below(256);
    auto fill_scales = [&](uint64_t off, uint64_t bytes) {
        for (uint64_t i = 0; i < bytes; i += 2) {
            uint16_t h = f16::float_to_half((float)(0.002 + 0.01 * rng.uni()));
            std::memcpy(&b[off + i], &h, 2);
        }
    };
    fill_scales(GU_SCALES_OFF, GU_SCALES_BYTES);
    fill_scales(D_SCALES_OFF, D_SCALES_BYTES);
    return b;
}
}  // namespace

void test_q2e_geometry() {
    CHECK(BLOB == 1382400);
    CHECK(SC_GU == 40 && SC_D == 10);
    CHECK(H / QKA == 80 && FF / QKA == 20);
    CHECK(QK == 2 * QKA);
    CHECK(H % QK == 0 && FF % QK == 0);
    CHECK(ROW_GU == 640 && ROW_D == 160);
    CHECK(GU_CODES_OFF + GU_CODES_BYTES == GU_SCALES_OFF);
    CHECK(GU_SCALES_OFF + GU_SCALES_BYTES == D_CODES_OFF);
    CHECK(D_CODES_OFF + D_CODES_BYTES == D_SCALES_OFF);
    CHECK(D_SCALES_OFF + D_SCALES_BYTES == BLOB);
    // closed form of the total from the theory
    CHECK(BLOB == 2ull * FF * ROW_GU + 2ull * FF * SC_GU * 2 + (uint64_t)H * ROW_D + (uint64_t)H * SC_D * 2);
    for (uint32_t r = 0; r < FF; ++r) {
        CHECK(gate_slot(r) != up_slot(r));
        CHECK(up_slot(r) < 2 * FF);
        CHECK(gate_slot(r) == 2 * r && up_slot(r) == 2 * r + 1);
    }
    // accessors stay in the blob and end exactly at section ends
    std::vector<uint8_t> blob(BLOB);
    const uint8_t* base = blob.data();
    CHECK(gu_codes(base, 2 * FF - 1) + ROW_GU == base + GU_SCALES_OFF);
    CHECK(gu_scales(base, 2 * FF - 1) + SC_GU * 2 == base + D_CODES_OFF);
    CHECK(down_codes(base, H - 1) + ROW_D == base + D_SCALES_OFF);
    CHECK(down_scales(base, H - 1) + SC_D * 2 == base + BLOB);
    for (uint32_t nb = 1; nb < 40; ++nb)
        for (uint32_t b = 0; b < nb; ++b)
            for (uint32_t h = 0; h < 2; ++h) CHECK(2 * b + h < 2 * nb);
}

void test_q2e_codes() {
    for (uint32_t c0 = 0; c0 < 4; ++c0) for (uint32_t c1 = 0; c1 < 4; ++c1)
    for (uint32_t c2 = 0; c2 < 4; ++c2) for (uint32_t c3 = 0; c3 < 4; ++c3) {
        const uint32_t p = pack4(c0, c1, c2, c3);
        CHECK(p < 256);
        CHECK(code_of(p, 0) == c0 && code_of(p, 1) == c1 && code_of(p, 2) == c2 && code_of(p, 3) == c3);
    }
    for (uint32_t byte = 0; byte < 256; ++byte)
        for (uint32_t k = 0; k < 4; ++k) {
            CHECK(code_of(byte, k) < 4);
            CHECK(code_of(byte, k) == byte / (1u << (2 * k)) % 4);
        }
    for (uint32_t c = 0; c < 4; ++c) CHECK(decode_code(c) >= -1 && decode_code(c) <= 2);
    CHECK(decode_code(0) == -1 && decode_code(1) == 0 && decode_code(2) == 1 && decode_code(3) == 2);
    for (uint32_t b = 0; b < 5; ++b)
        for (uint32_t j = 0; j < 64; ++j) {
            CHECK(code_byte_index(b, j) < (b + 1) * 16);
            CHECK(code_byte_index(b, j) >= b * 16);
        }
    // unpack_block agrees with the per-code accessor
    testh::Rng rng(7);
    for (int it = 0; it < 50; ++it) {
        uint8_t blk[16], out[64];
        for (auto& v : blk) v = (uint8_t)rng.below(256);
        unpack_block(blk, out);
        for (uint32_t j = 0; j < 64; ++j) CHECK(out[j] == block_code(blk, 0, j));
    }
}

void test_q2e_offset_identity() {
    // sum (c-1)*q == sum c*q - sum q, exactly over integers, for every half of random blocks (incl. extreme q)
    testh::Rng rng(11);
    for (int it = 0; it < 2000; ++it) {
        uint8_t codes[16 * 3];
        int8_t q[32];
        for (auto& v : codes) v = (uint8_t)rng.below(256);
        const int mode = it % 4;
        for (auto& v : q) v = mode == 0 ? (int8_t)127 : mode == 1 ? (int8_t)-127 : (int8_t)((int)rng.below(255) - 127);
        int32_t sq = 0;
        for (auto v : q) sq += v;
        for (uint32_t b = 0; b < 3; ++b)
            for (uint32_t h = 0; h < 2; ++h)
                CHECK(half_decoded_dot(codes, b, h, q) == half_code_dot(codes, b, h, q) - sq);
    }
    // no int32 overflow: worst-case block dot
    CHECK((int64_t)3 * 127 * 64 < (int64_t)1 << 31);
    uint8_t all3[16]; std::memset(all3, 0xFF, sizeof all3);
    int8_t q127[32]; std::memset(q127, 127, sizeof q127);
    CHECK(half_code_dot(all3, 0, 0, q127) == 3 * 127 * 32);
}

void test_q2e_quant() {
    testh::Rng rng(21);
    for (int it = 0; it < 300; ++it) {
        const uint32_t n = 32 * (1 + rng.below(8));
        std::vector<float> x(n);
        const float scale = it % 5 == 0 ? 1e-6f : it % 5 == 1 ? 1e4f : 3.f;
        for (auto& v : x) v = (float)rng.gauss() * scale;
        if (it % 7 == 0) std::fill(x.begin(), x.begin() + 32, 0.f);  // all-zero chunk
        ActQ a = act_quant(x.data(), n);
        for (uint32_t c = 0; c < n / 32; ++c) {
            int32_t s = 0;
            for (uint32_t i = 0; i < 32; ++i) {
                const int v = a.q[c * 32 + i];
                CHECK(v >= -127 && v <= 127);
                s += v;
                if (a.d[c] > 0) CHECK(std::fabs((float)v * a.d[c] - x[c * 32 + i]) <= a.d[c] * 0.5f * 1.001f + 1e-30f);
                else CHECK(v == 0);
            }
            CHECK(s == a.hx[c]);
            CHECK(std::abs(s) <= 127 * 32);
        }
    }
    // peak maps to +-127 exactly
    std::vector<float> x(32, 0.f); x[3] = -5.f; x[9] = 2.5f;
    ActQ a = act_quant(x.data(), 32);
    CHECK(a.q[3] == -127);
    CHECK(a.q[9] == 64 || a.q[9] == 63);
    // clamp helper
    for (int v = -400; v <= 400; ++v) {
        const int c = clamp127(v);
        CHECK(c >= -127 && c <= 127);
        if (v >= -127 && v <= 127) CHECK(c == v);
    }
}

void test_q2e_row_dot() {
    testh::Rng rng(31);
    for (int it = 0; it < 40; ++it) {
        const uint32_t nb = 1 + rng.below(12);
        std::vector<uint8_t> codes(nb * 16), sc(nb * 2);
        std::vector<float> wd(nb), x(nb * 64);
        for (auto& v : codes) v = (uint8_t)rng.below(256);
        for (uint32_t b = 0; b < nb; ++b) {
            const uint16_t h = f16::float_to_half((float)(0.01 + rng.uni()));
            std::memcpy(&sc[2 * b], &h, 2);
            wd[b] = f16::half_to_float(h);
        }
        for (auto& v : x) v = (float)rng.gauss();
        ActQ a = act_quant(x.data(), nb * 64);
        const float got = row_dot(codes.data(), sc.data(), nb, a.q.data(), a.d.data(), a.hx.data());
        // independent double reference with decoded weights (c-1)*d against dequantised activations
        double ref = 0, mag = 0;
        for (uint32_t b = 0; b < nb; ++b)
            for (uint32_t j = 0; j < 64; ++j) {
                const uint32_t c = (codes[b * 16 + j / 4] >> (2 * (j % 4))) & 3;
                const uint32_t ch = 2 * b + j / 32;
                const double xh = (double)a.q[b * 64 + j] * (double)a.d[ch];
                ref += ((double)c - 1.0) * wd[b] * xh;
                mag += std::fabs(2.0 * wd[b] * xh);
            }
        CHECK_MSG(std::fabs((double)got - ref) <= 1e-5 * (mag + 1.0), "got=%g ref=%g", (double)got, ref);
    }
    // zero activations give exactly zero
    std::vector<uint8_t> codes(16, 0xAB), sc(2, 0x00);
    uint16_t h = f16::float_to_half(1.f); std::memcpy(sc.data(), &h, 2);
    std::vector<float> z(64, 0.f);
    ActQ a = act_quant(z.data(), 64);
    CHECK(row_dot(codes.data(), sc.data(), 1, a.q.data(), a.d.data(), a.hx.data()) == 0.f);
}

void test_q2e_multi() {
    testh::Rng rng(41);
    for (uint32_t nt = 1; nt <= 8; ++nt) {
        const uint32_t nb = 7;
        std::vector<uint8_t> codes(nb * 16), sc(nb * 2);
        for (auto& v : codes) v = (uint8_t)rng.below(256);
        for (uint32_t b = 0; b < nb; ++b) { uint16_t h = f16::float_to_half((float)(0.05 + rng.uni())); std::memcpy(&sc[2 * b], &h, 2); }
        std::vector<int8_t> q((size_t)nt * nb * 64);
        std::vector<float> d((size_t)nt * nb * 2);
        std::vector<int32_t> hx((size_t)nt * nb * 2);
        for (uint32_t t = 0; t < nt; ++t) {
            std::vector<float> x(nb * 64);
            for (auto& v : x) v = (float)rng.gauss();
            act_quant(x.data(), nb * 64, q.data() + (size_t)t * nb * 64, d.data() + (size_t)t * nb * 2, hx.data() + (size_t)t * nb * 2);
        }
        float out[8];
        row_dot_multi(codes.data(), sc.data(), nb, nt, q.data(), nb * 64, d.data(), hx.data(), nb * 2, out);
        for (uint32_t t = 0; t < nt; ++t) {
            const float single = row_dot(codes.data(), sc.data(), nb, q.data() + (size_t)t * nb * 64, d.data() + (size_t)t * nb * 2, hx.data() + (size_t)t * nb * 2);
            CHECK(std::memcmp(&single, &out[t], 4) == 0);
        }
    }
}

void test_q2e_dispatch_masks() {
    for (uint32_t nb = 0; nb <= 8; ++nb) {
        CHECK(tail_mask(nb) < 256);
        CHECK(__builtin_popcount(tail_mask(nb)) == (int)nb);
        // mask bits are the low nb bits
        for (uint32_t i = 0; i < 8; ++i) CHECK(((tail_mask(nb) >> i) & 1u) == (i < nb ? 1u : 0u));
    }
    CHECK(tail_mask(9) == 0xFF && tail_mask(100) == 0xFF);
    for (uint32_t nblocks = 1; nblocks <= 60; ++nblocks) {
        uint32_t covered = 0;
        for (uint32_t b0 = 0; b0 < nblocks; b0 += group_size(nblocks, b0)) {
            const uint32_t g = group_size(nblocks, b0);
            CHECK(g >= 1 && g <= 8 && b0 + g <= nblocks);
            covered += g;
        }
        CHECK(covered == nblocks);
    }
    for (uint32_t n = 0; n < 40; ++n) {
        CHECK(falls_back(n) == !(n >= 1 && n <= 8));
        if (n >= 1 && n <= 8) CHECK(dispatch_nt(n) == n);
        if (n > 8) CHECK(dispatch_nt(n) == 8);
    }
}

void test_q2e_cpu() {
    for (int m = 0; m < 32; ++m) {
        CpuFeatures f;
        f.avx512f = m & 1; f.avx512bw = m & 2; f.avx512vl = m & 4; f.avx512_vnni = m & 8; f.avx512_vbmi = m & 16;
        CHECK(f.usable() == (m == 31));
        // round trip through cpuid register bits
        uint32_t ebx = (f.avx512f ? 1u << 16 : 0) | (f.avx512bw ? 1u << 30 : 0) | (f.avx512vl ? 1u << 31 : 0);
        uint32_t ecx = (f.avx512_vnni ? 1u << 11 : 0) | (f.avx512_vbmi ? 1u << 1 : 0);
        const CpuFeatures g = detect(7, ebx, ecx);
        CHECK(g.avx512f == f.avx512f && g.avx512bw == f.avx512bw && g.avx512vl == f.avx512vl);
        CHECK(g.avx512_vnni == f.avx512_vnni && g.avx512_vbmi == f.avx512_vbmi);
        for (uint32_t leaf = 0; leaf < 7; ++leaf) CHECK(!detect(leaf, 0xFFFFFFFFu, 0xFFFFFFFFu).usable());
    }
    for (uint32_t i = 0; i < 32; ++i) CHECK(!bit(0, i));
    CHECK(detect(7, 0xFFFFFFFFu, 0xFFFFFFFFu).usable());
    CHECK(!detect(7, 0, 0).usable());
    const CpuFeatures host = detect_host();
    (void)host.usable();  // must not crash
}

void test_q2e_forward() {
    const std::vector<uint8_t> blob = make_blob(51);
    testh::Rng rng(52);
    const uint32_t NT = 10;
    std::vector<float> x((size_t)NT * H);
    for (auto& v : x) v = (float)rng.gauss();

    // float reference with decoded weights and unquantised activations (token 0)
    std::vector<float> act(FF);
    auto dotw = [&](const uint8_t* codes, const uint8_t* sc, uint32_t nb, const float* in) {
        double s = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            const double wd = f16::half_to_float(load_u16(sc + 2 * b));
            for (uint32_t j = 0; j < 64; ++j) s += (block_code(codes, b, j) - 1.0) * wd * in[b * 64 + j];
        }
        return s;
    };
    for (uint32_t r = 0; r < FF; ++r) {
        const double g = dotw(gu_codes(blob.data(), gate_slot(r)), gu_scales(blob.data(), gate_slot(r)), SC_GU, x.data());
        const double u = dotw(gu_codes(blob.data(), up_slot(r)), gu_scales(blob.data(), up_slot(r)), SC_GU, x.data());
        act[r] = (float)(g / (1.0 + std::exp(-g)) * u);
    }
    std::vector<float> ref(H), out(H);
    for (uint32_t r = 0; r < H; ++r) ref[r] = (float)dotw(down_codes(blob.data(), r), down_scales(blob.data(), r), SC_D, act.data());
    expert_forward(blob.data(), x.data(), out.data());
    double num = 0, den = 0;
    for (uint32_t r = 0; r < H; ++r) { num += (double)(out[r] - ref[r]) * (out[r] - ref[r]); den += (double)ref[r] * ref[r]; }
    CHECK_MSG(std::sqrt(num / (den + 1e-30)) < 0.05, "relative error %g", std::sqrt(num / (den + 1e-30)));

    // multi-token path == token-by-token path, bit for bit, for 1..8 (fast) and 9,10 (fallback)
    std::vector<float> single((size_t)NT * H);
    for (uint32_t t = 0; t < NT; ++t) expert_forward(blob.data(), x.data() + (size_t)t * H, single.data() + (size_t)t * H);
    for (uint32_t nt : {1u, 2u, 3u, 5u, 8u, 9u, 10u}) {
        std::vector<float> multi((size_t)nt * H, -1.f);
        expert_forward_multi(blob.data(), x.data(), nt, multi.data());
        CHECK(std::memcmp(multi.data(), single.data(), (size_t)nt * H * sizeof(float)) == 0);
    }
}
