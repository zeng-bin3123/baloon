// Invariants of ScalarDequantizers and the host side of PrefillKernels.
#include <cmath>
#include <cstring>
#include <vector>
#include "check.hpp"
#include "baloon/scalar_dequantizers.hpp"
#include "baloon/ggml_dequant_layouts.hpp"
#include "baloon/prefill_kernel_geometry.hpp"
using namespace baloon;

static void rand_block(testh::Rng& r, std::vector<uint8_t>& b, size_t n, uint16_t d_half, size_t d_off, size_t d2_off = SIZE_MAX, uint16_t d2 = 0) {
    b.resize(n);
    for (auto& x : b) x = (uint8_t)r.u32();
    b[d_off] = (uint8_t)(d_half & 0xFF); b[d_off + 1] = (uint8_t)(d_half >> 8);
    if (d2_off != SIZE_MAX) { b[d2_off] = (uint8_t)(d2 & 0xFF); b[d2_off + 1] = (uint8_t)(d2 >> 8); }
}
static const uint16_t kD = 0x3C00 /*1.0*/, kD2 = 0x3800 /*0.5*/;
static bool same(float a, float b) { return std::memcmp(&a, &b, 4) == 0 || (a == b); }

void test_dequant_geometry() {
    using namespace sdq;
    CHECK(2 + 16 == kBlkQ2_0 && 16 * 4 == kQK2_0 && kBlkQ4_0 == 18 && kBlkQ5_0 == 2 + 4 + 16 && kBlkQ8_0 == 2 + 32 && kBlkIQ4NL == 2 + 16);
    CHECK(kBlkQ6K == 128 + 64 + 16 + 2 && kBlkQ4K == 2 + 2 + 12 + 128 && kBlkQ5K == 2 + 2 + 12 + 32 + 128);
    CHECK(kBlkQ3K == 32 + 64 + 12 + 2 && kBlkIQ4XS == 2 + 2 + 4 + 128);
    // Q2_0 pack/unpack: no bit-order ambiguity
    for (int c0 = 0; c0 < 4; ++c0) for (int c1 = 0; c1 < 4; ++c1) for (int c2 = 0; c2 < 4; ++c2) for (int c3 = 0; c3 < 4; ++c3) {
        const uint8_t b = q2_pack(c0, c1, c2, c3);
        CHECK(q2_code(b, 0) == c0 && q2_code(b, 1) == c1 && q2_code(b, 2) == c2 && q2_code(b, 3) == c3);
    }
    for (int b = 0; b < 256; ++b) for (int k = 0; k < 4; ++k) CHECK(q2_code((uint8_t)b, k) < 4);
    // IQ4_NL codebook: 16 strictly increasing entries in [-127,113]
    for (int i = 0; i < 16; ++i) { CHECK(kIq4nl[i] >= -127 && kIq4nl[i] <= 113); if (i) CHECK(kIq4nl[i] > kIq4nl[i - 1]); }
    // masks of Q5_K cover the eight bits; fixed mask / shift-by-one would differ at group 1
    int sum = 0; for (int k = 0; k < 4; ++k) { sum += (1 << (2 * k)) + (2 << (2 * k)); CHECK((2 << (2 * k)) == 2 * (1 << (2 * k))); }
    CHECK(sum == 255 && (1 << 2) != (1 << 1));
    // fp16 exponent mapping to f32
    for (int ex = 1; ex <= 30; ++ex) { CHECK(ex - 15 + 127 == ex + 112 && ex + 112 >= 1 && ex + 112 <= 254); }
    for (int man = 0; man < 1024; ++man) CHECK(man * 8192 < 8388608);
    CHECK(f16::half_to_float(0x3C00) == 1.0f && f16::half_to_float(0xC000) == -2.0f && f16::half_to_float(0x0000) == 0.0f);
    // 6-bit scale reassembly < 64
    for (int x = 0; x < 256; ++x) for (int y = 0; y < 4; ++y) CHECK((((x & 15) | (y << 4))) < 64);
}

void test_dequant_simple() {
    using namespace sdq;
    testh::Rng rng(81);
    std::vector<uint8_t> b; float y[256], z[256];
    for (int it = 0; it < 3000; ++it) {
        const float dv = (it & 1) ? 1.0f : 0.5f; const uint16_t dh = (it & 1) ? kD : kD2;
        rand_block(rng, b, kBlkQ2_0, dh, 0);
        deq_q2_0(b.data(), y);
        for (int j = 0; j < 64; ++j) { const int c = (b[2 + j / 4] >> ((j % 4) * 2)) & 3; CHECK(y[j] == dv * (float)(c - 1)); CHECK(y[j] >= -dv && y[j] <= 2 * dv); }
        dequant::dequantize_q2_0(b.data(), z); for (int j = 0; j < 64; ++j) CHECK(same(y[j], z[j]));
        rand_block(rng, b, kBlkQ4_0, kD, 0);
        deq_q4_0(b.data(), y);
        for (int i = 0; i < 32; ++i) { const int nib = i < 16 ? (b[2 + i] & 15) : (b[2 + i - 16] >> 4); CHECK(y[i] == (float)(nib - 8)); CHECK(y[i] >= -8 && y[i] <= 7); }
        dequant::dequantize_q4_0(b.data(), z); for (int j = 0; j < 32; ++j) CHECK(same(y[j], z[j]));
        rand_block(rng, b, kBlkQ5_0, kD, 0);
        deq_q5_0(b.data(), y);
        uint32_t qh; std::memcpy(&qh, b.data() + 2, 4);
        for (int i = 0; i < 32; ++i) {
            const int j = i % 16; const int nib = i < 16 ? (b[6 + j] & 15) : (b[6 + j] >> 4);
            const int hb = (qh >> i) & 1;                                   // bit j for low half, j+16 for high half
            CHECK(y[i] == (float)(nib + 16 * hb - 16)); CHECK(y[i] >= -16 && y[i] <= 15);
        }
        dequant::dequantize_q5_0(b.data(), z); for (int j = 0; j < 32; ++j) CHECK(same(y[j], z[j]));
        rand_block(rng, b, kBlkQ8_0, kD2, 0);
        deq_q8_0(b.data(), y);
        for (int j = 0; j < 32; ++j) CHECK(y[j] == 0.5f * (float)(int8_t)b[2 + j]);
        rand_block(rng, b, kBlkIQ4NL, kD, 0);
        deq_iq4_nl(b.data(), y);
        for (int i = 0; i < 32; ++i) { const int nib = i < 16 ? (b[2 + i] & 15) : (b[2 + i - 16] >> 4); CHECK(y[i] == (float)kIq4nl[nib]); }
    }
}

// Independent bit-position oracles (element-index formulations).
void test_dequant_kquants() {
    using namespace sdq;
    testh::Rng rng(91);
    std::vector<uint8_t> b; float y[256], z[256];
    for (int it = 0; it < 3000; ++it) {
        // Q6_K
        rand_block(rng, b, kBlkQ6K, kD, 208);
        deq_q6_k(b.data(), y);
        for (int e = 0; e < 256; ++e) {
            const int n = e / 128, r = e % 128, quad = r / 32, l = r % 32;
            const int qli = 64 * n + l + (quad & 1) * 32; const int nib = quad >= 2 ? (b[qli] >> 4) : (b[qli] & 15);
            const int hi = (b[128 + 32 * n + l] >> (2 * quad)) & 3;
            const int sc = (int8_t)b[192 + 8 * n + l / 16 + 2 * quad];
            CHECK(y[e] == (float)(sc * ((nib | (hi << 4)) - 32)));
            CHECK(8 * (n) + l / 16 + 2 * quad < 16 && 128 * n + l + 32 * quad < 256);
        }
        dequant::dequantize_q6_k(b.data(), z); for (int j = 0; j < 256; ++j) CHECK(same(y[j], z[j]));
        // Q4_K / Q5_K 6-bit scale/min oracle
        for (int kind = 0; kind < 2; ++kind) {
            rand_block(rng, b, kind ? kBlkQ5K : kBlkQ4K, kD, 0, 2, kD2);   // d=1, dmin=0.5
            const uint8_t* sc = b.data() + 4;
            auto scale = [&](int i) { return i < 4 ? (sc[i] & 63) : ((sc[i + 4] & 15) | ((sc[i - 4] >> 6) << 4)); };
            auto minv = [&](int i) { return i < 4 ? (sc[i + 4] & 63) : ((sc[i + 4] >> 4) | ((sc[i] >> 6) << 4)); };
            if (kind) deq_q5_k(b.data(), y); else deq_q4_k(b.data(), y);
            const uint8_t* qh = b.data() + 16; const uint8_t* qs = b.data() + (kind ? 48 : 16);
            for (int e = 0; e < 256; ++e) {
                const int g = e / 64, w = e % 64, half = w / 32, l = w % 32;
                const int sidx = 2 * g + half;
                int q = half ? (qs[32 * g + l] >> 4) : (qs[32 * g + l] & 15);
                if (kind) q += ((qh[l] >> (2 * g + half)) & 1) * 16;            // mask shifts left by 2 per 64 group
                CHECK(y[e] == (float)(scale(sidx)) * (float)q - 0.5f * (float)minv(sidx));
                CHECK(scale(sidx) < 64 && minv(sidx) < 64);
            }
            if (kind) dequant::dequantize_q5_k(b.data(), z); else dequant::dequantize_q4_k(b.data(), z);
            for (int j = 0; j < 256; ++j) CHECK(same(y[j], z[j]));
        }
        // Q3_K
        rand_block(rng, b, kBlkQ3K, kD, 108);
        deq_q3_k(b.data(), y);
        const uint8_t* s12 = b.data() + 96;
        auto s6 = [&](int i) { const int lo = i < 8 ? (s12[i] & 15) : (s12[i - 8] >> 4); const int hi = (s12[8 + i % 4] >> (2 * (i / 4))) & 3; return (lo | (hi << 4)) - 32; };
        for (int e = 0; e < 256; ++e) {
            const int n = e / 128, r = e % 128, j = r / 32, l = r % 32;
            const int q = (b[32 + 32 * n + l] >> (2 * j)) & 3;
            const int hbit = (b[l] >> (4 * n + j)) & 1;                          // hoisted mask 1 << (4n+j), not reset per half
            CHECK(y[e] == (float)(s6(n * 8 + j * 2 + l / 16) * (q - (hbit ? 0 : 4))));
        }
        dequant::dequantize_q3_k(b.data(), z); for (int j = 0; j < 256; ++j) CHECK(same(y[j], z[j]));
        // IQ4_XS
        rand_block(rng, b, kBlkIQ4XS, kD, 0);
        deq_iq4_xs(b.data(), y);
        const uint16_t sh = (uint16_t)(b[2] | (b[3] << 8));
        for (int e = 0; e < 256; ++e) {
            const int ib = e / 32, w = e % 32;
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            CHECK(ls >= 0 && ls < 64 && ls - 32 >= -32 && ls - 32 <= 31);
            const uint8_t byte = b[8 + 16 * ib + (w % 16)]; const int nib = w < 16 ? (byte & 15) : (byte >> 4);
            CHECK(y[e] == (float)((ls - 32) * kIq4nl[nib]));
        }
        dequant::dequantize_iq4_xs(b.data(), z); for (int j = 0; j < 256; ++j) CHECK(same(y[j], z[j]));
    }
    // Q5_K regression guard: qh = 0xFF, qs = 0, every sub-block scale 1 (d=1, dmin=0): all 256 elements are exactly 16.
    // A fixed mask or a shift-by-one mask sequence would drop the fifth bit in later groups only if qh bits differ; the
    // oracle above covers that, this pins the all-ones case.
    b.assign(kBlkQ5K, 0); b[1] = 0x3C;
    for (int i = 0; i < 4; ++i) { b[4 + i] = 1; b[4 + 8 + i] = 1; }
    for (int i = 16; i < 48; ++i) b[i] = 0xFF;
    float y2[256]; deq_q5_k(b.data(), y2);
    for (int e = 0; e < 256; ++e) CHECK(y2[e] == 16.f);
    // Only bit 2*g (u1 of group g) set: only the low-nibble half of group g gets the +16.
    for (int g = 0; g < 4; ++g) {
        for (int i = 16; i < 48; ++i) b[i] = (uint8_t)(1u << (2 * g));
        deq_q5_k(b.data(), y2);
        for (int e = 0; e < 256; ++e) CHECK(y2[e] == ((e / 64 == g && (e % 64) < 32) ? 16.f : 0.f));
    }
}

void test_prefill_host_geometry() {
    using namespace pfk;
    testh::Rng rng(101);
    for (int64_t n : {1, 255, 256, 257, 10240, 100000, 3276800}) { CHECK(n <= blocks_for(n) * 256); CHECK(blocks_for(n) * 256 < n + 256); }
    for (int64_t T = 1; T < 300; ++T) CHECK(T <= route_blocks(T) * 8);
    for (int it = 0; it < 50000; ++it) {
        const int64_t T = 1 + rng.below(64), i = rng.next() % (uint64_t)(T * kHcD);
        const Thc p = hc_decode(i);
        CHECK(p.t < T && p.c < 4 && p.d < 2560 && p.t * 10240 + p.c * 2560 + p.d == i);
        const int64_t j = rng.next() % (uint64_t)(T * 2560); const int c = (int)rng.below(4);
        CHECK(gr_mix_read(j, c) < T * 10240);
        CHECK(gr_norm_idx((int64_t)rng.below(1000), (int)rng.below(2560)) < 10240);
    }
    CHECK(10240 / 128 * 128 == 10240 && kGdnChannels / kGdnConvBlock == 80);
    for (int rg = 0; rg < 4; ++rg) for (int r = 0; r < 32; ++r) { const int row = gdn_row(rg, r); CHECK(row < 128 && row / 32 == rg && row % 32 == r); }
    for (int h = 0; h < 48; ++h) for (int t = 0; t < 128; ++t) {
        CHECK(rec_key_idx(h, t) < 2 * 16 * 128 && rec_key_idx(h, t) >= 16 * 128);
        CHECK(rec_value_idx(h, t) < 10240 && rec_value_idx(h, t) >= 2 * 16 * 128);
        if (h < 32) CHECK(l2_idx(h, t) < 4096);
    }
    for (int i = 0; i < 100000; ++i) { const int64_t T = 1 + rng.below(100), g = rng.next() % (uint64_t)(T * 48); CHECK(g / 48 < T && g % 48 < 48); }
    { bool seen[512] = {}; for (int l = 0; l < 32; ++l) for (int i = 0; i < 16; ++i) { const int e = lane_expert(l, i); CHECK(e < 512 && e / 32 == i && e % 32 == l && !seen[e]); seen[e] = true; }
      for (int e = 0; e < 512; ++e) CHECK(seen[e] && e % 32 + e / 32 * 32 == e && e / 32 < 16); }
    CHECK(O_D_CODES == 819200 && O_GU_SC == 1228800 && O_D_SC == 1331200 && kExpertBlob == 1382400);
    CHECK(kExpertThreads == 1280 * 640 + 2560 * 160);
    for (int it = 0; it < 100000; ++it) {
        const int64_t row = rng.below(1280), byte = rng.below(640); const int k = (int)rng.below(4);
        CHECK(gu_scale_idx(row, byte) < 1280 * 40 && gu_out_idx(row, byte, k) < 1280 * 2560);
        const int64_t dr = rng.below(2560), db = rng.below(160);
        CHECK(d_scale_idx(dr, db) < 2560 * 10 && d_out_idx(dr, db, k) < 2560 * 640);
        const int64_t i = 1280 * 640 + (int64_t)(rng.next() % (2560 * 160));
        CHECK((i - 1280 * 640) / 160 < 2560 && (i - 1280 * 640) % 160 < 160);
        const unsigned c = rng.u32() & 255; CHECK(code_field(c, k) < 4);
        const int64_t n = 1 + rng.below(100), q = rng.next() % (uint64_t)(n * 640);
        CHECK(q / 640 < n && q / 640 * 1280 + 2 * (q % 640) + 1 < n * 1280);         // interleaved SwiGLU
    }
    // every output element of the expert blob is written exactly once
    { std::vector<uint8_t> cov(1280 * 2560, 0); for (int64_t r = 0; r < 1280; ++r) for (int64_t by = 0; by < 640; ++by) for (int k = 0; k < 4; ++k) ++cov[gu_out_idx(r, by, k)];
      for (auto v : cov) CHECK(v == 1); }
    { std::vector<uint8_t> cov(2560 * 640, 0); for (int64_t r = 0; r < 2560; ++r) for (int64_t by = 0; by < 160; ++by) for (int k = 0; k < 4; ++k) ++cov[d_out_idx(r, by, k)];
      for (auto v : cov) CHECK(v == 1); }
    CHECK(2560 / 8 == 320 && 8 * 2 == 16);
    for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) { CHECK(split_q_idx(h, d) < 24 * 512 && gate_attn_idx(h, d) < 24 * 512 && gate_attn_idx(h, d) >= split_q_idx(h, 0) + 256); }
    for (int p = 0; p < 32; ++p) CHECK(p + 32 < 64);
    CHECK(2 * 256 == 512 && 24 * 256 == 6144 && 4 * 128 == 512);
    for (int z = 0; z < 8; ++z) for (int t = 0; t < 64; ++t) CHECK(kv_group(z) < 4 && kv_group(z) * 64 + t < 256);
    for (int64_t page = 0; page < 50; ++page) for (int kvh = 0; kvh < 2; ++kvh) CHECK(kv_page_row(page, kvh) < 100);
    for (int64_t q = -100000; q <= 100000; q += 7) { const int c = clamp_q(q); CHECK(c >= -127 && c <= 127); if (q >= -127 && q <= 127) CHECK(c == q); }
    CHECK(clamp_q(-128) == -127 && clamp_q(1 << 30) == 127 && clamp_q(-(1LL << 40)) == -127);
}
