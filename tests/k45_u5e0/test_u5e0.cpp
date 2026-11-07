// Unit u5e0736f9b8 tests: LAYER 1 statements of IqAvx512Rows, Q2Avx2Rows and IqQuantizedDotKernels, checked on the
// production headers and on the production kernels (src/cpu/iq_rows_avx512.cpp built with the AVX-512 reference ISA flags,
// src/cpu/q2_avx2_rows.cpp built with -mavx2 -mf16c -mfma).
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../unit14/check.hpp"
#include "baloon/iq_dot_geometry.hpp"
#include "baloon/iq_quant_expert_layout.hpp"
#include "baloon/iquant_avx512_rows.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/kern_geometry.hpp"
#include "baloon/q2_avx2_rows.hpp"

using namespace baloon;
using testh::Rng;

static const uint32_t kIq5[5] = {16, 17, 18, 21, 22};

// ============================================================================================ IqAvx512Rows
static void test_iq512_layout() {
    for (uint32_t ty = 0; ty < 64; ++ty) {
        const bool want = ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22;
        CHECK(iquant512::supported(ty) == want);
        CHECK(kern::iq512_supported(ty) == want);
        CHECK(kern::iqgeo::supported(ty) == want);
        CHECK((iquant512::bytes_of(ty) > 0) == want);
        CHECK(kern::iqgeo::block_bytes(ty) == iquant512::bytes_of(ty));
    }
    CHECK(!iquant512::supported(29));   // IQ1_M stays on ggml-cpu
    CHECK(!iquant512::supported(42));   // Q2_0 is not an i-quant
    CHECK(iquant512::bytes_of(16) == 66 && iquant512::bytes_of(17) == 74 && iquant512::bytes_of(18) == 98 &&
          iquant512::bytes_of(21) == 110 && iquant512::bytes_of(22) == 82);
    CHECK(iquant512::T_IQ2_XXS == 16 && iquant512::T_IQ2_XS == 17 && iquant512::T_IQ3_XXS == 18 && iquant512::T_IQ3_S == 21 &&
          iquant512::T_IQ2_S == 22);
    CHECK(4 * 64 == (int)iquant512::QK_K);
    CHECK(2 + 64 == (int)iquant512::bytes_of(16));
    CHECK(2 + 64 + 8 == (int)iquant512::bytes_of(17));
    CHECK(2 + 64 + 32 == (int)iquant512::bytes_of(18));
    CHECK(2 + 64 + 8 + 32 + 4 == (int)iquant512::bytes_of(21));
    CHECK(2 + 32 + 32 + 8 + 8 == (int)iquant512::bytes_of(22));
    CHECK(iquant512::n_blocks(2048) == 8);
}

static void test_iq512_reads() {
    using iquant512::chunk_fields;
    for (uint32_t ty : kIq5) {
        const uint32_t bytes = iquant512::bytes_of(ty);
        for (uint32_t j = 0; j < 4; ++j) {
            const iquant512::ChunkFields c = chunk_fields(ty, j);
            for (uint32_t a = 0; a < iquant512::F_COUNT; ++a) {
                if (c.f[a].len == 0) continue;
                CHECK(c.f[a].offset >= 2);                                   // never inside the fp16 scale
                CHECK(c.f[a].offset + c.f[a].len <= bytes);                  // inside the block
                CHECK(kern::iqgeo::chunk_read_end(ty, j) <= bytes);
                for (uint32_t b = a + 1; b < iquant512::F_COUNT; ++b) {      // fields of one chunk do not overlap
                    if (c.f[b].len == 0) continue;
                    const bool disjoint = c.f[a].offset + c.f[a].len <= c.f[b].offset || c.f[b].offset + c.f[b].len <= c.f[a].offset;
                    CHECK(disjoint);
                }
                for (uint32_t j2 = j + 1; j2 < 4; ++j2) {                    // the same field of two chunks does not overlap
                    const iquant512::ChunkFields c2 = chunk_fields(ty, j2);
                    CHECK(c.f[a].offset + c.f[a].len <= c2.f[a].offset);
                }
            }
        }
    }
    for (uint32_t j = 0; j < 4; ++j) {   // exact offsets of the theory
        auto f16 = iquant512::chunk_fields(16, j), f17 = iquant512::chunk_fields(17, j), f22 = iquant512::chunk_fields(22, j),
             f18 = iquant512::chunk_fields(18, j), f21 = iquant512::chunk_fields(21, j);
        CHECK(f16.f[iquant512::F_GRID].offset == 2 + 16 * j && f16.f[iquant512::F_GRID].len == 16);
        CHECK(f17.f[iquant512::F_GRID].offset == 2 + 16 * j && f17.f[iquant512::F_SCALES].offset == 66 + 2 * j &&
              f17.f[iquant512::F_SCALES].len == 2);
        CHECK(f22.f[iquant512::F_GRID].offset == 2 + 8 * j && f22.f[iquant512::F_GRID].len == 8);
        CHECK(f22.f[iquant512::F_SIGNS].offset == 34 + 8 * j && f22.f[iquant512::F_SIGNS].len == 8);
        CHECK(f22.f[iquant512::F_QH].offset == 66 + 2 * j && f22.f[iquant512::F_SCALES].offset == 74 + 2 * j);
        CHECK(f18.f[iquant512::F_GRID].offset == 2 + 16 * j && f18.f[iquant512::F_SIGNSCALE].offset == 66 + 8 * j &&
              f18.f[iquant512::F_SIGNSCALE].len == 8);
        CHECK(f21.f[iquant512::F_QH].offset == 66 + 2 * j && f21.f[iquant512::F_SIGNS].offset == 74 + 8 * j &&
              f21.f[iquant512::F_SCALES].offset == 106 + j && f21.f[iquant512::F_SCALES].len == 1);
    }
    CHECK(74 + 32 == 106);   // IQ3_S sign region ends where the scale bytes begin
    CHECK(34 + 32 == 66);    // IQ2_S sign region ends where qh begins
}

static void test_iq512_signs_scales_batches() {
    for (int y = -127; y <= 127; ++y) {
        for (int b = 0; b < 2; ++b) {
            const int n = iquant512::apply_sign(b != 0, y);
            CHECK(n >= -127 && n <= 127);
            CHECK(iquant512::apply_sign(b != 0, n) == y);
            CHECK(kern::iqgeo::apply_sign(b != 0, y) == n);
        }
        CHECK(iquant512::apply_sign(false, y) == y);
        CHECK(iquant512::apply_sign(true, y) == -y);
    }
    for (uint32_t l = 0; l < 32; ++l) {
        CHECK(iquant512::scale_of_lane(l) < 4);
        CHECK(kern::iqgeo::scale_lane(l) == iquant512::scale_of_lane(l));
        for (uint32_t s = 0; s < 4; ++s) CHECK((iquant512::scale_of_lane(l) == s) == (8 * s <= l && l < 8 * s + 8));
    }
    for (uint32_t s4 = 0; s4 < 16; ++s4) CHECK(iquant512::odd_scale(s4) % 2 == 1 && iquant512::odd_scale(s4) <= 31);
    for (uint32_t nt = 0; nt <= 100; ++nt) {   // groups of at most four cover each token once
        uint32_t covered = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += iquant512::batch_size(nt, t0)) {
            const uint32_t g = iquant512::batch_size(nt, t0);
            CHECK(g >= 1 && g <= 4);
            CHECK(g == kern::iqgeo::group_size(nt, t0));
            CHECK(t0 == covered);
            covered += g;
        }
        CHECK(covered == nt);
    }
    // int16 pair sums of the vector multiply-add cannot saturate for the magnitudes used by the grids
    const int mags2[3] = {8, 25, 43};
    for (int a0 : mags2) for (int a1 : mags2)
        for (int b0 = -127; b0 <= 127; ++b0)
            for (int b1 = -127; b1 <= 127; ++b1)
                CHECK(!iquant512::maddubs_would_saturate((uint8_t)a0, (int8_t)b0, (uint8_t)a1, (int8_t)b1));
    CHECK(!iquant512::maddubs_would_saturate(62, 127, 62, 127) && !iquant512::maddubs_would_saturate(62, -127, 62, -127));
}


// Independent scalar decoder of the five i-quant block formats (llama.cpp block layouts), used as the reference for the
// AVX-512 kernels. Returns d_w * d_act * sum_i m_i * mag_i * sign_i * q_i with m_i the per-format scale multiplier.
struct RefTables { const uint64_t* g2xxs; const uint64_t* g2xs; const uint64_t* g2s; const uint32_t* g3xxs; const uint32_t* g3s; };
static uint8_t ref_ksign(uint32_t v7) { return (uint8_t)(v7 | ((__builtin_popcount(v7) & 1u) << 7)); }
static double ref_row_dot(const RefTables& T, uint32_t ty, const uint8_t* row, uint32_t nb, const kern::Q8KBlock* act) {
    double total = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t)b * iquant512::bytes_of(ty);
        uint16_t h; std::memcpy(&h, blk, 2);
        const double d = (double)_cvtsh_ss(h) * (double)act[b].d;
        long isum[8][4] = {};   // [ib32][l] sums of mag*sign*q (before scaling)
        double mult[8][4];      // scale multiplier per (ib32, l)
        for (uint32_t ib = 0; ib < 8; ++ib) for (uint32_t l = 0; l < 4; ++l) {
            uint8_t mag[8]; uint8_t sg; double m;
            if (ty == 16) {
                const uint8_t* qs = blk + 2 + 8 * ib; uint32_t a1; std::memcpy(&a1, qs + 4, 4);
                const uint64_t g = T.g2xxs[qs[l]]; std::memcpy(mag, &g, 8);
                sg = ref_ksign((a1 >> (7 * l)) & 127u); m = (2.0 * (a1 >> 28) + 1) / 8.0;
            } else if (ty == 17) {
                uint16_t qv; std::memcpy(&qv, blk + 2 + 8 * ib + 2 * l, 2);
                const uint64_t g = T.g2xs[qv & 511u]; std::memcpy(mag, &g, 8);
                sg = ref_ksign(qv >> 9); const uint8_t sc = blk[66 + ib];
                m = (2.0 * (l < 2 ? (sc & 0xf) : (sc >> 4)) + 1) / 8.0;
            } else if (ty == 22) {
                const uint8_t* qs = blk + 2; const uint8_t* sgn = blk + 34; const uint8_t* qh = blk + 66; const uint8_t sc = blk[74 + ib];
                const uint32_t idx = qs[4 * ib + l] | ((uint32_t)(qh[ib] << (8 - 2 * l)) & 0x300u);
                const uint64_t g = T.g2s[idx]; std::memcpy(mag, &g, 8);
                sg = sgn[4 * ib + l]; m = (2.0 * (l < 2 ? (sc & 0xf) : (sc >> 4)) + 1) / 8.0;
            } else if (ty == 18) {
                const uint8_t* qs = blk + 2 + 8 * ib; uint32_t a; std::memcpy(&a, blk + 66 + 4 * ib, 4);
                const uint32_t g1 = T.g3xxs[qs[2 * l]], g2 = T.g3xxs[qs[2 * l + 1]]; std::memcpy(mag, &g1, 4); std::memcpy(mag + 4, &g2, 4);
                sg = ref_ksign((a >> (7 * l)) & 127u); m = (2.0 * (a >> 28) + 1) / 4.0;
            } else {   // 21
                const uint8_t* qs = blk + 2 + 8 * ib; const uint8_t qh = blk[66 + ib]; const uint8_t* sgn = blk + 74 + 4 * ib;
                const uint8_t sc = blk[106 + ib / 2];
                const uint32_t i1 = qs[2 * l] | ((uint32_t)(qh << (8 - 2 * l)) & 256u), i2 = qs[2 * l + 1] | ((uint32_t)(qh << (7 - 2 * l)) & 256u);
                const uint32_t g1 = T.g3s[i1], g2 = T.g3s[i2]; std::memcpy(mag, &g1, 4); std::memcpy(mag + 4, &g2, 4);
                sg = sgn[l]; m = 1.0 + 2.0 * ((ib & 1) ? (sc >> 4) : (sc & 0xf));
            }
            long a = 0;
            for (uint32_t k = 0; k < 8; ++k) { const int y = act[b].qs[ib * 32 + l * 8 + k]; a += (long)mag[k] * ((sg >> k) & 1 ? -y : y); }
            isum[ib][l] = a; mult[ib][l] = m;
        }
        double acc = 0;
        for (uint32_t ib = 0; ib < 8; ++ib) for (uint32_t l = 0; l < 4; ++l) acc += mult[ib][l] * (double)isum[ib][l];
        total += d * acc;
    }
    return total;
}

// Synthetic grid tables (bounded magnitudes): random weights, random blocks; the production kernels must agree with themselves
// across every token batching and must leave unsupported types / unwritten lanes untouched.
static void test_iq512_kernels() {
    Rng rng(0x5e0);
    static uint64_t g2xxs[256], g2xs[512], g2s[1024];
    static uint32_t g3xxs[256], g3s[512];
    const uint8_t m2[3] = {8, 25, 43};
    for (auto* g : {g2xxs}) for (int i = 0; i < 256; ++i) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)m2[rng.below(3)] << (8 * k); g[i] = v; }
    for (int i = 0; i < 512; ++i) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)m2[rng.below(3)] << (8 * k); g2xs[i] = v; }
    for (int i = 0; i < 1024; ++i) { uint64_t v = 0; for (int k = 0; k < 8; ++k) v |= (uint64_t)m2[rng.below(3)] << (8 * k); g2s[i] = v; }
    const uint8_t m3[4] = {4, 12, 20, 28};
    for (int i = 0; i < 256; ++i) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)m3[rng.below(4)] << (8 * k); g3xxs[i] = v; }
    for (int i = 0; i < 512; ++i) { uint32_t v = 0; for (int k = 0; k < 4; ++k) v |= (uint32_t)m3[rng.below(4)] << (8 * k); g3s[i] = v; }
    kern::IqTables t;
    t.iq2xxs_grid = g2xxs; t.iq2xs_grid = g2xs; t.iq2s_grid = g2s; t.iq3xxs_grid = g3xxs; t.iq3s_grid = g3s;
    kern::iq512_set_tables(t);
    for (uint32_t ty : kIq5) CHECK(kern::iq512_tables_ready(ty));
    CHECK(!kern::iq512_tables_ready(29) && !kern::iq512_tables_ready(42));

    const uint32_t nb = 3, n = nb * 256, rows = 5, max_nt = 9;
    std::vector<float> xs((size_t)max_nt * n);
    for (auto& v : xs) v = (float)rng.gauss();
    std::vector<kern::Q8KBlock> acts((size_t)max_nt * nb);
    for (uint32_t t = 0; t < max_nt; ++t) iquant512::q8k_quantize(xs.data() + (size_t)t * n, n, acts.data() + (size_t)t * nb);
    for (uint32_t ty : kIq5) {
        const size_t rb = (size_t)nb * iquant512::bytes_of(ty);
        std::vector<uint8_t> w((size_t)(2 * rows) * rb);
        for (auto& b : w) b = (uint8_t)rng.below(256);
        for (size_t r = 0; r < (size_t)2 * rows; ++r)       // sane fp16 block scales
            for (uint32_t b = 0; b < nb; ++b) { const uint16_t h = (uint16_t)_cvtss_sh(0.01f + 0.001f * (float)rng.below(50), 0); std::memcpy(&w[r * rb + b * iquant512::bytes_of(ty)], &h, 2); }
        const size_t os = rows + 3;
        std::vector<float> single((size_t)max_nt * os, -7.f);
        for (uint32_t t = 0; t < max_nt; ++t) kern::iq512_rows(ty, w.data(), rb, 0, rows, nb, acts.data() + (size_t)t * nb, 1, single.data() + (size_t)t * os, os);
        const RefTables RT{g2xxs, g2xs, g2s, g3xxs, g3s};
        for (uint32_t t = 0; t < max_nt; ++t) for (uint32_t r = 0; r < rows; ++r) {   // independent scalar reference
            const double want = ref_row_dot(RT, ty, w.data() + r * rb, nb, acts.data() + (size_t)t * nb);
            const double got = single[(size_t)t * os + r];
            CHECK(std::fabs(got - want) <= 2e-3 * (1.0 + std::fabs(want)));
        }
        bool nonzero = false;
        for (uint32_t t = 0; t < max_nt; ++t) for (uint32_t r = 0; r < rows; ++r) nonzero |= single[(size_t)t * os + r] != 0.f && std::isfinite(single[(size_t)t * os + r]);
        CHECK(nonzero);
        for (uint32_t nt = 1; nt <= max_nt; ++nt) {
            std::vector<float> multi((size_t)max_nt * os, -7.f);
            kern::iq512_rows(ty, w.data(), rb, 0, rows, nb, acts.data(), nt, multi.data(), os);
            for (uint32_t t = 0; t < max_nt; ++t)
                for (uint32_t r = 0; r < os; ++r) {
                    const float got = multi[(size_t)t * os + r];
                    if (t < nt && r < rows) CHECK(std::memcmp(&got, &single[(size_t)t * os + r], 4) == 0);   // same result for any batching
                    else CHECK(got == -7.f);                                                                    // nothing else is written
                }
        }
        // row sub-range [r0, r1) writes exactly those rows
        std::vector<float> part((size_t)4 * os, -7.f);
        kern::iq512_rows(ty, w.data(), rb, 1, 4, nb, acts.data(), 4, part.data(), os);
        for (uint32_t t = 0; t < 4; ++t) for (uint32_t r = 0; r < os; ++r) {
            const float got = part[(size_t)t * os + r];
            if (r >= 1 && r < 4) CHECK(std::memcmp(&got, &single[(size_t)t * os + r], 4) == 0); else CHECK(got == -7.f);
        }
        // gate/up: act = silu(gate) * up with gate row r, up row n_ff + r
        const uint32_t n_ff = rows;
        std::vector<float> gu((size_t)6 * os, -7.f);
        kern::iq512_gu_rows(ty, w.data(), rb, n_ff, 0, n_ff, nb, acts.data(), 6, gu.data(), os);
        std::vector<float> g((size_t)6 * os), u((size_t)6 * os);
        kern::iq512_rows(ty, w.data(), rb, 0, n_ff, nb, acts.data(), 6, g.data(), os);
        kern::iq512_rows(ty, w.data() + (size_t)n_ff * rb, rb, 0, n_ff, nb, acts.data(), 6, u.data(), os);
        for (uint32_t t = 0; t < 6; ++t) for (uint32_t r = 0; r < n_ff; ++r) {
            const float gg = g[(size_t)t * os + r], uu = u[(size_t)t * os + r];
            const float want = gg / (1.f + std::exp(-gg)) * uu;
            CHECK(std::fabs(gu[(size_t)t * os + r] - want) <= 1e-5f * (1.f + std::fabs(want)));
        }
    }
    float untouched[4] = {1.f, 2.f, 3.f, 4.f};
    kern::iq512_rows(29, nullptr, 0, 0, 1, 1, nullptr, 1, untouched, 1);
    kern::iq512_rows(42, nullptr, 0, 0, 1, 1, nullptr, 1, untouched, 1);
    CHECK(untouched[0] == 1.f && untouched[3] == 4.f);
}

// ============================================================================================== Q2Avx2Rows
static float h2f_(uint16_t h) { return _cvtsh_ss(h); }

static void test_q2_unpack_and_bounds() {
    CHECK(q2rows::built_with_required_isa());
    Rng rng(0x602);
    for (int trial = 0; trial < 4000; ++trial) {
        uint8_t codes[16], out[64];
        for (auto& c : codes) c = (uint8_t)rng.below(256);
        q2rows::unpack_codes_64(codes, out);
        for (uint32_t v = 0; v < 64; ++v) {
            const uint32_t want = (codes[v / 4] / (1u << (2 * (v % 4)))) % 4;
            CHECK(out[v] == want);
            CHECK(out[v] < 4);
            CHECK(q2rows::code_byte(v) == v / 4 && q2rows::code_shift(v) == 2 * (v % 4));
            if (v < 32) CHECK(q2rows::code_byte(v) < 8); else CHECK(q2rows::code_byte(v) >= 8 && q2rows::code_byte(v) < 16);
        }
        for (uint32_t i = 0; i < 16; ++i)   // lossless: four codes recombine to the byte
            CHECK((uint32_t)out[4 * i] + 4u * out[4 * i + 1] + 16u * out[4 * i + 2] + 64u * out[4 * i + 3] == codes[i]);
    }
    for (uint32_t b = 0; b < 256; ++b) {
        uint8_t codes[16], out[64];
        std::memset(codes, (int)b, 16);
        q2rows::unpack_codes_64(codes, out);
        CHECK(out[0] == b % 4 && out[1] == b / 4 % 4 && out[2] == b / 16 % 4 && out[3] == b / 64 % 4);
    }
    CHECK(16 * 4 == 64 && q2rows::BLOCK_BYTES == 18 && q2rows::BLOCK_VALUES == 64 && q2rows::CHUNK == 32 && q2rows::MAX_GROUP == 4);
    // the pair sum of two code*activation products stays within +-762
    int lo = 0, hi = 0;
    for (int c0 = 0; c0 <= 3; ++c0) for (int c1 = 0; c1 <= 3; ++c1)
        for (int q0 = -127; q0 <= 127; ++q0) for (int q1 = -127; q1 <= 127; ++q1) {
            const int s = c0 * q0 + c1 * q1;
            lo = std::min(lo, s); hi = std::max(hi, s);
        }
    CHECK(lo == -762 && hi == 762);
    // code-offset correction: centred dot == raw dot - activation sum
    Rng r2(0x603);
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 1 + (int)r2.below(64);
        long off = 0, raw = 0, sum = 0;
        for (int i = 0; i < n; ++i) { const int c = (int)r2.below(4), x = (int)r2.below(255) - 127; off += (long)(c - 1) * x; raw += (long)c * x; sum += x; }
        CHECK(off == raw - sum);
    }
    // reduction tree
    for (int trial = 0; trial < 500; ++trial) {
        long v[8]; for (auto& x : v) x = (long)r2.below(2000) - 1000;
        CHECK((((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))) == v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7]);
    }
}

static void test_q2_quantizer() {
    Rng rng(0x604);
    const uint32_t chunks = 16, n = chunks * 32;
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<float> x(n);
        const float amp = trial % 7 == 0 ? 0.f : (float)std::exp(rng.gauss() * 2.0);
        for (auto& v : x) v = (trial % 11 == 0 ? 1.f : (float)rng.gauss()) * amp;
        if (trial % 5 == 0) x[rng.below(n)] *= 50.f;
        std::vector<int8_t> q(n); std::vector<float> d(chunks); std::vector<int32_t> hx(chunks);
        q2rows::activation_quantizer(x.data(), n, q.data(), d.data(), hx.data());
        for (uint32_t c = 0; c < chunks; ++c) {
            float amax = 0.f; for (uint32_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[c * 32 + i]));
            const float dd = amax / 127.f, id = dd != 0.f ? 1.f / dd : 0.f;
            CHECK(d[c] == dd);
            int32_t s = 0;
            for (uint32_t i = 0; i < 32; ++i) {
                const float tt = x[c * 32 + i] * id;
                float tr = std::trunc(tt); const float fr = tt - tr;
                if (std::fabs(fr) >= 0.5f) tr += tt < 0 ? -1.f : 1.f;
                const int want = (int)std::max(-127.f, std::min(127.f, tr));
                CHECK(q[c * 32 + i] == want);                         // truncation after +-0.5, then clamp
                CHECK(q[c * 32 + i] != -128); // -128 never appears
                s += q[c * 32 + i];
            }
            CHECK(hx[c] == s);
        }
    }
}

static void test_q2_rows() {
    Rng rng(0x605);
    const uint32_t nb = 5, rows = 6, max_nt = 11, n = nb * 64, chunks = nb * 2;
    std::vector<uint8_t> w((size_t)rows * nb * 18);
    for (auto& b : w) b = (uint8_t)rng.below(256);
    for (size_t blk = 0; blk < (size_t)rows * nb; ++blk) { const uint16_t h = (uint16_t)_cvtss_sh(0.02f + 0.003f * (float)rng.below(40), 0); std::memcpy(&w[blk * 18], &h, 2); }
    const size_t q_stride = n + 64, ch_stride = chunks + 3;
    std::vector<int8_t> q((size_t)max_nt * q_stride, 0); std::vector<float> xd((size_t)max_nt * ch_stride, 0.f); std::vector<int32_t> hx((size_t)max_nt * ch_stride, 0);
    for (uint32_t t = 0; t < max_nt; ++t) {
        std::vector<float> x(n); for (auto& v : x) v = (float)rng.gauss();
        q2rows::activation_quantizer(x.data(), n, q.data() + (size_t)t * q_stride, xd.data() + (size_t)t * ch_stride, hx.data() + (size_t)t * ch_stride);
    }
    auto ref = [&](uint32_t r, uint32_t t) {
        double acc = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* blk = &w[((size_t)r * nb + b) * 18];
            uint16_t h; std::memcpy(&h, blk, 2);
            for (uint32_t i = 0; i < 64; ++i) {
                const int code = (blk[2 + i / 4] >> (2 * (i % 4))) & 3;
                acc += (double)h2f_(h) * xd[(size_t)t * ch_stride + 2 * b + i / 32] * (double)(code - 1) * q[(size_t)t * q_stride + b * 64 + i];
            }
        }
        return acc;
    };
    for (uint32_t nt = 1; nt <= max_nt; ++nt) {
        const size_t os = rows + 2;
        std::vector<float> out((size_t)max_nt * os, -9.f);
        q2rows::gguf_rows_multi(w.data(), 0, rows, nb, nt, q.data(), q_stride, xd.data(), hx.data(), ch_stride, out.data(), os);
        for (uint32_t t = 0; t < max_nt; ++t) for (uint32_t r = 0; r < os; ++r) {
            const float got = out[(size_t)t * os + r];
            if (t < nt && r < rows) { const double want = ref(r, t); CHECK(std::fabs(got - want) <= 1e-3 * (1.0 + std::fabs(want))); }
            else CHECK(got == -9.f);
        }
    }
    for (uint32_t t = 0; t < 4; ++t) {   // single-token entry points agree with the reference too
        std::vector<float> o(rows);
        q2rows::row_range(w.data(), 0, rows, nb, q.data() + (size_t)t * q_stride, xd.data() + (size_t)t * ch_stride, hx.data() + (size_t)t * ch_stride, o.data());
        if (t == 0) for (uint32_t r = 0; r < rows; ++r) CHECK(std::fabs(o[r] - ref(r, 0)) <= 1e-3 * (1.0 + std::fabs(ref(r, 0))));
    }
    for (uint32_t nt = 0; nt <= 64; ++nt) {   // groups of four cover every token once; group_size in 1..4
        uint32_t cov = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += q2rows::MAX_GROUP) { const uint32_t g = q2rows::group_size(nt, t0); CHECK(g >= 1 && g <= 4); cov += g; }
        CHECK(cov == nt);
    }
    for (uint32_t g = 0; g < 8; ++g) for (uint32_t t = 0; t < 4; ++t) CHECK(4 * g + t < 4 * (g + 1));
    for (uint32_t b = 0; b < 50; ++b) {
        CHECK(q2rows::block_start(b) + 18 == q2rows::block_start(b + 1));
        for (uint32_t i = 0; i < 64; ++i) CHECK(q2rows::chunk_of(b, i) == (b * 64 + i) / 32);
    }
}

// ================================================================================== IqQuantizedDotKernels
static void test_iqdot() {
    for (uint32_t t = 0; t < 64; ++t) {
        const bool want = t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11;
        CHECK(iqdot::supported(t) == want);
        CHECK((iqdot::row_bytes(t, 256) > 0) == want);
        if (!want) for (uint64_t n : {0ull, 64ull, 256ull, 4096ull}) CHECK(iqdot::row_bytes(t, n) == 0);
    }
    CHECK(iqdot::row_bytes(16, 256) == 66 && iqdot::row_bytes(17, 256) == 74 && iqdot::row_bytes(18, 256) == 98 &&
          iqdot::row_bytes(20, 256) == 144 && iqdot::row_bytes(21, 256) == 110 && iqdot::row_bytes(22, 256) == 82 &&
          iqdot::row_bytes(29, 256) == 56 && iqdot::row_bytes(23, 256) == 136 && iqdot::row_bytes(11, 256) == 110 &&
          iqdot::row_bytes(42, 256) == 72);
    for (uint32_t ty : kIq5) for (uint64_t n = 256; n <= 4096; n += 256) CHECK(iqdot::row_bytes(ty, n) == (uint64_t)iquant512::bytes_of(ty) * (n / 256));
    for (uint64_t n = 64; n <= 1024; n += 64) CHECK(iqdot::row_bytes(42, n) == (uint64_t)q2rows::BLOCK_BYTES * (n / 64));
    for (uint64_t n = 256; n <= 4096; n += 256) CHECK(n / 256 * 256 == n);

    // sign expansion
    for (uint32_t v = 0; v < 128; ++v) {
        CHECK(iqdot::ksigns(v) / 128 % 2 == iqdot::popc8(v) % 2);
        CHECK(iqdot::ksigns(v) % 128 == v);
        CHECK(iqdot::ksigns(v) == (v | ((uint32_t)__builtin_popcount(v) & 1u) << 7));
    }
    for (uint32_t s = 0; s < 256; ++s) for (uint32_t k = 0; k < 4; ++k) CHECK((iqdot::broadcast4(s) >> (8 * k)) % 256 == s);

    // warp call mapping
    Rng rng(0x705);
    for (int trial = 0; trial < 20000; ++trial) {
        const uint64_t ipb = 1 + rng.below(8), nb = 1 + rng.below(64), k = rng.below((uint32_t)(nb * ipb));
        const iqdot::Call c = iqdot::decompose_call(k, ipb);
        CHECK(c.block < nb);
        CHECK(c.block * ipb + c.part == k);
        CHECK(c.part < ipb);
        const iq_layout::Call c2 = iq_layout::decompose_call(k, ipb);
        CHECK(c2.block_index == c.block && c2.intra_block_offset == c.part);
        const uint64_t step = rng.below(3);
        CHECK(step == 0 || iqdot::iqs_of(k, ipb, step) < step * ipb);
        CHECK(k % 32 + 32 * (k / 32) == k);
    }
    // format traits
    for (uint32_t t : {16u, 17u, 18u, 20u, 21u, 22u, 29u, 42u}) CHECK(iqdot::traits_ok(t));
    for (uint32_t t : {0u, 11u, 23u, 30u, 41u, 43u}) CHECK(!iqdot::traits_ok(t));
    CHECK(iqdot::traits(20).qk == 32 && iqdot::traits(20).ipb == 2 && iqdot::traits(29).step == 1 && iqdot::traits(42).qk == 64 &&
          iqdot::traits(16).ipb == 8 && iqdot::traits(16).step == 2);
    CHECK(iq_layout::calls_per_block_iq256 == 8 && iq_layout::calls_per_block_iq32 == 2);

    // native expert layout
    const uint32_t types[] = {16, 17, 18, 21, 22, 29, 42, 11, 23};
    for (int trial = 0; trial < 400; ++trial) {
        const uint32_t gu = types[rng.below(9)], d = types[rng.below(9)];
        const uint64_t e = 256 * (1 + rng.below(16)), f = 256 * (1 + rng.below(16));
        const iqdot::ExpertLayout L = iqdot::expert_layout(gu, d, e, f);
        CHECK(L.gu_row == iqdot::row_bytes(gu, e) && L.d_row == iqdot::row_bytes(d, f));
        CHECK(L.down_off == L.up_off + f * L.gu_row);
        CHECK(L.bytes == L.down_off + e * L.d_row);
        for (int k = 0; k < 8; ++k) {
            const uint64_t r = rng.below((uint32_t)f), r2 = rng.below((uint32_t)e);
            CHECK(r * L.gu_row + L.gu_row <= L.up_off);
            CHECK(L.up_off + r * L.gu_row + L.gu_row <= L.down_off);
            CHECK(L.down_off + r2 * L.d_row + L.d_row <= L.bytes);
        }
        iq_layout::Layout M{e, f, L.gu_row, L.d_row};
        CHECK(M.up_offset() == L.up_off && M.down_offset() == L.down_off && M.bytes() == L.bytes);
        const uint32_t nff = (uint32_t)f;
        CHECK(2 * nff <= 8 * ((2 * nff + 7) / 8));
        for (uint32_t row = nff; row < 2 * nff; row += 97) CHECK(row - nff < nff);
    }
    // scratch sizing
    for (uint64_t x = 0; x < 3000; ++x) { CHECK(iqdot::align256(x) >= x); CHECK(iqdot::align256(x) % 256 == 0); CHECK(iqdot::align256(x) == iq_layout::align256(x)); CHECK(iqdot::align256(x) - x < 256); }
    for (uint64_t cap = 1; cap <= 16; ++cap) for (uint64_t f = 32; f <= 2048; f += 32) {
        CHECK(iqdot::grouped_scratch_bytes(cap, f) >= 3 * (cap * f * 4));
        CHECK(iqdot::grouped_scratch_bytes(cap, f) >= 3 * (cap * f * 4) + cap * (f / 32) * 36);
    }
    CHECK(iqdot::grouped_scratch_bytes(4, 512) == 3 * 8192 + 2304);
    // dequantizer thread maps
    bool seen[256] = {};
    for (uint32_t tid = 0; tid < 32; ++tid) {
        const uint32_t o = iqdot::dq8_offset(tid);
        CHECK(o + 8 <= 256);
        CHECK(o % 8 == 0 && !seen[o]);
        seen[o] = true;
        for (uint32_t t2 = tid + 1; t2 < 32; ++t2) CHECK(iqdot::dq8_offset(t2) != o);
        CHECK(iqdot::dq_q2_0_offset(tid) == 8 * tid);
        for (uint32_t j = 0; j < 4; ++j) CHECK(iqdot::dq_iq4_nl_hi(tid, j) < 32);
    }
    for (uint32_t o = 0; o < 256; o += 8) CHECK(seen[o]);   // the 32 threads tile the 256-value superblock exactly
    for (uint32_t r = 0; r < 40; ++r) for (uint32_t p = 0; p < 2; ++p) {
        CHECK(iqdot::gu_interleave(r, p) == iq_layout::gu_row(r, p));
        for (uint32_t r2 = 0; r2 < 40; ++r2) for (uint32_t p2 = 0; p2 < 2; ++p2)
            if (iqdot::gu_interleave(r, p) == iqdot::gu_interleave(r2, p2)) CHECK(r == r2 && p == p2);
    }
    for (int c = 0; c < 4; ++c) CHECK(iq_layout::q2_0_decode((uint32_t)c) == c - 1);
}

int main() {
    test_iq512_layout();
    test_iq512_reads();
    test_iq512_signs_scales_batches();
    test_iq512_kernels();
    test_q2_unpack_and_bounds();
    test_q2_quantizer();
    test_q2_rows();
    test_iqdot();
    std::printf("checks=%ld failures=%ld\n", testh::checks(), testh::failures());
    if (testh::failures() != 0) { std::printf("U5E0 TESTS FAILED\n"); return 1; }
    std::printf("U5E0 TESTS PASSED\n");
    return 0;
}

