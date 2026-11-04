// unit95 host tests: LAYER 1 invariants of F16BitsRounding, DequantBlockLayouts, GatedResidual, plus the artifact
// dequantizers against the independent scalar references of earlier units.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "check.hpp"
#include "baloon/dequant_artifact.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/f16_conversion.hpp"
#include "baloon/gr_gated_residual.hpp"
#include "baloon/gr_up_geometry.hpp"
#include "baloon/scalar_dequantizers.hpp"

using namespace baloon;

// ---------------------------------------------------------------- F16BitsRounding
static void test_rne_step() {
    for (uint32_t h = 0; h < 64; ++h)
        for (uint32_t rem = 0; rem < 8192; ++rem) {
            const uint32_t r = f16bits::rne_step(h, rem);
            if (rem < 4096) CHECK(r == h);
            if (rem > 4096) CHECK(r == h + 1);
            if (rem == 4096) CHECK(r % 2 == 0);
            CHECK(r >= h && r <= h + 1);
        }
}
static void test_widen_exhaustive() {
    int bad = 0, bad_nan = 0, bad_sub = 0;
    for (uint32_t h = 0; h < 65536; ++h) {
        const float a = f16bits::widen_f16_bits((uint16_t)h), b = f16::half_to_float((uint16_t)h);
        const bool nan = ((h >> 10) & 31) == 31 && (h & 1023);
        if (nan) { if (!(a != a)) ++bad_nan; continue; }
        if (f16bits::as_u32(a) != f16bits::as_u32(b)) ++bad;
        // round trip: every non-NaN half survives widen -> round
        if (f16bits::round_f32_to_f16_bits(a) != (uint16_t)h) ++bad;
        // subnormal: value is m * 2^-24 exactly, and the exponent field is 113 - s for s = shifts
        if (((h >> 10) & 31) == 0 && (h & 1023)) {
            const double want = std::ldexp((double)(h & 1023), -24);
            if (std::fabs((double)a) != want) ++bad_sub;
            const int s = 10 - (int)std::floor(std::log2((double)(h & 1023)));
            if ((int)((f16bits::as_u32(a) >> 23) & 255) != 113 - s) ++bad_sub;
        }
    }
    CHECK(bad == 0); CHECK(bad_nan == 0); CHECK(bad_sub == 0);
    CHECK(127 - 15 + 1 == 113);
}
static void test_round_vs_reference() {
    std::mt19937 rng(7);
    long bad = 0;
    for (int i = 0; i < 3000000; ++i) {
        const uint32_t u = (uint32_t)rng();
        const float f = f16bits::as_f32(u);
        const uint16_t a = f16bits::round_f32_to_f16_bits(f), b = f16::float_to_half(f);
        if (f != f) { if (!(((a >> 10) & 31) == 31 && (a & 1023))) ++bad; }
        else if (a != b) ++bad;
    }
    CHECK(bad == 0);
    // exact ties: mantissa remainder 4096 rounds to even for both parities
    for (uint32_t e = 1; e <= 30; ++e)
        for (uint32_t hm = 0; hm < 1024; hm += 1) {
            const uint32_t x = ((e - 15 + 127) << 23) | (hm << 13) | 4096u;
            const uint16_t r = f16bits::round_f32_to_f16_bits(f16bits::as_f32(x));
            CHECK(r % 2 == 0);
            CHECK(r == (uint16_t)(e * 1024 + hm + (hm & 1)));
        }
}
static void test_inf_nan_overflow() {
    for (uint32_t sign : {0u, 0x80000000u}) {
        const uint16_t s16 = (uint16_t)(sign >> 16);
        CHECK(f16bits::round_f32_to_f16_bits(f16bits::as_f32(sign | 0x7F800000u)) == (uint16_t)(s16 | 0x7C00));      // inf
        for (uint32_t man : {1u, 0x2000u, 0x400000u, 0x7FFFFFu}) {
            const uint16_t r = f16bits::round_f32_to_f16_bits(f16bits::as_f32(sign | 0x7F800000u | man));
            CHECK(((r >> 10) & 31) == 31 && (r & 1023) != 0);                                                        // NaN stays NaN
        }
        for (uint32_t rawexp = 143; rawexp <= 254; ++rawexp)
            for (uint32_t man : {0u, 1u, 0x7FFFFFu}) {
                const uint16_t r = f16bits::round_f32_to_f16_bits(f16bits::as_f32(sign | (rawexp << 23) | man));
                CHECK(r == (uint16_t)(s16 | 0x7C00));                                                                // finite overflow -> inf, not NaN
            }
    }
    CHECK(((int)255 - 127 + 15) >= 31);                       // the conflation hazard: raw exponent 255 satisfies exp >= 31
    CHECK((uint32_t)(1 - 15 + 127) == 113u);                  // signed evaluation is right...
    CHECK(f16bits::round_f32_to_f16_bits(65504.f) == 0x7BFF);
    CHECK(f16bits::round_f32_to_f16_bits(65520.f) == 0x7C00); // rounds up into inf (tie to even carries)
}
static void test_bf16() {
    std::mt19937 rng(11);
    for (int i = 0; i < 2000000; ++i) {
        const uint32_t u = (uint32_t)rng() & 0x7FFFFFFFu;                      // below 2^31 so +0x8000 cannot wrap
        const float f = f16bits::as_f32(u);
        if (f != f || std::isinf(f)) continue;
        const uint32_t i32 = u, hi = i32 >> 16, lo = i32 & 0xFFFF;
        const uint16_t r = f16bits::round_f32_to_bf16_bits(f);
        if (lo < 0x8000) CHECK(r == hi);
        else if (lo > 0x8000) CHECK(r == hi + 1);
        else CHECK(r % 2 == 0);
        CHECK(f16bits::round_f32_to_bf16_bits(f16bits::as_f32((uint32_t)hi << 16)) == hi);   // already bf16: unchanged
    }
    CHECK(artifact::bf16_to_f32(0x3F80) == 1.0f);
}

// ---------------------------------------------------------------- DequantBlockLayouts
struct Geo { int type, elems, bytes; };
static const Geo kGeo[] = {{0,1,4},{1,1,2},{30,1,2},{2,32,18},{3,32,20},{6,32,22},{7,32,24},{8,32,34},{9,32,36},{10,256,84},
    {11,256,110},{12,256,144},{13,256,176},{14,256,210},{16,256,66},{17,256,74},{18,256,98},{20,32,18},{21,256,110},{22,256,82},
    {23,256,136},{29,256,56},{42,64,18}};
static void test_layouts() {
    for (const Geo& g : kGeo) CHECK(256 % g.elems == 0);                         // every block divides a super-block
    for (int b : {18, 22, 34, 110, 210}) CHECK(b % 4 == 2);                      // 16-bit loads only
    for (int b : {36, 136, 144, 176}) CHECK(b % 4 == 0);
    CHECK(2 + 64 * 2 / 8 == 18);
    CHECK(32 + 64 + 12 + 2 == 110 && 32 + 64 == 96 && 96 + 12 == 108);
    CHECK(128 + 64 + 16 + 2 == 210 && 128 + 64 == 192 && 192 + 16 == 208);
    CHECK(4 + 12 + 32 + 128 == 176 && 4 + 12 == 16 && 16 + 32 == 48);
    CHECK(4 + 12 + 128 == 144);
    CHECK(2 + 2 + 4 + 128 == 136 && 2 + 2 == 4 && 4 + 4 == 8);
    CHECK(2 + 4 + 16 == 22 && 2 + 4 == 6);
    // the repository constants agree with the layout theory
    CHECK(dequant::kQ3KBytes == 110 && dequant::kQ3KQsOff == 32 && dequant::kQ3KScalesOff == 96 && dequant::kQ3KDOff == 108);
    CHECK(dequant::kQ6KBytes == 210 && dequant::kQ6KQhOff == 128 && dequant::kQ6KScOff == 192 && dequant::kQ6KDOff == 208);
    CHECK(dequant::kQ5KBytes == 176 && dequant::kQ5KQhOff == 16 && dequant::kQ5KQsOff == 48);
    CHECK(dequant::kQ4KBytes == 144 && dequant::kQ4KQsOff == 16);
    CHECK(dequant::kIq4XsBytes == 136 && dequant::kIq4XsScalesHOff == 2 && dequant::kIq4XsScalesLOff == 4 && dequant::kIq4XsQsOff == 8);
    CHECK(dequant::kQ5_0Bytes == 22 && dequant::kQ2_0Bytes == 18 && dequant::kQ2_0Elems == 64);
    // row bytes: n / elems * bytes; whole blocks; weights total does not overflow
    for (const Geo& g : kGeo)
        for (int k = 1; k < 50; ++k) CHECK((size_t)(k * g.elems) / g.elems * g.bytes == (size_t)k * g.bytes);
    CHECK(artifact::row_bytes("q4_k", 4096) == 16 * 144 && artifact::row_bytes("q6_k", 512) == 2 * 210 && artifact::row_bytes("q2_0", 128) == 36);
    CHECK(artifact::row_bytes("nope", 64) == 0);
    for (uint64_t row : {1ull, 1000ull, 1ull << 20}) for (uint64_t nout : {1ull, 7ull, 4096ull}) {
        const uint64_t M = UINT64_MAX;
        if (row <= M / nout) CHECK(row * nout <= M);
    }
    // value ranges of the integer codes
    for (int c = 0; c < 16; ++c) { CHECK(c - 8 >= -8 && c - 8 <= 7); for (int b = 0; b < 2; ++b) CHECK(c + 16 * b - 16 >= -16 && c + 16 * b - 16 <= 15); }
    for (int n = 0; n < 16; ++n) for (int h = 0; h < 4; ++h) CHECK(n + 16 * h - 32 >= -32 && n + 16 * h - 32 <= 31);
    for (int t = 0; t < 4; ++t) for (int m : {0, 4}) CHECK(t - m >= -4 && t - m <= 3);
    for (int c = 0; c < 4; ++c) CHECK(c - 1 >= -1 && c - 1 <= 2);
    for (int u : {1, 4, 16, 64}) CHECK(2 * u < 256 && (u & (2 * u)) == 0);
    // 6-bit scale / min of Q4_K / Q5_K for every byte pattern of the neighbours
    for (int a = 0; a < 256; ++a) for (int b = 0; b < 256; b += 3) {
        uint8_t q[12]; for (int i = 0; i < 12; ++i) q[i] = (uint8_t)((i % 2) ? a : b);
        for (int j = 0; j < 8; ++j) { uint8_t d, m; dequant::get_scale_min_k4(j, q, &d, &m); CHECK(d < 64 && m < 64); }
    }
    for (int lo = 0; lo < 256; ++lo) for (int hi = 0; hi < 256; ++hi) { const int ls = lo % 16 + 16 * (hi % 4); CHECK(ls < 64 && ls - 32 >= -32 && ls - 32 <= 31); }
}
static void put_h(uint8_t* p, float v) { const uint16_t h = f16bits::round_f32_to_f16_bits(v); p[0] = h & 255; p[1] = h >> 8; }
static void test_artifact_vs_sdq() {
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> U(-2.f, 2.f);
    struct T { const char* name; int e, b; int d0, d1; void (*ref)(const uint8_t*, float*); };
    const T ts[] = {{"q2_0",64,18,0,-1,sdq::deq_q2_0},{"q4_0",32,18,0,-1,sdq::deq_q4_0},{"q5_0",32,22,0,-1,sdq::deq_q5_0},
        {"q8_0",32,34,0,-1,sdq::deq_q8_0},{"iq4_nl",32,18,0,-1,sdq::deq_iq4_nl},{"q3_k",256,110,108,-1,sdq::deq_q3_k},
        {"q4_k",256,144,0,2,sdq::deq_q4_k},{"q5_k",256,176,0,2,sdq::deq_q5_k},{"q6_k",256,210,208,-1,sdq::deq_q6_k},
        {"iq4_xs",256,136,0,-1,sdq::deq_iq4_xs}};
    for (const T& t : ts) {
        const int nblk = 6, n = nblk * t.e;
        std::vector<uint8_t> in((size_t)nblk * t.b);
        for (int trial = 0; trial < 40; ++trial) {
            for (auto& x : in) x = (uint8_t)rng();
            for (int b = 0; b < nblk; ++b) { put_h(&in[(size_t)b * t.b + t.d0], U(rng)); if (t.d1 >= 0) put_h(&in[(size_t)b * t.b + t.d1], U(rng)); }
            std::vector<float> a(n), r(n);
            CHECK(artifact::dequantize_row(t.name, in.data(), a.data(), n));
            for (int b = 0; b < nblk; ++b) t.ref(&in[(size_t)b * t.b], &r[(size_t)b * t.e]);
            int bad = 0;
            for (int i = 0; i < n; ++i) if (std::fabs(a[i] - r[i]) > 1e-5f * (1.f + std::fabs(r[i]))) ++bad;
            CHECK_MSG(bad == 0, "%s trial %d: %d mismatches", t.name, trial, bad);
        }
        if (t.e > 1) { std::vector<float> tmp(t.e); CHECK(!artifact::dequantize_row(t.name, in.data(), tmp.data(), (size_t)t.e - 1)); }
    }
    CHECK(!artifact::dequantize_row("bogus", nullptr, nullptr, 0));
    // f32 / f16 / bf16 rows
    float f[4] = {1.5f, -2.f, 0.f, 1e-6f}; uint8_t raw[16]; std::memcpy(raw, f, 16); float o[4];
    artifact::deq_f32(raw, o, 4); CHECK(std::memcmp(o, f, 16) == 0);
    uint8_t h[8]; for (int i = 0; i < 4; ++i) { const uint16_t v = f16bits::round_f32_to_f16_bits(f[i]); h[2*i] = v & 255; h[2*i+1] = v >> 8; }
    artifact::deq_f16(h, o, 4); CHECK(o[0] == 1.5f && o[1] == -2.f && o[2] == 0.f);
    uint8_t bb[8]; for (int i = 0; i < 4; ++i) { const uint16_t v = f16bits::round_f32_to_bf16_bits(f[i]); bb[2*i] = v & 255; bb[2*i+1] = v >> 8; }
    artifact::deq_bf16(bb, o, 4); CHECK(o[0] == 1.5f && o[1] == -2.f && std::fabs(o[3] - 1e-6f) < 1e-8f);
}

// ---------------------------------------------------------------- GatedResidual
static void test_gr() {
    using namespace baloon::gr;
    CHECK(kN * kHc == 10240 && kD == 10240);
    CHECK(kLr / kWarps == 40 && kDownBlocks * kWarps == kLr);
    CHECK(80 * 32 == kN && 160 * 16 == kN && kUpBlocks == 80);
    CHECK(320 / 8 == 40 && 40 - 32 == 8);
    CHECK(10240 / 8 == 1280 && 1280 / 32 == 40);
    CHECK(10240 / 2560 == 4 && 2560 / 8 / 32 == 10);
    // down rows, reduction ownership, gate rows, lane coverage
    for (int k = 0; k < 320; ++k) CHECK(k / 8 < 40 && (k / 8) * 8 + k % 8 == k);
    for (int i = 0; i < 10240; ++i) {
        CHECK(i == (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 && (i % 256) / 32 < 8);
        CHECK(i / 8 < (10240 + 7) / 8 && i == (i / 8) * 8 + i % 8);
    }
    for (int j = 0; j < 320; ++j) CHECK(j == j % 32 + 32 * (j / 32) && j / 32 < 10);
    // up kernel: every (block, row) in range, bijection onto [0, 10240)
    std::vector<int> hit(kD, 0);
    for (int b = 0; b < 80; ++b) for (int r = 0; r < 128; ++r) { const int i = up_index(b, r); CHECK(i >= 0 && i < kD); if (i >= 0 && i < kD) ++hit[i]; }
    int bad = 0; for (int i = 0; i < kD; ++i) { if (hit[i] != 1) ++bad; CHECK(up_index(up_block_of(i), up_row_of(i)) == i && up_block_of(i) < 80 && up_row_of(i) < 128); }
    CHECK(bad == 0);
    // residual write decode and mean index
    for (int i = 0; i < kD; ++i) CHECK(i / kN < kHc && i % kN < kN && (i / kN) * kN + i % kN == i);
    for (int c = 0; c < kHc; ++c) for (int d = 0; d < kN; ++d) CHECK(c * kN + d < kHc * kN);
    // workspace
    for (uint64_t s = 0; s < 5000; ++s) { const uint64_t a = align16(s); CHECK(a >= s && a < s + 16 && a % 16 == 0); CHECK(7 + s <= 7 + a); }
    CHECK(workspace_bytes() == 104320);
    const uint64_t sizes[5] = {10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4};
    const Workspace w = workspace_layout(sizes);
    CHECK(w.total == 104320);
    for (int i = 0; i < 5; ++i) { CHECK(w.off[i] % 16 == 0); if (i) CHECK(w.off[i] >= w.off[i - 1] + w.size[i - 1]); }
    CHECK(640 < 10240 * 2);                                    // the old xq bug: 640 bytes is strictly too small
    CHECK(8 * 4 <= 8 * 8);                                     // float scratch fits in the double scratch
    // write / mean reference semantics
    std::vector<float> R(kD), bo(kN), wt(kHc), out(kD);
    for (int i = 0; i < kD; ++i) R[i] = (float)(i % 13) * 0.25f;
    for (int d = 0; d < kN; ++d) bo[d] = (float)(d % 7) - 3.f;
    for (int c = 0; c < kHc; ++c) wt[c] = 0.5f + (float)c;
    write(R.data(), bo.data(), wt.data(), out.data(), kN, kHc);
    int wbad = 0; for (int i = 0; i < kD; ++i) if (out[i] != R[i] + bo[i % kN] * wt[i / kN]) ++wbad;
    CHECK(wbad == 0);
}

int main() {
    test_rne_step(); test_widen_exhaustive(); test_round_vs_reference(); test_inf_nan_overflow(); test_bf16();
    test_layouts(); test_artifact_vs_sdq(); test_gr();
    std::printf("unit95 host: %ld checks, %ld failures\n", testh::checks(), testh::failures());
    return testh::failures() ? 1 : 0;
}
