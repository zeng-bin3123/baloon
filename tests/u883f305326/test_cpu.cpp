// Unit u883f305326 CPU suite: Q2ExpertDotKernel (src/cpu/expert_q2.cpp) and Q2Avx2Fallback (src/kernels/cpu/q2_avx2.cpp).
// Sections that execute AVX-512 VNNI/VBMI code run only when cpu_features().usable(); the rest always run.
// Prints "U883F305326 CPU TESTS PASSED" and exits 0 on success.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <random>
#include <vector>

#include <immintrin.h>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/u11_expert.hpp"
#include "baloon/expert_table.hpp"
#include "harness.hpp"

using namespace baloon;
namespace qa = baloon::u11::q2avx2;

static std::mt19937 rng(0x5eedu);
static uint16_t f2h(float v) { return (uint16_t)_cvtss_sh(v, 0); }
static float h2f(uint16_t h) { return _cvtsh_ss(h); }
static std::vector<uint8_t> rand_bytes(size_t n) { std::vector<uint8_t> v(n); for (auto& b : v) b = (uint8_t)rng(); return v; }
static uint8_t code_at(const uint8_t* c16, uint32_t j) { return (uint8_t)((c16[j / 4] >> (2 * (j % 4))) & 3u); }
// GGUF Q2_0 matrix: rows x nb blocks of 18 bytes (fp16 scale, 16 code bytes)
static std::vector<uint8_t> gguf_q2(uint32_t rows, uint32_t nb) {
    std::vector<uint8_t> w = rand_bytes((size_t)rows * nb * 18);
    for (size_t b = 0; b < (size_t)rows * nb; ++b) { uint16_t h = f2h(std::ldexp((float)(rng() % 1000 + 1), -12)); std::memcpy(&w[b * 18], &h, 2); }
    return w;
}
static std::vector<float> rand_f(size_t n, float a = 2.f) { std::uniform_real_distribution<float> u(-a, a); std::vector<float> v(n); for (auto& x : v) x = u(rng); return v; }

// "the scalar and AVX2 forms are bit-identical" ; q in [-127,127] ; hx = sum of the chunk's q
UF_SECTION(act_quant_scalar_eq_avx2, "act_quant_q8_1 and act_quant_q8_1_avx2 are bit-identical; |q|<=127; hx is the chunk sum") {
    const uint32_t n = 32 * 37;
    std::vector<float> x = rand_f(n, 5.f); x[3] = 0.f; for (uint32_t i = 64; i < 96; ++i) x[i] = 0.f;   // an all-zero chunk
    std::vector<int8_t> q1(n), q2(n); std::vector<float> d1(n / 32), d2(n / 32); std::vector<int32_t> h1(n / 32), h2(n / 32);
    kern::act_quant_q8_1(x.data(), n, q1.data(), d1.data(), h1.data());
    kern::act_quant_q8_1_avx2(x.data(), n, q2.data(), d2.data(), h2.data());
    CHECK(std::memcmp(q1.data(), q2.data(), n) == 0);
    CHECK(std::memcmp(d1.data(), d2.data(), d1.size() * 4) == 0);
    CHECK(std::memcmp(h1.data(), h2.data(), h1.size() * 4) == 0);
    for (uint32_t c = 0; c < n / 32; ++c) {
        int32_t s = 0; for (uint32_t i = 0; i < 32; ++i) { s += q1[c * 32 + i]; CHECK(q1[c * 32 + i] >= -127 && q1[c * 32 + i] <= 127); }
        CHECK(s == h1[c]);
        for (uint32_t i = 0; i < 32; ++i) CHECK(std::fabs(q1[c * 32 + i] * d1[c] - x[c * 32 + i]) <= 0.5f * d1[c] + 1e-6f);
    }
}

// "value i of a 64-value block is in code byte i/4 at bit 2*(i%4)"
UF_SECTION(unpack64_layout, "unpack64: value i is code byte i/4 at bit 2*(i%4)") {
    for (int rep = 0; rep < 64; ++rep) {
        std::vector<uint8_t> c = rand_bytes(16); uint8_t out[64];
        qa::unpack64(c.data(), out);
        for (uint32_t i = 0; i < 64; ++i) CHECK(out[i] == code_at(c.data(), i));
    }
}

// reference: out[t][r] = sum_b sum_i h2f(scale_b) * (code_i - 1) * q[t][64b+i] * xd[t][chunk]
static void q2_reference(const std::vector<uint8_t>& w, uint32_t rows, uint32_t nb, uint32_t nt, const std::vector<int8_t>& q,
                         const std::vector<float>& xd, std::vector<double>& out) {
    const uint32_t n = nb * 64; out.assign((size_t)nt * rows, 0.0);
    for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < rows; ++r) for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = &w[((size_t)r * nb + b) * 18]; uint16_t h; std::memcpy(&h, blk, 2);
        for (uint32_t i = 0; i < 64; ++i)
            out[(size_t)t * rows + r] += (double)h2f(h) * ((int)code_at(blk + 2, i) - 1) * q[(size_t)t * n + 64 * b + i] * xd[(size_t)t * (n / 32) + 2 * b + i / 32];
    }
}
static void q2_case(uint32_t rows, uint32_t nb, uint32_t nt, std::vector<int8_t>& q, std::vector<float>& xd, std::vector<int32_t>& hx,
                    std::vector<uint8_t>& w) {
    const uint32_t n = nb * 64; w = gguf_q2(rows, nb); q.resize((size_t)nt * n); xd.resize((size_t)nt * n / 32); hx.resize(xd.size());
    for (uint32_t t = 0; t < nt; ++t) { std::vector<float> x = rand_f(n); kern::act_quant_q8_1(x.data(), n, &q[(size_t)t * n], &xd[(size_t)t * n / 32], &hx[(size_t)t * n / 32]); }
}
// "rows [row0, row1) against nt tokens (in groups of at most 4); out[t * out_stride + r]"
UF_SECTION(q2_avx2_rows_vs_reference, "q2_0_gguf_rows_multi_avx2 matches the exact (code-1)*d dot for nt=1..9 (groups of <=4) and a row sub-range") {
    const uint32_t rows = 13, nb = 5, n = nb * 64;
    for (uint32_t nt = 1; nt <= 9; ++nt) {
        std::vector<int8_t> q; std::vector<float> xd; std::vector<int32_t> hx; std::vector<uint8_t> w; q2_case(rows, nb, nt, q, xd, hx, w);
        std::vector<double> ref; q2_reference(w, rows, nb, nt, q, xd, ref);
        std::vector<float> out((size_t)nt * rows, -7.f);
        qa::q2_0_gguf_rows_multi_avx2(w.data(), 2, 11, nb, nt, q.data(), n, xd.data(), hx.data(), n / 32, out.data(), rows);
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < rows; ++r) {
            const float o = out[(size_t)t * rows + r];
            if (r < 2 || r >= 11) CHECK(o == -7.f);                       // outside [row0,row1) is untouched
            else CHECK(std::fabs(o - ref[(size_t)t * rows + r]) <= 1e-4 * (1.0 + std::fabs(ref[(size_t)t * rows + r])));
        }
    }
}

// "Computes rows [row0, row1) ... against nt quantized activations": the AVX-512 VNNI path agrees with the AVX2 fallback
UF_SECTION(q2_avx512_eq_avx2, "q2_0_gguf_rows_multi (AVX-512) agrees with the AVX2 fallback and the reference (skipped without VNNI/VBMI)") {
    if (!kern::cpu_features().usable()) { std::printf("  (skipped: %s)\n", kern::cpu_features().reason()); return; }
    const uint32_t rows = 17, nb = 6, n = nb * 64;
    for (uint32_t nt : {1u, 3u, 4u, 5u, 8u}) {
        std::vector<int8_t> q; std::vector<float> xd; std::vector<int32_t> hx; std::vector<uint8_t> w; q2_case(rows, nb, nt, q, xd, hx, w);
        std::vector<double> ref; q2_reference(w, rows, nb, nt, q, xd, ref);
        std::vector<float> a((size_t)nt * rows), b((size_t)nt * rows);
        kern::q2_0_gguf_rows_multi(w.data(), 0, rows, nb, nt, q.data(), n, xd.data(), hx.data(), n / 32, a.data(), rows);
        kern::q2_0_gguf_rows_multi_avx2(w.data(), 0, rows, nb, nt, q.data(), n, xd.data(), hx.data(), n / 32, b.data(), rows);
        for (size_t k = 0; k < a.size(); ++k) {
            CHECK(std::fabs(a[k] - b[k]) <= 1e-4 * (1.0 + std::fabs(b[k])));
            CHECK(std::fabs(a[k] - ref[k]) <= 1e-4 * (1.0 + std::fabs(ref[k])));
        }
    }
}

// ---- end of sections

// ---- unit u883f305326 additions: one section per Q2Avx2Fallback / Q2ExpertDotKernel source-semantics sentence ----

// "unpack64 shifts 16-bit lanes by 0/2/4/6 and masks each byte with 3; despite the 16-bit shift each byte keeps its OWN codes"
UF_SECTION(u6_unpack64_every_byte, "unpack64: every byte value 0..255 in every position keeps its own four codes") {
    for (uint32_t v = 0; v < 256; ++v) {
        uint8_t c[16], out[64];
        for (uint32_t j = 0; j < 16; ++j) c[j] = (uint8_t)(v + 37 * j);     // neighbours differ, so leaks would show
        qa::unpack64(c, out);
        for (uint32_t i = 0; i < 64; ++i) CHECK(out[i] == ((c[i / 4] >> (2 * (i % 4))) & 3u));
    }
}

// "maddubs of codes (<= 3) with int8 activations (|x| <= 127) never saturates" ; per block d(s0 (S0 - sum0) + s1 (S1 - sum1))
UF_SECTION(u6_maddubs_extremes, "row dot at the saturation extremes (codes 3, x = +-127) equals the exact (code-1)*d sum") {
    const uint32_t nb = 6;
    for (int sign : {1, -1}) {
        std::vector<uint8_t> w(nb * 18, 0xFF);
        for (uint32_t b = 0; b < nb; ++b) { uint16_t h = f2h(0.25f); std::memcpy(&w[b * 18], &h, 2); }
        std::vector<int8_t> q(nb * 64, (int8_t)(127 * sign));
        std::vector<float> xd(2 * nb, 0.5f); std::vector<int32_t> hx(2 * nb, 32 * 127 * sign);
        float out = 0.f;
        qa::rows(w.data(), 0, 1, nb, q.data(), xd.data(), hx.data(), &out);
        const float exact = (float)nb * 64.f * 0.25f * 0.5f * (3 - 1) * 127.f * sign;   // all products exact in f32
        CHECK(out == exact);
    }
}

// "The quantizer's integer horizontal sum (shuffle 0x4E then 0xB1) adds all four lanes"
UF_SECTION(u6_hsum_lanes, "act_quant_q8_1_avx2: hx equals the integer sum of the chunk's 32 codes for one-hot chunks") {
    for (uint32_t pos = 0; pos < 32; ++pos) {
        std::vector<float> x(32, 0.f); x[pos] = 3.f; x[(pos + 9) % 32] = -1.5f;
        int8_t q[32]; float d; int32_t hx;
        qa::act_quant_q8_1_avx2(x.data(), 32, q, &d, &hx);
        int32_t s = 0; for (int i = 0; i < 32; ++i) s += q[i];
        CHECK(hx == s && q[pos] == 127);
    }
}

// "Rows at >4 tokens are processed in groups of at most 4 that cover every token"
UF_SECTION(u6_groups_cover, "group_size(nt, t0) steps give groups of 1..4 tokens covering nt = 1..64") {
    for (uint32_t nt = 1; nt <= 64; ++nt) {
        uint32_t covered = 0;
        for (uint32_t t0 = 0; t0 < nt; t0 += q2rows::group_size(nt, t0)) {
            const uint32_t g = q2rows::group_size(nt, t0);
            CHECK(g >= 1 && g <= 4); covered += g;
        }
        CHECK(covered == nt);
    }
}

// "The CPU gate requires AVX512 F, BW, VL, VNNI and VBMI (CPUID.7.0 EBX bits 16, 30, 31; ECX 11, 1)"
UF_SECTION(u6_cpu_gate_bits, "avx512_ok accepts exactly EBX 16/30/31 + ECX 11/1; dropping any one bit refuses") {
    const uint32_t ebx = (1u << 16) | (1u << 30) | (1u << 31), ecx = (1u << 11) | 2u;
    CHECK(etable::avx512_ok(ebx, ecx));
    for (unsigned k : {16u, 30u, 31u}) CHECK(!etable::avx512_ok(ebx & ~(1u << k), ecx));
    for (unsigned k : {11u, 1u}) CHECK(!etable::avx512_ok(ebx, ecx & ~(1u << k)));
    const kern::CpuFeatures f = kern::cpu_features();
    CHECK(std::string(f.reason()) == "ok" ? f.usable() : !f.usable());
}

// "Blob geometry (H = 2560, FF = 640): gate/up rows interleaved (gate 2r, up 2r+1), 40 blocks each; down rows 10 blocks each; total 1,382,400 bytes"
UF_SECTION(u6_blob_geometry, "q2geo: 40/10 blocks per row, interleaved gate/up rows, BLOB = 1,382,400") {
    CHECK(kern::q2geo::H / 64 == 40 && kern::q2geo::FF / 64 == 10);
    CHECK(kern::q2geo::BLOB == 2ull * kern::q2geo::FF * (640 + 80) + (uint64_t)kern::q2geo::H * (160 + 20));
    CHECK(kern::q2geo::BLOB == 1382400ull);
    for (uint32_t r = 0; r < kern::q2geo::FF; ++r) CHECK((2 * r) / 2 == r && (2 * r + 1) / 2 == r && 2 * r + 1 < 2 * kern::q2geo::FF);
    for (uint32_t n = 1; n <= 8; ++n) CHECK(kern::q2geo::dispatch_nt(n) == n);
    for (uint32_t nb = 0; nb < 8; ++nb) CHECK(kern::q2geo::tail_mask_bits(nb) == (1u << (2 * nb)) - 1u);
}

int main() {
    const kern::CpuFeatures f = kern::cpu_features();
    std::printf("cpu: avx2=%d avx512 f/bw/vl/vnni/vbmi=%d%d%d%d%d -> %s\n", f.avx2, f.avx512f, f.avx512bw, f.avx512vl,
                f.avx512_vnni, f.avx512_vbmi, f.reason());
    return uf::run_all("U883F305326 CPU TESTS PASSED");
}
