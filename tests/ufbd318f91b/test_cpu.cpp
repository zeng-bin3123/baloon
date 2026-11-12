// Unit ufbd318f91b CPU suite: Q2ExpertDotKernel (src/cpu/expert_q2.cpp) and Q2Avx2Fallback (src/kernels/cpu/q2_avx2.cpp).
// Sections that execute AVX-512 VNNI/VBMI code run only when cpu_features().usable(); the rest always run.
// Prints "UFBD318F91B CPU TESTS PASSED" and exits 0 on success.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <immintrin.h>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/q2_avx2_rows.hpp"
#include "baloon/u11_expert.hpp"
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

int main() {
    const kern::CpuFeatures f = kern::cpu_features();
    std::printf("cpu: avx2=%d avx512 f/bw/vl/vnni/vbmi=%d%d%d%d%d -> %s\n", f.avx2, f.avx512f, f.avx512bw, f.avx512vl,
                f.avx512_vnni, f.avx512_vbmi, f.reason());
    return uf::run_all("UFBD318F91B CPU TESTS PASSED");
}
