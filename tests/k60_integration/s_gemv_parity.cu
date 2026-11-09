// s_gemv_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// s_gemv_parity: runs selected invariant checks of k14gpu_misc/test_misc.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k14gpu_misc/test_misc.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("s_gemv_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("s_gemv_parity: cannot open GPU 0\n"); return 2; }
    cpu_features();
    test_sgemv();
    std::printf("s_gemv_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("s_gemv_parity FAILED\n"); return 1; }
    std::printf("s_gemv_parity PASSED\n");
    return 0;
}

// ============================================================================================================
// Unit-34 integration layer. Everything below is host-side reference code and invariant checks for this
// executable; the legacy body above keeps running unchanged and its verdict is combined with this one.
// ============================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#define IG_UNUSED [[maybe_unused]]
IG_UNUSED static long ig_checks = 0;
IG_UNUSED static long ig_failures = 0;
IG_UNUSED static void ig_check(bool ok, const char* what) { ++ig_checks; if (!ok) { ++ig_failures; std::printf("  invariant violated: %s\n", what); } }
IG_UNUSED static double ig_rel_l1(const float* a, const float* b, size_t n) {
    double num = 0, den = 0; for (size_t i = 0; i < n; ++i) { num += std::fabs((double)a[i] - (double)b[i]); den += std::fabs((double)b[i]); }
    return den > 0 ? num / den : num;
}

// The legacy body runs unchanged with its output untouched (an external launch tracer may read this process's stdout).
template <class F> IG_UNUSED static int ig_legacy(F f) { return f(); }
IG_UNUSED static int ig_finish(const char* tag, int legacy_rc) {
    std::printf("%s: port-layer checks=%ld violations=%ld legacy_rc=%d\n", tag, ig_checks, ig_failures, legacy_rc);
    if (ig_checks == 0 || ig_failures != 0) { std::printf("%s: port-layer invariants NOT satisfied\n", tag); return 1; }
    if (legacy_rc != 0) return legacy_rc;
    std::printf("%s: port layer PASS\n", tag); return 0;
}
#include "baloon/scalar_dequantizers.hpp"
#include "baloon/port_kernels_ref.hpp"
#include "baloon/f16_bits.hpp"
struct IgCase { const char* name; int n_in, n_out; };
static const int ig_kScales = 40;
static void ig_put_f16(uint8_t* p, float d) { const uint16_t h = baloon::f16bits::f16_from_f32(d); p[0] = h & 255; p[1] = h >> 8; }
static float ig_round_f16(float d) { return baloon::f16bits::f32_from_f16(baloon::f16bits::f16_from_f32(d)); }
// Each *_to_raw packs (scale, codes) into the GGUF block layout; each *_deq is the value the block must decode to.
static void ig_q4_0_to_raw(float d, const int* q, uint8_t* raw) { ig_put_f16(raw, d); for (int j = 0; j < 16; ++j) raw[2 + j] = (uint8_t)(q[j] | (q[j + 16] << 4)); }
static void ig_q4_0_deq(float d, const int* q, float* y) { const float r = ig_round_f16(d); for (int j = 0; j < 32; ++j) y[j] = r * (float)(q[j] - 8); }
static void ig_q8_0_to_raw(float d, const int* q, uint8_t* raw) { ig_put_f16(raw, d); for (int j = 0; j < 32; ++j) raw[2 + j] = (uint8_t)(int8_t)q[j]; }
static void ig_q8_0_deq(float d, const int* q, float* y) { const float r = ig_round_f16(d); for (int j = 0; j < 32; ++j) y[j] = r * (float)q[j]; }
static void ig_q2_0_to_raw(float d, const int* c, uint8_t* raw) { ig_put_f16(raw, d); for (int j = 0; j < 16; ++j) raw[2 + j] = baloon::sdq::q2_pack(c[4 * j], c[4 * j + 1], c[4 * j + 2], c[4 * j + 3]); }
static void ig_q2_0_deq(float d, const int* c, float* y) { const float r = ig_round_f16(d); for (int j = 0; j < 64; ++j) y[j] = r * (float)(c[j] - 1); }
static void ig_iq4_to_raw(float d, const int* nib, uint8_t* raw) { ig_q4_0_to_raw(d, nib, raw); }
static void ig_iq4_deq(float d, const int* nib, float* y) { const float r = ig_round_f16(d); for (int j = 0; j < 32; ++j) y[j] = r * (float)baloon::sdq::kIq4nl[nib[j]]; }
static bool ig_same(const float* a, const float* b, int n) { for (int i = 0; i < n; ++i) if (a[i] != b[i]) return false; return true; }
// Q4_K block with d=1, dmin=1/2, sub-block scales s_i and mins m_i for i<4 (the rest zero): value = d*s*nibble - dmin*m.
static void ig_test_q4k() {
    uint8_t blk[144] = {0}; ig_put_f16(blk, 1.f); ig_put_f16(blk + 2, 0.5f);
    const int s[4] = {1, 2, 3, 4}, m[4] = {4, 3, 2, 1};
    for (int i = 0; i < 4; ++i) { blk[4 + i] = (uint8_t)s[i]; blk[8 + i] = (uint8_t)m[i]; }
    for (int l = 0; l < 128; ++l) blk[16 + l] = (uint8_t)((l % 16) | (((l + 5) % 16) << 4));
    float y[256]; baloon::sdq::deq_q4_k(blk, y);
    bool ok = true;
    for (int g = 0; g < 2; ++g) for (int l = 0; l < 32; ++l) {
        const uint8_t q = blk[16 + g * 32 + l]; const int i0 = 2 * g, i1 = 2 * g + 1;
        if (y[g * 64 + l] != 1.f * s[i0] * (float)(q & 15) - 0.5f * m[i0]) ok = false;
        if (y[g * 64 + 32 + l] != 1.f * s[i1] * (float)(q >> 4) - 0.5f * m[i1]) ok = false;
    }
    bool tail = true; for (int i = 128; i < 256; ++i) if (y[i] != 0.f) tail = false;
    ig_check(ok, "Q4_K sub-block scale/min decode"); ig_check(tail, "Q4_K sub-blocks with zero scale and min decode to zero");
}
static double ig_bench_s2_gemv(int reps, double* same_err) {
    namespace rk = baloon::refk; std::mt19937 rng(77); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int n_in = ig_kScales * 64, n_out = 16; rk::Codebook cb;
    std::vector<uint8_t> codes((size_t)n_out * rk::s2_row_bytes(n_in)); for (auto& b : codes) b = (uint8_t)(rng() & 255);
    std::vector<float> sc((size_t)n_out * ig_kScales); for (auto& s : sc) s = 0.1f + std::fabs(U(rng));
    std::vector<float> x(n_in); for (auto& v : x) v = U(rng); std::vector<float> a(n_out), b(n_out);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { rk::s2_gemv(codes.data(), sc.data(), cb, x.data(), a.data(), n_in, n_out); rk::s2_gemv_fast(codes.data(), sc.data(), cb, x.data(), b.data(), n_in, n_out); }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    *same_err = ig_rel_l1(a.data(), b.data(), n_out); return ms;
}
static void ig_run() {
    std::mt19937 rng(5); uint8_t raw[40]; float y[64], z[64]; int c[64];
    const float d = 0.37f;
    for (int j = 0; j < 32; ++j) c[j] = rng() % 16; ig_q4_0_to_raw(d, c, raw); baloon::sdq::deq_q4_0(raw, y); ig_q4_0_deq(d, c, z); ig_check(ig_same(y, z, 32), "Q4_0 raw block decodes to the packed codes");
    for (int j = 0; j < 32; ++j) c[j] = (int)(rng() % 255) - 127; ig_q8_0_to_raw(d, c, raw); baloon::sdq::deq_q8_0(raw, y); ig_q8_0_deq(d, c, z); ig_check(ig_same(y, z, 32), "Q8_0 raw block decodes to the packed codes");
    for (int j = 0; j < 64; ++j) c[j] = rng() % 4; ig_q2_0_to_raw(d, c, raw); baloon::sdq::deq_q2_0(raw, y); ig_q2_0_deq(d, c, z); ig_check(ig_same(y, z, 64), "Q2_0 raw block decodes to the packed codes");
    for (int j = 0; j < 32; ++j) c[j] = rng() % 16; ig_iq4_to_raw(d, c, raw); baloon::sdq::deq_iq4_nl(raw, y); ig_iq4_deq(d, c, z); ig_check(ig_same(y, z, 32), "IQ4_NL raw block decodes to the packed codes");
    ig_test_q4k();
    double err = 1; const double ms = ig_bench_s2_gemv(20, &err);
    std::printf("s_gemv_parity: s2 reference+fast gemv 20 reps %.2f ms\n", ms);
    ig_check(err == 0.0, "fast S2 gemv equals the reference exactly");
    const IgCase cases[] = {{"s2-row", ig_kScales * 64, 16}};
    ig_check(cases[0].n_in == 2560, "case width");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("s_gemv_parity", rc); }
