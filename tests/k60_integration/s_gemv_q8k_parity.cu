// s_gemv_q8k_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// s_gemv_q8k_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("s_gemv_q8k_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("s_gemv_q8k_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_formats();
    std::printf("s_gemv_q8k_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("s_gemv_q8k_parity FAILED\n"); return 1; }
    std::printf("s_gemv_q8k_parity PASSED\n");
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
struct IgCase { int n_in, n_out; unsigned seed; };
static const int8_t ig_kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const int ig_Q8K_BYTES = 292;   // float d + int8 q[256] + int16 bsums[16]
// Code of element i of an array of Q8_K blocks.
static int ig_q8k_at(const uint8_t* blocks, int64_t i) { return (int)(int8_t)blocks[(i / 256) * ig_Q8K_BYTES + 4 + (i % 256)]; }
static float ig_q8k_scale(const uint8_t* blocks, int64_t i) { float d; std::memcpy(&d, blocks + (i / 256) * ig_Q8K_BYTES, 4); return d; }
// Weight at element i of an IQ4_NL row (18-byte blocks of 32: low nibbles are elements 0..15, high nibbles 16..31).
static float ig_weight_at(const uint8_t* row, int64_t i) {
    const uint8_t* b = row + (i / 32) * 18; const int j = (int)(i % 32);
    const float d = baloon::sdq::rd_f16(b); const uint8_t q = b[2 + (j % 16)];
    return d * (float)ig_kIq4Nl[j < 16 ? (q & 0xF) : (q >> 4)];
}
static void ig_run_case(const IgCase& c) {
    std::mt19937 rng(c.seed); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int nb = c.n_in / 32;
    std::vector<uint8_t> W((size_t)c.n_out * nb * 18);
    for (size_t b = 0; b < (size_t)c.n_out * nb; ++b) { const uint16_t h = 0x2E66; W[b * 18] = h & 255; W[b * 18 + 1] = h >> 8; for (int j = 0; j < 16; ++j) W[b * 18 + 2 + j] = (uint8_t)(rng() & 255); }
    std::vector<uint8_t> A((size_t)(c.n_in / 256) * ig_Q8K_BYTES, 0);
    for (int b = 0; b < c.n_in / 256; ++b) { const float d = 0.01f; std::memcpy(&A[(size_t)b * ig_Q8K_BYTES], &d, 4); for (int i = 0; i < 256; ++i) A[(size_t)b * ig_Q8K_BYTES + 4 + i] = (uint8_t)(int8_t)((int)(rng() % 255) - 127); }
    std::vector<float> ya(c.n_out), yb(c.n_out);
    for (int r = 0; r < c.n_out; ++r) {
        const uint8_t* row = W.data() + (size_t)r * nb * 18; double a = 0, bq = 0;
        for (int i = 0; i < c.n_in; ++i) a += (double)ig_weight_at(row, i) * ig_q8k_at(A.data(), i) * ig_q8k_scale(A.data(), i);
        std::vector<float> deq(c.n_in); for (int b = 0; b < nb; ++b) baloon::sdq::deq_iq4_nl(row + (size_t)b * 18, deq.data() + b * 32);
        for (int i = 0; i < c.n_in; ++i) bq += (double)deq[i] * ig_q8k_at(A.data(), i) * ig_q8k_scale(A.data(), i);
        ya[r] = (float)a; yb[r] = (float)bq;
    }
    ig_check(ig_rel_l1(ya.data(), yb.data(), c.n_out) < 1e-6, "element accessors agree with the block dequantiser");
}
static void ig_run() {
    for (int i = 0; i < 16; ++i) ig_check(ig_kIq4Nl[i] == baloon::sdq::kIq4nl[i], "IQ4_NL table matches the library table");
    ig_check(ig_Q8K_BYTES == 4 + 256 + 32, "Q8_K block bytes");
    for (const IgCase& c : {IgCase{256, 4, 1}, IgCase{512, 8, 2}, IgCase{2048, 5, 3}}) ig_run_case(c);
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("s_gemv_q8k_parity", rc); }
