// bf16_gemv_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// bf16_gemv_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("bf16_gemv_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("bf16_gemv_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_dense();
    std::printf("bf16_gemv_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("bf16_gemv_parity FAILED\n"); return 1; }
    std::printf("bf16_gemv_parity PASSED\n");
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
#include "baloon/port_kernels_ref.hpp"
IG_UNUSED static float ig_bf16_to_f32(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }
IG_UNUSED static uint16_t ig_f32_to_bf16(float f) { uint32_t b; std::memcpy(&b, &f, 4); return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16); }
// Double-accumulated reference: y[o] = sum_i bf16(W[o][i]) * x[i].
static void ig_reference(const uint16_t* w, const float* x, float* y, int n_in, int n_out) {
    for (int o = 0; o < n_out; ++o) { double acc = 0; for (int i = 0; i < n_in; ++i) acc += (double)ig_bf16_to_f32(w[(size_t)o * n_in + i]) * x[i]; y[o] = (float)acc; }
}
static void ig_run() {
    std::mt19937 rng(7); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int n = 64;
    std::vector<uint16_t> I((size_t)n * n, 0); for (int i = 0; i < n; ++i) I[(size_t)i * n + i] = 0x3F80;
    std::vector<float> x(n), y(n); for (auto& v : x) v = U(rng);
    ig_reference(I.data(), x.data(), y.data(), n, n);
    ig_check(ig_rel_l1(y.data(), x.data(), n) == 0.0, "identity matrix returns x exactly");
    const int ni = 256, no = 48;
    std::vector<uint16_t> W((size_t)no * ni); for (auto& v : W) v = ig_f32_to_bf16(U(rng));
    std::vector<float> xx(ni); for (auto& v : xx) v = U(rng);
    std::vector<float> ya(no), yb(no);
    ig_reference(W.data(), xx.data(), ya.data(), ni, no);
    baloon::refk::bf16_gemv(W.data(), xx.data(), yb.data(), ni, no);
    ig_check(ig_rel_l1(yb.data(), ya.data(), no) < 1e-4, "library bf16 gemv matches the double reference");
    std::vector<float> x2(ni), y2(no); for (int i = 0; i < ni; ++i) x2[i] = 2.f * xx[i];
    ig_reference(W.data(), x2.data(), y2.data(), ni, no);
    for (int o = 0; o < no; ++o) ya[o] *= 2.f;
    ig_check(ig_rel_l1(y2.data(), ya.data(), no) < 1e-6, "gemv is linear in x");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("bf16_gemv_parity", rc); }
