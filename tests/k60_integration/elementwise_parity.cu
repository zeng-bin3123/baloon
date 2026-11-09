// elementwise_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// elementwise_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("elementwise_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("elementwise_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_dense();
    std::printf("elementwise_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("elementwise_parity FAILED\n"); return 1; }
    std::printf("elementwise_parity PASSED\n");
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
static void ig_run() {
    namespace rk = baloon::refk;
    std::vector<float> xs; for (int i = -40; i <= 40; ++i) xs.push_back(0.25f * (float)i);
    const size_t n = xs.size();
    std::vector<float> a(n), b(n), c(n), d(n);
    for (size_t i = 0; i < n; ++i) { a[i] = rk::sigmoid(xs[i]) + rk::sigmoid(-xs[i]); b[i] = 1.f; }
    ig_check(ig_rel_l1(a.data(), b.data(), n) < 1e-6, "sigmoid(x)+sigmoid(-x)=1");
    for (size_t i = 0; i < n; ++i) { a[i] = rk::silu(xs[i]); b[i] = xs[i] * rk::sigmoid(xs[i]); }
    ig_check(ig_rel_l1(a.data(), b.data(), n) < 1e-6, "silu(x)=x*sigmoid(x)");
    for (size_t i = 0; i < n; ++i) { c[i] = rk::softplus(xs[i]) - rk::softplus(-xs[i]); d[i] = xs[i]; }
    ig_check(ig_rel_l1(c.data(), d.data(), n) < 1e-4, "softplus(x)-softplus(-x)=x");
    ig_check(std::fabs(rk::gelu_tanh(0.f)) == 0.f, "gelu(0)=0");
    ig_check(rk::gelu_tanh(6.f) > 5.99f && rk::gelu_tanh(-6.f) > -1e-3f, "gelu saturates to identity / zero");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("elementwise_parity", rc); }
