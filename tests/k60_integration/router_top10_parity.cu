// router_top10_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// router_top10_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("router_top10_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("router_top10_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_moe();
    std::printf("router_top10_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("router_top10_parity FAILED\n"); return 1; }
    std::printf("router_top10_parity PASSED\n");
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
static const double ig_RENORM_CLAMP = 1.0 / 16384.0;
// Independent reference: double softmax, order by (prob desc, index asc), renormalise over the selected ten.
static int ig_reference(const float* logits, int n, int* idx, float* w) {
    if (n < 1) return 0;
    const int k = std::min(10, n);
    double mx = -1e300; for (int i = 0; i < n; ++i) mx = std::max(mx, (double)logits[i]);
    std::vector<double> p(n); double s = 0; for (int i = 0; i < n; ++i) { p[i] = std::exp((double)logits[i] - mx); s += p[i]; }
    for (auto& e : p) e /= s;
    std::vector<int> o(n); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
    double sel = 0; for (int j = 0; j < k; ++j) sel += p[o[j]];
    sel = std::max(sel, ig_RENORM_CLAMP);
    for (int j = 0; j < k; ++j) { idx[j] = o[j]; w[j] = (float)(p[o[j]] / sel); }
    return k;
}
static void ig_run_case(int n, unsigned seed, bool ties) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-4.f, 4.f);
    std::vector<float> lg(n); for (auto& v : lg) v = ties ? 1.f : U(rng);
    int ia[10], ib[10]; float wa[10], wb[10];
    const int ka = ig_reference(lg.data(), n, ia, wa);
    const int kb = baloon::refk::native_router_top10(lg.data(), n, ib, wb);
    ig_check(ka == kb && ka == std::min(10, n), "selected count is min(10, n)");
    bool same = ka == kb; for (int j = 0; j < std::min(ka, kb); ++j) same = same && ia[j] == ib[j];
    ig_check(same, "selected indices match (ties go to the lower index)");
    double sum = 0; bool mono = true; for (int j = 0; j < kb; ++j) { sum += wb[j]; if (j && wb[j] > wb[j - 1] * (1 + 1e-6f)) mono = false; }
    ig_check(std::fabs(sum - 1.0) < 1e-4, "routing weights sum to one");
    ig_check(mono, "routing weights are non-increasing");
    ig_check(ig_rel_l1(wa, wb, std::min(ka, kb)) < 1e-4, "weights match the double reference");
    if (ties) { bool lowest = true; for (int j = 0; j < kb; ++j) lowest = lowest && ib[j] == j; ig_check(lowest, "all-equal logits select experts 0..9"); }
}
static void ig_run() {
    for (int n : {1, 7, 10, 11, 64, 256}) ig_run_case(n, 100u + (unsigned)n, false);
    ig_run_case(64, 1, true);
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("router_top10_parity", rc); }
