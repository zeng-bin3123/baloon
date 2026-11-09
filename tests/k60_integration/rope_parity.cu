// rope_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// rope_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("rope_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("rope_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_rope_qsa_ple();
    std::printf("rope_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("rope_parity FAILED\n"); return 1; }
    std::printf("rope_parity PASSED\n");
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
    std::mt19937 rng(5); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int hd = 16, half = hd / 2;
    auto cs_sn = [&](float theta, std::vector<float>& cs, std::vector<float>& sn) { cs.assign(half, 0.f); sn.assign(half, 0.f); for (int i = 0; i < half; ++i) { cs[i] = std::cos(theta * (i + 1)); sn[i] = std::sin(theta * (i + 1)); } };
    std::vector<float> x(hd); for (auto& v : x) v = U(rng);
    std::vector<float> cs, sn, y = x;
    cs_sn(0.f, cs, sn); baloon::refk::native_rope_apply(y.data(), 1, hd, hd, cs.data(), sn.data());
    ig_check(ig_rel_l1(y.data(), x.data(), hd) == 0.0, "angle 0 is the identity");
    cs_sn(0.37f, cs, sn); y = x; baloon::refk::native_rope_apply(y.data(), 1, hd, hd, cs.data(), sn.data());
    double n0 = 0, n1 = 0; for (int i = 0; i < hd; ++i) { n0 += (double)x[i] * x[i]; n1 += (double)y[i] * y[i]; }
    ig_check(std::fabs(n0 - n1) < 1e-5 * n0, "rotation preserves the norm");
    std::vector<float> z = x, ca, sa, cb, sb, cab, sab;
    cs_sn(0.2f, ca, sa); cs_sn(0.15f, cb, sb); cs_sn(0.35f, cab, sab);
    baloon::refk::native_rope_apply(z.data(), 1, hd, hd, ca.data(), sa.data());
    baloon::refk::native_rope_apply(z.data(), 1, hd, hd, cb.data(), sb.data());
    std::vector<float> w = x; baloon::refk::native_rope_apply(w.data(), 1, hd, hd, cab.data(), sab.data());
    ig_check(ig_rel_l1(z.data(), w.data(), hd) < 1e-5, "rotations compose: R(a)R(b)=R(a+b)");
    // partial rotation leaves the tail untouched
    std::vector<float> p = x; std::vector<float> c8(4), s8(4); for (int i = 0; i < 4; ++i) { c8[i] = std::cos(0.5f); s8[i] = std::sin(0.5f); }
    baloon::refk::native_rope_apply(p.data(), 1, hd, 8, c8.data(), s8.data());
    ig_check(std::memcmp(p.data() + 8, x.data() + 8, 8 * sizeof(float)) == 0, "dims beyond n_rot are untouched");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("rope_parity", rc); }
