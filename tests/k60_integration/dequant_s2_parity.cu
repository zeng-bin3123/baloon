// dequant_s2_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// dequant_s2_parity: runs selected invariant checks of k14gpu_misc/test_misc.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k14gpu_misc/test_misc.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("dequant_s2_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("dequant_s2_parity: cannot open GPU 0\n"); return 2; }
    cpu_features();
    test_dequant();
    test_s2();
    std::printf("dequant_s2_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("dequant_s2_parity FAILED\n"); return 1; }
    std::printf("dequant_s2_parity PASSED\n");
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
static const int ig_QK = baloon::refk::kS2Group;   // values per scale group
static const int ig_kScales = 40;                  // scale groups of a 2560-wide row
static void ig_run() {
    namespace rk = baloon::refk;
    std::mt19937 rng(3); std::uniform_real_distribution<float> U(0.01f, 2.f);
    const int n = ig_kScales * ig_QK;
    ig_check(n == 2560 && rk::s2_row_bytes(n) == 640, "a 2560-wide S2 row is 640 code bytes + 40 scales");
    std::vector<uint8_t> codes(rk::s2_row_bytes(n)); for (auto& b : codes) b = (uint8_t)(rng() & 255);
    std::vector<float> sc(ig_kScales); for (auto& s : sc) s = U(rng);
    rk::Codebook cb; std::vector<float> out(n);
    rk::dequant_s2_ref(codes.data(), sc.data(), cb, out.data(), n);
    bool ok = true;
    for (int e = 0; e < n; ++e) {
        const int c = (codes[e >> 2] >> ((e & 3) * 2)) & 3;
        const float lv = c == 0 ? -3.f : c == 1 ? -1.f : c == 2 ? 1.f : 3.f;
        if (out[e] != lv * sc[e / ig_QK]) ok = false;
    }
    ig_check(ok, "every value is level[code] * scale[group]");
    bool in_set = true; for (int e = 0; e < n; ++e) { const float a = std::fabs(out[e]) / sc[e / ig_QK]; if (std::fabs(a - 1.f) > 1e-5f && std::fabs(a - 3.f) > 1e-5f) in_set = false; }
    ig_check(in_set, "magnitudes are 1 or 3 times the group scale");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("dequant_s2_parity", rc); }
