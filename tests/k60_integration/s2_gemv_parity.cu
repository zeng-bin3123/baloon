// s2_gemv_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// s2_gemv_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("s2_gemv_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("s2_gemv_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_s2();
    std::printf("s2_gemv_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("s2_gemv_parity FAILED\n"); return 1; }
    std::printf("s2_gemv_parity PASSED\n");
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
static const int ig_QK = baloon::refk::kS2Group;
static const int ig_kScales = 40;
static void ig_run() {
    namespace rk = baloon::refk;
    std::mt19937 rng(9); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int n_in = ig_kScales * ig_QK, n_out = 6;
    std::vector<uint8_t> codes((size_t)n_out * rk::s2_row_bytes(n_in)); for (auto& b : codes) b = (uint8_t)(rng() & 255);
    std::vector<float> sc((size_t)n_out * ig_kScales); for (auto& s : sc) s = 0.05f + 0.5f * std::fabs(U(rng));
    std::vector<float> x(n_in); for (auto& v : x) v = U(rng);
    rk::Codebook cb;
    std::vector<float> y0(n_out), y1(n_out), y2(n_out), yd(n_out);
    rk::s2_gemv(codes.data(), sc.data(), cb, x.data(), y0.data(), n_in, n_out);
    rk::s2_gemv_fast(codes.data(), sc.data(), cb, x.data(), y1.data(), n_in, n_out);
    rk::s2_gemv_quads(codes.data(), sc.data(), cb, x.data(), y2.data(), n_in, n_out);
    ig_check(std::memcmp(y0.data(), y1.data(), n_out * sizeof(float)) == 0, "table variant has identical summation order to the reference");
    ig_check(ig_rel_l1(y2.data(), y0.data(), n_out) < 1e-4, "quad variant agrees up to fp32 reassociation");
    for (int r = 0; r < n_out; ++r) {
        std::vector<float> row(n_in); const uint8_t* cr = codes.data() + (size_t)r * rk::s2_row_bytes(n_in);
        std::vector<float> rs(sc.begin() + (size_t)r * ig_kScales, sc.begin() + (size_t)(r + 1) * ig_kScales);
        rk::dequant_s2_ref(cr, rs.data(), cb, row.data(), n_in);
        double a = 0; for (int i = 0; i < n_in; ++i) a += (double)row[i] * x[i]; yd[r] = (float)a;
    }
    ig_check(ig_rel_l1(y0.data(), yd.data(), n_out) < 1e-4, "gemv equals dot of the dequantised rows");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("s2_gemv_parity", rc); }
