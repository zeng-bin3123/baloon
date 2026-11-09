// s2_gemv_q8_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// s2_gemv_q8_parity: runs selected invariant checks of k14gpu_misc/test_misc.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k14gpu_misc/test_misc.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("s2_gemv_q8_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("s2_gemv_q8_parity: cannot open GPU 0\n"); return 2; }
    cpu_features();
    test_s2();
    std::printf("s2_gemv_q8_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("s2_gemv_q8_parity FAILED\n"); return 1; }
    std::printf("s2_gemv_q8_parity PASSED\n");
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
    std::mt19937 rng(21); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int G = rk::kS2Group, n_in = 10 * G, n_out = 5, gs = n_in / G;
    std::vector<uint8_t> codes((size_t)n_out * rk::s2_row_bytes(n_in)); for (auto& b : codes) b = (uint8_t)(rng() & 255);
    std::vector<float> sc((size_t)n_out * gs); for (auto& s : sc) s = 0.1f + std::fabs(U(rng));
    std::vector<float> x(n_in); for (auto& v : x) v = U(rng);
    std::vector<int8_t> xq(n_in); std::vector<float> xs(gs), xd(n_in);
    for (int g = 0; g < gs; ++g) {
        float am = 0; for (int e = g * G; e < (g + 1) * G; ++e) am = std::max(am, std::fabs(x[e]));
        xs[g] = am / 127.f; const float inv = am > 0 ? 127.f / am : 0.f;
        for (int e = g * G; e < (g + 1) * G; ++e) { int q = (int)std::nearbyint(x[e] * inv); q = std::max(-127, std::min(127, q)); xq[e] = (int8_t)q; xd[e] = xq[e] * xs[g]; }
    }
    rk::Codebook cb; std::vector<float> yq(n_out), yf(n_out), yx(n_out);
    rk::s2_gemv_q8(codes.data(), sc.data(), cb, xq.data(), xs.data(), yq.data(), n_in, n_out);
    rk::s2_gemv(codes.data(), sc.data(), cb, xd.data(), yf.data(), n_in, n_out);
    ig_check(ig_rel_l1(yq.data(), yf.data(), n_out) < 1e-4, "integer-activation gemv equals the float gemv on the dequantised activation");
    rk::s2_gemv(codes.data(), sc.data(), cb, x.data(), yx.data(), n_in, n_out);
    ig_check(ig_rel_l1(yq.data(), yx.data(), n_out) < 0.05, "quantisation error against the exact activation stays small");
    bool inr = true; for (auto q : xq) if (q < -127) inr = false;
    ig_check(inr, "activation codes stay in [-127, 127]");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("s2_gemv_q8_parity", rc); }
