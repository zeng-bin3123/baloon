// gr_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// gr_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("gr_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("gr_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_gr();
    std::printf("gr_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("gr_parity FAILED\n"); return 1; }
    std::printf("gr_parity PASSED\n");
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
#include "baloon/gr_gated_residual.hpp"
#include "baloon/f16_bits.hpp"
enum IgActMode { IG_ACT_FP32 = 0, IG_ACT_BF16 = 1 };
struct IgOpts { bool selftest = true; bool fp32 = false; int n = 256, hc = 4; };
static IgOpts ig_parse(int argc, char** argv) {
    IgOpts o; for (int i = 1; i < argc; ++i) { if (!std::strcmp(argv[i], "--fp32")) o.fp32 = true; else if (!std::strcmp(argv[i], "--selftest")) o.selftest = true; }
    return o;
}
static IgActMode ig_select_activation_mode(const IgOpts& o) { return o.fp32 || baloon::refk::gr_fp32_flag().load() ? IG_ACT_FP32 : IG_ACT_BF16; }
static const char* ig_activation_mode_name(IgActMode m) { return m == IG_ACT_FP32 ? "fp32" : "bf16"; }
static uint16_t ig_bf16_bits(float f) { return baloon::f16bits::round_f32_to_bf16_bits(f); }
static float ig_to_bf16(float f) { uint32_t b = (uint32_t)ig_bf16_bits(f) << 16; float r; std::memcpy(&r, &b, 4); return r; }
// The scalar contract of the activation path: bf16 mode rounds every activation to bf16 first, fp32 mode is the identity.
static float ig_scalar_activation_contract(float x, IgActMode m) { return m == IG_ACT_BF16 ? ig_to_bf16(x) : x; }
static double ig_rel_terms(double a, double b, double* num, double* den) { *num = std::fabs(a - b); *den = std::fabs(b); return *den; }
static double ig_rel_diff(double a, double b) { double nu, de; ig_rel_terms(a, b, &nu, &de); return de > 0 ? nu / de : nu; }
// Gated residual write, via the layout helper: out[s][c] = R[s][c] + w[s] * block[c].
static void ig_reference(const float* R, const float* block, const float* w, float* out, int n, int hc) { baloon::gr::write(R, block, w, out, n, hc); }
static void ig_run(const IgOpts& o) {
    const IgActMode mode = ig_select_activation_mode(o);
    std::printf("gr_parity: activation mode %s\n", ig_activation_mode_name(mode));
    ig_check(!std::strcmp(ig_activation_mode_name(IG_ACT_FP32), "fp32") && !std::strcmp(ig_activation_mode_name(IG_ACT_BF16), "bf16"), "mode names");
    ig_check(ig_scalar_activation_contract(1.2345678f, IG_ACT_FP32) == 1.2345678f, "fp32 contract is the identity");
    ig_check(ig_scalar_activation_contract(1.f, IG_ACT_BF16) == 1.f && ig_scalar_activation_contract(1.0000001f, IG_ACT_BF16) == 1.f, "bf16 contract rounds to 8 bits of mantissa");
    ig_check(ig_bf16_bits(1.f) == 0x3F80 && ig_bf16_bits(-2.f) == 0xC000, "bf16 bit patterns");
    ig_check(ig_rel_diff(1.1, 1.0) > 0.0999 && ig_rel_diff(1.1, 1.0) < 0.1001 && ig_rel_diff(0.0, 0.0) == 0.0, "relative difference");
    std::mt19937 rng(13); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int n = o.n, hc = o.hc;
    std::vector<float> R((size_t)hc * n), blk(n), w(hc), a((size_t)hc * n), b((size_t)hc * n);
    for (auto& v : R) v = U(rng); for (auto& v : blk) v = U(rng); for (auto& v : w) v = U(rng);
    ig_reference(R.data(), blk.data(), w.data(), a.data(), n, hc);
    baloon::refk::gr_write(R.data(), blk.data(), w.data(), b.data(), n, hc);
    ig_check(ig_rel_l1(a.data(), b.data(), a.size()) < 1e-6, "gated residual write agrees with the library");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); const IgOpts o = ig_parse(argc, argv); ig_run(o); return ig_finish("gr_parity", rc); }
