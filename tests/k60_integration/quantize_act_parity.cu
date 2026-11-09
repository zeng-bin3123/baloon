// quantize_act_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// quantize_act_parity: runs selected invariant checks of k14gpu_misc/test_misc.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k14gpu_misc/test_misc.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("quantize_act_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("quantize_act_parity: cannot open GPU 0\n"); return 2; }
    cpu_features();
    test_quantize_act();
    std::printf("quantize_act_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("quantize_act_parity FAILED\n"); return 1; }
    std::printf("quantize_act_parity PASSED\n");
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
#include "baloon/f16_bits.hpp"
static float ig_f16_bits_to_f32(uint16_t h) { return baloon::f16bits::f32_from_f16(h); }
static uint16_t ig_f32_to_f16_bits(float f) { return baloon::f16bits::f16_from_f32(f); }
static int ig_nearest_int_host(float f) { return (int)std::nearbyint(f); }
struct IgQ8 { std::vector<int8_t> q; std::vector<uint16_t> d; };
// Q8_0: per-32 block, fp16 scale d = amax/127, q = nearest(x/d).
static IgQ8 ig_reference_q8_0(const float* x, int n) {
    IgQ8 r; r.q.resize(n); r.d.resize(n / 32);
    for (int b = 0; b < n / 32; ++b) {
        float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i]));
        const float d = am / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        r.d[b] = ig_f32_to_f16_bits(d);
        for (int i = 0; i < 32; ++i) r.q[b * 32 + i] = (int8_t)ig_nearest_int_host(x[b * 32 + i] * id);
    }
    return r;
}
struct IgQ8K { float d; int8_t q[256]; int16_t bsums[16]; };
// Q8_K: 256-wide block, d = 1/iscale with iscale = -128/(signed value of largest magnitude), 16 partial sums.
static IgQ8K ig_reference_q8_K(const float* x) {
    IgQ8K r{}; float amax = 0, mx = 0;
    for (int i = 0; i < 256; ++i) { const float a = std::fabs(x[i]); if (a > amax) { amax = a; mx = x[i]; } }
    if (amax == 0.f) { r.d = 0.f; return r; }
    const float is = -128.f / mx;
    for (int i = 0; i < 256; ++i) { const int v = ig_nearest_int_host(is * x[i]); r.q[i] = (int8_t)std::min(127, v); }
    for (int j = 0; j < 16; ++j) { int s = 0; for (int l = 0; l < 16; ++l) s += r.q[j * 16 + l]; r.bsums[j] = (int16_t)s; }
    r.d = 1.f / is; return r;
}
static void ig_run_case(int n, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-3.f, 3.f);
    std::vector<float> x(n); for (auto& v : x) v = U(rng);
    IgQ8 r = ig_reference_q8_0(x.data(), n);
    bool err_ok = true, rng_ok = true;
    for (int i = 0; i < n; ++i) {
        const float d = ig_f16_bits_to_f32(r.d[i / 32]);
        if (r.q[i] < -127) rng_ok = false;
        if (std::fabs(x[i] - d * r.q[i]) > 0.5f * d * 1.01f + 2e-3f * std::fabs(x[i]) + 1e-6f) err_ok = false;
    }
    ig_check(rng_ok, "Q8_0 codes stay in [-127,127]");
    ig_check(err_ok, "Q8_0 reconstruction error is at most half a step");
    std::vector<uint8_t> lib((size_t)(n / 32) * baloon::refk::kQ81Bytes);
    baloon::refk::quantize_q8_1_rows(x.data(), 1, n, lib.data());
    bool same_d = true, near_q = true;
    for (int b = 0; b < n / 32; ++b) {
        uint16_t dh; std::memcpy(&dh, lib.data() + (size_t)b * baloon::refk::kQ81Bytes, 2);
        if (dh != r.d[b]) same_d = false;
        for (int i = 0; i < 32; ++i) if (std::abs((int)(int8_t)lib[(size_t)b * baloon::refk::kQ81Bytes + 4 + i] - (int)r.q[b * 32 + i]) > 1) near_q = false;
    }
    ig_check(same_d, "scales equal the library's Q8_1 scales");
    ig_check(near_q, "codes equal the library's up to a tie");
}
static void ig_run_case_k(unsigned seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-5.f, 5.f);
    std::vector<float> x(256); for (auto& v : x) v = U(rng);
    IgQ8K r = ig_reference_q8_K(x.data());
    bool err_ok = true, sums_ok = true;
    for (int i = 0; i < 256; ++i) { const float lim = (r.q[i] == 127 ? 1.0f : 0.5f) * std::fabs(r.d) * 1.01f + 1e-5f; if (std::fabs(x[i] - r.d * r.q[i]) > lim) err_ok = false; }   // only the +127 clamp may cost a full step
    for (int j = 0; j < 16; ++j) { int s = 0; for (int l = 0; l < 16; ++l) s += r.q[j * 16 + l]; if (s != r.bsums[j]) sums_ok = false; }
    int am = 0; for (int i = 1; i < 256; ++i) if (std::fabs(x[i]) > std::fabs(x[am])) am = i;
    ig_check(err_ok, "Q8_K reconstruction error is at most half a step (one step where the +127 clamp applies)");
    ig_check(sums_ok, "Q8_K partial sums equal the sums of the codes");
    ig_check(r.q[am] == -128, "the largest-magnitude element maps to -128");
}
static void ig_run() {
    ig_check(ig_f32_to_f16_bits(1.f) == 0x3C00 && ig_f16_bits_to_f32(0x3C00) == 1.f, "f16 bit conversions");
    ig_check(ig_nearest_int_host(2.5f) == 2 && ig_nearest_int_host(3.5f) == 4 && ig_nearest_int_host(-2.5f) == -2, "nearest int rounds half to even");
    for (int n : {32, 64, 256, 1024}) ig_run_case(n, 40u + (unsigned)n);
    for (unsigned s = 0; s < 8; ++s) ig_run_case_k(900u + s);
    float z[256] = {0}; IgQ8K zk = ig_reference_q8_K(z); ig_check(zk.d == 0.f, "all-zero block has zero scale");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("quantize_act_parity", rc); }
