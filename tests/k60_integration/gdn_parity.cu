// gdn_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// gdn_parity: runs selected invariant checks of k32_gpu_kernels/test_kg.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k32_gpu_kernels/test_kg.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("gdn_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("gdn_parity: cannot open GPU 0\n"); return 2; }
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_gdn();
    std::printf("gdn_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("gdn_parity FAILED\n"); return 1; }
    std::printf("gdn_parity PASSED\n");
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
static float ig_sigmoid_f(float x) { return 1.f / (1.f + std::exp(-x)); }
// Depthwise causal conv for one token, written window-first: window = [hist..., x]; hist slides left.
static void ig_ref_conv(float* hist, const float* x, const float* w, float* out, int channels, int taps) {
    const int ht = taps - 1;
    for (int c = 0; c < channels; ++c) {
        std::vector<float> win(hist + (size_t)c * ht, hist + (size_t)c * ht + ht); win.push_back(x[c]);
        float a = 0.f; for (int t = 0; t < taps; ++t) a += w[(size_t)c * taps + t] * win[t];
        out[c] = a;
        for (int t = 0; t < ht; ++t) hist[(size_t)c * ht + t] = win[t + 1];
    }
}
// Delta-rule step in outer-product form: S <- e^g S; S <- S + k (beta (v - S^T k))^T; out = scale S^T q.
static void ig_ref_step(float* S, const float* q, const float* k, const float* v, float g, float beta, float scale, float* out, int dk, int dv) {
    const float dec = std::exp(g);
    for (int i = 0; i < dk * dv; ++i) S[i] *= dec;
    std::vector<double> delta(dv);
    for (int j = 0; j < dv; ++j) { double kv = 0; for (int i = 0; i < dk; ++i) kv += (double)S[(size_t)i * dv + j] * k[i]; delta[j] = beta * ((double)v[j] - kv); }
    for (int i = 0; i < dk; ++i) for (int j = 0; j < dv; ++j) S[(size_t)i * dv + j] += (float)(k[i] * delta[j]);
    for (int j = 0; j < dv; ++j) { double o = 0; for (int i = 0; i < dk; ++i) o += (double)S[(size_t)i * dv + j] * q[i]; out[j] = (float)(o * scale); }
}
static void ig_run() {
    std::mt19937 rng(11); std::uniform_real_distribution<float> U(-1.f, 1.f);
    ig_check(ig_sigmoid_f(0.f) == 0.5f, "sigmoid(0)=1/2");
    const int C = 24, taps = 4, ht = taps - 1;
    std::vector<float> w(C * taps), h1(C * ht, 0.f), h2(C * ht, 0.f), x(C), o1(C), o2(C);
    for (auto& v : w) v = U(rng);
    float worst = 0.f;
    for (int step = 0; step < 6; ++step) {
        for (auto& v : x) v = U(rng);
        ig_ref_conv(h1.data(), x.data(), w.data(), o1.data(), C, taps);
        baloon::refk::gdn_conv_step(h2.data(), x.data(), w.data(), o2.data(), C, taps);
        worst = std::max(worst, (float)ig_rel_l1(o1.data(), o2.data(), C));
    }
    ig_check(worst < 1e-5f && ig_rel_l1(h1.data(), h2.data(), h1.size()) < 1e-6, "conv step matches the library over a sequence");
    // a filter that is 1 on the newest tap returns the current input
    std::vector<float> wid(C * taps, 0.f); for (int c = 0; c < C; ++c) wid[c * taps + ht] = 1.f;
    std::fill(h1.begin(), h1.end(), 0.5f); for (auto& v : x) v = U(rng);
    ig_ref_conv(h1.data(), x.data(), wid.data(), o1.data(), C, taps);
    ig_check(ig_rel_l1(o1.data(), x.data(), C) == 0.0, "newest-tap filter is the identity");
    const int dk = 16, dv = 12;
    std::vector<float> S1(dk * dv), q(dk), k(dk), v(dv), oa(dv), ob(dv);
    for (auto& e : S1) e = U(rng); for (auto& e : q) e = U(rng); for (auto& e : k) e = U(rng); for (auto& e : v) e = U(rng);
    std::vector<float> S2 = S1, S0 = S1;
    ig_ref_step(S1.data(), q.data(), k.data(), v.data(), -0.3f, 0.7f, 0.25f, oa.data(), dk, dv);
    baloon::refk::gdn_step_host(S2.data(), q.data(), k.data(), v.data(), -0.3f, 0.7f, 0.25f, ob.data(), dk, dv);
    ig_check(ig_rel_l1(oa.data(), ob.data(), dv) < 1e-4 && ig_rel_l1(S1.data(), S2.data(), S1.size()) < 1e-4, "delta-rule step matches the library");
    S1 = S0; ig_ref_step(S1.data(), q.data(), k.data(), v.data(), -0.3f, 0.f, 1.f, oa.data(), dk, dv);
    std::vector<float> Sd(S0.size()); for (size_t i = 0; i < Sd.size(); ++i) Sd[i] = S0[i] * std::exp(-0.3f);
    ig_check(ig_rel_l1(S1.data(), Sd.data(), Sd.size()) < 1e-6, "beta=0 only decays the state");
    std::fill(k.begin(), k.end(), 0.f); k[0] = 1.f; S1 = S0;
    ig_ref_step(S1.data(), q.data(), k.data(), v.data(), 0.f, 1.f, 1.f, oa.data(), dk, dv);
    ig_check(ig_rel_l1(S1.data(), v.data(), dv) < 1e-6, "unit key with beta=1 overwrites its row with v");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("gdn_parity", rc); }
