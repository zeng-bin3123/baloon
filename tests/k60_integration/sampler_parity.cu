// sampler_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
// sampler_parity: runs selected invariant checks of k14gpu_misc/test_misc.cu on the device (its main is renamed; --selftest is accepted).
#define main orig_test_main
#include "../k14gpu_misc/test_misc.cu"
#undef main
static int legacy_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("sampler_parity: no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("sampler_parity: cannot open GPU 0\n"); return 2; }
    cpu_features();
    test_sampler();
    std::printf("sampler_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("sampler_parity FAILED\n"); return 1; }
    std::printf("sampler_parity PASSED\n");
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
#include "baloon/sampler_chain_contracts.hpp"
static const int ig_hist_len = 64;
// Reference cut: smallest prefix of the (descending) probabilities whose mass reaches top_p, never fewer than min_keep.
static size_t ig_reference_cut(const std::vector<double>& p, double top_p, size_t min_keep) {
    double c = 0; size_t cut = p.size();
    for (size_t i = 0; i < p.size(); ++i) { c += p[i]; if (c >= top_p) { cut = i + 1; break; } }
    return baloon::sampler::cut_after(cut, min_keep, p.size());
}
// Reference greedy pick after repetition / frequency / presence penalties over the last ig_hist_len tokens.
static uint32_t ig_reference_pick(const std::vector<float>& logits, const std::vector<uint32_t>& hist, float rep, float freq, float pres) {
    std::vector<uint32_t> cnt(logits.size(), 0);
    const size_t start = hist.size() > (size_t)ig_hist_len ? hist.size() - ig_hist_len : 0;
    for (size_t i = start; i < hist.size(); ++i) if (hist[i] < cnt.size()) ++cnt[hist[i]];
    uint32_t best = 0; float bv = -INFINITY;
    for (uint32_t t = 0; t < logits.size(); ++t) {
        float l = logits[t];
        if (cnt[t]) { l = l <= 0.f ? l * rep : l / rep; l -= (float)cnt[t] * freq + pres; }
        if (l > bv) { bv = l; best = t; }
    }
    return best;
}
static void ig_run_one(unsigned seed) {
    namespace sp = baloon::sampler;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> U(-3.f, 3.f);
    const uint32_t V = 200; std::vector<float> lg(V); for (auto& v : lg) v = U(rng);
    std::vector<uint32_t> hist(100); for (auto& t : hist) t = rng() % V;
    const float rep = 1.3f, freq = 0.2f, pres = 0.4f;
    std::vector<uint32_t> cnt(V, 0);
    const size_t st = sp::window_start(ig_hist_len, hist.size());
    for (size_t i = st; i < hist.size(); ++i) ++cnt[hist[i]];
    std::vector<float> pen(V); for (uint32_t t = 0; t < V; ++t) pen[t] = sp::apply_penalties(lg[t], cnt[t], rep, freq, pres);
    ig_check(sp::greedy_argmax(pen.data(), V) == ig_reference_pick(lg, hist, rep, freq, pres), "penalised greedy pick matches the reference");
    ig_check(sp::window_len(ig_hist_len, hist.size()) == (size_t)ig_hist_len && sp::window_len(ig_hist_len, 10) == 10, "history window length");
}
static void ig_run() {
    namespace sp = baloon::sampler;
    for (unsigned s = 0; s < 16; ++s) ig_run_one(70u + s);
    std::vector<float> tie(50, 1.f); ig_check(sp::greedy_argmax(tie.data(), 50) == 0, "greedy ties go to the lowest index");
    ig_check(sp::apply_penalties(2.f, 0, 1.5f, 1.f, 1.f) == 2.f, "count 0 leaves the logit alone");
    ig_check(sp::apply_penalties(2.f, 1, 2.f, 0.f, 0.f) == 1.f && sp::apply_penalties(-2.f, 1, 2.f, 0.f, 0.f) == -4.f, "repetition divides positives and multiplies non-positives");
    std::vector<double> p = {0.5, 0.25, 0.15, 0.07, 0.03};
    ig_check(ig_reference_cut(p, 0.7, 1) == 2 && ig_reference_cut(p, 0.7, 4) == 4 && ig_reference_cut(p, 2.0, 1) == 5, "top-p cut honours min_keep and the table size");
    ig_check(sp::effective_top_k(1000) == sp::kMaxTopK && sp::effective_top_k(5) == 5, "top-k clamps to the kernel maximum");
}
int main(int argc, char** argv) { int rc = ig_legacy([&] { return legacy_main(argc, argv); }); ig_run(); return ig_finish("sampler_parity", rc); }
