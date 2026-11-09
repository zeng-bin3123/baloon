// qsa_parity: legacy invariant body (unchanged, entry renamed) followed by the unit-34 integration layer.
#define main legacy_main
#include "../gpu_unit7/test_ilu_qsa.cu"
#undef main

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
struct IgAlt { const char* name; double rel; };
struct IgDev { int ordinal = -1; bool ok = false; };
static const double ig_REF_R = 1e-5;
static const float ig_kOracle[8] = {0.f, 1.f, -1.f, 2.f, -2.f, 0.5f, -0.5f, 3.f};
static const int ig_kOracleN = 8;
static IgDev ig_open_dev() { IgDev d; int n = 0; if (cudaGetDeviceCount(&n) == cudaSuccess && n > 0 && cudaSetDevice(0) == cudaSuccess) { d.ordinal = 0; d.ok = true; } return d; }
static double ig_as_d(float f) { return (double)f; }
static float ig_as_f(double d) { return (float)d; }
static double ig_pct(double num, double den) { return den != 0 ? 100.0 * num / den : 0.0; }
static void ig_rel_terms(const float* a, const float* b, size_t n, double* num, double* den) { *num = 0; *den = 0; for (size_t i = 0; i < n; ++i) { *num += std::fabs(ig_as_d(a[i]) - ig_as_d(b[i])); *den += std::fabs(ig_as_d(b[i])); } }
static bool ig_require(bool cond, const char* what) { ig_check(cond, what); return cond; }
static void ig_report(const char* name, double rel, double tol) { std::printf("  %-34s rel_l1=%.3e (tol %.1e)\n", name, rel, tol); ig_check(rel <= tol, name); }
static const char* ig_verdict() { return ig_failures == 0 ? "consistent" : "inconsistent"; }
// a*b+c with the product rounded first (no fused multiply-add).
static float ig_mul_add_plain(float a, float b, float c) { volatile float p = a * b; return p + c; }
// Single-query attention: softmax(q.k/sqrt(d)) V over n cells, in double.
static void ig_ref_attend(const float* q, const float* K, const float* V, int n, int d, float* out) {
    std::vector<double> s(n); double mx = -1e300;
    for (int c = 0; c < n; ++c) { double a = 0; for (int i = 0; i < d; ++i) a += (double)q[i] * K[(size_t)c * d + i]; s[c] = a / std::sqrt((double)d); mx = std::max(mx, s[c]); }
    double z = 0; for (auto& v : s) { v = std::exp(v - mx); z += v; }
    for (int i = 0; i < d; ++i) { double a = 0; for (int c = 0; c < n; ++c) a += s[c] / z * V[(size_t)c * d + i]; out[i] = (float)a; }
}
static void ig_ref_gate(float* attn, const float* gate, int n) { for (int i = 0; i < n; ++i) attn[i] *= (float)(1.0 / (1.0 + std::exp(-(double)gate[i]))); }
// Block score = max over the block's cells of q.k (all dots first, then the per-block max).
static void ig_ref_indexer_scores(const float* q, const float* kidx, int n_cells, int block, int d, float* scores) {
    std::vector<float> dots(n_cells); for (int c = 0; c < n_cells; ++c) { float a = 0; for (int i = 0; i < d; ++i) a += q[i] * kidx[(size_t)c * d + i]; dots[c] = a; }
    for (int b = 0; b * block < n_cells; ++b) { float m = -INFINITY; for (int c = b * block; c < std::min(n_cells, (b + 1) * block); ++c) m = std::max(m, dots[c]); scores[b] = m; }
}
// Pooled block keys: raw sums, then means.
static void ig_ref_pooled_raw(const float* kidx, int n_cells, int block, int d, float* out) {
    const int nb = (n_cells + block - 1) / block; std::fill(out, out + (size_t)nb * d, 0.f);
    for (int c = 0; c < n_cells; ++c) for (int i = 0; i < d; ++i) out[(size_t)(c / block) * d + i] += kidx[(size_t)c * d + i];
}
static void ig_ref_pooled(const float* kidx, int n_cells, int block, int d, float* out) {
    ig_ref_pooled_raw(kidx, n_cells, block, d, out); const int nb = (n_cells + block - 1) / block;
    for (int b = 0; b < nb; ++b) { const int cnt = std::min(block, n_cells - b * block); for (int i = 0; i < d; ++i) out[(size_t)b * d + i] /= (float)cnt; }
}
struct IgRefCache { std::vector<float> pooled; int n_cells = 0, block = 0, d = 0;
    void build(const float* kidx, int nc, int bl, int dd) { n_cells = nc; block = bl; d = dd; pooled.assign((size_t)((nc + bl - 1) / bl) * dd, 0.f); ig_ref_pooled(kidx, nc, bl, dd, pooled.data()); } };
static void ig_ref_rms_norm(const float* x, const float* w, float* out, int n, float eps) { double s = 0; for (int i = 0; i < n; ++i) s += (double)x[i] * x[i]; const double r = 1.0 / std::sqrt(s / n + eps); for (int i = 0; i < n; ++i) out[i] = (float)(x[i] * r * w[i]); }
static void ig_ref_rope_neox(float* x, int n_rot, const float* cs, const float* sn) { const int h = n_rot / 2; for (int i = 0; i < h; ++i) { const float a = x[i], b = x[i + h]; x[i] = a * cs[i] - b * sn[i]; x[i + h] = b * cs[i] + a * sn[i]; } }
static std::vector<int> ig_ref_select(const float* scores, int nb, int k) {
    std::vector<int> o(nb); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
    o.resize(std::min(std::max(k, 0), nb)); return o;
}
static void ig_run() {
    namespace rk = baloon::refk;
    const IgDev dev = ig_open_dev(); ig_require(dev.ok, "a GPU device opens");
    std::mt19937 rng(23); std::uniform_real_distribution<float> U(-1.f, 1.f);
    const int n_cells = 77, block = 8, d = 16, nb = (n_cells + block - 1) / block;
    std::vector<float> q(d), K((size_t)n_cells * d), V((size_t)n_cells * d), s1(nb), s2(nb);
    for (auto& v : q) v = U(rng); for (auto& v : K) v = U(rng); for (auto& v : V) v = U(rng);
    ig_ref_indexer_scores(q.data(), K.data(), n_cells, block, d, s1.data()); rk::qsa_block_scores(q.data(), K.data(), n_cells, block, d, s2.data());
    double nu, de; ig_rel_terms(s1.data(), s2.data(), nb, &nu, &de); ig_report("indexer block scores", de > 0 ? nu / de : nu, ig_REF_R);
    const std::vector<int> a = ig_ref_select(s1.data(), nb, 4), b = rk::qsa_block_topk(s2.data(), nb, 4);
    ig_require(a == b, "top-k block selection matches (ties to the lower index)");
    IgRefCache rc; rc.build(K.data(), n_cells, block, d); std::vector<float> raw((size_t)nb * d); ig_ref_pooled_raw(K.data(), n_cells, block, d, raw.data());
    double e = 0; for (int i = 0; i < d; ++i) e = std::max(e, (double)std::fabs(raw[i] / block - rc.pooled[i]));
    ig_report("pooled = raw / count", e, 1e-6);
    ig_require(std::fabs(rc.pooled[(size_t)(nb - 1) * d] * (n_cells - (nb - 1) * block) - raw[(size_t)(nb - 1) * d]) < 1e-4, "the short last block divides by its own cell count");
    std::vector<float> o1(d), one(d); ig_ref_attend(q.data(), K.data(), V.data(), 1, d, one.data());
    ig_report("single cell attends to its value", ig_rel_l1(one.data(), V.data(), d), 1e-6);
    std::vector<float> zk((size_t)n_cells * d, 0.f), mean(d, 0.f); ig_ref_attend(q.data(), zk.data(), V.data(), n_cells, d, o1.data());
    for (int c = 0; c < n_cells; ++c) for (int i = 0; i < d; ++i) mean[i] += V[(size_t)c * d + i] / n_cells;
    ig_report("equal scores average the values", ig_rel_l1(o1.data(), mean.data(), d), 1e-5);
    std::vector<float> g1(ig_kOracle, ig_kOracle + ig_kOracleN), g2 = g1, gate(ig_kOracleN, 0.f), half(ig_kOracleN);
    ig_ref_gate(g1.data(), gate.data(), ig_kOracleN); for (int i = 0; i < ig_kOracleN; ++i) half[i] = 0.5f * ig_kOracle[i];
    ig_report("zero gate halves the attention", ig_rel_l1(g1.data(), half.data(), ig_kOracleN), 1e-6);
    std::vector<float> gl(ig_kOracleN); for (int i = 0; i < ig_kOracleN; ++i) gl[i] = 0.3f * (float)i - 1.f;
    std::vector<float> g3 = g2; ig_ref_gate(g2.data(), gl.data(), ig_kOracleN); rk::native_qsa_gate_apply(g3.data(), gl.data(), ig_kOracleN);
    ig_report("gate matches the library", ig_rel_l1(g2.data(), g3.data(), ig_kOracleN), 1e-6);
    const int hd = 16; std::vector<float> w(hd, 1.f), x(hd), n1(hd), n2(hd); for (auto& v : x) v = U(rng);
    ig_ref_rms_norm(x.data(), w.data(), n1.data(), hd, 1e-6f); rk::native_qsa_rms_norm_weighted(x.data(), w.data(), n2.data(), 1, hd, 1e-6f);
    ig_report("rms norm matches the library", ig_rel_l1(n1.data(), n2.data(), hd), 1e-5);
    std::vector<float> cs(hd / 2), sn(hd / 2); for (int i = 0; i < hd / 2; ++i) { cs[i] = std::cos(0.3f * (i + 1)); sn[i] = std::sin(0.3f * (i + 1)); }
    std::vector<float> r1 = x, r2 = x; ig_ref_rope_neox(r1.data(), hd, cs.data(), sn.data()); rk::native_rope_apply(r2.data(), 1, hd, hd, cs.data(), sn.data());
    ig_report("neox rope matches the library", ig_rel_l1(r1.data(), r2.data(), hd), 1e-6);
    const IgAlt alts[2] = {{"plain mul-add", std::fabs(ig_mul_add_plain(1.0000001f, 3.f, -3.f))}, {"percent", ig_pct(1, 4)}};
    ig_require(alts[1].rel == 25.0 && alts[0].rel >= 0.0, "alternate-contract helpers");
    ig_require(ig_as_f(ig_as_d(1.5f)) == 1.5f, "float/double conversions are exact for floats");
    std::printf("qsa_parity: port layer %s\n", ig_verdict());
}
int main() { int rc = ig_legacy([&] { return legacy_main(); }); ig_run(); return ig_finish("qsa_parity", rc); }
