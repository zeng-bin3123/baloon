// test_ilu_gdn.cu - GPU invariant tests of the warp-per-column GDN step and the preprocessing kernels.
// Every check is against a double-precision host reference or a LAYER-1 statement (slide, coverage,
// partition, rejection of aliased spans, softplus large branch). Exit code is non-zero on any failure.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/ilu_gdn_kernels.hpp"

using namespace baloon;
namespace gi = baloon::ilu_gdn;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

// Device buffer with canary pads on both sides to catch out-of-bounds writes.
struct Dev {
    static constexpr size_t kPad = 256;  // floats
    float* base = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        CUOK(cudaMalloc(&base, (count + 2 * kPad) * sizeof(float)));
        std::vector<float> c(count + 2 * kPad); uint32_t pat = 0xA5A5A5A5u;
        float f; std::memcpy(&f, &pat, 4); for (auto& x : c) x = f;
        CUOK(cudaMemcpy(base, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
    }
    Dev(const std::vector<float>& h) : Dev(h.size()) { CUOK(cudaMemcpy(ptr(), h.data(), n * 4, cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(base); }
    float* ptr() const { return base + kPad; }
    std::vector<float> get() const { std::vector<float> h(n); CUOK(cudaMemcpy(h.data(), ptr(), n * 4, cudaMemcpyDeviceToHost)); return h; }
    bool canary_ok() const {
        std::vector<float> a(kPad), b(kPad);
        CUOK(cudaMemcpy(a.data(), base, kPad * 4, cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(b.data(), base + kPad + n, kPad * 4, cudaMemcpyDeviceToHost));
        const uint32_t pat = 0xA5A5A5A5u;
        for (size_t i = 0; i < kPad; ++i) { uint32_t x, y; std::memcpy(&x, &a[i], 4); std::memcpy(&y, &b[i], 4); if (x != pat || y != pat) return false; }
        return true;
    }
};

static double max_rel(const std::vector<float>& g, const std::vector<double>& r) {
    double mx = 0, sc = 1e-30;
    for (size_t i = 0; i < r.size(); ++i) { mx = std::fmax(mx, std::fabs((double)g[i] - r[i])); sc = std::fmax(sc, std::fabs(r[i])); }
    return mx / sc;
}
static bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0; }
static void sync() { CUOK(cudaDeviceSynchronize()); }

static std::mt19937 rng(20261103);
static std::vector<float> randv(size_t n, float s = 1.f) { std::normal_distribution<float> nd(0.f, s); std::vector<float> v(n); for (auto& x : v) x = nd(rng); return v; }

static void test_step(int hk, int hv, float scale) {
    const int S = 128;
    auto st = randv((size_t)S * hv * S, 0.1f), q = randv((size_t)hk * S, 0.1f), k = randv((size_t)hk * S, 0.1f), v = randv((size_t)hv * S);
    std::vector<float> gate(hv), beta(hv);
    for (auto& x : gate) x = -std::fabs(randv(1)[0]) - 0.01f;           // decay strictly below 1: every state element changes
    for (auto& x : beta) x = 0.5f + 0.4f * std::tanh(randv(1)[0]);
    std::vector<double> rs(st.begin(), st.end()), ro((size_t)hv * S);
    for (int h = 0; h < hv; ++h) {
        const int src = h % hk; const double dec = std::exp((double)gate[h]);
        for (int j = 0; j < S; ++j) {
            double sk = 0;
            for (int i = 0; i < S; ++i) { double s = rs[((size_t)i * hv + h) * S + j] * dec; rs[((size_t)i * hv + h) * S + j] = s; sk += s * k[src * S + i]; }
            const double d = (v[h * S + j] - sk) * beta[h]; double dot = 0;
            for (int i = 0; i < S; ++i) { double s = rs[((size_t)i * hv + h) * S + j] + k[src * S + i] * d; rs[((size_t)i * hv + h) * S + j] = s; dot += s * q[src * S + i]; }
            ro[(size_t)h * S + j] = dot * scale;
        }
    }
    Dev ds(st), dq(q), dk(k), dv(v), dg(gate), db(beta), dout((size_t)hv * S);
    CHECK(gi::step(ds.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), dg.ptr(), db.ptr(), dout.ptr(), S, hk, hv, scale, nullptr) == gi::kOk);
    sync();
    const auto os = ds.get(), oo = dout.get();
    const double e_state = max_rel(os, rs), e_out = max_rel(oo, ro);
    std::printf("step hk=%d hv=%d scale=%g: state rel=%.3g out rel=%.3g\n", hk, hv, scale, e_state, e_out);
    CHECK(e_state < 1e-4); CHECK(e_out < 1e-4);
    CHECK(ds.canary_ok() && dout.canary_ok() && dq.canary_ok() && dk.canary_ok() && dv.canary_ok());
    // every one of the 128 rows x 128 columns of every head was written (shard/column coverage)
    // A skipped write leaves the element bit-identical to the input although the reference moved it; a lone
    // bit-identical result where the reference barely moved is rounding coincidence and is not a coverage miss.
    size_t missed = 0, bad = 0;
    for (size_t i = 0; i < st.size(); ++i) {
        if (os[i] == st[i] && std::fabs(rs[i] - (double)st[i]) > 1e-4) ++missed;
        if (std::fabs((double)os[i] - rs[i]) > 1e-5 + 1e-4 * std::fabs(rs[i])) ++bad;
    }
    CHECK(missed == 0);
    CHECK(bad == 0);   // elementwise agreement with the double-precision reference
    // determinism: identical input gives bitwise identical output
    Dev ds2(st), dout2((size_t)hv * S);
    CHECK(gi::step(ds2.ptr(), dq.ptr(), dk.ptr(), dv.ptr(), dg.ptr(), db.ptr(), dout2.ptr(), S, hk, hv, scale, nullptr) == gi::kOk);
    sync(); CHECK(bitwise_equal(ds2.get(), os) && bitwise_equal(dout2.get(), oo));
}

static void test_step_rejections() {
    const int hk = 16, hv = 48, S = 128;
    Dev st(randv((size_t)S * hv * S)), q(randv(hk * S)), k(randv(hk * S)), v(randv(hv * S)), g(std::vector<float>(hv, -0.1f)), b(std::vector<float>(hv, 0.5f)), out((size_t)hv * S);
    const auto before = st.get();
    // state aliased with output, output inside state, output over an input: all rejected, nothing launched
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), st.ptr(), S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), st.ptr() + 4096, S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), v.ptr(), S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), q.ptr(), S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    // bad geometry
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), 64, hk, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, 0, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, 10, hv, 1.f, nullptr) == gi::kInvalid);  // 48 % 10 != 0
    CHECK(gi::step(st.ptr(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, hk, 65536, 1.f, nullptr) == gi::kInvalid);
    // misaligned and null spans
    CHECK(gi::step(st.ptr(), (const float*)((const char*)q.ptr() + 2), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    CHECK(gi::step(st.ptr(), nullptr, k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, hk, hv, 1.f, nullptr) == gi::kInvalid);
    sync();
    CHECK(bitwise_equal(st.get(), before));
    // inputs may alias each other (q == k): accepted
    CHECK(gi::step(st.ptr(), k.ptr(), k.ptr(), v.ptr(), g.ptr(), b.ptr(), out.ptr(), S, hk, hv, 1.f, nullptr) == gi::kOk);
    sync();
}

static void test_conv() {
    for (int C : {8192 + 5, 1, 255, 256, 257, 65535}) {
        auto h = randv((size_t)C * 3), w = randv((size_t)C * 4);
        Dev dh(h), dw(w), dx((size_t)C), dout((size_t)C);
        std::vector<std::vector<float>> xs; std::vector<float> hist = h; bool all_ok = true; double worst = 0;
        for (int step = 0; step < 4; ++step) {
            auto x = randv((size_t)C); xs.push_back(x);
            CUOK(cudaMemcpy(dx.ptr(), x.data(), (size_t)C * 4, cudaMemcpyHostToDevice));
            std::vector<double> ro(C); std::vector<float> nh(hist.size());
            for (int c = 0; c < C; ++c) {
                const double acc = (double)hist[c * 3] * w[c * 4] + (double)hist[c * 3 + 1] * w[c * 4 + 1] + (double)hist[c * 3 + 2] * w[c * 4 + 2] + (double)x[c] * w[c * 4 + 3];
                ro[c] = acc / (1.0 + std::exp(-acc));
                nh[c * 3] = hist[c * 3 + 1]; nh[c * 3 + 1] = hist[c * 3 + 2]; nh[c * 3 + 2] = x[c];
            }
            all_ok &= gi::conv_silu(dh.ptr(), dx.ptr(), dw.ptr(), dout.ptr(), C, nullptr) == gi::kOk; sync();
            worst = std::fmax(worst, max_rel(dout.get(), ro));
            CHECK(bitwise_equal(dh.get(), nh));       // history slide is exact: [h1,h2,x]
            hist = nh;
        }
        CHECK(all_ok); CHECK(worst < 1e-5);
        // after three consecutive inputs the history holds exactly those inputs, oldest first
        const auto fin = dh.get(); bool slid = true;
        for (int c = 0; c < C; ++c) slid &= fin[c * 3] == xs[1][c] && fin[c * 3 + 1] == xs[2][c] && fin[c * 3 + 2] == xs[3][c];
        CHECK(slid);
        CHECK(dh.canary_ok() && dout.canary_ok());
        std::printf("conv C=%d worst rel=%.3g\n", C, worst);
    }
    Dev dh(std::vector<float>(12, 0.f)), dw(std::vector<float>(16, 0.f)), dx(4), dout(4);
    CHECK(gi::conv_silu(dh.ptr(), dx.ptr(), dw.ptr(), dout.ptr(), 0, nullptr) == gi::kInvalid);
    CHECK(gi::conv_silu(dh.ptr(), dx.ptr(), dw.ptr(), dout.ptr(), 65536, nullptr) == gi::kInvalid);
    CHECK(gi::conv_silu(dh.ptr(), dx.ptr(), dw.ptr(), dh.ptr(), 1, nullptr) == gi::kInvalid);       // out aliases history
    CHECK(gi::conv_silu(dh.ptr(), dh.ptr() + 2, dw.ptr(), dout.ptr(), 1, nullptr) == gi::kInvalid); // x overlaps history
    // history [p, p+12) directly followed by x at p+12 is disjoint and must be accepted (theory: history_then_raw_disjoint)
    CHECK(gi::conv_silu(dh.ptr(), dh.ptr() + 3, dw.ptr(), dout.ptr(), 1, nullptr) == gi::kOk);
    sync();
}

static void test_norm() {
    for (int rows : {1, 7, 100, 513}) for (int rms = 0; rms < 2; ++rms) {
        const float eps = 1e-6f; auto x = randv((size_t)rows * 128, 2.f);
        std::vector<double> r(x.size());
        for (int i = 0; i < rows; ++i) {
            double s = 0; for (int c = 0; c < 128; ++c) s += (double)x[i * 128 + c] * x[i * 128 + c];
            const double inv = rms ? 1.0 / std::sqrt(s / 128.0 + eps) : 1.0 / std::sqrt(s + eps);
            for (int c = 0; c < 128; ++c) r[i * 128 + c] = x[i * 128 + c] * inv;
        }
        Dev dx(x);
        CHECK(gi::row_norm(dx.ptr(), rows, eps, rms != 0, nullptr) == gi::kOk); sync();
        const auto o = dx.get(); const double e = max_rel(o, r);
        CHECK(e < 1e-5); CHECK(dx.canary_ok());
        if (!rms) for (int i = 0; i < rows; ++i) { double s = 0; for (int c = 0; c < 128; ++c) s += (double)o[i * 128 + c] * o[i * 128 + c]; CHECK(std::fabs(s - 1.0) < 1e-4); }  // unit L2 norm
        else      for (int i = 0; i < rows; ++i) { double s = 0; for (int c = 0; c < 128; ++c) s += (double)o[i * 128 + c] * o[i * 128 + c]; CHECK(std::fabs(s / 128.0 - 1.0) < 1e-4); }  // unit RMS
        std::printf("row_norm rows=%d rms=%d rel=%.3g\n", rows, rms, e);
    }
    Dev dx(128);
    CHECK(gi::row_norm(dx.ptr(), 0, 1e-6f, false, nullptr) == gi::kInvalid);
    CHECK(gi::row_norm(dx.ptr(), 65536, 1e-6f, false, nullptr) == gi::kInvalid);
}

static void test_out_norm() {
    for (int rows : {1, 48, 300}) {
        const float eps = 1e-6f; auto o = randv((size_t)rows * 128), z = randv((size_t)rows * 128, 2.f), w = randv(128, 0.1f);
        for (auto& a : w) a += 1.f;
        std::vector<double> r(o.size());
        for (int i = 0; i < rows; ++i) {
            double s = 0; for (int c = 0; c < 128; ++c) s += (double)o[i * 128 + c] * o[i * 128 + c];
            const double inv = 1.0 / std::sqrt(s / 128.0 + eps);
            for (int c = 0; c < 128; ++c) r[i * 128 + c] = o[i * 128 + c] * inv * w[c] / (1.0 + std::exp(-(double)z[i * 128 + c]));
        }
        Dev dO(o), dZ(z), dW(w), dY(o.size());
        CHECK(gi::out_rms_gate(dO.ptr(), dZ.ptr(), dW.ptr(), dY.ptr(), rows, eps, nullptr) == gi::kOk); sync();
        const double e = max_rel(dY.get(), r); CHECK(e < 1e-5); CHECK(dY.canary_ok());
        std::printf("out_norm rows=%d rel=%.3g\n", rows, e);
    }
    Dev a(128), b(128), c(128), y(128);
    CHECK(gi::out_rms_gate(a.ptr(), b.ptr(), c.ptr(), a.ptr(), 1, 1e-6f, nullptr) == gi::kInvalid);   // y aliases o
    CHECK(gi::out_rms_gate(a.ptr(), b.ptr(), c.ptr(), y.ptr(), 0, 1e-6f, nullptr) == gi::kInvalid);
}

static void test_beta_gate() {
    for (int n : {1, 48, 255, 257, 5000}) {
        auto raw = randv((size_t)n, 4.f); std::vector<double> r(n);
        for (int i = 0; i < n; ++i) r[i] = 1.0 / (1.0 + std::exp(-(double)raw[i]));
        Dev dr(raw), dout((size_t)n);
        CHECK(gi::beta_sigmoid(dr.ptr(), dout.ptr(), n, nullptr) == gi::kOk); sync();
        const auto o = dout.get(); CHECK(max_rel(o, r) < 1e-6); CHECK(dout.canary_ok());
        for (float x : o) CHECK(x >= 0.f && x <= 1.f);
    }
    // gate: softplus selector, large branch returns the input unchanged
    std::vector<float> alpha = {-30.f, -5.f, 0.f, 5.f, 19.9f, 20.f, 20.1f, 25.f, 80.f, 1000.f}, bias(alpha.size(), 0.f), aneg(alpha.size(), -1.f);
    const int n = (int)alpha.size();
    Dev da(alpha), dbias(bias), dan(aneg), dout((size_t)n);
    CHECK(gi::gate_from_alpha(da.ptr(), dbias.ptr(), dan.ptr(), dout.ptr(), n, nullptr) == gi::kOk); sync();
    const auto o = dout.get();
    for (int i = 0; i < n; ++i) {
        const double v = alpha[i], ref = -(v > 20.0 ? v : std::log1p(std::exp(v)));
        CHECK(std::fabs(o[i] - ref) <= 1e-5 * std::fabs(ref) + 1e-30);
        if (v > 20.0) CHECK(o[i] == -alpha[i]);        // exact pass-through above the threshold
        CHECK(std::isfinite(o[i]));
    }
    for (int i = 0; i < n; ++i) CHECK(std::fabs(gi::softplus_sel(alpha[i]) + o[i]) <= 1e-5f * std::fabs(o[i]) + 1e-30f);  // host selector agrees
    // bias and per-head multiplier both enter
    std::vector<float> a2 = {1.f, 2.f, 3.f}, b2 = {0.5f, -2.f, 30.f}, m2 = {-2.f, -0.5f, -1.f};
    Dev d2a(a2), d2b(b2), d2m(m2), d2o(3);
    CHECK(gi::gate_from_alpha(d2a.ptr(), d2b.ptr(), d2m.ptr(), d2o.ptr(), 3, nullptr) == gi::kOk); sync();
    const auto o2 = d2o.get();
    for (int i = 0; i < 3; ++i) { const double v = (double)a2[i] + b2[i], ref = (v > 20 ? v : std::log1p(std::exp(v))) * m2[i]; CHECK(std::fabs(o2[i] - ref) <= 1e-5 * std::fabs(ref) + 1e-12); }
    CHECK(gi::beta_sigmoid(dout.ptr(), dout.ptr(), n, nullptr) == gi::kInvalid);                  // in-place is rejected
    CHECK(gi::gate_from_alpha(da.ptr(), dbias.ptr(), dan.ptr(), da.ptr(), n, nullptr) == gi::kInvalid);
    CHECK(gi::beta_sigmoid(dout.ptr(), dout.ptr() + 100000, 0, nullptr) == gi::kInvalid);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    std::printf("visible GPUs: %d\n", ndev);
    if (ndev < 1) { std::printf("FAIL: no GPU visible\n"); return 2; }
    for (int d = 0; d < ndev; ++d) {
        CUOK(cudaSetDevice(d));
        cudaDeviceProp p; CUOK(cudaGetDeviceProperties(&p, d));
        std::printf("== device %d: %s (warpSize=%d)\n", d, p.name, p.warpSize);
        test_step(16, 48, 1.f); test_step(48, 48, 1.f); test_step(8, 32, 0.088388347648f); test_step(1, 3, 1.f);
        test_step_rejections(); test_conv(); test_norm(); test_out_norm(); test_beta_gate();
    }
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    std::printf(g_fail ? "UNIT-6 GPU TESTS FAILED\n" : "UNIT-6 GPU TESTS PASSED\n");
    return g_fail ? 1 : 0;
}
