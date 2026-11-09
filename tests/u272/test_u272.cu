// Unit u2726a197a5 tests: invariants of NativeGdnStep / NativeGdnPreprocess (addressing cover, head pairing,
// state/output byte sizes, span validity + overlap semantics, count guards, history slide, softplus branch,
// norm geometry) checked against double-precision host references on the device.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "baloon/ilu_native_gdn.hpp"

using namespace baloon::ilu_nat;
static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s @%d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

template <class T> static T* up(const std::vector<T>& h) {
    T* d = nullptr; cudaMalloc(&d, h.size() * sizeof(T)); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n); cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return h;
}
static double rel(const std::vector<float>& g, const std::vector<double>& r) {
    double mx = 0, sc = 1e-30;
    for (size_t i = 0; i < r.size(); ++i) { mx = std::fmax(mx, std::fabs(g[i] - r[i])); sc = std::fmax(sc, std::fabs(r[i])); }
    return mx / sc;
}
static std::mt19937 rng(1234);
static std::vector<float> rnd(size_t n, float a, float b) {
    std::uniform_real_distribution<float> d(a, b); std::vector<float> v(n); for (auto& x : v) x = d(rng); return v;
}

static void test_span_math() {
    using step_impl::overlap; using step_impl::valid_span;
    // overlap: symmetric, adjacent disjoint, self-overlap nonempty, empty inside reported, empty at edge not
    for (int t = 0; t < 2000; ++t) {
        uint64_t a = rng() % 100, an = rng() % 50, b = rng() % 100, bn = rng() % 50;
        CHECK(overlap(a, an, b, bn) == overlap(b, bn, a, an), "overlap symmetry");
        if (a + an <= b) CHECK(!overlap(a, an, b, bn), "disjoint_of_le");
    }
    CHECK(!overlap(10, 5, 15, 7), "adjacent spans must be disjoint");
    CHECK(overlap(10, 5, 10, 5), "nonempty span overlaps itself");
    CHECK(overlap(12, 0, 10, 5), "empty span strictly inside is reported (no emptiness guard)");
    CHECK(!overlap(10, 0, 10, 5) && !overlap(15, 0, 10, 5), "empty span at the edge is not reported");
    // valid_span
    CHECK(valid_span(8, 16), "plain valid");
    CHECK(!valid_span(0, 16), "null rejected");
    CHECK(!valid_span(6, 16), "misaligned rejected");
    CHECK(!valid_span(UINT64_MAX - 3, 8), "wrapping span rejected");
    CHECK(valid_span(UINT64_MAX - 7, 4), "span ending inside the address space is valid");
    CHECK(valid_span(1ull << 40, UINT64_MAX - (1ull << 40)), "exactly reaching the top is valid");
    CHECK(!valid_span(1ull << 40, UINT64_MAX - (1ull << 40) + 1), "one past the top invalid");
    CHECK(pre_impl::valid(8,  4) && !pre_impl::valid(0,  4) && !pre_impl::valid(2,  4) &&
          !pre_impl::valid(UINT64_MAX - 3,  8), "pre valid");
    CHECK(pre_impl::disjoint(0x100,  0x40, 0x140,  0x40) && !pre_impl::disjoint(0x100,  0x41, 0x140,  0x40), "pre disjoint");
    // layout_disjoint3 on cumulative layout
    for (int t = 0; t < 500; ++t) {
        uint64_t a = (rng() % 1000) * 4, n1 = (rng() % 50) * 4, n2 = (rng() % 50) * 4, n3 = (rng() % 50) * 4;
        CHECK(pre_impl::disjoint(a,  n1, a + n1,  n2) && pre_impl::disjoint(a,  n1, a + n1 + n2,  n3) &&
              pre_impl::disjoint(a + n1,  n2, a + n1 + n2,  n3), "cumulative layout disjoint");
    }
    // geometry guards
    unsigned g = 0;
    CHECK(pre_impl::count_and_stream(1, (void*)1) && pre_impl::count_and_stream(65535, (void*)1), "count bounds accepted");
    CHECK(!pre_impl::count_and_stream(0, (void*)1) && !pre_impl::count_and_stream(65536, (void*)1) && !pre_impl::count_and_stream(-3, (void*)1), "count bounds rejected");
    CHECK(!pre_impl::count_and_stream(10, nullptr), "null stream rejected");
    CHECK(pre_impl::norm_geometry(7, 128, 256, &g) && g == 7, "norm geometry ok");
    CHECK(!pre_impl::norm_geometry(7, 64, 256, &g) && !pre_impl::norm_geometry(7, 128, 128, &g) && !pre_impl::norm_geometry(0, 128, 256, &g), "norm geometry bad");
    for (int c : {1, 255, 256, 257, 1000, 65535}) {
        int blocks = (c + 255) / 256;
        CHECK(blocks >= 1 && blocks <= 256 && c <= blocks * 256, "grid bounds/covers count=%d", c);
    }
}

static int test_step(cudaStream_t st, int hk, int hv, bool zero_beta) {
    const int S = 128;
    auto h_state = rnd((size_t)S * hv * S, -0.3f, 0.3f), q = rnd((size_t)hk * S, -0.3f, 0.3f), k = rnd((size_t)hk * S, -0.3f, 0.3f);
    auto v = rnd((size_t)hv * S, -1.f, 1.f), gate = rnd(hv, -1.5f, 0.f), beta = rnd(hv, 0.2f, 0.9f);
    if (zero_beta) for (auto& b : beta) b = 0.f;
    std::vector<double> rs(h_state.begin(), h_state.end()), ro((size_t)hv * S);
    const double scale = 1.0 / std::sqrt((double)S);
    for (int h = 0; h < hv; ++h) {
        const int src = h % hk; const double dec = std::exp((double)gate[h]);
        for (int j = 0; j < S; ++j) {
            double sk = 0;
            for (int i = 0; i < S; ++i) { double s = rs[((size_t)i * hv + h) * S + j] * dec; rs[((size_t)i * hv + h) * S + j] = s; sk += s * k[(size_t)src * S + i]; }
            double d = (v[(size_t)h * S + j] - sk) * beta[h], dot = 0;
            for (int i = 0; i < S; ++i) { double s = rs[((size_t)i * hv + h) * S + j] + k[(size_t)src * S + i] * d; rs[((size_t)i * hv + h) * S + j] = s; dot += s * q[(size_t)src * S + i]; }
            ro[(size_t)h * S + j] = dot * scale;
        }
    }
    float *ds = up(h_state), *dq = up(q), *dk = up(k), *dv = up(v), *dg = up(gate), *db = up(beta), *dout = nullptr;
    CK(cudaMalloc(&dout, (size_t)hv * S * 4));
    CK(cudaMemset(dout, 0xFF, (size_t)hv * S * 4));
    int rc = native_gdn_step(ds, dq, dk, dv, dg, db, dout, S, hk, hv, st);
    CHECK(rc == 0, "step rc=%d hk=%d hv=%d", rc, hk, hv);
    CK(cudaStreamSynchronize(st));
    double eo = rel(down(dout, (size_t)hv * S), ro), es = rel(down(ds, h_state.size()), rs);
    printf("step hk=%-3d hv=%-3d beta0=%d  rel o=%.3g state=%.3g\n", hk, hv, (int)zero_beta, eo, es);
    CHECK(eo < 1e-4, "step output mismatch %.3g (every column covered exactly once)", eo);
    CHECK(es < 1e-4, "step state mismatch %.3g (every row covered exactly once)", es);
    cudaFree(ds); cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(db); cudaFree(dout);
    return 0;
}

static int test_step_rejects(cudaStream_t st) {
    const int hk = 2, hv = 4, S = 128;
    const size_t sb = (size_t)S * S * hv * 4;
    char* big = nullptr; CK(cudaMalloc(&big, sb + (1 << 20)));
    CK(cudaMemset(big, 0, sb + (1 << 20)));
    float* state = (float*)big;
    float* o = (float*)(big + sb);             float* q = (float*)(big + sb + 4096);
    float* k = (float*)(big + sb + 8192);      float* v = (float*)(big + sb + 12288);
    float* gate = (float*)(big + sb + 16384);  float* beta = (float*)(big + sb + 20480);
    CK(cudaMemset(o, 0x7F, 2048));
    std::vector<unsigned char> before(2048), after(2048);
    cudaMemcpy(before.data(), o, 2048, cudaMemcpyDeviceToHost);
    auto untouched = [&](const char* what) {
        cudaDeviceSynchronize(); cudaMemcpy(after.data(), o, 2048, cudaMemcpyDeviceToHost);
        CHECK(before == after, "%s: rejected call must not write", what);
    };
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, hk, hv, st) == 0, "baseline accepted");
    CK(cudaStreamSynchronize(st));
    CK(cudaMemset(o, 0x7F, 2048)); cudaMemcpy(before.data(), o, 2048, cudaMemcpyDeviceToHost);
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, 64, hk, hv, st) == kReject, "S != 128");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, 3, hv, st) == kReject, "h_v %% h_k != 0");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, 0, hv, st) == kReject, "h_k == 0");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, hk, 0, st) == kReject, "h_v == 0");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, 1, 65536, st) == kReject, "h_v > 65535");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, hk, hv, nullptr) == kReject, "null stream");
    CHECK(native_gdn_step(nullptr, q, k, v, gate, beta, o, S, hk, hv, st) == kReject, "null state");
    CHECK(native_gdn_step(state, q, k, v, gate, nullptr, o, S, hk, hv, st) == kReject, "null beta");
    CHECK(native_gdn_step((float*)(big + 2), q, k, v, gate, beta, o, S, hk, hv, st) == kReject, "misaligned state");
    CHECK(native_gdn_step(state, (float*)((char*)q + 2), k, v, gate, beta, o, S, hk, hv, st) == kReject, "misaligned q");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, (float*)((char*)o + 1), S, hk, hv, st) == kReject, "misaligned o");
    // writable overlap: o inside state, o overlapping q, state overlapping beta
    CHECK(native_gdn_step(state, q, k, v, gate, beta, state + 1024, S, hk, hv, st) == kReject, "o inside state");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, q, S, hk, hv, st) == kReject, "o aliases q");
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, hk, hv, st) == 0, "disjoint accepted again");
    CK(cudaStreamSynchronize(st));
    CK(cudaMemset(o, 0x7F, 2048)); cudaMemcpy(before.data(), o, 2048, cudaMemcpyDeviceToHost);
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, hk, hv, st) == 0, "ok");
    // read spans may alias each other (q == k)
    CHECK(native_gdn_step(state, q, q, v, gate, beta, o, S, hk, hv, st) == 0, "read-read aliasing is allowed");
    CK(cudaStreamSynchronize(st));
    // a rejected call leaves o untouched
    CK(cudaMemset(o, 0x7F, 2048)); cudaMemcpy(before.data(), o, 2048, cudaMemcpyDeviceToHost);
    CHECK(native_gdn_step(state, q, k, v, gate, beta, o, S, 3, hv, st) == kReject, "reject");
    untouched("step");
    // enable switch
    CHECK(native_gdn_enabled(), "enabled by default");
    native_gdn_set_enabled(false); CHECK(!native_gdn_enabled(), "disabled"); native_gdn_set_enabled(true); CHECK(native_gdn_enabled(), "re-enabled");
    cudaFree(big);
    return 0;
}

static int test_pre(cudaStream_t st) {
    // ---- conv + silu with history slide
    {
        const int C = 1000;
        auto hist = rnd(C * 3, -1, 1), x = rnd(C, -2, 2), w = rnd(C * 4, -1, 1);
        std::vector<double> ro(C); std::vector<float> rh(C * 3);
        for (int c = 0; c < C; ++c) {
            double acc = (double)hist[c * 3] * w[c * 4] + (double)hist[c * 3 + 1] * w[c * 4 + 1] + (double)hist[c * 3 + 2] * w[c * 4 + 2] + (double)x[c] * w[c * 4 + 3];
            ro[c] = acc / (1.0 + std::exp(-acc));
            rh[c * 3] = hist[c * 3 + 1]; rh[c * 3 + 1] = hist[c * 3 + 2]; rh[c * 3 + 2] = x[c];
        }
        float *dh = up(hist), *dx = up(x), *dw = up(w), *dout = nullptr; CK(cudaMalloc(&dout, C * 4));
        int rc = native_gdn_conv_silu(dh, dx, dw, dout, C, st); CHECK(rc == 0, "conv rc=%d", rc);
        CK(cudaStreamSynchronize(st));
        double e = rel(down(dout, C), ro); printf("conv_silu rel=%.3g\n", e); CHECK(e < 1e-5, "conv_silu mismatch");
        auto gh = down(dh, C * 3); bool same = true; for (int i = 0; i < C * 3; ++i) same &= gh[i] == rh[i];
        CHECK(same, "history must become [h1,h2,x] exactly");
        // sliding twice == window of the last 3 inputs
        auto x2 = rnd(C, -2, 2); float* dx2 = up(x2);
        CHECK(native_gdn_conv_silu(dh, dx2, dw, dout, C, st) == 0, "conv 2");
        CK(cudaStreamSynchronize(st));
        auto gh2 = down(dh, C * 3); same = true;
        for (int c = 0; c < C; ++c) same &= gh2[c * 3] == rh[c * 3 + 1] && gh2[c * 3 + 1] == x[c] && gh2[c * 3 + 2] == x2[c];
        CHECK(same, "second slide");
        // rejects: count guards, aliasing, null stream
        CHECK(native_gdn_conv_silu(dh, dx, dw, dout, 0, st) == kReject, "count 0");
        CHECK(native_gdn_conv_silu(dh, dx, dw, dout, 65536, st) == kReject, "count 65536");
        CHECK(native_gdn_conv_silu(dh, dx, dw, dout, C, nullptr) == kReject, "null stream");
        CHECK(native_gdn_conv_silu(dh, dx, dw, dh, C, st) == kReject, "out aliases hist");
        CHECK(native_gdn_conv_silu(dh, dh + 3, dw, dout, C, st) == kReject, "x overlaps hist");
        cudaFree(dh); cudaFree(dx); cudaFree(dx2); cudaFree(dw); cudaFree(dout);
    }
    // ---- max count accepted (65535 channels)
    {
        const int C = 65535;
        float *h, *x, *w, *o; CK(cudaMalloc(&h, C * 12)); CK(cudaMalloc(&x, C * 4)); CK(cudaMalloc(&w, C * 16)); CK(cudaMalloc(&o, C * 4));
        CK(cudaMemset(h, 0, C * 12)); CK(cudaMemset(x, 0, C * 4)); CK(cudaMemset(w, 0, C * 16));
        CHECK(native_gdn_conv_silu(h, x, w, o, C, st) == 0, "65535 accepted"); CK(cudaStreamSynchronize(st));
        cudaFree(h); cudaFree(x); cudaFree(w); cudaFree(o);
    }
    // ---- l2 norm with scale, and its unit-norm invariant
    {
        const int R = 37;
        auto in = rnd((size_t)R * 128, -3, 3); const float eps = 1e-6f, scale = 0.25f;
        std::vector<double> ro((size_t)R * 128);
        for (int r = 0; r < R; ++r) { double s = 0; for (int c = 0; c < 128; ++c) s += (double)in[r * 128 + c] * in[r * 128 + c];
            double inv = 1.0 / std::sqrt(s + eps); for (int c = 0; c < 128; ++c) ro[r * 128 + c] = in[r * 128 + c] * inv * scale; }
        float *di = up(in), *dout = nullptr; CK(cudaMalloc(&dout, in.size() * 4));
        int rc = native_gdn_l2_norm(di, dout, R, eps, scale, st); CHECK(rc == 0, "l2 rc=%d", rc);
        CK(cudaStreamSynchronize(st));
        auto got = down(dout, in.size()); double e = rel(got, ro); printf("l2_norm rel=%.3g\n", e); CHECK(e < 1e-5, "l2 mismatch");
        for (int r = 0; r < R; ++r) { double s = 0; for (int c = 0; c < 128; ++c) s += (double)got[r * 128 + c] * got[r * 128 + c];
            CHECK(std::fabs(std::sqrt(s) - scale) < 1e-4, "row %d norm %.6f != scale", r, std::sqrt(s)); }
        CHECK(native_gdn_l2_norm(di, di, R, eps, scale, st) == kReject, "in-place rejected (writable span overlaps input)");
        CHECK(native_gdn_l2_norm(di, dout, 0, eps, scale, st) == kReject, "rows 0");
        CHECK(native_gdn_l2_norm(di, dout, 65536, eps, scale, st) == kReject, "rows 65536");
        cudaFree(di); cudaFree(dout);
    }
    // ---- beta sigmoid + gate softplus (branch above 20)
    {
        const int N = 777;
        auto raw = rnd(N, -30, 30), bias = rnd(N, -1, 1), a = rnd(N, -2, 0);
        raw[0] = 25.f; bias[0] = 0.f; raw[1] = 20.f; bias[1] = 0.f; raw[2] = 19.99f; raw[3] = 20.01f; raw[4] = -60.f;
        std::vector<double> rb(N), rg(N);
        for (int i = 0; i < N; ++i) { rb[i] = 1.0 / (1.0 + std::exp(-(double)raw[i]));
            double v = (double)(raw[i] + bias[i]); rg[i] = (v > 20.0 ? v : std::log1p(std::exp(v))) * a[i]; }
        float *dr = up(raw), *db = up(bias), *da = up(a), *ob = nullptr, *og = nullptr; CK(cudaMalloc(&ob, N * 4)); CK(cudaMalloc(&og, N * 4));
        CHECK(native_gdn_beta_gate(dr, ob, N, st) == 0, "beta rc"); CHECK(native_gdn_gate(dr, db, da, og, N, st) == 0, "gate rc");
        CK(cudaStreamSynchronize(st));
        auto gb = down(ob, N), gg = down(og, N);
        double eb = rel(gb, rb), eg = rel(gg, rg); printf("beta rel=%.3g gate rel=%.3g\n", eb, eg);
        CHECK(eb < 1e-5, "beta mismatch"); CHECK(eg < 1e-5, "gate mismatch");
        CHECK(gg[0] == 25.f * a[0], "softplus large branch must return v unchanged (v=25)");
        CHECK(std::isfinite(gg[3]) && std::isfinite(gb[4]) && gb[4] >= 0.f, "no overflow at the extremes");
        CHECK(native_gdn_beta_gate(dr, dr, N, st) == kReject, "beta alias"); CHECK(native_gdn_gate(dr, db, da, da, N, st) == kReject, "gate alias");
        CHECK(native_gdn_gate(dr, db, da, og, 0, st) == kReject, "gate count 0");
        cudaFree(dr); cudaFree(db); cudaFree(da); cudaFree(ob); cudaFree(og);
    }
    // ---- out norm with gamma and sigmoid gate
    {
        const int R = 33; const float eps = 1e-6f;
        auto o = rnd((size_t)R * 128, -2, 2), z = rnd((size_t)R * 128, -4, 4), gm = rnd(128, 0.5f, 1.5f);
        std::vector<double> ry((size_t)R * 128);
        for (int r = 0; r < R; ++r) { double s = 0; for (int c = 0; c < 128; ++c) s += (double)o[r * 128 + c] * o[r * 128 + c];
            double inv = 1.0 / std::sqrt(s / 128.0 + eps);
            for (int c = 0; c < 128; ++c) ry[r * 128 + c] = o[r * 128 + c] * inv * gm[c] / (1.0 + std::exp(-(double)z[r * 128 + c])); }
        float *dO = up(o), *dz = up(z), *dg = up(gm), *dy = nullptr; CK(cudaMalloc(&dy, o.size() * 4));
        int rc = native_gdn_out_norm(dO, dz, dg, dy, R, eps, st); CHECK(rc == 0, "out_norm rc=%d", rc);
        CK(cudaStreamSynchronize(st));
        double e = rel(down(dy, o.size()), ry); printf("out_norm rel=%.3g\n", e); CHECK(e < 1e-5, "out_norm mismatch");
        CHECK(native_gdn_out_norm(dO, dz, dg, dO, R, eps, st) == kReject, "y aliases o");
        CHECK(native_gdn_out_norm(dO, dz, dy, dy, R, eps, st) == kReject, "y aliases gamma");
        cudaFree(dO); cudaFree(dz); cudaFree(dg); cudaFree(dy);
    }
    return 0;
}

int main() {
    int n = 0; CK(cudaGetDeviceCount(&n)); printf("devices: %d\n", n);
    CHECK(n > 0, "no GPU device");
    cudaStream_t st; CK(cudaStreamCreate(&st));
    test_span_math();
    // identities of the theory (LAYER 1) on plain integers
    for (int c = 0; c < 128; ++c) { int z = c / 4, y = c % 4; CHECK(z < 32 && y < 4 && z * 4 + y == c, "col cover"); }
    for (int i = 0; i < 128; ++i) { int r = i / 32, l = i % 32; CHECK(r < 4 && l < 32 && r * 32 + l == i, "row cover"); }
    CHECK(128 * 128 * 48 * 4 == 65536 * 48 && 128 * 48 * 4 == 512 * 48, "state/output byte sizes");
    if (test_step(st, 2, 8, false)) return 2;
    if (test_step(st, 4, 12, false)) return 2;
    if (test_step(st, 12, 12, false)) return 2;
    if (test_step(st, 1, 6, true)) return 2;     // beta = 0: state only decays; pure addressing check
    if (test_step_rejects(st)) return 2;
    if (test_pre(st)) return 2;
    CK(cudaStreamDestroy(st));
    if (g_fail) { printf("U272 UNIT TESTS FAILED (%d)\n", g_fail); return 1; }
    printf("U272 UNIT TESTS PASSED\n");
    return 0;
}
