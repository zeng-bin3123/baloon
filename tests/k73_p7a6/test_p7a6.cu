// Unit p7a60e039c0: device invariants of the dense / elementwise / rope kernels, checked against independent host references.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kg_kernels.hpp"

using namespace baloon::kg;
namespace fb = baloon::f16bits;
static int g_fail = 0, g_checks = 0;
static std::mt19937_64 rng(777);
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static void ok(bool c, const char* w) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", w); } }
static void near(double a, double b, double tol, const char* w) {
    ++g_checks;
    if (!(std::fabs(a - b) <= tol * (1.0 + std::fabs(b)))) { ++g_fail; std::printf("FAIL: %s (%.9g vs %.9g)\n", w, a, b); }
}
template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { cudaMalloc(&p, n_ * sizeof(T)); cudaMemset(p, 0, n_ * sizeof(T)); }
    Dev(const std::vector<T>& v) : n(v.size()) { cudaMalloc(&p, n * sizeof(T)); cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
    operator T*() const { return p; }
};
static uint16_t bf(float f) { return fb::round_f32_to_bf16_bits(f); }
static float fbf(uint16_t b) { return fb::bf16_bits_to_f32(b); }

static void test_gemv() {
    for (int n_out : {5, 64, 131}) {
        const int n_in = 300;
        std::vector<uint16_t> x(n_in), w((size_t)n_out * n_in);
        for (auto& v : x) v = bf(rf(-1, 1));
        for (auto& v : w) v = bf(rf(-1, 1));
        std::vector<double> ref(n_out, 0.0);
        for (int o = 0; o < n_out; ++o) for (int i = 0; i < n_in; ++i) ref[o] += (double)fbf(x[i]) * fbf(w[(size_t)o * n_in + i]);
        Dev<uint16_t> dx(x), dw(w); Dev<float> dy(n_out), dy2(n_out), dy3(n_out);
        ok(bf16_gemv(dx, dw, dy, n_in, n_out) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "bf16_gemv runs");
        ok(bf16_gemv_split(dx, dw, dy2, n_in, n_out, 64) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "split 64 runs");
        ok(bf16_gemv_split(dx, dw, dy3, n_in, n_out, 256) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess, "split 256 runs");
        auto a = dy.get(), b = dy2.get(), c = dy3.get();
        for (int o = 0; o < n_out; ++o) {
            near(a[o], ref[o], 1e-4, "gemv vs host");
            near(b[o], ref[o], 1e-4, "gemv split64 vs host");
            near(c[o], ref[o], 1e-4, "gemv split256 vs host");
        }
    }
    Dev<uint16_t> dx(4), dw(8); Dev<float> dy(2);
    ok(bf16_gemv_split(dx, dw, dy, 4, 2, 48) == cudaErrorInvalidValue, "split rejects non-power-of-two tpr");
    ok(bf16_gemv_split(dx, dw, dy, 4, 2, 2048) == cudaErrorInvalidValue, "split rejects tpr > 1024");
    ok(bf16_gemv(nullptr, dw, dy, 4, 2) == cudaErrorInvalidValue, "gemv rejects null");
    ok(bf16_gemv(dx, dw, dy, 0, 2) == cudaErrorInvalidValue, "gemv rejects n_in 0");
}

static void test_elementwise() {
    const int n = 1000;
    std::vector<float> a(n), b(n);
    for (int i = 0; i < n; ++i) { a[i] = rf(-8, 8); b[i] = rf(-8, 8); }
    { Dev<float> da(a), db(b); add_inplace(da, db, n); cudaDeviceSynchronize(); auto r = da.get();
      for (int i = 0; i < n; ++i) if (r[i] != a[i] + b[i]) { ok(false, "add exact"); break; } ok(true, "add"); }
    { Dev<float> da(a); scale_inplace(da, 0.37f, n); cudaDeviceSynchronize(); auto r = da.get();
      for (int i = 0; i < n; ++i) if (r[i] != a[i] * 0.37f) { ok(false, "scale exact"); break; } ok(true, "scale"); }
    { Dev<float> da(a); silu_inplace(da, n); cudaDeviceSynchronize(); auto r = da.get();
      for (int i = 0; i < n; ++i) near(r[i], a[i] / (1.0 + std::exp(-(double)a[i])), 1e-5, "silu"); }
    { std::vector<float> s = a; s[0] = 1.00390625f; s[1] = 1.01171875f; s[2] = -0.0f; s[3] = 3.4e38f;   // ties + sign + overflow edge
      Dev<float> ds(s); Dev<uint16_t> db16(n), dh(n);
      f32_to_bf16_bulk(ds, db16, n); f32_to_f16_bulk(ds, dh, n); cudaDeviceSynchronize();
      auto rb = db16.get(), rh = dh.get(); bool eb = true, eh = true;
      for (int i = 0; i < n; ++i) { if (rb[i] != fb::round_f32_to_bf16_bits(s[i])) eb = false; if (rh[i] != fb::f16_from_f32(s[i])) eh = false; }
      ok(eb, "f32->bf16 bulk equals scalar rounding"); ok(eh, "f32->f16 bulk equals scalar rounding"); }
    { const int vocab = 6, cols = 70; std::vector<float> tab(vocab * cols); for (auto& v : tab) v = rf(-1, 1);
      std::vector<int32_t> tok = {2, -1, 6, 5, 100000}; Dev<float> dt(tab); Dev<int32_t> dk(tok); Dev<float> out(tok.size() * cols);
      ok(embedding_gather(dt, vocab, cols, dk, (long long)tok.size(), out) == cudaSuccess, "gather runs"); cudaDeviceSynchronize();
      auto r = out.get(); bool good = true;
      for (size_t t = 0; t < tok.size(); ++t) for (int c = 0; c < cols; ++c) {
          const bool in = tok[t] >= 0 && tok[t] < vocab; const float want = in ? tab[(size_t)tok[t] * cols + c] : 0.f;
          if (r[t * cols + c] != want) good = false; }
      ok(good, "embedding gather: out-of-range ids give zero rows"); }
    { const int h = 48; std::vector<float> al(h), lg(h), dt(h), g_ref(h);
      for (int i = 0; i < h; ++i) { al[i] = rf(-30, 30); lg[i] = rf(-2, 1); dt[i] = rf(-1, 1);
          const double x = (double)al[i] + dt[i]; g_ref[i] = (float)(-std::exp((double)lg[i]) * (x > 20 ? x : std::log1p(std::exp(x)))); }
      Dev<float> da(al), dl(lg), dd(dt), dg(h); gdn_gate(da, dl, dd, dg, h); cudaDeviceSynchronize(); auto r = dg.get();
      for (int i = 0; i < h; ++i) { near(r[i], g_ref[i], 1e-4, "gdn gate"); ok(r[i] <= 0.f, "gdn gate is non-positive"); } }
    { const int rows = 3, cols = 517; const float eps = 1e-6f; std::vector<float> x(rows * cols), w(cols);
      for (auto& v : x) v = rf(-3, 3); for (auto& v : w) v = rf(0.5f, 1.5f);
      Dev<float> dx(x), dw(w), dy(x.size()); rms_norm_weighted(dx, dw, dy, rows, cols, eps); cudaDeviceSynchronize(); auto r = dy.get();
      for (int rr = 0; rr < rows; ++rr) { double ss = 0; for (int c = 0; c < cols; ++c) ss += (double)x[rr * cols + c] * x[rr * cols + c];
          const double inv = 1.0 / std::sqrt(ss / cols + eps);
          for (int c = 0; c < cols; ++c) near(r[rr * cols + c], x[rr * cols + c] * inv * w[c], 1e-4, "rms norm"); } }
}

static void test_doorbell() {
    Dev<uint32_t> bell(1), status(1);
    uint32_t h = 0;
    doorbell_publish(bell, 5); doorbell_ring(bell); doorbell_ring(bell); cudaDeviceSynchronize();
    cudaMemcpy(&h, bell, 4, cudaMemcpyDeviceToHost); ok(h == 7, "publish(5) then two rings = 7");
    doorbell_wait(bell, 7, 1000000, status); cudaDeviceSynchronize();
    cudaMemcpy(&h, status, 4, cudaMemcpyDeviceToHost); ok(h == 1, "wait on reached target reports success");
    doorbell_wait(bell, 8, 1000, status); cudaDeviceSynchronize();
    cudaMemcpy(&h, status, 4, cudaMemcpyDeviceToHost); ok(h == 0, "wait on unreachable target times out with status 0");
}

static void test_mrope() {
    const int bad_neg[3] = {4, -1, 5};
    const int sec[3] = {6, 5, 5};
    ok(mrope_table_set(nullptr), "default restore accepted");
    ok(mrope_table(0, 16) == 0 && mrope_table(15, 16) == 0, "default: all frequencies on component 0");
    ok(!mrope_table_set(bad_neg), "negative middle section rejected");
    const int neg0[3] = {-1, 8, 8}; ok(!mrope_table_set(neg0), "negative first section rejected");
    const int neg2[3] = {8, 9, -1}; ok(!mrope_table_set(neg2), "negative last section rejected");
    ok(mrope_table(15, 16) == 0, "rejected sections leave the default unchanged");
    ok(mrope_table_set(sec), "valid sections accepted");
    for (int i = 0; i < 16; ++i) ok(mrope_table(i, 16) == (i < 6 ? 0 : (i < 11 ? 1 : 2)), "section ownership");
    ok(!mrope_table_set(bad_neg), "negative rejected after valid set");
    ok(mrope_table(7, 16) == 1, "failed set keeps previous sections");
    // native_rope_apply: position 0 is the identity; sections must sum to rot/2
    const int n_tok = 2, nh = 2, hd = 32, rot = 32;
    std::vector<float> x(n_tok * nh * hd); for (auto& v : x) v = rf(-1, 1);
    std::vector<int32_t> p0(n_tok * 3, 0); Dev<float> dx(x); Dev<int32_t> dp(p0);
    ok(native_rope_apply(dx, dp, n_tok, nh, hd, rot, 10000.f) == cudaSuccess, "rope apply with {6,5,5}");
    ok(dx.get() == x, "rope at position 0 is the identity");
    const int off[3] = {6, 5, 4}; ok(mrope_table_set(off), "sum-mismatch sections are storable");
    ok(native_rope_apply(dx, dp, n_tok, nh, hd, rot, 10000.f) != cudaSuccess, "sections not summing to rot/2 rejected at apply");
    mrope_table_set(nullptr);
    // default text rope: rotation preserves pair norms and the position of component 0 only is used
    std::vector<int32_t> p1 = {3, 99, 99, 4, 77, 77}; Dev<float> dy(x); Dev<int32_t> dq(p1);
    ok(native_rope_apply(dy, dq, n_tok, nh, hd, rot, 10000.f) == cudaSuccess, "default rope apply");
    auto r = dy.get(); bool norm_ok = true; const int half = rot / 2;
    for (int th = 0; th < n_tok * nh; ++th) for (int i = 0; i < half; ++i) {
        const double a0 = x[th * hd + i], a1 = x[th * hd + i + half], b0 = r[th * hd + i], b1 = r[th * hd + i + half];
        if (std::fabs(a0 * a0 + a1 * a1 - b0 * b0 - b1 * b1) > 1e-4) norm_ok = false;
        const int pos = (th / nh) == 0 ? 3 : 4; const double ang = pos * std::pow(10000.0, -2.0 * i / rot);
        if (std::fabs(b0 - (a0 * std::cos(ang) - a1 * std::sin(ang))) > 2e-4) norm_ok = false;
    }
    ok(norm_ok, "default rope equals the NeoX rotation by component-0 position");
    // overlap of x and positions is rejected
    ok(native_rope_apply((float*)dq.p, dq, 1, 1, 6, 6, 10000.f) != cudaSuccess, "x/pos overlap rejected");
}

int main() {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 1) { std::printf("no GPU device\n"); return 2; }
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("cannot open device 0\n"); return 2; }
    test_gemv(); test_elementwise(); test_doorbell(); test_mrope();
    std::printf("%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail) { std::printf("P7A6 UNIT TESTS FAILED\n"); return 1; }
    std::printf("P7A6 UNIT TESTS PASSED\n");
    return 0;
}
