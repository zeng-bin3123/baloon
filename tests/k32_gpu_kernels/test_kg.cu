// Unit 32 invariants of the src/kernels/gpu kernels, checked on the device against independent host references:
// exact decode / rounding laws (Q2_0, bf16/f16 bits, i-quant sign parity, Q8_1), schedule independence (gemv schedules, fused vs
// unfused, native vs plain are equal), tie-breaking and renormalisation of the router, validation / overlap rejection of the native
// entry points, the gated-residual workspace geometry (104320 bytes) and block-format row geometry.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/scalar_dequantizers.hpp"

using namespace baloon;
using namespace baloon::kg;
static cudaStream_t g_s = nullptr;   // explicit non-null stream for native_gdn_* (null is rejected)
namespace fb = baloon::f16bits;

static int g_fail = 0, g_checks = 0;
static std::mt19937_64 rng(12345);
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static uint32_t ru(uint32_t n) { return (uint32_t)(rng() % n); }
static void ok(bool c, const char* what) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", what); } }
static void okf(bool c, const char* what, double a, double b) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s (%.9g vs %.9g)\n", what, a, b); } }

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { if (n) cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    Dev(const Dev&) = delete; Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); if (n) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
    void set(const std::vector<T>& v) { cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); }
    operator T*() const { return p; }
};
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess; }
static std::vector<float> vf(size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = rf(lo, hi); return v; }
static std::vector<uint16_t> vbf(size_t n, float s) { std::vector<uint16_t> v(n); for (auto& x : v) x = fb::round_f32_to_bf16_bits(rf(-s, s)); return v; }
static float bf(uint16_t b) { return fb::bf16_bits_to_f32(b); }
static float rbf(float x) { return bf(fb::round_f32_to_bf16_bits(x)); }
static float rh(float x) { return fb::f32_from_f16(fb::f16_from_f32(x)); }
static void near_vec(const std::vector<float>& a, const std::vector<double>& r, double rtol, double atol, const char* name) {
    double worst = 0; size_t wi = 0; bool good = a.size() == r.size();
    for (size_t i = 0; good && i < a.size(); ++i) {
        const double e = std::fabs((double)a[i] - r[i]) - (atol + rtol * std::fabs(r[i]));
        if (e > worst || std::isnan(a[i])) { worst = e; wi = i; }
        if (std::isnan(a[i])) good = false;
    }
    okf(good && worst <= 0, name, good && wi < a.size() ? a[wi] : 0.0, good && wi < r.size() ? r[wi] : 0.0);
}
static void near_vec_f(const std::vector<float>& a, const std::vector<float>& b, double rtol, double atol, const char* name) {
    std::vector<double> r(b.begin(), b.end()); near_vec(a, r, rtol, atol, name);
}
static double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu_d(double x) { return x * sigmoid_d(x); }

// ------------------------------------------------------------------------------------------------------ dense / elementwise
static void test_dense() {
    const int n_in = 1000;
    auto x = vbf(n_in, 1.f);
    for (int n_out : {5, 70}) {
        auto w = vbf((size_t)n_in * n_out, 1.f);
        Dev<uint16_t> dx(x), dw(w); Dev<float> dy(n_out);
        ok(bf16_gemv(dx, dw, dy, n_in, n_out) == cudaSuccess && sync_ok(), "bf16_gemv launch");
        std::vector<double> r(n_out, 0.0);
        for (int o = 0; o < n_out; ++o) for (int i = 0; i < n_in; ++i) r[o] += (double)bf(x[i]) * bf(w[(size_t)o * n_in + i]);
        near_vec(dy.get(), r, 1e-4, 1e-3, n_out < 64 ? "bf16_gemv naive path" : "bf16_gemv warp path");
        for (int tpr : {64, 128, 256}) {
            Dev<float> ds(n_out);
            ok(bf16_gemv_split(dx, dw, ds, n_in, n_out, tpr) == cudaSuccess && sync_ok(), "bf16_gemv_split launch");
            near_vec(ds.get(), r, 1e-4, 1e-3, "bf16_gemv_split matches reference for every tpr");
        }
        Dev<float> dm(n_out);
        std::vector<float> xf(n_in);
        Dev<float> dxf(xf); for (int i = 0; i < n_in; ++i) xf[i] = bf(x[i]);
        dxf.set(xf);
        ok(bf16_gemv_fp32_mmvf(dw, dxf, dm, n_out, n_in) == cudaSuccess && sync_ok(), "mmvf launch");
        near_vec(dm.get(), r, 1e-4, 1e-3, "bf16_gemv_fp32_mmvf matches reference");
    }
    Dev<float> dy(4);
    ok(bf16_gemv(nullptr, nullptr, dy, 10, 4) != cudaSuccess, "bf16_gemv rejects null");
    { Dev<uint16_t> a(10), b(40); ok(bf16_gemv_split(a, b, dy, 10, 4, 96) != cudaSuccess, "split rejects non power-of-two tpr"); }

    // Q2_0 dequantization: element i = d * (code - 1)
    const int nb = 100;
    std::vector<uint8_t> blk((size_t)nb * 18);
    std::vector<float> ref((size_t)nb * 64);
    for (int b = 0; b < nb; ++b) {
        const uint16_t d = fb::f16_from_f32(rf(-2.f, 2.f));
        blk[b * 18] = d & 0xff; blk[b * 18 + 1] = d >> 8;
        for (int k = 0; k < 16; ++k) blk[b * 18 + 2 + k] = (uint8_t)ru(256);
        for (int i = 0; i < 64; ++i) ref[(size_t)b * 64 + i] = fb::f32_from_f16(d) * (float)(((blk[b * 18 + 2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    }
    Dev<uint8_t> db(blk); Dev<float> dout((size_t)nb * 64);
    ok(dequant_s2(db, dout, nb) == cudaSuccess && sync_ok(), "dequant_s2 launch");
    ok(dout.get() == ref, "dequant_s2 equals d*(code-1) exactly");
    { auto o = dout.get(); std::vector<float> one(64); for (int i = 0; i < 64; ++i) one[i] = o[i]; (void)one; }

    // rounding laws (literal values, not derived from the library under test)
    std::vector<float> f = {1.0f, 0.f, -0.f, 1.00390625f, 1.01171875f, 65504.f, 1e10f, 1e-10f, INFINITY};
    uint32_t tie1 = 0x3F808000u, tie2 = 0x3F818000u; float t1, t2; std::memcpy(&t1, &tie1, 4); std::memcpy(&t2, &tie2, 4);
    f.push_back(t1); f.push_back(t2);
    Dev<float> df(f); Dev<uint16_t> dbf(f.size()), dh(f.size());
    ok(f32_to_bf16_bulk(df, dbf, f.size()) == cudaSuccess && f32_to_f16_bulk(df, dh, f.size()) == cudaSuccess && sync_ok(), "bulk convert launch");
    auto B = dbf.get(), H = dh.get();
    ok(B[0] == 0x3F80, "bf16(1.0) = 0x3F80");
    ok(B[9] == 0x3F80, "bf16 tie 0x3F808000 rounds to even (0x3F80)");
    ok(B[10] == 0x3F82, "bf16 tie 0x3F818000 rounds to even (0x3F82)");
    ok(H[0] == 0x3C00, "f16(1.0) = 0x3C00");
    ok(H[5] == 0x7BFF, "f16(65504) = max finite");
    ok(H[6] == 0x7C00, "f16 finite overflow is +inf");
    ok(H[2] == 0x8000, "f16(-0) keeps the sign");
    ok(H[8] == 0x7C00, "f16(inf) stays inf");

    // add / scale / silu
    auto a = vf(777, -3, 3), b = vf(777, -3, 3);
    Dev<float> da(a), dbb(b);
    ok(add_inplace(da, dbb, 777) == cudaSuccess && scale_inplace(da, 0.5f, 777) == cudaSuccess && sync_ok(), "add/scale launch");
    std::vector<float> r1(777); for (int i = 0; i < 777; ++i) r1[i] = (a[i] + b[i]) * 0.5f;
    near_vec_f(da.get(), r1, 1e-6, 1e-6, "add then scale");
    Dev<float> ds(a);
    ok(silu_inplace(ds, 777) == cudaSuccess && sync_ok(), "silu launch");
    std::vector<double> r2(777); for (int i = 0; i < 777; ++i) r2[i] = silu_d(a[i]);
    near_vec(ds.get(), r2, 1e-5, 1e-6, "silu");
    ok(add_inplace(nullptr, nullptr, 5) != cudaSuccess && add_inplace(nullptr, nullptr, 0) == cudaSuccess, "add_inplace validation");

    // embedding gather: out-of-range ids give zero rows
    const int vocab = 10, cols = 33;
    auto tab = vf((size_t)vocab * cols, -1, 1);
    std::vector<int32_t> toks = {3, 0, 9, -1, 10, 55};
    Dev<float> dt(tab), dg(toks.size() * cols); Dev<int32_t> dtk(toks);
    ok(embedding_gather(dt, vocab, cols, dtk, toks.size(), dg) == cudaSuccess && sync_ok(), "embedding_gather launch");
    auto g = dg.get(); bool eg = true;
    for (size_t t = 0; t < toks.size(); ++t) for (int c = 0; c < cols; ++c) {
        const float e = (toks[t] >= 0 && toks[t] < vocab) ? tab[(size_t)toks[t] * cols + c] : 0.f;
        if (g[t * cols + c] != e) eg = false;
    }
    ok(eg, "embedding_gather rows and zero rows for bad ids");

    // gdn gate and weighted rms norm
    auto al = vf(40, -3, 3), alog = vf(40, -2, 1), dtb = vf(40, -1, 1);
    Dev<float> dal(al), dlog(alog), ddt(dtb), dgate(40);
    ok(gdn_gate(dal, dlog, ddt, dgate, 40) == cudaSuccess && sync_ok(), "gdn_gate launch");
    std::vector<double> rg(40); for (int i = 0; i < 40; ++i) { const double z = (double)al[i] + dtb[i]; rg[i] = -std::exp((double)alog[i]) * (z > 20 ? z : std::log1p(std::exp(z))); }
    near_vec(dgate.get(), rg, 1e-5, 1e-6, "gdn_gate = -exp(a_log) * softplus(alpha + dt_bias)");
    const int rows = 3, rc = 517;
    auto rx = vf((size_t)rows * rc, -2, 2), rw = vf(rc, 0.5f, 1.5f);
    Dev<float> drx(rx), drw(rw), dry((size_t)rows * rc);
    ok(rms_norm_weighted(drx, drw, dry, rows, rc, 1e-6f) == cudaSuccess && sync_ok(), "rms_norm_weighted launch");
    std::vector<double> rr((size_t)rows * rc);
    for (int r = 0; r < rows; ++r) { double s = 0; for (int i = 0; i < rc; ++i) s += (double)rx[r * rc + i] * rx[r * rc + i]; const double inv = 1.0 / std::sqrt(s / rc + 1e-6); for (int i = 0; i < rc; ++i) rr[r * rc + i] = rx[r * rc + i] * inv * rw[i]; }
    near_vec(dry.get(), rr, 1e-5, 1e-6, "rms_norm_weighted");

    // doorbell: ring counts, publish stores, wait is bounded
    Dev<uint32_t> bell(1), st(1);
    for (int i = 0; i < 3; ++i) doorbell_ring(bell);
    sync_ok(); ok(bell.get()[0] == 3, "doorbell_ring x3 -> 3");
    doorbell_publish(bell, 7); sync_ok(); ok(bell.get()[0] == 7, "doorbell_publish stores");
    doorbell_wait(bell, 7, 1000, st); sync_ok(); ok(st.get()[0] == 1, "doorbell_wait succeeds when the word reached the target");
    doorbell_wait(bell, 100, 1000, st); ok(sync_ok(), "doorbell_wait returns after max_spins"); ok(st.get()[0] == 0, "doorbell_wait reports timeout");
}

// ------------------------------------------------------------------------------------------------------ gated delta net
static void test_gdn() {
    const int S = 128, h_k = 2, h_v = 4;
    auto st0 = vf((size_t)S * h_v * S, -0.5f, 0.5f), q = vf(h_k * S, -1, 1), k = vf(h_k * S, -1, 1), v = vf(h_v * S, -1, 1);
    auto gate = vf(h_v, -1.5f, -0.1f), beta = vf(h_v, 0.1f, 0.9f), z = vf(h_v * S, -2, 2), nw = vf(S, 0.5f, 1.5f);
    Dev<float> dq(q), dk(k), dv(v), dgate(gate), dbeta(beta), dz(z), dnw(nw);
    Dev<float> sA(st0), sB(st0), sC(st0), oA((size_t)h_v * S), oB((size_t)h_v * S);
    ok(gdn_step(sA, dq, dk, dv, dgate, dbeta, oA, S, h_k, h_v) == cudaSuccess && sync_ok(), "gdn_step launch");
    // host recurrence in double
    std::vector<double> ro((size_t)h_v * S); std::vector<float> rst = st0;
    for (int h = 0; h < h_v; ++h) for (int j = 0; j < S; ++j) {
        const int src = h % h_k; const double dec = std::exp((double)gate[h]);
        double sk = 0; std::vector<double> col(S);
        for (int i = 0; i < S; ++i) { col[i] = (double)st0[(size_t)i * h_v * S + h * S + j] * dec; sk += col[i] * k[src * S + i]; }
        const double d = ((double)v[h * S + j] - sk) * beta[h]; double o = 0;
        for (int i = 0; i < S; ++i) { col[i] += k[src * S + i] * d; o += col[i] * q[src * S + i]; rst[(size_t)i * h_v * S + h * S + j] = (float)col[i]; }
        ro[h * S + j] = o;
    }
    near_vec(oA.get(), ro, 1e-4, 1e-4, "gdn_step output = decay, contract, rank-1 update, read with updated state");
    near_vec_f(sA.get(), rst, 1e-4, 1e-4, "gdn_step updated state");
    ok(native_gdn_step(sB, dq, dk, dv, dgate, dbeta, oB, S, h_k, h_v, g_s) == cudaSuccess && sync_ok(), "native_gdn_step launch");
    ok(sB.get() == sA.get() && oB.get() == oA.get(), "native_gdn_step is bit-identical to gdn_step");
    ok(native_gdn_step(sB, sB, dk, dv, dgate, dbeta, oB, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects state overlapping q");
    ok(native_gdn_step(sB, dq, dk, dv, dgate, dbeta, oB, 64, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects S != 128");
    Dev<float> yF((size_t)h_v * S), yU((size_t)h_v * S);
    ok(fused_gdn_step_norm(sC, dq, dk, dv, dgate, dbeta, dz, dnw, yF, h_k, h_v, 1e-6f) == cudaSuccess && sync_ok(), "fused_gdn_step_norm launch");
    ok(gdn_out_norm(oA, dz, dnw, yU, h_v, S, 1e-6f) == cudaSuccess && sync_ok(), "gdn_out_norm launch");
    near_vec_f(yF.get(), yU.get(), 1e-4, 1e-5, "fused step+norm equals step then out_norm");
    ok(sC.get() == sA.get(), "fused step leaves the same state");
    // conv + l2
    const int C = (2 * h_k + h_v) * S, dc = 4;
    auto cs0 = vf((size_t)C * (dc - 1), -1, 1), xin = vf(C, -1, 1), kW = vf((size_t)C * dc, -1, 1);
    Dev<float> c1(cs0), c2(cs0), dx(xin), dW(kW), o1(C), o2(C);
    ok(gdn_conv_step(c1, dx, dW, o1, C, dc) == cudaSuccess && gdn_l2_norm(o1, 2 * h_k, S, 1e-6f) == cudaSuccess && sync_ok(), "conv + l2 launch");
    ok(fused_gdn_conv_l2(c2, dx, dW, o2, h_k, h_v, dc, 1e-6f) == cudaSuccess && sync_ok(), "fused_gdn_conv_l2 launch");
    near_vec_f(o2.get(), o1.get(), 1e-4, 1e-5, "fused conv+l2 equals conv then l2 on q|k heads");
    ok(c1.get() == c2.get(), "conv history identical after fused step");
    { auto o = o2.get(); bool unit = true; for (int hd = 0; hd < 2 * h_k; ++hd) { double s = 0; for (int i = 0; i < S; ++i) s += (double)o[hd * S + i] * o[hd * S + i]; if (std::fabs(s - 1.0) > 1e-4) unit = false; } ok(unit, "q and k heads have unit L2 norm"); }
    // ab
    auto a_ = vf(h_v, -2, 2), b_ = vf(h_v, -2, 2), alog = vf(h_v, -1, 1), dtb = vf(h_v, -1, 1);
    Dev<float> da(a_), db(b_), dl(alog), dd(dtb), be(h_v), ga(h_v);
    ok(fused_gdn_ab(da, db, dl, dd, be, ga, h_v) == cudaSuccess && sync_ok(), "fused_gdn_ab launch");
    std::vector<double> rb(h_v), rg(h_v);
    for (int h = 0; h < h_v; ++h) { rb[h] = sigmoid_d(b_[h]); const double zz = (double)a_[h] + dtb[h]; rg[h] = -std::exp((double)alog[h]) * (zz > 20 ? zz : std::log1p(std::exp(zz))); }
    near_vec(be.get(), rb, 1e-5, 1e-6, "fused ab: beta = sigmoid(b)"); near_vec(ga.get(), rg, 1e-5, 1e-6, "fused ab: gate");
    Dev<float> bb(b_);
    ok(gdn_beta_gate(bb, h_v) == cudaSuccess && native_gdn_beta_gate(db, h_v, g_s) == cudaSuccess && sync_ok(), "beta gate launches");
    ok(bb.get() == db.get(), "native beta gate = plain beta gate");
    // native pre/post
    Dev<float> ng(h_v);
    ok(native_gdn_gate(da, dd, dl, ng, h_v, g_s) == cudaSuccess && sync_ok(), "native_gdn_gate launch");
    near_vec(ng.get(), rg, 1e-5, 1e-6, "native_gdn_gate");
    ok(native_gdn_gate(da, dd, dl, da, h_v, g_s) != cudaSuccess, "native_gdn_gate rejects output overlapping an input");
    Dev<float> c3(cs0), o3(C);
    ok(native_gdn_conv_silu(c3, dx, dW, o3, C, dc, g_s) == cudaSuccess && sync_ok(), "native_gdn_conv_silu launch");
    std::vector<double> rc(C);
    for (int c = 0; c < C; ++c) { double a = 0; for (int i = 0; i < dc - 1; ++i) a += (double)cs0[(size_t)c * (dc - 1) + i] * kW[(size_t)c * dc + i]; a += (double)xin[c] * kW[(size_t)c * dc + dc - 1]; rc[c] = silu_d(a); }
    near_vec(o3.get(), rc, 1e-4, 1e-5, "native_gdn_conv_silu");
    auto xl = vf(3 * 128, -2, 2); Dev<float> dxl(xl);
    ok(native_gdn_l2_norm(dxl, 3, 128, 1e-12f, g_s) == cudaSuccess && sync_ok(), "native_gdn_l2_norm launch");
    { auto o = dxl.get(); bool unit = true; for (int r = 0; r < 3; ++r) { double s = 0; for (int i = 0; i < 128; ++i) s += (double)o[r * 128 + i] * o[r * 128 + i]; if (std::fabs(s - 1) > 1e-4) unit = false; } ok(unit, "native_gdn_l2_norm gives unit rows"); }
    Dev<float> yn((size_t)h_v * S);
    ok(native_gdn_out_norm(oA, dz, dnw, yn, h_v, 1e-6f, g_s) == cudaSuccess && sync_ok(), "native_gdn_out_norm launch");
    near_vec_f(yn.get(), yU.get(), 1e-4, 1e-5, "native_gdn_out_norm equals gdn_out_norm");
    ok(native_gdn_out_norm(oA, dz, dnw, oA, h_v, 1e-6f, g_s) != cudaSuccess, "native_gdn_out_norm rejects y aliasing o");
    ok(native_gdn_l2_norm(nullptr, 3, 200, 1e-6f, g_s) != cudaSuccess, "native_gdn_l2_norm rejects null");
    native_gdn_set_enabled(false); ok(!native_gdn_enabled(), "native gdn switch off"); native_gdn_set_enabled(true); ok(native_gdn_enabled(), "native gdn switch on");
}

// ------------------------------------------------------------------------------------------------------ gated residual
static void gr_ref(const std::vector<float>& R, const std::vector<uint16_t>& down, const std::vector<uint16_t>& up, const std::vector<uint16_t>& inj,
                   int n, int hc, int lr, float eps, std::vector<double>& x, std::vector<double>& w) {
    const int hd = n * hc; std::vector<float> xn(hd), xq(hd), lo(lr), lq(lr);
    for (int c = 0; c < hc; ++c) { float s = 0; for (int d = 0; d < n; ++d) s += R[c * n + d] * R[c * n + d]; const float inv = 1.0f / std::sqrt(s / (float)n + eps); for (int d = 0; d < n; ++d) { xn[c * n + d] = R[c * n + d] * inv; xq[c * n + d] = rbf(xn[c * n + d]); } }
    for (int k = 0; k < lr; ++k) { double a = 0; for (int i = 0; i < hd; ++i) a += (double)bf(down[(size_t)k * hd + i]) * xq[i]; lo[k] = (float)silu_d(a); lq[k] = rbf(lo[k]); }
    x.assign(n, 0.0); w.assign(hc, 0.0);
    std::vector<float> gated(hd);
    for (int i = 0; i < hd; ++i) { double a = 0; for (int k = 0; k < lr; ++k) a += (double)bf(up[(size_t)i * lr + k]) * lq[k]; gated[i] = xn[i] * (float)sigmoid_d(a); }
    for (int d = 0; d < n; ++d) { float s = 0; for (int c = 0; c < hc; ++c) s += gated[c * n + d]; x[d] = s / (float)hc; }
    for (int c = 0; c < hc; ++c) { double a = 0; for (int d = 0; d < n; ++d) a += (double)bf(inj[(size_t)c * n + d]) * x[d]; w[c] = a; }
}
static void test_gr() {
    { // workspace geometry of the real configuration
        GrDims d{2560, 4, 320};
        ok(gr_workspace_bytes(d) == 104320, "gr workspace = 104320 bytes at n=2560, hc=4, lr=320");
        Dev<uint8_t> buf(104320); GrWs ws;
        ok(gr_workspace_init(buf, 104320, d, &ws), "gr_workspace_init fits exactly");
        ok(!gr_workspace_init(buf, 104319, d, &ws), "gr_workspace_init rejects one byte too little");
        gr_workspace_init(buf, 104320, d, &ws);
        ok((uint8_t*)ws.xq - (uint8_t*)ws.xn == 40960 && (uint8_t*)ws.lq - (uint8_t*)ws.xq == 20480 && (uint8_t*)ws.gated - (uint8_t*)ws.lq == 640 &&
           (uint8_t*)ws.lo - (uint8_t*)ws.gated == 40960 && ws.bytes == 104320, "workspace regions are consecutive, 16-aligned, disjoint (xq has 2 bytes per element)");
        { GrDims o{5, 1, 3}; ok(gr_workspace_bytes(o) == 112, "odd-size workspace: every region rounded up to 16 bytes (32+16+16+32+16)");
          Dev<uint8_t> b2(112); GrWs w2; ok(gr_workspace_init(b2, 112, o, &w2), "odd workspace init");
          ok((uint8_t*)w2.xq - (uint8_t*)w2.xn == 32 && (uint8_t*)w2.lq - (uint8_t*)w2.xq == 16 && (uint8_t*)w2.gated - (uint8_t*)w2.lq == 16 && (uint8_t*)w2.lo - (uint8_t*)w2.gated == 32 && w2.bytes == 112, "odd workspace offsets are 16-aligned"); }
        ok(fused_gr_supported(2560, 4, 320) && !fused_gr_supported(2560, 4, 321) && !fused_gr_supported(64, 4, 16), "fused_gr_supported geometry");
    }
    { // generic gr_read against the host
        const int n = 64, hc = 4, lr = 16, hd = n * hc; GrDims d{n, hc, lr};
        auto R = vf(hd, -2, 2); auto down = vbf((size_t)lr * hd, 0.3f), up = vbf((size_t)hd * lr, 0.8f), inj = vbf((size_t)hc * n, 0.5f);
        Dev<float> dR(R), dx(n), dw(hc); Dev<uint16_t> dd(down), du(up), di(inj);
        Dev<uint8_t> buf(gr_workspace_bytes(d)); GrWs ws; gr_workspace_init(buf, buf.n, d, &ws);
        ok(gr_read(dR, dd, du, di, ws, d, 1e-6f, dx, dw) == cudaSuccess && sync_ok(), "gr_read launch");
        std::vector<double> rx, rw; gr_ref(R, down, up, inj, n, hc, lr, 1e-6f, rx, rw);
        near_vec(dx.get(), rx, 2e-3, 1e-4, "gr_read block input"); near_vec(dw.get(), rw, 2e-3, 1e-3, "gr_read injection weights");
        gr_set_fp32_activations(true); Dev<float> dx2(n), dw2(hc);
        ok(gr_read(dR, dd, du, di, ws, d, 1e-6f, dx2, dw2) == cudaSuccess && sync_ok(), "gr_read fp32-activation mode launch");
        gr_set_fp32_activations(false); gr_set_native_mmvf(true);
        ok(gr_read(dR, dd, du, di, ws, d, 1e-6f, dx2, dw2) == cudaSuccess && sync_ok(), "gr_read native mmvf mode launch");
        gr_set_native_mmvf(false);
        // write: out[i] = R[i] + block_out[i % n] * w[i / n]
        auto bo = vf(n, -1, 1), ww = vf(hc, -1, 1); Dev<float> dbo(bo), dww(ww), dout(hd);
        ok(gr_write(dR, dbo, dww, dout, n, hc) == cudaSuccess && sync_ok(), "gr_write launch");
        std::vector<double> rwr(hd); for (int i = 0; i < hd; ++i) rwr[i] = (double)R[i] + (double)bo[i % n] * ww[i / n];
        near_vec(dout.get(), rwr, 1e-6, 1e-6, "gr_write: every stream adds the same block output scaled by its weight");
    }
    { // fused kernels at the fixed geometry
        const int n = 2560, hc = 4, lr = 320, hd = n * hc; GrDims d{n, hc, lr};
        auto R = vf(hd, -2, 2); auto down = vbf((size_t)lr * hd, 0.05f), up = vbf((size_t)hd * lr, 0.3f), inj = vbf((size_t)hc * n, 0.2f);
        Dev<float> dR(R), dx(n), dw(hc), fx(n), fw(hc), lo(lr); Dev<uint16_t> dd(down), du(up), di(inj);
        Dev<uint8_t> buf(gr_workspace_bytes(d)); GrWs ws; gr_workspace_init(buf, buf.n, d, &ws);
        ok(gr_read(dR, dd, du, di, ws, d, 1e-6f, dx, dw) == cudaSuccess && fused_gr_read(dR, dd, du, di, lo, fx, fw, 1e-6f) == cudaSuccess && sync_ok(), "fused/plain gr read launch");
        near_vec_f(fx.get(), dx.get(), 3e-3, 2e-4, "fused_gr_read block input equals gr_read"); near_vec_f(fw.get(), dw.get(), 3e-3, 2e-3, "fused_gr_read injection equals gr_read");
        const int T = 3; auto RT = vf((size_t)T * hd, -2, 2); Dev<float> dRT(RT), mlo((size_t)T * lr), mx((size_t)T * n), mw((size_t)T * hc);
        GrMulti m; m.R = dRT; m.down = dd; m.up = du; m.inject = di; m.lo = mlo; m.x_out = mx; m.w_out = mw; m.eps = 1e-6f; m.T = T;
        ok(fused_gr_read_multi(m) == cudaSuccess && sync_ok(), "fused_gr_read_multi launch");
        auto MX = mx.get(), MW = mw.get(); bool same = true;
        for (int t = 0; t < T; ++t) {
            std::vector<float> Rt(RT.begin() + (size_t)t * hd, RT.begin() + (size_t)(t + 1) * hd); Dev<float> dRt(Rt), x1(n), w1(hc), l1(lr);
            fused_gr_read(dRt, dd, du, di, l1, x1, w1, 1e-6f); sync_ok(); auto X = x1.get(), W = w1.get();
            for (int i = 0; i < n; ++i) if (std::fabs(X[i] - MX[(size_t)t * n + i]) > 1e-4 * (1 + std::fabs(X[i]))) same = false;
            for (int c = 0; c < hc; ++c) if (std::fabs(W[c] - MW[t * hc + c]) > 1e-3 * (1 + std::fabs(W[c]))) same = false;
        }
        ok(same, "multi-token fused read equals per-token fused read");
        GrMulti bad = m; bad.T = 33; ok(fused_gr_read_multi(bad) != cudaSuccess, "fused multi rejects T > 32");
    }
    { // native norm / post
        const int rows = 3, cols = 200; auto x = vf(rows * cols, -2, 2), w = vf(cols, 0.5f, 1.5f);
        Dev<float> dx(x), dw(w), dy(rows * cols);
        ok(native_gr_rms_norm_weighted(dx, dw, dy, rows, cols, 1e-6f) == cudaSuccess && sync_ok(), "native rms norm launch");
        std::vector<double> r(rows * cols);
        for (int q = 0; q < rows; ++q) { double s = 0; for (int i = 0; i < cols; ++i) s += (double)x[q * cols + i] * x[q * cols + i]; const double inv = 1.0 / std::sqrt(s / cols + 1e-6); for (int i = 0; i < cols; ++i) r[q * cols + i] = x[q * cols + i] * inv * w[i]; }
        near_vec(dy.get(), r, 1e-5, 1e-6, "native_gr_rms_norm_weighted");
        ok(native_gr_rms_norm_weighted(dx, dw, dx, rows, cols, 1e-6f) == cudaSuccess && sync_ok(), "in-place (aliased) norm allowed");
        near_vec(dx.get(), r, 1e-5, 1e-6, "aliased norm gives the same result");
        ok(native_gr_rms_norm_weighted(dx, dw, (float*)dx.p + 1, rows, cols, 1e-6f) != cudaSuccess, "partially overlapping norm output rejected");
        const int n = 50, hc = 4; auto R = vf(n * hc, -1, 1), bo = vf(n, -1, 1), ww = vf(hc, -1, 1), bias = vf(hc, -1, 1), lg = vf(n * hc, -3, 3), xn = vf(n * hc, -1, 1);
        Dev<float> dR(R), dbo(bo), dww(ww), dbi(bias), dout(n * hc), dlg(lg), dxn(xn), dga(n * hc);
        ok(native_gr_post(dR, dbo, dww, dbi, 0.5f, dout, n, hc) == cudaSuccess && sync_ok(), "native_gr_post launch");
        std::vector<double> rp(n * hc); for (int i = 0; i < n * hc; ++i) rp[i] = (double)R[i] + (double)bo[i % n] * ((double)ww[i / n] * 0.5 + bias[i / n]);
        near_vec(dout.get(), rp, 1e-5, 1e-6, "native_gr_post with bias");
        ok(native_gr_post(dR, dbo, dww, nullptr, 2.f, dout, n, hc) == cudaSuccess && sync_ok(), "native_gr_post zero-bias launch");
        for (int i = 0; i < n * hc; ++i) rp[i] = (double)R[i] + (double)bo[i % n] * ((double)ww[i / n] * 2.0);
        near_vec(dout.get(), rp, 1e-5, 1e-6, "native_gr_post without bias");
        ok(native_gr_pre_gated(dxn, dlg, dga, n * hc) == cudaSuccess && sync_ok(), "pre_gated launch");
        std::vector<double> rgd(n * hc); for (int i = 0; i < n * hc; ++i) rgd[i] = xn[i] * sigmoid_d(lg[i]);
        near_vec(dga.get(), rgd, 1e-5, 1e-6, "native_gr_pre_gated = xn * sigmoid(logit)");
        ok(native_gr_down_silu(dlg, n * hc) == cudaSuccess && sync_ok(), "down_silu launch");
        std::vector<double> rs(n * hc); for (int i = 0; i < n * hc; ++i) rs[i] = silu_d(lg[i]);
        near_vec(dlg.get(), rs, 1e-5, 1e-6, "native_gr_down_silu");
        ok(native_gr_post(nullptr, dbo, dww, nullptr, 1.f, dout, n, hc) != cudaSuccess && native_gr_down_silu(dlg, 0) != cudaSuccess, "native gr ops validate arguments");
    }
}

// ------------------------------------------------------------------------------------------------------ MoE
static void router_ref(const std::vector<float>& lg, int T, int E, int k, std::vector<int>& ids, std::vector<float>& wt) {
    ids.assign((size_t)T * k, -1); wt.assign((size_t)T * k, 0.f);
    for (int t = 0; t < T; ++t) {
        const float* l = &lg[(size_t)t * E]; float mx = -INFINITY; for (int e = 0; e < E; ++e) mx = std::max(mx, l[e]);
        std::vector<double> ex(E); double sum = 0; for (int e = 0; e < E; ++e) { ex[e] = std::exp((double)l[e] - (double)mx); sum += ex[e]; }
        const float inv = (float)(1.0 / sum); std::vector<float> p(E); for (int e = 0; e < E; ++e) p[e] = (float)(ex[e] * (double)inv);
        std::vector<char> taken(E, 0); std::vector<float> ch(k);
        for (int i = 0; i < k; ++i) { int bi = -1; for (int e = 0; e < E; ++e) { if (taken[e]) continue; if (bi < 0 || p[e] > p[bi]) bi = e; } taken[bi] = 1; ids[(size_t)t * k + i] = bi; ch[i] = p[bi]; }
        double s = 0; for (int i = 0; i < k; ++i) s += ch[i]; const double sc = std::max<double>(s, 6.103515625e-05);
        for (int i = 0; i < k; ++i) wt[(size_t)t * k + i] = (float)((double)ch[i] / sc);
    }
}
static void test_moe() {
    const int T = 5, E = 64, k = 10; auto lg = vf((size_t)T * E, -4, 4);
    Dev<float> dl(lg), w1((size_t)T * k), w2((size_t)T * k); Dev<int32_t> i1((size_t)T * k), i2((size_t)T * k);
    ok(router_top10(dl, T, E, k, i1, w1) == cudaSuccess && native_router_top10(dl, T, E, k, i2, w2) == cudaSuccess && sync_ok(), "router launches");
    std::vector<int> rid; std::vector<float> rwt; router_ref(lg, T, E, k, rid, rwt);
    auto I1 = i1.get(); std::vector<int> I1v(I1.begin(), I1.end());
    ok(I1v == rid, "router_top10 ids = descending probability, lowest index on ties");
    near_vec_f(w1.get(), rwt, 1e-4, 1e-6, "router_top10 weights renormalised");
    ok(i1.get() == i2.get() && w1.get() == w2.get(), "native router is bit-identical to router_top10");
    { auto W = w1.get(); bool one = true; for (int t = 0; t < T; ++t) { double s = 0; for (int i = 0; i < k; ++i) s += W[t * k + i]; if (std::fabs(s - 1) > 1e-5) one = false; } ok(one, "selected weights sum to 1"); }
    { std::vector<float> z(16, 0.f); Dev<float> dz(z), wz(4), wz2(4); Dev<int32_t> iz(4), iz2(4);
      router_top10(dz, 1, 16, 4, iz, wz); native_router_top10(dz, 1, 16, 4, iz2, wz2); sync_ok();
      auto I = iz.get(); auto W = wz.get(); auto I2 = iz2.get();
      ok(I[0] == 0 && I[1] == 1 && I[2] == 2 && I[3] == 3, "router_top10: all-equal logits select experts 0..k-1");
      ok(I2[0] == 0 && I2[1] == 1 && I2[2] == 2 && I2[3] == 3, "native router: all-equal logits select experts 0..k-1");
      okf(std::fabs(W[0] - 0.25f) < 1e-6f, "equal logits give weight 1/k", W[0], 0.25); }
    { Dev<float> big(600); Dev<float> bw(10); Dev<int32_t> bi(10); ok(native_router_top10(big, 1, 600, 10, bi, bw) != cudaSuccess, "native router rejects n_expert > 512"); ok(native_router_top10(big, 1, 64, 10, bi, (float*)bi.p, 0) != cudaSuccess, "native router rejects overlapping outputs"); }
    // shared expert
    const int n = 64, ff = 96, Tn = 2; auto x = vf((size_t)Tn * n, -1, 1); auto gw = vbf((size_t)ff * n, 0.4f), uw = vbf((size_t)ff * n, 0.4f), dw = vbf((size_t)n * ff, 0.4f), sw = vbf(n, 0.4f);
    Dev<float> dx(x), out((size_t)Tn * n), out1(n); Dev<uint16_t> dgw(gw), duw(uw), ddw(dw), dsw(sw); Dev<uint8_t> scr(shared_expert_scratch_bytes(Tn, n, ff));
    auto ref_se = [&](bool native, int t) {
        std::vector<float> h(ff); std::vector<double> y(n);
        for (int f = 0; f < ff; ++f) { double g = 0, u = 0; for (int i = 0; i < n; ++i) { g += (double)bf(gw[(size_t)f * n + i]) * x[t * n + i]; u += (double)bf(uw[(size_t)f * n + i]) * x[t * n + i]; }
            const float gf = (float)g, uf = (float)u; float hv = (float)silu_d(gf) * uf; h[f] = native ? rbf(hv) : rh(hv); }
        double gs = 0; for (int i = 0; i < n; ++i) gs += (double)x[t * n + i] * bf(sw[i]); double gate = sigmoid_d(gs); if (native) gate = rbf((float)gate);
        for (int o = 0; o < n; ++o) { double a = 0; for (int f = 0; f < ff; ++f) a += (double)bf(dw[(size_t)o * ff + f]) * h[f]; y[o] = a * gate; }
        return y; };
    for (int mode = 0; mode < 2; ++mode) {
        shared_expert_set_native_bf16(mode == 1);
        ok(shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, scr, scr.n) == cudaSuccess && sync_ok(), "shared_expert_multi launch");
        auto O = out.get();
        for (int t = 0; t < Tn; ++t) { auto r = ref_se(mode == 1, t); std::vector<float> ot(O.begin() + t * n, O.begin() + (t + 1) * n); near_vec(ot, r, mode ? 8e-3 : 3e-3, 2e-3, mode ? "shared expert (native bf16)" : "shared expert (f16 hidden)"); }
        std::vector<float> x0(x.begin(), x.begin() + n); Dev<float> dx0(x0);
        ok(shared_expert(dx0, n, ff, dgw, duw, ddw, dsw, out1, scr, scr.n) == cudaSuccess && sync_ok(), "shared_expert launch");
        std::vector<float> o0(O.begin(), O.begin() + n); near_vec_f(out1.get(), o0, 1e-5, 1e-6, "single-token shared expert equals the first multi-token row");
    }
    shared_expert_set_native_bf16(false);
    ok(shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, scr, 16) != cudaSuccess, "shared_expert rejects too small scratch");
    // combine
    const int ne = 300, kk = 6; auto parts = vf((size_t)kk * ne, -1, 1), wts = vf(kk, 0, 1), sh = vf(ne, -1, 1);
    Dev<float> dp(parts), dwt(wts), dsh(sh), y1(ne), y2(ne), y3(ne);
    ok(moe_combine(dp, dwt, dsh, y1, ne, kk, 1) == cudaSuccess && native_moe_combine(dp, dwt, dsh, y2, ne, kk, 1) == cudaSuccess && moe_combine(dp, dwt, nullptr, y3, ne, kk, 0) == cudaSuccess && sync_ok(), "combine launches");
    std::vector<double> rc(ne), rc0(ne); for (int j = 0; j < ne; ++j) { double a = 0; for (int e = 0; e < kk; ++e) a += (double)wts[e] * parts[(size_t)e * ne + j]; rc0[j] = a; rc[j] = a + sh[j]; }
    near_vec(y1.get(), rc, 1e-5, 1e-6, "moe_combine adds the shared output plain"); near_vec(y2.get(), rc, 1e-5, 1e-6, "native_moe_combine"); near_vec(y3.get(), rc0, 1e-5, 1e-6, "moe_combine without shared");
    ok(native_moe_combine(dp, dwt, dsh, dp, ne, kk, 1) != cudaSuccess, "native_moe_combine rejects y overlapping parts");
}

// ------------------------------------------------------------------------------------------------------ Q2_0 gemv family
static void test_s2() {
    const int n_in = 1024, n_out = 37, nb = n_in / 64;
    std::vector<uint8_t> w((size_t)n_out * nb * 18);
    for (size_t b = 0; b < (size_t)n_out * nb; ++b) { const uint16_t d = fb::f16_from_f32(rf(-0.5f, 0.5f)); w[b * 18] = d & 0xff; w[b * 18 + 1] = d >> 8; for (int i = 0; i < 16; ++i) w[b * 18 + 2 + i] = (uint8_t)ru(256); }
    auto x = vf(n_in, -1, 1); std::vector<float> xh(n_in); for (int i = 0; i < n_in; ++i) xh[i] = rh(x[i]);
    auto dot = [&](const std::vector<float>& xv) { std::vector<double> r(n_out, 0.0);
        for (int o = 0; o < n_out; ++o) for (int b = 0; b < nb; ++b) { const uint8_t* p = &w[((size_t)o * nb + b) * 18]; const double d = fb::f32_from_f16((uint16_t)(p[0] | (p[1] << 8)));
            for (int i = 0; i < 64; ++i) r[o] += d * (double)(((p[2 + i / 4] >> (2 * (i % 4))) & 3) - 1) * xv[b * 64 + i]; }
        return r; };
    Dev<uint8_t> dw(w); Dev<float> dx(x), y1(n_out), y2(n_out), y3(n_out);
    ok(s2_gemv(dw, dx, y1, n_in, n_out) == cudaSuccess && s2_gemv_quads(dw, dx, y2, n_in, n_out) == cudaSuccess && s2_gemv_fast(dw, dx, y3, n_in, n_out) == cudaSuccess && sync_ok(), "s2 gemv launches");
    near_vec(y1.get(), dot(x), 1e-4, 1e-4, "s2_gemv"); near_vec(y2.get(), dot(x), 1e-4, 1e-4, "s2_gemv_quads");
    near_vec(y3.get(), dot(xh), 1e-4, 1e-4, "s2_gemv_fast equals s2_gemv on f16-rounded x");
    // Q8_0 activation (fp16 d + 32 int8)
    std::vector<uint8_t> xq((size_t)(n_in / 32) * 34); std::vector<float> xd(n_in);
    for (int b = 0; b < n_in / 32; ++b) { float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i])); const uint16_t dh = fb::f16_from_f32(am / 127.f); const float d = fb::f32_from_f16(dh); xq[b * 34] = dh & 0xff; xq[b * 34 + 1] = dh >> 8;
        for (int i = 0; i < 32; ++i) { int q = d > 0 ? (int)std::lround(x[b * 32 + i] / d) : 0; q = std::max(-127, std::min(127, q)); xq[b * 34 + 2 + i] = (uint8_t)(int8_t)q; xd[b * 32 + i] = d * (float)q; } }
    Dev<uint8_t> dxq(xq); Dev<float> y4(n_out);
    ok(s2_gemv_q8(dw, dxq, y4, n_in, n_out) == cudaSuccess && sync_ok(), "s2_gemv_q8 launch");
    near_vec(y4.get(), dot(xd), 2e-4, 1e-4, "s2_gemv_q8 equals the Q2_0 x dequantised-Q8_0 product");
    ok(s2_gemv(dw, dx, y1, 1000, n_out) != cudaSuccess, "s2_gemv rejects n_in % 64 != 0");
    ok(s2_gemv_fast(dw, dx, y1, 64 * 400, n_out) != cudaSuccess, "s2_gemv_fast rejects n_in beyond shared capacity");
}

// ------------------------------------------------------------------------------------------------------ rotary / attention / ple
static void rope_host(std::vector<float>& x, const std::vector<int>& pos3, int n_tok, int nh, int hd, int rot, float theta, const int* comp) {
    const int half = rot / 2;
    for (int t = 0; t < n_tok; ++t) for (int h = 0; h < nh; ++h) for (int i = 0; i < half; ++i) {
        const double a = (double)pos3[t * 3 + comp[i]] * std::pow((double)theta, -2.0 * i / rot); const double c = std::cos(a), s = std::sin(a);
        float* p = &x[((size_t)t * nh + h) * hd]; const double x0 = p[i], x1 = p[i + half]; p[i] = (float)(x0 * c - x1 * s); p[i + half] = (float)(x0 * s + x1 * c); }
}
static void test_rope_qsa_ple() {
    const int n_tok = 4, nh = 3, hd = 64, rot = 32, half = 16, n_pos = 16; const float theta = 10000.f;
    auto x = vf((size_t)n_tok * nh * hd, -1, 1); std::vector<int32_t> pos = {0, 3, 7, 15}; std::vector<int> pos3, comp0(half, 0);
    for (int p : pos) { pos3.push_back(p); pos3.push_back(p + 1); pos3.push_back(p + 2); }
    Dev<float> tab((size_t)n_pos * half * 2), dx(x), dn(x); Dev<int32_t> dpos(pos);
    ok(build_rope_table(tab, n_pos, rot, theta) == cudaSuccess && rope_neox_apply(dx, tab, dpos, n_tok, nh, hd, rot) == cudaSuccess && sync_ok(), "rope launches");
    std::vector<float> ref = x; std::vector<int> p1; for (int p : pos) { p1.push_back(p); p1.push_back(0); p1.push_back(0); }
    rope_host(ref, p1, n_tok, nh, hd, rot, theta, comp0.data());
    near_vec_f(dx.get(), ref, 1e-4, 1e-4, "rope_neox_apply equals the host rotation");
    { auto o = dx.get(); bool same0 = true, pres = true, tail = true;
      for (int i = 0; i < nh * hd; ++i) if (o[i] != x[i]) same0 = false;
      for (int t = 0; t < n_tok; ++t) for (int h = 0; h < nh; ++h) { const size_t b = ((size_t)t * nh + h) * hd; for (int i = 0; i < half; ++i) { const double a = (double)x[b + i] * x[b + i] + (double)x[b + i + half] * x[b + i + half], c = (double)o[b + i] * o[b + i] + (double)o[b + i + half] * o[b + i + half]; if (std::fabs(a - c) > 1e-4) pres = false; } for (int i = rot; i < hd; ++i) if (o[b + i] != x[b + i]) tail = false; }
      ok(same0, "position 0 is the identity"); ok(pres, "rotation preserves every pair norm"); ok(tail, "dims beyond rot are untouched"); }
    Dev<int32_t> dp3(std::vector<int32_t>(pos3.begin(), pos3.end()));
    mrope_table_set(nullptr);
    ok(native_rope_apply(dn, dp3, n_tok, nh, hd, rot, theta) == cudaSuccess, "native rope launch");
    near_vec_f(dn.get(), dx.get(), 1e-5, 1e-5, "native rope with default sections equals table rope");
    const int sec[3] = {6, 5, 5}; mrope_table_set(sec); Dev<float> dm(x);
    ok(native_rope_apply(dm, dp3, n_tok, nh, hd, rot, theta) == cudaSuccess, "mrope launch");
    std::vector<int> comp(half); for (int i = 0; i < half; ++i) { comp[i] = i < 6 ? 0 : (i < 11 ? 1 : 2); ok(mrope_table(i, half) == comp[i], "mrope_table sections"); }
    ref = x; rope_host(ref, pos3, n_tok, nh, hd, rot, theta, comp.data()); near_vec_f(dm.get(), ref, 1e-4, 1e-4, "mrope uses a different position component per frequency section");
    const int bad[3] = {6, 5, 4}; mrope_table_set(bad); ok(native_rope_apply(dm, dp3, n_tok, nh, hd, rot, theta) != cudaSuccess, "sections must sum to rot/2");
    mrope_table_set(nullptr);
    ok(rope_neox_apply(dx, tab, dpos, n_tok, nh, hd, 33) != cudaSuccess, "odd rot rejected");
    native_rope_set_enabled(false); ok(!native_rope_enabled(), "rope switch"); native_rope_set_enabled(true);

    // qsa gate / norm
    auto o = vf(500, -2, 2), g = vf(500, -3, 3); Dev<float> dout(o), dg(g);
    ok(native_qsa_gate_apply(dout, dg, 500) == cudaSuccess && sync_ok(), "qsa gate launch");
    std::vector<double> rg(500); for (int i = 0; i < 500; ++i) rg[i] = o[i] * sigmoid_d(g[i]); near_vec(dout.get(), rg, 1e-5, 1e-6, "qsa gate = out * sigmoid(gate)");
    ok(native_qsa_gate_apply(dout, (float*)dout.p + 3, 100) != cudaSuccess, "qsa gate rejects overlapping buffers");
    const int rows = 6, cols = 128; auto qx = vf(rows * cols, -2, 2), qw = vf(cols, 0.5f, 1.5f); Dev<float> dqx(qx), dqw(qw);
    ok(native_qsa_rms_norm_weighted(dqx, dqw, rows, cols, 1e-6f) == cudaSuccess && sync_ok(), "qsa norm launch");
    std::vector<double> rq(rows * cols); for (int r = 0; r < rows; ++r) { double s = 0; for (int i = 0; i < cols; ++i) s += (double)qx[r * cols + i] * qx[r * cols + i]; const double inv = 1.0 / std::sqrt(s / cols + 1e-6); for (int i = 0; i < cols; ++i) rq[r * cols + i] = qx[r * cols + i] * inv * qw[i]; }
    near_vec(dqx.get(), rq, 1e-5, 1e-6, "qsa per-row weighted rms norm");
    native_qsa_set_enabled(false); ok(!native_qsa_enabled(), "qsa switch"); native_qsa_set_enabled(true);
    // indexer append
    const int cap = 4; std::vector<float> sentinel(cap * 128, 7.f); Dev<float> cache(sentinel), norms(std::vector<float>(cap, -1.f)); auto key = vf(128, -1, 1); Dev<float> dkey(key);
    ok(native_qsa_indexer_append(cache, norms, dkey, 2, cap) == cudaSuccess && sync_ok(), "indexer append launch");
    { auto C = cache.get(); auto N = norms.get(); double s = 0, kn = 0; for (int i = 0; i < 128; ++i) { s += (double)C[2 * 128 + i] * C[2 * 128 + i]; kn += (double)key[i] * key[i]; }
      okf(std::fabs(s - 1) < 1e-5, "appended key row is unit length", s, 1); okf(std::fabs(N[2] - std::sqrt(kn)) < 1e-4, "norm of the original key recorded", N[2], std::sqrt(kn));
      bool untouched = true; for (int i = 0; i < 128; ++i) if (C[i] != 7.f || C[3 * 128 + i] != 7.f) untouched = false; ok(untouched && N[0] == -1.f, "other rows untouched"); }
    Dev<float> zk(std::vector<float>(128, 0.f)); native_qsa_indexer_append(cache, norms, zk, 1, cap); sync_ok();
    { auto C = cache.get(); bool z = true; for (int i = 0; i < 128; ++i) if (C[128 + i] != 0.f || std::isnan(C[128 + i])) z = false; ok(z && norms.get()[1] == 0.f, "zero key stored as zeros, norm 0"); }
    ok(native_qsa_indexer_append(cache, norms, dkey, cap, cap) != cudaSuccess && native_qsa_indexer_append(cache, norms, nullptr, 0, cap) != cudaSuccess, "indexer append validates position and pointers");
    native_qsa_indexer_set_enabled(false); ok(!native_qsa_indexer_enabled(), "indexer switch"); native_qsa_indexer_set_enabled(true);

    // ple postops (width not a multiple of the block)
    const int pr = 3, pc = 300; auto gt = vf(pr * pc, -3, 3), pl = vf(pr * pc, -1, 1), pj = vf(pr * pc, -2, 2), rs = vf(pr * pc, -1, 1), nw = vf(pc, 0.5f, 1.5f);
    std::vector<float> sc = {0.7f};
    Dev<float> dgt(gt), dpl(pl), dt(pr * pc), dpj(pj), dnw(nw), drs(rs), dsc(sc), dout2(pr * pc);
    ok(native_ple_postops(dgt, dpl, dt, dpj, dnw, drs, dsc, dout2, pr, pc, 1e-6f) == cudaSuccess && sync_ok(), "ple postops launch");
    std::vector<double> rt(pr * pc), ro(pr * pc);
    for (int i = 0; i < pr * pc; ++i) { const double gg = gt[i]; rt[i] = 0.5 * gg * (1.0 + std::tanh(0.7978845608028654 * (gg + 0.044715 * gg * gg * gg))) * pl[i]; }
    for (int r = 0; r < pr; ++r) { double s = 0; for (int i = 0; i < pc; ++i) s += (double)pj[r * pc + i] * pj[r * pc + i]; const double inv = 1.0 / std::sqrt(s / pc + 1e-6); for (int i = 0; i < pc; ++i) ro[r * pc + i] = ((double)rs[r * pc + i] + pj[r * pc + i] * inv * nw[i]) * 0.7f; }
    near_vec(dt.get(), rt, 1e-4, 1e-5, "ple gate = gelu_tanh(gate) * ple"); near_vec(dout2.get(), ro, 1e-4, 1e-5, "ple out = (resid + rmsnorm(proj) * w) * layer_scalar");
    ok(native_ple_postops(dgt, dpl, dt, dpj, dnw, drs, dsc, drs, pr, pc, 1e-6f) != cudaSuccess, "ple postops rejects out aliasing resid");
}

// ------------------------------------------------------------------------------------------------------ block formats
static const int NF = 10;
static const Fmt kFmts[NF] = {Fmt::Q2_0, Fmt::Q3_K, Fmt::IQ2_XXS, Fmt::IQ2_XS, Fmt::IQ3_XXS, Fmt::IQ4_NL, Fmt::IQ3_S, Fmt::IQ2_S, Fmt::IQ4_XS, Fmt::IQ1_M};
static const int kElems[NF] = {64, 256, 256, 256, 256, 32, 256, 256, 256, 256};
static const int kBytes[NF] = {18, 110, 66, 74, 98, 18, 110, 82, 136, 56};
static void put16(uint8_t* p, uint16_t v) { p[0] = v & 0xff; p[1] = v >> 8; }
static uint32_t get16(const uint8_t* p) { return p[0] | (p[1] << 8); }
static uint32_t get32(const uint8_t* p) { return get16(p) | (get16(p + 2) << 16); }
// random block with a finite scale
static void rand_block(Fmt f, uint8_t* b, int bytes) {
    for (int i = 0; i < bytes; ++i) b[i] = (uint8_t)ru(256);
    const uint16_t d = fb::f16_from_f32(rf(0.05f, 1.f));
    if (f == Fmt::Q3_K) put16(b + 108, d);
    else if (f == Fmt::IQ1_M) { uint16_t* s = nullptr; (void)s; uint16_t sc[4]; for (int k = 0; k < 4; ++k) sc[k] = (uint16_t)get16(b + 48 + 2 * k);
        sc[0] = (sc[0] & 0x0fff) | ((d & 0xf) << 12); sc[1] = (sc[1] & 0x0fff) | (((d >> 4) & 0xf) << 12); sc[2] = (sc[2] & 0x0fff) | (((d >> 8) & 0xf) << 12); sc[3] = (sc[3] & 0x0fff) | (((d >> 12) & 0xf) << 12);
        for (int k = 0; k < 4; ++k) put16(b + 48 + 2 * k, sc[k]); }
    else put16(b, d);
}
struct HostGrids { std::vector<uint64_t> g2xxs, g2xs, g2s, g1s; std::vector<uint32_t> g3xxs, g3s; };
static uint8_t ksign(int i) { return (uint8_t)(i | ((__builtin_popcount(i) & 1) << 7)); }
// group-wise decode of one block, structured like the format definition (independent of the per-element device decoder)
static void host_decode(Fmt f, const uint8_t* b, float* y, const HostGrids& H) {
    switch (f) {
        case Fmt::Q2_0: sdq::deq_q2_0(b, y); return;
        case Fmt::IQ4_NL: sdq::deq_iq4_nl(b, y); return;
        case Fmt::IQ4_XS: sdq::deq_iq4_xs(b, y); return;
        case Fmt::Q3_K: sdq::deq_q3_k(b, y); return;
        case Fmt::IQ2_XXS: { const float d = fb::f32_from_f16((uint16_t)get16(b)); const uint8_t* q = b + 2;
            for (int ib = 0; ib < 8; ++ib) { const uint32_t aux = get32(q + 4); const float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) { const uint8_t s = ksign((aux >> (7 * l)) & 127); const uint64_t gr = H.g2xxs[q[l]]; for (int j = 0; j < 8; ++j) y[j] = db * (float)((gr >> (8 * j)) & 0xff) * ((s >> j) & 1 ? -1.f : 1.f); y += 8; } q += 8; } return; }
        case Fmt::IQ2_XS: { const float d = fb::f32_from_f16((uint16_t)get16(b)); const uint8_t* q = b + 2; const uint8_t* sc = b + 66;
            for (int ib = 0; ib < 8; ++ib) { const float db0 = d * (0.5f + (float)(sc[ib] & 0xf)) * 0.25f, db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) { const uint32_t qq = get16(q + 2 * l); const uint8_t s = ksign(qq >> 9); const uint64_t gr = H.g2xs[qq & 511]; const float db = l < 2 ? db0 : db1; for (int j = 0; j < 8; ++j) y[j] = db * (float)((gr >> (8 * j)) & 0xff) * ((s >> j) & 1 ? -1.f : 1.f); y += 8; } q += 8; } return; }
        case Fmt::IQ2_S: { const float d = fb::f32_from_f16((uint16_t)get16(b)); const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sc = b + 74; const uint8_t* sg = qs + 32;
            for (int ib = 0; ib < 8; ++ib) { const float db0 = d * (0.5f + (float)(sc[ib] & 0xf)) * 0.25f, db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) { const uint64_t gr = H.g2s[qs[l] | (((uint32_t)qh[ib] << (8 - 2 * l)) & 0x300)]; const float db = l < 2 ? db0 : db1; for (int j = 0; j < 8; ++j) y[j] = db * (float)((gr >> (8 * j)) & 0xff) * ((sg[l] >> j) & 1 ? -1.f : 1.f); y += 8; } qs += 4; sg += 4; } return; }
        case Fmt::IQ3_XXS: { const float d = fb::f32_from_f16((uint16_t)get16(b)); const uint8_t* q = b + 2; const uint8_t* sa = q + 64;
            for (int ib = 0; ib < 8; ++ib) { const uint32_t aux = get32(sa + 4 * ib); const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                for (int l = 0; l < 4; ++l) { const uint8_t s = ksign((aux >> (7 * l)) & 127); const uint32_t g1 = H.g3xxs[q[2 * l]], g2 = H.g3xxs[q[2 * l + 1]];
                    for (int j = 0; j < 4; ++j) { y[j] = db * (float)((g1 >> (8 * j)) & 0xff) * ((s >> j) & 1 ? -1.f : 1.f); y[j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * ((s >> (j + 4)) & 1 ? -1.f : 1.f); } y += 8; } q += 8; } return; }
        case Fmt::IQ3_S: { const float d = fb::f32_from_f16((uint16_t)get16(b)); const uint8_t* qs = b + 2; const uint8_t* qh = b + 66; const uint8_t* sg = b + 74; const uint8_t* sc = b + 106;
            for (int ib = 0; ib < 8; ib += 2) { const float db1 = d * (1 + 2 * (sc[ib / 2] & 0xf)), db2 = d * (1 + 2 * (sc[ib / 2] >> 4));
                for (int half = 0; half < 2; ++half) { const float db = half ? db2 : db1;
                    for (int l = 0; l < 4; ++l) { const uint32_t g1 = H.g3s[qs[2 * l] | (((uint32_t)qh[0] << (8 - 2 * l)) & 256)], g2 = H.g3s[qs[2 * l + 1] | (((uint32_t)qh[0] << (7 - 2 * l)) & 256)];
                        for (int j = 0; j < 4; ++j) { y[j] = db * (float)((g1 >> (8 * j)) & 0xff) * ((sg[l] >> j) & 1 ? -1.f : 1.f); y[j + 4] = db * (float)((g2 >> (8 * j)) & 0xff) * ((sg[l] >> (j + 4)) & 1 ? -1.f : 1.f); } y += 8; }
                    qs += 8; sg += 4; qh += 1; } } return; }
        case Fmt::IQ1_M: { const uint8_t* qs = b; const uint8_t* qh = b + 32; const uint16_t sc[4] = {(uint16_t)get16(b + 48), (uint16_t)get16(b + 50), (uint16_t)get16(b + 52), (uint16_t)get16(b + 54)};
            const uint16_t db16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)); const float d = fb::f32_from_f16(db16);
            for (int ib = 0; ib < 8; ++ib) { const float dl1 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1), dl2 = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
                const uint32_t idx[4] = {(uint32_t)qs[0] | (((uint32_t)qh[0] << 8) & 0x700), (uint32_t)qs[1] | (((uint32_t)qh[0] << 4) & 0x700), (uint32_t)qs[2] | (((uint32_t)qh[1] << 8) & 0x700), (uint32_t)qs[3] | (((uint32_t)qh[1] << 4) & 0x700)};
                const float delta[4] = {(qh[0] & 0x08) ? -0.125f : 0.125f, (qh[0] & 0x80) ? -0.125f : 0.125f, (qh[1] & 0x08) ? -0.125f : 0.125f, (qh[1] & 0x80) ? -0.125f : 0.125f};
                for (int l = 0; l < 4; ++l) { const uint64_t gr = H.g1s[idx[l]]; const float dl = l < 2 ? dl1 : dl2; for (int j = 0; j < 8; ++j) y[j] = dl * ((float)(int8_t)((gr >> (8 * j)) & 0xff) + delta[l]); y += 8; }
                qs += 4; qh += 2; } return; }
    }
}
static void q8_host(const std::vector<float>& x, std::vector<uint8_t>& q, std::vector<float>& deq) {
    const int nb = (int)x.size() / 32; q.assign((size_t)nb * 36, 0); deq.assign(x.size(), 0.f);
    for (int b = 0; b < nb; ++b) { float am = 0; for (int i = 0; i < 32; ++i) am = std::max(am, std::fabs(x[b * 32 + i])); const float d = am / 127.f, id = d > 0 ? 1.f / d : 0.f; int sum = 0;
        for (int i = 0; i < 32; ++i) { const int v = (int)std::lround((double)(x[b * 32 + i] * id)); sum += v; q[b * 36 + 4 + i] = (uint8_t)(int8_t)v; }
        for (int i = 0; i < 32; ++i) deq[b * 32 + i] = fb::f32_from_f16(fb::f16_from_f32(d)) * (float)(int8_t)q[b * 36 + 4 + i];
        put16(&q[b * 36], fb::f16_from_f32(d)); put16(&q[b * 36 + 2], fb::f16_from_f32(d * (float)sum)); }
}
static void test_formats() {
    // geometry
    for (int i = 0; i < NF; ++i) { ok(iq_block_elems(kFmts[i]) == kElems[i] && iq_block_bytes(kFmts[i]) == kBytes[i], "block geometry of a format"); ok(iq_row_bytes(kFmts[i], (long long)kElems[i] * 3) == 3LL * kBytes[i] && iq_row_bytes(kFmts[i], kElems[i] + 1) == -1, "row bytes = blocks * block bytes, partial block rejected"); }
    ok(is_iq(Fmt::IQ2_S) && is_iq(Fmt::IQ4_NL) && !is_iq(Fmt::Q2_0) && !is_iq(Fmt::Q3_K), "is_iq classification");
    HostGrids H; H.g2xxs.resize(256); H.g2xs.resize(512); H.g2s.resize(1024); H.g1s.resize(2048); H.g3xxs.resize(256); H.g3s.resize(512);
    for (auto& v : H.g2xxs) v = rng(); for (auto& v : H.g2xs) v = rng(); for (auto& v : H.g2s) v = rng(); for (auto& v : H.g1s) { v = 0; for (int j = 0; j < 8; ++j) v |= (uint64_t)(uint8_t)(int8_t)((int)ru(3) - 1) << (8 * j); }
    for (auto& v : H.g3xxs) v = (uint32_t)rng(); for (auto& v : H.g3s) v = (uint32_t)rng();
    // magnitude grids hold small values in the real formats; keep products finite and comparable
    for (auto& v : H.g2xxs) v &= 0x0f0f0f0f0f0f0f0full; for (auto& v : H.g2xs) v &= 0x0f0f0f0f0f0f0f0full; for (auto& v : H.g2s) v &= 0x0f0f0f0f0f0f0f0full;
    for (auto& v : H.g3xxs) v &= 0x0f0f0f0fu; for (auto& v : H.g3s) v &= 0x0f0f0f0fu;
    Dev<uint64_t> a2xxs(H.g2xxs), a2xs(H.g2xs), a2s(H.g2s), a1s(H.g1s); Dev<uint32_t> a3xxs(H.g3xxs), a3s(H.g3s);
    IqGrids G; G.iq2xxs = a2xxs; G.iq2xs = a2xs; G.iq2s = a2s; G.iq3xxs = a3xxs; G.iq3s = a3s; G.iq1s = a1s;
    { // sign parity law through a literal block: IQ2_XXS, d=1, grid[0]=all ones, sign index 1 -> bits {0,7} negative
        std::vector<uint8_t> blk(66, 0); put16(blk.data(), 0x3C00); blk[2 + 8 + 0] = 0;  /* second group idx 0 */
        // group 0: aux32 (bytes 4..7 of the group) = 1 -> sign index 1 for l = 0, ls = 0
        blk[2 + 4] = 1; std::vector<uint64_t> g(256, 0); g[0] = 0x0101010101010101ull; Dev<uint64_t> dg(g); IqGrids G2; G2.iq2xxs = dg; Dev<uint8_t> db(blk); Dev<float> o(256);
        ok(iq_dequant_f32(Fmt::IQ2_XXS, db, o, 256, G2) == cudaSuccess && sync_ok(), "literal iq2_xxs launch"); auto y = o.get();
        bool lit = true; for (int j = 0; j < 8; ++j) { const float e = (j == 0 || j == 7) ? -0.125f : 0.125f; if (y[j] != e) lit = false; } ok(lit, "ksigns: index 1 -> sign bits 0 and 7 (parity bit), magnitude 0.5*0.25*d");
    }
    for (int fi = 0; fi < NF; ++fi) {
        const Fmt f = kFmts[fi]; const int E = kElems[fi], B = kBytes[fi], nbk = 6 * 256 / E; const long long n = (long long)nbk * E;
        std::vector<uint8_t> w((size_t)nbk * B); for (int b = 0; b < nbk; ++b) rand_block(f, &w[(size_t)b * B], B);
        std::vector<float> ref(n); for (int b = 0; b < nbk; ++b) host_decode(f, &w[(size_t)b * B], &ref[(size_t)b * E], H);
        Dev<uint8_t> dw(w); Dev<float> d32(n); Dev<uint16_t> d16(n);
        ok(iq_dequant_f32(f, dw, d32, n, G) == cudaSuccess && iq_dequant_f16(f, dw, d16, n, G) == cudaSuccess && sync_ok(), "iq dequant launch");
        near_vec_f(d32.get(), ref, 1e-6, 1e-6, "dequantized elements equal the group-wise host decoder");
        { auto h = d16.get(); bool eq = true; for (long long i = 0; i < n; ++i) if (h[i] != fb::f16_from_f32(ref[i]) && std::fabs(fb::f32_from_f16(h[i]) - ref[i]) > 1e-3 * std::fabs(ref[i])) eq = false; ok(eq, "f16 dequant is the f16 rounding of the f32 decode"); }
        ok(iq_dequant_f32(f, dw, d32, n + 1, G) != cudaSuccess, "dequant rejects a partial block");
        if (is_iq(f) && f != Fmt::IQ4_NL && f != Fmt::IQ4_XS) { IqGrids none; ok(!iq_supported(f, none) && iq_dequant_f32(f, dw, d32, n, none) != cudaSuccess, "grid-based format without its grid is unsupported"); }
        // mmvq against the dequantized rows and the host-decoded Q8_1 activation
        const int rows = 5, cols = 512, nb = cols / E; std::vector<uint8_t> mw((size_t)rows * nb * B); for (size_t b = 0; b < (size_t)rows * nb; ++b) rand_block(f, &mw[b * B], B);
        std::vector<float> wf((size_t)rows * cols); for (size_t b = 0; b < (size_t)rows * nb; ++b) host_decode(f, &mw[b * B], &wf[b * E], H);
        auto x = vf(cols, -1, 1); std::vector<uint8_t> q8; std::vector<float> xd; q8_host(x, q8, xd);
        Dev<uint8_t> dmw(mw), dq8(std::vector<uint8_t>((size_t)(cols / 32) * 36)); Dev<float> dx(x), dy(rows);
        ok(quantize_q8_1_rows(dx, dq8, 1, cols) == cudaSuccess && sync_ok(), "quantize_q8_1 launch");
        { auto g = dq8.get(); size_t bad = 0; for (size_t i = 0; i < g.size(); ++i) if (g[i] != q8[i]) ++bad; ok(bad == 0, "device Q8_1 blocks equal the host quantizer (d, s, int8)"); }
        ok(iq_mmvq(f, dmw, dq8, dy, rows, cols, G) == cudaSuccess && sync_ok(), "iq_mmvq launch");
        std::vector<double> r(rows, 0.0), mag(rows, 0.0); for (int o = 0; o < rows; ++o) for (int c = 0; c < cols; ++c) { r[o] += (double)wf[(size_t)o * cols + c] * xd[c]; mag[o] += std::fabs((double)wf[(size_t)o * cols + c] * xd[c]); }
        { auto y = dy.get(); bool good = true; for (int o = 0; o < rows; ++o) if (std::fabs(y[o] - r[o]) > 1e-4 * mag[o] + 1e-4) good = false; ok(good, "iq_mmvq = dequantized row . dequantized Q8_1 activation"); }
        // gate+up dequant and embedding gather
        Dev<uint16_t> gu((size_t)2 * rows * cols); ok(iq_dequant_gu_f16(f, dmw, dmw, gu, rows, cols, G) == cudaSuccess && sync_ok(), "gate+up dequant launch");
        { auto g = gu.get(); bool good = true; for (int i = 0; i < rows * cols; ++i) if (g[i] != g[(size_t)rows * cols + i]) good = false; ok(good, "gate rows then up rows (same source -> identical halves)"); }
        std::vector<int32_t> ids = {2, 0, rows, -1}; Dev<int32_t> dids(ids); Dev<float> emb((size_t)ids.size() * cols);
        ok(iq_embed_rows(f, dmw, rows, dids, (int)ids.size(), cols, emb, G) == cudaSuccess && sync_ok(), "embed rows launch");
        { auto e = emb.get(); bool good = true; for (int c = 0; c < cols; ++c) { if (std::fabs(e[c] - wf[(size_t)2 * cols + c]) > 1e-6 * std::fabs(wf[2 * cols + c]) + 1e-6) good = false; if (e[2 * cols + c] != 0.f || e[3 * cols + c] != 0.f) good = false; } ok(good, "embed rows decode the selected row, bad ids give zero rows"); }
    }
    { // Q8_1 literal laws
        std::vector<float> x(64, 0.f); for (int i = 0; i < 32; ++i) x[i] = (i == 5) ? 127.f : (i == 6 ? -63.5f : 0.f); Dev<float> dx(x); Dev<uint8_t> dq(2 * 36);
        quantize_q8_1_rows(dx, dq, 1, 64); sync_ok(); auto q = dq.get();
        ok(fb::f32_from_f16((uint16_t)get16(&q[0])) == 1.f && (int8_t)q[4 + 5] == 127 && (int8_t)q[4 + 6] == -64, "Q8_1: d = amax/127, ties round away from zero");
        ok(fb::f32_from_f16((uint16_t)get16(&q[2])) == 63.f, "Q8_1: s = d * sum(q)");
        ok(fb::f32_from_f16((uint16_t)get16(&q[36])) == 0.f && (int8_t)q[36 + 4] == 0, "Q8_1: all-zero block has d = 0 and zero quants");
    }
    { // expert evaluation: layout, scratch, result
        const Fmt f = Fmt::IQ4_NL; const int n_embd = 64, n_ff = 96; ExpertLayout L;
        ok(native_expert_layout(f, n_embd, n_ff, &L) && L.gate_off == 0 && L.up_off == (size_t)(n_embd / 32 * 18) * n_ff && L.down_off == 2 * L.up_off && L.total == L.down_off + (size_t)(n_ff / 32 * 18) * n_embd, "expert layout gate | up | down");
        ok(!native_expert_layout(f, 33, n_ff, &L), "expert layout rejects partial blocks");
        HostGrids H0; std::vector<uint8_t> blob(L.total); const int B = 18; for (size_t b = 0; b < L.total / B; ++b) rand_block(f, &blob[b * B], B);
        auto x = vf(n_embd, -1, 1); Dev<uint8_t> db(blob), scr(native_expert_scratch_bytes(n_embd, n_ff)); Dev<float> dx(x), dout(n_embd); IqGrids none;
        ok(native_expert_grouped(f, db, dx, n_embd, n_ff, dout, scr, scr.n, none) == cudaSuccess && sync_ok(), "native_expert_grouped launch");
        auto mv = [&](size_t off, int rows, int cols, const std::vector<float>& xd) { std::vector<float> o(rows); for (int r = 0; r < rows; ++r) { double a = 0; std::vector<float> row(cols); for (int b = 0; b < cols / 32; ++b) host_decode(f, &blob[off + ((size_t)r * (cols / 32) + b) * B], &row[b * 32], H0); for (int c = 0; c < cols; ++c) a += (double)row[c] * xd[c]; o[r] = (float)a; } return o; };
        std::vector<uint8_t> q; std::vector<float> xd; q8_host(x, q, xd);
        auto g = mv(L.gate_off, n_ff, n_embd, xd), u = mv(L.up_off, n_ff, n_embd, xd); std::vector<float> h(n_ff); for (int i = 0; i < n_ff; ++i) h[i] = (float)silu_d(g[i]) * u[i];
        std::vector<float> hd; q8_host(h, q, hd); auto o = mv(L.down_off, n_embd, n_ff, hd); near_vec_f(dout.get(), o, 1e-2, 2e-2, "native expert = quantize, gate/up, SwiGLU, quantize, down");
        ok(native_expert_grouped(f, db, dx, n_embd, n_ff, dout, scr, 8, none) != cudaSuccess, "native expert rejects too small scratch");
    }
}

int main() {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU\n"); return 2; }
    cudaSetDevice(0);
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_dense(); test_gdn(); test_gr(); test_moe(); test_s2(); test_rope_qsa_ple(); test_formats();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT 32 TESTS FAILED\n"); return 1; }
    std::printf("UNIT 32 TESTS PASSED\n");
    return 0;
}
