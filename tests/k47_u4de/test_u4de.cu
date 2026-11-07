// Unit u4decb4e657 invariants of the PRODUCTION kernels in src/kernels/gpu (gated delta net: gdn_core / gdn_fused / gdn_native /
// gdn_native_pre; gated residual: gr_core / gr_fused / gr_native_*; moe_native), checked on the device against independent
// host references in double precision. Invariants (theory LAYER 1):
//   * every state cell (i, head, col) is owned by exactly one thread: one decay step with beta = 0 scales each cell exactly once
//   * the recurrence equals the host delta rule (decay, contraction with k, rank-1 update, read-out with the UPDATED state)
//   * modulo head pairing head % h_k; h_k | h_v and h_v <= 65535 are required by the native step
//   * the conv history slides: after three steps the history is exactly the last three inputs; tap order oldest..newest
//   * L2 / output norms: unit rows, width exactly 128, finite non-negative epsilon, count in [1, 65535]
//   * spans: non-null, float-aligned, non-wrapping, written spans disjoint from inputs
//   * gated residual: workspace regions disjoint, 16-byte aligned, 104320 bytes at the real geometry (xq sized hc_dim*2 bytes);
//     fused read (T = 1 and multi-token) equals the unfused read; token t of a batch equals the single-token result
//   * write: out[i] = R[i] + block_out[i % n] * w[i / n]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kg_kernels.hpp"

using namespace baloon::kg;
static cudaStream_t g_s = nullptr;   // explicit non-null stream for native_gdn_* (null is rejected)
namespace fb = baloon::f16bits;

static int g_fail = 0, g_checks = 0;
static std::mt19937_64 rng(4242);
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static void ok(bool c, const char* what) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", what); } }

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
static double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sigm(x); }
static double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

// worst relative-or-absolute error of a against r
static size_t g_wi = 0; static double g_wa = 0, g_wr = 0;
static double worst_err(const std::vector<float>& a, const std::vector<double>& r, double atol) {
    if (a.size() != r.size()) return 1e30;
    double w = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i])) { g_wi = i; g_wa = a[i]; g_wr = r[i]; return 1e30; }
        const double e = std::fabs((double)a[i] - r[i]) / (atol + std::fabs(r[i]));
        if (e > w) { w = e; g_wi = i; g_wa = a[i]; g_wr = r[i]; }
    }
    return w;
}
static void near(const std::vector<float>& a, const std::vector<double>& r, double rtol, double atol, const char* what) {
    const double w = worst_err(a, r, atol);
    ++g_checks;
    if (!(w <= rtol)) { ++g_fail; std::printf("FAIL: %s (worst scaled error %.3g > %.3g at index %zu: got %.9g want %.9g)\n", what, (double)w, (double)rtol, g_wi, g_wa, g_wr); }
}
static void near_f(const std::vector<float>& a, const std::vector<float>& b, double rtol, double atol, const char* what) {
    std::vector<double> r(b.begin(), b.end()); near(a, r, rtol, atol, what);
}

// ---- host references -------------------------------------------------------------------------------------------------------
// state (S, h_v, S): cell (i, h, j) at (i*h_v + h)*S + j
static void ref_step(std::vector<double>& st, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                     const std::vector<float>& gate, const std::vector<float>& beta, std::vector<double>& o, int S, int h_k, int h_v) {
    o.assign((size_t)h_v * S, 0.0);
    for (int h = 0; h < h_v; ++h) {
        const int src = h % h_k;
        const double dec = std::exp((double)gate[h]);
        for (int j = 0; j < S; ++j) {
            double sk = 0;
            for (int i = 0; i < S; ++i) { double& c = st[((size_t)i * h_v + h) * S + j]; c *= dec; sk += c * k[(size_t)src * S + i]; }
            const double d = ((double)v[(size_t)h * S + j] - sk) * beta[h];
            double dot = 0;
            for (int i = 0; i < S; ++i) { double& c = st[((size_t)i * h_v + h) * S + j]; c += k[(size_t)src * S + i] * d; dot += c * q[(size_t)src * S + i]; }
            o[(size_t)h * S + j] = dot;
        }
    }
}
static std::vector<float> to_f(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

// ---- gated delta net ---------------------------------------------------------------------------------------------------------
static void test_gdn_step() {
    for (int cfg = 0; cfg < 3; ++cfg) {
        const int S = 128, h_k = cfg == 0 ? 2 : (cfg == 1 ? 3 : 5), h_v = cfg == 0 ? 4 : (cfg == 1 ? 6 : 5);
        const size_t ns = (size_t)S * h_v * S;
        // (1) ownership: beta = 0, decay 1/2 -> every cell scaled exactly once (twice would give 1/4, never -> 1)
        {
            std::vector<float> st0(ns); for (size_t i = 0; i < ns; ++i) st0[i] = (float)(i % 4093) + 1.f;
            std::vector<float> gate(h_v, -0.6931471806f), beta(h_v, 0.f);
            Dev<float> dq(vf((size_t)h_k * S, -1, 1)), dk(vf((size_t)h_k * S, -1, 1)), dv(vf((size_t)h_v * S, -1, 1)), dg(gate), db(beta), o((size_t)h_v * S);
            Dev<float> sa(st0), sb(st0), sc(st0), od((size_t)h_v * S), oe((size_t)h_v * S), of((size_t)h_v * S);
            ok(gdn_step(sa, dq, dk, dv, dg, db, od, S, h_k, h_v) == cudaSuccess, "gdn_step launch (ownership)");
            ok(native_gdn_step(sb, dq, dk, dv, dg, db, oe, S, h_k, h_v, g_s) == cudaSuccess, "native_gdn_step launch (ownership)");
            ok(sync_ok(), "ownership kernels run");
            for (int w = 0; w < 2; ++w) {
                const auto got = (w == 0 ? sa : sb).get();
                bool good = true;
                for (size_t i = 0; i < ns && good; ++i) good = std::fabs(got[i] - 0.5f * st0[i]) <= 1e-5f * st0[i];
                ok(good, w == 0 ? "gdn_step touches every state cell exactly once" : "native_gdn_step touches every state cell exactly once");
            }
            (void)sc; (void)of; (void)o;
        }
        // (2) full recurrence against the host delta rule, three consecutive steps (state carried between launches)
        {
            std::vector<float> st0 = vf(ns, -0.3f, 0.3f);
            std::vector<double> sref(st0.begin(), st0.end());
            Dev<float> sa(st0), sb(st0), sc(st0);
            Dev<float> od((size_t)h_v * S), oe((size_t)h_v * S), of((size_t)h_v * S);
            for (int step = 0; step < 3; ++step) {
                auto q = vf((size_t)h_k * S, -1, 1), k = vf((size_t)h_k * S, -0.2f, 0.2f), v = vf((size_t)h_v * S, -1, 1);
                auto gate = vf(h_v, -0.5f, -0.01f), beta = vf(h_v, 0.1f, 0.9f);
                Dev<float> dq(q), dk(k), dv(v), dg(gate), db(beta);
                std::vector<double> oref;
                ref_step(sref, q, k, v, gate, beta, oref, S, h_k, h_v);
                ok(gdn_step(sa, dq, dk, dv, dg, db, od, S, h_k, h_v) == cudaSuccess, "gdn_step launch");
                ok(native_gdn_step(sb, dq, dk, dv, dg, db, oe, S, h_k, h_v, g_s) == cudaSuccess, "native_gdn_step launch");
                // fused: y = gated RMS norm of the read-out; checked against the host norm of oref
                std::vector<float> z = vf((size_t)h_v * S, -2, 2), nw = vf(S, 0.5f, 1.5f);
                Dev<float> dz(z), dnw(nw), dy((size_t)h_v * S);
                ok(fused_gdn_step_norm(sc, dq, dk, dv, dg, db, dz, dnw, dy, h_k, h_v, 1e-6f) == cudaSuccess, "fused_gdn_step_norm launch");
                ok(sync_ok(), "gdn step kernels run");
                near(od.get(), oref, 2e-3, 1e-4, "gdn_step output equals the host delta rule");
                near(sa.get().size() ? sa.get() : std::vector<float>(), sref, 2e-3, 1e-4, "gdn_step state equals the host delta rule");
                ok(oe.get() == od.get() && sb.get() == sa.get(), "native_gdn_step is bit-identical to gdn_step");
                near(sc.get(), sref, 2e-3, 1e-4, "fused_gdn_step_norm state equals the host delta rule");
                std::vector<double> yref((size_t)h_v * S);
                for (int h = 0; h < h_v; ++h) {
                    double ss = 0; for (int j = 0; j < S; ++j) ss += oref[(size_t)h * S + j] * oref[(size_t)h * S + j];
                    const double inv = 1.0 / std::sqrt(ss / S + 1e-6);
                    for (int j = 0; j < S; ++j) yref[(size_t)h * S + j] = oref[(size_t)h * S + j] * inv * nw[j] * sigm(z[(size_t)h * S + j]);
                }
                near(dy.get(), yref, 3e-3, 1e-4, "fused_gdn_step_norm output = gated RMS norm of the read-out");
            }
        }
    }
}

static void test_gdn_validation() {
    const int S = 128, h_k = 2, h_v = 4;
    Dev<float> st((size_t)S * h_v * S), q((size_t)4 * S), k((size_t)4 * S), v((size_t)h_v * S), g(8), b(8), o((size_t)h_v * S);   // q,k,g,b padded so a wrongly accepted call stays in bounds
    ok(native_gdn_step(st, q, k, v, g, b, o, S, h_k, h_v, g_s) == cudaSuccess && sync_ok(), "native_gdn_step accepts a valid call");
    ok(native_gdn_step(st, q, k, v, g, b, o, S, h_k, h_v, nullptr) == cudaErrorInvalidValue, "native_gdn_step rejects a null stream (otherwise valid call)");
    ok(native_gdn_step(st, q, k, v, g, b, o, 64, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects S != 128");
    ok(native_gdn_step(st, q, k, v, g, b, o, S, 0, h_v, g_s) != cudaSuccess, "native_gdn_step rejects h_k = 0");
    ok(native_gdn_step(st, q, k, v, g, b, o, S, 3, 4, g_s) != cudaSuccess, "native_gdn_step rejects h_v % h_k != 0");
    ok(native_gdn_step(st, q, k, v, g, b, o, S, 1, 65536, g_s) != cudaSuccess, "native_gdn_step rejects h_v > 65535 (before touching memory)");
    ok(native_gdn_step(nullptr, q, k, v, g, b, o, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects a null span");
    ok(native_gdn_step((float*)((char*)(float*)st + 1), q, k, v, g, b, o, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects a misaligned state span");
    ok(native_gdn_step(st, (const float*)((const char*)(float*)q + 2), k, v, g, b, o, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects a misaligned q span");
    ok(native_gdn_step((float*)(UINTPTR_MAX - 3), q, k, v, g, b, o, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects a wrapping span");
    ok(native_gdn_step(st, st, k, v, g, b, o, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects state overlapping q");
    ok(native_gdn_step(st, q, k, v, g, b, g, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects output overlapping gate");
    ok(native_gdn_step(st, q, k, v, g, b, v, S, h_k, h_v, g_s) != cudaSuccess, "native_gdn_step rejects output overlapping v");
    // adjacent (non-overlapping) spans are fine: split one buffer
    Dev<float> big((size_t)S * h_v * S + (size_t)h_k * S * 2 + (size_t)h_v * S * 2 + 2 * h_v);
    float* bp = big;
    float* s2 = bp; float* q2 = s2 + (size_t)S * h_v * S; float* k2 = q2 + (size_t)h_k * S; float* v2 = k2 + (size_t)h_k * S;
    float* g2 = v2 + (size_t)h_v * S; float* b2 = g2 + h_v; float* o2 = b2 + h_v;
    ok(native_gdn_step(s2, q2, k2, v2, g2, b2, o2, S, h_k, h_v, g_s) == cudaSuccess && sync_ok(), "native_gdn_step accepts tightly packed disjoint spans");
}

static void test_gdn_conv() {
    const int dc = 4, C = 3 * 128 * 2;   // two heads each of q, k, v
    // (1) sliding: three steps leave exactly the three inputs in order (oldest .. newest)
    {
        Dev<float> cs((size_t)C * 3), out(C);
        Dev<float> w(vf((size_t)C * dc, -1, 1));
        std::vector<std::vector<float>> xs;
        for (int t = 0; t < 3; ++t) { xs.push_back(vf(C, -1, 1)); Dev<float> dx(xs.back()); ok(gdn_conv_step(cs, dx, w, out, C, dc) == cudaSuccess, "gdn_conv_step launch"); }
        ok(sync_ok(), "conv runs");
        const auto h = cs.get();
        bool good = true;
        for (int c = 0; c < C && good; ++c) for (int i = 0; i < 3; ++i) good = good && h[(size_t)c * 3 + i] == xs[i][c];
        ok(good, "conv history after three steps is exactly the last three inputs in order");
    }
    // (2) tap order: window [h0,h1,h2,x] with weights w0..w3 (history oldest first); fused conv+L2 and native conv+SiLU agree
    {
        std::vector<float> hist = vf((size_t)C * 3, -1, 1), w = vf((size_t)C * dc, -1, 1), x = vf(C, -1, 1);
        std::vector<double> ref(C), refs(C);
        for (int c = 0; c < C; ++c) {
            double a = 0; for (int i = 0; i < 3; ++i) a += (double)hist[(size_t)c * 3 + i] * w[(size_t)c * dc + i];
            a += (double)x[c] * w[(size_t)c * dc + 3]; ref[c] = a; refs[c] = silu(a);
        }
        Dev<float> c1(hist), c2(hist), c3(hist), dw(w), dx(x), o1(C), o2(C), o3(C);
        ok(gdn_conv_step(c1, dx, dw, o1, C, dc) == cudaSuccess, "conv plain launch");
        ok(native_gdn_conv_silu(c2, dx, dw, o2, C, dc, g_s) == cudaSuccess, "native conv+silu launch");
        const int h_k = 2, h_v = 2;     // channels = (2*h_k + h_v) * 128 = 768 = C
        ok(fused_gdn_conv_l2(c3, dx, dw, o3, h_k, h_v, dc, 1e-12f) == cudaSuccess, "fused conv+L2 launch");
        ok(sync_ok(), "conv variants run");
        near(o1.get(), ref, 1e-4, 1e-5, "conv window uses taps oldest..newest then the new input");
        near(o2.get(), refs, 1e-4, 1e-5, "native conv+silu equals silu(conv)");
        ok(c1.get() == c2.get() && c2.get() == c3.get(), "all conv variants slide the history identically");
        // fused: q and k heads (first 2*h_k) unit L2, v heads untouched
        const auto f = o3.get();
        std::vector<double> rf3(C);
        for (int hd = 0; hd < 2 * h_k + h_v; ++hd) {
            double ss = 0; for (int j = 0; j < 128; ++j) ss += ref[(size_t)hd * 128 + j] * ref[(size_t)hd * 128 + j];
            const double inv = hd < 2 * h_k ? 1.0 / std::sqrt(ss + 1e-12) : 1.0;
            for (int j = 0; j < 128; ++j) rf3[(size_t)hd * 128 + j] = ref[(size_t)hd * 128 + j] * inv;
        }
        near(f, rf3, 1e-4, 1e-5, "fused conv+L2 normalises exactly the q and k heads");
    }
    // validation of the native conv
    {
        Dev<float> cs((size_t)C * 3), x(C), w((size_t)C * dc), out(C);
        ok(native_gdn_conv_silu(cs, x, w, out, C, dc, g_s) == cudaSuccess && sync_ok(), "native conv accepts a valid call");
        ok(native_gdn_conv_silu(cs, x, w, x, C, dc, g_s) != cudaSuccess, "native conv rejects output overlapping the input");
        ok(native_gdn_conv_silu(cs, x, w, out, 0, dc, g_s) != cudaSuccess, "native conv rejects zero channels");
        ok(native_gdn_conv_silu(cs, x, w, out, 65536, dc, g_s) != cudaSuccess, "native conv rejects a count above 65535");
        ok(native_gdn_conv_silu(cs, x, w, out, C, 1, g_s) != cudaSuccess, "native conv rejects d_conv < 2");
        ok(native_gdn_conv_silu(nullptr, x, w, out, C, dc, g_s) != cudaSuccess, "native conv rejects a null history");
        ok(native_gdn_conv_silu(cs, x, w, out, C, dc, nullptr) == cudaErrorInvalidValue, "native conv rejects a null stream (otherwise valid call)");
    }
}

static void test_gdn_norms() {
    // L2 rows
    {
        const int rows = 7; auto x = vf((size_t)rows * 128, -3, 3);
        std::vector<double> ref(x.size());
        for (int r = 0; r < rows; ++r) {
            double ss = 0; for (int i = 0; i < 128; ++i) ss += (double)x[r * 128 + i] * x[r * 128 + i];
            for (int i = 0; i < 128; ++i) ref[r * 128 + i] = x[r * 128 + i] / std::sqrt(ss + 1e-6);
        }
        Dev<float> a(x), b(x);
        ok(gdn_l2_norm(a, rows, 128, 1e-6f) == cudaSuccess && native_gdn_l2_norm(b, rows, 128, 1e-6f, g_s) == cudaSuccess && sync_ok(), "l2 launches");
        near(a.get(), ref, 1e-4, 1e-6, "gdn_l2_norm equals host");
        near(b.get(), ref, 1e-4, 1e-6, "native_gdn_l2_norm equals host");
        ok(native_gdn_l2_norm(b, rows, 127, 1e-6f, g_s) != cudaSuccess, "native l2 rejects width != 128");
        ok(native_gdn_l2_norm(b, rows, 256, 1e-6f, g_s) != cudaSuccess, "native l2 rejects width 256");
        ok(native_gdn_l2_norm(b, 0, 128, 1e-6f, g_s) != cudaSuccess, "native l2 rejects zero rows");
        ok(native_gdn_l2_norm(b, 65536, 128, 1e-6f, g_s) != cudaSuccess, "native l2 rejects more than 65535 rows");
        ok(native_gdn_l2_norm(b, rows, 128, std::numeric_limits<float>::quiet_NaN(), g_s) != cudaSuccess, "native l2 rejects NaN epsilon");
        ok(native_gdn_l2_norm(b, rows, 128, std::numeric_limits<float>::infinity(), g_s) != cudaSuccess, "native l2 rejects infinite epsilon");
        ok(native_gdn_l2_norm(b, rows, 128, -1e-6f, g_s) != cudaSuccess, "native l2 rejects negative epsilon");
        ok(native_gdn_l2_norm(b, rows, 128, 0.f, g_s) == cudaSuccess && sync_ok(), "native l2 accepts zero epsilon");
        ok(native_gdn_l2_norm((float*)((char*)(float*)b + 1), rows, 128, 1e-6f, g_s) != cudaSuccess, "native l2 rejects a misaligned span");
        ok(native_gdn_l2_norm(nullptr, rows, 128, 1e-6f, g_s) != cudaSuccess, "native l2 rejects null");
        ok(native_gdn_l2_norm(b, rows, 128, 1e-6f, nullptr) == cudaErrorInvalidValue, "native l2 rejects a null stream (otherwise valid call)");
    }
    // output norm
    {
        const int h_v = 5; auto o = vf((size_t)h_v * 128, -2, 2), z = vf((size_t)h_v * 128, -3, 3), nw = vf(128, 0.5f, 1.5f);
        std::vector<double> ref(o.size());
        for (int h = 0; h < h_v; ++h) {
            double ss = 0; for (int i = 0; i < 128; ++i) ss += (double)o[h * 128 + i] * o[h * 128 + i];
            const double inv = 1.0 / std::sqrt(ss / 128 + 1e-6);
            for (int i = 0; i < 128; ++i) ref[h * 128 + i] = o[h * 128 + i] * inv * nw[i] * sigm(z[h * 128 + i]);
        }
        Dev<float> d_o(o), d_z(z), d_w(nw), y1(o.size()), y2(o.size());
        ok(gdn_out_norm(d_o, d_z, d_w, y1, h_v, 128, 1e-6f) == cudaSuccess && native_gdn_out_norm(d_o, d_z, d_w, y2, h_v, 1e-6f, g_s) == cudaSuccess && sync_ok(), "out_norm launches");
        near(y1.get(), ref, 1e-4, 1e-6, "gdn_out_norm equals host");
        near(y2.get(), ref, 1e-4, 1e-6, "native_gdn_out_norm equals host");
        ok(native_gdn_out_norm(d_o, d_z, d_w, d_o, h_v, 1e-6f, g_s) != cudaSuccess, "native out_norm rejects y aliasing o");
        ok(native_gdn_out_norm(d_o, d_z, d_w, y2, 0, 1e-6f, g_s) != cudaSuccess, "native out_norm rejects zero heads");
        ok(native_gdn_out_norm(d_o, d_z, d_w, y2, 65536, 1e-6f, g_s) != cudaSuccess, "native out_norm rejects more than 65535 heads");
        ok(native_gdn_out_norm(d_o, d_z, d_w, y2, h_v, std::numeric_limits<float>::quiet_NaN(), g_s) != cudaSuccess, "native out_norm rejects NaN epsilon");
        ok(native_gdn_out_norm(d_o, d_z, d_w, y2, h_v, -1.f, g_s) != cudaSuccess, "native out_norm rejects negative epsilon");
        ok(native_gdn_out_norm(d_o, d_z, d_w, y2, h_v, 1e-6f, nullptr) == cudaErrorInvalidValue, "native out_norm rejects a null stream (otherwise valid call)");
    }
    // beta sigmoid, gate softplus, ab
    {
        const int n = 48;
        auto bt = vf(n, -6, 6), al = vf(n, -3, 3), dt = vf(n, -1, 1), alog = vf(n, -2, 1);
        std::vector<double> rb(n), rg(n);
        for (int i = 0; i < n; ++i) { rb[i] = sigm(bt[i]); rg[i] = -std::exp((double)alog[i]) * softplus((double)al[i] + dt[i]); }
        Dev<float> b1(bt), b2(bt), d_a(al), d_dt(dt), d_al(alog), g1(n), g2(n), ab_beta(n), ab_g(n), d_b(bt);
        ok(gdn_beta_gate(b1, n) == cudaSuccess && native_gdn_beta_gate(b2, n, g_s) == cudaSuccess && sync_ok(), "beta gates launch");
        near(b1.get(), rb, 1e-5, 1e-7, "gdn_beta_gate = sigmoid");
        near(b2.get(), rb, 1e-5, 1e-7, "native_gdn_beta_gate = sigmoid");
        ok(native_gdn_gate(d_a, d_dt, d_al, g1, n, g_s) == cudaSuccess && sync_ok(), "native gate launches");
        near(g1.get(), rg, 1e-4, 1e-6, "native_gdn_gate = -exp(a_log) * softplus(alpha + dt_bias)");
        ok(fused_gdn_ab(d_a, d_b, d_al, d_dt, ab_beta, ab_g, n) == cudaSuccess && sync_ok(), "fused ab launches");
        near(ab_beta.get(), rb, 1e-5, 1e-7, "fused_gdn_ab beta = sigmoid(b)");
        near(ab_g.get(), rg, 1e-4, 1e-6, "fused_gdn_ab gate = -exp(a_log) * softplus(a + dt_bias)");
        {   // decode_ab_proj: projection of x by bf16 [w_alpha; w_beta] rows, then gate / sigmoid
            for (int cfg = 0; cfg < 3; ++cfg) {
                const int hv = cfg == 0 ? 32 : (cfg == 1 ? 7 : 33), ne = cfg == 0 ? 2560 : (cfg == 1 ? 8 : 520);   // 2*hv not a multiple of 8; n_embd/8 < 64 and not a multiple of 64
                auto wa = vbf((size_t)hv * ne, 0.5f), wb = vbf((size_t)hv * ne, 0.5f);
                auto px = vf(ne, -1, 1), pal = vf(hv, -1, 1), pdt = vf(hv, -1, 1);
                std::vector<double> rbeta(hv), rgate(hv);
                for (int h = 0; h < hv; ++h) {
                    double da = 0, db = 0;
                    for (int k = 0; k < ne; ++k) { da += (double)bf(wa[(size_t)h * ne + k]) * px[k]; db += (double)bf(wb[(size_t)h * ne + k]) * px[k]; }
                    rgate[h] = -std::exp((double)pal[h]) * softplus(da + pdt[h]);
                    rbeta[h] = sigm(db);
                }
                Dev<uint16_t> dwa(wa), dwb(wb); Dev<float> dx(px), dal(pal), ddt(pdt), pb(hv), pg(hv);
                ok(fused_gdn_ab(dwa, dwb, dx, dal, ddt, pb, pg, hv, ne) == cudaSuccess && sync_ok(), "fused_gdn_ab projection launches");
                near(pg.get(), rgate, 1e-4, 1e-5, "fused_gdn_ab projection: g[h] from row h of w_alpha (all 2*h_v rows covered, none twice)");
                near(pb.get(), rbeta, 1e-4, 1e-5, "fused_gdn_ab projection: beta[h] from row h_v + h of the beta weights");
            }
            // each weight row is read by exactly one wave: a one-hot row pattern isolates the row index and the chunk/lane mapping
            const int hv = 5, ne = 4096; std::vector<uint16_t> wa((size_t)hv * ne, 0), wb((size_t)hv * ne, 0); std::vector<float> px(ne);
            for (int k = 0; k < ne; ++k) px[k] = 0.001f * (float)(k % 97 + 1);
            for (int h = 0; h < hv; ++h) { wa[(size_t)h * ne + 8 * h + 3] = fb::round_f32_to_bf16_bits(1.f); wb[(size_t)h * ne + ne - 1 - h] = fb::round_f32_to_bf16_bits(1.f); }
            std::vector<float> zero(hv, 0.f), one(hv, 0.f);
            std::vector<double> rbeta(hv), rgate(hv);
            for (int h = 0; h < hv; ++h) { rgate[h] = -std::exp(0.0) * softplus((double)px[8 * h + 3]); rbeta[h] = sigm((double)px[ne - 1 - h]); }
            Dev<uint16_t> dwa(wa), dwb(wb); Dev<float> dx(px), dal(zero), ddt(zero), pb(hv), pg(hv);
            ok(fused_gdn_ab(dwa, dwb, dx, dal, ddt, pb, pg, hv, ne) == cudaSuccess && sync_ok(), "fused_gdn_ab one-hot launches");
            near(pg.get(), rgate, 1e-4, 1e-6, "fused_gdn_ab one-hot: alpha row h selects x[8h+3]");
            near(pb.get(), rbeta, 1e-4, 1e-6, "fused_gdn_ab one-hot: beta row selects x[n_embd-1-h] (last chunk, high half)");
            ok(fused_gdn_ab(dwa, dwb, dx, dal, ddt, pb, pg, 0, ne) != cudaSuccess, "fused_gdn_ab projection rejects h_v = 0");
            ok(fused_gdn_ab(dwa, dwb, dx, dal, ddt, pb, pg, hv, ne - 4) != cudaSuccess, "fused_gdn_ab projection rejects n_embd % 8 != 0");
            ok(fused_gdn_ab(dwa, dwb, dx, dal, ddt, pb, nullptr, hv, ne) != cudaSuccess, "fused_gdn_ab projection rejects null output");
            ok(fused_gdn_ab((const uint16_t*)((const char*)dwa.p + 2), dwb, dx, dal, ddt, pb, pg, hv, ne) != cudaSuccess, "fused_gdn_ab projection rejects misaligned weights");
        }
        ok(native_gdn_beta_gate(b2, 0, g_s) != cudaSuccess, "native beta gate rejects zero count");
        ok(native_gdn_beta_gate(b2, 65536, g_s) != cudaSuccess, "native beta gate rejects a count above 65535");
        ok(native_gdn_gate(d_a, d_dt, d_al, d_a, n, g_s) != cudaSuccess, "native gate rejects output overlapping an input");
        ok(native_gdn_gate(d_a, d_dt, d_al, g1, 0, g_s) != cudaSuccess, "native gate rejects zero count");
        ok(native_gdn_beta_gate(b2, n, nullptr) == cudaErrorInvalidValue, "native beta gate rejects a null stream (otherwise valid call)");
        ok(native_gdn_gate(d_a, d_dt, d_al, g1, n, nullptr) == cudaErrorInvalidValue, "native gate rejects a null stream (otherwise valid call)");
        // coverage of every element across a launch that is not a multiple of the block size
        const int n2 = 1000;
        auto big = vf(n2, -5, 5); std::vector<double> rbig(n2); for (int i = 0; i < n2; ++i) rbig[i] = sigm(big[i]);
        Dev<float> d_big(big); ok(gdn_beta_gate(d_big, n2) == cudaSuccess && sync_ok(), "beta gate launches (n=1000)");
        near(d_big.get(), rbig, 1e-5, 1e-7, "beta gate covers every element when n is not a multiple of the block");
    }
}

// ---- gated residual ----------------------------------------------------------------------------------------------------------
struct GrHost {
    int n, hc, lr; float eps;
    std::vector<double> xn, lo, gated, x, w;
};
static void ref_gr_read(const std::vector<float>& R, const std::vector<uint16_t>& down, const std::vector<uint16_t>& up, const std::vector<uint16_t>& inj, GrHost& g) {
    const int n = g.n, hc = g.hc, lr = g.lr, D = n * hc;
    g.xn.assign(D, 0); g.lo.assign(lr, 0); g.gated.assign(D, 0); g.x.assign(n, 0); g.w.assign(hc, 0);
    std::vector<double> xq(D);
    for (int c = 0; c < hc; ++c) {
        double ss = 0; for (int d = 0; d < n; ++d) ss += (double)R[c * n + d] * R[c * n + d];
        const double inv = 1.0 / std::sqrt(ss / n + g.eps);
        for (int d = 0; d < n; ++d) { g.xn[c * n + d] = R[c * n + d] * inv; xq[c * n + d] = rbf((float)g.xn[c * n + d]); }
    }
    std::vector<double> hq(lr);
    for (int k = 0; k < lr; ++k) {
        double a = 0; for (int i = 0; i < D; ++i) a += (double)bf(down[(size_t)k * D + i]) * xq[i];
        g.lo[k] = silu(a); hq[k] = rbf((float)g.lo[k]);
    }
    for (int i = 0; i < D; ++i) {
        double a = 0; for (int k = 0; k < lr; ++k) a += (double)bf(up[(size_t)i * lr + k]) * hq[k];
        g.gated[i] = g.xn[i] * sigm(a);
    }
    for (int d = 0; d < n; ++d) { double s = 0; for (int c = 0; c < hc; ++c) s += g.gated[c * n + d]; g.x[d] = s / hc; }
    for (int c = 0; c < hc; ++c) { double a = 0; for (int d = 0; d < n; ++d) a += (double)bf(inj[(size_t)c * n + d]) * g.x[d]; g.w[c] = a; }
}

static void test_gr_workspace() {
    const GrDims real{2560, 4, 320};
    ok(gr_workspace_bytes(real) == 104320, "gr workspace is 104320 bytes at the real geometry");
    std::vector<uint8_t> buf(gr_workspace_bytes(real) + 64);
    GrWs ws{};
    ok(gr_workspace_init(buf.data(), buf.size(), real, &ws), "gr_workspace_init accepts a sufficient buffer");
    const uint8_t* base = buf.data();
    struct Reg { const char* name; const void* p; size_t bytes; };
    const size_t D = 10240;
    const Reg regs[5] = {{"xn", ws.xn, D * 4}, {"xq", ws.xq, D * 2}, {"lq", ws.lq, 320 * 2}, {"gated", ws.gated, D * 4}, {"lo", ws.lo, 320 * 4}};
    bool aligned = true, inside = true, disjoint = true;
    for (int i = 0; i < 5; ++i) {
        const size_t off = (size_t)((const uint8_t*)regs[i].p - base);
        aligned = aligned && off % 16 == 0;
        inside = inside && off + regs[i].bytes <= ws.bytes;
        for (int j = i + 1; j < 5; ++j) {
            const size_t oj = (size_t)((const uint8_t*)regs[j].p - base);
            disjoint = disjoint && (off + regs[i].bytes <= oj || oj + regs[j].bytes <= off);
        }
    }
    ok(aligned, "gr workspace regions are 16-byte aligned");
    ok(inside, "gr workspace regions lie inside the workspace (xq holds hc_dim*2 bytes, not hc_lr*2)");
    ok(disjoint, "gr workspace regions are pairwise disjoint");
    ok(ws.bytes == 104320, "gr workspace reports the carved byte count");
    GrWs ws2{};
    ok(!gr_workspace_init(buf.data(), gr_workspace_bytes(real) - 1, real, &ws2), "gr_workspace_init rejects a buffer one byte short");
    ok(!gr_workspace_init(nullptr, buf.size(), real, &ws2), "gr_workspace_init rejects null");
    // odd geometry: rounding to 16 bytes grows each region by < 16 bytes
    const GrDims odd{37, 3, 5};
    const size_t D2 = 111, sum = (D2 * 4 + 15) / 16 * 16 + (D2 * 2 + 15) / 16 * 16 + (5 * 2 + 15) / 16 * 16 + (D2 * 4 + 15) / 16 * 16 + (5 * 4 + 15) / 16 * 16;
    ok(gr_workspace_bytes(odd) == sum, "gr workspace rounds every region to 16 bytes");
    ok(fused_gr_supported(2560, 4, 320) && !fused_gr_supported(2560, 4, 321) && !fused_gr_supported(2048, 4, 320) && !fused_gr_supported(2560, 2, 320),
       "fused_gr_supported accepts exactly the fixed geometry");
}

static void test_gr_read_write() {
    const int n = 2560, hc = 4, lr = 320, D = n * hc;
    const float eps = 1e-6f;
    auto down = vbf((size_t)lr * D, 0.05f), up = vbf((size_t)D * lr, 0.25f), inj = vbf((size_t)hc * n, 0.1f);
    Dev<uint16_t> d_down(down), d_up(up), d_inj(inj);
    // unfused vs host
    const int T = 5;
    std::vector<float> Rall = vf((size_t)T * D, -2, 2);
    std::vector<std::vector<double>> xs(T), ws_(T);
    for (int t = 0; t < T; ++t) {
        std::vector<float> R(Rall.begin() + (size_t)t * D, Rall.begin() + (size_t)(t + 1) * D);
        GrHost g{n, hc, lr, eps, {}, {}, {}, {}, {}};
        ref_gr_read(R, down, up, inj, g);
        xs[t] = g.x; ws_[t] = g.w;
        Dev<float> dR(R), x(n), w(hc);
        Dev<uint8_t> work(gr_workspace_bytes({n, hc, lr}));
        GrWs ws{}; const bool init_ok = gr_workspace_init(work, work.n, {n, hc, lr}, &ws);
        ok(init_ok, "gr workspace init (device)");
        gr_set_fp32_activations(false); gr_set_native_mmvf(false);
        ok(gr_read(dR, d_down, d_up, d_inj, ws, {n, hc, lr}, eps, x, w) == cudaSuccess && sync_ok(), "gr_read launch");
        near(x.get(), g.x, 5e-3, 0.5, "gr_read block input equals the host reference");
        near(w.get(), g.w, 5e-3, 0.5, "gr_read injection weights equal the host reference");
        // fused single token
        Dev<float> lo(lr), fx(n), fw(hc);
        ok(fused_gr_read(dR, d_down, d_up, d_inj, lo, fx, fw, eps) == cudaSuccess && sync_ok(), "fused_gr_read launch");
        near(fx.get(), g.x, 5e-3, 0.5, "fused_gr_read block input equals the host reference");
        near(fw.get(), g.w, 5e-3, 0.5, "fused_gr_read injection weights equal the host reference");
        near(lo.get(), g.lo, 5e-3, 0.5, "fused_gr_read hidden vector equals the host reference");
    }
    // multi-token: token t of the batch equals the single-token result; T=1 multi equals fused_gr_read exactly
    {
        Dev<float> dR(Rall), lo((size_t)T * lr), x((size_t)T * n), w((size_t)T * hc);
        GrMulti m{}; m.R = dR; m.down = d_down; m.up = d_up; m.inject = d_inj; m.lo = lo; m.x_out = x; m.w_out = w; m.eps = eps; m.T = T;
        ok(fused_gr_read_multi(m) == cudaSuccess && sync_ok(), "fused_gr_read_multi launch (T=5)");
        const auto gx = x.get(), gw = w.get();
        for (int t = 0; t < T; ++t) {
            std::vector<float> xt(gx.begin() + (size_t)t * n, gx.begin() + (size_t)(t + 1) * n), wt(gw.begin() + (size_t)t * hc, gw.begin() + (size_t)(t + 1) * hc);
            near(xt, xs[t], 5e-3, 0.5, "multi-token block input equals the host reference of its own token");
            {   // same token through the single-token fused kernel: isolates the multi-token (16-column) variant from bf16 rounding noise
                std::vector<float> Rt(Rall.begin() + (size_t)t * D, Rall.begin() + (size_t)(t + 1) * D);
                Dev<float> dRt(Rt), lo_t(lr), x_t(n), w_t(hc);
                ok(fused_gr_read(dRt, d_down, d_up, d_inj, lo_t, x_t, w_t, eps) == cudaSuccess && sync_ok(), "single-token fused launch (cross-check)");
                near_f(xt, x_t.get(), 1e-4, 1.0, "multi-token block input equals the single-token fused result of its own token");
            }
            near(wt, ws_[t], 5e-3, 0.5, "multi-token injection weights equal the host reference of its own token");
        }
        Dev<float> r1(std::vector<float>(Rall.begin(), Rall.begin() + D)), lo1(lr), x1(n), w1(hc), lo2(lr), x2(n), w2(hc);
        ok(fused_gr_read(r1, d_down, d_up, d_inj, lo1, x1, w1, eps) == cudaSuccess, "fused single launch");
        GrMulti m1{}; m1.R = r1; m1.down = d_down; m1.up = d_up; m1.inject = d_inj; m1.lo = lo2; m1.x_out = x2; m1.w_out = w2; m1.eps = eps; m1.T = 1;
        ok(fused_gr_read_multi(m1) == cudaSuccess && sync_ok(), "fused multi (T=1) launch");
        ok(x1.get() == x2.get() && w1.get() == w2.get() && lo1.get() == lo2.get(), "fused_gr_read is the T=1 case of fused_gr_read_multi");
        // limits
        GrMulti bad = m; bad.T = 0; ok(fused_gr_read_multi(bad) != cudaSuccess, "fused multi rejects T = 0");
        bad = m; bad.T = 33; ok(fused_gr_read_multi(bad) != cudaSuccess, "fused multi rejects T > 32");
        bad = m; bad.down = nullptr; ok(fused_gr_read_multi(bad) != cudaSuccess, "fused multi rejects a null weight");
        bad = m; bad.up = (const uint16_t*)((const char*)(const uint16_t*)d_up + 2); ok(fused_gr_read_multi(bad) != cudaSuccess, "fused multi rejects 16-byte-misaligned weights");
        Dev<float> lo32((size_t)32 * lr), x32((size_t)32 * n), w32((size_t)32 * hc), R32(vf((size_t)32 * D, -1, 1));
        GrMulti m32{}; m32.R = R32; m32.down = d_down; m32.up = d_up; m32.inject = d_inj; m32.lo = lo32; m32.x_out = x32; m32.w_out = w32; m32.eps = eps; m32.T = 32;
        ok(fused_gr_read_multi(m32) == cudaSuccess && sync_ok(), "fused multi accepts T = 32");
    }
    // write: out[i] = R[i] + block_out[i % n] * w[i / n]
    {
        auto R = vf((size_t)D, -2, 2), bo = vf(n, -2, 2), w = vf(hc, -2, 2);
        std::vector<double> ref(D); for (int i = 0; i < D; ++i) ref[i] = (double)R[i] + (double)bo[i % n] * w[i / n];
        Dev<float> dR(R), dbo(bo), dw(w), out(D);
        ok(gr_write(dR, dbo, dw, out, n, hc) == cudaSuccess && sync_ok(), "gr_write launch");
        near(out.get(), ref, 1e-6, 1.0, "gr_write: out[i] = R[i] + block_out[i % n] * w[i / n]");
        ok(gr_write(dR, dbo, dw, out, 0, hc) != cudaSuccess && gr_write(nullptr, dbo, dw, out, n, hc) != cudaSuccess, "gr_write rejects bad arguments");
        // native post with bias/scale
        auto bias = vf(hc, -1, 1); Dev<float> db(bias), out2(D);
        std::vector<double> ref2(D); for (int i = 0; i < D; ++i) ref2[i] = (double)R[i] + (double)bo[i % n] * ((double)w[i / n] * 0.5 + bias[i / n]);
        ok(native_gr_post(dR, dbo, dw, db, 0.5f, out2, n, hc) == cudaSuccess && sync_ok(), "native_gr_post launch");
        near(out2.get(), ref2, 1e-6, 1.0, "native_gr_post = R + block_out * (w*scale + bias)");
        Dev<float> out3(D);
        std::vector<double> ref3(D); for (int i = 0; i < D; ++i) ref3[i] = (double)R[i] + (double)bo[i % n] * ((double)w[i / n] * 1.0);
        ok(native_gr_post(dR, dbo, dw, nullptr, 1.0f, out3, n, hc) == cudaSuccess && sync_ok(), "native_gr_post (no bias) launch");
        near(out3.get(), ref3, 1e-6, 1.0, "native_gr_post with zero bias equals gr_write");
        near_f(out3.get(), out.get(), 1e-6, 1.0, "native_gr_post (scale 1, no bias) equals gr_write");
    }
    // elementwise helpers
    {
        const int m = 10240 + 77;
        auto xn = vf(m, -2, 2), lg = vf(m, -4, 4);
        std::vector<double> rg(m), rs(m);
        for (int i = 0; i < m; ++i) { rg[i] = (double)xn[i] * sigm(lg[i]); rs[i] = silu(lg[i]); }
        Dev<float> dx(xn), dl(lg), go(m), h(lg);
        ok(native_gr_pre_gated(dx, dl, go, m) == cudaSuccess && native_gr_down_silu(h, m) == cudaSuccess && sync_ok(), "gr elementwise launches");
        near(go.get(), rg, 1e-5, 1e-6, "native_gr_pre_gated = xn * sigmoid(logit)");
        near(h.get(), rs, 1e-5, 1e-6, "native_gr_down_silu = silu in place");
        ok(native_gr_pre_gated(dx, dl, go, 0) != cudaSuccess && native_gr_down_silu(nullptr, 5) != cudaSuccess, "gr elementwise reject bad arguments");
        // weighted rms norm, in place and out of place
        const int rows = 3, cols = 2560; auto xr = vf((size_t)rows * cols, -3, 3), wr = vf(cols, 0.5f, 1.5f);
        std::vector<double> rr((size_t)rows * cols);
        for (int r = 0; r < rows; ++r) {
            double ss = 0; for (int i = 0; i < cols; ++i) ss += (double)xr[r * cols + i] * xr[r * cols + i];
            const double inv = 1.0 / std::sqrt(ss / cols + 1e-6);
            for (int i = 0; i < cols; ++i) rr[r * cols + i] = xr[r * cols + i] * inv * wr[i];
        }
        Dev<float> dxr(xr), dwr(wr), yo((size_t)rows * cols), yi(xr);
        ok(native_gr_rms_norm_weighted(dxr, dwr, yo, rows, cols, 1e-6f) == cudaSuccess && native_gr_rms_norm_weighted(yi, dwr, yi, rows, cols, 1e-6f) == cudaSuccess && sync_ok(), "gr rms norm launches");
        near(yo.get(), rr, 1e-4, 1e-6, "native_gr_rms_norm_weighted equals host");
        ok(yi.get() == yo.get(), "in-place weighted rms norm equals the out-of-place result");
        ok(native_gr_rms_norm_weighted(dxr, dwr, (float*)((char*)(float*)dxr + 16), rows, cols, 1e-6f) != cudaSuccess, "weighted rms norm rejects partial overlap");
    }
}

// ---- MoE combine ----------------------------------------------------------------------------------------------------------------
static void test_moe_combine() {
    const int n = 2560, k = 10;
    auto parts = vf((size_t)n * k, -1, 1), w = vf(k, 0, 1), sh = vf(n, -1, 1);
    std::vector<double> ref(n), refn(n);
    for (int j = 0; j < n; ++j) { double a = 0; for (int e = 0; e < k; ++e) a += (double)w[e] * parts[(size_t)e * n + j]; refn[j] = a; ref[j] = a + sh[j]; }
    Dev<float> dp(parts), dw(w), dsh(sh), y1(n), y2(n);
    ok(native_moe_combine(dp, dw, dsh, y1, n, k, 1) == cudaSuccess && native_moe_combine(dp, dw, nullptr, y2, n, k, 0) == cudaSuccess && sync_ok(), "native_moe_combine launches");
    near(y1.get(), ref, 1e-4, 1e-5, "native_moe_combine with shared expert");
    near(y2.get(), refn, 1e-4, 1e-5, "native_moe_combine without shared expert");
    ok(native_moe_combine(dp, dw, dsh, dp, n, k, 1) != cudaSuccess, "native_moe_combine rejects y overlapping parts");
    ok(native_moe_combine(dp, dw, dsh, dsh, n, k, 1) != cudaSuccess, "native_moe_combine rejects y overlapping shared");
    ok(native_moe_combine(dp, dw, dsh, y1, 0, k, 1) != cudaSuccess && native_moe_combine(dp, dw, dsh, y1, n, 0, 1) != cudaSuccess, "native_moe_combine rejects empty shapes");
    ok(native_moe_combine((const float*)((const char*)(float*)dp + 1), dw, dsh, y1, n, k, 1) != cudaSuccess, "native_moe_combine rejects misaligned parts");
    ok(native_moe_combine(dp, dw, dsh, (float*)(UINTPTR_MAX - 3), n, k, 1) != cudaSuccess, "native_moe_combine rejects a wrapping span");
    native_moe_combine_set_enabled(false); ok(!native_moe_combine_enabled(), "moe combine switch off");
    native_moe_combine_set_enabled(true); ok(native_moe_combine_enabled(), "moe combine switch on");
    native_gdn_set_enabled(false); ok(!native_gdn_enabled(), "native gdn switch off");
    native_gdn_set_enabled(true); ok(native_gdn_enabled(), "native gdn switch on");
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("NO GPU DEVICE\n"); return 2; }
    cudaSetDevice(0);
    if (cudaStreamCreate(&g_s) != cudaSuccess) { std::printf("stream create failed\n"); return 2; }
    test_gdn_step();
    test_gdn_validation();
    test_gdn_conv();
    test_gdn_norms();
    test_gr_workspace();
    test_gr_read_write();
    test_moe_combine();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("U4DE GPU TESTS FAILED\n"); return 1; }
    std::printf("U4DE GPU TESTS PASSED\n");
    return 0;
}

