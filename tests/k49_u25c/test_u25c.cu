// test_u25c.cu - device checks of src/kernels/gpu/verify_window_kernels.cu against CPU references: every launcher, edge cases,
// rejected arguments and the LAYER 1 statements of VerifyWindowKernels (replay/keep semantics, earlier-token-only history taps,
// device-side keep count, grid-stride coverage). Requires a device (opens /dev/iluvatar* through the runtime).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "baloon/verify_window_kernels.hpp"
#include "invariants.hpp"

using namespace baloon;
namespace vw = baloon::vwin;

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { if (cudaMalloc((void**)&p, std::max<size_t>(1, count) * sizeof(T)) != cudaSuccess) p = nullptr; cudaMemset(p, 0, std::max<size_t>(1, count) * sizeof(T)); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { if (p) cudaFree(p); }
    std::vector<T> get() const { std::vector<T> h(n); cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }
    void put(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
};
static int dev_sync() { return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1; }
using u25c::Rng;
static std::vector<float> randv(Rng& r, size_t n, float sc = 1.f) { std::vector<float> v(n); for (auto& x : v) x = (float)r.gauss() * sc; return v; }
static bool close(float a, float b, float rt = 2e-5f, float at = 2e-6f) { return std::fabs(a - b) <= at + rt * std::fabs(b); }
static bool close_all(const std::vector<float>& a, const std::vector<float>& b, float rt = 2e-5f, float at = 2e-6f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!close(a[i], b[i], rt, at)) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------- conv / commit
static void test_conv() {
    Rng rng(21);
    const int hk = 2, hv = 4, C = vw::S * (2 * hk + hv);
    for (int T = 1; T <= 8; ++T) for (float eps : {1e-6f, 100.f}) {   // eps = 100 makes the max(norm, eps) clamp the active branch
        auto x = randv(rng, (size_t)T * C), hist = randv(rng, 3 * (size_t)C), w = randv(rng, (size_t)C * 4);
        Dev<float> dx(x), dh(hist), dw(w), dout((size_t)T * C);
        CHECK(vw::gdn_conv_l2_multi(dx.p, dh.p, dw.p, dout.p, T, C, hk, hv, eps, nullptr) == 0 && dev_sync() == 0);
        std::vector<float> ref((size_t)T * C);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                float acc = 0.f;
                for (int i = 0; i < 3; ++i) { const int e = t + i; acc += (e < 3 ? hist[(size_t)e * C + c] : x[(size_t)(e - 3) * C + c]) * w[(size_t)c * 4 + i]; }
                acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
                ref[(size_t)t * C + c] = acc;
            }
            for (int head = 0; head < 2 * hk; ++head) {
                double ss = 0; for (int j = 0; j < vw::S; ++j) { const float v = ref[(size_t)t * C + head * vw::S + j]; ss += (double)v * v; }
                const float nrm = std::max((float)std::sqrt(ss), eps);
                for (int j = 0; j < vw::S; ++j) ref[(size_t)t * C + head * vw::S + j] /= nrm;
            }
        }
        CHECK(close_all(dout.get(), ref, 1e-4f, 1e-5f));
        CHECK(dh.get() == hist);                                              // hist is read only
        // token t depends only on x_0..x_t: changing a later token leaves earlier outputs bit-identical
        if (T >= 2) {
            auto x2 = x; for (int c = 0; c < C; ++c) x2[(size_t)(T - 1) * C + c] += 1.f;
            Dev<float> dx2(x2), dout2((size_t)T * C);
            CHECK(vw::gdn_conv_l2_multi(dx2.p, dh.p, dw.p, dout2.p, T, C, hk, hv, eps, nullptr) == 0 && dev_sync() == 0);
            const auto a = dout.get(), b = dout2.get();
            CHECK(std::memcmp(a.data(), b.data(), (size_t)(T - 1) * C * sizeof(float)) == 0);
        }
    }
    // rejected arguments
    Dev<float> a(16), b(16), c(16), d(16);
    CHECK(vw::gdn_conv_l2_multi(a.p, b.p, c.p, d.p, 0, C, hk, hv, 1e-6f, nullptr) == -1);
    CHECK(vw::gdn_conv_l2_multi(a.p, b.p, c.p, d.p, 9, C, hk, hv, 1e-6f, nullptr) == -1);
    CHECK(vw::gdn_conv_l2_multi(a.p, b.p, c.p, d.p, 1, C + 1, hk, hv, 1e-6f, nullptr) == -1);   // C must be 128*(2hk+hv)
    CHECK(vw::gdn_conv_l2_multi(a.p, b.p, c.p, d.p, 1, C, hk, 3, 1e-6f, nullptr) == -1);        // hk | hv
    CHECK(vw::gdn_conv_l2_multi(a.p, b.p, c.p, d.p, 1, C, hk, hv, 0.f, nullptr) == -1);
    CHECK(vw::gdn_conv_l2_multi(nullptr, b.p, c.p, d.p, 1, C, hk, hv, 1e-6f, nullptr) == -1);
}

static void test_commit() {
    Rng rng(22);
    const int C = 300;
    for (int T = 1; T <= 8; ++T) for (int n : {-3, 0, 1, 2, 3, 4, T, T + 5}) {
        auto x = randv(rng, (size_t)T * C), hist = randv(rng, 3 * (size_t)C);
        Dev<float> dx(x), dh(hist);
        Dev<int32_t> dn(std::vector<int32_t>{n});
        CHECK(vw::gdn_conv_commit(dx.p, dh.p, dn.p, T, C, nullptr) == 0 && dev_sync() == 0);
        const int m = vw::keep_clamp(n, T);
        std::vector<float> ref = hist;
        if (m > 0) for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) {
            const int i = m + j;                             // entry of [hist | x_0 .. x_{m-1}]
            ref[(size_t)j * C + c] = i < 3 ? hist[(size_t)i * C + c] : x[(size_t)(i - 3) * C + c];
        }
        CHECK(dh.get() == ref);
        if (m > 0 && m >= 3) { const auto got = dh.get(); for (int j = 0; j < 3; ++j) CHECK(got[(size_t)j * C + 5] == x[(size_t)(m - 3 + j) * C + 5]); }   // last three KEPT inputs only
    }
    Dev<float> a(8), b(8); Dev<int32_t> n(1);
    CHECK(vw::gdn_conv_commit(a.p, b.p, nullptr, 2, 8, nullptr) == -1 && vw::gdn_conv_commit(a.p, b.p, n.p, 0, 8, nullptr) == -1 && vw::gdn_conv_commit(a.p, b.p, n.p, 9, 8, nullptr) == -1);
}

static void test_ab() {
    Rng rng(23);
    for (int hv : {1, 5, 32}) for (int T : {1, 3, 8}) {
        const int D = 200;
        auto xin = randv(rng, (size_t)T * D), wab = randv(rng, (size_t)2 * hv * D, 0.2f), dt = randv(rng, hv), a = randv(rng, hv);
        Dev<float> dxin(xin), dwab(wab), ddt(dt), da(a), dg((size_t)T * hv), db((size_t)T * hv);
        CHECK(vw::gdn_ab_multi(dxin.p, dwab.p, ddt.p, da.p, dg.p, db.p, T, D, hv, nullptr) == 0 && dev_sync() == 0);
        std::vector<float> g((size_t)T * hv), b((size_t)T * hv);
        for (int t = 0; t < T; ++t) for (int r = 0; r < 2 * hv; ++r) {
            double acc = 0; for (int d = 0; d < D; ++d) acc += (double)xin[(size_t)t * D + d] * wab[(size_t)r * D + d];
            if (r < hv) { const double z = acc + dt[r]; g[(size_t)t * hv + r] = (float)((z > 20 ? z : std::log1p(std::exp(z))) * a[r]); }
            else b[(size_t)t * hv + r - hv] = (float)(1.0 / (1.0 + std::exp(-acc)));
        }
        CHECK(close_all(dg.get(), g, 1e-4f, 1e-4f) && close_all(db.get(), b, 1e-4f, 1e-5f));
    }
    Dev<float> z(8);
    CHECK(vw::gdn_ab_multi(z.p, z.p, z.p, z.p, z.p, z.p, 0, 8, 1, nullptr) == -1 && vw::gdn_ab_multi(z.p, z.p, z.p, z.p, z.p, z.p, 1, 8, 0, nullptr) == -1);
}

// ---------------------------------------------------------------------------------------------- state step + norm
struct StepIn { std::vector<float> state, q, k, v, gate, beta, nw; };
static StepIn make_step(Rng& rng, int T, int hk, int hv) {
    StepIn s;
    s.state = randv(rng, (size_t)vw::S * hv * vw::S, 0.3f);
    s.q = randv(rng, (size_t)T * hk * vw::S, 0.3f); s.k = randv(rng, (size_t)T * hk * vw::S, 0.3f); s.v = randv(rng, (size_t)T * hv * vw::S);
    s.gate = randv(rng, (size_t)T * hv, 0.3f); for (auto& g : s.gate) g = -std::fabs(g);
    s.beta = randv(rng, (size_t)T * hv); for (auto& b : s.beta) b = 1.f / (1.f + std::exp(-b));
    s.nw = randv(rng, vw::S, 0.5f); for (auto& w : s.nw) w += 1.f;
    return s;
}
// CPU model of the T-token step (double accumulation); returns outputs [T][hv][S] with `sentinel` for t < t_out_begin and the
// state after `keep` tokens (keep <= 0: unchanged).
static void ref_step(const StepIn& in, int T, int hk, int hv, float scale, float eps, int keep, int t_out_begin, float sentinel,
                     std::vector<float>& out, std::vector<float>& state_after) {
    const int S = vw::S;
    std::vector<double> st(in.state.begin(), in.state.end());
    state_after = in.state;
    out.assign((size_t)T * hv * S, sentinel);
    for (int h = 0; h < hv; ++h) {
        const int qh = h % hk;
        for (int t = 0; t < T; ++t) {
            const double decay = std::exp((double)in.gate[(size_t)t * hv + h]);
            std::vector<double> kv(S, 0.0), o(S, 0.0), d(S);
            for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) {
                double& s = st[((size_t)i * hv + h) * S + j];
                s *= decay; kv[j] += s * in.k[((size_t)t * hk + qh) * S + i];
            }
            for (int j = 0; j < S; ++j) d[j] = ((double)in.v[((size_t)t * hv + h) * S + j] - kv[j]) * in.beta[(size_t)t * hv + h];
            for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) {
                double& s = st[((size_t)i * hv + h) * S + j];
                s += (double)in.k[((size_t)t * hk + qh) * S + i] * d[j]; o[j] += s * in.q[((size_t)t * hk + qh) * S + i];
            }
            if (keep > 0 && t + 1 == keep) for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) state_after[((size_t)i * hv + h) * S + j] = (float)st[((size_t)i * hv + h) * S + j];
            if (t >= t_out_begin) {
                double ss = 0; for (int j = 0; j < S; ++j) { o[j] *= scale; ss += o[j] * o[j]; }
                const double r = 1.0 / std::sqrt(ss / S + eps);
                for (int j = 0; j < S; ++j) out[((size_t)t * hv + h) * S + j] = (float)(o[j] * r * in.nw[j]);
            }
        }
    }
}
static int run_step(const StepIn& in, int T, int hk, int hv, float scale, float eps, int keep_dev, bool use_keep, int t_out_begin,
                    float sentinel, std::vector<float>& out, std::vector<float>& state) {
    Dev<float> dst(in.state), dq(in.q), dk(in.k), dv(in.v), dg(in.gate), db(in.beta), dnw(in.nw);
    Dev<float> dout((size_t)T * hv * vw::S);
    dout.put(std::vector<float>((size_t)T * hv * vw::S, sentinel));
    Dev<int32_t> dn(std::vector<int32_t>{keep_dev});
    const int rc = vw::gdn_step_norm_multi(dst.p, dq.p, dk.p, dv.p, dg.p, db.p, dnw.p, dout.p, T, hk, hv, scale, eps, use_keep ? dn.p : nullptr, t_out_begin, nullptr);
    if (rc != 0) return rc;
    if (dev_sync()) return 1000;
    out = dout.get(); state = dst.get();
    return 0;
}
static void test_step() {
    Rng rng(24);
    const float scale = 1.f / std::sqrt(128.f), eps = 1e-6f;
    for (int hk : {1, 2}) for (int hv : {2, 4}) for (int T : {1, 2, 4, 8}) {
        const StepIn in = make_step(rng, T, hk, hv);
        for (int keep : {0, 1, 2, T, T + 3}) for (int tob : {0, T / 2}) {
            std::vector<float> out, st, rout, rst;
            CHECK(run_step(in, T, hk, hv, scale, eps, keep, true, tob, -777.f, out, st) == 0);
            const int m = vw::keep_clamp(keep, T);
            ref_step(in, T, hk, hv, scale, eps, m, tob, -777.f, rout, rst);
            CHECK(close_all(out, rout, 2e-3f, 2e-4f));
            CHECK(close_all(st, rst, 2e-3f, 2e-4f));
            if (m == 0) CHECK(st == in.state);                                     // no keep count: the state is never modified
            for (int t = 0; t < tob; ++t) for (int i = 0; i < hv * vw::S; ++i) CHECK(out[(size_t)t * hv * vw::S + i] == -777.f);   // replayed tokens write no output
        }
        { std::vector<float> out, st; CHECK(run_step(in, T, hk, hv, scale, eps, 5, false, 0, 0.f, out, st) == 0 && st == in.state); }   // null keep pointer
    }
    // all T tokens produce the same outputs whatever the keep count (the keep count only affects the state)
    { const StepIn in = make_step(rng, 6, 2, 4); std::vector<float> o1, o2, s1, s2;
      CHECK(run_step(in, 6, 2, 4, scale, eps, 0, true, 0, 0.f, o1, s1) == 0 && run_step(in, 6, 2, 4, scale, eps, 3, true, 0, 0.f, o2, s2) == 0);
      CHECK(o1 == o2); }
    // multi-token step equals T single-token steps (same operation order): state kept after each token
    { const int T = 5, hk = 2, hv = 4; StepIn in = make_step(rng, T, hk, hv);
      std::vector<float> out, st;
      CHECK(run_step(in, T, hk, hv, scale, eps, T, true, 0, 0.f, out, st) == 0);
      StepIn cur = in; std::vector<float> outs((size_t)T * hv * vw::S);
      for (int t = 0; t < T; ++t) {
          StepIn one = cur;
          one.q.assign(in.q.begin() + (size_t)t * hk * vw::S, in.q.begin() + (size_t)(t + 1) * hk * vw::S);
          one.k.assign(in.k.begin() + (size_t)t * hk * vw::S, in.k.begin() + (size_t)(t + 1) * hk * vw::S);
          one.v.assign(in.v.begin() + (size_t)t * hv * vw::S, in.v.begin() + (size_t)(t + 1) * hv * vw::S);
          one.gate.assign(in.gate.begin() + (size_t)t * hv, in.gate.begin() + (size_t)(t + 1) * hv);
          one.beta.assign(in.beta.begin() + (size_t)t * hv, in.beta.begin() + (size_t)(t + 1) * hv);
          std::vector<float> o1, s1;
          CHECK(run_step(one, 1, hk, hv, scale, eps, 1, true, 0, 0.f, o1, s1) == 0);
          std::copy(o1.begin(), o1.end(), outs.begin() + (size_t)t * hv * vw::S);
          cur.state = s1;
      }
      CHECK(close_all(out, outs, 1e-5f, 1e-6f) && close_all(st, cur.state, 1e-5f, 1e-6f)); }
    // rejected arguments
    { const StepIn in = make_step(rng, 1, 1, 1); Dev<float> a(8);
      std::vector<float> o, s;
      CHECK(run_step(in, 0, 1, 2, scale, eps, 0, true, 0, 0.f, o, s) == -1);
      CHECK(run_step(in, 9, 1, 2, scale, eps, 0, true, 0, 0.f, o, s) == -1);
      CHECK(run_step(in, 1, 2, 3, scale, eps, 0, true, 0, 0.f, o, s) == -1);       // hk must divide hv
      CHECK(run_step(in, 1, 1, 2, scale, eps, 0, true, -1, 0.f, o, s) == -1);      // t_out_begin >= 0
      CHECK(run_step(in, 1, 1, 2, scale, -1.f, 0, true, 0, 0.f, o, s) == -1); }
}

// ---------------------------------------------------------------------------------------------- gathers
static void test_gathers() {
    Rng rng(25);
    // embedding gather with group scales, every code width
    for (int bits : {1, 2, 4, 8}) for (int group : {1, 8, 64}) {
        const int rows = 7, dim = 64, n = 5;
        const int rb = (dim * bits + 7) / 8, groups = dim / group;
        std::vector<uint8_t> table((size_t)rows * rb); for (auto& b : table) b = (uint8_t)rng.below(256);
        std::vector<float> scales((size_t)rows * groups); for (auto& s : scales) s = (float)rng.uni() + 0.1f;
        std::vector<int32_t> ids = {0, 6, 3, -1, 7};
        Dev<uint8_t> dt(table); Dev<float> ds(scales); Dev<int32_t> di(ids); Dev<float> dout((size_t)n * dim);
        CHECK(vw::embedding_gather_dev(dt.p, ds.p, di.p, dout.p, n, rows, dim, bits, group, nullptr) == 0 && dev_sync() == 0);
        const auto got = dout.get();
        for (int r = 0; r < n; ++r) for (int k = 0; k < dim; ++k) {
            float want = 0.f;
            if (ids[r] >= 0 && ids[r] < rows) want = (float)((int)vw::unpack_code(&table[(size_t)ids[r] * rb], k, bits) - (1 << (bits - 1))) * scales[(size_t)ids[r] * groups + k / group];
            CHECK(got[(size_t)r * dim + k] == want);
        }
    }
    { Dev<uint8_t> t(8); Dev<float> s(8); Dev<int32_t> i(8); Dev<float> o(8);
      CHECK(vw::embedding_gather_dev(t.p, s.p, i.p, o.p, 1, 1, 64, 3, 8, nullptr) == -1);        // unsupported width
      CHECK(vw::embedding_gather_dev(t.p, s.p, i.p, o.p, 1, 1, 64, 4, 7, nullptr) == -1); }       // group must divide dim
    // blob fetch: more than one grid of work (grid-stride), invalid ids give zeros
    for (int per : {1, 33, 4096}) {
        const int nblobs = 6, n = 9;
        std::vector<uint8_t> src((size_t)nblobs * per); for (auto& b : src) b = (uint8_t)(1 + rng.below(255));
        std::vector<int32_t> ids = {5, 0, 2, -1, 6, 3, 3, 100, 1};
        Dev<uint8_t> ds(src); Dev<int32_t> di(ids); Dev<uint8_t> dd((size_t)n * per);
        CHECK(vw::fetch_blobs(ds.p, di.p, dd.p, n, nblobs, per, nullptr) == 0 && dev_sync() == 0);
        const auto got = dd.get();
        for (int k = 0; k < n; ++k) for (int o = 0; o < per; ++o) CHECK(got[(size_t)k * per + o] == ((ids[k] >= 0 && ids[k] < nblobs) ? src[(size_t)ids[k] * per + o] : 0));
    }
    { const int rows = 5, row16 = 300, n = 4; std::vector<uint16_t> src((size_t)rows * row16); for (auto& b : src) b = (uint16_t)(1 + rng.below(65535));
      std::vector<int32_t> ids = {4, -2, 0, 5};
      Dev<uint16_t> ds(src); Dev<int32_t> di(ids); Dev<uint16_t> dd((size_t)n * row16);
      CHECK(vw::gather_rows(ds.p, di.p, dd.p, n, rows, row16, nullptr) == 0 && dev_sync() == 0);
      const auto got = dd.get();
      for (int k = 0; k < n; ++k) for (int o = 0; o < row16; ++o) CHECK(got[(size_t)k * row16 + o] == ((ids[k] >= 0 && ids[k] < rows) ? src[(size_t)ids[k] * row16 + o] : 0)); }
    // copy_indexed: stride beyond 64 blocks * 256 threads (grid-stride loop), device-side index, out of range -> zeros
    for (int stride : {1, 255, 256, 257, 16384, 16385, 70000}) for (int idx : {0, 2, 3, -1}) {
        const int nrows = 3;
        std::vector<float> src = randv(rng, (size_t)nrows * stride);
        Dev<float> ds(src); Dev<int32_t> di(std::vector<int32_t>{idx}); Dev<float> dd(stride);
        CHECK(vw::copy_indexed(ds.p, di.p, dd.p, stride, nrows, nullptr) == 0 && dev_sync() == 0);
        const auto got = dd.get();
        bool ok = true;
        for (int i = 0; i < stride; ++i) ok &= got[i] == ((idx >= 0 && idx < nrows) ? src[(size_t)idx * stride + i] : 0.f);
        CHECK(ok);
    }
}

static void test_streams() {
    Rng rng(26);
    for (int T : {1, 3, 8}) for (int hc : {1, 4}) {
        const int n = 37;
        auto x = randv(rng, (size_t)T * n), R0 = randv(rng, (size_t)T * n * hc);
        Dev<float> dx(x), dR(R0);
        CHECK(vw::broadcast_streams(dx.p, dR.p, T, n, hc, nullptr) == 0 && dev_sync() == 0);
        auto got = dR.get(); bool ok = true;
        for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == x[(i / ((size_t)n * hc)) * n + i % n];
        CHECK(ok);
        dR.put(R0);
        CHECK(vw::add_streams_broadcast(dR.p, dx.p, T, n, hc, nullptr) == 0 && dev_sync() == 0);
        got = dR.get(); ok = true;
        for (size_t i = 0; i < got.size(); ++i) ok &= got[i] == R0[i] + x[(i / ((size_t)n * hc)) * n + i % n];
        CHECK(ok);
    }
    Dev<float> a(8);
    CHECK(vw::broadcast_streams(a.p, a.p, 0, 1, 1, nullptr) == -1 && vw::add_streams_broadcast(nullptr, a.p, 1, 1, 1, nullptr) == -1);
}

static void test_small() {
    // rebase_ptrs: base + k*bytes for 128 blobs; more than 128 is rejected
    for (int n : {1, 16, 127, 128}) {
        Dev<uint64_t> d(n);
        CHECK(vw::rebase_ptrs(d.p, 0x1000, 1382400, n, nullptr) == 0 && dev_sync() == 0);
        const auto g = d.get(); bool ok = true;
        for (int k = 0; k < n; ++k) ok &= g[k] == 0x1000ull + (uint64_t)k * 1382400ull;
        CHECK(ok);
    }
    { Dev<uint64_t> d(200); CHECK(vw::rebase_ptrs(d.p, 0, 1, 129, nullptr) == -1 && vw::rebase_ptrs(d.p, 0, 1, 0, nullptr) == -1); }
    // dense steps
    for (int T = 1; T <= 8; ++T) for (int c0 : {0, 3, 100}) {
        Dev<int32_t> d(4 * T);
        CHECK(vw::dense_steps(d.p, c0, T, nullptr) == 0 && dev_sync() == 0);
        const auto g = d.get();
        for (int t = 0; t < T; ++t) { const int c = c0 + t; CHECK(g[4 * t] == c && g[4 * t + 1] == c + 1 && g[4 * t + 2] == (c + 1) / 4 && g[4 * t + 3] == c + 1); }
    }
    { Dev<int32_t> d(40); CHECK(vw::dense_steps(d.p, 0, 9, nullptr) == -1 && vw::dense_steps(d.p, -1, 2, nullptr) == -1); }
    // window ids
    for (int nkv : {0, 1, 5, 64, 65, 300}) for (int win : {1, 8, 64, 100}) {
        Dev<int32_t> ids(win), wd(1);
        CHECK(vw::window_ids(ids.p, wd.p, nkv, win, nullptr) == 0 && dev_sync() == 0);
        const int st = vw::win_start(nkv, win), width = vw::win_width(nkv, win);
        CHECK(wd.get()[0] == width);
        const auto g = ids.get();
        for (int j = 0; j < win; ++j) CHECK(g[j] == (j < width ? st + j : -1));
    }
    // map ids (n <= 64)
    { std::vector<int32_t> map = {10, 11, 12, 13}; Rng rng(5);
      for (int n : {1, 7, 64}) { std::vector<int32_t> ids(n); for (auto& v : ids) v = (int32_t)rng.below(8) - 2;
          Dev<int32_t> di(ids), dm(map), dout(n);
          CHECK(vw::map_ids(di.p, dm.p, dout.p, n, 4, nullptr) == 0 && dev_sync() == 0);
          const auto g = dout.get(); for (int i = 0; i < n; ++i) CHECK(g[i] == (ids[i] >= 0 && ids[i] < 4 ? map[ids[i]] : -1)); }
      Dev<int32_t> di(70), dm(4), dout(70); CHECK(vw::map_ids(di.p, dm.p, dout.p, 65, 4, nullptr) == -1); }
    // ident hits (n <= 1024), including a count that needs more than one pass of a 256-thread block
    { Rng rng(6);
      for (int n : {1, 64, 255, 256, 257, 1000, 1024}) { std::vector<int32_t> ids(n); int want = 0;
          for (int i = 0; i < n; ++i) { ids[i] = rng.below(3) ? i : (int32_t)rng.below(n) + (rng.below(2) ? 0 : 5000); want += ids[i] == i; }
          Dev<int32_t> di(ids), dh(1);
          CHECK(vw::ident_hits(di.p, n, dh.p, nullptr) == 0 && dev_sync() == 0 && dh.get()[0] == want); }
      Dev<int32_t> di(1100), dh(1); CHECK(vw::ident_hits(di.p, 1025, dh.p, nullptr) == -1); }
}

static void test_select() {
    Rng rng(27);
    // top prob: lowest index of the maximum, prob = 1 / sum exp(l - max); vocab not a multiple of the block
    for (int vocab : {1, 2, 100, 511, 512, 513, 5000}) for (int rows : {1, 3}) {
        auto lg = randv(rng, (size_t)rows * vocab, 3.f);
        if (vocab >= 8) { lg[3] = lg[7] = 50.f; }                           // tie: the lowest index wins
        Dev<float> dl(lg), dp(rows); Dev<int32_t> dt(rows);
        CHECK(vw::row_top_prob(dl.p, dt.p, dp.p, rows, vocab, nullptr) == 0 && dev_sync() == 0);
        const std::vector<int32_t> tk = dt.get(); const std::vector<float> pr = dp.get();
        for (int r = 0; r < rows; ++r) {
            const float* row = &lg[(size_t)r * vocab];
            int am = 0; for (int j = 1; j < vocab; ++j) if (row[j] > row[am]) am = j;
            double sum = 0; for (int j = 0; j < vocab; ++j) sum += std::exp((double)row[j] - row[am]);
            CHECK(tk[r] == am && close(pr[r], (float)(1.0 / sum), 1e-4f, 1e-7f));
            CHECK(pr[r] > 0.f && pr[r] <= 1.f + 1e-6f);
        }
    }
    // mtp select: lowest index of the maximum probability, token copied from that row
    for (int n : {1, 2, 5, 256, 300}) for (int tie : {0, 1}) {
        std::vector<float> pr(n); std::vector<int32_t> tk(n);
        for (int i = 0; i < n; ++i) { pr[i] = (float)rng.uni() * 0.9f; tk[i] = 1000 + i; }
        if (tie && n >= 2) { pr[n - 1] = pr[n / 2] = 0.99f; }
        Dev<float> dp(pr); Dev<int32_t> dt(tk), sel(1), tout(1);
        CHECK(vw::mtp_select(dt.p, dp.p, n, sel.p, tout.p, nullptr) == 0 && dev_sync() == 0);
        int am = 0; for (int i = 1; i < n; ++i) if (pr[i] > pr[am]) am = i;
        CHECK(sel.get()[0] == am && tout.get()[0] == 1000 + am);
    }
    // wait flag: reached immediately, or timed out
    { Dev<uint32_t> f(std::vector<uint32_t>{5}); Dev<int32_t> st(1);
      CHECK(vw::wait_flag_ge(f.p, 5, 100, st.p, nullptr) == 0 && dev_sync() == 0 && st.get()[0] == 1);
      CHECK(vw::wait_flag_ge(f.p, 6, 100, st.p, nullptr) == 0 && dev_sync() == 0 && st.get()[0] == 0);
      CHECK(vw::wait_flag_ge(f.p, 1, 0, st.p, nullptr) == -1); }
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("FAIL no device\n"); return 2; }
    cudaSetDevice(0);
    std::printf("devices: %d\n", n);
    u25c::run_all();                       // LAYER 1 host statements of all six modules (production code)
    test_conv(); test_commit(); test_ab(); test_step(); test_gathers(); test_streams(); test_small(); test_select();
    if (g_fail) { std::printf("U25C TESTS FAILED (%d checks)\n", g_fail); return 1; }
    std::printf("U25C TESTS PASSED\n");
    return 0;
}
