// test_prefill_batch.cu - GPU tests of the batched-prefill kernels against CPU references: bf16/f16 conversion is exact
// round-to-nearest-even, norms/mixes/writes/broadcast follow their index decomposition and write every element,
// GDN gates/conv/l2/recurrence match a sequential reference, routing returns the top-10 by logit with softmax weights,
// expert dequant equals (code-1)*scale element by element in f32 and f16, SwiGLU/gather/combine/split/gate/rope/kv_append
// match, and int8 KV never produces -128. Non-zero exit on any failure (2 when no device can be opened).
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/prefill_batch_api.hpp"
#include "baloon/prefill_gpu_kernels.hpp"
using namespace baloon;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { CUOK(cudaMalloc((void**)&p, n * sizeof(T))); CUOK(cudaMemset(p, 0, n * sizeof(T))); }
    Dev(const std::vector<T>& h) : n(h.size()) { CUOK(cudaMalloc((void**)&p, n * sizeof(T))); CUOK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> h(n); CUOK(cudaDeviceSynchronize()); CUOK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    void fill_nan() { CUOK(cudaMemset(p, 0xFF, n * sizeof(T))); }
};
static std::mt19937 rng(12345);
static std::vector<float> rnd(size_t n, float s = 1.f) { std::uniform_real_distribution<float> d(-s, s); std::vector<float> v(n); for (auto& x : v) x = d(rng); return v; }
static bool near(float a, float b, float rt = 2e-4f, float at = 2e-5f) { return std::fabs(a - b) <= at + rt * std::fabs(b); }
static bool all_near(const std::vector<float>& a, const std::vector<float>& b, float rt = 2e-4f, float at = 2e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!(near(a[i], b[i], rt, at))) { if (g_fail < 5) std::printf("  mismatch at %zu: got %g want %g\n", i, a[i], b[i]); return false; }
    return true;
}
static float sg(float x) { return 1.f / (1.f + std::exp(-x)); }

static void test_conversions() {
    auto x = rnd(100003, 50.f);
    x[0] = 0.f; x[1] = -0.f; x[2] = 1.f; x[3] = 65504.f; x[4] = 1e-8f; x[5] = 3.14159f;
    Dev<float> dx(x); Dev<uint16_t> yb(x.size()), yh(x.size());
    CHECK(pfb::to_bf16(dx.p, yb.p, (int64_t)x.size()) == 0); CHECK(pfb::to_f16(dx.p, yh.p, (int64_t)x.size()) == 0);
    auto b = yb.get(), h = yh.get();
    size_t badb = 0;
    for (size_t i = 0; i < x.size(); ++i) badb += (b[i] != prefill_kernels::bf_from_float(x[i]));
    CHECK(badb == 0);
    size_t badh = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const float back = prefill_kernels::f16_bits_to_f32(h[i]);
        if (std::fabs(x[i]) < 60000.f && std::fabs(x[i]) > 7e-5f) badh += !(std::fabs(back - x[i]) <= std::fabs(x[i]) * 4.9e-4f);
    }
    CHECK(badh == 0);
    CHECK(pfb::to_bf16(nullptr, yb.p, 4) == -1); CHECK(pfb::to_bf16(dx.p, yb.p, 0) == -1);
}

static void test_norm_and_hc() {
    for (int T : {1, 3, 7}) {
        const size_t N = pfb::kEmbN, D = pfb::kHcWidth;
        auto res = rnd(T * D), coef = rnd(T * 6), blk = rnd(T * N), e = rnd(T * N);
        Dev<float> dr(res), dc(coef), db(blk), de(e), y(T * N), w(T * D), nrm(T * D), bc(T * D);
        // gr_mix
        CHECK(pfb::gr_mix(dr.p, dc.p, 6, y.p, T) == 0);
        auto gy = y.get(); std::vector<float> ry(T * N);
        for (int t = 0; t < T; ++t) for (size_t d = 0; d < N; ++d) { float a = 0; for (int c = 0; c < 4; ++c) a += coef[t * 6 + c] * res[t * D + c * N + d]; ry[t * N + d] = a; }
        CHECK(all_near(gy, ry));
        // gr_write
        w.fill_nan();
        CHECK(pfb::gr_write(dr.p, dc.p, 6, db.p, w.p, T) == 0);
        auto gw = w.get(); std::vector<float> rw(T * D);
        for (int t = 0; t < T; ++t) for (int c = 0; c < 4; ++c) for (size_t d = 0; d < N; ++d) rw[t * D + c * N + d] = res[t * D + c * N + d] + coef[t * 6 + c] * blk[t * N + d];
        CHECK(all_near(gw, rw));                                       // NaN fill: every element was written
        // gr_broadcast
        bc.fill_nan();
        CHECK(pfb::gr_broadcast(de.p, bc.p, T) == 0);
        auto gb = bc.get(); bool ok = true;
        for (int t = 0; t < T; ++t) for (int c = 0; c < 4; ++c) for (size_t d = 0; d < N; ++d) ok &= (gb[t * D + c * N + d] == e[t * N + d]);
        CHECK(ok);
        // gr_norm
        nrm.fill_nan();
        CHECK(pfb::gr_norm(dr.p, nrm.p, T, 1e-6f) == 0);
        auto gn = nrm.get(); std::vector<float> rn(T * D);
        for (int t = 0; t < T; ++t) { double s = 0; for (size_t i = 0; i < D; ++i) s += (double)res[t * D + i] * res[t * D + i]; const float k = 1.f / std::sqrt((float)(s / D) + 1e-6f); for (size_t i = 0; i < D; ++i) rn[t * D + i] = res[t * D + i] * k; }
        CHECK(all_near(gn, rn, 1e-3f, 1e-4f));
    }
    CHECK(pfb::gr_mix(nullptr, nullptr, 4, nullptr, 1) == -1);
    { Dev<float> a(8), b(8), c(8); CHECK(pfb::gr_mix(a.p, b.p, 3, c.p, 1) == -1); }   // ld < 4 rejected
    // rms_rows and silu
    const int rows = 5, dim = 2560;
    auto x = rnd(rows * dim), wt = rnd(dim);
    Dev<float> dx(x), dw(wt), dy(rows * dim);
    CHECK(pfb::rms_rows(dx.p, dw.p, dy.p, rows, dim, 1e-6f) == 0);
    auto gy = dy.get(); std::vector<float> ry(rows * dim);
    for (int r = 0; r < rows; ++r) { double s = 0; for (int d = 0; d < dim; ++d) s += (double)x[r * dim + d] * x[r * dim + d]; float k = 1.f / std::sqrt((float)(s / dim) + 1e-6f); for (int d = 0; d < dim; ++d) ry[r * dim + d] = x[r * dim + d] * k * wt[d]; }
    CHECK(all_near(gy, ry, 1e-3f, 1e-4f));
    auto sx = rnd(12345, 8.f); Dev<float> dsx(sx), dsy(sx.size());
    CHECK(pfb::gr_silu(dsx.p, dsy.p, (int64_t)sx.size()) == 0);
    auto gs = dsy.get(); std::vector<float> rs(sx.size()); for (size_t i = 0; i < sx.size(); ++i) rs[i] = sx[i] * sg(sx[i]);
    CHECK(all_near(gs, rs));
}

static void test_gdn() {
    const int T = 9, C = pfb::kConvC, H = pfb::kValHeads, S = pfb::kStateS;
    // gates
    auto a = rnd(T * H, 3.f), b = rnd(T * H, 3.f), al = rnd(H, 1.f), dt = rnd(H, 1.f);
    Dev<float> da(a), db(b), dal(al), ddt(dt), g(T * H), be(T * H);
    CHECK(pfb::gdn_gates(da.p, db.p, dal.p, ddt.p, g.p, be.p, T) == 0);
    auto gg = g.get(), gb = be.get(); std::vector<float> rg(T * H), rb(T * H);
    for (int i = 0; i < T * H; ++i) { const int h = i % H; const float x = a[i] + dt[h]; rg[i] = -std::exp(al[h]) * (x > 20.f ? x : std::log1p(std::exp(x))); rb[i] = sg(b[i]); }
    CHECK(all_near(gg, rg)); CHECK(all_near(gb, rb));
    for (float v : gg) CHECK(v <= 0.f);                                // decay exponent is never positive
    for (float v : gb) CHECK(v > 0.f && v < 1.f);
    // conv: two chunks must equal one pass (state carries over), and match the sequential reference
    auto x = rnd((size_t)T * C), w = rnd((size_t)C * 4, 0.5f), st0 = rnd((size_t)C * 3, 0.5f);
    Dev<float> dx(x), dw(w), dst(st0), dout((size_t)T * C);
    CHECK(pfb::gdn_conv(dx.p, dw.p, dst.p, dout.p, T) == 0);
    auto go = dout.get(), gst = dst.get(); std::vector<float> ro((size_t)T * C), rst(st0);
    for (int c = 0; c < C; ++c) { float s0 = st0[c * 3], s1 = st0[c * 3 + 1], s2 = st0[c * 3 + 2];
        for (int t = 0; t < T; ++t) { const float v = x[(size_t)t * C + c]; const float acc = w[c * 4] * s0 + w[c * 4 + 1] * s1 + w[c * 4 + 2] * s2 + w[c * 4 + 3] * v; ro[(size_t)t * C + c] = acc * sg(acc); s0 = s1; s1 = s2; s2 = v; }
        rst[c * 3] = s0; rst[c * 3 + 1] = s1; rst[c * 3 + 2] = s2; }
    CHECK(all_near(go, ro)); CHECK(all_near(gst, rst));
    { Dev<float> s2(st0), o2((size_t)T * C), x1(std::vector<float>(x.begin(), x.begin() + (size_t)4 * C)), x2(std::vector<float>(x.begin() + (size_t)4 * C, x.end()));
      Dev<float> o1((size_t)4 * C), o3((size_t)(T - 4) * C);
      CHECK(pfb::gdn_conv(x1.p, dw.p, s2.p, o1.p, 4) == 0); CHECK(pfb::gdn_conv(x2.p, dw.p, s2.p, o3.p, T - 4) == 0);
      auto p1 = o1.get(), p2 = o3.get(); p1.insert(p1.end(), p2.begin(), p2.end());
      CHECK(all_near(p1, ro)); CHECK(all_near(s2.get(), rst)); }
    // l2 over the 32 q/k heads only (v untouched)
    auto qkv = rnd((size_t)T * C, 2.f); Dev<float> dq(qkv);
    CHECK(pfb::gdn_l2(dq.p, T, 1e-6f) == 0);
    auto gq = dq.get(); bool okq = true;
    for (int t = 0; t < T; ++t) for (int h = 0; h < 32; ++h) { double s = 0; for (int i = 0; i < S; ++i) { const float v = qkv[(size_t)t * C + h * S + i]; s += (double)v * v; } const float k = 1.f / std::sqrt((float)s + 1e-6f);
        for (int i = 0; i < S; ++i) okq &= near(gq[(size_t)t * C + h * S + i], qkv[(size_t)t * C + h * S + i] * k, 1e-3f, 1e-5f); }
    CHECK(okq);
    for (int t = 0; t < T; ++t) for (int i = 32 * S; i < C; ++i) CHECK(gq[(size_t)t * C + i] == qkv[(size_t)t * C + i]);
    // recurrence vs sequential reference (state layout [row][head][col]); chunked == whole
    auto rq = gq;                                                      // normalised q/k, raw v
    auto state0 = rnd((size_t)S * H * S, 0.1f);
    auto gdec = rg, gbeta = rb;
    Dev<float> dqq(rq), dgd(gdec), dgb(gbeta), dstate(state0), dro((size_t)T * pfb::kValWidth);
    const float scale = 1.f / std::sqrt((float)S);
    CHECK(pfb::gdn_recurrence(dqq.p, dgd.p, dgb.p, dstate.p, dro.p, T, scale) == 0);
    auto gout = dro.get(), gstate = dstate.get();
    std::vector<float> rs(state0), rout((size_t)T * pfb::kValWidth);
    for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        const int src = h % pfb::kKeyHeads; const float decay = std::exp(rg[t * H + h]), bt = rb[t * H + h];
        const float* q = &rq[(size_t)t * C + src * S]; const float* k = &rq[(size_t)t * C + pfb::kKeyHeads * S + src * S]; const float* v = &rq[(size_t)t * C + 2 * pfb::kKeyHeads * S + h * S];
        for (int c = 0; c < S; ++c) {
            float kv = 0; for (int r = 0; r < S; ++r) kv += rs[((size_t)r * H + h) * S + c] * k[r]; kv *= decay;
            const float delta = (v[c] - kv) * bt; float o = 0;
            for (int r = 0; r < S; ++r) { float& s = rs[((size_t)r * H + h) * S + c]; s = s * decay + k[r] * delta; o += s * q[r]; }
            rout[(size_t)t * pfb::kValWidth + h * S + c] = scale * o; } }
    CHECK(all_near(gout, rout, 5e-3f, 5e-4f)); CHECK(all_near(gstate, rs, 5e-3f, 5e-4f));
}

static void test_route() {
    for (int T : {1, 5, 33}) {
        auto lg = rnd((size_t)T * 512, 4.f);
        for (int t = 0; t < T; ++t) { lg[(size_t)t * 512 + 5] = lg[(size_t)t * 512 + 400] = 9.f; }   // a tie: lowest index must win
        Dev<float> dl(lg), w((size_t)T * 10); Dev<int32_t> ids((size_t)T * 10);
        CHECK(pfb::route(dl.p, ids.p, w.p, T) == 0);
        auto gi = ids.get(); auto gw = w.get();
        for (int t = 0; t < T; ++t) {
            std::vector<int> ord(512); std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return lg[(size_t)t * 512 + x] > lg[(size_t)t * 512 + y]; });
            double s = 0; float m = lg[(size_t)t * 512 + ord[0]]; for (int r = 0; r < 10; ++r) s += std::exp(lg[(size_t)t * 512 + ord[r]] - m);
            double sum = 0;
            for (int r = 0; r < 10; ++r) {
                CHECK(gi[(size_t)t * 10 + r] == ord[r]);                // distinct, ranked, ties to lowest index
                CHECK(gi[(size_t)t * 10 + r] >= 0 && gi[(size_t)t * 10 + r] < 512);
                CHECK(near(gw[(size_t)t * 10 + r], (float)(std::exp(lg[(size_t)t * 512 + ord[r]] - m) / s), 1e-3f, 1e-6f));
                sum += gw[(size_t)t * 10 + r];
            }
            CHECK(std::fabs(sum - 1.0) < 1e-4);
        }
    }
}

static void test_blob() {
    std::vector<uint8_t> blob(pfb::kBlobBytes);
    for (auto& v : blob) v = (uint8_t)rng();
    for (int64_t i = 0; i < 1280 * 40; ++i) { const uint16_t h = prefill_kernels::bf_from_float(0.f); (void)h; const float s = 0.01f + (float)(rng() % 100) * 0.002f; uint16_t hb = 0;
        { uint32_t u; std::memcpy(&u, &s, 4); const uint32_t sg_ = u >> 31, ex = ((u >> 23) & 0xFF) - 127 + 15, mn = (u >> 13) & 0x3FF; hb = (uint16_t)((sg_ << 15) | (ex << 10) | mn); }
        std::memcpy(&blob[pfb::kGuScaleOff + i * 2], &hb, 2); }
    for (int64_t i = 0; i < 2560 * 10; ++i) { const float s = 0.02f + (float)(rng() % 50) * 0.004f; uint32_t u; std::memcpy(&u, &s, 4);
        const uint16_t hb = (uint16_t)(((u >> 31) << 15) | ((((u >> 23) & 0xFF) - 127 + 15) << 10) | ((u >> 13) & 0x3FF)); std::memcpy(&blob[pfb::kDownScaleOff + i * 2], &hb, 2); }
    Dev<uint8_t> db(blob); Dev<float> gu(1280 * 2560), dn(2560 * 640); Dev<uint16_t> guh(1280 * 2560), dnh(2560 * 640);
    gu.fill_nan(); dn.fill_nan();
    CHECK(pfb::blob_dequant(db.p, gu.p, dn.p) == 0); CHECK(pfb::blob_dequant_f16(db.p, guh.p, dnh.p) == 0);
    auto g = gu.get(), d = dn.get(); auto gh = guh.get(), dh = dnh.get();
    size_t bad = 0, badh = 0;
    for (int64_t row = 0; row < 1280; ++row) for (int64_t col = 0; col < 2560; ++col) { const float w = prefill_kernels::gu_weight(blob.data(), row, col); bad += !(g[row * 2560 + col] == w);
        badh += !(std::fabs(prefill_kernels::f16_bits_to_f32(gh[row * 2560 + col]) - w) <= std::fabs(w) * 1e-3f + 1e-6f); }
    for (int64_t row = 0; row < 2560; ++row) for (int64_t col = 0; col < 640; ++col) { const float w = prefill_kernels::down_weight(blob.data(), row, col); bad += !(d[row * 640 + col] == w);
        badh += !(std::fabs(prefill_kernels::f16_bits_to_f32(dh[row * 640 + col]) - w) <= std::fabs(w) * 1e-3f + 1e-6f); }
    CHECK(bad == 0); CHECK(badh == 0);                                  // NaN fill: every element written, none out of place
}

static void test_moe_misc() {
    // swiglu
    const int R = 37; auto x = rnd((size_t)R * 1280, 4.f); Dev<float> dx(x), dy((size_t)R * 640);
    CHECK(pfb::swiglu_interleaved(dx.p, dy.p, R) == 0);
    auto gy = dy.get(); std::vector<float> ry((size_t)R * 640);
    for (int r = 0; r < R; ++r) for (int k = 0; k < 640; ++k) { const float gt = x[(size_t)r * 1280 + 2 * k], up = x[(size_t)r * 1280 + 2 * k + 1]; ry[(size_t)r * 640 + k] = gt * sg(gt) * up; }
    CHECK(all_near(gy, ry));
    auto ga = rnd(10001, 4.f), up = rnd(10001, 4.f); Dev<float> dga(ga), dup(up), dpy(ga.size());
    CHECK(pfb::swiglu_pair(dga.p, dup.p, dpy.p, (int64_t)ga.size()) == 0);
    auto pgy = dpy.get(); std::vector<float> pry(ga.size()); for (size_t i = 0; i < ga.size(); ++i) pry[i] = ga[i] * sg(ga[i]) * up[i];
    CHECK(all_near(pgy, pry));
    // gather rows of 2560 bf16 = 320 uint4
    const int src_rows = 50, n = 20, per = 320;
    std::vector<uint8_t> src((size_t)src_rows * per * 16); for (auto& v : src) v = (uint8_t)rng();
    std::vector<int32_t> idx(n); for (auto& i : idx) i = (int32_t)(rng() % src_rows);
    Dev<uint8_t> ds(src), dd((size_t)n * per * 16); Dev<int32_t> di(idx);
    CHECK(pfb::gather_rows16(ds.p, di.p, dd.p, n, per) == 0);
    auto gd = dd.get(); bool ok = true;
    for (int r = 0; r < n; ++r) ok &= std::memcmp(&gd[(size_t)r * per * 16], &src[(size_t)idx[r] * per * 16], (size_t)per * 16) == 0;
    CHECK(ok);
    // combine
    const int T = 6; auto sh = rnd((size_t)T * 2560), ex = rnd((size_t)T * 10 * 2560), w = rnd((size_t)T * 10, 0.3f);
    Dev<float> dsh(sh), dex(ex), dw(w), dout((size_t)T * 2560);
    CHECK(pfb::moe_combine(dsh.p, dex.p, dw.p, dout.p, T) == 0);
    auto go = dout.get(); std::vector<float> ro((size_t)T * 2560);
    for (int t = 0; t < T; ++t) for (int d = 0; d < 2560; ++d) { float a = sh[(size_t)t * 2560 + d]; for (int k = 0; k < 10; ++k) a += w[t * 10 + k] * ex[((size_t)t * 10 + k) * 2560 + d]; ro[(size_t)t * 2560 + d] = a; }
    CHECK(all_near(go, ro));
}

static void test_qsa() {
    const int T = 4; auto qf = rnd((size_t)T * 12288, 3.f), at = rnd((size_t)T * 6144);
    Dev<float> dqf(qf), dat(at), dq((size_t)T * 6144), dg((size_t)T * 6144);
    dq.fill_nan(); dg.fill_nan();
    CHECK(pfb::split_q(dqf.p, dq.p, T) == 0); CHECK(pfb::gate_attn(dat.p, dqf.p, dg.p, T) == 0);
    auto gq = dq.get(), gg = dg.get(); bool ok = true;
    for (int t = 0; t < T; ++t) for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) {
        ok &= gq[(size_t)t * 6144 + h * 256 + d] == qf[(size_t)t * 12288 + h * 512 + d];
        ok &= near(gg[(size_t)t * 6144 + h * 256 + d], at[(size_t)t * 6144 + h * 256 + d] * sg(qf[(size_t)t * 12288 + h * 512 + 256 + d])); }
    CHECK(ok);
    // rope: preserves pair norms, identity at position 0, matches reference
    const int heads = 3, hd = 128; auto x = rnd((size_t)T * heads * hd); std::vector<int32_t> pos = {0, 1, 17, 1000};
    Dev<float> dx(x); Dev<int32_t> dp(pos);
    CHECK(pfb::rope(dx.p, dp.p, T, heads, hd, 10000.f) == 0);
    auto gx = dx.get(); ok = true;
    for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) { const size_t o = ((size_t)t * heads + h) * hd;
        for (int p = 0; p < 32; ++p) { const float ang = (float)pos[t] * std::pow(10000.f, -(float)(2 * p) / 64.f); const float c = std::cos(ang), s = std::sin(ang);
            ok &= near(gx[o + p], x[o + p] * c - x[o + p + 32] * s, 2e-3f, 2e-3f); ok &= near(gx[o + p + 32], x[o + p] * s + x[o + p + 32] * c, 2e-3f, 2e-3f);
            ok &= near(gx[o + p] * gx[o + p] + gx[o + p + 32] * gx[o + p + 32], x[o + p] * x[o + p] + x[o + p + 32] * x[o + p + 32], 1e-3f, 1e-4f); }
        for (int d = 64; d < hd; ++d) ok &= gx[o + d] == x[o + d];          // dims past the rotary block untouched
        if (pos[t] == 0) for (int d = 0; d < 64; ++d) ok &= near(gx[o + d], x[o + d]); }
    CHECK(ok);
    // kv_append: scatter by row, per-64 scale, clamp never -128, dequant error bounded by scale/2
    auto kx = rnd((size_t)T * 256, 5.f); for (int g = 0; g < 4; ++g) kx[g * 64 + 3] = -5.f * (g + 1);   // force an extreme negative value
    std::vector<int32_t> rows = {7, 2, 9, 0}; const int R = 10;
    Dev<float> dkx(kx), dsc((size_t)R * 4); Dev<int32_t> dr(rows); Dev<int8_t> dkq((size_t)R * 256);
    CHECK(pfb::kv_append(dkx.p, dr.p, dkq.p, dsc.p, T) == 0);
    auto kq = dkq.get(); auto sc = dsc.get(); ok = true;
    for (int t = 0; t < T; ++t) for (int g = 0; g < 4; ++g) {
        float amax = 0; for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(kx[(size_t)t * 256 + g * 64 + i]));
        const float scale = amax / 127.f; ok &= near(sc[(size_t)rows[t] * 4 + g], scale, 1e-5f, 1e-9f);
        for (int i = 0; i < 64; ++i) { const int q = kq[(size_t)rows[t] * 256 + g * 64 + i]; ok &= (q >= -127 && q <= 127);
            ok &= std::fabs(q * scale - kx[(size_t)t * 256 + g * 64 + i]) <= scale * 0.5f + 1e-6f; } }
    CHECK(ok);
    for (int8_t q : kq) CHECK(q != -128);
}

int main() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) { std::printf("no CUDA device: %s\n", cudaGetErrorString(e)); return 2; }
    CUOK(cudaSetDevice(0));
    test_conversions(); test_norm_and_hc(); test_gdn(); test_route(); test_blob(); test_moe_misc(); test_qsa();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("PREFILL BATCH GPU TESTS FAILED\n"); return 1; }
    std::printf("PREFILL BATCH GPU TESTS PASSED\n");
    return 0;
}
