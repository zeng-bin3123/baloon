// Unit pf7b804cf95: invariants of the batched prompt-path kernels (src/prefill/kernels.cu), checked on the device against
// independent host references: exact data-movement laws, rounding laws of bf16/f16, softmax/top-10 routing with lowest-id ties,
// the 2-bit blob dequantisation, int8 KV quantisation bounds, and chunk-split invariance of the sequential GDN kernels.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/prefill_kernels_api.hpp"

using namespace baloon::pk;
namespace fb = baloon::f16bits;
static int g_fail = 0, g_checks = 0;
static std::mt19937_64 rng(777);
static float rf(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static void ok(bool c, const char* w) { ++g_checks; if (!c) { ++g_fail; std::printf("FAIL: %s\n", w); } }
template <class T> struct Dev {
    T* p = nullptr; size_t n;
    explicit Dev(size_t n_) : n(n_) { cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T)); cudaMemset(p, 0, (n ? n : 1) * sizeof(T)); }
    Dev(const std::vector<T>& v) : Dev(v.size()) { cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice); }
    Dev(const Dev&) = delete; Dev& operator=(const Dev&) = delete;
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
    operator T*() const { return p; }
};
static std::vector<float> vf(size_t n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = rf(lo, hi); return v; }
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess; }
static double sg(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static void near(const std::vector<float>& a, const std::vector<double>& r, double rtol, double atol, const char* name) {
    bool good = a.size() == r.size();
    for (size_t i = 0; good && i < a.size(); ++i)
        if (std::isnan(a[i]) || std::fabs((double)a[i] - r[i]) > atol + rtol * std::fabs(r[i])) {
            std::printf("  %s: idx %zu got %.9g want %.9g\n", name, i, (double)a[i], (double)r[i]); good = false; }
    ok(good, name);
}

int main() {
    int dev = 0; if (cudaGetDeviceCount(&dev) != cudaSuccess || dev < 1) { std::printf("no GPU\n"); return 1; }
    cudaSetDevice(0);
    const int T = 3;

    // ---- element-wise ----
    { const int64_t n = 5000; auto x = vf(n, -8, 8), y = vf(n, -3, 3);
      Dev<float> dx(x), dy(y), o(n);
      ok(gr_silu(dx, o, n) == 0 && sync_ok(), "gr_silu launch");
      std::vector<double> r(n); for (int64_t i = 0; i < n; ++i) r[i] = x[i] * sg(x[i]);
      near(o.get(), r, 1e-5, 1e-6, "gr_silu values");
      ok(swiglu_pair(dx, dy, o, n) == 0 && sync_ok(), "swiglu_pair launch");
      for (int64_t i = 0; i < n; ++i) r[i] = x[i] * sg(x[i]) * y[i];
      near(o.get(), r, 1e-5, 1e-6, "swiglu_pair values");
      ok(gate_attn(dx, dy, o, n) == 0 && sync_ok(), "gate_attn launch");
      for (int64_t i = 0; i < n; ++i) r[i] = x[i] * sg(y[i]);
      near(o.get(), r, 1e-5, 1e-6, "gate_attn = attn*sigmoid(gate)");
      // swiglu_interleaved: (g,u) interleaved along the row
      const int rows = 7, w = 33; auto in = vf((size_t)rows * 2 * w, -4, 4); Dev<float> di(in), oo((size_t)rows * w);
      ok(swiglu_interleaved(di, oo, rows, w) == 0 && sync_ok(), "swiglu_interleaved launch");
      std::vector<double> ri((size_t)rows * w);
      for (int a = 0; a < rows; ++a) for (int k = 0; k < w; ++k) { double g = in[(size_t)a * 2 * w + 2 * k], u = in[(size_t)a * 2 * w + 2 * k + 1]; ri[(size_t)a * w + k] = g * sg(g) * u; }
      near(oo.get(), ri, 1e-5, 1e-6, "swiglu_interleaved values");
      // to_bf16 / to_f16: exact rounding laws
      std::vector<float> sp = x; sp[0] = 0.f; sp[1] = -0.f; sp[2] = 65504.f; sp[3] = 1e-8f; sp[4] = 1.00048828125f;
      Dev<float> ds(sp); Dev<uint16_t> ob(n), oh(n);
      ok(to_bf16(ds, ob, n) == 0 && to_f16(ds, oh, n) == 0 && sync_ok(), "to_bf16/to_f16 launch");
      auto hb = ob.get(), hh = oh.get(); bool eb = true, eh = true;
      for (int64_t i = 0; i < n; ++i) { eb &= hb[i] == fb::round_f32_to_bf16_bits(sp[i]); eh &= hh[i] == fb::f16_from_f32(sp[i]); }
      ok(eb, "to_bf16 == round-to-nearest-even bf16"); ok(eh, "to_f16 == host f16 conversion");
    }

    // ---- hyper-connection ----
    { auto x = vf((size_t)T * D, -2, 2); Dev<float> dx(x), o((size_t)T * D);
      ok(gr_norm(dx, o, T, 1e-6f) == 0 && sync_ok(), "gr_norm launch");
      auto h = o.get(); bool good = true;
      for (int t = 0; t < T; ++t) { double s = 0; for (int i = 0; i < D; ++i) s += (double)h[(size_t)t * D + i] * h[(size_t)t * D + i]; good &= std::fabs(s / D - 1.0) < 1e-3; }
      ok(good, "gr_norm: mean square over the 4 streams is 1");
      const int ld = 8; auto coef = vf((size_t)T * ld, -1, 1); Dev<float> dc(coef), om((size_t)T * N);
      ok(gr_mix(dx, dc, ld, om, T) == 0 && sync_ok(), "gr_mix launch");
      std::vector<double> r((size_t)T * N);
      for (int t = 0; t < T; ++t) for (int d = 0; d < N; ++d) { double a = 0; for (int c = 0; c < HC; ++c) a += (double)coef[t * ld + c] * x[(size_t)t * D + (size_t)c * N + d]; r[(size_t)t * N + d] = a; }
      near(om.get(), r, 1e-5, 1e-5, "gr_mix values");
      auto y = vf((size_t)T * N, -2, 2); auto res = vf((size_t)T * D, -2, 2); Dev<float> dy(y), dr(res), ow((size_t)T * D);
      ok(gr_write(dr, dy, dc, ld, ow, T) == 0 && sync_ok(), "gr_write launch");
      std::vector<double> rw((size_t)T * D);
      for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) for (int d = 0; d < N; ++d) { size_t i = (size_t)t * D + (size_t)c * N + d; rw[i] = res[i] + (double)coef[t * ld + c] * y[(size_t)t * N + d]; }
      near(ow.get(), rw, 1e-5, 1e-5, "gr_write values");
      Dev<float> ob((size_t)T * D);
      ok(gr_broadcast(om, ob, T) == 0 && sync_ok(), "gr_broadcast launch");
      auto hb = ob.get(), hm = om.get(); bool eq = true;
      for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) for (int d = 0; d < N; ++d) eq &= hb[(size_t)t * D + (size_t)c * N + d] == hm[(size_t)t * N + d];
      ok(eq, "gr_broadcast copies each token row to all 4 streams exactly");
    }

    // ---- MoE ----
    { auto lg = vf((size_t)T * kExperts, -5, 5);
      for (int e : {10, 100, 300}) lg[e] = 9.f;              // equal maxima in token 0: tie-break by lowest id
      Dev<float> dl(lg); Dev<int32_t> ids((size_t)T * kRoutedK); Dev<float> w((size_t)T * kRoutedK);
      ok(route(dl, ids, w, T) == 0 && sync_ok(), "route launch");
      auto hi = ids.get(); auto hw = w.get(); bool good = true, order = true, sums = true;
      for (int t = 0; t < T; ++t) {
          std::vector<int> idx(kExperts); for (int e = 0; e < kExperts; ++e) idx[e] = e;
          std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lg[(size_t)t * kExperts + a] > lg[(size_t)t * kExperts + b]; });
          double s = 0; for (int k = 0; k < kRoutedK; ++k) { good &= hi[t * kRoutedK + k] == idx[k]; s += hw[t * kRoutedK + k]; if (k) order &= hw[t * kRoutedK + k] <= hw[t * kRoutedK + k - 1] + 1e-7f; }
          sums &= std::fabs(s - 1.0) < 1e-5;
      }
      ok(good, "route: top-10 ids in descending order, ties to the lowest expert id");
      ok(order && sums, "route: weights are a descending softmax summing to 1");
      Dev<int> bad(1); auto ids2 = hi; ids2[0] = kExperts; ids2[1] = -1; Dev<int32_t> di2(ids2);
      ok(route_ids_bad(di2, (int64_t)ids2.size(), bad) == 0 && sync_ok(), "route_ids_bad launch");
      ok(bad.get()[0] == 2, "route_ids_bad counts ids outside [0, 512)");
      Dev<int> bad0(1); ok(route_ids_bad(ids, (int64_t)T * kRoutedK, bad0) == 0 && sync_ok() && bad0.get()[0] == 0, "route_ids_bad: valid ids count 0");
      // moe_combine
      auto eo = vf((size_t)T * kRoutedK * N, -1, 1), sh = vf((size_t)T * N, -1, 1), ww = vf((size_t)T * kRoutedK, 0, 1);
      Dev<float> de(eo), ds(sh), dw(ww), oc((size_t)T * N);
      ok(moe_combine(de, dw, ds, oc, T) == 0 && sync_ok(), "moe_combine launch");
      std::vector<double> r((size_t)T * N);
      for (int t = 0; t < T; ++t) for (int d = 0; d < N; ++d) { double a = sh[(size_t)t * N + d]; for (int k = 0; k < kRoutedK; ++k) a += (double)ww[t * kRoutedK + k] * eo[((size_t)t * kRoutedK + k) * N + d]; r[(size_t)t * N + d] = a; }
      near(oc.get(), r, 1e-5, 1e-5, "moe_combine = shared + sum w*expert");
      // gather_rows16
      const int rows = 6, width = 24, take = 5; std::vector<uint16_t> src((size_t)rows * width); for (auto& v : src) v = (uint16_t)rng();
      std::vector<int32_t> ix = {5, 0, 3, 3, 1}; Dev<uint16_t> dsrc(src), dd((size_t)take * width); Dev<int32_t> dix(ix);
      ok(gather_rows16(dsrc, dix, dd, take, width) == 0 && sync_ok(), "gather_rows16 launch");
      auto g = dd.get(); bool eq = true;
      for (int r2 = 0; r2 < take; ++r2) for (int j = 0; j < width; ++j) eq &= g[(size_t)r2 * width + j] == src[(size_t)ix[r2] * width + j];
      ok(eq, "gather_rows16 copies the indexed rows bit-exactly");
      // blob_dequant: (code-1)*f16 scale, f16 variant == f16 of f32 variant
      std::vector<uint8_t> blob((size_t)kBlobBytes); for (auto& b : blob) b = (uint8_t)rng();
      auto setscales = [&](int64_t off, int64_t count) { for (int64_t i = 0; i < count; ++i) { uint16_t h = fb::f16_from_f32(rf(0.01f, 2.f)); blob[off + 2 * i] = h & 255; blob[off + 2 * i + 1] = h >> 8; } };
      setscales(kGuScaleOff, 1280 * 40); setscales(kDownScaleOff, 2560 * 10);
      Dev<uint8_t> db(blob); Dev<float> gu((size_t)1280 * 2560), dn((size_t)2560 * 640); Dev<uint16_t> guh((size_t)1280 * 2560), dnh((size_t)2560 * 640);
      ok(blob_dequant(db, gu, dn) == 0 && blob_dequant_f16(db, guh, dnh) == 0 && sync_ok(), "blob_dequant launches");
      auto hg = gu.get(), hd = dn.get(); auto hgh = guh.get(), hdh = dnh.get(); bool vals = true, f16eq = true;
      std::mt19937_64 pick(5);
      for (int s = 0; s < 20000; ++s) {
          bool isgu = pick() & 1; int64_t width2 = isgu ? 2560 : 640, rows2 = isgu ? 1280 : 2560, bytes = isgu ? 640 : 160, scales = isgu ? 40 : 10;
          int64_t row = pick() % rows2, col = pick() % width2;
          uint8_t c = blob[(isgu ? kGuOff : kDownOff) + row * bytes + col / 4];
          int64_t si = row * scales + col / 64; const uint8_t* sc = &blob[isgu ? kGuScaleOff : kDownScaleOff];
          float sv = fb::f32_from_f16((uint16_t)(sc[2 * si] | (sc[2 * si + 1] << 8)));
          float want = sv * (float)((int)((c >> (2 * (col % 4))) & 3) - 1);
          float got = (isgu ? hg : hd)[row * width2 + col]; uint16_t gh = (isgu ? hgh : hdh)[row * width2 + col];
          vals &= got == want; f16eq &= gh == fb::f16_from_f32(want);
      }
      ok(vals, "blob_dequant: value = (2-bit code - 1) * f16 group scale (64 columns per scale)");
      ok(f16eq, "blob_dequant_f16 == f16 rounding of blob_dequant");
    }

    // ---- QSA helpers ----
    { const int rows = 5, width = 301; auto x = vf((size_t)rows * width, -3, 3), wt = vf(width, 0.5f, 1.5f);
      Dev<float> dx(x), dw(wt), o((size_t)rows * width);
      ok(rms_rows(dx, dw, o, rows, width, 1e-6f) == 0 && sync_ok(), "rms_rows launch");
      std::vector<double> r((size_t)rows * width);
      for (int a = 0; a < rows; ++a) { double s = 0; for (int i = 0; i < width; ++i) s += (double)x[(size_t)a * width + i] * x[(size_t)a * width + i]; double inv = 1.0 / std::sqrt(s / width + 1e-6);
          for (int i = 0; i < width; ++i) r[(size_t)a * width + i] = x[(size_t)a * width + i] * inv * wt[i]; }
      near(o.get(), r, 1e-4, 1e-5, "rms_rows values");
      // split_q: 24 heads of (q 256 | gate 256)
      auto qf = vf((size_t)T * 12288, -1, 1); Dev<float> dq(qf), q((size_t)T * 6144), gt((size_t)T * 6144);
      ok(split_q(dq, q, gt, T) == 0 && sync_ok(), "split_q launch");
      auto hq = q.get(), hg = gt.get(); bool eq = true;
      for (int t = 0; t < T; ++t) for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) { eq &= hq[(size_t)t * 6144 + h * 256 + d] == qf[(size_t)t * 12288 + h * 512 + d]; eq &= hg[(size_t)t * 6144 + h * 256 + d] == qf[(size_t)t * 12288 + h * 512 + 256 + d]; }
      ok(eq, "split_q separates q and gate per head exactly");
      // rope: position 0 is the identity, rotation preserves pair norms, dims >= 64 untouched
      const int heads = 2, hd = 128; auto xr = vf((size_t)T * heads * hd, -1, 1); std::vector<int32_t> pos = {0, 7, 1000};
      Dev<float> dxr(xr); Dev<int32_t> dp(pos);
      ok(rope(dxr, dp, T, heads, hd, 10000.f) == 0 && sync_ok(), "rope launch");
      auto hr = dxr.get(); bool id0 = true, nrm = true, tail = true;
      for (int t = 0; t < T; ++t) for (int h = 0; h < heads; ++h) { size_t b = ((size_t)t * heads + h) * hd;
          for (int p = 0; p < 32; ++p) { double a0 = xr[b + p], b0 = xr[b + p + 32], a1 = hr[b + p], b1 = hr[b + p + 32];
              if (t == 0) id0 &= (a1 == xr[b + p] && b1 == xr[b + p + 32]);
              nrm &= std::fabs(a0 * a0 + b0 * b0 - a1 * a1 - b1 * b1) < 1e-4; }
          for (int d = 64; d < hd; ++d) tail &= hr[b + d] == xr[b + d]; }
      ok(id0, "rope: position 0 is the identity"); ok(nrm, "rope: rotation preserves each (p, p+32) pair norm"); ok(tail, "rope: dims >= 64 untouched");
      // kv_append: int8 groups of 64, scale = amax/127, |code| <= 127, zero group has scale 1
      auto k = vf((size_t)T * 256, -5, 5); for (int i = 64; i < 128; ++i) k[i] = 0.f;   // token 0, group 1 all zero
      Dev<float> dk(k); const int64_t base = 2; Dev<int8_t> kq((size_t)(base + T) * 256); Dev<float> ks((size_t)(base + T) * 4);
      ok(kv_append(dk, kq, ks, base, T) == 0 && sync_ok(), "kv_append launch");
      auto hq8 = kq.get(); auto hs = ks.get(); bool sc = true, bound = true, recon = true, untouched = true;
      for (int t = 0; t < T; ++t) for (int g = 0; g < 4; ++g) { float am = 0; for (int i = 0; i < 64; ++i) am = std::max(am, std::fabs(k[(size_t)t * 256 + g * 64 + i])); float want = am > 0 ? am / 127.f : 1.f;
          sc &= std::fabs(hs[(size_t)(base + t) * 4 + g] - want) <= 1e-6f * want;
          for (int i = 0; i < 64; ++i) { int c = hq8[(size_t)(base + t) * 256 + g * 64 + i]; bound &= c >= -127 && c <= 127; recon &= std::fabs(c * want - k[(size_t)t * 256 + g * 64 + i]) <= want * 0.5f + 1e-6f; } }
      for (size_t i = 0; i < (size_t)base * 256; ++i) untouched &= hq8[i] == 0;
      ok(sc, "kv_append: scale = amax/127 (1 for an all-zero group)"); ok(bound, "kv_append: codes within +-127"); ok(recon, "kv_append: reconstruction error <= scale/2"); ok(untouched, "kv_append: rows before base_row untouched");
    }

    // ---- gated delta net ----
    { auto a = vf((size_t)T * HV, -3, 3), b = vf((size_t)T * HV, -3, 3), al = vf(HV, -1, 1), db = vf(HV, -1, 1);
      Dev<float> da(a), dbb(b), dal(al), ddb(db), g((size_t)T * HV), be((size_t)T * HV);
      ok(gdn_gates(da, dbb, dal, ddb, g, be, T) == 0 && sync_ok(), "gdn_gates launch");
      auto hg = g.get(), hbe = be.get(); bool gok = true, bok = true;
      for (int i = 0; i < T * HV; ++i) { int h = i % HV; double z = a[i] + db[h], sp = z > 20 ? z : std::log1p(std::exp(z));
          gok &= std::fabs(hg[i] - (-std::exp((double)al[h]) * sp)) < 1e-5 + 1e-5 * std::fabs(hg[i]); bok &= std::fabs(hbe[i] - sg(b[i])) < 1e-6; gok &= hg[i] <= 0; }
      ok(gok, "gdn_gates: g = -exp(A_log)*softplus(a+dt_bias) <= 0"); ok(bok, "gdn_gates: beta = sigmoid(b)");
      // conv: matches the width-4 causal host reference and is chunk-split invariant (state carries)
      const int TT = 6; auto x = vf((size_t)TT * C, -2, 2), w = vf((size_t)C * 4, -1, 1), st0 = vf((size_t)C * 3, -1, 1);
      Dev<float> dx(x), dw(w), s1(st0), o1((size_t)TT * C), s2(st0), o2((size_t)TT * C);
      ok(gdn_conv(dx, dw, s1, o1, TT) == 0 && sync_ok(), "gdn_conv launch");
      ok(gdn_conv(dx, dw, s2, o2, 2) == 0 && gdn_conv((float*)dx + (size_t)2 * C, dw, s2, (float*)o2 + (size_t)2 * C, TT - 2) == 0 && sync_ok(), "gdn_conv chunked launch");
      auto h1 = o1.get(); std::vector<double> r((size_t)TT * C);
      for (int c = 0; c < C; ++c) { double s[3] = {st0[(size_t)c * 3], st0[(size_t)c * 3 + 1], st0[(size_t)c * 3 + 2]};
          for (int t = 0; t < TT; ++t) { double xv = x[(size_t)t * C + c]; double acc = w[(size_t)c * 4] * s[0] + w[(size_t)c * 4 + 1] * s[1] + w[(size_t)c * 4 + 2] * s[2] + w[(size_t)c * 4 + 3] * xv;
              r[(size_t)t * C + c] = acc * sg(acc); s[0] = s[1]; s[1] = s[2]; s[2] = xv; } }
      near(h1, r, 1e-4, 1e-5, "gdn_conv = causal depthwise width-4 conv + silu");
      ok(h1 == o2.get(), "gdn_conv: chunked == whole (conv state carries across calls)"); ok(s1.get() == s2.get(), "gdn_conv: final state equal after chunking");
      // l2: unit norm per q/k head, v untouched
      auto qkv = vf((size_t)T * C, -2, 2); Dev<float> dq(qkv); ok(gdn_l2(dq, T, 1e-12f) == 0 && sync_ok(), "gdn_l2 launch");
      auto hq = dq.get(); bool un = true, vt = true;
      for (int t = 0; t < T; ++t) { for (int hd2 = 0; hd2 < 2 * HK; ++hd2) { double s = 0; for (int i = 0; i < S; ++i) s += (double)hq[(size_t)t * C + hd2 * S + i] * hq[(size_t)t * C + hd2 * S + i]; un &= std::fabs(s - 1.0) < 1e-4; }
          for (int i = 2 * HK * S; i < C; ++i) vt &= hq[(size_t)t * C + i] == qkv[(size_t)t * C + i]; }
      ok(un, "gdn_l2: every q/k head has unit L2 norm"); ok(vt, "gdn_l2: v channels untouched");
      // recurrence: chunk-split invariance, finite, zero gate/beta -> state decays/zero output
      const int TR = 4; auto q2 = vf((size_t)TR * C, -1, 1); for (int t = 0; t < TR; ++t) for (int hd2 = 0; hd2 < 2 * HK; ++hd2) { double s = 0; for (int i = 0; i < S; ++i) s += (double)q2[(size_t)t * C + hd2 * S + i] * q2[(size_t)t * C + hd2 * S + i]; float inv = 1.f / std::sqrt((float)s); for (int i = 0; i < S; ++i) q2[(size_t)t * C + hd2 * S + i] *= inv; }
      auto gg = vf((size_t)TR * HV, -1, -0.01f), bb = vf((size_t)TR * HV, 0.1f, 0.9f);
      Dev<float> dq2(q2), dg(gg), dbt(bb), sa((size_t)HV * S * S), sb((size_t)HV * S * S), oa((size_t)TR * HV * S), ob((size_t)TR * HV * S);
      ok(gdn_recurrence(dq2, dg, dbt, sa, oa, TR) == 0 && sync_ok(), "gdn_recurrence launch");
      ok(gdn_recurrence(dq2, dg, dbt, sb, ob, 1) == 0 && gdn_recurrence((float*)dq2 + C, (float*)dg + HV, (float*)dbt + HV, sb, (float*)ob + (size_t)HV * S, TR - 1) == 0 && sync_ok(), "gdn_recurrence chunked launch");
      auto ho = oa.get(); bool fin = true; for (float v : ho) fin &= std::isfinite(v);
      ok(fin, "gdn_recurrence: finite outputs"); ok(ho == ob.get(), "gdn_recurrence: chunked == whole (state carries)"); ok(sa.get() == sb.get(), "gdn_recurrence: final state equal after chunking");
      bool nz = false; for (float v : ho) nz |= v != 0.f; ok(nz, "gdn_recurrence: outputs not identically zero");
      // layout: nonzero initial state in [row][head][col]; compare outputs and final state with a double-precision host reference
      { const int TL = 2; auto s0 = vf((size_t)S * HV * S, -0.5f, 0.5f); Dev<float> ds(s0), dol((size_t)TL * HV * S);
        ok(gdn_recurrence(dq2, dg, dbt, ds, dol, TL) == 0 && sync_ok(), "gdn_recurrence layout launch");
        std::vector<double> rs(s0.begin(), s0.end()), ro((size_t)TL * HV * S);
        auto at = [&](int row, int h, int col) -> size_t { return ((size_t)row * HV + h) * S + col; };
        for (int t = 0; t < TL; ++t) for (int h = 0; h < HV; ++h) { const int hk = h % HK; const float* base = &q2[(size_t)t * C];
            double gt = std::exp((double)gg[(size_t)t * HV + h]), bt = bb[(size_t)t * HV + h];
            for (int row = 0; row < S; ++row) { double kv = 0; for (int c = 0; c < S; ++c) { rs[at(row, h, c)] *= gt; kv += rs[at(row, h, c)] * base[HK * S + hk * S + c]; }
                double dl = bt * (base[2 * HK * S + h * S + row] - kv), o = 0;
                for (int c = 0; c < S; ++c) { rs[at(row, h, c)] += dl * base[HK * S + hk * S + c]; o += rs[at(row, h, c)] * base[hk * S + c]; }
                ro[((size_t)t * HV + h) * S + row] = o / std::sqrt((double)S); } }
        auto gsd = ds.get(), god = dol.get();
        near(god, ro, 1e-3, 1e-4, "gdn_recurrence: outputs match host reference");
        near(gsd, rs, 1e-3, 1e-4, "gdn_recurrence: state layout is [row][head][col]"); }
    }

    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT PF7B804CF95 TESTS FAILED\n"); return 1; }
    std::printf("UNIT PF7B804CF95 TESTS PASSED\n");
    return 0;
}
