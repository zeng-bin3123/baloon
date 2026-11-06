// Unit 19 invariants for port_scalar.hpp / port_kernels_ref.hpp (host reference math of the decode kernels).
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "baloon/gr_gated_residual.hpp"
#include "baloon/port_kernels_ref.hpp"
#include "baloon/port_scalar.hpp"
using namespace baloon::refk;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) <= tol * (1.f + std::fabs(b)); }
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float fl(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t rng = 12345;
static float rnd() { rng = rng * 1664525u + 1013904223u; return (float)((rng >> 8) & 0xFFFF) / 32768.f - 1.f; }

int main() {
    // ---- bf16: RNE on the 16 dropped bits
    CHECK(bf16_from_f32(1.0f) == 0x3F80);
    CHECK(bf16_from_f32(fl(0x3F808000u)) == 0x3F80);        // tie, even stays
    CHECK(bf16_from_f32(fl(0x3F818000u)) == 0x3F82);        // tie, odd rounds up to even
    CHECK(bf16_from_f32(fl(0x3F808001u)) == 0x3F81);        // above half rounds up
    CHECK(bf16_from_f32(fl(0x3F807FFFu)) == 0x3F80);        // below half rounds down
    CHECK(bf16_from_f32(-2.0f) == 0xC000);
    CHECK(bf16_from_f32(INFINITY) == 0x7F80 && bf16_from_f32(-INFINITY) == 0xFF80);
    { uint16_t n = bf16_from_f32(NAN); CHECK((n & 0x7F80) == 0x7F80 && (n & 0x7F) != 0); }
    { uint16_t n = bf16_from_f32(fl(0x7F800001u)); CHECK((n & 0x7F80) == 0x7F80 && (n & 0x7F) != 0); }   // signalling NaN must not collapse to inf
    { uint16_t n = bf16_from_f32(fl(0xFF800001u)); CHECK(n >> 15 == 1 && (n & 0x7F) != 0); }
    CHECK(bf16_from_f32(fl(0x7F7FFFFFu)) == 0x7F80);        // max finite fp32 overflows to inf
    for (uint32_t b = 0; b < 0x10000; b += 7) {             // exact round trip of every non-NaN bf16
        if ((b & 0x7F80) == 0x7F80 && (b & 0x7F)) continue;
        CHECK(bf16_from_f32(f32_from_bf16((uint16_t)b)) == b);
    }
    CHECK(f32_from_bf16(0x3F80) == 1.0f && bits(f32_from_bf16(0x8000)) == 0x80000000u);

    // ---- RoPE
    { float o0, o1; rope_neox_pair(3.f, 4.f, 0.6f, 0.8f, &o0, &o1);
      CHECK(near(o0, 3 * 0.6f - 4 * 0.8f) && near(o1, 3 * 0.8f + 4 * 0.6f) && near(o0 * o0 + o1 * o1, 25.f)); }
    { float cs[8], sn[8]; build_rope_table(cs, sn, 16, 0, 10000.f);
      for (int i = 0; i < 8; ++i) CHECK(cs[i] == 1.f && sn[i] == 0.f);
      build_rope_table(cs, sn, 16, 5, 10000.f);
      for (int i = 0; i < 8; ++i) CHECK(near(cs[i] * cs[i] + sn[i] * sn[i], 1.f));
      CHECK(near(cs[0], std::cos(5.0f)) && near(sn[0], std::sin(5.0f)));
      CHECK(near(sn[1], (float)std::sin(5.0 * std::pow(10000.0, -2.0 / 16.0))));
      float x[20]; for (int i = 0; i < 20; ++i) x[i] = rnd();
      float y[20]; std::memcpy(y, x, sizeof x);
      rope_neox_apply(y, 16, cs, sn);
      float nx = 0, ny = 0; for (int i = 0; i < 16; ++i) { nx += x[i] * x[i]; ny += y[i] * y[i]; }
      CHECK(near(nx, ny));
      for (int i = 16; i < 20; ++i) CHECK(x[i] == y[i]);        // dims past n_rot pass through
      float h[40]; for (int i = 0; i < 20; ++i) h[i] = h[20 + i] = x[i];
      native_rope_apply(h, 2, 20, 16, cs, sn);
      for (int i = 0; i < 20; ++i) CHECK(h[i] == y[i] && h[20 + i] == y[i]); }

    // ---- M-RoPE
    { const int sec[3] = {3, 2, 3}; const int p3[3] = {10, 20, 30};
      CHECK(mrope_pos(p3, sec, 0) == 10 && mrope_pos(p3, sec, 2) == 10 && mrope_pos(p3, sec, 3) == 20 && mrope_pos(p3, sec, 4) == 20 &&
            mrope_pos(p3, sec, 5) == 30 && mrope_pos(p3, sec, 7) == 30);
      const int bad[3] = {1, -1, 2}; CHECK(!mrope_table_set(bad));
      CHECK(mrope_table_set(sec));
      float c3[8], s3[8], c1[8], s1[8];
      const int same[3] = {7, 7, 7};
      mrope_table(c3, s3, 16, same, 10000.f); build_rope_table(c1, s1, 16, 7, 10000.f);
      for (int i = 0; i < 8; ++i) CHECK(near(c3[i], c1[i]) && near(s3[i], s1[i]));
      mrope_table(c3, s3, 16, p3, 10000.f);
      float cA[8], sA[8], cB[8], sB[8], cC[8], sC[8];
      build_rope_table(cA, sA, 16, 10, 10000.f); build_rope_table(cB, sB, 16, 20, 10000.f); build_rope_table(cC, sC, 16, 30, 10000.f);
      for (int i = 0; i < 8; ++i) { const float* c = i < 3 ? cA : i < 5 ? cB : cC; CHECK(near(c3[i], c[i])); } }

    // ---- KV-Q8 cell sizing
    CHECK(KV_Q8_GROUP == 64 && kv_q8_bytes_per_cell(8, 128) == 8 * 128 + 8 * 2 * 2);
    CHECK(kv_q8_bytes_per_cell(8, 100) == 0 && kv_q8_bytes_per_cell(0, 128) == 0);

    // ---- switches
    CHECK(native_gdn_enabled() && native_moe_combine_enabled() && native_qsa_enabled() && native_qsa_indexer_enabled() &&
          native_qsa_score_enabled() && native_rope_enabled() && native_router_enabled());
    native_rope_set_enabled(false); CHECK(!native_rope_enabled() && native_router_enabled()); native_rope_set_enabled(true);
    native_router_set_enabled(false); CHECK(!native_router_enabled() && native_rope_enabled()); native_router_set_enabled(true);
    native_gdn_set_enabled(false); native_qsa_set_enabled(false); native_qsa_indexer_set_enabled(false);
    native_qsa_score_set_enabled(false); native_moe_combine_set_enabled(false);
    CHECK(!native_gdn_enabled() && !native_qsa_enabled() && !native_qsa_indexer_enabled() && !native_qsa_score_enabled() && !native_moe_combine_enabled());
    native_gdn_set_enabled(true); native_qsa_set_enabled(true); native_qsa_indexer_set_enabled(true);
    native_qsa_score_set_enabled(true); native_moe_combine_set_enabled(true);

    // ---- bf16 GEMV
    { const int ni = 130, no = 5; std::vector<uint16_t> w(ni * no); std::vector<float> x(ni), y(no), y1(no), ys(no), wf(ni * no);
      for (int i = 0; i < ni * no; ++i) { w[i] = bf16_from_f32(rnd()); wf[i] = f32_from_bf16(w[i]); }
      for (auto& v : x) v = rnd();
      bf16_gemv_fp32_mmvf(w.data(), x.data(), y.data(), ni, no);
      for (int r = 0; r < no; ++r) { float a = 0; for (int i = 0; i < ni; ++i) a += wf[r * ni + i] * x[i]; CHECK(y[r] == a); }
      bf16_gemv_split(w.data(), x.data(), y1.data(), ni, no, 1);
      for (int r = 0; r < no; ++r) CHECK(y1[r] == y[r]);
      for (int s : {2, 3, 7, 200}) { bf16_gemv_split(w.data(), x.data(), ys.data(), ni, no, s); for (int r = 0; r < no; ++r) CHECK(near(ys[r], y[r], 1e-4f)); } }

    // ---- S2 GEMV
    { const int ni = 192, no = 4, gs = ni / 64; std::vector<uint8_t> codes(no * ni / 4); std::vector<float> sc(no * gs), x(ni), y(no), yf(no), yq(no), xd(ni);
      for (auto& c : codes) { rng = rng * 1664525u + 1013904223u; c = (uint8_t)(rng >> 24); }
      for (auto& s : sc) s = 0.5f + std::fabs(rnd());
      std::vector<int8_t> xq(ni); std::vector<float> xs(gs);
      for (int g = 0; g < gs; ++g) { xs[g] = 0.25f * (g + 1); for (int e = g * 64; e < (g + 1) * 64; ++e) { xq[e] = (int8_t)(rnd() * 100); x[e] = xq[e] * xs[g]; } }
      Codebook cb;
      s2_gemv(codes.data(), sc.data(), cb, x.data(), y.data(), ni, no);
      s2_gemv_fast(codes.data(), sc.data(), cb, x.data(), yf.data(), ni, no);
      for (int r = 0; r < no; ++r) CHECK(y[r] == yf[r]);
      s2_gemv_quads(codes.data(), sc.data(), cb, x.data(), yq.data(), ni, no);
      for (int r = 0; r < no; ++r) CHECK(near(y[r], yq[r], 1e-3f));
      s2_gemv_q8(codes.data(), sc.data(), cb, xq.data(), xs.data(), yq.data(), ni, no);
      for (int r = 0; r < no; ++r) CHECK(near(y[r], yq[r], 1e-3f));
      std::vector<float> row(ni);
      for (int r = 0; r < no; ++r) {
          dequant_s2(codes.data() + r * ni / 4, sc.data() + r * gs, cb, row.data(), ni);
          double a = 0; for (int e = 0; e < ni; ++e) a += (double)row[e] * x[e];
          CHECK(near((float)a, y[r], 1e-3f));
          for (int e = 0; e < ni; ++e) CHECK(row[e] == cb.levels[(codes[r * ni / 4 + e / 4] >> ((e % 4) * 2)) & 3] * sc[r * gs + e / 64]);
      } }

    // ---- GDN
    { float hist[3] = {1, 2, 3}, w[4] = {1, 10, 100, 1000}, x = 4, out;
      gdn_conv_step(hist, &x, w, &out, 1, 4);
      CHECK(out == 1 * 1 + 10 * 2 + 100 * 3 + 1000 * 4 && hist[0] == 2 && hist[1] == 3 && hist[2] == 4);
      float h2[3] = {1, 2, 3}; native_gdn_conv_silu(h2, &x, w, &out, 1, 4); CHECK(near(out, silu(4321.f))); }
    { float a[3] = {0.f, 1.f, -2.f}, b[3] = {0.f, 3.f, -3.f}, dt[3] = {0.1f, 0.f, 0.5f}, A[3] = {0.f, 0.5f, -1.f}, g[3], be[3], g2[3], be2[3];
      native_gdn_beta_gate(b, a, dt, A, be, g, 3); fused_gdn_ab(b, a, dt, A, be2, g2, 3);
      for (int i = 0; i < 3; ++i) { CHECK(g[i] < 0.f && be[i] > 0.f && be[i] < 1.f && g[i] == g2[i] && be[i] == be2[i]); }
      CHECK(near(be[0], 0.5f)); CHECK(near(g[0], -std::log1p(std::exp(0.1f))));
      float g3[3]; native_gdn_gate(a, dt, A, g3, 3); for (int i = 0; i < 3; ++i) CHECK(g3[i] == g[i]); }
    { float v[2 * 8]; for (auto& e : v) e = rnd() * 5; native_gdn_l2_norm(v, 2, 8, 1e-12f);
      for (int h = 0; h < 2; ++h) { float s = 0; for (int i = 0; i < 8; ++i) s += v[h * 8 + i] * v[h * 8 + i]; CHECK(near(s, 1.f, 1e-4f)); } }
    { float st[4] = {1, 2, 3, 4}, q[2] = {1, 0}, k[2] = {0, 1}, v[2] = {5, 6}, o[2];   // dk=2, dv=2, beta=0: only decay
      gdn_step_host(st, q, k, v, 0.f, 0.f, 1.f, o, 2, 2);
      CHECK(st[0] == 1 && st[3] == 4 && o[0] == 1 && o[1] == 2);
      float st2[4] = {0, 0, 0, 0};                                                      // empty state, beta=1: stores v at key k
      gdn_step_host(st2, k, k, v, -1.f, 1.f, 1.f, o, 2, 2);
      CHECK(near(st2[2], 5.f) && near(st2[3], 6.f) && near(o[0], 5.f) && near(o[1], 6.f)); }
    { GdnShapes s; s.hk = 2; s.hv = 4; s.dk = 4; s.dv = 4; CHECK(s.ok() && s.conv_channels() == 2 * 2 * 4 + 4 * 4);
      GdnShapes bad = s; bad.hv = 3; CHECK(!bad.ok());
      const int C = s.conv_channels(); std::vector<float> hist(C * 3), h2, x(C), w(C * 4), o1(C), o2(C);
      for (auto& e : hist) e = rnd(); h2 = hist; for (auto& e : x) e = rnd(); for (auto& e : w) e = rnd();
      fused_gdn_conv_l2(hist.data(), x.data(), w.data(), o1.data(), s, 1e-6f);
      native_gdn_conv_silu(h2.data(), x.data(), w.data(), o2.data(), C, 4);
      native_gdn_l2_norm(o2.data(), 2, 4, 1e-6f); native_gdn_l2_norm(o2.data() + 8, 2, 4, 1e-6f);
      for (int i = 0; i < C; ++i) CHECK(o1[i] == o2[i]);
      for (int i = 0; i < C * 3; ++i) CHECK(hist[i] == h2[i]);
      std::vector<float> st((size_t)4 * 4 * 4), st2; for (auto& e : st) e = rnd(); st2 = st;
      float g[4], be[4], z[16], nw[4], out[16], raw[16]; for (auto& e : g) e = -0.1f; for (auto& e : be) e = 0.7f; for (auto& e : z) e = rnd(); for (auto& e : nw) e = 1.f + rnd() * 0.1f;
      fused_gdn_step_norm(st.data(), o1.data(), g, be, z, nw, out, s, 1e-6f);
      const float* qq = o1.data(); const float* kk = qq + 8; const float* vv = kk + 8;
      for (int h = 0; h < 4; ++h) gdn_step_host(st2.data() + h * 16, qq + (h % 2) * 4, kk + (h % 2) * 4, vv + h * 4, g[h], be[h], 0.5f, raw + h * 4, 4, 4);
      float ref[16]; native_gdn_out_norm(raw, z, nw, ref, 4, 4, 1e-6f);
      for (int i = 0; i < 16; ++i) CHECK(out[i] == ref[i]);
      for (size_t i = 0; i < st.size(); ++i) CHECK(st[i] == st2[i]); }
    { float o[4] = {3, 4, 0, 0}, z[4] = {0, 0, 0, 0}, w[4] = {1, 1, 1, 1}, out[4];    // silu(0) = 0 -> output 0
      native_gdn_out_norm(o, z, w, out, 1, 4, 1e-6f); for (float e : out) CHECK(e == 0.f);
      float z2[4] = {20, 20, 20, 20}; native_gdn_out_norm(o, z2, w, out, 1, 4, 0.f);
      CHECK(near(out[0], 3.f / std::sqrt(25.f / 4.f) * silu(20.f)) && near(out[1], 4.f / std::sqrt(25.f / 4.f) * silu(20.f))); }

    // ---- gated residual
    { const int n = 6, hc = 3; std::vector<float> R(n * hc), blk(n), w(hc), o1(n * hc), o2(n * hc);
      for (auto& e : R) e = rnd(); for (auto& e : blk) e = rnd(); for (auto& e : w) e = rnd();
      native_gr_post(R.data(), blk.data(), w.data(), o1.data(), n, hc); baloon::gr::write(R.data(), blk.data(), w.data(), o2.data(), n, hc);
      for (int i = 0; i < n * hc; ++i) CHECK(o1[i] == o2[i]);
      float lg[3] = {0, 0, 0}, mix[6]; native_gr_pre_gated(R.data(), lg, mix, n, hc);
      for (int d = 0; d < n; ++d) CHECK(near(mix[d], 0.5f * (R[d] + R[n + d] + R[2 * n + d])));
      float big[3] = {50, -50, -50}; native_gr_pre_gated(R.data(), big, mix, n, hc); for (int d = 0; d < n; ++d) CHECK(near(mix[d], R[d], 1e-4f));
      float xin[4] = {3, 4, 0, 0}, nw[4] = {2, 2, 2, 2}, no[4]; native_gr_rms_norm_weighted(xin, nw, no, 4, 0.f);
      CHECK(near(no[0], 2 * 3 / 2.5f) && near(no[1], 2 * 4 / 2.5f));
      float gt[3] = {-100, 0, 100}, up[3] = {1, 2, 3}, dn[3]; native_gr_down_silu(gt, up, dn, 3);
      CHECK(near(dn[0], 0.f, 1e-3f) && dn[1] == 0.f && near(dn[2], 300.f)); }
    { const int T = 3, n = 8, hc = 4; std::vector<float> R(T * hc * n), lg(T * hc), nw(n), out(T * n, -7.f), one(n);
      for (auto& e : R) e = rnd(); for (auto& e : lg) e = rnd(); for (auto& e : nw) e = 1.f + rnd() * 0.2f;
      FusedGrArgs a; a.R = R.data(); a.logits = lg.data(); a.norm_w = nw.data(); a.out = out.data(); a.n = n; a.hc = hc; a.T = T;
      CHECK(kFusedGrMaxT == 8 && fused_gr_supported(8, n, hc) && !fused_gr_supported(9, n, hc) && !fused_gr_supported(0, n, hc));
      CHECK(fused_gr_read_multi(a));
      for (int t = 0; t < T; ++t) { float mix[n]; native_gr_pre_gated(R.data() + t * hc * n, lg.data() + t * hc, mix, n, hc);
        native_gr_rms_norm_weighted(mix, nw.data(), one.data(), n, a.eps); for (int i = 0; i < n; ++i) CHECK(out[t * n + i] == one[i]); }
      std::vector<float> o2(T * n, -7.f); a.out = o2.data(); a.T = 9; CHECK(!fused_gr_read_multi(a));
      for (float e : o2) CHECK(e == -7.f); }
    { GrShapes s; s.n = 100; s.hc = 4; GrShapes z; CHECK(s.ok() && !z.ok() && gr_workspace_bytes(z) == 0);
      const uint64_t b = gr_workspace_bytes(s); CHECK(b > 0 && b % 16 == 0 && b >= 4ull * (2 * 100 + 2 * 4 + 400));
      gr_set_fp32_activations(true); gr_set_native_mmvf(false); gr_set_fp32_activations(false); gr_set_native_mmvf(true); }

    // ---- short flash step
    { const int H = 4, KV = 2, D = 4, N = 5; std::vector<float> q(H * D), k(N * KV * D, 0.f), v(N * KV * D), o(H * D);
      for (auto& e : q) e = rnd(); for (auto& e : v) e = rnd();
      CHECK(native_flash_attn_short_step(q.data(), k.data(), v.data(), o.data(), H, KV, D, N, 0.5f) == NativeFlashAttnStatus::Ok);
      for (int h = 0; h < H; ++h) for (int i = 0; i < D; ++i) { float m = 0; for (int c = 0; c < N; ++c) m += v[(c * KV + h / 2) * D + i]; CHECK(near(o[h * D + i], m / N)); }
      CHECK(native_flash_attn_short_step(q.data(), k.data(), v.data(), o.data(), 3, 2, D, N, 1.f) == NativeFlashAttnStatus::BadShape);
      CHECK(native_flash_attn_short_step(q.data(), k.data(), v.data(), o.data(), H, KV, D, 0, 1.f) == NativeFlashAttnStatus::BadShape);
      CHECK(native_flash_attn_short_step(q.data(), k.data(), v.data(), o.data(), H, KV, D, kFlashShortMaxCells + 1, 1.f) == NativeFlashAttnStatus::TooLong);
      CHECK(kFlashShortMaxCells == 256);   // pinned literally to the theory bound, independent of the constant's definition
      { const int NB = 256, NT = 257; std::vector<float> kb((size_t)NT * KV * D, 0.f), vb((size_t)NT * KV * D, 1.f);
        CHECK(native_flash_attn_short_step(q.data(), kb.data(), vb.data(), o.data(), H, KV, D, 256, 1.f) == NativeFlashAttnStatus::Ok);
        CHECK(native_flash_attn_short_step(q.data(), kb.data(), vb.data(), o.data(), H, KV, D, 257, 1.f) == NativeFlashAttnStatus::TooLong);
        (void)NB; (void)NT; }
      std::vector<float> k2(N * KV * D, 0.f); for (int i = 0; i < D; ++i) k2[(2 * KV + 0) * D + i] = 100.f; float q1[16] = {0}; for (int h = 0; h < H; ++h) q1[h * D] = 1.f;
      k2[(2 * KV + 0) * D] = 100.f;
      native_flash_attn_short_step(q1, k2.data(), v.data(), o.data(), H, KV, D, N, 1.f);       // cell 2 dominates for heads 0,1 (kv head 0)
      for (int i = 0; i < D; ++i) CHECK(near(o[i], v[(2 * KV + 0) * D + i], 1e-3f)); }

    // ---- QSA select, router
    { const int d = 2, nc = 7, bc = 3; float kidx[nc * d], q[2] = {1, 0}, sc[3];
      for (int c = 0; c < nc; ++c) { kidx[c * d] = (float)c; kidx[c * d + 1] = 9.f; }
      qsa_block_scores(q, kidx, nc, bc, d, sc); CHECK(sc[0] == 2 && sc[1] == 5 && sc[2] == 6);
      auto t = qsa_block_topk(sc, 3, 2); CHECK(t.size() == 2 && t[0] == 2 && t[1] == 1);
      float tie[4] = {1, 1, 1, 0}; auto u = qsa_block_topk(tie, 4, 3); CHECK(u[0] == 0 && u[1] == 1 && u[2] == 2);
      CHECK(qsa_block_topk(tie, 4, 99).size() == 4 && qsa_block_topk(tie, 4, 0).empty());
      float ga[3] = {2, 2, 2}, gt[3] = {-100, 0, 100}; native_qsa_gate_apply(ga, gt, 3); CHECK(near(ga[0], 0.f, 1e-3f) && near(ga[1], 1.f) && near(ga[2], 2.f));
      float x[4] = {3, 4, 3, 4}, w[2] = {1, 1}, o[4]; native_qsa_rms_norm_weighted(x, w, o, 2, 2, 0.f); CHECK(near(o[0], o[2]) && near(o[0], 3.f / std::sqrt(12.5f))); }
    { const int E = 64; std::vector<float> lg(E); for (auto& e : lg) e = rnd() * 3; lg[17] = lg[40] = 5.f; int idx[10]; float w[10];
      CHECK(native_router_top10(lg.data(), E, idx, w) == 10);
      float s = 0; for (int j = 0; j < 10; ++j) { s += w[j]; CHECK(w[j] > 0); for (int i = j + 1; i < 10; ++i) CHECK(idx[i] != idx[j]); if (j) CHECK(w[j] <= w[j - 1] + 1e-7f); }
      CHECK(near(s, 1.f, 1e-4f) && idx[0] == 17 && idx[1] == 40);                             // tie -> lower index first
      for (int j = 0; j < 10; ++j) for (int e = 0; e < E; ++e) { bool in = false; for (int i = 0; i < 10; ++i) in |= idx[i] == e; if (!in) CHECK(lg[e] <= lg[idx[j]] + 1e-6f); }
      CHECK(native_router_top10(lg.data(), 4, idx, w) == 4 && native_router_top10(lg.data(), 0, idx, w) == 0); }

    // ---- MoE combine, shared expert, PLE
    { float r[6] = {1, 2, 3, 4, 5, 6}, w[2] = {0.25f, 0.75f}, sh[3] = {10, 20, 30}, o[3];
      native_moe_combine(r, w, sh, o, 2, 3); CHECK(near(o[0], 10 + 0.25f + 3.f) && near(o[2], 30 + 0.75f + 4.5f));
      native_moe_combine(r, w, nullptr, o, 2, 3); CHECK(near(o[1], 0.5f + 3.75f)); }
    { const int E = 4, F = 3, T = 3; std::vector<float> g(F * E), u(F * E), dn(E * F), gv(E), x(T * E), o1(T * E), o2(T * E), sc(shared_expert_scratch_bytes(F, T) / 4);
      for (auto& e : g) e = rnd(); for (auto& e : u) e = rnd(); for (auto& e : dn) e = rnd(); for (auto& e : gv) e = rnd(); for (auto& e : x) e = rnd();
      NativeSharedWeights w; w.gate = g.data(); w.up = u.data(); w.down = dn.data(); w.n_embd = E; w.n_ff = F;
      CHECK(shared_expert_scratch_bytes(F, T) == 2ull * 4 * F * T && shared_expert_scratch_bytes(0, T) == 0);
      shared_expert_multi(w, x.data(), o1.data(), sc.data(), T);
      for (int t = 0; t < T; ++t) { shared_expert(w, x.data() + t * E, o2.data() + t * E, sc.data()); for (int i = 0; i < E; ++i) CHECK(o1[t * E + i] == o2[t * E + i]); }
      float zero[4] = {0, 0, 0, 0}, oz[4]; shared_expert(w, zero, oz, sc.data()); for (float e : oz) CHECK(e == 0.f);
      w.gate_vec = gv.data(); std::vector<float> o3(E); shared_expert(w, x.data(), o3.data(), sc.data());
      float s = 0; for (int i = 0; i < E; ++i) s += gv[i] * x[i];
      for (int i = 0; i < E; ++i) CHECK(near(o3[i], o2[i] * sigmoid(s)));
      shared_expert_set_native_bf16(true); shared_expert_set_native_bf16(false); }
    { float gate[4] = {0, 0, 0, 0}, ple[4] = {1, 2, 3, 4}, tmp[4], hid[4] = {1, -1, 2, -2}, pw[4] = {1, 1, 1, 1}, out[4];
      NativePlePostopsBuffers b; b.gate = gate; b.ple = ple; b.tmp = tmp; b.n = 4;
      native_ple_postops(b, hid, pw, 1e-6f, out); for (int i = 0; i < 4; ++i) CHECK(out[i] == hid[i]);   // gelu(0) = 0 -> post branch vanishes
      float g2[4] = {1, 1, 1, 1}; b.gate = g2; native_ple_postops(b, hid, pw, 1e-6f, out);
      float sq = 0; for (int i = 0; i < 4; ++i) { float t = gelu_tanh(1.f) * ple[i]; sq += t * t; }
      for (int i = 0; i < 4; ++i) CHECK(near(out[i], hid[i] + gelu_tanh(1.f) * ple[i] / std::sqrt(sq / 4 + 1e-6f))); }

    // ---- IQ4 decode, MMVQ, Q8_1, expert layout
    CHECK(iq_supported(20) && iq_supported(23) && !iq_supported(2));
    CHECK(iq_row_bytes(20, 64) == 36 && iq_row_bytes(23, 512) == 272 && iq_row_bytes(23, 48) == 0 && iq_row_bytes(20, 0) == 0 && iq_row_bytes(8, 64) == 0);
    { uint8_t blk[18]; const uint16_t d = baloon::f16::float_to_half(0.5f); std::memcpy(blk, &d, 2);
      for (int j = 0; j < 16; ++j) blk[2 + j] = (uint8_t)(j | ((15 - j) << 4));
      float o[32]; iq_dequant_f32(20, blk, o, 32);
      for (int j = 0; j < 16; ++j) CHECK(o[j] == 0.5f * kIq4Values[j] && o[16 + j] == 0.5f * kIq4Values[15 - j]);
      uint16_t oh[32]; iq_dequant_f16(20, blk, oh, 32); for (int j = 0; j < 32; ++j) CHECK(oh[j] == baloon::f16::float_to_half(o[j]));
      CHECK(kIq4Values[0] == -127 && kIq4Values[7] == -10 && kIq4Values[8] == 1 && kIq4Values[15] == 113); }
    { uint8_t blk[136]; std::memset(blk, 0, sizeof blk); const uint16_t d = baloon::f16::float_to_half(2.0f); std::memcpy(blk, &d, 2);
      uint16_t sh = 0; sh |= (2u << 2 * 3); std::memcpy(blk + 2, &sh, 2);           // sub-block 3: high 2 bits of scale = 2
      blk[4 + 1] = 0x50;                                                             // sub-block 3 (odd): low nibble in high half of byte 1 -> 5
      for (int j = 0; j < 16; ++j) blk[8 + 3 * 16 + j] = (uint8_t)(8 | (9 << 4));    // codes 8 / 9
      float o[256]; iq_dequant_f32(23, blk, o, 256);
      const float dl3 = 2.0f * (float)((5 | (2 << 4)) - 32);                         // ls = 37
      for (int j = 0; j < 16; ++j) CHECK(o[3 * 32 + j] == dl3 * 1 && o[3 * 32 + 16 + j] == dl3 * 13);
      for (int j = 0; j < 32; ++j) CHECK(o[j] == 2.0f * (0 - 32) * kIq4Values[0]);   // ls=0 -> d*(-32)*value[0]
    }
    { const int ni = 64, no = 3; std::vector<uint8_t> w(no * iq_row_bytes(20, ni)); std::vector<float> x(ni), y(no), row(ni);
      for (int r = 0; r < no; ++r) { uint8_t* b = w.data() + r * 36; for (int k = 0; k < 2; ++k) { uint16_t d = baloon::f16::float_to_half(0.01f * (r + 1 + k)); std::memcpy(b + k * 18, &d, 2); for (int j = 0; j < 16; ++j) b[k * 18 + 2 + j] = (uint8_t)(rnd() * 127 + 128); } }
      for (auto& e : x) e = rnd();
      CHECK(iq_mmvq(20, w.data(), x.data(), y.data(), ni, no) && !iq_mmvq(20, w.data(), x.data(), y.data(), 48, no));
      for (int r = 0; r < no; ++r) { iq_dequant_f32(20, w.data() + r * 36, row.data(), ni); float a = 0; for (int i = 0; i < ni; ++i) a += row[i] * x[i]; CHECK(near(y[r], a)); }
      int32_t ids[3] = {2, -1, 3}; std::vector<float> em(3 * ni, 9.f); iq_embed_rows(20, w.data(), no, ids, 3, ni, em.data());
      iq_dequant_f32(20, w.data() + 2 * 36, row.data(), ni); for (int i = 0; i < ni; ++i) CHECK(em[i] == row[i] && em[ni + i] == 0.f && em[2 * ni + i] == 0.f);
      std::vector<uint8_t> gu(2 * 2 * 36, 0); std::memcpy(gu.data(), w.data(), 3 * 36); std::vector<uint16_t> gh(2 * ni), uh(2 * ni);
      iq_dequant_gu_f16(20, w.data(), gh.data(), uh.data(), 1, ni);                 // one ff row: gate = row0, up = row1
      iq_dequant_f32(20, w.data(), row.data(), ni); for (int i = 0; i < ni; ++i) CHECK(gh[i] == baloon::f16::float_to_half(row[i]));
      iq_dequant_f32(20, w.data() + 36, row.data(), ni); for (int i = 0; i < ni; ++i) CHECK(uh[i] == baloon::f16::float_to_half(row[i])); }
    { float x[32]; for (int i = 0; i < 32; ++i) x[i] = 0.f; x[0] = 127.f; x[1] = 0.5f; x[2] = -0.5f; x[3] = 1.4f; x[4] = -1.6f;
      uint8_t blk[36]; quantize_q8_1_rows(x, 1, 32, blk); uint16_t dh, sh; std::memcpy(&dh, blk, 2); std::memcpy(&sh, blk + 2, 2); const int8_t* q = (const int8_t*)(blk + 4);
      CHECK(baloon::f16::half_to_float(dh) == 1.f && q[0] == 127 && q[1] == 1 && q[2] == -1 && q[3] == 1 && q[4] == -2);   // ties away from zero
      CHECK(baloon::f16::half_to_float(sh) == 127.f + 1 - 1 + 1 - 2);
      std::vector<float> xr(2 * 64); for (auto& e : xr) e = rnd() * 3; std::vector<uint8_t> qb(2 * 2 * 36); quantize_q8_1_rows(xr.data(), 2, 64, qb.data());
      for (int r = 0; r < 2; ++r) for (int b = 0; b < 2; ++b) { const uint8_t* p = qb.data() + (r * 2 + b) * 36; uint16_t h; std::memcpy(&h, p, 2); float d = baloon::f16::half_to_float(h);
        for (int i = 0; i < 32; ++i) CHECK(std::fabs(((const int8_t*)(p + 4))[i] * d - xr[r * 64 + b * 32 + i]) <= d * 0.6f); }
      uint8_t zb[36]; float zx[32] = {0}; quantize_q8_1_rows(zx, 1, 32, zb); for (int i = 0; i < 36; ++i) CHECK(zb[i] == 0); }
    { const NativeExpertLayout l = native_expert_layout(20, 64, 32); CHECK(l.ok && l.gu_row_bytes == 36 && l.down_row_bytes == 18 && l.up_off == 32 * 36 && l.down_off == 2 * 32 * 36 && l.bytes == 2 * 32 * 36 + 64 * 18);
      CHECK(!native_expert_layout(20, 48, 32).ok && !native_expert_layout(8, 64, 32).ok && native_expert_scratch_bytes(64, 32) == 4ull * (64 + 64 + 64));
      std::vector<uint8_t> blob(l.bytes), blob2(l.bytes); for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t)(rnd() * 100 + 128);
      auto setd = [&](std::vector<uint8_t>& b, int64_t off, int64_t rb, int rows) { for (int r = 0; r < rows; ++r) for (int k = 0; k < rb / 18; ++k) { uint16_t d = baloon::f16::float_to_half(0.002f); std::memcpy(&b[off + r * rb + k * 18], &d, 2); } };
      setd(blob, 0, 36, 64); setd(blob, l.down_off, 18, 64); blob2 = blob;
      std::vector<float> x(64), o1(64), o2(64), oa(64); for (auto& e : x) e = rnd();
      const uint8_t* e1[1] = {blob.data()}; const uint8_t* e2[2] = {blob.data(), blob2.data()}; float w1[1] = {1.f}, w2[2] = {0.25f, 0.75f};
      CHECK(native_expert_grouped(20, e1, w1, 1, x.data(), o1.data(), 64, 32) && !native_expert_grouped(8, e1, w1, 1, x.data(), o1.data(), 64, 32));
      CHECK(native_expert_grouped(20, e2, w2, 2, x.data(), o2.data(), 64, 32));            // identical experts, weights sum to 1 -> same output
      for (int i = 0; i < 64; ++i) CHECK(near(o1[i], o2[i], 1e-4f));
      float w0[1] = {0.f}; native_expert_grouped(20, e1, w0, 1, x.data(), oa.data(), 64, 32); for (float e : oa) CHECK(e == 0.f);
      std::vector<float> h(32), row(64), manual(64, 0.f);
      for (int f = 0; f < 32; ++f) { float g = 0, u = 0; iq_dequant_f32(20, blob.data() + (2 * f) * 36, row.data(), 64); for (int i = 0; i < 64; ++i) g += row[i] * x[i];
        iq_dequant_f32(20, blob.data() + (2 * f + 1) * 36, row.data(), 64); for (int i = 0; i < 64; ++i) u += row[i] * x[i]; h[f] = silu(g) * u; }
      for (int d = 0; d < 64; ++d) { std::vector<float> r2(32); iq_dequant_f32(20, blob.data() + l.down_off + d * 18, r2.data(), 32); for (int f = 0; f < 32; ++f) manual[d] += r2[f] * h[f]; }
      for (int d = 0; d < 64; ++d) CHECK(near(o1[d], manual[d], 1e-4f)); }

    // ---- plain-named entry points equal their native_* forms; indexer append
    { const int ni = 8, no = 2; uint16_t w[16]; float x[8], y1[2], y2[2];
      for (auto& e : w) e = bf16_from_f32(rnd()); for (auto& e : x) e = rnd();
      bf16_gemv(w, x, y1, ni, no); bf16_gemv_fp32_mmvf(w, x, y2, ni, no); CHECK(y1[0] == y2[0] && y1[1] == y2[1]);
      float b[2] = {0.3f, -0.2f}, a[2] = {0.1f, 0.4f}, dt[2] = {0.f, 0.1f}, A[2] = {0.f, 0.2f}, be1[2], g1[2], be2[2], g2[2];
      gdn_beta_gate(b, a, dt, A, be1, g1, 2); native_gdn_beta_gate(b, a, dt, A, be2, g2, 2);
      CHECK(be1[0] == be2[0] && g1[1] == g2[1]);
      float st1[4] = {1, 2, 3, 4}, st2[4] = {1, 2, 3, 4}, st3[4] = {1, 2, 3, 4}, q[2] = {1, 2}, k[2] = {0.5f, -1}, v[2] = {3, 4}, o1[2], o2[2], o3[2];
      gdn_step(st1, q, k, v, -0.3f, 0.6f, 0.7f, o1, 2, 2); native_gdn_step(st2, q, k, v, -0.3f, 0.6f, 0.7f, o2, 2, 2); gdn_step_host(st3, q, k, v, -0.3f, 0.6f, 0.7f, o3, 2, 2);
      for (int i = 0; i < 4; ++i) CHECK(st1[i] == st3[i] && st2[i] == st3[i]);
      CHECK(o1[0] == o3[0] && o2[1] == o3[1]);
      float vv[8], vw[8], z[8], nw[8] = {1, 1, 1, 1, 1, 1, 1, 1}, r1[8], r2[8]; for (auto& e : vv) e = rnd(); for (auto& e : z) e = rnd(); for (int i = 0; i < 8; ++i) vw[i] = vv[i];
      gdn_l2_norm(vv, 2, 4, 1e-6f); native_gdn_l2_norm(vw, 2, 4, 1e-6f); for (int i = 0; i < 8; ++i) CHECK(vv[i] == vw[i]);
      gdn_out_norm(vv, z, nw, r1, 2, 4, 1e-6f); native_gdn_out_norm(vv, z, nw, r2, 2, 4, 1e-6f); for (int i = 0; i < 8; ++i) CHECK(r1[i] == r2[i]);
      const int n = 4, hc = 2; float R[8], lg[2] = {0.2f, -0.5f}, nrw[4] = {1, 1, 1, 1}, out1[4], mix[4], out2[4]; for (auto& e : R) e = rnd();
      gr_read(R, lg, nrw, out1, n, hc, 1e-6f); native_gr_pre_gated(R, lg, mix, n, hc); native_gr_rms_norm_weighted(mix, nrw, out2, n, 1e-6f);
      for (int i = 0; i < n; ++i) CHECK(out1[i] == out2[i]);
      float blk[4], ww[2] = {0.5f, 2.f}, wo1[8], wo2[8]; for (auto& e : blk) e = rnd();
      gr_write(R, blk, ww, wo1, n, hc); baloon::gr::write(R, blk, ww, wo2, n, hc); for (int i = 0; i < 8; ++i) CHECK(wo1[i] == wo2[i]);
      GrWorkspace ws; GrShapes bad; GrShapes ok; ok.n = 10; ok.hc = 2;
      CHECK(!gr_workspace_init(ws, bad) && ws.empty() && gr_workspace_init(ws, ok) && ws.size() * 4 == gr_workspace_bytes(ok));
      float cache[12] = {0}, key[3] = {1, 2, 3};
      CHECK(native_qsa_indexer_append(cache, 4, 2, key, 3) && cache[6] == 1 && cache[8] == 3 && cache[0] == 0 && cache[5] == 0 && cache[9] == 0);
      CHECK(!native_qsa_indexer_append(cache, 4, 4, key, 3) && !native_qsa_indexer_append(cache, 4, -1, key, 3));
      float qq[2] = {1, 0}, ki[14], s1[3], s2[3]; for (int c = 0; c < 7; ++c) { ki[c * 2] = (float)c; ki[c * 2 + 1] = 9; }
      native_qsa_score(qq, ki, 7, 3, 2, s1); qsa_block_scores(qq, ki, 7, 3, 2, s2); CHECK(s1[0] == s2[0] && s1[2] == s2[2] && s1[1] == 5);
      float lgt[12]; for (auto& e : lgt) e = rnd(); int i1[10], i2[10]; float w1[10], w2[10];
      CHECK(router_top10(lgt, 12, i1, w1) == 10 && native_router_top10(lgt, 12, i2, w2) == 10);
      for (int i = 0; i < 10; ++i) CHECK(i1[i] == i2[i] && w1[i] == w2[i]); }

    if (fails) { std::printf("%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("PORT UNIT 19 TESTS PASSED\n");
    return 0;
}
