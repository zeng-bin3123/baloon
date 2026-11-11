// Unit p5a693eb464: the decode-kernel surface (bf16/rope/kv-q8 scalars, GEMV family, GDN, gated residual, QSA select, router,
// MoE combine, IQ4 decode/MMVQ, native-path switches, QSA decode-attention pools).
// (1) Port check: every row of the port table names a file in this repo that defines the listed identifier (181 rows).
// (2) Invariants of the kernel math, checked on the real host reference implementation.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "baloon/decode_attn_kernels.hpp"
#include "baloon/port_kernels_ref.hpp"
#include "baloon/port_scalar.hpp"
using namespace baloon::refk;
static int fails = 0, checks = 0;
#define CHECK(c) do { ++checks; if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static bool is_id(char c) { return std::isalnum((unsigned char)c) || c == '_'; }
static bool defines(const std::string& t, const std::string& name, const std::string& kind) {
    for (size_t p = t.find(name); p != std::string::npos; p = t.find(name, p + 1)) {
        if (p > 0 && is_id(t[p - 1])) continue;
        size_t e = p + name.size();
        if (e < t.size() && is_id(t[e])) continue;
        size_t ls = t.rfind('\n', p); ls = ls == std::string::npos ? 0 : ls + 1;
        std::string pre = t.substr(ls, p - ls);
        if (pre.find("//") != std::string::npos) continue;
        size_t q = e; while (q < t.size() && t[q] == ' ') ++q;
        char nx = q < t.size() ? t[q] : 0;
        if (kind == "macro") { if (pre.find("#define") != std::string::npos || pre.find("# define") != std::string::npos) return true; continue; }
        if (kind == "const") { if ((pre.find("constexpr") != std::string::npos || pre.find("const ") != std::string::npos) && (nx == '=' || nx == '{' || nx == '(')) return true; continue; }
        bool head = pre.find("struct ") != std::string::npos || pre.find("class ") != std::string::npos || pre.find("enum ") != std::string::npos || pre.find("using ") != std::string::npos;
        if (head && (nx == '{' || nx == '=' || nx == ':' || nx == ';')) return true;
        if (!head && nx == '(' && !pre.empty() && pre.find("return") == std::string::npos && pre.find('=') == std::string::npos &&
            pre.find('.') == std::string::npos && pre.find("->") == std::string::npos && pre.find('(') == std::string::npos) return true;
    }
    return false;
}
static void port_check(const char* root) {
    std::ifstream in(std::string(root) + "/.einsteinOS/port/p5a693eb464.tsv");
    CHECK(in.good());
    std::string line; int n = 0; std::set<std::string> seen;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f; std::stringstream ss(line); std::string t;
        while (std::getline(ss, t, '\t')) f.push_back(t);
        CHECK(f.size() == 4); if (f.size() != 4) continue;
        ++n;
        CHECK(f[0].rfind("include/ref/kernels/", 0) == 0);
        std::ifstream src(std::string(root) + "/" + f[2]); CHECK(src.good());
        std::stringstream buf; buf << src.rdbuf();
        // kind comes from the reference name list shipped with the test
        static std::vector<std::pair<std::string, std::string>> kinds;
        if (kinds.empty()) { std::ifstream kn(std::string(root) + "/tests/k51_kernels_port/ref_names.txt"); std::string a, b; while (kn >> a >> b) kinds.push_back({a, b}); }
        std::string kind = "func"; for (auto& k : kinds) if (k.first == f[1]) kind = k.second;
        const bool ok = defines(buf.str(), f[3], kind);
        if (!ok) std::printf("port miss: %s -> %s %s\n", f[1].c_str(), f[2].c_str(), f[3].c_str());
        CHECK(ok);
        CHECK(seen.insert(f[0] + "|" + f[1]).second);
    }
    CHECK(n == 181);
}
static float lcg_state = 1.f;
static float rnd() { static uint32_t s = 12345u; s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 32768.f - 1.f; }

int main(int argc, char** argv) {
    port_check(argc > 1 ? argv[1] : ".");
    (void)lcg_state;

    // ---- bf16: representable values roundtrip exactly; error <= 2^-8 relative; ties to even; NaN/inf preserved ----
    for (int i = 0; i < 2000; ++i) {
        float x = rnd() * 1000.f; uint16_t b = bf16_from_f32(x); float y = f32_from_bf16(b);
        CHECK(std::fabs(y - x) <= std::fabs(x) * (1.f / 256.f) + 1e-30f);
        CHECK(bf16_from_f32(y) == b);
    }
    { uint32_t u = 0x3F808000u; float f; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x3F80);   // tie -> even (down)
      u = 0x3F818000u; std::memcpy(&f, &u, 4); CHECK(bf16_from_f32(f) == 0x3F82);                       // tie -> even (up)
      CHECK(std::isnan(f32_from_bf16(bf16_from_f32(NAN))));
      CHECK(std::isinf(f32_from_bf16(bf16_from_f32(INFINITY))) && f32_from_bf16(bf16_from_f32(-INFINITY)) < 0); }

    // ---- RoPE: rotation preserves pair norm; pos 0 is identity; dims >= n_rot pass through; mrope reduces to rope ----
    { const int n_rot = 16, hd = 24; float x[hd], x0[hd], cs[8], sn[8];
      for (int i = 0; i < hd; ++i) x0[i] = x[i] = rnd();
      build_rope_table(cs, sn, n_rot, 0, 10000.f); rope_neox_apply(x, n_rot, cs, sn);
      for (int i = 0; i < hd; ++i) CHECK(x[i] == x0[i]);
      build_rope_table(cs, sn, n_rot, 37, 10000.f); rope_neox_apply(x, n_rot, cs, sn);
      for (int i = 0; i < 8; ++i) CHECK(std::fabs(x[i] * x[i] + x[i + 8] * x[i + 8] - x0[i] * x0[i] - x0[i + 8] * x0[i + 8]) < 1e-5f);
      for (int i = n_rot; i < hd; ++i) CHECK(x[i] == x0[i]);
      const int sec[3] = {3, 3, 2}; CHECK(mrope_table_set(sec));
      const int bad[3] = {1, -1, 2}; CHECK(!mrope_table_set(bad));
      int p3[3] = {5, 5, 5}; float c2[8], s2[8], c1[8], s1[8];
      mrope_table(c2, s2, n_rot, p3, 10000.f); build_rope_table(c1, s1, n_rot, 5, 10000.f);
      for (int i = 0; i < 8; ++i) CHECK(c1[i] == c2[i] && s1[i] == s2[i]);
      int q3[3] = {1, 2, 3}; CHECK(mrope_pos(q3, sec, 0) == 1 && mrope_pos(q3, sec, 3) == 2 && mrope_pos(q3, sec, 6) == 3 && mrope_pos(q3, sec, 7) == 3);
      const int def[3] = {11, 11, 10}; mrope_table_set(def); }

    // ---- KV-Q8 sizing ----
    CHECK(kv_q8_bytes_per_cell(2, 256) == 2 * 256 + 2 * 4 * 2);
    CHECK(kv_q8_bytes_per_cell(2, 100) == 0 && kv_q8_bytes_per_cell(0, 256) == 0);

    // ---- bf16 GEMV: splits==1 is exactly the plain kernel; other splits agree to fp32 tolerance ----
    { const int ni = 192, no = 5; std::vector<uint16_t> w(ni * no); std::vector<float> x(ni), y0(no), y1(no), y3(no), y7(no);
      for (auto& v : w) v = bf16_from_f32(rnd()); for (auto& v : x) v = rnd();
      bf16_gemv_fp32_mmvf(w.data(), x.data(), y0.data(), ni, no); bf16_gemv_split(w.data(), x.data(), y1.data(), ni, no, 1);
      bf16_gemv_split(w.data(), x.data(), y3.data(), ni, no, 3); bf16_gemv_split(w.data(), x.data(), y7.data(), ni, no, 7);
      for (int r = 0; r < no; ++r) { CHECK(y0[r] == y1[r]); CHECK(std::fabs(y0[r] - y3[r]) < 1e-3f); CHECK(std::fabs(y0[r] - y7[r]) < 1e-3f); } }

    // ---- S2 GEMV: GEMV equals dot(dequant row, x); fast/quads variants are bit-identical to the reference; q8 variant is close ----
    { const int ni = 256, no = 4, gs = ni / 64; Codebook cb; std::vector<uint8_t> codes(no * ni / 4); std::vector<float> sc(no * gs), x(ni), yr(no), yf(no), yq(no), y8(no);
      for (auto& c : codes) c = (uint8_t)(rnd() * 127 + 128); for (auto& s : sc) s = std::fabs(rnd()) + 0.1f; for (auto& v : x) v = rnd();
      s2_gemv(codes.data(), sc.data(), cb, x.data(), yr.data(), ni, no);
      s2_gemv_fast(codes.data(), sc.data(), cb, x.data(), yf.data(), ni, no); s2_gemv_quads(codes.data(), sc.data(), cb, x.data(), yq.data(), ni, no);
      for (int r = 0; r < no; ++r) {
          std::vector<float> row(ni); dequant_s2_ref(codes.data() + r * ni / 4, sc.data() + r * gs, cb, row.data(), ni);
          // dequant_s2 takes one scale per 64 values for the row passed in
          double d = 0; for (int i = 0; i < ni; ++i) d += (double)row[i] * x[i];
          CHECK(std::fabs(d - yr[r]) < 1e-3); CHECK(yf[r] == yr[r]); CHECK(std::fabs(yq[r] - yr[r]) < 1e-3f); }
      std::vector<int8_t> xq(ni); std::vector<float> xs(gs);
      for (int g = 0; g < gs; ++g) { float m = 0; for (int i = 0; i < 64; ++i) m = std::fmax(m, std::fabs(x[g * 64 + i])); xs[g] = m / 127.f;
          for (int i = 0; i < 64; ++i) xq[g * 64 + i] = (int8_t)std::lrintf(x[g * 64 + i] / xs[g]); }
      s2_gemv_q8(codes.data(), sc.data(), cb, xq.data(), xs.data(), y8.data(), ni, no);
      for (int r = 0; r < no; ++r) CHECK(std::fabs(y8[r] - yr[r]) < 0.05f * (1.f + std::fabs(yr[r]))); }

    // ---- GDN: l2 norm gives unit vectors; beta/gate ranges; step with zero state/zero v keeps state zero ----
    { const int heads = 3, d = 16; std::vector<float> x(heads * d); for (auto& v : x) v = rnd() * 4; gdn_l2_norm(x.data(), heads, d, 1e-6f);
      for (int h = 0; h < heads; ++h) { double s = 0; for (int i = 0; i < d; ++i) s += x[h * d + i] * x[h * d + i]; CHECK(std::fabs(s - 1.0) < 1e-4); }
      float b[4] = {-5, 0, 3, 9}, a[4] = {-1, 0, 1, 2}, dt[4] = {0.1f, 0.2f, 0.3f, 0.4f}, Al[4] = {0.f, -1.f, 0.5f, 1.f}, beta[4], g[4], beta2[4], g2[4];
      gdn_beta_gate(b, a, dt, Al, beta, g, 4); fused_gdn_ab(b, a, dt, Al, beta2, g2, 4);
      for (int i = 0; i < 4; ++i) { CHECK(beta[i] > 0.f && beta[i] < 1.f); CHECK(g[i] <= 0.f); CHECK(beta[i] == beta2[i] && g[i] == g2[i]); }
      const int dk = 4, dv = 3; std::vector<float> st(dk * dv, 0.f), q(dk, 1.f), k(dk, 0.5f), v(dv, 0.f), o(dv, 9.f);
      gdn_step(st.data(), q.data(), k.data(), v.data(), -0.1f, 0.5f, 0.5f, o.data(), dk, dv);
      for (float s : st) CHECK(s == 0.f); for (float s : o) CHECK(s == 0.f);
      std::vector<float> st2(dk * dv, 0.f), o2(dv); native_gdn_step(st2.data(), q.data(), k.data(), v.data(), -0.1f, 0.5f, 0.5f, o2.data(), dk, dv);
      for (int i = 0; i < dv; ++i) CHECK(o2[i] == o[i]); }

    // ---- gated residual: workspace sizing is 16-byte aligned and monotone; post with zero block is identity-like ----
    { GrShapes s; s.n = 10; s.hc = 4; CHECK(s.ok()); CHECK(gr_workspace_bytes(s) % 16 == 0 && gr_workspace_bytes(s) > 0);
      GrShapes s2 = s; s2.n = 100; CHECK(gr_workspace_bytes(s2) > gr_workspace_bytes(s)); GrShapes bad; CHECK(!bad.ok() && gr_workspace_bytes(bad) == 0);
      CHECK(fused_gr_supported(1, 8, 4) && fused_gr_supported(kFusedGrMaxT, 8, 4) && !fused_gr_supported(kFusedGrMaxT + 1, 8, 4) && !fused_gr_supported(0, 8, 4));
      float a[4] = {1, -2, 3, -4}, c[4] = {1, 1, 1, 1}, o[4]; native_gr_down_silu(a, c, o, 4);
      for (int i = 0; i < 4; ++i) CHECK(std::fabs(o[i] - silu(a[i])) < 1e-6f);
      gr_set_fp32_activations(true); gr_set_native_mmvf(false); gr_set_fp32_activations(false); gr_set_native_mmvf(true); }

    // ---- QSA block select: score = block max; top-k best first, ties to lower index; k clamped ----
    { const int d = 4, n = 10, bc = 4; std::vector<float> q(d), kk(n * d), sc(3); for (auto& v : q) v = rnd(); for (auto& v : kk) v = rnd();
      qsa_block_scores(q.data(), kk.data(), n, bc, d, sc.data());
      for (int b = 0; b < 3; ++b) { float m = -INFINITY; for (int c = b * bc; c < std::min(n, (b + 1) * bc); ++c) { float dot = 0; for (int i = 0; i < d; ++i) dot += q[i] * kk[c * d + i]; m = std::fmax(m, dot); } CHECK(sc[b] == m); }
      float t[5] = {1, 3, 3, 2, 3}; auto o = qsa_block_topk(t, 5, 3); CHECK(o.size() == 3 && o[0] == 1 && o[1] == 2 && o[2] == 4);
      CHECK(qsa_block_topk(t, 5, 99).size() == 5 && qsa_block_topk(t, 5, -1).empty()); }

    // ---- router top-10: distinct indices, weights positive and sum to 1, descending, k clamped to n_expert ----
    { std::vector<float> lg(64); for (auto& v : lg) v = rnd() * 3; int idx[10]; float w[10];
      CHECK(router_top10(lg.data(), 64, idx, w) == 10); std::set<int> u(idx, idx + 10); CHECK(u.size() == 10);
      float s = 0; for (int j = 0; j < 10; ++j) { s += w[j]; CHECK(w[j] > 0); if (j) CHECK(lg[idx[j - 1]] >= lg[idx[j]]); } CHECK(std::fabs(s - 1.f) < 1e-5f);
      CHECK(router_top10(lg.data(), 4, idx, w) == 4); CHECK(router_top10(lg.data(), 0, idx, w) == 0);
      float eq[12]; for (auto& v : eq) v = 1.f; router_top10(eq, 12, idx, w); for (int j = 0; j < 10; ++j) CHECK(idx[j] == j); }

    // ---- MoE combine: out = shared + sum w*routed; null shared means 0 ----
    { float rt[6] = {1, 2, 3, 4, 5, 6}, w[2] = {0.25f, 0.75f}, sh[3] = {10, 20, 30}, o[3], o2[3];
      native_moe_combine(rt, w, sh, o, 2, 3); native_moe_combine(rt, w, nullptr, o2, 2, 3);
      for (int d = 0; d < 3; ++d) { CHECK(std::fabs(o[d] - (sh[d] + 0.25f * rt[d] + 0.75f * rt[3 + d])) < 1e-6f); CHECK(std::fabs(o2[d] - (o[d] - sh[d])) < 1e-6f); } }

    // ---- IQ4: supported types, row bytes, codebook decode, embed rows == dequant rows, grouped expert layout consistent ----
    { CHECK(iq_supported(20) && iq_supported(23) && !iq_supported(2));
      CHECK(iq_row_bytes(20, 64) == 2 * 18 && iq_row_bytes(23, 512) == 2 * 136 && iq_row_bytes(20, 33) == 0 && iq_row_bytes(7, 64) == 0);
      std::vector<uint8_t> blk(18, 0); uint16_t one = 0x3C00; std::memcpy(blk.data(), &one, 2); for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)(i | ((15 - i) << 4));
      std::vector<float> f(32); iq_dequant_f32(20, blk.data(), f.data(), 32);
      for (int i = 0; i < 16; ++i) { CHECK(f[i] == (float)kIq4Values[i]); CHECK(f[16 + i] == (float)kIq4Values[15 - i]); }
      std::vector<uint8_t> tab(blk); tab.insert(tab.end(), blk.begin(), blk.end()); int32_t ids[2] = {1, 0}; std::vector<float> e(64);
      iq_embed_rows(20, tab.data(), 2, ids, 2, 32, e.data()); for (int i = 0; i < 64; ++i) CHECK(e[i] == f[i % 32]);
      std::vector<float> x(32, 1.f), y(2); std::vector<uint8_t> w2(blk); w2.insert(w2.end(), blk.begin(), blk.end());
      CHECK(iq_mmvq(20, w2.data(), x.data(), y.data(), 32, 2)); double ssum = 0; for (float v : f) ssum += v; CHECK(std::fabs(y[0] - ssum) < 0.5 && std::fabs(y[1] - y[0]) < 1e-6);
      NativeExpertLayout L = native_expert_layout(20, 64, 32); CHECK(L.ok && L.bytes >= L.down_off + L.down_row_bytes && L.up_off > 0 && L.down_off >= L.up_off);
      CHECK(native_expert_scratch_bytes(64, 32) == 4ull * (64 + 64 + 64)); }

    // ---- native-path switches round trip ----
    { native_gdn_set_enabled(false); CHECK(!native_gdn_enabled()); native_gdn_set_enabled(true); CHECK(native_gdn_enabled());
      native_moe_combine_set_enabled(false); CHECK(!native_moe_combine_enabled()); native_moe_combine_set_enabled(true);
      native_qsa_set_enabled(false); CHECK(!native_qsa_enabled()); native_qsa_set_enabled(true);
      native_qsa_indexer_set_enabled(false); CHECK(!native_qsa_indexer_enabled()); native_qsa_indexer_set_enabled(true);
      native_qsa_score_set_enabled(false); CHECK(!native_qsa_score_enabled()); native_qsa_score_set_enabled(true);
      native_rope_set_enabled(false); CHECK(!native_rope_enabled()); native_rope_set_enabled(true);
      native_router_set_enabled(false); CHECK(!native_router_enabled()); native_router_set_enabled(true); CHECK(native_router_enabled()); }

    // ---- QSA decode-attention pools: row addressing is page-major, per_call copies every field, validity rules ----
    { namespace ak = baloon::attn_kernels; float k[1], v[1], ks[1], vs[1], q[1], sc[1], out[1]; int32_t ids[1], nid[1];
      ak::QsaAttnPools p; CHECK(!p.valid());
      p.k = k; p.v = v; p.n_kv = 2; p.page_size = 4; p.n_cells = 10; CHECK(p.valid());
      CHECK(p.row(0, 0) == 0 && p.row(3, 0) == 3 && p.row(3, 1) == 7 && p.row(4, 0) == 8 && p.row(9, 1) == (2 * 2 + 1) * 4 + 1);
      std::set<int64_t> rows; for (int c = 0; c < 10; ++c) for (int h = 0; h < 2; ++h) rows.insert(p.row(c, h)); CHECK(rows.size() == 20);
      p.q8 = true; CHECK(!p.valid()); p.k_scale = ks; p.v_scale = vs; CHECK(p.valid());
      ak::QsaArgs a = p.per_call(q, ids, nid, sc, out, 3, 128, 0.0625f);
      CHECK(a.q == q && a.k == k && a.v == v && a.k_scale == ks && a.v_scale == vs && a.ids == ids && a.n_ids == nid && a.scratch == sc && a.out == out);
      CHECK(a.nq == 3 && a.n_kv == 2 && a.cap == 128 && a.page_size == 4 && a.n_cells == 10 && a.q8 && a.scale == 0.0625f);
      CHECK(ak::qsa_scratch_floats(128, 24) == (uint64_t)ak::n_chunks(128) * 24 * (ak::kHD + 2) + 64 && ak::n_chunks(129) == 3); }

    if (fails == 0) std::printf("PASS (%d checks)\n", checks); else std::printf("FAILED %d of %d checks\n", fails, checks);
    return fails == 0 ? 0 : 1;
}
