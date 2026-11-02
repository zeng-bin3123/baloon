// Unit 4 invariant tests (host reference). Each CHECK failure makes the program exit non-zero.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include "../src/ix_qsa_decode.h"
#include "../src/ix_qsa_select.h"
#include "../src/ix_quantize_act.h"
#include "../src/ix_router_top10.h"
#include "../src/ix_verify_kernels.h"
#include "../src/ix_sampler.h"
#include "../src/ix_s2_expert.h"
#include "../src/ix_sgemv.h"
static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static std::mt19937 rng(1234);
static float rnd(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

static void test_decode() {
    CHECK(qsa_n_chunks(64) == 1 && qsa_n_chunks(65) == 2 && qsa_n_chunks(0) == 0);
    for (int cap = 0; cap < 500; ++cap) CHECK(cap <= qsa_n_chunks(cap) * 64);
    CHECK(qsa_n_here(10, 0) == 10 && qsa_n_here(10, 1) == 0 && qsa_n_here(200, 1) == 64 && qsa_n_here(200, 3) == 8);
    CHECK(qsa_scratch_floats(130, 24) == 3L * 24 * 258 + 64);
    for (int h = 0; h < 24; ++h) CHECK(qsa_kv_head(h) * 12 + qsa_local_head(h) == h && qsa_kv_head(h) < 2);
    const int hd = 8;
    for (int n : {1, 63, 64, 65, 200, 333}) {
        std::vector<float> s(n), v((size_t)n * hd);
        for (auto& x : s) { x = rnd(-5, 5); }
        for (auto& x : v) { x = rnd(-1, 1); }
        int nc = qsa_n_chunks(n) + 2;  // two trailing empty chunks must be skipped
        std::vector<float> m(nc), l(nc), acc((size_t)nc * hd);
        for (int c = 0; c < nc; ++c) qsa_partial(s.data() + std::min(c * 64, n), v.data() + (size_t)std::min(c * 64, n) * hd, qsa_n_here(n, c), hd, &m[c], &l[c], &acc[(size_t)c * hd]);
        CHECK(m[nc - 1] == -FLT_MAX && l[nc - 1] == 0.f);
        std::vector<float> out(hd); qsa_merge(m.data(), l.data(), acc.data(), nc, hd, out.data());
        double mx = -1e30; for (float x : s) mx = std::max(mx, (double)x);
        std::vector<double> ref(hd, 0.0); double L = 0; for (int i = 0; i < n; ++i) { double p = exp(s[i] - mx); L += p; for (int d = 0; d < hd; ++d) ref[d] += p * v[(size_t)i * hd + d]; }
        for (int d = 0; d < hd; ++d) CHECK(fabs(out[d] - ref[d] / L) < 1e-5);
    }
    { float m[2] = {-FLT_MAX, -FLT_MAX}, l[2] = {0, 0}, a[16] = {0}, o[8]; qsa_merge(m, l, a, 2, 8, o); for (int d = 0; d < 8; ++d) CHECK(o[d] == 0.f); }
}
static std::vector<int> brute_select(const std::vector<uint32_t>& key, int n_kv, int width) {
    std::vector<int> cells(n_kv); for (int c = 0; c < n_kv; ++c) cells[c] = c;
    std::stable_sort(cells.begin(), cells.end(), [&](int a, int b) { return key[a / 4] > key[b / 4]; });
    cells.resize(std::min(width, n_kv)); std::sort(cells.begin(), cells.end()); return cells;
}
static void test_select() {
    for (int n = 0; n < 40; ++n) CHECK(qsa_block_weight(n, n / 4, n / 4) == n % 4);
    for (int n = 0; n < 40; ++n) CHECK((n / 4) * 4 + qsa_block_weight(n, n / 4, n / 4) == n);
    for (int b = 0; b < 20; ++b) for (int c = 0; c < 4; ++c) CHECK(qsa_cell_id(b, c) / 4 == b && qsa_cell_id(b, c) % 4 == c);
    float hd[3] = {-1.f, 2.f, 3.f}; CHECK(qsa_block_score(hd, 3, false) == 5.f && qsa_block_score(hd, 3, true) == 5.f + 1e9f);
    CHECK(qsa_ordered_key(-1.f) < qsa_ordered_key(0.f) && qsa_ordered_key(0.f) < qsa_ordered_key(2.f) && qsa_ordered_key(2.f) < qsa_ordered_key(3.f));
    for (int trial = 0; trial < 300; ++trial) {
        int n_kv = 1 + (int)(rng() % 300), width = 1 + (int)(rng() % 120); int nb = (n_kv + 3) / 4;
        std::vector<uint32_t> key(nb); int levels = 1 + (int)(rng() % 6);  // few levels => many ties
        for (auto& k : key) k = qsa_ordered_key((float)(int)(rng() % levels)) ;
        if (trial % 3 == 0) for (auto& k : key) k = (uint32_t)rng();
        std::vector<int> sel(std::min(width, n_kv) + 8); int got = qsa_radix_select(key.data(), nb, n_kv, width, sel.data());
        CHECK(got == std::min(width, n_kv));
        std::vector<int> s(sel.begin(), sel.begin() + got); std::sort(s.begin(), s.end());
        CHECK(std::set<int>(s.begin(), s.end()).size() == s.size());
        CHECK(s == brute_select(key, n_kv, width));
    }
}
static void test_quant() {
    CHECK(IX_Q80_BYTES == 2 + 32 && IX_Q8K_BYTES == 4 + 256 + 2 * 16);
    CHECK(ix4_nearest_int(2.5f) == 2 && ix4_nearest_int(3.5f) == 4 && ix4_nearest_int(-1.2f) == -1);
    CHECK(ix4_round_away(2.5f) == 3.f && ix4_round_away(-2.5f) == -3.f);
    CHECK(ix4_f16_to_f32(ix4_f32_to_f16(1.0f)) == 1.0f && ix4_f16_to_f32(ix4_f32_to_f16(-0.5f)) == -0.5f && ix4_f32_to_f16(1.0f) == 0x3c00);
    for (int t = 0; t < 200; ++t) {
        float x[256]; float sc = powf(10.f, rnd(-2, 2)); for (auto& v : x) v = rnd(-1, 1) * sc;
        uint8_t b[34]; quantize_q8_0_block(x, b); float y[32]; dequantize_q8_0_block(b, y);
        uint16_t h; memcpy(&h, b, 2); float d = ix4_f16_to_f32(h);
        for (int i = 0; i < 32; ++i) CHECK(fabsf(y[i] - x[i]) <= 0.5f * d * 1.01f + 1e-3f * sc);
        float ds; int8_t q[32]; quantize_q8_0_f32scale_block(x, &ds, q);
        for (int i = 0; i < 32; ++i) { CHECK(q[i] >= -127 && q[i] <= 127); CHECK(fabsf(q[i] * ds - x[i]) <= 0.5f * ds * 1.001f + 1e-6f); }
        uint8_t k[292]; quantize_q8_K_block(x, k); float yk[256]; dequantize_q8_K_block(k, yk); float dk; memcpy(&dk, k, 4);
        int16_t bs[16]; memcpy(bs, k + 260, 32);
        for (int j = 0; j < 16; ++j) { int s = 0; for (int i = 0; i < 16; ++i) s += (int8_t)k[4 + j * 16 + i]; CHECK(s == bs[j]); }
        for (int i = 0; i < 256; ++i) { CHECK((int8_t)k[4 + i] <= 127); CHECK(fabsf(yk[i] - x[i]) <= 0.5f * fabsf(dk) * 1.01f + 1e-5f * sc); }
    }
    float z[256] = {0}; uint8_t b[34]; quantize_q8_0_block(z, b); float y[32]; dequantize_q8_0_block(b, y); for (float v : y) CHECK(v == 0.f);
    uint8_t k[292]; quantize_q8_K_block(z, k); float yk[256]; dequantize_q8_K_block(k, yk); for (float v : yk) CHECK(v == 0.f);
    CHECK(2560 / 32 == 80 && 2560 / 256 == 10);
}
static void test_router() {
    for (int n : {1, 8, 31, 32, 33, 64, 256, 512}) {
        CHECK(ix4_round16(n) >= n && ix4_round16(n) % 16 == 0);
        CHECK(router_threads(n) % 32 == 0 && router_threads(n) >= std::min(n, 512));
    }
    CHECK(router_smem_bytes(512) == 512 + 4096 + 2048 && router_threads(512) == 512);
    for (int trial = 0; trial < 200; ++trial) {
        int n = 16 + (int)(rng() % 500), k = std::min(n, 10); std::vector<float> lg(n);
        int levels = 1 + (int)(rng() % 5); for (auto& v : lg) v = trial % 2 ? (float)(int)(rng() % levels) : rnd(-4, 4);
        std::vector<int> ids(k); std::vector<float> w(k); std::vector<double> p(n); std::vector<unsigned char> tk(n);
        router_top10_ref(lg.data(), n, k, ids.data(), w.data(), p.data(), tk.data());
        std::vector<int> ord(n); for (int i = 0; i < n; ++i) ord[i] = i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return lg[a] > lg[b]; });  // stable descending argsort
        float sum = 0; for (int j = 0; j < k; ++j) sum += w[j];
        for (int j = 0; j < k; ++j) CHECK(ids[j] == ord[j]);
        CHECK(fabsf(sum - 1.f) < 1e-4f);
    }
    for (int a = 0; a < 6; ++a) for (int b = 0; b < 6; ++b) {
        float va = a / 2, vb = b / 2; CHECK(!router_better(va, a, va, a));
        if (a != b) CHECK(router_better(va, a, vb, b) != router_better(vb, b, va, a));
    }
}
static void test_verify() {
    for (int j = 0; j < 4; ++j) { CHECK(verify_conv_src(0, j) == j); }
    for (int t = 3; t < 10; ++t) for (int j = 0; j < 4; ++j) CHECK(verify_conv_src(t, j) >= 3);
    for (int j = 0; j < 3; ++j) CHECK(verify_commit_src(0, j) == j);
    for (int n = 0; n < 8; ++n) {   // commit keeps last three of [h0 h1 h2 x0..x_{n-1}]
        std::vector<int> cat = {-3, -2, -1}; for (int i = 0; i < n; ++i) cat.push_back(i);
        for (int j = 0; j < 3; ++j) CHECK(cat[verify_commit_src(n, j)] == cat[cat.size() - 3 + j]);
    }
    for (int nkv = 0; nkv < 50; ++nkv) for (int w : {1, 8, 32}) {
        CHECK(verify_wstart(nkv, w) + verify_wwidth(nkv, w) == nkv); CHECK(verify_wwidth(nkv, w) <= w || nkv <= w);
    }
    for (int pos = 0; pos < 40; ++pos) { VerifyDense s = verify_dense_step(pos); CHECK(s.n_kv == pos + 1 && s.n_bid == s.n_kv / 4 && s.n_bid * 4 <= s.n_kv && s.width == pos + 1); }
    for (int n : {1, 7, 2560}) for (int hc : {1, 4}) for (int i : {0, n - 1, n, n * hc - 1}) if (i < n * hc) { CHECK(verify_bcast_src(i, n) < n && verify_bcast_stream(i, n) < hc); CHECK(verify_bcast_stream(i, n) * n + verify_bcast_src(i, n) == i); }
    for (int k = 0; k < 20; ++k) CHECK(verify_rebase(1000, 48, k + 1) == verify_rebase(1000, 48, k) + 48);
    for (int hv = 1; hv < 40; ++hv) CHECK(verify_ab_grid(hv) * 8 >= 2 * hv);
    CHECK(128 / 4 == 32);
}
static void test_sampler() {
    IxValIdx a{1.f, 5}, b{1.f, 2}, c{2.f, 9};
    CHECK(sampler_merge(a, b).i == 2 && sampler_merge(b, a).i == 2 && sampler_merge(a, c).i == 9 && sampler_merge(c, a).i == 9);
    for (uint32_t c0 : {0u, 255u, 256u, 0xffffffffu, 123456789u}) { float u = sampler_uniform24(c0); CHECK(u >= 0.f && u < 1.f); }
    for (int h = 0; h < 20; ++h) for (int ln : {0, 5, 64}) { int L = sampler_hlen(ln, h); CHECK(L <= h && (h - L) + L == h); }
    for (int cut = 0; cut <= 10; ++cut) { CHECK(sampler_clamp_cut(cut, 3, 10) <= 10); CHECK(sampler_clamp_cut(cut, 3, 10) >= 3); }
    CHECK(sampler_clamp_cut(0, 5, 2) == 2);
    CHECK(sampler_presence_term(0) == 0 && sampler_presence_term(7) == 1);
    for (float p : {1.f, 1.1f, 2.f}) { CHECK(sampler_repeat_pen(3.f, p) <= 3.f); CHECK(sampler_repeat_pen(-3.f, p) <= -3.f); CHECK(sampler_repeat_pen(0.f, p) == 0.f); }
}

static void test_s2() {
    CHECK(S2_BLOB_BYTES == 1382400 && S2_O_DCODES < S2_O_GUSC && S2_O_GUSC < S2_O_DSC && S2_O_DSC < S2_BLOB_BYTES);
    for (int c = 0; c < 256; ++c) { uint8_t b = (uint8_t)c; int r = 0; for (int k = 0; k < 4; ++k) { CHECK(s2_code(&b, k) == ((c >> (2 * k)) & 3)); r += s2_code(&b, k) << (2 * k); } CHECK(r == c); }
    for (int i = 0; i < 100; ++i) CHECK(s2_slot_row(i) * 2 + (s2_slot_is_up(i) ? 1 : 0) == i);
    for (int n : {1, 5}) for (int h = 0; h < n; ++h) for (int r : {0, 639}) { CHECK(h * 640 + r < n * 640); CHECK(n * 640 + h * 640 + r >= n * 640 && n * 640 + h * 640 + r < 2 * n * 640); }
    // bias decomposition: sum (c-1)x == sum c x - sum x
    for (int t = 0; t < 50; ++t) {
        uint8_t codes[8]; int8_t x[32]; for (auto& b : codes) b = (uint8_t)rng(); for (auto& v : x) v = (int8_t)((int)(rng() % 255) - 127);
        int ref = 0; for (int i = 0; i < 32; ++i) ref += (s2_code(codes, i) - 1) * x[i];
        CHECK(s2_dot_bias(codes, 0, x, 32) == ref);
    }
    // full FFN vs float-dequantised path on a random blob
    std::vector<uint8_t> blob(S2_BLOB_BYTES); for (auto& b : blob) b = (uint8_t)rng();
    uint16_t* sc = (uint16_t*)(blob.data() + S2_O_GUSC); for (long i = 0; i < 2 * 640 * 40; ++i) sc[i] = ix4_f32_to_f16(rnd(0.001f, 0.01f));
    sc = (uint16_t*)(blob.data() + S2_O_DSC); for (long i = 0; i < 2560 * 10; ++i) sc[i] = ix4_f32_to_f16(rnd(0.001f, 0.01f));
    std::vector<float> x(S2_H); for (auto& v : x) v = rnd(-1, 1);
    std::vector<int8_t> xq(S2_H); std::vector<float> dx(S2_H / 32), xd(S2_H);
    for (int b = 0; b < S2_H / 32; ++b) { uint8_t blk[34]; quantize_q8_0_block(&x[b * 32], blk); uint16_t h; memcpy(&h, blk, 2); dx[b] = ix4_f16_to_f32(h); for (int i = 0; i < 32; ++i) { xq[b * 32 + i] = (int8_t)blk[2 + i]; xd[b * 32 + i] = dx[b] * xq[b * 32 + i]; } }
    std::vector<float> gu(2 * S2_FF), out(S2_H); s2_expert_ffn(blob.data(), xq.data(), dx.data(), gu.data(), out.data());
    for (int i : {0, 1, 2, 3, 700, 1279}) {   // gate/up rows against explicit dequant
        double ref = 0; for (int e = 0; e < S2_H; ++e) ref += (double)(s2_code(blob.data(), (long)i * S2_H + e) - 1) * ix4_f16_to_f32(((uint16_t*)(blob.data() + S2_O_GUSC))[(long)i * 40 + e / 64]) * xd[e];
        float got = gu[(s2_slot_is_up(i) ? S2_FF : 0) + s2_slot_row(i)]; CHECK(fabs(got - ref) <= 1e-3 * (1 + fabs(ref)));
    }
    bool finite = true; for (float v : out) finite = finite && std::isfinite(v); CHECK(finite);
}
static void test_sgemv() {
    CHECK(sgemv_per_byte(2) == 4 && sgemv_per_byte(4) == 2 && sgemv_per_byte(8) == 1);
    for (int i = 1; i < 16; ++i) CHECK(kIq4NlTable[i - 1] < kIq4NlTable[i]);
    CHECK(sgemv_act_ok(640, 32) && !sgemv_act_ok(640, 256) && sgemv_act_ok(2560, 256));
    for (int i = 0; i < 1024; i += 16) for (int g : {16, 32, 64}) { CHECK((i + 15) / g == i / g); CHECK(i % 256 + 16 <= 256 && i % 32 + 16 <= 32); }
    for (int bits : {2, 4, 8}) for (int group : {16, 32, 64}) for (int nl = 0; nl < 2; ++nl) for (int use_off = 0; use_off < 2; ++use_off) {
        if (nl && bits != 4) continue;
        int n = 1024; std::vector<uint8_t> codes(n * bits / 8); for (auto& b : codes) b = (uint8_t)rng();
        std::vector<float> sc(n / group), off(n / group), y(n); for (auto& v : sc) v = rnd(0.01f, 0.1f); for (auto& v : off) v = rnd(-0.5f, 0.5f); for (auto& v : y) v = rnd(-1, 1);
        float a = sgemv_row_naive(codes.data(), bits, sc.data(), use_off ? off.data() : nullptr, group, nl, y.data(), n);
        float b = sgemv_row_warp(codes.data(), bits, sc.data(), use_off ? off.data() : nullptr, group, nl, y.data(), n);
        CHECK(fabsf(a - b) <= 1e-3f * (1.f + fabsf(a)));
    }
    uint8_t byte = 0xE4; // codes 0,1,2,3 at 2 bits; low nibble 4, high nibble 14 at 4 bits
    for (int k = 0; k < 4; ++k) CHECK(sgemv_code(&byte, k, 2) == k);
    CHECK(sgemv_code(&byte, 0, 4) == 4 && sgemv_code(&byte, 1, 4) == 14 && sgemv_code(&byte, 0, 8) == 0xE4);
}
static void test_sampler_chain() {
    float lg[5] = {1.f, 3.f, 3.f, -2.f, 0.f}; CHECK(sampler_greedy(lg, 5) == 1);   // tie -> lowest index
    int hist[4] = {1, 1, 3, 0}; float l2[5] = {1.f, 3.f, 3.f, -2.f, 0.f};
    sampler_penalties(l2, hist, 4, 3, 2.f, 0.f, 0.f);   // window = last 3 = {1,3,0}
    CHECK(l2[1] == 1.5f && l2[3] == -4.f && l2[0] == 0.5f && l2[2] == 3.f);
    CHECK(sampler_greedy(l2, 5) == 2);
    float l3[2] = {1.f, 0.f}; int h3[2] = {0, 0}; sampler_penalties(l3, h3, 2, 0, 1.f, 0.5f, 0.25f); CHECK(l3[0] == 1.f && l3[1] == 0.f);
}

static void test_sampler_pick() {
    for (uint64_t c = 0; c < 2000; ++c) { float u = sampler_draw(42, c); CHECK(u >= 0.f && u < 1.f); CHECK(u == sampler_draw(42, c)); }
    CHECK(sampler_draw(42, 0) != sampler_draw(42, 1) || sampler_draw(42, 1) != sampler_draw(42, 2));
    double m = 0; for (uint64_t c = 0; c < 20000; ++c) m += sampler_draw(7, c); m /= 20000; CHECK(fabs(m - 0.5) < 0.02);
    float lg[6] = {2.f, 5.f, 4.f, 1.f, 5.f, -1.f}; int ord[6]; double pr[6];
    for (float u : {0.f, 0.3f, 0.99f}) { int t = sampler_sample(lg, 6, 2, 1.f, 1.f, 1, u, ord, pr); CHECK(t == 1 || t == 4); }   // top_k=2 keeps the two 5.0 logits
    CHECK(sampler_sample(lg, 6, 0, 1e-6f, 1.f, 1, 0.9f, ord, pr) == 1);                // min_keep=1 keeps best (lowest index of tie)
    CHECK(sampler_sample(lg, 6, 0, 1.f, 1.f, 1, 0.f, ord, pr) == 1);
    int hit[6] = {0}; for (uint64_t c = 0; c < 6000; ++c) hit[sampler_sample(lg, 6, 0, 1.f, 1.f, 1, sampler_draw(3, c), ord, pr)]++;
    CHECK(hit[1] > 0 && hit[4] > 0 && hit[0] > 0 && hit[1] > hit[3]);
}
int main() {
    test_decode(); test_select(); test_quant(); test_router(); test_verify(); test_sampler(); test_s2(); test_sgemv(); test_sampler_chain(); test_sampler_pick();
    if (fails) { printf("unit4: %d FAILED checks\n", fails); return 1; }
    printf("unit4: all checks passed\n"); return 0;
}
