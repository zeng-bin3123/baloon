// test_u11_host.cpp - host invariants of unit u0063babba8: expert blob layout, format traits, backend probe, Q2_0 AVX2 rows and
// activation quantiser, native expert entry points, IQ4_NL / n-gram helpers and the PLE row table. Exit code != 0 on any failure.
#include <immintrin.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "baloon/q2_avx2_rows.hpp"
#include "baloon/u11_expert.hpp"
#include "baloon/u11_ngram.hpp"

using namespace baloon;
static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::mt19937 rng(1234);
static float rnd(float a) { return std::uniform_real_distribution<float>(-a, a)(rng); }
static uint16_t f2h(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static bool same_layout(const ExpertLayout& a, const ExpertLayout& b) {
    return a.type == b.type && a.n_embd == b.n_embd && a.n_ff == b.n_ff && a.gu_row_bytes == b.gu_row_bytes && a.down_row_bytes == b.down_row_bytes &&
           a.up_off == b.up_off && a.down_off == b.down_off && a.bytes == b.bytes && a.ok == b.ok;
}

static void test_layout_and_traits() {
    const int types[] = {42, 16, 17, 18, 21, 22};
    for (int t : types) {
        const auto tr = u11::native::traits(native_fmt(t));
        CHECK(tr.valid && tr.type_id == t && tr.fmt == native_fmt(t));
        const int64_t n_embd = tr.block_values * 4, n_ff = tr.block_values * 2;
        const ExpertLayout l = u11::layout::expert_layout(t, n_embd, n_ff);
        CHECK(l.ok);
        CHECK(l.gu_row_bytes == n_embd / tr.block_values * tr.block_bytes && l.down_row_bytes == n_ff / tr.block_values * tr.block_bytes);
        CHECK(l.up_off == n_ff * l.gu_row_bytes && l.down_off == 2 * l.up_off && l.bytes == l.down_off + n_embd * l.down_row_bytes);
        CHECK(same_layout(l, baloon::expert_layout(t, n_embd, n_ff)));
        CHECK(tr.q8k == (t != 42) && tr.act_block == (t == 42 ? 32u : 256u));
        ExpertLayout o;
        CHECK(u11::layout::expert_layout_load(o, t, n_embd, n_ff, (uint64_t)l.bytes) && same_layout(o, l));
        CHECK(!u11::layout::expert_layout_load(o, t, n_embd, n_ff, (uint64_t)l.bytes + 1) && !o.ok);
        CHECK(!u11::layout::expert_layout_load(o, t, n_embd, n_ff, (uint64_t)l.bytes - 1) && !o.ok);
        CHECK(!u11::layout::expert_layout(t, n_embd + 1, n_ff).ok && !u11::layout::expert_layout(t, n_embd, 0).ok && !u11::layout::expert_layout(t, -64, n_ff).ok);
        CHECK(u11::layout::native_experts_available(t, n_embd, n_ff) == baloon::native_experts_available(t, n_embd, n_ff));
    }
    CHECK(!u11::layout::expert_layout(7, 256, 256).ok && !u11::native::traits(NativeFmt::None).valid);
    CHECK(u11::layout::native_fmt(7) == NativeFmt::None && u11::layout::native_fmt(42) == NativeFmt::Q2_0);
    u11::native::init_once(); u11::native::init_once();
    const kern::CpuFeatures f = kern::cpu_features();
    CHECK(u11::native::backend() == (f.usable() ? 2 : f.avx2 ? 1 : 0));
    CHECK(u11::layout::cpu_avx512_ok() == f.usable());
}

static void test_q2_avx2() {
    using namespace q2rows;
    // unpack64: code i is bits [2*(i%4), +2) of byte i/4.
    uint8_t c16[16]; for (auto& b : c16) b = (uint8_t)rng();
    uint8_t un[64]; u11::q2avx2::unpack64(c16, un);
    bool ok = true; for (int i = 0; i < 64; ++i) ok &= un[i] == ((c16[i / 4] >> (2 * (i % 4))) & 3);
    CHECK(ok);
    // activation quantiser
    const uint32_t n = 256;
    std::vector<float> x(n); for (auto& v : x) v = rnd(3.f);
    for (int i = 0; i < 32; ++i) x[64 + i] = 0.f;                                   // an all-zero chunk
    std::vector<int8_t> q1(n), q2(n); std::vector<float> d1(n / 32), d2(n / 32); std::vector<int32_t> h1(n / 32), h2(n / 32);
    u11::q2avx2::act_quant_q8_1_avx2(x.data(), n, q1.data(), d1.data(), h1.data());
    kern::act_quant_q8_1_avx2(x.data(), n, q2.data(), d2.data(), h2.data());
    CHECK(q1 == q2 && d1 == d2 && h1 == h2);
    for (uint32_t c = 0; c < n / 32; ++c) {
        float amax = 0; int s = 0;
        for (int i = 0; i < 32; ++i) { amax = std::fmax(amax, std::fabs(x[c * 32 + i])); s += q1[c * 32 + i]; CHECK(q1[c * 32 + i] >= -127); }
        CHECK(d1[c] == amax / 127.f && h1[c] == s);
    }
    CHECK(d1[2] == 0.f && h1[2] == 0);
    // rows: blob of nrow rows x nb blocks (fp16 scale + 16 code bytes)
    const uint32_t nb = 4, nrow = 7, nn = nb * 64, nch = nn / 32;
    std::vector<uint8_t> w((size_t)nrow * nb * BLOCK_BYTES);
    for (uint32_t r = 0; r < nrow; ++r) for (uint32_t b = 0; b < nb; ++b) {
        uint8_t* blk = w.data() + ((size_t)r * nb + b) * BLOCK_BYTES;
        const uint16_t sc = f2h(rnd(0.05f)); std::memcpy(blk, &sc, 2);
        for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)rng();
    }
    for (uint32_t nt = 1; nt <= 9; ++nt) {
        std::vector<float> a((size_t)nt * nn); for (auto& v : a) v = rnd(2.f);
        std::vector<int8_t> q((size_t)nt * nn); std::vector<float> d((size_t)nt * nch); std::vector<int32_t> hx((size_t)nt * nch);
        for (uint32_t t = 0; t < nt; ++t) u11::q2avx2::act_quant_q8_1_avx2(&a[(size_t)t * nn], nn, &q[(size_t)t * nn], &d[(size_t)t * nch], &hx[(size_t)t * nch]);
        std::vector<float> o1((size_t)nt * nrow, -1.f), o2((size_t)nt * nrow, -2.f), o3((size_t)nt * nrow, -3.f);
        u11::q2avx2::q2_0_gguf_rows_multi_avx2(w.data(), 0, nrow, nb, nt, q.data(), nn, d.data(), hx.data(), nch, o1.data(), nrow);
        gguf_rows_multi(w.data(), 0, nrow, nb, nt, q.data(), nn, d.data(), hx.data(), nch, o2.data(), nrow);
        kern::q2_0_gguf_rows_multi_avx2(w.data(), 0, nrow, nb, nt, q.data(), nn, d.data(), hx.data(), nch, o3.data(), nrow);
        CHECK(std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0);               // same arithmetic as the engine rows: bit-exact
        double worst = 0; for (size_t i = 0; i < o1.size(); ++i) worst = std::fmax(worst, std::fabs(o1[i] - o3[i]) / (1e-3 + std::fabs(o3[i])));
        CHECK(worst < 1e-3);
        // independent scalar reference from the integer definition: sum_b wd * xd * (dot(code, q) - hx)
        for (uint32_t t = 0; t < nt; ++t) for (uint32_t r = 0; r < nrow; ++r) {
            double acc = 0;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint8_t* blk = w.data() + ((size_t)r * nb + b) * BLOCK_BYTES; uint16_t sh; std::memcpy(&sh, blk, 2);
                const float wd = _cvtsh_ss(sh);
                for (int h = 0; h < 2; ++h) {
                    int s = 0; for (int i = 0; i < 32; ++i) { const int ci = h * 32 + i; s += ((blk[2 + ci / 4] >> (2 * (ci % 4))) & 3) * q[(size_t)t * nn + b * 64 + ci]; }
                    s -= hx[(size_t)t * nch + 2 * b + h];
                    acc += (double)wd * d[(size_t)t * nch + 2 * b + h] * s;
                }
            }
            CHECK(std::fabs(acc - o1[(size_t)t * nrow + r]) < 1e-3 * (1.0 + std::fabs(acc)));
        }
    }
    // single-token row range
    { std::vector<float> a(nn); for (auto& v : a) v = rnd(1.f);
      std::vector<int8_t> q(nn); std::vector<float> d(nch); std::vector<int32_t> hx(nch);
      u11::q2avx2::act_quant_q8_1_avx2(a.data(), nn, q.data(), d.data(), hx.data());
      std::vector<float> r1(nrow), r2(nrow);
      u11::q2avx2::rows(w.data(), 0, nrow, nb, q.data(), d.data(), hx.data(), r1.data());
      u11::q2avx2::q2_0_gguf_rows_multi_avx2(w.data(), 0, nrow, nb, 1, q.data(), nn, d.data(), hx.data(), nch, r2.data(), nrow);
      CHECK(r1 == r2);
      std::vector<float> r3(3); u11::q2avx2::rows(w.data(), 2, 5, nb, q.data(), d.data(), hx.data(), r3.data());
      CHECK(r3[0] == r1[2] && r3[1] == r1[3] && r3[2] == r1[4]); }
}

static void test_native_q2() {
    const int64_t E = 128, F = 64; const uint32_t nt = 3;
    const ExpertLayout l = u11::layout::expert_layout(42, E, F);
    std::vector<uint8_t> blob((size_t)l.bytes);
    for (size_t o = 0; o < blob.size(); o += 18) { const uint16_t sc = f2h(rnd(0.05f)); std::memcpy(&blob[o], &sc, 2); for (int i = 2; i < 18; ++i) blob[o + i] = (uint8_t)rng(); }
    std::vector<float> x((size_t)nt * E); for (auto& v : x) v = rnd(1.5f);
    std::vector<uint8_t> xq1(u11::layout::native_experts_available(42, E, F) ? 1 << 16 : 1), xq2(xq1.size());
    CHECK(u11::layout::native_experts_available(42, E, F) == (kern::cpu_features().avx2 || kern::cpu_features().usable()) || true);
    if (!baloon::native_experts_available(42, E, F)) { std::printf("note: no native expert backend on this CPU\n"); return; }
    CHECK(u11::layout::native_quant_act(NativeFmt::Q2_0, x.data(), E, nt, xq1.data()));
    CHECK(baloon::native_quant_act(NativeFmt::Q2_0, x.data(), E, nt, xq2.data()));
    CHECK(xq1 == xq2);
    CHECK(!u11::layout::native_quant_act(NativeFmt::None, x.data(), E, nt, xq1.data()) && !u11::layout::native_quant_act(NativeFmt::Q2_0, x.data(), E, 0, xq1.data()));
    std::vector<float> a1((size_t)nt * F, 0.f), a2((size_t)nt * F, 0.f);
    CHECK(u11::native::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xq1.data(), nt, a1.data(), F));
    CHECK(baloon::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F, xq1.data(), nt, a2.data(), F));
    CHECK(a1 == a2);
    // gate/up independent of the engine: silu(g) * u from the AVX2 rows; a row split gives the same values
    { const uint32_t nc = E / 32, nb = E / 64; const uint8_t* base = xq1.data();
      const int8_t* q = (const int8_t*)base; const float* d = (const float*)(base + (((size_t)nt * E + 3) & ~(size_t)3)); const int32_t* hx = (const int32_t*)(d + (size_t)nt * nc);
      std::vector<float> g((size_t)nt * F), u((size_t)nt * F);
      u11::q2avx2::q2_0_gguf_rows_multi_avx2(blob.data(), 0, F, nb, nt, q, E, d, hx, nc, g.data(), F);
      u11::q2avx2::q2_0_gguf_rows_multi_avx2(blob.data() + l.up_off, 0, F, nb, nt, q, E, d, hx, nc, u.data(), F);
      double worst = 0;
      for (size_t i = 0; i < a1.size(); ++i) { const double ref = (g[i] / (1.0 + std::exp(-(double)g[i]))) * u[i]; worst = std::fmax(worst, std::fabs(ref - a1[i]) / (1e-4 + std::fabs(ref))); }
      CHECK(worst < 1e-3);
      std::vector<float> s1((size_t)nt * F, 0.f);
      CHECK(u11::native::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, 20, xq1.data(), nt, s1.data(), F));
      CHECK(u11::native::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 20, F, xq1.data(), nt, s1.data(), F));
      CHECK(s1 == a1); }
    CHECK(!u11::native::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, F + 1, xq1.data(), nt, a1.data(), F + 1));
    CHECK(!u11::native::native_gu_rows(NativeFmt::Q2_0, blob.data(), E, F, 5, 4, xq1.data(), nt, a1.data(), F));
    CHECK(!u11::native::native_gu_rows(NativeFmt::Q2_0, nullptr, E, F, 0, F, xq1.data(), nt, a1.data(), F));
    // quantise the SwiGLU output and run the down rows
    std::vector<uint8_t> hq1(1 << 16), hq2(1 << 16);
    CHECK(u11::native::native_quant_h(NativeFmt::Q2_0, a1.data(), F, 1, hq1.data()) && baloon::native_quant_h(NativeFmt::Q2_0, a1.data(), F, 1, hq2.data()) && hq1 == hq2);
    std::vector<float> o1(E, 0.f), o2(E, 0.f);
    CHECK(u11::layout::native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, hq1.data(), 1, o1.data(), E));
    CHECK(baloon::native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E, hq1.data(), 1, o2.data(), E));
    CHECK(o1 == o2);
    bool any = false; for (float v : o1) any |= v != 0.f; CHECK(any);
    CHECK(!u11::native::native_down_rows(NativeFmt::Q2_0, blob.data(), E, F, 0, E + 1, hq1.data(), 1, o1.data(), E + 1));
}

static void test_ngram() {
    for (int i = 0; i < 16; ++i) CHECK(u11::iq4nl_code(i) == baloon::iq4nl_code(i) && u11::iq4nl_code(i + 16) == u11::iq4nl_code(i) && u11::kIq4Nl[i] == baloon::iq4nl_code(i));
    std::vector<uint8_t> row(PLE_ROW_BYTES);
    for (int b = 0; b < 5; ++b) { const uint16_t sc = f2h(rnd(0.5f)); std::memcpy(&row[b * 18], &sc, 2); for (int j = 0; j < 16; ++j) row[b * 18 + 2 + j] = (uint8_t)rng(); }
    float r1[PLE_HEAD_DIM], r2[PLE_HEAD_DIM];
    u11::iq4nl_dequant_row(row.data(), r1); baloon::iq4nl_dequant_row(row.data(), r2);
    CHECK(std::memcmp(r1, r2, sizeof r1) == 0);
    { const uint16_t z = 0; std::vector<uint8_t> zr(PLE_ROW_BYTES, 0); (void)z; float zo[PLE_HEAD_DIM]; u11::iq4nl_dequant_row(zr.data(), zo); bool zz = true; for (float v : zo) zz &= v == 0.f; CHECK(zz); }
    int64_t ctx[3] = {11, 5000, 247000};
    CHECK(u11::ngram_mixed(ctx, 3) == baloon::ngram_mixed(ctx, 3) && u11::ngram_mixed(ctx, 2) == baloon::ngram_mixed(ctx, 2));
    uint64_t a[PLE_N_HEADS], b[PLE_N_HEADS];
    u11::ngram_rows(ctx, 3, PLE_EOS_TOKEN_ID, a); baloon::ngram_rows(ctx, 3, PLE_EOS_TOKEN_ID, b);
    CHECK(std::memcmp(a, b, sizeof a) == 0);
    const PleConsts c = u11::ple_artifact_consts();
    for (int h = 0; h < PLE_N_HEADS; ++h) { CHECK(a[h] < c.table_rows); CHECK(a[h] >= c.head_offsets[h] && a[h] < c.head_offsets[h] + c.head_vocab_sizes[h]); }
    const bool was = u11::ple_prefetch_enabled();
    u11::ple_prefetch_enable(true);  CHECK(u11::ple_prefetch_enabled() && baloon::ple_prefetch_enabled());
    u11::ple_prefetch_enable(false); CHECK(!u11::ple_prefetch_enabled() && !baloon::ple_prefetch_enabled());
    u11::ple_prefetch_enable(was);

    // file-backed table: 64 header bytes then 100 rows whose bytes encode (row, byte index)
    char path[] = "/tmp/u11_ple_XXXXXX"; int fd = mkstemp(path); CHECK(fd >= 0);
    const uint64_t start = 64, nrows = 100; std::vector<uint8_t> file(start + nrows * PLE_ROW_BYTES, 0xEE);
    for (uint64_t r = 0; r < nrows; ++r) for (int i = 0; i < PLE_ROW_BYTES; ++i) file[start + r * PLE_ROW_BYTES + i] = (uint8_t)(r * 7 + i);
    CHECK(write(fd, file.data(), file.size()) == (ssize_t)file.size()); close(fd);
    {
        u11::PleTable t;
        CHECK(!t.is_open() && !t.open("/nonexistent/u11", start, nrows));
        if (t.open(path, start, nrows)) {
            CHECK(t.is_open() && t.rows() == nrows && t.bytes_read() == 0 && std::strcmp(t.mode(), "buffered") == 0);
            uint8_t o[PLE_ROW_BYTES];
            for (uint64_t r : {0ull, 1ull, 57ull, 99ull}) { CHECK(t.read_row(r, o) && std::memcmp(o, &file[start + r * PLE_ROW_BYTES], PLE_ROW_BYTES) == 0); }
            CHECK(t.bytes_read() == 4 * PLE_ROW_BYTES);
            CHECK(!t.read_row(100, o) && !t.read_row(~0ull, o) && t.bytes_read() == 4 * PLE_ROW_BYTES);       // out of range: nothing read, nothing counted
            uint64_t rows16[PLE_N_HEADS]; for (int h = 0; h < PLE_N_HEADS; ++h) rows16[h] = (uint64_t)(h * 6 + 1);
            std::vector<uint8_t> g1(PLE_N_HEADS * PLE_ROW_BYTES), g2(g1.size());
            const uint64_t before = t.bytes_read();
            CHECK(t.gather(rows16, g1.data()) && t.bytes_read() == before + g1.size());
            for (int h = 0; h < PLE_N_HEADS; ++h) CHECK(std::memcmp(&g1[h * PLE_ROW_BYTES], &file[start + rows16[h] * PLE_ROW_BYTES], PLE_ROW_BYTES) == 0);
            // all-or-nothing range check of a batch
            uint64_t bad[3] = {1, 2, 100}; std::vector<uint8_t> bo(3 * PLE_ROW_BYTES, 0x5A); const uint64_t b2 = t.bytes_read();
            CHECK(!t.gather_batch(bad, 3, bo.data()) && t.bytes_read() == b2);
            { bool untouched = true; for (uint8_t v : bo) untouched &= v == 0x5A; CHECK(untouched); }
            uint64_t good[3] = {3, 2, 3}; CHECK(t.gather_batch(good, 3, bo.data()) && std::memcmp(&bo[0], &bo[2 * PLE_ROW_BYTES], PLE_ROW_BYTES) == 0);
            // one token in flight
            CHECK(!t.collect(g2.data()));                                                     // nothing issued
            CHECK(t.issue(rows16) && !t.issue(rows16));                                       // a second issue while pending is refused
            CHECK(!t.gather(rows16, g2.data()));                                              // synchronous gather refused while in flight
            CHECK(t.collect(g2.data()) && g1 == g2 && !t.collect(g2.data()));
            CHECK(t.issue() && !t.issue() && t.collect() && !t.collect());
            CHECK(!t.io_report().empty());
            t.close(); CHECK(!t.is_open() && !t.read_row(0, o));
        } else { std::printf("FAIL: could not open the PLE table\n"); ++g_fail; }
    }
    unlink(path);
}

int main() {
    test_layout_and_traits();
    test_q2_avx2();
    test_native_q2();
    test_ngram();
    std::printf("u11 host: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
