// Unit-14 CPU kernel tests: Q2_0 expert paths, activation quantizers, GGUF rows, i-quant rows, expert pool.
// Built with the AVX-512 ISA flags and -ffp-contract=off. Exits non-zero on any failed check.
#include <immintrin.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/kern_cpu_pool.hpp"
#include "baloon/q2_expert_vnni_kernel.hpp"

using namespace baloon::kern;

static int g_fail = 0, g_checks = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
    } while (0)

static std::mt19937 rng(12345);
static float urand(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
static float h2fl(uint16_t h) { return _cvtsh_ss(h); }

static std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b(BLOB);
    for (uint64_t i = 0; i < O_GU_SCALES; ++i) b[i] = (uint8_t)rng();
    for (uint64_t i = O_GU_SCALES; i < BLOB; i += 2) { const uint16_t h = f2h(urand(0.005f, 0.05f)); std::memcpy(&b[i], &h, 2); }
    return b;
}
static std::vector<float> make_x(size_t n, float s = 1.f) { std::vector<float> x(n); for (auto& v : x) v = urand(-s, s); return x; }

static bool bit_equal(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

static void test_features() {
    const CpuFeatures f = cpu_features();
    CHECK(f.usable());
    CHECK(std::strlen(f.reason()) == 0);
    for (int miss = 0; miss < 5; ++miss) {
        CpuFeatures g;
        g.avx512f = g.avx512bw = g.avx512vl = g.avx512_vnni = g.avx512_vbmi = true;
        (miss == 0 ? g.avx512f : miss == 1 ? g.avx512bw : miss == 2 ? g.avx512vl : miss == 3 ? g.avx512_vnni : g.avx512_vbmi) = false;
        CHECK(!g.usable());
        CHECK(std::strlen(g.reason()) > 0);
    }
    cpu_require_expert_support();   // aborts on an unsupported machine
}

static void test_geometry_consts() {
    CHECK(H == 2560 && FF == 640 && NE == 512 && QK == 64 && QKA == 32 && BB == 18 && MAXC == 80 && MAXT == 8);
    CHECK(ROW_GU == 640 && ROW_D == 160 && SC_GU == 40 && SC_D == 10 && BLOB == 1382400);
    CHECK(O_GU_CODES == 0 && O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200);
    CHECK(sizeof(ActQ) == H + MAXC * 4 + MAXC * 4);
}

static void test_act_quant() {
    for (int rep = 0; rep < 200; ++rep) {
        const uint32_t n = 32 * (1 + rng() % 80);
        std::vector<float> x(n);
        for (auto& v : x) {
            switch (rng() % 6) {
                case 0: v = 0.f; break;
                case 1: v = (float)((int)(rng() % 9) - 4) * 0.5f; break;   // exact ties after scaling
                case 2: v = urand(-1e4f, 1e4f); break;
                default: v = urand(-1.f, 1.f); break;
            }
        }
        std::vector<int8_t> q1(n), q2(n);
        std::vector<float> d1(n / 32), d2(n / 32);
        std::vector<int32_t> h1(n / 32), h2(n / 32);
        act_quant_q8_1(x.data(), n, q1.data(), d1.data(), h1.data());
        act_quant_q8_1_avx2(x.data(), n, q2.data(), d2.data(), h2.data());
        CHECK(q1 == q2);
        CHECK(bit_equal(d1.data(), d2.data(), n / 32));
        CHECK(h1 == h2);
        bool inr = true;
        for (uint32_t c = 0; c < n / 32; ++c) {
            int s = 0;
            for (int i = 0; i < 32; ++i) { inr &= q1[c * 32 + i] >= -127; s += q1[c * 32 + i]; }
            inr &= (s == h1[c]);
        }
        CHECK(inr);
    }
    // all-zero chunk: d = 0, q = 0
    std::vector<float> z(64, 0.f);
    int8_t q[64]; float d[2]; int32_t h[2];
    act_quant_q8_1(z.data(), 64, q, d, h);
    CHECK(d[0] == 0.f && h[0] == 0 && q[5] == 0);
}

static void test_expert() {
    const auto blob = make_blob();
    ExpertScratch* sc = new ExpertScratch;
    ExpertScratchMulti* sm = new ExpertScratchMulti;
    for (int rep = 0; rep < 3; ++rep) {
        const auto x = make_x(H, rep == 2 ? 30.f : 1.f);
        std::vector<float> o_sc(H), o_v(H), o_ref(H);
        s2_expert_scalar(blob.data(), x.data(), o_sc.data());
        s2_expert_vnni(blob.data(), x.data(), o_v.data(), *sc);
        CHECK(bit_equal(o_sc.data(), o_v.data(), H));
        {   // the older reference header keeps the gate/up scales right after the gate/up codes: relocate the planes
            std::vector<uint8_t> old(BLOB);
            const uint64_t gu_c = O_D_CODES, d_c = O_GU_SCALES - O_D_CODES, gu_s = O_D_SCALES - O_GU_SCALES, d_s = BLOB - O_D_SCALES;
            std::memcpy(&old[0], &blob[O_GU_CODES], gu_c);
            std::memcpy(&old[gu_c], &blob[O_GU_SCALES], gu_s);
            std::memcpy(&old[gu_c + gu_s], &blob[O_D_CODES], d_c);
            std::memcpy(&old[gu_c + gu_s + d_c], &blob[O_D_SCALES], d_s);
            baloon::q2_expert::expert_forward(old.data(), x.data(), o_ref.data());
        }
        double worst = 0;
        for (uint32_t i = 0; i < H; ++i) worst = std::max(worst, (double)std::fabs(o_ref[i] - o_v[i]) / (std::fabs(o_ref[i]) + 1e-3));
        CHECK(worst < 1e-3);
        // pre-quantized entry point
        ActQ xq;
        act_quant_q8_1(x.data(), H, xq.q, xq.d, xq.hx);
        std::vector<float> o_q(H);
        s2_expert_vnni_q(blob.data(), xq, o_q.data(), *sc);
        CHECK(bit_equal(o_q.data(), o_v.data(), H));
    }
    // multi token: every width 1..8 equals token-by-token; 9 and 11 fall back and still equal
    const auto xs = make_x((size_t)11 * H);
    std::vector<float> single((size_t)11 * H);
    for (int t = 0; t < 11; ++t) s2_expert_scalar(blob.data(), xs.data() + (size_t)t * H, single.data() + (size_t)t * H);
    for (uint32_t nt : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u}) {
        std::vector<float> o((size_t)nt * H, -1.f);
        s2_expert_vnni_multi(blob.data(), xs.data(), nt, o.data(), *sm);
        CHECK(bit_equal(o.data(), single.data(), (size_t)nt * H));
    }
    // row-range pieces tile: gu rows [0,300) + [300,640) == whole
    ActQ xq;
    act_quant_q8_1(xs.data(), H, xq.q, xq.d, xq.hx);
    std::vector<float> a1(FF), a2(FF);
    s2_expert_gu_rows(blob.data(), xq, 0, FF, a1.data());
    s2_expert_gu_rows(blob.data(), xq, 0, 300, a2.data());
    s2_expert_gu_rows(blob.data(), xq, 300, FF, a2.data());
    CHECK(bit_equal(a1.data(), a2.data(), FF));
    // oracle contract: quantization changes (fp16 scales), and vnni still equals the scalar oracle path
    expert_set_oracle_q8_0(true);
    CHECK(expert_oracle_q8_0_enabled());
    {
        std::vector<float> o_sc(H), o_v(H);
        s2_expert_scalar(blob.data(), xs.data(), o_sc.data());
        s2_expert_vnni(blob.data(), xs.data(), o_v.data(), *sc);
        CHECK(bit_equal(o_sc.data(), o_v.data(), H));
        int8_t q[32]; float d[1]; int32_t h[1];
        act_quant_q8_1(xs.data(), 32, q, d, h);
        CHECK(h2fl(f2h(d[0])) == d[0]);   // scale is exactly an fp16 value
        std::vector<float> fast(H);
        expert_set_oracle_q8_0(false);
        s2_expert_scalar(blob.data(), xs.data(), fast.data());
        CHECK(!bit_equal(fast.data(), o_sc.data(), H));   // the oracle form really is a different quantization
    }
    expert_set_oracle_q8_0(false);
    CHECK(!expert_oracle_q8_0_enabled());
    delete sc; delete sm;
}

// Independent model of GGUF rows: double accumulation of (c-1) * w_scale * xhat * x_scale.
static void test_gguf_rows() {
    const uint32_t nb = 40, rows = 13;
    std::vector<uint8_t> w((size_t)rows * nb * BB);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t b = 0; b < nb; ++b) {
            uint8_t* blk = &w[((size_t)r * nb + b) * BB];
            const uint16_t h = f2h(urand(0.01f, 0.1f));
            std::memcpy(blk, &h, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t)rng();
        }
    for (uint32_t nt : {1u, 2u, 5u, 8u, 9u, 11u}) {
        const auto x = make_x((size_t)nt * nb * 64);
        std::vector<int8_t> q((size_t)nt * nb * 64);
        std::vector<float> xd((size_t)nt * nb * 2);
        std::vector<int32_t> hx((size_t)nt * nb * 2);
        for (uint32_t t = 0; t < nt; ++t) act_quant_q8_1(x.data() + (size_t)t * nb * 64, nb * 64, q.data() + (size_t)t * nb * 64, xd.data() + (size_t)t * nb * 2, hx.data() + (size_t)t * nb * 2);
        std::vector<float> o1((size_t)nt * rows, -7.f), o2((size_t)nt * rows, -7.f);
        q2_0_gguf_rows_multi(w.data(), 0, rows, nb, nt, q.data(), (size_t)nb * 64, xd.data(), hx.data(), (size_t)nb * 2, o1.data(), rows);
        q2_0_gguf_rows_multi_avx2(w.data(), 0, rows, nb, nt, q.data(), (size_t)nb * 64, xd.data(), hx.data(), (size_t)nb * 2, o2.data(), rows);
        CHECK(bit_equal(o1.data(), o2.data(), (size_t)nt * rows));
        double worst = 0;
        for (uint32_t t = 0; t < nt; ++t)
            for (uint32_t r = 0; r < rows; ++r) {
                double e = 0, mag = 0;
                for (uint32_t b = 0; b < nb; ++b) {
                    const uint8_t* blk = &w[((size_t)r * nb + b) * BB];
                    uint16_t h; std::memcpy(&h, blk, 2);
                    const double wd = h2fl(h);
                    for (uint32_t j = 0; j < 64; ++j) {
                        const int c = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
                        const size_t ch = (size_t)t * nb * 2 + 2 * b + j / 32;
                        const double v = wd * (c - 1) * q[(size_t)t * nb * 64 + b * 64 + j] * xd[ch];
                        e += v; mag += std::fabs(v);
                    }
                }
                worst = std::max(worst, std::fabs(e - o1[(size_t)t * rows + r]) / (mag + 1e-9));
            }
        CHECK(worst < 1e-5);
    }
    // sub-range: rows [3,9) only touch those outputs
    const auto x = make_x(nb * 64);
    std::vector<int8_t> q(nb * 64); std::vector<float> xd(nb * 2); std::vector<int32_t> hx(nb * 2);
    act_quant_q8_1(x.data(), nb * 64, q.data(), xd.data(), hx.data());
    std::vector<float> all(rows), part(rows, 99.f);
    q2_0_gguf_rows_multi(w.data(), 0, rows, nb, 1, q.data(), 0, xd.data(), hx.data(), 0, all.data(), rows);
    q2_0_gguf_rows_multi(w.data(), 3, 9, nb, 1, q.data(), 0, xd.data(), hx.data(), 0, part.data(), rows);
    for (uint32_t r = 0; r < rows; ++r) CHECK(r >= 3 && r < 9 ? part[r] == all[r] : part[r] == 99.f);
}

// ---- i-quant rows against an element-wise model ---------------------------------------------------------------
struct Grids {
    std::vector<uint64_t> g2xxs, g2xs, g2s;
    std::vector<uint32_t> g3xxs, g3s;
    Grids() : g2xxs(256), g2xs(512), g2s(1024), g3xxs(256), g3s(512) {
        auto m8 = [&](int mx) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(rng() % (mx + 1)) << (8 * i); return v; };
        auto m4 = [&]() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)(1 + 2 * (rng() % 8)) << (8 * i); return v; };
        for (auto& v : g2xxs) v = m8(43);
        for (auto& v : g2xs) v = m8(43);
        for (auto& v : g2s) v = m8(43);
        for (auto& v : g3xxs) v = m4();
        for (auto& v : g3s) v = m4();
    }
};
static uint8_t ks(uint32_t i) { return (uint8_t)(i | ((__builtin_popcount(i) & 1) << 7)); }

// Dequantized values (scale-free of d) of one 256 block: val[i] = K/8 * scale(i) * sign * mag, written from the format
// definitions (ggml block layouts), independent of the kernel's chunk decode.
static void model_block(const Grids& g, uint32_t ty, const uint8_t* b, float* out) {
    uint16_t hd; std::memcpy(&hd, b, 2);
    const float d = h2fl(hd);
    for (int ib = 0; ib < 8; ++ib) {
        float v[32]; int sc[2];
        if (ty == 16) {
            const uint8_t* p = b + 2 + 8 * ib; uint32_t a1; std::memcpy(&a1, p + 4, 4);
            for (int l = 0; l < 4; ++l) { const uint64_t gr = g.g2xxs[p[l]]; const uint8_t sg = ks((a1 >> (7 * l)) & 127);
                for (int i = 0; i < 8; ++i) v[8 * l + i] = ((gr >> (8 * i)) & 0xff) * ((sg >> i) & 1 ? -1.f : 1.f); }
            sc[0] = sc[1] = 2 * (a1 >> 28) + 1;
        } else if (ty == 17) {
            for (int l = 0; l < 4; ++l) { uint16_t w; std::memcpy(&w, b + 2 + 8 * ib + 2 * l, 2); const uint64_t gr = g.g2xs[w & 511]; const uint8_t sg = ks(w >> 9);
                for (int i = 0; i < 8; ++i) v[8 * l + i] = ((gr >> (8 * i)) & 0xff) * ((sg >> i) & 1 ? -1.f : 1.f); }
            const uint8_t s = b[66 + ib]; sc[0] = 2 * (s & 15) + 1; sc[1] = 2 * (s >> 4) + 1;
        } else if (ty == 22) {
            const uint8_t* qs = b + 2 + 4 * ib; const uint8_t* sg = b + 34 + 4 * ib; const uint32_t qh = b[66 + ib];
            for (int l = 0; l < 4; ++l) { const uint64_t gr = g.g2s[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
                for (int i = 0; i < 8; ++i) v[8 * l + i] = ((gr >> (8 * i)) & 0xff) * ((sg[l] >> i) & 1 ? -1.f : 1.f); }
            const uint8_t s = b[74 + ib]; sc[0] = 2 * (s & 15) + 1; sc[1] = 2 * (s >> 4) + 1;
        } else if (ty == 18) {
            const uint8_t* qs = b + 2 + 8 * ib; uint32_t a; std::memcpy(&a, b + 66 + 4 * ib, 4);
            for (int l = 0; l < 4; ++l) { const uint8_t sg = ks((a >> (7 * l)) & 127);
                for (int h = 0; h < 2; ++h) { const uint32_t gr = g.g3xxs[qs[2 * l + h]];
                    for (int i = 0; i < 4; ++i) v[8 * l + 4 * h + i] = ((gr >> (8 * i)) & 0xff) * ((sg >> (4 * h + i)) & 1 ? -1.f : 1.f); } }
            sc[0] = sc[1] = 2 * (a >> 28) + 1;
        } else {   // 21
            const uint8_t* qs = b + 2 + 8 * ib; const uint32_t qh = b[66 + ib]; const uint8_t* sg = b + 74 + 4 * ib;
            for (int l = 0; l < 4; ++l)
                for (int h = 0; h < 2; ++h) { const uint32_t gr = g.g3s[qs[2 * l + h] | ((qh << (8 - 2 * l - h)) & 256)];
                    for (int i = 0; i < 4; ++i) v[8 * l + 4 * h + i] = ((gr >> (8 * i)) & 0xff) * ((sg[l] >> (4 * h + i)) & 1 ? -1.f : 1.f); }
            const uint8_t s = b[106 + ib / 2]; sc[0] = sc[1] = 2 * ((ib & 1) ? (s >> 4) : (s & 15)) + 1;
        }
        const float K = ty == 18 ? 0.25f : ty == 21 ? 1.f : 0.125f;
        for (int i = 0; i < 32; ++i) out[32 * ib + i] = d * K * sc[i / 16] * v[i];
    }
}

static void test_iq_rows() {
    Grids g;
    IqTables t;
    t.iq2xxs_grid = g.g2xxs.data(); t.iq2xs_grid = g.g2xs.data(); t.iq2s_grid = g.g2s.data();
    t.iq3xxs_grid = g.g3xxs.data(); t.iq3s_grid = g.g3s.data();
    iq512_set_tables(t);
    for (uint32_t ty : {16u, 17u, 18u, 21u, 22u}) {
        CHECK(iq512_supported(ty));
        CHECK(iq512_tables_ready(ty));
        const uint32_t bytes = iqgeo::block_bytes(ty), nb = 4, rows = 7;
        std::vector<uint8_t> w((size_t)rows * nb * bytes);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t b = 0; b < nb; ++b) {
                uint8_t* blk = &w[((size_t)r * nb + b) * bytes];
                for (uint32_t i = 0; i < bytes; ++i) blk[i] = (uint8_t)rng();
                const uint16_t h = f2h(urand(0.01f, 0.1f)); std::memcpy(blk, &h, 2);
            }
        for (uint32_t nt : {1u, 2u, 3u, 4u, 5u, 7u, 9u}) {
            std::vector<float> x((size_t)nt * nb * 256);
            for (auto& v : x) v = urand(-2.f, 2.f);
            std::vector<Q8KBlock> acts((size_t)nt * nb);
            for (uint32_t tk = 0; tk < nt; ++tk) baloon::iquant512::q8k_quantize(x.data() + (size_t)tk * nb * 256, nb * 256, acts.data() + (size_t)tk * nb);
            std::vector<float> out((size_t)nt * rows, 5.f);
            iq512_rows(ty, w.data(), (size_t)nb * bytes, 0, rows, nb, acts.data(), nt, out.data(), rows);
            double worst = 0;
            for (uint32_t tk = 0; tk < nt; ++tk)
                for (uint32_t r = 0; r < rows; ++r) {
                    double e = 0, mag = 0;
                    for (uint32_t b = 0; b < nb; ++b) {
                        float val[256];
                        model_block(g, ty, &w[((size_t)r * nb + b) * bytes], val);
                        const Q8KBlock& a = acts[(size_t)tk * nb + b];
                        for (int i = 0; i < 256; ++i) { const double v = (double)val[i] * a.qs[i] * a.d; e += v; mag += std::fabs(v); }
                    }
                    worst = std::max(worst, std::fabs(e - out[(size_t)tk * rows + r]) / (mag + 1e-9));
                }
            CHECK(worst < 1e-5);
            if (worst >= 1e-5) std::printf("  iq type %u nt %u worst %.3g\n", ty, nt, worst);
        }
        // gate/up pair: layout gate rows | up rows
        const uint32_t n_ff = 4;
        std::vector<uint8_t> w2((size_t)2 * n_ff * nb * bytes);
        for (auto& v : w2) v = (uint8_t)rng();
        for (uint32_t r = 0; r < 2 * n_ff; ++r)
            for (uint32_t b = 0; b < nb; ++b) { const uint16_t h = f2h(urand(0.01f, 0.1f)); std::memcpy(&w2[((size_t)r * nb + b) * bytes], &h, 2); }
        const uint32_t nt = 3;
        std::vector<float> x((size_t)nt * nb * 256);
        for (auto& v : x) v = urand(-1.f, 1.f);
        std::vector<Q8KBlock> acts((size_t)nt * nb);
        for (uint32_t tk = 0; tk < nt; ++tk) baloon::iquant512::q8k_quantize(x.data() + (size_t)tk * nb * 256, nb * 256, acts.data() + (size_t)tk * nb);
        std::vector<float> gg((size_t)nt * 2 * n_ff), act((size_t)nt * n_ff, 0.f);
        iq512_rows(ty, w2.data(), (size_t)nb * bytes, 0, 2 * n_ff, nb, acts.data(), nt, gg.data(), 2 * n_ff);
        iq512_gu_rows(ty, w2.data(), (size_t)nb * bytes, n_ff, 0, n_ff, nb, acts.data(), nt, act.data(), n_ff);
        bool ok = true;
        for (uint32_t tk = 0; tk < nt; ++tk)
            for (uint32_t r = 0; r < n_ff; ++r) {
                const float gv = gg[(size_t)tk * 2 * n_ff + r], uv = gg[(size_t)tk * 2 * n_ff + n_ff + r];
                const float e = gv / (1.f + std::exp(-gv)) * uv;
                ok &= std::fabs(e - act[(size_t)tk * n_ff + r]) <= 1e-5f * (std::fabs(e) + 1e-3f);
            }
        CHECK(ok);
    }
    CHECK(!iq512_supported(15) && !iq512_supported(19) && !iq512_supported(23));
}

// ---- pool -----------------------------------------------------------------------------------------------------
static void test_pool() {
    CHECK(!physical_cores().empty());
    CHECK(pin_current_thread(physical_cores()[0]) || true);
    restore_thread_affinity();
    const auto blob = make_blob();
    const uint32_t NJ = 6;
    std::vector<std::vector<float>> xs(NJ);
    for (auto& x : xs) x = make_x(H);
    std::vector<std::vector<float>> want(NJ, std::vector<float>(H));
    for (uint32_t j = 0; j < NJ; ++j) s2_expert_scalar(blob.data(), xs[j].data(), want[j].data());
    for (auto cfg : std::vector<std::tuple<uint32_t, bool, bool>>{{1, false, false}, {2, false, true}, {7, false, false}, {3, true, true}, {0, true, false}}) {
        ExpertPool pool(std::get<0>(cfg), std::get<1>(cfg), std::get<2>(cfg));
        CHECK(pool.workers() >= 1);
        if (std::get<0>(cfg) > 0) CHECK(pool.workers() == std::get<0>(cfg));
        std::vector<ExpertScratch*> scr(NJ);
        for (auto& s : scr) s = new ExpertScratch;
        std::vector<std::vector<float>> out(NJ, std::vector<float>(H));
        std::vector<ExpertJob> jobs(NJ);
        for (uint32_t j = 0; j < NJ; ++j) jobs[j] = {blob.data(), xs[j].data(), out[j].data(), scr[j]};
        for (int rep = 0; rep < 40; ++rep) {   // many batches: the park / publish protocol must never lose or repeat a task
            for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
            pool.run(jobs.data(), NJ);
            for (uint32_t j = 0; j < NJ; ++j) CHECK(bit_equal(out[j].data(), want[j].data(), H));
            for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
            CHECK(pool.run_split(jobs.data(), 1 + rep % 6));
            for (uint32_t j = 0; j < 1 + (uint32_t)(rep % 6); ++j) CHECK(bit_equal(out[j].data(), want[j].data(), H));
        }
        std::vector<ExpertJob> many(ExpertPool::kMaxSplit + 1, jobs[0]);
        CHECK(!pool.run_split(many.data(), (uint32_t)many.size()));   // beyond capacity: refused, nothing done
        for (auto& s : scr) delete s;
    }
    // multi token, different token counts per expert, experts spanning task boundaries
    {
        ExpertPool pool(5, false, true);
        const uint32_t nts[4] = {1, 8, 3, 5};
        std::vector<std::vector<float>> in(4), out(4), ref(4);
        std::vector<ExpertScratchMulti*> sm(4);
        std::vector<ExpertJobMulti> jobs(4);
        for (int e = 0; e < 4; ++e) {
            in[e] = make_x((size_t)nts[e] * H);
            out[e].assign((size_t)nts[e] * H, -1.f);
            ref[e].resize((size_t)nts[e] * H);
            for (uint32_t t = 0; t < nts[e]; ++t) s2_expert_scalar(blob.data(), in[e].data() + (size_t)t * H, ref[e].data() + (size_t)t * H);
            sm[e] = new ExpertScratchMulti;
            jobs[e].blob = blob.data(); jobs[e].x = in[e].data(); jobs[e].nt = nts[e]; jobs[e].out = out[e].data(); jobs[e].scratch = sm[e];
        }
        for (int rep = 0; rep < 10; ++rep) {
            for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
            CHECK(pool.run_split_multi(jobs.data(), 4));
            for (int e = 0; e < 4; ++e) CHECK(bit_equal(out[e].data(), ref[e].data(), (size_t)nts[e] * H));
        }
        expert_set_oracle_q8_0(true);
        CHECK(!pool.run_split_multi(jobs.data(), 4));   // oracle contract never uses the split path
        expert_set_oracle_q8_0(false);
        CHECK(!pool.run_split_multi_native(jobs.data(), 4));   // Q2_0 jobs are not native
        for (auto s : sm) delete s;
    }
    // native i-quant experts through the pool
    {
        Grids g;
        IqTables t;
        t.iq2xxs_grid = g.g2xxs.data(); t.iq2xs_grid = g.g2xs.data(); t.iq2s_grid = g.g2s.data();
        t.iq3xxs_grid = g.g3xxs.data(); t.iq3s_grid = g.g3s.data();
        iq512_set_tables(t);
        const uint32_t n_embd = 512, n_ff = 256, E = 3, gt = 17, dt = 22;
        const size_t gu_rb = (size_t)(n_embd / 256) * iqgeo::block_bytes(gt), d_rb = (size_t)(n_ff / 256) * iqgeo::block_bytes(dt);
        std::vector<std::vector<uint8_t>> guw(E), dw(E);
        std::vector<std::vector<float>> in(E), out(E), ref(E);
        const uint32_t nts[3] = {2, 6, 1};
        std::vector<ExpertJobMulti> jobs(E);
        for (uint32_t e = 0; e < E; ++e) {
            guw[e].resize(2 * n_ff * gu_rb); dw[e].resize(n_embd * d_rb);
            for (auto& v : guw[e]) v = (uint8_t)rng();
            for (auto& v : dw[e]) v = (uint8_t)rng();
            for (uint32_t r = 0; r < 2 * n_ff; ++r) { const uint16_t h = f2h(urand(0.01f, 0.05f)); for (uint32_t b = 0; b < n_embd / 256; ++b) std::memcpy(&guw[e][r * gu_rb + b * iqgeo::block_bytes(gt)], &h, 2); }
            for (uint32_t r = 0; r < n_embd; ++r) { const uint16_t h = f2h(urand(0.01f, 0.05f)); std::memcpy(&dw[e][r * d_rb], &h, 2); }
            in[e] = make_x((size_t)nts[e] * n_embd);
            out[e].assign((size_t)nts[e] * n_embd, -1.f);
            ref[e].assign((size_t)nts[e] * n_embd, 0.f);
            // single-thread reference with the same building blocks
            std::vector<Q8KBlock> x8((size_t)nts[e] * 2), a8(nts[e]);
            for (uint32_t tk = 0; tk < nts[e]; ++tk) baloon::iquant512::q8k_quantize(in[e].data() + (size_t)tk * n_embd, n_embd, x8.data() + (size_t)tk * 2);
            std::vector<float> act((size_t)nts[e] * n_ff);
            iq512_gu_rows(gt, guw[e].data(), gu_rb, n_ff, 0, n_ff, 2, x8.data(), nts[e], act.data(), n_ff);
            for (uint32_t tk = 0; tk < nts[e]; ++tk) baloon::iquant512::q8k_quantize(act.data() + (size_t)tk * n_ff, n_ff, a8.data() + tk);
            iq512_rows(dt, dw[e].data(), d_rb, 0, n_embd, 1, a8.data(), nts[e], ref[e].data(), n_embd);
            jobs[e].gu_type = gt; jobs[e].d_type = dt; jobs[e].gu_w = guw[e].data(); jobs[e].d_w = dw[e].data();
            jobs[e].gu_row_bytes = gu_rb; jobs[e].d_row_bytes = d_rb; jobs[e].n_ff = n_ff; jobs[e].n_embd = n_embd;
            jobs[e].nt = nts[e]; jobs[e].x = in[e].data(); jobs[e].out = out[e].data();
        }
        ExpertPool pool(4, false, true);
        for (int rep = 0; rep < 10; ++rep) {
            for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
            CHECK(pool.run_split_multi_native(jobs.data(), E));
            for (uint32_t e = 0; e < E; ++e) CHECK(bit_equal(out[e].data(), ref[e].data(), (size_t)nts[e] * n_embd));
        }
    }
}

int main() {
    test_features();
    test_geometry_consts();
    test_act_quant();
    test_expert();
    test_gguf_rows();
    test_iq_rows();
    test_pool();
    std::printf("checks: %d, failures: %d\n", g_checks, g_fail);
    if (g_fail) { std::printf("CPU KERNEL TESTS FAILED\n"); return 1; }
    std::printf("CPU KERNEL TESTS PASSED\n");
    return 0;
}
