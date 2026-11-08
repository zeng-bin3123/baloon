// GPU tests of the unit-14 misc kernels: dequantization, activation quantizers, S-family GEMV, sampler, resident S2 experts
// (against the CPU Q2_0 expert kernel), PLE block. Each device result is compared with an independent host model.
#include "../k14gpu_common.h"

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kern_ple.hpp"

using namespace baloon::kern;
namespace sdq = baloon::sdq;

static void put_h(uint8_t* p, float f) { const uint16_t h = h_of(f); p[0] = (uint8_t)(h & 255); p[1] = (uint8_t)(h >> 8); }
static void host_deq(int type, const uint8_t* b, float* y) {
    switch (type) {
        case 2: sdq::deq_q4_0(b, y); break; case 6: sdq::deq_q5_0(b, y); break; case 8: sdq::deq_q8_0(b, y); break;
        case 20: sdq::deq_iq4_nl(b, y); break; case 11: sdq::deq_q3_k(b, y); break; case 12: sdq::deq_q4_k(b, y); break;
        case 13: sdq::deq_q5_k(b, y); break; case 14: sdq::deq_q6_k(b, y); break; case 23: sdq::deq_iq4_xs(b, y); break;
        case 42: sdq::deq_q2_0(b, y); break;
    }
}
static void rand_block(Rng& r, int type, uint8_t* b, int bytes) {
    for (int i = 0; i < bytes; ++i) b[i] = (uint8_t)r.next();
    switch (type) {
        case 2: case 6: case 8: case 20: case 42: case 23: put_h(b, 0.01f + (float)r.uni() * 0.05f); break;
        case 11: put_h(b + 108, 0.01f + (float)r.uni() * 0.02f); break;
        case 12: case 13: put_h(b, 0.01f + (float)r.uni() * 0.02f); put_h(b + 2, 0.01f + (float)r.uni() * 0.02f); break;
        case 14: put_h(b + 208, 0.005f + (float)r.uni() * 0.01f); break;
    }
}

// ============================================================================================== dequant
static void test_dequant() {
    Rng r(11);
    const int types[] = {2, 6, 8, 20, 11, 12, 13, 14, 23, 42};
    for (int type : types) {
        const auto* we = baloon::kern::mmvqgeo::find_weight(type);
        const int be = (int)we->elems, bb = (int)we->bytes, nb = 9;
        const int64_t n = (int64_t)nb * be;
        CHECK(dequant_bf16_supported(type));
        std::vector<uint8_t> w((size_t)nb * bb);
        for (int b = 0; b < nb; ++b) rand_block(r, type, &w[(size_t)b * bb], bb);
        std::vector<float> want((size_t)n);
        for (int b = 0; b < nb; ++b) host_deq(type, &w[(size_t)b * bb], &want[(size_t)b * be]);
        uint8_t* dw = dup(w);
        float* d32 = dalloc<float>((size_t)n);
        uint16_t* d16 = dalloc<uint16_t>((size_t)n);
        uint16_t* dbf = dalloc<uint16_t>((size_t)n);
        CHECK(dequant_f32(type, dw, d32, n, 0) == cudaSuccess);
        CHECK(dequant_f16(type, dw, d16, n, 0) == cudaSuccess);
        CHECK(dequant_bf16(type, dw, dbf, n, 0) == cudaSuccess);
        sync();
        const auto g32 = fetch(d32, (size_t)n); const auto g16 = fetch(d16, (size_t)n); const auto gbf = fetch(dbf, (size_t)n);
        bool ok32 = true, ok16 = true, okbf = true;
        for (int64_t i = 0; i < n; ++i) {
            ok32 &= near_tol(g32[i], want[i], 1e-6, 2e-5);
            ok16 &= near_tol(f_of(g16[i]), want[i], 1e-4, 2e-3);
            okbf &= near_tol(baloon::f16bits::bf16_bits_to_f32(gbf[i]), want[i], 1e-4, 1e-2);
        }
        CHECK(ok32); CHECK(ok16); CHECK(okbf);
        if (!(ok32 && ok16 && okbf)) std::printf("  dequant type %d mismatch\n", type);
        // shapes that are not whole blocks are refused
        CHECK(dequant_f32(type, dw, d32, n - 1, 0) != cudaSuccess);
        cudaFree(dw); cudaFree(d32); cudaFree(d16); cudaFree(dbf);
    }
    // plain conversions are exact
    std::vector<float> f(100); for (auto& v : f) v = r.sym(10.f);
    std::vector<uint16_t> hf(100), bf(100);
    for (int i = 0; i < 100; ++i) { hf[i] = h_of(f[i]); bf[i] = baloon::f16bits::round_f32_to_bf16_bits(f[i]); }
    float* df = dup(f); uint16_t* dh = dup(hf); uint16_t* db = dup(bf); float* o32 = dalloc<float>(100); uint16_t* ob = dalloc<uint16_t>(100);
    CHECK(dequant_f32(0, df, o32, 100, 0) == cudaSuccess); sync();
    CHECK(fetch(o32, 100) == f);
    CHECK(dequant_f32(1, dh, o32, 100, 0) == cudaSuccess); sync();
    { auto g = fetch(o32, 100); bool ok = true; for (int i = 0; i < 100; ++i) ok &= (g[i] == f_of(hf[i])); CHECK(ok); }
    CHECK(dequant_f32(30, db, o32, 100, 0) == cudaSuccess); sync();
    { auto g = fetch(o32, 100); bool ok = true; for (int i = 0; i < 100; ++i) ok &= (g[i] == baloon::f16bits::bf16_bits_to_f32(bf[i])); CHECK(ok); }
    CHECK(dequant_bf16(0, df, ob, 100, 0) == cudaSuccess); sync();
    CHECK(fetch(ob, 100) == bf);
    CHECK(!dequant_bf16_supported(16) && !dequant_bf16_supported(99));
    cudaFree(df); cudaFree(dh); cudaFree(db); cudaFree(o32); cudaFree(ob);
}

// ============================================================================================== quantize_act
static void test_quantize_act() {
    Rng r(12);
    const int64_t nb = 37;
    std::vector<float> x((size_t)nb * 256);
    for (auto& v : x) v = r.sym(3.f);
    x[5] = 0.f;
    for (int i = 0; i < 256; ++i) x[256 * 3 + i] = 0.f;   // an all-zero block: d = 0, q = 0
    float* dx = dup(x);
    uint8_t* q0 = dalloc<uint8_t>((size_t)nb * 8 * 34);
    uint8_t* q0c = dalloc<uint8_t>((size_t)nb * 8 * 34);
    uint8_t* qk = dalloc<uint8_t>((size_t)nb * 292);
    CHECK(quantize_q8_0_ggml_rule(dx, q0, nb * 8, 0) == cudaSuccess);
    CHECK(quantize_q8_0_cpu_rule(dx, q0c, nb * 8, 0) == cudaSuccess);
    CHECK(quantize_q8_k(dx, qk, nb, 0) == cudaSuccess);
    sync();
    const auto g0 = fetch(q0, (size_t)nb * 8 * 34), g0c = fetch(q0c, (size_t)nb * 8 * 34), gk = fetch(qk, (size_t)nb * 292);
    bool ok0 = true, ok0c = true, same = (g0 == g0c);
    for (int64_t b = 0; b < nb * 8; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        uint16_t dh; std::memcpy(&dh, &g0[b * 34], 2);
        ok0 &= (dh == baloon::f16bits::f16_from_f32(d));
        for (int i = 0; i < 32; ++i) {
            const int want = std::max(-128, std::min(127, (int)std::nearbyint(x[b * 32 + i] * id)));
            ok0 &= ((int8_t)g0[b * 34 + 2 + i] == want);
            const int wantc = std::max(-127, std::min(127, want));
            ok0c &= ((int8_t)g0c[b * 34 + 2 + i] == wantc) && wantc >= -127;
        }
    }
    CHECK(ok0); CHECK(ok0c); CHECK(same);
    bool okk = true;
    for (int64_t b = 0; b < nb; ++b) {
        float amax = 0.f;
        for (int i = 0; i < 256; ++i) amax = std::max(amax, std::fabs(x[b * 256 + i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        float dd; std::memcpy(&dd, &gk[b * 292], 4);
        okk &= (dd == d);
        for (int g = 0; g < 16; ++g) {
            int s = 0;
            for (int j = 0; j < 16; ++j) {
                const int want = std::max(-127, std::min(127, (int)std::nearbyint(x[b * 256 + 16 * g + j] * id)));
                okk &= ((int8_t)gk[b * 292 + 4 + 16 * g + j] == want);
                s += want;
            }
            int16_t bs; std::memcpy(&bs, &gk[b * 292 + 260 + 2 * g], 2);
            okk &= (bs == s) && std::abs(s) <= 2032;
        }
    }
    CHECK(okk);
    CHECK(quantize_q8_k(nullptr, qk, 1, 0) != cudaSuccess);
    cudaFree(dx); cudaFree(q0); cudaFree(q0c); cudaFree(qk);
}

// ============================================================================================== s_gemv
static const int kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static int dec(int bits, uint32_t c) { return bits == 2 ? (int)c - 1 : bits == 4 ? kIq4[c] : (int)(int8_t)c; }
static uint32_t code_at(const uint8_t* row, int bits, int64_t i) { const int64_t bit = i * bits; return ((uint32_t)row[bit >> 3] >> (bit & 7)) & ((1u << bits) - 1u); }

static void test_sgemv() {
    Rng r(13);
    CHECK(q8k_form_ok(256) && q8k_form_ok(2560) && !q8k_form_ok(128) && !q8k_form_ok(0));
    for (int bits : {2, 4, 8})
        for (int gs : {4, 5, 6})
            for (int n_in : {512, 768, 2304})
                for (int n_out : {1, 9, 33}) {
                    const int ng = n_in >> gs;
                    std::vector<uint8_t> codes((size_t)n_out * n_in * bits / 8);
                    for (auto& c : codes) c = (uint8_t)r.next();
                    std::vector<uint16_t> sc((size_t)n_out * ng);
                    for (auto& s : sc) s = h_of(0.005f + (float)r.uni() * 0.02f);
                    std::vector<float> x(n_in);
                    for (auto& v : x) v = r.sym(2.f);
                    uint8_t* dc = dup(codes); uint16_t* ds = dup(sc); float* dx = dup(x);
                    float* dy = dalloc<float>(n_out);
                    // float activations
                    CHECK(s_gemv(bits, dc, ds, dx, dy, n_in, n_out, gs, 0) == cudaSuccess);
                    sync();
                    auto y = fetch(dy, n_out);
                    bool ok = true;
                    for (int o = 0; o < n_out; ++o) {
                        double e = 0, mag = 0;
                        for (int i = 0; i < n_in; ++i) {
                            const double v = (double)f_of(sc[(size_t)o * ng + (i >> gs)]) * dec(bits, code_at(&codes[(size_t)o * n_in * bits / 8], bits, i)) * x[i];
                            e += v; mag += std::fabs(v);
                        }
                        ok &= std::fabs(e - y[o]) <= 1e-4 * mag + 1e-6;
                    }
                    CHECK(ok);
                    // Q8_K / Q8_0 activations quantized on the device; expected values use the device-produced integers
                    const int nbk = n_in / 256;
                    if (n_in % 256 == 0) {
                        uint8_t* dq = dalloc<uint8_t>((size_t)nbk * 292);
                        CHECK(quantize_q8_k(dx, dq, nbk, 0) == cudaSuccess);
                        sync();
                        auto qb = fetch(dq, (size_t)nbk * 292);
                        auto expect = [&](int o) {
                            double e = 0, mag = 0;
                            for (int i = 0; i < n_in; ++i) {
                                float d; std::memcpy(&d, &qb[(size_t)(i / 256) * 292], 4);
                                const int q = (int8_t)qb[(size_t)(i / 256) * 292 + 4 + i % 256];
                                const double v = (double)f_of(sc[(size_t)o * ng + (i >> gs)]) * dec(bits, code_at(&codes[(size_t)o * n_in * bits / 8], bits, i)) * q * d;
                                e += v; mag += std::fabs(v);
                            }
                            return std::make_pair(e, mag);
                        };
                        CHECK(s_gemv_q8k(bits, dc, ds, dq, dy, n_in, n_out, gs, 0) == cudaSuccess);
                        sync();
                        auto yk = fetch(dy, n_out);
                        bool okk = true;
                        for (int o = 0; o < n_out; ++o) { auto em = expect(o); okk &= std::fabs(em.first - yk[o]) <= 1e-4 * em.second + 1e-6; }
                        CHECK(okk);
                        float* dp = dalloc<float>((size_t)8 * n_out);
                        float* dy2 = dalloc<float>(n_out);
                        for (int splits : {1, 2, 3, 5}) {
                            CHECK(s_gemv_q8k_split(bits, dc, ds, dq, dy2, dp, n_in, n_out, gs, splits, 0) == cudaSuccess);
                            CHECK(s_gemv_split(bits, dc, ds, dq, 1, dy2, dp, n_in, n_out, gs, splits, 0) == cudaSuccess);
                            sync();
                            auto ys = fetch(dy2, n_out);
                            bool oks = true;
                            for (int o = 0; o < n_out; ++o) { auto em = expect(o); oks &= std::fabs(em.first - ys[o]) <= 1e-4 * em.second + 1e-6; }
                            CHECK(oks);
                            if (splits == 1) CHECK(ys == yk);   // one slice + zero = the same lane order: bit-identical
                        }
                        cudaFree(dq); cudaFree(dp); cudaFree(dy2);
                    }
                    {   // Q8_0 activations
                        const int nb0 = n_in / 32;
                        uint8_t* dq = dalloc<uint8_t>((size_t)nb0 * 34);
                        CHECK(quantize_q8_0_cpu_rule(dx, dq, nb0, 0) == cudaSuccess);
                        sync();
                        auto qb = fetch(dq, (size_t)nb0 * 34);
                        float* dp = dalloc<float>((size_t)4 * n_out); float* dy2 = dalloc<float>(n_out);
                        for (int splits : {1, 3}) {
                            CHECK(s_gemv_q8_0_split(bits, dc, ds, dq, dy2, dp, n_in, n_out, gs, splits, 0) == cudaSuccess);
                            CHECK(s_gemv_split_async(bits, dc, ds, dq, 0, dy2, dp, n_in, n_out, gs, splits, 0) == cudaSuccess);
                            sync();
                            auto ys = fetch(dy2, n_out);
                            bool oks = true;
                            for (int o = 0; o < n_out; ++o) {
                                double e = 0, mag = 0;
                                for (int i = 0; i < n_in; ++i) {
                                    const double d = f_of((uint16_t)(qb[(size_t)(i / 32) * 34] | (qb[(size_t)(i / 32) * 34 + 1] << 8)));
                                    const int q = (int8_t)qb[(size_t)(i / 32) * 34 + 2 + i % 32];
                                    const double v = (double)f_of(sc[(size_t)o * ng + (i >> gs)]) * dec(bits, code_at(&codes[(size_t)o * n_in * bits / 8], bits, i)) * q * d;
                                    e += v; mag += std::fabs(v);
                                }
                                oks &= std::fabs(e - ys[o]) <= 1e-4 * mag + 1e-6;
                            }
                            CHECK(oks);
                        }
                        cudaFree(dq); cudaFree(dp); cudaFree(dy2);
                    }
                    cudaFree(dc); cudaFree(ds); cudaFree(dx); cudaFree(dy);
                }
    // validation: groups of 4 are refused (one scale is read for sixteen elements), as are ragged lengths and bad bits
    uint8_t* a = dalloc<uint8_t>(1 << 16); float* y = dalloc<float>(64); float* p = dalloc<float>(512);
    CHECK(s_gemv_q8k_split(2, a, (const uint16_t*)a, a, y, p, 512, 4, 2, 2, 0) != cudaSuccess);
    CHECK(s_gemv_q8_0_split(2, a, (const uint16_t*)a, a, y, p, 512, 4, 2, 2, 0) != cudaSuccess);
    CHECK(s_gemv(3, a, (const uint16_t*)a, (const float*)a, y, 512, 4, 4, 0) != cudaSuccess);
    CHECK(s_gemv(2, a, (const uint16_t*)a, (const float*)a, y, 520, 4, 4, 0) != cudaSuccess);
    CHECK(s_gemv_q8k(2, a, (const uint16_t*)a, a, y, 512 + 128, 4, 4, 0) != cudaSuccess);
    CHECK(s_gemv_q8k_split(2, a, (const uint16_t*)a, a, y, p, 512, 4, 4, 0, 0) != cudaSuccess);
    cudaFree(a); cudaFree(y); cudaFree(p);
}

// ============================================================================================== sampler
static std::vector<int32_t> sample(const std::vector<float>& logits, int rows, int nv, const std::vector<int32_t>& hist, int hstride,
                                   const std::vector<int32_t>& hlen, const SamplerParams& p, uint64_t seed, uint64_t ctr) {
    float* dl = dup(logits); float* ws = dalloc<float>((size_t)rows * nv); int32_t* dout = dalloc<int32_t>(rows);
    int32_t* dh = hist.empty() ? nullptr : dup(hist); int32_t* dn = hlen.empty() ? nullptr : dup(hlen);
    CHECK(sample_tokens(dl, rows, nv, dh, hstride, dn, p, seed, ctr, ws, dout, 0) == cudaSuccess);
    sync();
    auto o = fetch(dout, rows);
    cudaFree(dl); cudaFree(ws); cudaFree(dout); if (dh) cudaFree(dh); if (dn) cudaFree(dn);
    return o;
}

static void test_sampler() {
    SamplerParams g; g.temperature = 0.f;
    // greedy: ties go to the lowest index, whatever the block layout
    {
        const int nv = 1000;
        std::vector<float> l(nv, 0.f); l[700] = 5.f; l[333] = 5.f; l[999] = 5.f;
        CHECK(sample(l, 1, nv, {}, 0, {}, g, 1, 0)[0] == 333);
        std::vector<float> ninf(nv, -INFINITY);
        CHECK(sample(ninf, 1, nv, {}, 0, {}, g, 1, 0)[0] == 0);   // nothing finite: token 0
        std::vector<float> nan(nv, 0.f); nan[10] = NAN; nan[11] = 0.5f;
        CHECK(sample(nan, 1, nv, {}, 0, {}, g, 1, 0)[0] == 11);   // NaN never wins
        // several rows at once
        std::vector<float> two((size_t)2 * nv, 0.f); two[17] = 1.f; two[nv + 900] = 2.f;
        const auto o = sample(two, 2, nv, {}, 0, {}, g, 1, 0);
        CHECK(o[0] == 17 && o[1] == 900);
    }
    // penalties
    {
        const int nv = 8;
        std::vector<float> l(nv, -5.f); l[0] = 1.0f; l[1] = 0.9f;
        std::vector<int32_t> hist = {0, 0, 0, 7, 7, 7, 7, 7}, hlen = {3};
        SamplerParams p = g; p.repeat_penalty = 2.f;
        CHECK(sample(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);                 // 1.0 / 2 < 0.9
        p = g; p.frequency_penalty = 0.1f;
        CHECK(sample(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);                 // 1.0 - 3 * 0.1 = 0.7 < 0.9
        p = g; p.presence_penalty = 0.05f;
        CHECK(sample(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 0);                 // 1.0 - 0.05 = 0.95 > 0.9: a boolean, not the count
        p = g; p.presence_penalty = 0.15f;
        CHECK(sample(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);
        // history window is the TAIL of the row
        std::vector<int32_t> h2 = {0, 0, 1, 1, 1, 1, 1, 1}, n2 = {4};   // valid: 0 0 1 1
        std::vector<float> l2(nv, -5.f); l2[0] = 1.0f; l2[1] = 1.1f;
        p = g; p.repeat_penalty = 4.f; p.last_n = 2;                           // window = [1, 1]: only token 1 penalized
        CHECK(sample(l2, 1, nv, h2, 8, n2, p, 1, 0)[0] == 0);
        p.last_n = 4;                                                          // window = [0, 0, 1, 1]: both penalized: 0.25 vs 0.275
        CHECK(sample(l2, 1, nv, h2, 8, n2, p, 1, 0)[0] == 1);
        // a negative logit is multiplied: the penalty never raises it
        std::vector<float> l3(nv, -5.f); l3[2] = -0.1f; l3[3] = -0.15f;
        std::vector<int32_t> h3 = {2, 0, 0, 0, 0, 0, 0, 0}, n3 = {1};
        p = g; p.repeat_penalty = 2.f;
        CHECK(sample(l3, 1, nv, h3, 8, n3, p, 1, 0)[0] == 3);                    // -0.1 * 2 = -0.2 < -0.15
    }
    // stochastic sampling
    {
        const int nv = 4096;
        Rng r(14);
        std::vector<float> l(nv); for (auto& v : l) v = r.sym(4.f);
        // top_k = 1 is argmax
        SamplerParams p; p.temperature = 0.8f; p.top_k = 1;
        int best = 0; for (int i = 1; i < nv; ++i) if (l[i] > l[best]) best = i;
        for (uint64_t c = 0; c < 5; ++c) CHECK(sample(l, 1, nv, {}, 0, {}, p, 7, c)[0] == best);
        // top_k = 5: always inside the five best, deterministic per (seed, counter), different counters differ sometimes
        std::vector<int> idx(nv); for (int i = 0; i < nv; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
        p.top_k = 5; p.temperature = 1.5f;
        int distinct = 0; std::vector<int> seen;
        for (uint64_t c = 0; c < 60; ++c) {
            const int t = sample(l, 1, nv, {}, 0, {}, p, 99, c)[0];
            CHECK(std::find(idx.begin(), idx.begin() + 5, t) != idx.begin() + 5);
            CHECK(sample(l, 1, nv, {}, 0, {}, p, 99, c)[0] == t);
            if (std::find(seen.begin(), seen.end(), t) == seen.end()) { seen.push_back(t); ++distinct; }
        }
        CHECK(distinct >= 3);
        // flat logits over 4 tokens: roughly uniform
        std::vector<float> flat(4, 0.f); int cnt[4] = {0, 0, 0, 0};
        SamplerParams q; q.temperature = 1.f; q.top_k = 4;
        for (uint64_t c = 0; c < 1200; ++c) ++cnt[sample(flat, 1, 4, {}, 0, {}, q, 5, c)[0]];
        std::printf("  flat counts %d %d %d %d, top5 distinct %d\n", cnt[0], cnt[1], cnt[2], cnt[3], distinct);
        for (int k = 0; k < 4; ++k) CHECK(cnt[k] > 220 && cnt[k] < 380);
        // tiny top_p keeps only the best, min_keep widens the cut to three survivors
        std::vector<float> l4 = {3.f, 2.9f, 2.8f, -4.f, -4.f, -4.f};
        SamplerParams t; t.temperature = 1.f; t.top_k = 6; t.top_p = 0.01f; t.min_keep = 1;
        bool only0 = true; for (uint64_t c = 0; c < 80; ++c) only0 &= (sample(l4, 1, 6, {}, 0, {}, t, 3, c)[0] == 0);
        CHECK(only0);
        t.min_keep = 3;
        int c3[6] = {0, 0, 0, 0, 0, 0};
        for (uint64_t c = 0; c < 300; ++c) ++c3[sample(l4, 1, 6, {}, 0, {}, t, 3, c)[0]];
        CHECK(c3[0] > 0 && c3[1] > 0 && c3[2] > 0 && c3[3] == 0 && c3[4] == 0 && c3[5] == 0);
    }
    SamplerParams bad; bad.repeat_penalty = 0.f;
    float* a = dalloc<float>(64); int32_t* o = dalloc<int32_t>(1);
    CHECK(sample_tokens(a, 1, 8, nullptr, 0, nullptr, bad, 1, 0, a, o, 0) != cudaSuccess);
    CHECK(sample_tokens(nullptr, 1, 8, nullptr, 0, nullptr, g, 1, 0, a, o, 0) != cudaSuccess);
    cudaFree(a); cudaFree(o);
}

// ============================================================================================== s2 experts
static void test_s2() {
    using namespace baloon::kern;
    Rng r(15);
    const uint64_t slot_bytes = q2geo::BLOB + 64;
    const int n_tok = 3, K = 4, NE = 512;
    std::vector<uint8_t> pool(2 * slot_bytes, 0);
    for (int s = 0; s < 2; ++s) {
        uint8_t* b = &pool[s * slot_bytes];
        for (uint64_t i = 0; i < q2geo::O_GU_SCALES; ++i) b[i] = (uint8_t)r.next();
        for (uint64_t i = q2geo::O_GU_SCALES; i < q2geo::BLOB; i += 2) put_h(b + i, 0.005f + (float)r.uni() * 0.045f);
    }
    std::vector<int32_t> slot_of(NE, -1); slot_of[3] = 0; slot_of[17] = 1;
    const std::vector<int32_t> ids = {3, 100, 17, 5,   9, 8, 7, 6,   17, 3, 200, 300};
    std::vector<float> x((size_t)n_tok * q2geo::H), w(n_tok * K), out0((size_t)n_tok * q2geo::H, 0.5f);
    for (auto& v : x) v = r.sym(1.f);
    for (auto& v : w) v = 0.1f + (float)r.uni() * 0.5f;
    // host model: CPU Q2_0 expert kernel per resident pair, combined in router-slot order
    std::vector<float> want = out0;
    std::vector<int> hits_per_tok(n_tok, 0);
    for (int t = 0; t < n_tok; ++t) {
        std::vector<float> acc(q2geo::H, 0.f), o(q2geo::H);
        for (int k = 0; k < K; ++k) {
            const int e = ids[t * K + k], s = slot_of[e];
            if (s < 0) continue;
            ++hits_per_tok[t];
            s2_expert_scalar(&pool[s * slot_bytes], &x[(size_t)t * q2geo::H], o.data());
            for (int i = 0; i < q2geo::H; ++i) acc[i] += w[t * K + k] * o[i];
        }
        for (int i = 0; i < q2geo::H; ++i) want[(size_t)t * q2geo::H + i] += acc[i];
    }
    double maxabs = 0; for (float v : want) maxabs = std::max(maxabs, (double)std::fabs(v));
    uint8_t* dpool = dup(pool); int32_t* dslot = dup(slot_of); int32_t* dids = dup(ids); float* dx = dup(x); float* dw = dup(w);
    const size_t sb = moe_hit_grouped_scratch_bytes(n_tok, K);
    {   // scratch size: 16-byte regions of gate-major buffer, int8 image, chunk scales / sums, inputs, per-router-slot outputs, flags
        const size_t P = n_tok * K; auto r16 = [](size_t n) { return (n + 15) / 16 * 16; };
        const size_t exp = r16(P * 640 * 8) + r16(P * 640) + 2 * r16(P * 20 * 4) + r16((size_t)n_tok * 2560) + 2 * r16((size_t)n_tok * 80 * 4) + r16(P * 2560 * 4) + r16(P * 4);
        CHECK(sb == exp && sb % 16 == 0);
        CHECK(moe_hit_grouped_scratch_bytes(0, 4) == 0);
    }
    void* scratch = nullptr; CUOK(cudaMalloc(&scratch, sb));
    int32_t* hit_slot = dalloc<int32_t>(n_tok * K); int32_t* hit_dst = dalloc<int32_t>(n_tok * K); int32_t* n_hit = dalloc<int32_t>(n_tok);
    CHECK(moe_hit_select_multi(dids, dslot, n_tok, K, hit_slot, hit_dst, n_hit, 0) == cudaSuccess);
    sync();
    {   // ballot-prefix compaction: dense positions in router order, unused entries -1
        const auto hs = fetch(hit_slot, n_tok * K), hd = fetch(hit_dst, n_tok * K), nh = fetch(n_hit, n_tok);
        CHECK(nh[0] == 2 && nh[1] == 0 && nh[2] == 2);
        CHECK(hs[0] == 0 && hd[0] == 0 && hs[1] == 1 && hd[1] == 2 && hs[2] == -1 && hs[3] == -1);
        CHECK(hs[4] == -1 && hs[7] == -1);
        CHECK(hs[8] == 1 && hd[8] == 0 && hs[9] == 0 && hd[9] == 1 && hs[10] == -1);
    }
    const int ntot = n_tok * q2geo::H;
    auto run_and_check = [&](const char* name, auto&& launch, double tol_frac) {
        float* dout = dup(out0);
        CHECK(launch(dout) == cudaSuccess);
        sync();
        const auto got = fetch(dout, ntot);
        bool ok = true;
        for (int i = 0; i < ntot; ++i) ok &= std::fabs(got[i] - want[i]) <= tol_frac * maxabs;
        CHECK(ok);
        if (!ok) std::printf("  s2 path %s mismatch\n", name);
        cudaFree(dout);
        return got;
    };
    const auto fast = run_and_check("fast", [&](float* o) { return moe_hit_grouped_s2_multi(dpool, slot_bytes, dx, hit_slot, hit_dst, n_tok, K, dw, o, scratch, 0); }, 5e-3);
    const auto cpuo = run_and_check("cpu_order", [&](float* o) { return moe_hit_grouped_s2_cpu_order(dpool, slot_bytes, dx, hit_slot, hit_dst, n_tok, K, dw, o, scratch, 0); }, 2e-4);
    (void)cpuo;
    // grouping by resident expert: stable order (ascending expert, token, router slot)
    int32_t* ps = dalloc<int32_t>(n_tok * K); int32_t* pt = dalloc<int32_t>(n_tok * K); int32_t* pd = dalloc<int32_t>(n_tok * K);
    CHECK(moe_group_resident(dids, dslot, n_tok, K, ps, pt, pd, 0) == cudaSuccess);
    sync();
    {
        const auto s = fetch(ps, n_tok * K), t = fetch(pt, n_tok * K), d = fetch(pd, n_tok * K);
        const int es[4] = {0, 0, 1, 1}, et[4] = {0, 2, 0, 2}, ed[4] = {0, 1, 2, 0};
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok &= (s[i] == es[i] && t[i] == et[i] && d[i] == ed[i]);
        for (int i = 4; i < n_tok * K; ++i) ok &= (s[i] == -1);
        CHECK(ok);
    }
    const auto grouped = run_and_check("grouped", [&](float* o) { return moe_grouped_s2(dpool, slot_bytes, dx, ps, pt, pd, n_tok, K, dw, o, scratch, 0); }, 5e-3);
    CHECK(grouped == fast);   // the result does not depend on how the pairs were formed
    // single token: host-known hit count and device-known hit count give token 0 of the multi run, bit for bit
    {
        int32_t* h1s = dalloc<int32_t>(K); int32_t* h1d = dalloc<int32_t>(K); int32_t* n1 = dalloc<int32_t>(1);
        CHECK(moe_hit_select(dids, dslot, K, h1s, h1d, n1, 0) == cudaSuccess);
        sync();
        CHECK(fetch(n1, 1)[0] == 2);
        const size_t sb1 = moe_hit_grouped_scratch_bytes(1, K);
        void* s1 = nullptr; CUOK(cudaMalloc(&s1, sb1));
        std::vector<float> o0(out0.begin(), out0.begin() + q2geo::H);
        float* dout = dup(o0);
        CHECK(moe_hit_grouped_s2(dpool, slot_bytes, dx, h1s, h1d, 2, dw, K, dout, s1, 0) == cudaSuccess);
        sync();
        auto g1 = fetch(dout, q2geo::H);
        CHECK(std::equal(g1.begin(), g1.end(), fast.begin()));
        float* dout2 = dup(o0);
        CHECK(moe_hit_grouped_s2_dev(dpool, slot_bytes, dx, h1s, h1d, n1, K, dw, K, dout2, s1, 0) == cudaSuccess);
        sync();
        auto g2 = fetch(dout2, q2geo::H);
        CHECK(g2 == g1);
        // moe_hit_add on its own adds the held weighted sum once more
        std::vector<float> z(q2geo::H, 0.f); float* dz = dup(z);
        CHECK(moe_hit_add(dw, 1, K, dz, s1, 0) == cudaSuccess);
        sync();
        auto gz = fetch(dz, q2geo::H);
        bool ok = true; for (int i = 0; i < q2geo::H; ++i) ok &= near_tol(gz[i] + 0.5f, g1[i], 1e-5, 1e-5);
        CHECK(ok);
        cudaFree(h1s); cudaFree(h1d); cudaFree(n1); cudaFree(s1); cudaFree(dout); cudaFree(dout2); cudaFree(dz);
    }
    // a token whose experts are all non-resident is left untouched
    {
        bool untouched = true;
        for (int i = 0; i < q2geo::H; ++i) untouched &= (fast[(size_t)1 * q2geo::H + i] == 0.5f);
        CHECK(untouched);
    }
    // validation
    CHECK(moe_hit_select(dids, dslot, 65, hit_slot, hit_dst, n_hit, 0) != cudaSuccess);
    CHECK(moe_group_resident(dids, dslot, 9, K, ps, pt, pd, 0) != cudaSuccess);   // more tokens than GMAX
    CHECK(moe_hit_grouped_s2_multi(dpool, 1000, dx, hit_slot, hit_dst, n_tok, K, dw, dx, scratch, 0) != cudaSuccess);   // slot stride below one blob
    cudaFree(dpool); cudaFree(dslot); cudaFree(dids); cudaFree(dx); cudaFree(dw); cudaFree(scratch);
    cudaFree(hit_slot); cudaFree(hit_dst); cudaFree(n_hit); cudaFree(ps); cudaFree(pt); cudaFree(pd);
}

// ============================================================================================== PLE
static float bfr(float f) { return baloon::f16bits::bf16_bits_to_f32(baloon::f16bits::round_f32_to_bf16_bits(f)); }

static void test_ple() {
    using namespace baloon::kern;
    Rng r(16);
    const int NE_ = 2560, HC = 4, HCD = 10240, NH = 9;
    CHECK(ple_block_available());
    CHECK(ple_block_scratch_bytes() == 223152);
    std::vector<float> convw((size_t)4 * HCD), normw(NE_), hist((size_t)NH * HCD), newrow(HCD), hidden(NE_);
    std::vector<uint16_t> gatew((size_t)HC * NE_), projw((size_t)NE_ * NE_);
    for (auto& v : convw) v = r.sym(0.4f);
    for (auto& v : normw) v = 1.f + r.sym(0.1f);
    for (auto& v : hist) v = r.sym(1.f);
    for (auto& v : newrow) v = r.sym(1.f);
    for (auto& v : hidden) v = r.sym(1.f);
    for (auto& v : gatew) v = baloon::f16bits::round_f32_to_bf16_bits(r.sym(0.03f));
    for (auto& v : projw) v = baloon::f16bits::round_f32_to_bf16_bits(r.sym(0.03f));
    // host model of the block
    std::vector<double> conv(HCD), nrm(HCD), s3(HCD), gate(HC);
    for (int c = 0; c < HCD; ++c) {
        double a = 0;
        for (int k = 0; k < 4; ++k) {
            const int row = 9 - (3 - k) * 3;   // 0, 3, 6, 9
            a += (double)convw[k + 4 * c] * (row == 9 ? newrow[c] : hist[row + 9 * c]);
        }
        conv[c] = a;
    }
    for (int s = 0; s < HC; ++s) {
        double ss = 0; for (int i = 0; i < NE_; ++i) ss += conv[s * NE_ + i] * conv[s * NE_ + i];
        const double rr = 1.0 / std::sqrt(ss / NE_ + 1e-6);
        for (int i = 0; i < NE_; ++i) nrm[s * NE_ + i] = conv[s * NE_ + i] * rr * normw[i];
        double lg = 0; for (int j = 0; j < NE_; ++j) lg += (double)baloon::f16bits::bf16_bits_to_f32(gatew[(size_t)s * NE_ + j]) * hidden[j];
        { const double m = std::fabs(lg) > 1e-6 ? std::fabs(lg) : 1e-6; const double sg = lg > 0 ? 1.0 : (lg < 0 ? -1.0 : 0.0); gate[s] = 1.0 / (1.0 + std::exp(-sg * std::sqrt(m))); }
    }
    for (int i = 0; i < HCD; ++i) s3[i] = (hidden[i % NE_] * gate[i / NE_] + nrm[i]) + newrow[i];
    std::vector<double> act(NE_);
    for (int j = 0; j < NE_; ++j) { double s = 0; for (int k = 0; k < HC; ++k) s += s3[k * NE_ + j]; act[j] = s / (1.0 + std::exp(-s)); }

    float* dcw = dup(convw); float* dnw = dup(normw); float* dh = dup(hist); float* dn = dup(newrow); float* dhid = dup(hidden);
    uint16_t* dgw = dup(gatew); uint16_t* dpw = dup(projw);
    float* dout = dalloc<float>(NE_); float* dgate = dalloc<float>(HC);
    void* scratch = nullptr; CUOK(cudaMalloc(&scratch, ple_block_scratch_bytes()));
    PleWeights w; w.conv_w = dcw; w.norm_w = dnw; w.gate_w = dgw; w.proj_w = dpw;
    PleOut po; po.out = dout; po.gate = dgate;
    for (int bf : {0, 1})
        for (int post : {0, 1}) {
            ple_set_native_bf16(bf != 0);
            ple_set_native_postops(post != 0);
            CHECK(ple_native_postops_enabled() == (post != 0));
            CHECK(ple_block(w, dn, dh, dhid, po, scratch, 0) == cudaSuccess);
            sync();
            const auto o = fetch(dout, NE_), g = fetch(dgate, HC);
            bool ok = true, okg = true;
            for (int s = 0; s < HC; ++s) okg &= near_tol(g[s], gate[s], 1e-4, 1e-3);
            CHECK(okg);
            for (int oo = 0; oo < NE_; ++oo) {
                double e = post ? hidden[oo] : 0.0;
                for (int j = 0; j < NE_; ++j) e += (double)baloon::f16bits::bf16_bits_to_f32(projw[(size_t)oo * NE_ + j]) * (bf ? (double)bfr((float)act[j]) : act[j]);
                ok &= near_tol(o[oo], e, 5e-3, 5e-3);
            }
            CHECK(ok);
            if (!ok) std::printf("  ple mismatch (bf16=%d postops=%d)\n", bf, post);
        }
    ple_set_native_bf16(false); ple_set_native_postops(false);
    // the block does not modify the history; advance commits the new row (shift left, append)
    {
        CHECK(fetch(dh, hist.size()) == hist);
        CHECK(ple_history_advance(dh, dn, 0) == cudaSuccess);
        sync();
        const auto h2 = fetch(dh, hist.size());
        bool ok = true;
        for (int c = 0; c < HCD; ++c) {
            for (int rr = 0; rr + 1 < NH; ++rr) ok &= (h2[rr + 9 * c] == hist[rr + 1 + 9 * c]);
            ok &= (h2[8 + 9 * c] == newrow[c]);
        }
        CHECK(ok);
    }
    // overlap checks
    {
        PleOut bad = po;
        CHECK(ple_block(w, dn, dh, (const float*)scratch, bad, scratch, 0) != cudaSuccess);                       // hidden inside the scratch
        CHECK(ple_block(w, dn, dh, dhid, po, (void*)dhid, 0) != cudaSuccess);                                      // scratch on top of hidden
        PleOut al = po; al.out = dhid;                                                                             // out == hidden is allowed
        CHECK(ple_block(w, dn, dh, dhid, al, scratch, 0) == cudaSuccess);
        sync();
        CHECK(ple_history_advance(dh, dh, 0) != cudaSuccess);
        PleWeights nw = w; nw.proj_w = nullptr;
        CHECK(ple_block(nw, dn, dh, dhid, po, scratch, 0) != cudaSuccess);
    }
    cudaFree(dcw); cudaFree(dnw); cudaFree(dh); cudaFree(dn); cudaFree(dhid); cudaFree(dgw); cudaFree(dpw); cudaFree(dout); cudaFree(dgate); cudaFree(scratch);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    CHECK(ndev > 0);
    CUOK(cudaSetDevice(0));
    cpu_features();
    test_dequant();      std::printf("dequant done (checks %ld, fail %ld)\n", g_checks, g_fail);
    test_quantize_act(); std::printf("quantize done (checks %ld, fail %ld)\n", g_checks, g_fail);
    test_sgemv();        std::printf("s_gemv done (checks %ld, fail %ld)\n", g_checks, g_fail);
    test_sampler();      std::printf("sampler done (checks %ld, fail %ld)\n", g_checks, g_fail);
    test_s2();           std::printf("s2 done (checks %ld, fail %ld)\n", g_checks, g_fail);
    test_ple();          std::printf("ple done (checks %ld, fail %ld)\n", g_checks, g_fail);
    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("MISC GPU TESTS FAILED\n"); return 1; }
    std::printf("MISC GPU TESTS PASSED\n");
    return 0;
}
