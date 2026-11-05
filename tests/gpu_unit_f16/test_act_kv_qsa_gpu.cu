// GPU tests: Q8 KV append/gather vs the host reference, Q8_0/Q8_K quantisation invariants, fp16 KV round trip, indexer
// scores, exact top-512 (set, order, tie rule), attention vs a double-precision softmax, gate, and step == explicit.
// Exit 0 all pass, 1 a check failed, 2 no device.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/f16_bits_rounding.hpp"
#include "baloon/kv_q8_quantization.hpp"
#include "../../src/kernels/cuda/act_kv_qsa_dev.h"
using namespace baloon;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA '%s' line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)
template <class T> static T* up(const std::vector<T>& v) { T* d; if (cudaMalloc((void**)&d, std::max<size_t>(1, v.size() * sizeof(T))) != cudaSuccess) return nullptr; cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
template <class T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> v(n); cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return v; }
template <class T> static T* dalloc(size_t n) { T* d; cudaMalloc((void**)&d, std::max<size_t>(1, n * sizeof(T))); cudaMemset(d, 0, std::max<size_t>(1, n * sizeof(T))); return d; }

static void test_kv_q8() {
    const int kvh = 3, hd = 128, ps = 4, groups = hd / 64, npages = 6, tl = 3;
    std::vector<int32_t> table = {4, 1, 5};
    const size_t rows = (size_t)npages * kvh * ps;
    std::vector<int8_t> hc(rows * hd, 0); std::vector<uint16_t> hs(rows * groups, 0);
    int8_t* dc = dalloc<int8_t>(rows * hd); uint16_t* ds = dalloc<uint16_t>(rows * groups); int32_t* dt = up(table);
    std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<std::vector<float>> toks;
    for (int pos = 0; pos < ps * tl; ++pos) {
        std::vector<float> x((size_t)kvh * hd);
        for (auto& v : x) v = nd(rng);
        if (pos == 2) for (int i = 0; i < 64; ++i) x[i] = 0.f;  // an all-zero group
        toks.push_back(x);
        float* dx = up(x);
        CHECK(kvq8dev::kv_append_q8_step(dx, dc, ds, dt, pos, kvh, hd, ps, tl, npages) == 0);
        cudaFree(dx);
        kvq8::append_token(x.data(), hc.data(), hs.data(), table.data(), pos, kvh, hd, ps);
    }
    CHECK(down(dc, rows * hd) == hc);
    CHECK(down(ds, rows * groups) == hs);
    // rejected geometry launches nothing
    CHECK(kvq8dev::kv_append_q8_step(nullptr, dc, ds, dt, 0, kvh, hd, ps, tl, npages) == -1);
    CHECK(kvq8dev::kv_append_q8_step(toks[0].data(), dc, ds, dt, ps * tl, kvh, hd, ps, tl, npages) == -1);
    // gather == dequantised host values; invariant |x - deq| <= scale/2 + eps
    std::vector<int32_t> ids = {7, 0, -1, 11, 2};
    int32_t* di = up(ids); float* dout = dalloc<float>(ids.size() * kvh * hd);
    CHECK(kvq8dev::kv_gather_q8_step(dc, ds, dt, di, (int)ids.size(), kvh, hd, ps, tl, npages, dout) == 0);
    auto out = down(dout, ids.size() * kvh * hd);
    for (size_t a = 0; a < ids.size(); ++a)
        for (int h = 0; h < kvh; ++h)
            for (int d = 0; d < hd; ++d) {
                const float got = out[(a * kvh + h) * hd + d];
                if (ids[a] < 0) { CHECK(got == 0.f); continue; }
                const auto ad = kvq8::append_address(table.data(), ids[a], h, kvh, ps);
                const float want = kvq8::dequant_value(hc[(size_t)ad.row * hd + d], hs[(size_t)ad.row * groups + d / 64]);
                CHECK(got == want);
                const float x = toks[ids[a]][(size_t)h * hd + d];
                const float sc = f16::half_to_float(hs[(size_t)ad.row * groups + d / 64]);
                CHECK(std::fabs(x - got) <= sc * 0.51f + 1e-6f);
            }
    cudaFree(dc); cudaFree(ds); cudaFree(dt); cudaFree(di); cudaFree(dout);
}

static void test_quant() {
    const int nb = 64;
    std::mt19937 rng(9); std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> x((size_t)nb * 256);
    for (auto& v : x) v = nd(rng);
    for (int i = 0; i < 256; ++i) x[i] = 0.f;  // zero Q8_K block, zero Q8_0 blocks
    float* dx = up(x);
    // Q8_0
    uint8_t* d0 = dalloc<uint8_t>((size_t)nb * 8 * 34);
    CHECK(actq::quantize_q8_0(dx, d0, nb * 8) == 0);
    auto q0 = down(d0, (size_t)nb * 8 * 34);
    float* dy = dalloc<float>(x.size());
    CHECK(actq::dequant_q8_0(d0, dy, nb * 8) == 0);
    auto y0 = down(dy, x.size());
    for (int b = 0; b < nb * 8; ++b) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const uint16_t dh = (uint16_t)(q0[b * 34] | (q0[b * 34 + 1] << 8));
        CHECK(dh == f16::float_to_half(amax / 127.f));
        const float d = f16::half_to_float(dh);
        for (int i = 0; i < 32; ++i) {
            CHECK((int8_t)q0[b * 34 + 2 + i] >= -127 && (int8_t)q0[b * 34 + 2 + i] <= 127);
            CHECK(std::fabs(x[b * 32 + i] - y0[b * 32 + i]) <= d * 0.5f * 1.01f + amax * 1e-3f + 1e-7f);
        }
    }
    // scaled == quantise(x*scale)
    std::vector<float> xs(x.size()); const float sc = 0.37f;
    for (size_t i = 0; i < x.size(); ++i) xs[i] = x[i] * sc;
    float* dxs = up(xs); uint8_t* d1 = dalloc<uint8_t>((size_t)nb * 8 * 34); uint8_t* d2 = dalloc<uint8_t>((size_t)nb * 8 * 34);
    CHECK(actq::quantize_q8_0_scaled(dx, sc, d1, nb * 8) == 0);
    CHECK(actq::quantize_q8_0(dxs, d2, nb * 8) == 0);
    CHECK(down(d1, (size_t)nb * 8 * 34) == down(d2, (size_t)nb * 8 * 34));
    // Q8_K
    uint8_t* dk = dalloc<uint8_t>((size_t)nb * 292);
    CHECK(actq::quantize_q8_K(dx, dk, nb) == 0);
    auto qk = down(dk, (size_t)nb * 292);
    CHECK(actq::dequant_q8_K(dk, dy, nb) == 0);
    auto yk = down(dy, x.size());
    for (int b = 0; b < nb; ++b) {
        const uint8_t* ib = &qk[(size_t)b * 292];
        float d; std::memcpy(&d, ib, 4);
        int16_t bs[16]; std::memcpy(bs, ib + 260, 32);
        float amax = 0; for (int i = 0; i < 256; ++i) amax = std::max(amax, std::fabs(x[b * 256 + i]));
        if (amax == 0.f) { CHECK(d == 0.f); }
        for (int j = 0; j < 16; ++j) {
            int s = 0; for (int l = 0; l < 16; ++l) s += (int8_t)ib[4 + j * 16 + l];
            CHECK(s == bs[j]);
        }
        for (int i = 0; i < 256; ++i) {
            CHECK((int8_t)ib[4 + i] >= -127);
            CHECK(std::fabs(x[b * 256 + i] - yk[b * 256 + i]) <= std::fabs(d) * 0.5f * 1.01f + 1e-6f);
        }
        if (amax > 0.f) {  // the largest-magnitude value maps to exactly -127 (iscale = -127/xmax)
            int imax = 0; for (int i = 0; i < 256; ++i) if (std::fabs(x[b * 256 + i]) > std::fabs(x[b * 256 + imax])) imax = i;
            CHECK((int8_t)ib[4 + imax] == -127);
        }
    }
    CHECK(actq::quantize_q8_K(nullptr, dk, 1) == -1);
    cudaFree(dx); cudaFree(dxs); cudaFree(d0); cudaFree(d1); cudaFree(d2); cudaFree(dk); cudaFree(dy);
}

static void test_qsa() {
    const int dim = 128, ncell = 3000;
    std::mt19937 rng(21); std::normal_distribution<float> nd(0.f, 1.f);
    auto rv = [&](size_t n) { std::vector<float> v(n); for (auto& a : v) a = nd(rng); return v; };
    // --- fp16 KV append / gather
    uint16_t* kc = dalloc<uint16_t>((size_t)ncell * dim); uint16_t* vc = dalloc<uint16_t>((size_t)ncell * dim);
    std::vector<uint16_t> hk((size_t)ncell * dim, 0), hv((size_t)ncell * dim, 0);
    std::vector<int> poss = {0, 1, 5, 17, 1000, 2999};
    for (int p : poss) {
        auto k = rv(dim), v = rv(dim);
        if (p == 17) { k[0] = 1e9f; k[1] = -INFINITY; v[2] = 65504.f; v[3] = 1e-9f; }
        float *dk = up(k), *dv = up(v);
        if (p % 2 == 0) CHECK(qsak::kv_append(dk, dv, kc, vc, p, dim, ncell) == 0);
        else { CHECK(qsak::step_upload(p, 0) == 0); CHECK(qsak::kv_append_step(dk, dv, kc, vc, dim, ncell) == 0); }
        for (int i = 0; i < dim; ++i) {
            hk[(size_t)p * dim + i] = f16bits::round_f32_to_f16_bits(k[i]);
            hv[(size_t)p * dim + i] = f16bits::round_f32_to_f16_bits(v[i]);
        }
        cudaFree(dk); cudaFree(dv);
    }
    CHECK(down(kc, hk.size()) == hk); CHECK(down(vc, hv.size()) == hv);
    CHECK(qsak::kv_append(nullptr, nullptr, kc, vc, 0, dim, ncell) == -1);
    CHECK(qsak::kv_append(up(rv(dim)), up(rv(dim)), kc, vc, ncell, dim, ncell) == -1);
    std::vector<int32_t> gid = {2999, 17, -1, 5, 3000, 0};
    int32_t* dg = up(gid); float *gk = dalloc<float>(gid.size() * dim), *gv = dalloc<float>(gid.size() * dim);
    CHECK(qsak::kv_gather(kc, vc, dg, (int)gid.size(), dim, ncell, gk, gv) == 0);
    auto ok = down(gk, gid.size() * dim), ov = down(gv, gid.size() * dim);
    for (size_t a = 0; a < gid.size(); ++a)
        for (int i = 0; i < dim; ++i) {
            const bool valid = gid[a] >= 0 && gid[a] < ncell;
            const float wk = valid ? f16::half_to_float(hk[(size_t)gid[a] * dim + i]) : 0.f;
            const float wv = valid ? f16::half_to_float(hv[(size_t)gid[a] * dim + i]) : 0.f;
            CHECK(std::memcmp(&ok[a * dim + i], &wk, 4) == 0); CHECK(std::memcmp(&ov[a * dim + i], &wv, 4) == 0);
        }
    CHECK(qsak::step_upload_width(0, (int)gid.size()) == 0);
    float *gk2 = dalloc<float>(gid.size() * dim), *gv2 = dalloc<float>(gid.size() * dim);
    CHECK(qsak::kv_gather_step(kc, vc, dg, (int)gid.size(), dim, ncell, gk2, gv2) == 0);
    CHECK(down(gk2, gid.size() * dim) == ok); CHECK(down(gv2, gid.size() * dim) == ov);

    // --- indexer append + scoring
    const int nkv = 700, cap = 768;
    float* keys = dalloc<float>((size_t)cap * qsak::kKeyDim);
    std::vector<float> hkeys((size_t)cap * qsak::kKeyDim, 0.f);
    for (int p = 0; p < nkv; ++p) {
        auto kk = rv(qsak::kKeyDim); float* d = up(kk);
        CHECK(qsak::indexer_key_append(d, keys, p, cap) == 0);
        std::copy(kk.begin(), kk.end(), hkeys.begin() + (size_t)p * qsak::kKeyDim); cudaFree(d);
    }
    CHECK(down(keys, hkeys.size()) == hkeys);
    CHECK(qsak::indexer_key_append(up(rv(qsak::kKeyDim)), keys, cap, cap) == -1);
    auto q = rv(qsak::kIdxHeads * qsak::kKeyDim); std::vector<float> hw = {0.5f, 1.f, 0.25f, 2.f};
    float *dq = up(q), *dhw = up(hw), *dsc = dalloc<float>(cap);
    CHECK(qsak::qsa_index(dq, dhw, keys, dsc, nkv, cap) == 0);
    auto sc = down(dsc, cap);
    std::vector<float> ref(cap);
    for (int c = 0; c < cap; ++c) {
        if (c >= nkv) { CHECK(std::isinf(sc[c]) && sc[c] < 0); continue; }
        double s = 0;
        for (int h = 0; h < 4; ++h) { double dot = 0; for (int d = 0; d < 128; ++d) dot += (double)q[h * 128 + d] * hkeys[(size_t)c * 128 + d]; s += hw[h] * (dot > 0.0 ? dot : 0.0); }
        ref[c] = (float)s; CHECK(std::fabs(sc[c] - s) <= 1e-3 * (1.0 + std::fabs(s)));
    }
    CHECK(qsak::step_upload(0, nkv) == 0);
    float* dsc2 = dalloc<float>(cap);
    CHECK(qsak::qsa_index_step(dq, dhw, keys, dsc2, cap) == 0);
    CHECK(down(dsc2, cap) == sc);

    // --- exact top-512 on random scores, heavy ties, n < 512, n == 0
    for (int mode = 0; mode < 5; ++mode) {
        int n = mode == 3 ? 100 : (mode == 4 ? 0 : 2500);
        std::vector<float> s(std::max(n, 1));
        for (int i = 0; i < n; ++i) s[i] = mode == 1 ? (float)(rng() % 7) : (mode == 2 ? (float)(rng() % 3) - 1.f : nd(rng));
        if (mode == 0 && n > 10) { s[3] = -0.f; s[4] = 0.f; s[5] = -INFINITY; }
        float* ds = up(s); int32_t* dids = dalloc<int32_t>(512); int32_t* dn = dalloc<int32_t>(1);
        CHECK(qsak::topk_512(ds, n, dids, dn) == 0);
        auto ids = down(dids, 512); int nout = down(dn, 1)[0];
        const int kk = std::min(n, 512);
        CHECK(nout == kk);
        std::vector<int> ord(n); std::iota(ord.begin(), ord.end(), 0);
        auto key = [&](int i) { uint32_t u; std::memcpy(&u, &s[i], 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); };
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return key(a) > key(b); });
        std::vector<int> want(ord.begin(), ord.begin() + kk); std::sort(want.begin(), want.end());
        for (int j = 0; j < 512; ++j) CHECK(ids[j] == (j < kk ? want[j] : -1));
        // topk_512_step takes n from the step record
        CHECK(qsak::step_upload(0, n) == 0);
        int32_t* dids2 = dalloc<int32_t>(512); int32_t* dn2 = dalloc<int32_t>(1);
        CHECK(qsak::topk_512_step(ds, dids2, dn2) == 0);
        CHECK(down(dids2, 512) == ids); CHECK(down(dn2, 1)[0] == nout);
        cudaFree(ds); cudaFree(dids); cudaFree(dn); cudaFree(dids2); cudaFree(dn2);
    }

    // --- attention
    const int nsel = 300; std::vector<int32_t> sel(nsel);
    for (int i = 0; i < nsel; ++i) sel[i] = (i * 7) % ncell;
    sel[10] = -1; sel[11] = ncell + 5;
    auto qa = rv(dim); float scale = 0.09f;
    float* dqa = up(qa); int32_t* dsel = up(sel); float* dout = dalloc<float>(dim);
    CHECK(qsak::qsa_attend(dqa, kc, vc, dsel, nsel, ncell, dim, scale, dout) == 0);
    auto o = down(dout, dim);
    std::vector<double> lg(nsel, -1e30); double m = -1e30;
    for (int i = 0; i < nsel; ++i) if (sel[i] >= 0 && sel[i] < ncell) {
        double s = 0; for (int d = 0; d < dim; ++d) s += (double)qa[d] * f16::half_to_float(hk[(size_t)sel[i] * dim + d]);
        lg[i] = s * scale; m = std::max(m, lg[i]);
    }
    double den = 0; for (int i = 0; i < nsel; ++i) if (lg[i] > -1e29) den += std::exp(lg[i] - m);
    for (int d = 0; d < dim; ++d) {
        double a = 0;
        for (int i = 0; i < nsel; ++i) if (lg[i] > -1e29) a += std::exp(lg[i] - m) * f16::half_to_float(hv[(size_t)sel[i] * dim + d]);
        a /= den; CHECK(std::fabs(o[d] - a) <= 2e-4 * (1.0 + std::fabs(a)));
    }
    CHECK(qsak::step_upload_width(0, nsel) == 0);
    float* dout2 = dalloc<float>(dim);
    CHECK(qsak::qsa_attend_step(dqa, kc, vc, dsel, ncell, dim, scale, dout2) == 0);
    CHECK(down(dout2, dim) == o);
    CHECK(qsak::qsa_attend(dqa, kc, vc, dsel, qsak::kMaxIds + 1, ncell, dim, scale, dout) == -1);
    // nothing valid -> zeros
    std::vector<int32_t> none = {-1, ncell};
    int32_t* dn0 = up(none); CHECK(qsak::qsa_attend(dqa, kc, vc, dn0, 2, ncell, dim, scale, dout) == 0);
    for (float v : down(dout, dim)) CHECK(v == 0.f);

    // --- gate
    const int ng = 1000; auto base = rv(ng), gate = rv(ng);
    std::vector<uint16_t> gh(ng); for (int i = 0; i < ng; ++i) gh[i] = f16bits::round_f32_to_f16_bits(gate[i] * 4.f);
    float *db = up(base), *db2 = up(base), *dgf = up(gate); uint16_t* dgh = up(gh);
    CHECK(qsak::qsa_gate_apply(db, dgh, ng) == 0); CHECK(qsak::qsa_gate_apply_f32(db2, dgf, ng) == 0);
    auto r1 = down(db, ng), r2 = down(db2, ng);
    for (int i = 0; i < ng; ++i) {
        const double g1 = f16::half_to_float(gh[i]);
        CHECK(std::fabs(r1[i] - base[i] / (1.0 + std::exp(-g1))) <= 1e-5 * (1 + std::fabs(base[i])));
        CHECK(std::fabs(r2[i] - base[i] / (1.0 + std::exp(-(double)gate[i]))) <= 1e-5 * (1 + std::fabs(base[i])));
        CHECK(std::fabs(r2[i]) <= std::fabs(base[i]) + 1e-6f);  // sigmoid < 1 never amplifies
    }
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no device\n"); return 2; }
    CU(cudaFree(0));
    test_kv_q8(); test_quant(); test_qsa();
    CU(cudaDeviceSynchronize());
    std::printf("act/kv/qsa gpu test: %d failures\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
