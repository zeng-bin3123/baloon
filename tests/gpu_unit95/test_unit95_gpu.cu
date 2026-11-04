// test_unit95_gpu.cu - device tests: Q8_0/Q8_K activation quantizers, Q8 KV cache append/gather, QSA step kernels
// (fp16 caches, indexer, top-512 with ties, attention, gate). Every check compares with a host reference or asserts an
// invariant (code range, bsums, ascending unique ids, step == non-step). Exit 0 only if all pass; 2 if no device.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/device_kv_qsa_api.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_quantization.hpp"
using namespace baloon;
using namespace baloon::dk;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <typename T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T))); return p; }
template <typename T> static T* dup(const std::vector<T>& h) { T* d = dalloc<T>(h.size()); if (!h.empty()) CUOK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); if (n) CUOK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
static uint16_t h16(float f) { return f16bits::round_f32_to_f16_bits(f); }
static float f16f(uint16_t h) { return f16bits::widen_f16_bits(h); }

// ------------------------------------------------------------------ activation quantizers
static void test_quantize() {
    std::mt19937 rng(5);
    std::normal_distribution<float> N(0.f, 3.f);
    const int n = 256 * 37;
    std::vector<float> x(n);
    for (auto& v : x) v = N(rng);
    for (int i = 0; i < 256; ++i) x[i] = 0.f;                       // an all-zero block
    float* dx = dup(x);
    // Q8_0
    uint8_t* dq = dalloc<uint8_t>((size_t)n / 32 * Q8_0_BYTES);
    CHECK(quantize_q8_0(dx, dq, n, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto q = down(dq, (size_t)n / 32 * Q8_0_BYTES);
    int bad_scale = 0, bad_code = 0;
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b * 32 + i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        const uint16_t hs = (uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8));
        if (std::abs((int)hs - (int)h16(d)) > 1) ++bad_scale;   // division may differ by one fp16 ulp
        for (int i = 0; i < 32; ++i) {
            const int want = (int)std::nearbyint(x[b * 32 + i] * id), got = (int8_t)q[b * 34 + 2 + i];
            if (std::abs(want - got) > 1) ++bad_code;
            CHECK(got >= -127 && got <= 127);
        }
    }
    CHECK(bad_scale == 0); CHECK(bad_code == 0);
    float* dy = dalloc<float>(n);
    CHECK(dequant_q8_0(dq, dy, n, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto y = down(dy, n);
    int bad_err = 0;
    for (int b = 0; b < n / 32; ++b) { const float d = f16f((uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8)));
        for (int i = 0; i < 32; ++i) if (std::fabs(y[b * 32 + i] - x[b * 32 + i]) > 0.5f * d * 1.02f + 1e-6f + std::fabs(x[b * 32 + i]) * 1e-3f) ++bad_err; }
    CHECK(bad_err == 0);
    // scaled: quantizing x with scale s equals quantizing s*x
    std::vector<float> xs(n); for (int i = 0; i < n; ++i) xs[i] = x[i] * 0.25f;
    float* dxs = dup(xs); uint8_t* dq2 = dalloc<uint8_t>((size_t)n / 32 * Q8_0_BYTES); uint8_t* dq3 = dalloc<uint8_t>((size_t)n / 32 * Q8_0_BYTES);
    CHECK(quantize_q8_0_scaled(dx, 0.25f, dq2, n, 0) == 0); CHECK(quantize_q8_0(dxs, dq3, n, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    CHECK(down(dq2, (size_t)n / 32 * 34) == down(dq3, (size_t)n / 32 * 34));
    // Q8_K
    uint8_t* dk8 = dalloc<uint8_t>((size_t)n / 256 * Q8K_BYTES);
    CHECK(quantize_q8_K(dx, dk8, n, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto k = down(dk8, (size_t)n / 256 * Q8K_BYTES);
    int bad_bs = 0, bad_range = 0, bad_dq = 0;
    for (int b = 0; b < n / 256; ++b) {
        const uint8_t* p = &k[(size_t)b * Q8K_BYTES];
        float d; std::memcpy(&d, p, 4);
        int16_t bs[16]; std::memcpy(bs, p + 4 + 256, 32);
        for (int j = 0; j < 16; ++j) { int s = 0; for (int i = 0; i < 16; ++i) s += (int8_t)p[4 + j * 16 + i]; if (s != bs[j]) ++bad_bs; }
        float amax = 0, mx = 0; for (int i = 0; i < 256; ++i) if (std::fabs(x[b * 256 + i]) > amax) { amax = std::fabs(x[b * 256 + i]); mx = x[b * 256 + i]; }
        if (amax == 0.f) { CHECK(d == 0.f); for (int i = 0; i < 256; ++i) CHECK((int8_t)p[4 + i] == 0); continue; }
        CHECK(std::fabs(d - (-mx / 127.f)) <= 1e-5f * std::fabs(d));    // the extreme value maps to -127
        CHECK((int8_t)p[4 + [&]{ int a = 0; for (int i = 0; i < 256; ++i) if (x[b*256+i] == mx) { a = i; break; } return a; }()] == -127);
        for (int i = 0; i < 256; ++i) { const int c = (int8_t)p[4 + i]; if (c < -128 || c > 127) ++bad_range; }
        for (int i = 0; i < 256; ++i) if (std::fabs((float)(int8_t)p[4 + i] * d - x[b * 256 + i]) > std::fabs(d) * 0.51f + 1e-6f) ++bad_dq;
    }
    CHECK(bad_bs == 0); CHECK(bad_range == 0); CHECK(bad_dq == 0);
    CHECK(dequant_q8_K(dk8, dy, n, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    y = down(dy, n);
    int bad_k = 0; for (int i = 0; i < n; ++i) { float d; std::memcpy(&d, &k[(size_t)(i / 256) * Q8K_BYTES], 4); if (std::fabs(y[i] - d * (float)(int8_t)k[(size_t)(i / 256) * Q8K_BYTES + 4 + i % 256]) > 1e-6f) ++bad_k; }
    CHECK(bad_k == 0);
    CHECK(quantize_q8_0(dx, dq, 33, 0) == -1); CHECK(quantize_q8_K(dx, dk8, 255, 0) == -1); CHECK(quantize_q8_0(nullptr, dq, 32, 0) == -1);
    cudaFree(dx); cudaFree(dy); cudaFree(dq); cudaFree(dq2); cudaFree(dq3); cudaFree(dk8); cudaFree(dxs);
}

// ------------------------------------------------------------------ KV Q8 cache
static void test_kv_q8() {
    const int kv_heads = 2, head_dim = 128, page_size = 16, n_table = 8, n_pages = 8;
    KvQ8Shape s{kv_heads, head_dim, page_size, n_table};
    const int groups = head_dim / 64;
    const size_t rows = (size_t)n_pages * kv_heads * page_size;
    std::vector<int32_t> table = {5, 2, 7, 0, 3, 1, 6, 4};                  // a permutation: logical page -> physical page
    int32_t* dtab = dup(table);
    int8_t* dcodes = dalloc<int8_t>(rows * head_dim); uint16_t* dscales = dalloc<uint16_t>(rows * groups);
    CUOK(cudaMemset(dcodes, 0, rows * head_dim)); CUOK(cudaMemset(dscales, 0, rows * groups * 2));
    std::vector<int8_t> hcodes(rows * head_dim, 0); std::vector<uint16_t> hscales(rows * groups, 0);
    std::mt19937 rng(9); std::normal_distribution<float> N(0.f, 2.f);
    int32_t* dpos = dalloc<int32_t>(1);
    const int n_tok = 70;
    std::vector<std::vector<float>> toks;
    for (int pos = 0; pos < n_tok; ++pos) {
        std::vector<float> v((size_t)kv_heads * head_dim);
        for (auto& e : v) e = N(rng);
        if (pos == 3) std::fill(v.begin(), v.begin() + 64, 0.f);             // an all-zero group: scale 0, codes 0
        toks.push_back(v);
        float* dsrc = dup(v);
        CUOK(cudaMemcpy(dpos, &pos, 4, cudaMemcpyHostToDevice));
        CHECK(kv_append_q8_step(s, dsrc, dcodes, dscales, dtab, dpos, 0) == 0);
        CUOK(cudaDeviceSynchronize());
        cudaFree(dsrc);
        kvq8::append_token(v.data(), hcodes.data(), hscales.data(), table.data(), pos, kv_heads, head_dim, page_size);
    }
    auto codes = down(dcodes, rows * head_dim); auto scales = down(dscales, rows * groups);
    int bad_scale = 0, bad_code = 0, big = 0, other = 0;
    for (size_t r = 0; r < rows; ++r) {
        for (int g = 0; g < groups; ++g) if (std::abs((int)scales[r * groups + g] - (int)hscales[r * groups + g]) > 1) ++bad_scale;
        for (int d = 0; d < head_dim; ++d) { const int a = codes[r * head_dim + d], b = hcodes[r * head_dim + d]; if (std::abs(a - b) > 1) ++bad_code; if (a < -127 || a > 127) ++big; }
    }
    CHECK(bad_scale == 0); CHECK(bad_code == 0); CHECK(big == 0); (void)other;
    // untouched cells (positions >= n_tok) stay zero: append writes only its own row
    for (int pos = n_tok; pos < n_table * page_size; ++pos) for (int h = 0; h < kv_heads; ++h) {
        const auto a = kvq8::append_address(table.data(), pos, h, kv_heads, page_size);
        for (int d = 0; d < head_dim; ++d) if (codes[(size_t)a.row * head_dim + d] != 0) ++other;
    }
    CHECK(other == 0);
    // gather a shuffled id list; roundtrip error <= half a step of the group scale
    std::vector<int32_t> ids; for (int i = 0; i < 40; ++i) ids.push_back((i * 37 + 11) % n_tok);
    int32_t* dids = dup(ids); float* dout = dalloc<float>(ids.size() * kv_heads * head_dim);
    CHECK(kv_gather_q8_step(s, (int)ids.size(), dids, nullptr, dcodes, dscales, dtab, dout, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto out = down(dout, ids.size() * kv_heads * head_dim);
    int bad_rt = 0;
    for (size_t i = 0; i < ids.size(); ++i) for (int h = 0; h < kv_heads; ++h) for (int d = 0; d < head_dim; ++d) {
        const float want = toks[ids[i]][(size_t)h * head_dim + d], got = out[(i * kv_heads + h) * head_dim + d];
        const float amax = [&]{ float m = 0; for (int e = 0; e < 64; ++e) m = std::max(m, std::fabs(toks[ids[i]][(size_t)h * head_dim + (d / 64) * 64 + e])); return m; }();
        if (std::fabs(got - want) > amax / 127.f * 0.55f + amax * 1e-3f + 1e-6f) ++bad_rt;
    }
    CHECK(bad_rt == 0);
    // n_ids from device caps the count; rows past it are untouched
    int32_t n3 = 5; int32_t* dn = dup(std::vector<int32_t>{n3});
    CUOK(cudaMemset(dout, 0, ids.size() * kv_heads * head_dim * 4));
    CHECK(kv_gather_q8_step(s, (int)ids.size(), dids, dn, dcodes, dscales, dtab, dout, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto o2 = down(dout, ids.size() * kv_heads * head_dim);
    int tail = 0; for (size_t i = (size_t)n3 * kv_heads * head_dim; i < o2.size(); ++i) tail += o2[i] != 0.f;
    CHECK(tail == 0);
    KvQ8Shape badshape{2, 100, 16, 8};
    CHECK(kv_append_q8_step(badshape, nullptr, dcodes, dscales, dtab, dpos, 0) == -1);
    CHECK(kv_append_q8_step(badshape, dout, dcodes, dscales, dtab, dpos, 0) == -1);   // head_dim not a multiple of 64
    cudaFree(dtab); cudaFree(dcodes); cudaFree(dscales); cudaFree(dpos); cudaFree(dids); cudaFree(dout); cudaFree(dn);
}

// ------------------------------------------------------------------ QSA
static uint32_t okey(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static std::vector<int32_t> ref_topk(const std::vector<float>& sc, int n) {
    std::vector<int32_t> idx(n); std::iota(idx.begin(), idx.end(), 0);
    if (n > kTopK) {
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return okey(sc[a]) > okey(sc[b]); });
        idx.resize(kTopK);
        std::sort(idx.begin(), idx.end());
    }
    return idx;
}
static void test_topk() {
    std::mt19937 rng(21);
    for (int n : {0, 1, 100, 511, 512, 513, 1000, 3000, 5000}) {
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> sc(n);
            std::normal_distribution<float> N(0.f, 4.f);
            for (int i = 0; i < n; ++i) sc[i] = mode == 0 ? N(rng) : (mode == 1 ? (float)(int)(N(rng)) : 1.0f);   // random / many ties / all equal
            float* dsc = dup(sc); int32_t* dids = dalloc<int32_t>(kTopK + 8); int32_t* dno = dalloc<int32_t>(1);
            CUOK(cudaMemset(dids, 0xFF, (kTopK + 8) * 4));
            CHECK(topk_512(dsc, n, dids, dno, 0) == 0);
            CUOK(cudaDeviceSynchronize());
            const auto want = ref_topk(sc, n);
            int32_t no = down(dno, 1)[0];
            CHECK(no == (int)want.size());
            auto got = down(dids, want.size());
            CHECK(got == want);
            // invariants: ids unique, ascending, in range; every selected key >= every unselected key
            bool asc = true; for (size_t i = 1; i < got.size(); ++i) asc &= got[i] > got[i - 1];
            CHECK(asc);
            if (n > kTopK) { uint32_t mn = ~0u; for (int id : got) mn = std::min(mn, okey(sc[id])); std::vector<char> in(n, 0); for (int id : got) in[id] = 1;
                bool ok = true; for (int i = 0; i < n; ++i) if (!in[i] && okey(sc[i]) > mn) ok = false; CHECK(ok); }
            cudaFree(dsc); cudaFree(dids); cudaFree(dno);
        }
    }
    CHECK(topk_512(nullptr, 5, nullptr, nullptr, 0) == -1);
}

static void test_qsa_pipeline() {
    QsaShape s{4, 64, 3, 32};
    const int n_kv = 1500, row = s.head_dim, cap = 2048;
    std::mt19937 rng(33); std::normal_distribution<float> N(0.f, 1.f);
    std::vector<float> krows((size_t)n_kv * row), vrows((size_t)n_kv * row), ikeys((size_t)n_kv * s.index_dim);
    for (auto& e : krows) e = N(rng); for (auto& e : vrows) e = N(rng); for (auto& e : ikeys) e = N(rng);
    uint16_t* dK = dalloc<uint16_t>((size_t)cap * row); uint16_t* dV = dalloc<uint16_t>((size_t)cap * row); uint16_t* dI = dalloc<uint16_t>((size_t)cap * s.index_dim);
    CUOK(cudaMemset(dK, 0, (size_t)cap * row * 2)); CUOK(cudaMemset(dV, 0, (size_t)cap * row * 2)); CUOK(cudaMemset(dI, 0, (size_t)cap * s.index_dim * 2));
    for (int p = 0; p < n_kv; ++p) {
        float* a = dup(std::vector<float>(krows.begin() + (size_t)p * row, krows.begin() + (size_t)(p + 1) * row));
        float* b = dup(std::vector<float>(vrows.begin() + (size_t)p * row, vrows.begin() + (size_t)(p + 1) * row));
        float* c = dup(std::vector<float>(ikeys.begin() + (size_t)p * s.index_dim, ikeys.begin() + (size_t)(p + 1) * s.index_dim));
        if (p % 2 == 0) { CHECK(kv_append(a, dK, row, p, 0) == 0); }
        else { CHECK(qsa_step_fill(p, n_kv, 0, 0) == 0); CHECK(kv_append_step(a, dK, row, 0) == 0); }   // step form reads pos from scratch
        CHECK(kv_append(b, dV, row, p, 0) == 0);
        CHECK(indexer_key_append(c, dI, s.index_dim, p, 0) == 0);
        CUOK(cudaDeviceSynchronize());
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    auto hK = down(dK, (size_t)n_kv * row);
    int bad_app = 0; for (size_t i = 0; i < hK.size(); ++i) bad_app += hK[i] != h16(krows[i]);
    CHECK(bad_app == 0);
    // gather: fp16 roundtrip
    std::vector<int32_t> ids = {0, 7, 1499, 300, 301, 2}; int32_t* dids = dup(ids); float* dg = dalloc<float>(ids.size() * row);
    CHECK(kv_gather(dK, dids, (int)ids.size(), row, dg, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto g = down(dg, ids.size() * row);
    int bad_g = 0; for (size_t i = 0; i < ids.size(); ++i) for (int d = 0; d < row; ++d) bad_g += g[i * row + d] != f16f(h16(krows[(size_t)ids[i] * row + d]));
    CHECK(bad_g == 0);
    CHECK(qsa_step_fill(0, 0, (int)ids.size(), 0) == 0);
    float* dg2 = dalloc<float>(ids.size() * row);
    CHECK(kv_gather_step(dK, dids, (int)ids.size(), row, dg2, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    CHECK(down(dg2, ids.size() * row) == g);                                   // step form == plain form
    // indexer
    std::vector<float> iq((size_t)s.index_heads * s.index_dim), hw(s.index_heads);
    for (auto& e : iq) e = N(rng); for (auto& e : hw) e = N(rng);
    float* diq = dup(iq); float* dhw = dup(hw); float* dsc = dalloc<float>(cap); float* dsc2 = dalloc<float>(cap);
    CHECK(qsa_index(s, diq, dhw, dI, n_kv, dsc, 0) == 0);
    CHECK(qsa_step_fill(0, n_kv, 0, 0) == 0);
    CHECK(qsa_index_step(s, diq, dhw, dI, cap, dsc2, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto sc = down(dsc, n_kv), sc2 = down(dsc2, n_kv);
    int bad_idx = 0, bad_step = 0;
    for (int j = 0; j < n_kv; ++j) {
        double tot = 0;
        for (int h = 0; h < s.index_heads; ++h) { double dot = 0; for (int d = 0; d < s.index_dim; ++d) dot += (double)iq[(size_t)h * s.index_dim + d] * f16f(h16(ikeys[(size_t)j * s.index_dim + d])); tot += hw[h] * (dot > 0.0 ? dot : 0.0); }
        if (std::fabs(sc[j] - tot) > 1e-3 * (1.0 + std::fabs(tot))) ++bad_idx;
        if (sc[j] != sc2[j]) ++bad_step;
    }
    CHECK(bad_idx == 0); CHECK(bad_step == 0);
    // top-512 on the indexer scores (step form), then attention over the selection
    int32_t* dsel = dalloc<int32_t>(kTopK); int32_t* dno = dalloc<int32_t>(1);
    CHECK(topk_512_step(dsc, dsel, dno, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    CHECK(down(dno, 1)[0] == kTopK);
    auto sel = down(dsel, kTopK);
    CHECK(sel == ref_topk(sc, n_kv));
    std::vector<float> q((size_t)s.n_heads * s.head_dim); for (auto& e : q) e = N(rng);
    float* dq = dup(q); float* dout = dalloc<float>(q.size()); float* dout2 = dalloc<float>(q.size());
    const float scale = 1.f / std::sqrt((float)s.head_dim);
    CHECK(qsa_attend(s, dq, dK, dV, dsel, kTopK, scale, dout, 0) == 0);
    CHECK(qsa_step_fill(0, n_kv, kTopK, 0) == 0);
    CHECK(qsa_attend_step(s, dq, dK, dV, dsel, scale, dout2, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto o = down(dout, q.size()), o2 = down(dout2, q.size());
    int bad_att = 0; double rowsum_dev = 0;
    for (int h = 0; h < s.n_heads; ++h) {
        std::vector<double> lg(kTopK); double m = -1e30;
        for (int j = 0; j < kTopK; ++j) { double dot = 0; for (int d = 0; d < s.head_dim; ++d) dot += (double)q[(size_t)h * s.head_dim + d] * f16f(h16(krows[(size_t)sel[j] * row + d])); lg[j] = dot * scale; if ((double)lg[j] > (double)m) m = lg[j]; }
        double den = 0; for (auto& l : lg) { l = std::exp(l - m); den += l; }
        for (int d = 0; d < s.head_dim; ++d) { double acc = 0; for (int j = 0; j < kTopK; ++j) acc += lg[j] * f16f(h16(vrows[(size_t)sel[j] * row + d])); acc /= den;
            if (std::fabs(o[(size_t)h * s.head_dim + d] - acc) > 2e-3 * (1.0 + std::fabs(acc))) ++bad_att; }
    }
    CHECK(bad_att == 0); CHECK(o == o2); (void)rowsum_dev;
    // n_sel = 0 gives zeros; one selected row gives exactly that V row (softmax of a single logit is 1)
    CHECK(qsa_attend(s, dq, dK, dV, dsel, 0, scale, dout, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    { auto z = down(dout, q.size()); bool allz = true; for (float e : z) allz &= e == 0.f; CHECK(allz); }
    CHECK(qsa_attend(s, dq, dK, dV, dsel, 1, scale, dout, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    { auto one = down(dout, q.size()); int bad1 = 0; for (int h = 0; h < s.n_heads; ++h) for (int d = 0; d < s.head_dim; ++d) if (std::fabs(one[(size_t)h * s.head_dim + d] - f16f(h16(vrows[(size_t)sel[0] * row + d]))) > 1e-5f) ++bad1; CHECK(bad1 == 0); }
    CHECK(qsa_attend(s, dq, dK, dV, dsel, kTopK + 1, scale, dout, 0) == -1);
    // gate
    const int ng = 1000; std::vector<float> at(ng), gt(ng); for (int i = 0; i < ng; ++i) { at[i] = N(rng); gt[i] = N(rng) * 3.f; }
    std::vector<uint16_t> gh(ng); for (int i = 0; i < ng; ++i) gh[i] = h16(gt[i]);
    float* dat = dup(at); uint16_t* dgh = dup(gh); float* dgf = dup(gt); float* dgo = dalloc<float>(ng); float* dgo2 = dalloc<float>(ng);
    CHECK(qsa_gate_apply(dat, dgh, dgo, ng, 0) == 0); CHECK(qsa_gate_apply_f32(dat, dgf, dgo2, ng, 0) == 0);
    CUOK(cudaDeviceSynchronize());
    auto go = down(dgo, ng), go2 = down(dgo2, ng);
    int bad_gate = 0;
    for (int i = 0; i < ng; ++i) {
        const double w1 = at[i] / (1.0 + std::exp(-(double)f16f(gh[i]))), w2 = at[i] / (1.0 + std::exp(-(double)gt[i]));
        if (std::fabs(go[i] - w1) > 1e-4 * (1 + std::fabs(w1))) ++bad_gate;
        if (std::fabs(go2[i] - w2) > 1e-4 * (1 + std::fabs(w2))) ++bad_gate;
        if (!(std::fabs(go2[i]) <= std::fabs(at[i]) * 1.0001f)) ++bad_gate;       // sigmoid in (0,1) never amplifies
    }
    CHECK(bad_gate == 0);
    CHECK(qsa_gate_apply(dat, dgh, dgo, 0, 0) == -1);
    CHECK(step_upload_raw(nullptr, nullptr, 0, 0) == -1);
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_quantize(); test_kv_q8(); test_topk(); test_qsa_pipeline();
    CUOK(cudaDeviceSynchronize());
    std::printf("unit95 gpu: %ld checks, %ld failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
