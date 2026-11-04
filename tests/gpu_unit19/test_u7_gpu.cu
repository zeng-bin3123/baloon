// test_u7_gpu.cu - GPU invariant tests of the activation quantizers (quantize_act.cu), the Q8 KV cache kernels (kv_q8.cu)
// and the QSA decode-step kernels (qsa.cu). Each check compares against an independent host reference or a stated
// invariant (clamping, group max, paged addressing, ties-to-lowest-index top-k, softmax normalisation, masking).
// Exit code is non-zero on any failure.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/act_quant_kernels.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_kernels.hpp"
#include "baloon/kv_q8_quantization.hpp"
#include "baloon/qsa_step_kernels.hpp"

using namespace baloon;
namespace fb = baloon::f16bits;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { if (++g_fail <= 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T>
struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { CUOK(cudaMalloc((void**)&p, (count ? count : 1) * sizeof(T))); CUOK(cudaMemset(p, 0, (count ? count : 1) * sizeof(T))); }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { if (n) CUOK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T>& h) { CUOK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); if (n) CUOK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    void fill_bytes(int v) { CUOK(cudaMemset(p, v, n * sizeof(T))); }
};

static std::mt19937_64 rng(12345);
static float urand(float lo, float hi) { return lo + (hi - lo) * (float)((rng() >> 11) * (1.0 / 9007199254740992.0)); }
static int irand(int n) { return (int)(rng() % (uint64_t)n); }

// ======================================================= activation quantizers ==============================================
static void host_q8_0(const float* x, float scale, uint8_t* o) {
    float v[32], amax = 0.f;
    for (int j = 0; j < 32; ++j) { v[j] = x[j] * scale; amax = std::max(amax, std::fabs(v[j])); }
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t hb = fb::f16_from_f32(d);
    o[0] = (uint8_t)(hb & 255); o[1] = (uint8_t)(hb >> 8);
    for (int j = 0; j < 32; ++j) { int q = (int)std::round(v[j] * id); q = std::max(-127, std::min(127, q)); o[2 + j] = (uint8_t)(int8_t)q; }
}
static void host_q8_K(const float* x, uint8_t* o) {
    int8_t* qs = (int8_t*)(o + 4); int16_t* bs = (int16_t*)(o + 260);
    float mx = 0.f, amax = 0.f;
    for (int j = 0; j < 256; ++j) if (std::fabs(x[j]) > amax) { amax = std::fabs(x[j]); mx = x[j]; }
    float d = 0.f;
    if (amax == 0.f) { std::memset(o + 4, 0, 288); }
    else {
        const float iscale = -127.f / mx;
        for (int j = 0; j < 16; ++j) { int s = 0; for (int l = 0; l < 16; ++l) { int q = (int)std::nearbyint(iscale * x[j * 16 + l]); q = std::min(127, q); qs[j * 16 + l] = (int8_t)q; s += q; } bs[j] = (int16_t)s; }
        d = 1.f / iscale;
    }
    std::memcpy(o, &d, 4);
}

static std::vector<float> act_input(size_t n, int mode) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) {
        switch (mode) {
            case 0: x[i] = urand(-3.f, 3.f); break;
            case 1: x[i] = urand(-0.2f, 0.2f); break;
            case 2: x[i] = urand(-3e4f, 3e4f); break;
            default: x[i] = (i / 32) % 3 == 0 ? 0.f : urand(-1.f, 1.f); break;   // some all-zero blocks
        }
    }
    return x;
}

static void test_act_quant() {
    const size_t n = 256 * 37;
    for (int mode = 0; mode < 4; ++mode) {
        const std::vector<float> x = act_input(n, mode);
        Dev<float> dx(x);
        // Q8_0
        Dev<uint8_t> q0(n / 32 * actq::kBlockQ8_0Bytes);
        CHECK(actq::quantize_q8_0(dx.p, q0.p, (int64_t)n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<uint8_t> g0 = q0.get(), r0(g0.size());
        for (size_t b = 0; b < n / 32; ++b) host_q8_0(&x[b * 32], 1.f, &r0[b * 34]);
        CHECK(g0 == r0);
        Dev<float> back(n);
        CHECK(actq::dequant_q8_0(q0.p, back.p, (int64_t)n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const std::vector<float> y = back.get();
        for (size_t b = 0; b < n / 32; ++b) {
            const float d = fb::f32_from_f16((uint16_t)(g0[b * 34] | (g0[b * 34 + 1] << 8)));
            float amax = 0.f; for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
            for (int j = 0; j < 32; ++j) {
                const int8_t q = (int8_t)g0[b * 34 + 2 + j];
                CHECK(q >= -127 && q <= 127);
                // reconstruction error <= half a step of the unrounded scale, plus the fp16 rounding of the stored scale
                CHECK(std::fabs(y[b * 32 + j] - x[b * 32 + j]) <= amax / 254.f * 1.002f + amax * 1.2e-3f + 1e-30f);
            }
            if (amax == 0.f) CHECK(d == 0.f);
        }
        // Q8_0 scaled == quantizing the pre-scaled input
        const float sc = 0.37f;
        Dev<uint8_t> q0s(n / 32 * 34);
        CHECK(actq::quantize_q8_0_scaled(dx.p, sc, q0s.p, (int64_t)n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<uint8_t> gs = q0s.get(), rs(gs.size());
        for (size_t b = 0; b < n / 32; ++b) host_q8_0(&x[b * 32], sc, &rs[b * 34]);
        CHECK(gs == rs);
        // Q8_K
        Dev<uint8_t> qk(n / 256 * actq::kBlockQ8KBytes);
        CHECK(actq::quantize_q8_K(dx.p, qk.p, (int64_t)n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<uint8_t> gk = qk.get(), rk(gk.size());
        for (size_t b = 0; b < n / 256; ++b) host_q8_K(&x[b * 256], &rk[b * 292]);
        CHECK(gk == rk);
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* o = &gk[b * 292];
            float d; std::memcpy(&d, o, 4);
            int mx_i = 0;
            for (int j = 1; j < 256; ++j) if (std::fabs(x[b * 256 + j]) > std::fabs(x[b * 256 + mx_i])) mx_i = j;
            for (int j = 0; j < 16; ++j) {
                int16_t bs; std::memcpy(&bs, o + 260 + 2 * j, 2);
                int s = 0; for (int l = 0; l < 16; ++l) s += (int8_t)o[4 + j * 16 + l];
                CHECK(bs == s);                                           // bsums are the sums of the codes
            }
            for (int j = 0; j < 256; ++j) { const int q = (int8_t)o[4 + j]; CHECK(q >= -127 && q <= 127); }
            if (std::fabs(x[b * 256 + mx_i]) > 0.f) CHECK((int8_t)o[4 + mx_i] == -127);   // the max-magnitude element maps to -127
            else CHECK(d == 0.f);
        }
        Dev<float> backk(n);
        CHECK(actq::dequant_q8_K(qk.p, backk.p, (int64_t)n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const std::vector<float> yk = backk.get();
        for (size_t b = 0; b < n / 256; ++b) {
            float d; std::memcpy(&d, &gk[b * 292], 4);
            float amax = 0.f; for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[b * 256 + j]));
            for (int j = 0; j < 256; ++j) {
                CHECK(yk[b * 256 + j] == d * (float)(int8_t)gk[b * 292 + 4 + j]);
                CHECK(std::fabs(yk[b * 256 + j] - x[b * 256 + j]) <= amax / 254.f * 1.002f + 1e-30f);
            }
        }
    }
    // rejected launches
    Dev<float> a(64); Dev<uint8_t> o(1024);
    CHECK(actq::quantize_q8_0(a.p, o.p, 33, nullptr) == -1);
    CHECK(actq::quantize_q8_0(nullptr, o.p, 32, nullptr) == -1);
    CHECK(actq::quantize_q8_K(a.p, o.p, 128, nullptr) == -1);
    CHECK(actq::dequant_q8_K(o.p, a.p, 100, nullptr) == -1);
    CHECK(actq::quantize_q8_0_scaled(a.p, INFINITY, o.p, 32, nullptr) == -1);
    CHECK(actq::quantize_q8_0(a.p, o.p, 0, nullptr) == 0);
}

// ======================================================= Q8 KV cache ========================================================
static void test_kv_q8() {
    const int kv_heads = 2, head_dim = 128, page_size = 16, n_table = 8, n_pages = 10, groups = head_dim / 64;
    std::vector<int32_t> table(n_table);
    { std::vector<int32_t> pg(n_pages); std::iota(pg.begin(), pg.end(), 0); std::shuffle(pg.begin(), pg.end(), rng); for (int i = 0; i < n_table; ++i) table[i] = pg[i]; }
    const size_t rows = (size_t)n_pages * kv_heads * page_size;
    std::vector<int8_t> hc(rows * head_dim, 0x55);
    std::vector<uint16_t> hs(rows * groups, 0x5555);
    Dev<int8_t> dc(rows * head_dim); Dev<uint16_t> ds(rows * groups); Dev<int32_t> dt(table);
    dc.fill_bytes(0x55); CUOK(cudaMemset(ds.p, 0x55, rows * groups * 2));
    const int ncells = n_table * page_size;
    for (int pos = 0; pos < ncells; pos += 1) {
        std::vector<float> src((size_t)kv_heads * head_dim);
        const float amp = (pos % 7 == 0) ? 0.f : (pos % 5 == 0 ? 1e-4f : (pos % 3 == 0 ? 500.f : 2.f));
        for (auto& v : src) v = urand(-amp, amp);
        if (pos % 11 == 3) std::fill(src.begin(), src.begin() + 64, 0.f);   // a zero group inside a live row
        Dev<float> dsrc(src);
        CHECK(kvq8dev::kv_append_q8_step(dsrc.p, dc.p, ds.p, dt.p, pos, kv_heads, head_dim, page_size, n_table, n_pages, nullptr) == 0);
        kvq8::append_token(src.data(), hc.data(), hs.data(), table.data(), pos, kv_heads, head_dim, page_size);
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(dc.get() == hc);            // codes, including every untouched canary row
    CHECK(ds.get() == hs);            // scales
    for (size_t i = 0; i < hc.size(); ++i) if (hc[i] == 0x55) continue;   // (canary rows are covered by the equality above)
    // gather
    std::vector<int32_t> ids;
    for (int i = 0; i < 90; ++i) ids.push_back(irand(ncells));
    ids.push_back(-1); ids.push_back(ncells); ids.push_back(ncells + 40); ids.push_back(0); ids.push_back(ncells - 1);
    const int n_ids = (int)ids.size();
    Dev<int32_t> di(ids); Dev<float> out((size_t)n_ids * kv_heads * head_dim);
    CHECK(kvq8dev::kv_gather_q8_step(dc.p, ds.p, dt.p, di.p, n_ids, kv_heads, head_dim, page_size, n_table, n_pages, out.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const std::vector<float> got = out.get();
    for (int i = 0; i < n_ids; ++i)
        for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                float want = 0.f;
                if (ids[i] >= 0 && ids[i] < ncells) {
                    const kvq8::AppendAddr a = kvq8::append_address(table.data(), ids[i], h, kv_heads, page_size);
                    want = kvq8::dequant_value(hc[(size_t)a.row * head_dim + d], hs[(size_t)a.row * groups + d / 64]);
                }
                CHECK(got[((size_t)i * kv_heads + h) * head_dim + d] == want);
            }
    // quantization invariants on the stored data: |code| <= 127 and the group max maps to +-127 (non-empty groups)
    for (size_t r = 0; r < rows; ++r) {
        if (hs[r * groups] == 0x5555) continue;
        for (int g = 0; g < groups; ++g) {
            int mx = 0;
            for (int t = 0; t < 64; ++t) mx = std::max(mx, std::abs((int)hc[r * head_dim + g * 64 + t]));
            CHECK(mx <= 127);
            if (fb::f32_from_f16(hs[r * groups + g]) != 0.f) CHECK(mx == 127); else CHECK(mx == 0);
        }
    }
    // a page index outside the pool is skipped (nothing written), bad arguments are rejected
    std::vector<int32_t> bad_table = table; bad_table[2] = n_pages + 3;
    Dev<int32_t> dbad(bad_table); Dev<float> one((size_t)kv_heads * head_dim);
    std::vector<int8_t> before = dc.get();
    CHECK(kvq8dev::kv_append_q8_step(one.p, dc.p, ds.p, dbad.p, 2 * page_size + 1, kv_heads, head_dim, page_size, n_table, n_pages, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    CHECK(dc.get() == before);
    CHECK(kvq8dev::kv_append_q8_step(one.p, dc.p, ds.p, dt.p, -1, kv_heads, head_dim, page_size, n_table, n_pages, nullptr) == -1);
    CHECK(kvq8dev::kv_append_q8_step(one.p, dc.p, ds.p, dt.p, ncells, kv_heads, head_dim, page_size, n_table, n_pages, nullptr) == -1);
    CHECK(kvq8dev::kv_append_q8_step(one.p, dc.p, ds.p, dt.p, 0, kv_heads, 96, page_size, n_table, n_pages, nullptr) == -1);
    CHECK(kvq8dev::kv_append_q8_step(nullptr, dc.p, ds.p, dt.p, 0, kv_heads, head_dim, page_size, n_table, n_pages, nullptr) == -1);
    CHECK(kvq8dev::kv_gather_q8_step(dc.p, ds.p, dt.p, nullptr, 1, kv_heads, head_dim, page_size, n_table, n_pages, out.p, nullptr) == -1);
    CHECK(kvq8dev::kv_gather_q8_step(dc.p, ds.p, dt.p, di.p, 0, kv_heads, head_dim, page_size, n_table, n_pages, out.p, nullptr) == 0);
}

// ======================================================= QSA decode step ====================================================
namespace qs = baloon::qsa_step;
static const int KVH = 2, GROUP = 3, HD = 64, PS = 16, NT = 40, NP = 48, NCELLS_MAX = NT * PS;

struct Cache {
    std::vector<int32_t> table;
    std::vector<uint16_t> k, v;     // host mirrors of the fp16 caches
    Dev<int32_t>* dt = nullptr; Dev<uint16_t>* dk = nullptr; Dev<uint16_t>* dv = nullptr;
};
static size_t row_of(const std::vector<int32_t>& table, int c, int h) { return ((size_t)table[c / PS] * KVH + h) * PS + c % PS; }

static void test_step_state() {
    int32_t* s = qs::step_scratch();
    CHECK(s != nullptr && s == qs::step_scratch());
    auto rd = [&]() { int32_t h[8]; CUOK(cudaMemcpy(h, s, sizeof h, cudaMemcpyDeviceToHost)); return std::vector<int32_t>(h, h + 8); };
    CHECK(qs::step_upload(700, nullptr) == 0);
    std::vector<int32_t> a = rd();
    CHECK(a[0] == 700 && a[1] == 701 && a[2] == 512);                 // width = min(n_kv, 512)
    CHECK(qs::step_upload(100, nullptr) == 0);
    a = rd();
    CHECK(a[0] == 100 && a[1] == 101 && a[2] == 101);
    CHECK(qs::step_upload(0, nullptr) == 0);
    a = rd();
    CHECK(a[0] == 0 && a[1] == 1 && a[2] == 1);
    CHECK(qs::step_upload_width(7, nullptr) == 0);
    CHECK(rd()[2] == 7);
    CHECK(qs::step_upload_width(0, nullptr) == -1 && qs::step_upload_width(513, nullptr) == -1);
    CHECK(qs::step_upload(-1, nullptr) == -1);
    CHECK(std::strlen(qs::last_error()) > 0);
    int32_t raw[3] = {9, 8, 6};
    CHECK(qs::step_upload_raw(raw, 3, nullptr) == 0);
    a = rd();
    CHECK(a[0] == 9 && a[1] == 8 && a[2] == 6);
    CHECK(qs::step_upload_raw(raw, 0, nullptr) == -1 && qs::step_upload_raw(raw, 9, nullptr) == -1 && qs::step_upload_raw(nullptr, 1, nullptr) == -1);
}

static void test_kv_and_index_and_attend() {
    Cache c;
    c.table.resize(NT);
    { std::vector<int32_t> pg(NP); std::iota(pg.begin(), pg.end(), 0); std::shuffle(pg.begin(), pg.end(), rng); for (int i = 0; i < NT; ++i) c.table[i] = pg[i]; }
    const size_t rows = (size_t)NP * KVH * PS;
    c.k.assign(rows * HD, 0x5555); c.v.assign(rows * HD, 0x5555);
    Dev<int32_t> dt(c.table); Dev<uint16_t> dk(rows * HD), dv(rows * HD);
    CUOK(cudaMemset(dk.p, 0x55, rows * HD * 2)); CUOK(cudaMemset(dv.p, 0x55, rows * HD * 2));
    const int n_cells = 600;
    Dev<float> dkey((size_t)KVH * HD), dval((size_t)KVH * HD);
    for (int pos = 0; pos < n_cells; ++pos) {
        std::vector<float> k((size_t)KVH * HD), v(k.size());
        for (auto& x : k) x = urand(-4.f, 4.f);
        for (auto& x : v) x = urand(-4.f, 4.f);
        if (pos % 50 == 0) k[3] = 70000.f;               // overflows fp16 -> infinity in the cache (never NaN)
        dkey.put(k); dval.put(v);
        if (pos % 2 == 0) {
            CHECK(qs::kv_append(dkey.p, dval.p, dk.p, dv.p, dt.p, pos, KVH, HD, PS, NT, NP, nullptr) == 0);
        } else {
            CHECK(qs::step_upload(pos, nullptr) == 0);
            CHECK(qs::kv_append_step(dkey.p, dval.p, dk.p, dv.p, dt.p, KVH, HD, PS, NT, NP, nullptr) == 0);
        }
        for (int h = 0; h < KVH; ++h)
            for (int d = 0; d < HD; ++d) {
                c.k[row_of(c.table, pos, h) * HD + d] = fb::f16_from_f32(k[(size_t)h * HD + d]);
                c.v[row_of(c.table, pos, h) * HD + d] = fb::f16_from_f32(v[(size_t)h * HD + d]);
            }
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(dk.get() == c.k);          // exact fp16 bits + untouched canary rows (addressing)
    CHECK(dv.get() == c.v);
    CHECK(c.k[row_of(c.table, 50, 0) * HD + 3] == 0x7C00);

    // ---- gather (host n_ids) and gather_step (n_ids = step[3]) ----
    std::vector<int32_t> ids;
    for (int i = 0; i < 40; ++i) ids.push_back(irand(n_cells));
    ids.push_back(-1); ids.push_back(NCELLS_MAX); ids.push_back(NCELLS_MAX + 9);
    const int n_ids = (int)ids.size();
    Dev<int32_t> di(ids); Dev<float> go((size_t)n_ids * KVH * HD);
    CHECK(qs::kv_gather(dk.p, dt.p, di.p, n_ids, KVH, HD, PS, NT, NP, go.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    std::vector<float> g = go.get();
    for (int i = 0; i < n_ids; ++i)
        for (int h = 0; h < KVH; ++h)
            for (int d = 0; d < HD; ++d) {
                const float want = (ids[i] >= 0 && ids[i] < NCELLS_MAX) ? fb::f32_from_f16(c.k[row_of(c.table, ids[i], h) * HD + d]) : 0.f;
                const float gotv = g[((size_t)i * KVH + h) * HD + d];
                CHECK(gotv == want || (std::isnan(want) && std::isnan(gotv)));
            }
    {   // step gather writes only the first step[3] slots
        int32_t raw[4] = {5, 6, 7, 10};
        CHECK(qs::step_upload_raw(raw, 4, nullptr) == 0);
        go.fill_bytes(0xAB);
        CHECK(qs::kv_gather_step(dk.p, dt.p, di.p, n_ids, KVH, HD, PS, NT, NP, go.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<float> g2 = go.get();
        for (int i = 0; i < n_ids; ++i)
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < HD; ++d) {
                    const float gotv = g2[((size_t)i * KVH + h) * HD + d];
                    if (i < 10) { const float want = (ids[i] >= 0 && ids[i] < NCELLS_MAX) ? fb::f32_from_f16(c.k[row_of(c.table, ids[i], h) * HD + d]) : 0.f; CHECK(gotv == want || (std::isnan(want) && std::isnan(gotv))); }
                    else { uint32_t b; std::memcpy(&b, &gotv, 4); CHECK(b == 0xABABABABu); }
                }
    }

    // ---- indexer: key append, scoring, top-512 ----
    const int max_cells = 3000;
    std::vector<float> hkeys((size_t)max_cells * 128, 0.f);
    Dev<float> dkeys((size_t)max_cells * 128); Dev<float> onekey(128);
    for (int pos = 0; pos < max_cells; ++pos) {
        std::vector<float> key(128);
        for (auto& x : key) x = (pos % 9 == 0) ? 0.5f : urand(-1.f, 1.f);       // every ninth key identical: ties among scores
        onekey.put(key);
        if (pos % 2) { CHECK(qs::step_upload(pos, nullptr) == 0); CHECK(qs::indexer_key_append_step(onekey.p, dkeys.p, max_cells, nullptr) == 0); }
        else CHECK(qs::indexer_key_append(onekey.p, dkeys.p, pos, max_cells, nullptr) == 0);
        std::copy(key.begin(), key.end(), hkeys.begin() + (size_t)pos * 128);
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(dkeys.get() == hkeys);
    CHECK(qs::indexer_key_append(onekey.p, dkeys.p, max_cells, max_cells, nullptr) == -1);
    CHECK(qs::indexer_key_append(onekey.p, dkeys.p, -1, max_cells, nullptr) == -1);

    std::vector<float> hq(4 * 128), hhw(4);
    for (auto& x : hq) x = urand(-1.f, 1.f);
    for (auto& x : hhw) x = urand(0.1f, 1.5f);
    Dev<float> dq(hq), dhw(hhw);
    auto ref_scores = [&](int n) {
        std::vector<double> s(n);
        for (int cidx = 0; cidx < n; ++cidx) {
            double acc = 0;
            for (int h = 0; h < 4; ++h) { double dot = 0; for (int d = 0; d < 128; ++d) dot += (double)hq[h * 128 + d] * hkeys[(size_t)cidx * 128 + d]; acc += hhw[h] * std::max(dot, 0.0); }
            s[cidx] = acc;
        }
        return s;
    };
    Dev<float> dsc(max_cells);
    for (int n : {1, 5, 600, 2999}) {
        dsc.fill_bytes(0);
        CHECK(qs::qsa_index(dq.p, dhw.p, dkeys.p, dsc.p, n, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const std::vector<float> got = dsc.get();
        const std::vector<double> want = ref_scores(n);
        for (int i = 0; i < n; ++i) CHECK(std::fabs(got[i] - want[i]) <= 1e-4 * (1.0 + std::fabs(want[i])));
        for (int i = n; i < max_cells; ++i) CHECK(got[i] == 0.f);   // nothing past n is written
        CHECK(qs::step_upload(n - 1, nullptr) == 0);
        dsc.fill_bytes(0);
        CHECK(qs::qsa_index_step(dq.p, dhw.p, dkeys.p, dsc.p, max_cells, nullptr) == 0);   // n = step[1] = n
        CUOK(cudaDeviceSynchronize());
        const std::vector<float> got2 = dsc.get();
        for (int i = 0; i < n; ++i) CHECK(got2[i] == got[i]);
        for (int i = n; i < max_cells; ++i) CHECK(got2[i] == 0.f);
    }
    CHECK(qs::qsa_index(dq.p, dhw.p, nullptr, dsc.p, 4, nullptr) == -1);
    CHECK(qs::qsa_index(dq.p, dhw.p, dkeys.p, dsc.p, -3, nullptr) == -1);

    // top-512: exact reference = sort by (order key desc, index asc), take 512, report ascending
    auto ref_topk = [&](const std::vector<float>& s) {
        const int n = (int)s.size();
        std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return qs::order_key_host(s[a]) > qs::order_key_host(s[b]); });
        std::vector<int32_t> out(512, -1);
        const int cnt = std::min(n, 512);
        std::vector<int> sel(idx.begin(), idx.begin() + cnt); std::sort(sel.begin(), sel.end());
        for (int i = 0; i < cnt; ++i) out[i] = sel[i];
        return out;
    };
    Dev<int32_t> dout(512), dn(1);
    for (int kind = 0; kind < 5; ++kind)
        for (int n : {1, 100, 511, 512, 513, 1000, 2051, 3000}) {
            std::vector<float> s(n);
            for (int i = 0; i < n; ++i) {
                switch (kind) {
                    case 0: s[i] = urand(-10.f, 10.f); break;
                    case 1: s[i] = (float)irand(4); break;                       // heavy ties
                    case 2: s[i] = 1.5f; break;                                  // all equal: lowest 512 indices
                    case 3: s[i] = -urand(0.f, 1e-3f) - (i % 3 == 0 ? 0.f : 5.f); break;   // negatives
                    default: s[i] = (i % 7 == 0) ? 3.f : (float)irand(2000) / 7.f; break;
                }
            }
            Dev<float> ds(s);
            dout.fill_bytes(0x7F); dn.fill_bytes(0x7F);
            CHECK(qs::topk_512(ds.p, n, dout.p, dn.p, nullptr) == 0);
            CUOK(cudaDeviceSynchronize());
            CHECK(dout.get() == ref_topk(s));
            CHECK(dn.get()[0] == std::min(n, 512));
            CHECK(qs::step_upload(n - 1, nullptr) == 0);
            dout.fill_bytes(0x7F); dn.fill_bytes(0x7F);
            CHECK(qs::topk_512_step(ds.p, dout.p, dn.p, nullptr) == 0);
            CUOK(cudaDeviceSynchronize());
            CHECK(dout.get() == ref_topk(s));
            int32_t st[8]; CUOK(cudaMemcpy(st, qs::step_scratch(), sizeof st, cudaMemcpyDeviceToHost));
            CHECK(st[3] == std::min(n, 512));                                    // n_sel published for the later kernels
        }
    CHECK(qs::topk_512(nullptr, 4, dout.p, dn.p, nullptr) == -1);
    CHECK(qs::topk_512(dsc.p, -1, dout.p, dn.p, nullptr) == -1);

    // ---- attention ----
    const int n_heads = KVH * GROUP;
    std::vector<float> hq2((size_t)n_heads * HD);
    for (auto& x : hq2) x = urand(-1.f, 1.f);
    Dev<float> dq2(hq2), dout2((size_t)n_heads * HD);
    const float scale = 1.f / std::sqrt((float)HD);
    auto ref_attend = [&](const std::vector<int32_t>& sel, int nsel) {
        std::vector<double> o((size_t)n_heads * HD, 0.0);
        for (int hd = 0; hd < n_heads; ++hd) {
            const int kvh = hd / GROUP;
            std::vector<double> sc; std::vector<int> cells;
            for (int j = 0; j < nsel; ++j) {
                if (sel[j] < 0 || sel[j] >= NCELLS_MAX) continue;
                double dot = 0;
                for (int d = 0; d < HD; ++d) dot += (double)hq2[(size_t)hd * HD + d] * fb::f32_from_f16(c.k[row_of(c.table, sel[j], kvh) * HD + d]);
                sc.push_back(dot * scale); cells.push_back(sel[j]);
            }
            if (sc.empty()) continue;
            const double mx = *std::max_element(sc.begin(), sc.end());
            double sum = 0; for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
            for (size_t j = 0; j < sc.size(); ++j)
                for (int d = 0; d < HD; ++d) o[(size_t)hd * HD + d] += sc[j] / sum * fb::f32_from_f16(c.v[row_of(c.table, cells[j], kvh) * HD + d]);
        }
        return o;
    };
    // k cache has +inf entries at cells % 50 == 0 (dim 3 of head 0); keep attended cells away from them so the reference is finite
    auto pick = [&](int count, bool with_bad) {
        std::vector<int32_t> s;
        while ((int)s.size() < count) { const int cell = irand(n_cells); if (cell % 50 != 0) s.push_back(cell); }
        if (with_bad && count >= 4) { s[1] = -1; s[2] = NCELLS_MAX + 5; s[3] = -77; }
        return s;
    };
    for (int count : {1, 7, 64, 300, 512}) {
        std::vector<int32_t> sel = pick(count, true);
        sel.resize(512, -1);
        Dev<int32_t> dsel(sel), dns(std::vector<int32_t>{count});
        dout2.fill_bytes(0xFF);
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dns.p, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<float> got = dout2.get();
        std::vector<double> want = ref_attend(sel, count);
        for (size_t i = 0; i < got.size(); ++i) CHECK(std::fabs(got[i] - want[i]) <= 2e-4 * (1.0 + std::fabs(want[i])));
        int32_t raw[4] = {0, 0, 0, count};
        CHECK(qs::step_upload_raw(raw, 4, nullptr) == 0);
        dout2.fill_bytes(0xFF);
        CHECK(qs::qsa_attend_step(dq2.p, dk.p, dv.p, dt.p, dsel.p, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        std::vector<float> got2 = dout2.get();
        for (size_t i = 0; i < got.size(); ++i) CHECK(got2[i] == got[i]);
    }
    {   // nothing valid -> zeros; nsel = 0 -> zeros
        std::vector<int32_t> sel(512, -1);
        Dev<int32_t> dsel(sel), dns(std::vector<int32_t>{5});
        dout2.fill_bytes(0xFF);
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dns.p, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        for (float x : dout2.get()) CHECK(x == 0.f);
        Dev<int32_t> dz(std::vector<int32_t>{0});
        dout2.fill_bytes(0xFF);
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dz.p, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        for (float x : dout2.get()) CHECK(x == 0.f);
    }
    {   // softmax weights sum to one: with V == constant the output is that constant
        std::vector<uint16_t> ones(rows * HD, fb::f16_from_f32(0.5f));
        Dev<uint16_t> dvone(ones);
        std::vector<int32_t> sel = pick(200, false); sel.resize(512, -1);
        Dev<int32_t> dsel(sel), dns(std::vector<int32_t>{200});
        CHECK(qs::qsa_attend(dq2.p, dk.p, dvone.p, dt.p, dsel.p, dns.p, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        for (float x : dout2.get()) CHECK(std::fabs(x - 0.5f) <= 1e-5f);
    }
    {
        std::vector<int32_t> sel(512, 0);
        Dev<int32_t> dsel(sel), dns(std::vector<int32_t>{1});
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dns.p, KVH, GROUP, 300, PS, NT, NP, scale, dout2.p, nullptr) == -1);   // head_dim > 256
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, nullptr, KVH, GROUP, HD, PS, NT, NP, scale, dout2.p, nullptr) == -1);   // no n_sel
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dns.p, KVH, 0, HD, PS, NT, NP, scale, dout2.p, nullptr) == -1);
        CHECK(qs::qsa_attend(dq2.p, dk.p, dv.p, dt.p, dsel.p, dns.p, KVH, GROUP, HD, PS, NT, NP, NAN, dout2.p, nullptr) == -1);
        CHECK(qs::kv_append(dkey.p, dval.p, dk.p, dv.p, dt.p, NCELLS_MAX, KVH, HD, PS, NT, NP, nullptr) == -1);
        CHECK(qs::kv_gather(nullptr, dt.p, di.p, 1, KVH, HD, PS, NT, NP, go.p, nullptr) == -1);
    }
}

static void test_gate() {
    const int n = 1000;
    std::vector<float> o(n), gf(n); std::vector<uint16_t> gh(n);
    for (int i = 0; i < n; ++i) { o[i] = urand(-2.f, 2.f); gf[i] = urand(-12.f, 12.f); gh[i] = fb::f16_from_f32(gf[i]); }
    Dev<float> d1(o), d2(o), dgf(gf); Dev<uint16_t> dgh(gh);
    CHECK(qs::qsa_gate_apply_f32(d1.p, dgf.p, n, nullptr) == 0);
    CHECK(qs::qsa_gate_apply(d2.p, dgh.p, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const std::vector<float> a = d1.get(), b = d2.get();
    for (int i = 0; i < n; ++i) {
        const double w1 = o[i] / (1.0 + std::exp(-(double)gf[i]));
        const double w2 = o[i] / (1.0 + std::exp(-(double)fb::f32_from_f16(gh[i])));
        CHECK(std::fabs(a[i] - w1) <= 1e-5 * (1.0 + std::fabs(w1)));
        CHECK(std::fabs(b[i] - w2) <= 1e-5 * (1.0 + std::fabs(w2)));
        CHECK(std::fabs(a[i]) <= std::fabs(o[i]) * 1.000001f);          // sigmoid in (0,1) never amplifies
    }
    CHECK(qs::qsa_gate_apply(nullptr, dgh.p, 4, nullptr) == -1);
    CHECK(qs::qsa_gate_apply_f32(d1.p, dgf.p, -1, nullptr) == -1);
    CHECK(qs::qsa_gate_apply_f32(d1.p, dgf.p, 0, nullptr) == 0);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no CUDA device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    qs::set_debug_sync(true);
    test_act_quant();       std::printf("act_quant done (%ld checks, %ld failures)\n", g_checks, g_fail);
    test_kv_q8();           std::printf("kv_q8 done (%ld checks, %ld failures)\n", g_checks, g_fail);
    test_step_state();      std::printf("step_state done (%ld checks, %ld failures)\n", g_checks, g_fail);
    test_kv_and_index_and_attend(); std::printf("qsa done (%ld checks, %ld failures)\n", g_checks, g_fail);
    test_gate();
    CUOK(cudaDeviceSynchronize());
    std::printf("%ld checks, %ld failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
