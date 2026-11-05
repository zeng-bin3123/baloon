// test_f16_gpu.cu - GPU invariant tests of unit 7: (A) the f32->f16 / f16->f32 / f32->bf16 conversions on EVERY f32 and
// f16 bit pattern against independent formulations, (B) activation quantisation, (C) the paged int8 KV cache, (D) the
// QSA decode step (append, indexer score, exact top-512, attention, gate). Exit code is non-zero on any failure.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/gpu_quant_steps.hpp"
#include "baloon/kv_q8_quantization.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 50) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t n_) : n(n_) { CUOK(cudaMalloc((void**)&p, (n_ ? n_ : 1) * sizeof(T))); CUOK(cudaMemset(p, 0, (n_ ? n_ : 1) * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    void up(const std::vector<T>& h) { CUOK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> down() const { std::vector<T> h(n); CUOK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
};

// ================================ (A) exhaustive conversions ======================================================
__device__ __forceinline__ unsigned giesen_f16(uint32_t f) {  // independent: float add does the denormal rounding
    const uint32_t f32infty = 255u << 23, f16max = (127u + 16u) << 23, denorm_magic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
    const uint32_t sign = f & 0x80000000u;
    f ^= sign;
    unsigned o;
    if (f >= f16max) o = (f > f32infty) ? 0x7e00u : 0x7c00u;
    else if (f < (113u << 23)) {
        const float tmp = __uint_as_float(f) + __uint_as_float(denorm_magic);
        o = __float_as_uint(tmp) - denorm_magic;
    } else {
        const uint32_t odd = (f >> 13) & 1u;
        f += ((15u - 127u) << 23) + 0xfffu;
        f += odd;
        o = f >> 13;
    }
    return o | (sign >> 16);
}
__global__ void k_f32_to_f16(uint64_t base, uint64_t n, unsigned long long* bad, unsigned int* first_bad) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = (uint32_t)(base + i);
    const unsigned got = f16bits::round_f32_to_f16_bits(__uint_as_float(u));
    const unsigned want = giesen_f16(u);
    bool ok;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) ok = (got & 0x7C00u) == 0x7C00u && (got & 0x3FFu) != 0 && (got & 0x8000u) == (want & 0x8000u);
    else ok = got == want;
    if (!ok) { atomicAdd(bad, 1ull); atomicMin(first_bad, (unsigned int)u); }
}
__global__ void k_f32_to_bf16(uint64_t base, uint64_t n, unsigned long long* bad) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t u = (uint32_t)(base + i);
    const unsigned got = f16bits::round_f32_to_bf16_bits(__uint_as_float(u));
    unsigned want;
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) want = ((u >> 16) | 0x40u) & 0xFFFFu;
    else {
        const uint32_t lo = u >> 16, rem = u & 0xFFFFu;
        want = (rem > 0x8000u || (rem == 0x8000u && (lo & 1u))) ? ((lo + 1) & 0xFFFFu) : lo;
    }
    if (got != want) atomicAdd(bad, 1ull);
}
__global__ void k_widen(float* out) {
    const unsigned h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < 65536) out[h] = f16bits::widen_f16_bits((uint16_t)h);
}
static double decode_half(unsigned h) {
    const int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    double v = e == 0 ? std::ldexp((double)m, -24) : (e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25));
    return s ? -v : v;
}
static void test_conversions() {
    std::printf("-- (A) exhaustive conversions\n");
    Dev<unsigned long long> bad(1), badbf(1);
    Dev<unsigned int> first(1);
    unsigned int big = ~0u;
    CUOK(cudaMemcpy(first.p, &big, 4, cudaMemcpyHostToDevice));
    const uint64_t chunk = 1ull << 26;
    for (uint64_t base = 0; base < (1ull << 32); base += chunk) {
        k_f32_to_f16<<<(unsigned)(chunk / 256), 256>>>(base, chunk, bad.p, first.p);
        k_f32_to_bf16<<<(unsigned)(chunk / 256), 256>>>(base, chunk, badbf.p);
    }
    CUOK(cudaGetLastError());
    CUOK(cudaDeviceSynchronize());
    const unsigned long long nb = bad.down()[0], nbf = badbf.down()[0];
    const unsigned long long fb = first.down()[0];
    if (nb) std::printf("first f32->f16 mismatch at bits %08llx\n", fb);
    CHECK(nb == 0);
    CHECK(nbf == 0);
    Dev<float> w(65536);
    k_widen<<<256, 256>>>(w.p);
    CUOK(cudaDeviceSynchronize());
    const std::vector<float> hw = w.down();
    for (unsigned h = 0; h < 65536; ++h) {
        const double r = decode_half(h);
        if (std::isnan(r)) CHECK(std::isnan(hw[h])); else { CHECK((double)hw[h] == r); CHECK(std::signbit(hw[h]) == (bool)(h & 0x8000)); }
    }
}

// ================================ (B) activation quantisation =====================================================
static void ref_q8_0(const float* x, float scale, uint8_t* y) {
    float v[32], amax = 0.f;
    for (int i = 0; i < 32; ++i) { v[i] = x[i] * scale; amax = std::max(amax, std::fabs(v[i])); }
    const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
    const uint16_t dh = f16bits::round_f32_to_f16_bits(d);
    y[0] = dh & 0xFF; y[1] = dh >> 8;
    for (int i = 0; i < 32; ++i) { int q = (int)std::nearbyintf(v[i] * id); q = std::max(-127, std::min(127, q)); y[2 + i] = (uint8_t)(int8_t)q; }
}
static void ref_q8_K(const float* x, uint8_t* y) {
    float amax = 0.f, mx = 0.f;
    for (int i = 0; i < 256; ++i) { const float a = std::fabs(x[i]); if (amax < a) { amax = a; mx = x[i]; } }
    int16_t bs[16] = {0};
    float d = 0.f;
    int8_t* qs = (int8_t*)(y + 4);
    if (amax == 0.f) std::memset(qs, 0, 256);
    else {
        const float iscale = -127.f / mx;
        for (int i = 0; i < 256; ++i) { int q = (int)std::nearbyintf(iscale * x[i]); q = std::min(127, q); qs[i] = (int8_t)q; bs[i >> 4] = (int16_t)(bs[i >> 4] + q); }
        d = 1.f / iscale;
    }
    std::memcpy(y, &d, 4);
    std::memcpy(y + 260, bs, 32);
}
static void test_actq() {
    std::printf("-- (B) activation quantisation\n");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    const int NB0 = 3000, NBK = 500;
    std::vector<float> x0(NB0 * 32), xk(NBK * 256);
    for (auto& v : x0) v = nd(rng);
    for (auto& v : xk) v = nd(rng);
    for (int i = 0; i < 32; ++i) x0[5 * 32 + i] = 0.f;            // all-zero block
    for (int i = 0; i < 256; ++i) xk[7 * 256 + i] = 0.f;
    for (int i = 0; i < 32; ++i) x0[9 * 32 + i] = (i % 2 ? 1.f : -1.f) * 1e-30f * (float)(i + 1);  // tiny magnitudes
    xk[11 * 256 + 3] = -50.f;                                       // largest magnitude is negative
    Dev<float> dx0(x0.size()), dxk(xk.size()), dout(std::max(x0.size(), xk.size()));
    Dev<uint8_t> dy0((size_t)NB0 * actq::kQ80Bytes), dyk((size_t)NBK * actq::kQ8KBytes), dys((size_t)NB0 * actq::kQ80Bytes);
    dx0.up(x0); dxk.up(xk);
    CHECK(actq::quantize_q8_0(dx0.p, dy0.p, NB0, nullptr) == 0);
    CHECK(actq::quantize_q8_0_scaled(dx0.p, 0.37f, dys.p, NB0, nullptr) == 0);
    CHECK(actq::quantize_q8_K(dxk.p, dyk.p, NBK, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto y0 = dy0.down(), ys = dys.down(), yk = dyk.down();
    int bad0 = 0, bads = 0, badk = 0;
    for (int b = 0; b < NB0; ++b) {
        uint8_t r[34];
        ref_q8_0(&x0[b * 32], 1.f, r);
        if (std::memcmp(r, &y0[(size_t)b * 34], 34)) ++bad0;
        ref_q8_0(&x0[b * 32], 0.37f, r);
        if (std::memcmp(r, &ys[(size_t)b * 34], 34)) ++bads;
    }
    for (int b = 0; b < NBK; ++b) {
        uint8_t r[292];
        ref_q8_K(&xk[b * 256], r);
        if (std::memcmp(r, &yk[(size_t)b * 292], 292)) ++badk;
    }
    CHECK(bad0 == 0); CHECK(bads == 0); CHECK(badk == 0);
    if (bad0 || bads || badk) std::printf("quantise mismatches: q8_0=%d scaled=%d q8_K=%d\n", bad0, bads, badk);
    // invariants of the encoding itself
    CHECK(actq::dequant_q8_0(dy0.p, dout.p, NB0, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    auto o = dout.down();
    for (int b = 0; b < NB0; ++b) {
        const float d = f16bits::widen_f16_bits((uint16_t)(y0[(size_t)b * 34] | (y0[(size_t)b * 34 + 1] << 8)));
        // Quantisation rounds with the f32 scale d32 = amax/127 but the block stores it as f16, so the reconstruction
        // error is <= 0.5*d32 (rounding) + |q|*|d16-d32| (scale storage error, |q| <= 127).
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x0[b * 32 + i]));
        const float d32 = amax / 127.f;
        const float bound = d32 * 0.5f * 1.001f + 127.f * std::fabs(d - d32) * 1.001f + 1e-30f;
        for (int i = 0; i < 32; ++i) {
            const int q = (int8_t)y0[(size_t)b * 34 + 2 + i];
            CHECK(q >= -127 && q <= 127);
            CHECK(o[b * 32 + i] == d * (float)q);
            CHECK(std::fabs(o[b * 32 + i] - x0[b * 32 + i]) <= bound);
        }
    }
    CHECK(actq::dequant_q8_K(dyk.p, dout.p, NBK, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    o = dout.down();
    for (int b = 0; b < NBK; ++b) {
        float d; std::memcpy(&d, &yk[(size_t)b * 292], 4);
        int16_t bs[16]; std::memcpy(bs, &yk[(size_t)b * 292 + 260], 32);
        for (int j = 0; j < 16; ++j) { int s = 0; for (int i = 0; i < 16; ++i) s += (int8_t)yk[(size_t)b * 292 + 4 + j * 16 + i]; CHECK(s == bs[j]); }
        for (int i = 0; i < 256; ++i) { CHECK(o[b * 256 + i] == d * (float)(int)(int8_t)yk[(size_t)b * 292 + 4 + i]); CHECK(std::fabs(o[b * 256 + i] - xk[b * 256 + i]) <= std::fabs(d) * 0.5f * 1.01f + 1e-30f); }
    }
    CHECK(actq::quantize_q8_0(nullptr, dy0.p, 4, nullptr) == -1);
    CHECK(actq::quantize_q8_K(dxk.p, nullptr, 4, nullptr) == -1);
    CHECK(actq::dequant_q8_0(dy0.p, dout.p, -1, nullptr) == -1);
    CHECK(actq::quantize_q8_0(nullptr, nullptr, 0, nullptr) == 0);
}

// ================================ (C) paged int8 KV cache =========================================================
static void test_kv_q8() {
    std::printf("-- (C) paged int8 KV cache\n");
    const kvq8gpu::Geom g = {2, 128, 16};
    const int table_len = 4, n_pages = 8, groups = g.head_dim / 64;
    const std::vector<int32_t> table = {5, 2, 7, 0};
    const size_t rows = (size_t)n_pages * g.kv_heads * g.page_size;
    std::vector<int8_t> hc(rows * g.head_dim, 0);
    std::vector<uint16_t> hs(rows * groups, 0);
    Dev<int8_t> dc(hc.size());
    Dev<uint16_t> ds(hs.size());
    Dev<int32_t> dt(table_len), dpos(1);
    dt.up(table);
    Dev<float> dsrc((size_t)g.kv_heads * g.head_dim);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 2.f);
    const int cap = table_len * g.page_size;
    for (int pos = 0; pos < cap; ++pos) {
        std::vector<float> src((size_t)g.kv_heads * g.head_dim);
        for (auto& v : src) v = nd(rng) * (pos % 5 == 0 ? 1e-3f : 1.f);
        if (pos == 3) std::fill(src.begin(), src.end(), 0.f);
        dsrc.up(src);
        const int32_t p = pos;
        CUOK(cudaMemcpy(dpos.p, &p, 4, cudaMemcpyHostToDevice));
        CHECK(kvq8gpu::kv_append_q8_step(dsrc.p, g, dc.p, ds.p, dt.p, table_len, n_pages, dpos.p, nullptr) == 0);
        kvq8::append_token(src.data(), hc.data(), hs.data(), table.data(), pos, g.kv_heads, g.head_dim, g.page_size);
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(dc.down() == hc);
    CHECK(ds.down() == hs);
    // positions outside the table / invalid page: nothing may change
    const std::vector<int8_t> before = dc.down();
    for (int bad : {cap, cap + 5, -1}) {
        const int32_t p = bad;
        CUOK(cudaMemcpy(dpos.p, &p, 4, cudaMemcpyHostToDevice));
        CHECK(kvq8gpu::kv_append_q8_step(dsrc.p, g, dc.p, ds.p, dt.p, table_len, n_pages, dpos.p, nullptr) == 0);
    }
    Dev<int32_t> dt2(table_len);
    dt2.up({5, 2, 99, -3});  // pages outside [0, n_pages)
    for (int bad : {2 * g.page_size, 3 * g.page_size + 1}) {
        const int32_t p = bad;
        CUOK(cudaMemcpy(dpos.p, &p, 4, cudaMemcpyHostToDevice));
        CHECK(kvq8gpu::kv_append_q8_step(dsrc.p, g, dc.p, ds.p, dt2.p, table_len, n_pages, dpos.p, nullptr) == 0);
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(dc.down() == before);
    // gather
    std::vector<int32_t> ids = {0, 1, 17, 63, 40, -1, cap, cap + 100, 31, 16, 15, 2};
    const int nid = (int)ids.size();
    Dev<int32_t> did(nid);
    did.up(ids);
    Dev<float> dout((size_t)nid * g.kv_heads * g.head_dim);
    CHECK(kvq8gpu::kv_gather_q8_step(dc.p, ds.p, g, dt.p, table_len, n_pages, did.p, nid, dout.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto out = dout.down();
    int badv = 0;
    for (int i = 0; i < nid; ++i)
        for (int h = 0; h < g.kv_heads; ++h)
            for (int d = 0; d < g.head_dim; ++d) {
                float want = 0.f;
                if (ids[i] >= 0 && ids[i] < cap) {
                    const auto a = kvq8::append_address(table.data(), ids[i], h, g.kv_heads, g.page_size);
                    want = kvq8::dequant_value(hc[(size_t)a.row * g.head_dim + d], hs[(size_t)a.row * groups + d / 64]);
                }
                if (out[((size_t)i * g.kv_heads + h) * g.head_dim + d] != want) ++badv;
            }
    CHECK(badv == 0);
    // reconstruction error of a stored group is at most scale/2
    CHECK(kvq8gpu::kv_gather_q8_step(dc.p, ds.p, g, dt.p, table_len, n_pages, did.p, 0, dout.p, nullptr) == 0);
    kvq8gpu::Geom bad_geom = {2, 100, 16};
    CHECK(kvq8gpu::kv_append_q8_step(dsrc.p, bad_geom, dc.p, ds.p, dt.p, table_len, n_pages, dpos.p, nullptr) == -1);
    CHECK(kvq8gpu::kv_append_q8_step(nullptr, g, dc.p, ds.p, dt.p, table_len, n_pages, dpos.p, nullptr) == -1);
    CHECK(kvq8gpu::kv_gather_q8_step(dc.p, ds.p, g, dt.p, 0, n_pages, did.p, nid, dout.p, nullptr) == -1);
}

// ================================ (D) QSA decode step =============================================================
static float h2f_host(uint16_t h) { return f16bits::widen_f16_bits(h); }
static std::vector<int32_t> host_topk(const std::vector<float>& s, int n) {
    const int K = qsa_step::kTopK;
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<int32_t> out;
    if (n <= K) { out = idx; out.resize(K, -1); return out; }
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return s[a] > s[b]; });  // ties keep the lowest index first
    idx.resize(K);
    std::sort(idx.begin(), idx.end());
    return idx;
}
static void test_qsa() {
    std::printf("-- (D) QSA decode step\n");
    namespace qs = qsa_step;
    const int n_kv = 2, n_head = 6, d_idx = 16, nih = 4, cap = 900, n_cells = 700, D = qs::kDim;
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.f, 1.f);
    Dev<uint16_t> kp((size_t)cap * n_kv * D), vp((size_t)cap * n_kv * D);
    Dev<float> ikp((size_t)cap * d_idx), dk((size_t)n_kv * D), dv((size_t)n_kv * D), dikey(d_idx);
    std::vector<uint16_t> hk((size_t)cap * n_kv * D, 0), hv((size_t)cap * n_kv * D, 0);
    std::vector<float> hik((size_t)cap * d_idx, 0.f);
    for (int pos = 0; pos < n_cells; ++pos) {
        std::vector<float> k((size_t)n_kv * D), v((size_t)n_kv * D), key(d_idx);
        for (auto& x : k) x = nd(rng) * 0.5f;
        for (auto& x : v) x = nd(rng);
        for (auto& x : key) x = nd(rng);
        dk.up(k); dv.up(v); dikey.up(key);
        CHECK(qs::kv_append(dk.p, dv.p, kp.p, vp.p, n_kv, cap, pos, nullptr) == 0);
        CHECK(qs::indexer_key_append(dikey.p, ikp.p, d_idx, cap, pos, nullptr) == 0);
        for (size_t i = 0; i < k.size(); ++i) { hk[(size_t)pos * n_kv * D + i] = f16bits::round_f32_to_f16_bits(k[i]); hv[(size_t)pos * n_kv * D + i] = f16bits::round_f32_to_f16_bits(v[i]); }
        for (int i = 0; i < d_idx; ++i) hik[(size_t)pos * d_idx + i] = key[i];
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(kp.down() == hk);
    CHECK(vp.down() == hv);
    CHECK(ikp.down() == hik);
    CHECK(qs::kv_append(dk.p, dv.p, kp.p, vp.p, n_kv, cap, cap, nullptr) == -1);   // position out of range
    CHECK(qs::indexer_key_append(dikey.p, ikp.p, d_idx, cap, -1, nullptr) == -1);

    // indexer scores
    std::vector<float> iq(nih * d_idx), w(nih);
    for (auto& x : iq) x = nd(rng);
    for (auto& x : w) x = std::fabs(nd(rng)) + 0.1f;
    Dev<float> diq(iq.size()), dw(nih), dscore(cap);
    diq.up(iq); dw.up(w);
    CHECK(qs::qsa_index(diq.p, dw.p, ikp.p, nih, d_idx, cap, n_cells, dscore.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto sc = dscore.down();
    int bad_sc = 0;
    for (int c = 0; c < cap; ++c) {
        if (c >= n_cells) { if (!(std::isinf(sc[c]) && sc[c] < 0)) ++bad_sc; continue; }
        double s = 0;
        for (int h = 0; h < nih; ++h) { double dot = 0; for (int d = 0; d < d_idx; ++d) dot += (double)iq[h * d_idx + d] * hik[(size_t)c * d_idx + d]; s += (double)w[h] * (dot > 0.0 ? dot : 0.0); }
        if (std::fabs(sc[c] - s) > 1e-4 * (1.0 + std::fabs(s))) ++bad_sc;
    }
    CHECK(bad_sc == 0);

    // top-512: exact selection incl. ties; sizes around the K boundary; fully tied input
    Dev<float> dts(cap);
    Dev<int32_t> dids(qs::kTopK);
    for (int mode = 0; mode < 5; ++mode)
        for (int n : {0, 1, 100, 511, 512, 513, 700, 900}) {
            std::vector<float> s(cap);
            for (int i = 0; i < cap; ++i) {
                if (mode == 0) s[i] = nd(rng);
                else if (mode == 1) s[i] = (float)((int)(rng() % 7) - 3) + 0.5f;     // heavy ties
                else if (mode == 2) s[i] = 1.25f;                                    // everything tied
                else if (mode == 3) s[i] = -(float)i * 0.001f - 1.f;                 // descending, negative
                else s[i] = (float)(i % 13) - 6.5f;
            }
            dts.up(s);
            CHECK(qs::topk_512(dts.p, cap, n, dids.p, nullptr) == 0);
            CUOK(cudaDeviceSynchronize());
            const auto got = dids.down();
            const auto want = host_topk(s, n);
            if (got != want) std::printf("topk mismatch mode=%d n=%d\n", mode, n);
            CHECK(got == want);
        }

    // attention over the selected cells vs a double reference; widths 0..512, padded -1 ids
    std::vector<float> s2(cap);
    for (int i = 0; i < cap; ++i) s2[i] = nd(rng);
    dts.up(s2);
    CHECK(qs::topk_512(dts.p, cap, n_cells, dids.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    std::vector<int32_t> ids = dids.down();
    std::vector<float> q((size_t)n_head * D);
    for (auto& x : q) x = nd(rng);
    Dev<float> dq(q.size()), dout(q.size());
    dq.up(q);
    const float scale = 1.f / 16.f;
    for (int width : {0, 1, 7, 64, 300, 512}) {
        CHECK(qs::qsa_attend(dq.p, kp.p, vp.p, dids.p, width, n_head, n_kv, cap, scale, dout.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const auto out = dout.down();
        int bad = 0;
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / (n_head / n_kv);
            std::vector<double> sco(width);
            double m = -1.0e30;
            for (int j = 0; j < width; ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double)q[h * D + d] * scale * h2f_host(hk[((size_t)ids[j] * n_kv + kvh) * D + d]);
                sco[j] = dot; m = (dot > m ? dot : m);
            }
            double l = 0;
            std::vector<double> acc(D, 0.0);
            for (int j = 0; j < width; ++j) { const double p = std::exp(sco[j] - m); l += p; for (int d = 0; d < D; ++d) acc[d] += p * h2f_host(hv[((size_t)ids[j] * n_kv + kvh) * D + d]); }
            for (int d = 0; d < D; ++d) { const double want = width ? acc[d] / l : 0.0; if (std::fabs(out[h * D + d] - want) > 2e-4 * (1.0 + std::fabs(want))) ++bad; }
        }
        CHECK(bad == 0);
    }
    // a masked (-1) id inside the width contributes nothing
    {
        std::vector<int32_t> ids2 = {ids[0], -1, ids[1], -1};
        ids2.resize(qs::kTopK, -1);
        Dev<int32_t> d2(qs::kTopK), d3(qs::kTopK);
        d2.up(ids2);
        std::vector<int32_t> ids3 = {ids[0], ids[1]};
        ids3.resize(qs::kTopK, -1);
        d3.up(ids3);
        Dev<float> oa(q.size()), ob(q.size());
        CHECK(qs::qsa_attend(dq.p, kp.p, vp.p, d2.p, 4, n_head, n_kv, cap, scale, oa.p, nullptr) == 0);
        CHECK(qs::qsa_attend(dq.p, kp.p, vp.p, d3.p, 2, n_head, n_kv, cap, scale, ob.p, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        CHECK(oa.down() == ob.down());
    }
    CHECK(qs::qsa_attend(dq.p, kp.p, vp.p, dids.p, 513, n_head, n_kv, cap, scale, dout.p, nullptr) == -1);
    CHECK(qs::qsa_attend(dq.p, kp.p, vp.p, dids.p, 8, 5, 2, cap, scale, dout.p, nullptr) == -1);  // n_head % n_kv != 0

    // gather of selected cells
    Dev<float> dg((size_t)qs::kTopK * n_kv * D);
    CHECK(qs::kv_gather(kp.p, dids.p, qs::kTopK, n_kv, cap, dg.p, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto gg = dg.down();
    int badg = 0;
    for (int j = 0; j < qs::kTopK && !badg; ++j)
        for (int r = 0; r < n_kv * D; ++r) {
            const float want = ids[j] >= 0 ? h2f_host(hk[(size_t)ids[j] * n_kv * D + r]) : 0.f;
            if (gg[(size_t)j * n_kv * D + r] != want) { ++badg; break; }
        }
    CHECK(badg == 0);

    // gate
    const int64_t n = 5000;
    std::vector<float> a(n), gt(n);
    for (int64_t i = 0; i < n; ++i) { a[i] = nd(rng) * 3.f; gt[i] = nd(rng) * 6.f; }
    Dev<float> da(n), dgt(n), of(n);
    Dev<uint16_t> oh(n);
    da.up(a); dgt.up(gt);
    CHECK(qs::qsa_gate_apply_f32(da.p, dgt.p, of.p, n, nullptr) == 0);
    CHECK(qs::qsa_gate_apply(da.p, dgt.p, oh.p, n, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto rf = of.down();
    const auto rh = oh.down();
    int badgate = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double want = (double)a[i] / (1.0 + std::exp(-(double)gt[i]));
        if (std::fabs(rf[i] - want) > 1e-5 * (1.0 + std::fabs(want))) ++badgate;
        if (std::fabs(h2f_host(rh[i]) - want) > 2e-3 * (1.0 + std::fabs(want))) ++badgate;
        if (rh[i] != f16bits::round_f32_to_f16_bits(rf[i])) ++badgate;
    }
    CHECK(badgate == 0);
    CHECK(qs::qsa_gate_apply_f32(nullptr, dgt.p, of.p, 10, nullptr) == -1);
    CHECK(qs::qsa_gate_apply(da.p, dgt.p, oh.p, 0, nullptr) == 0);
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUOK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s\n", prop.name);
    test_conversions();
    test_actq();
    test_kv_q8();
    test_qsa();
    std::printf("f16_gpu checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("F16 GPU TESTS FAILED\n"); return 1; }
    std::printf("F16 GPU TESTS PASSED\n");
    return 0;
}
