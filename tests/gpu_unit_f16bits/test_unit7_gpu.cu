// test_unit7_gpu.cu - GPU invariants of the f16/bf16 bit conversions (device == host bit for bit), the activation
// quantizers (Q8_0 ggml/CPU rules, Q8_K, bsums), the Q8 KV cache (append/gather through a permuted page table), and the
// QSA step kernels (step state, fp16 KV, indexer pooling and scoring, top-k with lowest-index ties, attention, gate).
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/f16_bits.hpp"
#include "baloon/kv_q8_kernels.hpp"
#include "baloon/kv_q8_quantization.hpp"
#include "baloon/qsa_cache_indexer.hpp"
#include "baloon/qsa_step_kernels.hpp"
#include "baloon/quantize_act_kernels.hpp"

using namespace baloon;
namespace fb = baloon::f16bits;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <typename T>
struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { CUOK(cudaMalloc((void**)&p, (count ? count : 1) * sizeof(T))); CUOK(cudaMemset(p, 0, (count ? count : 1) * sizeof(T))); }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { if (n) CUOK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> h(n); if (n) CUOK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    void fill_bytes(int v) { CUOK(cudaMemset(p, v, n * sizeof(T))); }
};
static void sync() { CUOK(cudaDeviceSynchronize()); }
static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ---- 1. device conversions == host conversions ---------------------------------------------------------------
__global__ void k_enc(const float* x, uint16_t* o, uint16_t* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { o[i] = fb::f16_from_f32(x[i]); b[i] = fb::round_f32_to_bf16_bits(x[i]); }
}
__global__ void k_dec(const uint16_t* h, float* o) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) o[i] = fb::f32_from_f16(h[i]);
}
static void test_device_conversions() {
    std::vector<uint16_t> all(65536);
    std::iota(all.begin(), all.end(), 0);
    Dev<uint16_t> dh(all);
    Dev<float> df(65536);
    k_dec<<<256, 256>>>(dh.p, df.p);
    sync();
    const auto dec = df.get();
    for (uint32_t h = 0; h < 65536; ++h) {
        const float hv = fb::f32_from_f16((uint16_t)h);
        uint32_t a, b;
        std::memcpy(&a, &dec[h], 4); std::memcpy(&b, &hv, 4);
        CHECK(a == b);
    }
    std::mt19937_64 rng(99);
    const int n = 1 << 20;
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = (i % 4 == 0) ? flt((uint32_t)rng()) : (float)((rng() & 1 ? -1.0 : 1.0) * std::ldexp(1.0 + (double)(rng() & 0x7FFFFF) / 8388608.0, (int)(rng() % 44) - 28));
    Dev<float> dx(x);
    Dev<uint16_t> dout(n), dbf(n);
    k_enc<<<n / 256, 256>>>(dx.p, dout.p, dbf.p, n);
    sync();
    const auto o = dout.get(), bf = dbf.get();
    long bad = 0;
    for (int i = 0; i < n; ++i) if (o[i] != fb::f16_from_f32(x[i]) || bf[i] != fb::round_f32_to_bf16_bits(x[i])) ++bad;
    CHECK(bad == 0);
}

// ---- 2. activation quantizers --------------------------------------------------------------------------------
static int nearest_int_host(float x) { return (int)std::nearbyint(x); }
static void test_quantize_act() {
    namespace qa = quant_act;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 3.f);
    const int64_t n = 32 * 4096;
    std::vector<float> x(n);
    for (auto& v : x) v = nd(rng);
    for (int j = 0; j < 32; ++j) x[32 * 10 + j] = 0.f;                                   // an all-zero block
    x[32 * 11 + 5] = 1000.f;                                                              // an outlier block
    Dev<float> dx(x);
    Dev<uint8_t> q80(n / 32 * qa::kQ8_0Bytes);

    CHECK(qa::quantize_q8_0(dx.p, q80.p, n, nullptr) == 0);
    sync();
    auto q = q80.get();
    long mism = 0, total = 0;
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        const uint16_t dh = (uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8));
        CHECK(std::abs((int)dh - (int)fb::f16_from_f32(d)) <= 1);
        const float dd = fb::f32_from_f16(dh);
        for (int j = 0; j < 32; ++j) {
            const int c = (int8_t)q[b * 34 + 2 + j];
            const int r = (int)std::round(x[b * 32 + j] * id);
            ++total; mism += std::abs(c - r) > 1; 
            CHECK(c >= -128 && c <= 127);
            CHECK(std::fabs(c * dd - x[b * 32 + j]) <= dd * 0.6f + amax * 2e-3f + 1e-30f);   // error bound ~ d/2 + fp16 scale error
        }
    }
    CHECK(mism == 0);
    // dequant kernel agrees with the layout
    Dev<float> yd(n);
    CHECK(qa::dequant_q8_0(q80.p, yd.p, n, nullptr) == 0);
    sync();
    const auto y = yd.get();
    for (int64_t i = 0; i < n; i += 7) {
        const int64_t b = i / 32;
        const uint16_t dh = (uint16_t)(q[b * 34] | (q[b * 34 + 1] << 8));
        CHECK(y[i] == fb::f32_from_f16(dh) * (float)(int8_t)q[b * 34 + 2 + i % 32]);
    }

    // CPU-parity rule: symmetric range, scale applied first
    const float sc = 0.37f;
    CHECK(qa::quantize_q8_0_scaled(dx.p, q80.p, n, sc, nullptr) == 0);
    sync();
    q = q80.get();
    mism = 0;
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j] * sc));
        const float d = amax / 127.f;
        for (int j = 0; j < 32; ++j) {
            const int c = (int8_t)q[b * 34 + 2 + j];
            CHECK(c >= -127 && c <= 127);                                                 // never -128
            const int r = d != 0.f ? nearest_int_host(x[b * 32 + j] * sc / d) : 0;
            mism += std::abs(c - r) > 1;
        }
        if (amax == 0.f) for (int j = 0; j < 32; ++j) CHECK((int8_t)q[b * 34 + 2 + j] == 0);
    }
    CHECK(mism == 0);

    // Q8_K
    const int64_t nk = 256 * 512;
    std::vector<float> xk(nk);
    for (auto& v : xk) v = nd(rng);
    for (int j = 0; j < 256; ++j) xk[256 * 3 + j] = 0.f;
    Dev<float> dxk(xk);
    Dev<uint8_t> qk(nk / 256 * qa::kQ8KBytes);
    CHECK(qa::quantize_q8_K(dxk.p, qk.p, nk, nullptr) == 0);
    sync();
    const auto kq = qk.get();
    for (int64_t b = 0; b < nk / 256; ++b) {
        const uint8_t* blk = &kq[b * 292];
        float d; std::memcpy(&d, blk, 4);
        const int8_t* qs = (const int8_t*)(blk + 4);
        int16_t bs[16]; std::memcpy(bs, blk + 260, 32);
        float amax = 0.f, mx = 0.f;
        for (int j = 0; j < 256; ++j) if (std::fabs(xk[b * 256 + j]) > amax) { amax = std::fabs(xk[b * 256 + j]); mx = xk[b * 256 + j]; }
        for (int k = 0; k < 16; ++k) {
            int s = 0;
            for (int l = 0; l < 16; ++l) s += qs[k * 16 + l];
            CHECK(bs[k] == s);                                                            // bsums are the exact block sums
            CHECK(std::abs(s) <= 2032);                                                   // fits an int16 with room
        }
        if (amax == 0.f) { CHECK(d == 0.f); for (int j = 0; j < 256; ++j) CHECK(qs[j] == 0); continue; }
        const float iscale = -127.f / mx;
        long m2 = 0;
        for (int j = 0; j < 256; ++j) {
            CHECK(qs[j] >= -127 && qs[j] <= 127);
            const int r = std::min(127, nearest_int_host(iscale * xk[b * 256 + j]));
            m2 += std::abs(qs[j] - r) > 1;
            CHECK(std::fabs(qs[j] * d - xk[b * 256 + j]) <= std::fabs(d) * 0.6f + 1e-6f);
        }
        CHECK(m2 == 0);
        CHECK(std::fabs(d - 1.f / iscale) <= std::fabs(d) * 1e-5f);
    }
    Dev<float> yk(nk);
    CHECK(qa::dequant_q8_K(qk.p, yk.p, nk, nullptr) == 0);
    sync();
    const auto ykh = yk.get();
    for (int64_t i = 0; i < nk; i += 11) {
        float d; std::memcpy(&d, &kq[(i / 256) * 292], 4);
        CHECK(ykh[i] == d * (float)(int8_t)kq[(i / 256) * 292 + 4 + i % 256]);
    }
    // rejections
    CHECK(qa::quantize_q8_0(dx.p, q80.p, 33, nullptr) == -1);
    CHECK(qa::quantize_q8_K(nullptr, qk.p, 256, nullptr) == -1);
    CHECK(qa::quantize_q8_0_scaled(dx.p, q80.p, 32, NAN, nullptr) == -1);
    CHECK(qa::dequant_q8_K(qk.p, yk.p, 0, nullptr) == -1);
}

// ---- 3. Q8 KV cache ------------------------------------------------------------------------------------------
static void test_kv_q8() {
    const int kv_heads = 3, head_dim = 128, page_size = 8, n_pages = 6;
    const int64_t pool_rows = (int64_t)n_pages * kv_heads * page_size, groups = head_dim / 64;
    const std::vector<int32_t> table = {4, 0, 5, 2, 1, 3};                                // permuted physical pages
    Dev<int32_t> dt(table);
    Dev<int8_t> codes(pool_rows * head_dim);
    Dev<uint16_t> scales(pool_rows * groups);
    codes.fill_bytes(0x55);
    scales.fill_bytes(0x55);
    std::vector<int8_t> hc((size_t)pool_rows * head_dim, 0x55);
    std::vector<uint16_t> hs((size_t)pool_rows * groups, 0x5555);
    std::mt19937 rng(8);
    std::normal_distribution<float> nd(0.f, 2.f);
    const int n_tok = 40;
    std::vector<std::vector<float>> toks(n_tok, std::vector<float>(kv_heads * head_dim));
    for (int pos = 0; pos < n_tok; ++pos) {
        for (auto& v : toks[pos]) v = nd(rng);
        if (pos == 7) std::fill(toks[pos].begin(), toks[pos].end(), 0.f);                 // zero group: scale 0, codes 0
        Dev<float> ds(toks[pos]);
        CHECK(kv_q8_dev::kv_append_q8_step(ds.p, codes.p, scales.p, dt.p, (int64_t)table.size(), pos, kv_heads, head_dim, page_size, pool_rows, nullptr) == 0);
        sync();
        kvq8::append_token(toks[pos].data(), hc.data(), hs.data(), table.data(), pos, kv_heads, head_dim, page_size);
    }
    const auto gc = codes.get();
    const auto gs = scales.get();
    long worse = 0, untouched_bad = 0;
    std::vector<char> touched((size_t)pool_rows, 0);
    for (int pos = 0; pos < n_tok; ++pos)
        for (int h = 0; h < kv_heads; ++h) touched[kvq8::append_address(table.data(), pos, h, kv_heads, page_size).row] = 1;
    for (int64_t r = 0; r < pool_rows; ++r) {
        if (!touched[r]) {                                                                // rows never appended keep the canary
            for (int d = 0; d < head_dim; ++d) untouched_bad += gc[r * head_dim + d] != 0x55;
            for (int g = 0; g < groups; ++g) untouched_bad += gs[r * groups + g] != 0x5555;
            continue;
        }
        for (int g = 0; g < groups; ++g) {
            CHECK(std::abs((int)gs[r * groups + g] - (int)hs[r * groups + g]) <= 1);
            for (int j = 0; j < 64; ++j) {
                const int c = gc[r * head_dim + g * 64 + j];
                CHECK(c >= -127 && c <= 127);                                             // never -128
                worse += std::abs(c - (int)hc[r * head_dim + g * 64 + j]) > 1;
            }
        }
    }
    CHECK(worse == 0);
    CHECK(untouched_bad == 0);
    // gather: quantization error bound, masked ids, order of the [id][head][dim] layout
    std::vector<int32_t> ids = {0, 39, 7, 15, -1, 16, 1000, 23};
    Dev<int32_t> di(ids);
    Dev<float> out(ids.size() * kv_heads * head_dim);
    CHECK(kv_q8_dev::kv_gather_q8_step(codes.p, scales.p, dt.p, (int64_t)table.size(), di.p, (int)ids.size(), kv_heads, head_dim, page_size, pool_rows, out.p, nullptr) == 0);
    sync();
    const auto og = out.get();
    for (size_t k = 0; k < ids.size(); ++k)
        for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const float got = og[(k * kv_heads + h) * head_dim + d];
                if (ids[k] < 0 || ids[k] >= n_tok) { CHECK(got == 0.f || ids[k] >= n_tok); continue; }
                const float want = toks[ids[k]][h * head_dim + d];
                float amax = 0.f;
                for (int j = 0; j < 64; ++j) amax = std::max(amax, std::fabs(toks[ids[k]][h * head_dim + (d / 64) * 64 + j]));
                CHECK(std::fabs(got - want) <= amax / 127.f * 0.55f + amax * 1e-3f + 1e-7f);
            }
    // rejections: nothing launched
    CHECK(kv_q8_dev::kv_append_q8_step(nullptr, codes.p, scales.p, dt.p, 6, 0, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1);
    CHECK(kv_q8_dev::kv_append_q8_step((const float*)codes.p, codes.p, scales.p, dt.p, 6, 0, kv_heads, 100, page_size, pool_rows, nullptr) == -1);  // head_dim % 64
    CHECK(kv_q8_dev::kv_append_q8_step((const float*)codes.p, codes.p, scales.p, dt.p, 6, 6 * page_size, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1);  // pos past table
    CHECK(kv_q8_dev::kv_gather_q8_step(codes.p, scales.p, dt.p, 6, nullptr, 3, kv_heads, head_dim, page_size, pool_rows, out.p, nullptr) == -1);
}

// ---- 4. QSA step kernels -------------------------------------------------------------------------------------
namespace qs = baloon::qsa_step;
static double h2d(uint16_t h) { return (double)fb::f32_from_f16(h); }

static void test_step_state() {
    Dev<int32_t> ds(4);
    for (int pos : {0, 1, 2, 3, 4, 5, 100, 2046, 2047, 2048, 2049, 2050, 40000}) {
        CHECK(qs::step_upload(ds.p, pos, qs::kTopK, qs::kBlockR, nullptr) == 0);
        sync();
        const auto s = ds.get();
        const qs::Step e = qs::make_step(pos);
        const auto c = qsa_cache::step_fill((uint32_t)pos);
        CHECK(s[0] == pos && s[1] == e.n_kv && s[2] == e.n_bid && s[3] == e.width);
        CHECK((uint32_t)s[1] == c.n_kv && (uint32_t)s[2] == c.n_bid && (uint32_t)s[3] == c.width);
        CHECK(s[1] == pos + 1 && s[2] * 4 <= s[1] && s[1] < (s[2] + 1) * 4 && s[3] <= s[1] && s[3] <= qs::kMaxWidth);
        if (pos + 1 < qs::kMaxWidth) CHECK(s[3] == pos + 1);                              // identity below the bound
        CHECK(s[3] == std::min(pos + 1, 2051));
    }
    const qs::Step raw{9, 8, 7, 6};
    CHECK(qs::step_upload_raw(ds.p, raw, nullptr) == 0);
    auto s = ds.get();
    CHECK(s[0] == 9 && s[1] == 8 && s[2] == 7 && s[3] == 6);
    CHECK(qs::step_upload_width(ds.p, 3, nullptr) == 0);
    s = ds.get();
    CHECK(s[0] == 9 && s[1] == 8 && s[2] == 7 && s[3] == 3);
    CHECK(qs::step_upload(nullptr, 3, 2048, 4, nullptr) == -1 && qs::step_upload(ds.p, -1, 2048, 4, nullptr) == -1);
    CHECK(qs::step_upload_width(ds.p, -2, nullptr) == -1);
    CHECK(qs::qsa_step_fill(ds.p, 0, 4, nullptr) == -1);
    void* a = qs::step_scratch(1024);
    CHECK(a != nullptr && cudaMemset(a, 1, 1024) == cudaSuccess);
    void* b = qs::step_scratch(16);
    CHECK(b == a);                                                                         // never shrinks / reallocates
    void* c2 = qs::step_scratch(1 << 20);
    CHECK(c2 != nullptr && cudaMemset(c2, 2, 1 << 20) == cudaSuccess);
}

static void test_fp16_kv() {
    const int kv_heads = 2, head_dim = 96, page_size = 4, n_pages = 5;
    const int64_t pool_rows = (int64_t)n_pages * kv_heads * page_size;
    const std::vector<int32_t> table = {3, 1, 4, 0, 2};
    Dev<int32_t> dt(table);
    Dev<uint16_t> pool(pool_rows * head_dim);
    pool.fill_bytes(0x7B);
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 40.f);
    const int n_tok = 18;
    std::vector<std::vector<float>> toks(n_tok, std::vector<float>(kv_heads * head_dim));
    std::vector<char> touched(pool_rows, 0);
    for (int pos = 0; pos < n_tok; ++pos) {
        for (auto& v : toks[pos]) v = nd(rng);
        toks[pos][0] = 1e6f;                                                               // overflows fp16 -> +inf
        toks[pos][1] = 6e-8f;                                                              // fp16 subnormal range
        Dev<float> ds(toks[pos]);
        CHECK(qs::kv_append_step(ds.p, pool.p, dt.p, (int64_t)table.size(), pos, kv_heads, head_dim, page_size, pool_rows, nullptr) == 0);
        sync();
        for (int h = 0; h < kv_heads; ++h) touched[((int64_t)table[pos / page_size] * kv_heads + h) * page_size + pos % page_size] = 1;
    }
    const auto hp = pool.get();
    for (int pos = 0; pos < n_tok; ++pos)
        for (int h = 0; h < kv_heads; ++h) {
            const int64_t row = ((int64_t)table[pos / page_size] * kv_heads + h) * page_size + pos % page_size;
            for (int d = 0; d < head_dim; ++d) CHECK(hp[row * head_dim + d] == fb::f16_from_f32(toks[pos][h * head_dim + d]));
        }
    long untouched_bad = 0;
    for (int64_t r = 0; r < pool_rows; ++r)
        if (!touched[r]) for (int d = 0; d < head_dim; ++d) untouched_bad += hp[r * head_dim + d] != 0x7B7B;
    CHECK(untouched_bad == 0);
    CHECK(hp[(((int64_t)table[0] * kv_heads) * page_size) * head_dim] == 0x7C00);          // finite overflow became +inf, not NaN
    const std::vector<int32_t> ids = {17, 0, 5, -3, 9, 77};
    Dev<int32_t> di(ids);
    Dev<float> out(ids.size() * kv_heads * head_dim);
    CHECK(qs::kv_gather_step(pool.p, dt.p, (int64_t)table.size(), di.p, (int)ids.size(), kv_heads, head_dim, page_size, pool_rows, out.p, nullptr) == 0);
    sync();
    const auto og = out.get();
    for (size_t k = 0; k < ids.size(); ++k)
        for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
                const float got = og[(k * kv_heads + h) * head_dim + d];
                if (ids[k] < 0 || ids[k] >= (int)table.size() * page_size) { CHECK(got == 0.f); continue; }
                const float want = fb::f32_from_f16(fb::f16_from_f32(toks[ids[k]][h * head_dim + d]));
                CHECK(std::memcmp(&got, &want, 4) == 0);
            }
    CHECK(qs::kv_append_step(nullptr, pool.p, dt.p, 5, 0, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1);
    CHECK(qs::kv_append_step((const float*)pool.p, pool.p, dt.p, 5, 5 * page_size, kv_heads, head_dim, page_size, pool_rows, nullptr) == -1);
    CHECK(qs::kv_gather_step(pool.p, dt.p, 5, nullptr, 3, kv_heads, head_dim, page_size, pool_rows, out.p, nullptr) == -1);
}

static void test_indexer() {
    const int D = qs::kKeyDim, max_cells = 64;
    std::mt19937 rng(40);
    std::normal_distribution<float> nd(0.f, 1.f);
    Dev<float> tail(3 * D), pooled((max_cells / 4 + 2) * D);
    std::vector<std::vector<float>> keys(12, std::vector<float>(D));
    const float eps = 1e-6f;
    for (int pos = 0; pos < 12; ++pos) {
        for (auto& v : keys[pos]) v = nd(rng);
        Dev<float> dk(keys[pos]);
        CHECK(qs::indexer_key_append(dk.p, tail.p, pooled.p, pos, max_cells, eps, nullptr) == 0);
        sync();
        const auto t = tail.get();
        if (pos % 4 < 3) for (int d = 0; d < D; ++d) CHECK(t[(pos % 4) * D + d] == keys[pos][d]);   // raw tail slot
        if (pos % 4 == 3) {                                                                        // block completes: pooled row
            const int b = pos / 4;
            const auto pl = pooled.get();
            std::vector<double> mean(D);
            double ss = 0;
            for (int d = 0; d < D; ++d) {
                mean[d] = (keys[4 * b][d] + keys[4 * b + 1][d] + keys[4 * b + 2][d] + keys[pos][d]) * 0.25;
                ss += mean[d] * mean[d];
            }
            const double inv = 1.0 / std::sqrt(ss / D + eps);
            double rms = 0;
            for (int d = 0; d < D; ++d) {
                CHECK(std::fabs(pl[(size_t)b * D + d] - mean[d] * inv) <= 1e-4 + 1e-4 * std::fabs(mean[d] * inv));
                rms += (double)pl[(size_t)b * D + d] * pl[(size_t)b * D + d];
            }
            CHECK(std::fabs(std::sqrt(rms / D) - 1.0) < 1e-3);                                     // unit RMS
        }
    }
    CHECK(qs::indexer_key_append(nullptr, tail.p, pooled.p, 0, max_cells, eps, nullptr) == -1);
    CHECK(qs::indexer_key_append((const float*)tail.p, tail.p, pooled.p, max_cells, max_cells, eps, nullptr) == -1);
    CHECK(qs::indexer_key_append((const float*)tail.p, tail.p, pooled.p, 0, max_cells, 0.f, nullptr) == -1);

    // scoring
    for (int n_kv : {1, 3, 4, 5, 8, 17, 64, 101}) {
        const int nb = n_kv / 4 + (n_kv % 4 != 0), rows = nb;
        std::vector<float> pl((size_t)rows * D), q(4 * D), hw(4);
        for (auto& v : pl) v = nd(rng) * 0.2f;
        for (auto& v : q) v = nd(rng) * 0.2f;
        for (auto& v : hw) v = std::fabs(nd(rng)) + 0.1f;
        Dev<float> dp(pl), dq(q), dw(hw), dc(n_kv);
        CHECK(qs::qsa_index_step(dp.p, dq.p, dw.p, n_kv, rows, dc.p, nullptr) == 0);
        sync();
        const auto cs = dc.get();
        for (int j = 0; j < n_kv; ++j) {
            const int b = j / 4;
            double s = 0;
            for (int h = 0; h < 4; ++h) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double)q[h * D + d] * pl[(size_t)b * D + d];
                s += hw[h] * std::max<double>(dot, 0.0);
            }
            const bool tail_cell = j >= (n_kv / 4) * 4;
            if (tail_cell) { CHECK(std::fabs((double)cs[j] - (s + 1e9)) <= 256.0); CHECK(cs[j] >= 0.99e9f); }   // +1e9 on the tail only
            else { CHECK(std::fabs((double)cs[j] - s) <= 2e-4 * std::fabs(s) + 1e-4); CHECK(cs[j] < 1e6f); }
            if (j % 4) CHECK(cs[j] == cs[j - 1]);                                                  // cells of a block share a score
        }
    }
    CHECK(qs::qsa_index_step(nullptr, nullptr, nullptr, 4, 1, nullptr, nullptr) == -1);
    Dev<float> z(8);
    CHECK(qs::qsa_index_step(z.p, z.p, z.p, 17, 2, z.p, nullptr) == -1);                   // rows < blocks needed
}

static void test_topk() {
    std::mt19937 rng(61);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Case { int n; int levels; };                                                    // levels > 0: heavy ties
    for (Case cs : {Case{5, 0}, Case{100, 0}, Case{2051, 0}, Case{2052, 0}, Case{4000, 0}, Case{4000, 8}, Case{32768, 0},
                    Case{32768, 5}, Case{3000, 1}, Case{600, 3}}) {
        std::vector<float> sc(cs.n);
        for (auto& v : sc) v = cs.levels ? (float)(int)(rng() % cs.levels) - 2.f : nd(rng);
        const int width = std::min(cs.n, qs::kMaxWidth);
        Dev<float> d(sc);
        Dev<int32_t> ids(width), nout(1);
        CHECK(qs::topk_512_step(d.p, cs.n, width, ids.p, nout.p, nullptr) == 0);
        sync();
        const auto got = ids.get();
        const int n_out = nout.get()[0];
        std::vector<uint32_t> keys(cs.n);
        for (int i = 0; i < cs.n; ++i) keys[i] = qsa_cache::order_key_f(sc[i]);
        const auto want = qsa_cache::topk_ids(keys, (uint32_t)width);
        CHECK(n_out == (int)want.size() && n_out == std::min(cs.n, width));
        bool same = n_out == (int)want.size();
        for (int i = 0; same && i < n_out; ++i) same = got[i] == (int32_t)want[i];
        CHECK(same);                                                                      // exact set, ascending, lowest-index ties
        for (int i = 1; i < n_out; ++i) CHECK(got[i] > got[i - 1]);
        for (int i = n_out; i < width; ++i) CHECK(got[i] == -1);                           // padding
    }
    // a smaller width than the cell count and a width of one
    std::vector<float> sc(700);
    for (auto& v : sc) v = nd(rng);
    Dev<float> d(sc);
    for (int width : {1, 2, 64, 699}) {
        Dev<int32_t> ids(width), nout(1);
        CHECK(qs::topk_512_step(d.p, 700, width, ids.p, nout.p, nullptr) == 0);
        sync();
        std::vector<uint32_t> keys(700);
        for (int i = 0; i < 700; ++i) keys[i] = qsa_cache::order_key_f(sc[i]);
        const auto want = qsa_cache::topk_ids(keys, (uint32_t)width);
        const auto got = ids.get();
        CHECK(nout.get()[0] == width);
        for (int i = 0; i < width; ++i) CHECK(got[i] == (int32_t)want[i]);
    }
    Dev<int32_t> ids(8), nout(1);
    CHECK(qs::topk_512_step(d.p, 700, 0, ids.p, nout.p, nullptr) == -1);
    CHECK(qs::topk_512_step(d.p, 700, qs::kMaxWidth + 1, ids.p, nout.p, nullptr) == -1);
    CHECK(qs::topk_512_step(nullptr, 700, 4, ids.p, nout.p, nullptr) == -1);
}

static void run_attend(int n_head, int n_kv_heads, int head_dim, int n_sel, bool mask_all, uint32_t seed) {
    const int page_size = 16, n_pages = 160;
    const int64_t pool_rows = (int64_t)n_pages * n_kv_heads * page_size;
    std::vector<int32_t> table(n_pages);
    std::iota(table.begin(), table.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(table.begin(), table.end(), rng);
    const int n_cells = n_pages * page_size;
    std::normal_distribution<float> nd(0.f, 1.f);
    Dev<int32_t> dt(table);
    Dev<uint16_t> kp(pool_rows * head_dim), vp(pool_rows * head_dim);
    std::vector<std::vector<float>> K(n_cells, std::vector<float>(n_kv_heads * head_dim)), V = K;
    const int filled = std::min(n_cells, std::max(n_sel + 8, 64));
    for (int pos = 0; pos < filled; ++pos) {
        for (auto& v : K[pos]) v = nd(rng);
        for (auto& v : V[pos]) v = nd(rng);
        Dev<float> a(K[pos]), b(V[pos]);
        CHECK(qs::kv_append_step(a.p, kp.p, dt.p, n_pages, pos, n_kv_heads, head_dim, page_size, pool_rows, nullptr) == 0);
        CHECK(qs::kv_append_step(b.p, vp.p, dt.p, n_pages, pos, n_kv_heads, head_dim, page_size, pool_rows, nullptr) == 0);
    }
    std::vector<int32_t> ids(n_sel);
    std::vector<int32_t> perm(filled);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < n_sel; ++i) ids[i] = mask_all ? -1 : (i % 9 == 4 ? -1 : perm[i % filled]);   // some masked ids
    std::vector<float> q(n_head * head_dim);
    for (auto& v : q) v = nd(rng);
    const float scale = 1.0f / std::sqrt((float)head_dim);
    Dev<float> dq(q), dout(n_head * head_dim);
    Dev<int32_t> di(ids);
    CHECK(qs::qsa_attend_step(dq.p, kp.p, vp.p, dt.p, n_pages, di.p, n_sel, n_head, n_kv_heads, head_dim, page_size, pool_rows, scale, dout.p, nullptr) == 0);
    sync();
    const auto got = dout.get();
    const int G = n_head / n_kv_heads;
    for (int h = 0; h < n_head; ++h) {
        const int kvh = h / G;
        std::vector<double> s(n_sel, -1e30);
        double m = -1e30;
        for (int i = 0; i < n_sel; ++i) {
            if (ids[i] < 0) continue;
            double dot = 0;
            for (int d = 0; d < head_dim; ++d)
                dot += (double)q[h * head_dim + d] * scale * (double)fb::f32_from_f16(fb::f16_from_f32(K[ids[i]][kvh * head_dim + d]));
            s[i] = dot; m = std::max(m, dot);
        }
        double L = 0;
        std::vector<double> acc(head_dim, 0.0);
        for (int i = 0; i < n_sel; ++i) {
            if (ids[i] < 0) continue;
            const double e = std::exp(s[i] - m);
            L += e;
            for (int d = 0; d < head_dim; ++d) acc[d] += e * (double)fb::f32_from_f16(fb::f16_from_f32(V[ids[i]][kvh * head_dim + d]));
        }
        for (int d = 0; d < head_dim; ++d) {
            const double want = L > 0 ? acc[d] / L : 0.0;                                   // nothing attended -> zeros
            CHECK(std::fabs(got[h * head_dim + d] - want) <= 2e-4 + 2e-4 * std::fabs(want));
        }
    }
}
static void test_attend() {
    run_attend(12, 1, 256, 130, false, 1);                                                 // GQA 12 q heads / kv head, HD 256
    run_attend(24, 2, 64, 300, false, 2);
    run_attend(4, 4, 32, 1, false, 3);                                                     // single cell: weight exactly 1
    run_attend(12, 1, 64, 64, true, 4);                                                    // fully masked -> zeros
    run_attend(12, 1, 128, 2051, false, 5);                                                // the largest selection
    Dev<float> z(16);
    Dev<uint16_t> zp(16);
    Dev<int32_t> zi(4);
    CHECK(qs::qsa_attend_step(z.p, zp.p, zp.p, zi.p, 4, zi.p, 0, 12, 1, 64, 16, 16, 1.f, z.p, nullptr) == -1);        // n_sel == 0
    CHECK(qs::qsa_attend_step(z.p, zp.p, zp.p, zi.p, 4, zi.p, 2052, 12, 1, 64, 16, 16, 1.f, z.p, nullptr) == -1);     // beyond the cap
    CHECK(qs::qsa_attend_step(z.p, zp.p, zp.p, zi.p, 4, zi.p, 4, 10, 4, 64, 16, 16, 1.f, z.p, nullptr) == -1);        // heads % kv != 0
}

static void test_gate() {
    const int64_t n = 5000;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> x(n), g(n);
    std::vector<uint16_t> gh(n);
    for (int64_t i = 0; i < n; ++i) { x[i] = nd(rng); g[i] = nd(rng); gh[i] = fb::f16_from_f32(g[i]); }
    Dev<float> dx1(x), dx2(x), dg(g);
    Dev<uint16_t> dgh(gh);
    CHECK(qs::qsa_gate_apply_f32(dx1.p, dg.p, n, nullptr) == 0);
    CHECK(qs::qsa_gate_apply(dx2.p, dgh.p, n, nullptr) == 0);
    sync();
    const auto a = dx1.get(), b = dx2.get();
    for (int64_t i = 0; i < n; ++i) {
        const double s1 = 1.0 / (1.0 + std::exp(-(double)g[i]));
        const double s2 = 1.0 / (1.0 + std::exp(-h2d(gh[i])));
        CHECK(std::fabs(a[i] - x[i] * s1) <= 1e-5 * std::fabs(x[i]) + 1e-7);
        CHECK(std::fabs(b[i] - x[i] * s2) <= 1e-5 * std::fabs(x[i]) + 1e-7);
        CHECK(std::fabs(a[i]) <= std::fabs(x[i]) * (1.0 + 1e-6));                          // a sigmoid gate never amplifies
    }
    CHECK(qs::qsa_gate_apply(nullptr, dgh.p, n, nullptr) == -1 && qs::qsa_gate_apply_f32(dx1.p, dg.p, 0, nullptr) == -1);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no GPU\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_device_conversions(); std::printf("[done] device conversions   fail=%ld\n", g_fail);
    test_quantize_act();       std::printf("[done] activation quantize  fail=%ld\n", g_fail);
    test_kv_q8();              std::printf("[done] kv q8                fail=%ld\n", g_fail);
    test_step_state();         std::printf("[done] step state           fail=%ld\n", g_fail);
    test_fp16_kv();            std::printf("[done] fp16 kv              fail=%ld\n", g_fail);
    test_indexer();            std::printf("[done] indexer              fail=%ld\n", g_fail);
    test_topk();               std::printf("[done] topk                 fail=%ld\n", g_fail);
    test_attend();             std::printf("[done] attend               fail=%ld\n", g_fail);
    test_gate();               std::printf("[done] gate                 fail=%ld\n", g_fail);
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT7 GPU TESTS FAILED\n"); return 1; }
    std::printf("UNIT7 GPU TESTS PASSED\n");
    return 0;
}
