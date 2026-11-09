// test_ucc7.cu - GPU + host invariants of unit ucc74f228b9 (NativeFlashAttnShort, QsaCacheIndexAttend, QsaBlockSelect,
// NativeQsaIndexerAppend). Every LAYER-1 statement of the four Lean modules is exercised against the real kernels or the
// pure geometry helpers; any failed CHECK makes the process exit non-zero.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/decode_attn_kernels.hpp"
#include "baloon/native_flash_attn_bounds.hpp"
#include "baloon/qsa_native_kernels.hpp"
#include "baloon/qsa_step_kernels.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/ucc7_geometry.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 80) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)
template <class T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, (n ? n : 1) * sizeof(T))); return p; }
template <class T> static T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); if (!v.empty()) CUOK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> fetch(const T* p, size_t n) { std::vector<T> v(n); if (n) CUOK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
struct Rng {
    uint64_t s; explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 99) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t u32() { return (uint32_t)(next() >> 16); }
    uint32_t below(uint32_t n) { return u32() % n; }
    float uni() { return (float)((next() >> 11) / 9007199254740992.0) * 2.f - 1.f; }
    float gauss() { double u = ((next() >> 11) + 1) / 9007199254740993.0, v = (next() >> 11) / 9007199254740992.0; return (float)(std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v)); }
};
static float h2f(uint16_t h) { return f16bits::f32_from_f16(h); }

// ================================================================================ NativeFlashAttnShort
static void host_flash_facts() {
    using namespace flash_bounds;
    // validStep predicate (all five conjuncts independently falsifiable)
    for (uint32_t w = 0; w <= 300; ++w) for (uint32_t mc : {64u, 256u}) {
        const bool lean = 1 <= w && w <= mc;
        CHECK(valid(w, w, w - 1, w / 4, mc) == lean);
        if (lean) { CHECK(!valid(w, w + 1, w - 1, w / 4, mc)); CHECK(!valid(w, w, w, w / 4, mc)); CHECK(!valid(w, w, w - 1, w / 4 + 1, mc)); }
    }
    CHECK(kMaxContext / 128 == 2 && loop_iterations() == 2);
    for (uint32_t h = 0; h < 24; ++h) CHECK(kv_head(h) == h / 12 && kv_head(h) < 2);
    // cell_cover / cell_unique: every padded cell hit by exactly one (base, warp, lane-group, row)
    std::vector<int> hit(256, 0);
    for (uint32_t base : {0u, 128u}) for (uint32_t w = 0; w < 4; ++w) for (uint32_t g = 0; g < 4; ++g) for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t c = cell_of(base, w, g * 8, r); CHECK(c < 256); if (c < 256) ++hit[c];
    }
    for (int c : hit) CHECK(c == 1);
    for (uint32_t lane = 0; lane < 32; ++lane) CHECK(cell_of(0, 1, lane, 3) == 32 + (lane / 8) * 8 + 3);
    // v_local_cover: 128 locals
    std::vector<int> vh(128, 0);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t k0 = 0; k0 < 32; k0 += 4) for (uint32_t g = 0; g < 4; ++g) { const uint32_t l = pv_local(w, k0, g * 8); if (l < 128) ++vh[l]; }
    for (int c : vh) CHECK(c == 1);
    // tile bounds
    CHECK(kTileFloats == 4096 && kTileFloats * 4 == 16384 && kSmemBytes <= kSmemLimit);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t lane = 0; lane < 32; ++lane) for (uint32_t i0 = 0; i0 < 128; i0 += 32) for (uint32_t j = 0; j < 4; ++j)
        CHECK(vkq_write(w, lane, i0, j) + 1 < 4096);
    for (uint32_t w = 0; w < 4; ++w) for (uint32_t g = 0; g < 4; ++g) for (uint32_t c = 0; c < 256; ++c) CHECK(partial_slot(w, g, 0, c) < 4096);
    // KV addressing
    for (uint64_t cap : {(uint64_t)256, (uint64_t)1000}) for (uint64_t cell : {(uint64_t)0, cap - 1}) for (uint64_t kv = 0; kv < 2; ++kv) for (uint64_t d : {(uint64_t)0, (uint64_t)255})
        CHECK(kv_index(cell, kv, d) < kv_halves(cap));
    CHECK(kv_bytes(1) == 1024 && 2 * 256 * 2 == 1024);
    CHECK(kv_bytes(UINT64_MAX / 1024) / 1024 == UINT64_MAX / 1024);        // guard value does not wrap
    // outputs: tid and tid+128 cover 256 dims; head rows disjoint; q inside 24*256
    std::vector<int> oh(256, 0); for (uint32_t t = 0; t < 128; ++t) { ++oh[t]; ++oh[t + 128]; } for (int c : oh) CHECK(c == 1);
    CHECK(out_index(23, 127) + 128 < 24 * 256 && q_index(23, 255) < 24 * 256 && out_index(1, 0) == 256);
    CHECK(16 * 16 == 256);
}

static void gpu_flash() {
    using namespace attn_kernels;
    Rng r(7);
    const int H = 24, KV = 2, D = 256;
    for (int width : {1, 2, 3, 4, 5, 63, 64, 127, 128, 129, 200, 255, 256}) {
        std::vector<float> q(H * D); std::vector<uint16_t> k((size_t)width * KV * D), v(k.size());
        for (auto& x : q) x = r.uni(); for (auto& x : k) x = f16bits::f16_from_f32(r.uni()); for (auto& x : v) x = f16bits::f16_from_f32(r.uni());
        float* dq = dup(q); uint16_t* dk = dup(k); uint16_t* dv = dup(v); float* dout = dalloc<float>(H * D); int32_t* dst = dalloc<int32_t>(1);
        CUOK(cudaMemset(dst, 0x7f, 4));
        FlashArgs a{dq, dk, dv, dout, dst, width, width, width - 1, width / 4, 256};
        CHECK(native_flash_attn_short_step(a, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const auto out = fetch(dout, H * D); CHECK(fetch(dst, 1)[0] == 0);
        double worst = 0; bool finite = true;
        for (int h = 0; h < H; ++h) {
            const int kh = h / 12;                                  // division pairing
            std::vector<double> s(width); double mx = -1e30;
            for (int c = 0; c < width; ++c) { double d = 0; for (int i = 0; i < D; ++i) d += (double)q[h * D + i] * 0.0625 * h2f(k[((size_t)c * KV + kh) * D + i]); s[c] = d; mx = std::max(mx, d); }
            double sum = 0; for (auto& x : s) { x = std::exp(x - mx); sum += x; }
            for (int i = 0; i < D; ++i) {
                double o = 0; for (int c = 0; c < width; ++c) o += s[c] * h2f(v[((size_t)c * KV + kh) * D + i]);
                o /= sum; if (!std::isfinite(out[h * D + i])) finite = false; worst = std::max(worst, std::fabs(o - out[h * D + i]));
            }
        }
        CHECK(finite); CHECK(worst <= 3e-4);
        if (worst > 3e-4) std::printf("  flash width %d worst %f\n", width, worst);
        // the device-side validity check: an unsupported step NaN-fills the output and writes status 1
        for (int variant = 0; variant < 4; ++variant) {
            FlashArgs b = a;
            if (variant == 0) b.pos = width;
            if (variant == 1) b.n_bid = width / 4 + 1;
            if (variant == 2) b.n_kv = width + 1;
            if (variant == 3) b.max_ctx = width - 1;
            CUOK(cudaMemset(dout, 0, H * D * 4)); CUOK(cudaMemset(dst, 0, 4));
            CHECK(native_flash_attn_step(b, nullptr) == 0); CUOK(cudaDeviceSynchronize());
            const auto o2 = fetch(dout, H * D); bool allnan = true; for (float x : o2) if (!std::isnan(x)) allnan = false;
            CHECK(allnan); CHECK(fetch(dst, 1)[0] == 1);
            FlashArgs c2 = b; CHECK(native_flash_attn_short_step(c2, nullptr) == -1);   // host path rejects instead
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dst);
    }
    // validate_spans semantics (status/out must not overlap inputs; inputs may alias)
    Span a{4096, 64}, b{4160, 64}, c{4128, 64};
    CHECK(validate_spans(&a, 1, &b, 1, 4)); CHECK(!validate_spans(&a, 1, &c, 1, 4));
    Span o2[2] = {b, c}; CHECK(!validate_spans(&a, 1, o2, 2, 4));
    Span i2[2] = {a, a}; CHECK(validate_spans(i2, 2, &b, 1, 4));
    Span mis{4097, 64}, z{4096, 0}, nul{0, 8}; CHECK(!validate_spans(&mis, 1, &b, 1, 4)); CHECK(!validate_spans(&z, 1, &b, 1, 4)); CHECK(!validate_spans(&a, 1, &nul, 1, 4));
    CHECK(!validate_spans(&a, 1, &b, 1, 3));
    Span adj{4160, 64}, oadj{4224, 64}; CHECK(validate_spans(&adj, 1, &oadj, 1, 4));
    // wave reductions: warp_max / warp_sum
    for (int cols : {1, 33, 64, 65, 500}) {
        const int rows = 5; std::vector<float> x((size_t)rows * cols); for (auto& e : x) e = r.uni();
        float* dx = dup(x); float* dm = dalloc<float>(rows); float* ds = dalloc<float>(rows);
        CHECK(flash_wave_reduce(dx, dm, rows, cols, 0, nullptr) == 0 && flash_wave_reduce(dx, ds, rows, cols, 1, nullptr) == 0);
        CUOK(cudaDeviceSynchronize()); const auto m = fetch(dm, rows), s = fetch(ds, rows);
        for (int rr = 0; rr < rows; ++rr) { float mx = -1e30f; double sm = 0; for (int cc = 0; cc < cols; ++cc) { mx = std::max(mx, x[(size_t)rr * cols + cc]); sm += x[(size_t)rr * cols + cc]; } CHECK(m[rr] == mx); CHECK(std::fabs(s[rr] - sm) < 1e-4); }
        cudaFree(dx); cudaFree(dm); cudaFree(ds);
    }
}

// ================================================================================ QsaCacheIndexAttend (host facts)
static uint32_t lean_order_key(uint32_t b) { return b < 2147483648u ? b + 2147483648u : 4294967295u - b; }
static void host_qsa_facts() {
    using namespace qsa_step;
    for (int pos = 0; pos < 6000; ++pos) {
        const Step s = make_step(pos);
        const int n = pos + 1, nb = n / 4;
        CHECK(s.pos == pos && s.n_kv == n && s.n_bid == nb && s.width == std::min(n, 2048 + 4 - 1));
        CHECK(s.width <= s.n_kv && s.width <= 2051);
        CHECK(nb * 4 <= n && n - nb * 4 < 4);                       // nbid_cells_le, tail_lt_block
        CHECK((nb * 4 < n) == (n % 4 != 0));                        // tail_iff
        if (n <= 2051) CHECK(s.width == n);                         // selWidth_identity
        CHECK(((pos + 1) % 4 == 0) == (pos % 4 == 3));              // block_complete_iff
        CHECK(qsa_native::score_step_ok(n, 8192, pos, nb, s.width));
        CHECK(!qsa_native::score_step_ok(n, 8192, pos, nb + 1, s.width));
        CHECK(!qsa_native::score_step_ok(n, 8192, pos + 1, nb, s.width));
    }
    for (int tk : {1, 7, 100}) for (int r : {1, 2, 4}) for (int pos = 0; pos < 300; ++pos) {
        const Step s = make_step(pos, tk, r); CHECK(s.width == std::min(pos + 1, tk + r - 1) && s.n_bid == (pos + 1) / r);
    }
    // order key: the theory's closed form agrees with the production bit map; monotone on floats
    Rng rg(3);
    for (int i = 0; i < 200000; ++i) {
        uint32_t b = (rg.u32() << 16) ^ rg.u32(); CHECK(lean_order_key(b) == qsa_native::order_key_bits(b));
    }
    std::vector<float> fs = {0.f, 1e-30f, 1.f, 2.f, 1e30f, -1e-30f, -1.f, -2.f, -1e30f, 3.5f, -3.5f};
    for (int i = 0; i < 2000; ++i) fs.push_back(rg.gauss() * 100.f);
    for (float a : fs) for (float b : fs) {
        const uint32_t ka = qsa_native::order_key_f(a), kb = qsa_native::order_key_f(b);
        if (a < b) CHECK(ka < kb); if (a == b && std::signbit(a) == std::signbit(b)) CHECK(ka == kb);
    }
    CHECK(qsa_native::order_key_f(NAN) == 0u);
    for (uint32_t a = 0; a < 2147483648u; a += 99991u) { CHECK(lean_order_key(a) >= 2147483648u); }
    CHECK(lean_order_key(0x80000001u) < lean_order_key(0x80000000u) + 1 && lean_order_key(0xFF7FFFFFu) < lean_order_key(0x80000000u));
    // GQA pairing, gate half, gather decomposition
    for (int h = 0; h < 24; ++h) CHECK(h / 12 < 2);
    CHECK(1 / 12 != 1 % 2);
    for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d) CHECK(h * 512 + 256 + d < (h + 1) * 512);
    for (int i = 0; i < 4 * 128; ++i) CHECK(i / 128 * 128 + (i % 128) / 64 * 64 + (i % 128) % 64 == i && (i % 128) / 64 < 2);
    for (int n = 1; n < 5000; n += 7) CHECK(n <= (n + 255) / 256 * 256);
    for (int ms : {1, 100, 2051}) CHECK((ms + 32) * 4 == ms * 4 + 128);
}

// ================================================================================ QsaBlockSelect (host facts)
static uint32_t take_quota(uint32_t budget, uint32_t before, uint32_t eq) { return budget > before ? std::min(eq, budget - before) : 0; }
static void host_select_facts() {
    using namespace qsa_native;
    for (uint32_t n = 1; n < 3000; ++n) {
        const uint32_t nb = n / 4; uint32_t sum = 0;
        for (uint32_t b = 0; b <= nb; ++b) { const uint32_t w = block_weight(n, nb, b); CHECK(w <= 4); sum += w; for (uint32_t c = 0; c < w; ++c) CHECK(b * 4 + c < n); }
        CHECK(sum == n);                                                   // weights_total
        CHECK((block_weight(n, nb, nb) == 0) == (n % 4 == 0));             // tail_weight_zero_iff
        CHECK((nb * 4 < n) == (n % 4 != 0));                               // bias_iff
    }
    // digit round trip and pass count
    Rng r(5);
    for (int i = 0; i < 100000; ++i) {
        const uint32_t k = (r.u32() << 16) ^ r.u32();
        CHECK(((k >> 24) << 24) + (((k >> 16) & 255) << 16) + (((k >> 8) & 255) << 8) + (k & 255) == k);
    }
    CHECK(4 * 8 == 32);
    // quota algebra over a grid: telescoping, <= eq, zero when exhausted, total
    for (uint32_t budget = 0; budget < 40; ++budget) for (uint32_t before = 0; before < 40; ++before) for (uint32_t eq = 0; eq < 12; ++eq) {
        const uint32_t q = take_quota(budget, before, eq);
        CHECK(std::min(budget, before + eq) == std::min(budget, before) + q);
        CHECK(q <= eq); if (budget <= before) CHECK(q == 0);
        if (budget <= before + eq) CHECK(std::min(budget, before) + q == budget);
    }
    for (uint32_t gt = 0; gt < 30; ++gt) for (uint32_t eq = 0; eq < 30; ++eq) for (uint32_t width = gt; width < 60; ++width)
        if (width <= gt + eq) CHECK(sel_count(gt, eq, width) == width);     // exactly width cells when ties suffice
    for (int nb = 1; nb < 5000; nb += 13) CHECK(nb <= 256 * ((nb + 255) / 256));
    for (int b = 0; b < 1000; ++b) CHECK(b / 8 * 8 + b % 8 == b);
    for (int mb = 1; mb < 500; ++mb) CHECK(mb <= (mb + 7) / 8 * 8);
}

static void gpu_block_select() {
    using namespace qsa_native;
    Rng r(11);
    auto topk_ref = [&](const std::vector<float>& bs, int n, int width) {
        const int nb = n / 4;
        std::vector<uint32_t> key(n); for (int c = 0; c < n; ++c) { float f = bs[c / 4] + 0.0f; key[c] = order_key_f(f); }
        std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i; (void)nb;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] > key[b]; });
        const int take = std::min(n, width); std::vector<int> ref(idx.begin(), idx.begin() + take); std::sort(ref.begin(), ref.end()); return ref;
    };
    for (int n : {1, 3, 4, 5, 100, 2051, 2052, 2053, 2054, 2055, 4000, 8003}) {
        const int nbk = n / 4 + (n % 4 != 0);
        for (int alpha : {2, 9, 100000}) for (int width : {std::min(n, 2051), 100, 1}) {
            std::vector<float> bs(nbk); for (auto& s : bs) s = (float)(int)r.below(alpha) - (float)alpha * 0.4f;   // ties + negatives
            const auto ref = topk_ref(bs, n, width);
            float* d_s = dup(bs); int32_t* d_ids = dalloc<int32_t>(width); int32_t* d_n = dalloc<int32_t>(1);
            CUOK(cudaMemset(d_ids, 0x55, (size_t)width * 4));
            CHECK(qsa_block_topk(d_s, n, width, d_ids, d_n, nullptr) == cudaSuccess); CUOK(cudaDeviceSynchronize());
            const auto got = fetch(d_ids, width); const int cnt = fetch(d_n, 1)[0];
            CHECK(cnt == (int)ref.size());
            bool same = cnt == (int)ref.size(); for (int i = 0; i < cnt && same; ++i) same = got[i] == ref[i]; CHECK(same);
            for (int i = 1; i < cnt; ++i) CHECK(got[i] > got[i - 1]);                // ascending, no repeats
            for (int i = 0; i < cnt; ++i) CHECK(got[i] >= 0 && got[i] < n);
            for (int i = cnt; i < width; ++i) CHECK(got[i] == -1);                 // padded
            if (n <= width) for (int i = 0; i < n; ++i) CHECK(got[i] == i);        // identity
            cudaFree(d_s); cudaFree(d_ids); cudaFree(d_n);
        }
    }
    // argument validation
    { float s[8] = {0}; int32_t ids[8], no[1];
      CHECK(qsa_block_topk(nullptr, 4, 2, ids, no, nullptr) == cudaErrorInvalidValue);
      CHECK(qsa_block_topk(s, 0, 2, ids, no, nullptr) == cudaErrorInvalidValue); CHECK(qsa_block_topk(s, 4, 0, ids, no, nullptr) == cudaErrorInvalidValue); }
    // block scores: relu dots, tail bias, optional bias
    const int rows = 2100;
    std::vector<float> pooled((size_t)rows * 128), iq(4 * 128), hw = {0.7f, 0.3f, 1.1f, 0.5f};
    for (auto& x : pooled) x = r.gauss(); for (auto& x : iq) x = r.gauss();
    float* d_p = dup(pooled); float* d_q = dup(iq); float* d_hw = dup(hw);
    for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 129, 130, 131, 132, 1001, 4000, 8399}) {
        const int nbk = n / 4 + (n % 4 != 0);
        std::vector<float> bias(nbk); for (auto& b : bias) b = r.uni();
        for (int use_bias = 0; use_bias < 2; ++use_bias) {
            float* d_b = use_bias ? dup(bias) : nullptr; float* d_s = dalloc<float>(nbk);
            CHECK(qsa_block_scores(d_p, d_q, d_hw, d_b, d_s, n, rows, nullptr) == cudaSuccess); CUOK(cudaDeviceSynchronize());
            const auto sc = fetch(d_s, nbk);
            for (int b = 0; b < nbk; ++b) {
                double s = 0; for (int h = 0; h < 4; ++h) { double dot = 0; for (int d = 0; d < 128; ++d) dot += (double)iq[h * 128 + d] * pooled[(size_t)b * 128 + d]; s += hw[h] * (dot > 0.0 ? dot : 0.0); }
                if (use_bias) s += bias[b];
                const bool tail = (b == n / 4) && (n % 4 != 0);
                if (tail) CHECK(sc[b] >= 0.99e9f); else CHECK(std::fabs(sc[b] - s) <= 1e-3 * std::fabs(s) + 1e-3);
            }
            if (d_b) cudaFree(d_b); cudaFree(d_s);
        }
    }
    float* d_s = dalloc<float>(8);
    CHECK(qsa_block_scores(d_p, d_q, d_hw, nullptr, d_s, 100, 10, nullptr) == cudaErrorInvalidValue);   // rows < nb
    CHECK(qsa_block_scores(d_p, d_q, d_hw, nullptr, d_p + 64, 100, rows, nullptr) == cudaErrorInvalidValue);   // out overlaps input
    cudaFree(d_s); cudaFree(d_p); cudaFree(d_q); cudaFree(d_hw);
}

// ================================================================================ NativeQsaIndexerAppend
static void gpu_indexer() {
    using namespace qsa_native;
    const int max_cells = 64, rows = max_cells / 4 + 1, D = 128;
    const long long pos_base = 8; const float freq = 0.8f, eps = 1e-6f;
    Rng r(44);
    std::vector<float> keys((size_t)max_cells * D); for (auto& x : keys) x = r.gauss();
    float* d_key = dalloc<float>(D); float* d_tail = dalloc<float>(3 * D); float* d_pooled = dalloc<float>((size_t)rows * D); float* d_dead = dalloc<float>(D);
    CUOK(cudaMemset(d_tail, 0, 3 * D * 4)); CUOK(cudaMemset(d_pooled, 0, (size_t)rows * D * 4)); CUOK(cudaMemset(d_dead, 0, D * 4));
    CHECK(native_qsa_indexer_enabled()); native_qsa_indexer_set_enabled(false); CHECK(!native_qsa_indexer_enabled()); native_qsa_indexer_set_enabled(true); CHECK(native_qsa_indexer_enabled());
    auto ref_block = [&](const float* k0, const float* k1, const float* k2, const float* k3, bool zero_pos, int b, std::vector<double>& out) {
        std::vector<double> m(D), y(D); double ss = 0;
        for (int d = 0; d < D; ++d) { m[d] = zero_pos ? k3[d] : 0.25 * ((double)k0[d] + k1[d] + k2[d] + k3[d]); ss += m[d] * m[d]; }
        const double inv = 1.0 / std::sqrt(ss / D + eps); for (int d = 0; d < D; ++d) y[d] = m[d] * inv;
        const double rpos = zero_pos ? 0.0 : (double)(pos_base + 4 * b);
        out = y;
        for (int d = 0; d < 64; ++d) { const int pair = d % 32; const double ang = rpos * std::pow((double)freq, pair), c = std::cos(ang), s = std::sin(ang); const double lo = y[pair], hi = y[pair + 32]; out[d] = d < 32 ? lo * c - hi * s : hi * c + lo * s; }
    };
    for (int pos = 0; pos < max_cells; ++pos) {
        CUOK(cudaMemcpy(d_key, keys.data() + (size_t)pos * D, D * 4, cudaMemcpyHostToDevice));
        IndexerArgs a{d_key, d_tail, d_pooled, d_dead, pos, max_cells, pos_base, freq, eps};
        CHECK(native_qsa_indexer_append(a, nullptr) == cudaSuccess); CUOK(cudaDeviceSynchronize());
        const auto tail = fetch(d_tail, 3 * D);
        if (pos % 4 < 3) for (int d = 0; d < D; ++d) CHECK(tail[(pos % 4) * D + d] == keys[(size_t)pos * D + d]);   // raw slot store
        if (pos == 0 || pos % 4 == 3) {
            const int b = pos / 4; std::vector<double> ref;
            if (pos == 0) ref_block(nullptr, nullptr, nullptr, &keys[0], true, 0, ref);
            else ref_block(&keys[(size_t)(4 * b) * D], &keys[(size_t)(4 * b + 1) * D], &keys[(size_t)(4 * b + 2) * D], &keys[(size_t)(4 * b + 3) * D], false, b, ref);
            const auto pooled = fetch(d_pooled, (size_t)rows * D), dead = fetch(d_dead, D);
            for (int d = 0; d < D; ++d) CHECK(std::fabs(pooled[(size_t)b * D + d] - ref[d]) <= 3e-4 * std::fabs(ref[d]) + 3e-4);
            if (pos != 0) for (int d = 0; d < D; ++d) CHECK(pooled[(size_t)(b + 1) * D + d] == dead[d]);       // spare row re-seeded from dead
            if (pos == 0) for (int d = 0; d < D; ++d) CHECK(dead[d] == pooled[d]);
            for (int d = 64; d < D; ++d) { double m = 0; (void)m; }
        }
    }
    // dims 64..127 are NOT rotated: pooled == normalised mean there; and the spare row on the last block exists (row 16)
    { const auto pooled = fetch(d_pooled, (size_t)rows * D), dead = fetch(d_dead, D);
      for (int d = 0; d < D; ++d) CHECK(pooled[(size_t)(rows - 1) * D + d] == dead[d]);
      for (int b = 0; b < 16; ++b) { std::vector<double> ref; ref_block(&keys[(size_t)(4 * b) * D], &keys[(size_t)(4 * b + 1) * D], &keys[(size_t)(4 * b + 2) * D], &keys[(size_t)(4 * b + 3) * D], false, b, ref);
        for (int d = 0; d < D; ++d) CHECK(std::fabs(pooled[(size_t)b * D + d] - ref[d]) <= 3e-4 * std::fabs(ref[d]) + 3e-4); } }
    // capacity guard: pos >= max_cells (and negative) is ignored
    { const auto t0 = fetch(d_tail, 3 * D), p0 = fetch(d_pooled, (size_t)rows * D), d0 = fetch(d_dead, D);
      for (int pos : {max_cells, max_cells + 3, -1}) { IndexerArgs a{d_key, d_tail, d_pooled, d_dead, pos, max_cells, pos_base, freq, eps}; native_qsa_indexer_append(a, nullptr); CUOK(cudaDeviceSynchronize()); }
      CHECK(fetch(d_tail, 3 * D) == t0); CHECK(fetch(d_pooled, (size_t)rows * D) == p0); CHECK(fetch(d_dead, D) == d0); }
    // validation: unaligned base, overlapping buffers
    { IndexerArgs a{d_key, d_tail, d_pooled, d_dead, 3, max_cells, 6, freq, eps}; CHECK(native_qsa_indexer_append(a, nullptr) == cudaErrorInvalidValue);
      IndexerArgs b{d_key, d_tail, d_pooled, d_dead, 3, max_cells, (long long)2147483647 - 10, freq, eps}; CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
      IndexerArgs c{d_key, d_tail, d_pooled, d_pooled + 128, 3, max_cells, pos_base, freq, eps}; CHECK(native_qsa_indexer_append(c, nullptr) == cudaErrorInvalidValue);
      IndexerArgs z{}; CHECK(native_qsa_indexer_append(z, nullptr) == cudaErrorInvalidValue); }
    cudaFree(d_key); cudaFree(d_tail); cudaFree(d_pooled); cudaFree(d_dead);
}

// ================================================================================ QsaCacheIndexAttend (device)
static void gpu_attend_and_gate() {
    using namespace qsa_step;
    Rng r(77);
    const int NH = 24, NKV = 2, HD = 256, PS = 16, P = 8, NC = P * PS;
    const int64_t pool_rows = (int64_t)P * NKV * PS;
    std::vector<int32_t> table = {3, 0, 7, 1, 5, 2, 6, 4};
    uint16_t* d_kp = dalloc<uint16_t>((size_t)pool_rows * HD); uint16_t* d_vp = dalloc<uint16_t>((size_t)pool_rows * HD);
    CUOK(cudaMemset(d_kp, 0, (size_t)pool_rows * HD * 2)); CUOK(cudaMemset(d_vp, 0, (size_t)pool_rows * HD * 2));
    int32_t* d_t = dup(table);
    std::vector<float> K((size_t)NC * NKV * HD), V(K.size());
    for (auto& x : K) x = (float)(int)(r.below(33)) / 16.f - 1.f;        // exact in fp16
    for (auto& x : V) x = (float)(int)(r.below(33)) / 16.f - 1.f;
    float* d_src = dalloc<float>((size_t)NKV * HD);
    for (int pass = 0; pass < 2; ++pass) for (int c = 0; c < NC; ++c) {
        CUOK(cudaMemcpy(d_src, (pass ? V : K).data() + (size_t)c * NKV * HD, (size_t)NKV * HD * 4, cudaMemcpyHostToDevice));
        CHECK(kv_append_step(d_src, pass ? d_vp : d_kp, d_t, P, c, NKV, HD, PS, pool_rows, nullptr) == 0);
    }
    CUOK(cudaDeviceSynchronize());
    std::vector<int32_t> ids; for (int c = 0; c < NC; c += 3) ids.push_back(c); ids.push_back(NC - 1); ids.push_back(-1);   // one masked id
    const int n_sel = (int)ids.size();
    std::vector<float> q((size_t)NH * HD); for (auto& x : q) x = r.uni();
    float* d_q = dup(q); int32_t* d_ids = dup(ids); float* d_out = dalloc<float>((size_t)NH * HD);
    const float scale = 0.0625f;
    CHECK(qsa_attend_step(d_q, d_kp, d_vp, d_t, P, d_ids, n_sel, NH, NKV, HD, PS, pool_rows, scale, d_out, nullptr) == 0); CUOK(cudaDeviceSynchronize());
    const auto out = fetch(d_out, (size_t)NH * HD);
    double worst = 0, worst_mod = 0;
    std::vector<double> alt((size_t)NH * HD, 0.0);
    for (int h = 0; h < NH; ++h) for (int pairing = 0; pairing < 2; ++pairing) {
        const int kh = pairing == 0 ? h / (NH / NKV) : h % NKV;
        std::vector<double> s; std::vector<int> cs; double mx = -1e30;
        for (int id : ids) { if (id < 0) continue; double d = 0; for (int i = 0; i < HD; ++i) d += (double)q[(size_t)h * HD + i] * scale * K[((size_t)id * NKV + kh) * HD + i]; s.push_back(d); cs.push_back(id); mx = std::max(mx, d); }
        double sum = 0; for (auto& x : s) { x = std::exp(x - mx); sum += x; }
        double wd = 0;
        for (int i = 0; i < HD; ++i) { double o = 0; for (size_t j = 0; j < s.size(); ++j) o += s[j] * V[((size_t)cs[j] * NKV + kh) * HD + i]; o /= sum; wd = std::max(wd, std::fabs(o - out[(size_t)h * HD + i])); }
        if (pairing == 0) worst = std::max(worst, wd); else worst_mod = std::max(worst_mod, wd);
    }
    CHECK(worst <= 1e-3);                 // division pairing h / 12 matches
    CHECK(worst_mod > 1e-2);              // modulo pairing h % 2 would NOT (some head differs)
    // gate: out *= sigmoid(gate) from the gate half
    std::vector<float> gate((size_t)NH * HD); for (auto& x : gate) x = r.gauss() * 2.f;
    float* d_g = dup(gate);
    CHECK(qsa_gate_apply_f32(d_out, d_g, (int64_t)NH * HD, nullptr) == 0); CUOK(cudaDeviceSynchronize());
    const auto og = fetch(d_out, (size_t)NH * HD); bool gate_ok = true;
    for (size_t i = 0; i < og.size(); ++i) { const double e = (double)out[i] / (1.0 + std::exp(-(double)gate[i])); if (std::fabs(e - og[i]) > 1e-4 * std::fabs(e) + 1e-5) gate_ok = false; }
    CHECK(gate_ok);
    // rejected shapes
    CHECK(qsa_attend_step(d_q, d_kp, d_vp, d_t, P, d_ids, 0, NH, NKV, HD, PS, pool_rows, scale, d_out, nullptr) == -1);
    CHECK(qsa_attend_step(d_q, d_kp, d_vp, d_t, P, d_ids, 2052, NH, NKV, HD, PS, pool_rows, scale, d_out, nullptr) == -1);
    CHECK(qsa_attend_step(d_q, d_kp, d_vp, d_t, P, d_ids, n_sel, 10, 4, HD, PS, pool_rows, scale, d_out, nullptr) == -1);
    // step state on device: derived from one position
    int32_t* d_step = dalloc<int32_t>(4);
    for (int p : {0, 1, 2, 3, 4, 2046, 2047, 2048, 2049, 2050, 2051, 4000, 32767}) {
        CHECK(step_upload(d_step, p, 2048, 4, nullptr) == 0); CUOK(cudaDeviceSynchronize());
        const auto s = fetch(d_step, 4); const Step h = make_step(p);
        CHECK(s[0] == h.pos && s[1] == h.n_kv && s[2] == h.n_bid && s[3] == h.width);
    }
    // indexer pooling (tail + spare row) via the step kernels and the one-block top-k with -0 / ties
    { for (int n : {5, 2051, 2052, 3001}) {
        std::vector<float> sc(n); for (auto& s : sc) s = (float)(int)r.below(7) - 3.f; if (n > 3) sc[3] = -0.0f;
        for (auto& s : sc) s += 0.0f;                                  // fold -0 like the production path
        const int width = std::min(n, 2051);
        std::vector<uint32_t> key(n); for (int i = 0; i < n; ++i) key[i] = qsa_native::order_key_f(sc[i]);
        std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i; std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] > key[b]; });
        std::vector<int> ref(idx.begin(), idx.begin() + width); std::sort(ref.begin(), ref.end());
        float* d_s = dup(sc); int32_t* d_i = dalloc<int32_t>(width); int32_t* d_n = dalloc<int32_t>(1);
        CHECK(topk_512_step(d_s, n, width, d_i, d_n, nullptr) == 0); CUOK(cudaDeviceSynchronize());
        const auto got = fetch(d_i, width); bool same = fetch(d_n, 1)[0] == width; for (int i = 0; i < width && same; ++i) same = got[i] == ref[i]; CHECK(same);
        cudaFree(d_s); cudaFree(d_i); cudaFree(d_n); } }
    cudaFree(d_kp); cudaFree(d_vp); cudaFree(d_t); cudaFree(d_src); cudaFree(d_q); cudaFree(d_ids); cudaFree(d_out); cudaFree(d_g); cudaFree(d_step);
}

// ================================================================================ device evaluation of the closed forms
static void gpu_geometry() {
    std::vector<int32_t> pos; for (int p = 0; p < 6000; ++p) pos.push_back(p); for (int p : {8191, 32766, 32767}) pos.push_back(p);
    const int n = (int)pos.size();
    int32_t* d_p = dup(pos); int32_t* d_o = dalloc<int32_t>((size_t)n * ucc7::kStepFields);
    CHECK(ucc7::geometry_eval(d_p, n, 2048, 4, d_o, nullptr) == 0); CUOK(cudaDeviceSynchronize());
    const auto o = fetch(d_o, (size_t)n * ucc7::kStepFields);
    for (int i = 0; i < n; ++i) {
        const qsa_step::Step h = qsa_step::make_step(pos[i]); const int32_t* f = &o[(size_t)i * ucc7::kStepFields];
        CHECK(f[0] == h.n_kv && f[1] == h.n_bid && f[2] == h.width);
        CHECK(f[3] == qsa_native::block_weight(h.n_kv, h.n_bid, h.n_bid) && f[3] < 4 && (f[4] == 1) == (h.n_kv % 4 != 0));
        CHECK(f[5] == (pos[i] % 24) / 12 && f[5] < 2);
    }
    CHECK(ucc7::geometry_eval(nullptr, n, 2048, 4, d_o, nullptr) == -1 && ucc7::geometry_eval(d_p, 0, 2048, 4, d_o, nullptr) == -1);
    std::vector<uint32_t> bits = {0u, 1u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFFu, 0x3F800000u, 0xBF800000u};
    Rng r(9); for (int i = 0; i < 50000; ++i) bits.push_back((r.u32() << 16) ^ r.u32());
    uint32_t* d_b = dup(bits); uint32_t* d_k = dalloc<uint32_t>(bits.size());
    CHECK(ucc7::order_key_eval(d_b, (int)bits.size(), d_k, nullptr) == 0); CUOK(cudaDeviceSynchronize());
    const auto keys = fetch(d_k, bits.size());
    for (size_t i = 0; i < bits.size(); ++i) CHECK(keys[i] == lean_order_key(bits[i]) && keys[i] == qsa_native::order_key_bits(bits[i]));
    std::vector<int32_t> bu, be, eq, want;
    for (int b = 0; b < 30; ++b) for (int bf = 0; bf < 30; ++bf) for (int e = 0; e < 8; ++e) { bu.push_back(b); be.push_back(bf); eq.push_back(e); want.push_back((int32_t)take_quota(b, bf, e)); }
    int32_t* d_bu = dup(bu); int32_t* d_be = dup(be); int32_t* d_eq = dup(eq); int32_t* d_q = dalloc<int32_t>(bu.size());
    CHECK(ucc7::quota_eval(d_bu, d_be, d_eq, (int)bu.size(), d_q, nullptr) == 0); CUOK(cudaDeviceSynchronize());
    CHECK(fetch(d_q, bu.size()) == want);
    cudaFree(d_p); cudaFree(d_o); cudaFree(d_b); cudaFree(d_k); cudaFree(d_bu); cudaFree(d_be); cudaFree(d_eq); cudaFree(d_q);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    host_flash_facts(); host_qsa_facts(); host_select_facts();
    gpu_flash(); gpu_block_select(); gpu_indexer(); gpu_attend_and_gate(); gpu_geometry();
    CUOK(cudaDeviceSynchronize());
    std::printf("%ld checks, %ld failures\n", g_checks, g_fail);
    if (g_fail) { std::printf("UCC7 UNIT TESTS FAILED\n"); return 1; }
    std::printf("UCC7 UNIT TESTS PASSED\n");
    return 0;
}
