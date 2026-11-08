// test_u293.cu - unit u293ad55355: host invariants (invariants.hpp) + device checks of the native PLE post-ops, the QSA indexer
// append, the tile block scorer and the radix top-k against double-precision CPU references and the prior units' kernels.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "invariants.hpp"
#include "baloon/ilu_qsa_kernels.hpp"

using namespace baloon;

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { ++g_fail; std::printf("FAIL %s:%d  %s -> %d\n", __FILE__, __LINE__, #x, (int)e_); } } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t u32() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 16); }
    float uni() { return (u32() & 0xFFFFFF) / 16777216.0f; }                 // [0,1)
    float range(float a, float b) { return a + (b - a) * uni(); }
    float gauss() { float x = 0; for (int i = 0; i < 12; ++i) x += uni(); return x - 6.f; }
};
static bool near(double a, double b, double atol, double rtol) { return std::fabs(a - b) <= atol + rtol * std::fabs(b); }
// Score tolerance: ordinary rows within 1e-3 + 2e-6*|ref|; tail rows (~1e9, float spacing 64) within 128.
static bool score_ok(double a, double ref) { return std::fabs(a - ref) <= 1e-3 + 2e-6 * std::fabs(ref) + (std::fabs(ref) > 5e8 ? 128.0 : 0.0); }

template <class T> static T* dalloc(size_t n) { T* p = nullptr; CU(cudaMalloc((void**)&p, n * sizeof(T))); CU(cudaMemset(p, 0, n * sizeof(T))); return p; }
template <class T> static T* dup(const std::vector<T>& v) { T* p = dalloc<T>(v.size()); CU(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; }
template <class T> static std::vector<T> fetch(const T* p, size_t n) { std::vector<T> v(n); CU(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
static void sync() { CU(cudaDeviceSynchronize()); }

static uint16_t f2h(float f) {  // round to nearest even
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFFu;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t)(sign | r);
}

// ============================================================================================ PLE post-ops
static void test_ple_postops() {
    using namespace ple_native;
    Rng rng(11);
    std::vector<float> normed(kD), gw(kD), value(kN), hist((size_t)kD * kHist), hidden(kD);
    std::vector<uint16_t> cw((size_t)kD * kTaps);
    for (int i = 0; i < kD; ++i) {
        const int s = i / kN;
        normed[i] = rng.range(0.5f, 1.5f);
        // stream 0: s ~ +51, stream 1: ~ -51, stream 2: ~ +2.5, stream 3: ~ -0.8 (away from the sqrt kink at 0)
        const float lo[4] = {0.01f, -0.03f, 0.0005f, -0.0005f}, hi[4] = {0.03f, -0.01f, 0.0015f, -0.0001f};
        gw[i] = rng.range(lo[s], hi[s]);
        hidden[i] = rng.gauss();
    }
    for (auto& v : value) v = rng.gauss();
    for (auto& v : hist) v = rng.gauss();
    for (auto& v : cw) v = f2h(rng.gauss() * 0.5f);

    float *d_norm = dup(normed), *d_gw = dup(gw), *d_val = dup(value), *d_hist = dup(hist), *d_hid = dup(hidden);
    uint16_t* d_cw = dup(cw);
    float *d_gate = dalloc<float>(kH), *d_gated = dalloc<float>(kD), *d_resid = dalloc<float>(kD);
    Args a{d_norm, d_gw, d_val, d_cw, d_hist, d_hid, d_gate, d_gated, d_resid};
    CHECK(native_ple_postops(a, nullptr) == cudaSuccess);
    sync();
    const auto gate = fetch(d_gate, kH), gated = fetch(d_gated, kD), resid = fetch(d_resid, kD);

    for (int s = 0; s < kH; ++s) {
        double sc = 0; for (int d = 0; d < kN; ++d) sc += (double)normed[s * kN + d] * gw[s * kN + d];
        CHECK(gate[s] > 0.f && gate[s] < 1.f);
        CHECK(near(gate[s], gate_of((float)sc), 1e-4, 1e-4));
    }
    CHECK(gate[0] > 0.5f && gate[1] < 0.5f);   // sign of the score survives the signed sqrt
    int bad_g = 0, bad_r = 0;
    for (int i = 0; i < kD; ++i) {
        if (gated[i] != value[i % kN] * gate[i / kN]) ++bad_g;                       // broadcast law
        double acc = 0;
        for (int k = 0; k < kTaps; ++k) {
            const double tap = k < 3 ? hist[hist_index(hist_row(k), i)] : normed[i];   // rows 0, 3, 6 and the fresh value
            acc += (double)half_to_float(cw[weight_index(i, k)]) * tap;
        }
        const double act = acc / (1.0 + std::exp(-acc));
        const double ref = (double)hidden[i] + ((double)gated[i] + act);
        if (!near(resid[i], ref, 2e-4, 2e-4)) ++bad_r;
    }
    CHECK(bad_g == 0);
    CHECK(bad_r == 0);

    // the history is not modified
    CHECK(fetch(d_hist, hist.size()) == hist);

    // exception 1: resid == hidden (in place) gives bitwise the same result
    {
        float* h2 = dup(hidden);
        Args b = a; b.hidden = h2; b.resid = h2; b.gated = dalloc<float>(kD); b.gate = dalloc<float>(kH);
        CHECK(native_ple_postops(b, nullptr) == cudaSuccess); sync();
        CHECK(fetch(h2, kD) == resid);
        cudaFree(h2); cudaFree(b.gated); cudaFree(b.gate);
    }
    // exception 2: resid == gated
    {
        float* g2 = dalloc<float>(kD);
        Args b = a; b.gated = g2; b.resid = g2; b.gate = dalloc<float>(kH);
        CHECK(native_ple_postops(b, nullptr) == cudaSuccess); sync();
        CHECK(fetch(g2, kD) == resid);
        cudaFree(g2); cudaFree(b.gate);
    }
    // both at once would clobber hidden: rejected, nothing written
    {
        float* h2 = dup(hidden);
        Args b = a; b.hidden = h2; b.resid = h2; b.gated = h2;
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        CHECK(fetch(h2, kD) == hidden);
        cudaFree(h2);
    }
    // everything else must be disjoint: partial overlaps and other aliases are rejected
    {
        float* big = dalloc<float>(3 * (size_t)kD);
        Args b = a; b.resid = big; b.hidden = big + 1;                 // partial overlap output/input
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.resid = d_norm;                                      // output aliasing another input
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.gated = d_val;                                       // output aliasing an input of different size
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.gate = d_gated;                                      // two outputs aliased (not the allowed pair)
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.gated = big; b.resid = big + 5;                      // outputs partially overlapping
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.normed = nullptr;
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.conv_w = (const uint16_t*)((const char*)d_cw + 1);   // misaligned fp16 weights
        CHECK(native_ple_postops(b, nullptr) == cudaErrorInvalidValue);
        cudaFree(big);
    }
    for (void* p : {(void*)d_norm, (void*)d_gw, (void*)d_val, (void*)d_hist, (void*)d_hid, (void*)d_cw, (void*)d_gate, (void*)d_gated, (void*)d_resid}) cudaFree(p);
}

// ============================================================================================ indexer append
namespace {
struct IdxRef {
    int max_cells;
    std::vector<float> tail, pooled, dead;
    explicit IdxRef(int mc) : max_cells(mc), tail(3 * 128, 0.f), pooled((size_t)(mc / 4 + 1) * 128, 0.f), dead(128, 0.f) {}
    void append(const std::vector<float>& key, int pos, long long base, float freq, float eps) {
        const int slot = pos & 3;
        if (slot < 3) std::copy(key.begin(), key.end(), tail.begin() + slot * 128);
        if (slot != 3 && pos != 0) return;
        std::vector<double> mean(128), y(128), out(128);
        for (int d = 0; d < 128; ++d)
            mean[d] = pos == 0 ? key[d] : ((double)tail[d] + tail[128 + d] + tail[256 + d] + key[d]) * 0.25;
        double ss = 0; for (double m : mean) ss += m * m;
        const double inv = 1.0 / std::sqrt(ss / 128.0 + eps);
        for (int d = 0; d < 128; ++d) { y[d] = mean[d] * inv; out[d] = y[d]; }
        const double rpos = pos == 0 ? 0.0 : (double)(base + 4ll * (pos >> 2));
        for (int p = 0; p < 32; ++p) {
            const double ang = rpos * std::pow((double)freq, (double)p), cs = std::cos(ang), sn = std::sin(ang);
            out[p] = y[p] * cs - y[p + 32] * sn;
            out[p + 32] = y[p + 32] * cs + y[p] * sn;
        }
        if (pos == 0) { for (int d = 0; d < 128; ++d) { pooled[d] = (float)out[d]; dead[d] = (float)out[d]; } }
        else {
            const int b = pos >> 2;
            for (int d = 0; d < 128; ++d) { pooled[(size_t)b * 128 + d] = (float)out[d]; pooled[(size_t)(b + 1) * 128 + d] = dead[d]; }
        }
    }
};
}  // namespace

static void test_indexer() {
    using namespace qsa_native;
    const int max_cells = 64;
    const long long base = 8;
    const float freq = 0.9f, eps = 1e-6f;
    const size_t rows = max_cells / 4 + 1;
    Rng rng(21);
    IdxRef ref(max_cells);
    float *d_tail = dalloc<float>(3 * 128), *d_pooled = dalloc<float>(rows * 128), *d_dead = dalloc<float>(128), *d_key = dalloc<float>(128);
    float *p_tail = dalloc<float>(3 * 128), *p_pooled = dalloc<float>(rows * 128), *p_dead = dalloc<float>(128);   // prior unit's kernel
    for (int pos = 0; pos < max_cells; ++pos) {
        std::vector<float> key(128);
        for (auto& v : key) v = rng.gauss();
        CU(cudaMemcpy(d_key, key.data(), 512, cudaMemcpyHostToDevice));
        IndexerArgs a{d_key, d_tail, d_pooled, d_dead, pos, max_cells, base, freq, eps};
        CHECK(native_qsa_indexer_append(a, nullptr) == cudaSuccess);
        ilu_qsa::AppendArgs pa{d_key, p_tail, p_pooled, p_dead, pos, max_cells, (int)rows, base, freq, eps};
        CHECK(ilu_qsa::pool_append(pa, nullptr) == ilu_qsa::kOk);
        sync();
        ref.append(key, pos, base, freq, eps);
        const auto tail = fetch(d_tail, 3 * 128), pooled = fetch(d_pooled, rows * 128), dead = fetch(d_dead, 128);
        const auto ptail = fetch(p_tail, 3 * 128), ppooled = fetch(p_pooled, rows * 128), pdead = fetch(p_dead, 128);
        int bad = 0, badp = 0;
        for (size_t i = 0; i < tail.size(); ++i) { if (tail[i] != ref.tail[i]) ++bad; if (!near(tail[i], ptail[i], 1e-6, 1e-6)) ++badp; }
        for (size_t i = 0; i < pooled.size(); ++i) { if (!near(pooled[i], ref.pooled[i], 2e-4, 2e-4)) ++bad; if (!near(pooled[i], ppooled[i], 1e-5, 1e-5)) ++badp; }
        for (size_t i = 0; i < dead.size(); ++i) { if (!near(dead[i], ref.dead[i], 2e-4, 2e-4)) ++bad; if (!near(dead[i], pdead[i], 1e-5, 1e-5)) ++badp; }
        if (bad || badp) std::printf("  indexer pos %d: %d vs reference, %d vs prior kernel\n", pos, bad, badp);
        CHECK(bad == 0);
        CHECK(badp == 0);
    }
    // rejections (nothing is written)
    {
        const auto before = fetch(d_pooled, rows * 128);
        IndexerArgs a{d_key, d_tail, d_pooled, d_dead, 3, max_cells, base, freq, eps}, b;
        b = a; b.pos_base = 6;                                   CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.pos = max_cells;                                CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.pos = -1;                                       CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.pos_base = 2147483647ll - 1;  b.pos_base -= b.pos_base % 4; b.max_cells = 64;
        CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);                      // int32 overflow of rope positions
        b = a; b.eps = -1.f;                                     CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.freq_step = 0.f;                                CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.key = nullptr;                                  CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        b = a; b.dead = d_tail + 128;                            CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);   // dead inside tail
        b = a; b.key = d_pooled + 128;                           CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);   // key inside pooled
        b = a; b.pooled = d_tail;                                CHECK(native_qsa_indexer_append(b, nullptr) == cudaErrorInvalidValue);
        sync();
        CHECK(fetch(d_pooled, rows * 128) == before);
    }
    CHECK(native_qsa_indexer_enabled());
    native_qsa_indexer_set_enabled(false); CHECK(!native_qsa_indexer_enabled());
    native_qsa_indexer_set_enabled(true);
    for (float* p : {d_tail, d_pooled, d_dead, d_key, p_tail, p_pooled, p_dead}) cudaFree(p);
}

// ============================================================================================ score tiles
static double ref_block_score(const std::vector<float>& pooled, const std::vector<float>& iq, const std::vector<float>& hw,
                              const std::vector<float>* bias, int row, int n) {
    double s = 0;
    for (int h = 0; h < 4; ++h) {
        double dot = 0;
        for (int d = 0; d < 128; ++d) dot += (double)pooled[(size_t)row * 128 + d] * iq[h * 128 + d];
        s += (double)hw[h] * (dot > 0.0 ? dot : 0.0);
    }
    if (bias) s += (*bias)[row];
    if (row == n / 4 && n % 4 > 0) s += 1.0e9;
    return s;
}

static void test_score() {
    using namespace qsa_native;
    const int max_cells = 4100, max_blocks = max_cells / 4 + 1;
    Rng rng(31);
    std::vector<float> pooled((size_t)max_blocks * 128), iq(4 * 128), hw(4), bias(max_blocks);
    for (auto& v : pooled) v = rng.gauss() * 0.3f;
    for (auto& v : iq) v = rng.gauss() * 0.3f;
    for (auto& v : hw) v = rng.range(0.2f, 1.f);
    for (auto& v : bias) v = rng.range(-1.f, 1.f);
    float *d_p = dup(pooled), *d_iq = dup(iq), *d_hw = dup(hw), *d_bias = dup(bias);
    float *d_cell = dalloc<float>(max_cells), *d_blk = dalloc<float>(max_blocks), *d_cell2 = dalloc<float>(max_cells), *d_blk2 = dalloc<float>(max_blocks);
    const float sentinel = -777.f;
    auto fill = [&](float* p, size_t n) { std::vector<float> v(n, sentinel); CU(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice)); };
    for (int n : {1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 100, 127, 128, 129, 1000, 2051, 2052, 2053, 4096, 4099, 4100}) {
        for (int use_bias = 0; use_bias < 2; ++use_bias) {
            const int nb = n / 4 + (n % 4 != 0), pos = n - 1, nbid = n / 4, width = std::min(n, 2051);
            fill(d_cell, max_cells); fill(d_blk, max_blocks);
            ScoreArgs a{d_p, d_iq, d_hw, use_bias ? d_bias : nullptr, d_blk, d_cell, pos, n, nbid, width, max_cells, max_blocks};
            CHECK(qsa_native::native_qsa_score(a, nullptr) == cudaSuccess);
            ilu_qsa::mma_block_score(d_p, d_iq, d_hw, use_bias ? d_bias : nullptr, d_blk2, d_cell2, pos, n, nbid, width, max_cells, max_blocks, nullptr);
            sync();
            const auto cell = fetch(d_cell, max_cells), blk = fetch(d_blk, max_blocks), cell2 = fetch(d_cell2, n), blk2 = fetch(d_blk2, nb);
            int bad = 0, bad_prior = 0;
            for (int r = 0; r < nb; ++r) {
                const double ref = ref_block_score(pooled, iq, hw, use_bias ? &bias : nullptr, r, n);
                if (!score_ok(blk[r], ref)) ++bad;
                if (!score_ok(blk[r], blk2[r])) ++bad_prior;
                for (int i = 0; i < 4 && r * 4 + i < n; ++i) { if (cell[r * 4 + i] != blk[r]) ++bad; if (!score_ok(cell[r * 4 + i], cell2[r * 4 + i])) ++bad_prior; }
            }
            for (int c = n; c < max_cells; ++c) if (cell[c] != sentinel) ++bad;       // cells >= n untouched
            for (int r = nb; r < max_blocks; ++r) if (blk[r] != sentinel) ++bad;      // rows >= nb untouched
            if (n % 4 != 0) CHECK(blk[nbid] > 9.0e8f);                                  // tail bias fires exactly when n % 4 != 0
            if (n % 4 == 0 && nb > 0) CHECK(blk[nb - 1] < 9.0e8f);
            if (bad || bad_prior) std::printf("  score n=%d bias=%d: %d vs ref, %d vs prior\n", n, use_bias, bad, bad_prior);
            CHECK(bad == 0);
            CHECK(bad_prior == 0);
        }
    }
    // step / geometry rejections write nothing
    {
        const int n = 100, pos = 99, nbid = 25, width = 100;
        fill(d_cell, max_cells);
        auto rejected = [&](ScoreArgs a) {
            const bool r = qsa_native::native_qsa_score(a, nullptr) == cudaErrorInvalidValue;
            sync();
            const auto cell = fetch(d_cell, max_cells);
            bool untouched = true; for (float v : cell) untouched = untouched && v == sentinel;
            return r && untouched;
        };
        ScoreArgs g{d_p, d_iq, d_hw, nullptr, nullptr, d_cell, pos, n, nbid, width, max_cells, max_blocks};
        ScoreArgs b;
        CHECK(qsa_native::native_qsa_score(g, nullptr) == cudaSuccess); sync(); fill(d_cell, max_cells);
        b = g; b.pos = pos + 1;                  CHECK(rejected(b));
        b = g; b.pos = pos - 1;                  CHECK(rejected(b));
        b = g; b.n_bid = nbid + 1;               CHECK(rejected(b));
        b = g; b.width = width - 1;              CHECK(rejected(b));
        b = g; b.width = width + 1;              CHECK(rejected(b));
        b = g; b.max_blocks = max_blocks - 1;    CHECK(rejected(b));
        b = g; b.max_blocks = max_blocks + 1;    CHECK(rejected(b));
        b = g; b.max_cells = n - 1;              CHECK(rejected(b));
        b = g; b.n = 0; b.pos = -1; b.n_bid = 0; CHECK(rejected(b));
        b = g; b.cell_scores = d_p;              // output over the pooled input
        CHECK(qsa_native::native_qsa_score(b, nullptr) == cudaErrorInvalidValue);
        b = g; b.block_scores = d_cell;          // two outputs aliased
        CHECK(qsa_native::native_qsa_score(b, nullptr) == cudaErrorInvalidValue);
        b = g; b.hw = nullptr;
        CHECK(qsa_native::native_qsa_score(b, nullptr) == cudaErrorInvalidValue);
        // width 2051 cap: n = 3000 requires width 2051
        ScoreArgs w{d_p, d_iq, d_hw, nullptr, nullptr, d_cell, 2999, 3000, 750, 2051, max_cells, max_blocks};
        CHECK(qsa_native::native_qsa_score(w, nullptr) == cudaSuccess);
        w.width = 3000; CHECK(qsa_native::native_qsa_score(w, nullptr) == cudaErrorInvalidValue);
        sync();
    }
    CHECK(native_qsa_score_enabled());
    native_qsa_score_set_enabled(false); CHECK(!native_qsa_score_enabled());
    native_qsa_score_set_enabled(true);
    for (float* p : {d_p, d_iq, d_hw, d_bias, d_cell, d_blk, d_cell2, d_blk2}) cudaFree(p);
}

// ============================================================================================ block scores + top-k
static std::vector<int> ref_topk(const std::vector<float>& scores, int n_kv, int width) {
    std::vector<int> ids;
    if (n_kv <= width) { for (int i = 0; i < n_kv; ++i) ids.push_back(i); return ids; }
    std::vector<int> order(n_kv);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const uint32_t ka = qsa_native::order_key_f(scores[a / 4]), kb = qsa_native::order_key_f(scores[b / 4]);
        return ka != kb ? ka > kb : a < b;
    });
    ids.assign(order.begin(), order.begin() + width);
    std::sort(ids.begin(), ids.end());
    return ids;
}

static void test_select() {
    using namespace qsa_native;
    Rng rng(41);
    // block scores vs CPU and vs the prior kernel
    {
        const int rows = 600;
        std::vector<float> pooled((size_t)rows * 128), iq(4 * 128), hw(4), bias(rows);
        for (auto& v : pooled) v = rng.gauss() * 0.3f;
        for (auto& v : iq) v = rng.gauss() * 0.3f;
        for (auto& v : hw) v = rng.range(0.2f, 1.f);
        for (auto& v : bias) v = rng.range(-1.f, 1.f);
        float *d_p = dup(pooled), *d_iq = dup(iq), *d_hw = dup(hw), *d_b = dup(bias), *d_s = dalloc<float>(rows), *d_s2 = dalloc<float>(rows);
        for (int n : {1, 4, 5, 6, 7, 8, 100, 1001, 2399, 2400}) for (int use_bias = 0; use_bias < 2; ++use_bias) {
            const int nb = n / 4 + (n % 4 != 0);
            CHECK(qsa_block_scores(d_p, d_iq, d_hw, use_bias ? d_b : nullptr, d_s, n, rows, nullptr) == cudaSuccess);
            CHECK(ilu_qsa::block_score(d_p, d_iq, d_hw, use_bias ? d_b : nullptr, d_s2, n, rows, nullptr) == ilu_qsa::kOk);
            sync();
            const auto s = fetch(d_s, nb), s2 = fetch(d_s2, nb);
            int bad = 0;
            for (int r = 0; r < nb; ++r) {
                const double ref = ref_block_score(pooled, iq, hw, use_bias ? &bias : nullptr, r, n);
                if (!score_ok(s[r], ref)) ++bad;
                if (!score_ok(s[r], s2[r])) ++bad;
            }
            CHECK(bad == 0);
        }
        CHECK(qsa_block_scores(d_p, d_iq, d_hw, nullptr, d_s, 4 * rows + 1, rows, nullptr) == cudaErrorInvalidValue);     // rows < nb
        CHECK(qsa_block_scores(d_p, d_iq, d_hw, nullptr, d_p + 128, 8, rows, nullptr) == cudaErrorInvalidValue);          // output inside input
        CHECK(qsa_block_scores(nullptr, d_iq, d_hw, nullptr, d_s, 8, rows, nullptr) == cudaErrorInvalidValue);
        CHECK(qsa_block_scores(d_p, d_iq, d_hw, nullptr, d_s, 0, rows, nullptr) == cudaErrorInvalidValue);
        sync();
        for (float* p : {d_p, d_iq, d_hw, d_b, d_s, d_s2}) cudaFree(p);
    }
    // exact top-k
    const int max_n = 9000, max_w = 2051;
    float* d_scores = dalloc<float>(max_n / 4 + 2);
    int32_t* d_ids = dalloc<int32_t>(max_w);
    int32_t* d_n = dalloc<int32_t>(1);
    int total = 0, mism = 0;
    for (int mode = 0; mode < 6; ++mode) {
        for (int n : {1, 3, 4, 5, 16, 100, 2051, 2052, 2053, 2055, 2056, 2060, 4000, 8191, 9000}) {
            for (int width : {1, 2, 3, 7, 100, 2051}) {
                const int nb = n / 4 + (n % 4 != 0);
                std::vector<float> sc(nb);
                for (int b = 0; b < nb; ++b) {
                    switch (mode) {
                        case 0: sc[b] = std::floor(rng.range(-3.f, 3.f)) * 0.5f; break;     // many ties, both signs
                        case 1: sc[b] = rng.gauss(); break;                                  // distinct
                        case 2: sc[b] = 1.25f; break;                                        // all equal
                        case 3: { const uint32_t r = rng.u32() % 10; sc[b] = r == 0 ? NAN : r == 1 ? INFINITY : r == 2 ? -INFINITY : r == 3 ? -0.f : r == 4 ? 0.f : std::floor(rng.gauss()); break; }
                        case 4: sc[b] = NAN; break;                                          // all NaN
                        default: sc[b] = (b % 7 == 0) ? 3.f : (b % 5 == 0 ? 3.f : -1.f); break;
                    }
                }
                if (mode == 5 && nb > 0) sc[nb - 1] = 1.0e9f;                              // tail-bias style score
                CU(cudaMemcpy(d_scores, sc.data(), nb * 4, cudaMemcpyHostToDevice));
                std::vector<int32_t> fill(max_w, -5); CU(cudaMemcpy(d_ids, fill.data(), max_w * 4, cudaMemcpyHostToDevice));
                CHECK(qsa_block_topk(d_scores, n, width, d_ids, d_n, nullptr) == cudaSuccess);
                sync();
                const int n_out = fetch(d_n, 1)[0];
                const auto ids = fetch(d_ids, width);
                const auto ref = ref_topk(sc, n, width);
                ++total;
                bool ok = n_out == (int)std::min(n, width);
                for (int i = 0; ok && i < width; ++i) ok = ids[i] == (i < (int)ref.size() ? ref[i] : -1);
                if (!ok) { ++mism; if (mism < 10) std::printf("  topk mismatch mode=%d n=%d width=%d n_out=%d\n", mode, n, width, n_out); }
            }
        }
    }
    CHECK(mism == 0);
    std::printf("  topk cases: %d, mismatches: %d\n", total, mism);
    // NaN scores lose to everything finite; with room for all finite cells no NaN cell is chosen
    {
        std::vector<float> sc(100, NAN); for (int b = 0; b < 40; ++b) sc[b] = (float)b;
        CU(cudaMemcpy(d_scores, sc.data(), 100 * 4, cudaMemcpyHostToDevice));
        CHECK(qsa_block_topk(d_scores, 400, 160, d_ids, d_n, nullptr) == cudaSuccess); sync();
        const auto ids = fetch(d_ids, 160);
        int nan_cells = 0; for (int i : ids) if (i >= 160) ++nan_cells;
        CHECK(nan_cells == 0 && fetch(d_n, 1)[0] == 160);
    }
    // rejections
    CHECK(qsa_block_topk(nullptr, 10, 4, d_ids, d_n, nullptr) == cudaErrorInvalidValue);
    CHECK(qsa_block_topk(d_scores, 0, 4, d_ids, d_n, nullptr) == cudaErrorInvalidValue);
    CHECK(qsa_block_topk(d_scores, 10, 0, d_ids, d_n, nullptr) == cudaErrorInvalidValue);
    CHECK(qsa_block_topk(d_scores, 10, 4, (int32_t*)d_scores, d_n, nullptr) == cudaErrorInvalidValue);   // ids over scores
    CHECK(qsa_block_topk(d_scores, 10, 4, d_ids, d_ids + 1, nullptr) == cudaErrorInvalidValue);          // n_out inside ids
    sync();
    for (void* p : {(void*)d_scores, (void*)d_ids, (void*)d_n}) cudaFree(p);
}

int main() {
    std::printf("== host invariants ==\n");
    u293::run_all_host();
    std::printf("host failures so far: %d\n", g_fail);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { ++g_fail; std::printf("FAIL no CUDA device\n"); }
    else {
        CU(cudaSetDevice(0));
        std::printf("== PLE post-ops ==\n");     test_ple_postops();
        std::printf("== indexer append ==\n");   test_indexer();
        std::printf("== score tiles ==\n");      test_score();
        std::printf("== select / top-k ==\n");   test_select();
        sync();
    }
    if (g_fail == 0) { std::printf("U293 TESTS PASSED\n"); return 0; }
    std::printf("U293 TESTS FAILED (%d)\n", g_fail);
    return 1;
}
