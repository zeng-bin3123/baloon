// Unit uca8d125571 GPU tests: every kernel of the unit against a host model on the real device.
// Exit code 0 only if a device was opened and every check passed.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/act_quant_kernels.hpp"
#include "baloon/f16_bits.hpp"
#include "baloon/qsa_kernels.hpp"
#include "baloon/qsa_native_kernels.hpp"
#include "baloon/verify_window_kernels.hpp"

using namespace baloon;
using qsa_step::Step;

static long g_checks = 0, g_fail = 0;
#define CHECK(c)                                                                                                  \
    do {                                                                                                          \
        ++g_checks;                                                                                               \
        if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); }        \
    } while (0)
#define CUCHK(x)                                                                                                  \
    do {                                                                                                          \
        const cudaError_t e_ = (x);                                                                               \
        if (e_ != cudaSuccess) { std::printf("CUDA ERROR %d at %s:%d (%s)\n", (int)e_, __FILE__, __LINE__, #x); std::exit(2); } \
    } while (0)

static std::mt19937 rng(12345u);
static uint32_t rnd() { return rng(); }
static float rf(float lo, float hi) { return lo + (hi - lo) * ((rnd() >> 8) / 16777216.0f); }

template <class T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { CUCHK(cudaMalloc((void**)&p, std::max<size_t>(count, 1) * sizeof(T))); CUCHK(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T))); }
    explicit Dev(const std::vector<T>& h) : n(h.size()) { CUCHK(cudaMalloc((void**)&p, std::max<size_t>(h.size(), 1) * sizeof(T))); if (!h.empty()) CUCHK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const { std::vector<T> h(n); if (n) CUCHK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    void set(const std::vector<T>& h) { CUCHK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); }
};
static void sync() { CUCHK(cudaDeviceSynchronize()); }
static bool near(float a, float b, float rel = 2e-4f, float abs_ = 2e-5f) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::fabs(a - b) <= abs_ + rel * std::max(std::fabs(a), std::fabs(b));
}
static uint16_t h16(float f) { return f16bits::f16_from_f32(f); }
static float f32h(uint16_t h) { return f16bits::f32_from_f16(h); }

// ---------------------------------------------------------------------------------------------------------------------
// QSA cache step, KV cache, indexer, selection
// ---------------------------------------------------------------------------------------------------------------------
static void test_step_and_kv() {
    using namespace qsa_dev;
    const Shapes sh = qsa_step::real_shapes(16, 4096);
    Step* ds = step_scratch(0);
    CHECK(ds != nullptr);
    for (int pos : {0, 3, 4, 100, 2049, 2050, 4095}) {
        CHECK(qsa_step_fill(pos, sh, ds, nullptr) == kOk);
        Step h{};
        CUCHK(cudaMemcpy(&h, ds, sizeof(Step), cudaMemcpyDeviceToHost));
        const Step e = qsa_step::step_fill(pos, 2048, 4);
        CHECK(h.pos == e.pos && h.n_kv == e.n_kv && h.n_bid == e.n_bid && h.width == e.width);
    }
    CHECK(step_upload_width(7, ds, nullptr) == kOk);
    Step h{};
    CUCHK(cudaMemcpy(&h, ds, sizeof(Step), cudaMemcpyDeviceToHost));
    CHECK(h.width == 7 && h.pos == 4095);  // only the width field changed
    CHECK(qsa_step_fill(-1, sh, ds, nullptr) == kInvalid);

    // paged KV cache with a permuted page table
    const int P = sh.page_size, H = sh.n_head_kv, HD = sh.head_dim, cells = sh.max_cells, nPages = cells / P;
    std::vector<int32_t> table(nPages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    Dev<int32_t> dtab(table);
    const size_t rows = (size_t)nPages * H * P;
    Dev<uint16_t> kp(rows * HD), vp(rows * HD);
    Dev<float> k((size_t)H * HD), v((size_t)H * HD);
    std::vector<std::vector<float>> hk(cells), hv(cells);
    for (int pos : {0, 1, 15, 16, 17, 255, 1000, 4095}) {
        hk[pos].resize((size_t)H * HD); hv[pos].resize((size_t)H * HD);
        for (auto& x : hk[pos]) x = rf(-3, 3);
        for (auto& x : hv[pos]) x = rf(-3, 3);
        k.set(hk[pos]); v.set(hv[pos]);
        if (pos % 2 == 0) {
            CHECK(kv_append(k.p, v.p, kp.p, vp.p, dtab.p, pos, sh, nullptr) == kOk);
        } else {
            CHECK(qsa_step_fill(pos, sh, ds, nullptr) == kOk);
            CHECK(kv_append_step(k.p, v.p, kp.p, vp.p, dtab.p, ds, sh, nullptr) == kOk);
        }
    }
    sync();
    CHECK(kv_append(k.p, v.p, kp.p, vp.p, dtab.p, cells, sh, nullptr) == kInvalid);  // out of capacity
    const auto kph = kp.get();
    const auto vph = vp.get();
    for (int pos : {0, 1, 15, 16, 17, 255, 1000, 4095}) {
        for (int h2 = 0; h2 < H; ++h2) {
            const int64_t row = qsa_step::kv_row(table[pos / P], H, h2, P, pos);
            bool okk = true, okv = true;
            for (int d = 0; d < HD; ++d) {
                okk &= kph[row * HD + d] == h16(hk[pos][h2 * HD + d]);
                okv &= vph[row * HD + d] == h16(hv[pos][h2 * HD + d]);
            }
            CHECK(okk && okv);
        }
    }
    // gather (explicit n and step width)
    const std::vector<int32_t> ids = {4095, 0, 1000, 16, 255, 17, 15, 1};
    Dev<int32_t> dids(ids);
    Dev<float> out(ids.size() * H * HD);
    CHECK(kv_gather(kp.p, dtab.p, dids.p, 6, sh, out.p, nullptr) == kOk);
    sync();
    auto og = out.get();
    for (int i = 0; i < 6; ++i) for (int h2 = 0; h2 < H; ++h2) {
        const int64_t row = qsa_step::kv_row(table[ids[i] / P], H, h2, P, ids[i]);
        bool ok = true;
        for (int d = 0; d < HD; ++d) ok &= og[((size_t)i * H + h2) * HD + d] == f32h(kph[row * HD + d]);
        CHECK(ok);
    }
    Step st = qsa_step::step_fill(4095, 2048, 4);
    st.width = 3;
    CHECK(step_upload(st, ds, nullptr) == kOk);
    Dev<float> out2(ids.size() * H * HD);
    CHECK(kv_gather_step(vp.p, dtab.p, dids.p, ds, sh, out2.p, nullptr) == kOk);
    sync();
    og = out2.get();
    for (int i = 0; i < 3; ++i) {
        const int64_t row = qsa_step::kv_row(table[ids[i] / P], H, 1, P, ids[i]);
        bool ok = true;
        for (int d = 0; d < HD; ++d) ok &= og[((size_t)i * H + 1) * HD + d] == f32h(vph[row * HD + d]);
        CHECK(ok);
    }
    // only `width` cells are written (index 3 untouched = zero)
    bool untouched = true;
    for (int d = 0; d < HD; ++d) untouched &= og[(size_t)3 * H * HD + d] == 0.f;
    CHECK(untouched);
}

static void test_indexer_and_select() {
    using namespace qsa_dev;
    // pooling schedule: tail holds r-1 rows, pooled[pos/4] = mean of the 4 keys when pos % 4 == 3
    Dev<float> tail(3 * 128), pooled(8 * 128), key(128);
    std::vector<std::vector<float>> keys(12, std::vector<float>(128));
    for (int pos = 0; pos < 12; ++pos) {
        for (auto& x : keys[pos]) x = rf(-2, 2);
        key.set(keys[pos]);
        CHECK(indexer_key_append(key.p, tail.p, pooled.p, pos, 4, nullptr) == kOk);
    }
    sync();
    const auto pl = pooled.get();
    for (int b = 0; b < 3; ++b) for (int d = 0; d < 128; ++d) {
        const float e = (keys[b * 4][d] + keys[b * 4 + 1][d] + keys[b * 4 + 2][d] + keys[b * 4 + 3][d]) / 4.f;
        CHECK(near(pl[b * 128 + d], e, 1e-5f, 1e-6f));
    }
    bool zero_rest = true;
    for (size_t i = 3 * 128; i < pl.size(); ++i) zero_rest &= pl[i] == 0.f;
    CHECK(zero_rest);  // blocks that never completed stay untouched

    // cell scores from block scores + exact top-k
    auto ref_topk = [](const std::vector<float>& s, int width) {
        std::vector<int> idx(s.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return (s[a] + 0.f) > (s[b] + 0.f); });
        std::vector<int32_t> o(idx.begin(), idx.begin() + std::min<size_t>(width, idx.size()));
        std::sort(o.begin(), o.end());
        return o;
    };
    const int ns[] = {1, 5, 100, 2051, 2052, 5000, 32768};
    int variant = 0;
    for (int n : ns) {
        const int nb = n / 4;
        std::vector<float> bs(nb + 1);
        for (auto& x : bs) {
            switch (variant % 3) {
                case 0: x = rf(-5, 5); break;
                case 1: x = (float)(int)(rnd() % 4) - 1.f; break;       // many ties
                default: x = (rnd() & 1) ? -0.0f : 0.0f; break;         // all tied (+/-0 fold)
            }
        }
        ++variant;
        if (n % 4 != 0) bs[nb] += 1e9f;  // tail bias
        Dev<float> dbs(bs), cs(std::max(n, 1));
        CHECK(qsa_index(dbs.p, cs.p, n, 4, nullptr) == kOk);
        sync();
        const auto csh = cs.get();
        bool map_ok = true;
        for (int j = 0; j < n; ++j) map_ok &= csh[j] == bs[std::min(j / 4, nb)];
        CHECK(map_ok);
        const int width = std::min(n, 2051);
        std::vector<float> cell(csh.begin(), csh.begin() + n);
        const auto want = ref_topk(cell, width);
        Dev<int32_t> ids(2100), nout(1);
        CHECK(topk_512(cs.p, n, width, ids.p, nout.p, nullptr) == kOk);
        sync();
        const auto got = ids.get();
        CHECK(nout.get()[0] == (int)want.size());
        CHECK(std::equal(want.begin(), want.end(), got.begin()));
        // device-step form
        Step* ds = step_scratch(1);
        const qsa_step::Step s = qsa_step::step_fill(n - 1, 2048, 4);
        CHECK(step_upload(s, ds, nullptr) == kOk);
        Dev<int32_t> ids2(2100), nout2(1);
        CHECK(topk_512_step(cs.p, ds, ids2.p, nout2.p, nullptr) == kOk);
        sync();
        CHECK(ids2.get() == got && nout2.get()[0] == nout.get()[0]);
        // tail cells are always selected when a partial block exists
        if (n % 4 != 0 && n > 2051) {
            bool all_tail = true;
            for (int j = nb * 4; j < n; ++j) all_tail &= std::binary_search(got.begin(), got.begin() + width, j);
            CHECK(all_tail);
        }
    }
    CHECK(topk_512(nullptr, 5, 3, nullptr, nullptr, nullptr) == kInvalid);
    {   // block scoring + block-weighted top-k
        for (int n : {1, 4, 7, 64, 1001, 6000}) {
            const int nb = n / 4, rows = nb + 1;
            std::vector<float> pooledh((size_t)rows * 128), iq(4 * 128), hw(4);
            for (auto& x : pooledh) x = rf(-1, 1);
            for (auto& x : iq) x = rf(-1, 1);
            for (auto& x : hw) x = rf(0.1f, 1.f);
            Dev<float> dp(pooledh), diq(iq), dhw(hw), sc(rows);
            CHECK(qsa_block_scores(dp.p, diq.p, dhw.p, sc.p, n, 4, rows, nullptr) == kOk);
            CHECK(qsa_block_scores(dp.p, diq.p, dhw.p, sc.p, n, 4, rows - 1, nullptr) == kInvalid);  // too few pooled rows
            sync();
            const auto g = sc.get();
            for (int b = 0; b < rows; ++b) {
                float e = 0.f;
                for (int h = 0; h < 4; ++h) { float dot = 0.f; for (int d = 0; d < 128; ++d) dot += pooledh[(size_t)b * 128 + d] * iq[h * 128 + d]; e += hw[h] * std::max(dot, 0.f); }
                if (b == nb && n % 4 != 0) e += 1e9f;
                CHECK(near(g[b], e, 1e-4f, 1e-4f) || (e > 1e8f && near(g[b], e, 1e-6f, 1.f)));
            }
            const int width = std::min(n, 2051);
            std::vector<float> cell(n);
            for (int j = 0; j < n; ++j) cell[j] = g[std::min(j / 4, nb)];
            const auto want = ref_topk(cell, width);
            Dev<int32_t> ids(2100), nout(1);
            CHECK(qsa_block_topk(sc.p, n, 4, width, ids.p, nout.p, nullptr) == kOk);
            sync();
            CHECK(nout.get()[0] == (int)want.size());
            CHECK(std::equal(want.begin(), want.end(), ids.get().begin()));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Attention kernels
// ---------------------------------------------------------------------------------------------------------------------
struct Pool16 { std::vector<uint16_t> k, v; };
static void ref_attend(const std::vector<float>& q, const std::function<float(int, int, int)>& K, const std::function<float(int, int, int)>& V,
                       const int32_t* ids, int n, float scale, int n_head, int n_kv, std::vector<float>& out) {
    out.assign((size_t)n_head * 256, 0.f);
    for (int h = 0; h < n_head; ++h) {
        const int kvh = h / (n_head / n_kv);
        std::vector<double> s(n);
        double m = -1e300;
        for (int i = 0; i < n; ++i) { double a = 0; for (int d = 0; d < 256; ++d) a += (double)q[h * 256 + d] * scale * K(ids[i], kvh, d); s[i] = a; m = std::max(m, a); }
        double L = 0; std::vector<double> acc(256, 0.0);
        for (int i = 0; i < n; ++i) { const double p = std::exp(s[i] - m); L += p; for (int d = 0; d < 256; ++d) acc[d] += p * V(ids[i], kvh, d); }
        for (int d = 0; d < 256; ++d) out[h * 256 + d] = n > 0 ? (float)(acc[d] / L) : 0.f;
    }
}

static void test_attention() {
    using namespace qsa_dev;
    const int H = 2, P = 16, cells = 4096, nPages = cells / P;
    const Shapes sh = qsa_step::real_shapes(P, cells);
    std::vector<int32_t> table(nPages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    Dev<int32_t> dtab(table);
    const size_t rows = (size_t)nPages * H * P;
    std::vector<uint16_t> kh(rows * 256), vh(rows * 256);
    for (auto& x : kh) x = h16(rf(-1, 1));
    for (auto& x : vh) x = h16(rf(-1, 1));
    Dev<uint16_t> kp(kh), vp(vh);
    auto rowof = [&](int cell, int kvh) { return qsa_step::kv_row(table[cell / P], H, kvh, P, cell); };
    auto Kf = [&](int cell, int kvh, int d) { return f32h(kh[rowof(cell, kvh) * 256 + d]); };
    auto Vf = [&](int cell, int kvh, int d) { return f32h(vh[rowof(cell, kvh) * 256 + d]); };
    std::vector<float> qh(24 * 256);
    for (auto& x : qh) x = rf(-1, 1);
    Dev<float> dq(qh);
    const float scale = 0.0625f;
    for (int n : {1, 7, 64, 65, 700, 2051}) {
        std::vector<int32_t> ids(2051);
        std::vector<int32_t> perm(cells);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < 2051; ++i) ids[i] = perm[i];
        Dev<int32_t> dids(ids);
        Dev<float> out(24 * 256);
        CHECK(qsa_attend(dq.p, kp.p, vp.p, dtab.p, dids.p, n, scale, sh, out.p, nullptr) == kOk);
        sync();
        std::vector<float> want;
        ref_attend(qh, Kf, Vf, ids.data(), n, scale, 24, 2, want);
        const auto got = out.get();
        bool ok = true;
        for (size_t i = 0; i < want.size(); ++i) ok &= near(got[i], want[i], 5e-4f, 5e-5f);
        CHECK(ok);
        // step form (width from the device record)
        Step st = qsa_step::step_fill(n - 1, 2048, 4);
        Step* ds = step_scratch(2);
        step_upload(st, ds, nullptr);
        Dev<float> out2(24 * 256);
        CHECK(qsa_attend_step(dq.p, kp.p, vp.p, dtab.p, dids.p, ds, scale, sh, out2.p, nullptr) == kOk);
        sync();
        CHECK(out2.get() == got);
    }
    // split-K decode attention: fp16 pools, identity page map, batch of queries with ragged n_ids
    {
        const int nq = 7, cap = 2051;
        const std::vector<int32_t> nids = {0, 1, 63, 64, 65, 129, 2051};
        std::vector<float> q((size_t)nq * 24 * 256);
        for (auto& x : q) x = rf(-1, 1);
        std::vector<int32_t> ids((size_t)nq * cap);
        for (auto& x : ids) x = (int32_t)(rnd() % cells);
        // identity page map: pool row (c / P * H + kvh) * P + c % P
        std::vector<uint16_t> k2(rows * 256), v2(rows * 256);
        for (auto& x : k2) x = h16(rf(-1, 1));
        for (auto& x : v2) x = h16(rf(-1, 1));
        Dev<float> dq2(q);
        Dev<int32_t> dids(ids), dn(nids);
        Dev<uint16_t> dk(k2), dv(v2);
        Dev<float> scratch(nq * qsa_dev::qsa_decode_attn_scratch_floats(cap, 24)), out((size_t)nq * 24 * 256);
        out.set(std::vector<float>(out.n, -1.f));
        QsaAttnPools pools{dk.p, dv.p, nullptr, nullptr, nullptr, P, H, false};
        CHECK(qsa_decode_attn_batch(pools, dq2.p, dids.p, dn.p, nq, cap, scale, scratch.p, out.p, nullptr) == kOk);
        sync();
        const auto got = out.get();
        auto rid = [&](int cell, int kvh) { return ((int64_t)(cell / P) * H + kvh) * P + cell % P; };
        for (int z = 0; z < nq; ++z) {
            std::vector<float> qz(q.begin() + (size_t)z * 24 * 256, q.begin() + (size_t)(z + 1) * 24 * 256), want;
            ref_attend(qz, [&](int c, int kvh, int d) { return f32h(k2[rid(c, kvh) * 256 + d]); },
                       [&](int c, int kvh, int d) { return f32h(v2[rid(c, kvh) * 256 + d]); }, &ids[(size_t)z * cap], nids[z], scale, 24, 2, want);
            bool ok = true;
            for (size_t i = 0; i < want.size(); ++i) ok &= near(got[(size_t)z * 24 * 256 + i], want[i], 5e-4f, 5e-5f);
            CHECK(ok);  // n_ids == 0 gives zeros
        }
        // step form uses step->width
        Step* ds = step_scratch(3);
        Step st = qsa_step::step_fill(699, 2048, 4);
        step_upload(st, ds, nullptr);
        Dev<float> out1(24 * 256);
        CHECK(qsa_decode_attn_step(pools, dq2.p, dids.p, ds, cap, scale, scratch.p, out1.p, nullptr) == kOk);
        sync();
        std::vector<float> qz(q.begin(), q.begin() + 24 * 256), want;
        ref_attend(qz, [&](int c, int kvh, int d) { return f32h(k2[rid(c, kvh) * 256 + d]); },
                   [&](int c, int kvh, int d) { return f32h(v2[rid(c, kvh) * 256 + d]); }, ids.data(), 700, scale, 24, 2, want);
        const auto g1 = out1.get();
        bool ok = true;
        for (size_t i = 0; i < want.size(); ++i) ok &= near(g1[i], want[i], 5e-4f, 5e-5f);
        CHECK(ok);
        // int8 pools with group scales
        std::vector<int8_t> k8(rows * 256), v8(rows * 256);
        std::vector<float> ks(rows * 4), vs(rows * 4);
        for (auto& x : k8) x = (int8_t)((int)(rnd() % 255) - 127);
        for (auto& x : v8) x = (int8_t)((int)(rnd() % 255) - 127);
        for (auto& x : ks) x = rf(0.001f, 0.02f);
        for (auto& x : vs) x = rf(0.001f, 0.02f);
        Dev<int8_t> dk8(k8), dv8(v8);
        Dev<float> dks(ks), dvs(vs), out8((size_t)nq * 24 * 256);
        QsaAttnPools p8{dk8.p, dv8.p, dks.p, dvs.p, nullptr, P, H, true};
        CHECK(qsa_decode_attn_batch(p8, dq2.p, dids.p, dn.p, nq, cap, scale, scratch.p, out8.p, nullptr) == kOk);
        sync();
        const auto g8 = out8.get();
        for (int z = 0; z < nq; ++z) {
            std::vector<float> qz8(q.begin() + (size_t)z * 24 * 256, q.begin() + (size_t)(z + 1) * 24 * 256), w8;
            ref_attend(qz8, [&](int c, int kvh, int d) { return (float)k8[rid(c, kvh) * 256 + d] * ks[rid(c, kvh) * 4 + d / 64]; },
                       [&](int c, int kvh, int d) { return (float)v8[rid(c, kvh) * 256 + d] * vs[rid(c, kvh) * 4 + d / 64]; }, &ids[(size_t)z * cap], nids[z], scale, 24, 2, w8);
            bool ok8 = true;
            for (size_t i = 0; i < w8.size(); ++i) ok8 &= near(g8[(size_t)z * 24 * 256 + i], w8[i], 1e-3f, 1e-4f);
            CHECK(ok8);
        }
        CHECK(qsa_decode_attn_batch(p8, dq2.p, dids.p, dn.p, 0, cap, scale, scratch.p, out8.p, nullptr) == kInvalid);
    }
    // short-context flash kernel
    {
        const int maxCtx = 256;
        std::vector<uint16_t> fk((size_t)maxCtx * 2 * 256), fv((size_t)maxCtx * 2 * 256);
        for (auto& x : fk) x = h16(rf(-1, 1));
        for (auto& x : fv) x = h16(rf(-1, 1));
        Dev<uint16_t> dfk(fk), dfv(fv);
        Dev<int32_t> status(1);
        Step* ds = step_scratch(4);
        for (int width : {1, 4, 5, 127, 128, 129, 255, 256}) {
            Step st = qsa_step::step_fill(width - 1, 2048, 4);
            step_upload(st, ds, nullptr);
            Dev<float> out(24 * 256);
            CHECK(native_flash_attn_short_step(dq.p, dfk.p, dfv.p, ds, maxCtx, out.p, status.p, nullptr) == kOk);
            sync();
            std::vector<int32_t> ids(width);
            std::iota(ids.begin(), ids.end(), 0);
            std::vector<float> want;
            ref_attend(qh, [&](int c, int kvh, int d) { return f32h(fk[((size_t)c * 2 + kvh) * 256 + d]); },
                       [&](int c, int kvh, int d) { return f32h(fv[((size_t)c * 2 + kvh) * 256 + d]); }, ids.data(), width, 0.0625f, 24, 2, want);
            const auto got = out.get();
            bool ok = true;
            for (size_t i = 0; i < want.size(); ++i) ok &= near(got[i], want[i], 5e-4f, 5e-5f);
            CHECK(ok);
            CHECK(status.get()[0] == 0);
        }
        // inconsistent step -> NaN output and status 1
        Step bad = qsa_step::step_fill(9, 2048, 4);
        bad.n_kv = 11;
        step_upload(bad, ds, nullptr);
        Dev<float> outb(24 * 256);
        CHECK(native_flash_attn_short_step(dq.p, dfk.p, dfv.p, ds, maxCtx, outb.p, status.p, nullptr) == kOk);
        sync();
        const auto gb = outb.get();
        bool all_nan = true;
        for (float x : gb) all_nan &= std::isnan(x);
        CHECK(all_nan && status.get()[0] == 1);
        Step big = qsa_step::step_fill(300, 2048, 4);   // wider than max_context
        step_upload(big, ds, nullptr);
        status.set({0});
        CHECK(native_flash_attn_short_step(dq.p, dfk.p, dfv.p, ds, maxCtx, outb.p, status.p, nullptr) == kOk);
        sync();
        CHECK(status.get()[0] == 1);
        CHECK(native_flash_attn_short_step(dq.p, dfk.p, dfv.p, ds, 0, outb.p, status.p, nullptr) == kInvalid);
        CHECK(native_flash_attn_short_step(dq.p, dfk.p, dfv.p, ds, maxCtx, (float*)dq.p, status.p, nullptr) == kInvalid);  // out aliases q
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Tile scorer
// ---------------------------------------------------------------------------------------------------------------------
static float tf32_h(float x) { uint32_t u = f16bits::bits_of(x) & 0xFFFFE000u; return f16bits::float_of(u); }
static void test_native_score() {
    using namespace qsa_dev;
    const int maxCells = 4096, rows = maxCells / 4 + 1;
    std::vector<float> pooled((size_t)rows * 128), iq(4 * 128), hw(4);
    for (auto& x : pooled) x = rf(-1, 1);
    for (auto& x : iq) x = rf(-1, 1);
    for (auto& x : hw) x = rf(0.1f, 1.f);
    Dev<float> dp(pooled), diq(iq), dhw(hw), cs(maxCells);
    Step* ds = step_scratch(5);
    ScoreArgs a{dp.p, diq.p, dhw.p, cs.p, ds, rows, maxCells};
    native_qsa_score_set_enabled(false);
    CHECK(!native_qsa_score_enabled() && native_qsa_score(a, nullptr) == kInvalid);
    native_qsa_score_set_enabled(true);
    CHECK(native_qsa_score_enabled());
    for (int n : {1, 3, 4, 5, 100, 2051, 4093, 4096}) {
        cs.set(std::vector<float>(maxCells, -7.f));
        step_upload(qsa_step::step_fill(n - 1, 2048, 4), ds, nullptr);
        CHECK(native_qsa_score(a, nullptr) == kOk);
        sync();
        const auto got = cs.get();
        const int full = n / 4;
        bool ok = true;
        for (int j = 0; j < n; ++j) {
            const int row = j / 4;
            float dot[4];
            for (int h = 0; h < 4; ++h) {
                float a0 = 0.f, a1 = 0.f;
                for (int k = 0; k < 64; ++k) a0 = std::fmaf(tf32_h(pooled[(size_t)row * 128 + k]), tf32_h(iq[h * 128 + k]), a0);
                for (int k = 64; k < 128; ++k) a1 = std::fmaf(tf32_h(pooled[(size_t)row * 128 + k]), tf32_h(iq[h * 128 + k]), a1);
                dot[h] = a0 + a1;
            }
            float e = 0.f;
            for (int h = 0; h < 4; ++h) e += hw[h] * std::max(dot[h], 0.f);
            if (row == full && n % 4 != 0) e += 1e9f;
            ok &= near(got[j], e, 1e-4f, 1e-4f) || (e > 1e8f && std::fabs(got[j] - e) <= 64.f);
        }
        CHECK(ok);
        bool untouched = true;
        for (int j = n; j < maxCells; ++j) untouched &= got[j] == -7.f;
        CHECK(untouched);  // cells >= n are never written
    }
    {   // inconsistent step: nothing is written
        cs.set(std::vector<float>(maxCells, -7.f));
        Step bad = qsa_step::step_fill(99, 2048, 4);
        bad.n_bid += 1;
        step_upload(bad, ds, nullptr);
        CHECK(native_qsa_score(a, nullptr) == kOk);
        sync();
        bool untouched = true;
        for (float x : cs.get()) untouched &= x == -7.f;
        CHECK(untouched);
    }
    ScoreArgs overlap = a;
    overlap.cell_score = dp.p;  // output aliases the input keys
    CHECK(native_qsa_score(overlap, nullptr) == kInvalid);
    ScoreArgs small = a;
    small.rows = rows - 1;
    CHECK(native_qsa_score(small, nullptr) == kInvalid);
    native_qsa_score_set_enabled(false);
}

// ---------------------------------------------------------------------------------------------------------------------
// int8 KV + activation quantizers
// ---------------------------------------------------------------------------------------------------------------------
static void test_q8_and_quant() {
    using namespace qsa_dev;
    const Shapes sh = qsa_step::real_shapes(16, 4096);
    const int H = 2, P = 16, nPages = 4096 / P;
    std::vector<int32_t> table(nPages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    Dev<int32_t> dtab(table);
    const size_t rows = (size_t)nPages * H * P;
    Dev<int8_t> kp(rows * 256), vp(rows * 256);
    Dev<float> ksc(rows * 4), vsc(rows * 4), k(H * 256), v(H * 256);
    Step* ds = step_scratch(6);
    std::vector<int> poss = {0, 5, 16, 999, 4095};
    std::vector<std::vector<float>> hk(4096), hv(4096);
    for (int pos : poss) {
        hk[pos].resize(H * 256); hv[pos].resize(H * 256);
        for (int i = 0; i < H * 256; ++i) { const float mag = (i / 64 == 1) ? 0.f : (float)std::pow(10.0, rf(-2, 1)); hk[pos][i] = mag * rf(-1, 1); hv[pos][i] = rf(-4, 4); }
        k.set(hk[pos]); v.set(hv[pos]);
        step_upload(qsa_step::step_fill(pos, 2048, 4), ds, nullptr);
        CHECK(kv_append_q8_step(k.p, v.p, kp.p, vp.p, ksc.p, vsc.p, dtab.p, ds, sh, nullptr) == kOk);
    }
    sync();
    const auto kq = kp.get();
    const auto ksh = ksc.get();
    const auto vq = vp.get();
    const auto vsh = vsc.get();
    for (int pos : poss) for (int h = 0; h < H; ++h) {
        const int64_t row = qsa_step::kv_row(table[pos / P], H, h, P, pos);
        for (int g = 0; g < 4; ++g) {
            float am = 0.f, am2 = 0.f;
            for (int d = 0; d < 64; ++d) { am = std::max(am, std::fabs(hk[pos][h * 256 + g * 64 + d])); am2 = std::max(am2, std::fabs(hv[pos][h * 256 + g * 64 + d])); }
            const float sc = am / 127.f, sc2 = am2 / 127.f;
            CHECK(ksh[row * 4 + g] == sc && vsh[row * 4 + g] == sc2);
            const float id = sc != 0.f ? 1.f / sc : 0.f, id2 = sc2 != 0.f ? 1.f / sc2 : 0.f;
            bool ok = true;
            for (int d = 0; d < 64; ++d) {
                const float q1 = std::min(127.f, std::max(-127.f, std::nearbyint(hk[pos][h * 256 + g * 64 + d] * id)));
                const float q2 = std::min(127.f, std::max(-127.f, std::nearbyint(hv[pos][h * 256 + g * 64 + d] * id2)));
                ok &= kq[row * 256 + g * 64 + d] == (int8_t)q1 && vq[row * 256 + g * 64 + d] == (int8_t)q2;
                ok &= std::fabs((float)kq[row * 256 + g * 64 + d] * sc - hk[pos][h * 256 + g * 64 + d]) <= sc * 0.5001f + 1e-12f;
            }
            CHECK(ok);
        }
    }
    const std::vector<int32_t> ids = {999, 0, 4095, 16};
    Dev<int32_t> dids(ids);
    step_upload(Step{4095, 4096, 1024, 3}, ds, nullptr);
    Dev<float> out(ids.size() * H * 256);
    CHECK(kv_gather_q8_step(kp.p, ksc.p, dtab.p, dids.p, ds, sh, out.p, nullptr) == kOk);
    sync();
    const auto og = out.get();
    for (int i = 0; i < 3; ++i) for (int h = 0; h < H; ++h) {
        const int64_t row = qsa_step::kv_row(table[ids[i] / P], H, h, P, ids[i]);
        bool ok = true;
        for (int d = 0; d < 256; ++d) ok &= og[((size_t)i * H + h) * 256 + d] == (float)kq[row * 256 + d] * ksh[row * 4 + d / 64];
        CHECK(ok);
    }
    // activation quantizers vs bit-exact host references
    const int rowsN = 3, n = 512;
    std::vector<float> x((size_t)rowsN * n);
    for (auto& f : x) f = (float)std::pow(10.0, rf(-3, 2)) * rf(-1, 1);
    for (int i = 0; i < 64; ++i) x[i] = 0.f;  // all-zero blocks
    x[300] = 5.f; x[301] = -5.f;              // equal magnitudes, opposite signs: lowest index wins
    Dev<float> dx(x);
    Dev<uint8_t> q80((size_t)rowsN * n / 32 * 34), q80s(q80.n), q8k((size_t)rowsN * n / 256 * 292);
    CHECK(act_dev::quantize_q8_0(dx.p, q80.p, rowsN, n, nullptr) == act_dev::kOk);
    CHECK(act_dev::quantize_q8_0_scaled(dx.p, q80s.p, rowsN, n, 0.37f, nullptr) == act_dev::kOk);
    CHECK(act_dev::quantize_q8_K(dx.p, q8k.p, rowsN, n, nullptr) == act_dev::kOk);
    CHECK(act_dev::quantize_q8_K(dx.p, q8k.p, rowsN, 100, nullptr) == act_dev::kInvalid);  // ragged row
    sync();
    const auto g0 = q80.get(), g0s = q80s.get(), gk = q8k.get();
    bool ok0 = true, ok0s = true, okk = true;
    for (size_t b = 0; b < (size_t)rowsN * n / 32; ++b) {
        uint8_t ref[34], refs[34];
        act_quant::ref_quantize_q8_0(&x[b * 32], ref);
        act_quant::ref_quantize_q8_0(&x[b * 32], refs, 0.37f);
        ok0 &= std::memcmp(ref, &g0[b * 34], 34) == 0;
        ok0s &= std::memcmp(refs, &g0s[b * 34], 34) == 0;
    }
    for (size_t b = 0; b < (size_t)rowsN * n / 256; ++b) {
        uint8_t ref[292];
        act_quant::ref_quantize_q8_K(&x[b * 256], ref);
        okk &= std::memcmp(ref, &gk[b * 292], 292) == 0;
    }
    CHECK(ok0 && ok0s && okk);
    Dev<float> y0(x.size()), yk(x.size());
    CHECK(act_dev::dequant_q8_0(q80.p, y0.p, rowsN, n, nullptr) == act_dev::kOk);
    CHECK(act_dev::dequant_q8_K(q8k.p, yk.p, rowsN, n, nullptr) == act_dev::kOk);
    sync();
    const auto h0 = y0.get(), hk2 = yk.get();
    bool d0 = true, dk = true;
    for (size_t b = 0; b < (size_t)rowsN * n / 32; ++b) {
        uint16_t dh; std::memcpy(&dh, &g0[b * 34], 2);
        for (int i = 0; i < 32; ++i) d0 &= h0[b * 32 + i] == f32h(dh) * (float)(int8_t)g0[b * 34 + 2 + i];
    }
    for (size_t b = 0; b < (size_t)rowsN * n / 256; ++b) {
        float d; std::memcpy(&d, &gk[b * 292], 4);
        for (int i = 0; i < 256; ++i) dk &= hk2[b * 256 + i] == d * (float)(int8_t)gk[b * 292 + 4 + i];
    }
    CHECK(d0 && dk);
}

// ---------------------------------------------------------------------------------------------------------------------
// Verify-window kernels
// ---------------------------------------------------------------------------------------------------------------------
static void test_verify() {
    using namespace verify_dev;
    {   // streams
        const int T = 3, n = 40, hc = 4;
        std::vector<float> x((size_t)T * n), R((size_t)T * n * hc), d((size_t)T * n);
        for (auto& f : x) f = rf(-1, 1);
        for (auto& f : R) f = rf(-1, 1);
        for (auto& f : d) f = rf(-1, 1);
        Dev<float> dx(x), dR((size_t)T * n * hc), dd(d), dR2(R);
        CHECK(broadcast_streams(dx.p, dR.p, T, n, hc, nullptr) == kOk);
        CHECK(add_streams_broadcast(dR2.p, dd.p, T, n, hc, nullptr) == kOk);
        sync();
        const auto g = dR.get(), g2 = dR2.get();
        bool ok = true, ok2 = true;
        for (size_t i = 0; i < g.size(); ++i) { const size_t t = i / (n * hc); ok &= g[i] == x[t * n + i % n]; ok2 &= g2[i] == R[i] + d[t * n + i % n]; }
        CHECK(ok && ok2);
        CHECK(broadcast_streams(dx.p, dR.p, kVerifyMaxT + 1, n, hc, nullptr) == kInvalid);
    }
    {   // gathers
        const int rowsN = 10, row = 33;
        std::vector<float> src((size_t)rowsN * row);
        for (auto& f : src) f = rf(-1, 1);
        std::vector<int32_t> idx = {9, 0, 4, 4};
        Dev<float> dsrc(src), dst(idx.size() * row);
        Dev<int32_t> didx(idx);
        CHECK(copy_indexed(dsrc.p, didx.p, dst.p, 4, row, nullptr) == kOk);
        sync();
        const auto g = dst.get();
        bool ok = true;
        for (size_t i = 0; i < idx.size(); ++i) for (int o = 0; o < row; ++o) ok &= g[i * row + o] == src[(size_t)idx[i] * row + o];
        CHECK(ok);
        std::vector<uint16_t> s16((size_t)rowsN * 20);
        for (auto& x : s16) x = (uint16_t)rnd();
        std::vector<int32_t> ids16 = {3, -1, 9, 10, 0};
        Dev<uint16_t> ds16(s16), dd16(ids16.size() * 20);
        Dev<int32_t> di16(ids16);
        CHECK(gather_rows(ds16.p, di16.p, 5, 20, rowsN, dd16.p, nullptr) == kOk);
        sync();
        const auto g16 = dd16.get();
        bool ok16 = true;
        for (size_t i = 0; i < ids16.size(); ++i) for (int o = 0; o < 20; ++o) ok16 &= g16[i * 20 + o] == ((ids16[i] >= 0 && ids16[i] < rowsN) ? s16[(size_t)ids16[i] * 20 + o] : 0);
        CHECK(ok16);  // out-of-range ids give zero rows
        const size_t bb = 77;
        std::vector<uint8_t> blobs(bb * 6);
        for (auto& b : blobs) b = (uint8_t)rnd();
        std::vector<int32_t> bi = {5, 2, 2, 0};
        Dev<uint8_t> dbl(blobs), dout(bi.size() * bb);
        Dev<int32_t> dbi(bi);
        CHECK(fetch_blobs(dbl.p, dbi.p, 4, bb, dout.p, nullptr) == kOk);
        sync();
        const auto gb = dout.get();
        bool okb = true;
        for (size_t i = 0; i < bi.size(); ++i) okb &= std::memcmp(&gb[i * bb], &blobs[(size_t)bi[i] * bb], bb) == 0;
        CHECK(okb);
        for (int bits : {2, 4, 8}) {
            const int cols = 70, per = 8 / bits, vocab = 9;
            const size_t rb = (cols + per - 1) / per;
            std::vector<uint8_t> tab(rb * vocab);
            for (auto& b : tab) b = (uint8_t)rnd();
            std::vector<float> sc(vocab);
            for (auto& s : sc) s = rf(0.01f, 2.f);
            std::vector<int32_t> eids = {8, 0, 3};
            Dev<uint8_t> dtab(tab);
            Dev<float> dsc(sc), eo(eids.size() * cols);
            Dev<int32_t> de(eids);
            CHECK(embedding_gather_dev(dtab.p, dsc.p, de.p, 3, cols, bits, eo.p, nullptr) == kOk);
            sync();
            const auto ge = eo.get();
            bool oke = true;
            for (size_t i = 0; i < eids.size(); ++i) for (int c = 0; c < cols; ++c) {
                const int code = (tab[(size_t)eids[i] * rb + c / per] >> ((c % per) * bits)) & ((1 << bits) - 1);
                oke &= ge[i * cols + c] == (float)(code - (1 << (bits - 1))) * sc[eids[i]];
            }
            CHECK(oke);
        }
        CHECK(embedding_gather_dev(nullptr, nullptr, nullptr, 1, 1, 3, nullptr, nullptr) == kInvalid);
    }
}

static void test_verify_steps_and_accept() {
    using namespace verify_dev;
    {   // dense_steps / window_ids
        Dev<Step> steps(8);
        CHECK(dense_steps(2045, 8, steps.p, nullptr) == kOk);
        sync();
        const auto g = steps.get();
        for (int t = 0; t < 8; ++t) { const int c = 2045 + t; CHECK(g[t].pos == c && g[t].n_kv == c + 1 && g[t].n_bid == (c + 1) / 4 && g[t].width == c + 1); }
        CHECK(dense_steps(0, 0, steps.p, nullptr) == kInvalid);
        for (auto nw : {std::pair<int, int>{10, 4}, {3, 8}, {1000, 256}, {8, 8}}) {
            Dev<Step> st(1);
            st.set({Step{nw.first - 1, nw.first, nw.first / 4, std::min(nw.first, 2051)}});
            Dev<int32_t> ids(300);
            CHECK(window_ids(st.p, nw.second, ids.p, 300, nullptr) == kOk);
            sync();
            const int start = nw.first > nw.second ? nw.first - nw.second : 0, width = nw.first - start;
            const auto gi = ids.get();
            bool ok = true;
            for (int j = 0; j < 300; ++j) ok &= gi[j] == (j < width ? start + j : -1);
            CHECK(ok);
            CHECK(st.get()[0].width == width && st.get()[0].n_kv == nw.first);
        }
    }
    {   // ident_hits, map_ids, mtp_select, rebase_ptrs, row_top_prob, wait_flag_ge
        for (int lead : {0, 1, 5, 12}) {
            std::vector<int32_t> a(12), b(12);
            for (int i = 0; i < 12; ++i) { a[i] = (int)(rnd() % 1000); b[i] = i < lead ? a[i] : a[i] + 1; }
            Dev<int32_t> da(a), db(b), hit(1);
            CHECK(ident_hits(da.p, db.p, 12, hit.p, nullptr) == kOk);
            sync();
            CHECK(hit.get()[0] == lead);
        }
        const std::vector<int32_t> tab = {10, 20, 30, 40}, in = {3, 0, 4, -1, 2};
        Dev<int32_t> dt(tab), di(in), dout(in.size());
        CHECK(map_ids(di.p, dt.p, 4, dout.p, 5, nullptr) == kOk);
        sync();
        CHECK(dout.get() == std::vector<int32_t>({40, 10, -1, -1, 30}));
        std::vector<float> rows(5 * 13);
        for (auto& f : rows) f = rf(-1, 1);
        Dev<float> drows(rows), dst(13);
        for (int r : {0, 4, 5, -1}) {
            Dev<int32_t> dr(std::vector<int32_t>{r});
            dst.set(std::vector<float>(13, 9.f));
            CHECK(mtp_select(drows.p, dr.p, 5, 13, dst.p, nullptr) == kOk);
            sync();
            const auto g = dst.get();
            bool ok = true;
            for (int i = 0; i < 13; ++i) ok &= g[i] == ((r >= 0 && r < 5) ? rows[(size_t)r * 13 + i] : 0.f);
            CHECK(ok);
        }
        Dev<uint8_t> base(4096);
        Dev<void*> ptrs(9);
        CHECK(rebase_ptrs(base.p, 100, ptrs.p, 9, nullptr) == kOk);
        sync();
        const auto gp = ptrs.get();
        bool okp = true;
        for (int k = 0; k < 9; ++k) okp &= (uintptr_t)gp[k] == (uintptr_t)base.p + (size_t)k * 100;
        CHECK(okp);
        const int rowsN = 4, V = 1000;
        std::vector<float> lg((size_t)rowsN * V);
        for (auto& f : lg) f = rf(-5, 5);
        lg[1 * V + 17] = 50.f; lg[1 * V + 900] = 50.f;                       // tie: lowest index wins
        for (int i = 0; i < V; ++i) lg[2 * V + i] = 1.f;                      // all equal: id 0, prob 1/V
        Dev<float> dl(lg), prob(rowsN);
        Dev<int32_t> id(rowsN);
        CHECK(row_top_prob(dl.p, rowsN, V, id.p, prob.p, nullptr) == kOk);
        sync();
        const auto gid = id.get();
        const auto gpb = prob.get();
        for (int r = 0; r < rowsN; ++r) {
            int bi = 0;
            for (int i = 1; i < V; ++i) if (lg[(size_t)r * V + i] > lg[(size_t)r * V + bi]) bi = i;
            double sum = 0;
            for (int i = 0; i < V; ++i) sum += std::exp((double)lg[(size_t)r * V + i] - lg[(size_t)r * V + bi]);
            CHECK(gid[r] == bi);
            CHECK(near(gpb[r], (float)(1.0 / sum), 1e-4f, 1e-7f));
        }
        CHECK(gid[1] == 17 && gid[2] == 0);
        Dev<int32_t> flag(std::vector<int32_t>{5}), status(1);
        status.set({-1});
        CHECK(wait_flag_ge(flag.p, 5, 1000, status.p, nullptr) == kOk);
        sync();
        CHECK(status.get()[0] == 0);
        CHECK(wait_flag_ge(flag.p, 6, 1000, status.p, nullptr) == kOk);
        sync();
        CHECK(status.get()[0] == 1);  // never reached: gives up
    }
}

static float silu_h(float x) { return x / (1.f + std::exp(-x)); }
static void test_gdn_window() {
    using namespace verify_dev;
    const int hk = 2, hv = 4, T = 5, C = (2 * hk + hv) * 128, n_embd = 300;
    {   // alpha / beta
        std::vector<float> x((size_t)T * n_embd), dt(hv), a(hv);
        std::vector<uint16_t> W((size_t)2 * hv * n_embd);
        for (auto& f : x) f = rf(-1, 1);
        for (auto& f : dt) f = rf(-1, 1);
        for (auto& f : a) f = -rf(0.1f, 2.f);
        for (auto& w : W) w = f16bits::bf16_bits_from_f32(rf(-1, 1));
        Dev<float> dx(x), ddt(dt), da(a), gate((size_t)T * hv), beta((size_t)T * hv);
        Dev<uint16_t> dW(W);
        CHECK(gdn_ab_multi(dx.p, dW.p, ddt.p, da.p, gate.p, beta.p, T, n_embd, hv, nullptr) == kOk);
        sync();
        const auto gg = gate.get(), gb = beta.get();
        bool ok = true;
        for (int t = 0; t < T; ++t) for (int row = 0; row < 2 * hv; ++row) {
            double acc = 0;
            for (int i = 0; i < n_embd; ++i) acc += (double)x[(size_t)t * n_embd + i] * f16bits::f32_from_bf16(W[(size_t)row * n_embd + i]);
            if (row < hv) { const double v = acc + dt[row]; const double sp = v > 20 ? v : std::log1p(std::exp(v)); ok &= near(gg[(size_t)t * hv + row], (float)(sp * a[row]), 1e-4f, 1e-5f); }
            else ok &= near(gb[(size_t)t * hv + row - hv], (float)(1.0 / (1.0 + std::exp(-acc))), 1e-4f, 1e-6f);
        }
        CHECK(ok);
    }
    {   // conv window + l2, then commit
        std::vector<float> hist((size_t)C * 3), x((size_t)T * C), w((size_t)C * 4);
        for (auto& f : hist) f = rf(-1, 1);
        for (auto& f : x) f = rf(-1, 1);
        for (auto& f : w) f = rf(-1, 1);
        Dev<float> dh(hist), dx(x), dw(w), out((size_t)T * C);
        const float eps = 1e-6f;
        CHECK(gdn_conv_l2_multi(dh.p, dx.p, dw.p, out.p, T, hk, hv, eps, nullptr) == kOk);
        sync();
        CHECK(dh.get() == hist);  // the conv step never touches the history
        std::vector<float> want((size_t)T * C);
        auto seq = [&](int c, int i) { return i < 3 ? hist[(size_t)c * 3 + i] : x[(size_t)(i - 3) * C + c]; };
        for (int t = 0; t < T; ++t) for (int c = 0; c < C; ++c) {
            float acc = 0.f;
            for (int j = 0; j < 3; ++j) acc += seq(c, t + j) * w[(size_t)c * 4 + j];
            acc += x[(size_t)t * C + c] * w[(size_t)c * 4 + 3];
            want[(size_t)t * C + c] = silu_h(acc);
        }
        for (int t = 0; t < T; ++t) for (int hh = 0; hh < 2 * hk; ++hh) {
            double ss = 0;
            for (int d = 0; d < 128; ++d) ss += (double)want[(size_t)t * C + hh * 128 + d] * want[(size_t)t * C + hh * 128 + d];
            const float inv = (float)(1.0 / std::sqrt(ss + eps));
            for (int d = 0; d < 128; ++d) want[(size_t)t * C + hh * 128 + d] *= inv;
        }
        const auto g = out.get();
        bool ok = true;
        for (size_t i = 0; i < g.size(); ++i) ok &= near(g[i], want[i], 1e-3f, 1e-5f);
        CHECK(ok);
        for (int n : {0, 1, 2, 3, 5}) {
            Dev<float> dh2(hist);
            CHECK(gdn_conv_commit(dh2.p, dx.p, n, C, nullptr) == kOk);
            sync();
            const auto gh = dh2.get();
            bool okc = true;
            for (int c = 0; c < C; ++c) for (int j = 0; j < 3; ++j) okc &= gh[(size_t)c * 3 + j] == (n <= 0 ? hist[(size_t)c * 3 + j] : seq(c, n + j));
            CHECK(okc);
        }
        CHECK(gdn_conv_commit(nullptr, nullptr, 0, C, nullptr) == kOk);  // no-op for n <= 0 even with null buffers
    }
    {   // fused step + norm vs a sequential host model; replay and keep-count
        const size_t SS = 128;
        std::vector<float> state0(SS * hv * SS), q((size_t)T * hk * SS), k((size_t)T * hk * SS), v((size_t)T * hv * SS), gate((size_t)T * hv), beta((size_t)T * hv);
        for (auto& f : state0) f = rf(-0.5f, 0.5f);
        for (auto& f : q) f = rf(-1, 1);
        for (auto& f : k) f = rf(-0.2f, 0.2f);
        for (auto& f : v) f = rf(-1, 1);
        for (auto& f : gate) f = -rf(0.01f, 0.5f);
        for (auto& f : beta) f = rf(0.1f, 0.9f);
        const float scale = 0.0884f, eps = 1e-6f;
        auto run_host = [&](int upto, std::vector<float>& st, std::vector<float>& outh) {
            st = state0;
            outh.assign((size_t)T * hv * SS, 0.f);
            for (int t = 0; t < upto; ++t) for (int h = 0; h < hv; ++h) {
                const int qh = h % hk;
                const float decay = std::exp(gate[(size_t)t * hv + h]);
                std::vector<float> o(SS);
                for (size_t col = 0; col < SS; ++col) {
                    double kv = 0;
                    for (size_t i = 0; i < SS; ++i) { float& s = st[(i * hv + h) * SS + col]; s *= decay; kv += (double)s * k[((size_t)t * hk + qh) * SS + i]; }
                    const double d = (v[((size_t)t * hv + h) * SS + col] - kv) * beta[(size_t)t * hv + h];
                    double od = 0;
                    for (size_t i = 0; i < SS; ++i) { float& s = st[(i * hv + h) * SS + col]; s += (float)(k[((size_t)t * hk + qh) * SS + i] * d); od += (double)s * q[((size_t)t * hk + qh) * SS + i]; }
                    o[col] = (float)(od * scale);
                }
                double ss = 0;
                for (float f : o) ss += (double)f * f;
                const float inv = (float)(1.0 / std::sqrt(ss / 128.0 + eps));
                for (size_t col = 0; col < SS; ++col) outh[((size_t)t * hv + h) * SS + col] = o[col] * inv;
            }
        };
        Dev<float> dq(q), dk(k), dv(v), dg(gate), db(beta);
        struct Case { int keep_n; bool has_keep; int t_out; };
        for (Case cs : {Case{0, false, 0}, Case{T, true, 0}, Case{3, true, 0}, Case{2, true, 2}, Case{0, true, 0}}) {
            Dev<float> dst(state0), out((size_t)T * hv * SS);
            out.set(std::vector<float>(out.n, -9.f));
            Dev<int32_t> keep(std::vector<int32_t>{cs.keep_n});
            CHECK(gdn_step_norm_multi(dst.p, dq.p, dk.p, dv.p, dg.p, db.p, out.p, T, hk, hv, scale, eps, cs.has_keep ? keep.p : nullptr, cs.t_out, nullptr) == kOk);
            sync();
            std::vector<float> st, outh;
            run_host(T, st, outh);
            const auto go = out.get();
            bool ok = true;
            for (int t = 0; t < T; ++t) for (size_t i = 0; i < (size_t)hv * SS; ++i) {
                const size_t idx = (size_t)t * hv * SS + i;
                ok &= (t < cs.t_out) ? go[idx] == -9.f : near(go[idx], outh[idx], 2e-3f, 2e-4f);  // replayed tokens write no output
            }
            CHECK(ok);
            // state: committed after exactly `keep` tokens, never without a keep-count
            std::vector<float> want_state;
            std::vector<float> dummy;
            if (cs.has_keep && cs.keep_n > 0) run_host(cs.keep_n, want_state, dummy); else want_state = state0;
            const auto gs = dst.get();
            bool oks = true;
            for (size_t i = 0; i < gs.size(); ++i) oks &= near(gs[i], want_state[i], 2e-3f, 2e-4f);
            CHECK(oks);
        }
        CHECK(gdn_step_norm_multi(nullptr, dq.p, dk.p, dv.p, dg.p, db.p, nullptr, T, hk, hv, scale, eps, nullptr, 0, nullptr) == kInvalid);
        CHECK(gdn_step_norm_multi(dq.p, dq.p, dk.p, dv.p, dg.p, db.p, dq.p, T, hk, 3, scale, eps, nullptr, 0, nullptr) == kInvalid);  // h_v % h_k != 0
    }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) { std::printf("NO GPU DEVICE\n"); return 2; }
    CUCHK(cudaSetDevice(0));
    CUCHK(cudaFree(nullptr));  // force context creation: opens the device node
    std::printf("devices: %d\n", count);
    test_step_and_kv();
    test_indexer_and_select();
    test_attention();
    test_native_score();
    test_q8_and_quant();
    test_verify();
    test_verify_steps_and_accept();
    test_gdn_window();
    sync();
    std::printf("gpu checks=%ld failures=%ld\n", g_checks, g_fail);
    std::printf(g_fail == 0 ? "UNIT19 GPU TESTS PASSED\n" : "UNIT19 GPU TESTS FAILED\n");
    return g_fail == 0 ? 0 : 1;
}
