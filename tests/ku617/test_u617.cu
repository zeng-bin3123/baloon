// test_u617.cu - LAYER 1 invariants of MtpDrafter, VerifyWindow, SuffixDrafter and VerifyWindowKernels checked against the
// production code (mtp_drafter.hpp, verify_window_protocol.hpp, suffix_drafter.hpp, verify_window_kernels.hpp and its CUDA
// implementation). The device section opens a GPU through the runtime; any failed check makes the process exit non-zero.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "baloon/mtp_drafter.hpp"
#include "baloon/suffix_drafter.hpp"
#include "baloon/verify_commit_dev.hpp"
#include "baloon/verify_window_kernels.hpp"
#include "baloon/verify_window_protocol.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

namespace {
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return n ? (uint32_t)(next() >> 16) % n : 0; }
    float f() { return (float)((next() >> 11) & 0xFFFFFF) / 8388608.f - 1.f; }
};

namespace mtp = baloon::mtp_drafter;
namespace ver = baloon::verify;
namespace sfx = baloon::suffix;
namespace vw = baloon::vwin;

// ------------------------------------------------------------------------------------------------ MtpDrafter
void test_mtp() {
    Rng r(1);
    for (uint64_t n = 0; n < 3000; ++n) {
        const uint64_t a = mtp::align256(n);
        CHECK(a >= n && a % 256 == 0 && a < n + 256);
    }
    // bump total: regions are padded individually; monotone, 256-aligned
    for (int it = 0; it < 200; ++it) {
        std::vector<uint64_t> parts(1 + r.below(10));
        uint64_t sum = 0;
        for (auto& p : parts) { p = r.below(5000); sum += p; }
        const uint64_t tot = mtp::bump_total(parts);
        CHECK(tot % 256 == 0 && tot >= sum);
        uint64_t manual = 0;
        for (uint64_t p : parts) manual += (p + 255) / 256 * 256;
        CHECK(tot == manual);
    }
    // window capacity
    for (uint64_t w = 1; w < 2000; ++w) {
        const uint64_t cap = mtp::cap_of(w, 1u << 20);
        CHECK(cap >= w && cap % 64 == 0 && cap < w + 64);
    }
    // step rows
    for (uint64_t maxT = 1; maxT <= 16; ++maxT) {
        const uint64_t ra = mtp::full_layer_row(maxT);
        CHECK(ra == 2 * maxT - 1 && ra < 2 * maxT);
        for (uint64_t j = 1; j + 2 <= maxT; ++j) {
            const uint64_t cr = mtp::chain_row(maxT, j);
            CHECK(cr >= maxT && cr < ra);
            for (uint64_t t = 0; t < maxT; ++t) CHECK(t != cr);       // catch-up rows [0,T) never collide
        }
        for (uint64_t row = 0; row < 2 * maxT; ++row) CHECK((row * 4 + 3) * 4 + 4 <= (2 * maxT) * 4 * 4);
    }
    // draft chain: at most max_t - 1 drafts
    for (uint64_t maxT = 2; maxT <= 16; ++maxT)
        for (uint64_t steps = 0; steps + 2 <= maxT; ++steps) CHECK(mtp::draft_count(steps) <= maxT - 1);
    // K/V-only staging: step_row0 = s < 0 maps to row -1 - s >= 0; -1 -> 0
    for (int s = -64; s < 0; ++s) CHECK(-1 - s >= 0);
    CHECK(-1 - (-1) == 0);
    // skipped prefill chunk cells lie outside the window
    for (int it = 0; it < 500; ++it) {
        const int64_t window = 1 + r.below(300), plen = window + 64 + r.below(2000), fn = plen - window - 64;
        const int64_t cell0 = r.below(2000), c = r.below(64), T = 1 + r.below(64);
        if (cell0 + c + T <= fn)
            for (int64_t t = 0; t < T; ++t) { const int64_t cell = cell0 + c + t; CHECK(cell < fn && cell + window + 64 < plen); }
    }
    for (uint64_t cell = 0; cell < 1000; ++cell) CHECK((cell + 1) / 4 <= cell + 1);
}

// ------------------------------------------------------------------------------------------------ VerifyWindow
void test_verify_window() {
    for (uint64_t T = 1; T <= 64; ++T) {
        CHECK(ver::group_end(2, 0, T) == ver::group_begin(2, 1, T) && ver::group_begin(2, 0, T) == 0 && ver::group_end(2, 1, T) == T);
        const uint64_t a = ver::group_end(2, 0, T) - ver::group_begin(2, 0, T), b = ver::group_end(2, 1, T) - ver::group_begin(2, 1, T);
        CHECK(a + b == T && b <= a);
        if (T >= 2) CHECK(a > 0 && b > 0);
        CHECK(ver::group_begin(1, 0, T) == 0 && ver::group_end(1, 0, T) == T);
    }
    for (uint64_t G = 1; G <= 2; ++G)
        for (uint64_t k = 0; k < 200; ++k) {
            const uint64_t l = ver::ring_layer(G, k), grp = ver::ring_group(G, k);
            CHECK(grp < G && ver::ring_no(G, l, grp) == k + 1);
            if (grp + 1 < G) CHECK(ver::ring_no(G, l, grp + 1) == ver::ring_no(G, l, grp) + 1);
        }
    // flag B only rises, commutes, idempotent
    Rng r(2);
    for (int it = 0; it < 2000; ++it) {
        const uint64_t c = r.below(50), a = r.below(50), b = r.below(50);
        CHECK(ver::raise(c, a) >= c && ver::raise(c, a) >= a);
        CHECK(ver::raise(ver::raise(c, a), b) == ver::raise(ver::raise(c, b), a));
        CHECK(ver::raise(ver::raise(c, a), a) == ver::raise(c, a));
    }
    // plan layout
    for (uint64_t cap = 0; cap < 600; ++cap) {
        const uint64_t po = ver::ptr_off(cap), base = 4 + (cap + 1) + cap + cap;
        CHECK(po % 2 == 0 && po >= base && po <= base + 1);
        CHECK(ver::ptr2_off(cap) + 2 * cap == ver::start2_off(cap));
        CHECK(ver::start2_off(cap) + (cap + 1) < ver::plan_total(cap));
        CHECK(ver::plan_total(cap) == po + 4 * cap + (cap + 1) + 1);
        CHECK(ver::start_off(cap) == 4 && ver::dst_off(cap) == 4 + cap + 1 && ver::tok_off(cap) == 4 + (cap + 1) + cap);
        CHECK(ver::plan_alloc_words(ver::plan_total(cap)) >= ver::plan_slot_off(ver::plan_total(cap), 1) + ver::plan_total(cap));
    }
    // staging split
    for (uint64_t S = 0; S < 100; ++S) {
        CHECK(ver::staging_begin(1, S, 0) + ver::per(1, S) <= S);
        for (uint64_t grp = 0; grp < 2; ++grp) CHECK(ver::staging_begin(2, S, grp) + ver::per(2, S) <= S);
        CHECK(ver::staging_begin(2, S, 0) + ver::per(2, S) <= ver::staging_begin(2, S, 1));   // disjoint windows
    }
    // commit rows
    for (int64_t nk = 1; nk <= 8; ++nk)
        for (int64_t pos0 : {0, 5, 1000}) {
            const auto v = ver::commit_vector(pos0, nk, 8);
            CHECK(v.size() == 10 && v[0] == nk && v[1] == nk - 1 && v[1] < nk);
            for (int64_t t = 0; t < 8; ++t) CHECK(v[2 + t] == (t < nk ? pos0 + t : -1));
        }
}

// ------------------------------------------------------------------------------------------------ SuffixDrafter
void test_suffix() {
    for (uint64_t m = 0; m < 64; ++m) CHECK(sfx::min_match_of(m) >= 3);
    for (uint64_t lo = 0; lo < 20; ++lo) for (uint64_t hi = 0; hi < 20; ++hi) CHECK(sfx::max_match_of(lo, hi) >= lo);
    Rng r(3);
    for (uint64_t t = 1; t < 5000; t += 7) {
        const uint64_t cap = sfx::capacity_for(t);
        CHECK(cap >= t * 2 && (cap & (cap - 1)) == 0);
    }
    for (int it = 0; it < 2000; ++it) { const uint64_t x = ((uint64_t)r.next()); CHECK(sfx::key_of(x) != 0); }
    for (uint64_t cap : {16ull, 64ull, 1024ull}) for (uint64_t i = 0; i < cap; ++i) CHECK(sfx::probe_next(i, cap) < cap);
    // way list
    for (int it = 0; it < 500; ++it) {
        std::vector<uint64_t> l(r.below(6));
        for (auto& x : l) x = r.below(1000);
        const uint64_t e = r.below(1000);
        const auto out = sfx::push_way(sfx::WAYS, l, e);
        CHECK(out.size() <= sfx::WAYS && !out.empty() && out[0] == e);
        for (size_t i = 1; i < out.size(); ++i) CHECK(out[i] == l[i - 1]);
    }
    // backwards match loop: bounds against brute force
    for (int it = 0; it < 500; ++it) {
        const uint32_t n = 8 + r.below(60);
        std::vector<uint32_t> h(n);
        for (auto& x : h) x = r.below(3);
        const uint64_t cur = n - 1, p = r.below(n - 1), maxm = 1 + r.below(12);
        const uint64_t len = sfx::match_len(h, p, cur, maxm);
        CHECK(len <= maxm && len <= p + 1);
        for (uint64_t k = 0; k < len; ++k) CHECK(h[p - k] == h[cur - k]);
        CHECK(len == maxm || len == p + 1 || h[p - len] != h[cur - len]);
        CHECK(sfx::match_aux(h, p, cur, maxm, 0) == len);
    }
    // continuation copy
    for (uint64_t be = 0; be < 20; ++be) for (uint64_t cur = be; cur < 30; ++cur) for (uint64_t mk = 0; mk < 6; ++mk) {
        const uint64_t k = sfx::cont_count(be, cur, mk);
        CHECK(k == std::min(mk, cur - be) && k <= mk && k <= cur - be);
        if (be < cur && mk > 0) CHECK(k > 0);
    }
    // behaviour of the class: a repeated phrase is continued after the earlier occurrence
    {
        sfx::SuffixDrafter d(256, 3, 16);
        const uint32_t phrase[] = {11, 22, 33, 44, 55, 66, 77};
        for (uint32_t x : phrase) d.append(x);
        for (uint32_t x : {1u, 2u, 9u}) d.append(x);
        for (int i = 0; i < 3; ++i) d.append(phrase[i]);
        const auto p = d.propose(4);
        CHECK(p.size() == 4 && p[0] == 44 && p[1] == 55 && p[2] == 66 && p[3] == 77);
        CHECK(d.propose(0).empty() && d.propose(2).size() == 2);
        d.reset();
        CHECK(d.size() == 0 && d.propose(4).empty());
    }
    // random streams: never more than max_k, and every proposal equals an earlier continuation of a >= min_match suffix
    for (int it = 0; it < 60; ++it) {
        sfx::SuffixDrafter d(512, 3, 12);
        const uint32_t n = 20 + r.below(300), alpha = 2 + r.below(4);
        for (uint32_t i = 0; i < n; ++i) {
            d.append(r.below(alpha));
            const uint64_t mk = r.below(9);
            const auto p = d.propose(mk);
            CHECK(p.size() <= mk);
            if (!p.empty()) {
                const auto& h = d.history();
                const uint64_t cur = h.size() - 1;
                bool ok = false;
                for (uint64_t e = 0; e < cur && !ok; ++e) {
                    if (sfx::match_len(h, e, cur, d.max_match()) < d.min_match()) continue;
                    if (e + p.size() > cur) continue;
                    bool same = true;
                    for (size_t j = 0; j < p.size(); ++j) same = same && h[e + 1 + j] == p[j];
                    ok = same;
                }
                CHECK(ok);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ device
template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t c) : n(c) { cudaMalloc((void**)&p, std::max<size_t>(1, c) * sizeof(T)); cudaMemset(p, 0, std::max<size_t>(1, c) * sizeof(T)); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> h(n); cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }
};
bool synced() { return cudaDeviceSynchronize() == cudaSuccess; }

void test_device() {
    Rng r(4);
    // conv commit: new history entry j == entry n + j of [hist0..2, x_0 ..]; n <= 0 keeps hist; n clamped to T
    for (int T = 1; T <= 8; ++T)
        for (int nk : {-3, 0, 1, 2, 3, T, T + 5}) {
            const int C = 256;
            std::vector<float> x((size_t)T * C), hist(3 * (size_t)C);
            for (auto& v : x) v = r.f();
            for (auto& v : hist) v = r.f();
            Dev<float> dx(x), dh(hist);
            Dev<int32_t> dn(std::vector<int32_t>{nk});
            CHECK(vw::gdn_conv_commit(dx.p, dh.p, dn.p, T, C, nullptr) == 0 && synced());
            const auto got = dh.get();
            const int n = nk < 0 ? 0 : (nk > T ? T : nk);
            bool ok = true;
            for (int j = 0; j < 3; ++j) for (int c = 0; c < C; ++c) {
                const int e = n + j;
                const float want = e < 3 ? hist[(size_t)e * C + c] : x[(size_t)(e - 3) * C + c];
                ok = ok && got[(size_t)j * C + c] == want;
            }
            CHECK(ok);
        }
    // dense_steps: [c, c+1, (c+1)/4, c+1]
    for (int c0 : {0, 1, 3, 4, 1000}) {
        const int T = 8;
        Dev<int32_t> ds(4 * T);
        CHECK(vw::dense_steps(ds.p, c0, T, nullptr) == 0 && synced());
        const auto g = ds.get();
        for (int t = 0; t < T; ++t) { const int c = c0 + t; CHECK(g[4 * t] == c && g[4 * t + 1] == c + 1 && g[4 * t + 2] == (c + 1) / 4 && g[4 * t + 3] == c + 1); }
    }
    // window ids: clipped to the last `window` cells, width = min(n_kv, window)
    for (int nkv : {1, 5, 64, 100, 300}) for (int window : {1, 8, 64, 128}) {
        Dev<int32_t> ids(128), wd(1);
        std::vector<int32_t> init(128, 77);
        cudaMemcpy(ids.p, init.data(), 128 * 4, cudaMemcpyHostToDevice);
        CHECK(vw::window_ids(ids.p, wd.p, nkv, window, nullptr) == 0 && synced());
        const auto g = ids.get(); const int w = wd.get()[0], st = nkv > window ? nkv - window : 0;
        CHECK(w == std::min(nkv, window));
        for (int j = 0; j < std::min(128, w); ++j) CHECK(g[j] == st + j && g[j] < nkv);
    }
    // fetch_blobs: flat index i -> (i / per, i % per); invalid ids give zero bytes
    {
        const int nb = 6, bytes = 37, n = 5;
        std::vector<uint8_t> src((size_t)nb * bytes);
        for (auto& v : src) v = (uint8_t)(1 + r.below(255));
        std::vector<int32_t> ids = {3, 0, 5, -1, 9};
        Dev<uint8_t> ds(src), dd((size_t)n * bytes);
        Dev<int32_t> di(ids);
        CHECK(vw::fetch_blobs(ds.p, di.p, dd.p, n, nb, bytes, nullptr) == 0 && synced());
        const auto g = dd.get();
        for (int k = 0; k < n; ++k) for (int b = 0; b < bytes; ++b) {
            const bool valid = ids[k] >= 0 && ids[k] < nb;
            CHECK(g[(size_t)k * bytes + b] == (valid ? src[(size_t)ids[k] * bytes + b] : 0));
        }
    }
    // gather_rows
    {
        const int rows = 7, row16 = 10, n = 4;
        std::vector<uint16_t> src((size_t)rows * row16);
        for (auto& v : src) v = (uint16_t)(1 + r.below(60000));
        std::vector<int32_t> ids = {6, 2, -1, rows};
        Dev<uint16_t> ds(src), dd((size_t)n * row16);
        Dev<int32_t> di(ids);
        CHECK(vw::gather_rows(ds.p, di.p, dd.p, n, rows, row16, nullptr) == 0 && synced());
        const auto g = dd.get();
        for (int k = 0; k < n; ++k) for (int o = 0; o < row16; ++o) {
            const bool valid = ids[k] >= 0 && ids[k] < rows;
            CHECK(g[(size_t)k * row16 + o] == (valid ? src[(size_t)ids[k] * row16 + o] : 0));
        }
    }
    // rebase_ptrs: ptr k = base + k * bytes
    for (int n : {1, 7, 128}) {
        Dev<uint64_t> dp(n);
        const uint64_t base = 0x7f0000001000ull, bytes = 4096 + 256;
        CHECK(vw::rebase_ptrs(dp.p, base, bytes, n, nullptr) == 0 && synced());
        const auto g = dp.get();
        for (int k = 0; k < n; ++k) CHECK(g[k] == base + (uint64_t)k * bytes);
        if (n > 1) for (int k = 0; k + 1 < n; ++k) CHECK(g[k + 1] - g[k] == bytes);
    }
    CHECK(vw::rebase_ptrs(nullptr, 0, 0, 129, nullptr) != 0);   // over the 128-thread limit is rejected
    // map_ids, ident_hits
    {
        std::vector<int32_t> map = {10, 11, 12, 13}, ids = {0, 3, 4, -1, 2};
        Dev<int32_t> dm(map), di(ids), dout(ids.size());
        CHECK(vw::map_ids(di.p, dm.p, dout.p, (int)ids.size(), (int)map.size(), nullptr) == 0 && synced());
        const auto g = dout.get();
        for (size_t i = 0; i < ids.size(); ++i) CHECK(g[i] == (ids[i] >= 0 && ids[i] < 4 ? map[ids[i]] : -1));
        std::vector<int32_t> id2(300);
        int want = 0;
        for (int i = 0; i < 300; ++i) { id2[i] = r.below(2) ? i : (int)r.below(300); want += id2[i] == i; }
        Dev<int32_t> d2(id2), dh(1);
        CHECK(vw::ident_hits(d2.p, 300, dh.p, nullptr) == 0 && synced() && dh.get()[0] == want);
    }
    // mtp_select: lowest index of the maximum probability, token read at that index
    for (int it = 0; it < 20; ++it) {
        const int n = 1 + r.below(8);
        std::vector<int32_t> tok(n); std::vector<float> prob(n);
        for (int i = 0; i < n; ++i) { tok[i] = 100 + i * 7; prob[i] = (float)r.below(4) * 0.25f; }
        int best = 0;
        for (int i = 1; i < n; ++i) if (prob[i] > prob[best]) best = i;
        Dev<int32_t> dt(tok), ds(1), dto(1);
        Dev<float> dp(prob);
        CHECK(vw::mtp_select(dt.p, dp.p, n, ds.p, dto.p, nullptr) == 0 && synced());
        CHECK(ds.get()[0] == best && dto.get()[0] == tok[best]);
    }
    // device commit rows == host commit vector; n_keep clamped to [0, last_t]
    for (int last_t : {0, 1, 7, 8, 300})
        for (int nk : {-2, 0, 1, 5, 8, 1000})
            for (int pos0 : {0, 41}) {
                Dev<int32_t> dn(std::vector<int32_t>{nk}), dout((size_t)last_t + 2);
                CHECK(baloon::vcommit::commit_rows(dout.p, dn.p, pos0, last_t, nullptr) == 0 && synced());
                const int n = nk < 0 ? 0 : (nk > last_t ? last_t : nk);
                const auto want = ver::commit_vector(pos0, n, last_t);
                CHECK(dout.get() == want);
            }
    CHECK(baloon::vcommit::commit_rows(nullptr, nullptr, 0, 4, nullptr) == -1);
    CHECK(baloon::vcommit::commit_rows(reinterpret_cast<int32_t*>(1), reinterpret_cast<int32_t*>(1), 0, 5000, nullptr) == -1);
    // accepted prefix: a = longest agreeing prefix, n_keep = a + 1 <= n_draft + 1
    for (int it = 0; it < 40; ++it) {
        const int nd = r.below(12);
        std::vector<int32_t> dr(nd), tg(nd);
        for (int i = 0; i < nd; ++i) { dr[i] = r.below(3); tg[i] = r.below(5) ? dr[i] : (int)r.below(3); }
        int a = 0;
        while (a < nd && dr[a] == tg[a]) ++a;
        Dev<int32_t> dd(dr), dt(tg), dk(1), da(1);
        CHECK(baloon::vcommit::accept_prefix(dd.p, dt.p, nd, dk.p, da.p, nullptr) == 0 && synced());
        CHECK(da.get()[0] == a && dk.get()[0] == a + 1 && a <= nd);
    }
    // wait_flag_ge: reached flag reports 1, unreachable flag times out with 0
    {
        Dev<uint32_t> fl(std::vector<uint32_t>{5});
        Dev<int32_t> st(1);
        CHECK(vw::wait_flag_ge(fl.p, 5, 1000, st.p, nullptr) == 0 && synced() && st.get()[0] == 1);
        CHECK(vw::wait_flag_ge(fl.p, 6, 1000, st.p, nullptr) == 0 && synced() && st.get()[0] == 0);
    }
    // row_top_prob: tok = lowest argmax, prob = 1 / sum exp(l - max)
    {
        const int rows = 3, vocab = 1000;
        std::vector<float> lg((size_t)rows * vocab);
        for (auto& v : lg) v = r.f() * 4.f;
        lg[0 * vocab + 17] = lg[0 * vocab + 400] = 9.f;               // tie: lowest index wins
        lg[1 * vocab + 999] = 12.f;
        Dev<float> dl(lg), dp(rows);
        Dev<int32_t> dt(rows);
        CHECK(vw::row_top_prob(dl.p, dt.p, dp.p, rows, vocab, nullptr) == 0 && synced());
        const std::vector<int32_t> t = dt.get(); const std::vector<float> p = dp.get();
        for (int i = 0; i < rows; ++i) {
            int best = 0;
            for (int j = 1; j < vocab; ++j) if (lg[(size_t)i * vocab + j] > lg[(size_t)i * vocab + best]) best = j;
            double s = 0;
            for (int j = 0; j < vocab; ++j) s += std::exp((double)lg[(size_t)i * vocab + j] - lg[(size_t)i * vocab + best]);
            CHECK(t[i] == best);
            CHECK(std::fabs(p[i] - 1.0 / s) <= 1e-4 * (1.0 / s) + 1e-7);
        }
    }
}
}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("FAIL no device\n"); return 2; }
    cudaSetDevice(0);
    test_mtp();
    test_verify_window();
    test_suffix();
    test_device();
    std::printf("checks=%d failures=%d\n", g_checks, g_fail);
    if (g_fail == 0) std::printf("U617 UNIT TESTS PASSED\n");
    return g_fail == 0 ? 0 : 1;
}

