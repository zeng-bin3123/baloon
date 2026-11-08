// test_gpu.cu - unit ucbadf2af54 device checks: router_top10 (block-wide argmax, ties -> lowest index, 2^-14 clamp, thread/smem
// sizing), shared_expert / shared_expert_multi (scratch carving, 1..8 tokens, row scaling), moe_combine (k <= 64, plain shared add)
// and the sampler (penalty window = tail of the history, presence boolean, greedy ties -> lowest index, top-k / top-p bounds).
// Every check compares against a CPU reference written from the theory; a failing check makes the process exit non-zero.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "baloon/f16_bits.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"
#include "baloon/router_top_k_selection.hpp"
#include "baloon/shared_expert_layout.hpp"

using namespace baloon;

static int g_fail = 0, g_n = 0;
#define CHECK(c) do { ++g_n; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T> struct Dev {
    T* p = nullptr; size_t n = 0;
    explicit Dev(size_t count) : n(count) { CU(cudaMalloc((void**)&p, std::max<size_t>(count, 1) * sizeof(T))); CU(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { CU(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    std::vector<T> get() const { std::vector<T> v(n); CU(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
    operator T*() const { return p; }
};
static bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess; }

static uint16_t to_bf16(float f) { return f16bits::round_f32_to_bf16_bits(f); }
static float from_bf16(uint16_t b) { return f16bits::bf16_bits_to_f32(b); }
static float round_bf(float f) { return from_bf16(to_bf16(f)); }
static float round_h(float f) { return f16bits::f32_from_f16(f16bits::f16_from_f32(f)); }

// ------------------------------------------------------------------------------------------------ router
// CPU reference: the same float softmax recipe, then k stable-argmax passes (lowest index wins ties), clamp 2^-14.
static void router_ref(const std::vector<float>& lg, int T, int E, int k, std::vector<int>& ids, std::vector<float>& wt) {
    ids.assign((size_t)T * k, 0); wt.assign((size_t)T * k, 0.f);
    for (int t = 0; t < T; ++t) {
        const float* l = &lg[(size_t)t * E];
        float mx = -INFINITY; for (int e = 0; e < E; ++e) mx = std::max(mx, l[e]);
        std::vector<float> p(E); float sum = 0.f;
        for (int e = 0; e < E; ++e) { p[e] = std::exp(l[e] - mx); sum += p[e]; }
        const float inv = 1.0f / sum; for (int e = 0; e < E; ++e) p[e] *= inv;
        std::vector<char> taken(E, 0); std::vector<float> ch(k);
        for (int i = 0; i < k; ++i) {
            int bi = -1;
            for (int e = 0; e < E; ++e) { if (taken[e]) continue; if (bi < 0 || p[e] > p[bi]) bi = e; }
            taken[bi] = 1; ids[(size_t)t * k + i] = bi; ch[i] = p[bi];
        }
        float s = 0.f; for (int i = 0; i < k; ++i) s += ch[i];
        const float sc = std::max(s, 6.103515625e-05f);
        for (int i = 0; i < k; ++i) wt[(size_t)t * k + i] = ch[i] / sc;
    }
}
static void run_router(const std::vector<float>& lg, int T, int E, int k, const char* what) {
    Dev<float> dl(lg), dw((size_t)T * k); Dev<int32_t> di((size_t)T * k);
    CHECK(kg::router_top10(dl, T, E, k, di, dw) == cudaSuccess && sync_ok());
    std::vector<int> rid; std::vector<float> rw; router_ref(lg, T, E, k, rid, rw);
    const auto gi = di.get(); const auto gw = dw.get();
    bool ids_ok = true, w_ok = true;
    for (size_t i = 0; i < rid.size(); ++i) { if (gi[i] != rid[i]) ids_ok = false; if (std::fabs(gw[i] - rw[i]) > 1e-5f + 1e-4f * std::fabs(rw[i])) w_ok = false; }
    if (!ids_ok || !w_ok) std::printf("  router mismatch (%s): T=%d E=%d k=%d ids_ok=%d w_ok=%d\n", what, T, E, k, ids_ok, w_ok);
    CHECK(ids_ok); CHECK(w_ok);
    for (int t = 0; t < T; ++t) {                                                                         // distinct ids, weights sum to 1
        std::set<int> u; double s = 0;
        for (int i = 0; i < k; ++i) { u.insert(gi[(size_t)t * k + i]); s += gw[(size_t)t * k + i]; }
        CHECK((int)u.size() == k && std::fabs(s - 1.0) < 1e-4);
    }
}
static void test_router() {
    std::mt19937 rng(7);
    auto rnd = [&](int n, float lo, float hi) { std::vector<float> v(n); for (auto& x : v) x = lo + (hi - lo) * (float)(rng() % 100000) / 100000.f; return v; };
    auto quant = [&](int n, int levels) { std::vector<float> v(n); for (auto& x : v) x = (float)(int)(rng() % levels) * 0.5f; return v; };   // many exact ties
    // expert counts around the warp/thread boundaries: 10 -> 32 threads, 96 -> 96 threads (partial wave), 130 -> 160, 512 -> 512, 600 -> 512
    for (int E : {10, 33, 64, 96, 130, 257, 512, 600}) {
        const int k = std::min(E, 10);
        run_router(rnd(5 * E, -4.f, 4.f), 5, E, k, "random");
        run_router(quant(5 * E, 3), 5, E, k, "ties");
    }
    run_router(rnd(512, -3.f, 3.f), 1, 512, 1, "k=1");
    run_router(quant(3 * 130, 4), 3, 130, 64, "k=64");
    run_router(std::vector<float>(512, 0.25f), 1, 512, 10, "all equal");
    {   // all-equal logits: the lowest indices win, in order
        std::vector<float> z(130, 1.f); Dev<float> dz(z), dw(10); Dev<int32_t> di(10);
        CHECK(kg::router_top10(dz, 1, 130, 10, di, dw) == cudaSuccess && sync_ok());
        const auto I = di.get(); const auto W = dw.get();
        for (int i = 0; i < 10; ++i) { CHECK(I[i] == i); CHECK(std::fabs(W[i] - 0.1f) < 1e-6f); }
    }
    {   // a tie between experts in different threads: the lower index wins, whatever thread holds it
        std::vector<float> l(512, 0.f); l[500] = 5.f; l[37] = 5.f; l[255] = 5.f; l[256] = 5.f;
        Dev<float> dl(l), dw(4); Dev<int32_t> di(4);
        CHECK(kg::router_top10(dl, 1, 512, 4, di, dw) == cudaSuccess && sync_ok());
        const auto I = di.get(); CHECK(I[0] == 37 && I[1] == 255 && I[2] == 256 && I[3] == 500);
    }
    {   // the native router agrees with the generic one on 512 experts, k = 10 (same selection, same weights)
        auto l = rnd(2 * 512, -4.f, 4.f);
        Dev<float> dl(l), w1(20), w2(20); Dev<int32_t> i1(20), i2(20);
        CHECK(kg::router_top10(dl, 2, 512, 10, i1, w1) == cudaSuccess && kg::native_router_top10(dl, 2, 512, 10, i2, w2) == cudaSuccess && sync_ok());
        CHECK(i1.get() == i2.get());
        const auto a = w1.get(), b = w2.get(); bool ok = true; for (size_t i = 0; i < a.size(); ++i) ok = ok && std::fabs(a[i] - b[i]) < 1e-6f; CHECK(ok);
    }
    {   // argument guards
        Dev<float> dl(1024); Dev<float> dw(100); Dev<int32_t> di(100);
        CHECK(kg::router_top10(dl, 1, 100, 65, di, dw) != cudaSuccess);
        CHECK(kg::router_top10(dl, 1, 100, 0, di, dw) != cudaSuccess);
        CHECK(kg::router_top10(dl, 1, 8, 9, di, dw) != cudaSuccess);
        CHECK(kg::router_top10(dl, 0, 8, 2, di, dw) != cudaSuccess);
        CHECK(kg::router_top10(nullptr, 1, 8, 2, di, dw) != cudaSuccess);
        CHECK(kg::router_top10(dl, 1, 8, 2, di, dw) == cudaSuccess && sync_ok());
    }
}

// ------------------------------------------------------------------------------------------------ shared expert / combine
static std::vector<uint16_t> bfv(std::mt19937& rng, size_t n, float amp) { std::vector<uint16_t> v(n); for (auto& x : v) x = to_bf16(amp * ((float)(rng() % 20001) / 10000.f - 1.f)); return v; }
static void test_shared_expert() {
    std::mt19937 rng(11);
    struct Shape { int n, ff; };
    for (Shape sh : {Shape{64, 96}, Shape{128, 640}, Shape{48, 33}}) {
        const int n = sh.n, ff = sh.ff;
        auto gw = bfv(rng, (size_t)ff * n, 0.4f), uw = bfv(rng, (size_t)ff * n, 0.4f), dw = bfv(rng, (size_t)n * ff, 0.4f), sw = bfv(rng, n, 0.4f);
        Dev<uint16_t> dgw(gw), duw(uw), ddw(dw), dsw(sw);
        for (int Tn : {1, 2, 5, 8}) {
            std::vector<float> x((size_t)Tn * n); for (auto& v : x) v = (float)(rng() % 20001) / 10000.f - 1.f;
            Dev<float> dx(x), out((size_t)Tn * n);
            const size_t need = kg::shared_expert_scratch_bytes(Tn, n, ff);
            CHECK(need == shared_expert::scratch_bytes((uint64_t)Tn, (uint64_t)ff));
            Dev<uint8_t> scr(need);
            for (int mode = 0; mode < 2; ++mode) {
                kg::shared_expert_set_native_bf16(mode == 1);
                CHECK(kg::shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, scr, need) == cudaSuccess && sync_ok());
                const auto O = out.get();
                bool ok = true; double worst = 0;
                for (int t = 0; t < Tn; ++t) {
                    std::vector<float> h(ff);
                    for (int f = 0; f < ff; ++f) {
                        double g = 0, u = 0;
                        for (int i = 0; i < n; ++i) { g += (double)from_bf16(gw[(size_t)f * n + i]) * x[(size_t)t * n + i]; u += (double)from_bf16(uw[(size_t)f * n + i]) * x[(size_t)t * n + i]; }
                        const float gf = (float)g, uf = (float)u;
                        const float hv = (gf / (1.f + std::exp(-gf))) * uf;
                        h[f] = mode ? round_bf(hv) : round_h(hv);
                    }
                    double gs = 0; for (int i = 0; i < n; ++i) gs += (double)x[(size_t)t * n + i] * from_bf16(sw[i]);
                    float gate = (float)(1.0 / (1.0 + std::exp(-gs))); if (mode) gate = round_bf(gate);
                    for (int o = 0; o < n; ++o) {
                        double a = 0; for (int f = 0; f < ff; ++f) a += (double)from_bf16(dw[(size_t)o * ff + f]) * h[f];
                        const double want = a * gate, got = O[(size_t)t * n + o];
                        const double err = std::fabs(got - want); worst = std::max(worst, err);
                        if (err > 6e-3 + 2e-2 * std::fabs(want)) ok = false;
                    }
                }
                if (!ok) std::printf("  shared expert mismatch n=%d ff=%d T=%d mode=%d worst=%g\n", n, ff, Tn, mode, worst);
                CHECK(ok);
            }
            kg::shared_expert_set_native_bf16(false);
            if (Tn == 5) {   // each row of a multi-token launch equals the single-token launch of that row
                CHECK(kg::shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, scr, need) == cudaSuccess && sync_ok());
                const auto O = out.get();
                for (int t = 0; t < Tn; ++t) {
                    std::vector<float> xt(x.begin() + (size_t)t * n, x.begin() + (size_t)(t + 1) * n); Dev<float> dxt(xt), o1(n);
                    Dev<uint8_t> s1(kg::shared_expert_scratch_bytes(1, n, ff));
                    CHECK(kg::shared_expert(dxt, n, ff, dgw, duw, ddw, dsw, o1, s1, s1.n) == cudaSuccess && sync_ok());
                    const auto r = o1.get(); bool same = true;
                    for (int o = 0; o < n; ++o) same = same && std::fabs(r[o] - O[(size_t)t * n + o]) < 1e-5f + 1e-5f * std::fabs(r[o]);
                    CHECK(same);
                }
            }
            // guards: one byte short of the carved scratch, a misaligned scratch, token counts outside 1..8
            CHECK(kg::shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, scr, need - 1) != cudaSuccess);
            CHECK(kg::shared_expert_multi(dx, Tn, n, ff, dgw, duw, ddw, dsw, out, (uint8_t*)scr.p + 8, need) != cudaSuccess);
        }
        std::vector<float> x9((size_t)9 * n, 0.5f); Dev<float> dx9(x9), o9((size_t)9 * n); Dev<uint8_t> s9(kg::shared_expert_scratch_bytes(9, n, ff) + 64);
        CHECK(kg::shared_expert_multi(dx9, 9, n, ff, dgw, duw, ddw, dsw, o9, s9, s9.n) != cudaSuccess);
        CHECK(kg::shared_expert_multi(dx9, 0, n, ff, dgw, duw, ddw, dsw, o9, s9, s9.n) != cudaSuccess);
    }
}
static void test_combine() {
    std::mt19937 rng(13);
    for (int ne : {1, 127, 128, 129, 300, 2560}) for (int k : {1, 6, 10, 64}) {
        std::vector<float> parts((size_t)k * ne), w(k), sh(ne);
        for (auto& v : parts) v = (float)(rng() % 2001) / 1000.f - 1.f;
        for (auto& v : w) v = (float)(rng() % 1001) / 1000.f;
        for (auto& v : sh) v = (float)(rng() % 2001) / 1000.f - 1.f;
        Dev<float> dp(parts), dw(w), dsh(sh), y1(ne), y2(ne);
        CHECK(kg::moe_combine(dp, dw, dsh, y1, ne, k, 1) == cudaSuccess && kg::moe_combine(dp, dw, nullptr, y2, ne, k, 0) == cudaSuccess && sync_ok());
        const auto A = y1.get(), B = y2.get(); bool ok = true;
        for (int j = 0; j < ne; ++j) {
            double a = 0; for (int e = 0; e < k; ++e) a += (double)w[e] * parts[(size_t)e * ne + j];
            ok = ok && std::fabs(B[j] - a) < 1e-4 && std::fabs(A[j] - (a + sh[j])) < 1e-4;                // shared added plain, never weighted
        }
        CHECK(ok);
    }
    Dev<float> dp(100 * 8), dw(100), dsh(8), y(8);
    CHECK(kg::moe_combine(dp, dw, dsh, y, 8, 65, 1) != cudaSuccess);                                      // longer lists are refused, not truncated
    CHECK(kg::moe_combine(dp, dw, dsh, y, 8, 0, 1) != cudaSuccess);
    CHECK(kg::moe_combine(dp, dw, nullptr, y, 8, 3, 1) != cudaSuccess);
    CHECK(kg::moe_combine(dp, dw, dsh, y, 8, 64, 1) == cudaSuccess && sync_ok());
}

// ------------------------------------------------------------------------------------------------ sampler
static std::vector<int32_t> sample(const std::vector<float>& logits, int rows, int V, const std::vector<int32_t>& hist, int stride, const std::vector<int32_t>& hlen,
                                   const kern::SamplerParams& p, uint64_t seed, uint64_t counter) {
    Dev<float> dl(logits), ws((size_t)rows * V); Dev<int32_t> dh(hist.empty() ? std::vector<int32_t>(1, 0) : hist), dhl(hlen.empty() ? std::vector<int32_t>(1, 0) : hlen), out(rows);
    CHECK(kern::sample_tokens(dl, rows, V, hist.empty() ? nullptr : (const int32_t*)dh, stride, hlen.empty() ? nullptr : (const int32_t*)dhl, p, seed, counter, ws, out, 0) == cudaSuccess && sync_ok());
    return out.get();
}
static void test_sampler() {
    std::mt19937 rng(17);
    kern::SamplerParams greedy; greedy.temperature = 0.f; greedy.last_n = 0; greedy.repeat_penalty = 1.f;
    {   // greedy: ties -> the lowest index, over vocabularies spanning many 256-thread strides
        for (int V : {1, 7, 255, 256, 257, 1000, 5000, 32768}) {
            std::vector<float> l(V); for (auto& x : l) x = (float)(int)(rng() % 3);
            int want = 0; for (int i = 1; i < V; ++i) if (l[i] > l[want]) want = i;
            CHECK(sample(l, 1, V, {}, 0, {}, greedy, 1, 0)[0] == want);
        }
        std::vector<float> l(4096, 0.f); l[4000] = 3.f; l[17] = 3.f; l[2048] = 3.f;
        CHECK(sample(l, 1, 4096, {}, 0, {}, greedy, 1, 0)[0] == 17);
        const int rows = 6, V = 777; std::vector<float> m((size_t)rows * V); for (auto& x : m) x = (float)(int)(rng() % 4);
        const auto g = sample(m, rows, V, {}, 0, {}, greedy, 1, 0);
        for (int r = 0; r < rows; ++r) { int want = 0; for (int i = 1; i < V; ++i) if (m[(size_t)r * V + i] > m[(size_t)r * V + want]) want = i; CHECK(g[r] == want); }
    }
    {   // penalties: only tokens with count > 0, repeat multiplies non-positive / divides positive, presence is a boolean
        const int V = 8; std::vector<float> l = {5.f, 4.f, 3.f, -1.f, -2.f, 0.f, 0.f, 0.f};
        kern::SamplerParams p = greedy; p.last_n = 16;
        std::vector<int32_t> h(16, 0); h[0] = 0; h[1] = 0; h[2] = 0;                                      // token 0 seen 3 times
        std::vector<int32_t> len = {3};
        p.repeat_penalty = 1.f; p.presence_penalty = 0.f; p.frequency_penalty = 0.f;
        CHECK(sample(l, 1, V, h, 16, len, p, 1, 0)[0] == 0);
        p.presence_penalty = 1.5f;                                                                        // 5 - 1.5 = 3.5 < 4: token 1 wins (the count is NOT multiplied in)
        CHECK(sample(l, 1, V, h, 16, len, p, 1, 0)[0] == 1);
        p.presence_penalty = 0.4f; p.frequency_penalty = 0.f;                                             // 5 - 0.4 = 4.6 > 4: still token 0
        CHECK(sample(l, 1, V, h, 16, len, p, 1, 0)[0] == 0);
        p.presence_penalty = 0.f; p.frequency_penalty = 0.4f;                                             // 5 - 3*0.4 = 3.8 < 4: token 1 wins
        CHECK(sample(l, 1, V, h, 16, len, p, 1, 0)[0] == 1);
        p.frequency_penalty = 0.f; p.repeat_penalty = 2.f;                                                // 5 / 2 = 2.5 < 4
        CHECK(sample(l, 1, V, h, 16, len, p, 1, 0)[0] == 1);
        std::vector<float> neg = {-1.f, -1.5f, -9.f, -9.f, -9.f, -9.f, -9.f, -9.f}; std::vector<int32_t> h2(16, 0); std::vector<int32_t> len2 = {1};
        CHECK(sample(neg, 1, V, h2, 16, len2, p, 1, 0)[0] == 1);                                          // -1 * 2 = -2 < -1.5: for a non-positive logit the multiply makes it worse
        p.repeat_penalty = 1.f;
        // the window is the TAIL of the history: an old occurrence outside last_n is not penalised
        std::vector<int32_t> hh(8, 3); hh[6] = 0; hh[7] = 1;                                             // history 3 3 3 3 3 3 0 1
        p.last_n = 2; p.presence_penalty = 10.f;
        std::vector<float> l2 = {5.f, 4.f, 3.f, 9.f, 0.f, 0.f, 0.f, 0.f};
        CHECK(sample(l2, 1, V, hh, 8, {8}, p, 1, 0)[0] == 3);                                             // 3 is outside the 2-token tail: untouched, best
        p.last_n = 8;
        CHECK(sample(l2, 1, V, hh, 8, {8}, p, 1, 0)[0] == 2);                                             // whole history: 3, 0, 1 penalised, 2 wins
        p.last_n = 0;
        CHECK(sample(l2, 1, V, hh, 8, {8}, p, 1, 0)[0] == 3);                                             // last_n = 0: nothing penalised
    }
    {   // sampled path: top_k = 1 is greedy; draws stay inside the top-k; top_p keeps the head; deterministic per (seed, counter)
        const int V = 600; std::vector<float> l(V); for (auto& x : l) x = (float)(rng() % 1000) / 100.f;
        int best = 0; for (int i = 1; i < V; ++i) if (l[i] > l[best]) best = i;
        kern::SamplerParams p; p.temperature = 1.f; p.top_k = 1; p.top_p = 1.f; p.min_keep = 1; p.last_n = 0; p.repeat_penalty = 1.f;
        for (uint64_t c = 0; c < 20; ++c) CHECK(sample(l, 1, V, {}, 0, {}, p, 99, c)[0] == best);
        p.top_k = 5;
        std::vector<int> order(V); for (int i = 0; i < V; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return l[a] > l[b]; });
        std::set<int> top(order.begin(), order.begin() + 5);
        std::set<int> seen;
        for (uint64_t c = 0; c < 200; ++c) { const int t = sample(l, 1, V, {}, 0, {}, p, 5, c)[0]; CHECK(top.count(t) == 1); seen.insert(t); }
        CHECK(seen.size() >= 2);
        CHECK(sample(l, 1, V, {}, 0, {}, p, 5, 7) == sample(l, 1, V, {}, 0, {}, p, 5, 7));
        p.top_k = 64; p.top_p = 1e-6f; p.min_keep = 1;                                                    // a vanishing nucleus keeps only the head
        for (uint64_t c = 0; c < 20; ++c) CHECK(sample(l, 1, V, {}, 0, {}, p, 5, c)[0] == best);
        // two equal best logits and a tiny rest: each is drawn about half the time (the uniform is in [0,1))
        std::vector<float> e(50, -30.f); e[3] = 2.f; e[40] = 2.f;
        kern::SamplerParams q; q.temperature = 1.f; q.top_k = 2; q.top_p = 1.f; q.min_keep = 1; q.last_n = 0;
        int c3 = 0, N = 400; for (int c = 0; c < N; ++c) { const int t = sample(e, 1, 50, {}, 0, {}, q, 123, (uint64_t)c)[0]; CHECK(t == 3 || t == 40); c3 += (t == 3); }
        CHECK(c3 > N * 30 / 100 && c3 < N * 70 / 100);
        // top_k above the capacity is clamped to 64: survivors never exceed 64 distinct tokens
        q.top_k = 1000; std::set<int> s64; std::vector<float> flat(300, 1.f);
        for (int c = 0; c < 600; ++c) s64.insert(sample(flat, 1, 300, {}, 0, {}, q, 4, (uint64_t)c)[0]);
        CHECK(s64.size() <= 64);
        for (int t : s64) CHECK(t < 64);                                                                  // ties keep the lowest indices
    }
    {   // invalid arguments
        Dev<float> dl(16), ws(16); Dev<int32_t> out(1);
        kern::SamplerParams bad = greedy; bad.repeat_penalty = 0.f;
        CHECK(kern::sample_tokens(dl, 1, 16, nullptr, 0, nullptr, bad, 1, 0, ws, out, 0) != cudaSuccess);
        CHECK(kern::sample_tokens(dl, 0, 16, nullptr, 0, nullptr, greedy, 1, 0, ws, out, 0) != cudaSuccess);
    }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no CUDA device\n"); return 2; }
    CU(cudaSetDevice(0));
    test_router(); test_shared_expert(); test_combine(); test_sampler();
    std::printf("%d device checks, %d failed\n", g_n, g_fail);
    if (g_fail) { std::printf("UCBADF TESTS FAILED\n"); return 1; }
    std::printf("UCBADF TESTS PASSED\n");
    return 0;
}
