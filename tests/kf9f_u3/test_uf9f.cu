// test_uf9f.cu - invariants of NativeRouterTop10, SamplerChain, ExpertCacheResidency and ExpertSourceDispatch
// (lean/uf9f/*.lean) checked against the real GPU kernels and host models. Exits non-zero on any violation.
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "baloon/core_expert_cache.hpp"
#include "baloon/core_expert_source.hpp"
#include "baloon/kern_cuda_misc.hpp"
#include "baloon/kg_kernels.hpp"

namespace kg = baloon::kg;
namespace kern = baloon::kern;
namespace core = baloon::core;
static long checks = 0, fails = 0;
static void check(bool ok, const char* what) { ++checks; if (!ok) { ++fails; std::printf("  VIOLATED: %s\n", what); } }
template <class T> static T* dup(const std::vector<T>& v) { T* d; cudaMalloc(&d, v.size() * sizeof(T)); cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }

// ------------------------------------------------------------------------------------------------ router
static void router_run(const std::vector<float>& lg, int nt, int ne, int k, std::vector<int32_t>& ids, std::vector<float>& w, bool* ok = nullptr) {
    float* dl = dup(lg); int32_t* di; float* dw;
    cudaMalloc(&di, nt * k * 4); cudaMalloc(&dw, nt * k * 4);
    cudaError_t e = kg::native_router_top10(dl, nt, ne, k, di, dw);
    cudaError_t s = cudaDeviceSynchronize();
    if (ok) *ok = (e == cudaSuccess && s == cudaSuccess);
    ids.assign(nt * k, -1); w.assign(nt * k, -1.f);
    cudaMemcpy(ids.data(), di, nt * k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(w.data(), dw, nt * k * 4, cudaMemcpyDeviceToHost);
    cudaFree(dl); cudaFree(di); cudaFree(dw);
}

static void test_router() {
    std::printf("[router]\n");
    const int NE = 512, K = 10, NT = 5;
    std::mt19937 rng(7);
    // distinct logits (multiples of 0.01): order is strict, so ids must equal the sorted-by-value order.
    std::vector<float> lg(NT * NE);
    for (int t = 0; t < NT; ++t) {
        std::vector<int> perm(NE); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
        for (int e = 0; e < NE; ++e) lg[t * NE + e] = perm[e] * 0.01f - 2.f;
    }
    std::vector<int32_t> ids; std::vector<float> w; bool ok = false;
    router_run(lg, NT, NE, K, ids, w, &ok);
    check(ok, "router launch succeeds");
    for (int t = 0; t < NT; ++t) {
        std::vector<int> ord(NE); std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { const float x = lg[t * NE + a], y = lg[t * NE + b]; return x > y || (x == y && a < b); });
        double mx = lg[t * NE + ord[0]], sum = 0;
        for (int e = 0; e < NE; ++e) sum += std::exp((double)lg[t * NE + e] - mx);
        double ssel = 0; for (int i = 0; i < K; ++i) ssel += std::exp((double)lg[t * NE + ord[i]] - mx) / sum;
        double wsum = 0; bool idok = true, wok = true, unique = true;
        for (int i = 0; i < K; ++i) {
            idok &= ids[t * K + i] == ord[i];
            const double ref = std::exp((double)lg[t * NE + ord[i]] - mx) / sum / ssel;
            wok &= std::fabs(w[t * K + i] - ref) < 2e-4;
            wsum += w[t * K + i];
            for (int j = 0; j < i; ++j) unique &= ids[t * K + i] != ids[t * K + j];
        }
        check(idok, "router: ids are the ten largest, in descending order");
        check(wok, "router: weights = softmax renormalised over the ten selected");
        check(std::fabs(wsum - 1.0) < 1e-4, "router: selected weights sum to 1");
        check(unique, "router: ids are distinct (single writer per expert)");
        bool desc = true; for (int i = 1; i < K; ++i) desc &= w[t * K + i - 1] >= w[t * K + i];
        check(desc, "router: weights non-increasing");
    }
    // all equal -> ids 0..9, weights 0.1 (tie rule: smaller expert id wins)
    std::vector<float> eq(NE, 0.5f);
    router_run(eq, 1, NE, K, ids, w);
    bool tie = true; for (int i = 0; i < K; ++i) tie &= ids[i] == i && std::fabs(w[i] - 0.1f) < 1e-5f;
    check(tie, "router: all-equal logits pick ids 0..9");
    // planted tie across lanes and slots: experts 450, 12, 300, 7 share the max -> order 7, 12, 300, 450
    std::vector<float> pt(NE, 0.f);
    for (int e : {450, 12, 300, 7}) pt[e] = 5.f;
    for (int e = 0; e < NE; ++e) if (pt[e] == 0.f) pt[e] = -1.f - e * 0.001f;
    router_run(pt, 1, NE, K, ids, w);
    check(ids[0] == 7 && ids[1] == 12 && ids[2] == 300 && ids[3] == 450, "router: ties resolve to the smaller expert id, across lanes/slots");
    check(ids[4] == 0 && ids[5] == 1, "router: after the planted tie the next-largest follow");
    // expert id = lane + 32*slot roundtrip: a single hot expert is found wherever it sits
    bool all_found = true;
    for (int e = 0; e < NE; e += 37) { std::vector<float> one(NE, 0.f); one[e] = 9.f; router_run(one, 1, NE, K, ids, w); all_found &= ids[0] == e; }
    check(all_found, "router: single hot expert found at any position");
    // span validation: null, misaligned, overlapping rejected; end-to-end adjacent accepted
    float* dl; cudaMalloc(&dl, NE * 4 + 16); cudaMemset(dl, 0, NE * 4 + 16);
    char* buf; cudaMalloc(&buf, 256); cudaMemset(buf, 0, 256);
    int32_t* i0 = (int32_t*)buf; float* w0 = (float*)(buf + 40);
    check(kg::native_router_top10(dl, 1, NE, K, i0, w0) == cudaSuccess, "router: adjacent (touching) spans accepted");
    cudaDeviceSynchronize();
    check(kg::native_router_top10(dl, 1, NE, K, i0, (float*)(buf + 36)) != cudaSuccess, "router: overlapping ids/weights rejected");
    check(kg::native_router_top10(dl, 1, NE, K, i0, (float*)(buf + 42)) != cudaSuccess, "router: misaligned weights rejected");
    check(kg::native_router_top10((const float*)((char*)dl + 2), 1, NE, K, i0, w0) != cudaSuccess, "router: misaligned logits rejected");
    check(kg::native_router_top10(nullptr, 1, NE, K, i0, w0) != cudaSuccess, "router: null logits rejected");
    check(kg::native_router_top10(dl, 1, NE, K, nullptr, w0) != cudaSuccess, "router: null ids rejected");
    check(kg::native_router_top10(dl, 1, NE, K, (int32_t*)((char*)dl + 8), w0) != cudaSuccess, "router: ids inside logits rejected");
    cudaGetLastError();
    cudaFree(dl); cudaFree(buf);
    bool en = kg::native_router_enabled();
    kg::native_router_set_enabled(false); check(!kg::native_router_enabled(), "router: set_enabled(false)");
    kg::native_router_set_enabled(true); check(kg::native_router_enabled(), "router: set_enabled(true)");
    kg::native_router_set_enabled(en);
}

// ------------------------------------------------------------------------------------------------ sampler
struct SRun { std::vector<int32_t> tok; bool ok; };
static SRun sample(const std::vector<float>& lg, int rows, int V, const std::vector<int32_t>& hist, int stride, const std::vector<int32_t>& hlen,
                   const kern::SamplerParams& p, uint64_t seed, uint64_t ctr) {
    float* dl = dup(lg); int32_t* dh = dup(hist.empty() ? std::vector<int32_t>(1, 0) : hist); int32_t* dn = dup(hlen);
    float* ws; int32_t* out; cudaMalloc(&ws, (size_t)rows * V * 4); cudaMalloc(&out, rows * 4);
    cudaError_t e = kern::sample_tokens(dl, rows, V, dh, stride, dn, p, seed, ctr, ws, out, 0);
    cudaError_t s = cudaDeviceSynchronize();
    SRun r; r.ok = e == cudaSuccess && s == cudaSuccess; r.tok.assign(rows, -2);
    cudaMemcpy(r.tok.data(), out, rows * 4, cudaMemcpyDeviceToHost);
    cudaFree(dl); cudaFree(dh); cudaFree(dn); cudaFree(ws); cudaFree(out);
    return r;
}

static void test_sampler() {
    std::printf("[sampler]\n");
    const int V = 1000, R = 4;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 2.f);
    std::vector<float> lg(R * V);
    for (auto& x : lg) x = nd(rng);
    kern::SamplerParams g; g.temperature = 0.f; g.repeat_penalty = 1.f; g.presence_penalty = 0.f; g.frequency_penalty = 0.f;
    std::vector<int32_t> nohist(R * 4, 0), nolen(R, 0);
    SRun r = sample(lg, R, V, nohist, 4, nolen, g, 1, 0);
    check(r.ok, "greedy launch");
    bool am = true;
    for (int t = 0; t < R; ++t) am &= r.tok[t] == (int)(std::max_element(lg.begin() + t * V, lg.begin() + (t + 1) * V) - (lg.begin() + t * V));
    check(am, "greedy (temperature<=0) is the argmax - temperature is not applied (inv_t=0 would zero every logit)");
    g.temperature = -1.f;
    r = sample(lg, R, V, nohist, 4, nolen, g, 1, 0);
    bool am2 = true; for (int t = 0; t < R; ++t) am2 &= r.tok[t] == (int)(std::max_element(lg.begin() + t * V, lg.begin() + (t + 1) * V) - (lg.begin() + t * V));
    check(am2, "negative temperature is also greedy");
    // ties go to the lowest index (ties planted across threads of the strided scan)
    std::vector<float> tl(R * V, -3.f);
    for (int t = 0; t < R; ++t) { tl[t * V + 997] = 4.f; tl[t * V + 513] = 4.f; tl[t * V + 255] = 4.f; tl[t * V + 770] = 4.f; }
    g.temperature = 0.f;
    r = sample(tl, R, V, nohist, 4, nolen, g, 1, 0);
    bool low = true; for (int t = 0; t < R; ++t) low &= r.tok[t] == 255;
    check(low, "greedy ties resolve to the lowest index");
    std::vector<float> flat(R * V, 1.f);
    r = sample(flat, R, V, nohist, 4, nolen, g, 1, 0);
    bool z = true; for (int t = 0; t < R; ++t) z &= r.tok[t] == 0;
    check(z, "all-equal row answers index 0");
    // penalties: top 5.0 (token 10), runner-up 4.0 (token 20), rest low
    auto two = [&](float a, float b) { std::vector<float> l(V, -10.f); l[10] = a; l[20] = b; return l; };
    kern::SamplerParams p; p.temperature = 0.f; p.repeat_penalty = 2.f; p.presence_penalty = 0.f; p.frequency_penalty = 0.f; p.last_n = 64;
    std::vector<int32_t> h(8, 0); std::vector<int32_t> hl(1, 1); h[0] = 10;
    r = sample(two(5.f, 4.f), 1, V, h, 8, hl, p, 1, 0);
    check(r.tok[0] == 20, "repeat penalty divides a positive logit (5/2 < 4)");
    r = sample(two(-1.f, -1.5f), 1, V, h, 8, hl, p, 1, 0);
    check(r.tok[0] == 20, "repeat penalty multiplies a negative logit (-1*2 < -1.5): naive division would invert");
    std::vector<int32_t> none(1, 0);
    r = sample(two(5.f, 4.f), 1, V, h, 8, none, p, 1, 0);
    check(r.tok[0] == 10, "count 0 -> no penalty");
    // presence is boolean, frequency scales with count
    p.repeat_penalty = 1.f; p.presence_penalty = 1.f; p.frequency_penalty = 0.f;
    std::vector<int32_t> h1(8, 0), h3(8, 0); h1[0] = 10; h3[0] = h3[1] = h3[2] = 10;
    std::vector<int32_t> l1(1, 1), l3(1, 3);
    SRun a = sample(two(5.f, 4.5f), 1, V, h1, 8, l1, p, 1, 0), b = sample(two(5.f, 4.5f), 1, V, h3, 8, l3, p, 1, 0);
    check(a.tok[0] == 20 && b.tok[0] == 20, "presence penalty applied once (same for count 1 and 3)");
    p.presence_penalty = 0.f; p.frequency_penalty = 0.4f;
    a = sample(two(5.f, 4.5f), 1, V, h1, 8, l1, p, 1, 0); b = sample(two(5.f, 4.5f), 1, V, h3, 8, l3, p, 1, 0);
    check(a.tok[0] == 10 && b.tok[0] == 20, "frequency penalty scales with the count (1*0.4 keeps top, 3*0.4 drops it)");
    // history window is the TAIL: [10, 1, 2, 3], last_n=3 excludes 10; last_n=4 includes it
    p.repeat_penalty = 2.f; p.frequency_penalty = 0.f;
    std::vector<int32_t> ht = {10, 1, 2, 3, 0, 0, 0, 0}; std::vector<int32_t> l4(1, 4);
    p.last_n = 3; a = sample(two(5.f, 4.f), 1, V, ht, 8, l4, p, 1, 0);
    p.last_n = 4; b = sample(two(5.f, 4.f), 1, V, ht, 8, l4, p, 1, 0);
    check(a.tok[0] == 10 && b.tok[0] == 20, "penalty window is the last min(last_n, hist_len) entries");
    p.last_n = 1000; b = sample(two(5.f, 4.f), 1, V, ht, 8, l4, p, 1, 0);
    check(b.tok[0] == 20, "last_n larger than history is clamped to the history");
    p.last_n = 0; a = sample(two(5.f, 4.f), 1, V, ht, 8, l4, p, 1, 0);
    check(a.tok[0] == 10, "last_n = 0 disables penalties");
    // sampled path
    kern::SamplerParams s; s.temperature = 1.f; s.top_k = 1; s.top_p = 1.f; s.min_keep = 1; s.repeat_penalty = 1.f; s.last_n = 0;
    r = sample(lg, R, V, nohist, 4, nolen, s, 3, 5);
    bool k1 = true; for (int t = 0; t < R; ++t) k1 &= r.tok[t] == (int)(std::max_element(lg.begin() + t * V, lg.begin() + (t + 1) * V) - (lg.begin() + t * V));
    check(r.ok && k1, "top_k=1 sampling equals the argmax");
    s.top_k = 5000;  // clamped to 64
    SRun d1 = sample(lg, R, V, nohist, 4, nolen, s, 99, 17), d2 = sample(lg, R, V, nohist, 4, nolen, s, 99, 17);
    check(d1.ok && d1.tok == d2.tok, "same (seed, counter) -> same draw (counter-based, deterministic)");
    bool in64 = true;
    for (int t = 0; t < R; ++t) {
        std::vector<int> ord(V); std::iota(ord.begin(), ord.end(), 0);
        std::partial_sort(ord.begin(), ord.begin() + 64, ord.end(), [&](int x, int y) { return lg[t * V + x] > lg[t * V + y]; });
        in64 &= std::find(ord.begin(), ord.begin() + 64, d1.tok[t]) != ord.begin() + 64;
    }
    check(in64, "top_k clamped to 64: the draw is among the 64 largest logits");
    // uniform draws vary over counters, always in range
    std::vector<float> fl(V, 0.f); kern::SamplerParams hi; hi.temperature = 2.f; hi.top_k = 64; hi.top_p = 1.f; hi.min_keep = 1; hi.repeat_penalty = 1.f; hi.last_n = 0;
    std::vector<int> seen; bool inr = true;
    for (int c = 0; c < 64; ++c) { SRun q = sample(lg, 1, V, nohist, 4, nolen, hi, 1234, c); inr &= q.ok && q.tok[0] >= 0 && q.tok[0] < V; seen.push_back(q.tok[0]); }
    std::sort(seen.begin(), seen.end());
    check(inr, "sampled tokens are valid vocabulary indices");
    check(std::unique(seen.begin(), seen.end()) - seen.begin() > 3, "uniform draws vary across counters (Philox stream is not constant)");
    // top_p: a tiny nucleus with min_keep=1 keeps only the top token
    kern::SamplerParams tp = hi; tp.top_p = 1e-4f; tp.temperature = 1.f;
    bool tpok = true;
    for (int c = 0; c < 16; ++c) { SRun q = sample(lg, 1, V, nohist, 4, nolen, tp, 5, c); tpok &= q.tok[0] == (int)(std::max_element(lg.begin(), lg.begin() + V) - lg.begin()); }
    check(tpok, "top_p cut keeps at least min_keep survivors (and no more than the nucleus)");
}

// ------------------------------------------------------------------------------------------------ expert cache
static void test_cache() {
    std::printf("[expert cache]\n");
    // per-layer ranges tile 0..slots exactly, for many geometries
    bool tile = true, lohi = true;
    for (int64_t nl = 1; nl <= 12; ++nl) for (int64_t slots = nl; slots <= 200; slots += 7) {
        core::ExpertCache c; if (!c.open(nl, 16, slots, 64, true)) { tile = false; continue; }
        int64_t prev = 0;
        for (int64_t l = 0; l < nl; ++l) {
            auto r = c.layer_slot_range(l);
            lohi &= r.first <= r.second;
            tile &= r.first == prev; prev = r.second;
            tile &= r.first == l * (slots / nl);
            if (l + 1 < nl) tile &= r.second == (l + 1) * (slots / nl);
        }
        tile &= prev == slots;
        auto o = c.layer_slot_range(nl); lohi &= o.first == 0 && o.second == 0;
    }
    check(tile, "per-layer ranges tile [0, slots) exactly; last layer takes the remainder");
    check(lohi, "lo <= hi; out-of-range layer -> empty range");
    // admission: never evicts, never exceeds the per-layer bound
    core::ExpertCache c; check(c.open(6, 16, 26, 64, true), "open per-layer cache");
    bool bounded = true, noevict = true; int64_t total = 0;
    for (int64_t l = 0; l < 6; ++l) {
        auto rg = c.layer_slot_range(l); std::vector<int32_t> got;
        for (int64_t e = 0; e < 16; ++e) {
            int32_t s = c.admit(l, e);
            if (s != core::kNotResident) { bounded &= s >= rg.first && s < rg.second; got.push_back(s); ++total; }
        }
        bounded &= (int64_t)got.size() == rg.second - rg.first;
        for (int64_t e = 0; e < 16; ++e) { int32_t s = c.slot_of(l, e); if (s != core::kNotResident && std::find(got.begin(), got.end(), s) == got.end()) noevict = false; }
    }
    check(bounded, "admissions stay within the layer's range and fill it exactly");
    check(total == 26, "per-layer quotas sum to the arena: all slots usable, none beyond");
    check(noevict, "admit never evicts: earlier residents stay resident");
    int32_t before = c.slot_of(0, 0); c.admit(0, 15); check(c.slot_of(0, 0) == before, "refused admission leaves the table unchanged");
    check(c.admit(0, 3) == c.slot_of(0, 3) || c.slot_of(0, 3) == core::kNotResident, "re-admission is consistent with the table");
    check(c.slot_of(-1, 0) == core::kNotResident && c.slot_of(6, 0) == core::kNotResident && c.slot_of(0, 16) == core::kNotResident && c.slot_of(0, -1) == core::kNotResident, "slot_of out of range -> kNotResident (-1)");
    check(c.admit(6, 0) == core::kNotResident && c.admit(0, 16) == core::kNotResident, "admit out of range -> kNotResident");
    check(core::kNotResident == -1, "kNotResident == -1");
    // global mode: one counter <= slots
    core::ExpertCache g; g.open(48, 512, 256, 32, false);
    int64_t got = 0; bool onlyEarly = true;
    for (int64_t l = 0; l < 48; ++l) for (int64_t e = 0; e < 10; ++e) { if (g.admit(l, e) != core::kNotResident) { ++got; onlyEarly &= l <= 25; } }
    check(got == 256 && onlyEarly, "global arrival-order admission: 256 slots consumed inside 26 layers of 10 experts");
    // alignment of sized slots
    core::ExpertCache sz; check(sz.open_sized(2, 4, {100, 256, 257, 1}, true), "open_sized");
    bool al = true; bool mono = true; const uint8_t* b0 = sz.device_slot(0);
    for (int64_t i = 0; i < 4; ++i) { const uint8_t* p = sz.device_slot(i); al &= p != nullptr && ((p - b0) % 256) == 0; if (i) mono &= p - sz.device_slot(i - 1) >= (int64_t)sz.slot_bytes(i - 1); }
    check(al && mono, "sized slots start 256-byte aligned and never overlap");
    check(sz.device_slot(4) == nullptr && sz.device_slot(-1) == nullptr, "device_slot out of range -> nullptr");
    bool a256 = true; for (int64_t n = 0; n < 3000; ++n) { int64_t a = core::cache_align256(n); a256 &= a >= n && a % 256 == 0 && a < n + 256; }
    check(a256, "align256: >= n, multiple of 256, < n + 256");
    // fills copy at most the blob and verify
    core::ExpertCache f; f.open(2, 4, 8, 64, true);
    std::vector<uint8_t> src(200); for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 1);
    int32_t s0 = f.admit(0, 1);
    check(s0 != core::kNotResident && f.fill_slot_blocking(s0, src.data(), 64) && f.verify_slot(s0, src.data(), 64), "fill_slot_blocking + verify_slot");
    int32_t s1 = f.admit(1, 2);
    f.fill_slot(s1, src.data(), 200);   // oversized request is bounded by the blob size
    f.sync();
    check(f.verify_slot(s1, src.data(), 64) && f.pending_fills() == 0, "fill_slot bounded by blob size; sync clears pending");
    check(f.arena_bytes() == 8 * 64, "uniform arena = slots * blob");
    f.close(); check(!f.is_open(), "close");
    // profile validation
    std::vector<std::pair<int32_t, int32_t>> pr = {{0, 1}, {3, 7}, {2, 0}}, outp;
    auto enc = core::encode_expert_profile(4, 8, pr);
    check(core::parse_expert_profile(enc.data(), enc.size(), 4, 8, 3, outp) && outp == pr, "profile roundtrip accepted");
    check(!core::parse_expert_profile(enc.data(), enc.size(), 5, 8, 3, outp), "profile with wrong n_layers rejected");
    check(!core::parse_expert_profile(enc.data(), enc.size(), 4, 9, 3, outp), "profile with wrong n_expert rejected");
    check(!core::parse_expert_profile(enc.data(), enc.size(), 4, 8, 2, outp), "profile with n_ranked > slots rejected");
    check(!core::parse_expert_profile(enc.data(), enc.size() - 3, 4, 8, 3, outp), "truncated profile rejected");
    auto bad = core::encode_expert_profile(4, 8, {{0, 1}, {4, 0}}); check(!core::parse_expert_profile(bad.data(), bad.size(), 4, 8, 3, outp), "pair with layer out of range rejected");
    bad = core::encode_expert_profile(4, 8, {{0, 8}}); check(!core::parse_expert_profile(bad.data(), bad.size(), 4, 8, 3, outp), "pair with expert out of range rejected");
    char tmpl[] = "/tmp/uf9f_profileXXXXXX"; int fd = mkstemp(tmpl);
    if (fd >= 0) { (void)!write(fd, enc.data(), enc.size()); close(fd); outp.clear(); check(core::read_expert_profile(tmpl, 4, 8, 3, outp) && outp == pr, "read_expert_profile from file"); unlink(tmpl); }
    check(!core::read_expert_profile("/nonexistent/uf9f.strp", 4, 8, 3, outp), "read_expert_profile missing file -> false");
    check(core::profile_header_ok(4, 8, 3, 3, 4, 8) && !core::profile_header_ok(4, 8, 4, 3, 4, 8), "headerOk: geometry and n_ranked <= want");
    // design-note constants (Lean: slots4105_fits_*)
    check((uint64_t)4105 * 1382400 <= (uint64_t)5413 * 1048576 && (uint64_t)4105 * 1382400 <= (uint64_t)6414 * 1048576, "4105-slot arena fits the quoted budgets");
}

// ------------------------------------------------------------------------------------------------ expert source / dispatch
struct Sink : core::GpuPlanSink { int calls = 0; size_t last_jobs = 0; void plan(int64_t, const core::ExpertDispatch& d) override { ++calls; last_jobs = d.jobs.size(); } };

static void test_source() {
    std::printf("[expert source]\n");
    const int64_t NL = 3, NE = 4, B = 16;
    std::vector<uint8_t> file(NL * NE * B); for (size_t i = 0; i < file.size(); ++i) file[i] = (uint8_t)(i / B * 5 + (i % B));
    core::ArenaExpertSource ar; check(ar.open_bytes(file.data(), file.size(), NL, NE, B), "arena open_bytes");
    bool addr = true; for (int64_t l = 0; l < NL; ++l) for (int64_t e = 0; e < NE; ++e) { const uint8_t* p = ar.blob(l, e); addr &= p && std::memcmp(p, &file[(l * NE + e) * B], B) == 0; }
    check(addr, "blob(l, e) is the (l*n_expert + e)-th blob");
    check(ar.blob(0, NE) == nullptr, "expert == n_expert rejected (would alias blob(1,0))");
    check(ar.blob(1, 0) != nullptr && ar.blob(NL, 0) == nullptr && ar.blob(-1, 0) == nullptr && ar.blob(0, -1) == nullptr, "layer / negative bounds checked");
    check(ar.arena_bytes() >= file.size() + B, "arena is at least one blob longer than the file");
    check(ar.file_bytes() == file.size(), "file_bytes = n_layers*n_expert*blob");
    uint64_t r0 = ar.reads(); ar.blob(0, 1); ar.blob(0, NE); check(ar.reads() == r0 + 1, "only accepted lookups bump reads()");
    check(ar.device_alias() != nullptr || !ar.pinned(), "device alias present whenever pinned");
    ar.close();
    char tmpl[] = "/tmp/uf9f_expertsXXXXXX"; int fd = mkstemp(tmpl);
    if (fd >= 0) {
        (void)!write(fd, file.data(), file.size()); close(fd);
        core::FileExpertSource fs; check(fs.open(tmpl, NL, NE, B), "file source open");
        bool fa = true; for (int64_t l = 0; l < NL; ++l) for (int64_t e = 0; e < NE; ++e) { const uint8_t* p = fs.blob(l, e); fa &= p && std::memcmp(p, &file[(l * NE + e) * B], B) == 0; }
        check(fa, "file source blobs match");
        check(fs.blob(0, NE) == nullptr && fs.blob(NL, 0) == nullptr, "file source rejects out-of-range");
        fs.close();
        core::FileExpertSource wrong; check(!wrong.open(tmpl, NL, NE + 1, B), "file of the wrong size rejected");
        core::ArenaExpertSource aw; check(aw.open(tmpl, NL, NE, B), "arena open from file"); aw.close();
        unlink(tmpl);
    }
    // hit/miss split
    std::vector<bool> hit = {true, false, false, true, false, false, true, false, false, false};
    const int K = (int)hit.size(); int nmiss = 0; for (bool h : hit) nmiss += !h;
    std::vector<int32_t> ex(K); for (int i = 0; i < K; ++i) ex[i] = i % 4;
    core::ExpertDispatch d;
    bool allpn = true;
    for (int pn = 0; pn <= 256; pn += 16) {
        check(core::expert_hit_run(hit, pn, d, ex.data()), "expert_hit_run accepts pcie_num in [0,256]");
        const size_t m = (size_t)nmiss * pn / 256;
        allpn &= d.gpu_pcie.size() == m && m <= (size_t)nmiss;
        allpn &= d.gpu_hits.size() + d.jobs.size() + d.gpu_pcie.size() == (size_t)K;     // every entry is computed exactly once
        allpn &= d.jobs.size() == (size_t)nmiss - m;
        std::vector<int> seen(K, 0); for (int32_t i : d.gpu_hits) ++seen[i]; for (auto& j : d.jobs) ++seen[j.out_index]; for (auto& j : d.gpu_pcie) ++seen[j.out_index];
        for (int i = 0; i < K; ++i) allpn &= seen[i] == 1;
        // the PCIe jobs are the LAST m misses in routing order
        if (!d.gpu_pcie.empty() && !d.jobs.empty()) allpn &= d.jobs.back().out_index < d.gpu_pcie.front().out_index;
        for (auto& j : d.jobs) allpn &= !hit[j.out_index];
        for (int32_t i : d.gpu_hits) allpn &= hit[i];
    }
    check(allpn, "split: hits + CPU jobs + PCIe = k, each router entry exactly once, pcie share = nmiss*pn/256 (last misses)");
    check(!core::expert_hit_run(hit, 257, d) && !core::expert_hit_run(hit, -1, d), "pcie_num outside [0,256] rejected");
    check(core::pcie_share_of(7, 128) == 3 && core::pcie_share_of(7, 256) == 7 && core::pcie_share_of(7, 0) == 0, "pcie_share_of examples");
    bool share = true; for (uint64_t n = 0; n <= 40; ++n) for (uint64_t pn = 0; pn <= 256; ++pn) share &= core::pcie_share_of(n, pn) <= n; check(share, "pcie share never exceeds the misses");
    // row zero-padding: cpu + gpu rows partition the answer
    // pool dispatch resolves blobs, calls the sink once
    core::ArenaExpertSource a2; a2.open_bytes(file.data(), file.size(), NL, NE, B);
    Sink sink; core::ExpertDispatch d2;
    check(core::expert_pool_dispatch(a2, &sink, 1, ex.data(), hit, 128, d2) && sink.calls == 1, "expert_pool_dispatch ok, sink planned once");
    bool resolved = true; for (auto& j : d2.jobs) resolved &= j.blob == a2.blob(1, j.expert); for (auto& j : d2.gpu_pcie) resolved &= j.blob == a2.blob(1, j.expert);
    check(resolved && !d2.failed, "dispatch resolves every missed blob");
    std::vector<int32_t> badex = ex; badex[1] = (int32_t)NE;   // first miss, out of range
    core::ExpertDispatch d3;
    check(!core::expert_pool_dispatch(a2, &sink, 1, badex.data(), hit, 0, d3) && d3.failed && sink.calls == 1, "bad expert id -> failed, sink not called");
    check(!core::expert_pool_dispatch(a2, nullptr, 1, nullptr, hit, 0, d3) && d3.failed, "null experts with non-empty router -> failed");
    // multi: a failure latches - later tokens are not dispatched
    std::vector<std::vector<int32_t>> mex = {ex, badex, ex, ex};
    std::vector<std::vector<bool>> mh(4, hit);
    std::vector<core::ExpertDispatch> mo;
    check(!core::expert_pool_dispatch_multi(a2, nullptr, 0, mex, mh, 0, mo) && mo.size() == 2 && !mo[0].failed && mo[1].failed, "multi: failure latches, later tokens not dispatched");
    std::vector<std::vector<int32_t>> okex = {ex, ex, ex};
    std::vector<std::vector<bool>> okh(3, hit);
    check(core::expert_pool_dispatch_multi(a2, nullptr, 0, okex, okh, 64, mo) && mo.size() == 3, "multi: all tokens dispatched");
    check(!core::expert_pool_dispatch_multi(a2, nullptr, 0, okex, std::vector<std::vector<bool>>(2, hit), 0, mo), "multi: mismatched token counts rejected");
    // GGUF chunked load: regions tile the blob, 16 experts per chunk
    const int64_t up = 8, blob = 24, ne = 40;
    std::vector<uint8_t> gate(ne * up), upw(ne * up), down(ne * up), dst(ne * blob, 0xEE);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = (uint8_t)(i + 1); upw[i] = (uint8_t)(i + 100); down[i] = (uint8_t)(i + 200); }
    int64_t chunks = 0;
    check(core::load_experts_gguf(gate.data(), upw.data(), down.data(), ne, up, 2 * up, blob, dst.data(), &chunks) && chunks == 3, "load_experts_gguf: ceil(40/16) = 3 chunks");
    bool lg = true; for (int64_t e = 0; e < ne; ++e) { lg &= !std::memcmp(&dst[e * blob], &gate[e * up], up) && !std::memcmp(&dst[e * blob + up], &upw[e * up], up) && !std::memcmp(&dst[e * blob + 2 * up], &down[e * up], up); }
    check(lg, "load_experts_gguf: blob = [gate | up | down]");
    check(!core::load_experts_gguf(gate.data(), upw.data(), down.data(), ne, up, 3 * up, blob, dst.data()), "load_experts_gguf rejects down_off != 2*up_off");
    // artifact constants
    check(3ull * 640 * 2560 * 9 / 32 == 1382400ull && 48ull * 512 * 1382400 == 33973862400ull && 48ull * 10 * 1382400 == 663552000ull && 512ull * 1382400 == 707788800ull, "artifact geometry constants");
}

int main() {
    int n = 0; cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n < 1) { std::printf("no GPU device (%s)\n", cudaGetErrorString(e)); return 2; }
    cudaSetDevice(0);
    void* probe; if (cudaMalloc(&probe, 1 << 20) != cudaSuccess) { std::printf("cannot open GPU\n"); return 2; } cudaFree(probe);
    test_router(); test_sampler(); test_cache(); test_source();
    std::printf("uf9f: %ld checks, %ld violated\n", checks, fails);
    return fails ? 1 : 0;
}
