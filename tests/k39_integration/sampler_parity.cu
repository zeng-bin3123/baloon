// sampler_parity.cu — GPU sampler invariant tests: greedy argmax, repeat/frequency/presence
// penalties, and stochastic top-k/top-p sampling via sample_tokens (src/cu/sampler.cu).
// Sentence 1: run() allocates GPU logit/workspace/output buffers, calls sample_tokens, and
//   returns the selected token ids so each test can CHECK the result against a CPU reference.
// Sentence 2: reference_pick() is the CPU argmax reference for the greedy (temperature <= 0)
//   path: ties are broken by lowest index, matching the kernel's behaviour.
// Sentence 3: hist_len() extracts the valid history length for a given row from the hlen vector.
// Sentence 4: test_sampler_greedy() checks that temperature <= 0 selects the lowest-index
//   maximum-logit token, that all-(-inf) logits produce token 0, that NaN logits never win,
//   and that multiple rows are sampled independently.
// Sentence 5: test_sampler_penalties() checks repeat_penalty, frequency_penalty, and
//   presence_penalty against hand-computed expected results, and verifies that last_n
//   correctly controls the history window used by all three penalty types.
// Sentence 6: test_sampler_stochastic() checks that top_k=1 equals argmax, that top_k=5
//   always produces a token from the five best candidates, that the same (seed, counter)
//   gives the same token (determinism), that flat logits produce a roughly uniform distribution,
//   and that top_p with min_keep trims and widens the candidate set correctly.
// Sentence 7: test_sampler_bad_args() checks that sample_tokens rejects repeat_penalty = 0
//   and a null logits pointer, returning a non-cudaSuccess error code in both cases.
#include "../k14gpu_common.h"
#include "baloon/kern_cuda_misc.hpp"
#include <algorithm>
#include <numeric>

using namespace baloon::kern;

// Sentence 3: hist_len() extracts the valid history length for a given row from the hlen vector.
static int32_t hist_len(const std::vector<int32_t>& hlen, int row) {
    return (row < (int)hlen.size()) ? hlen[row] : 0;
}

// Sentence 1: run() allocates GPU logit/workspace/output buffers, calls sample_tokens, and
// returns the selected token ids so each test can CHECK the result against a CPU reference.
static std::vector<int32_t> run(
    const std::vector<float>& logits, int rows, int nv,
    const std::vector<int32_t>& hist, int hstride,
    const std::vector<int32_t>& hlen, const SamplerParams& p,
    uint64_t seed, uint64_t ctr)
{
    float*   dl  = dup(logits);
    float*   ws  = dalloc<float>((size_t)rows * nv);
    int32_t* dout = dalloc<int32_t>(rows);
    int32_t* dh  = hist.empty() ? nullptr : dup(hist);
    int32_t* dn  = hlen.empty() ? nullptr : dup(hlen);
    CHECK(sample_tokens(dl, rows, nv, dh, hstride, dn, p, seed, ctr, ws, dout, 0) == cudaSuccess);
    sync();
    auto o = fetch(dout, rows);
    cudaFree(dl); cudaFree(ws); cudaFree(dout);
    if (dh) cudaFree(dh);
    if (dn) cudaFree(dn);
    return o;
}

// Sentence 2: reference_pick() is the CPU argmax reference for the greedy (temperature <= 0)
// path: ties are broken by lowest index, matching the kernel's behaviour.
static int reference_pick(const std::vector<float>& logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// Sentence 4: test_sampler_greedy() checks that temperature <= 0 selects the lowest-index
// maximum-logit token, that all-(-inf) logits produce token 0, that NaN logits never win,
// and that multiple rows are sampled independently.
static void test_sampler_greedy() {
    SamplerParams g; g.temperature = 0.f;
    const int nv = 1000;
    // ties go to the lowest index, whatever the block layout
    std::vector<float> l(nv, 0.f); l[700] = 5.f; l[333] = 5.f; l[999] = 5.f;
    CHECK(run(l, 1, nv, {}, 0, {}, g, 1, 0)[0] == 333);
    CHECK(run(l, 1, nv, {}, 0, {}, g, 1, 0)[0] == reference_pick(l, nv));
    // all -inf: nothing finite, kernel falls back to token 0
    std::vector<float> ninf(nv, -INFINITY);
    CHECK(run(ninf, 1, nv, {}, 0, {}, g, 1, 0)[0] == 0);
    // NaN never wins: token 11 wins because nan[10] = NAN is ignored
    std::vector<float> nan(nv, 0.f); nan[10] = NAN; nan[11] = 0.5f;
    CHECK(run(nan, 1, nv, {}, 0, {}, g, 1, 0)[0] == 11);
    // several rows at once: each row is sampled independently (greedy per row)
    std::vector<float> two((size_t)2 * nv, 0.f); two[17] = 1.f; two[nv + 900] = 2.f;
    const auto o = run(two, 2, nv, {}, 0, {}, g, 1, 0);
    CHECK(o[0] == 17 && o[1] == 900);
}

// Sentence 5: test_sampler_penalties() checks repeat_penalty, frequency_penalty, and
// presence_penalty against hand-computed expected results, and verifies that last_n
// correctly controls the history window used by all three penalty types.
static void test_sampler_penalties() {
    SamplerParams g; g.temperature = 0.f;
    const int nv = 8;
    std::vector<float> l(nv, -5.f); l[0] = 1.0f; l[1] = 0.9f;
    // history = {0,0,0,7,7,7,7,7}, hlen = 3 (only the first 3 entries are valid: 0,0,0)
    std::vector<int32_t> hist = {0, 0, 0, 7, 7, 7, 7, 7}, hlen = {3};
    // repeat_penalty divides logit of previously-seen tokens: 1.0/2 = 0.5 < 0.9 → token 1
    SamplerParams p = g; p.repeat_penalty = 2.f;
    CHECK(run(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);
    // frequency_penalty subtracts count * penalty: 1.0 - 3*0.1 = 0.7 < 0.9 → token 1
    p = g; p.frequency_penalty = 0.1f;
    CHECK(run(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);
    // presence_penalty subtracts once per distinct token (bool, not count): 1.0 - 0.05 = 0.95 > 0.9 → token 0
    p = g; p.presence_penalty = 0.05f;
    CHECK(run(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 0);
    p = g; p.presence_penalty = 0.15f;
    CHECK(run(l, 1, nv, hist, 8, hlen, p, 1, 0)[0] == 1);
    // last_n controls the history window: last_n = 2 uses the TAIL [1,1] of valid=[0,0,1,1]
    std::vector<int32_t> h2 = {0, 0, 1, 1, 1, 1, 1, 1}, n2 = {4};  // valid: last 4 → 0 0 1 1
    std::vector<float> l2(nv, -5.f); l2[0] = 1.0f; l2[1] = 1.1f;
    p = g; p.repeat_penalty = 4.f; p.last_n = 2;   // window=[1,1]: only token 1 penalised → 0 wins
    CHECK(run(l2, 1, nv, h2, 8, n2, p, 1, 0)[0] == 0);
    p.last_n = 4;                                    // window=[0,0,1,1]: both penalised → 0.25 vs 0.275 → 1 wins
    CHECK(run(l2, 1, nv, h2, 8, n2, p, 1, 0)[0] == 1);
    // a negative logit is multiplied by repeat_penalty: the penalty never raises it
    std::vector<float> l3(nv, -5.f); l3[2] = -0.1f; l3[3] = -0.15f;
    std::vector<int32_t> h3 = {2, 0, 0, 0, 0, 0, 0, 0}, n3 = {1};
    p = g; p.repeat_penalty = 2.f;
    CHECK(run(l3, 1, nv, h3, 8, n3, p, 1, 0)[0] == 3);    // -0.1 * 2 = -0.2 < -0.15
}

// Sentence 6: test_sampler_stochastic() checks that top_k=1 equals argmax, that top_k=5
// always produces a token from the five best candidates, that the same (seed, counter)
// gives the same token (determinism), that flat logits produce a roughly uniform distribution,
// and that top_p with min_keep trims and widens the candidate set correctly.
static void test_sampler_stochastic() {
    const int nv = 4096;
    Rng r(14);
    std::vector<float> l(nv); for (auto& v : l) v = r.sym(4.f);
    // top_k = 1 is argmax regardless of seed/counter
    SamplerParams p; p.temperature = 0.8f; p.top_k = 1;
    int best = 0; for (int i = 1; i < nv; ++i) if (l[i] > l[best]) best = i;
    for (uint64_t c = 0; c < 5; ++c) CHECK(run(l, 1, nv, {}, 0, {}, p, 7, c)[0] == best);
    // top_k = 5: result always lies within the five best tokens; different counters differ sometimes
    std::vector<int> idx(nv); for (int i = 0; i < nv; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
    p.top_k = 5; p.temperature = 1.5f;
    int distinct = 0; std::vector<int> seen;
    for (uint64_t c = 0; c < 60; ++c) {
        const int t = run(l, 1, nv, {}, 0, {}, p, 99, c)[0];
        CHECK(std::find(idx.begin(), idx.begin() + 5, t) != idx.begin() + 5);
        CHECK(run(l, 1, nv, {}, 0, {}, p, 99, c)[0] == t);  // same (seed, counter) → same token
        if (std::find(seen.begin(), seen.end(), t) == seen.end()) { seen.push_back(t); ++distinct; }
    }
    CHECK(distinct >= 3);
    // flat logits over 4 tokens: roughly uniform (chi-sq style sanity check)
    std::vector<float> flat(4, 0.f); int cnt[4] = {0, 0, 0, 0};
    SamplerParams q; q.temperature = 1.f; q.top_k = 4;
    for (uint64_t c = 0; c < 1200; ++c) ++cnt[run(flat, 1, 4, {}, 0, {}, q, 5, c)[0]];
    std::printf("  flat counts %d %d %d %d, top5 distinct %d\n", cnt[0], cnt[1], cnt[2], cnt[3], distinct);
    for (int k = 0; k < 4; ++k) CHECK(cnt[k] > 220 && cnt[k] < 380);
    // top_p = 0.01 with min_keep = 1: only the highest-probability token survives the cut
    std::vector<float> l4 = {3.f, 2.9f, 2.8f, -4.f, -4.f, -4.f};
    SamplerParams t; t.temperature = 1.f; t.top_k = 6; t.top_p = 0.01f; t.min_keep = 1;
    bool only0 = true; for (uint64_t c = 0; c < 80; ++c) only0 &= (run(l4, 1, 6, {}, 0, {}, t, 3, c)[0] == 0);
    CHECK(only0);
    // min_keep = 3: at least 3 tokens survive the top-p cut even when top_p is tight
    t.min_keep = 3;
    int c3[6] = {0, 0, 0, 0, 0, 0};
    for (uint64_t c = 0; c < 300; ++c) ++c3[run(l4, 1, 6, {}, 0, {}, t, 3, c)[0]];
    CHECK(c3[0] > 0 && c3[1] > 0 && c3[2] > 0 && c3[3] == 0 && c3[4] == 0 && c3[5] == 0);
}

// Sentence 7: test_sampler_bad_args() checks that sample_tokens rejects repeat_penalty = 0
// and a null logits pointer, returning a non-cudaSuccess error code in both cases.
static void test_sampler_bad_args() {
    SamplerParams bad; bad.repeat_penalty = 0.f;
    SamplerParams g;   g.temperature      = 0.f;
    float*   a = dalloc<float>(64);
    int32_t* o = dalloc<int32_t>(1);
    CHECK(sample_tokens(a, 1, 8, nullptr, 0, nullptr, bad, 1, 0, a, o, 0) != cudaSuccess);
    CHECK(sample_tokens(nullptr, 1, 8, nullptr, 0, nullptr, g, 1, 0, a, o, 0) != cudaSuccess);
    cudaFree(a); cudaFree(o);
}

// test_sampler: top-level entry point exercising sample_tokens / sampler_kernel
// (src/cu/sampler.cu). Runs all sentence sub-tests (greedy, penalties, stochastic,
// bad_args) so the port-map entry
//   tests/k39_integration/sampler_parity.cu  test_sampler  src/cu/sampler.cu  sampler_kernel
// resolves to a named symbol in this translation unit.
static void test_sampler() {
    test_sampler_greedy();
    test_sampler_penalties();
    test_sampler_stochastic();
    test_sampler_bad_args();
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        std::printf("sampler_parity: no GPU device\n"); return 2;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("sampler_parity: cannot open GPU 0\n"); return 2;
    }
    test_sampler();
    std::printf("sampler_parity: checks=%ld failures=%ld\n", (long)g_checks, (long)g_fail);
    if (g_checks == 0 || g_fail != 0) { std::printf("sampler_parity FAILED\n"); return 1; }
    std::printf("sampler_parity PASSED\n");
    return 0;
}
