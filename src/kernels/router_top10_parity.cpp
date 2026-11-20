// router_top10_parity.cpp — CPU reference for the top-10 router (softmax + renorm + clamp).
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

static constexpr float RENORM_CLAMP = 1.0f; // weight clamp after top-k renormalisation

// reference: top-k selection with softmax + renorm + clamp, lowest-index tie-break.
static void reference(const float* logits, int n_experts, int top_k,
                      std::vector<int>& ids, std::vector<float>& weights) {
    // Softmax
    float mx = *std::max_element(logits, logits + n_experts);
    std::vector<float> s(n_experts);
    double sum = 0.0;
    for (int i = 0; i < n_experts; ++i) { s[i] = std::exp(logits[i] - mx); sum += s[i]; }
    for (int i = 0; i < n_experts; ++i) s[i] = (float)(s[i] / sum);
    // Top-k: stable sort by (score desc, index asc) = lowest-index tie-break
    std::vector<int> idx(n_experts);
    for (int i = 0; i < n_experts; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){ return s[a] > s[b]; });
    ids.resize(top_k); weights.resize(top_k);
    for (int i = 0; i < top_k; ++i) ids[i] = idx[i];
    // Renorm selected weights
    double wsum = 0.0;
    for (int i = 0; i < top_k; ++i) wsum += s[ids[i]];
    for (int i = 0; i < top_k; ++i) {
        weights[i] = (float)std::min((double)RENORM_CLAMP, (double)s[ids[i]] / (wsum > 0 ? wsum : 1.0));
    }
}

static int g_rt_fail = 0;
static void run_case(int n_experts, int top_k, const float* logits) {
    std::vector<int> ids; std::vector<float> weights;
    reference(logits, n_experts, top_k, ids, weights);
    // Invariant: weights sum ≤ 1 + eps (clamped renorm)
    double wsum = 0.0;
    for (float w : weights) wsum += w;
    if (wsum > 1.0 + 1e-5) { std::printf("FAIL router renorm wsum=%g > 1\n", wsum); ++g_rt_fail; }
    // Invariant: ids are unique
    std::vector<int> sorted = ids; std::sort(sorted.begin(), sorted.end());
    for (int i = 1; i < (int)sorted.size(); ++i)
        if (sorted[i] == sorted[i-1]) { std::printf("FAIL router duplicate id %d\n", sorted[i]); ++g_rt_fail; }
}

static void check(int n_experts, int top_k) {
    std::vector<float> logits(n_experts);
    for (int i = 0; i < n_experts; ++i) logits[i] = (float)(i % 7) * 0.3f - 1.0f;
    run_case(n_experts, top_k, logits.data());
    // Tie case: all equal logits
    std::fill(logits.begin(), logits.end(), 0.0f);
    run_case(n_experts, top_k, logits.data());
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(64, 6); check(64, 10); check(128, 10);
    if (g_rt_fail) { std::printf("router_top10_parity FAILED (%d)\n", g_rt_fail); return 1; }
    std::printf("router_top10_parity PASSED\n"); return 0;
}
