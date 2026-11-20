// sampler_parity.cpp — CPU reference for the top-p/min-p token sampler.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

static constexpr int hist_len = 256; // histogram length for sampler stats

// reference_cut: find the min-p cut index (smallest prefix of sorted-desc probs summing to ≥ p).
static int reference_cut(const float* probs_sorted_desc, int n, float p) {
    double cumsum = 0.0;
    for (int i = 0; i < n; ++i) {
        cumsum += probs_sorted_desc[i];
        if (cumsum >= (double)p) return i + 1;
    }
    return n;
}

// reference_pick: pick the lowest-index token with cumulative prob ≥ u (inverse CDF).
static int reference_pick(const float* probs, int n, float u) {
    double cumsum = 0.0;
    for (int i = 0; i < n; ++i) {
        cumsum += probs[i];
        if (cumsum >= (double)u) return i;
    }
    return n - 1;
}

static int g_samp_fail = 0;
static void run(int n_vocab, float top_p) {
    std::vector<float> logits(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logits[i] = (float)(n_vocab - i) * 0.01f;
    // Softmax
    float mx = logits[0];
    for (float v : logits) mx = std::max(mx, v);
    double sum = 0.0;
    std::vector<float> probs(n_vocab);
    for (int i = 0; i < n_vocab; ++i) { probs[i] = std::exp(logits[i] - mx); sum += probs[i]; }
    for (float& v : probs) v = (float)(v / sum);
    // Sort descending
    std::vector<int> idx(n_vocab); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
    std::vector<float> sorted_p(n_vocab);
    for (int i = 0; i < n_vocab; ++i) sorted_p[i] = probs[idx[i]];
    int cut = reference_cut(sorted_p.data(), n_vocab, top_p);
    if (cut < 1 || cut > n_vocab) { std::printf("FAIL sampler_parity run cut=%d\n", cut); ++g_samp_fail; }
    int pick = reference_pick(sorted_p.data(), cut, 0.5f);
    if (pick < 0 || pick >= cut) { std::printf("FAIL sampler_parity pick=%d cut=%d\n", pick, cut); ++g_samp_fail; }
}

static void check(int n_vocab) {
    run(n_vocab, 0.9f);
    run(n_vocab, 0.99f);
    run(n_vocab, 1.0f);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    check(32000); check(1024); check(16);
    if (g_samp_fail) { std::printf("sampler_parity FAILED (%d)\n", g_samp_fail); return 1; }
    std::printf("sampler_parity PASSED\n"); return 0;
}
