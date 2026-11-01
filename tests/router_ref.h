// Host reference for the router: softmax in double -> float probs -> top-k (ties: lowest index) -> renormalise.
// Mirrors docs/semantics.md as implemented by Strata's kernel (float p = (float)(ex*inv); weights/sum in double).
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
inline void router_ref(const float* l, int n_expert, int k, int* ids, float* w) {
    float mx = -INFINITY; for (int e = 0; e < n_expert; ++e) mx = std::fmax(mx, l[e]);
    std::vector<double> ex(n_expert); double sum = 0.0;
    for (int e = 0; e < n_expert; ++e) { ex[e] = std::exp((double)l[e] - (double)mx); sum += ex[e]; }
    const float inv = (float)(1.0 / sum);
    std::vector<float> p(n_expert); for (int e = 0; e < n_expert; ++e) p[e] = (float)(ex[e] * inv);
    std::vector<int> ord(n_expert); std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return p[a] > p[b]; });
    double s = 0.0; for (int i = 0; i < k; ++i) { ids[i] = ord[i]; w[i] = p[ord[i]]; s += (double)w[i]; }
    const double sc = std::fmax(s, 6.103515625e-05);
    for (int i = 0; i < k; ++i) w[i] = (float)((double)w[i] / sc);
}
