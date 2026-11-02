// Sampler helpers recoverable from the visible theory: argmax merge, Philox uniform, penalty window, min_keep clamp, penalties.
#pragma once
#include "ix_u4_common.h"
struct IxValIdx { float v; int i; };
IX4_HD IxValIdx sampler_merge(IxValIdx a, IxValIdx b) { return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a; }
IX4_HD float sampler_uniform24(uint32_t c0) { return (float)(c0 >> 8) * (1.0f / 16777216.0f); }
IX4_HD int sampler_hlen(int last_n, int hist_len) { return last_n < hist_len ? last_n : hist_len; }
IX4_HD int sampler_clamp_cut(int cut, int min_keep, int n) { return cut < min_keep ? (min_keep < n ? min_keep : n) : cut; }
IX4_HD int sampler_presence_term(int count) { return count > 0 ? 1 : 0; }
IX4_HD float sampler_repeat_pen(float logit, float p) { return logit <= 0.f ? logit * p : logit / p; }
// Chain: penalties -> top_k -> top_p -> temperature -> pick. Greedy = global argmax of penalised logits (temperature NOT applied).
IX4_HD void sampler_penalties(float* logits, const int* hist, int hist_len, int last_n, float repeat, float freq, float presence) {
    int L = sampler_hlen(last_n, hist_len); const int* w = hist + (hist_len - L);
    for (int i = 0; i < L; ++i) {
        int t = w[i]; bool first = true; for (int j = 0; j < i; ++j) if (w[j] == t) first = false;
        if (!first) continue;
        int count = 0; for (int j = 0; j < L; ++j) count += (w[j] == t);
        logits[t] = sampler_repeat_pen(logits[t], repeat);
        logits[t] -= (float)count * freq + (float)sampler_presence_term(count) * presence;
    }
}
IX4_HD int sampler_greedy(const float* logits, int n) { IxValIdx b{logits[0], 0}; for (int i = 1; i < n; ++i) b = sampler_merge(b, IxValIdx{logits[i], i}); return b.i; }
// Philox4x32-10 counter-based RNG; uniform uses the top 24 bits of word 0 so the draw is in [0,1).
IX4_HD void sampler_philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        uint64_t p0 = (uint64_t)0xD2511F53u * c[0], p1 = (uint64_t)0xCD9E8D57u * c[2];
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ c[1] ^ k0, n2 = (uint32_t)(p0 >> 32) ^ c[3] ^ k1;
        c[0] = n0; c[1] = (uint32_t)p1; c[2] = n2; c[3] = (uint32_t)p0; k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
}
IX4_HD float sampler_draw(uint64_t seed, uint64_t counter) { uint32_t c[4] = {(uint32_t)counter, (uint32_t)(counter >> 32), 0u, 0u}; sampler_philox(c, (uint32_t)seed, (uint32_t)(seed >> 32)); return sampler_uniform24(c[0]); }
// Candidate list sorted descending by logit (ties: lower index first). cand_idx/cand_logit have n entries; returns survivors.
// top_k (0 = off) -> softmax over survivors at temperature T -> top_p cut (clamped by min_keep) -> pick with uniform u.
inline int sampler_sample(const float* logits, int n, int top_k, float top_p, float temp, int min_keep, float u, int* order, double* prob) {
    for (int i = 0; i < n; ++i) order[i] = i;
    // insertion-free selection sort is O(n^2); use simple stable sort via index compare
    for (int i = 1; i < n; ++i) { int t = order[i]; int j = i - 1; while (j >= 0 && (logits[order[j]] < logits[t])) { order[j + 1] = order[j]; --j; } order[j + 1] = t; }
    int m = (top_k > 0 && top_k < n) ? top_k : n;
    double mx = logits[order[0]] / temp, s = 0.0;
    for (int i = 0; i < m; ++i) { prob[i] = exp((double)logits[order[i]] / temp - mx); s += prob[i]; }
    for (int i = 0; i < m; ++i) prob[i] /= s;
    int cut = m; double c = 0.0; for (int i = 0; i < m; ++i) { c += prob[i]; if (c >= top_p) { cut = i + 1; break; } }
    cut = sampler_clamp_cut(cut, min_keep, m);
    double z = 0.0; for (int i = 0; i < cut; ++i) z += prob[i];
    double r = (double)u * z, a = 0.0; for (int i = 0; i < cut; ++i) { a += prob[i]; if (r < a) return order[i]; }
    return order[cut - 1];
}
