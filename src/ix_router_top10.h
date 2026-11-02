// RouterTop10 reference: softmax (double exp, ascending double sum), k argmax passes (larger value, then lower index), renorm clamp 2^-14.
#pragma once
#include "ix_u4_common.h"
IX4_HD int ix4_round16(int n) { return (n + 15) / 16 * 16; }
IX4_HD int router_smem_bytes(int n) { return ix4_round16(n) + n * 8 + n * 4; }
IX4_HD int router_threads(int n) { int m = n < 512 ? n : 512; return (m + 31) / 32 * 32; }
IX4_HD bool router_better(float v, int i, float v2, int i2) { return v > v2 || (v == v2 && i < i2); }
// logits[n] -> ids[k], w[k]. taken must hold n bytes (scratch).
inline void router_top10_ref(const float* logits, int n, int k, int* ids, float* w, double* p, unsigned char* taken) {
    double mx = (double)logits[0]; for (int i = 1; i < n; ++i) mx = (double)logits[i] > mx ? (double)logits[i] : mx;
    double s = 0.0; for (int i = 0; i < n; ++i) { p[i] = exp((double)logits[i] - mx); s += p[i]; }
    for (int i = 0; i < n; ++i) { p[i] /= s; taken[i] = 0; }
    float sum = 0.f;
    for (int j = 0; j < k; ++j) {
        int best = -1; for (int i = 0; i < n; ++i) if (!taken[i] && (best < 0 || router_better((float)p[i], i, (float)p[best], best))) best = i;
        taken[best] = 1; ids[j] = best; w[j] = (float)p[best]; sum += w[j];
    }
    float den = sum > 6.103515625e-5f ? sum : 6.103515625e-5f;
    for (int j = 0; j < k; ++j) w[j] /= den;
}
