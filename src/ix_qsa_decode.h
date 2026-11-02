// QsaDecodeAttention: split-K decode attention (64-cell chunks, online-softmax partials, merge). Reference semantics.
#pragma once
#include "ix_u4_common.h"
#define IX_QSA_CHUNK 64
#define IX_QSA_HEAD_DIM 256
#define IX_QSA_GQA 12
IX4_HD int qsa_n_chunks(int cap) { return (cap + 63) / 64; }
IX4_HD int qsa_n_here(int n, int c) { int r = n - c * 64; return r < 64 ? (r > 0 ? r : 0) : 64; }
IX4_HD long qsa_scratch_floats(int cap, int n_head) { return (long)qsa_n_chunks(cap) * n_head * (IX_QSA_HEAD_DIM + 2) + 64; }
IX4_HD int qsa_kv_head(int h) { return h / IX_QSA_GQA; }
IX4_HD int qsa_local_head(int h) { return h % IX_QSA_GQA; }
// One (chunk, head) partial. scores[i] already scaled; v is [n_here][hd]. Empty chunk: m=-FLT_MAX, l=0, acc untouched (zeroed).
IX4_HD void qsa_partial(const float* scores, const float* v, int n_here, int hd, float* m, float* l, float* acc) {
    for (int d = 0; d < hd; ++d) acc[d] = 0.f;
    if (n_here <= 0) { *m = -FLT_MAX; *l = 0.f; return; }
    float mx = -FLT_MAX; for (int i = 0; i < n_here; ++i) mx = scores[i] > mx ? scores[i] : mx;
    float s = 0.f;
    for (int i = 0; i < n_here; ++i) { float p = expf(scores[i] - mx); s += p; for (int d = 0; d < hd; ++d) acc[d] += p * v[(long)i * hd + d]; }
    *m = mx; *l = s;
}
// Merge: M = max m; L = sum l*exp(m-M); out = (sum acc*exp(m-M))/L, or 0 if L == 0. Empty chunks skipped.
IX4_HD void qsa_merge(const float* m, const float* l, const float* acc, int nc, int hd, float* out) {
    float M = -FLT_MAX; bool any = false;
    for (int c = 0; c < nc; ++c) if (!(l[c] == 0.f && m[c] == -FLT_MAX)) { any = true; M = m[c] > M ? m[c] : M; }
    float L = 0.f; if (any) for (int c = 0; c < nc; ++c) if (!(l[c] == 0.f && m[c] == -FLT_MAX)) L += l[c] * expf(m[c] - M);
    for (int d = 0; d < hd; ++d) {
        float a = 0.f;
        if (L != 0.f) for (int c = 0; c < nc; ++c) if (!(l[c] == 0.f && m[c] == -FLT_MAX)) a += acc[(long)c * hd + d] * expf(m[c] - M);
        out[d] = L != 0.f ? a / L : 0.f;
    }
}
