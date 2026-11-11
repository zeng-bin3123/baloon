// port_kernels_ref.hpp - host reference math for the decode kernels: GEMV (bf16, S-family 2-bit), GDN pre/post ops,
// gated residual, flash short step, QSA block select, router top-10, IQ4 block decode + MMVQ, shared expert, PLE post-ops.
// Plain scalar C++ (fp32 accumulate, fixed summation order) so device kernels can be checked against it.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "baloon/f16_conversion.hpp"
#include "baloon/port_scalar.hpp"

namespace baloon {
namespace refk {

inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
inline float gelu_tanh(float x) { return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x))); }

// ================================================================== bf16 GEMV
// y[r] = sum_i bf16(w[r*n_in+i]) * x[i], fp32 accumulate in index order.
inline void bf16_gemv_fp32_mmvf(const uint16_t* w, const float* x, float* y, int n_in, int n_out) {
    for (int r = 0; r < n_out; ++r) {
        float a = 0.f;
        for (int i = 0; i < n_in; ++i) a += f32_from_bf16(w[(int64_t)r * n_in + i]) * x[i];
        y[r] = a;
    }
}
// Same product with the reduction axis cut into `splits` contiguous chunks (partial sums added in chunk order).
inline void bf16_gemv_split(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int splits) {
    if (splits < 1) splits = 1;
    const int chunk = (n_in + splits - 1) / splits;
    for (int r = 0; r < n_out; ++r) {
        float total = 0.f;
        for (int s = 0; s < splits; ++s) {
            const int lo = s * chunk, hi = std::min(n_in, lo + chunk);
            float a = 0.f;
            for (int i = lo; i < hi; ++i) a += f32_from_bf16(w[(int64_t)r * n_in + i]) * x[i];
            total += a;
        }
        y[r] = total;
    }
}

// ================================================================== S-family 2-bit GEMV
enum class SForm { S2 = 2, S4 = 4, S8 = 8 };
struct Codebook { int bits = 2; float levels[16] = {-3.f, -1.f, 1.f, 3.f}; };
constexpr int kS2Group = 64;   // values per scale group (matches the S2 row layout: 2560 values -> 40 scales)

inline int s2_row_bytes(int n_in) { return n_in / 4; }
inline int s2_code(const uint8_t* row, int e) { return (row[e >> 2] >> ((e & 3) * 2)) & 3; }

// Expand one S2 row (codes: 2 bits, element e in byte e/4 at shift 2*(e%4); one fp32 scale per 64 values).
inline void dequant_s2_ref(const uint8_t* codes, const float* scales, const Codebook& cb, float* out, int n_in) {
    for (int e = 0; e < n_in; ++e) out[e] = cb.levels[s2_code(codes, e)] * scales[e / kS2Group];
}
// Reference: group-wise (sum_codes level*x) * scale.
inline void s2_gemv(const uint8_t* codes, const float* scales, const Codebook& cb, const float* x, float* y, int n_in, int n_out) {
    const int gs = n_in / kS2Group;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = codes + (int64_t)r * s2_row_bytes(n_in);
        float acc = 0.f;
        for (int g = 0; g < gs; ++g) {
            float a = 0.f;
            for (int e = g * kS2Group; e < (g + 1) * kS2Group; ++e) a += cb.levels[s2_code(row, e)] * x[e];
            acc += a * scales[(int64_t)r * gs + g];
        }
        y[r] = acc;
    }
}
// Byte-at-a-time variant (4 codes per byte through a 256-entry table); identical summation order to s2_gemv.
inline void s2_gemv_fast(const uint8_t* codes, const float* scales, const Codebook& cb, const float* x, float* y, int n_in, int n_out) {
    float lut[256][4];
    for (int b = 0; b < 256; ++b) for (int k = 0; k < 4; ++k) lut[b][k] = cb.levels[(b >> (2 * k)) & 3];
    const int gs = n_in / kS2Group;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = codes + (int64_t)r * s2_row_bytes(n_in);
        float acc = 0.f;
        for (int g = 0; g < gs; ++g) {
            float a = 0.f;
            for (int e = g * kS2Group; e < (g + 1) * kS2Group; e += 4) {
                const float* l = lut[row[e >> 2]];
                a += l[0] * x[e]; a += l[1] * x[e + 1]; a += l[2] * x[e + 2]; a += l[3] * x[e + 3];
            }
            acc += a * scales[(int64_t)r * gs + g];
        }
        y[r] = acc;
    }
}
// Quad variant: four partial accumulators (one per code slot of a byte) merged at group end. Same value up to fp32 reassociation.
inline void s2_gemv_quads(const uint8_t* codes, const float* scales, const Codebook& cb, const float* x, float* y, int n_in, int n_out) {
    const int gs = n_in / kS2Group;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = codes + (int64_t)r * s2_row_bytes(n_in);
        float acc = 0.f;
        for (int g = 0; g < gs; ++g) {
            float q[4] = {0.f, 0.f, 0.f, 0.f};
            for (int e = g * kS2Group; e < (g + 1) * kS2Group; e += 4)
                for (int k = 0; k < 4; ++k) q[k] += cb.levels[(row[e >> 2] >> (2 * k)) & 3] * x[e + k];
            acc += ((q[0] + q[1]) + (q[2] + q[3])) * scales[(int64_t)r * gs + g];
        }
        y[r] = acc;
    }
}
// Integer activations: x_q (int8) with one fp32 scale per 64 values. Levels must be integers; per-group dot is exact in int32.
inline void s2_gemv_q8(const uint8_t* codes, const float* scales, const Codebook& cb, const int8_t* xq, const float* xs, float* y, int n_in, int n_out) {
    const int gs = n_in / kS2Group;
    for (int r = 0; r < n_out; ++r) {
        const uint8_t* row = codes + (int64_t)r * s2_row_bytes(n_in);
        float acc = 0.f;
        for (int g = 0; g < gs; ++g) {
            int32_t d = 0;
            for (int e = g * kS2Group; e < (g + 1) * kS2Group; ++e) d += (int32_t)cb.levels[s2_code(row, e)] * (int32_t)xq[e];
            acc += (float)d * scales[(int64_t)r * gs + g] * xs[g];
        }
        y[r] = acc;
    }
}

// ================================================================== GDN
struct GdnShapes {
    int hk = 16, hv = 32, dk = 128, dv = 128, conv_taps = 4;
    int conv_channels() const { return 2 * hk * dk + hv * dv; }
    int hist_taps() const { return conv_taps - 1; }
    bool ok() const { return hk > 0 && hv > 0 && dk > 0 && dv > 0 && conv_taps >= 2 && hv % hk == 0; }
};
// Depthwise causal conv for one token: window = [h0..h_{T-2}, x]; out = sum_t w[c][t]*window[t]; hist slides left by one.
inline void gdn_conv_step(float* hist, const float* x, const float* w, float* out, int channels, int taps) {
    const int ht = taps - 1;
    for (int c = 0; c < channels; ++c) {
        float* h = hist + (int64_t)c * ht;
        float a = 0.f;
        for (int t = 0; t < ht; ++t) a += w[(int64_t)c * taps + t] * h[t];
        a += w[(int64_t)c * taps + ht] * x[c];
        for (int t = 0; t + 1 < ht; ++t) h[t] = h[t + 1];
        h[ht - 1] = x[c];
        out[c] = a;
    }
}
inline void native_gdn_conv_silu(float* hist, const float* x, const float* w, float* out, int channels, int taps) {
    gdn_conv_step(hist, x, w, out, channels, taps);
    for (int c = 0; c < channels; ++c) out[c] = silu(out[c]);
}
// g = -exp(A_log) * softplus(a + dt_bias)
inline void native_gdn_gate(const float* a, const float* dt_bias, const float* A_log, float* g, int n) {
    for (int i = 0; i < n; ++i) g[i] = -std::exp(A_log[i]) * softplus(a[i] + dt_bias[i]);
}
inline void native_gdn_beta_gate(const float* b, const float* a, const float* dt_bias, const float* A_log, float* beta, float* g, int n) {
    for (int i = 0; i < n; ++i) beta[i] = sigmoid(b[i]);
    native_gdn_gate(a, dt_bias, A_log, g, n);
}
// Per-head L2 normalisation: x / sqrt(sum x^2 + eps).
inline void native_gdn_l2_norm(float* x, int heads, int d, float eps) {
    for (int h = 0; h < heads; ++h) {
        float* p = x + (int64_t)h * d; float s = 0.f;
        for (int i = 0; i < d; ++i) s += p[i] * p[i];
        const float inv = 1.f / std::sqrt(s + eps);
        for (int i = 0; i < d; ++i) p[i] *= inv;
    }
}
// Gated RMS norm per head: out = rmsnorm(o) * w * silu(z)
inline void native_gdn_out_norm(const float* o, const float* z, const float* w, float* out, int heads, int d, float eps) {
    for (int h = 0; h < heads; ++h) {
        const float* p = o + (int64_t)h * d; float s = 0.f;
        for (int i = 0; i < d; ++i) s += p[i] * p[i];
        const float inv = 1.f / std::sqrt(s / (float)d + eps);
        for (int i = 0; i < d; ++i) out[(int64_t)h * d + i] = p[i] * inv * w[i] * silu(z[(int64_t)h * d + i]);
    }
}
// One delta-rule step for one value head. state[dk][dv]; q,k: dk; v: dv.
inline void gdn_step_host(float* state, const float* q, const float* k, const float* v, float g, float beta, float scale, float* out, int dk, int dv) {
    const float decay = std::exp(g);
    for (int i = 0; i < dk * dv; ++i) state[i] *= decay;
    std::vector<float> d(dv);
    for (int j = 0; j < dv; ++j) {
        float kv = 0.f;
        for (int i = 0; i < dk; ++i) kv += state[(int64_t)i * dv + j] * k[i];
        d[j] = (v[j] - kv) * beta;
    }
    for (int i = 0; i < dk; ++i) for (int j = 0; j < dv; ++j) state[(int64_t)i * dv + j] += k[i] * d[j];
    for (int j = 0; j < dv; ++j) {
        float o = 0.f;
        for (int i = 0; i < dk; ++i) o += state[(int64_t)i * dv + j] * q[i];
        out[j] = o * scale;
    }
}
// Fused: both gates in one pass (identical to the unfused pair).
inline void fused_gdn_ab(const float* b, const float* a, const float* dt_bias, const float* A_log, float* beta, float* g, int n) {
    for (int i = 0; i < n; ++i) { beta[i] = sigmoid(b[i]); g[i] = -std::exp(A_log[i]) * softplus(a[i] + dt_bias[i]); }
}
// Fused: conv+silu, then L2-normalise the q and k slices of the channel vector (layout q | k | v).
inline void fused_gdn_conv_l2(float* hist, const float* x, const float* w, float* out, const GdnShapes& s, float eps) {
    native_gdn_conv_silu(hist, x, w, out, s.conv_channels(), s.conv_taps);
    native_gdn_l2_norm(out, s.hk, s.dk, eps);
    native_gdn_l2_norm(out + (int64_t)s.hk * s.dk, s.hk, s.dk, eps);
}
// Fused: delta step for every value head (q/k head = h % hk is shared across value groups) followed by the gated out-norm.
inline void fused_gdn_step_norm(float* state, const float* qkv, const float* g, const float* beta, const float* z, const float* nw,
                                float* out, const GdnShapes& s, float eps) {
    const float* q = qkv; const float* k = qkv + (int64_t)s.hk * s.dk; const float* v = k + (int64_t)s.hk * s.dk;
    std::vector<float> raw((size_t)s.hv * s.dv);
    const float scale = 1.f / std::sqrt((float)s.dk);
    for (int h = 0; h < s.hv; ++h) {
        const int kh = h % s.hk;
        gdn_step_host(state + (int64_t)h * s.dk * s.dv, q + (int64_t)kh * s.dk, k + (int64_t)kh * s.dk, v + (int64_t)h * s.dv,
                      g[h], beta[h], scale, raw.data() + (int64_t)h * s.dv, s.dk, s.dv);
    }
    native_gdn_out_norm(raw.data(), z, nw, out, s.hv, s.dv, eps);
}

// ================================================================== gated residual (hc parallel streams of width n)
struct GrShapes { int n = 0, hc = 4; bool ok() const { return n > 0 && hc > 0; } };
using GrWorkspace = std::vector<float>;
constexpr int kFusedGrMaxT = 8;
inline std::atomic<bool>& gr_fp32_flag() { static std::atomic<bool> f{false}; return f; }
inline std::atomic<bool>& gr_mmvf_flag() { static std::atomic<bool> f{true}; return f; }
inline void gr_set_fp32_activations(bool on) { gr_fp32_flag().store(on); }
inline void gr_set_native_mmvf(bool on) { gr_mmvf_flag().store(on); }
// Workspace floats: pre-mix (n), normed (n), gate logits (hc), gates (hc), post (n*hc).
inline uint64_t gr_workspace_bytes(const GrShapes& s) {
    if (!s.ok()) return 0;
    auto al = [](uint64_t b) { return (b + 15) / 16 * 16; };
    return al(4ull * s.n) + al(4ull * s.n) + al(4ull * s.hc) + al(4ull * s.hc) + al(4ull * s.n * s.hc);
}
inline void native_gr_rms_norm_weighted(const float* x, const float* w, float* out, int n, float eps) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    const float inv = 1.f / std::sqrt(s / (float)n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}
// Collapse the streams with per-stream gates: out[d] = sum_c sigmoid(logit[c]) * R[c*n+d].
inline void native_gr_pre_gated(const float* R, const float* logits, float* out, int n, int hc) {
    for (int d = 0; d < n; ++d) {
        float a = 0.f;
        for (int c = 0; c < hc; ++c) a += sigmoid(logits[c]) * R[(int64_t)c * n + d];
        out[d] = a;
    }
}
// Write back: out[c*n+d] = R[c*n+d] + block[d] * w[c]   (same contract as the gated-residual write).
inline void native_gr_post(const float* R, const float* block, const float* w, float* out, int n, int hc) {
    for (int c = 0; c < hc; ++c) for (int d = 0; d < n; ++d) out[(int64_t)c * n + d] = R[(int64_t)c * n + d] + block[d] * w[c];
}
inline void native_gr_down_silu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}
struct FusedGrArgs {
    const float* R = nullptr;        // [T][hc*n] streams
    const float* logits = nullptr;   // [T][hc]
    const float* norm_w = nullptr;   // [n]
    float* out = nullptr;            // [T][n]
    int n = 0, hc = 0, T = 1;
    float eps = 1e-6f;
};
inline bool fused_gr_supported(int T, int n, int hc) { return T >= 1 && T <= kFusedGrMaxT && n > 0 && hc > 0; }
// One token: gated collapse of the streams, then weighted RMS norm.
inline void fused_gr_read(const FusedGrArgs& a, int t) {
    std::vector<float> tmp(a.n);
    native_gr_pre_gated(a.R + (int64_t)t * a.hc * a.n, a.logits + (int64_t)t * a.hc, tmp.data(), a.n, a.hc);
    native_gr_rms_norm_weighted(tmp.data(), a.norm_w, a.out + (int64_t)t * a.n, a.n, a.eps);
}
// T tokens; returns false (and touches nothing) when the shape is outside what the fused path supports.
inline bool fused_gr_read_multi(const FusedGrArgs& a) {
    if (!fused_gr_supported(a.T, a.n, a.hc)) return false;
    for (int t = 0; t < a.T; ++t) fused_gr_read(a, t);
    return true;
}

// ================================================================== short flash step (single query, GQA)
constexpr int kFlashShortMaxCells = 256;  // theory: short-context kernel valid for width in [1, 256]
inline NativeFlashAttnStatus native_flash_attn_short_step(const float* q, const float* k, const float* v, float* out, int heads, int kv_heads,
                                                          int hd, int n_cells, float scale) {
    if (heads <= 0 || kv_heads <= 0 || hd <= 0 || heads % kv_heads != 0 || n_cells < 1) return NativeFlashAttnStatus::BadShape;
    if (n_cells > kFlashShortMaxCells) return NativeFlashAttnStatus::TooLong;
    const int grp = heads / kv_heads;
    std::vector<float> s(n_cells);
    for (int h = 0; h < heads; ++h) {
        const int kh = h / grp;
        float mx = -INFINITY;
        for (int c = 0; c < n_cells; ++c) {
            float d = 0.f;
            for (int i = 0; i < hd; ++i) d += q[(int64_t)h * hd + i] * k[((int64_t)c * kv_heads + kh) * hd + i];
            s[c] = d * scale; mx = std::max(mx, s[c]);
        }
        float sum = 0.f;
        for (int c = 0; c < n_cells; ++c) { s[c] = std::exp(s[c] - mx); sum += s[c]; }
        for (int i = 0; i < hd; ++i) {
            float o = 0.f;
            for (int c = 0; c < n_cells; ++c) o += s[c] * v[((int64_t)c * kv_heads + kh) * hd + i];
            out[(int64_t)h * hd + i] = o / sum;
        }
    }
    return NativeFlashAttnStatus::Ok;
}

// ================================================================== QSA helpers
inline void native_qsa_gate_apply(float* attn, const float* gate, int n) { for (int i = 0; i < n; ++i) attn[i] *= sigmoid(gate[i]); }
inline void native_qsa_rms_norm_weighted(const float* x, const float* w, float* out, int heads, int hd, float eps) {
    for (int h = 0; h < heads; ++h) native_gr_rms_norm_weighted(x + (int64_t)h * hd, w, out + (int64_t)h * hd, hd, eps);
}
// Block score = max over the block's cells of q . k_cell (last block may be short).
inline void qsa_block_scores(const float* q, const float* kidx, int n_cells, int block_cells, int d, float* scores) {
    const int nb = (n_cells + block_cells - 1) / block_cells;
    for (int b = 0; b < nb; ++b) {
        float m = -INFINITY;
        for (int c = b * block_cells; c < std::min(n_cells, (b + 1) * block_cells); ++c) {
            float dot = 0.f;
            for (int i = 0; i < d; ++i) dot += q[i] * kidx[(int64_t)c * d + i];
            m = std::max(m, dot);
        }
        scores[b] = m;
    }
}
// Indices of the k best scores, best first; ties go to the lower index. k is clamped to nb.
inline std::vector<int> qsa_block_topk(const float* scores, int nb, int k) {
    std::vector<int> idx(nb);
    for (int i = 0; i < nb; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    idx.resize(std::min(std::max(k, 0), nb));
    return idx;
}

// ================================================================== router top-10
constexpr int kRouterK = 10;
// Softmax over all experts, pick the 10 best (ties -> lower index), renormalise to sum 1 (divisor clamped at 1/16384).
inline int native_router_top10(const float* logits, int n_expert, int* idx, float* w) {
    if (n_expert < 1) return 0;
    const int k = std::min(kRouterK, n_expert);
    float mx = -INFINITY;
    for (int i = 0; i < n_expert; ++i) mx = std::max(mx, logits[i]);
    std::vector<float> p(n_expert); float sum = 0.f;
    for (int i = 0; i < n_expert; ++i) { p[i] = std::exp(logits[i] - mx); sum += p[i]; }
    for (int i = 0; i < n_expert; ++i) p[i] /= sum;
    std::vector<int> o = qsa_block_topk(p.data(), n_expert, k);
    float s = 0.f;
    for (int j = 0; j < k; ++j) s += p[o[j]];
    s = std::max(s, 1.f / 16384.f);
    for (int j = 0; j < k; ++j) { idx[j] = o[j]; w[j] = p[o[j]] / s; }
    return k;
}

// ================================================================== MoE combine / shared expert / PLE post-ops
// out = shared + sum_j w[j] * routed[j]
inline void native_moe_combine(const float* routed, const float* w, const float* shared, float* out, int k, int n) {
    for (int d = 0; d < n; ++d) {
        float a = shared ? shared[d] : 0.f;
        for (int j = 0; j < k; ++j) a += w[j] * routed[(int64_t)j * n + d];
        out[d] = a;
    }
}
struct NativeSharedWeights {
    const float* gate = nullptr; const float* up = nullptr; const float* down = nullptr;   // [n_ff][n_embd], [n_ff][n_embd], [n_embd][n_ff]
    const float* gate_vec = nullptr;                                                         // optional [n_embd] sigmoid output gate
    int n_embd = 0, n_ff = 0;
};
inline std::atomic<bool>& shared_bf16_flag() { static std::atomic<bool> f{false}; return f; }
inline void shared_expert_set_native_bf16(bool on) { shared_bf16_flag().store(on); }
inline uint64_t shared_expert_scratch_bytes(int n_ff, int T) { return n_ff > 0 && T > 0 ? 2ull * 4 * n_ff * T : 0; }
inline void shared_expert(const NativeSharedWeights& w, const float* x, float* out, float* scratch) {
    float* g = scratch; float* u = scratch + w.n_ff;
    for (int f = 0; f < w.n_ff; ++f) {
        float a = 0.f, b = 0.f;
        for (int i = 0; i < w.n_embd; ++i) { a += w.gate[(int64_t)f * w.n_embd + i] * x[i]; b += w.up[(int64_t)f * w.n_embd + i] * x[i]; }
        g[f] = silu(a) * b; (void)u;
    }
    float og = 1.f;
    if (w.gate_vec) { float s = 0.f; for (int i = 0; i < w.n_embd; ++i) s += w.gate_vec[i] * x[i]; og = sigmoid(s); }
    for (int d = 0; d < w.n_embd; ++d) {
        float a = 0.f;
        for (int f = 0; f < w.n_ff; ++f) a += w.down[(int64_t)d * w.n_ff + f] * g[f];
        out[d] = a * og;
    }
}
// T tokens, each independent: result equals T single-token calls.
inline void shared_expert_multi(const NativeSharedWeights& w, const float* x, float* out, float* scratch, int T) {
    for (int t = 0; t < T; ++t) shared_expert(w, x + (int64_t)t * w.n_embd, out + (int64_t)t * w.n_embd, scratch + (int64_t)t * 2 * w.n_ff);
}
struct NativePlePostopsBuffers { const float* gate = nullptr; const float* ple = nullptr; float* tmp = nullptr; int n = 0; };
// out = hidden + rmsnorm(gelu(gate) * ple) * post_w
inline void native_ple_postops(const NativePlePostopsBuffers& b, const float* hidden, const float* post_w, float eps, float* out) {
    for (int i = 0; i < b.n; ++i) b.tmp[i] = gelu_tanh(b.gate[i]) * b.ple[i];
    std::vector<float> nrm(b.n);
    native_gr_rms_norm_weighted(b.tmp, post_w, nrm.data(), b.n, eps);
    for (int i = 0; i < b.n; ++i) out[i] = hidden[i] + nrm[i];
}
inline void native_rope_apply(float* x, int heads, int hd, int n_rot, const float* cs, const float* sn) {
    for (int h = 0; h < heads; ++h) rope_neox_apply(x + (int64_t)h * hd, n_rot, cs, sn);
}

// ================================================================== IQ4 block decode + MMVQ + Q8_1 rows
constexpr int kTypeIq4Nl = 20, kTypeIq4Xs = 23;
constexpr int8_t kIq4Values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
inline bool iq_supported(int type) { return type == kTypeIq4Nl || type == kTypeIq4Xs; }
inline int iq_block_elems(int type) { return type == kTypeIq4Nl ? 32 : type == kTypeIq4Xs ? 256 : 0; }
inline int iq_block_bytes(int type) { return type == kTypeIq4Nl ? 18 : type == kTypeIq4Xs ? 136 : 0; }
// Bytes of one row of n_in values (0 when unsupported or n_in is not a whole number of blocks).
inline int64_t iq_row_bytes(int type, int64_t n_in) {
    if (!iq_supported(type) || n_in <= 0 || n_in % iq_block_elems(type)) return 0;
    return n_in / iq_block_elems(type) * iq_block_bytes(type);
}
inline void iq_dequant_f32(int type, const uint8_t* src, float* out, int64_t n) {
    if (iq_row_bytes(type, n) == 0) return;
    const int64_t nb = n / iq_block_elems(type);
    for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* p = src + b * iq_block_bytes(type);
        uint16_t dh; std::memcpy(&dh, p, 2);
        const float d = f16::half_to_float(dh);
        if (type == kTypeIq4Nl) {
            for (int j = 0; j < 16; ++j) {
                out[b * 32 + j] = d * kIq4Values[p[2 + j] & 15];
                out[b * 32 + 16 + j] = d * kIq4Values[p[2 + j] >> 4];
            }
        } else {
            uint16_t sh; std::memcpy(&sh, p + 2, 2);
            const uint8_t* sl = p + 4; const uint8_t* qs = p + 8;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * (float)(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    out[b * 256 + ib * 32 + j] = dl * kIq4Values[qs[ib * 16 + j] & 15];
                    out[b * 256 + ib * 32 + 16 + j] = dl * kIq4Values[qs[ib * 16 + j] >> 4];
                }
            }
        }
    }
}
inline void iq_dequant_f16(int type, const uint8_t* src, uint16_t* out, int64_t n) {
    std::vector<float> t((size_t)std::max<int64_t>(n, 0));
    iq_dequant_f32(type, src, t.data(), n);
    for (int64_t i = 0; i < n; ++i) out[i] = f16::float_to_half(t[i]);
}
// Gate/up rows are interleaved (row 2r = gate, 2r+1 = up); split into two fp16 matrices [n_ff][n_in].
inline void iq_dequant_gu_f16(int type, const uint8_t* src, uint16_t* gate, uint16_t* up, int n_ff, int n_in) {
    const int64_t rb = iq_row_bytes(type, n_in);
    for (int r = 0; r < n_ff; ++r) {
        iq_dequant_f16(type, src + (int64_t)(2 * r) * rb, gate + (int64_t)r * n_in, n_in);
        iq_dequant_f16(type, src + (int64_t)(2 * r + 1) * rb, up + (int64_t)r * n_in, n_in);
    }
}
// Gather rows `ids` of an embedding table into fp32; ids outside [0, n_rows) yield a zero row.
inline void iq_embed_rows(int type, const uint8_t* table, int64_t n_rows, const int32_t* ids, int n_ids, int n_in, float* out) {
    const int64_t rb = iq_row_bytes(type, n_in);
    for (int i = 0; i < n_ids; ++i) {
        float* o = out + (int64_t)i * n_in;
        if (ids[i] < 0 || ids[i] >= n_rows || rb == 0) { for (int j = 0; j < n_in; ++j) o[j] = 0.f; continue; }
        iq_dequant_f32(type, table + (int64_t)ids[i] * rb, o, n_in);
    }
}
// y[r] = dequant(row r) . x in fp32.
inline bool iq_mmvq(int type, const uint8_t* w, const float* x, float* y, int n_in, int n_out) {
    const int64_t rb = iq_row_bytes(type, n_in);
    if (rb == 0) return false;
    std::vector<float> row(n_in);
    for (int r = 0; r < n_out; ++r) {
        iq_dequant_f32(type, w + (int64_t)r * rb, row.data(), n_in);
        float a = 0.f;
        for (int i = 0; i < n_in; ++i) a += row[i] * x[i];
        y[r] = a;
    }
    return true;
}
// Q8_1 blocks of 36 bytes: fp16 d, fp16 s = d * sum(q), int8 q[32]; q = round-half-away(x / d), d = amax / 127.
constexpr int kQ81Bytes = 36;
inline void quantize_q8_1_rows(const float* x, int n_rows, int n_in, uint8_t* out) {
    const int nb = n_in / 32;
    for (int r = 0; r < n_rows; ++r) for (int b = 0; b < nb; ++b) {
        const float* p = x + (int64_t)r * n_in + b * 32; float amax = 0.f;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(p[i]));
        const float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        int8_t q[32]; int sum = 0;
        for (int i = 0; i < 32; ++i) { q[i] = (int8_t)std::lround(p[i] * id); sum += q[i]; }
        uint8_t* o = out + ((int64_t)r * nb + b) * kQ81Bytes;
        const uint16_t dh = f16::float_to_half(d), sh = f16::float_to_half(d * (float)sum);
        std::memcpy(o, &dh, 2); std::memcpy(o + 2, &sh, 2); std::memcpy(o + 4, q, 32);
    }
}

// Expert blob: gate/up interleaved rows [2*n_ff][n_in], then down rows [n_embd][n_ff] (same quant type).
struct NativeExpertLayout { int64_t gu_row_bytes = 0, down_row_bytes = 0, up_off = 0, down_off = 0, bytes = 0; bool ok = false; };
inline NativeExpertLayout native_expert_layout(int type, int n_embd, int n_ff) {
    NativeExpertLayout l;
    l.gu_row_bytes = iq_row_bytes(type, n_embd); l.down_row_bytes = iq_row_bytes(type, n_ff);
    if (!l.gu_row_bytes || !l.down_row_bytes) return l;
    l.up_off = (int64_t)n_ff * l.gu_row_bytes;
    l.down_off = 2 * l.up_off;
    l.bytes = l.down_off + (int64_t)n_embd * l.down_row_bytes;
    l.ok = true;
    return l;
}
inline uint64_t native_expert_scratch_bytes(int n_embd, int n_ff) { return 4ull * (n_embd + 2 * (uint64_t)n_ff + n_embd); }
// out = sum_j w[j] * down_j( silu(gate_j x) * (up_j x) ), experts addressed by blob pointers.
inline bool native_expert_grouped(int type, const uint8_t* const* experts, const float* w, int k, const float* x, float* out, int n_embd, int n_ff) {
    const NativeExpertLayout l = native_expert_layout(type, n_embd, n_ff);
    if (!l.ok) return false;
    std::vector<float> row_e(std::max(n_embd, n_ff)), h(n_ff);
    for (int d = 0; d < n_embd; ++d) out[d] = 0.f;
    for (int j = 0; j < k; ++j) {
        const uint8_t* b = experts[j];
        for (int f = 0; f < n_ff; ++f) {
            float g = 0.f, u = 0.f;
            iq_dequant_f32(type, b + (int64_t)(2 * f) * l.gu_row_bytes, row_e.data(), n_embd);
            for (int i = 0; i < n_embd; ++i) g += row_e[i] * x[i];
            iq_dequant_f32(type, b + (int64_t)(2 * f + 1) * l.gu_row_bytes, row_e.data(), n_embd);
            for (int i = 0; i < n_embd; ++i) u += row_e[i] * x[i];
            h[f] = silu(g) * u;
        }
        for (int d = 0; d < n_embd; ++d) {
            iq_dequant_f32(type, b + l.down_off + (int64_t)d * l.down_row_bytes, row_e.data(), n_ff);
            float a = 0.f;
            for (int f = 0; f < n_ff; ++f) a += row_e[f] * h[f];
            out[d] += w[j] * a;
        }
    }
    return true;
}

// ================================================================== plain-named entry points (same math as the native_* forms above)
inline void bf16_gemv(const uint16_t* w, const float* x, float* y, int n_in, int n_out) { bf16_gemv_fp32_mmvf(w, x, y, n_in, n_out); }
inline void gdn_beta_gate(const float* b, const float* a, const float* dt_bias, const float* A_log, float* beta, float* g, int n) {
    native_gdn_beta_gate(b, a, dt_bias, A_log, beta, g, n);
}
inline void gdn_l2_norm(float* x, int heads, int d, float eps) { native_gdn_l2_norm(x, heads, d, eps); }
inline void gdn_out_norm(const float* o, const float* z, const float* w, float* out, int heads, int d, float eps) {
    native_gdn_out_norm(o, z, w, out, heads, d, eps);
}
inline void gdn_step(float* state, const float* q, const float* k, const float* v, float g, float beta, float scale, float* out, int dk, int dv) {
    gdn_step_host(state, q, k, v, g, beta, scale, out, dk, dv);
}
inline void native_gdn_step(float* state, const float* q, const float* k, const float* v, float g, float beta, float scale, float* out, int dk, int dv) {
    gdn_step_host(state, q, k, v, g, beta, scale, out, dk, dv);
}
// Gated residual read for one token: gated collapse of the streams, then weighted RMS norm.
inline void gr_read(const float* R, const float* logits, const float* norm_w, float* out, int n, int hc, float eps) {
    std::vector<float> mix(n);
    native_gr_pre_gated(R, logits, mix.data(), n, hc);
    native_gr_rms_norm_weighted(mix.data(), norm_w, out, n, eps);
}
inline void gr_write(const float* R, const float* block, const float* w, float* out, int n, int hc) { native_gr_post(R, block, w, out, n, hc); }
// Size the workspace for the shape (zero-filled); false for an invalid shape.
inline bool gr_workspace_init(GrWorkspace& ws, const GrShapes& s) {
    const uint64_t b = gr_workspace_bytes(s);
    if (b == 0) { ws.clear(); return false; }
    ws.assign((size_t)(b / 4), 0.f);
    return true;
}
// Append one key row at cell `pos` of the indexer cache; false when pos is outside [0, n_cells).
inline bool native_qsa_indexer_append(float* cache, int n_cells, int pos, const float* k, int d) {
    if (pos < 0 || pos >= n_cells || d <= 0) return false;
    for (int i = 0; i < d; ++i) cache[(int64_t)pos * d + i] = k[i];
    return true;
}
inline void native_qsa_score(const float* q, const float* kidx, int n_cells, int block_cells, int d, float* scores) {
    qsa_block_scores(q, kidx, n_cells, block_cells, d, scores);
}
inline int router_top10(const float* logits, int n_expert, int* idx, float* w) { return native_router_top10(logits, n_expert, idx, w); }

}  // namespace refk
}  // namespace baloon
