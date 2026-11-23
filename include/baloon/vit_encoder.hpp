// vit_encoder.hpp - host reference implementation of the ViT vision encoder for Qwen3.5-MoE.
// Each function maps to one sentence of the visual forward-pass scope:
//   (1) vit_patch_embed    — projects image patches via patch_embed.proj (3-D conv)
//   (2) vit_add_pos_embed  — adds pos_embed
//   (3) vit_block_forward  — passes through one block: norm1 → QKV → proj → norm2 → MLP
//   (4) vit_merger_mlp     — applies the two-layer merger MLP (norm → linear_fc1 → GELU → linear_fc2)
//   (5) vit_encode         — full encoder run to yield per-patch hidden-state vectors
//
// All weights are BF16; all arithmetic uses float32 accumulators.
// Header-only; no external dependencies beyond port_scalar.hpp and the C++ standard library.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "baloon/port_scalar.hpp"

namespace baloon {
namespace vit {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// LayerNorm (scale + bias), standard ViT variant (eps=1e-6).
// x and out may alias only when x == out.
inline void vit_layer_norm(const float* x, int n,
                            const float* scale, const float* bias,
                            float* out, float eps = 1e-6f) {
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) { sum += x[i]; sum2 += (double)x[i] * x[i]; }
    const float mean    = (float)(sum  / n);
    const float var     = (float)(sum2 / n) - mean * mean;
    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; ++i)
        out[i] = (x[i] - mean) * inv_std * scale[i] + bias[i];
}

// BF16 vector → float vector.
inline void vit_load_bf16(const uint16_t* src, int n, float* dst) {
    for (int i = 0; i < n; ++i) dst[i] = refk::f32_from_bf16(src[i]);
}

// BF16 GEMV with optional BF16 bias: out[r] = sum_j w[r*cols+j]*x[j] + (b ? f32(b[r]) : 0).
inline void vit_gemv(const uint16_t* w, const uint16_t* b,
                      const float* x, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        float a = b ? refk::f32_from_bf16(b[r]) : 0.f;
        const uint16_t* wr = w + (int64_t)r * cols;
        for (int j = 0; j < cols; ++j) a += refk::f32_from_bf16(wr[j]) * x[j];
        out[r] = a;
    }
}

// GELU activation (tanh approximation; matches the ViT MLP in Qwen3.5-MoE).
inline float vit_gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ---------------------------------------------------------------------------
// (1) vit_patch_embed — projects image patches via patch_embed.proj (3-D conv)
//
// The 3-D convolution with kernel [out_ch, in_ch, t_p, h_p, w_p] applied to
// non-overlapping patches is equivalent to an independent linear projection of
// each flattened patch vector of length in_flat = in_ch * t_p * h_p * w_p.
//
// w     : BF16 [out_ch, in_flat]       — patch_embed.proj.weight, row-major
// b     : BF16 [out_ch]                — patch_embed.proj.bias
// x     : float[n_patches, in_flat]   — pre-extracted, channel-last patches
// out   : float[n_patches, out_ch]    — patch embeddings (written, not added)
// ---------------------------------------------------------------------------
inline void vit_patch_embed(const uint16_t* w, const uint16_t* b,
                             int out_ch, int in_flat,
                             const float* x, int n_patches,
                             float* out) {
    for (int p = 0; p < n_patches; ++p)
        vit_gemv(w, b, x + (int64_t)p * in_flat, out_ch, in_flat,
                 out + (int64_t)p * out_ch);
}

// ---------------------------------------------------------------------------
// (2) vit_add_pos_embed — adds pos_embed
//
// Looks up each patch's position ID in the learnable table and adds the
// corresponding BF16 row to the patch embedding (in-place).
//
// table   : BF16 [max_pos, d_model]  — model.visual.pos_embed.weight
// d_model : embedding width (1152)
// pos_ids : int[n_patches]           — 2-D position index per patch
// x       : float[n_patches, d_model] — patch embeddings, modified in-place
// ---------------------------------------------------------------------------
inline void vit_add_pos_embed(const uint16_t* table, int d_model,
                               const int* pos_ids, int n_patches,
                               float* x) {
    for (int p = 0; p < n_patches; ++p) {
        const uint16_t* row = table + (int64_t)pos_ids[p] * d_model;
        float* xp = x + (int64_t)p * d_model;
        for (int i = 0; i < d_model; ++i)
            xp[i] += refk::f32_from_bf16(row[i]);
    }
}

// ---------------------------------------------------------------------------
// (3) vit_block_forward — one block: norm1 → QKV → proj → norm2 → MLP
//
// Standard ViT transformer block with pre-norm and residual connections.
// Attention: multi-head scaled dot-product, no causal mask (all patches visible).
// MLP: linear_fc1 → GELU → linear_fc2 (same as Qwen3.5-MoE ViT blocks).
//
// All weight matrices stored as BF16 row-major; arithmetic in float32.
// ---------------------------------------------------------------------------
struct VitBlockWeights {
    const uint16_t* norm1_w;  // BF16 [d_model]          — model.visual.blocks.i.norm1.weight
    const uint16_t* norm1_b;  // BF16 [d_model]          — model.visual.blocks.i.norm1.bias
    const uint16_t* qkv_w;   // BF16 [3*d_model, d_model] — model.visual.blocks.i.attn.qkv.weight
    const uint16_t* qkv_b;   // BF16 [3*d_model]        — model.visual.blocks.i.attn.qkv.bias
    const uint16_t* proj_w;  // BF16 [d_model, d_model] — model.visual.blocks.i.attn.proj.weight
    const uint16_t* proj_b;  // BF16 [d_model]          — model.visual.blocks.i.attn.proj.bias
    const uint16_t* norm2_w; // BF16 [d_model]          — model.visual.blocks.i.norm2.weight
    const uint16_t* norm2_b; // BF16 [d_model]          — model.visual.blocks.i.norm2.bias
    const uint16_t* mlp_w1;  // BF16 [ffn_dim, d_model] — model.visual.blocks.i.mlp.linear_fc1.weight
    const uint16_t* mlp_b1;  // BF16 [ffn_dim]          — model.visual.blocks.i.mlp.linear_fc1.bias
    const uint16_t* mlp_w2;  // BF16 [d_model, ffn_dim] — model.visual.blocks.i.mlp.linear_fc2.weight
    const uint16_t* mlp_b2;  // BF16 [d_model]          — model.visual.blocks.i.mlp.linear_fc2.bias
};

// Passes x through one transformer block in-place.
// scratch: caller-supplied float buffer of at least
//   n_patches*(d_model + 3*d_model + d_model + ffn_dim) + n_patches*n_patches floats.
inline void vit_block_forward(const VitBlockWeights& bw,
                               int d_model, int n_heads, int ffn_dim,
                               int n_patches,
                               float* x,
                               float* scratch) {
    const int head_dim = d_model / n_heads;
    // scratch layout
    float* norm_buf  = scratch;
    float* qkv_buf   = norm_buf + (int64_t)n_patches * d_model;
    float* attn_buf  = qkv_buf  + (int64_t)n_patches * 3 * d_model;
    float* ffn_buf   = attn_buf + (int64_t)n_patches * d_model;
    float* score_buf = ffn_buf  + (int64_t)n_patches * ffn_dim; // n_patches*n_patches floats

    // Load norm1 scale/bias into float
    std::vector<float> ns1(d_model), nb1(d_model);
    vit_load_bf16(bw.norm1_w, d_model, ns1.data());
    vit_load_bf16(bw.norm1_b, d_model, nb1.data());

    // (a) Layer norm 1 — applied to each patch row
    for (int p = 0; p < n_patches; ++p)
        vit_layer_norm(x + (int64_t)p * d_model, d_model,
                       ns1.data(), nb1.data(),
                       norm_buf + (int64_t)p * d_model);

    // (b) QKV projection: qkv_buf[p] = norm_buf[p] @ qkv_w.T + qkv_b
    for (int p = 0; p < n_patches; ++p)
        vit_gemv(bw.qkv_w, bw.qkv_b,
                 norm_buf + (int64_t)p * d_model,
                 3 * d_model, d_model,
                 qkv_buf + (int64_t)p * 3 * d_model);

    // (c) Multi-head scaled dot-product attention
    const float scale = 1.0f / std::sqrt((float)head_dim);
    // zero attn_buf
    std::fill(attn_buf, attn_buf + (int64_t)n_patches * d_model, 0.f);
    for (int h = 0; h < n_heads; ++h) {
        // per-head score matrix: score[i, j] = dot(Q[i,h], K[j,h]) * scale
        for (int i = 0; i < n_patches; ++i) {
            const float* qi = qkv_buf + (int64_t)i * 3 * d_model + h * head_dim;
            for (int j = 0; j < n_patches; ++j) {
                const float* kj = qkv_buf + (int64_t)j * 3 * d_model + d_model + h * head_dim;
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) dot += qi[d] * kj[d];
                score_buf[(int64_t)i * n_patches + j] = dot * scale;
            }
            // softmax over row i
            float* row = score_buf + (int64_t)i * n_patches;
            float mx = row[0];
            for (int j = 1; j < n_patches; ++j) if (row[j] > mx) mx = row[j];
            float s = 0.f;
            for (int j = 0; j < n_patches; ++j) { row[j] = std::exp(row[j] - mx); s += row[j]; }
            for (int j = 0; j < n_patches; ++j) row[j] /= s;
        }
        // accumulate: attn_buf[i, h*head_dim + d] = sum_j score[i,j] * V[j, h*head_dim + d]
        for (int i = 0; i < n_patches; ++i) {
            const float* si = score_buf + (int64_t)i * n_patches;
            float* ai = attn_buf + (int64_t)i * d_model + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                float v = 0.f;
                for (int j = 0; j < n_patches; ++j)
                    v += si[j] * (qkv_buf[(int64_t)j * 3 * d_model + 2 * d_model + h * head_dim + d]);
                ai[d] = v;
            }
        }
    }

    // (d) Output projection + residual: x[p] += attn_buf[p] @ proj_w.T + proj_b
    for (int p = 0; p < n_patches; ++p) {
        vit_gemv(bw.proj_w, bw.proj_b,
                 attn_buf + (int64_t)p * d_model, d_model, d_model,
                 norm_buf + (int64_t)p * d_model); // reuse norm_buf as temp
        for (int i = 0; i < d_model; ++i)
            x[(int64_t)p * d_model + i] += norm_buf[(int64_t)p * d_model + i];
    }

    // (e) Layer norm 2
    std::vector<float> ns2(d_model), nb2(d_model);
    vit_load_bf16(bw.norm2_w, d_model, ns2.data());
    vit_load_bf16(bw.norm2_b, d_model, nb2.data());
    for (int p = 0; p < n_patches; ++p)
        vit_layer_norm(x + (int64_t)p * d_model, d_model,
                       ns2.data(), nb2.data(),
                       norm_buf + (int64_t)p * d_model);

    // (f) MLP: ffn_buf[p] = GELU(norm_buf[p] @ mlp_w1.T + mlp_b1)
    for (int p = 0; p < n_patches; ++p) {
        vit_gemv(bw.mlp_w1, bw.mlp_b1,
                 norm_buf + (int64_t)p * d_model, ffn_dim, d_model,
                 ffn_buf + (int64_t)p * ffn_dim);
        for (int i = 0; i < ffn_dim; ++i)
            ffn_buf[(int64_t)p * ffn_dim + i] = vit_gelu(ffn_buf[(int64_t)p * ffn_dim + i]);
    }

    // (g) MLP output projection + residual: x[p] += ffn_buf[p] @ mlp_w2.T + mlp_b2
    for (int p = 0; p < n_patches; ++p) {
        vit_gemv(bw.mlp_w2, bw.mlp_b2,
                 ffn_buf + (int64_t)p * ffn_dim, d_model, ffn_dim,
                 norm_buf + (int64_t)p * d_model); // reuse as temp
        for (int i = 0; i < d_model; ++i)
            x[(int64_t)p * d_model + i] += norm_buf[(int64_t)p * d_model + i];
    }
}

// Minimum scratch buffer size (in floats) for vit_block_forward.
inline int64_t vit_block_scratch_floats(int d_model, int ffn_dim, int n_patches) {
    return (int64_t)n_patches * (d_model + 3 * d_model + d_model + ffn_dim)
         + (int64_t)n_patches * n_patches;
}

// ---------------------------------------------------------------------------
// (4) vit_merger_mlp — norm → linear_fc1 → GELU → linear_fc2
//
// Spatial merging step that reduces n_patches patch tokens into
// n_tokens = n_patches / merge_sq output tokens of dimension out_dim.
//
// Steps per output token (covering merge_sq = merge_size² input patches):
//   a) Apply LayerNorm to each of the merge_sq patches using merger.norm weights.
//   b) Concatenate the merge_sq normalised patch vectors → merged_dim = merge_sq * d_model.
//   c) Apply linear_fc1 (BF16) → GELU.
//   d) Apply linear_fc2 (BF16) → out_dim (language model hidden size, e.g. 2048).
//
// Patches are assumed laid out in row-major spatial order: the merge_size×merge_size
// neighbourhood of patch p covers indices [p, p+1, …, p+merge_sq-1] in x.
// ---------------------------------------------------------------------------
struct VitMergerWeights {
    const uint16_t* norm_w;  // BF16 [d_model]            — model.visual.merger.norm.weight
    const uint16_t* norm_b;  // BF16 [d_model]            — model.visual.merger.norm.bias
    const uint16_t* fc1_w;   // BF16 [merged_dim, merged_dim] — model.visual.merger.linear_fc1.weight
    const uint16_t* fc1_b;   // BF16 [merged_dim]         — model.visual.merger.linear_fc1.bias
    const uint16_t* fc2_w;   // BF16 [out_dim, merged_dim] — model.visual.merger.linear_fc2.weight
    const uint16_t* fc2_b;   // BF16 [out_dim]            — model.visual.merger.linear_fc2.bias
};

// merge_sq = merge_size * merge_size (e.g. 4 for merge_size=2).
// merged_dim = merge_sq * d_model (e.g. 4*1152 = 4608).
// n_patches must be divisible by merge_sq.
// out: float[n_tokens, out_dim] where n_tokens = n_patches / merge_sq.
// scratch: at least n_tokens * (merged_dim + merged_dim + out_dim) floats.
inline void vit_merger_mlp(const VitMergerWeights& mw,
                            int d_model, int merge_sq, int out_dim,
                            const float* x, int n_patches,
                            float* out,
                            float* scratch) {
    const int merged_dim = merge_sq * d_model;
    const int n_tokens   = n_patches / merge_sq;

    // scratch layout: norm_buf[n_patches*d_model], merged[n_tokens*merged_dim], fc1[n_tokens*merged_dim]
    float* norm_buf = scratch;
    float* merged   = norm_buf + (int64_t)n_patches * d_model;
    float* fc1_buf  = merged   + (int64_t)n_tokens  * merged_dim;

    // Load merger norm scale/bias into float
    std::vector<float> ns(d_model), nb(d_model);
    vit_load_bf16(mw.norm_w, d_model, ns.data());
    vit_load_bf16(mw.norm_b, d_model, nb.data());

    // (a) Apply norm to every patch
    for (int p = 0; p < n_patches; ++p)
        vit_layer_norm(x + (int64_t)p * d_model, d_model,
                       ns.data(), nb.data(),
                       norm_buf + (int64_t)p * d_model);

    // (b) Concatenate merge_sq consecutive normalised patches → one merged_dim vector
    for (int t = 0; t < n_tokens; ++t) {
        float* dst = merged + (int64_t)t * merged_dim;
        for (int s = 0; s < merge_sq; ++s)
            std::memcpy(dst + (int64_t)s * d_model,
                        norm_buf + (int64_t)(t * merge_sq + s) * d_model,
                        d_model * sizeof(float));
    }

    // (c) linear_fc1 → GELU
    for (int t = 0; t < n_tokens; ++t) {
        vit_gemv(mw.fc1_w, mw.fc1_b,
                 merged + (int64_t)t * merged_dim,
                 merged_dim, merged_dim,
                 fc1_buf + (int64_t)t * merged_dim);
        for (int i = 0; i < merged_dim; ++i)
            fc1_buf[(int64_t)t * merged_dim + i] =
                vit_gelu(fc1_buf[(int64_t)t * merged_dim + i]);
    }

    // (d) linear_fc2 → out_dim (per-patch hidden-state vectors)
    for (int t = 0; t < n_tokens; ++t)
        vit_gemv(mw.fc2_w, mw.fc2_b,
                 fc1_buf + (int64_t)t * merged_dim,
                 out_dim, merged_dim,
                 out + (int64_t)t * out_dim);
}

// Minimum scratch buffer size (in floats) for vit_merger_mlp.
inline int64_t vit_merger_scratch_floats(int d_model, int merge_sq, int out_dim, int n_patches) {
    const int merged_dim = merge_sq * d_model;
    const int n_tokens   = n_patches / merge_sq;
    (void)out_dim;
    return (int64_t)n_patches * d_model            // norm_buf
         + (int64_t)n_tokens  * merged_dim         // merged
         + (int64_t)n_tokens  * merged_dim;        // fc1_buf (out is caller's)
}

// ---------------------------------------------------------------------------
// (5) vit_encode — full encoder run to yield per-patch hidden-state vectors
//
// Chains all five sub-operations:
//   patch_embed → add_pos_embed → 27 × block_forward → merger_mlp
//
// Returns n_tokens = n_patches / (merge_size * merge_size).
//
// out is written with n_tokens float vectors of length cfg.out_dim each.
// scratch must hold at least vit_encode_scratch_floats(cfg, n_patches) floats.
// ---------------------------------------------------------------------------
struct VitConfig {
    int d_model    = 1152;   // ViT hidden width
    int depth      = 27;     // number of transformer blocks
    int n_heads    = 16;     // attention heads
    int ffn_dim    = 4304;   // MLP intermediate width
    int in_ch      = 3;      // image channels
    int t_patch    = 2;      // temporal patch size
    int s_patch    = 16;     // spatial patch size
    int merge_size = 2;      // spatial merge factor (merge_size × merge_size patches → 1 token)
    int out_dim    = 2048;   // language model hidden size (merger output)
    int max_pos    = 2304;   // positional embedding table rows
};

// Maximum number of transformer blocks supported.
static constexpr int kVitMaxDepth = 32;

struct VitWeights {
    const uint16_t* patch_embed_w;        // BF16 [d_model, in_ch*t*s*s]
    const uint16_t* patch_embed_b;        // BF16 [d_model]
    const uint16_t* pos_embed;            // BF16 [max_pos, d_model]
    VitBlockWeights blocks[kVitMaxDepth]; // per-block weights (depth ≤ kVitMaxDepth)
    VitMergerWeights merger;
};

// Minimum scratch buffer size (in floats) for vit_encode.
inline int64_t vit_encode_scratch_floats(const VitConfig& cfg, int n_patches) {
    const int in_flat  = cfg.in_ch * cfg.t_patch * cfg.s_patch * cfg.s_patch;
    const int merge_sq = cfg.merge_size * cfg.merge_size;
    // embed_buf + max(block_scratch, merger_scratch)
    const int64_t embed_sz  = (int64_t)n_patches * cfg.d_model;
    const int64_t block_sz  = vit_block_scratch_floats(cfg.d_model, cfg.ffn_dim, n_patches);
    const int64_t merger_sz = vit_merger_scratch_floats(cfg.d_model, merge_sq, cfg.out_dim, n_patches);
    (void)in_flat;
    return embed_sz + (block_sz > merger_sz ? block_sz : merger_sz);
}

// Runs the full ViT encoder on n_patches pre-extracted patches and writes
// n_tokens = n_patches / merge_sq output hidden-state vectors to `out`.
// patches: float[n_patches, in_ch * t_patch * s_patch * s_patch]
// pos_ids: int[n_patches]
// out    : float[n_tokens, cfg.out_dim]
// scratch: float[vit_encode_scratch_floats(cfg, n_patches)]
inline int vit_encode(const VitWeights& w, const VitConfig& cfg,
                       const float* patches, int n_patches, const int* pos_ids,
                       float* out,
                       float* scratch) {
    const int in_flat  = cfg.in_ch * cfg.t_patch * cfg.s_patch * cfg.s_patch;
    const int merge_sq = cfg.merge_size * cfg.merge_size;
    const int n_tokens = n_patches / merge_sq;

    // embed_buf lives at the front of scratch; block/merger scratch follows
    float* embed_buf = scratch;
    float* work      = embed_buf + (int64_t)n_patches * cfg.d_model;

    // (1) Project patches via patch_embed.proj (3-D conv as GEMV)
    vit_patch_embed(w.patch_embed_w, w.patch_embed_b,
                    cfg.d_model, in_flat, patches, n_patches, embed_buf);

    // (2) Add learnable positional embeddings
    vit_add_pos_embed(w.pos_embed, cfg.d_model, pos_ids, n_patches, embed_buf);

    // (3) Pass through depth transformer blocks
    for (int i = 0; i < cfg.depth && i < kVitMaxDepth; ++i)
        vit_block_forward(w.blocks[i], cfg.d_model, cfg.n_heads, cfg.ffn_dim,
                          n_patches, embed_buf, work);

    // (4) Merger MLP: norm → linear_fc1 → GELU → linear_fc2 → out_dim hidden states
    vit_merger_mlp(w.merger, cfg.d_model, merge_sq, cfg.out_dim,
                   embed_buf, n_patches, out, work);

    return n_tokens;
}

}  // namespace vit
}  // namespace baloon

