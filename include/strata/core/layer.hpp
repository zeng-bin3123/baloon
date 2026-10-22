#pragma once
// strata/core/layer.hpp — GDN & MoE layer primitives
// Target: Iluvatar BI-V100 (ivcore10)
//
// GDN layer op order (from ref/gdn.py::gdn_decode):
//   qkv = W_qkv · x
//   conv → SiLU  (SiLU on WHOLE conv output BEFORE split)
//   q, k = l2_norm(·)  (v is NOT normalised)
//   β = σ(W_β · x)
//   gate = softplus(W_α · x + dt) · ssm_a
//   recurrent step (gdn_step)
//   y = rms_norm(o, ssm_norm) · σ(z)
//   out = W_ssm_out · y
//
// Critical ordering properties:
//   1. SiLU applied BEFORE the split (to entire 10240-wide conv output)
//   2. v is NOT normalized (l2_norm on q, k only: 2 rows of 16 heads, not 48 of v)
//   3. q scale is 1/√S, applied by CALLER after the norm
//   4. State is transposed: (S, h_v, S) here vs (S, S, h_v) in reference
//
// MoE: out = shared_expert(x) + Σ w_j · expert_{ids[j]}(x)
//   Router: BF16 activation → GEMV → top-k → ids, weights
//   8.1e-3 logit error without BF16 can flip top-10 selection
//
// Activation contract: attn_qkv uses Q8_K, ssm_alpha/ssm_beta use BF16

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "strata/core/types_fwd.hpp"

namespace strata {

// Forward declarations
class WeightTable;

// ─── Configuration selectors ──────────────────────────────────────
// Configure before capture. Idempotent: b = b.
void layer_set_native_bf16(bool enabled);
void layer_set_native_flash_attn_short(bool enabled);

// ─── GdnBuffers ───────────────────────────────────────────────────
// Every buffer one GDN layer needs, carved from an arena ONCE.
// Two quantized activation images per layer (which format a weight
// wants is a property of that weight, not of the layer):
//   x_q8k  ∈ block_q8_K^{n_embd}  — for K-quant weights
//   x_q8_0 ∈ block_q8_0^{n_embd}  — for Q8_0 weights
//   x_bf16 ∈ bf16^{n_embd}        — for BF16 weights
//
// QKV path: qkv, conv_out, h ∈ R^C where C = conv_channels = 10240
// Per-head scalars: alpha, beta, gate ∈ R^{h_v}
// Recurrence: o ∈ R^{h_v · S}, z, y ∈ R^{value_dim}
// State (outlives call): state ∈ R^{S × h_v × S}, conv_state ∈ R^{C × (d_conv - 1)}
struct GdnBuffers {
    uint8_t* x_q8k = nullptr;
    uint8_t* x_q8_0 = nullptr;
    uint16_t* x_bf16 = nullptr;

    // QKV path
    float* qkv = nullptr;
    float* conv_out = nullptr;
    float* h = nullptr;

    // Per-head scalars
    float* alpha = nullptr;
    float* beta = nullptr;
    float* gate = nullptr;

    // Recurrence
    float* o = nullptr;
    float* z = nullptr;
    float* y = nullptr;

    // Output quantized images
    uint8_t* y_q8k = nullptr;
    uint8_t* y_q8_0 = nullptr;

    // Persistent state (outlives individual layer calls)
    float* state = nullptr;
    float* conv_state = nullptr;
};

// Total bytes for one GDN layer's buffers, 16-byte aligned between regions.
uint64_t gdn_buffers_bytes(const Geometry& g);

// Carve device memory at `base` into the fields of `b`.
// Returns number of bytes consumed.
uint64_t gdn_buffers_init(const Geometry& g, void* base, GdnBuffers& b);

// Zero state and conv_state. Matches reference zeros() initialization.
void gdn_buffers_zero_state(const GdnBuffers& b, const Geometry& g, void* stream);

// ─── gdn_layer ────────────────────────────────────────────────────
// One GDN layer: full op sequence from x → out.
// `mixed` = input activation, `out` = output buffer.
bool gdn_layer(const WeightTable& tables, const Geometry& g,
               int64_t layer, const GdnBuffers& b,
               const float* mixed, float* out, void* stream,
               std::string& err);

// ─── Doorbell ─────────────────────────────────────────────────────
// Signal between GPU graph and host expert pool.
struct Doorbell {
    volatile int32_t* h_flag = nullptr;  // host-mapped flag
    int32_t* d_flag = nullptr;           // device-side alias
    int64_t layer = -1;
    int64_t n_expert = 0;
};

// ─── MoEBuffers ───────────────────────────────────────────────────
// Buffers for Mixture-of-Experts block.
//   x_bf16: router activation (BF16 critical: 8.1e-3 error without it)
//   x_f16: shared expert activation
//   logits, ids, weights: router output
//   shared: shared expert output
struct MoEBuffers {
    uint16_t* x_bf16 = nullptr;
    uint16_t* x_f16 = nullptr;
    float* logits = nullptr;
    int32_t* ids = nullptr;
    float* weights = nullptr;
    float* shared = nullptr;
    float* sh_scratch = nullptr;
    uint8_t* x_q8_0 = nullptr;
    uint8_t* x_q8k = nullptr;
};

// ─── moe_layer ────────────────────────────────────────────────────
// Full MoE block: route → dispatch → combine.
// `parts` is (k, n_embd) device f32, filled by CALLER.
bool moe_layer(const WeightTable& tables, const Geometry& g,
               int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, const float* parts, float* out,
               void* stream, std::string& err,
               const Doorbell* db = nullptr);

// ─── moe_route ────────────────────────────────────────────────────
// Split MoE: route only. x → bf16 → router GEMV → top-k → (ids, weights) → ring doorbell.
bool moe_route(const WeightTable& tables, const Geometry& g,
               int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, void* stream, std::string& err,
               const Doorbell* db = nullptr);

// ─── moe_finish ───────────────────────────────────────────────────
// Split MoE: shared expert + combine parts.
// CRITICAL: parts MUST be from THIS layer's router.
bool moe_finish(const WeightTable& tables, const Geometry& g,
                int64_t layer, int64_t k, const MoEBuffers& b,
                const float* x, const float* parts, float* out,
                void* stream, std::string& err);

// ─── moe_shared ───────────────────────────────────────────────────
// Shared expert only: x → b.shared
bool moe_shared(const WeightTable& tables, const Geometry& g,
                int64_t layer, const MoEBuffers& b,
                const float* x, void* stream, std::string& err);

// ─── moe_combine_parts ───────────────────────────────────────────
// out = shared + Σ w_j · parts[j]
bool moe_combine_parts(const Geometry& g, int64_t layer, int64_t k,
                        const MoEBuffers& b, const float* parts,
                        float* out, void* stream, std::string& err);

// ─── Shared-early scheduling (Plan v0.3 P3) ──────────────────────
// Shared expert runs at END of pre[l] (after doorbell ring),
// so GPU computes while host runs CPU pool.  post[l] then only combines.
// Same kernels, same inputs, same stream order → bitwise unchanged result.
void layer_set_shared_early(bool enabled);
bool layer_shared_early();

}  // namespace strata

