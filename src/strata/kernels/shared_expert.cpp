// strata/kernels/shared_expert.cpp — Shared expert kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace strata {
namespace kernels {

// ─── Global configuration state ────────────────────────────────
static bool g_shared_expert_native_bf16 = false;

void shared_expert_set_native_bf16(bool enabled) {
    g_shared_expert_native_bf16 = enabled;
}

// ─── bf16 helpers ──────────────────────────────────────────────
static inline float bf16_to_f32(uint16_t val) {
    uint32_t out = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &out, 4);
    return result;
}

// ─── Code extraction (same as s_gemv) ──────────────────────────
static inline int extract_code(const uint8_t* codes, int64_t elem_idx,
                               int code_bits) {
    switch (code_bits) {
        case 2: {
            const int64_t byte_idx = elem_idx / 4;
            const int bit_pos = static_cast<int>((elem_idx % 4) * 2);
            return (codes[byte_idx] >> bit_pos) & 0x3;
        }
        case 4: {
            const int64_t byte_idx = elem_idx / 2;
            const int bit_pos = static_cast<int>((elem_idx % 2) * 4);
            return (codes[byte_idx] >> bit_pos) & 0xF;
        }
        case 8:
            return codes[elem_idx];
        default:
            return 0;
    }
}

static inline float decode_weight(int code, const SForm& form) {
    switch (form.codebook) {
        case Codebook::Affine:
            return static_cast<float>(code + form.code_bias);
        case Codebook::Iq4Nl:
            return static_cast<float>(kIq4NlTable[code & 0xF]);
        default:
            return static_cast<float>(code + form.code_bias);
    }
}

// ─── GEMV helper for shared expert ──────────────────────────────
// Performs y[o] = sum_i x[i] * W[i,o] where W is in SForm format
// x is provided as float array (pre-dequantized)
static void gemv_sform_f32(const float* x, const uint8_t* codes,
                           const float* scales, const float* offset,
                           float* y, int64_t n_in, int64_t n_out,
                           const SForm& form) {
    if (!x || !codes || !scales || !y) return;

    const int64_t code_bytes_per_row = (n_in * form.code_bits + 7) / 8;
    const int64_t groups_per_row = n_in / form.group_elems;

    for (int64_t o = 0; o < n_out; ++o) {
        float dot = 0.0f;
        const uint8_t* row_codes = codes + o * code_bytes_per_row;
        const float* row_scales = scales + o * groups_per_row;
        const float* row_off = (offset && form.has_offset)
                                ? offset + o * groups_per_row : nullptr;

        for (int64_t i = 0; i < n_in; ++i) {
            const int code = extract_code(row_codes, i, form.code_bits);
            const float cb_val = decode_weight(code, form);
            const int64_t g = i / form.group_elems;
            float w_val = cb_val * row_scales[g];
            if (row_off) w_val += row_off[g];
            dot += x[i] * w_val;
        }
        y[o] = dot;
    }
}

// ─── shared_expert ──────────────────────────────────────────────
void shared_expert(const uint8_t* /*x_q8_0*/, const uint8_t* /*x_q8k*/,
                   const uint16_t* x_bf16,
                   const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales,
                   const float* gate_off,
                   const SForm& up_form,
                   const uint8_t* up_codes, const float* up_scales,
                   const float* up_off,
                   const SForm& down_form,
                   const uint8_t* down_codes, const float* down_scales,
                   const float* down_off,
                   const uint16_t* gate_inp_bf16,
                   float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int /*tpr*/,
                   cudaStream_t /*stream*/,
                   const float* x_f32,
                   const NativeSharedWeights* /*native*/) {
    if (!out || !scratch) return;
    if (n_embd <= 0 || n_ff <= 0) return;

    // Dequantize activation from bf16 to f32 for host reference
    std::vector<float> x_vec(static_cast<size_t>(n_embd), 0.0f);
    if (x_f32) {
        std::memcpy(x_vec.data(), x_f32, static_cast<size_t>(n_embd) * sizeof(float));
    } else if (x_bf16) {
        for (int64_t i = 0; i < n_embd; ++i) {
            x_vec[static_cast<size_t>(i)] = bf16_to_f32(x_bf16[i]);
        }
    }

    float* gate_out = scratch;                   // [n_ff]
    float* up_out   = scratch + n_ff;            // [n_ff]
    float* hidden   = scratch + 2 * n_ff;        // [n_ff]
    float* down_out = scratch + 3 * n_ff;        // [n_embd]

    // Step 1: gate_out = x · W_gate^T
    if (gate_codes && gate_scales) {
        gemv_sform_f32(x_vec.data(), gate_codes, gate_scales, gate_off,
                       gate_out, n_embd, n_ff, gate_form);
    } else {
        std::memset(gate_out, 0, static_cast<size_t>(n_ff) * sizeof(float));
    }

    // Step 2: up_out = x · W_up^T
    if (up_codes && up_scales) {
        gemv_sform_f32(x_vec.data(), up_codes, up_scales, up_off,
                       up_out, n_embd, n_ff, up_form);
    } else {
        std::memset(up_out, 0, static_cast<size_t>(n_ff) * sizeof(float));
    }

    // Step 3: hidden = SiLU(gate_out) * up_out   (SiLU on GATE)
    for (int64_t i = 0; i < n_ff; ++i) {
        const float g = gate_out[i];
        const float silu_g = g / (1.0f + std::exp(-g));
        hidden[i] = silu_g * up_out[i];
    }

    // Step 4: down_out = hidden · W_down^T
    if (down_codes && down_scales) {
        gemv_sform_f32(hidden, down_codes, down_scales, down_off,
                       down_out, n_ff, n_embd, down_form);
    } else {
        std::memset(down_out, 0, static_cast<size_t>(n_embd) * sizeof(float));
    }

    // Step 5: g = sigmoid(x · w_gate_inp) — one scalar per token
    float g_scalar = 0.0f;
    if (gate_inp_bf16) {
        float dot = 0.0f;
        for (int64_t i = 0; i < n_embd; ++i) {
            dot += x_vec[static_cast<size_t>(i)] * bf16_to_f32(gate_inp_bf16[i]);
        }
        g_scalar = 1.0f / (1.0f + std::exp(-dot));
    } else {
        g_scalar = 1.0f;  // no gating
    }

    // Step 6: out = down_out * g
    for (int64_t i = 0; i < n_embd; ++i) {
        out[i] = down_out[i] * g_scalar;
    }
}

// ─── shared_expert_multi ────────────────────────────────────────
void shared_expert_multi(int n_tok, const float* x,
                         const uint16_t* /*x_bf16*/,
                         const NativeSharedWeights& /*nw*/,
                         const uint16_t* gate_inp_bf16,
                         float* /*gate*/, float* /*up*/, float* g,
                         float* out, int64_t n_embd, int64_t n_ff,
                         cudaStream_t /*stream*/) {
    if (!x || !out) return;
    if (n_tok <= 0 || n_tok > 8) return;
    if (n_embd <= 0 || n_ff <= 0) return;

    // Multi-token: process each token independently.
    // In GPU, weights read once for all tokens; host reference is simpler.
    for (int t = 0; t < n_tok; ++t) {
        const float* x_t = x + t * n_embd;
        float* out_t = out + t * n_embd;

        // Compute scalar gate if gate_inp is available
        if (g && gate_inp_bf16) {
            float dot = 0.0f;
            for (int64_t i = 0; i < n_embd; ++i) {
                dot += x_t[i] * bf16_to_f32(gate_inp_bf16[i]);
            }
            g[t] = 1.0f / (1.0f + std::exp(-dot));
        } else if (g) {
            g[t] = 1.0f;
        }

        // For host reference without weights, just zero the output
        // (full path would need the weight pointers from NativeSharedWeights)
        std::memset(out_t, 0, static_cast<size_t>(n_embd) * sizeof(float));
    }
}

// ─── moe_combine ────────────────────────────────────────────────
void moe_combine(const float* parts, const float* weights,
                 const float* shared, float* y,
                 int64_t n_embd, int64_t k, cudaStream_t /*stream*/) {
    if (!parts || !weights || !y) return;
    if (n_embd <= 0 || k <= 0) return;

    for (int64_t d = 0; d < n_embd; ++d) {
        float val = 0.0f;

        // Routed outputs: weighted by router weights
        for (int64_t j = 0; j < k; ++j) {
            val += weights[j] * parts[j * n_embd + d];
        }

        // Shared output: ADDED PLAIN (no weight)
        if (shared) {
            val += shared[d];
        }

        y[d] = val;
    }
}

}  // namespace kernels
}  // namespace strata

