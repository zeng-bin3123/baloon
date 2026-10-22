// strata/kernels/ple.cpp — PLE (Piecewise Linear Embedding) kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.

#include "strata/kernels/ple.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

namespace strata {
namespace kernels {

// ─── Global configuration state ────────────────────────────────────
static bool g_ple_native_bf16 = false;
static bool g_ple_native_postops = false;
static PleShapes g_ple_shapes;

void ple_set_shapes(const PleShapes& shapes) {
    g_ple_shapes = shapes;
}

const PleShapes& ple_get_shapes() {
    return g_ple_shapes;
}

void ple_set_native_bf16(bool enabled) {
    g_ple_native_bf16 = enabled;
}

void ple_set_native_postops(bool enabled) {
    g_ple_native_postops = enabled;
}

bool ple_native_postops_enabled() {
    return g_ple_native_postops;
}

bool ple_block_available() {
    // On host fallback, PLE block is always available.
    return true;
}

// ─── bf16 helper ───────────────────────────────────────────────────
static inline float bf16_to_f32(uint16_t val) {
    uint32_t out = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &out, 4);
    return result;
}

// ─── grouped_norm ──────────────────────────────────────────────────
// Groups of elements are normalized independently, then scaled by gamma.
// hc_dim = hc * group_size.  Each group is L2-normalized, then scaled.
// NO (1+w) folding: output = normalized * gamma.
static void grouped_norm(const float* input, const float* gamma,
                         int64_t hc_dim, int64_t hc, float* output) {
    if (hc <= 0 || hc_dim <= 0) return;
    const int64_t group_size = hc_dim / hc;

    for (int64_t c = 0; c < hc; ++c) {
        const float* in_group  = input  + c * group_size;
        const float* gam_group = gamma  + c * group_size;
        float* out_group       = output + c * group_size;

        // L2 norm of the group
        float sq_sum = 0.0f;
        for (int64_t d = 0; d < group_size; ++d) {
            sq_sum += in_group[d] * in_group[d];
        }
        const float inv_norm = 1.0f / std::sqrt(sq_sum + 1e-6f);

        for (int64_t d = 0; d < group_size; ++d) {
            out_group[d] = in_group[d] * inv_norm * gam_group[d];
        }
    }
}

// ─── Scratch sizing ────────────────────────────────────────────────
// We need: key[hc_dim] + value[n_embd] + gate[hc] + gated[hc_dim]
//        + normalized[hc_dim] + conv[hc_dim] + matvec temp
// Conservative: 512K covers all intermediate buffers for reasonable dims.
static constexpr uint64_t PLE_SCRATCH_BYTES = 512 * 1024;

uint64_t ple_block_scratch_bytes() {
    return PLE_SCRATCH_BYTES;
}

// ─── ple_block ─────────────────────────────────────────────────────
// Process one PLE token.  Full math from the theory specification.
void ple_block(const float* emb, const float* hidden,
               const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, cudaStream_t /*stream*/) {
    if (!emb || !hidden || !hist_rows || !out.result || !scratch) return;

    const PleShapes& s = g_ple_shapes;
    if (!s.valid()) {
        // Shapes not configured — minimal passthrough for safety.
        out.result[0] = hidden[0];
        return;
    }

    const int64_t n_embd = s.n_embd;
    const int64_t hc     = s.hc;
    const int64_t hc_dim = s.hc_dim;
    const int64_t gsize  = s.group_size();
    const int64_t ng     = s.ng_hist;

    // Scratch layout
    auto* scratch_bytes = static_cast<float*>(scratch);
    float* key_buf   = scratch_bytes;
    float* value_buf = key_buf + hc_dim;
    float* gate_buf  = value_buf + n_embd;
    float* gated_buf = gate_buf + hc;
    float* norm_buf  = gated_buf + hc_dim;
    float* conv_buf  = norm_buf + hc_dim;

    // ── Step 1: key = grouped_norm(W_key · emb, gamma_key) ──
    // W_key projection: key_buf = W_key · emb (using S2 codes or bf16 alternate)
    // For host reference with S2 codes + scales:
    if (w.key_bf16) {
        // BF16 path: key_buf[i] = sum_d w_key_bf16[i * n_embd + d] * emb[d]
        for (int64_t i = 0; i < hc_dim; ++i) {
            float dot = 0.0f;
            for (int64_t d = 0; d < n_embd; ++d) {
                dot += bf16_to_f32(w.key_bf16[i * n_embd + d]) * emb[d];
            }
            key_buf[i] = dot;
        }
    } else if (w.key_codes && w.key_scales) {
        // S2 quantized path: identity codebook dequant + matvec
        // For small test dimensions, do element-wise extraction
        for (int64_t i = 0; i < hc_dim; ++i) {
            float dot = 0.0f;
            for (int64_t d = 0; d < n_embd; ++d) {
                const int64_t idx = i * n_embd + d;
                const int64_t byte_idx = idx / 4;  // 2-bit codes, 4 per byte
                const int bit_pos = static_cast<int>((idx % 4) * 2);
                int code = 0;
                if (w.key_codes) {
                    code = (w.key_codes[byte_idx] >> bit_pos) & 0x3;
                }
                const int64_t scale_group = idx / 32;  // typical group size
                float scale = w.key_scales[scale_group];
                dot += static_cast<float>(code) * scale * emb[d];
            }
            key_buf[i] = dot;
        }
    } else {
        // No weights — zero key
        std::memset(key_buf, 0, static_cast<size_t>(hc_dim) * sizeof(float));
    }
    // Apply grouped_norm with gamma_key
    if (w.norm_key) {
        grouped_norm(key_buf, w.norm_key, hc_dim, hc, key_buf);
    }
    if (out.key) std::memcpy(out.key, key_buf, static_cast<size_t>(hc_dim) * sizeof(float));

    // ── Step 2: query = grouped_norm(hidden, gamma_query) ──
    float* query_buf = conv_buf + hc_dim;  // reuse more scratch
    if (w.norm_query) {
        grouped_norm(hidden, w.norm_query, hc_dim, hc, query_buf);
    } else {
        std::memcpy(query_buf, hidden, static_cast<size_t>(hc_dim) * sizeof(float));
    }

    // ── Step 3: value = W_value · emb ──
    if (w.value_bf16) {
        for (int64_t i = 0; i < n_embd; ++i) {
            float dot = 0.0f;
            for (int64_t d = 0; d < n_embd; ++d) {
                dot += bf16_to_f32(w.value_bf16[i * n_embd + d]) * emb[d];
            }
            value_buf[i] = dot;
        }
    } else {
        // No value weights — pass through emb
        std::memcpy(value_buf, emb, static_cast<size_t>(n_embd) * sizeof(float));
    }
    if (out.value) std::memcpy(out.value, value_buf, static_cast<size_t>(n_embd) * sizeof(float));

    // ── Step 4: s[c] = (sum_d key[c,d] * query[c,d]) / sqrt(n_embd) ──
    // ── Step 5: gate[c] = sigmoid(sign(s[c]) * sqrt(max(|s[c]|, 1e-6))) ──
    const float inv_sqrt_embd = 1.0f / std::sqrt(static_cast<float>(n_embd));
    for (int64_t c = 0; c < hc; ++c) {
        float dot = 0.0f;
        for (int64_t d = 0; d < gsize; ++d) {
            dot += key_buf[c * gsize + d] * query_buf[c * gsize + d];
        }
        const float sc = dot * inv_sqrt_embd;
        // signed sqrt gate: sign(s) * sqrt(max(|s|, 1e-6))
        const float abs_s = std::fabs(sc);
        const float signed_sqrt = (sc >= 0.0f ? 1.0f : -1.0f)
                                  * std::sqrt(std::max(abs_s, 1e-6f));
        gate_buf[c] = 1.0f / (1.0f + std::exp(-signed_sqrt));
    }
    if (out.gate) std::memcpy(out.gate, gate_buf, static_cast<size_t>(hc) * sizeof(float));

    // ── Step 6: gated[c,d] = value[d] * gate[c] ──
    for (int64_t c = 0; c < hc; ++c) {
        for (int64_t d = 0; d < gsize; ++d) {
            // value is n_embd-wide; distribute across groups
            // Each group c has gsize elements at offset c*gsize
            // value[d] in the context of gated[c,d] uses the same d index
            int64_t val_idx = c * gsize + d;
            float val = (val_idx < n_embd) ? value_buf[val_idx] : 0.0f;
            gated_buf[c * gsize + d] = val * gate_buf[c];
        }
    }
    if (out.gated) std::memcpy(out.gated, gated_buf, static_cast<size_t>(hc_dim) * sizeof(float));

    // ── Step 7: normalized = grouped_norm(gated, gamma_conv) ──
    if (w.norm_conv) {
        grouped_norm(gated_buf, w.norm_conv, hc_dim, hc, norm_buf);
    } else {
        std::memcpy(norm_buf, gated_buf, static_cast<size_t>(hc_dim) * sizeof(float));
    }
    if (out.normalized) std::memcpy(out.normalized, norm_buf, static_cast<size_t>(hc_dim) * sizeof(float));

    // ── Step 8: conv on NORMALIZED (not gated!) ──
    // conv[c] = sum_k W_conv1d[k,c] * padded[hist + t]_{k,c}
    // History layout (ROW-FASTEST): hist[row + ng_hist * channel]
    // Padding: [hist rows | normalized] forms the input window
    // W_conv1d: F16, GGML-native layout kW[k + PLE_CONV_KERNEL * c]
    for (int64_t c = 0; c < hc_dim; ++c) {
        float sum = 0.0f;
        // The conv window has PLE_CONV_KERNEL taps.
        // Last (ng_hist) history rows + current normalized form the padded input.
        // We use the last (PLE_CONV_KERNEL - 1) history rows + current normalized.
        for (int k = 0; k < PLE_CONV_KERNEL; ++k) {
            float input_val;
            if (k < PLE_CONV_KERNEL - 1) {
                // Read from history: use the last (PLE_CONV_KERNEL-1) rows
                // History row index: ng - (PLE_CONV_KERNEL - 1) + k
                int64_t row = ng - (PLE_CONV_KERNEL - 1) + k;
                if (row >= 0 && row < ng) {
                    // ROW-FASTEST: hist[row + ng_hist * channel]
                    input_val = hist_rows[row + ng * c];
                } else {
                    input_val = 0.0f;  // padding
                }
            } else {
                // Last tap: current normalized value
                input_val = norm_buf[c];
            }

            // Weight: GGML-native kW[k + PLE_CONV_KERNEL * c]
            float w_val = 0.0f;
            if (w.conv1d_f16) {
                w_val = f32_from_f16(w.conv1d_f16[k + PLE_CONV_KERNEL * c]);
            }
            sum += input_val * w_val;
        }
        conv_buf[c] = sum;
    }
    if (out.conv) std::memcpy(out.conv, conv_buf, static_cast<size_t>(hc_dim) * sizeof(float));

    // ── Step 9: result = hidden + gated + silu(conv) ──
    for (int64_t i = 0; i < hc_dim; ++i) {
        // SiLU: x / (1 + exp(-x))
        const float cv = conv_buf[i];
        const float silu_cv = cv / (1.0f + std::exp(-cv));
        out.result[i] = hidden[i] + gated_buf[i] + silu_cv;
    }
}

// ─── ple_history_advance ───────────────────────────────────────────
// FIFO shift in row-fastest layout.
//
//   hist[r, c] ← hist[r+1, c]  for r in [0, ng_hist-2], c in [0, hc_dim)
//   hist[ng_hist-1, c] ← normalized[c]   for all c
//
// Row-fastest layout: hist[row + ng_hist * channel]
// Dimensions from ple_get_shapes().
void ple_history_advance(float* hist, const float* normalized,
                         cudaStream_t /*stream*/) {
    if (!hist || !normalized) return;

    const PleShapes& s = g_ple_shapes;
    if (!s.valid()) return;

    const int64_t ng     = s.ng_hist;
    const int64_t hc_dim = s.hc_dim;

    // For each channel, shift the FIFO: drop oldest row, append normalized
    // Layout: hist[row + ng_hist * channel]
    for (int64_t c = 0; c < hc_dim; ++c) {
        // Shift rows 1..ng-1 down to 0..ng-2
        for (int64_t r = 0; r < ng - 1; ++r) {
            hist[r + ng * c] = hist[(r + 1) + ng * c];
        }
        // Write normalized as newest row
        hist[(ng - 1) + ng * c] = normalized[c];
    }
}

}  // namespace kernels
}  // namespace strata
