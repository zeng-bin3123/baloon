#pragma once
// strata/kernels/ple.hpp — PLE (Piecewise Linear Embedding) block kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// PLE block: processes one token through a key-value gated convolution block.
//
//   key = grouped_norm(W_key · emb, gamma_key)
//   query = grouped_norm(hidden, gamma_query)
//   value = W_value · emb
//   s[c] = (sum_d key[c,d] * query[c,d]) / sqrt(n_embd)
//   gate[c] = sigmoid(sign(s[c]) * sqrt(max(|s[c]|, 1e-6)))
//   gated[c,d] = value[d] * gate[c]
//   normalized = grouped_norm(gated, gamma_conv)
//   conv[c] = sum_k W_conv1d[k,c] * padded[hist + t]_{k,c}
//   result = hidden + gated + silu(conv)
//
// CRITICAL: conv input is `normalized` (grouped_norm of gated), NOT gated itself.
// CRITICAL: History layout is ROW-FASTEST: hist[row + NG_HIST * channel].
//
// ple_history_advance: FIFO shift of history buffer (in-place, capture-safe).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace strata {
namespace kernels {

// PLE constants
constexpr int PLE_CONV_KERNEL = 4;   // conv1d kernel width
constexpr int PLE_NG_HIST = 3;       // conv history rows (d_conv - 1)

// ─── PleWeights ────────────────────────────────────────────────────
// Weight references for one PLE block.
//
// W_key: S2-quantized [n_embd, hc_dim], codes + F32 scales (widened from FP16 at load)
// W_value: BF16 [n_embd, n_embd]
// gamma_key, gamma_query, gamma_conv: F32 [hc_dim], NO (1+w) folding
// W_conv1d: F16 [PLE_CONV_KERNEL × hc_dim], GGML-native layout kW[k + 4c]
//
// Optional alternate key paths: native Q2_0, or BF16 [n_embd, hc_dim].
struct PleWeights {
    const uint8_t*  key_codes      = nullptr;   // S2 codes
    const float*    key_scales     = nullptr;   // S2 F32 scales (widened from fp16)
    const uint16_t* value_bf16     = nullptr;   // BF16 [n_embd, n_embd]
    const float*    norm_key       = nullptr;   // gamma_key, F32 [hc_dim]
    const float*    norm_query     = nullptr;   // gamma_query, F32 [hc_dim]
    const float*    norm_conv      = nullptr;   // gamma_conv, F32 [hc_dim]
    const uint16_t* conv1d_f16     = nullptr;   // F16 [PLE_CONV_KERNEL × hc_dim]

    // Optional alternate key paths
    const void*     key_native_data   = nullptr;
    int             key_native_type   = -1;     // -1 = not used
    void*           key_native_q8_1   = nullptr;
    const uint16_t* key_bf16          = nullptr; // BF16 alternate
};

// ─── PleOut ────────────────────────────────────────────────────────
// Intermediate and final outputs of the PLE block pipeline.
// Any pointer may be null (caller only requesting final result).
struct PleOut {
    float* key        = nullptr;   // [hc_dim]
    float* value      = nullptr;   // [n_embd]
    float* gate       = nullptr;   // [hc] (one per channel group)
    float* gated      = nullptr;   // [hc_dim]
    float* normalized = nullptr;   // [hc_dim]
    float* conv       = nullptr;   // [hc_dim]
    float* result     = nullptr;   // [hc_dim] (final output)
};

// ─── ple_block ─────────────────────────────────────────────────────
// Process one PLE token.
//
// Given:
//   emb ∈ R^{n_embd} (n-gram embedding)
//   hidden ∈ R^{hc_dim} (residual input)
//   hist_rows ∈ R^{NG_HIST × hc_dim} (row-fastest, oldest first)
//
// Steps:
//   1. key = grouped_norm(W_key · emb, gamma_key)
//   2. query = grouped_norm(hidden, gamma_query)
//   3. value = W_value · emb
//   4. s[c] = (sum_d key[c,d] · query[c,d]) / sqrt(n_embd)
//   5. gate[c] = sigmoid(sign(s[c]) · sqrt(max(|s[c]|, 1e-6)))
//   6. gated[c,d] = value[d] * gate[c]
//   7. normalized = grouped_norm(gated, gamma_conv)
//   8. conv[c] = sum_k W_conv1d[k,c] · padded[hist + t]_{k,c}
//      CRITICAL: conv input is `normalized`, NOT gated itself.
//   9. result = hidden + gated + silu(conv)
//
// emb: [n_embd] float
// hidden: [hc_dim] float
// hist_rows: [NG_HIST × hc_dim] float, ROW-FASTEST layout
// w: PLE weights
// out: PLE outputs (any pointer may be null)
// scratch: device scratch buffer (from ple_block_scratch_bytes)
uint64_t ple_block_scratch_bytes();

void ple_block(const float* emb, const float* hidden,
               const float* hist_rows, const PleWeights& w,
               PleOut& out, void* scratch, cudaStream_t stream);

// ─── ple_history_advance ───────────────────────────────────────────
// FIFO shift in row-fastest layout:
//   hist[r, c] ← hist[r+1, c]  for r in [0, NG_HIST-2], c in [0, hc_dim)
//   hist[NG_HIST-1, c] ← normalized[c]   for all c
//
// One thread per channel; in-place, allocation-free, capture-safe.
// Precondition: hist and normalized are nonnull, disjoint device regions.
void ple_history_advance(float* hist, const float* normalized,
                         cudaStream_t stream);

// ─── Configuration flags ───────────────────────────────────────────
// Set before session capture; captured graphs retain selection.

// Selects pinned CUDA BF16/F32 MMVF for PLE value projection.
void ple_set_native_bf16(bool enabled);

// Selects pinned CUDA post-projection kernels.
void ple_set_native_postops(bool enabled);

// Query current native postops state.
bool ple_native_postops_enabled();

// Whether PLE block is available on this hardware.
bool ple_block_available();

}  // namespace kernels
}  // namespace strata

