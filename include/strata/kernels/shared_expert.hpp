#pragma once
// strata/kernels/shared_expert.hpp — Shared expert and MoE combination kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// shared_expert_set_native_bf16: configure BF16/F32 MMVF path.
//   true ⟹ pinned CUDA BF16/F32 MMVF + FP32 sigmoid for scalar gate.
//   Requires optional unrounded x_f32 input.
//
// NativeSharedWeights: optional native GGUF projections.
//   gate_type, up_type, down_type: -1 = absent (fallback to canonical).
//   Each projection independent; any active entry selects pinned FP32 SwiGLU.
//   q8_1: caller-owned scratch, sized by native_q8_1_bytes.
//
// shared_expert:
//   h = SiLU(x · W_gate^T) ⊙ (x · W_up^T)     (SiLU on GATE)
//   h = h · W_down^T
//   g = sigmoid(x · w_gate_inp)                  (one SCALAR per token)
//   out = h * g[:,None]
//   Result ADDED to routed output, not weighted against it.
//   x_q8_0, x_q8k: TWO quantized images of same activation.
//   Which one used: per weight's SForm::act_kind.
//   CRITICAL: ffn_down_shexp is IQ4_NL/Q4_0/Q5_0/Q8_0/Q2_0 (no K-quant)
//   ⟹ contract is Q8_0, not Q8_K; n_in=640 not multiple of 256.
//
// shared_expert_multi: multi-token variant (n_tok ≤ 8).
//   Multi-column MMVQ: weights read once for all tokens.
//   Each token's output bitwise identical to shared_expert on that token.
//
// moe_combine: MoE final combination.
//   y[d] = sum_j w[j] * parts[j,d] + shared[d]
//   ROUTED outputs: router-WEIGHTED.  SHARED output: ADDED PLAIN (no weight).
//   Weights already renormalised (sum to 1).  Do NOT normalise again.

#include <cuda_runtime.h>
#include <cstdint>

#include "strata/kernels/s_gemv.hpp"

namespace strata {
namespace kernels {

// ─── Configuration ──────────────────────────────────────────────
// shared_expert_set_native_bf16: configure BF16/F32 MMVF path.
//   true ⟹ pinned CUDA BF16/F32 MMVF + FP32 sigmoid for scalar gate.
//   Requires optional unrounded x_f32 input.
void shared_expert_set_native_bf16(bool enabled);

// ─── NativeSharedWeights ────────────────────────────────────────
// Optional native GGUF projections for shared expert.
//
// gate_type, up_type, down_type: GGUF type id (-1 = absent, fallback to canonical).
// Each projection independent; any active entry selects pinned FP32 SwiGLU.
// q8_1: caller-owned scratch, sized by native_q8_1_bytes.
struct NativeSharedWeights {
    int gate_type = -1;
    int up_type   = -1;
    int down_type = -1;
    const void* gate_data = nullptr;
    const void* up_data   = nullptr;
    const void* down_data = nullptr;
    void* q8_1 = nullptr;
};

// ─── shared_expert ──────────────────────────────────────────────
// The shared expert (host-callable).
//
// Math:
//   h = SiLU(x · W_gate^T) ⊙ (x · W_up^T)     (SiLU on GATE)
//   h = h · W_down^T
//   g = sigmoid(x · w_gate_inp)                  (one SCALAR per token)
//   out = h * g[:,None]
//
// Result ADDED to routed output, not weighted against it.
//
// x_q8_0, x_q8k: TWO quantized images of same activation.
// Which one used: per weight's SForm::act_kind.
//
// CRITICAL: ffn_down_shexp is IQ4_NL/Q4_0/Q5_0/Q8_0/Q2_0 (no K-quant)
//   ⟹ contract is Q8_0, not Q8_K; n_in=640 not multiple of 256.
void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k,
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
                   int64_t n_embd, int64_t n_ff, int tpr,
                   cudaStream_t stream,
                   const float* x_f32 = nullptr,
                   const NativeSharedWeights* native = nullptr);

// ─── shared_expert_multi ────────────────────────────────────────
// Multi-token variant (n_tok ≤ 8, verify window).
//
// Multi-column MMVQ: weights read once for all tokens.
// Each token's output bitwise identical to shared_expert on that token.
void shared_expert_multi(int n_tok, const float* x,
                         const uint16_t* x_bf16,
                         const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16,
                         float* gate, float* up, float* g,
                         float* out, int64_t n_embd, int64_t n_ff,
                         cudaStream_t stream);

// ─── moe_combine ────────────────────────────────────────────────
// MoE final combination.
//
//   y[d] = sum_j w[j] * parts[j,d] + shared[d]
//
// ROUTED outputs: router-WEIGHTED.  SHARED output: ADDED PLAIN (no weight).
// Weights from router are ALREADY renormalised (sum to 1).
// Do NOT normalise again here.
//
// parts ∈ R^{k × n_embd},  w ∈ R^k,  shared ∈ R^{n_embd} ∪ {null}.
void moe_combine(const float* parts, const float* weights,
                 const float* shared, float* y,
                 int64_t n_embd, int64_t k, cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

