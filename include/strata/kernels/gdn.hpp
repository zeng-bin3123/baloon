#pragma once
// strata/kernels/gdn.hpp — GDN (Gated Delta Net) kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// GdnShapes: (S, h_k, h_v) with h_v ≡ 0 (mod h_k).
//   Real model: S=128, h_k=16, h_v=48.
//   Derived: C = 2*S*h_k + S*h_v = 10240 channels.
//
// gdn_step: delta-net recurrence for one token.
//   State layout: (S, h_v, S) with j fastest.
//   Invariant: decay BEFORE rank-1 update (Property 5).
//   Invariant: head pairing is MODULO: h_hat = h % h_k (Property 8/11).
//   Parallelism: each (j, h) column is independent.
//
// gdn_conv_step: 1-D depthwise convolution, GGML-native weight layout.
//   Postcondition: conv_state shifted (FIFO).
//
// gdn_l2_norm: per-row L2 normalization.
//   CRITICAL: eps added to SQUARED NORM (sum), NOT to mean. No 1/width.
//
// gdn_beta_gate: in-place sigmoid on beta (before gdn_step).
//
// gdn_out_norm: per-head RMS norm + ssm_norm + sigmoid(z) gating.
//   CRITICAL: uses SIGMOID (not SiLU) for z gating. This is qwen4exp.

#include <cuda_runtime.h>
#include <cstdint>

namespace strata {
namespace kernels {

// ─── GdnShapes ─────────────────────────────────────────────────────
// (S, h_k, h_v) ∈ Z_{>0}^3, h_v ≡ 0 (mod h_k).
// Real artifact: S=128, h_k=16, h_v=48.
// Derived: C = 2*S*h_k + S*h_v = 10240.
struct GdnShapes {
    int64_t S   = 0;    // state size (recurrence width)
    int64_t h_k = 0;    // number of key heads
    int64_t h_v = 0;    // number of value heads (must be divisible by h_k)

    // Total channels: 2*S*h_k + S*h_v
    int64_t channels() const { return 2 * S * h_k + S * h_v; }

    // Validate: S > 0, h_k > 0, h_v > 0, h_v % h_k == 0
    bool valid() const {
        return S > 0 && h_k > 0 && h_v > 0 && (h_v % h_k == 0);
    }
};

// ─── gdn_step ──────────────────────────────────────────────────────
// Delta-net recurrence for one token.  State layout: (S, h_v, S) with j fastest.
//
// For each head h in [0, h_v), with paired key head h_hat = h % h_k:
//   1. dec_h = exp(gate_h)
//   2. state[i, h, j] *= dec_h         ∀ i, j           (DECAY FIRST)
//   3. sk[h, j] = sum_i state[i,h,j] * k[h_hat, i]
//   4. d[h, j] = (v[h, j] - sk[h, j]) * beta_h
//   5. state[i, h, j] += k[h_hat, i] * d[h, j]          (rank-1 update)
//   6. o[h, j] = sum_i state[i,h,j] * q[h_hat, i]
//
// state: [S × h_v × S] float (persistent across tokens)
// q: [h_k × S] float
// k: [h_k × S] float
// v: [h_v × S] float
// gate: [h_v] float (from gdn_gate)
// beta: [h_v] float (AFTER sigmoid via gdn_beta_gate)
// o: [h_v × S] float (output)
void gdn_step(float* state, const float* q, const float* k,
              const float* v, const float* gate, const float* beta,
              float* o, const GdnShapes& shapes, cudaStream_t stream);

// ─── gdn_conv_step ─────────────────────────────────────────────────
// 1-D depthwise convolution for one token.
//
//   out[c] = sum_{i=0}^{d_conv-1} inp[i][c] * kW[c * d_conv + i]
//
// where inp = [conv_state | x] (oldest state rows, then new input x).
// Weight layout: GGML-native, kW[c * d_conv + i] = tap i of channel c.
// Postcondition: conv_state shifted — oldest row dropped, x becomes newest.
//
// conv_state: [(d_conv-1) × channels] float (mutated in place)
// x: [channels] float (new input)
// kW: [channels × d_conv] float (GGML layout)
// out: [channels] float (convolution output)
void gdn_conv_step(float* conv_state, const float* x, const float* kW,
                   float* out, int64_t channels, int64_t d_conv,
                   cudaStream_t stream);

// ─── gdn_l2_norm ───────────────────────────────────────────────────
// Per-row L2 normalization (in-place).
//
//   x[r, :] = x[r, :] / sqrt(sum_j x[r,j]^2 + eps)
//
// CRITICAL: eps is added to the SQUARED NORM (sum), NOT to the mean.
// There is NO division by width — this is L2 norm, not RMS norm.
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps,
                 cudaStream_t stream);

// ─── gdn_beta_gate ─────────────────────────────────────────────────
// In-place sigmoid on beta, applied BEFORE gdn_step.
//   beta[h] = 1 / (1 + exp(-beta[h])),  h in [0, h_v)
// Without this, the delta-net write strength is unbounded and signed.
void gdn_beta_gate(float* beta, int64_t h_v, cudaStream_t stream);

// ─── gdn_out_norm ──────────────────────────────────────────────────
// Per-head output normalization with sigmoid gating.
//
//   y[h, :] = rms_norm(o[h, :], eps) * ssm_norm[h, :] * sigmoid(z[h, :])
//
// rms_norm(v, eps) = v / sqrt((1/|v|) * sum_i v_i^2 + eps)
// CRITICAL: uses SIGMOID (not SiLU) for z gating.  This is qwen4exp.
// Norm is per-ROW (per head), not over the whole (h_v, S) tensor.
//
// o: [h_v × S] float (output from gdn_step)
// z: [h_v × S] float (gate input)
// ssm_norm: [h_v × S] float (learned norm weights)
// out: [h_v × S] float (normalized gated output)
void gdn_out_norm(const float* o, const float* z, const float* ssm_norm,
                  float* out, int64_t h_v, int64_t S, float eps,
                  cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

