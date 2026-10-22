// strata/kernels/gdn.cpp — GDN (Gated Delta Net) kernel implementations
// Target: Iluvatar BI-V100 (ivcore10)
//
// Host-side reference implementations.  On real hardware, dispatch to CUDA kernels.

#include "strata/kernels/gdn.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>

namespace strata {
namespace kernels {

// ─── gdn_step ──────────────────────────────────────────────────────
// Delta-net recurrence for one token.
//
// State layout: (S, h_v, S) with j fastest.
//   state[i, h, j] is at index: i * (h_v * S) + h * S + j
//
// Invariant: decay BEFORE rank-1 update (Property 5).
// Invariant: head pairing is MODULO: h_hat = h % h_k (Property 8/11).
// Parallelism: each (j, h) column is independent; no cross-column coupling.
void gdn_step(float* state, const float* q, const float* k,
              const float* v, const float* gate, const float* beta,
              float* o, const GdnShapes& s, cudaStream_t /*stream*/) {
    if (!state || !q || !k || !v || !gate || !beta || !o) return;
    if (!s.valid()) return;

    const int64_t S   = s.S;
    const int64_t h_k = s.h_k;
    const int64_t h_v = s.h_v;

    // For each value head h:
    for (int64_t h = 0; h < h_v; ++h) {
        const int64_t h_hat = h % h_k;  // paired key head (MODULO, not interleave)
        const float dec = std::exp(gate[h]);

        // Step 1-2: Apply decay to state
        //   state[i, h, j] *= dec  for all i, j
        for (int64_t i = 0; i < S; ++i) {
            for (int64_t j = 0; j < S; ++j) {
                state[i * (h_v * S) + h * S + j] *= dec;
            }
        }

        // Step 3: sk[h, j] = sum_i state[i,h,j] * k[h_hat, i]
        // Step 4: d[h, j] = (v[h, j] - sk[h, j]) * beta[h]
        // Step 5: state[i, h, j] += k[h_hat, i] * d[h, j]
        for (int64_t j = 0; j < S; ++j) {
            // Compute sk
            float sk = 0.0f;
            for (int64_t i = 0; i < S; ++i) {
                sk += state[i * (h_v * S) + h * S + j] * k[h_hat * S + i];
            }
            // Compute delta
            const float d = (v[h * S + j] - sk) * beta[h];
            // Rank-1 update
            for (int64_t i = 0; i < S; ++i) {
                state[i * (h_v * S) + h * S + j] += k[h_hat * S + i] * d;
            }
        }

        // Step 6: o[h, j] = sum_i state[i,h,j] * q[h_hat, i]
        for (int64_t j = 0; j < S; ++j) {
            float sum = 0.0f;
            for (int64_t i = 0; i < S; ++i) {
                sum += state[i * (h_v * S) + h * S + j] * q[h_hat * S + i];
            }
            o[h * S + j] = sum;
        }
    }
}

// ─── gdn_conv_step ─────────────────────────────────────────────────
// 1-D depthwise convolution for one token.
//
// Weight layout: GGML-native, kW[c * d_conv + i] = tap i of channel c.
// Postcondition: conv_state shifted (oldest row dropped, x becomes newest).
void gdn_conv_step(float* conv_state, const float* x, const float* kW,
                   float* out, int64_t channels, int64_t d_conv,
                   cudaStream_t /*stream*/) {
    if (!conv_state || !x || !kW || !out || channels <= 0 || d_conv <= 1)
        return;

    const int64_t hist_rows = d_conv - 1;

    for (int64_t c = 0; c < channels; ++c) {
        float sum = 0.0f;

        // Convolve over history rows (taps 0..hist_rows-1)
        for (int64_t i = 0; i < hist_rows; ++i) {
            sum += conv_state[i * channels + c] * kW[c * d_conv + i];
        }
        // Tap for the new input x (tap index = hist_rows = d_conv-1)
        sum += x[c] * kW[c * d_conv + hist_rows];

        out[c] = sum;
    }

    // Shift conv_state: drop oldest row, append x as newest
    // Move rows: row[i] ← row[i+1] for i in [0, hist_rows-2]
    if (hist_rows > 1) {
        std::memmove(conv_state, conv_state + channels,
                     static_cast<size_t>((hist_rows - 1) * channels) * sizeof(float));
    }
    // Write x as newest row
    std::memcpy(conv_state + (hist_rows - 1) * channels, x,
                static_cast<size_t>(channels) * sizeof(float));
}

// ─── gdn_l2_norm ───────────────────────────────────────────────────
// Per-row L2 normalization (in-place).
//   x[r, :] = x[r, :] / sqrt(sum_j x[r,j]^2 + eps)
//
// CRITICAL: eps added to SQUARED NORM (sum), NOT to the mean.
// NO division by width — this is L2 norm, NOT RMS norm.
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps,
                 cudaStream_t /*stream*/) {
    if (!x || rows <= 0 || cols <= 0) return;

    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;

        float sq_sum = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            sq_sum += row[c] * row[c];
        }
        // eps added to squared norm, NOT to mean
        const float inv_norm = 1.0f / std::sqrt(sq_sum + eps);

        for (int64_t c = 0; c < cols; ++c) {
            row[c] *= inv_norm;
        }
    }
}

// ─── gdn_beta_gate ─────────────────────────────────────────────────
// In-place sigmoid.
void gdn_beta_gate(float* beta, int64_t h_v, cudaStream_t /*stream*/) {
    if (!beta || h_v <= 0) return;

    for (int64_t h = 0; h < h_v; ++h) {
        beta[h] = 1.0f / (1.0f + std::exp(-beta[h]));
    }
}

// ─── gdn_out_norm ──────────────────────────────────────────────────
// Per-head output normalization with sigmoid gating.
//
//   y[h, :] = rms_norm(o[h, :], eps) * ssm_norm[h, :] * sigmoid(z[h, :])
//
// CRITICAL: uses SIGMOID (not SiLU) for z gating.
// Norm is per-ROW (per head).
void gdn_out_norm(const float* o, const float* z, const float* ssm_norm,
                  float* out, int64_t h_v, int64_t S, float eps,
                  cudaStream_t /*stream*/) {
    if (!o || !z || !ssm_norm || !out || h_v <= 0 || S <= 0) return;

    for (int64_t h = 0; h < h_v; ++h) {
        const float* o_row = o + h * S;
        const float* z_row = z + h * S;
        const float* n_row = ssm_norm + h * S;
        float* out_row = out + h * S;

        // RMS norm of o[h, :]
        float sq_sum = 0.0f;
        for (int64_t j = 0; j < S; ++j) {
            sq_sum += o_row[j] * o_row[j];
        }
        const float mean_sq = sq_sum / static_cast<float>(S);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (int64_t j = 0; j < S; ++j) {
            const float normed = o_row[j] * inv_rms;
            // SIGMOID (not SiLU) for z gating
            const float sig_z = 1.0f / (1.0f + std::exp(-z_row[j]));
            out_row[j] = normed * n_row[j] * sig_z;
        }
    }
}

}  // namespace kernels
}  // namespace strata

