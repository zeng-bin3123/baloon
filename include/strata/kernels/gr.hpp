#pragma once
// strata/kernels/gr.hpp — GR (Grouped Residual) kernels
// Target: Iluvatar BI-V100 (ivcore10)
//
// GrShapes: (n_embd, hc, hc_lr).
//   Real model: n_embd=2560, hc=4, hc_lr=320.
//   Residual is a STACK of hc streams, each n_embd wide.
//
// GrWorkspace: device-memory scratch for gr_read.
//   CRITICAL: must be DEVICE memory, not host.
//
// gr_read (build_hc_mix):  collapses hc streams to one vector.
//   xn[c] = RMSNorm(R[c], w_norm[c]) * (1+implicit)  (w_norm stored as 1+w)
//   lo = SiLU(bf16(xn) · w_down^T / hc)  (division INSIDE SiLU)
//   gated[c,d] = xn[c,d] * sigmoid(lo · w_up^T[c,d])
//   mixed = mean_c(gated[c])
//   inject[c] = bf16(xn) · w_inject^T[c]  (null w_inject → untouched)
//
// gr_write (build_hc_combine):
//   R_out[c,d] = R[c,d] + block_out[d] * w[c]
//   w[c] = 2 * sigmoid(inject[c] / hc)
//   Note: 2*sigmoid centers gate on 1; zero injection → plain residual add.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace strata {
namespace kernels {

// ─── Configuration ─────────────────────────────────────────────────
// gr_set_fp32_activations: selects activation precision for gr_read.
//   false → BF16 rounding (default), true → FP32 activations.
//   Configure before session capture; captured graphs retain selected kernels.
void gr_set_fp32_activations(bool enabled);

// gr_set_native_mmvf: enables native MMVF kernel path.
//   true → FP32 activations (overrides gr_set_fp32_activations).
//   Requires even hc * n_embd and hc_lr (MMVF reads pairs).
//   Also selects pinned weighted F32 RMSNorm and nonlinear/combine arithmetic.
//   gr_write selects pinned sigmoid/scatter under same setting.
void gr_set_native_mmvf(bool enabled);

// ─── GrShapes ──────────────────────────────────────────────────────
// (n_embd, hc, hc_lr) ∈ Z^3.
// Real model: n_embd=2560, hc=4, hc_lr=320.
// Residual is a STACK of hc streams, each n_embd wide.
struct GrShapes {
    int64_t n_embd = 0;    // embedding dimension per stream
    int64_t hc     = 0;    // number of residual streams (head count)
    int64_t hc_lr  = 0;    // low-rank bottleneck dimension

    // Total residual width = hc * n_embd.
    int64_t residual_width() const { return hc * n_embd; }

    bool valid() const {
        return n_embd > 0 && hc > 0 && hc_lr > 0;
    }
};

// ─── GrWorkspace ───────────────────────────────────────────────────
// DEVICE-memory scratch for gr_read.
//
// Regions:
//   xn:    float[hc * n_embd]       — unrounded activated streams
//   xq:    uint16_t[hc * n_embd]    — bf16(xn), default contract
//   lq:    uint16_t[hc_lr]          — bf16(lo), default contract
//   gated: float[hc * n_embd]       — xn * sigmoid(gate), before mean
//   lo:    float[hc_lr]             — SiLU bottleneck, FP32 contract
//
// Invariant: bytes > 0 implies valid; gr_read CHECKS it.
// CRITICAL: must be DEVICE memory, not host.
struct GrWorkspace {
    float*    xn    = nullptr;
    uint16_t* xq    = nullptr;
    uint16_t* lq    = nullptr;
    float*    gated = nullptr;
    float*    lo    = nullptr;
    size_t    bytes = 0;       // total allocation, 0 = uninitialized
};

// ─── gr_workspace_init / gr_workspace_bytes ────────────────────────
// Carve base (device memory) into workspace regions, 16-byte aligned.
// base = null → size-only query (returns needed bytes, out untouched).
size_t gr_workspace_init(const GrShapes& shapes, void* base,
                         GrWorkspace& out);

inline size_t gr_workspace_bytes(const GrShapes& shapes) {
    GrWorkspace w;
    return gr_workspace_init(shapes, nullptr, w);
}

// ─── gr_read ───────────────────────────────────────────────────────
// build_hc_mix: ONE token, collapses hc streams to one vector.
//
// R ∈ R^{hc × n_embd}, stream c starts at c * n_embd.
//
// Steps:
//   1. xn[c] = RMSNorm(R[c], w_norm[c]) * (1+implicit)  (w_norm stored as 1+w)
//   2. lo = SiLU(bf16(xn) · w_down^T / hc)   (division INSIDE SiLU)
//   3. gated[c,d] = xn[c,d] * sigmoid(lo · w_up^T[c,d])
//   4. mixed = mean_c(gated[c])
//   5. inject[c] = bf16(xn) · w_inject^T[c]  (null w_inject → untouched)
//
// Weight layout:
//   w_down: (hc_lr, hc * n_embd), bf16
//   w_up:   (hc * n_embd, hc_lr), bf16
//   NO permutation: matrices taken as manifest stores them.
//
// FP32 contract (opt-in): replace bf16(xn) and bf16(lo) with xn and lo.
void gr_read(const float* R, const float* w_norm,
             const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps,
             const GrShapes& shapes, const GrWorkspace& ws,
             float* mixed, float* inject, cudaStream_t stream);

// ─── gr_write ──────────────────────────────────────────────────────
// build_hc_combine: scatter block output into residual streams.
//
//   R_out[c, d] = R[c, d] + block_out[d] * w[c]
//   w[c] = 2 * sigmoid(inject[c] / hc)
//
// Note: 2*sigmoid centres gate on 1; zero injection → plain residual add.
// Block output added to EVERY stream identically; only weight is per-stream.
void gr_write(const float* R, const float* block_out,
              const float* inject, const GrShapes& shapes,
              float* R_out, cudaStream_t stream);

}  // namespace kernels
}  // namespace strata

