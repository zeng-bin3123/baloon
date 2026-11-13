// gr.hpp - gated residual (hyper-connection) reference: gr_read, gr_write, workspace helpers.
// GrShapes, GrWorkspace, gr_read, gr_write, gr_workspace_bytes, gr_workspace_init,
// gr_set_fp32_activations, gr_set_native_mmvf.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace ref_gr {

// Shape parameters for a gated-residual block.
struct GrShapes {
    int n_embd;  // vocabulary / hidden dimension
    int hc;      // number of residual streams (hyper-connection count)
    int hc_lr;   // low-rank bottleneck width
};

// Workspace pointer table filled by gr_workspace_init.
struct GrWorkspace {
    float* xn;     // region 0: hc*n_embd floats (stream-normalised activations)
    float* xq;     // region 1: hc*n_embd floats (down projection input scratch)
    float* lq;     // region 2: hc_lr floats (low-rank projection)
    float* gated;  // region 3: hc*n_embd floats (gated stream values)
    float* lo;     // region 4: hc_lr floats (SiLU'd down projection)
    uint64_t total_bytes;
};

// Return byte count needed for the workspace (5 regions, each 16-byte rounded).
uint64_t gr_workspace_bytes(const GrShapes& sh);

// Carve workspace into the five regions.  ws must point to gr_workspace_bytes(sh) bytes.
void gr_workspace_init(void* ws, const GrShapes& sh, GrWorkspace* out);

// Set whether fp32 activations are used in the down projection (default: false → bf16).
void gr_set_fp32_activations(bool on);

// Set whether the native MMVF (native matrix-vector fused) path is active.
void gr_set_native_mmvf(bool on);

// Read one token through the gated-residual block.
// R: residual streams [hc][n_embd]; w_down, w_up, w_inject (optional): weight matrices.
// Output is the mean-mixed gated output in work->gated averaged over streams.
int gr_read(const float* R, const float* w_down, const float* w_up, const float* w_inject,
             const GrShapes& sh, GrWorkspace* work, void* stream);

// Write block output back to residual: out[i] = R[i] + block_out[i%n] * 2*sigmoid(inject[i/n]/hc).
int gr_write(float* out, const float* R, const float* block_out, const float* inject,
              const GrShapes& sh, void* stream);

}  // namespace ref_gr
}  // namespace baloon

