// kern_ple.hpp - GPU half of the PLE block: depthwise causal dilated convolution over [history | new] rows, per-stream RMS
// norm, stream gates, gated broadcast, three-way add, stream sum + SiLU, bf16 projection, history advance.
// Fixed geometry (see plegeo): n_embd = 2560, hc = 4 streams, hc_dim = 10240, conv kernel 4 / dilation 3, 9 history rows.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace kern {

struct PleWeights {
    const float* conv_w = nullptr;      // 4 * hc_dim, tap k of channel c at k + 4*c
    const float* norm_w = nullptr;      // n_embd RMS weights (shared by the four streams)
    const uint16_t* gate_w = nullptr;   // bf16, hc rows of n_embd: row s gates stream s
    const uint16_t* proj_w = nullptr;   // bf16, n_embd rows of n_embd (output x input)
};

struct PleOut {
    float* out = nullptr;               // n_embd results
    float* gate = nullptr;              // optional: hc gate values (may be null)
};

// True when a CUDA device is usable for the block.
bool ple_block_available();
// Bytes of the caller's workspace (16-byte aligned regions: float workspace | activation image | bf16 copy | slack).
size_t ple_block_scratch_bytes();

// out = [hidden +] project(silu(sum over streams of gated/normed conv output)). `history` is the row-fastest state
// (row + 9*channel, 9 rows) and is NOT modified; call ple_history_advance to commit the new row afterwards.
// new_row: hc_dim values of the current token; hidden: n_embd. scratch: ple_block_scratch_bytes() bytes, disjoint from
// every other buffer (checked).
cudaError_t ple_block(const PleWeights& w, const float* new_row, const float* history, const float* hidden, const PleOut& out,
                      void* scratch, cudaStream_t stream);
// Shift every channel column left by one and append new_row[c] (row-fastest layout).
cudaError_t ple_history_advance(float* history, const float* new_row, cudaStream_t stream);

// Projection consumes bf16-rounded activations (true) or the float activations (false, default).
void ple_set_native_bf16(bool on);
// Residual add of `hidden` is done inside the block (true) or left to the caller (false, default).
void ple_set_native_postops(bool on);
bool ple_native_postops_enabled();

}  // namespace kern
}  // namespace baloon
