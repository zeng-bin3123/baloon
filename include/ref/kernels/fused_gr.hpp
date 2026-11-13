// fused_gr.hpp - fused gated-residual read at the fixed geometry n_embd=2560, hc=4, hc_lr=320.
// FusedGrArgs, fused_gr_read, fused_gr_read_multi, fused_gr_supported, kFusedGrMaxT.
#pragma once
#include <cstdint>

namespace baloon {
namespace ref_fused_gr {

// Maximum number of tokens the multi-token variant handles.
constexpr int kFusedGrMaxT = 32;

// Arguments for one fused gr_read call (single or multi-token).
struct FusedGrArgs {
    const float*    R;           // residual streams [hc][n_embd] (read)
    const uint16_t* w_down;      // bf16 down-projection weights [hc_lr][hc*n_embd/hc_lr]
    const uint16_t* w_up;        // bf16 up-projection weights
    const uint16_t* w_inject;    // bf16 injection weights (may be null)
    const float*    inj_prev;    // previous layer injection logits for apply (may be null)
    const float*    bo_prev;     // previous layer block output for apply (may be null)
    float*          inject_out;  // injection logit output (may be null)
    float*          R_out;       // residual output [hc][n_embd]
    int             n_embd;      // must be 2560 for supported()
    int             hc;          // must be 4 for supported()
    int             hc_lr;       // must be 320 for supported()
    bool            apply;       // fold in previous gr_write
};

// Returns true iff the fixed geometry (2560/4/320) is met, enabling the fused path.
bool fused_gr_supported(int n_embd, int hc, int hc_lr);

// Single-token fused gr_read.  Returns 0 on success, negative on bad args.
int fused_gr_read(const FusedGrArgs& args, void* stream);

// Multi-token fused gr_read (1 <= n_tok <= kFusedGrMaxT).
int fused_gr_read_multi(const FusedGrArgs& args, int n_tok, void* stream);

}  // namespace ref_fused_gr
}  // namespace baloon

