#pragma once
// strata/core/mtp.hpp — Multi-Token Prediction draft layer on GPU
// Target: Iluvatar BI-V100 (ivcore10)
//
// Architecture per cell i:
//   e_i = fc_emb(rms(embed(tok_i)) * (1 + w))
//   h_i = fc_hidden_s(rms_10240(R_i) * (1 + w))   per stream s
//   R_i = h_i + e_i → attn HC → QSA (own K/V) → MLP HC → MoE(512, top-10, shared)
//   R_i += y * 2*sigmoid(inject / hc) → final mixer → head
//
// Cell i pairs main model's residual at position i with token at i+1.
// Output predicts token at position i+2.
//
// VRAM ≈ 0.9 GB (K/V + buffers + 512 expert blobs at 708 MB)

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "strata/core/types_fwd.hpp"

namespace strata {

// Forward declarations
class WeightTable;

// ─── MtpDrafter ──────────────────────────────────────────────────────
// GPU-based multi-token prediction drafter.
//
//   load(rt_dir, geom, state, max_t) → {ok, err}
//     Post: VRAM ≈ 0.9 GB allocated
//
//   prefill(R_rows, next_tokens, n, cell0) → {ok, err}
//     Processes cells [cell0, cell0+n) in batches of max_t
//
//   draft(T, tokens, pos, anchor, drafts) → {ok, err}
//     Pre:  catch-up over T cells from pos, then draft chain from row anchor
//     Post: drafts[j] = argmax P_MTP(v | R_{pos+anchor+1+j})  j∈[0, T-1)
//
//   draft_first(T, R_row, tok, cell, drafts) → {ok, err}
//     Single-cell initialization then T-1 draft steps
class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();

    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    bool load(const std::string& rt_dir, const Geometry& geom,
              SessionState& state, int max_t, std::string& err,
              int64_t window = 32768);

    void set_prompt_len(int64_t len) { prompt_len_ = len; }
    uint64_t vram_bytes() const { return vram_used_; }

    bool bind(const WeightTable& wt, const HeadProjection* head,
              const float* window_R, std::string& err);

    bool prefill(const float* R_rows, const int32_t* next_tokens,
                 int64_t n_cells, int64_t cell0, std::string& err);

    bool draft(int T, const int32_t* tokens, int64_t pos, int anchor_idx,
               int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f,
               int* n_drafts = nullptr);

    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell,
                     int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f,
                     int* n_drafts = nullptr);

    // Timing accumulators (ms)
    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    bool capture_prefill(int T, std::string& err);
    bool capture_round(int T, std::string& err);
    bool capture_step(int idx, std::string& err);

    cudaGraphExec_t step_exec_[9] = {};

    // Weight accessors
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const Geometry* geom_ = nullptr;
    SessionState* state_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const HeadProjection* head_ = nullptr;
    const float* window_R_ = nullptr;

    int max_t_ = 0;
    int64_t n_vocab_ = 0;
    int64_t prompt_len_ = 0;
    uint64_t vram_used_ = 0;

    cudaStream_t stream_ = nullptr;
    // Device buffers allocated during load — layout TBD by later units.
};

}  // namespace strata

