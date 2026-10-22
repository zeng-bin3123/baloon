// strata/core/mtp.cpp — MTP drafter (skeleton, GPU kernels in later units)
#include "strata/core/mtp.hpp"
#include "strata/core/weights.hpp"

namespace strata {

MtpDrafter::~MtpDrafter() {
    for (int i = 0; i < 9; ++i) {
        if (step_exec_[i]) { cudaGraphExecDestroy(step_exec_[i]); step_exec_[i] = nullptr; }
    }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

bool MtpDrafter::load(const std::string& rt_dir, const Geometry& geom,
                      SessionState& state, int max_t, std::string& err,
                      int64_t window) {
    geom_ = &geom;
    state_ = &state;
    max_t_ = max_t;
    err = "mtp load: GPU buffer allocation not yet implemented";
    return false;
}

bool MtpDrafter::bind(const WeightTable& wt, const HeadProjection* head,
                      const float* window_R, std::string& err) {
    wt_ = &wt;
    head_ = head;
    window_R_ = window_R;
    return true;
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens,
                         int64_t n_cells, int64_t cell0, std::string& err) {
    err = "mtp prefill: not yet implemented";
    return false;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t pos,
                       int anchor_idx, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    err = "mtp draft: not yet implemented";
    return false;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token,
                             int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    err = "mtp draft_first: not yet implemented";
    return false;
}

bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs,
                                std::string& err) {
    err = "mtp record_forward: not yet implemented";
    return false;
}

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    err = "mtp capture_prefill: not yet implemented";
    return false;
}

bool MtpDrafter::capture_round(int T, std::string& err) {
    err = "mtp capture_round: not yet implemented";
    return false;
}

bool MtpDrafter::capture_step(int idx, std::string& err) {
    err = "mtp capture_step: not yet implemented";
    return false;
}

const float* MtpDrafter::f32(const char* name) const {
    if (!wt_) return nullptr;
    auto* ref = wt_->find(name);
    return ref ? static_cast<const float*>(ref->data) : nullptr;
}

const uint16_t* MtpDrafter::bf16(const char* name) const {
    if (!wt_) return nullptr;
    auto* ref = wt_->find(name);
    return ref ? static_cast<const uint16_t*>(ref->data) : nullptr;
}

const void* MtpDrafter::q8(const char* name) const {
    if (!wt_) return nullptr;
    auto* ref = wt_->find(name);
    return ref ? ref->data : nullptr;
}

}  // namespace strata

