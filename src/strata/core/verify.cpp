// strata/core/verify.cpp — Speculative verifier (skeleton, GPU kernels in later units)
#include "strata/core/verify.hpp"

namespace strata {

Verifier::~Verifier() {
    for (int i = 0; i < 9; ++i) {
        if (exec_[i]) { cudaGraphExecDestroy(exec_[i]); exec_[i] = nullptr; }
    }
    if (commit_exec_) { cudaGraphExecDestroy(commit_exec_); commit_exec_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    // residual_buf_ freed by whoever allocated it (session state or mapped alloc)
}

bool Verifier::init(const WeightTable& wt, const Geometry& geom,
                    SessionState& state, const ExpertCacheHits& hits,
                    const HeadProjection* head, int max_t, std::string& err) {
    wt_ = &wt;
    geom_ = &geom;
    state_ = &state;
    hits_ = hits;
    head_ = head;
    max_t_ = max_t;
    // Stream and buffer allocation deferred to later units.
    err = "verifier init: GPU kernel registration not yet implemented";
    return false;
}

bool Verifier::run(int T, const int32_t* tokens, int64_t pos0,
                   ExpertPoolFn pool, void* user,
                   int32_t* out, std::string& err) {
    err = "verifier run: not yet implemented";
    return false;
}

bool Verifier::commit(int n_keep, std::string& err) {
    err = "verifier commit: not yet implemented";
    return false;
}

const float* Verifier::final_R(int idx) const {
    // Returns pointer into residual_buf_ offset by idx * (hc * n_embd)
    // Geometry not available yet — will be filled in later units.
    return residual_buf_;  // placeholder
}

bool Verifier::capture(int T, std::string& err) {
    err = "verifier capture: not yet implemented";
    return false;
}

bool Verifier::capture_commit(std::string& err) {
    err = "verifier capture_commit: not yet implemented";
    return false;
}

bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {
    err = "verifier record_window: not yet implemented";
    return false;
}

}  // namespace strata

