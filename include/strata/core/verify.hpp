#pragma once
// strata/core/verify.hpp — Speculative verification window
// Target: Iluvatar BI-V100 (ivcore10)
//
// Verifies T draft tokens in one captured graph.
// Every kernel is either the single-token kernel per token, or a
// multi-token kernel whose per-token arithmetic IS the single-token
// kernel's — so a draft is accepted iff greedy decode would have produced it.
//
// CPU expert cost: ~1.75× (T=2), 2.4× (T=3), 3.05× (T=4) of single-token misses.

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "strata/core/types_fwd.hpp"

namespace strata {

// Forward declarations
class WeightTable;

// ─── Expert callback ─────────────────────────────────────────────────
// Called by verifier to fetch expert activations from CPU expert pool.
//   x_f:   input activations  (n_tok × k floats)
//   ids:   expert indices     (n_tok × k int32s)
//   n_tok: number of tokens
//   k:     experts per token
//   out:   output buffer       (n_tok × hidden_dim floats)
//   layer: which layer
using ExpertPoolFn = void (*)(void* user, const float* x_f,
                              const int32_t* ids, int64_t n_tok,
                              int64_t k, float* out, int64_t layer);

// ─── ExpertCacheHits ─────────────────────────────────────────────────
// Points into the on-device expert cache for cache-hit expert lookups.
struct ExpertCacheHits {
    const int32_t* d_results = nullptr;   // hit/miss results on device
    const uint8_t* cache_base = nullptr;  // base of cached expert blobs
    int64_t blob_bytes = 0;               // bytes per expert blob
};

// ─── PlanSink ────────────────────────────────────────────────────────
// Collects kernel-launch plan decisions made during verify/commit.
struct PlanSink {
    int pcie_mode = 0;
    // Extensible — future units may add fields for profiling/tracing.
};

// ─── Verifier ────────────────────────────────────────────────────────
// Speculative verify: runs T tokens through the full model in one
// captured graph, producing greedy argmax outputs.
//
//   run(T, tokens[0..T), pos0, pool, user) → out[0..T)
//     Pre:  T ≤ max_t, tokens at positions [pos0, pos0+T)
//     Post: out[t] = argmax_v P(· | tokens[0..t])   (bit-exact with single-token)
//     Cost: dense weights read ONCE for T tokens
//
//   commit(n_keep)
//     Pre:  1 ≤ n_keep ≤ T
//     Post: GDN state advanced by replaying first n_keep tokens.
//           Indexer key tail restored from snapshot, accepted keys re-appended.
//           K/V cells past accepted prefix simply overwritten on next use.
//
//   final_R(t): R^{hc × n_embd} on device, valid until next run.
class Verifier {
public:
    Verifier() = default;
    ~Verifier();

    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;

    bool init(const WeightTable& wt, const Geometry& geom,
              SessionState& state, const ExpertCacheHits& hits,
              const HeadProjection* head, int max_t, std::string& err);

    bool run(int T, const int32_t* tokens, int64_t pos0,
             ExpertPoolFn pool, void* user,
             int32_t* out, std::string& err);

    bool commit(int n_keep, std::string& err);

    const float* final_R(int idx) const;
    const float* final_R_all() const { return residual_buf_; }

    PlanSink* plan_sink() { return &plan_; }
    void set_split(bool on) { split_enabled_ = on; }
    void set_pcie_mode(int mode) { plan_.pcie_mode = mode; }

    // Timing accumulators (ms)
    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;

private:
    bool capture(int T, std::string& err);
    bool capture_commit(std::string& err);
    bool record_window(int T, cudaStream_t cs, std::string& err);

    const WeightTable* wt_ = nullptr;
    const Geometry* geom_ = nullptr;
    SessionState* state_ = nullptr;
    ExpertCacheHits hits_;
    const HeadProjection* head_ = nullptr;

    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t exec_[9] = {};       // one per T value
    cudaGraphExec_t commit_exec_ = nullptr;

    // Mapped staging: host pointer + device alias
    float* residual_buf_ = nullptr;

    PlanSink plan_;
    bool split_enabled_ = false;
};

}  // namespace strata

