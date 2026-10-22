#pragma once
// strata/core/session.hpp — Session state, graph capture, token loop
// Target: Iluvatar BI-V100 (ivcore10)
//
// Architecture: 48 layers total, 36 GDN + 12 QSA (is_qsa_layer splits them).
//
// SessionState holds everything a sequence needs that is NOT a weight:
//   - Per-layer GDN/QSA buffers and persistent state
//   - MoE and block buffers
//   - Residual R ∈ R^{hc × n_embd}
//   - PLE state (layer-1 module; LEDGER L123)
//
// Graph capture: one CUDA graph per layer (because a graph bakes POINTERS).
// Measured: engine is launch-bound, not bandwidth-bound.
//   ~0.028 ms/kernel; captured graph: 1.585 ms vs 2.393 ms = 1.51× speedup.
//
// P2.T10: ZERO token-path allocations.
// P2.X3: NO cudaStreamSynchronize in the loop.

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "strata/core/layer.hpp"
#include "strata/core/types_fwd.hpp"

namespace strata {

// Forward declarations
class WeightTable;

// QSA state and buffers — defined in a later unit.
struct QsaState {
    float* k_cache = nullptr;
    float* v_cache = nullptr;
    int64_t max_cells = 0;
    int64_t pos = 0;
};

struct QsaBuffers {
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn = nullptr;
    float* out = nullptr;
};

// Block-level buffers (residual, norm scratch)
struct BlockBuffers {
    float* norm_out = nullptr;
    float* residual_scratch = nullptr;
    float* mixer_out = nullptr;
    float* ffn_out = nullptr;
};

// NG_HIST for PLE
inline constexpr int64_t NG_HIST = 16;

namespace kernels {
// PLE run descriptor — defined in a later unit.
struct PleRun {
    bool enabled = false;
    int64_t hc_dim = 0;
    float* weights = nullptr;
};
}  // namespace kernels

// ─── SessionData ──────────────────────────────────────────────────
// Everything a sequence needs that is NOT a weight.
// Architecture: 48 layers total, 36 GDN + 12 QSA.
// PLE state (LEDGER L123): layer-1 module that WAS NOT CALLED (why C1 failed).
struct SessionData {
    int64_t max_cells = 0;

    // GDN layer buffers + persistent state
    GdnBuffers gdn;
    float* gdn_state = nullptr;

    // QSA state and buffers
    QsaState* qsa_states = nullptr;
    void* qsa_state_arena = nullptr;
    QsaBuffers qsa_bufs;
    void* qsa_buf_arena = nullptr;

    // MoE buffers
    MoEBuffers moe;
    void* moe_arena = nullptr;

    // Block buffers
    BlockBuffers block;
    void* block_arena = nullptr;

    // Residual: R ∈ R^{hc × n_embd}
    float* R = nullptr;

    // Experts per token
    int64_t k = 10;

    // Doorbell for GPU ↔ host signaling
    const Doorbell* db = nullptr;

    // ── PLE state (LEDGER L123) ──────────────────────────────────
    // Layer-1 module. ple_prev[2] = last two tokens (oldest first, -1 where none).
    // ple_hist = NG_HIST rows of hc_dim NORMALIZED history, ROW-FASTEST.
    int32_t ple_prev[2] = {-1, -1};
    int32_t ple_token = -1;
    float* ple_hist = nullptr;
    kernels::PleRun ple;
};

// ─── session_bytes / session_init / session_zero ──────────────────
// P2.T10: ZERO token-path allocations. session_bytes sizes everything
// at once before any token runs.

uint64_t session_bytes(const Geometry& g, int64_t max_cells, int64_t k);
uint64_t session_init(const Geometry& g, int64_t max_cells, int64_t k,
                      void* base, SessionData& s);
void session_zero(SessionData& s, const Geometry& g,
                  const float* R_init, void* stream);

// ─── session_token ────────────────────────────────────────────────
// Layers 0..47 in order, each a block_layer.
// `parts` is (k, n_embd) DEVICE memory, filled by caller.
// sync_every_layer: inserts device sync + error check (P2.X3 forbids in fast path).
bool session_token(const WeightTable& tables, const Geometry& g,
                   int64_t pos, int32_t pos_base,
                   SessionData& s, const float* parts,
                   void* stream, bool sync_every_layer,
                   std::string& err);

// ─── SessionGraphs ────────────────────────────────────────────────
// One CUDA graph per layer (graph bakes pointers; each layer has
// different weights AND state).
//
// Split graphs (R0.11): 5 prefix graphs preP[0..4], plus preA (mixer)
// and preB (FFN front + router).
//
// Diagnostic counters:
//   rings_mid_graph: layers where ring observed while graph STILL RUNNING
//   ms_to_ring: launch → ring observed (cumulative)
//   ms_host: ring observed → next pre launched (GPU IDLE during this)
struct SessionGraphs {
    cudaGraphExec_t* execs = nullptr;    // pre graph per layer
    int64_t n = 0;                       // number of layers
    cudaGraphExec_t* posts = nullptr;    // post graph per layer
    bool captured = false;
    float* parts_dev = nullptr;

    // Overlap diagnostics
    int64_t rings_mid_graph = 0;
    double ms_to_ring = 0;

    // Split graph support (R0.11)
    cudaGraphExec_t* preP[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaGraphExec_t* preA = nullptr;
    cudaGraphExec_t* preB = nullptr;
    bool split_captured = false;

    // Counters
    int64_t calls_total = 0;
    double ms_host = 0;
};

// ─── session_capture / session_replay ─────────────────────────────
// Idempotent. Position reaches QSA via fixed-address pinned staging.

bool session_capture(const WeightTable& tables, const Geometry& g,
                     SessionData& s, const float* parts,
                     SessionGraphs& gr, std::string& err,
                     bool split = false);

// One token by replaying captured graphs.
// Identical arithmetic to session_token; ~2000 kernel launches → 48 graph launches.
bool session_replay(const Geometry& g, int64_t pos, int32_t pos_base,
                    SessionData& s, SessionGraphs& gr,
                    void* stream, std::string& err);

// Replays BOTH pre[l] and post[l] per layer.
// TRUE per-token GPU floor (session_replay omits 48 post graphs).
bool session_replay_full(const Geometry& g, int64_t pos, int32_t pos_base,
                         SessionData& s, SessionGraphs& gr,
                         void* stream, std::string& err);

// ─── Stage timing via prefix differencing ─────────────────────────
// stage_ms[k] = prefix[k] - prefix[k-1], k = 1..5
// R0.11: Fifth prefix bug — prefix 5 had ~7 more nodes than prefix 4.
// Residual saved/restored between prefixes.

bool session_replay_stage_prefixes(const Geometry& g, int64_t pos, int32_t pos_base,
                                   SessionData& s, SessionGraphs& gr,
                                   void* stream, std::vector<double>& stage_ms,
                                   std::vector<double>& mixer_per_layer,
                                   std::string& err);

bool session_replay_stage_sweep(const Geometry& g, int64_t pos, int32_t pos_base,
                                SessionData& s, SessionGraphs& gr,
                                void* stream, int k, double& ms,
                                std::string& err);

bool session_replay_stages(const Geometry& g, int64_t pos, int32_t pos_base,
                           SessionData& s, SessionGraphs& gr,
                           void* stream, double& ms_mixer, double& ms_ffn,
                           double& ms_post, std::string& err);

bool session_replay_stages_per_layer(const Geometry& g, int64_t pos, int32_t pos_base,
                                     SessionData& s, SessionGraphs& gr,
                                     void* stream,
                                     std::vector<double>& mixer_per_layer,
                                     double& ms_ffn, double& ms_post,
                                     std::string& err);

void session_graphs_free(SessionGraphs& gr);

// ─── SessionLoopScratch ───────────────────────────────────────────
// Per-token scratch, MUST NOT be reallocated per token (P2.T10).
// The old code allocated per token:
//   1. cudaHostAlloc + cudaFreeHost (RAII) → implicit device sync
//   2. cudaEventCreate/Destroy per token
//   3. Host affinity change per token
struct SessionLoopScratch {
    float* y_miss = nullptr;       // pinned host staging for pool answer
    size_t parts_bytes = 0;
    cudaEvent_t probe = nullptr;   // CUDA event for doorbell probe
    long long pinned_core = -1;    // host affinity to restore
    bool pinned = false;

    bool init(size_t parts_bytes_in, std::string& err);
    void free();
};

// ─── Pool callback ────────────────────────────────────────────────
// (user, x_f, ids, weights, n_embd, k) → out[k × n_embd]
using PoolFn = void (*)(void* user, const float* x_f,
                        const int32_t* ids, const float* weights,
                        int64_t n_embd, int64_t k, float* out);

// Hit callback — notifies caller which experts were cache hits.
using HitFn = void (*)(void* user, int64_t layer,
                       const int32_t* ids, int64_t k);

// ─── session_loop ─────────────────────────────────────────────────
// The host loop (P2.S5):
//   ∀ l ∈ [0,47]: launch G[l] → poll doorbell → read miss list →
//   dispatch pool → wait → write y_miss → launch G[l+1]
//
// NO cudaStreamSynchronize in the loop (P2.X3).
// Overlap = one-layer delay: layer publishes x_f at ~71% of its own
// duration; pool has remaining 29%.
//
// dump_layers: HOST buffer of (n_layers+1) × hc × n_embd floats.
// Slot 0 = initial residual, slot l+1 = R after layer l.
bool session_loop(const Geometry& g, int64_t pos, int32_t pos_base,
                  SessionData& s, SessionGraphs& gr,
                  PoolFn pool, HitFn hits, void* user,
                  bool overlap, void* stream, std::string& err,
                  float* dump_layers = nullptr,
                  SessionLoopScratch* scratch = nullptr);

// ─── TokenGraph (Plan v0.3 P3) ───────────────────────────────────
// Whole token as ONE graph. Under WDDM every graph launch costs
// ~0.3-0.4 ms submission latency; per-layer loop = 96 launches.
//
// Between pre[l] and parts copy: a one-thread doorbell_wait kernel
// spins ON THE DEVICE until host writes the ring value.
struct TokenGraph {
    cudaGraphExec_t exec = nullptr;
    bool captured = false;
    int64_t n_layers = 0;
    const float* y_src = nullptr;   // host staging pointer baked into graph
    size_t parts_bytes = 0;
    int64_t calls = 0;
    double ms_wait = 0;
    double ms_pool = 0;
    int64_t flushes = 0;
};

// ─── TokenHits (Plan v0.3 P4) ────────────────────────────────────
// VRAM expert tier inside the token graph.
// Residency is STATIC during a token: n_layers × n_expert device table.
struct TokenHits {
    const int32_t* d_res = nullptr;       // residency table on device
    int64_t n_expert = 0;
    const uint8_t* cache_base = nullptr;
    int64_t blob = 0;                     // bytes per expert blob
    int32_t* d_slot = nullptr;
    int32_t* d_dst = nullptr;
    int32_t* d_count = nullptr;
    uint8_t* x_q8 = nullptr;
    float* x_scale = nullptr;
    void* scratch = nullptr;
    float* hit_out = nullptr;

    bool on() const {
        return d_res && cache_base && d_slot && d_dst &&
               d_count && x_q8 && x_scale && scratch && hit_out;
    }
};

// ─── session_capture_token / session_run_token ────────────────────
// Captures the whole-token graph. Host only polls rings, runs pool,
// writes flags — no driver call in the loop.
// Micro-benchmark: 5.2 μs per handoff.

bool session_capture_token(const WeightTable& tables, const Geometry& g,
                           SessionData& s, float* parts_dev,
                           const float* y_miss_host, size_t parts_bytes,
                           TokenGraph& tg, std::string& err,
                           const TokenHits* hits = nullptr);

bool session_run_token(const Geometry& g, int64_t pos, int32_t pos_base,
                       SessionData& s, TokenGraph& tg,
                       PoolFn pool, void* user, float* y_miss_host,
                       void* stream, std::string& err);

void token_graph_free(TokenGraph& tg);

}  // namespace strata

