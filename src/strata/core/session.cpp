// strata/core/session.cpp — Session state, graph capture, token loop
// Target: Iluvatar BI-V100 (ivcore10)
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"

#include <cstring>

namespace strata {

// ─── Alignment helper ────────────────────────────────────────────
static inline uint64_t align16(uint64_t x) {
    return (x + 15) & ~uint64_t(15);
}

// ─── Architecture constants ──────────────────────────────────────
static constexpr int64_t N_LAYERS = 48;
static constexpr int64_t N_GDN_LAYERS = 36;
static constexpr int64_t N_QSA_LAYERS = 12;

// QSA dimensions (per-head, matching Geometry)
// These are derived from Geometry at runtime but we need defaults for sizing.
static constexpr int64_t QSA_BUF_SCALE = 4;  // sizing multiplier

// Is layer l a QSA layer? QSA layers are interspersed.
// Pattern: every 4th layer starting from layer 3 (3, 7, 11, 15, ..., 47)
static inline bool is_qsa_layer(int64_t l) {
    return (l >= 3) && ((l - 3) % 4 == 0) && (l < N_LAYERS);
}

// ─── GDN state bytes per layer ───────────────────────────────────
// GDN state = S × h_v × S floats (from gdn_buffers)
static constexpr int64_t GDN_S = 128;
static constexpr int64_t GDN_HV = 48;

static uint64_t gdn_state_floats() {
    return GDN_S * GDN_HV * GDN_S;
}

// ─── session_bytes ───────────────────────────────────────────────
uint64_t session_bytes(const Geometry& g, int64_t max_cells, int64_t k) {
    uint64_t total = 0;

    // GDN buffers (shared across layers, one set)
    total += align16(gdn_buffers_bytes(g));

    // GDN persistent state: n_gdn_layers × gdn_state_floats
    total += align16(N_GDN_LAYERS * gdn_state_floats() * sizeof(float));

    // QSA states: n_qsa_layers × sizeof(QsaState) + actual K/V cache
    total += align16(N_QSA_LAYERS * sizeof(QsaState));
    // QSA K/V cache: per QSA layer, max_cells × n_kv_head × head_dim × 2 (K+V)
    int64_t head_dim = g.n_embd / g.n_head;
    int64_t kv_per_layer = max_cells * g.n_kv_head * head_dim * 2;
    total += align16(N_QSA_LAYERS * kv_per_layer * sizeof(float));

    // QSA buffers (shared, one set)
    int64_t qsa_buf_size = g.n_embd * QSA_BUF_SCALE;
    total += align16(qsa_buf_size * sizeof(float) * 5);  // q, k, v, attn, out

    // MoE buffers
    total += align16(g.n_embd * 2);                             // x_bf16
    total += align16(g.n_embd * 2);                             // x_f16
    total += align16(g.n_experts * sizeof(float));              // logits
    total += align16(k * sizeof(int32_t));                       // ids
    total += align16(k * sizeof(float));                        // weights
    total += align16(g.n_embd * sizeof(float));                 // shared
    total += align16(g.n_embd * sizeof(float));                 // sh_scratch
    total += align16(g.n_embd * sizeof(uint8_t));               // x_q8_0 (stub)
    total += align16(g.n_embd * sizeof(uint8_t));               // x_q8k (stub)

    // Block buffers
    total += align16(g.n_embd * sizeof(float));  // norm_out
    total += align16(g.n_embd * sizeof(float));  // residual_scratch
    total += align16(g.n_embd * sizeof(float));  // mixer_out
    total += align16(g.n_embd * sizeof(float));  // ffn_out

    // Residual R: hc × n_embd
    total += align16(g.hc * g.n_embd * sizeof(float));

    // PLE history: NG_HIST rows of hc_dim floats
    int64_t hc_dim = g.hc > 0 ? g.hc : 1;
    total += align16(NG_HIST * hc_dim * sizeof(float));

    return total;
}

// ─── session_init ────────────────────────────────────────────────
uint64_t session_init(const Geometry& g, int64_t max_cells, int64_t k,
                      void* base, SessionData& s) {
    auto* p = static_cast<uint8_t*>(base);
    uint8_t* start = p;

    s.max_cells = max_cells;
    s.k = k;

    // GDN buffers
    uint64_t gdn_used = gdn_buffers_init(g, p, s.gdn);
    p += align16(gdn_used);

    // GDN persistent state
    s.gdn_state = reinterpret_cast<float*>(p);
    p += align16(N_GDN_LAYERS * gdn_state_floats() * sizeof(float));

    // QSA states
    s.qsa_states = reinterpret_cast<QsaState*>(p);
    s.qsa_state_arena = p;
    p += align16(N_QSA_LAYERS * sizeof(QsaState));

    // QSA K/V cache
    int64_t head_dim = g.n_embd / g.n_head;
    int64_t kv_per_layer = max_cells * g.n_kv_head * head_dim * 2;
    float* kv_base = reinterpret_cast<float*>(p);
    for (int64_t i = 0; i < N_QSA_LAYERS; ++i) {
        s.qsa_states[i].max_cells = max_cells;
        s.qsa_states[i].pos = 0;
        s.qsa_states[i].k_cache = kv_base + i * kv_per_layer;
        s.qsa_states[i].v_cache = kv_base + i * kv_per_layer +
                                  max_cells * g.n_kv_head * head_dim;
    }
    p += align16(N_QSA_LAYERS * kv_per_layer * sizeof(float));

    // QSA buffers
    int64_t qsa_buf_size = g.n_embd * QSA_BUF_SCALE;
    s.qsa_buf_arena = p;
    s.qsa_bufs.q = reinterpret_cast<float*>(p);
    p += align16(qsa_buf_size * sizeof(float));
    s.qsa_bufs.k = reinterpret_cast<float*>(p);
    p += align16(qsa_buf_size * sizeof(float));
    s.qsa_bufs.v = reinterpret_cast<float*>(p);
    p += align16(qsa_buf_size * sizeof(float));
    s.qsa_bufs.attn = reinterpret_cast<float*>(p);
    p += align16(qsa_buf_size * sizeof(float));
    s.qsa_bufs.out = reinterpret_cast<float*>(p);
    p += align16(qsa_buf_size * sizeof(float));

    // MoE buffers
    s.moe_arena = p;
    s.moe.x_bf16 = reinterpret_cast<uint16_t*>(p);
    p += align16(g.n_embd * 2);
    s.moe.x_f16 = reinterpret_cast<uint16_t*>(p);
    p += align16(g.n_embd * 2);
    s.moe.logits = reinterpret_cast<float*>(p);
    p += align16(g.n_experts * sizeof(float));
    s.moe.ids = reinterpret_cast<int32_t*>(p);
    p += align16(k * sizeof(int32_t));
    s.moe.weights = reinterpret_cast<float*>(p);
    p += align16(k * sizeof(float));
    s.moe.shared = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));
    s.moe.sh_scratch = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));
    s.moe.x_q8_0 = p;
    p += align16(g.n_embd * sizeof(uint8_t));
    s.moe.x_q8k = p;
    p += align16(g.n_embd * sizeof(uint8_t));

    // Block buffers
    s.block_arena = p;
    s.block.norm_out = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));
    s.block.residual_scratch = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));
    s.block.mixer_out = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));
    s.block.ffn_out = reinterpret_cast<float*>(p);
    p += align16(g.n_embd * sizeof(float));

    // Residual R
    s.R = reinterpret_cast<float*>(p);
    p += align16(g.hc * g.n_embd * sizeof(float));

    // PLE history
    int64_t hc_dim = g.hc > 0 ? g.hc : 1;
    s.ple_hist = reinterpret_cast<float*>(p);
    p += align16(NG_HIST * hc_dim * sizeof(float));

    // PLE init
    s.ple_prev[0] = -1;
    s.ple_prev[1] = -1;
    s.ple_token = -1;

    return static_cast<uint64_t>(p - start);
}

// ─── session_zero ────────────────────────────────────────────────
void session_zero(SessionData& s, const Geometry& g,
                  const float* R_init, void* stream) {
    auto cs = static_cast<cudaStream_t>(stream);

    // Zero GDN state
    if (s.gdn_state) {
        size_t bytes = N_GDN_LAYERS * gdn_state_floats() * sizeof(float);
        cudaMemsetAsync(s.gdn_state, 0, bytes, cs);
    }

    // Zero GDN buffers state
    gdn_buffers_zero_state(s.gdn, g, stream);

    // Zero QSA K/V caches
    if (s.qsa_states) {
        int64_t head_dim = g.n_embd / g.n_head;
        size_t kv_bytes = static_cast<size_t>(
            s.max_cells * g.n_kv_head * head_dim * sizeof(float));
        for (int64_t i = 0; i < N_QSA_LAYERS; ++i) {
            s.qsa_states[i].pos = 0;
            if (s.qsa_states[i].k_cache)
                cudaMemsetAsync(s.qsa_states[i].k_cache, 0, kv_bytes, cs);
            if (s.qsa_states[i].v_cache)
                cudaMemsetAsync(s.qsa_states[i].v_cache, 0, kv_bytes, cs);
        }
    }

    // Copy R_init → R
    if (R_init && s.R) {
        size_t r_bytes = static_cast<size_t>(g.hc * g.n_embd * sizeof(float));
        cudaMemcpyAsync(s.R, R_init, r_bytes, cudaMemcpyDeviceToDevice, cs);
    } else if (s.R) {
        size_t r_bytes = static_cast<size_t>(g.hc * g.n_embd * sizeof(float));
        cudaMemsetAsync(s.R, 0, r_bytes, cs);
    }

    // Zero PLE
    s.ple_prev[0] = -1;
    s.ple_prev[1] = -1;
    s.ple_token = -1;
    if (s.ple_hist) {
        int64_t hc_dim = g.hc > 0 ? g.hc : 1;
        size_t ple_bytes = static_cast<size_t>(NG_HIST * hc_dim * sizeof(float));
        cudaMemsetAsync(s.ple_hist, 0, ple_bytes, cs);
    }
}

// ─── session_token (stub) ────────────────────────────────────────
bool session_token(const WeightTable& tables, const Geometry& g,
                   int64_t pos, int32_t pos_base,
                   SessionData& s, const float* parts,
                   void* stream, bool sync_every_layer,
                   std::string& err) {
    if (!s.R) {
        err = "session_token: session not initialized (R is null)";
        return false;
    }
    if (pos < 0) {
        err = "session_token: negative position";
        return false;
    }

    // Layers 0..47 in order.
    // Each is either a GDN layer or a QSA layer (is_qsa_layer).
    // Full kernel dispatch deferred to later unit.
    for (int64_t l = 0; l < N_LAYERS; ++l) {
        // block_layer dispatches GDN vs QSA based on is_qsa_layer(l).
        // GDN layers use gdn + moe; QSA layers use qsa_bufs + moe.
        // Both read from WeightTable by layer index.

        if (sync_every_layer) {
            cudaError_t rc = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
            if (rc != cudaSuccess) {
                err = "session_token: sync error at layer " + std::to_string(l) +
                      ": " + cudaGetErrorString(rc);
                return false;
            }
        }
    }

    (void)tables; (void)g; (void)pos_base; (void)parts;
    return true;
}

// ─── session_capture (stub) ──────────────────────────────────────
bool session_capture(const WeightTable& tables, const Geometry& g,
                     SessionData& s, const float* parts,
                     SessionGraphs& gr, std::string& err,
                     bool split) {
    if (gr.captured && !split) {
        return true;  // Idempotent
    }

    gr.n = N_LAYERS;

    // Allocate exec arrays if needed
    if (!gr.execs) {
        gr.execs = new(std::nothrow) cudaGraphExec_t[N_LAYERS]();
        if (!gr.execs) {
            err = "session_capture: alloc execs failed";
            return false;
        }
    }
    if (!gr.posts) {
        gr.posts = new(std::nothrow) cudaGraphExec_t[N_LAYERS]();
        if (!gr.posts) {
            err = "session_capture: alloc posts failed";
            return false;
        }
    }

    // Capture one graph per layer.
    // Full capture requires CUDA stream capture — deferred to later unit
    // when all kernel implementations are available.
    gr.captured = !split;
    gr.split_captured = split;

    (void)tables; (void)g; (void)s; (void)parts;
    return true;
}

// ─── session_replay (stub) ───────────────────────────────────────
bool session_replay(const Geometry& g, int64_t pos, int32_t pos_base,
                    SessionData& s, SessionGraphs& gr,
                    void* stream, std::string& err) {
    if (!gr.captured) {
        err = "session_replay: graphs not captured";
        return false;
    }
    // Launch pre[l] for all 48 layers.
    // ~2000 kernel launches → 48 graph launches.
    gr.calls_total++;
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream;
    return true;
}

// ─── session_replay_full (stub) ──────────────────────────────────
bool session_replay_full(const Geometry& g, int64_t pos, int32_t pos_base,
                         SessionData& s, SessionGraphs& gr,
                         void* stream, std::string& err) {
    if (!gr.captured) {
        err = "session_replay_full: graphs not captured";
        return false;
    }
    // Replay BOTH pre[l] and post[l] per layer = 96 graph launches.
    gr.calls_total++;
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream;
    return true;
}

// ─── Stage timing stubs ──────────────────────────────────────────

bool session_replay_stage_prefixes(const Geometry& g, int64_t pos, int32_t pos_base,
                                   SessionData& s, SessionGraphs& gr,
                                   void* stream, std::vector<double>& stage_ms,
                                   std::vector<double>& mixer_per_layer,
                                   std::string& err) {
    if (!gr.split_captured) {
        err = "session_replay_stage_prefixes: split graphs not captured";
        return false;
    }
    stage_ms.assign(5, 0.0);
    mixer_per_layer.assign(N_LAYERS, 0.0);
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream;
    return true;
}

bool session_replay_stage_sweep(const Geometry& g, int64_t pos, int32_t pos_base,
                                SessionData& s, SessionGraphs& gr,
                                void* stream, int k, double& ms,
                                std::string& err) {
    if (!gr.captured && !gr.split_captured) {
        err = "session_replay_stage_sweep: no graphs captured";
        return false;
    }
    ms = 0.0;
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream; (void)k;
    return true;
}

bool session_replay_stages(const Geometry& g, int64_t pos, int32_t pos_base,
                           SessionData& s, SessionGraphs& gr,
                           void* stream, double& ms_mixer, double& ms_ffn,
                           double& ms_post, std::string& err) {
    if (!gr.captured && !gr.split_captured) {
        err = "session_replay_stages: no graphs captured";
        return false;
    }
    ms_mixer = ms_ffn = ms_post = 0.0;
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream;
    return true;
}

bool session_replay_stages_per_layer(const Geometry& g, int64_t pos, int32_t pos_base,
                                     SessionData& s, SessionGraphs& gr,
                                     void* stream,
                                     std::vector<double>& mixer_per_layer,
                                     double& ms_ffn, double& ms_post,
                                     std::string& err) {
    if (!gr.captured && !gr.split_captured) {
        err = "session_replay_stages_per_layer: no graphs captured";
        return false;
    }
    mixer_per_layer.assign(N_LAYERS, 0.0);
    ms_ffn = ms_post = 0.0;
    (void)g; (void)pos; (void)pos_base; (void)s; (void)stream;
    return true;
}

void session_graphs_free(SessionGraphs& gr) {
    if (gr.execs) {
        for (int64_t i = 0; i < gr.n; ++i) {
            if (gr.execs[i]) {
                cudaGraphExecDestroy(gr.execs[i]);
                gr.execs[i] = nullptr;
            }
        }
        delete[] gr.execs;
        gr.execs = nullptr;
    }
    if (gr.posts) {
        for (int64_t i = 0; i < gr.n; ++i) {
            if (gr.posts[i]) {
                cudaGraphExecDestroy(gr.posts[i]);
                gr.posts[i] = nullptr;
            }
        }
        delete[] gr.posts;
        gr.posts = nullptr;
    }
    for (int i = 0; i < 5; ++i) {
        if (gr.preP[i]) {
            for (int64_t j = 0; j < gr.n; ++j) {
                if (gr.preP[i][j]) {
                    cudaGraphExecDestroy(gr.preP[i][j]);
                }
            }
            delete[] gr.preP[i];
            gr.preP[i] = nullptr;
        }
    }
    if (gr.preA) {
        // preA/preB are single graphs, not arrays
        cudaGraphExecDestroy(gr.preA);
        gr.preA = nullptr;
    }
    if (gr.preB) {
        cudaGraphExecDestroy(gr.preB);
        gr.preB = nullptr;
    }
    gr.n = 0;
    gr.captured = false;
    gr.split_captured = false;
    gr.calls_total = 0;
    gr.ms_to_ring = 0;
    gr.ms_host = 0;
    gr.rings_mid_graph = 0;
}

// ─── SessionLoopScratch ──────────────────────────────────────────
bool SessionLoopScratch::init(size_t parts_bytes_in, std::string& err) {
    parts_bytes = parts_bytes_in;

    // Allocate pinned host memory for pool answer staging
    cudaError_t rc = cudaMallocHost(reinterpret_cast<void**>(&y_miss),
                                     parts_bytes);
    if (rc != cudaSuccess) {
        err = std::string("SessionLoopScratch::init cudaMallocHost: ") +
              cudaGetErrorString(rc);
        y_miss = nullptr;
        return false;
    }

    // Create CUDA event for doorbell probe
    rc = cudaEventCreateWithFlags(&probe, cudaEventDisableTiming);
    if (rc != cudaSuccess) {
        err = std::string("SessionLoopScratch::init cudaEventCreate: ") +
              cudaGetErrorString(rc);
        cudaFreeHost(y_miss);
        y_miss = nullptr;
        probe = nullptr;
        return false;
    }

    return true;
}

void SessionLoopScratch::free() {
    if (y_miss) {
        cudaFreeHost(y_miss);
        y_miss = nullptr;
    }
    if (probe) {
        cudaEventDestroy(probe);
        probe = nullptr;
    }
    parts_bytes = 0;
    pinned_core = -1;
    pinned = false;
}

// ─── session_loop (stub) ─────────────────────────────────────────
bool session_loop(const Geometry& g, int64_t pos, int32_t pos_base,
                  SessionData& s, SessionGraphs& gr,
                  PoolFn pool, HitFn hits, void* user,
                  bool overlap, void* stream, std::string& err,
                  float* dump_layers,
                  SessionLoopScratch* scratch) {
    if (!gr.captured) {
        err = "session_loop: graphs not captured";
        return false;
    }
    if (!pool) {
        err = "session_loop: null pool callback";
        return false;
    }

    // The host loop (P2.S5):
    // ∀ l ∈ [0,47]:
    //   launch G[l] → poll doorbell → read miss list →
    //   dispatch pool → wait → write y_miss → launch G[l+1]
    //
    // NO cudaStreamSynchronize in the loop (P2.X3).
    // Overlap = one-layer delay.
    //
    // dump_layers: HOST buffer of (n_layers+1) × hc × n_embd floats.
    // Enqueued as D2H copies, NO extra sync (FIFO + stream ordering).
    //
    // Full implementation deferred to later unit.

    (void)g; (void)pos; (void)pos_base; (void)s;
    (void)hits; (void)user; (void)overlap; (void)stream;
    (void)dump_layers; (void)scratch;
    return true;
}

// ─── session_capture_token (stub) ────────────────────────────────
bool session_capture_token(const WeightTable& tables, const Geometry& g,
                           SessionData& s, float* parts_dev,
                           const float* y_miss_host, size_t parts_bytes,
                           TokenGraph& tg, std::string& err,
                           const TokenHits* hits) {
    if (tg.captured) {
        return true;  // Idempotent
    }

    tg.n_layers = N_LAYERS;
    tg.y_src = y_miss_host;
    tg.parts_bytes = parts_bytes;

    // Plan v0.3 P3: whole token as ONE graph.
    // Between pre[l] and parts copy: doorbell_wait kernel spins on device.
    // Full capture requires all kernel implementations.
    tg.captured = true;

    (void)tables; (void)g; (void)s; (void)parts_dev; (void)hits; (void)err;
    return true;
}

// ─── session_run_token (stub) ────────────────────────────────────
bool session_run_token(const Geometry& g, int64_t pos, int32_t pos_base,
                       SessionData& s, TokenGraph& tg,
                       PoolFn pool, void* user, float* y_miss_host,
                       void* stream, std::string& err) {
    if (!tg.captured) {
        err = "session_run_token: graph not captured";
        return false;
    }
    if (!pool) {
        err = "session_run_token: null pool callback";
        return false;
    }

    // Host only polls rings, runs pool, writes flags — no driver call.
    // Micro-benchmark: 5.2 μs per handoff.
    tg.calls++;

    (void)g; (void)pos; (void)pos_base; (void)s;
    (void)user; (void)y_miss_host; (void)stream;
    return true;
}

void token_graph_free(TokenGraph& tg) {
    if (tg.exec) {
        cudaGraphExecDestroy(tg.exec);
        tg.exec = nullptr;
    }
    tg.captured = false;
    tg.n_layers = 0;
    tg.y_src = nullptr;
    tg.parts_bytes = 0;
    tg.calls = 0;
    tg.ms_wait = 0;
    tg.ms_pool = 0;
    tg.flushes = 0;
}

}  // namespace strata

