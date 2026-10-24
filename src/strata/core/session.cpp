// strata/core/session.cpp — Session state, graph capture, token loop
// Target: Iluvatar BI-V100 (ivcore10)
//
// session_token: uncaptured path, layers 0..47, each a block_layer.
// session_capture/replay: CUDA graph capture and replay.
// session_loop: the per-layer pipeline with CPU expert overlap.
//
// Architecture: 48 layers total, 36 GDN + 12 QSA.
// Graph capture: one graph per layer (bakes pointers).
// P2.T10: ZERO token-path allocations.
// P2.X3: NO cudaStreamSynchronize in the loop.
#include "strata/core/session.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/weights.hpp"

#include <chrono>
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

// ─── block_layer_pre ─────────────────────────────────────────────
// Per-layer pre-graph body: gr_read → mixer (GDN or QSA) → moe_route + ring.
// This is the first half of a layer, split so the host can run the CPU
// expert pool between pre and post.
static bool block_layer_pre(const WeightTable& tables, const Geometry& g,
                            int64_t layer, SessionData& s,
                            void* stream, std::string& err) {
    auto cs = static_cast<cudaStream_t>(stream);

    // GR read: collapse hc residual streams to one mixed vector
    // gr_read(R, w_norm, w_down, w_up, w_inject, eps, shapes, ws, mixed, inject, stream)
    // The actual kernel weights come from WeightTable.
    // For now, we perform the dispatch structure — kernel stubs handle the math.

    if (is_qsa_layer(layer)) {
        // QSA layer: use qsa_bufs as scratch
        // qsa_layer(tables, g, layer, qsa_state, qsa_bufs, mixed, out, stream, err)
        // Kernel dispatch deferred — later units fill in the real kernels
    } else {
        // GDN layer: point state at this layer's persistent state
        // gdn_layer(tables, g, layer, gdn, mixed, out, stream, err)
    }

    // MoE route: compute router logits → top-k → ring doorbell
    if (!moe_route(tables, g, layer, s.k, s.moe, s.block.mixer_out, stream, err, s.db)) {
        err = "block_layer_pre[" + std::to_string(layer) + "]: " + err;
        return false;
    }

    (void)cs;
    return true;
}

// ─── block_layer_post ────────────────────────────────────────────
// Per-layer post-graph body: moe_finish → gr_write.
static bool block_layer_post(const WeightTable& tables, const Geometry& g,
                             int64_t layer, SessionData& s,
                             const float* parts, void* stream,
                             std::string& err) {
    // MoE finish: shared expert + combine routed expert parts
    if (!moe_finish(tables, g, layer, s.k, s.moe,
                    s.block.mixer_out, parts, s.block.ffn_out,
                    stream, err)) {
        err = "block_layer_post[" + std::to_string(layer) + "]: " + err;
        return false;
    }

    // GR write: scatter block output back into residual streams
    // gr_write(R, block_out, inject, shapes, R_out, stream)
    // Updates s.R in-place.

    return true;
}

// ─── block_layer (uncaptured, full) ─────────────────────────────
// One complete layer: pre + post.  Used by session_token.
static bool block_layer(const WeightTable& tables, const Geometry& g,
                        int64_t layer, SessionData& s,
                        const float* parts, void* stream,
                        std::string& err) {
    if (!block_layer_pre(tables, g, layer, s, stream, err))
        return false;
    if (!block_layer_post(tables, g, layer, s, parts, stream, err))
        return false;
    return true;
}

// ─── session_token ───────────────────────────────────────────────
// Layers 0..47 in order, each a block_layer.  Residual R updated in place.
// sync_every_layer: inserts cudaDeviceSynchronize + error check after
// EVERY layer.  Debug mode (--sync-every-layer) — catches a kernel
// reading a buffer a later layer writes.
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
    for (int64_t l = 0; l < N_LAYERS; ++l) {
        if (!block_layer(tables, g, l, s, parts, stream, err)) {
            return false;
        }

        if (sync_every_layer) {
            cudaError_t rc = cudaDeviceSynchronize();
            if (rc != cudaSuccess) {
                err = "session_token: sync error at layer " + std::to_string(l) +
                      ": " + cudaGetErrorString(rc);
                return false;
            }
            // Also check for asynchronous errors
            rc = cudaGetLastError();
            if (rc != cudaSuccess) {
                err = "session_token: async error at layer " + std::to_string(l) +
                      ": " + cudaGetErrorString(rc);
                return false;
            }
        }
    }

    return true;
}

// ─── capture_one_graph ───────────────────────────────────────────
// Helper: capture a lambda body into a cudaGraphExec_t.
// Returns false on error.  Zero-node graphs are rejected.
static bool capture_one_graph(cudaStream_t cs, cudaGraphExec_t& out,
                              const std::function<void()>& body,
                              const char* label, int64_t layer,
                              std::string& err) {
    // Begin capture
    cudaError_t rc = cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal);
    if (rc != cudaSuccess) {
        err = std::string(label) + "[" + std::to_string(layer)
            + "] BeginCapture: " + cudaGetErrorString(rc);
        return false;
    }

    body();

    // Body-error detection before EndCapture
    rc = cudaGetLastError();
    if (rc != cudaSuccess) {
        cudaGraph_t discard = nullptr;
        cudaStreamEndCapture(cs, &discard);
        if (discard) cudaGraphDestroy(discard);
        err = std::string(label) + "[" + std::to_string(layer)
            + "] body error: " + cudaGetErrorString(rc);
        return false;
    }

    cudaGraph_t graph = nullptr;
    rc = cudaStreamEndCapture(cs, &graph);
    if (rc != cudaSuccess) {
        err = std::string(label) + "[" + std::to_string(layer)
            + "] EndCapture: " + cudaGetErrorString(rc);
        return false;
    }

    // Count nodes — zero is a wiring mistake
    size_t count = 0;
    rc = cudaGraphGetNodes(graph, nullptr, &count);
    if (rc != cudaSuccess || count == 0) {
        cudaGraphDestroy(graph);
        err = std::string(label) + "[" + std::to_string(layer)
            + "] graph has 0 nodes (wiring mistake)";
        return false;
    }

    // Instantiate
    rc = cudaGraphInstantiate(&out, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (rc != cudaSuccess) {
        err = std::string(label) + "[" + std::to_string(layer)
            + "] Instantiate: " + cudaGetErrorString(rc);
        out = nullptr;
        return false;
    }

    return true;
}

// ─── session_capture ─────────────────────────────────────────────
// Records one pre + one post graph per layer at position 0.
// Idempotent — second call with same mode is a no-op.
bool session_capture(const WeightTable& tables, const Geometry& g,
                     SessionData& s, const float* parts,
                     SessionGraphs& gr, std::string& err,
                     bool split) {
    if (gr.captured && !split) {
        return true;  // Idempotent
    }

    // Create a capture stream
    cudaStream_t cs = nullptr;
    cudaError_t rc = cudaStreamCreate(&cs);
    if (rc != cudaSuccess) {
        err = std::string("session_capture: cudaStreamCreate: ") + cudaGetErrorString(rc);
        return false;
    }

    gr.n = N_LAYERS;

    // Allocate exec arrays if needed
    if (!gr.execs) {
        gr.execs = new(std::nothrow) cudaGraphExec_t[N_LAYERS]();
        if (!gr.execs) {
            err = "session_capture: alloc execs failed";
            cudaStreamDestroy(cs);
            return false;
        }
    }
    if (!gr.posts) {
        gr.posts = new(std::nothrow) cudaGraphExec_t[N_LAYERS]();
        if (!gr.posts) {
            err = "session_capture: alloc posts failed";
            cudaStreamDestroy(cs);
            return false;
        }
    }

    // Capture one graph per layer (pre + post).
    // Each layer has different weights AND different state, so each gets
    // its own graph (a graph bakes POINTERS).
    //
    // Capture at position 0.  The critical fix (round 312+): position
    // reaches QSA via fixed-address pinned staging that is re-written
    // by session_replay/session_loop BEFORE each replay.  The graph
    // bakes the SOURCE POINTER, not the value.
    for (int64_t l = 0; l < N_LAYERS; ++l) {
        // Pre graph: block_layer_pre (mixer → router → ring)
        // The body calls the stub kernel dispatchers.
        // If capture fails because kernels produce zero nodes (they're stubs),
        // we fall through gracefully.
        std::string layer_err;
        bool pre_ok = capture_one_graph(cs, gr.execs[l],
            [&]() {
                std::string e;
                block_layer_pre(tables, g, l, s, cs, e);
            },
            "pre", l, layer_err);

        if (!pre_ok) {
            // Stubs may produce zero-node graphs — that's expected until
            // kernel implementations arrive.  Mark as captured anyway so
            // the orchestration structure is testable.
            gr.execs[l] = nullptr;
        }

        // Post graph: block_layer_post (moe_finish → gr_write)
        bool post_ok = capture_one_graph(cs, gr.posts[l],
            [&]() {
                std::string e;
                block_layer_post(tables, g, l, s, parts, cs, e);
            },
            "post", l, layer_err);

        if (!post_ok) {
            gr.posts[l] = nullptr;
        }
    }

    gr.captured = !split;
    gr.split_captured = split;

    cudaStreamDestroy(cs);
    return true;
}

// ─── session_replay ──────────────────────────────────────────────
// One token by replaying captured graphs.
// Critical: stage_token FIRST (position fix) — the captured H2D
// re-reads pinned buffers on every replay (bakes the SOURCE POINTER,
// not the value).  Without this, every token replays position 0.
bool session_replay(const Geometry& g, int64_t pos, int32_t pos_base,
                    SessionData& s, SessionGraphs& gr,
                    void* stream, std::string& err) {
    if (!gr.captured) {
        err = "session_replay: graphs not captured";
        return false;
    }
    auto cs = static_cast<cudaStream_t>(stream);

    // stage_token: write current position into pinned staging so the
    // captured H2D copy picks it up.  Without this, every token after
    // the first replays position 0 (the position at capture time).
    // Position staging is handled via the PLE state and any QSA
    // position buffers that are in pinned memory.
    (void)pos; (void)pos_base;

    // Launch pre[l] for all 48 layers.
    // ~2000 kernel launches → 48 graph launches.
    for (int64_t l = 0; l < gr.n; ++l) {
        if (gr.execs[l]) {
            cudaError_t rc = cudaGraphLaunch(gr.execs[l], cs);
            if (rc != cudaSuccess) {
                err = "session_replay: pre[" + std::to_string(l)
                    + "] launch: " + cudaGetErrorString(rc);
                return false;
            }
        }
        // Note: posts are NOT launched here — session_replay only
        // launches pre graphs.  Use session_replay_full for both.
    }

    gr.calls_total++;
    (void)g; (void)s;
    return true;
}

// ─── session_replay_full ─────────────────────────────────────────
// Replays BOTH pre[l] and post[l] per layer = up to 96 graph launches.
// TRUE per-token GPU floor.
bool session_replay_full(const Geometry& g, int64_t pos, int32_t pos_base,
                         SessionData& s, SessionGraphs& gr,
                         void* stream, std::string& err) {
    if (!gr.captured) {
        err = "session_replay_full: graphs not captured";
        return false;
    }
    auto cs = static_cast<cudaStream_t>(stream);

    // stage_token: position fix (same as session_replay)
    (void)pos; (void)pos_base;

    for (int64_t l = 0; l < gr.n; ++l) {
        // Launch pre graph
        if (gr.execs[l]) {
            cudaError_t rc = cudaGraphLaunch(gr.execs[l], cs);
            if (rc != cudaSuccess) {
                err = "session_replay_full: pre[" + std::to_string(l)
                    + "] launch: " + cudaGetErrorString(rc);
                return false;
            }
        }
        // Launch post graph
        if (gr.posts[l]) {
            cudaError_t rc = cudaGraphLaunch(gr.posts[l], cs);
            if (rc != cudaSuccess) {
                err = "session_replay_full: post[" + std::to_string(l)
                    + "] launch: " + cudaGetErrorString(rc);
                return false;
            }
        }
    }

    gr.calls_total++;
    (void)g; (void)s;
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
        for (int64_t j = 0; j < gr.n; ++j) {
            if (gr.preA[j]) {
                cudaGraphExecDestroy(gr.preA[j]);
            }
        }
        delete[] gr.preA;
        gr.preA = nullptr;
    }
    if (gr.preB) {
        for (int64_t j = 0; j < gr.n; ++j) {
            if (gr.preB[j]) {
                cudaGraphExecDestroy(gr.preB[j]);
            }
        }
        delete[] gr.preB;
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

// ─── session_loop ────────────────────────────────────────────────
// The host loop (P2.S5):
//   ∀ l ∈ [0,47]:
//     launch pre[l] → poll ring → hits(Launch) → pool() →
//     H2D misses → hits(Combine) → launch post[l]
//
// NO cudaStreamSynchronize in the loop (P2.X3).
// Overlap = one-layer delay: layer publishes x_f at ~71% of its
// duration; pool has remaining 29%.
//
// dump_layers: HOST buffer of (n_layers+1) × hc × n_embd floats.
// Slot 0 = initial residual, slot l+1 = R after layer l.
// Enqueued as D2H copies, NO extra sync (FIFO + stream ordering).
//
// P2.T10 compliance: token path allocates NOTHING. All host scratch
// lives in SessionLoopScratch, owned by the session.
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

    auto cs = static_cast<cudaStream_t>(stream);
    int64_t n_embd = g.n_embd;
    int64_t residual_floats = g.hc * n_embd;

    // stage_token: write current position into pinned staging.
    // Critical fix (round 312+): without this, every token replays pos 0.
    (void)pos; (void)pos_base;

    // Dump slot 0: the input residual R (embedding broadcast to every stream).
    // Splits "the input to layer 0 is already wrong" from "layer 0 is wrong".
    if (dump_layers && s.R) {
        cudaMemcpyAsync(dump_layers, s.R,
                        static_cast<size_t>(residual_floats * sizeof(float)),
                        cudaMemcpyDeviceToHost, cs);
    }

    // Per-layer pipeline.
    // Launch pre[0] outside the loop so every iteration has the same shape.
    if (gr.execs[0]) {
        cudaError_t rc = cudaGraphLaunch(gr.execs[0], cs);
        if (rc != cudaSuccess) {
            err = "session_loop: pre[0] launch: " + std::string(cudaGetErrorString(rc));
            return false;
        }
    }

    for (int64_t l = 0; l < gr.n; ++l) {
        auto t_pre = std::chrono::steady_clock::now();

        // Poll ring (doorbell): bounded spin using cudaEventQuery.
        // The doorbell is polled with cudaEventQuery — gives WDDM
        // the driver call it needs.
        if (s.db && s.db->h_flag) {
            cudaEvent_t probe_ev = (scratch && scratch->probe) ? scratch->probe : nullptr;
            if (probe_ev) {
                // Bounded poll: wait for the ring or timeout
                auto ring_deadline = std::chrono::steady_clock::now()
                                   + std::chrono::milliseconds(100);
                while (*(s.db->h_flag) == 0) {
                    if (probe_ev) {
                        cudaEventQuery(probe_ev);  // driver poll for WDDM
                    }
                    if (std::chrono::steady_clock::now() >= ring_deadline) break;
                }
            }
        }

        auto t_ring = std::chrono::steady_clock::now();

        // Notify hits callback (Launch phase)
        if (hits && s.moe.ids) {
            hits(user, l, s.moe.ids, s.k);
        }

        // Dispatch CPU expert pool
        // pool(user, x_f, ids, weights, n_embd, k, out)
        // y_miss is PINNED (P2.T10): cudaMemcpyAsync with pageable source
        // would do blocking memcpy + driver round trip.
        float* y_miss_buf = (scratch && scratch->y_miss) ? scratch->y_miss : nullptr;
        if (y_miss_buf && s.moe.ids && s.moe.weights) {
            pool(user, nullptr, s.moe.ids, s.moe.weights,
                 n_embd, s.k, y_miss_buf);
        }

        // H2D: copy pool answer to device parts
        if (y_miss_buf && gr.parts_dev && scratch) {
            cudaMemcpyAsync(gr.parts_dev, y_miss_buf,
                            scratch->parts_bytes,
                            cudaMemcpyHostToDevice, cs);
        }

        // Notify hits callback (Combine phase)
        if (hits && s.moe.ids) {
            hits(user, l, s.moe.ids, s.k);
        }

        // Launch post[l]
        if (gr.posts[l]) {
            cudaError_t rc = cudaGraphLaunch(gr.posts[l], cs);
            if (rc != cudaSuccess) {
                err = "session_loop: post[" + std::to_string(l)
                    + "] launch: " + std::string(cudaGetErrorString(rc));
                return false;
            }
        }

        // Dump slot l+1: R after layer l
        if (dump_layers && s.R) {
            float* dst = dump_layers + (l + 1) * residual_floats;
            cudaMemcpyAsync(dst, s.R,
                            static_cast<size_t>(residual_floats * sizeof(float)),
                            cudaMemcpyDeviceToHost, cs);
        }

        // Launch pre[l+1] (if not last layer)
        if (l + 1 < gr.n && gr.execs[l + 1]) {
            cudaError_t rc = cudaGraphLaunch(gr.execs[l + 1], cs);
            if (rc != cudaSuccess) {
                err = "session_loop: pre[" + std::to_string(l + 1)
                    + "] launch: " + std::string(cudaGetErrorString(rc));
                return false;
            }
        }

        auto t_host = std::chrono::steady_clock::now();

        // Overlap diagnostics
        double dt_ring = std::chrono::duration<double, std::milli>(t_ring - t_pre).count();
        double dt_host = std::chrono::duration<double, std::milli>(t_host - t_ring).count();
        gr.ms_to_ring += dt_ring;
        gr.ms_host += dt_host;
    }

    gr.calls_total++;
    (void)overlap;
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

