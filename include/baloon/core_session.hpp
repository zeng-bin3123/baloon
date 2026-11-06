// core_session.hpp - one token through all layers (host model): session arena sizing/carving, per-layer GDN/QSA state
// indexing, stage-prefix timing differences and the doorbell-driven session loop.
// session_bytes = align256(sum of region sizes) + 4096 and session_init carves each region at a 256-byte aligned cursor,
// so the carve fits inside the reported size (<= 16 regions). GDN layers are numbered separately from QSA layers
// (l % 4 == 3 is QSA): 36 GDN states, 12 QSA states. Stage times are differences of consecutive prefix times.
// session_loop expects ring value l+1 from layer l; the dump holds (n_layers+1)*hc*n_embd floats.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "baloon/core_layer.hpp"

namespace baloon {
namespace core {

constexpr uint64_t SESSION_STATE_ALIGN = 256;
constexpr int kSessionRegions = 7;
constexpr int kSessionPrefixes = 5;

constexpr uint64_t align_up(uint64_t n, uint64_t a = SESSION_STATE_ALIGN) { return (n + a - 1) / a * a; }
uint64_t session_bytes(const std::vector<uint64_t>& parts);
constexpr uint64_t gdn_state_floats() { return 128ull * 48 * 128 + 10240ull * 3; }      // 817152 per GDN layer
int gdn_point_at(int layer);        // index among GDN layers, -1 for a QSA layer
int qsa_point_at(int layer);        // index among QSA layers, -1 for a GDN layer
uint64_t ple_hist_bytes(uint64_t n_tokens, uint64_t n_embd_ple);
constexpr uint64_t session_dump_floats(uint64_t n_layers, uint64_t hc, uint64_t n_embd) { return (n_layers + 1) * hc * n_embd; }

using LayerFn = std::function<void(int layer, int stage_prefix)>;

struct TokenGraph { std::vector<std::function<void()>> nodes; bool captured = false; };

struct Session {
    uint64_t bytes = 0;
    std::vector<uint8_t> arena;
    uint64_t off[kSessionRegions] = {};
    uint64_t size[kSessionRegions] = {};
    int n_layers = 0;
    int64_t token = 0;                              // the token the captured graphs process
    TokenGraph token_graph;                         // embed + all layers + head
    TokenGraph prefix_graph[kSessionPrefixes];      // layers with stage_prefix 1..5
    std::function<void(int64_t)> embed;
    LayerFn layer;
    std::function<void()> head;
    std::function<int64_t()> clock_ns;              // injectable clock (default: steady clock)
};

bool session_init(Session& s, const std::vector<uint64_t>& parts, int n_layers);
void session_zero(Session& s);
bool session_capture(Session& s, LayerFn layer, int n_layers);                       // prefix graphs 1..5
bool session_capture_token(Session& s, std::function<void(int64_t)> embed, LayerFn layer, std::function<void()> head);
void session_graphs_free(Session& s);
void token_graph_free(TokenGraph& g);

bool session_run_token(Session& s, int64_t token);                    // replay the captured token graph
bool session_token(Session& s, int64_t token, bool use_graph = true); // graph when captured, else the eager path
bool stage_token(Session& s, int64_t token, int stage_prefix);        // run the prefix graph k (1..5)

int64_t session_replay(Session& s, int reps);                         // total ns of reps token replays
int64_t session_replay_full(Session& s);                              // ns of the prefix-5 (full) graph
bool session_replay_stage_prefixes(Session& s, std::vector<int64_t>& prefix_ns);   // 5 prefix times (ns)
bool session_replay_stages(Session& s, std::vector<int64_t>& stage_ns);            // 5 stage times = prefix differences
bool session_replay_stage_sweep(Session& s, int reps, std::vector<int64_t>& stage_ns);  // min over reps per prefix
bool session_replay_stages_per_layer(Session& s, std::vector<std::vector<int64_t>>& per_layer);  // [layer][stage]

std::vector<int64_t> stage_diffs(const std::vector<int64_t>& prefix_ns);

// Scratch for the parts staging of the loop: k * n_embd * 4 bytes (non-empty for positive inputs).
struct SessionLoopScratch {
    std::vector<float> parts;
    uint64_t k = 0, n_embd = 0;
    bool init(uint64_t k_parts, uint64_t n_embd_);
    void free();
    uint64_t bytes() const { return k * n_embd * 4; }
};

constexpr bool spin_boundary(uint64_t spins) { return (spins + 1) % 1024 == 0; }
// Layer l waits for ring value l+1, then consume(l) runs. After each 1024-spin boundary `escalate` is called; if it
// returns false (or max_spins is exhausted) the loop gives up and returns false.
bool session_loop(Doorbell& ring, int n_layers, const std::function<void(int layer)>& consume,
                  uint64_t max_spins = UINT64_MAX, const std::function<bool()>& escalate = nullptr);

}  // namespace core
}  // namespace baloon
