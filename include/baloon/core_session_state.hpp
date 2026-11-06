// core_session_state.hpp - per-session state tying the arena, the graph set and the hit sinks together (host model).
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "baloon/core_hit_hook.hpp"
#include "baloon/core_layout.hpp"
#include "baloon/core_session.hpp"

namespace baloon {
namespace core {

// Router pool callbacks: single token (layer, experts[k] -> out) and a multi-token window (rows * k experts).
using PoolFn = std::function<bool(int layer, const int32_t* experts, int k, float* out)>;
using PoolMultiFn = std::function<bool(int layer, const int32_t* experts, int rows, int k, float* out)>;

// The captured graphs of one session: the full token graph and the five prefix graphs.
struct SessionGraphs {
    bool captured() const { return token.captured; }
    TokenGraph token;
    TokenGraph prefix[kSessionPrefixes];
    void free_all() { token_graph_free(token); for (auto& g : prefix) token_graph_free(g); }
};

struct SessionState {
    Session session;
    ModelGeometry geom;
    SessionGraphs graphs;
    TokenHits hits;
    HitFn hit;                                   // notified per routed pick when set
    int64_t position = 0;
    bool init(const std::vector<uint64_t>& parts) { return session_init(session, parts, geom.n_layers) && check_all(geom); }
    void pick(HitPhase ph, int layer, int expert) { if (hit) hit(ph, layer, expert); }
    void reset() { session_zero(session); hits.clear(); position = 0; }
};

}  // namespace core
}  // namespace baloon
