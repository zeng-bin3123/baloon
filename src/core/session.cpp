// session.cpp - host model of the per-token session graphs (see core_session.hpp).
#include "baloon/core_session.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#define REF_SPIN_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define REF_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
#define REF_SPIN_PAUSE() std::this_thread::yield()
#endif

namespace baloon {
namespace core {

uint64_t session_bytes(const std::vector<uint64_t>& parts) {
    uint64_t sum = 0;
    for (uint64_t p : parts) sum += p;
    return align_up(sum, SESSION_STATE_ALIGN) + 4096;
}

int gdn_point_at(int layer) {
    if (layer < 0 || is_qsa_layer(layer)) return -1;
    int n = 0;
    for (int i = 0; i < layer; ++i) n += is_qsa_layer(i) ? 0 : 1;      // GDN layers that precede it
    return n;
}
int qsa_point_at(int layer) {
    if (layer < 0 || !is_qsa_layer(layer)) return -1;
    int n = 0;
    for (int i = 0; i < layer; ++i) n += is_qsa_layer(i) ? 1 : 0;
    return n;
}
uint64_t ple_hist_bytes(uint64_t n_tokens, uint64_t n_embd_ple) { return align_up(n_tokens * n_embd_ple * 4, 16); }

bool session_init(Session& s, const std::vector<uint64_t>& parts, int n_layers) {
    if (parts.empty() || parts.size() > 16 || (int)parts.size() > kSessionRegions || n_layers <= 0) return false;
    const uint64_t total = session_bytes(parts);
    s.arena.assign((size_t)total, 0);
    s.bytes = total;
    s.n_layers = n_layers;
    uint64_t used = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        s.off[i] = used;                                 // each region starts at a 256-byte aligned cursor
        s.size[i] = parts[i];
        used = align_up(used + parts[i], SESSION_STATE_ALIGN);
    }
    return used <= total;
}

void session_zero(Session& s) { std::fill(s.arena.begin(), s.arena.end(), (uint8_t)0); }

void token_graph_free(TokenGraph& g) { g.nodes.clear(); g.captured = false; }
void session_graphs_free(Session& s) {
    token_graph_free(s.token_graph);
    for (auto& g : s.prefix_graph) token_graph_free(g);
    s.embed = nullptr; s.layer = nullptr; s.head = nullptr;
}

bool session_capture(Session& s, LayerFn layer, int n_layers) {
    if (!layer || n_layers <= 0) return false;
    s.n_layers = n_layers;
    s.layer = layer;
    for (int k = 1; k <= kSessionPrefixes; ++k) {
        TokenGraph& g = s.prefix_graph[k - 1];
        token_graph_free(g);
        for (int l = 0; l < n_layers; ++l) g.nodes.push_back([layer, l, k]() { layer(l, k); });
        g.captured = true;
    }
    return true;
}

bool session_capture_token(Session& s, std::function<void(int64_t)> embed, LayerFn layer, std::function<void()> head) {
    if (!layer || s.n_layers <= 0) return false;
    s.embed = embed; s.layer = layer; s.head = head;
    TokenGraph& g = s.token_graph;
    token_graph_free(g);
    Session* sp = &s;
    g.nodes.push_back([sp]() { if (sp->embed) sp->embed(sp->token); });
    for (int l = 0; l < s.n_layers; ++l) g.nodes.push_back([sp, l]() { sp->layer(l, 0); });
    g.nodes.push_back([sp]() { if (sp->head) sp->head(); });
    g.captured = true;
    return true;
}

static void run_graph(const TokenGraph& g) { for (auto& n : g.nodes) n(); }

bool session_run_token(Session& s, int64_t token) {
    if (!s.token_graph.captured) return false;
    s.token = token;
    run_graph(s.token_graph);
    return true;
}

bool session_token(Session& s, int64_t token, bool use_graph) {
    if (use_graph && s.token_graph.captured) return session_run_token(s, token);
    if (!s.layer || s.n_layers <= 0) return false;       // eager path: same order as the graph
    s.token = token;
    if (s.embed) s.embed(token);
    for (int l = 0; l < s.n_layers; ++l) s.layer(l, 0);
    if (s.head) s.head();
    return true;
}

bool stage_token(Session& s, int64_t token, int stage_prefix) {
    if (stage_prefix < 1 || stage_prefix > kSessionPrefixes || !s.prefix_graph[stage_prefix - 1].captured) return false;
    s.token = token;
    run_graph(s.prefix_graph[stage_prefix - 1]);
    return true;
}

static int64_t now_ns(Session& s) {
    if (s.clock_ns) return s.clock_ns();
    using C = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(C::now().time_since_epoch()).count();
}

int64_t session_replay(Session& s, int reps) {
    if (!s.token_graph.captured || reps < 1) return -1;
    const int64_t t0 = now_ns(s);
    for (int i = 0; i < reps; ++i) run_graph(s.token_graph);
    return now_ns(s) - t0;
}

int64_t session_replay_full(Session& s) {
    if (!s.prefix_graph[kSessionPrefixes - 1].captured) return -1;
    const int64_t t0 = now_ns(s);
    run_graph(s.prefix_graph[kSessionPrefixes - 1]);
    return now_ns(s) - t0;
}

std::vector<int64_t> stage_diffs(const std::vector<int64_t>& p) {
    std::vector<int64_t> d;
    for (size_t i = 0; i < p.size(); ++i) d.push_back(i == 0 ? p[0] : p[i] - p[i - 1]);   // stage i = t_{i+1} - t_i
    return d;
}

bool session_replay_stage_prefixes(Session& s, std::vector<int64_t>& prefix_ns) {
    prefix_ns.clear();
    for (int k = 1; k <= kSessionPrefixes; ++k) {
        if (!s.prefix_graph[k - 1].captured) { prefix_ns.clear(); return false; }
        const int64_t t0 = now_ns(s);
        run_graph(s.prefix_graph[k - 1]);
        prefix_ns.push_back(now_ns(s) - t0);
    }
    return true;
}

bool session_replay_stages(Session& s, std::vector<int64_t>& stage_ns) {
    std::vector<int64_t> p;
    if (!session_replay_stage_prefixes(s, p)) { stage_ns.clear(); return false; }
    stage_ns = stage_diffs(p);
    return true;
}

bool session_replay_stage_sweep(Session& s, int reps, std::vector<int64_t>& stage_ns) {
    stage_ns.clear();
    if (reps < 1) return false;
    std::vector<int64_t> best(kSessionPrefixes, INT64_MAX);
    for (int r = 0; r < reps; ++r) {
        std::vector<int64_t> p;
        if (!session_replay_stage_prefixes(s, p)) return false;
        for (int k = 0; k < kSessionPrefixes; ++k) best[(size_t)k] = std::min(best[(size_t)k], p[(size_t)k]);
    }
    stage_ns = stage_diffs(best);
    return true;
}

bool session_replay_stages_per_layer(Session& s, std::vector<std::vector<int64_t>>& per_layer) {
    per_layer.clear();
    if (!s.layer || s.n_layers <= 0) return false;
    for (int l = 0; l < s.n_layers; ++l) {
        std::vector<int64_t> p;
        for (int k = 1; k <= kSessionPrefixes; ++k) {
            const int64_t t0 = now_ns(s);
            s.layer(l, k);
            p.push_back(now_ns(s) - t0);
        }
        per_layer.push_back(stage_diffs(p));
    }
    return true;
}

bool SessionLoopScratch::init(uint64_t k_parts, uint64_t n_embd_) {
    free();
    if (k_parts == 0 || n_embd_ == 0) return false;
    k = k_parts; n_embd = n_embd_;
    parts.assign((size_t)(k * n_embd), 0.0f);
    return true;
}
void SessionLoopScratch::free() { parts.clear(); parts.shrink_to_fit(); k = n_embd = 0; }

bool session_loop(Doorbell& ring, int n_layers, const std::function<void(int)>& consume, uint64_t max_spins,
                  const std::function<bool()>& escalate) {
    if (!ring.word || n_layers <= 0) return false;
    uint64_t spins = 0;
    for (int l = 0; l < n_layers; ++l) {
        const uint64_t want = (uint64_t)l + 1;              // layer l publishes ring value l + 1
        while (doorbell_value(&ring) < want) {
            if (spins >= max_spins) return false;
            if (spin_boundary(spins)) {
                if (escalate && !escalate()) return false;
                std::this_thread::yield();
            } else {
                REF_SPIN_PAUSE();
            }
            ++spins;
        }
        if (consume) consume(l);
    }
    return true;
}

}  // namespace core
}  // namespace baloon
