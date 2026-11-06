// core_layout.hpp - model geometry and per-layer views (host model). The 48-layer pattern puts QSA at l % 4 == 3, so
// 36 GDN + 12 QSA; each layer owns one dense index within its own kind (gdn_point_at / qsa_point_at).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "baloon/core_graph.hpp"
#include "baloon/core_session.hpp"

namespace baloon {
namespace core {

struct ModelGeometry {
    int n_layers = 48;
    uint64_t n_embd = 2048, hc = 4, n_head = 16, head_dim = 256, n_head_kv = 2;
    int n_qsa() const { int n = 0; for (int l = 0; l < n_layers; ++l) n += is_qsa_layer(l); return n; }
    int n_gdn() const { return n_layers - n_qsa(); }
};

struct LayerView {
    int layer = 0;
    LayerType type = LayerType::Gdn;
    int point = -1;                 // index among layers of the same kind
    uint64_t state_floats = 0;      // recurrent state floats (GDN only)
};

inline LayerView layer_view(const ModelGeometry&, int l) {
    LayerView v; v.layer = l; v.type = layer_type_of(l);
    v.point = v.type == LayerType::Qsa ? qsa_point_at(l) : gdn_point_at(l);
    v.state_floats = v.type == LayerType::Gdn ? gdn_state_floats() : 0;
    return v;
}

// One layer is consistent when it lies in range, its point index is valid and of its own kind, and the attention shapes
// of a QSA layer are valid. `why` receives a reason on failure.
inline bool check_layer(const ModelGeometry& g, int l, std::string* why = nullptr) {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    if (l < 0 || l >= g.n_layers) return fail("layer out of range");
    const LayerView v = layer_view(g, l);
    if (v.point < 0) return fail("no point index");
    if (v.type == LayerType::Qsa) {
        if (gdn_point_at(l) != -1) return fail("QSA layer has a GDN index");
        if (!qsa_shapes(g.n_head, g.head_dim, g.n_head_kv).ok) return fail("bad attention shapes");
    } else if (qsa_point_at(l) != -1) return fail("GDN layer has a QSA index");
    return true;
}

// Whole model: every layer valid, point indices dense and unique per kind, kind counts add up.
inline bool check_all(const ModelGeometry& g, std::string* why = nullptr) {
    if (g.n_layers <= 0) { if (why) *why = "no layers"; return false; }
    std::vector<int> seen_g, seen_q;
    for (int l = 0; l < g.n_layers; ++l) {
        if (!check_layer(g, l, why)) return false;
        const LayerView v = layer_view(g, l);
        auto& s = v.type == LayerType::Qsa ? seen_q : seen_g;
        if (v.point != (int)s.size()) { if (why) *why = "points not dense"; return false; }
        s.push_back(v.point);
    }
    if ((int)seen_g.size() != g.n_gdn() || (int)seen_q.size() != g.n_qsa()) { if (why) *why = "kind counts"; return false; }
    return true;
}

}  // namespace core
}  // namespace baloon
