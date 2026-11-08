// session_schedule.hpp - one token through all layers: GDN state slice ownership, the pre[l]/post[l] launch order, stage-time
// differencing and the per-layer residual dump layout.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace session_schedule {

constexpr uint64_t kLayers = 48, kQsaPeriod = 4, kGdnLayers = 36;

// Number of GDN layers strictly before layer l (every layer with l % 4 == 3 is QSA): l - l / 4.
constexpr uint64_t gdn_index(uint64_t l) { return l - l / kQsaPeriod; }
constexpr bool is_qsa(uint64_t l) { return l % kQsaPeriod == kQsaPeriod - 1; }

enum class Kind { Pre, Post };
struct Ev {
    Kind kind; uint64_t layer;
    bool operator==(const Ev& o) const { return kind == o.kind && layer == o.layer; }
};
// The launch order of session_replay_full / session_loop: pre[l] then post[l] for every layer in turn.
inline std::vector<Ev> schedule(uint64_t n_layers) {
    std::vector<Ev> v;
    v.reserve((size_t)(2 * n_layers));
    for (uint64_t l = 0; l < n_layers; ++l) { v.push_back({Kind::Pre, l}); v.push_back({Kind::Post, l}); }
    return v;
}

// Five stage times from prefix times p1..p5 (p0 = 0): consecutive differences, so they telescope to p5.
inline std::vector<int64_t> stage_ms(int64_t p1, int64_t p2, int64_t p3, int64_t p4, int64_t p5) { return {p1, p2 - p1, p3 - p2, p4 - p3, p5 - p4}; }

// Residual dump: (n_layers + 1) slots of n = hc * n_embd floats; slot 0 = input, slot l+1 = residual after layer l.
constexpr uint64_t slot_off(uint64_t l, uint64_t n) { return l * n; }
constexpr uint64_t dump_floats(uint64_t n_layers, uint64_t n) { return (n_layers + 1) * n; }

// GDN recurrence + conv history floats of one layer.
constexpr uint64_t gdn_state_floats(uint64_t state_size, uint64_t v_heads, uint64_t d_conv, uint64_t conv_channels) {
    return state_size * v_heads * state_size + conv_channels * (d_conv - 1);
}

}  // namespace session_schedule
}  // namespace baloon
