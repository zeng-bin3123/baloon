// session_loop_contract.hpp - per-token host loop contracts: ordered layer loop, idempotent capture, one disjoint dump slot
// per layer, telescoping prefix timings, verify-window graph table bound.
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace session_loop {

template <class S, class Step> S run_layers(Step step, uint64_t n, S s) {
    for (uint64_t l = 0; l < n; ++l) s = step(l, s);
    return s;
}

// Capture state: set once; a second capture changes nothing.
template <class G> struct Capture {
    bool captured = false;
    G graphs{};
    void capture(const G& g) {
        if (captured) return;
        captured = true;
        graphs = g;
    }
};

// dump_layers: (n_layers+1) slots of a = hc*n_embd floats; slot l+1 holds R after layer l.
constexpr uint64_t dump_total(uint64_t a, uint64_t nl) { return (nl + 1) * a; }
constexpr uint64_t slot_base(uint64_t a, uint64_t l) { return (l + 1) * a; }

// Stage times from prefix timings P[0..n]: consecutive differences.
inline std::vector<int64_t> stage_times(const std::vector<int64_t>& P) {
    std::vector<int64_t> t;
    for (size_t k = 0; k + 1 < P.size(); ++k) t.push_back(P[k + 1] - P[k]);
    return t;
}

constexpr uint64_t kMaxWindow = 8;
constexpr uint64_t kExecSlots = 9;
constexpr bool exec_index_ok(uint64_t max_t, uint64_t T) { return max_t <= kMaxWindow && T <= max_t && T < kExecSlots; }

}  // namespace session_loop
}  // namespace baloon
