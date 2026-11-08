// moe_split_protocol.hpp - the MoE block split at the router: moe_route publishes ids/weights for THIS layer, the pool computes THIS
// layer's expert outputs, moe_finish combines them with THIS layer's weights plus the shared expert (added plain).
#pragma once
#include <cstdint>
#include <vector>

namespace baloon {
namespace moe_split {

constexpr int kMaxK = 64;
constexpr int kNativeExperts = 512, kNativeK = 10;

// sum_j w[j] * parts[j] + shared. Lists are zipped: the shorter one bounds the sum.
template <class T> T combine(const std::vector<T>& w, const std::vector<T>& parts, T shared) {
    T acc = T(0);
    for (size_t j = 0; j < w.size() && j < parts.size(); ++j) acc += w[j] * parts[j];
    return acc + shared;
}
// moe_finish = moe_shared ; moe_combine_parts (the shared output is an input here: it is a pure function of x, so running it
// before or after the pool changes nothing).
template <class T> T finish(T shared, const std::vector<T>& w, const std::vector<T>& parts) { return combine(w, parts, shared); }

// The correct loop combines layer l's expert outputs with layer l's weights; the delayed loop (L100) uses layer l+1's.
template <class T, class Route, class Outs, class Sh> T layer_out(Route route, Outs outs, Sh sh, uint64_t l) { return finish<T>(sh(l), route(l), outs(l)); }
template <class T, class Route, class Outs, class Sh> T stale_out(Route route, Outs outs, Sh sh, uint64_t l) { return finish<T>(sh(l), route(l + 1), outs(l)); }

// The k guard shared by moe_route, moe_finish and moe_combine_parts; the native router additionally needs 512 experts and k = 10.
constexpr bool k_valid(int64_t k) { return 1 <= k && k <= kMaxK; }
constexpr bool native_router_ok(int64_t n_expert, int64_t k) { return n_expert == kNativeExperts && k == kNativeK; }

// Mapped-pinned doorbell regions: x_f (n_embd f32), ids (k i32), weights (k f32), seq (u32), flag (u32).
constexpr uint64_t doorbell_bytes(uint64_t n_embd, uint64_t k) { return n_embd * 4 + k * 4 + k * 4 + 4 + 4; }
constexpr uint64_t parts_bytes(uint64_t k, uint64_t n_embd) { return k * n_embd * 4; }
// The ring sequence the host waits for after layer l is l + 1: strictly increasing, one ring per layer.
constexpr uint64_t ring_value(uint64_t l) { return l + 1; }

}  // namespace moe_split
}  // namespace baloon
