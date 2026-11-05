// generate_driver_accounting.hpp - position and row accounting of the `ref generate` driver: which positions produce
// tokens, the logits-dump row count, context fit, the batched-prefill split, the auto-sized expert cache, the PCIe
// fraction clamp, profile truncation, the routing-trace layer index and serve admission. Nat subtraction in the model
// truncates at 0; sat_sub reproduces that on unsigned values.
#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace baloon {
namespace generate {

inline uint64_t sat_sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

// positions and rows
inline uint64_t dump_positions(uint64_t n_prompt, uint64_t max_new) { return sat_sub(n_prompt, 1) + max_new; }
inline uint64_t last_pos(uint64_t n_prompt, uint64_t max_new) { return sat_sub(n_prompt, 2) + max_new; }
inline uint64_t first_produced_pos(uint64_t n_prompt) { return sat_sub(n_prompt, 1); }
// does the token at `pos` get produced (sampled) by the loop?
inline bool produces_token(uint64_t n_prompt, uint64_t max_new, uint64_t pos) {
    return pos >= first_produced_pos(n_prompt) && pos <= last_pos(n_prompt, max_new);
}
// request fits the context: n <= ctx - max_new
inline bool size_check(uint64_t n_tokens, uint64_t max_new, uint64_t max_context) { return n_tokens <= sat_sub(max_context, max_new); }

// prefill split
inline uint64_t n_batched(uint64_t n_prompt, uint64_t prefill_until) {
    return (0 < prefill_until && prefill_until < sat_sub(n_prompt, 1)) ? prefill_until : sat_sub(n_prompt, 1);
}

// auto-sized expert cache: (free - reserve) / blob slots, saturating at 0
inline uint64_t auto_slots(uint64_t free_bytes, uint64_t reserve, uint64_t blob) { return blob == 0 ? 0 : sat_sub(free_bytes, reserve) / blob; }
inline uint64_t profile_want(uint64_t profile_size, uint64_t slots) { return std::min(profile_size, slots); }

// PCIe fraction
inline int64_t clamp_pcie(int64_t x) { return x < 0 ? 0 : (x > 256 ? 256 : x); }
inline int64_t pcie_num(double frac) {
    if (!(frac == frac)) return 0;                                  // NaN -> 0
    const double r = std::floor(frac * 256.0 + 0.5);
    if (r <= -1e18) return 0;
    if (r >= 1e18) return 256;
    return clamp_pcie((int64_t)r);
}

// routing trace: the counter has already advanced past the layer
inline int64_t trace_layer(uint64_t layers_after) { return (int64_t)layers_after - 1; }

// serve admission: prompt + max_new + 8 must fit the context
inline bool serve_accepts(uint64_t n, uint64_t max_new, uint64_t ctx) { return !(ctx < n + max_new + 8); }

}  // namespace generate
}  // namespace baloon
