// generate_driver.hpp - decode-loop position accounting (teacher forcing then sampling), logits-dump rows, expert-cache
// auto sizing, cache-slot borrowing for the prompt path, PCIe share clamp and speculative draft-cell bounds.
#pragma once
#include <cmath>
#include <cstdint>

namespace baloon {
namespace gdrv {

// Tokens produced after processing position `pos`: none before the last prompt position.
inline int64_t produced_at(int64_t np, int64_t pos) { return np <= pos + 1 ? pos + 2 - np : 0; }
// The loop stops at the position where `max_new` tokens exist; rows written = np - 1 + max_new.
inline int64_t stop_position(int64_t np, int64_t max_new) { return np - 2 + max_new; }
inline int64_t dump_rows(int64_t np, int64_t max_new) { return np - 1 + max_new; }
inline bool context_ok(int64_t np, int64_t max_new, int64_t ctx) { return np >= 1 && max_new >= 1 && np + max_new <= ctx; }

inline int64_t slots_auto(int64_t free_b, int64_t reserve, int64_t blob) {
    if (blob <= 0) return 0;
    const int64_t d = free_b > reserve ? free_b - reserve : 0;
    return d / blob;
}
inline int64_t ceil_mul(int64_t need, int64_t blob) { return (need + blob - 1) / blob * blob; }
inline int64_t borrow_slots(int64_t need, int64_t blob) { return (need + blob - 1) / blob; }  // k = ceil(need/blob)
constexpr int64_t kDecodeSlotsFloor = 128;
inline bool can_borrow(int64_t slots, int64_t k) { return k + kDecodeSlotsFloor <= slots; }
inline int64_t first_lent(int64_t slots, int64_t k) { return slots - k; }

inline int clamp_pcie(double frac) {
    if (!(frac == frac)) return 0;  // NaN
    const double r = std::floor(frac * 256.0 + 0.5);
    return r < 0 ? 0 : (r > 256 ? 256 : (int)r);
}

// Speculative round: window T <= S, accepted a <= T-1, emitted a+1 <= S, draft cells p + a + j.
inline int64_t emitted(int64_t a) { return a + 1; }
inline bool spec_round_ok(int64_t T, int64_t S, int64_t a) { return T <= S && a + 1 <= T && a >= 0; }
inline int64_t next_window_start(int64_t p, int64_t a) { return p + a + 1; }
inline int64_t draft_cell(int64_t p, int64_t a, int64_t j) { return p + a + j; }
// FINDING (theory): draft cells can exceed max_context by up to S-3; K/V + RoPE need S cells of slack.
inline int64_t draft_cell_slack(int64_t S) { return S; }
inline bool serve_admit(int64_t n, int64_t max_new, int64_t ctx) { return n + max_new + 8 <= ctx; }

}  // namespace gdrv
}  // namespace baloon
