// gdn_reference.hpp - launch geometry and sliding-window semantics of the reference gated-delta-net kernels.
// The conv state holds d_conv-1 taps; kernel tap 0 reads the OLDEST tap, the new input lands in the last tap.
#pragma once
#include <cstdint>

namespace baloon {
namespace gdn_ref {

constexpr int kJThreads = 32;
constexpr int kMaxH = 1;
constexpr int kConvThreads = 256;
constexpr int kBetaThreads = 128;
constexpr int kMaxState = 128;

// Slot i of the conv state after one step (shift left, append x).
template <class T, class F> inline T slide(F st, T x, int dc, int i) { return i + 2 < dc ? st(i + 1) : x; }
// Tap i of the dot-product window: stored taps first, then the new input.
template <class T, class F> inline T window(F st, T x, int dc, int i) { return i < dc - 1 ? st(i) : x; }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
inline int step_col_blocks(int S) { return ceil_div(S, kJThreads); }
inline int head_blocks(int hv) { return (hv + kMaxH - 1) / kMaxH; }
inline int col_block(int j) { return j / kJThreads; }
inline int col_lane(int j) { return j % kJThreads; }
inline int src_head(int h, int hk) { return h % hk; }
inline int conv_blocks(int C) { return ceil_div(C, kConvThreads); }
inline int beta_blocks(int n) { return ceil_div(n, kBetaThreads); }
inline bool l2_width_ok(int n) { return n <= 32 * 32; }
inline bool staged_ok(int S) { return S <= kMaxState; }

}  // namespace gdn_ref
}  // namespace baloon
