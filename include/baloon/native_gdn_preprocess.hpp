// native_gdn_preprocess.hpp - geometry and span-safety arithmetic of the native GDN preprocessing kernels
// (conv+silu, l2-norm, beta sigmoid, gate softplus, out-norm). Conv uses four taps: history[3c..3c+2] and the
// input; history slides to [h1,h2,x]. Norm kernels use 256 threads per row, only threads < 128 are active.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace native_gdn_pre {

constexpr int kS = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kConvTaps = 4;
constexpr int kHistTaps = 3;
constexpr int kMaxCount = 65535;
constexpr uint64_t kUMax = UINT64_MAX;

template <class T> struct Hist3 { T h0, h1, h2; };
template <class T> inline Hist3<T> slide_hist(const Hist3<T>& h, T x) { return Hist3<T>{h.h1, h.h2, x}; }

inline int64_t weight_index(int64_t c, int tap) { return c * kConvTaps + tap; }
inline int64_t hist_index(int64_t c, int tap) { return c * kHistTaps + tap; }
inline int conv_blocks(int ch) { return (ch + kThreads - 1) / kThreads; }
inline int conv_block_of(int c) { return c / kThreads; }
inline bool count_ok(int n) { return n >= 1 && n <= kMaxCount; }
inline int active_warp(int col) { return col / 32; }
inline int64_t row_offset(int64_t b, int col) { return b * kS + col; }
inline bool partial_lane_ok(int lane) { return lane < kWarps; }

struct Span { uint64_t addr; uint64_t bytes; };
inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }
inline bool span_valid(const Span& s) { return s.bytes <= kUMax - s.addr; }
inline bool disjoint(const Span& x, const Span& y) { return !overlap(x.addr, x.bytes, y.addr, y.bytes); }

// Rsqrt argument of the l2 norm: sum/S + eps/S.
inline float l2_rsqrt_arg(float sum, float eps) { return sum / (float)kS + eps / (float)kS; }

}  // namespace native_gdn_pre
}  // namespace baloon
