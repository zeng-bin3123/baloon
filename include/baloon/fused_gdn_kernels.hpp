// fused_gdn_kernels.hpp - index arithmetic of the fused gated-delta-net kernels (conv+L2, alpha/beta, step+norm).
// step kernel: block = head, thread tid = rg*128+col owns rows rg*32..rg*32+31 of column col.
// conv+L2 kernel: one block per 128 channels, history of three taps slides by one.
// ab kernel: one warp per row of the stacked [alpha; beta] projection, 8 rows per block.
#pragma once
#include <cstdint>

namespace baloon {
namespace fused_gdn {

constexpr int kS = 128;              // state width
constexpr int kRowGroups = 4;
constexpr int kRowsPerThread = 32;
constexpr int kStepThreads = 512;
constexpr int kConvChannelsPerBlock = 128;
constexpr int kAbRowsPerBlock = 8;

inline int tid(int rg, int col) { return rg * kS + col; }
inline int tid_rg(int t) { return t / kS; }
inline int tid_col(int t) { return t % kS; }
inline int row_of(int rg, int r) { return rg * kRowsPerThread + r; }
inline int row_group(int row) { return row / kRowsPerThread; }
inline int row_offset(int row) { return row % kRowsPerThread; }

// State address (row, head, col), layout (S, h_v, S), col fastest.
inline int64_t addr(int64_t hv, int64_t row, int64_t head, int64_t col) { return (row * hv + head) * kS + col; }

inline int qk_head(int head, int hk) { return head % hk; }
inline int rg0_warp(int t) { return t / 32; }

// conv+L2 grid
inline int conv_blocks(int channels) { return channels / kConvChannelsPerBlock; }
inline bool conv_grid_exact(int channels) { return channels % kConvChannelsPerBlock == 0; }
inline bool conv_qk_block_ok(int b, int qk, int channels) { return b < qk && qk <= channels / kConvChannelsPerBlock && b < channels / kConvChannelsPerBlock; }

template <class T> struct Hist3 { T a, b, c; };
// [v0,v1,v2] with input x becomes [v1,v2,x].
template <class T> inline Hist3<T> slide3(const Hist3<T>& h, T x) { return Hist3<T>{h.b, h.c, x}; }

// alpha/beta kernel
inline int ab_blocks(int hv) { return (2 * hv + kAbRowsPerBlock - 1) / kAbRowsPerBlock; }
inline int ab_row(int b, int w) { return b * kAbRowsPerBlock + w; }
inline int ab_block_of(int row) { return row / kAbRowsPerBlock; }
inline int ab_warp_of(int row) { return row % kAbRowsPerBlock; }
inline bool ab_is_beta(int row, int hv) { return row >= hv; }
inline int ab_head_index(int row, int hv) { return row >= hv ? row - hv : row; }
inline int ab_chunks(int n_embd) { return n_embd / 8; }   // bf16 chunks of 8 values per uint4

}  // namespace fused_gdn
}  // namespace baloon
