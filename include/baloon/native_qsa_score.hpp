// native_qsa_score.hpp - tile-index arithmetic and guards of the tensor-core QSA block-scoring kernel.
// A block handles 32 pooled rows; two warps split the K=128 reduction (warp w takes columns w*32+lane and
// w*32+lane+64). Accumulator element l of tile ia lands at row warp*32 + ia*16 + (l/2)*8 + lane/4, head (lane%4)*2 + l%2.
#pragma once
#include <cstdint>

namespace baloon {
namespace native_qsa_score {

constexpr uint32_t kD = 128, kHeads = 4, kPaddedHeads = 8, kR = 4, kRows = 32, kWarps = 2, kStride = 36, kCombine = 68;
constexpr uint32_t kTopK = 2048;
constexpr uint32_t kCombineFloats = kPaddedHeads * kCombine;      // 544
constexpr uint32_t kTileFloats = 2 * 16 * kStride;                // 1152

inline uint32_t col_of(uint32_t w, uint32_t lane, uint32_t pass) { return w * 32 + lane + 64 * pass; }
inline uint32_t col_warp(uint32_t col) { return col / 32 % 2; }
inline uint32_t col_lane(uint32_t col) { return col % 32; }
inline uint32_t col_pass(uint32_t col) { return col / 64; }
inline uint32_t acc_row(uint32_t w, uint32_t ia, uint32_t l, uint32_t lane) { return w * 32 + ia * 16 + (l / 2) * 8 + lane / 4; }
inline uint32_t acc_local_row(uint32_t ia, uint32_t l, uint32_t lane) { return ia * 16 + (l / 2) * 8 + lane / 4; }
inline uint32_t acc_head(uint32_t l, uint32_t lane) { return (lane % 4) * 2 + l % 2; }
inline bool head_is_real(uint32_t head) { return head < kHeads; }
inline uint32_t combine_index(uint32_t head, uint32_t row) { return head * kCombine + row; }
inline uint32_t tile_index(uint32_t i, uint32_t lane) { return i * kStride + lane; }
inline uint32_t grid_blocks(uint32_t max_blocks) { return (max_blocks + kRows - 1) / kRows; }
inline uint32_t pooled_row(uint32_t block, uint32_t lane) { return block * kRows + lane; }

// guards
inline uint32_t width_guard(uint32_t n) { return n < kTopK + kR - 1 ? n : kTopK + kR - 1; }
inline bool step_consistent(uint32_t pos, uint32_t n, uint32_t n_bid, uint32_t width) {
    return n >= 1 && pos == n - 1 && n_bid == n / 4 && width == width_guard(n);
}
inline bool max_blocks_ok(uint32_t max_blocks, uint32_t max_cells) { return max_blocks == max_cells / 4 + 1; }
inline bool tail_bias_applies(uint32_t n) { return n % 4 != 0; }
inline uint32_t row_owner(uint32_t cell) { return cell / 4; }
inline uint32_t cell_begin(uint32_t row) { return 4 * row; }
inline uint32_t cell_end(uint32_t row, uint32_t n) { return n < 4 * (row + 1) ? n : 4 * (row + 1); }

}  // namespace native_qsa_score
}  // namespace baloon
