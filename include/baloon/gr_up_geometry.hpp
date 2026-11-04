// gr_up_geometry.hpp - geometry of the gated-residual up/mean kernels (hc = 4 streams, n_embd = 2560, hc_lr = 320).
#pragma once
#include <cstdint>
namespace baloon {
namespace gr {
constexpr int kN = 2560, kHc = 4, kD = kN * kHc, kLr = 320, kWarps = 8, kUpCols = 32, kUpRows = kUpCols * kHc;
constexpr int kUpBlocks = kN / kUpCols;            // 80 blocks of 32 columns
constexpr int kDownBlocks = kLr / kWarps;          // 40 blocks of 8 warps, one row per warp
// Fused up kernel: (block, row r in 0..127) addresses stream r/32, column blk*32 + r%32.
constexpr int up_index(int blk, int r) { return (r / kUpCols) * kN + blk * kUpCols + r % kUpCols; }
constexpr int up_block_of(int i) { return (i % kN) / kUpCols; }
constexpr int up_row_of(int i) { return (i / kN) * kUpCols + (i % kN) % kUpCols; }
constexpr uint64_t align16(uint64_t s) { return (s + 15) / 16 * 16; }
constexpr uint64_t workspace_bytes() {
    return align16(kD * 4) + align16(kD * 2) + align16(kLr * 2) + align16(kD * 4) + align16(kLr * 4);
}
}  // namespace gr
}  // namespace baloon
