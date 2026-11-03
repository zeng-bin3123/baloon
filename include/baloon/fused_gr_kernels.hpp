// fused_gr_kernels.hpp - launch geometry and index decomposition of the fused gated-residual (hyper-connection)
// read kernels at the fixed geometry n_embd=2560, hc=4, hc_lr=320. Streams are stream-major: i = c*N + d.
#pragma once
#include <cstdint>

namespace baloon {
namespace fused_gr {

constexpr int kN = 2560;
constexpr int kHc = 4;
constexpr int kD = kN * kHc;        // 10240
constexpr int kLr = 320;
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kDownBlocks = kLr / kWarps;            // 40 (+1 inject block)
constexpr int kUpBlocks = 80, kUpCols = 32;
constexpr int kUpMultiBlocks = 160, kUpMultiCols = 16;
constexpr int kTile = kN;                            // floats per tile
constexpr int kChunk = 8;                            // bf16 values per uint4
constexpr int kTileChunks = kTile / kChunk;          // 320
constexpr int kChunksPerLane = kTileChunks / 32;     // 10
constexpr int kRowChunks = kD / kChunk;              // 1280
constexpr int kLrChunks = kLr / kChunk;              // 40

inline int stream_of(int i) { return i / kN; }
inline int col_of(int i) { return i % kN; }
inline int flat(int c, int d) { return c * kN + d; }
inline bool vec4_same_stream(int t) { return (4 * t) / kN == (4 * t + 3) / kN; }
inline int down_row(int b, int w) { return b * kWarps + w; }
inline int tile_chunk(int lane, int q) { return lane + 32 * q; }
inline int up_weight_row(int c, int blk, int dd) { return c * kN + blk * kUpCols + dd; }
inline int up_row_stream(int r) { return r / 32; }
inline int up_row_col(int r) { return r % 32; }
inline int warp_of_row(int r) { return r % kWarps; }
inline int warp_step_of_row(int r) { return r / kWarps; }
inline bool second_pass_lane(int lane) { return lane < kLrChunks - 32; }
inline bool token_lane_ok(int k, int T) { return k < T && T <= 32; }
inline int64_t tile_bytes(int T) { return (int64_t)T * kTile * 4; }

}  // namespace fused_gr
}  // namespace baloon
