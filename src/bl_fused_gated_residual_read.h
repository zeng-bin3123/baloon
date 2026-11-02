// bl_fused_gated_residual_read.h - index geometry of the fused hyper-connection read kernels for the fixed
// geometry n_embd = 2560, hc = 4, hc_lr = 320.
#pragma once
#include "bl_nat.h"

namespace baloon::fused_gr {

constexpr nat N = 2560, HC = 4, D = N * HC, LR = 320, THREADS = 256, WARPS = THREADS / 32;
constexpr nat DOWN_BLOCKS = LR / WARPS, UP_COLS = 32, UP_BLOCKS = N / UP_COLS;
constexpr nat TILE = 2560, TQ = TILE / 8 / 32, UPM_COLS = 16, UPM_BLOCKS = N / UPM_COLS;
static_assert(D == 10240 && WARPS == 8 && DOWN_BLOCKS * WARPS == LR && UP_BLOCKS * UP_COLS == N, "geometry");
static_assert(UPM_BLOCKS * UPM_COLS == N && TQ == 10 && 32 * TQ * 8 == TILE && 4 * TILE == D && LR / 8 == 32 + 8, "tiles");

constexpr nat downRow(nat b, nat w) { return b * 8 + w; }
constexpr nat downChunk(nat lane, nat j) { return lane + 32 * j; }               // < 1280
constexpr nat tiledChunk(nat base, nat lane, nat q) { return base / 8 + lane + 32 * q; }
constexpr nat streamOf(nat flat) { return flat / N; }
constexpr nat colOf(nat flat) { return flat % N; }
constexpr nat tileCopyToken(nat i) { return i / 640; }
constexpr nat tileCopyOffset(nat i) { return i % 640; }
constexpr nat upIndex(nat blk, nat r) { return (r / 32) * N + blk * 32 + r % 32; }
constexpr nat upmIndex(nat blk, nat r) { return (r / 16) * N + blk * 16 + r % 16; }
constexpr nat mixedIndex(nat blk, nat t) { return blk * 32 + t; }
constexpr nat secondChunk(nat lane) { return 32 + lane; }  // exists only for lane < 8

constexpr bool grSupported(long n_embd, long hc, long hc_lr) { return n_embd == 2560 && hc == 4 && hc_lr == 320; }
constexpr bool tokenLaneOwned(nat T, nat maxT) { return T <= maxT && maxT <= 32; }

}  // namespace baloon::fused_gr
