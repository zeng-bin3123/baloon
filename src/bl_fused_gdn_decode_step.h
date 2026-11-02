// bl_fused_gdn_decode_step.h - index geometry and launch contracts of the fused gated-delta-net decode kernels.
#pragma once
#include "bl_nat.h"

namespace baloon::fused_gdn_decode {

constexpr nat S = 128;   // state dim
constexpr nat RG = 4;    // row groups per block
constexpr nat RPG = 32;  // rows per group
static_assert(RG * RPG == S && S * RG == 512 && S * RG / 32 == 16, "block geometry");

constexpr nat threadRow(nat rg, nat r) { return rg * RPG + r; }
constexpr nat tid(nat rg, nat col) { return rg * S + col; }
constexpr nat warpSlot(nat t) { return t / 32; }
// State layout (S, h_v, S): element (row, head, col) at (row*h_v + head)*S + col.
constexpr nat stateIndex(nat row, nat head, nat col, nat h_v) { return (row * h_v + head) * S + col; }
constexpr nat outIndex(nat head, nat col) { return head * S + col; }
constexpr nat qkHead(nat head, nat h_k) { return nmod(head, h_k); }

// conv + L2 kernel
constexpr nat convChannel(nat b, nat t) { return b * 128 + t; }
constexpr nat convHistIndex(nat c, nat i) { return c * 3 + i; }
constexpr nat convWeightIndex(nat c, nat i) { return c * 4 + i; }
constexpr bool l2Applies(nat c, nat qk_heads) { return c / 128 < qk_heads; }

// alpha/beta kernel: one warp per output row, 8 warps per 256-thread block.
constexpr nat abGridBlocks(nat h_v) { return (2 * h_v + 7) / 8; }
constexpr bool abIsAlpha(nat row, nat h_v) { return row < h_v; }
constexpr nat abLocalRow(nat row, nat h_v) { return row < h_v ? row : row - h_v; }

// Launch contracts.
constexpr bool stepNormArgsValid(nat h_k, nat h_v) { return h_k > 0 && h_v > 0 && h_v % h_k == 0; }
constexpr bool convArgsValid(nat channels, nat qk_heads) { return channels % 128 == 0 && qk_heads <= channels / 128; }

}  // namespace baloon::fused_gdn_decode
