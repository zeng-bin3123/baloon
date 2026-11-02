// bl_iquant_avx512_rows.h - format table, decoder offsets, index/scale/sign rules and token tiling of the
// i-quant (IQ2_XXS/XS, IQ3_XXS/S, IQ2_S) multi-token row dots against Q8_K activations.
#pragma once
#include "bl_nat.h"

namespace baloon::iquant_rows {

enum : nat { IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ3_S = 21, IQ2_S = 22 };

constexpr nat blockBytes(nat ty) {
  return ty == 16 ? 66 : ty == 17 ? 74 : ty == 18 ? 98 : ty == 21 ? 110 : ty == 22 ? 82 : 0;
}
constexpr bool supported(nat ty) { return ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22; }

// Byte offsets touched by the decoder for 64-value chunk j (< 4): [lo, hi) per field.
struct Span { nat lo, hi; };
constexpr Span xxsQs(nat j) { return {2 + 16 * j, 2 + 16 * j + 16}; }
constexpr Span xsScale(nat j) { return {66 + 2 * j, 66 + 2 * j + 2}; }
constexpr Span iq2sSign(nat j) { return {2 + 32 + 8 * j, 2 + 32 + 8 * j + 8}; }
constexpr Span iq2sQs(nat j) { return {2 + 8 * j, 2 + 8 * j + 8}; }
constexpr Span iq2sScale(nat j) { return {74 + 2 * j, 74 + 2 * j + 2}; }
constexpr Span iq3xxsWords(nat j) { return {66 + 8 * j, 66 + 8 * j + 8}; }
constexpr Span iq3sQh(nat j) { return {66 + 2 * j, 66 + 2 * j + 2}; }
constexpr Span iq3sSign(nat j) { return {74 + 8 * j, 74 + 8 * j + 8}; }
constexpr Span iq3sScale(nat j) { return {106 + j, 106 + j + 1}; }

// Index and scale rules.
constexpr nat signIndex(nat a, nat k) { return (a >> (7 * k)) & 127; }          // < 128 (ksigns size)
constexpr nat scaleFromNibble(nat s) { return 2 * s + 1; }                     // odd, <= 31 for s < 16
constexpr nat topNibble(nat a) { return (a & 0xFFFFFFFFull) >> 28; }            // < 16
constexpr nat iq2xsGridIndex(nat v) { return v & 511; }                         // < 512
constexpr nat iq2xsSignIndex(nat v) { return (v & 0xFFFF) >> 9; }               // < 128
constexpr nat iq2sIndex(nat q, nat h) { return q | ((h << 2) & 0x300); }       // < 1024 for q < 256
constexpr nat iq3sIndex(nat q, nat x, nat kk) { return q | (((x >> kk) & 1) << 8); }  // < 512

// Masked negate: involution preserving magnitude.
constexpr long applySign(bool m, long y) { return m ? -y : y; }

// Token tiling: down projection groups of 4; gate/up clamps to 8.
constexpr nat tileSize(nat nt, nat t0) { return nmin(4, sat_sub(nt, t0)); }
constexpr nat guTile(nat nt) { return nt >= 8 ? 8 : nt; }

constexpr nat blocksPerRow(nat n) { return n / 256; }  // QK_K = 256

}  // namespace baloon::iquant_rows
