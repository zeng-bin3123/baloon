/-!
# MODULE: dequant_block_geometry
Block geometry of the GGML quantization types handled by `dequant_bf16.cu`.

## Source semantics
Each supported type has (block elements, block bytes). Every block holds a whole number of 32-element
groups, and the byte sizes are the sums of the fields the kernel indexes (scales, mins, high bits, codes).
`group32` addresses group `gi` as block `gi / (be/32)`.

## Form template identifiers
geometry -> `geometry(type, be, bb)`; type ids: Q4_0=2, Q5_0=6, Q8_0=8, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14,
IQ4_NL=20, IQ4_XS=23, Q2_0=42.
-/

namespace Strata.DequantBlockGeometry

def geometry : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

-- § field sums

theorem q2_0_bytes : 2 + 64 / 4 = 18 := by decide
theorem q4_0_bytes : 2 + 32 / 2 = 18 := by decide
theorem q5_0_bytes : 2 + 4 + 32 / 2 = 22 := by decide
theorem q8_0_bytes : 2 + 32 = 34 := by decide
theorem q3_k_bytes : 32 + 64 + 12 + 2 = 110 := by decide
theorem q4_k_bytes : 2 + 2 + 12 + 128 = 144 := by decide
theorem q5_k_bytes : 2 + 2 + 12 + 32 + 128 = 176 := by decide
theorem q6_k_bytes : 128 + 64 + 16 + 2 = 210 := by decide
theorem iq4_xs_bytes : 2 + 2 + 4 + 128 = 136 := by decide

-- § whole groups

/-- Every supported block holds a whole number of 32-element groups. -/
theorem blocks_hold_whole_groups (t be bb : Nat) (h : geometry t = some (be, bb)) : be % 32 = 0 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Unsupported types have no geometry. -/
theorem unsupported_none : geometry 0 = none := by decide

-- § group addressing

/-- Q2_0: two 32-element groups per 64-element block, block index `gi / 2`. -/
theorem q2_0_block_of_group (gi : Nat) : gi / 2 * 2 ≤ gi ∧ gi < gi / 2 * 2 + 2 := by
  omega

/-- K-types: eight 32-element groups per 256-element block. -/
theorem k_block_of_group (gi : Nat) : gi / 8 * 8 ≤ gi ∧ gi < gi / 8 * 8 + 8 := by
  omega

/-- Blocks per row cover the row when `cols` is a multiple of the block size. -/
theorem blocks_cover_row (cols be : Nat) (hbe : 0 < be) (h : cols % be = 0) : cols / be * be = cols := by
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)

/-- Groups per row: `cols / 32`, and a 64-multiple row has an even group count. -/
theorem even_groups_for_q2_0 (cols : Nat) (h : cols % 64 = 0) : (cols / 32) % 2 = 0 := by
  omega

end Strata.DequantBlockGeometry
