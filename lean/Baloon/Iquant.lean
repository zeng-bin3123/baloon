-- Baloon.Iquant: i-quant (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) block layouts,
-- in-bounds read proofs, scale lanes and odd-scale generation.
-- Independent implementation proving the same mathematical facts as Ref.Iquant.

namespace Baloon.Iquant

def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

theorem supported_iff_sized : ∀ ty : Nat, ty < 64 → (supported ty = true ↔ 0 < blockBytes ty) := by
  decide

theorem block_byte_decomposition :
    blockBytes 16 = 2 + 64 ∧ blockBytes 17 = 2 + 64 + 8 ∧ blockBytes 22 = 2 + 64 + 8 + 8 ∧
    blockBytes 18 = 2 + 96 ∧ blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by decide

theorem bb16 : blockBytes 16 = 66 := by decide
theorem bb17 : blockBytes 17 = 74 := by decide
theorem bb18 : blockBytes 18 = 98 := by decide
theorem bb21 : blockBytes 21 = 110 := by decide
theorem bb22 : blockBytes 22 = 82 := by decide

theorem iq2xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  rw [bb16]; omega

theorem iq2xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  rw [bb17]; omega

theorem iq2s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 22 ∧
    74 + 2 * j + 1 < blockBytes 22 := by
  rw [bb22]; omega

theorem iq3xxs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ blockBytes 18 := by
  rw [bb18]; omega

theorem iq3s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  rw [bb21]; omega

def scaleLane (l : Nat) : Nat := l / 8
theorem scale_lane_lt : ∀ l : Nat, l < 32 → scaleLane l < 4 := by decide

theorem odd_scale (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

end Baloon.Iquant
