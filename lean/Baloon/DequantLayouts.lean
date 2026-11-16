-- Baloon.DequantLayouts: block geometry table, Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/Q5_0
-- struct field offsets and load-alignment proofs for the native MMVQ kernels.
-- Independent implementation proving the same mathematical facts as Ref.DequantLayouts.

namespace Baloon.DequantLayouts

def geom : Nat → Option (Nat × Nat)
  | 0  => some (1, 4)   | 1  => some (1, 2)   | 30 => some (1, 2)
  | 2  => some (32, 18) | 3  => some (32, 20)  | 6  => some (32, 22)
  | 7  => some (32, 24) | 8  => some (32, 34)  | 9  => some (32, 36)
  | 10 => some (256, 84)| 11 => some (256, 110)| 12 => some (256, 144)
  | 13 => some (256, 176)| 14 => some (256, 210)| 16 => some (256, 66)
  | 17 => some (256, 74)| 18 => some (256, 98) | 20 => some (32, 18)
  | 21 => some (256, 110)| 22 => some (256, 82)| 23 => some (256, 136)
  | 29 => some (256, 56)| 42 => some (64, 18)  | _  => none

theorem q2_0_geometry : (geom 42 = some (64, 18)) ∧ (2 + 64 * 2 / 8 = 18) := by { decide }

theorem supported_have_geometry :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29],
    (geom t).isSome = true := by { decide }

theorem elems_divide_superblock :
    ∀ t ∈ [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42],
    (geom t).map (fun p => 256 % p.1) = some 0 := by { decide }

theorem q3k_layout : (32 + 64 + 12 + 2 = 110) ∧ (32 + 64 = 96) ∧ (96 + 12 = 108) := by { decide }
theorem q6k_layout : (128 + 64 + 16 + 2 = 210) ∧ (128 + 64 = 192) ∧ (192 + 16 = 208) := by { decide }
theorem q5k_layout : (4 + 12 + 32 + 128 = 176) ∧ (4 + 12 = 16) ∧ (16 + 32 = 48) := by { decide }
theorem q4k_layout : (4 + 12 + 128 = 144) ∧ (4 + 12 = 16) := by { decide }
theorem iq4xs_layout : (2 + 2 + 4 + 128 = 136) ∧ (2 + 2 = 4) ∧ (4 + 4 = 8) := by { decide }
theorem q50_layout : (2 + 4 + 16 = 22) ∧ (2 + 4 = 6) := by { decide }

theorem stride_alignment :
    ([18, 22, 34, 110, 210].all (fun b => b % 4 == 2)) = true ∧
    ([36, 136, 144, 176].all (fun b => b % 4 == 0)) = true := by { decide }

end Baloon.DequantLayouts
