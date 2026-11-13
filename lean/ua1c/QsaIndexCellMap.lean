-- QsaIndexCellMap.lean
-- Formal model of src/kernels/cuda/qsa.cu and native_qsa_score.cu index cell mapping.
namespace Baloon.IndexMap

def cellLo (r b : Nat) : Nat := b * r

def cellHi (r nKv nBid b : Nat) : Nat := if b = nBid then nKv else min (b * r + r) nKv

def tailBiased (r nKv nBid b : Nat) : Prop := b = nBid ∧ nKv % r ≠ 0

theorem cell_owned (r nKv j : Nat) (hr : 0 < r) (hj : j < nKv) :
    j / r ≤ nKv / r ∧ cellLo r (j / r) ≤ j ∧ j < cellHi r nKv (nKv / r) (j / r) := by
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  refine ⟨Nat.div_le_div_right (Nat.le_of_lt hj), ?_, ?_⟩
  · unfold cellLo; omega
  · unfold cellHi; split
    · exact hj
    · exact Nat.lt_min.mpr ⟨by omega, hj⟩

theorem cell_owner_unique (r nKv nBid b j : Nat) (hr : 0 < r) (hb : b < nBid)
    (hlo : cellLo r b ≤ j) (hhi : j < cellHi r nKv nBid b) : b = j / r := by
  unfold cellLo at hlo; unfold cellHi at hhi
  have hne : b ≠ nBid := by omega
  simp only [hne, if_false] at hhi
  have hhi' : j < b * r + r := Nat.lt_of_lt_of_le hhi (Nat.min_le_left _ _)
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  rcases Nat.lt_trichotomy b (j / r) with h | h | h
  · have h1 : (b + 1) * r ≤ j / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1; omega
  · exact h
  · have h1 : (j / r + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1; omega

theorem aligned_tail_empty (r nKv : Nat) (h : nKv % r = 0) :
    cellHi r nKv (nKv / r) (nKv / r) = cellLo r (nKv / r) := by
  unfold cellHi cellLo
  have e1 := Nat.div_add_mod nKv r; rw [Nat.mul_comm] at e1
  simp only [if_true]; omega

theorem tailBiased_iff_nonempty (r nKv : Nat) :
    tailBiased r nKv (nKv / r) (nKv / r) ↔ cellLo r (nKv / r) < cellHi r nKv (nKv / r) (nKv / r) := by
  unfold tailBiased cellHi cellLo
  have e1 := Nat.div_add_mod nKv r; rw [Nat.mul_comm] at e1
  simp only [if_true, true_and]; omega

def reluSum : List Int → Int
  | [] => 0
  | d :: ds => max d 0 + reluSum ds

theorem reluSum_nonneg (ds : List Int) : 0 ≤ reluSum ds := by
  induction ds with
  | nil => simp [reluSum]
  | cons d ds ih => simp only [reluSum]; omega

theorem relu_before_sum_differs : reluSum [3, -5] ≠ max (3 + -5) 0 := by decide

theorem reluSum_ge_mem (ds : List Int) (d : Int) (h : d ∈ ds) : d ≤ reluSum ds := by
  induction ds with
  | nil => simp at h
  | cons x xs ih =>
    simp only [reluSum]
    have h0 := reluSum_nonneg xs
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

def nativeStepOk (maxCells pos n full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

theorem stepFill_nativeOk (maxCells pos : Nat) (h : pos + 1 ≤ maxCells) :
    nativeStepOk maxCells pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (2048 + 4 - 1)) := by
  unfold nativeStepOk; refine ⟨by omega, h, by omega, rfl, ?_⟩; split <;> omega

theorem native_grid_covers (maxCells n : Nat) (h : n ≤ maxCells) : n / 4 < maxCells / 4 + 1 := by
  omega

end Baloon.IndexMap
