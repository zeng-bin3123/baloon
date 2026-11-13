-- BlockWeightedSelect.lean
-- Formal model of src/kernels/cuda/qsa_select.cu block-weighted top-k selection.
namespace Baloon.BlockSelect

def weight (r nKv nBid b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

def sumW (r nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => sumW r nKv nBid m + weight r nKv nBid m

def cellOf (r b c : Nat) : Nat := b * r + c

def identitySel (nKv : Nat) : List Nat := List.range nKv

theorem sumW_completed (r nKv nBid m : Nat) (hm : m ≤ nBid) : sumW r nKv nBid m = m * r := by
  induction m with
  | zero => simp [sumW]
  | succ m ih =>
    have hlt : m < nBid := by omega
    simp only [sumW, weight, hlt, if_true, ih (by omega), Nat.succ_mul]

theorem sumW_total (r nKv : Nat) : sumW r nKv (nKv / r) (nKv / r + 1) = nKv := by
  have h1 := sumW_completed r nKv (nKv / r) (nKv / r) (Nat.le_refl _)
  have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
  simp only [sumW, weight, Nat.lt_irrefl, if_false, h1]; omega

theorem tail_weight_lt (r nKv : Nat) (hr : 0 < r) : weight r nKv (nKv / r) (nKv / r) < r := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r
  have h2 := Nat.mod_lt nKv hr
  rw [Nat.mul_comm] at h1; omega

theorem tail_weight_zero (r nKv : Nat) (h : nKv % r = 0) : weight r nKv (nKv / r) (nKv / r) = 0 := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r; rw [Nat.mul_comm] at h1; omega

theorem cellOf_lt (r nKv b c : Nat) (hb : b ≤ nKv / r) (hc : c < weight r nKv (nKv / r) b) :
    cellOf r b c < nKv := by
  unfold cellOf; unfold weight at hc
  by_cases hlt : b < nKv / r
  · simp only [hlt, if_true] at hc
    have h1 : (b + 1) * r ≤ nKv / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
    rw [Nat.succ_mul] at h1; omega
  · have hbe : b = nKv / r := by omega; subst hbe
    simp only [Nat.lt_irrefl, if_false] at hc
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r; omega

theorem cellOf_disjoint (r b b' c c' : Nat) (hc : c < r) (hc' : c' < r)
    (h : cellOf r b c = cellOf r b' c') : b = b' ∧ c = c' := by
  unfold cellOf at h
  rcases Nat.lt_trichotomy b b' with hlt | heq | hgt
  · have h1 : (b + 1) * r ≤ b' * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1; omega
  · subst heq; omega
  · have h1 : (b' + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1; omega

theorem identitySel_spec (nKv : Nat) :
    (identitySel nKv).length = nKv ∧ ∀ i, i < nKv → (identitySel nKv)[i]? = some i := by
  refine ⟨List.length_range, fun i hi => ?_⟩; simp [identitySel, hi]

theorem scoreSlot_lt (qi nq maxB b : Nat) (hq : qi < nq) (hb : b < maxB) : qi * maxB + b < nq * maxB := by
  have h1 : (qi + 1) * maxB ≤ nq * maxB := Nat.mul_le_mul_right maxB (Nat.succ_le_of_lt hq)
  rw [Nat.succ_mul] at h1; omega

end Baloon.BlockSelect
