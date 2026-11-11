/-!
# MODULE: block_weighted_select
SOURCES: src/kernels/cuda/qsa_select.cu
`block_scores_kernel` scores one entry per BLOCK (the spare `dead` key for `b = n_bid`), and
`block_topk_kernel` runs the radix select over blocks with each block weighted by its cell count,
expanding a selected block into its cells `b*R .. b*R + w - 1`.

## Source semantics
* `weight b = R` for completed blocks `b < n_bid`, and `n_kv - n_bid*R` for the tail block `b = n_bid`
  (zero when `n_kv` is block-aligned, which the kernel skips).
* The weights of blocks `0..n_bid` sum to exactly `n_kv`, so weighted counts over blocks equal cell
  counts — the block-level select agrees with the cell-level `topk_512`.
* The expanded cells of block `b` lie in `[b*R, b*R + R)`, are `< n_kv`, and different blocks' cells
  are disjoint.
* `n_kv ≤ width` short-circuits to the identity selection `0..n_kv-1`.
* The scores output index `qi * max_blocks + b` is inside `nq * max_blocks` for `b < max_blocks`.

## Form template identifiers
`R` → `r`, `weight` lambda → `weight`, `nb = n_bid + 1` → blocks, cell id `b*R + c` → `cellOf`,
identity path → `identitySel`, `out[qi*max_blocks+b]` → `scoreSlot`.
-/

namespace Strata.BlockSelect

/-- The cell count of block `b`. -/
def weight (r nKv nBid b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Total weight of blocks `0 .. m-1`. -/
def sumW (r nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => sumW r nKv nBid m + weight r nKv nBid m

/-- Cell `c` of block `b`. -/
def cellOf (r b c : Nat) : Nat := b * r + c

/-- The identity selection used when everything fits. -/
def identitySel (nKv : Nat) : List Nat := List.range nKv

/-- Completed blocks contribute `R` cells each. -/
theorem sumW_completed (r nKv nBid m : Nat) (hm : m ≤ nBid) : sumW r nKv nBid m = m * r := by
  induction m with
  | zero => simp [sumW]
  | succ m ih =>
    have hlt : m < nBid := by omega
    simp only [sumW, weight, hlt, if_true, ih (by omega), Nat.succ_mul]

/-- Blocks `0..n_bid` weigh exactly `n_kv` cells. -/
theorem sumW_total (r nKv : Nat) : sumW r nKv (nKv / r) (nKv / r + 1) = nKv := by
  have h1 := sumW_completed r nKv (nKv / r) (nKv / r) (Nat.le_refl _)
  have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
  simp only [sumW, weight, Nat.lt_irrefl, if_false, h1]
  omega

/-- The tail block weighs fewer than `R` cells. -/
theorem tail_weight_lt (r nKv : Nat) (hr : 0 < r) : weight r nKv (nKv / r) (nKv / r) < r := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r
  have h2 := Nat.mod_lt nKv hr
  rw [Nat.mul_comm] at h1
  omega

/-- A block-aligned cache has an empty tail block (the kernel's `w == 0` skip). -/
theorem tail_weight_zero (r nKv : Nat) (h : nKv % r = 0) : weight r nKv (nKv / r) (nKv / r) = 0 := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at h1
  omega

/-- Expanded cells of a block are real cells. -/
theorem cellOf_lt (r nKv b c : Nat) (hb : b ≤ nKv / r) (hc : c < weight r nKv (nKv / r) b) :
    cellOf r b c < nKv := by
  unfold cellOf
  unfold weight at hc
  by_cases hlt : b < nKv / r
  · simp only [hlt, if_true] at hc
    have h1 : (b + 1) * r ≤ nKv / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
    rw [Nat.succ_mul] at h1
    omega
  · have hbe : b = nKv / r := by omega
    subst hbe
    simp only [Nat.lt_irrefl, if_false] at hc
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
    omega

/-- Cells of different blocks never coincide. -/
theorem cellOf_disjoint (r b b' c c' : Nat) (hc : c < r) (hc' : c' < r)
    (h : cellOf r b c = cellOf r b' c') : b = b' ∧ c = c' := by
  unfold cellOf at h
  rcases Nat.lt_trichotomy b b' with hlt | heq | hgt
  · have h1 : (b + 1) * r ≤ b' * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq; omega
  · have h1 : (b' + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

/-- The identity path emits exactly `n_kv` ascending ids. -/
theorem identitySel_spec (nKv : Nat) :
    (identitySel nKv).length = nKv ∧ ∀ i, i < nKv → (identitySel nKv)[i]? = some i := by
  refine ⟨List.length_range, fun i hi => ?_⟩
  simp [identitySel, hi]

/-- Score slots of every query stay inside the `nq * max_blocks` scores buffer. -/
theorem scoreSlot_lt (qi nq maxB b : Nat) (hq : qi < nq) (hb : b < maxB) : qi * maxB + b < nq * maxB := by
  have h1 : (qi + 1) * maxB ≤ nq * maxB := Nat.mul_le_mul_right maxB (Nat.succ_le_of_lt hq)
  rw [Nat.succ_mul] at h1
  omega

end Strata.BlockSelect
