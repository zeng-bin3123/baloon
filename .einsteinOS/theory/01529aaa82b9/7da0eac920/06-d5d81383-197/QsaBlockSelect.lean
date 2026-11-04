/-!
# MODULE: qsa_block_select
Block-score top-k selection for quantized sparse attention: per-block cell weights, the 8-bit radix digit scan,
and the equal-key budget clamp.

## Source semantics
Blocks 0..n_bid cover n_kv cells in groups of R; the last block (index n_bid = n_kv / R) may be partial.
The radix select scans a histogram from digit 255 downward and stops at the first digit whose cumulative
count reaches `width`. Ties at the threshold are taken lowest-index first within a per-thread budget.

## Form template identifiers
- `R` ↔ `R` (indexer block size, 4); `nKv` ↔ `n_kv`; `nBid` ↔ `n_bid`; `width` ↔ `width`
- `hist` ↔ `hist[256]`; `scan` ↔ the digit loop in `block_topk_kernel`
-/

namespace Strata.QsaBlockSelect

-- § block weights

/-- Weight (cell count) of block `b`: `R` for full blocks, the remainder for the last one. -/
def blockWeight (R nKv nBid b : Nat) : Nat := if b < nBid then R else nKv - nBid * R

/-- Total weight of blocks `0..n-1`. -/
def totalWeight (R nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => totalWeight R nKv nBid n + blockWeight R nKv nBid n

/-- Blocks strictly below `n_bid` each carry exactly `R` cells. -/
theorem totalWeight_below (R nKv nBid : Nat) :
    ∀ n, n ≤ nBid → totalWeight R nKv nBid n = n * R := by
  intro n
  induction n with
  | zero => intro _; simp [totalWeight]
  | succ k ih =>
    intro h
    have hk : k < nBid := by omega
    have h1 := ih (by omega)
    have h2 : blockWeight R nKv nBid k = R := by simp [blockWeight, hk]
    simp only [totalWeight, h1, h2]
    rw [Nat.succ_mul]

/-- The `n_bid + 1` blocks together cover exactly `n_kv` cells when `n_bid = n_kv / R`. -/
theorem total_covers_kv (R nKv : Nat) (hR : 0 < R) :
    totalWeight R nKv (nKv / R) (nKv / R + 1) = nKv := by
  have h1 := totalWeight_below R nKv (nKv / R) (nKv / R) (Nat.le_refl _)
  have h2 : (nKv / R) * R ≤ nKv := Nat.div_mul_le_self nKv R
  have h3 : blockWeight R nKv (nKv / R) (nKv / R) = nKv - (nKv / R) * R := by
    simp [blockWeight]
  simp only [totalWeight, h1, h3]
  generalize (nKv / R) * R = m at *
  omega

/-- The last block weight is strictly less than `R` (it is the remainder `n_kv % R`). -/
theorem last_weight_lt (R nKv : Nat) (hR : 0 < R) :
    blockWeight R nKv (nKv / R) (nKv / R) < R := by
  have h3 : blockWeight R nKv (nKv / R) (nKv / R) = nKv - (nKv / R) * R := by
    simp [blockWeight]
  have h4 : nKv % R < R := Nat.mod_lt _ hR
  have h5 : nKv = R * (nKv / R) + nKv % R := (Nat.div_add_mod nKv R).symm
  rw [h3]
  rw [Nat.mul_comm] at h5
  generalize (nKv / R) * R = m at *
  omega

-- § radix digit scan

/-- Scan histogram digits from `k` down to 1; stop at the first digit reaching `width`, else digit 0. -/
def scan (hist : Nat → Nat) (width : Nat) : Nat → Nat → Nat × Nat
  | 0, cum => (0, cum)
  | k + 1, cum =>
    if width ≤ cum + hist (k + 1) then (k + 1, cum)
    else scan hist width k (cum + hist (k + 1))

/-- The scan never returns a digit above its starting digit. -/
theorem scan_digit_le (hist : Nat → Nat) (width : Nat) :
    ∀ k cum, (scan hist width k cum).1 ≤ k := by
  intro k
  induction k with
  | zero => intro cum; simp [scan]
  | succ k ih =>
    intro cum
    unfold scan
    split
    · exact Nat.le_refl _
    · have := ih (cum + hist (k + 1)); omega

/-- The cumulative count above the chosen digit only grows. -/
theorem scan_above_ge (hist : Nat → Nat) (width : Nat) :
    ∀ k cum, cum ≤ (scan hist width k cum).2 := by
  intro k
  induction k with
  | zero => intro cum; simp [scan]
  | succ k ih =>
    intro cum
    unfold scan
    split
    · exact Nat.le_refl _
    · have := ih (cum + hist (k + 1)); omega

/-- A nonzero chosen digit reaches the width: `above + hist[d] ≥ width`. -/
theorem scan_reaches (hist : Nat → Nat) (width : Nat) :
    ∀ k cum, 0 < (scan hist width k cum).1 →
      width ≤ (scan hist width k cum).2 + hist (scan hist width k cum).1 := by
  intro k
  induction k with
  | zero => intro cum h; simp [scan] at h
  | succ k ih =>
    intro cum h
    unfold scan at h ⊢
    split at h
    · rename_i hw
      simp only [hw, if_true]
    · rename_i hw
      simp only [hw, if_false]
      exact ih _ h

-- § equal-key budget

/-- Per-thread share of equal-key cells: the budget left after earlier threads, clamped into `[0, eq]`. -/
def myEq (budget before eq : Nat) : Nat := min eq (budget - before)

/-- A thread never selects more equal-key cells than it owns. -/
theorem myEq_le (budget before eq : Nat) : myEq budget before eq ≤ eq := by
  unfold myEq; omega

/-- A thread past the budget selects no equal-key cells. -/
theorem myEq_zero (budget before eq : Nat) (h : budget ≤ before) : myEq budget before eq = 0 := by
  unfold myEq; omega

/-- A thread fully inside the budget selects all of its equal-key cells. -/
theorem myEq_full (budget before eq : Nat) (h : before + eq ≤ budget) : myEq budget before eq = eq := by
  unfold myEq; omega

end Strata.QsaBlockSelect
