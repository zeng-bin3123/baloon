/-!
# MODULE: qsa_index_cell_map
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/native_qsa_score.cu
`qsa_index_kernel` and the native tensor-core `score_kernel`: score each pooled row as the sum over
indexer heads of `relu(dot)`, add the optional bias, add `+1e9` to the incomplete tail block, and copy
the row's score onto its cells.

## Source semantics
* Block `b` owns cells `[b*r, min(b*r + r, n_kv))`, and the tail block `b = n_bid` owns
  `[n_bid*r, n_kv)`. Every cell `j < n_kv` is owned by exactly block `j / r ≤ n_bid`.
* The `+1e9` tail bias applies iff `b = n_bid ∧ n_kv % r ≠ 0`; a block-aligned cache's tail block owns
  no cells, so the spare row never scores a real cell.
* Blocks `b > n_bid` of a constant capacity grid do nothing; native capacity `max/4 + 1` blocks
  always cover `n_bid + 1` blocks.
* The ReLU is per head, so the head sum is non-negative.
* The native kernel accepts a step only if it is exactly `qsa_step_fill` with `r = 4`, `top_k = 2048`.

## Form template identifiers
`lo`/`hi` → `cellLo`/`cellHi`, `n_bid` → `nBid`, `+1e9` condition → `tailBiased`,
`score += relu(s_dot[h])` → `reluSum`, native `step` check → `nativeStepOk`, `max_blocks` → `maxBlocks`.
-/

namespace Strata.IndexMap

/-- First cell of block `b`. -/
def cellLo (r b : Nat) : Nat := b * r

/-- One past the last cell of block `b`. -/
def cellHi (r nKv nBid b : Nat) : Nat := if b = nBid then nKv else min (b * r + r) nKv

/-- The kernel adds the tail bias to block `b`. -/
def tailBiased (r nKv nBid b : Nat) : Prop := b = nBid ∧ nKv % r ≠ 0

/-- Every cached cell is inside the range of block `j / r`. -/
theorem cell_owned (r nKv j : Nat) (hr : 0 < r) (hj : j < nKv) :
    j / r ≤ nKv / r ∧ cellLo r (j / r) ≤ j ∧ j < cellHi r nKv (nKv / r) (j / r) := by
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  refine ⟨Nat.div_le_div_right (Nat.le_of_lt hj), ?_, ?_⟩
  · unfold cellLo; omega
  · unfold cellHi
    split
    · exact hj
    · exact Nat.lt_min.mpr ⟨by omega, hj⟩

/-- A cell is inside at most one block's range: ranges are disjoint. -/
theorem cell_owner_unique (r nKv nBid b j : Nat) (hr : 0 < r) (hb : b < nBid)
    (hlo : cellLo r b ≤ j) (hhi : j < cellHi r nKv nBid b) : b = j / r := by
  unfold cellLo at hlo
  unfold cellHi at hhi
  have hne : b ≠ nBid := by omega
  simp only [hne, if_false] at hhi
  have hhi' : j < b * r + r := Nat.lt_of_lt_of_le hhi (Nat.min_le_left _ _)
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  rcases Nat.lt_trichotomy b (j / r) with h | h | h
  · have h1 : (b + 1) * r ≤ j / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1
    omega
  · exact h
  · have h1 : (j / r + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1
    omega

/-- A block-aligned cache's tail (spare) block owns no cells. -/
theorem aligned_tail_empty (r nKv : Nat) (h : nKv % r = 0) :
    cellHi r nKv (nKv / r) (nKv / r) = cellLo r (nKv / r) := by
  unfold cellHi cellLo
  have e1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at e1
  simp only [if_true]
  omega

/-- The tail bias is applied exactly when the tail block owns at least one cell. -/
theorem tailBiased_iff_nonempty (r nKv : Nat) :
    tailBiased r nKv (nKv / r) (nKv / r) ↔ cellLo r (nKv / r) < cellHi r nKv (nKv / r) (nKv / r) := by
  unfold tailBiased cellHi cellLo
  have e1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at e1
  simp only [if_true, true_and]
  omega

-- § head sum

/-- `Σ_h relu(dot_h)`, in head order. -/
def reluSum : List Int → Int
  | [] => 0
  | d :: ds => max d 0 + reluSum ds

/-- The per-head ReLU makes the head sum non-negative. -/
theorem reluSum_nonneg (ds : List Int) : 0 ≤ reluSum ds := by
  induction ds with
  | nil => simp [reluSum]
  | cons d ds ih => simp only [reluSum]; omega

/-- Summing BEFORE the ReLU can lose mass: the reference's per-head order matters. -/
theorem relu_before_sum_differs : reluSum [3, -5] ≠ max (3 + -5) 0 := by decide

/-- The relu sum dominates every single head's contribution. -/
theorem reluSum_ge_mem (ds : List Int) (d : Int) (h : d ∈ ds) : d ≤ reluSum ds := by
  induction ds with
  | nil => simp at h
  | cons x xs ih =>
    simp only [reluSum]
    have h0 := reluSum_nonneg xs
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

-- § native step contract and grid

/-- The native score kernel's guard on the step entries `(pos, n_kv, n_bid, width)`. -/
def nativeStepOk (maxCells pos n full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- `qsa_step_fill` with `r = 4`, `top_k = 2048` always passes the native guard. -/
theorem stepFill_nativeOk (maxCells pos : Nat) (h : pos + 1 ≤ maxCells) :
    nativeStepOk maxCells pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (2048 + 4 - 1)) := by
  unfold nativeStepOk
  refine ⟨by omega, h, by omega, rfl, ?_⟩
  split <;> omega

/-- The native capacity `max/4 + 1` blocks covers blocks `0..n_bid` for every admissible step. -/
theorem native_grid_covers (maxCells n : Nat) (h : n ≤ maxCells) : n / 4 < maxCells / 4 + 1 := by
  omega

end Strata.IndexMap
