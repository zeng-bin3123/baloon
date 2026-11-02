/-!
# MODULE: native_gdn_recurrence
Thread mapping and span contracts of the llama.cpp-style gated delta net step with one warp per
(head, column): each lane keeps four state rows in registers and the warp reduces the K^T S and Q^T S
contractions by xor shuffle.

## Source semantics
- Grid is (h_v, 1, S/4), block (32, 4): column `col = blockIdx.z*4 + threadIdx.y`, lane `l` owns rows
  `r*32 + l` for r in 0..3; row i of column (head, col) is at `(i*h_v + head)*S + col`.
- q/k head is `head % h_k`; the output is scaled by `1/sqrt(S)`.
- Required: stream given, S = 128, h_v in (0, 65535], h_k > 0, h_v % h_k = 0; state/output/inputs are aligned,
  state and output disjoint, and every input span is disjoint from state and output.

## Form template identifiers
- S -> state_dim, h_k -> n_qk_heads, h_v -> n_v_heads, g_val -> decay, delta_col -> delta
- native_gdn_step -> warp_delta_step, enabled -> native_gdn_flag
-/

namespace Strata.NativeGdn

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § column and row mapping

/-- Column of (z, y) with z < 32, y < 4 is below 128. -/
theorem column_lt (z y : Nat) (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by
  omega

/-- Distinct (z, y) give distinct columns. -/
theorem column_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4) (h : z * 4 + y = z' * 4 + y') :
    z = z' ∧ y = y' := by
  omega

/-- Every column is produced by some (z, y). -/
theorem column_cover (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ c = z * 4 + y :=
  ⟨c / 4, c % 4, by omega, by omega, by omega⟩

/-- Rows owned by lane l: r*32 + l for r < 4, all below 128. -/
theorem lane_row_lt (r l : Nat) (hr : r < 4) (hl : l < 32) : r * 32 + l < 128 := by
  omega

/-- The four register rows of 32 lanes tile all 128 rows exactly once. -/
theorem lane_row_inj (r l r' l' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by
  omega

/-- Every row is owned by some (register, lane). -/
theorem lane_row_cover (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ i = r * 32 + l :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- The warp reduction is over exactly the 32 lanes of one column. -/
theorem warp_is_lane_set : 4 * 32 = 128 := by decide

/-- A block of 32x4 threads has 128 threads. -/
theorem block_threads : 32 * 4 = 128 := by decide

-- § addressing

/-- State element address bound. -/
theorem state_elem_lt (i head col h_v : Nat) (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    (i * h_v + head) * 128 + col < 128 * h_v * 128 :=
  idx_lt (idx_lt hi hh) hc

/-- q/k element address bound for head `head % h_k`. -/
theorem qk_elem_lt (head i h_k : Nat) (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < h_k * 128 :=
  idx_lt (Nat.mod_lt _ hk) hi

/-- v / output address bound. -/
theorem vo_elem_lt (head col h_v : Nat) (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 :=
  idx_lt hh hc

-- § launch contract

/-- Shape validity (the checks before the span checks). -/
def validShape (S h_k h_v : Nat) : Prop := S = 128 ∧ 0 < h_k ∧ 0 < h_v ∧ h_v ≤ 65535 ∧ h_v % h_k = 0

/-- Valid shapes give an in-range q/k head for every v head. -/
theorem valid_shape_head_pair (S h_k h_v head : Nat) (h : validShape S h_k h_v) :
    head % h_k < h_k := by
  obtain ⟨_, hk, _, _, _⟩ := h
  exact Nat.mod_lt _ hk

/-- Valid shapes fit the grid.x limit. -/
theorem valid_shape_grid_x (S h_k h_v : Nat) (h : validShape S h_k h_v) : h_v ≤ 65535 := by
  obtain ⟨_, _, _, h4, _⟩ := h
  exact h4

/-- Byte counts of the state, q/k, v/output and per-head spans for S = 128. -/
def stateBytes (h_v : Nat) : Nat := 128 * 128 * h_v * 4
/-- q/k byte count. -/
def qkBytes (h_k : Nat) : Nat := 128 * h_k * 4
/-- v / output byte count. -/
def voBytes (h_v : Nat) : Nat := 128 * h_v * 4
/-- per-head scalar byte count. -/
def headBytes (h_v : Nat) : Nat := h_v * 4

/-- The state is 128 times larger than the output. -/
theorem state_is_128x_output (h_v : Nat) : stateBytes h_v = 128 * voBytes h_v := by
  unfold stateBytes voBytes
  omega

/-- Spans [a, a+an) and [b, b+bn) overlap iff each starts before the other ends. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Disjoint spans placed back to back do not overlap. -/
theorem adjacent_disjoint (a an bn : Nat) (hn : 0 < an) : ¬ overlaps a an (a + an) bn := by
  unfold overlaps
  omega

/-- The overlap relation is symmetric. -/
theorem overlaps_symm (a an b bn : Nat) : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  omega

end Strata.NativeGdn
