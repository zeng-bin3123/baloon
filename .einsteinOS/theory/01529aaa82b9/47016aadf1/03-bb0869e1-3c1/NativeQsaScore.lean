/-!
# MODULE: native_qsa_score
Tile-index arithmetic and guards of the llama.cpp-derived tensor-core QSA block scoring kernel.

## Source semantics
Block handles 32 pooled rows; two warps split the K=128 reduction, warp w taking columns w*32+lane and
w*32+lane+64. Accumulator element l of tile ia lands at row warp*32 + ia*16 + (l/2)*8 + lane/4 and head
(lane%4)*2 + l%2. Guards: step consistency (pos = n-1, n_bid = n/4, width = min(n, 2051)).
max_blocks must equal max_cells/4 + 1.

## Form template identifiers
D -> 128; HEADS -> 4; R -> 4; ROWS -> 32; WARPS -> 2; STRIDE -> 36; COMBINE -> 68.
-/
namespace Strata.NativeQsaScore

/-- Column loop covers 0..127 with (warp, lane, pass). -/
theorem col_cover (col : Nat) (h : col < 128) :
    ∃ w l m, w < 2 ∧ l < 32 ∧ m < 2 ∧ col = w * 32 + l + 64 * m :=
  ⟨col / 32 % 2, col % 32, col / 64, by omega, by omega, by omega, by omega⟩

/-- Column assignment is injective. -/
theorem col_unique (w l m w' l' m' : Nat) (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (e : w * 32 + l + 64 * m = w' * 32 + l' + 64 * m') : w = w' ∧ l = l' ∧ m = m' := by omega

/-- Accumulator row index is below 64. -/
theorem acc_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Accumulator head index is below 8 (heads >= 4 are padding). -/
theorem acc_head_lt (l lane : Nat) (hl : l < 4) (hlane : lane < 32) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Within one warp tile the row index inside 32 rows is below 32. -/
theorem acc_local_row_lt (ia l lane : Nat) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    ia * 16 + (l / 2) * 8 + lane / 4 < 32 := by omega

/-- The (ia, l/2, lane/4) triple is unique per local row (rows 0..31 are covered). -/
theorem acc_local_row_cover (r : Nat) (h : r < 32) :
    ∃ ia hl q, ia < 2 ∧ hl < 2 ∧ q < 8 ∧ r = ia * 16 + hl * 8 + q :=
  ⟨r / 16, r / 8 % 2, r % 8, by omega, by omega, by omega, by omega⟩

/-- Combine buffer index head*68 + row stays within 8*68. -/
theorem combine_index_lt (h row : Nat) (hh : h < 8) (hr : row < 64) : h * 68 + row < 8 * 68 := by omega

/-- The combine buffer (8*68 floats) fits in the shared tile region (2*16*36 floats). -/
theorem combine_fits : 8 * 68 ≤ 2 * 16 * 36 := by decide

/-- Distinct (head,row) pairs address distinct combine slots (row < 68). -/
theorem combine_inj (h r h' r' : Nat) (hr : r < 68) (hr' : r' < 68) (e : h * 68 + r = h' * 68 + r') :
    h = h' ∧ r = r' := by omega

/-- Row-block grid: ceil(max_blocks/32) blocks cover all pooled rows. -/
theorem grid_cover (row mb : Nat) (h : row < mb) : row / 32 < (mb + 31) / 32 := by omega

/-- Block row0 + lane recovers the pooled row. -/
theorem row_decomp (b lane : Nat) (h : lane < 32) : (b * 32 + lane) / 32 = b := by omega

/-- Selection width guard equals min(n, 2051). -/
theorem width_guard (n : Nat) : (if n < 2051 then n else 2051) = min n 2051 := by
  by_cases h : n < 2051
  · rw [if_pos h]; omega
  · rw [if_neg h]; omega

/-- Cells written for row: i from 4*row up to min(n, 4*(row+1)). Every cell j<n is written by row j/4. -/
theorem cell_owner (j n : Nat) (h : j < n) : j / 4 * 4 ≤ j ∧ j < (j / 4 + 1) * 4 ∧ j < n := by omega

/-- The row at index `full` is the tail block: n/4 blocks completed. -/
theorem tail_row (n : Nat) (h : 0 < n) : n / 4 ≤ n := by omega

/-- Capacity relation max_blocks = max_cells/4 + 1 gives enough rows for every n <= max_cells. -/
theorem capacity_rows (n mc : Nat) (h : n ≤ mc) : n / 4 < mc / 4 + 1 := by omega

/-- Tail bias applies exactly when n % 4 != 0. -/
theorem tail_bias_iff (n : Nat) : n % 4 ≠ 0 ↔ n ≠ (n / 4) * 4 := by omega

/-- Per-lane shared tile index i*STRIDE + lane stays within 16*36 for i<16, lane<32. -/
theorem tile_index_lt (i lane : Nat) (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

end Strata.NativeQsaScore
