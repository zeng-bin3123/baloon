/-!
# MODULE: native_qsa_score
Tiling arithmetic of the tensor-core indexer scoring kernel (128 dims, 4 heads, 4 cells per block).

## Source semantics
Two warps each accumulate 32 of every 64 K columns: column = warp*32 + lane + 64*iter covers 0..127. The accumulator fragment element l of lane maps to row warp*32 + ia*16 + (l/2)*8 + lane/4 and head (lane%4)*2 + l%2 inside a 68-stride combine buffer. Block rows score their cells: cell i belongs to row i/4; the incomplete tail row gets +1e9.

## Form template identifiers
score_kernel -> block_score, COMBINE -> combine_stride, ROWS -> rows_per_block
-/
namespace Strata.NativeQsaScore

-- § K coverage

/-- Every K column is visited by exactly one (warp, lane, iteration) in range. -/
theorem col_surj (c : Nat) (h : c < 128) : ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ w * 32 + l + it * 64 = c := by refine ⟨(c % 64) / 32, c % 32, c / 64, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The column map is injective. -/
theorem col_inj (w l it w' l' it' : Nat) (hw : w < 2) (hl : l < 32) (hi : it < 2) (hw' : w' < 2) (hl' : l' < 32) (hi' : it' < 2) (h : w * 32 + l + it * 64 = w' * 32 + l' + it' * 64) : w = w' ∧ l = l' ∧ it = it' := by omega

-- § fragment mapping

/-- Fragment rows are below 64 (two warps x 32 rows). -/
theorem frag_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) : w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Fragment heads are below 8. -/
theorem frag_head_lt (l lane : Nat) (hlane : lane < 32) (hl : l < 4) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Combine-buffer addresses stay inside 8 heads x 68. -/
theorem combine_lt (h i : Nat) (hh : h < 8) (hi : i < 68) : h * 68 + i < 8 * 68 := by omega

/-- Distinct (head, row) pairs map to distinct buffer slots. -/
theorem combine_inj (h i h' i' : Nat) (hi : i < 68) (hi' : i' < 68) (e : h * 68 + i = h' * 68 + i') : h = h' ∧ i = i' := by omega

-- § cells

/-- A cell below n belongs to a row no greater than n/4. -/
theorem cell_row_le (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by omega

/-- Rows partition cells in groups of 4. -/
theorem cell_row_range (i : Nat) : (i / 4) * 4 ≤ i ∧ i < (i / 4 + 1) * 4 := by omega

/-- The tail bonus applies exactly when the cache length is not a multiple of 4. -/
theorem tail_bonus_iff (n : Nat) : n % 4 ≠ 0 ↔ ¬ (n = n / 4 * 4) := by omega

/-- The block grid covers max_blocks rows at 32 rows per block. -/
theorem score_grid (mb : Nat) : mb ≤ (mb + 31) / 32 * 32 := by omega

/-- The block count matches the cell capacity. -/
theorem blocks_for_cells (mc : Nat) : mc / 4 + 1 ≥ (mc + 3) / 4 := by omega

end Strata.NativeQsaScore
