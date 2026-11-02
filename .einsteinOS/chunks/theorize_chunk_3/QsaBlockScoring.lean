/-!
# MODULE: qsa_block_scoring
Tile geometry, shared-memory layout, step validity and cell-export contracts of the QSA score kernel: for each
pooled block row, 4 indexer heads are scored with tensor-core tiles over K = 128, ReLU'd per head, summed,
biased, masked for the partial last block, and replicated to the R = 4 cells of the block.

## Source semantics
- Grid covers `max_blocks` rows in tiles of ROWS = 32; block = 2 warps (64 threads); warp w accumulates
  K columns `w*32+lane` and `w*32+lane+64`.
- Shared tile per warp is 16 rows x STRIDE 36 floats; the combine buffer is indexed `h*COMBINE + i`, with
  COMBINE = 68, i = warp*32 + ia*16 + (l/2)*8 + lane/4 and head `h = (lane%4)*2 + l%2`.
- Step validity: `n >= 1`, `n <= max_cells`, `pos == n-1`, `full == n/4`, `width == min(n, 2051)`; invalid steps
  produce no output. Rows `row > full` are skipped; cell export covers `i in [row*4, min(n, (row+1)*4))`.
- The partial last block (`row == full` and `n % 4 != 0`) is penalised with 1e9.

## Form template identifiers
- ROWS -> rows_per_tile, WARPS -> warps_per_block, STRIDE -> tile_stride, COMBINE -> combine_stride
- pooled -> pooled_blocks, cells -> cell_scores, kStepNKv -> kv_len, kStepNBid -> full_blocks
-/

namespace Strata.QsaScore

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § step validity

/-- The kernel's step validity predicate over integers. -/
def validStep (n maxc pos full width : Int) : Prop :=
  1 ≤ n ∧ n ≤ maxc ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- The kernel's block count: full blocks plus the partial block row. -/
theorem valid_full_le (n maxc pos full width : Int) (h : validStep n maxc pos full width) :
    full * 4 ≤ n := by
  obtain ⟨_, _, _, h4, _⟩ := h
  omega

/-- Valid steps have a positive context length. -/
theorem valid_pos_nonneg (n maxc pos full width : Int) (h : validStep n maxc pos full width) :
    0 ≤ pos := by
  obtain ⟨h1, _, h3, _, _⟩ := h
  omega

-- § grid and rows

/-- Grid covers max_blocks rows with 32 rows per block. -/
theorem grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 31) / 32 * 32 := by
  omega

/-- Row of (block, lane) is block*32 + lane; lane < 32 gives unique rows. -/
theorem row_inj (b l b' l' : Nat) (hl : l < 32) (hl' : l' < 32) (h : b * 32 + l = b' * 32 + l') :
    b = b' ∧ l = l' := by
  omega

/-- With `max_blocks = max_cells/4 + 1`, every row up to `full = n/4` exists when n <= max_cells. -/
theorem full_row_exists (n max_cells : Nat) (h : n ≤ max_cells) : n / 4 < max_cells / 4 + 1 := by
  omega

/-- A row the kernel scores (row <= full) is a valid pooled row. -/
theorem scored_row_lt (row full n max_cells : Nat) (hr : row ≤ full) (hf : full = n / 4)
    (hn : n ≤ max_cells) : row < max_cells / 4 + 1 := by
  omega

-- § K split across warps

/-- Column of (warp, lane, iteration) with 2 warps: warp*32 + lane + 64*iter. -/
theorem k_col_lt (w l it : Nat) (hw : w < 2) (hl : l < 32) (hi : it < 2) : w * 32 + l + 64 * it < 128 := by
  omega

/-- Distinct (warp, lane, iteration) triples read distinct columns. -/
theorem k_col_inj (w l it w' l' it' : Nat) (hw : w < 2) (hl : l < 32) (hw' : w' < 2) (hl' : l' < 32)
    (h : w * 32 + l + 64 * it = w' * 32 + l' + 64 * it') : w = w' ∧ l = l' ∧ it = it' := by
  omega

/-- Every K column of 128 is read exactly once. -/
theorem k_col_cover (c : Nat) (h : c < 128) :
    ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ c = w * 32 + l + 64 * it :=
  ⟨(c % 64) / 32, c % 32, c / 64, by omega, by omega, by omega, by omega⟩

/-- Each warp covers 64 columns in two strides of 32. -/
theorem warp_columns : 2 * 2 * 32 = 128 := by decide

-- § shared memory layout

/-- Shared tile floats for two warps: 2 * 16 * 36. -/
theorem shared_tile_floats : 2 * 16 * 36 = 1152 := by decide

/-- The combine buffer (8 heads x 68) is smaller than the shared tile it reuses. -/
theorem combine_fits_in_tile : 8 * 68 ≤ 2 * 16 * 36 := by decide

/-- Tile row of the A operand: (tid%16)*STRIDE + (tid/16)*4 + 3 stays inside 16 rows of STRIDE 36. -/
theorem load_a_in_tile (tid : Nat) (h : tid < 32) : (tid % 16) * 36 + (tid / 16) * 4 + 3 < 16 * 36 := by
  omega

/-- Tile row of the B operand: (tid%8)*STRIDE + ((tid/8)*4)%8 + 3 stays inside 8 rows. -/
theorem load_b_in_tile (tid : Nat) (h : tid < 32) : (tid % 8) * 36 + ((tid / 8) * 4) % 8 + 3 < 8 * 36 := by
  omega

/-- Tile fill: row i, lane l writes `i*36 + l` with l < 32, below the row stride. -/
theorem fill_in_stride (i l : Nat) (hi : i < 16) (hl : l < 32) : i * 36 + l < 16 * 36 := by
  omega

/-- Combine row index `warp*32 + ia*16 + (l/2)*8 + lane/4` is below 64 (< COMBINE = 68). -/
theorem combine_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by
  omega

/-- Combine head index `(lane%4)*2 + l%2` is below 8. -/
theorem combine_head_lt (lane l : Nat) : (lane % 4) * 2 + l % 2 < 8 := by
  omega

/-- Combine address bound: `h*68 + i < 544` for h < 8, i < 64. -/
theorem combine_addr_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 8 * 68 := by
  omega

/-- The two warp partials of row i sit at `h*68 + i` and `h*68 + 32 + i` for the combining warp-0 lane i < 32. -/
theorem warp_partial_addrs (h lane : Nat) (hh : h < 8) (hl : lane < 32) :
    h * 68 + lane < 8 * 68 ∧ h * 68 + 32 + lane < 8 * 68 := by
  omega

/-- Each (warp, ia, l-pair, lane/4) row of the combine buffer is hit by a unique (ia, l/2, lane/4, warp). -/
theorem combine_row_inj (w ia a lq w' ia' a' lq' : Nat) (ha : a < 2) (hlq : lq < 8) (ha' : a' < 2)
    (hlq' : lq' < 8) (hia : ia < 2) (hia' : ia' < 2)
    (h : w * 32 + ia * 16 + a * 8 + lq = w' * 32 + ia' * 16 + a' * 8 + lq') :
    w = w' ∧ ia = ia' ∧ a = a' ∧ lq = lq' := by
  omega

-- § cell export

/-- Cell index recombines: i / 4 is the row that exports it. -/
theorem cell_row (i : Nat) : 4 * (i / 4) + i % 4 = i := by
  omega

/-- A cell below `n` is exported by a row that is scored (row <= n/4). -/
theorem cell_row_scored (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

/-- The partial block row is the last scored row exactly when n % 4 != 0. -/
theorem partial_row_is_last (n : Nat) (h : n % 4 ≠ 0) : n / 4 * 4 < n := by
  omega

/-- Capacity precondition: `max_blocks = max_cells/4 + 1` is what the launcher requires. -/
def capacitiesOk (max_cells max_blocks : Nat) : Prop := 1 ≤ max_cells ∧ max_blocks = max_cells / 4 + 1

/-- Under the capacity precondition the cells array covers every exported cell. -/
theorem cells_cover_exports (max_cells max_blocks n i : Nat) (h : capacitiesOk max_cells max_blocks)
    (hn : n ≤ max_cells) (hi : i < n) : i < max_cells := by
  omega

end Strata.QsaScore
