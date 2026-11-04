/-!
# MODULE: native_qsa_score_tiles
Tensor-core scoring of pooled indexer rows against 4 query heads: per-head relu, head-ordered sum, optional bias,
`+1e9` for the incomplete tail block, and fan-out of each row score to its 4 cells.

## Source semantics
* 128-dim keys, 4 heads, 4 cells per row; 32 rows per CUDA block, 2 warps.
* Each warp accumulates K columns `warp*32 + lane` and `warp*32 + lane + 64`.
* Accumulator fragment element `l` of lane `lane` of warp `warp` lands in transposed shared slot
  `h*COMBINE + i` with `i = warp*32 + ia*16 + (l/2)*8 + lane/4`, `h = (lane%4)*2 + l%2`.
* Per-head relu is applied before the head sum; cells `[4*row, min(4*row+4, n))` receive the row score.
* The kernel is active only when the step state is consistent: `pos = n-1`, `full = n/4`, `width = min(n, 2051)`.

## Form template identifiers
* `D`, `HEADS`, `R`, `ROWS`, `WARPS`, `STRIDE`, `COMBINE` -> geometry constants
* `fragRow`, `fragHead`      -> accumulator fragment placement
* `relu`                     -> per-head relu
-/

namespace Strata.NativeQsaScore

/-- Key dimension. -/
def D : Nat := 128
/-- Number of indexer heads. -/
def HEADS : Nat := 4
/-- Cells per row. -/
def R : Nat := 4
/-- Rows per block. -/
def ROWS : Nat := 32
/-- Warps per block. -/
def WARPS : Nat := 2
/-- Shared tile row stride. -/
def STRIDE : Nat := 36
/-- Combine buffer stride. -/
def COMBINE : Nat := 68

/-- Row index of accumulator element `l` (fragment placement). -/
def fragRow (warp ia l lane : Nat) : Nat := warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4
/-- Head index of accumulator element `l`. -/
def fragHead (l lane : Nat) : Nat := (lane % 4) * 2 + l % 2

/-- Tile writes `i*STRIDE + lane` for `i<16, lane<32` stay inside one warp's `16*STRIDE` tile. -/
theorem tile_write_in_range (i lane : Nat) (hi : i < 16) (hl : lane < 32) :
    i * STRIDE + lane < 16 * STRIDE := by
  unfold STRIDE
  omega

/-- Fragment rows are below `WARPS*ROWS = 64`. -/
theorem frag_row_lt (warp ia l lane : Nat) (hw : warp < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    fragRow warp ia l lane < 64 := by
  unfold fragRow
  omega

/-- Fragment heads are below 8 (heads 4..7 are zero padding). -/
theorem frag_head_lt (l lane : Nat) : fragHead l lane < 8 := by
  unfold fragHead
  omega

/-- Combine writes `h*COMBINE + i` fit the shared array of `WARPS*16*STRIDE` floats. -/
theorem combine_in_range (warp ia l lane : Nat) (hw : warp < 2) (hia : ia < 2) (hl : l < 4)
    (hlane : lane < 32) :
    fragHead l lane * COMBINE + fragRow warp ia l lane < WARPS * 16 * STRIDE := by
  unfold fragHead fragRow COMBINE WARPS STRIDE
  omega

/-- Fragment placement is injective: distinct (warp, ia, l, lane) never write the same (row, head) slot. -/
theorem frag_injective (w1 i1 l1 n1 w2 i2 l2 n2 : Nat)
    (hw1 : w1 < 2) (hi1 : i1 < 2) (hl1 : l1 < 4) (hn1 : n1 < 32)
    (hw2 : w2 < 2) (hi2 : i2 < 2) (hl2 : l2 < 4) (hn2 : n2 < 32)
    (hr : fragRow w1 i1 l1 n1 = fragRow w2 i2 l2 n2) (hh : fragHead l1 n1 = fragHead l2 n2) :
    w1 = w2 ∧ i1 = i2 ∧ l1 = l2 ∧ n1 = n2 := by
  unfold fragRow fragHead at *
  omega

/-- K-column coverage: every column `col < 128` is handled by exactly one (warp, lane, iteration). -/
theorem k_column_coverage (col : Nat) (hc : col < D) :
    (col / 32) % 2 < 2 ∧ col % 32 < 32 ∧ (col / 32) / 2 < 2 ∧
    col = ((col / 32) % 2) * 32 + col % 32 + 64 * ((col / 32) / 2) := by
  unfold D at hc
  omega

/-- Per-head relu is nonnegative, hence the head sum is monotone in each head. -/
theorem relu_nonneg (v : Int) : 0 ≤ max v 0 := by
  omega

/-- Relu is idempotent. -/
theorem relu_idempotent (v : Int) : max (max v 0) 0 = max v 0 := by
  omega

/-- Cell fan-out: cell `i` is written by exactly row `i/4`. -/
theorem cell_row_iff (i row : Nat) : (row * R ≤ i ∧ i < (row + 1) * R) ↔ i / R = row := by
  unfold R
  omega

/-- Cells written for a row are bounded by both the row end and the cell count. -/
theorem cell_range_bound (i row n : Nat) (h1 : row * R ≤ i) (h2 : i < n) (h3 : i < (row + 1) * R) :
    i < n ∧ i / R = row := by
  refine ⟨h2, ?_⟩
  exact (cell_row_iff i row).mp ⟨h1, h3⟩

/-- The grid of `(max_blocks+31)/32` blocks covers every row. -/
theorem grid_covers_rows (max_blocks : Nat) : max_blocks ≤ ((max_blocks + ROWS - 1) / ROWS) * ROWS := by
  unfold ROWS
  omega

/-- With `n ≤ max_cells` and `max_blocks = max_cells/4 + 1`, the tail row `n/4` is a valid row. -/
theorem tail_row_valid (n max_cells : Nat) (h : n ≤ max_cells) : n / R < max_cells / R + 1 := by
  unfold R
  omega

/-- The incomplete-tail row holds exactly `n % 4` cells. -/
theorem tail_row_cells (n : Nat) : n - (n / R) * R = n % R := by
  unfold R
  omega

/-- `+1e9` applies iff the tail is nonempty, i.e. the last row is partial. -/
theorem tail_partial_iff (n : Nat) : n % R ≠ 0 ↔ (n / R) * R < n := by
  unfold R
  omega

/-- Selection width of the live graph: `min n 2051`, never larger than `n`. -/
theorem width_le (n : Nat) : min n 2051 ≤ n := by
  omega

end Strata.NativeQsaScore
