/-!
# MODULE: native_qsa_score
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (TF32 mma) block scoring for the QSA indexer, adapted from llama.cpp's mmf path: 32 pooled rows per
block, two warps splitting the 128-wide reduction, a combine buffer, and a reference-ordered epilogue
(relu per head, ordered head sum, optional bias, +1e9 on an incomplete tail) written onto the block's cells.

## Source semantics
* Fixed geometry: D = 128, 4 heads, R = 4 cells per block, top_k = 2048, so the width cap is 2051.
* A step is valid only if `pos = n-1`, `n_bid = n/R`, `width = min(n, 2051)`.
* Warp `w`, lane `l` accumulates columns `w*32 + l` and `w*32 + l + 64`.
* `max_blocks = max_cells / R + 1`; blocks with `row0 > n_bid` return.

## Form template identifiers
* `D` -> index_dim, `ROWS` -> rows_per_block, `STRIDE` -> smem_row_stride, `COMBINE` -> combine_stride.
-/

namespace Strata.NativeQsaScore

/-- Indexer dimension. -/
def D : Nat := 128
/-- Rows per block. -/
def ROWS : Nat := 32
/-- Shared row stride (floats). -/
def STRIDE : Nat := 36
/-- Combine buffer stride. -/
def COMBINE : Nat := 68

/-- Two warps of 32 lanes with two passes of 64 columns cover the 128 columns. -/
theorem col_cover (c : Nat) (h : c < 128) :
    ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ c = w * 32 + l + 64 * it := by
  refine ⟨(c % 64) / 32, c % 32, c / 64, ?_, ?_, ?_, ?_⟩ <;> omega

/-- Column ownership is unique, so partial sums are disjoint. -/
theorem col_unique (w l it w' l' it' : Nat) (hw : w < 2) (hw' : w' < 2) (hl : l < 32) (hl' : l' < 32)
    (h : w * 32 + l + 64 * it = w' * 32 + l' + 64 * it') : w = w' ∧ l = l' ∧ it = it' := by omega

/-- The shared tile is `2 * 16 * 36 = 1152` floats. -/
theorem shared_size : 2 * 16 * STRIDE = 1152 := by decide

/-- Staging writes `tile[i*36 + lane]` stay inside the warp's 16x36 tile. -/
theorem stage_in_tile (i lane : Nat) (hi : i < 16) (hl : lane < 32) : i * 36 + lane < 16 * 36 := by omega

/-- A warp's tile offset plus an in-tile index stays inside the shared array. -/
theorem warp_tile_in_shared (w idx : Nat) (hw : w < 2) (hi : idx < 16 * 36) :
    w * (16 * 36) + idx < 2 * 16 * 36 := by omega

/-- `ldmatrix` A source: row `t%16`, column offset `(t/16)*4 + k*8` with 4 consecutive floats stays in-tile. -/
theorem load_a_in_tile (t k : Nat) (ht : t < 32) (hk : k < 4) :
    (t % 16) * 36 + (t / 16) * 4 + k * 8 + 3 < 16 * 36 := by omega

/-- Combine buffer writes: row index `i = w*32 + ia*16 + (l/2)*8 + lane/4` is below 64. -/
theorem combine_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Combine head index `(lane%4)*2 + l%2` is below 8. -/
theorem combine_head_lt (l lane : Nat) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Combine address `h*68 + i` stays inside the shared array (`h < 8`, `i < 64`). -/
theorem combine_addr_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 1152 := by omega

/-- The second-warp partial is read at `h*68 + 32 + lane`, still inside the array. -/
theorem combine_second_lt (h lane : Nat) (hh : h < 8) (hl : lane < 32) : h * 68 + 32 + lane < 1152 := by omega

/-- Block count `ceil(max_blocks/32)` covers every block row. -/
theorem grid_covers (max_blocks : Nat) : max_blocks ≤ (max_blocks + 31) / 32 * 32 := by omega

/-- A row `<= full` lies in a launched block when `full < max_blocks`. -/
theorem row_in_grid (row full max_blocks : Nat) (h : row ≤ full) (hf : full < max_blocks) :
    row / 32 < (max_blocks + 31) / 32 := by omega

/-- With `max_blocks = max_cells/4 + 1` and `n <= max_cells`, the last block `n/4` is a valid row. -/
theorem last_block_valid (n max_cells : Nat) (h : n ≤ max_cells) : n / 4 < max_cells / 4 + 1 := by omega

/-- The validity width equals `min(n, 2051)`. -/
def widthOk (n width : Nat) : Prop := width = if n < 2051 then n else 2051

/-- A valid step width never exceeds the cap or `n`. -/
theorem widthOk_bounds (n width : Nat) (h : widthOk n width) : width ≤ n ∧ width ≤ 2051 := by
  unfold widthOk at h; split at h <;> omega

/-- The 2051 cap is `top_k + R - 1` for `top_k = 2048`. -/
theorem width_cap : 2048 + 4 - 1 = 2051 := by decide

/-- The relu stage is nonnegative. -/
theorem relu_nonneg (v : Int) : 0 ≤ max v 0 := by omega

/-- The relu never lowers a value. -/
theorem relu_ge (v : Int) : v ≤ max v 0 := by omega

/-- Cells written by row `r`: `[4r, min(n, 4r+4))`, which are disjoint across rows. -/
theorem row_cells_disjoint (r r' c c' : Nat) (hc : c < 4) (hc' : c' < 4)
    (h : r * 4 + c = r' * 4 + c') : r = r' ∧ c = c' := by omega

/-- Capacity relation used by validation: `max_blocks = max_cells/4 + 1` leaves room for the spare block. -/
theorem spare_room (max_cells : Nat) : max_cells / 4 + 1 > max_cells / 4 := by omega

end Strata.NativeQsaScore
