/-!
# MODULE: native_qsa_score_tiles
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (mma tf32) per-block indexer score kernel, bit-matched to llama.cpp's mmf path, writing
the per-cell scores for the full blocks and the tail.

## Source semantics
* Step is accepted iff 1 <= n <= max_cells, pos = n - 1, n_bid = n / 4, width = min(n, 2051).
* A block-row of 32 rows per CUDA block; rows 0..n_bid are scored.
* Row r writes cells [4r, min(4r + 4, n)); the tail row (r = n_bid) writes n % 4 cells and gets +1e9 iff n % 4 != 0.
* Two warps split K = 128: warp w handles columns w*32 + lane and w*32 + lane + 64.
* max_blocks must equal max_cells / 4 + 1.

## Form template identifiers
cellsWritten <- loop bound `i < n && i < (row+1)*R`, blocksFor <- (max_blocks + ROWS - 1) / ROWS.
-/

namespace Strata.NativeScore

-- § step acceptance

/-- The step check of `score_kernel` (R = 4, budget 2048 so width bound 2051). -/
def stepOk (n maxCells pos full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = min n 2051

/-- A step built from a position as `qsa_step_fill` builds it is accepted. -/
theorem stepOk_of_fill {pos maxCells : Nat} (h : pos + 1 ≤ maxCells) :
    stepOk (pos + 1) maxCells pos ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold stepOk
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The hard-coded 2051 is `top_k + r - 1` for the supported budget 2048. -/
theorem width_bound : 2048 + 4 - 1 = 2051 := by decide

-- § cells written per row

/-- Cells written by row `row`. -/
def cellsWritten (n row : Nat) : Nat := min 4 (n - row * 4)

/-- Full block rows write four cells. -/
theorem cellsWritten_full {n row : Nat} (h : row < n / 4) : cellsWritten n row = 4 := by
  unfold cellsWritten; omega

/-- The tail row writes `n % 4` cells. -/
theorem cellsWritten_tail {n : Nat} : cellsWritten n (n / 4) = n % 4 := by
  unfold cellsWritten; omega

/-- Rows are never wider than four cells. -/
theorem cellsWritten_le (n row : Nat) : cellsWritten n row ≤ 4 := by
  unfold cellsWritten; omega

/-- The +1e9 tail bias fires for the last row exactly when there are tail cells. -/
theorem tail_bias_iff {n : Nat} : n % 4 ≠ 0 ↔ 0 < cellsWritten n (n / 4) := by
  unfold cellsWritten; omega

-- § grid coverage

/-- Number of 32-row CUDA blocks. -/
def blocksFor (maxBlocks : Nat) : Nat := (maxBlocks + 31) / 32

/-- Every row below max_blocks belongs to a launched block. -/
theorem row_block_lt {row maxBlocks : Nat} (h : row < maxBlocks) : row / 32 < blocksFor maxBlocks := by
  unfold blocksFor; omega

/-- With `max_blocks = max_cells / 4 + 1` the scored rows `0..n_bid` all fit. -/
theorem scored_rows_fit {n maxCells : Nat} (h : n ≤ maxCells) :
    n / 4 < maxCells / 4 + 1 := by omega

-- § K split between the two warps

/-- Each of the 128 columns is handled by exactly the (warp, lane, half) given by this witness. -/
theorem column_covered {c : Nat} (hc : c < 128) :
    ∃ w l : Nat, w < 2 ∧ l < 32 ∧ (c = w * 32 + l ∨ c = w * 32 + l + 64) :=
  ⟨(c / 32) % 2, c % 32, by omega, by omega, by omega⟩

/-- The two warps' column sets are disjoint: distinct (w, l, half) give distinct columns. -/
theorem column_unique {w1 l1 h1 w2 l2 h2 : Nat} (hw1 : w1 < 2) (hl1 : l1 < 32) (hh1 : h1 < 2)
    (hw2 : w2 < 2) (hl2 : l2 < 32) (hh2 : h2 < 2)
    (h : w1 * 32 + l1 + 64 * h1 = w2 * 32 + l2 + 64 * h2) :
    w1 = w2 ∧ l1 = l2 ∧ h1 = h2 := by omega

end Strata.NativeScore
