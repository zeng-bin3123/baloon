/-!
# MODULE: native_qsa_score_tiles
Tensor-core scoring of pooled indexer blocks (`src/kernels/cuda/native_qsa_score.cu`): each CUDA block
scores 32 pooled rows with 4 heads over 128 dims using two warps, then expands each row's score to its 4 cells.

## Source semantics
* The grid is `ceil(max_blocks / 32)` blocks; rows `> full = n/4` return.
* The step is trusted only if `pos = n-1`, `full = n/4`, `width = min n 2051`, `1 <= n <= max_cells`.
* Columns are split between two warps: `col = warp*32 + lane + 64*k`, `k ∈ {0,1}`.
* Partial results land at `shared[h*68 + i]`, `i = warp*32 + ia*16 + (l/2)*8 + lane/4`, `h = (lane%4)*2 + l%2`.
* Row `r` writes cells `[4r, min n (4r+4))`; the last row gets `+1e9` iff `n % 4 != 0`.

## Form template identifiers
* `gridBlocks`   <- `(max_blocks + ROWS - 1) / ROWS`
* `colOf`        <- `warp*32 + lane + 64*k`
* `accRow`       <- `warp*ROWS + ia*16 + (l/2)*8 + lane/4`
* `accHead`      <- `(lane%4)*2 + l%2`
-/

namespace NativeQsaScoreTiles

-- § row coverage

/-- Number of 32-row blocks. -/
def gridBlocks (maxBlocks : Nat) : Nat := (maxBlocks + 31) / 32

/-- The grid reaches every row. -/
theorem grid_covers (maxBlocks : Nat) : maxBlocks ≤ gridBlocks maxBlocks * 32 := by
  unfold gridBlocks
  omega

/-- Every row is handled by exactly the (block, lane) pair `(row/32, row%32)`. -/
theorem row_roundtrip (row : Nat) : row / 32 * 32 + row % 32 = row := by
  omega

/-- That block index exists in the grid. -/
theorem row_block_lt (row maxBlocks : Nat) (h : row < maxBlocks) : row / 32 < gridBlocks maxBlocks := by
  unfold gridBlocks
  omega

/-- With `max_blocks = max_cells/4 + 1`, every completed-or-tail row of any valid step is inside the grid. -/
theorem full_row_lt (n maxCells : Nat) (h : n ≤ maxCells) : n / 4 < maxCells / 4 + 1 := by
  omega

-- § step trust

/-- The kernel's acceptance test on the step state. -/
def StepOk (maxCells pos n full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos + 1 = n ∧ full = n / 4 ∧ width = min n 2051

/-- A step filled from a position is accepted. -/
theorem stepOk_of_fill (maxCells pos : Nat) (h : pos + 1 ≤ maxCells) :
    StepOk maxCells pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold StepOk
  omega

-- § column split between warps

/-- Column handled by (warp, lane, iteration). -/
def colOf (warp lane k : Nat) : Nat := warp * 32 + lane + 64 * k

/-- The two warps with two iterations tile all 128 columns. -/
theorem colOf_lt (warp lane k : Nat) (hw : warp < 2) (hl : lane < 32) (hk : k < 2) :
    colOf warp lane k < 128 := by
  unfold colOf
  omega

/-- Every column is reached. -/
theorem colOf_surj (c : Nat) (hc : c < 128) :
    ∃ warp lane k, warp < 2 ∧ lane < 32 ∧ k < 2 ∧ colOf warp lane k = c := by
  refine ⟨c / 32 % 2, c % 32, c / 64, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · unfold colOf
    omega

/-- No column is reached twice. -/
theorem colOf_inj (w1 l1 k1 w2 l2 k2 : Nat) (hw1 : w1 < 2) (hl1 : l1 < 32) (hk1 : k1 < 2)
    (hw2 : w2 < 2) (hl2 : l2 < 32) (hk2 : k2 < 2)
    (h : colOf w1 l1 k1 = colOf w2 l2 k2) : w1 = w2 ∧ l1 = l2 ∧ k1 = k2 := by
  unfold colOf at h
  omega

-- § accumulator shuffle

/-- Row of the accumulator fragment element (warp, ia, l, lane). -/
def accRow (warp ia l lane : Nat) : Nat := warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4

/-- Head of the accumulator fragment element. -/
def accHead (l lane : Nat) : Nat := (lane % 4) * 2 + l % 2

/-- The row part lands inside the 64 combined rows. -/
theorem accRow_lt (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4)
    (hlane : lane < 32) : accRow warp ia l lane < 64 := by
  unfold accRow
  omega

/-- The head part lands inside the 8 combined heads. -/
theorem accHead_lt (l lane : Nat) : accHead l lane < 8 := by
  unfold accHead
  omega

/-- So the shared write `h*68 + i` is inside the `8*68` combine area, which fits in the `2*16*36` shared array. -/
theorem combine_index_lt (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4)
    (hlane : lane < 32) : accHead l lane * 68 + accRow warp ia l lane < 1152 := by
  have h1 := accRow_lt warp ia l lane hw hi hl hlane
  have h2 := accHead_lt l lane
  omega

-- § cell expansion

/-- Row `r` writes cells `4r .. 4r+3`; a cell `i` is written by exactly row `i / 4`. -/
theorem cell_row_unique (i row : Nat) : (row * 4 ≤ i ∧ i < row * 4 + 4) ↔ i / 4 = row := by
  omega

/-- Every cell below `n` is written by a row that exists (row ≤ n/4). -/
theorem cell_written (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

/-- The `+1e9` tail bonus applies iff the last row has fewer than 4 cells. -/
theorem tail_partial_iff (n : Nat) : n % 4 ≠ 0 ↔ n < (n / 4 + 1) * 4 ∧ n / 4 * 4 < n := by
  omega

end NativeQsaScoreTiles
