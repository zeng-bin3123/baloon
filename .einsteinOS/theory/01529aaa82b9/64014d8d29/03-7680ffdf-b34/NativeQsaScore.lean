/-!
# MODULE: native_qsa_score
SOURCES: src/kernels/cuda/native_qsa_score.cu
llama.cpp-order tensor-core (mma m16n8k8 tf32) scoring of pooled indexer keys against the 4 indexer query
heads, writing per-cell scores; guarded by a step-consistency predicate and disjoint-span validation.

## Source semantics
* The kernel runs only when the step is consistent: 1 <= n <= max_cells, pos = n-1, full = n/4,
  width = min(n, 2051); otherwise it returns without writing.
* Row `row <= full` is scored by block row/32, lane row%32; it writes cells [4*row, min(n, 4*row+4)).
* K = 128 is split between 2 warps; each covers columns warp*32+lane and warp*32+lane+64.
* The spare row (row == full) gets +1e9 iff n % 4 != 0.
* Input/output spans must be 4-aligned, bounded and pairwise disjoint.

## Form template identifiers
ROWS -> 32; WARPS -> 2; R -> 4; COMBINE -> 68; STRIDE -> 36.
-/

namespace NativeQsaScore

/-- Step consistency guard of `score_kernel`. -/
def stepValid (n pos full width maxCells : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- A step produced by the host `step_fill` for any in-capacity position passes the guard. -/
theorem stepValid_of_fill (pos maxCells : Nat) (h : pos + 1 ≤ maxCells) :
    stepValid (pos + 1) pos ((pos + 1) / 4) (if pos + 1 < 2051 then pos + 1 else 2051) maxCells := by
  unfold stepValid
  refine ⟨by omega, h, by omega, rfl, rfl⟩

/-- The guard's width is the QSA selection width `min n 2051` (top_k 2048 + r - 1). -/
theorem guard_width_is_min (n : Nat) : (if n < 2051 then n else 2051) = min n 2051 := by
  split <;> omega

-- § launch geometry

/-- Capacity convention: `max_blocks = max_cells/4 + 1`. -/
def maxBlocksOf (maxCells : Nat) : Nat := maxCells / 4 + 1

/-- Every scored row index fits inside the block capacity. -/
theorem full_lt_maxBlocks (n maxCells : Nat) (h : n ≤ maxCells) : n / 4 < maxBlocksOf maxCells := by
  unfold maxBlocksOf; omega

/-- A row `<= full` is covered by some block of the launch grid `ceil(max_blocks / 32)`. -/
theorem row_block_covered (row full maxCells : Nat) (hf : full < maxBlocksOf maxCells) (hr : row ≤ full) :
    row / 32 < (maxBlocksOf maxCells + 31) / 32 ∧ row = (row / 32) * 32 + row % 32 := by
  omega

/-- Every cell `j < n` is written by row `j/4 <= full`. -/
theorem cell_covered (j n : Nat) (h : j < n) : j / 4 ≤ n / 4 ∧ 4 * (j / 4) ≤ j ∧ j < 4 * (j / 4) + 4 := by
  omega

/-- Distinct rows write disjoint cells. -/
theorem rows_disjoint (r1 r2 c1 c2 : Nat) (h1 : c1 < 4) (h2 : c2 < 4) (h : r1 * 4 + c1 = r2 * 4 + c2) :
    r1 = r2 ∧ c1 = c2 := by omega

/-- Written cells are real cells: `i < n` and `i < 4*(row+1)` for the loop bound. -/
theorem written_cell_range (row n i : Nat) (hi : i < n ∧ i < (row + 1) * 4) (hl : row * 4 ≤ i) :
    i / 4 = row := by
  have : (row + 1) * 4 = row * 4 + 4 := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § K split between two warps

/-- Every K column is accumulated by exactly one (warp, lane, pass). -/
theorem k_columns_cover (col : Nat) (h : col < 128) :
    col = (col / 64) * 64 + ((col / 32) % 2) * 32 + col % 32 ∧
    (col / 32) % 2 < 2 ∧ col % 32 < 32 ∧ col / 64 < 2 := by omega

/-- The spare-row bias flag. -/
theorem tail_bias_flag (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

-- § shared-memory layouts

/-- The 4(8)-head by 64-row combine buffer index stays inside shared memory (8 * 68 <= 2 * 16 * 36). -/
theorem combine_in_shared (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 2 * 16 * 36 := by omega

/-- The combine index is injective (decode from the `68` stride). -/
theorem combine_injective (h1 h2 i1 i2 : Nat) (a : i1 < 64) (b : i2 < 64)
    (h : h1 * 68 + i1 = h2 * 68 + i2) : h1 = h2 ∧ i1 = i2 := by omega

/-- Row index written back by a lane: `warp*32 + ia*16 + (l/2)*8 + lane/4 < 64`; head index `< 8`. -/
theorem writeback_ranges (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4) (hn : lane < 32) :
    warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 ∧ (lane % 4) * 2 + l % 2 < 8 := by omega

/-- A warp's staging tile (16 rows of stride 36) fits its share of shared memory. -/
theorem tile_in_shared (i lane warp : Nat) (hi : i < 16) (hl : lane < 32) (hw : warp < 2) :
    warp * 16 * 36 + i * 36 + lane < 2 * 16 * 36 := by omega

-- § span validation

/-- Half-open interval overlap test used by `overlaps`. -/
def overlaps (x nx y ny : Nat) : Prop := x < y + ny ∧ y < x + nx

/-- Overlap is symmetric. -/
theorem overlaps_symm (x nx y ny : Nat) : overlaps x nx y ny ↔ overlaps y ny x nx := by
  unfold overlaps; omega

/-- Ranges that end before the other starts do not overlap. -/
theorem not_overlaps_of_le (x nx y ny : Nat) (h : x + nx ≤ y) : ¬ overlaps x nx y ny := by
  unfold overlaps; omega

/-- An empty span can only "overlap" by lying strictly inside the other span. -/
theorem empty_span_inside (x y ny : Nat) (h : overlaps x 0 y ny) : y < x ∧ x < y + ny := by
  unfold overlaps at h; omega

end NativeQsaScore
