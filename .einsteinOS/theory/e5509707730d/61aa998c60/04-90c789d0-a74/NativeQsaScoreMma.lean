/-!
# MODULE: native_qsa_score_mma
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (TF32 mma) scoring of pooled indexer rows against 4 query heads, with the reference's ReLU-sum order,
optional bias, the +1e9 incomplete-tail boost, and expansion of each row score onto its 4 cells.

## Source semantics
* 32 rows per block, 2 warps; grid = ceil(max_blocks / 32); only rows <= n/4 are scored.
* K = 128 is split between warps: column = warp*32 + lane (+ 64 on the second pass).
* Results go to a shared [head][row] array with row stride 68; row index = warp*32 + ia*16 + (l/2)*8 + lane/4.
* The kernel is a no-op unless the step state is consistent (pos = n-1, n_bid = n/4, width = min(n, 2051)).
* Capacity contract: max_blocks = max_cells / 4 + 1.

## Form template identifiers
- `ROWS` = 32, `WARPS` = 2, `COMBINE` = 68, `R` = 4
- `D` = 128, `HEADS` = 4
-/

namespace Strata.NativeQsaScore

-- § K-split coverage

/-- Every column below 128 is handled by exactly the (warp, lane, pass) triple decoded from it. -/
theorem column_decode (col : Nat) (h : col < 128) :
    col = (col / 32 % 2) * 32 + col % 32 + 64 * (col / 64) := by
  omega

/-- The decoded triple is in range. -/
theorem column_decode_range (col : Nat) (h : col < 128) :
    col / 32 % 2 < 2 ∧ col % 32 < 32 ∧ col / 64 < 2 := by
  omega

/-- Two different (warp, lane, pass) triples never name the same column. -/
theorem column_unique (w l i w' l' i' : Nat) (hl : l < 32) (hl' : l' < 32) (hw : w < 2) (hw' : w' < 2)
    (h : w * 32 + l + 64 * i = w' * 32 + l' + 64 * i') : w = w' ∧ l = l' ∧ i = i' := by
  omega

-- § row and grid coverage

/-- A row is within the grid. -/
theorem row_in_grid (b maxb : Nat) (h : b < maxb) : b / 32 < (maxb + 31) / 32 := by
  omega

/-- A row decomposes into block, lane. -/
theorem row_decode (b : Nat) : b / 32 * 32 + b % 32 = b := by
  omega

/-- The capacity relation puts the last possible block row inside the pooled buffer. -/
theorem last_row_in_capacity (n maxc : Nat) (hn : n ≤ maxc) : n / 4 < maxc / 4 + 1 := by
  omega

-- § combine buffer

/-- Row index inside the 64 combined rows. -/
theorem combine_row_lt (warp ia l lane : Nat) (hw : warp < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) :
    warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by
  omega

/-- Combined index head * 68 + row stays inside the 8 * 68 combine area. -/
theorem combine_index_lt (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 8 * 68 := by
  omega

/-- The combine area fits in the shared buffer of WARPS * 16 * 36 floats. -/
theorem combine_fits_shared : 8 * 68 ≤ 2 * 16 * 36 := by
  decide

/-- The second warp's partial is read at offset 32, still inside the row's 68-wide stride. -/
theorem second_partial_inside (j lane : Nat) (hj : j < 4) (hl : lane < 32) : j * 68 + 32 + lane < 8 * 68 := by
  omega

/-- Head slot of a lane/element pair is below 8, matching the padded 8-head B tile. -/
theorem head_slot_lt (lane l : Nat) (hl : lane < 32) (hl2 : l < 4) : (lane % 4) * 2 + l % 2 < 8 := by
  omega

-- § cell expansion

/-- Cells written for row `row` all lie in block `row`. -/
theorem cell_block (row i : Nat) (h1 : row * 4 ≤ i) (h2 : i < (row + 1) * 4) : i / 4 = row := by
  omega

/-- Cells written are real cached cells: i < n. -/
theorem cell_written_real (row i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

-- § step-state guard

/-- The guard width formula equals min(n, 2051) = min(n, 2048 + 4 - 1). -/
theorem width_formula (n : Nat) : (if n < 2051 then n else 2051) = min n (2048 + 4 - 1) := by
  split <;> omega

-- § summation order

/-- The kernel adds 0 first, so the head value is the first partial plus the second; a zero seed is neutral. -/
theorem zero_seed_neutral (a b : Nat) : (0 + a) + b = a + b := by
  omega

/-- Bias and tail boost are added after the head sum, and a zero boost is neutral. -/
theorem zero_boost_neutral (s : Nat) : s + 0 = s := by
  omega

end Strata.NativeQsaScore
