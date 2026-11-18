-- Baloon.NativeQsaScore: tile-index arithmetic and consistency guards of the
-- tensor-core (mma m16n8k8 tf32) QSA block-scoring kernel.
-- SOURCE: src/kernels/cuda/native_qsa_score.cu
-- Unit: u316bd6b9ec  ROWS=32 WARPS=2 R=4 COMBINE=68 STRIDE=36

namespace Baloon.NativeQsaScore

/-- Step consistency guard: 0 < n <= max_cells, pos = n-1, full = n/4, width = min(n, 2051). -/
def stepValid (n pos full width maxCells : Nat) : Prop :=
  0 < n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

theorem stepValid_of_fill (pos maxCells : Nat) (h : pos + 1 ≤ maxCells) :
    stepValid (pos + 1) pos ((pos + 1) / 4) (if pos + 1 < 2051 then pos + 1 else 2051) maxCells := by
  simp only [stepValid]; exact ⟨by omega, h, by omega, rfl, rfl⟩

theorem guard_width_is_min (n : Nat) : (if n < 2051 then n else 2051) = min n 2051 := by
  split_ifs with h <;> omega

/-- Capacity: max_blocks = 1 + max_cells / 4. -/
def maxBlocksOf (maxCells : Nat) : Nat := 1 + maxCells / 4

theorem full_lt_maxBlocks (n maxCells : Nat) (h : n ≤ maxCells) : n / 4 < maxBlocksOf maxCells := by
  simp only [maxBlocksOf]; omega

theorem row_block_covered (row full maxCells : Nat) (hf : full < maxBlocksOf maxCells) (hr : row ≤ full) :
    row / 32 < (maxBlocksOf maxCells + 31) / 32 ∧ row = (row / 32) * 32 + row % 32 :=
  ⟨by omega, by omega⟩

theorem cell_covered (j n : Nat) (h : j < n) :
    j / 4 ≤ n / 4 ∧ 4 * (j / 4) ≤ j ∧ j < 4 * (j / 4) + 4 :=
  ⟨by omega, by omega, by omega⟩

theorem rows_disjoint (r1 r2 c1 c2 : Nat) (h1 : c1 < 4) (h2 : c2 < 4) (h : r1 * 4 + c1 = r2 * 4 + c2) :
    r1 = r2 ∧ c1 = c2 :=
  ⟨by omega, by omega⟩

theorem written_cell_range (row n i : Nat) (hi : i < n ∧ i < (row + 1) * 4) (hl : row * 4 ≤ i) :
    i / 4 = row := by
  have expand : (row + 1) * 4 = row * 4 + 4 := by ring
  omega

theorem k_columns_cover (col : Nat) (h : col < 128) :
    col = (col / 64) * 64 + ((col / 32) % 2) * 32 + col % 32 ∧
    (col / 32) % 2 < 2 ∧ col % 32 < 32 ∧ col / 64 < 2 :=
  ⟨by omega, by omega, by omega, by omega⟩

theorem tail_bias_flag (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n :=
  Iff.intro (by omega) (by omega)

theorem combine_in_shared (h i : Nat) (hh : h < 8) (hi : i < 64) : h * 68 + i < 2 * 16 * 36 := by
  have bh : h * 68 ≤ 7 * 68 := Nat.mul_le_mul_right 68 (by omega)
  omega

theorem combine_injective (h1 h2 i1 i2 : Nat) (a : i1 < 64) (b : i2 < 64)
    (h : h1 * 68 + i1 = h2 * 68 + i2) : h1 = h2 ∧ i1 = i2 :=
  ⟨by omega, by omega⟩

theorem writeback_ranges (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4) (hn : lane < 32) :
    warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 ∧ (lane % 4) * 2 + l % 2 < 8 :=
  ⟨by omega, by omega⟩

theorem tile_in_shared (i lane warp : Nat) (hi : i < 16) (hl : lane < 32) (hw : warp < 2) :
    warp * 16 * 36 + i * 36 + lane < 2 * 16 * 36 := by
  have bw : warp * (16 * 36) ≤ 1 * (16 * 36) := Nat.mul_le_mul_right (16 * 36) (by omega)
  omega

/-- Half-open interval overlap: commuted order y < x+nx /\ x < y+ny. -/
def overlaps (x nx y ny : Nat) : Prop := y < x + nx ∧ x < y + ny

theorem overlaps_symm (x nx y ny : Nat) : overlaps x nx y ny ↔ overlaps y ny x nx := by
  simp only [overlaps]; constructor <;> intro ⟨h1, h2⟩ <;> exact ⟨h2, h1⟩

theorem not_overlaps_of_le (x nx y ny : Nat) (h : x + nx ≤ y) : ¬ overlaps x nx y ny := by
  simp only [overlaps]; omega

theorem empty_span_inside (x y ny : Nat) (h : overlaps x 0 y ny) : y < x ∧ x < y + ny := by
  simp only [overlaps] at h; exact ⟨h.1, h.2⟩

end Baloon.NativeQsaScore
