/-!
# MODULE: native_mmvq_geometry
Launch geometry and byte accounting of the native MMVQ kernels in `native_mmvq.cu`.

## Source semantics
* Block structs have fixed byte sizes (Q5_K 176, Q4_K 144, Q3_K 110, Q6_K 210, IQ4_XS 136, Q4_0/IQ4_NL/Q2_0 18,
  Q5_0 22, Q8_0 34, Q8_1 36).
* A launch uses 4 warps of 32 lanes; row `blockIdx.x*ROWS + i` is handled in block `blockIdx.x`.
* Blocks per iteration `BPI = VDR*WARPS*WARP/QI` and the small-K rule `n_in/QK < BPI`.
* `native_mmvq_weight_bytes = (n_in/elems)*bytes*n_out`; Q8_1 buffer = `ncols*(n_in/32)*36`.

## Form template identifiers
QK -> block_elems; QI -> ints_per_block; VDR -> values_per_thread; WARPS -> warps_per_block;
BPI -> blocks_per_iteration; ROWS -> rows_per_block.
-/

namespace NativeMmvqGeometry

-- § block byte sizes

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q4_K: dm 4 + scales 12 + qs 128. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q5_K: dm 4 + scales 12 + qh 32 + qs 128. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS: d 2 + scales_h 2 + scales_l 4 + qs 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q5_0: d 2 + qh 4 + qs 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_1: ds 4 + qs 32. -/
theorem q81_size : 4 + 32 = 36 := by decide

-- § iteration traits

/-- Blocks per iteration for the K formats with QI = 32, VDR = 2. -/
theorem bpi_q45k : 2 * 4 * 32 / 32 = 8 := by decide
/-- Q6_K: VDR = 1, QI = 32. -/
theorem bpi_q6k : 4 * 32 / 32 = 4 := by decide
/-- Q3_K: QI = 16. -/
theorem bpi_q3k : 4 * 32 / 16 = 8 := by decide
/-- IQ4_XS: VDR = 4. -/
theorem bpi_iq4xs : 4 * 4 * 32 / 32 = 16 := by decide
/-- Q2_0: QI = 2. -/
theorem bpi_q20 : 4 * 32 / 2 = 64 := by decide
/-- Q4_0 / Q5_0 / IQ4_NL small-K threshold in blocks (64) and elements (2048). -/
theorem small_k_q40 : 2 * 4 * 32 / 4 = 64 ∧ 64 * 32 = 2048 := by decide
/-- Q8_0 small-K threshold in blocks (32) and elements (1024). -/
theorem small_k_q80 : 2 * 4 * 32 / 8 = 32 ∧ 32 * 32 = 1024 := by decide

/-- For the 16-per-block formats the thread-to-block map stays below the block-per-iteration count. -/
theorem q3k_thread_block (tid : Nat) (h : tid < 128) : tid / 16 < 8 := by omega

/-- For Q6_K the thread-to-block map stays below four. -/
theorem q6k_thread_block (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- For IQ4_XS the thread-to-block map stays below 16. -/
theorem iq4xs_thread_block (tid : Nat) (h : tid < 128) : tid / 8 < 16 := by omega

/-- The IQ4_XS lane offset is a multiple of 4 below 32. -/
theorem iq4xs_kqs (tid : Nat) : 4 * (tid % 8) < 32 := by omega

-- § row grouping

/-- Grid size covers all rows when 4 rows share a block. -/
theorem row_group_covers (row n_out : Nat) (h : row < n_out) : row / 4 < (n_out + 3) / 4 := by
  omega

/-- A (block, warp) pair reconstructs the row. -/
theorem row_from_block_warp (row : Nat) : (row / 4) * 4 + row % 4 = row := by omega

/-- The warp index of a row is below 4. -/
theorem row_warp_lt (row : Nat) : row % 4 < 4 := by omega

-- § byte accounting

/-- Weight bytes of the matrix. -/
def weightBytes (elems bytes n_in n_out : Nat) : Nat := (n_in / elems) * bytes * n_out

/-- Row-major: n_out rows of identical size. -/
theorem weightBytes_eq (elems bytes n_in n_out : Nat) :
    weightBytes elems bytes n_in n_out = n_out * ((n_in / elems) * bytes) := by
  unfold weightBytes; rw [Nat.mul_comm]

/-- Row offset of a valid row stays inside the matrix. -/
theorem row_end_le (elems bytes n_in n_out row : Nat) (h : row < n_out) :
    (row + 1) * ((n_in / elems) * bytes) ≤ weightBytes elems bytes n_in n_out := by
  rw [weightBytes_eq]
  exact Nat.mul_le_mul_right _ h

/-- Columns quantize as one vector: block boundaries never straddle a column. -/
theorem quant_block_inside_column (ncols m : Nat) : (ncols * (32 * m)) / 32 = ncols * m := by
  rw [Nat.mul_left_comm]; omega

/-- Number of Q8_1 blocks of the whole batch. -/
theorem q8_1_blocks (ncols m : Nat) : ncols * (32 * m) = 32 * (ncols * m) := Nat.mul_left_comm _ _ _

end NativeMmvqGeometry
