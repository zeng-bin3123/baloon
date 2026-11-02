/-!
# MODULE: native_mmvq
Iteration geometry, block byte sizes and launch arithmetic of the native quantized matrix-vector kernels.

## Source semantics
Each of the 128 threads (4 warps x 32 lanes) starts at block tid/T and strides by BPI blocks per iteration, where T * BPI = 128. Small-K launches use 4 rows per block, otherwise 1. Q8_1 blocks are 36 bytes (4 + 32). Multi-column activations are contiguous, so ncols columns of n_in = 32k quantize as one vector of ncols*n_in elements.

## Form template identifiers
Q5KTraits.. -> format_traits, T -> threads_per_block_group, BPI -> blocks_per_iteration
-/
namespace Strata.NativeMmvq

-- § thread mapping

/-- A thread's starting block index is below the blocks-per-iteration stride. -/
theorem start_lt_stride (tid T BPI : Nat) (hT : 128 = T * BPI) (h : tid < 128) : tid / T < BPI := Nat.div_lt_of_lt_mul (by omega)

/-- Q5_K / Q4_K: T = 16, BPI = 8. -/
theorem q45k_shape : 16 * 8 = 128 := by decide
/-- Q2_0: T = 2, BPI = 64. -/
theorem q20_shape : 2 * 64 = 128 := by decide
/-- Q3_K: T = 16, BPI = 8. -/
theorem q3k_shape : 16 * 8 = 128 := by decide
/-- Q6_K: T = 32, BPI = 4. -/
theorem q6k_shape : 32 * 4 = 128 := by decide
/-- IQ4_XS: T = 8, BPI = 16. -/
theorem iq4xs_shape : 8 * 16 = 128 := by decide
/-- Small formats with Qi = 4: T = 2, BPI = 64. -/
theorem small4_shape : 2 * 64 = 128 := by decide
/-- Q8_0 with Qi = 8: T = 4, BPI = 32. -/
theorem small8_shape : 4 * 32 = 128 := by decide

-- § block byte sizes

/-- Q8_1 block: half2 + 32 codes. -/
theorem q81_bytes : 4 + 32 = 36 := by decide
/-- Q4_0 block. -/
theorem q40_bytes : 2 + 16 = 18 := by decide
/-- Q5_0 block. -/
theorem q50_bytes : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block. -/
theorem q80_bytes : 2 + 32 = 34 := by decide
/-- Q4_K block. -/
theorem q4k_bytes : 4 + 12 + 128 = 144 := by decide
/-- Q5_K block. -/
theorem q5k_bytes : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q6_K block. -/
theorem q6k_bytes : 128 + 64 + 16 + 2 = 210 := by decide
/-- Q3_K block. -/
theorem q3k_bytes : 32 + 64 + 12 + 2 = 110 := by decide
/-- IQ4_XS block. -/
theorem iq4xs_bytes : 2 + 2 + 4 + 128 = 136 := by decide

-- § launch arithmetic

/-- Small-K grid (4 rows per block) covers n_out. -/
theorem smallk_grid (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by omega

/-- Upstream multi layout (2 rows per block) covers n_out. -/
theorem upstream_grid (n_out : Nat) : n_out ≤ (n_out + 1) / 2 * 2 := by omega

/-- Quantizing ncols columns as one vector keeps block boundaries inside columns. -/
theorem multi_quant_blocks (k ncols : Nat) : (32 * k * ncols) / 32 = k * ncols := by rw [Nat.mul_assoc]; exact Nat.mul_div_cancel_left _ (by decide)

/-- Weight bytes: rows times blocks per row times block bytes. -/
def weightBytes (n_in elems bytes n_out : Nat) : Nat := n_in / elems * bytes * n_out

/-- Zero rows need zero weight bytes. -/
theorem weight_zero (n_in elems bytes : Nat) : weightBytes n_in elems bytes 0 = 0 := by simp [weightBytes]

/-- Multi-column output element j*n_out + row stays inside ncols*n_out. -/
theorem out_index_lt (j row n_out ncols : Nat) (hj : j < ncols) (hr : row < n_out) : j * n_out + row < ncols * n_out := by
  have h := Nat.mul_le_mul_right n_out (show j + 1 ≤ ncols by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

end Strata.NativeMmvq
