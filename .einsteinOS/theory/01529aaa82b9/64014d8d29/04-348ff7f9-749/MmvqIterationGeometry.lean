/-!
# MODULE: mmvq_iteration_geometry
SOURCES: src/kernels/cuda/native_mmvq.cu
Thread-to-block mapping, blocks-per-iteration, small-K thresholds, grid coverage, and multi-column strides of
the native (llama.cpp-transcribed) MMVQ kernels.

## Source semantics
* A 128-thread block (4 warps x 32) covers `BPI` weight blocks per iteration with `T` threads each, `T*BPI = 128`.
* The small-K rule (4 rows per block) triggers when `n_in / block_elems < BPI`.
* With `ncols` columns quantized as one vector, column `c` starts at activation block `c * (n_in/32)`.
* The upstream multi-column layout uses 2 warps for 5..8 columns with BPI halved.

## Form template identifiers
VDR -> values_per_dot_call ; QI -> dot_calls_per_block ; BPI -> blocks_per_iteration ; ROWS -> rows_per_cuda_block
-/

namespace Strata.MmvqGeometry

-- § threads per weight block and blocks per iteration

/-- (T threads per block, BPI blocks per iteration) for Q5_K, Q2_0, Q3_K, Q6_K, IQ4_XS, Q4_0/Q5_0/IQ4_NL, Q8_0. -/
def traits : List (Nat × Nat) := [(16, 8), (2, 64), (16, 8), (32, 4), (8, 16), (2, 64), (4, 32)]

/-- Every format assigns all 128 threads exactly once per iteration. -/
theorem threads_cover_block : ∀ p ∈ traits, p.1 * p.2 = 128 := by decide

/-- The `BPI` formulas of the source. -/
theorem bpi_values :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧
      4 * 4 * 32 / 32 = 16 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by
  decide

/-- Thread `tid < T*BPI` handles a block index below `BPI`. -/
theorem thread_block_lt (T BPI tid : Nat) (h : tid < T * BPI) : tid / T < BPI :=
  Nat.div_lt_of_lt_mul h

/-- The (block, slot) pair of a thread determines the thread. -/
theorem thread_decomp (T tid : Nat) : T * (tid / T) + tid % T = tid := Nat.div_add_mod tid T

-- § small-K thresholds

/-- Q4_0 / Q5_0 / IQ4_NL: small-K iff `n_in < 2048`. -/
theorem smallk_4 (n : Nat) (h : n % 32 = 0) : n / 32 < 64 ↔ n < 2048 := by omega

/-- Q8_0: small-K iff `n_in < 1024`. -/
theorem smallk_q8_0 (n : Nat) (h : n % 32 = 0) : n / 32 < 32 ↔ n < 1024 := by omega

/-- Q4_K / Q5_K / Q3_K: small-K iff `n_in < 2048`. -/
theorem smallk_k (n : Nat) (h : n % 256 = 0) : n / 256 < 8 ↔ n < 2048 := by omega

/-- Q2_0: small-K iff `n_in < 4096`. -/
theorem smallk_q2_0 (n : Nat) (h : n % 64 = 0) : n / 64 < 64 ↔ n < 4096 := by omega

/-- IQ4_XS: small-K iff `n_in < 4096`. -/
theorem smallk_iq4xs (n : Nat) (h : n % 256 = 0) : n / 256 < 16 ↔ n < 4096 := by omega

/-- Q6_K: small-K iff `n_in < 1024`. -/
theorem smallk_q6k (n : Nat) (h : n % 256 = 0) : n / 256 < 4 ↔ n < 1024 := by omega

-- § grids

/-- Small-K launches 4 rows per block and cover all rows. -/
theorem grid_rows4 (n : Nat) : n ≤ ((n + 3) / 4) * 4 := by omega

/-- The upstream multi-column layout uses 2 rows per block. -/
theorem grid_rows2 (n : Nat) : n ≤ ((n + 1) / 2) * 2 := by omega

/-- With 2 warps (5..8 columns) the blocks per iteration are halved and still use all 64 threads. -/
theorem two_warp_traits : ∀ p ∈ [(16, 4), (2, 32), (32, 2), (8, 8), (4, 16)], p.1 * p.2 = 64 := by decide

/-- Halved BPI values of the formula `BPI * NW / WARPS` for NW = 2. -/
theorem two_warp_bpi : 8 * 2 / 4 = 4 ∧ 64 * 2 / 4 = 32 ∧ 4 * 2 / 4 = 2 ∧ 16 * 2 / 4 = 8 ∧ 32 * 2 / 4 = 16 := by
  decide

-- § multi-column strides

/-- Column `c` of an `ncols`-column activation starts at block `c * m` when `n_in = 32 m`, with the
within-block position unchanged: whole columns never share a 32-element block. -/
theorem column_blocks (c m i : Nat) :
    (c * (32 * m) + i) / 32 = c * m + i / 32 ∧ (c * (32 * m) + i) % 32 = i % 32 := by
  have e : c * (32 * m) = 32 * (c * m) := by rw [Nat.mul_left_comm]
  rw [e]
  exact ⟨Nat.mul_add_div (by decide) _ _, Nat.mul_add_mod _ _ _⟩

/-- Q8_1 activation image size: `ncols * (n_in/32) * 36`. -/
theorem q81_bytes (ncols m : Nat) : ncols * m * 36 = ncols * (m * 36) := Nat.mul_assoc _ _ _

-- § dispatcher coverage

/-- The (type, block elements, block bytes) table of `native_mmvq_weight_bytes`. -/
def weightTable : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144), (13, 256, 176),
   (14, 256, 210), (23, 256, 136), (42, 64, 18), (16, 256, 66), (17, 256, 74), (18, 256, 98),
   (21, 256, 110), (22, 256, 82), (29, 256, 56)]

/-- Every type reported as supported has a weight-size entry. -/
theorem supported_in_table :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29], ∃ e ∈ weightTable, e.1 = t := by
  decide

/-- Block sizes divide `n_in` only if `n_in` is a multiple of 256, 64 or 32: all divide 256. -/
theorem table_elems_divide_256 : ∀ e ∈ weightTable, 256 % e.2.1 = 0 := by decide

end Strata.MmvqGeometry
