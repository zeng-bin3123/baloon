/-!
# MODULE: native_mmvq_q8_dot
Block geometry, thread-to-block mapping, shape validation and Q8_1 activation layout of the native
CUDA MMVQ kernels (Q2_0, Q3_K..Q6_K, IQ4_XS and the 32-element legacy formats).

## Source semantics
Weights are stored as fixed-size blocks; activations are quantized to Q8_1 blocks (36 bytes per 32 values).
Each kernel assigns `tid / T` as the block index and strides by `BPI` blocks, where `T * BPI = 128`
(4 warps x 32 lanes). Columns are contiguous Q8_1 vectors; every 32-element block stays inside one column.
Row groups are launched as `ceil(n_out / 4)` blocks of 4 rows for small K and `n_out` blocks otherwise.

## Form template identifiers
- `Q5KBlock/Q4KBlock/Q3KBlock/Q6KBlock/IQ4XSBlock/Q20Block/Q40Block/Q50Block/Q80Block/IQ4NLBlock` -> weight block
- `Q81Block` -> activation block
- `native_mmvq_weight_bytes` -> weightBytes, `validate_shape` -> validShape, `native_quantize_q8_1` -> q8_1Bytes
-/

namespace Strata.NativeMmvq

-- § constants

/-- Elements per K-quant super-block. -/
def QK : Nat := 256
/-- Elements per Q8_1 block. -/
def Q8K : Nat := 32
/-- Ints per K-quant block. -/
def QI : Nat := 32
/-- Vector dot ratio of the K-quant kernels. -/
def VDR : Nat := 2
/-- Warps per thread block. -/
def WARPS : Nat := 4
/-- Lanes per warp. -/
def WARP : Nat := 32

/-- Blocks per K iteration for the Q4_K / Q5_K kernels. -/
theorem k_quant_blocks_per_iter : VDR * WARPS * WARP / QI = 8 := by decide

/-- Small-K thresholds of the legacy 32-element kernels: 64 blocks for Qi=4, 32 for Qi=8. -/
theorem small_k_thresholds : 2 * WARPS * WARP / 4 = 64 ∧ 2 * WARPS * WARP / 8 = 32 := by decide

/-- Small-K element thresholds (2048 / 1024). -/
theorem small_k_element_thresholds : 64 * 32 = 2048 ∧ 32 * 32 = 1024 := by decide

-- § block layouts

/-- Q4_K block: dm(4) + scales(12) + qs(128). -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q5_K block: dm(4) + scales(12) + qh(32) + qs(128). -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q3_K block: hmask(32) + qs(64) + scales(12) + d(2). -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q6_K block: ql(128) + qh(64) + scales(16) + d(2). -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS block: d(2) + scales_h(2) + scales_l(4) + qs(128). -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Legacy blocks: Q4_0 18, Q5_0 22, Q8_0 34, Q2_0 18, Q8_1 36 bytes. -/
theorem legacy_sizes : 2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧ 4 + 32 = 36 := by decide
/-- Packed element counts: K-quant nibbles, Q3_K 2-bit codes, Q2_0 2-bit codes, Q4_0 nibbles. -/
theorem packed_element_counts : 128 * 2 = 256 ∧ 64 * 4 = 256 ∧ 16 * 4 = 64 ∧ 16 * 2 = 32 := by decide

-- § geometry and supported types

/-- (elements per block, bytes per block) for the non-IQ GGML types handled here. -/
def blockGeom : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Weight bytes of an `n_out x n_in` matrix, row-major blocks. -/
def weightBytes (t nIn nOut : Nat) : Option Nat :=
  (blockGeom t).map fun p => (nIn / p.1) * p.2 * nOut

/-- Q6_K geometry. -/
theorem blockGeom_q6k : blockGeom 14 = some (256, 210) := rfl

/-- Example weight size: Q8_0, 4096 inputs, 2 outputs. -/
theorem weightBytes_q8_0_example : weightBytes 8 4096 2 = some (128 * 34 * 2) := by decide

/-- Example weight size: Q5_K, 2560 inputs, 10 outputs. -/
theorem weightBytes_q5k_example : weightBytes 13 2560 10 = some (10 * 176 * 10) := by decide

/-- Unsupported types have no geometry. -/
theorem blockGeom_unsupported : blockGeom 0 = none := rfl

/-- Types dispatched to the native MMVQ entry points. -/
def supported (t : Nat) : Bool :=
  [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29].contains t

/-- Q4_0 is supported, type 3 is not. -/
theorem supported_examples : supported 2 = true ∧ supported 3 = false := by decide

/-- IQ types are supported but delegate geometry to `iq_row_bytes`. -/
theorem iq_supported_no_local_geometry : supported 16 = true ∧ blockGeom 16 = none := by decide

-- § shape validation

/-- `validate_shape`: positive, block-divisible `n_in` and `1 ≤ ncols ≤ 8`. -/
def validShape (nIn ncols blk : Nat) : Prop :=
  0 < nIn ∧ nIn % blk = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- A valid shape tiles exactly into blocks. -/
theorem validShape_tiles (nIn ncols blk : Nat) (h : validShape nIn ncols blk) :
    (nIn / blk) * blk = nIn :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2.1)

-- § Q8_1 activation layout

/-- Bytes of `ncols` Q8_1 activation columns. -/
def q8_1Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

/-- Column bytes scale linearly with the number of columns. -/
theorem q8_1Bytes_mul (nIn ncols : Nat) : q8_1Bytes nIn ncols = ncols * q8_1Bytes nIn 1 := by
  unfold q8_1Bytes
  rw [Nat.one_mul, Nat.mul_assoc]

/-- A column-major `ncols`-vector quantizes as one vector: total length stays a multiple of 32. -/
theorem total_multiple_of_32 (nIn ncols : Nat) (h : nIn % 32 = 0) : (nIn * ncols) % 32 = 0 := by
  rw [Nat.mul_mod, h]
  simp

/-- Block index of element `k` of column `j` is `j * (n_in/32) + k/32`: blocks never straddle columns. -/
theorem column_block_index (j m k : Nat) : (j * (32 * m) + k) / 32 = j * m + k / 32 := by
  have h : j * (32 * m) + k = k + 32 * (j * m) := by
    rw [Nat.mul_left_comm]
    omega
  rw [h, Nat.add_mul_div_left k (j * m) (by decide : 0 < 32)]
  omega

/-- Output index `j * n_out + row` of column `j` stays inside the `ncols * n_out` buffer. -/
theorem output_index_lt (j c r n : Nat) (hj : j < c) (hr : r < n) : j * n + r < c * n := by
  have h1 := Nat.mul_le_mul_right n (Nat.succ_le_of_lt hj)
  rw [Nat.succ_mul] at h1
  omega

-- § thread mapping

/-- Per-format `T * BPI = 128`: Q5_K/Q4_K, Q2_0, Q3_K, Q6_K, IQ4_XS, Q4_0-family, Q8_0. -/
theorem threads_times_bpi :
    16 * 8 = 128 ∧ 2 * 64 = 128 ∧ 16 * 8 = 128 ∧ 32 * 4 = 128 ∧ 8 * 16 = 128 ∧ 2 * 64 = 128 ∧ 4 * 32 = 128 := by
  decide

/-- Every lane's first block index `tid / T` is below the per-iteration block count. -/
theorem lane_block_lt (tid T B : Nat) (hT : 0 < T) (h : T * B = 128) (ht : tid < 128) :
    tid / T < B := by
  rw [Nat.div_lt_iff_lt_mul hT]
  have h' : B * T = 128 := by rw [Nat.mul_comm]; exact h
  omega

-- § launch geometry

/-- Small-K launches cover every row with 4 rows per block. -/
theorem small_k_blocks_cover (nOut : Nat) : nOut ≤ ((nOut + 3) / 4) * 4 := by omega

-- § affine corrections for Q4_0 / Q5_0

/-- The Q4_0 correction `8 * vdr / qi` equals 4, Q5_0's `16 * vdr / qi` equals 8. -/
theorem affine_correction_constants : 8 * 2 / 4 = 4 ∧ 16 * 2 / 4 = 8 := by decide

/-- Offset codes: `(c - 8) * q = c * q - 8 * q`, which is why the Q8_1 sum is consumed. -/
theorem affine_split (c q : Int) : (c - 8) * q = c * q - 8 * q := Int.sub_mul c 8 q

end Strata.NativeMmvq
