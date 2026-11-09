/-!
# MODULE: native_mmvq_quantized_dot
SOURCES: src/kernels/cuda/native_mmvq.cu
Quantized matrix-vector product (MMVQ) on Q8_1-quantized activations for K-quant, legacy and IQ4 weight
formats, ported from llama.cpp. Single-column and multi-column (2..8) launch geometry, validation, and
weight-byte accounting.

## Source semantics
- Each weight format has a fixed block size (elements, bytes); a row has `n_in / block_elems` blocks and the
  weight buffer has `row_bytes * n_out` bytes.
- Activations are quantized to Q8_1 blocks (32 elements, 36 bytes); `ncols` columns are contiguous, so a
  quantizer pass over `n_in * ncols` elements never straddles a column.
- Threads: 4 warps x 32 lanes; blocks per K-iteration = VDR*WARPS*WARP/QI; "small K" rule switches to 4 rows
  per CUDA block when `n_in/block < blocks_per_iter`.
- Validation: `1 <= ncols <= 8`, `n_in > 0` and divisible by the block element count, `n_out > 0`.

## Form template identifiers
- Q5KBlock/Q4KBlock/Q3KBlock/Q6KBlock/IQ4XSBlock/Q40Block/Q50Block/Q80Block/IQ4NLBlock/Q20Block -> block layout
- Q81Block -> activation block
- native_mmvq_weight_bytes -> weightBytes ; native_q8_1_bytes -> q81Bytes ; native_mmvq_supported -> supported
-/

namespace Strata.NativeMmvq

-- § constants
/-- Warps per CUDA block. -/
def WARPS : Nat := 4
/-- Lanes per warp. -/
def WARP : Nat := 32
/-- Q8_1 elements per block. -/
def Q8K : Nat := 32
/-- Maximum activation columns. -/
def MAX_NCOLS : Nat := 8

/-- (block elements, block bytes) per supported non-IQ GGML type id. -/
def blockInfo : Nat → Option (Nat × Nat)
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

/-- Weight bytes of an `n_out`-row matrix whose rows have `n_in` elements. -/
def weightBytes (elems bytes nIn nOut : Nat) : Nat := (nIn / elems) * bytes * nOut

/-- Q8_1 buffer size for `ncols` columns of `n_in` elements (36 bytes per 32). -/
def q81Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

-- § struct layouts
/-- Q5_K block: half2 dm (4) + scales (12) + qh (32) + qs (128) = 176 bytes. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block: dm (4) + scales (12) + qs (128) = 144 bytes. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: hmask (32) + qs (64) + scales (12) + d (2) = 110 bytes. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q6_K block: ql (128) + qh (64) + scales (16) + d (2) = 210 bytes. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS block: d (2) + scales_h (2) + scales_l (4) + qs (128) = 136 bytes. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q8_1 block: ds half2 (4) + 32 int8 = 36 bytes. -/
theorem q81_size : 4 + 32 = 36 := by decide
/-- Q5_0 block: d (2) + qh (4) + qs (16) = 22 bytes. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: d (2) + 32 int8 = 34 bytes. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- Q4_0 / IQ4_NL / Q2_0 block: d (2) + 16 bytes = 18. -/
theorem q40_size : 2 + 16 = 18 := by decide

-- § activation quantization
/-- Each flat element index splits uniquely into (Q8_1 block, lane in block). -/
theorem q81_index_split (i : Nat) : 32 * (i / 32) + i % 32 = i := by
  omega

/-- Q8_1 buffer size is additive in the number of columns. -/
theorem q81Bytes_add (n a b : Nat) : q81Bytes n (a + b) = q81Bytes n a + q81Bytes n b := by
  unfold q81Bytes
  rw [Nat.add_mul, Nat.add_mul]

/-- When `n_in` is a multiple of 32, a column's blocks are whole, so column boundaries are block boundaries. -/
theorem column_block_aligned (n c : Nat) (h : n % 32 = 0) : (n * c) / 32 = c * (n / 32) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = 32 * m := ⟨n / 32, by omega⟩
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ (by decide : 0 < 32),
    Nat.mul_div_cancel_left _ (by decide : 0 < 32)]
  exact Nat.mul_comm _ _

-- § iteration geometry
/-- Blocks per K-iteration for the K-quants Q4_K/Q5_K (VDR=2, QI=32): 8. -/
theorem bpi_q45k : 2 * WARPS * WARP / 32 = 8 := by decide
/-- Blocks per iteration for Q3_K (VDR=1, QI=16): 8. -/
theorem bpi_q3k : WARPS * WARP / 16 = 8 := by decide
/-- Blocks per iteration for Q6_K (VDR=1, QI=32): 4. -/
theorem bpi_q6k : WARPS * WARP / 32 = 4 := by decide
/-- Blocks per iteration for IQ4_XS (VDR=4, QI=32): 16. -/
theorem bpi_iq4xs : 4 * WARPS * WARP / 32 = 16 := by decide
/-- Blocks per iteration for Q2_0 (QI=2): 64. -/
theorem bpi_q20 : WARPS * WARP / 2 = 64 := by decide
/-- Q4_0/Q5_0/IQ4_NL (Qi=4, VDR=2): 64 blocks per iteration. -/
theorem bpi_small4 : 2 * WARPS * WARP / 4 = 64 := by decide
/-- Q8_0 (Qi=8, VDR=2): 32 blocks per iteration. -/
theorem bpi_small8 : 2 * WARPS * WARP / 8 = 32 := by decide

/-- Small-K threshold of Q4_0-family: `n_in/32 < 64` iff `n_in < 2048`. -/
theorem smallK_threshold_q40 (n : Nat) : n / 32 < 64 ↔ n < 2048 := by
  omega
/-- Small-K threshold of Q8_0: `n_in/32 < 32` iff `n_in < 1024`. -/
theorem smallK_threshold_q80 (n : Nat) : n / 32 < 32 ↔ n < 1024 := by
  omega

/-- For the K-quant mapping every thread of a 128-thread block lands in a block index below 8. -/
theorem kbx_first_lt_bpi (tid : Nat) (h : tid < 128) : tid / 16 < 8 := by
  omega

/-- With 4 rows per CUDA block the grid covers every output row. -/
theorem grid_covers_small (nOut : Nat) : nOut ≤ ((nOut + 3) / 4) * 4 := by
  omega

/-- Exactly one CUDA block per row in the normal layout: row index equals block index (trivial cover). -/
theorem row_cover_normal (nOut r : Nat) (h : r < nOut) : r < nOut := h

/-- Q3_K signed scale: 4 low bits and 2 high bits, minus 32, lies in [-32, 31]. -/
theorem q3k_scale_range (lo hi : Int) (hl0 : 0 ≤ lo) (hl1 : lo < 16) (hh0 : 0 ≤ hi) (hh1 : hi < 4) :
    -32 ≤ (lo + 16 * hi) - 32 ∧ (lo + 16 * hi) - 32 ≤ 31 := by
  omega

/-- Q6_K 6-bit code minus 32 lies in [-32, 31]. -/
theorem q6k_code_range (c : Int) (h0 : 0 ≤ c) (h1 : c < 64) : -32 ≤ c - 32 ∧ c - 32 ≤ 31 := by
  omega

/-- IQ4_XS combined scale: 4 low + 2 high bits minus 32 is in [-32, 31]. -/
theorem iq4xs_scale_range (lo hi : Int) (h0 : 0 ≤ lo) (h1 : lo < 16) (h2 : 0 ≤ hi) (h3 : hi < 4) :
    -32 ≤ (lo + 16 * hi) - 32 ∧ (lo + 16 * hi) - 32 ≤ 31 := by
  omega

-- § validation and weight sizes
/-- Column-count validation predicate. -/
def ncolsOk (n : Nat) : Prop := 1 ≤ n ∧ n ≤ MAX_NCOLS

/-- Multi-column path is used exactly for 2..8 columns. -/
theorem multi_iff (n : Nat) (h : ncolsOk n) : 1 < n ↔ 2 ≤ n ∧ n ≤ 8 := by
  unfold ncolsOk MAX_NCOLS at h
  omega

/-- Supported non-IQ types have a block entry; unsupported ids do not. -/
theorem blockInfo_unsupported : blockInfo 0 = none := by decide

/-- Q5_K weights: 256-element blocks, 176 bytes. -/
theorem blockInfo_q5k : blockInfo 13 = some (256, 176) := by decide

/-- Weight bytes factorise as row bytes times row count. -/
theorem weightBytes_row (e b nIn nOut : Nat) : weightBytes e b nIn nOut = ((nIn / e) * b) * nOut := by
  unfold weightBytes
  rfl

/-- Zero rows weigh nothing. -/
theorem weightBytes_zero_rows (e b nIn : Nat) : weightBytes e b nIn 0 = 0 := by
  simp [weightBytes]

/-- Adding rows adds bytes. -/
theorem weightBytes_add_rows (e b nIn r s : Nat) :
    weightBytes e b nIn (r + s) = weightBytes e b nIn r + weightBytes e b nIn s := by
  unfold weightBytes
  rw [Nat.mul_add]

end Strata.NativeMmvq
