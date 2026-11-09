/-!
# MODULE: native_mmvq_kernels
SOURCES: src/kernels/cuda/native_mmvq.cu
llama.cpp-pinned Q8_1 quantizer and MMVQ kernels (Q4_0/Q5_0/Q8_0/IQ4_NL/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/Q2_0) for
ncols = 1..8, plus the dispatcher, support predicate and weight-size computation.

## Source semantics
Each launch uses 4 warps x 32 lanes = 128 threads. Thread `tid` starts at weight block `tid / T` and strides
by BPI blocks, with BPI * T = 128. Small-K shapes use 4 rows per CUDA block, otherwise 1. Q8_1 activations are
36-byte blocks of 32 values; ncols columns are contiguous so quantizing ncols*n_in values never straddles a
column. Shapes require n_in divisible by the block size and 1 <= ncols <= 8.

## Form template identifiers
bpi/tt -> "blocks per iteration / threads per block"; validShape -> "shape contract";
weightBytes -> "weight byte count".
-/
namespace Strata.NativeMmvq

-- § geometry
/-- Values per weight block for supported types. -/
def blockElems (t : Nat) : Nat :=
  if t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 20 then 32 else if t = 42 then 64 else
  if t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 23 ∨ t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29 then 256
  else 0

/-- Bytes per weight block. -/
def blockBytes (t : Nat) : Nat :=
  if t = 2 then 18 else if t = 6 then 22 else if t = 8 then 34 else if t = 20 then 18
  else if t = 11 then 110 else if t = 12 then 144 else if t = 13 then 176 else if t = 14 then 210
  else if t = 23 then 136 else if t = 42 then 18 else if t = 16 then 66 else if t = 17 then 74
  else if t = 18 then 98 else if t = 21 then 110 else if t = 22 then 82 else if t = 29 then 56 else 0

/-- `native_mmvq_supported`. -/
def supported (t : Nat) : Bool :=
  t == 2 || t == 6 || t == 8 || t == 11 || t == 12 || t == 13 || t == 14 || t == 20 || t == 23 ||
  t == 42 || t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- `native_mmvq_weight_bytes` without the validation. -/
def weightBytes (t nIn nOut : Nat) : Nat := nIn / blockElems t * blockBytes t * nOut

/-- Every supported type has a non-empty block geometry. -/
theorem supported_has_geometry (t : Nat) (ht : t < 64) (h : supported t = true) :
    0 < blockElems t ∧ 0 < blockBytes t := by
  revert t
  decide

/-- Weight bytes scale linearly in the number of output rows. -/
theorem weightBytes_add_rows (t nIn a b : Nat) :
    weightBytes t nIn (a + b) = weightBytes t nIn a + weightBytes t nIn b := by
  unfold weightBytes
  rw [Nat.mul_add]

-- § shape contract
/-- `validate_shape` as a decidable predicate. -/
def validShape (nIn ncols e : Nat) : Bool :=
  decide (0 < nIn ∧ nIn % e = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8)

/-- `native_q8_1_bytes`: 36-byte blocks of 32 values per column. -/
def q8Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

/-- A valid shape has a non-empty Q8_1 image. -/
theorem q8Bytes_pos (nIn ncols : Nat) (h : validShape nIn ncols 32 = true) : 0 < q8Bytes nIn ncols := by
  unfold validShape at h
  simp at h
  unfold q8Bytes
  have h1 : 0 < nIn / 32 := by omega
  have h2 : 0 < ncols * (nIn / 32) := Nat.mul_pos (by omega) h1
  omega

/-- Quantizing ncols columns as one vector never splits a 32-block across columns. -/
theorem q8_block_within_column (m i : Nat) : i / 32 / m = i / (32 * m) := by
  exact Nat.div_div_eq_div_mul i 32 m

/-- Rounding the quantizer grid up to 256 threads covers every element. -/
theorem quant_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § iteration traits
/-- Blocks per iteration of the ncols = 1 loop. -/
def bpi (t : Nat) : Nat :=
  if t = 13 ∨ t = 12 ∨ t = 11 then 8 else if t = 14 then 4 else if t = 23 then 16
  else if t = 42 then 64 else if t = 8 then 32 else 64

/-- Lanes cooperating on one weight block (`T`). -/
def tt (t : Nat) : Nat :=
  if t = 13 ∨ t = 12 ∨ t = 11 then 16 else if t = 14 then 32 else if t = 23 then 8
  else if t = 42 then 2 else if t = 8 then 4 else 2

/-- BPI * T equals the 128 threads of a CUDA block for every multi-warp format. -/
theorem bpi_mul_tt (t : Nat) (ht : t < 64)
    (h : t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 20 ∨ t = 23 ∨ t = 42) :
    bpi t * tt t = 128 := by
  revert t
  decide

/-- Hence the starting block `tid / T` is always below the stride BPI: threads never alias a later iteration. -/
theorem start_block_lt_stride (B T tid : Nat) (h : B * T = 128) (ht : tid < 128) : tid / T < B := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  omega

/-- Small-K launches (4 rows per block) cover all rows. -/
theorem grid_small_covers (n : Nat) : n ≤ 4 * ((n + 3) / 4) ∧ 4 * ((n + 3) / 4) < n + 4 := by
  omega

/-- Upstream multi-column launches (2 rows per block) cover all rows. -/
theorem grid_upstream_covers (n : Nat) : n ≤ 2 * ((n + 1) / 2) ∧ 2 * ((n + 1) / 2) < n + 2 := by
  omega

/-- The shared cross-warp partial buffer fits 48 KiB for every ncols <= 8 (exact layout, 4 rows). -/
theorem shared_partial_fits (nc : Nat) (h : nc ≤ 8) : (4 - 1) * nc * 4 * 32 * 4 ≤ 49152 := by
  omega

-- § packed dot algebra
/-- `__dp4a` without accumulator. -/
def dp4 (a b : Fin 4 → Int) : Int := a 0 * b 0 + a 1 * b 1 + a 2 * b 2 + a 3 * b 3

/-- Affine Q4_0/Q5_0 correction: (v - k) . u = v . u - k * sum u. -/
theorem dp4_affine_correction (a b : Fin 4 → Int) (k : Int) :
    dp4 (fun i => a i - k) b = dp4 a b - k * (b 0 + b 1 + b 2 + b 3) := by
  unfold dp4
  simp only [Int.sub_mul, Int.mul_add]
  omega

end Strata.NativeMmvq
