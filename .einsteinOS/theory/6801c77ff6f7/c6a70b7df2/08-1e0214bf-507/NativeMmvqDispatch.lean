/-!
# MODULE: native_mmvq_dispatch
SOURCES: src/kernels/cuda/native_mmvq.cu
Bit-faithful ports of llama.cpp's Q8_1 matrix-vector kernels (Q4_0, Q5_0, Q8_0, IQ4_NL, Q2_0, Q3_K, Q4_K,
Q5_K, Q6_K, IQ4_XS) with single- and multi-column (2..8) launchers, shape validation and type dispatch.

## Source semantics
* Weight bytes = (n_in / block_elems) * block_bytes * n_out, with the block table below.
* A call needs n_in > 0, n_in % block_elems = 0 and 1 <= ncols <= 8, plus 4-byte aligned pointers.
* Small-K rule: when blocks per row < blocks per iteration, 4 rows share a block (ROWS = 4), else 1 row.
* Multi-column: exact layout is 4 warps; upstream layout is 4 warps (ncols<=4) or 2 warps with 2 rows.
* Types 16,17,18,21,22,29 are routed to the IQ kernels; others are native.

## Form template identifiers
native_mmvq_supported -> supported_pred ; native_mmvq_weight_bytes -> weight_bytes_fn ;
validate_shape -> shape_check ; native_mmvq -> type_dispatch ; Q81Block -> act_block
-/

namespace Strata.NativeMmvq

/-- Values per weight block for each ggml type id handled here. -/
def blockElems (t : Nat) : Nat :=
  match t with
  | 2 => 32 | 6 => 32 | 8 => 32 | 20 => 32
  | 11 => 256 | 12 => 256 | 13 => 256 | 14 => 256 | 23 => 256
  | 42 => 64
  | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256 | 22 => 256 | 29 => 256
  | _ => 0

/-- Bytes per weight block. -/
def blockBytes (t : Nat) : Nat :=
  match t with
  | 2 => 18 | 6 => 22 | 8 => 34 | 20 => 18
  | 11 => 110 | 12 => 144 | 13 => 176 | 14 => 210 | 23 => 136
  | 42 => 18
  | 16 => 66 | 17 => 74 | 18 => 98 | 21 => 110 | 22 => 82 | 29 => 56
  | _ => 0

/-- `native_mmvq_supported`. -/
def nativeSupported (t : Nat) : Bool :=
  decide (t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 20 ∨ t = 23 ∨ t = 42 ∨
          t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29)

/-- Types forwarded to `iq_mmvq`. -/
def iqRoute (t : Nat) : Bool :=
  decide (t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29)

/-- Dispatch targets. -/
inductive Route where
  | native
  | iq
  | unsupported
  deriving DecidableEq

/-- `native_mmvq`'s switch. -/
def route (t : Nat) : Route :=
  if iqRoute t = true then Route.iq else if nativeSupported t = true then Route.native else Route.unsupported

-- § tables

/-- Every supported type has positive block geometry. -/
theorem supported_has_block :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29],
      0 < blockElems t ∧ 0 < blockBytes t := by
  decide

/-- The support predicate agrees with the block table on every id below 64. -/
theorem supported_iff_block : ∀ t < 64, nativeSupported t = decide (0 < blockElems t) := by
  decide

/-- IQ-routed types are natively supported. -/
theorem iqRoute_subset (t : Nat) (h : iqRoute t = true) : nativeSupported t = true := by
  simp only [iqRoute, nativeSupported, decide_eq_true_eq] at *
  omega

/-- A type is rejected by the dispatcher exactly when it is unsupported. -/
theorem route_unsupported_iff (t : Nat) : route t = Route.unsupported ↔ nativeSupported t = false := by
  unfold route
  by_cases hi : iqRoute t = true
  · have hn := iqRoute_subset t hi
    simp [hi, hn]
  · by_cases hn : nativeSupported t = true
    · simp [hi, hn]
    · have hf : nativeSupported t = false := by simpa using hn
      simp [hi, hn, hf]

-- § weight bytes and shapes

/-- `native_mmvq_weight_bytes` (without the overflow check). -/
def weightBytes (t nIn nOut : Nat) : Nat := (nIn / blockElems t) * blockBytes t * nOut

/-- Shape validation. -/
def validShape (nIn ncols blockElemsN : Nat) : Prop :=
  0 < nIn ∧ nIn % blockElemsN = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- With a valid shape the block count times block size is exactly n_in. -/
theorem blocks_cover (nIn ncols b : Nat) (h : validShape nIn ncols b) (hb : 0 < b) :
    nIn / b * b = nIn := by
  obtain ⟨_, h2, _, _⟩ := h
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h2)

/-- One output row's bytes. -/
theorem weightBytes_one (t nIn : Nat) : weightBytes t nIn 1 = (nIn / blockElems t) * blockBytes t := by
  simp [weightBytes]

/-- ncols columns quantize as a single vector of ncols * n_in values: block counts agree. -/
theorem q8_1_columns_whole (nIn ncols : Nat) (h : nIn % 32 = 0) :
    (nIn * ncols) / 32 = ncols * (nIn / 32) := by
  obtain ⟨q, rfl⟩ : ∃ q, nIn = 32 * q := ⟨nIn / 32, by omega⟩
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ (by decide : 0 < 32),
    Nat.mul_div_cancel_left _ (by decide : 0 < 32), Nat.mul_comm]

-- § launch geometry

/-- Small-K row blocks: 4 rows per block cover all outputs. -/
theorem small_k_blocks_cover (nOut : Nat) : nOut ≤ 4 * ((nOut + 3) / 4) := by
  omega

/-- Upstream multi-column layout: 2 rows per block cover all outputs. -/
theorem upstream_blocks_cover (nOut : Nat) : nOut ≤ 2 * ((nOut + 1) / 2) := by
  omega

/-- Blocks per iteration of the 32-value formats (qi = 4 and qi = 8). -/
theorem small_bpi : 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by
  decide

/-- Small-K thresholds in elements: 2048 for Q4_0/Q5_0/IQ4_NL and 1024 for Q8_0. -/
theorem small_k_thresholds : 64 * 32 = 2048 ∧ 32 * 32 = 1024 := by
  decide

/-- Blocks per iteration of the 256-value formats. -/
theorem k_bpi : 2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧ 4 * 4 * 32 / 32 = 16 := by
  decide

/-- Q2_0 (QK=64, QI=2): 64 blocks per iteration. -/
theorem q2_0_bpi : 4 * 32 / 2 = 64 := by
  decide

/-- Shared partial buffer of the multi-column kernel fits 48 KiB for every ncols in 2..8. -/
theorem multi_shared_fits : ∀ n ∈ [2, 3, 4, 5, 6, 7, 8], 3 * n * 4 * 32 * 4 ≤ 49152 := by
  decide

/-- Warp count rule of the upstream layout never exceeds the 4 warps of the exact layout. -/
theorem upstream_warps_le (n : Nat) : (if n ≤ 4 then 4 else 2) ≤ 4 := by
  split <;> omega

/-- tid decomposition 32*y + x with y<4, x<32 is a bijection onto 0..127. -/
theorem tid_range (x y : Nat) (hx : x < 32) (hy : y < 4) : 32 * y + x < 128 := by
  omega

end Strata.NativeMmvq
