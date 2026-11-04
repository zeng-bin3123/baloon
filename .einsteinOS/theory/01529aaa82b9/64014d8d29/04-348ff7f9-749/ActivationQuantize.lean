/-!
# MODULE: activation_quantize
SOURCES: src/kernels/cuda/quantize_act.cu, src/kernels/cuda/iq_kernels.cu
Integer semantics of the activation quantizers: clamp ranges, Q8_0 / Q8_1 / Q8_K block layouts, the
`nearest_int` magic-constant trick, bsums bounds and launch grids.

## Source semantics
* Q8_0 block = fp16 d + 32 int8 = 34 bytes; Q8_1 = half2 ds + 32 int8 = 36 bytes; Q8_K = f32 d + 256 int8 +
  16 int16 bsums = 292 bytes.
* `nearest_int(x)` adds 1.5*2^23, reads the mantissa bits and subtracts 0x400000; exact for |x| < 2^22.
* ggml's reference clamps to [-128, 127]; the engine's CPU-parity variant clamps to [-127, 127].
* The sum of 16 int8 values in [-127,127] fits an int16.

## Form template identifiers
quantize_q8_0_kernel -> quantize_q8_0_ggml_rule ; quantize_q8_0_scaled_kernel -> quantize_q8_0_cpu_rule
nearest_int_dev -> round_to_int_magic ; bsums -> block_partial_sums
-/

namespace Strata.ActQuant

-- § clamps

/-- Clamp an integer into `[lo, hi]`. -/
def clampI (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- Clamp result is within bounds. -/
theorem clampI_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clampI lo hi v ∧ clampI lo hi v ≤ hi := by
  simp only [clampI]
  split_ifs <;> omega

/-- Clamp is the identity inside the range. -/
theorem clampI_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clampI lo hi v = v := by
  simp only [clampI]
  split_ifs <;> omega

/-- The CPU-parity rule uses a symmetric range, so negation is closed. -/
theorem symmetric_range_closed (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : -127 ≤ -v ∧ -v ≤ 127 := by
  omega

-- § block layouts

/-- Q8_0 layout: 2 + 32 = 34; element `j < 32` is at byte `2 + j`. -/
theorem q8_0_layout (j : Nat) (h : j < 32) : 2 + 32 = 34 ∧ 2 + j < 34 := by omega

/-- Q8_1 layout: 4 + 32 = 36; element `j < 32` is at byte `4 + j`. -/
theorem q8_1_layout (j : Nat) (h : j < 32) : 4 + 32 = 36 ∧ 4 + j < 36 := by omega

/-- Q8_K layout: 4 + 256 + 32 = 292, bsums start at 260 (even) and end at 292. -/
theorem q8_k_layout (j : Nat) (h : j < 16) :
    4 + 256 + 32 = 292 ∧ 260 % 2 = 0 ∧ 4 + 256 + 2 * j + 2 ≤ 292 := by omega

/-- Block starts keep the alignment their fields need: Q8_K `d` is 4-aligned, Q8_0 `d` is 2-aligned. -/
theorem block_start_alignment (b : Nat) : (b * 292) % 4 = 0 ∧ (b * 34) % 2 = 0 := by omega

/-- Q8_1 element addressing: flat index `i` is block `i / 32` slot `i % 32` and is recoverable. -/
theorem q81_index (i : Nat) : i = (i / 32) * 32 + i % 32 ∧ i % 32 < 32 := by omega

/-- A buffer of `n` elements (`n % 32 = 0`) has `n / 32` blocks and element `i < n` is in range. -/
theorem q81_in_range (i n : Nat) (hn : n % 32 = 0) (hi : i < n) : i / 32 < n / 32 := by omega

-- § bsums

/-- Sixteen values of magnitude at most 127 sum to at most 2032, which fits an int16. -/
theorem bsums_fit : 16 * 127 = 2032 ∧ 2032 ≤ 32767 := by decide

-- § nearest_int magic constant

/-- Mantissa recovery: with `m = (4194304 + n) mod 2^23`, `m - 0x400000 = n` for `-2^22 ≤ n < 2^22`. -/
theorem nearest_int_magic (n : Int) (h1 : -4194304 ≤ n) (h2 : n < 4194304) :
    ((4194304 + n) % 8388608) - 4194304 = n := by
  omega

/-- All int8 quant values (|x| ≤ 127) are far inside the magic constant's exact range. -/
theorem nearest_int_range : (127 : Int) < 4194304 := by decide

/-- The Q8_K quantizer clamps only from above (`min 127`); values of the scaled product are at most 127. -/
theorem q8k_min_only (v : Int) (h : v ≤ 127) : min 127 v = v := by omega

-- § launch grids

/-- One thread per Q8_0 block, 128 threads per block. -/
theorem q8_0_grid (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by omega

/-- One thread per Q8_K block, 64 threads per block. -/
theorem q8_k_grid (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by omega

/-- One thread per element, 256 threads per block (Q8_1). -/
theorem q8_1_grid (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by omega

/-- Columns quantized as one vector keep every 32-block inside a column when `n_in % 32 = 0`. -/
theorem columns_as_vector (n_in c i : Nat) (h : n_in % 32 = 0) (hi : i < n_in) :
    (c * n_in + i) / 32 = c * (n_in / 32) + i / 32 := by
  obtain ⟨m, rfl⟩ : ∃ m, n_in = 32 * m := ⟨n_in / 32, by omega⟩
  have e : c * (32 * m) = 32 * (c * m) := by rw [Nat.mul_left_comm]
  rw [e, Nat.mul_add_div (by decide)]
  have : 32 * m / 32 = m := by omega
  rw [this]

end Strata.ActQuant
