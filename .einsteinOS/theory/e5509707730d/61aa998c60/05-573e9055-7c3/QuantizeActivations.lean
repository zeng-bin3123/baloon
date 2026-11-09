/-!
# MODULE: quantize_activations
SOURCES: src/kernels/cuda/quantize_act.cu
Activation quantizers matching ggml's src1 conversion: Q8_0 (fp16 scale), Q8_0 with an extra fp32 scale array
and CPU rounding rule, and Q8_K (fp32 scale, 256 values, 16-element block sums), plus dequantizers.

## Source semantics
- Q8_0 block = 34 bytes (fp16 d, 32 int8); Q8_K block = 292 bytes (f32 d, 256 int8, 16 int16 bsums).
- `n` must be a multiple of the block size. Output integers are clamped; the scaled variant clamps to
  [-127, 127]; Q8_K clamps only above (min 127).
- ggml `nearest_int` adds 1.5*2^23 and masks the mantissa: the recovered value lies in [-2^22, 2^22).
- An all-zero block has d = 0 and all codes 0.

## Form template identifiers
- QK8_0 -> 32 ; QK_K -> 256 ; Q8K_BYTES -> 292 ; nearest_int_dev -> nearestInt
-/

namespace Strata.QuantizeAct

/-- Clamp an integer to `[lo, hi]`. -/
def clamp (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- Clamp result is within bounds when `lo ≤ hi`. -/
theorem clamp_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi v ∧ clamp lo hi v ≤ hi := by
  unfold clamp
  split_ifs <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp_id (lo hi v : Int) (h0 : lo ≤ v) (h1 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp
  split_ifs <;> omega

/-- Clamp is idempotent. -/
theorem clamp_idem (lo hi v : Int) (h : lo ≤ hi) : clamp lo hi (clamp lo hi v) = clamp lo hi v := by
  have := clamp_bounds lo hi v h
  exact clamp_id lo hi _ this.1 this.2

/-- The scaled quantizer's code range is symmetric: [-127, 127]. -/
theorem scaled_code_symmetric (v : Int) : -127 ≤ clamp (-127) 127 v ∧ clamp (-127) 127 v ≤ 127 := by
  exact clamp_bounds (-127) 127 v (by omega)

-- § block sizes
/-- Q8_0 block: 2 + 32 = 34 bytes. -/
theorem q8_0_bytes : 2 + 32 = 34 := by decide
/-- Q8_K block: 4 + 256 + 16*2 = 292 bytes. -/
theorem q8k_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- Blocks of an `n`-element vector for block size 32, when `n` divides evenly. -/
theorem q8_0_blocks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

/-- Blocks of an `n`-element vector for block size 256, when `n` divides evenly. -/
theorem q8k_blocks_exact (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by
  omega

/-- Total Q8_0 bytes of `n` elements. -/
def q8_0Total (n : Nat) : Nat := (n / 32) * 34

/-- Total Q8_K bytes of `n` elements. -/
def q8kTotal (n : Nat) : Nat := (n / 256) * 292

/-- Q8_0 total is additive over multiples of 32. -/
theorem q8_0Total_add (a b : Nat) (ha : a % 32 = 0) : q8_0Total (a + b) = q8_0Total a + q8_0Total b := by
  unfold q8_0Total
  have : (a + b) / 32 = a / 32 + b / 32 := by omega
  rw [this, Nat.add_mul]

/-- Q8_K total is additive over multiples of 256. -/
theorem q8kTotal_add (a b : Nat) (ha : a % 256 = 0) : q8kTotal (a + b) = q8kTotal a + q8kTotal b := by
  unfold q8kTotal
  have : (a + b) / 256 = a / 256 + b / 256 := by omega
  rw [this, Nat.add_mul]

-- § nearest_int
/-- Recovered integer of ggml's magic-number rounding from the low 23 mantissa bits. -/
def nearestInt (bits : Nat) : Int := ((bits % 8388608 : Nat) : Int) - 4194304

/-- The recovered value lies in [-2^22, 2^22). -/
theorem nearestInt_range (bits : Nat) : -4194304 ≤ nearestInt bits ∧ nearestInt bits < 4194304 := by
  unfold nearestInt
  have h : bits % 8388608 < 8388608 := Nat.mod_lt _ (by decide)
  omega

-- § block sums
/-- Sum of sixteen int8-range values fits in int16 (|sum| ≤ 16*128 = 2048). -/
theorem bsums_fit_int16
    (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int)
    (h0 : -128 ≤ a0 ∧ a0 ≤ 127) (h1 : -128 ≤ a1 ∧ a1 ≤ 127) (h2 : -128 ≤ a2 ∧ a2 ≤ 127)
    (h3 : -128 ≤ a3 ∧ a3 ≤ 127) (h4 : -128 ≤ a4 ∧ a4 ≤ 127) (h5 : -128 ≤ a5 ∧ a5 ≤ 127)
    (h6 : -128 ≤ a6 ∧ a6 ≤ 127) (h7 : -128 ≤ a7 ∧ a7 ≤ 127) (h8 : -128 ≤ a8 ∧ a8 ≤ 127)
    (h9 : -128 ≤ a9 ∧ a9 ≤ 127) (h10 : -128 ≤ a10 ∧ a10 ≤ 127) (h11 : -128 ≤ a11 ∧ a11 ≤ 127)
    (h12 : -128 ≤ a12 ∧ a12 ≤ 127) (h13 : -128 ≤ a13 ∧ a13 ≤ 127) (h14 : -128 ≤ a14 ∧ a14 ≤ 127)
    (h15 : -128 ≤ a15 ∧ a15 ≤ 127) :
    -32768 ≤ a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 ∧
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 ≤ 32767 := by
  omega

/-- Block sum index `j*16 + ii` stays inside the 256-element block. -/
theorem bsum_index_in_block (j ii : Nat) (hj : j < 16) (hi : ii < 16) : j * 16 + ii < 256 := by
  omega

/-- Thread launch: one thread per Q8_K block, 64 threads per CUDA block, grid covers all blocks. -/
theorem q8k_grid_covers (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by
  omega

/-- Thread launch for Q8_0: 128 threads per CUDA block cover all blocks. -/
theorem q8_0_grid_covers (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by
  omega

/-- The fp32 scales array has exactly one entry per 32-element chunk. -/
theorem scales_per_chunk (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by
  omega

end Strata.QuantizeAct
