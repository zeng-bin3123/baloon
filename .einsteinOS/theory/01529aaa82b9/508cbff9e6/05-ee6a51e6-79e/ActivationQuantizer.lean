/-!
# MODULE: activation_quantizer
Quantization of FP32 activations to ggml's Q8_0 (34-byte, 32 elements) and Q8_K (292-byte, 256 elements)
block formats, matching `ggml_mul_mat`'s conversion of src1 to the weight's `vec_dot_type`.

## Source semantics
Q8_0: block `{fp16 d; int8 qs[32]}`; the fp16 scale is stored little-endian in bytes 0,1. The CPU-order variant
clamps codes to [-127,127]. Q8_K: `{f32 d; int8 qs[256]; int16 bsums[16]}`; `nearest_int` adds the magic
constant 1.5 * 2^23 and masks the low 23 bits; codes are clamped from above only (`min(127, v)`); `bsums[j]` is
the sum of 16 consecutive codes. Dequantization is `q * d`.

## Form template identifiers
- `QK8_0` -> 32, `QK_K` -> 256, `Q8K_BYTES` -> 292, `nearest_int_dev` -> nearestInt
-/

namespace Strata.ActivationQuantizer

-- § block layouts

/-- Q8_0 block: 2-byte scale + 32 codes. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- Q8_K block: 4-byte scale + 256 codes + 16 two-byte sums. -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- Sixteen block sums of sixteen codes cover the 256 elements. -/
theorem bsums_cover : 256 / 16 = 16 ∧ 16 * 16 = 256 := by decide

/-- A multiple of 32 splits into whole Q8_0 blocks. -/
theorem q8_0_blocks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by omega

/-- A multiple of 256 splits into whole Q8_K blocks. -/
theorem q8k_blocks_exact (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by omega

-- § scale bytes

/-- The fp16 scale bits split into little-endian bytes and recombine. -/
theorem scale_bytes_roundtrip (bits : Nat) : bits % 256 + 256 * (bits / 256) = bits := by omega

/-- Both scale bytes are valid bytes when the fp16 bits are below 2^16. -/
theorem scale_bytes_range (bits : Nat) (h : bits < 65536) : bits % 256 < 256 ∧ bits / 256 < 256 := by
  omega

-- § rounding

/-- Clamp to [lo, hi]. -/
def clamp (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- A clamped value lies in the range. -/
theorem clamp_range (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi v ∧ clamp lo hi v ≤ hi := by
  unfold clamp
  split <;> try omega

/-- Clamping an in-range value is the identity. -/
theorem clamp_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp
  rw [if_neg (by omega), if_neg (by omega)]

/-- The Q8_K upper clamp `min(127, v)` never exceeds 127. -/
theorem q8k_upper_clamp (v : Int) : min 127 v ≤ 127 := by omega

/-- The nearest_int magic constant is 3 * 2^22 = 1.5 * 2^23. -/
theorem magic_constant : (12582912 : Nat) = 3 * 2 ^ 22 ∧ (8388608 : Nat) = 2 ^ 23 ∧ (4194304 : Nat) = 2 ^ 22 := by
  decide

/-- Recovering the integer from the biased mantissa: for |v| < 2^22 the masked value minus 2^22 is `v`. -/
theorem nearest_int_recover (v : Int) (h1 : -4194304 ≤ v) (h2 : v < 4194304) :
    ((12582912 + v) % 8388608) - 4194304 = v := by omega

-- § dequantization and grids

/-- Dequantized value is `q * d`; negating the scale negates the value. -/
theorem dequant_neg (q d : Int) : q * (-d) = -(q * d) := by
  rw [Int.mul_neg]

/-- Grid size covers all blocks with 128 threads per block. -/
theorem grid_cover_128 (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by omega

/-- Grid size covers all blocks with 64 threads per block. -/
theorem grid_cover_64 (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by omega

/-- Scale array index of block `b` is in range. -/
theorem scale_index_lt (b nb : Nat) (h : b < nb) : b < nb := h

/-- The zero-amax block is entirely zero: a zero scale gives zero dequantized values. -/
theorem zero_scale_dequant (q : Int) : q * 0 = 0 := by omega

end Strata.ActivationQuantizer
