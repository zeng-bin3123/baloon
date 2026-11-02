/-!
# MODULE: quantize_activation
Activation quantizers matching ggml: Q8_0 (34-byte blocks, fp16 scale), Q8_0 with fp32 scales (CPU-matching rule,
round half away from zero), and Q8_K (292-byte blocks). Also the dequantizers.

## Source semantics
Q8_0: d32 = amax/127; stored scale is fp16(d32); integers divide by d32. Scaled variant clamps to [-127,127].
Q8_K: iscale = -127/max (signed value at largest magnitude); only an upper clamp at 127; bsums per 16 elements.

## Form template identifiers
quantize_q8_0 -> act_quant_block32; quantize_q8_K -> act_quant_block256
-/
namespace Strata.QuantizeAct

/-- A Q8_0 block is a 2-byte scale plus 32 int8. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- A Q8_K block is f32 d, 256 int8, and 16 int16 partial sums. -/
theorem q8_K_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- Q8_K has 16 partial sums of 16 elements each. -/
theorem q8_K_bsums : 256 / 16 = 16 := by decide

/-- Consecutive Q8_0 blocks are 34 bytes apart. -/
theorem q8_0_stride (b : Nat) : (b + 1) * 34 = b * 34 + 34 := by omega

/-- Consecutive Q8_K blocks are 292 bytes apart. -/
theorem q8_K_stride (b : Nat) : (b + 1) * 292 = b * 292 + 292 := by omega

/-- Symmetric clamp used by the scaled quantizer lands in [-127, 127]. -/
theorem clamp_range (v : Int) : -127 ≤ max (-127) (min 127 v) ∧ max (-127) (min 127 v) ≤ 127 := by omega

/-- Q8_K upper-only clamp never exceeds 127. -/
theorem q8K_upper (v : Int) : min 127 v ≤ 127 := by omega

/-- The nearest-int magic constant 12582912 is 1.5 * 2^23. -/
theorem magic_constant : 12582912 = 3 * 4194304 ∧ 4194304 = 2 ^ 22 := by decide

/-- Grid sizing covers every block. -/
theorem grid_cover (nb t : Nat) (h : 0 < t) : nb ≤ ((nb + t - 1) / t) * t := by
  have := Nat.div_add_mod (nb + t - 1) t
  have h2 := Nat.mod_lt (nb + t - 1) h
  rw [Nat.mul_comm] at this
  omega

/-- Block count for 2560 elements in Q8_0 is 80. -/
theorem blocks_2560 : 2560 / 32 = 80 := by decide

/-- Block count for 2560 elements in Q8_K is 10. -/
theorem blocks_2560_K : 2560 / 256 = 10 := by decide

/-- Dequantization of a zero scale is zero. -/
theorem dequant_zero_scale (q : Int) : q * 0 = 0 := by omega

end Strata.QuantizeAct
