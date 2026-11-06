/-!
# MODULE: activation_quantize_blocks
SOURCES: src/kernels/cuda/quantize_act.cu
Quantizes FP32 activations into ggml's block_q8_0 (34 bytes / 32 values), a CPU-matching scaled variant, and
block_q8_K (292 bytes / 256 values), plus the matching dequantizers.

## Source semantics
* block_q8_0 = fp16 d (2 B) + 32 int8; block_q8_K = f32 d + 256 int8 + 16 int16 bsums = 292 B.
* Q8_0 reference clamps to [-128,127]; the CPU-matching variant clamps to [-127,127].
* Q8_K clamps with min(127) only (iscale = -127/max keeps products inside [-127,127]).
* `nearest_int` keeps the low 23 bits of (x + 1.5*2^23) and subtracts 2^22.
* Zero blocks write d = 0 and zeroed quants (and bsums, a deliberate divergence).
* Launch grids are ceil(nb / threads) blocks of 128 (Q8_0) or 64 (Q8_K) threads.

## Form template identifiers
quantize_q8_0 -> quant_q80 ; quantize_q8_0_scaled -> quant_q80_scaled ; quantize_q8_K -> quant_q8k ;
nearest_int_dev -> magic_round ; dequant_q8_0 -> dequant_q80 ; dequant_q8_K -> dequant_q8k
-/

namespace Strata.QuantizeAct

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

-- § block layouts

/-- block_q8_0 layout: 2-byte scale followed by 32 int8. -/
theorem q80_block_layout : 2 + 32 = 34 := by
  decide

/-- block_q8_K layout: f32 scale, 256 int8, 16 int16 partial sums. -/
theorem q8K_block_layout : 4 + 256 + 2 * 16 = 292 := by
  decide

/-- Offset of bsums inside block_q8_K. -/
theorem q8K_bsums_offset : 4 + 256 = 260 := by
  decide

/-- Whole 32-value blocks cover a multiple-of-32 buffer. -/
theorem q80_blocks_cover (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by
  omega

/-- Whole 256-value blocks cover a multiple-of-256 buffer. -/
theorem q8K_blocks_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by
  omega

/-- bsums group j and inner index ii address values inside the 256-value block. -/
theorem bsum_index_lt (j ii : Nat) (hj : j < 16) (hi : ii < 16) : j * 16 + ii < 256 := by
  omega

-- § clamps

/-- CPU-matching clamp, symmetric. -/
def clampScaled (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Reference clamp used by the ggml-matching Q8_0 quantizer. -/
def clampRef (v : Int) : Int := if v > 127 then 127 else if v < -128 then -128 else v

/-- Q8_K clamp: min only. -/
def clampK (v : Int) : Int := if v < 127 then v else 127

/-- The scaled clamp lands in [-127, 127]. -/
theorem clampScaled_range (v : Int) : -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampScaled
  split_ifs <;> omega

/-- The reference clamp lands in [-128, 127]. -/
theorem clampRef_range (v : Int) : -128 ≤ clampRef v ∧ clampRef v ≤ 127 := by
  unfold clampRef
  split_ifs <;> omega

/-- The two clamps agree whenever the input is not below -127. -/
theorem clampScaled_eq_clampRef (v : Int) (h : -127 ≤ v) : clampScaled v = clampRef v := by
  unfold clampScaled clampRef
  split_ifs <;> omega

/-- The min-only clamp never exceeds 127. -/
theorem clampK_le (v : Int) : clampK v ≤ 127 := by
  unfold clampK
  split_ifs <;> omega

/-- Inside [-127,127] the Q8_K clamp is the identity, which is why no lower clamp is needed. -/
theorem clampK_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampK v = v := by
  unfold clampK
  split_ifs <;> omega

-- § magic-number rounding

/-- Integer part recovered by `nearest_int`: low 23 bits minus 2^22. -/
def magicLow (i : Nat) : Int := ((i &&& 0x007fffff : Nat) : Int) - 0x00400000

/-- The recovered integer lies in [-2^22, 2^22 - 1]. -/
theorem magicLow_range (i : Nat) : -4194304 ≤ magicLow i ∧ magicLow i ≤ 4194303 := by
  unfold magicLow
  have : i &&& 0x007fffff ≤ 0x007fffff := Nat.and_le_right
  omega

/-- Partial sums of k values in [-127,127] are bounded by 127*k. -/
theorem sumTo_bound (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → -127 ≤ f i ∧ f i ≤ 127) :
    -127 * (n : Int) ≤ sumTo f n ∧ sumTo f n ≤ 127 * (n : Int) := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    simp only [sumTo]
    omega

/-- A 16-value bsum of int8 quants fits int16 (|sum| <= 2032). -/
theorem bsum_fits_int16 (f : Nat → Int) (h : ∀ i, i < 16 → -127 ≤ f i ∧ f i ≤ 127) :
    -32768 ≤ sumTo f 16 ∧ sumTo f 16 ≤ 32767 := by
  have := sumTo_bound f 16 h
  omega

-- § byte-level packing

/-- Two scale bytes recombine into a 16-bit value below 2^16. -/
theorem scale_bytes_lt (lo hi : Nat) (hl : lo < 256) (hh : hi < 256) : lo + 256 * hi < 65536 := by
  omega

/-- Splitting a 16-bit pattern into bytes and recombining is exact. -/
theorem scale_bytes_roundtrip (v : Nat) (h : v < 65536) : v % 256 + 256 * (v / 256) = v ∧ v / 256 < 256 := by
  omega

/-- Storing an int8 as a byte and reading it back is the identity. -/
def toU8 (q : Int) : Nat := (q % 256).toNat

/-- Signed reinterpretation of a byte. -/
def toI8 (u : Nat) : Int := if u < 128 then (u : Int) else (u : Int) - 256

/-- int8 -> uint8 -> int8 round trip. -/
theorem int8_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) : toI8 (toU8 q) = q := by
  unfold toI8 toU8
  split_ifs <;> omega

-- § launch geometry

/-- Q8_0 grid (128 threads per block) covers all blocks. -/
theorem q80_grid_covers (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by
  omega

/-- Q8_K grid (64 threads per block) covers all blocks. -/
theorem q8K_grid_covers (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by
  omega

end Strata.QuantizeAct
