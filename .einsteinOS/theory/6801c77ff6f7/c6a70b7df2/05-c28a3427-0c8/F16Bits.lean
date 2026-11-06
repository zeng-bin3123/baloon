/-!
# MODULE: f16_bits
SOURCES: include/strata/kernels/f16_bits.hpp
A bit-level model of `f16_from_f32` (round-to-nearest-even, saturating) and `f32_from_f16` (exact, subnormals
normalised), over natural numbers holding the raw bit patterns, with the regression vectors the header's tests use.

## Source semantics
* `f16_from_f32`: an f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value too large for fp16 saturates to inf
  (0x7C00 with the sign); tiny values underflow to a signed zero; subnormal results restore the implicit 1 and
  round at a moved point; normal results round from 13 surviving mantissa bits, and a rounding carry may bump the
  exponent (65520 → inf).
* `f32_from_f16` never rounds: subnormals are normalised by shifting the leading 1 to bit 10, giving f32 exponent
  field `113 - s`; it does not flush subnormals to zero.
* All claims here are on concrete bit vectors plus structural facts about the normalisation loop; the exhaustive
  65,536-input check is the header's numpy oracle test, not repeated here.

## Form template identifiers
f16FromF32Bits → fp16_encode_rne, f32FromF16Bits → fp16_decode_exact, normShift → subnormal_normalize,
regression vectors → fp16_oracle_vectors.
-/

namespace Strata.Kernels.F16Bits

-- § the model

/-- Round-to-nearest-even f32 → fp16 on raw bits (`x` is the f32 bit pattern; the result the fp16 bits). -/
def f16FromF32Bits (x : Nat) : Nat :=
  let sign := (x >>> 16) &&& 0x8000
  let rawexp := (x >>> 23) &&& 0xFF
  let man := x &&& 0x7FFFFF
  if rawexp = 0xFF then
    sign ||| 0x7C00 ||| (if man ≠ 0 then 0x200 else 0)
  else
    let e : Int := (rawexp : Int) - 127 + 15
    if e ≥ 31 then
      sign ||| 0x7C00
    else if e ≤ 0 then
      if e < -10 then
        sign
      else
        let man' := man ||| 0x800000
        let sh := (14 - e).toNat
        let h := (man' >>> sh) &&& 0x3FF
        let rem := man' &&& ((1 <<< sh) - 1)
        let half := 1 <<< (sh - 1)
        let h' := if rem > half ∨ (rem = half ∧ (h &&& 1) = 1) then h + 1 else h
        sign ||| h'
    else
      let h := sign ||| (e.toNat <<< 10) ||| (man >>> 13)
      let rem := man &&& 0x1FFF
      if rem > 0x1000 ∨ (rem = 0x1000 ∧ (h &&& 1) = 1) then h + 1 else h

/-- The subnormal normalisation loop: shift left until bit 10 is set; returns (mantissa, shift count). -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | f + 1, m, s => if (m &&& 0x400) = 0 then normShift f (m <<< 1) (s + 1) else (m, s)

/-- Exact fp16 → f32 on raw bits. -/
def f32FromF16Bits (h : Nat) : Nat :=
  let sign := (h &&& 0x8000) <<< 16
  let ex := (h >>> 10) &&& 0x1F
  let man := h &&& 0x3FF
  if ex = 0 then
    if man = 0 then
      sign
    else
      let r := normShift 10 man 0
      sign ||| ((113 - r.2) <<< 23) ||| ((r.1 &&& 0x3FF) <<< 13)
  else if ex = 31 then
    sign ||| 0x7F800000 ||| (man <<< 13)
  else
    sign ||| ((ex + 112) <<< 23) ||| (man <<< 13)

-- § structural facts about the normalisation loop

/-- The shift count grows by at most one per fuel step. -/
theorem normShift_snd_le (f : Nat) : ∀ m s : Nat, (normShift f m s).2 ≤ s + f := by
  induction f with
  | zero =>
    intro m s
    show s ≤ s + 0
    omega
  | succ f ih =>
    intro m s
    simp only [normShift]
    split
    · have := ih (m <<< 1) (s + 1)
      omega
    · show s ≤ s + (f + 1)
      omega

/-- With the header's fuel of 10 the shift count is at most 10, so the exponent field `113 - s` is in 103..113. -/
theorem decode_subnormal_exponent_range (m : Nat) :
    103 ≤ 113 - (normShift 10 m 0).2 ∧ 113 - (normShift 10 m 0).2 ≤ 113 := by
  have h := normShift_snd_le 10 m 0
  omega

-- § encode: regression vectors (every regime)

/-- 1.0f encodes to 0x3C00 (the value an earlier `__float2half` got wrong as 0x3C01). -/
theorem enc_one : f16FromF32Bits 0x3F800000 = 0x3C00 := by
  decide

/-- +inf stays +inf. -/
theorem enc_inf : f16FromF32Bits 0x7F800000 = 0x7C00 := by
  decide

/-- -inf stays -inf (sign kept). -/
theorem enc_neg_inf : f16FromF32Bits 0xFF800000 = 0xFC00 := by
  decide

/-- A quiet NaN stays a NaN (0x7E00). -/
theorem enc_nan : f16FromF32Bits 0x7FC00000 = 0x7E00 := by
  decide

/-- 65536.0 is finite but too large: it saturates to inf (the conflated version returned a NaN here). -/
theorem enc_overflow_65536 : f16FromF32Bits 0x47800000 = 0x7C00 := by
  decide

/-- 1e30 saturates to inf, not NaN. -/
theorem enc_overflow_1e30 : f16FromF32Bits 0x7149F2CA = 0x7C00 := by
  decide

/-- The largest fp16 value, 65504, is exact. -/
theorem enc_max_half : f16FromF32Bits 0x477FE000 = 0x7BFF := by
  decide

/-- 65520 is a tie that rounds to even upward, and the carry bumps the exponent into inf. -/
theorem enc_carry_to_inf : f16FromF32Bits 0x477FF000 = 0x7C00 := by
  decide

/-- 2^-24, the smallest fp16 subnormal, is exact. -/
theorem enc_min_subnormal : f16FromF32Bits 0x33800000 = 0x0001 := by
  decide

/-- 2^-25 is an exact tie between 0 and 2^-24 and rounds to even, i.e. to zero. -/
theorem enc_tie_to_even : f16FromF32Bits 0x33000000 = 0x0000 := by
  decide

/-- 1.5 * 2^-25 is above the tie and rounds up to 2^-24. -/
theorem enc_above_tie : f16FromF32Bits 0x33400000 = 0x0001 := by
  decide

/-- 2^-26 underflows to zero. -/
theorem enc_underflow : f16FromF32Bits 0x32800000 = 0x0000 := by
  decide

/-- -0.0 keeps its sign. -/
theorem enc_neg_zero : f16FromF32Bits 0x80000000 = 0x8000 := by
  decide

-- § decode: regression vectors

/-- The smallest subnormal decodes to 2^-24, not zero (flushing would be a 100% error). -/
theorem dec_min_subnormal : f32FromF16Bits 0x0001 = 0x33800000 := by
  decide

/-- The largest subnormal 0x03FF decodes exactly. -/
theorem dec_max_subnormal : f32FromF16Bits 0x03FF = 0x387FC000 := by
  decide

/-- The smallest normal 2^-14. -/
theorem dec_min_normal : f32FromF16Bits 0x0400 = 0x38800000 := by
  decide

/-- 1.0. -/
theorem dec_one : f32FromF16Bits 0x3C00 = 0x3F800000 := by
  decide

/-- +inf. -/
theorem dec_inf : f32FromF16Bits 0x7C00 = 0x7F800000 := by
  decide

/-- Signed zero. -/
theorem dec_neg_zero : f32FromF16Bits 0x8000 = 0x80000000 := by
  decide

-- § round trips on representative fp16 values

/-- Decoding then encoding returns the same bits for a subnormal. -/
theorem roundtrip_min_subnormal : f16FromF32Bits (f32FromF16Bits 0x0001) = 0x0001 := by
  decide

/-- Round trip for the largest subnormal. -/
theorem roundtrip_max_subnormal : f16FromF32Bits (f32FromF16Bits 0x03FF) = 0x03FF := by
  decide

/-- Round trip for a negative subnormal (sign preserved). -/
theorem roundtrip_neg_subnormal : f16FromF32Bits (f32FromF16Bits 0x8001) = 0x8001 := by
  decide

/-- Round trip for 1.0. -/
theorem roundtrip_one : f16FromF32Bits (f32FromF16Bits 0x3C00) = 0x3C00 := by
  decide

/-- Round trip for the largest finite fp16. -/
theorem roundtrip_max_half : f16FromF32Bits (f32FromF16Bits 0x7BFF) = 0x7BFF := by
  decide

/-- Round trip for the smallest normal. -/
theorem roundtrip_min_normal : f16FromF32Bits (f32FromF16Bits 0x0400) = 0x0400 := by
  decide

end Strata.Kernels.F16Bits
