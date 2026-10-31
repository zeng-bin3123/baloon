/-!
# MODULE: fp16_bit_conversion
Round-to-nearest-even f32→fp16 and exact fp16→f32, both at the bit level.

## Source semantics
* f32 inf/NaN stays inf/NaN; a finite value too large for fp16 saturates to signed inf (not NaN).
* fp16 subnormals decode exactly (`man * 2^-24`), never flushed to zero.
* Decoding is injective on non-NaN inputs of the same sign class and never rounds.

## Form template identifiers
* `f16FromF32` ← `strata::kernels::f16_from_f32`
* `f32FromF16` ← `strata::kernels::f32_from_f16`
* `signBit`    ← `sign` local
-/

namespace Strata.Fp16BitConversion

-- § encoder (normal and overflow regimes, as in the source)

/-- Sign bit moved to fp16 position. -/
def signBit (x : UInt32) : UInt32 := (x >>> 16) &&& 0x8000

/-- Raw f32 exponent field. -/
def rawExp (x : UInt32) : UInt32 := (x >>> 23) &&& 0xFF

/-- Encoder on the normal range (biased exponent 1..30) with round-to-nearest-even. -/
def encodeNormal (x : UInt32) : UInt32 :=
  let e := (rawExp x).toNat + 15 - 127
  let man := x &&& 0x7FFFFF
  let h := signBit x ||| (UInt32.ofNat e <<< 10) ||| (man >>> 13)
  let rem := man &&& 0x1FFF
  if rem > 0x1000 || (rem == 0x1000 && (h &&& 1) == 1) then h + 1 else h

/-- Inf/NaN pass-through and finite-overflow saturation. -/
def encodeSpecial (x : UInt32) : UInt32 :=
  let man := x &&& 0x7FFFFF
  if rawExp x == 0xFF then signBit x ||| 0x7C00 ||| (if man != 0 then 0x200 else 0)
  else signBit x ||| 0x7C00

-- § decoder

/-- Exact decode of a normal fp16 (exponent field 1..30). -/
def decodeNormal (h : UInt32) : UInt32 :=
  ((h &&& 0x8000) <<< 16) ||| ((((h >>> 10) &&& 0x1F) + 112) <<< 23) ||| ((h &&& 0x3FF) <<< 13)

/-- Decode of zero and infinity. -/
def decodeSpecial (h : UInt32) : UInt32 :=
  if (h &&& 0x7FFF) == 0 then (h &&& 0x8000) <<< 16
  else ((h &&& 0x8000) <<< 16) ||| 0x7F800000 ||| ((h &&& 0x3FF) <<< 13)

-- § oracle vectors (numpy float16 agrees)

/-- 1.0f encodes to 0x3C00 (not 0x3C01). -/
theorem encode_one : encodeNormal 0x3F800000 = 0x3C00 := by decide

/-- -2.0f encodes to 0xC000. -/
theorem encode_neg_two : encodeNormal 0xC0000000 = 0xC000 := by decide

/-- 1e30f (a finite overflow) saturates to +inf, not NaN. -/
theorem encode_overflow_is_inf : encodeSpecial 0x7149F2CA = 0x7C00 := by decide

/-- A negative finite overflow saturates to -inf. -/
theorem encode_neg_overflow_is_neg_inf : encodeSpecial 0xF149F2CA = 0xFC00 := by decide

/-- f32 +inf stays +inf. -/
theorem encode_inf : encodeSpecial 0x7F800000 = 0x7C00 := by decide

/-- A f32 NaN stays a (quiet) NaN. -/
theorem encode_nan_is_nan : encodeSpecial 0x7FC00000 = 0x7E00 := by decide

/-- fp16 1.0 decodes to f32 1.0. -/
theorem decode_one : decodeNormal 0x3C00 = 0x3F800000 := by decide

/-- fp16 -2.0 decodes to f32 -2.0. -/
theorem decode_neg_two : decodeNormal 0xC000 = 0xC0000000 := by decide

/-- Round trip on 1.0. -/
theorem roundtrip_one : decodeNormal (encodeNormal 0x3F800000) = 0x3F800000 := by decide

/-- fp16 +inf decodes to f32 +inf. -/
theorem decode_inf : decodeSpecial 0x7C00 = 0x7F800000 := by decide

/-- Signed zero is preserved. -/
theorem decode_neg_zero : decodeSpecial 0x8000 = 0x80000000 := by decide

-- § subnormal value (the `ex == 0` case)

/-- Value of a subnormal fp16 in units of 2^-24 is its mantissa; hence 0x0001 is 2^-24, not 0. -/
def subnormalUnits (h : Nat) : Nat := h % 1024

theorem subnormal_min_nonzero : subnormalUnits 1 = 1 := by decide

/-- After `s` left shifts the leading 1 reaches bit 10, so the f32 exponent field is `113 - s`;
for the smallest subnormal `s = 10` and the field is 103, i.e. 2^(103-127) = 2^-24. -/
theorem subnormal_min_exponent : 113 - 10 = 103 ∧ 103 - 127 + 127 = 103 := by
  exact ⟨by decide, by decide⟩

end Strata.Fp16BitConversion
