/-!
# MODULE: f16_bit_conversion
Round-to-nearest-even f32 -> fp16 bit conversion and exact fp16 -> f32 widening, written as explicit bit arithmetic.

## Source semantics
* f32 inf/NaN (raw exponent 255) stays inf/NaN; a FINITE value too large for fp16 saturates to inf (0x7C00).
* Subnormal results restore the implicit 1 and shift right by `14 - exp`, so the rounding point moves.
* Normal results round from 13 surviving mantissa bits with a sticky remainder; a carry may bump the exponent.
* fp16 -> f32 is exact, including subnormals (f32 exponent field `113 - s`).

## Form template identifiers
* f16_from_f32 -> `narrow_to_half`
* f32_from_f16 -> `widen_from_half`
* man / rem / sh -> `mantissa`, `remainder`, `shift`
-/

namespace Strata.F16

-- § Rounding

/-- Round-up decision of round-to-nearest-even: strictly above half, or exactly half with odd result. -/
def roundUp (rem half h : Nat) : Bool := decide (half < rem ∨ (rem = half ∧ h % 2 = 1))

/-- Below half never rounds up. -/
theorem roundUp_below (rem half h : Nat) (hr : rem < half) : roundUp rem half h = false := by
  unfold roundUp
  simp only [decide_eq_false_iff_not]
  omega

/-- Above half always rounds up. -/
theorem roundUp_above (rem half h : Nat) (hr : half < rem) : roundUp rem half h = true := by
  unfold roundUp
  simp only [decide_eq_true_eq]
  omega

/-- An exact tie with an even result stays. -/
theorem roundUp_tie_even (half h : Nat) (he : h % 2 = 0) : roundUp half half h = false := by
  unfold roundUp
  simp only [decide_eq_false_iff_not]
  omega

/-- An exact tie with an odd result goes up (to even). -/
theorem roundUp_tie_odd (half h : Nat) (ho : h % 2 = 1) : roundUp half half h = true := by
  unfold roundUp
  simp only [decide_eq_true_eq]
  exact Or.inr ⟨trivial, ho⟩

/-- The carry of an all-ones mantissa bumps the exponent field exactly by one. -/
theorem mantissa_carry_bumps_exponent (e : Nat) : e * 1024 + 1023 + 1 = (e + 1) * 1024 := by
  omega

/-- A carry out of the largest finite exponent (30) lands exactly on infinity, 0x7C00. -/
theorem carry_to_infinity : (30 + 1) * 1024 = 0x7C00 := by decide

-- § Classification

/-- The re-biased exponent `rawexp - 127 + 15`. -/
def rebias (rawexp : Int) : Int := rawexp - 127 + 15

/-- A finite f32 (raw exponent <= 254) overflows fp16 exactly when raw exponent >= 143. -/
theorem overflow_iff (rawexp : Int) (h0 : 0 ≤ rawexp) (h1 : rawexp ≤ 254) :
    31 ≤ rebias rawexp ↔ 143 ≤ rawexp := by
  unfold rebias
  omega

/-- Infinity as fp16 bits has exponent field 31 and zero mantissa. -/
theorem inf_bits : (0x7C00 >>> 10) &&& 0x1F = 31 ∧ 0x7C00 &&& 0x3FF = 0 := by decide

/-- The quiet-NaN pattern keeps a nonzero mantissa, so a NaN never collapses into infinity. -/
theorem nan_bits_nonzero_mantissa : (0x7C00 ||| 0x200) &&& 0x3FF ≠ 0 := by decide

/-- Subnormal shift `14 - exp` lies in [14, 24] when exp is in [-10, 0]. -/
theorem subnormal_shift_range (e : Int) (h0 : -10 ≤ e) (h1 : e ≤ 0) :
    14 ≤ 14 - e ∧ 14 - e ≤ 24 := by
  omega

/-- Below exponent -10 the result underflows to a signed zero; at -10 the shift is 24, the last case with a rounding point. -/
theorem underflow_threshold (e : Int) (h : e < -10) : 24 < 14 - e := by
  omega

-- § Widening

/-- 1.0 is 0x3C00: exponent field 15, which re-biases to f32 exponent 127, mantissa zero. -/
theorem one_bits : (0x3C00 >>> 10) &&& 0x1F = 15 ∧ 0x3C00 &&& 0x3FF = 0 ∧ 15 - 15 + 127 = 127 := by
  decide

/-- 0x3C01 is NOT 1.0: its mantissa is nonzero (the observed bad `__float2half` result). -/
theorem not_one_bits : 0x3C01 &&& 0x3FF ≠ 0 := by decide

/-- Normal widening: exponent fields 1..30 map to f32 exponents 113..142. -/
theorem widen_normal_exponent (ex : Nat) (h0 : 1 ≤ ex) (h1 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by
  omega

/-- Subnormal widening: after `s ≤ 10` shifts the f32 exponent field `113 - s` is at least 103. -/
theorem widen_subnormal_exponent (s : Nat) (hs : s ≤ 10) : 103 ≤ 113 - s := by
  omega

/-- The smallest subnormal 0x0001 needs s = 10, giving field 103, i.e. 2^(103-127) = 2^-24. -/
theorem smallest_subnormal : 113 - 10 = 103 ∧ 127 - 103 = 24 := by decide

end Strata.F16
