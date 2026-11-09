/-!
# MODULE: fp16_bits
SOURCES: include/strata/kernels/f16_bits.hpp
Bit-exact fp16 <-> f32 conversion modelled on bit fields: round-to-nearest-even encode with explicit
normal / subnormal / overflow regimes, and an exact decode.

## Source semantics
* encode: raw f32 exponent 255 stays inf/NaN; a FINITE value too large for fp16 saturates to inf (not NaN);
  a value below the subnormal range underflows to a signed zero; the round carry may bump the exponent.
* decode: exact for all 65536 inputs, subnormal `man * 2^-24` is renormalised (f32 exponent field `113 - s`).

## Form template identifiers
- `f16_from_f32` -> `encode_rne`; `f32_from_f16` -> `decode_exact`.
-/

namespace Strata.Fp16Bits

-- § field layout

/-- Sign bit of an fp16 pattern. -/
def sgn (h : Nat) : Nat := h / 32768

/-- 5-bit exponent field of an fp16 pattern. -/
def expo (h : Nat) : Nat := h / 1024 % 32

/-- 10-bit mantissa field of an fp16 pattern. -/
def mant (h : Nat) : Nat := h % 1024

/-- An fp16 pattern is exactly sign, exponent and mantissa fields. -/
theorem fields_recompose (h : Nat) (hh : h < 65536) :
    sgn h * 32768 + expo h * 1024 + mant h = h := by
  unfold sgn expo mant
  omega

/-- Finite normal exponents 1..30 map into f32 exponent fields 113..142, never 0 or 255. -/
theorem normal_exponent_range (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex + 112 ∧ ex + 112 ≤ 142 := by
  omega

/-- Subnormal renormalisation: after s ≥ 1 shifts (s ≤ 10) the f32 exponent field `113 - s` lies in 103..112. -/
theorem subnormal_exponent_range (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 10) :
    103 ≤ 113 - s ∧ 113 - s ≤ 112 := by
  omega

/-- The renormalised exponent is exactly the true exponent: `man * 2^-24` with leading 1 at bit 10-s equals
`2^(-14-s)(1+f)`, i.e. (10 - s) - 24 = -(14 + s), and the biased field is `-(14+s) + 127 = 113 - s`. -/
theorem subnormal_exponent_value (s : Nat) (h : s ≤ 10) :
    ((10 : Int) - s) - 24 = -((14 : Int) + s) ∧ (113 : Int) - s = -((14 : Int) + s) + 127 := by
  omega

/-- The smallest subnormal 0x0001 gets f32 exponent field 103, i.e. 2^(103-127) = 2^-24. -/
theorem smallest_subnormal : (113 - 10 : Nat) = 103 ∧ (103 : Int) - 127 = -24 := by
  decide

-- § encode

/-- Finite-normal case: exponent field `rawexp - 112`, mantissa from the top 10 of 23 bits, RNE carry. -/
def encNormal (s rawexp man : Nat) : Nat :=
  let h := s + (rawexp - 112) * 1024 + man / 8192
  let rem := man % 8192
  if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- Subnormal case with shift divisor `d = 2^sh`: mantissa with implicit 1 restored, RNE. -/
def encSub (s man d : Nat) : Nat :=
  let m := man + 8388608
  let h := m / d % 1024
  let rem := m % d
  if rem > d / 2 ∨ (rem = d / 2 ∧ h % 2 = 1) then s + h + 1 else s + h

/-- Encode from (sign bit value `s` = 0 or 32768, raw f32 exponent, 23-bit mantissa). -/
def f16FromF32 (s rawexp man : Nat) (d : Nat) : Nat :=
  if rawexp = 255 then s + 31744 + (if man ≠ 0 then 512 else 0)
  else if 143 ≤ rawexp then s + 31744
  else if rawexp ≤ 112 then
    (if rawexp < 102 then s else encSub s man d)
  else encNormal s rawexp man

/-- An f32 infinity stays an fp16 infinity. -/
theorem enc_inf (s d : Nat) : f16FromF32 s 255 0 d = s + 31744 := by
  simp [f16FromF32]

/-- An f32 NaN stays a NaN: exponent all ones and a nonzero mantissa. -/
theorem enc_nan (s man d : Nat) (h : man ≠ 0) : f16FromF32 s 255 man d = s + 31744 + 512 := by
  simp [f16FromF32, h]

/-- A FINITE value too large for fp16 saturates to inf, never to NaN (the bug the shared copy fixes). -/
theorem enc_overflow_is_inf (s rawexp man d : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp ≠ 255) :
    f16FromF32 s rawexp man d = s + 31744 := by
  simp [f16FromF32, h1, h2]

/-- A value below the subnormal range underflows to a signed zero. -/
theorem enc_underflow (s rawexp man d : Nat) (h : rawexp < 102) : f16FromF32 s rawexp man d = s := by
  have h1 : rawexp ≠ 255 := by omega
  have h2 : ¬ (143 ≤ rawexp) := by omega
  have h3 : rawexp ≤ 112 := by omega
  simp [f16FromF32, h1, h2, h3, h]

/-- In the normal regime the result never exceeds infinity (the carry can reach, not pass, 0x7C00). -/
theorem enc_normal_le_inf (s rawexp man : Nat) (hs : s = 0 ∨ s = 32768) (h1 : 113 ≤ rawexp)
    (h2 : rawexp ≤ 142) (hm : man < 8388608) : encNormal s rawexp man ≤ s + 31744 := by
  unfold encNormal
  dsimp only
  split <;> omega

/-- In the normal regime the result is at least the smallest normal number. -/
theorem enc_normal_ge_min (s rawexp man : Nat) (h1 : 113 ≤ rawexp) :
    s + 1024 ≤ encNormal s rawexp man := by
  unfold encNormal
  dsimp only
  split <;> omega

/-- A subnormal result never exceeds the smallest normal (the carry may promote into it). -/
theorem enc_sub_le_min_normal (s man d : Nat) : encSub s man d ≤ s + 1024 := by
  unfold encSub
  dsimp only
  have h2 : (man + 8388608) / d % 1024 < 1024 := Nat.mod_lt _ (by decide)
  split <;> omega

/-- A tie with an even truncated result is not rounded up (round-to-nearest-EVEN). -/
theorem enc_normal_tie_even (s rawexp man : Nat) (h : man % 8192 = 4096)
    (he : (s + (rawexp - 112) * 1024 + man / 8192) % 2 = 0) :
    encNormal s rawexp man = s + (rawexp - 112) * 1024 + man / 8192 := by
  unfold encNormal
  simp only [h, he]
  simp

/-- A tie with an odd truncated result rounds up. -/
theorem enc_normal_tie_odd (s rawexp man : Nat) (h : man % 8192 = 4096)
    (ho : (s + (rawexp - 112) * 1024 + man / 8192) % 2 = 1) :
    encNormal s rawexp man = s + (rawexp - 112) * 1024 + man / 8192 + 1 := by
  unfold encNormal
  simp only [h, ho]
  simp

-- § decode

/-- Decode a finite normal or infinite pattern (exponent field nonzero) to (sign, raw f32 exponent, 23-bit mantissa). -/
def decodeFields (h : Nat) : Nat × Nat × Nat :=
  (sgn h * 32768, (if expo h = 31 then 255 else expo h + 112), mant h * 8192)

/-- The decoded 23-bit mantissa fits in 23 bits (the fp16 mantissa is padded by 13 zero bits). -/
theorem decode_mantissa_lt (h : Nat) : (decodeFields h).2.2 < 8388608 := by
  unfold decodeFields mant
  simp only
  omega

/-- Finite normals decode to a raw exponent that is neither 0 nor 255. -/
theorem decode_normal_exponent (h : Nat) (h1 : 1 ≤ expo h) (h2 : expo h ≤ 30) :
    113 ≤ (decodeFields h).2.1 ∧ (decodeFields h).2.1 ≤ 142 := by
  unfold decodeFields
  simp only
  have : expo h ≠ 31 := by omega
  simp [this]
  omega

/-- encode ∘ decode is the identity on every finite normal fp16 value. -/
theorem roundtrip_normal (h : Nat) (hh : h < 65536) (h1 : 1 ≤ expo h) (h2 : expo h ≤ 30) (d : Nat) :
    f16FromF32 (decodeFields h).1 (decodeFields h).2.1 (decodeFields h).2.2 d = h := by
  have hne : expo h ≠ 31 := by omega
  have hrec := fields_recompose h hh
  have hm : mant h < 1024 := by unfold mant; omega
  have e1 : (decodeFields h).2.1 = expo h + 112 := by simp [decodeFields, hne]
  have e2 : (decodeFields h).2.2 = mant h * 8192 := rfl
  have e3 : (decodeFields h).1 = sgn h * 32768 := rfl
  rw [e1, e2, e3]
  have n1 : expo h + 112 ≠ 255 := by omega
  have n2 : ¬ (143 ≤ expo h + 112) := by omega
  have n3 : ¬ (expo h + 112 ≤ 112) := by omega
  have q1 : mant h * 8192 / 8192 = mant h := by omega
  have q2 : mant h * 8192 % 8192 = 0 := by omega
  simp only [f16FromF32, encNormal, n1, n2, n3, q1, q2]
  simp
  omega

/-- encode ∘ decode is the identity on infinity (exponent 31, mantissa 0). -/
theorem roundtrip_inf (h : Nat) (hh : h < 65536) (he : expo h = 31) (hm : mant h = 0) (d : Nat) :
    f16FromF32 (decodeFields h).1 (decodeFields h).2.1 (decodeFields h).2.2 d = h := by
  have hrec := fields_recompose h hh
  have e1 : (decodeFields h).2.1 = 255 := by simp [decodeFields, he]
  have e2 : (decodeFields h).2.2 = 0 := by simp [decodeFields, hm]
  have e3 : (decodeFields h).1 = sgn h * 32768 := rfl
  rw [e1, e2, e3]
  rw [he, hm] at hrec
  simp [f16FromF32]
  omega

end Strata.Fp16Bits
