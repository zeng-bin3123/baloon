/-!
# MODULE: half_float_bit_conversions
SOURCES: include/strata/kernels/f16_bits.hpp, include/strata/kernels/bf16_bits.hpp
Bit-exact f32 <-> fp16 and f32 <-> bf16 conversions, modelled on raw bit patterns as natural numbers
(fields extracted with `/` and `%` by powers of two, which equals the C++ shifts and masks on disjoint fields).

## Source semantics
* `bf16_from_f32(i) = ((i + ((i >> 16) & 1) + 0x7FFF) & 0xFFFF0000) >> 16` in uint32 arithmetic:
  round-to-nearest-even on the 16 dropped bits.  The `& 1` reads bit 16 (lowest KEPT bit); reading bit 15
  instead differs only on exact ties.
* `f32_from_bf16(h) = h << 16`: exact.
* `f16_from_f32`: RNE with three regimes. Raw exponent 255 (inf/NaN) stays inf/NaN; a FINITE value whose
  rebiased exponent is >= 31 saturates to inf (0x7C00), never NaN; exponent <= 0 goes subnormal, and below
  -10 flushes to a signed zero.  Normal mantissas round from 13 dropped bits; the carry may bump the exponent.
* `f32_from_f16`: exact for all 65,536 inputs, including subnormals (renormalised, never flushed).

## Form template identifiers
f16_from_f32 → f16FromF32, f32_from_f16 → f32FromF16, bf16_from_f32 → bf16FromF32,
f32_from_bf16 → f32FromBf16, the subnormal normalising loop → subShift,
the rejected bit-15 variant → bf16FromF32Bit15
-/

namespace Strata.HalfFloatBits

-- § bf16 model

/-- `bf16_from_f32` on the raw f32 bits `i`: add the RNE bias (bit 16 + 0x7FFF), keep the top 16 bits. -/
def bf16FromF32 (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) / 65536) % 65536

/-- The plausible-but-wrong variant that reads bit 15 instead of bit 16. -/
def bf16FromF32Bit15 (i : Nat) : Nat := ((i + (i / 32768) % 2 + 32767) / 65536) % 65536

/-- `f32_from_bf16`: the bf16 bits become the top half of an f32. -/
def f32FromBf16 (h : Nat) : Nat := h * 65536

/-- Wrapping the sum to uint32 and then masking/shifting equals dividing first and reducing mod 2^16,
so the `Nat` model above is exactly the uint32 computation. -/
theorem uint32_wrap_then_shift (x : Nat) : (x % 4294967296) / 65536 = (x / 65536) % 65536 := by
  omega

/-- bf16 → f32 → bf16 is the identity: the widening is exact and the RNE bias never carries. -/
theorem bf16_roundtrip (h : Nat) (hh : h < 65536) : bf16FromF32 (f32FromBf16 h) = h := by
  unfold bf16FromF32 f32FromBf16
  have h1 : h * 65536 / 65536 = h := by omega
  rw [h1]
  have hb : h % 2 + 32767 < 65536 := by
    have := Nat.mod_lt h (show 0 < 2 by decide)
    omega
  have hre : h * 65536 + h % 2 + 32767 = (h % 2 + 32767) + h * 65536 := by omega
  have h2 : (h * 65536 + h % 2 + 32767) / 65536 = h := by
    rw [hre, Nat.add_mul_div_right _ _ (show 0 < 65536 by decide), Nat.div_eq_of_lt hb, Nat.zero_add]
  rw [h2]
  exact Nat.mod_eq_of_lt hh

/-- Below the half-way point the dropped bits are simply truncated. -/
theorem bf16_below_half_truncates (i : Nat) (hl : i % 65536 < 32768) :
    bf16FromF32 i = (i / 65536) % 65536 := by
  unfold bf16FromF32
  have hq : (i + (i / 65536) % 2 + 32767) / 65536 = i / 65536 := by omega
  rw [hq]

/-- Above the half-way point the kept part is incremented (mod 2^16, matching uint32 wrap). -/
theorem bf16_above_half_rounds_up (i : Nat) (hl : 32768 < i % 65536) :
    bf16FromF32 i = (i / 65536 + 1) % 65536 := by
  unfold bf16FromF32
  have hq : (i + (i / 65536) % 2 + 32767) / 65536 = i / 65536 + 1 := by omega
  rw [hq]

/-- On an exact tie with an even kept part, nothing is added: ties go to even. -/
theorem bf16_tie_even_stays (i : Nat) (ht : i % 65536 = 32768) (he : (i / 65536) % 2 = 0) :
    bf16FromF32 i = (i / 65536) % 65536 := by
  unfold bf16FromF32; omega

/-- On an exact tie with an odd kept part, it is rounded up to the next (even) value. -/
theorem bf16_tie_odd_rounds_up (i : Nat) (ht : i % 65536 = 32768) (ho : (i / 65536) % 2 = 1) :
    bf16FromF32 i = (i / 65536 + 1) % 65536 := by
  unfold bf16FromF32; omega

/-- Every tie lands on an even bf16 pattern (the uint32 wrap at the very top also lands on 0, even). -/
theorem bf16_tie_result_even (i : Nat) (ht : i % 65536 = 32768) :
    bf16FromF32 i % 2 = 0 := by
  have hcase : (i / 65536) % 2 = 0 ∨ (i / 65536) % 2 = 1 := by omega
  cases hcase with
  | inl he =>
    rw [bf16_tie_even_stays i ht he]; omega
  | inr ho =>
    rw [bf16_tie_odd_rounds_up i ht ho]; omega

/-- The bit-15 variant agrees with the correct rule everywhere except on exact ties — which is why
end-to-end tests would never see it. -/
theorem bit15_agrees_off_ties (i : Nat) (hn : i % 65536 ≠ 32768) :
    bf16FromF32Bit15 i = bf16FromF32 i := by
  have hcase : i % 65536 < 32768 ∨ 32768 < i % 65536 := by omega
  unfold bf16FromF32Bit15
  cases hcase with
  | inl hl =>
    have hb : (i / 32768) % 2 = 0 := by omega
    have hq : (i + (i / 32768) % 2 + 32767) / 65536 = i / 65536 := by omega
    rw [hq, bf16_below_half_truncates i hl]
  | inr hl =>
    have hb : (i / 32768) % 2 = 1 := by omega
    have hq : (i + (i / 32768) % 2 + 32767) / 65536 = i / 65536 + 1 := by omega
    rw [hq, bf16_above_half_rounds_up i hl]

/-- ...and on a tie it is wrong: 0x00008000 must round to even 0, the bit-15 variant gives 1. -/
theorem bit15_wrong_on_tie : bf16FromF32 0x8000 = 0 ∧ bf16FromF32Bit15 0x8000 = 1 := by decide

-- § fp16 model

/-- Number of left shifts until bit 10 of a nonzero subnormal mantissa is set (the C++ `while` loop). -/
def subShiftAux (m : Nat) : Nat → Nat
  | 0 => 0
  | f + 1 => if 1024 ≤ m then 0 else 1 + subShiftAux (2 * m) f

/-- `s` in `f32_from_f16`'s subnormal branch. -/
def subShift (man : Nat) : Nat := subShiftAux man 11

/-- `f16_from_f32` on raw f32 bits. `re` is the raw exponent, `re - 112` the rebiased fp16 exponent. -/
def f16FromF32 (x : Nat) : Nat :=
  let s := (x / 2147483648) % 2 * 32768
  let re := (x / 8388608) % 256
  let man := x % 8388608
  if re = 255 then s + 31744 + (if man = 0 then 0 else 512)
  else if 143 ≤ re then s + 31744
  else if re ≤ 112 then
    if re < 102 then s
    else
      let m := man + 8388608
      let sh := 126 - re
      let h := (m / 2 ^ sh) % 1024
      let rem := m % 2 ^ sh
      let half := 2 ^ (sh - 1)
      if half < rem ∨ (rem = half ∧ h % 2 = 1) then s + h + 1 else s + h
  else
    let h := s + (re - 112) * 1024 + man / 8192
    let rem := man % 8192
    if 4096 < rem ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- `f32_from_f16` on raw fp16 bits: exact widening, subnormals renormalised. -/
def f32FromF16 (h : Nat) : Nat :=
  let sign := (h / 32768) % 2 * 2147483648
  let ex := (h / 1024) % 32
  let man := h % 1024
  if ex = 0 then
    if man = 0 then sign
    else
      let s := subShift man
      sign + (113 - s) * 8388608 + ((man * 2 ^ s) % 1024) * 8192
  else if ex = 31 then sign + 2139095040 + man * 8192
  else sign + (ex + 112) * 8388608 + man * 8192

/-- An f32 NaN (raw exponent 255, nonzero mantissa) stays a quiet NaN with its sign. -/
theorem f16_nan_stays_nan (x : Nat) (he : (x / 8388608) % 256 = 255) (hm : x % 8388608 ≠ 0) :
    f16FromF32 x = (x / 2147483648) % 2 * 32768 + 31744 + 512 := by
  simp only [f16FromF32, he, hm, if_true, if_false]

/-- An f32 infinity stays an infinity with its sign. -/
theorem f16_inf_stays_inf (x : Nat) (he : (x / 8388608) % 256 = 255) (hm : x % 8388608 = 0) :
    f16FromF32 x = (x / 2147483648) % 2 * 32768 + 31744 := by
  simp only [f16FromF32, he, hm, if_true]

/-- The round-198 fix: a FINITE value too large for fp16 saturates to inf, never NaN. -/
theorem f16_finite_overflow_saturates (x : Nat) (hf : (x / 8388608) % 256 ≠ 255)
    (ho : 143 ≤ (x / 8388608) % 256) :
    f16FromF32 x = (x / 2147483648) % 2 * 32768 + 31744 := by
  simp only [f16FromF32, hf, ho, if_true, if_false]

/-- The saturated result has a zero mantissa field, so it is an infinity and not a NaN. -/
theorem f16_finite_overflow_not_nan (x : Nat) (hf : (x / 8388608) % 256 ≠ 255)
    (ho : 143 ≤ (x / 8388608) % 256) : f16FromF32 x % 1024 = 0 := by
  rw [f16_finite_overflow_saturates x hf ho]; omega

/-- Below 2^-25 in magnitude (rebiased exponent < -10) the result is a signed zero. -/
theorem f16_underflow_signed_zero (x : Nat) (hu : (x / 8388608) % 256 < 102) :
    f16FromF32 x = (x / 2147483648) % 2 * 32768 := by
  have h1 : (x / 8388608) % 256 ≠ 255 := by omega
  have h2 : ¬ 143 ≤ (x / 8388608) % 256 := by omega
  have h3 : (x / 8388608) % 256 ≤ 112 := by omega
  simp only [f16FromF32, h1, h2, h3, hu, if_true, if_false]

/-- In the normal regime the result (minus its sign) is a normal fp16 or, after a carry, exactly inf. -/
theorem f16_normal_range (x : Nat) (h1 : 113 ≤ (x / 8388608) % 256) (h2 : (x / 8388608) % 256 < 143) :
    (x / 2147483648) % 2 * 32768 + 1024 ≤ f16FromF32 x ∧
      f16FromF32 x ≤ (x / 2147483648) % 2 * 32768 + 31744 := by
  have a1 : (x / 8388608) % 256 ≠ 255 := by omega
  have a2 : ¬ 143 ≤ (x / 8388608) % 256 := by omega
  have a3 : ¬ (x / 8388608) % 256 ≤ 112 := by omega
  simp only [f16FromF32, a1, a2, a3, if_false]
  split <;> omega

/-- Oracle vectors (numpy float16): 1.0, -1.0, 65504 (max), 65520 (ties up to inf), 65536, 1e30, inf, NaN,
2^-24 (smallest subnormal) and 2^-25 (a tie that rounds to even zero). -/
theorem f16_oracle_vectors :
    f16FromF32 0x3F800000 = 0x3C00 ∧ f16FromF32 0xBF800000 = 0xBC00 ∧
    f16FromF32 0x477FE000 = 0x7BFF ∧ f16FromF32 0x477FF000 = 0x7C00 ∧
    f16FromF32 0x47800000 = 0x7C00 ∧ f16FromF32 0x7149F2CA = 0x7C00 ∧
    f16FromF32 0x7F800000 = 0x7C00 ∧ f16FromF32 0x7FC00000 = 0x7E00 ∧
    f16FromF32 0x33800000 = 0x0001 ∧ f16FromF32 0x33000000 = 0x0000 := by
  decide

/-- Widening a normal fp16 gives the f32 with the same sign, rebiased exponent and shifted mantissa. -/
theorem f32FromF16_normal (sg ex man : Nat) (hs : sg < 2) (he1 : 1 ≤ ex) (he2 : ex ≤ 30)
    (hm : man < 1024) :
    f32FromF16 (sg * 32768 + ex * 1024 + man) =
      sg * 2147483648 + (ex + 112) * 8388608 + man * 8192 := by
  have h1 : (sg * 32768 + ex * 1024 + man) / 32768 % 2 = sg := by omega
  have h2 : (sg * 32768 + ex * 1024 + man) / 1024 % 32 = ex := by omega
  have h3 : (sg * 32768 + ex * 1024 + man) % 1024 = man := by omega
  have h4 : ex ≠ 0 := by omega
  have h5 : ex ≠ 31 := by omega
  simp only [f32FromF16, h1, h2, h3, h4, h5, if_false]

/-- fp16 → f32 → fp16 is the identity on every normal fp16 value: the widening is exact and the
narrowing sees zero dropped bits. -/
theorem f16_roundtrip_normal (sg ex man : Nat) (hs : sg < 2) (he1 : 1 ≤ ex) (he2 : ex ≤ 30)
    (hm : man < 1024) :
    f16FromF32 (f32FromF16 (sg * 32768 + ex * 1024 + man)) = sg * 32768 + ex * 1024 + man := by
  rw [f32FromF16_normal sg ex man hs he1 he2 hm]
  have a1 : (sg * 2147483648 + (ex + 112) * 8388608 + man * 8192) / 2147483648 % 2 = sg := by omega
  have a2 : (sg * 2147483648 + (ex + 112) * 8388608 + man * 8192) / 8388608 % 256 = ex + 112 := by
    omega
  have a3 : (sg * 2147483648 + (ex + 112) * 8388608 + man * 8192) % 8388608 = man * 8192 := by omega
  have b1 : ex + 112 ≠ 255 := by omega
  have b2 : ¬ 143 ≤ ex + 112 := by omega
  have b3 : ¬ ex + 112 ≤ 112 := by omega
  have b4 : man * 8192 / 8192 = man := by omega
  have b5 : man * 8192 % 8192 = 0 := Nat.mul_mod_left man 8192
  have b6 : ex + 112 - 112 = ex := by omega
  simp only [f16FromF32, a1, a2, a3, b1, b2, b3, b4, b5, b6, if_false]
  split <;> omega

/-- Zero (either sign) widens to the f32 zero of the same sign. -/
theorem f32FromF16_zero (sg : Nat) (hs : sg < 2) : f32FromF16 (sg * 32768) = sg * 2147483648 := by
  have h1 : sg * 32768 / 32768 % 2 = sg := by omega
  have h2 : sg * 32768 / 1024 % 32 = 0 := by omega
  have h3 : sg * 32768 % 1024 = 0 := by omega
  simp only [f32FromF16, h1, h2, h3, if_true]

/-- The smallest subnormal 0x0001 is NOT flushed: it widens to 2^-24 (f32 bits 0x33800000), the case the
two private copies got wrong. -/
theorem f32FromF16_min_subnormal : f32FromF16 0x0001 = 0x33800000 := by decide

/-- The largest subnormal 0x03FF widens to (1023/1024)·2^-14 and round-trips. -/
theorem f16_subnormal_roundtrip_max : f16FromF32 (f32FromF16 0x03FF) = 0x03FF := by decide

end Strata.HalfFloatBits
