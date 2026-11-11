/-!
# MODULE: fp16_bit_conversion
SOURCES: include/strata/kernels/f16_bits.hpp, include/strata/artifact/dequant.hpp
Bit-exact model of the fp16 → f32 widening (`f32_from_f16`, `fp16_to_fp32`) and of the
round-to-nearest-even f32 → fp16 narrowing (`f16_from_f32`), over raw bit patterns as naturals.

## Source semantics
* fp16 = 1 sign, 5 exponent, 10 mantissa bits; f32 = 1, 8, 23.
* widening is exact: normals rebias the exponent by +112; subnormals are normalised by shifting the
  mantissa left `s` times until bit 10 is set, giving f32 exponent field `113 - s`.
* narrowing distinguishes an f32 inf/NaN (raw exponent 255 → inf/NaN) from a FINITE overflow
  (→ signed inf); the earlier conflated version returned NaN for every finite overflow.
* rounding is round-to-nearest-even with a sticky remainder; the carry may bump the exponent.
* narrowing after widening is the identity on every normal fp16.

## Form template identifiers
f32_from_f16 / fp16_to_fp32 → `f32FromF16`; f16_from_f32 → `f16FromF32`; subnormal
`while (!(m & 0x400))` loop → `normAux`/`normalize`; conflated overflow test → `f16FromF32Conflated`.
-/

namespace Strata.Fp16Bits

-- § field extraction

/-- The C++ field extraction `(x >> k) & (2^n - 1)` is `x / 2^k % 2^n`. -/
theorem field_eq (x k n : Nat) : (x >>> k) &&& (2 ^ n - 1) = x / 2 ^ k % 2 ^ n := by
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]

-- § widening

/-- The subnormal normalisation loop with explicit fuel: shift left until bit 10 is set. -/
def normAux : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | f + 1, m, s => if m / 1024 % 2 = 1 then (m, s) else normAux f (m * 2) (s + 1)

/-- Normalise a nonzero 10-bit subnormal mantissa; returns (shifted mantissa, shift count). -/
def normalize (m : Nat) : Nat × Nat := normAux 11 m 0

/-- The loop only ever multiplies by two: the result is `m * 2^(shift)` relative to the start. -/
theorem normAux_spec (f m s : Nat) :
    (normAux f m s).1 = m * 2 ^ ((normAux f m s).2 - s) ∧ s ≤ (normAux f m s).2 := by
  induction f generalizing m s with
  | zero => simp [normAux]
  | succ f ih =>
    unfold normAux
    split
    · simp
    · obtain ⟨h1, h2⟩ := ih (m * 2) (s + 1)
      refine ⟨?_, by omega⟩
      rw [h1]
      have e : (normAux f (m * 2) (s + 1)).2 - s = ((normAux f (m * 2) (s + 1)).2 - (s + 1)) + 1 := by
        omega
      rw [e, Nat.pow_succ, Nat.mul_comm (2 ^ _) 2, Nat.mul_assoc]

/-- So the normalised mantissa is exactly `man * 2^s`, which is what makes the f32 value
`man * 2^-24` exact (exponent `113 - s - 127 - 10 = -24 - s`). -/
theorem normalize_scaled (m : Nat) : (normalize m).1 = m * 2 ^ (normalize m).2 := by
  have h := (normAux_spec 11 m 0).1
  simpa [normalize] using h

/-- Every nonzero subnormal mantissa normalises with bit 10 set and a shift of at most 10, so the
f32 exponent field `113 - s` is at least 103: the widened value is a normal f32. -/
theorem normalize_all :
    (List.range 1024).all (fun m => m = 0 ||
      ((normalize m).1 / 1024 % 2 == 1 && (normalize m).2 ≤ 10 && (normalize m).1 < 2048)) = true := by
  set_option maxRecDepth 200000 in decide

/-- fp16 bits → f32 bits, mirroring `f32_from_f16` field by field. -/
def f32FromF16 (h : Nat) : Nat :=
  let sign := h / 32768 % 2
  let ex := h / 1024 % 32
  let man := h % 1024
  if ex = 0 then
    if man = 0 then sign * 2 ^ 31
    else sign * 2 ^ 31 + (113 - (normalize man).2) * 2 ^ 23 + ((normalize man).1 % 1024) * 2 ^ 13
  else if ex = 31 then sign * 2 ^ 31 + 0x7F800000 + man * 2 ^ 13
  else sign * 2 ^ 31 + (ex + 112) * 2 ^ 23 + man * 2 ^ 13

-- § narrowing

/-- f32 bits → fp16 bits with round-to-nearest-even, mirroring `f16_from_f32`.
`raw` is the f32 exponent field; the fp16 exponent is `raw - 112`. -/
def f16FromF32 (x : Nat) : Nat :=
  let sign := x / 2 ^ 31 * 32768
  let raw := x / 2 ^ 23 % 256
  let man := x % 2 ^ 23
  if raw = 255 then sign + 0x7C00 + (if man ≠ 0 then 0x200 else 0)
  else if 143 ≤ raw then sign + 0x7C00
  else if raw ≤ 112 then
    if raw < 102 then sign
    else
      let m := man + 2 ^ 23
      let sh := 126 - raw
      let h := m / 2 ^ sh % 1024
      let rem := m % 2 ^ sh
      let half := 2 ^ (sh - 1)
      if rem > half ∨ (rem = half ∧ h % 2 = 1) then sign + h + 1 else sign + h
  else
    let h := sign + (raw - 112) * 1024 + man / 2 ^ 13
    let rem := man % 2 ^ 13
    if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- The conflated overflow test the header replaced: `exp >= 31` returns a NaN when the f32
mantissa is nonzero, even for a finite input. -/
def f16FromF32Conflated (x : Nat) : Nat :=
  let sign := x / 2 ^ 31 * 32768
  let raw := x / 2 ^ 23 % 256
  let man := x % 2 ^ 23
  if 143 ≤ raw then sign + 0x7C00 + (if man ≠ 0 then 0x200 else 0) else f16FromF32 x

-- § oracle vectors

/-- 1.0 narrows to 0x3C00 and 65504 (fp16 max) to 0x7BFF. -/
theorem narrow_one_and_max : f16FromF32 0x3F800000 = 0x3C00 ∧ f16FromF32 0x477FE000 = 0x7BFF := by
  decide

/-- Finite overflow saturates to +inf: 65536.0 and 1e30 both give 0x7C00. -/
theorem narrow_finite_overflow :
    f16FromF32 0x47800000 = 0x7C00 ∧ f16FromF32 0x7149F2CA = 0x7C00 := by decide

/-- Round 198's finding: the conflated test returns a NaN (0x7E00) for the finite value 1e30. -/
theorem conflated_returns_nan : f16FromF32Conflated 0x7149F2CA = 0x7E00 := by decide

/-- f32 inf stays inf and a quiet NaN stays a NaN, with the sign kept. -/
theorem narrow_inf_nan :
    f16FromF32 0x7F800000 = 0x7C00 ∧ f16FromF32 0xFF800000 = 0xFC00 ∧
      f16FromF32 0x7FC00000 = 0x7E00 := by decide

/-- Subnormal regime: 2^-24 is 0x0001, and the exact tie 2^-25 rounds to EVEN (zero). -/
theorem narrow_subnormal :
    f16FromF32 0x33800000 = 0x0001 ∧ f16FromF32 0x33000000 = 0x0000 := by decide

/-- Rounding carry bumps the exponent: the largest f32 below 2.0 narrows to 2.0 (0x4000). -/
theorem narrow_carry : f16FromF32 0x3FFFFFFF = 0x4000 := by decide

/-- Widening vectors: 0x3C00 → 1.0, 0x0001 → 2^-24 (not flushed to zero), 0x7C00 → inf. -/
theorem widen_vectors :
    f32FromF16 0x3C00 = 0x3F800000 ∧ f32FromF16 0x0001 = 0x33800000 ∧
      f32FromF16 0x03FF = 0x387FC000 ∧ f32FromF16 0x7C00 = 0x7F800000 := by decide

-- § round trip

/-- Narrowing after widening is the identity on every normal fp16 (exponent 1..30), any sign. -/
theorem roundtrip_normal (s ex man : Nat) (hs : s < 2) (he1 : 1 ≤ ex) (he2 : ex ≤ 30)
    (hm : man < 1024) :
    f16FromF32 (f32FromF16 (s * 32768 + ex * 1024 + man)) = s * 32768 + ex * 1024 + man := by
  have hsign : (s * 32768 + ex * 1024 + man) / 32768 % 2 = s := by omega
  have hex : (s * 32768 + ex * 1024 + man) / 1024 % 32 = ex := by omega
  have hman : (s * 32768 + ex * 1024 + man) % 1024 = man := by omega
  have hw : f32FromF16 (s * 32768 + ex * 1024 + man) =
      s * 2 ^ 31 + (ex + 112) * 2 ^ 23 + man * 2 ^ 13 := by
    simp only [f32FromF16, hsign, hex, hman]
    have h0 : ex ≠ 0 := by omega
    have h31 : ex ≠ 31 := by omega
    simp [h0, h31]
  rw [hw]
  simp only [f16FromF32]
  have hr : (s * 2 ^ 31 + (ex + 112) * 2 ^ 23 + man * 2 ^ 13) / 2 ^ 23 % 256 = ex + 112 := by
    omega
  have hsg : (s * 2 ^ 31 + (ex + 112) * 2 ^ 23 + man * 2 ^ 13) / 2 ^ 31 = s := by omega
  have hm2 : (s * 2 ^ 31 + (ex + 112) * 2 ^ 23 + man * 2 ^ 13) % 2 ^ 23 = man * 2 ^ 13 := by
    omega
  rw [hr, hsg, hm2]
  have a1 : ex + 112 ≠ 255 := by omega
  have a2 : ¬ 143 ≤ ex + 112 := by omega
  have a3 : ¬ ex + 112 ≤ 112 := by omega
  have a4 : man * 2 ^ 13 / 2 ^ 13 = man := by omega
  have a5 : man * 2 ^ 13 % 2 ^ 13 = 0 := by omega
  simp only [a1, a2, a3, a4, a5, if_false]
  have a6 : ex + 112 - 112 = ex := by omega
  rw [a6]
  have a7 : ¬ (0 > 4096 ∨ (0 = 4096 ∧ (s * 32768 + ex * 1024 + man) % 2 = 1)) := by omega
  simp only [a7, if_false]

end Strata.Fp16Bits
