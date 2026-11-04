/-!
# MODULE: kernels_f16_bits
Round-to-nearest-even f32 → fp16 and exact fp16 → f32, as integer bit manipulations: the rounding step, the
overflow / underflow / NaN regimes, the subnormal exponent arithmetic and the normal-value round trip.

## Source semantics
* `f16_from_f32` splits into: f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value whose rebased
  exponent is ≥ 31 saturates to inf (never NaN); rebased exponent < -10 underflows to signed zero;
  `-10 ≤ exp ≤ 0` is the subnormal path (restore the implicit 1, shift by `14 - exp`); otherwise the normal
  path rounds a 13-bit remainder to nearest even and the carry may bump the exponent.
* `f32_from_f16` is exact: a subnormal `man * 2^-24` with leading bit at position `p` becomes exponent field
  `103 + p`; normals map exponent `ex` to `ex + 112`; `ex = 31` maps to field 255.
* Normal fp16 values round-trip exactly through f32.

## Form template identifiers
f16_from_f32 → float_to_half_bits, f32_from_f16 → half_bits_to_float, STRATA_HD → host_device_attr.
-/

namespace Strata.Kernels.F16

-- § round to nearest even on a dropped remainder

/-- Round `m / p` to nearest, ties to even; `p` is the weight of the dropped bits. -/
def rneStep (m p : Nat) : Nat :=
  if (m % p) * 2 > p ∨ ((m % p) * 2 = p ∧ (m / p) % 2 = 1) then m / p + 1 else m / p

/-- The result is the floor quotient or the next integer. -/
theorem rneStep_bounds (m p : Nat) : m / p ≤ rneStep m p ∧ rneStep m p ≤ m / p + 1 := by
  unfold rneStep
  split <;> omega

/-- Remainder below half: round down. -/
theorem rneStep_down (m p : Nat) (h : (m % p) * 2 < p) : rneStep m p = m / p := by
  unfold rneStep
  rw [if_neg]
  omega

/-- Remainder above half: round up. -/
theorem rneStep_up (m p : Nat) (h : (m % p) * 2 > p) : rneStep m p = m / p + 1 := by
  unfold rneStep
  rw [if_pos]
  exact Or.inl h

/-- Exact tie, even quotient: stay. -/
theorem rneStep_tie_even (m p : Nat) (h : (m % p) * 2 = p) (he : (m / p) % 2 = 0) : rneStep m p = m / p := by
  unfold rneStep
  rw [if_neg]
  omega

/-- Exact tie, odd quotient: round up to the even neighbour. -/
theorem rneStep_tie_odd (m p : Nat) (h : (m % p) * 2 = p) (ho : (m / p) % 2 = 1) : rneStep m p = m / p + 1 := by
  unfold rneStep
  rw [if_pos]
  exact Or.inr ⟨h, ho⟩

/-- An exactly representable value (zero remainder) is unchanged. -/
theorem rneStep_exact (q p : Nat) (hp : 0 < p) : rneStep (q * p) p = q := by
  have h1 : (q * p) % p = 0 := Nat.mul_mod_left q p
  have h2 : (q * p) / p = q := Nat.mul_div_cancel q hp
  unfold rneStep
  rw [h1, h2, if_neg]
  omega

-- § normal path

/-- Normal path value before rounding: exponent field `e` (1..30), top ten mantissa bits `man / 8192`. -/
def normalPre (sign e man : Nat) : Nat := sign + e * 1024 + man / 8192

/-- Normal path with rounding applied to the 13 dropped bits (`p = 8192`). -/
def normalRounded (sign e man : Nat) : Nat := sign + e * 1024 + rneStep man 8192

/-- The rounded normal value never exceeds `sign + 0x7C00` (the carry from `e = 30` reaches inf, not NaN). -/
theorem normal_le_inf (sign e man : Nat) (he : e ≤ 30) (hm : man < 8388608) :
    normalRounded sign e man ≤ sign + 31744 := by
  have hb := rneStep_bounds man 8192
  unfold normalRounded
  omega

/-- A mantissa that is all ones with remainder above half carries into the exponent. -/
theorem normal_carry_into_exponent (sign e : Nat) :
    normalRounded sign e 8388607 = sign + (e + 1) * 1024 := by
  unfold normalRounded rneStep
  simp <;> omega

/-- The odd bit tested by the source (`h & 1`) is the lowest bit of the kept mantissa (`man / 8192`),
because the exponent field starts at bit 10. -/
theorem lsb_is_kept_mantissa_lsb (sign e man : Nat) (hs : sign = 0 ∨ sign = 32768) (he : e < 32) :
    (sign + e * 1024 + man / 8192) % 2 = (man / 8192) % 2 := by
  rcases hs with rfl | rfl <;> omega

-- § overflow, NaN, underflow regimes

/-- Raw f32 exponent field. -/
def rawExp (x : Nat) : Nat := (x / 8388608) % 256

/-- f32 mantissa. -/
def mant (x : Nat) : Nat := x % 8388608

/-- Sign bit moved to bit 15. -/
def signBits (x : Nat) : Nat := if x ≥ 2147483648 then 32768 else 0

/-- Model of the three regime branches of `f16_from_f32` (the normal and subnormal bodies are the
`normalRounded` and `subnormalRounded` models). Returns the 16-bit result. -/
def regime (x : Nat) : Nat :=
  if rawExp x = 255 then signBits x + 31744 + (if mant x ≠ 0 then 512 else 0)
  else if rawExp x ≥ 143 then signBits x + 31744
  else if rawExp x < 102 then signBits x
  else 0

/-- Infinity stays infinity. -/
theorem inf_stays_inf (x : Nat) (h : rawExp x = 255) (hm : mant x = 0) : regime x = signBits x + 31744 := by
  unfold regime
  simp [h, hm]

/-- A NaN input stays a NaN: the mantissa field is non-zero. -/
theorem nan_stays_nan (x : Nat) (h : rawExp x = 255) (hm : mant x ≠ 0) :
    regime x = signBits x + 31744 + 512 := by
  unfold regime
  simp [h, hm]

/-- A finite overflowing value saturates to a pure inf (mantissa bits zero), never to a NaN. -/
theorem finite_overflow_is_inf (x : Nat) (h1 : rawExp x < 255) (h2 : 143 ≤ rawExp x) :
    regime x = signBits x + 31744 ∧ regime x % 1024 = (signBits x) % 1024 := by
  have hs : signBits x = 0 ∨ signBits x = 32768 := by unfold signBits; split <;> simp
  unfold regime
  have h3 : rawExp x ≠ 255 := by omega
  simp only [h3, if_false, h2, if_true]
  constructor
  · first | trivial | rfl
  · rcases hs with h | h <;> rw [h] <;> omega

/-- Rebased exponent `rawexp - 112 ≥ 31` is the same as `rawexp ≥ 143`. -/
theorem overflow_threshold (rawexp : Nat) : (rawexp : Int) - 127 + 15 ≥ 31 ↔ rawexp ≥ 143 := by omega

/-- Rebased exponent `< -10` is the same as `rawexp < 102`. -/
theorem underflow_threshold (rawexp : Nat) : (rawexp : Int) - 127 + 15 < -10 ↔ rawexp < 102 := by omega

/-- Values deep below the subnormal range yield a signed zero. -/
theorem deep_underflow_is_signed_zero (x : Nat) (h : rawExp x < 102) : regime x = signBits x := by
  unfold regime
  have h1 : rawExp x ≠ 255 := by omega
  have h2 : ¬ (rawExp x ≥ 143) := by omega
  simp [h1, h2, h]

-- § subnormal path

/-- Subnormal rounding: the 24-bit significand `man + 2^23` shifted by `sh = 14 - exp` (weight `2^sh`). -/
def subnormalRounded (sign man p : Nat) : Nat := sign + rneStep (man + 8388608) p

/-- For `sh ≥ 14` the unrounded shifted significand is below 1024, so masking with 0x3FF is a no-op. -/
theorem subnormal_shift_lt (man p : Nat) (hm : man < 8388608) (hp : 16384 ≤ p) : (man + 8388608) / p < 1024 := by
  have h : 16384 * 1024 ≤ p * 1024 := Nat.mul_le_mul_right 1024 hp
  have : man + 8388608 < p * 1024 := by omega
  exact Nat.div_lt_of_lt_mul this

/-- Rounding can reach 1024, which is exactly the smallest normal fp16 (exponent field 1, mantissa 0). -/
theorem subnormal_carry_is_min_normal (man p : Nat) (hm : man < 8388608) (hp : 16384 ≤ p) :
    rneStep (man + 8388608) p ≤ 1024 := by
  have h1 := subnormal_shift_lt man p hm hp
  have h2 := rneStep_bounds (man + 8388608) p
  omega

/-- Shift amounts `14 - exp` over the subnormal exponents `-10 ≤ exp ≤ 0` are 14..24. -/
theorem subnormal_shift_range (exp : Int) (h1 : -10 ≤ exp) (h2 : exp ≤ 0) : 14 ≤ 14 - exp ∧ 14 - exp ≤ 24 := by
  omega

-- § exact decode

/-- Normal fp16 exponent field `ex ∈ 1..30` becomes f32 exponent field `ex + 112 ∈ 113..142`. -/
theorem decode_normal_exponent (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    (ex : Int) - 15 + 127 = ex + 112 ∧ 113 ≤ ex + 112 ∧ ex + 112 ≤ 142 := by omega

/-- Subnormal with leading bit at position `p ≤ 9` (value `man * 2^-24`): after `s = 10 - p` shifts the f32
exponent field is `113 - s`, which is `103 + p`, i.e. unbiased exponent `p - 24`. -/
theorem decode_subnormal_exponent (p : Nat) (hp : p ≤ 9) : 113 - (10 - p) = 103 + p ∧ 103 + p = 127 + p - 24 := by
  omega

/-- The shifted subnormal significand has bit 10 set and stays below 2048 when `2^p ≤ man < 2^(p+1)`. -/
theorem subnormal_normalised (man p : Nat) (hp : p ≤ 9) (h1 : 2 ^ p ≤ man) (h2 : man < 2 ^ (p + 1)) :
    1024 ≤ man * 2 ^ (10 - p) ∧ man * 2 ^ (10 - p) < 2048 := by
  have e : 2 ^ p * 2 ^ (10 - p) = 1024 := by
    rw [← Nat.pow_add]
    have : p + (10 - p) = 10 := by omega
    rw [this]
  have e2 : 2 ^ (p + 1) * 2 ^ (10 - p) = 2048 := by
    rw [← Nat.pow_add]
    have : p + 1 + (10 - p) = 11 := by omega
    rw [this]
  constructor
  · calc 1024 = 2 ^ p * 2 ^ (10 - p) := e.symm
      _ ≤ man * 2 ^ (10 - p) := Nat.mul_le_mul_right _ h1
  · calc man * 2 ^ (10 - p) < 2 ^ (p + 1) * 2 ^ (10 - p) :=
          Nat.mul_lt_mul_of_pos_right h2 (Nat.two_pow_pos _)
      _ = 2048 := e2

-- § round trip

/-- A normal fp16 mantissa embedded in f32 (`man << 13`) leaves a zero remainder, so re-rounding is exact. -/
theorem normal_roundtrip_mantissa (man : Nat) : rneStep (man * 8192) 8192 = man :=
  rneStep_exact man 8192 (by decide)

/-- Hence f16 → f32 → f16 is the identity on the mantissa and exponent of normal values. -/
theorem normal_roundtrip (sign ex man : Nat) :
    normalRounded sign ex (man * 8192) = sign + ex * 1024 + man := by
  unfold normalRounded
  rw [normal_roundtrip_mantissa]

end Strata.Kernels.F16
