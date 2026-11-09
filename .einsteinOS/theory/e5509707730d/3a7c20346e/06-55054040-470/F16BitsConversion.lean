/-!
# MODULE: f16_bits_conversion
SOURCES: include/strata/kernels/f16_bits.hpp
Hand-written IEEE binary16 conversions (`f16_from_f32` rounding to nearest even, `f32_from_f16` exact),
plus the bf16 round-to-nearest-even trick used in `ple.cu`.

## Source semantics
* Rebiased exponent is `rawexp - 127 + 15`. Raw exponent 255 stays inf/NaN; finite values with rebiased
  exponent >= 31 saturate to inf; rebiased exponent <= 0 enters the subnormal path; below -10 it flushes to
  a signed zero.
* Rounding adds one to the truncated mantissa when the discarded remainder exceeds half, or equals half
  and the truncated value is odd. A carry may bump the exponent.
* Decoding subnormals normalises the mantissa with `s` left shifts; the f32 exponent field is `113 - s`.

## Form template identifiers
`f16_from_f32` -> `encode_half`; `f32_from_f16` -> `decode_half`; `man` -> `mantissa`; `sh` -> `shift`;
`rem` -> `discarded_remainder`.
-/

namespace Strata.F16

-- § exponent classification

/-- Rebiased exponent `rawexp - 127 + 15`. -/
def rebias (rawexp : Nat) : Int := (rawexp : Int) - 127 + 15

/-- Finite overflow to infinity happens exactly when the raw exponent is at least 143. -/
theorem overflow_iff (r : Nat) : rebias r ≥ 31 ↔ 143 ≤ r := by
  unfold rebias
  omega

/-- The subnormal path is taken exactly when the raw exponent is at most 112. -/
theorem subnormal_iff (r : Nat) : rebias r ≤ 0 ↔ r ≤ 112 := by
  unfold rebias
  omega

/-- Flush to a signed zero happens exactly when the raw exponent is at most 101. -/
theorem underflow_iff (r : Nat) : rebias r < -10 ↔ r ≤ 101 := by
  unfold rebias
  omega

/-- In the subnormal path the shift `14 - exp` stays within 14..24, so the 32-bit shifts are defined. -/
theorem subnormal_shift_range (r : Nat) (h1 : 102 ≤ r) (h2 : r ≤ 112) :
    14 ≤ 14 - rebias r ∧ 14 - rebias r ≤ 24 := by
  unfold rebias
  omega

-- § round to nearest even

/-- Round-to-nearest-even step: bump `h` when the remainder is above half, or at half with `h` odd. -/
def rne (h rem half : Nat) : Nat :=
  if rem > half ∨ (rem = half ∧ h % 2 = 1) then h + 1 else h

/-- On an exact tie the rounded result is even. -/
theorem rne_tie_even (h rem half : Nat) (e : rem = half) : rne h rem half % 2 = 0 := by
  unfold rne
  split <;> omega

/-- Below half the value is truncated. -/
theorem rne_below (h rem half : Nat) (hl : rem < half) : rne h rem half = h := by
  unfold rne
  split <;> omega

/-- Above half the value is bumped. -/
theorem rne_above (h rem half : Nat) (hg : rem > half) : rne h rem half = h + 1 := by
  unfold rne
  split <;> omega

/-- Rounding moves the value by at most one unit upwards. -/
theorem rne_bounds (h rem half : Nat) : h ≤ rne h rem half ∧ rne h rem half ≤ h + 1 := by
  unfold rne
  split <;> omega

-- § normal path

/-- For a 23-bit mantissa the kept part fits in 10 bits and the remainder in 13 bits. -/
theorem normal_split (man : Nat) (h : man < 8388608) :
    man / 8192 < 1024 ∧ man % 8192 < 8192 ∧ man = 8192 * (man / 8192) + man % 8192 := by
  omega

/-- When the kept mantissa is all ones, the rounding carry bumps the exponent field by one. -/
theorem carry_bumps_exponent (e : Nat) : (e * 1024 + 1023) + 1 = (e + 1) * 1024 := by
  omega

/-- The largest finite exponent plus a rounding carry lands exactly on the infinity pattern 0x7C00. -/
theorem carry_to_inf : (30 * 1024 + 1023) + 1 = 0x7C00 := by decide

-- § subnormal path

/-- At the deepest subnormal shift (24) an input in `[2^23, 2^24)` truncates to zero and rounds to the
smallest subnormal exactly when it is strictly above the halfway point. -/
theorem subnormal_min_boundary (man : Nat) (h1 : 8388608 ≤ man) (h2 : man < 16777216) :
    man / 16777216 = 0 ∧ (rne 0 man 8388608 = 1 ↔ man > 8388608) := by
  unfold rne
  split <;> omega

-- § exact decode of subnormals

/-- Number of left shifts needed until the leading one reaches bit 10 (fuel-bounded). -/
def normShift : Nat → Nat → Nat
  | _, 0 => 0
  | m, fuel + 1 => if 1024 ≤ m then 0 else 1 + normShift (2 * m) fuel

/-- The shift count never exceeds the fuel. -/
theorem normShift_le (m fuel : Nat) : normShift m fuel ≤ fuel := by
  induction fuel generalizing m with
  | zero => simp [normShift]
  | succ n ih =>
    simp only [normShift]
    split
    · omega
    · have := ih (2 * m)
      omega

/-- The smallest subnormal mantissa 1 needs ten shifts. -/
theorem normShift_one : normShift 1 10 = 10 := by decide

/-- A mantissa already holding bit 9 needs a single shift. -/
theorem normShift_512 : normShift 512 10 = 1 := by decide

/-- With at most ten shifts the f32 exponent field `113 - s` stays in range 103..113. -/
theorem decode_exponent_range (s : Nat) (hs : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 113 := by
  omega

/-- The smallest subnormal decodes to exponent field 103, i.e. 2^(103-127) = 2^-24. -/
theorem smallest_subnormal_exponent : 113 - 10 + 24 = 127 := by decide

-- § bf16 rounding used by ple.cu

/-- `(i + lsb + 0x7FFF) >> 16` with `lsb` the lowest kept bit equals round-to-nearest-even of `i / 65536`. -/
theorem bf16_rounding (i : Nat) :
    (i + (i / 65536) % 2 + 32767) / 65536 =
      (if i % 65536 > 32768 ∨ (i % 65536 = 32768 ∧ (i / 65536) % 2 = 1)
        then i / 65536 + 1 else i / 65536) := by
  split <;> omega

end Strata.F16
