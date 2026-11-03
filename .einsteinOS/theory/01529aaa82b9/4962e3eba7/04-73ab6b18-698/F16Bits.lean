/-!
# MODULE: f16_bits
IEEE fp16 <-> f32 conversion by bits: round-to-nearest-even rule, exponent regions, overflow
saturation vs NaN, subnormal normalisation. Modelled over Nat.

## Source semantics
`f16_from_f32`: f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value too large for fp16
saturates to inf; exponent <= 0 is subnormal with a moving rounding point; below -10 underflows to
signed zero; the normal path rounds a 13-bit remainder to nearest even and the carry may bump the
exponent. `f32_from_f16` is exact, including subnormals.

## Form template identifiers
f16Round -> round_nearest_even ; special -> overflow_classify ; norm -> subnormal_normalise
-/

set_option maxRecDepth 100000

namespace Strata.F16

-- § Round to nearest even

/-- Rounding of `h` with discarded remainder `rem` against the half point `half`. -/
def f16Round (h rem half : Nat) : Nat :=
  if half < rem ∨ (rem = half ∧ h % 2 = 1) then h + 1 else h

/-- Rounding moves by at most one unit. -/
theorem round_cases (h r half : Nat) : f16Round h r half = h ∨ f16Round h r half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- Below the half point the value is truncated. -/
theorem round_below (h r half : Nat) (hl : r < half) : f16Round h r half = h := by
  unfold f16Round
  split <;> omega

/-- Above the half point it rounds up. -/
theorem round_above (h r half : Nat) (hl : half < r) : f16Round h r half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- Exact tie, even value: stays. -/
theorem round_tie_even (h half : Nat) (he : h % 2 = 0) : f16Round h half half = h := by
  unfold f16Round
  split <;> omega

/-- Exact tie, odd value: rounds up to the even neighbour. -/
theorem round_tie_odd (h half : Nat) (ho : h % 2 = 1) : f16Round h half half = h + 1 := by
  unfold f16Round
  split <;> omega

/-- After a tie the result is even. -/
theorem round_tie_result_even (h half : Nat) : f16Round h half half % 2 = 0 := by
  unfold f16Round
  split <;> omega

-- § Carry between mantissa and exponent

/-- Packed exponent and mantissa fields. -/
def pack (e m : Nat) : Nat := e * 1024 + m

/-- A mantissa carry bumps the exponent and clears the mantissa. -/
theorem carry_bumps (e : Nat) : pack e 1023 + 1 = pack (e + 1) 0 := by
  unfold pack; omega

/-- The largest finite fp16 rounds up to exactly +inf (0x7C00). -/
theorem carry_to_inf : pack 30 1023 + 1 = 0x7C00 := by unfold pack; decide

-- § Overflow classification

/-- Exponent and mantissa fields for the special/overflow cases, none for the finite path. -/
def special (rawexp man : Nat) : Option (Nat × Nat) :=
  if rawexp = 255 then some (31, if man ≠ 0 then 512 else 0)
  else if 143 ≤ rawexp then some (31, 0)
  else none

/-- The overflow threshold: raw exponent 143 is the first with fp16 exponent 31. -/
theorem overflow_threshold : 143 - 127 + 15 = 31 := rfl

/-- A finite overflowing value saturates to inf: exponent 31, mantissa 0. -/
theorem finite_overflow_inf (rawexp man : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp < 255) :
    special rawexp man = some (31, 0) := by
  unfold special
  rw [if_neg (by omega), if_pos h1]

/-- An f32 NaN stays a NaN: the mantissa field is non-zero. -/
theorem nan_stays_nan (man : Nat) (h : man ≠ 0) : special 255 man = some (31, 512) := by
  unfold special
  simp [h]

/-- An f32 inf stays an inf. -/
theorem inf_stays_inf : special 255 0 = some (31, 0) := by
  unfold special
  simp

/-- The conflated test of the older copies: any exponent >= 31 gives NaN when the mantissa is set. -/
def buggySpecial (man : Nat) : Nat × Nat := (31, if man ≠ 0 then 512 else 0)

/-- The old behaviour returned a NaN for a finite overflow with a non-zero mantissa. -/
theorem buggy_differs (rawexp man : Nat) (h1 : 143 ≤ rawexp) (h2 : rawexp < 255) (hm : man ≠ 0) :
    some (buggySpecial man) ≠ special rawexp man := by
  rw [finite_overflow_inf rawexp man h1 h2]
  unfold buggySpecial
  simp [hm]

-- § Subnormal and underflow regions

/-- Subnormal results (exponent in [-10, 0], raw exponent in [102, 112]) shift by 126 - rawexp. -/
theorem subnormal_shift (rawexp : Nat) (h1 : 102 ≤ rawexp) (h2 : rawexp ≤ 112) :
    14 ≤ 126 - rawexp ∧ 126 - rawexp ≤ 24 := by omega

/-- Raw exponent below 102 means fp16 exponent below -10: underflow to signed zero. -/
theorem underflow_threshold (rawexp : Nat) (h : rawexp < 102) : (rawexp : Int) - 112 < -10 := by
  omega

/-- The five regions partition all raw exponents 0..255. -/
def region (rawexp : Nat) : Nat :=
  if rawexp = 255 then 4 else if 143 ≤ rawexp then 3 else if 113 ≤ rawexp then 2
  else if 102 ≤ rawexp then 1 else 0

/-- Every region index is in 0..4. -/
theorem region_lt (r : Nat) : region r < 5 := by
  unfold region
  split <;> try omega
  split <;> try omega
  split <;> try omega
  split <;> omega

-- § Exact decode

/-- Subnormal normalisation loop with fuel: shifts left until bit 10 is set. -/
def norm : Nat → Nat → Nat × Nat
  | m, 0 => (m, 0)
  | m, f + 1 => if m &&& 1024 = 0 then ((norm (m * 2) f).1, (norm (m * 2) f).2 + 1) else (m, 0)

/-- For every non-zero 10-bit mantissa the loop ends with the leading one in bit 10, after at most
ten shifts: no subnormal input is flushed to zero. -/
theorem norm_ok : ∀ m < 1024, 1 ≤ m → (norm m 10).1 / 1024 = 1 ∧ (norm m 10).2 ≤ 10 := by
  decide

/-- The resulting f32 exponent field 113 - s stays a normal exponent (>= 103). -/
theorem norm_exp_field (s : Nat) (h : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 113 := by omega

/-- Smallest subnormal 0x0001 is 2^-24: s = 10 gives exponent field 103, and 103 - 127 = -24. -/
theorem min_subnormal : (113 - 10 : Int) - 127 = -24 := by decide

/-- Inf/NaN decode: exponent 31 maps to the all-ones f32 exponent 255. -/
theorem decode_special_exp : 0x7F800000 / 8388608 = 255 := by decide

/-- NaN payload shifts into the f32 mantissa without overflowing it. -/
theorem decode_payload (man : Nat) (h : man < 1024) : man * 8192 < 8388608 := by omega

end Strata.F16
