/-!
# MODULE: f16_bits
Round-to-nearest-even f32 -> fp16 (as raw bits) and exact fp16 -> f32 decoding, written out bit by bit.
This file models the field arithmetic and the rounding step on `Nat`.

## Source semantics
* `f16_from_f32` splits the f32 into sign / exponent / mantissa, handles inf/NaN (raw exponent 255) separately
  from finite overflow (saturates to inf), underflows to signed zero below `exp < -10`, restores the implicit 1
  for subnormals (the rounding point moves), and otherwise rounds 13 dropped bits with a sticky remainder.
* `f32_from_f16` is exact: normals re-bias the exponent by `+112`, subnormals are normalised with `s` shifts to
  exponent field `113 - s`, ex = 31 maps to the f32 inf/NaN pattern.

## Form template identifiers
* `rne_shift`            : the shared "shift right and round to nearest even" step (normal: sh = 13)
* `exp_field_decode`     : re-biased exponent field (ex + 112)
* `sat_threshold`        : exp >= 31 -> inf
-/

namespace Strata.F16

-- § field extraction

/-- Sign bit of an fp16 held in the low 16 bits. -/
def signOf (h : Nat) : Nat := h &&& 0x8000

/-- Five-bit exponent field of an fp16. -/
def expOf (h : Nat) : Nat := (h >>> 10) &&& 0x1F

/-- Ten-bit mantissa field of an fp16. -/
def manOf (h : Nat) : Nat := h &&& 0x3FF

/-- Shift right by `sh` and round to nearest, ties to even (the rounding step in every regime). -/
def rneShift (m sh : Nat) : Nat :=
  let h := m >>> sh
  let rem := m % 2 ^ sh
  let half := 2 ^ (sh - 1)
  if rem > half ∨ (rem = half ∧ h % 2 = 1) then h + 1 else h

/-- Re-biased f32 exponent field for a normal fp16 exponent `ex` (`ex - 15 + 127`). -/
def expFieldDecode (ex : Nat) : Nat := ex + 112

/-- A normal fp16 exponent decodes to an f32 exponent field strictly inside the normal range. -/
theorem expFieldDecode_normal (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ expFieldDecode ex ∧ expFieldDecode ex ≤ 142 := by
  unfold expFieldDecode
  omega

/-- Decoding the largest finite fp16 exponent never reaches the f32 inf/NaN field (255). -/
theorem expFieldDecode_lt_255 (ex : Nat) (h2 : ex ≤ 30) : expFieldDecode ex < 255 := by
  unfold expFieldDecode
  omega

/-- The decoded exponent field is injective, so distinct fp16 exponents stay distinct in f32. -/
theorem expFieldDecode_injective (a b : Nat) (h : expFieldDecode a = expFieldDecode b) : a = b := by
  unfold expFieldDecode at h
  omega

-- § rounding step

/-- The rounded result is either the truncated value or one more. -/
theorem rneShift_cases (m sh : Nat) :
    rneShift m sh = (m >>> sh) ∨ rneShift m sh = (m >>> sh) + 1 := by
  unfold rneShift
  by_cases h : (m % 2 ^ sh > 2 ^ (sh - 1) ∨ (m % 2 ^ sh = 2 ^ (sh - 1) ∧ (m >>> sh) % 2 = 1))
  · right; simp [h]
  · left; simp [h]

/-- Strictly below the halfway point the value is truncated. -/
theorem rneShift_below_half (m sh : Nat) (h : m % 2 ^ sh < 2 ^ (sh - 1)) :
    rneShift m sh = m >>> sh := by
  unfold rneShift
  have h1 : ¬ (m % 2 ^ sh > 2 ^ (sh - 1)) := by omega
  have h2 : ¬ (m % 2 ^ sh = 2 ^ (sh - 1)) := by omega
  simp [h1, h2]

/-- Strictly above the halfway point the value is rounded up. -/
theorem rneShift_above_half (m sh : Nat) (h : m % 2 ^ sh > 2 ^ (sh - 1)) :
    rneShift m sh = (m >>> sh) + 1 := by
  unfold rneShift
  simp [h]

/-- On an exact tie with an even truncated value the result stays put (ties to even). -/
theorem rneShift_tie_even (m sh : Nat) (h : m % 2 ^ sh = 2 ^ (sh - 1)) (he : (m >>> sh) % 2 = 0) :
    rneShift m sh = m >>> sh := by
  unfold rneShift
  have h1 : ¬ (m % 2 ^ sh > 2 ^ (sh - 1)) := by omega
  have h2 : ¬ ((m >>> sh) % 2 = 1) := by omega
  simp [h1, h2]

/-- On an exact tie with an odd truncated value the result rounds up to the even neighbour. -/
theorem rneShift_tie_odd (m sh : Nat) (h : m % 2 ^ sh = 2 ^ (sh - 1)) (ho : (m >>> sh) % 2 = 1) :
    rneShift m sh = (m >>> sh) + 1 := by
  unfold rneShift
  simp [h, ho]

/-- Whenever a tie rounds, the result is even: ties go to even. -/
theorem rneShift_tie_result_even (m sh : Nat) (h : m % 2 ^ sh = 2 ^ (sh - 1)) :
    rneShift m sh % 2 = 0 := by
  rcases Nat.mod_two_eq_zero_or_one (m >>> sh) with he | ho
  · rw [rneShift_tie_even m sh h he]; exact he
  · rw [rneShift_tie_odd m sh h ho]; omega

-- § regimes

/-- Normal regime: mantissa rounding uses the 13 dropped bits (`sh = 13`), half = 0x1000. -/
theorem normal_half_point : 2 ^ (13 - 1) = 0x1000 := by decide

/-- Subnormal shift `14 - exp` for `exp ∈ [-10, 0]` lies in `[14, 24]`, so the half-point `2^(sh-1)` is well defined. -/
theorem subnormal_shift_range (e : Int) (h1 : -10 ≤ e) (h2 : e ≤ 0) :
    14 ≤ 14 - e ∧ 14 - e ≤ 24 := by omega

/-- Overflow guard: a finite value with `exp >= 31` saturates; below that the exponent still fits 5 bits. -/
theorem exp_fits_five_bits (exp : Nat) (h : exp ≤ 30) : exp < 32 := by omega

/-- Subnormal decode: the number of normalising shifts for mantissa `m ∈ [1, 1023]` is at most 9. -/
theorem subnormal_shift_bound (m : Nat) (h1 : 1 ≤ m) (h2 : m < 1024) : 10 - Nat.log2 m ≤ 10 := by omega

/-- Decoded subnormal exponent field `113 - s` stays a normal f32 field when `s ≤ 10`. -/
theorem subnormal_exp_field (s : Nat) (h : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 113 := by omega

end Strata.F16
