/-!
# MODULE: f16_bits
Structural facts behind the hand-written fp16 <-> fp32 bit conversions.

## Source semantics
fp16 = 1 sign | 5 exponent | 10 mantissa bits. The decoder maps normals to f32 exponent `ex + 112`, subnormals to
`113 - s` after `s` normalising shifts, and inf/NaN (`ex = 31`) to 255. The encoder saturates finite overflow to inf,
rounds to nearest even, and a mantissa carry may bump the exponent (30 -> 31 gives inf).

## Form template identifiers
- f32_from_f16 -> exF / manF / signF
- f16_from_f32 -> overflow threshold, carry, subnormal shift range
-/
namespace Strata.F16Bits

def exF (h : Nat) : Nat := h / 1024 % 32
def manF (h : Nat) : Nat := h % 1024
def signF (h : Nat) : Nat := h / 32768 % 2

/-- The three fields tile the 16 bits exactly. -/
theorem fields_recompose (h : Nat) (hh : h < 65536) : h = signF h * 32768 + exF h * 1024 + manF h := by
  unfold signF exF manF; omega

/-- Exponent field is 5 bits. -/
theorem exF_lt (h : Nat) : exF h < 32 := by unfold exF; omega
/-- Mantissa field is 10 bits. -/
theorem manF_lt (h : Nat) : manF h < 1024 := by unfold manF; omega
/-- Sign is one bit. -/
theorem signF_le (h : Nat) : signF h ≤ 1 := by unfold signF; omega

/-- Normal fp16 exponents land in the normal f32 exponent range. -/
theorem normal_exp_field {ex : Nat} (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by omega

/-- Subnormal decode: after `1..10` normalising shifts the f32 exponent field is a valid normal exponent. -/
theorem subnormal_exp_field {s : Nat} (h1 : 1 ≤ s) (h2 : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s ≤ 112 := by omega

/-- Mantissa shifted into f32 position occupies only the low 23 bits. -/
theorem man_shift_fits {man : Nat} (h : man < 1024) : man * 8192 < 8388608 := by omega

/-- Assembled normal f32 (sign excluded) stays below 2^31. -/
theorem normal_assembled_lt {ex man : Nat} (h1 : ex ≤ 30) (h2 : man < 1024) :
    (ex - 15 + 127) * 8388608 + man * 8192 < 2147483648 := by omega

/-- Encoder overflow test: `exp = rawexp - 127 + 15 ≥ 31` iff `rawexp ≥ 143`. -/
theorem overflow_threshold (r : Nat) : ((r : Int) - 127 + 15 ≥ 31) ↔ 143 ≤ r := by omega

/-- Largest finite fp16 is 0x7BFF; one more is +inf 0x7C00, which is what a mantissa carry from there produces. -/
theorem carry_to_inf : (30 * 1024 + 1023) + 1 = 31 * 1024 := by decide

/-- Subnormal encode shift range for `exp ∈ [-10, 0]`. -/
theorem subnormal_shift_range (e : Int) (h1 : -10 ≤ e) (h2 : e ≤ 0) : 14 ≤ 14 - e ∧ 14 - e ≤ 24 := by omega

/-- Subnormal encode: a 24-bit mantissa shifted right by 14..24 fits the 10-bit field. -/
theorem subnormal_quotient_fits {X sh : Nat} (hX : X < 16777216) (h1 : 14 ≤ sh) : X / 2 ^ sh < 1024 := by
  have h2 : 2 ^ 14 ≤ 2 ^ sh := Nat.pow_le_pow_right (by decide) h1
  have h3 : (2 : Nat) ^ 14 = 16384 := by decide
  apply Nat.div_lt_of_lt_mul
  omega

/-- Round-to-nearest-even: an exact tie with an even result does not round up. -/
theorem tie_even_no_increment (half h : Nat) (he : h % 2 = 0) : ¬ (half < half ∨ (half = half ∧ h % 2 = 1)) := by
  omega

/-- An exact tie with an odd result does round up. -/
theorem tie_odd_increments (half h : Nat) (ho : h % 2 = 1) : (half < half ∨ (half = half ∧ h % 2 = 1)) := by
  omega

end Strata.F16Bits
