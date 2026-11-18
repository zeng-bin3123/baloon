-- Baloon.F16BitsRounding: round-to-nearest-even step for the fp16 normal path.
-- Port of the rounding rule in include/ref/artifact/dequant.hpp (REF_GGUF_MAIN_DISABLED
-- region) and include/baloon/f16_bits.hpp (rne_round).
-- Scope: rneStep, rne_below_half, rne_above_half, rne_tie_even, rne_le_succ

namespace Baloon.F16BitsRounding

/-- One RNE rounding step: `h` is the truncated half-precision mantissa, `rem` the
    13-bit dropped remainder.  The halfway point is 4096.  The increment is 1 when
    `2 * rem + h % 2 > 2 * 4096` (above half, or exactly at the tie with h odd),
    and 0 otherwise (below half, or at the tie with h already even). -/
def rneStep (h rem : Nat) : Nat :=
  h + if 2 * rem + h % 2 > 2 * 4096 then 1 else 0

/-- Below the halfway point the truncated value is kept. -/
theorem rne_below_half (h rem : Nat) (hr : rem < 4096) : rneStep h rem = h := by
  unfold rneStep; split <;> omega

/-- Above the halfway point the value is bumped by one ulp. -/
theorem rne_above_half (h rem : Nat) (hr : 4096 < rem) : rneStep h rem = h + 1 := by
  unfold rneStep; split <;> omega

/-- Exactly at the tie the result is always even (ties-to-even rule). -/
theorem rne_tie_even (h rem : Nat) (hr : rem = 4096) : rneStep h rem % 2 = 0 := by
  unfold rneStep; split <;> omega

/-- Rounding never moves the value by more than one unit in the last place. -/
theorem rne_le_succ (h rem : Nat) : h ≤ rneStep h rem ∧ rneStep h rem ≤ h + 1 := by
  unfold rneStep; split <;> omega

-- § the inf/NaN versus finite-overflow split

/-- The top-level classification of the encoder: `none` means "normal or subnormal path".
    Raw exponent 255 is tested FIRST so inf/NaN never falls into the finite-overflow branch.
    The overflow threshold `30 < rawexp - 112` is equivalent to `rawexp ≥ 143`. -/
def encodeTop (sign rawexp man : Nat) : Option Nat :=
  if rawexp = 255 then some (sign + 31744 + (if 0 < man then 512 else 0))
  else if 30 < (rawexp : Int) - 127 + 15 then some (sign + 31744)
  else none

/-- A finite overflow (rawexp ≠ 255, rawexp ≥ 143) saturates to a signed infinity:
    the mantissa field is zero, so the result is not a NaN. -/
theorem finite_overflow_is_inf (sign rawexp man : Nat) (hs : sign = 0 ∨ sign = 32768)
    (h1 : rawexp ≠ 255) (h3 : 143 ≤ rawexp) :
    ∃ v, encodeTop sign rawexp man = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  rw [if_neg h1, if_pos (show 30 < (rawexp : Int) - 127 + 15 by omega)]

/-- A NaN input (rawexp = 255, man ≠ 0) keeps a non-zero mantissa field in the result. -/
theorem nan_stays_nan (sign man : Nat) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    ∃ v, encodeTop sign 255 man = some v ∧ v % 1024 ≠ 0 := by
  refine ⟨sign + 31744 + 512, ?_, by omega⟩
  unfold encodeTop
  rw [if_pos rfl, if_pos (show 0 < man by omega)]

/-- An infinity input (rawexp = 255, man = 0) stays an infinity: mantissa field is zero. -/
theorem inf_stays_inf (sign : Nat) (hs : sign = 0 ∨ sign = 32768) :
    ∃ v, encodeTop sign 255 0 = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  have hm : ¬ 0 < (0 : Nat) := by omega
  rw [if_pos rfl, if_neg hm, Nat.add_zero]

/-- Why the inf/NaN branch must be tested before the overflow branch: raw exponent 255 also
    satisfies the half-exponent overflow condition `(rawexp : Int) - 127 + 15 ≥ 31`. -/
theorem conflation_hazard (rawexp : Nat) (h : rawexp = 255) : (rawexp : Int) - 127 + 15 ≥ 31 := by
  omega

end Baloon.F16BitsRounding
