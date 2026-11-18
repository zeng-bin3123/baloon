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

end Baloon.F16BitsRounding
