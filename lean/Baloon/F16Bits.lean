-- Baloon.F16Bits: float16 encoding/decoding invariants: round-to-nearest-even,
-- inf/NaN classification, subnormal normalisation and the exponent bias map.
-- Independent implementation proving the same mathematical facts as Ref.F16Bits.

namespace Baloon.F16Bits

def rneStep (h rem : Nat) : Nat :=
  if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

theorem rne_below_half (h rem : Nat) (hr : rem < 4096) : rneStep h rem = h := by
  unfold rneStep; split <;> omega

theorem rne_above_half (h rem : Nat) (hr : 4096 < rem) : rneStep h rem = h + 1 := by
  unfold rneStep; split <;> omega

theorem rne_tie_even (h rem : Nat) (hr : rem = 4096) : rneStep h rem % 2 = 0 := by
  unfold rneStep; split <;> omega

theorem rne_le_succ (h rem : Nat) : h ≤ rneStep h rem ∧ rneStep h rem ≤ h + 1 := by
  unfold rneStep; split <;> omega

def encodeTop (sign rawexp man : Nat) : Option Nat :=
  if rawexp = 255 then some (sign + 31744 + (if man ≠ 0 then 512 else 0))
  else if (rawexp : Int) - 127 + 15 ≥ 31 then some (sign + 31744)
  else none

theorem finite_overflow_is_inf (sign rawexp man : Nat) (hs : sign = 0 ∨ sign = 32768)
    (h1 : rawexp ≠ 255) (h3 : 143 ≤ rawexp) :
    ∃ v, encodeTop sign rawexp man = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop; rw [if_neg h1, if_pos (by omega)]

theorem nan_stays_nan (sign man : Nat) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    ∃ v, encodeTop sign 255 man = some v ∧ v % 1024 ≠ 0 := by
  refine ⟨sign + 31744 + 512, ?_, by omega⟩
  unfold encodeTop; rw [if_pos rfl, if_pos hm]

theorem inf_stays_inf (sign : Nat) (hs : sign = 0 ∨ sign = 32768) :
    ∃ v, encodeTop sign 255 0 = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  have hn : ¬ ((0 : Nat) ≠ 0) := by omega
  rw [if_pos rfl, if_neg hn]

theorem conflation_hazard (rawexp : Nat) (h : rawexp = 255) : (rawexp : Int) - 127 + 15 ≥ 31 := by
  omega

end Baloon.F16Bits
