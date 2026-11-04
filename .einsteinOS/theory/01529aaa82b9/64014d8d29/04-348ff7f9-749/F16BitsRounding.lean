/-!
# MODULE: f16_bits_rounding
SOURCES: include/strata/kernels/f16_bits.hpp, include/strata/artifact/dequant.hpp, src/kernels/cuda/ple.cu
Integer-level model of the hand-written fp16 <-> f32 and f32 -> bf16 conversions: round-to-nearest-even,
the inf/NaN versus finite-overflow split, subnormal normalisation and the exponent re-bias.

## Source semantics
* `f16_from_f32` rounds the 13 dropped mantissa bits to nearest, ties to even; the carry may bump the exponent.
* An f32 inf/NaN (raw exponent 255) stays inf/NaN; a FINITE value too large for fp16 saturates to inf (never NaN).
* `f32_from_f16` / `fp16_to_fp32` normalise subnormals by left-shifting until bit 10 is set; the f32 exponent
  field is then `113 - s`.  Unsigned wrap-around makes `ex - 15 + 127` correct only as a signed quantity.
* `bf16_bits` (ple.cu) adds `0x7FFF + lsb` and truncates: round-to-nearest-even on the upper 16 bits.

## Form template identifiers
f16_from_f32 -> round_f32_to_f16_bits ; f32_from_f16 -> widen_f16_bits ; bf16_bits -> round_f32_to_bf16_bits
rem -> dropped_mantissa_remainder ; rawexp -> f32_exponent_field ; man -> mantissa_field
-/

namespace Strata.F16Bits

-- § round-to-nearest-even on the dropped remainder

/-- One rounding step: `rem` is the dropped 13-bit remainder, `h` the truncated half-precision bits. -/
def rneStep (h rem : Nat) : Nat :=
  if rem > 4096 ∨ (rem = 4096 ∧ h % 2 = 1) then h + 1 else h

/-- Below the halfway point the truncated value is kept. -/
theorem rne_below_half (h rem : Nat) (hr : rem < 4096) : rneStep h rem = h := by
  unfold rneStep
  split <;> omega

/-- Above the halfway point the value is bumped by one. -/
theorem rne_above_half (h rem : Nat) (hr : 4096 < rem) : rneStep h rem = h + 1 := by
  unfold rneStep
  split <;> omega

/-- Exactly at the tie the result is always even (ties-to-even). -/
theorem rne_tie_even (h rem : Nat) (hr : rem = 4096) : rneStep h rem % 2 = 0 := by
  unfold rneStep
  split <;> omega

/-- Rounding never moves the value by more than one unit in the last place. -/
theorem rne_le_succ (h rem : Nat) : h ≤ rneStep h rem ∧ rneStep h rem ≤ h + 1 := by
  unfold rneStep
  split <;> omega

-- § the inf/NaN versus finite-overflow split

/-- The top-level classification of the encoder: `none` means "normal or subnormal path". -/
def encodeTop (sign rawexp man : Nat) : Option Nat :=
  if rawexp = 255 then some (sign + 31744 + (if man ≠ 0 then 512 else 0))
  else if (rawexp : Int) - 127 + 15 ≥ 31 then some (sign + 31744)
  else none

/-- A finite overflow saturates to a signed infinity: its mantissa field is zero, so it is not a NaN. -/
theorem finite_overflow_is_inf (sign rawexp man : Nat) (hs : sign = 0 ∨ sign = 32768)
    (h1 : rawexp ≠ 255) (h3 : 143 ≤ rawexp) :
    ∃ v, encodeTop sign rawexp man = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  rw [if_neg h1, if_pos (by omega)]

/-- A NaN input keeps a non-zero mantissa field. -/
theorem nan_stays_nan (sign man : Nat) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    ∃ v, encodeTop sign 255 man = some v ∧ v % 1024 ≠ 0 := by
  refine ⟨sign + 31744 + 512, ?_, by omega⟩
  unfold encodeTop
  rw [if_pos rfl, if_pos hm]

/-- An infinity input stays an infinity. -/
theorem inf_stays_inf (sign : Nat) (hs : sign = 0 ∨ sign = 32768) :
    ∃ v, encodeTop sign 255 0 = some v ∧ v % 1024 = 0 := by
  refine ⟨sign + 31744, ?_, by omega⟩
  unfold encodeTop
  have hn : ¬ ((0 : Nat) ≠ 0) := by omega
  rw [if_pos rfl, if_neg hn]

/-- Why the old test `exp >= 31` conflated the two cases: raw exponent 255 also satisfies it. -/
theorem conflation_hazard (rawexp : Nat) (h : rawexp = 255) : (rawexp : Int) - 127 + 15 ≥ 31 := by
  omega

-- § subnormal normalisation in the decoder

/-- The normalising loop of the decoder: shift left until bit 10 is set, counting shifts. -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | fuel + 1, m, s => if m < 1024 then normShift fuel (2 * m) (s + 1) else (m, s)

set_option maxRecDepth 100000 in
/-- For every non-zero subnormal mantissa the loop ends with bit 10 set after at most 10 shifts,
and the shifted value is `m * 2^s`. -/
theorem normShift_spec :
    ∀ m < 1024, 0 < m →
      1024 ≤ (normShift 10 m 0).1 ∧ (normShift 10 m 0).1 < 2048 ∧ (normShift 10 m 0).2 ≤ 10 ∧
        (normShift 10 m 0).1 = m * 2 ^ (normShift 10 m 0).2 := by
  decide

/-- The re-bias constant is 113. -/
theorem subnormal_bias : 127 - 15 + 1 = 113 := by decide

/-- After at most 10 shifts the f32 exponent field `113 - s` is a valid normal exponent. -/
theorem subnormal_exp_field (s : Nat) (hs : s ≤ 10) : 103 ≤ 113 - s ∧ 113 - s < 255 := by
  omega

/-- For normal halves the f32 exponent field is `ex + 112`, computed as a signed quantity. -/
theorem normal_exp_field (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    ((ex : Int) - 15 + 127 = (ex : Int) + 112) ∧ 113 ≤ (ex : Int) + 112 ∧ (ex : Int) + 112 ≤ 142 := by
  omega

/-- Evaluating `ex - 15 + 127` in truncated Nat arithmetic would be wrong; the C++ relies on unsigned wrap. -/
theorem nat_truncation_hazard : (1 - 15 + 127 : Nat) ≠ 1 + 112 := by decide

-- § bf16 rounding (ple.cu `bf16_bits`)

/-- `(i + ((i >> 16) & 1) + 0x7FFF) >> 16` as Nat arithmetic (inputs below 2^32 without carry-out). -/
def bf16Round (i : Nat) : Nat := (i + (i / 65536) % 2 + 32767) / 65536

/-- Values that are already bf16 are unchanged. -/
theorem bf16_exact (q : Nat) : bf16Round (65536 * q) = q := by
  unfold bf16Round
  omega

/-- Below the halfway point the result truncates. -/
theorem bf16_below_half (i : Nat) (h : i % 65536 < 32768) : bf16Round i = i / 65536 := by
  unfold bf16Round
  omega

/-- Above the halfway point the result rounds up. -/
theorem bf16_above_half (i : Nat) (h : 32768 < i % 65536) : bf16Round i = i / 65536 + 1 := by
  unfold bf16Round
  omega

/-- At the exact tie the result is even. -/
theorem bf16_tie_even (i : Nat) (h : i % 65536 = 32768) : bf16Round i % 2 = 0 := by
  unfold bf16Round
  obtain ⟨q, hq⟩ : ∃ q, i = 65536 * q + 32768 := ⟨i / 65536, by omega⟩
  subst hq
  have h1 : (65536 * q + 32768) / 65536 = q := by omega
  rw [h1]
  have h2 : (65536 * q + 32768 + q % 2 + 32767) / 65536 = q + q % 2 := by omega
  rw [h2]
  omega

end Strata.F16Bits
