/-!
# MODULE: fp16_bit_conversion
Round-to-nearest-even binary32 -> binary16 conversion and exact binary16 -> binary32 conversion, both
expressed purely on raw bit patterns (modelled on `Nat`, with the 32-bit / 16-bit widths implicit in the masks).

## Source semantics
* `f16_from_f32` : an f32 inf/NaN (raw exponent 255) stays inf/NaN (a NaN gets the quiet bit 0x200); a FINITE
  value too large for fp16 saturates to a signed inf (0x7C00) and is never turned into a NaN; values below
  half the smallest subnormal flush to a signed zero; subnormal results restore the implicit 1 and shift, with
  round-to-nearest-even at the moved rounding point; normal results round from 13 dropped bits, and the
  rounding carry may bump the exponent (up to inf).
* `f32_from_f16` : exact, no rounding. Zero keeps its sign, subnormals are normalised (never flushed),
  exponent 31 maps to f32 exponent 255 with the mantissa shifted left 13.
* Hence every non-NaN fp16 value round-trips bit-exactly, and NaNs stay NaNs.

## Form template identifiers
* `f16_from_f32`  -> `narrowConv`   (wide float bits -> narrow float bits, RNE)
* `f32_from_f16`  -> `widenConv`    (narrow float bits -> wide float bits, exact)
* `sign` / `rawexp` / `man` -> `signBits` / `expField` / `fracBits`
* `0x7C00` -> `narrowInf`, `0x200` -> `narrowQuietBit`, `0x7F800000` -> `wideInf`
-/

namespace Fp16Bits

-- § definitions

/-- Round-to-nearest-even f32 -> fp16 on raw bits. `exp = rawexp - 112` of the source is kept in `Nat`
form: `exp >= 31` is `rawexp >= 143`, `exp <= 0` is `rawexp <= 112`, `exp < -10` is `rawexp < 102`,
the subnormal shift `14 - exp` is `126 - rawexp`. -/
def f16FromF32 (x : Nat) : Nat :=
  let sign := (x >>> 16) &&& 0x8000
  let rawexp := (x >>> 23) &&& 0xFF
  let man := x &&& 0x7FFFFF
  if rawexp = 255 then sign ||| 0x7C00 ||| (if man ≠ 0 then 0x200 else 0)
  else if 143 ≤ rawexp then sign ||| 0x7C00
  else if rawexp ≤ 112 then
    if rawexp < 102 then sign
    else
      let man' := man ||| 0x800000
      let sh := 126 - rawexp
      let h := (man' >>> sh) &&& 0x3FF
      let rem := man' &&& (2 ^ sh - 1)
      let h' := if rem > 2 ^ (sh - 1) ∨ (rem = 2 ^ (sh - 1) ∧ h &&& 1 = 1) then h + 1 else h
      sign ||| h'
  else
    let h := sign ||| ((rawexp - 112) <<< 10) ||| (man >>> 13)
    let rem := man &&& 0x1FFF
    if rem > 0x1000 ∨ (rem = 0x1000 ∧ h &&& 1 = 1) then h + 1 else h

/-- Normalisation loop of the subnormal case: shift left until bit 10 is set, counting shifts. -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | fuel + 1, m, s => if m &&& 0x400 = 0 then normShift fuel (m <<< 1) (s + 1) else (m, s)

/-- Exact fp16 -> f32 on raw bits. -/
def f32FromF16 (h : Nat) : Nat :=
  let sign := (h &&& 0x8000) <<< 16
  let ex := (h >>> 10) &&& 0x1F
  let man := h &&& 0x3FF
  if ex = 0 then
    if man = 0 then sign
    else
      let r := normShift 16 man 0
      sign ||| ((113 - r.2) <<< 23) ||| ((r.1 &&& 0x3FF) <<< 13)
  else if ex = 31 then sign ||| 0x7F800000 ||| (man <<< 13)
  else sign ||| ((ex + 112) <<< 23) ||| (man <<< 13)

/-- A bit pattern is an fp16 NaN. -/
def IsNaN16 (h : Nat) : Prop := (h >>> 10) &&& 0x1F = 31 ∧ h &&& 0x3FF ≠ 0

instance (h : Nat) : Decidable (IsNaN16 h) := by unfold IsNaN16; infer_instance

-- § narrowing: case lemmas

/-- An f32 inf/NaN stays an inf/NaN: sign kept, quiet bit set iff the mantissa is nonzero. -/
theorem f16FromF32_special (x : Nat) (hr : (x >>> 23) &&& 0xFF = 255) :
    f16FromF32 x = ((x >>> 16) &&& 0x8000) ||| 0x7C00 ||| (if x &&& 0x7FFFFF ≠ 0 then 0x200 else 0) := by
  simp [f16FromF32, hr]

/-- A FINITE value too large for fp16 saturates to a signed inf (the NaN-conflation bug is excluded). -/
theorem f16FromF32_overflow_inf (x : Nat) (hne : (x >>> 23) &&& 0xFF ≠ 255)
    (hge : 143 ≤ (x >>> 23) &&& 0xFF) :
    f16FromF32 x = ((x >>> 16) &&& 0x8000) ||| 0x7C00 := by
  simp [f16FromF32, hne, hge]

/-- Values below half the smallest subnormal underflow to a signed zero. -/
theorem f16FromF32_underflow_zero (x : Nat) (hlt : (x >>> 23) &&& 0xFF < 102) :
    f16FromF32 x = (x >>> 16) &&& 0x8000 := by
  have h1 : (x >>> 23) &&& 0xFF ≠ 255 := by omega
  have h2 : ¬ (143 ≤ (x >>> 23) &&& 0xFF) := by omega
  have h3 : (x >>> 23) &&& 0xFF ≤ 112 := by omega
  simp [f16FromF32, h1, h2, h3, hlt]

-- § widening: case lemmas

/-- Signed zero widens to signed zero. -/
theorem f32FromF16_zero : f32FromF16 0 = 0 ∧ f32FromF16 0x8000 = 0x80000000 := by
  decide

/-- The smallest fp16 subnormal 0x0001 widens to 2^-24 (0x33800000), not to 0.0. -/
theorem f32FromF16_min_subnormal : f32FromF16 0x0001 = 0x33800000 := by
  decide

/-- Exponent 31 maps to f32 exponent 255 (inf/NaN) with the mantissa shifted left 13. -/
theorem f32FromF16_inf_nan (h : Nat) (hex : (h >>> 10) &&& 0x1F = 31) :
    f32FromF16 h = ((h &&& 0x8000) <<< 16) ||| 0x7F800000 ||| ((h &&& 0x3FF) <<< 13) := by
  simp [f32FromF16, hex]

-- § concrete vectors (rounding regimes)

/-- Exactly 1.0 converts to 0x3C00 (the failure mode seen with `__float2half` was 0x3C01). -/
theorem f16FromF32_one : f16FromF32 0x3F800000 = 0x3C00 := by decide

/-- Normal-range tie at an even mantissa rounds down; at an odd mantissa rounds up. -/
theorem f16FromF32_tie_even :
    f16FromF32 0x3F801000 = 0x3C00 ∧ f16FromF32 0x3F803000 = 0x3C02 := by decide

/-- Overflow regime: 65504 is the largest finite fp16, 65520 is the tie that rounds up to inf,
65536 and 1e30 saturate to inf (not NaN), -1e30 keeps its sign. -/
theorem f16FromF32_overflow_vectors :
    f16FromF32 0x477FE000 = 0x7BFF ∧ f16FromF32 0x477FF000 = 0x7C00 ∧
    f16FromF32 0x47800000 = 0x7C00 ∧ f16FromF32 0x7149F2CA = 0x7C00 ∧
    f16FromF32 0xF149F2CA = 0xFC00 := by decide

/-- Subnormal regime: 2^-24 is 0x0001, 2^-25 is a tie that rounds to even zero, just above it rounds
to 0x0001, 2^-26 flushes to zero, and -0.0 stays -0.0. -/
theorem f16FromF32_subnormal_vectors :
    f16FromF32 0x33800000 = 0x0001 ∧ f16FromF32 0x33000000 = 0x0000 ∧
    f16FromF32 0x33000001 = 0x0001 ∧ f16FromF32 0x32800000 = 0x0000 ∧
    f16FromF32 0x80000000 = 0x8000 := by decide

/-- f32 infinities and NaNs: +inf, -inf, a quiet NaN. -/
theorem f16FromF32_inf_nan_vectors :
    f16FromF32 0x7F800000 = 0x7C00 ∧ f16FromF32 0xFF800000 = 0xFC00 ∧
    f16FromF32 0x7FC00000 = 0x7E00 := by decide

-- § exhaustive properties over all 65,536 fp16 inputs

/-- Every non-NaN fp16 value survives widen-then-narrow bit-exactly (all 65,536 inputs checked). -/
theorem roundtrip_all :
    ∀ h : Fin 65536, ¬ IsNaN16 h.val → f16FromF32 (f32FromF16 h.val) = h.val := by
  have key : (List.range 65536).all (fun h =>
      decide (¬ IsNaN16 h → f16FromF32 (f32FromF16 h) = h)) = true := by
    decide +kernel
  intro h hn
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt)) hn

/-- NaNs stay NaNs through widen-then-narrow (the payload is not preserved, the class is). -/
theorem roundtrip_nan_stays_nan :
    ∀ h : Fin 65536, IsNaN16 h.val → IsNaN16 (f16FromF32 (f32FromF16 h.val)) := by
  have key : (List.range 65536).all (fun h =>
      decide (IsNaN16 h → IsNaN16 (f16FromF32 (f32FromF16 h)))) = true := by
    decide +kernel
  intro h hn
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt)) hn

/-- Widening preserves the sign bit and never flushes a nonzero value to zero. -/
theorem widen_sign_and_nonzero :
    ∀ h : Fin 65536, (f32FromF16 h.val >>> 31 = h.val >>> 15) ∧
      (h.val &&& 0x7FFF ≠ 0 → f32FromF16 h.val &&& 0x7FFFFFFF ≠ 0) := by
  have key : (List.range 65536).all (fun h =>
      decide ((f32FromF16 h >>> 31 = h >>> 15) ∧
        (h &&& 0x7FFF ≠ 0 → f32FromF16 h &&& 0x7FFFFFFF ≠ 0))) = true := by
    decide +kernel
  intro h
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt))

end Fp16Bits
