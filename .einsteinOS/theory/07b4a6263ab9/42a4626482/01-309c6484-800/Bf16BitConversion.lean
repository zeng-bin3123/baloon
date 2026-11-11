/-!
# MODULE: bf16_bit_conversion
SOURCES: include/strata/kernels/bf16_bits.hpp, include/strata/artifact/dequant.hpp, tools/canonical_xcheck.py
Model of bf16 ↔ f32 over raw bit patterns: the widening shift (`f32_from_bf16`, `bf16_to_fp32`,
`to_p32_bf16`) and ggml's round-to-nearest-even narrowing `(i + ((i >> 16) & 1) + 0x7FFF) >> 16`.

## Source semantics
* a bf16 is the top 16 bits of an f32, so widening is `h << 16` and is exact.
* narrowing adds `0x7FFF` plus the LOWEST KEPT bit (bit 16) and truncates: below half rounds down,
  above half rounds up, an exact tie rounds to the even neighbour, all in uint32 arithmetic.
* using bit 15 instead of bit 16 differs only on exact ties.
* narrowing after widening is the identity on all 65,536 bf16 patterns.

## Form template identifiers
bf16_from_f32 → `bf16FromF32`; f32_from_bf16 / bf16_to_fp32 / to_p32_bf16 → `f32FromBf16`;
the plausible bit-15 mistake → `bf16FromF32Bit15`; `low16` = `i & 0xFFFF`, `top16` = `i >> 16`.
-/

namespace Strata.Bf16Bits

-- § definitions

/-- bf16 bits → f32 bits: a left shift by 16. -/
def f32FromBf16 (h : Nat) : Nat := h <<< 16

/-- f32 bits → bf16 bits, round-to-nearest-even, with the uint32 wraparound made explicit. -/
def bf16FromF32 (i : Nat) : Nat := (i + i / 65536 % 2 + 0x7FFF) % 4294967296 / 65536 % 65536

/-- The variant that reads bit 15 instead of bit 16. -/
def bf16FromF32Bit15 (i : Nat) : Nat := (i + i / 32768 % 2 + 0x7FFF) % 4294967296 / 65536 % 65536

/-- The C++ `(i >> 16) & 1` is `i / 65536 % 2`. -/
theorem lowest_kept_bit (i : Nat) : (i >>> 16) &&& 1 = i / 65536 % 2 := by
  have h : (1 : Nat) = 2 ^ 1 - 1 := rfl
  rw [h, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]

-- § widening

/-- Widening is multiplication by 2^16: the low 16 bits of the f32 are zero. -/
theorem widen_eq (h : Nat) : f32FromBf16 h = h * 65536 := by
  unfold f32FromBf16; rw [Nat.shiftLeft_eq]

/-- Widening never changes the top half. -/
theorem widen_low_zero (h : Nat) : f32FromBf16 h % 65536 = 0 := by
  rw [widen_eq]; omega

-- § rounding regimes

/-- Below the top bf16 pattern the uint32 sum never wraps. -/
theorem no_wrap (i : Nat) (hi : i < 0xFFFF0000) :
    (i + i / 65536 % 2 + 0x7FFF) % 4294967296 = i + i / 65536 % 2 + 0x7FFF :=
  Nat.mod_eq_of_lt (by omega)

/-- Split an f32 pattern into its top half `q` and low half `r`. -/
theorem split (i : Nat) : ∃ q r, r < 65536 ∧ i = 65536 * q + r :=
  ⟨i / 65536, i % 65536, Nat.mod_lt _ (by decide), (Nat.div_add_mod i 65536).symm⟩

/-- The core computation once the input is split and no wrap occurs. -/
theorem core (q r b : Nat) (hb : b ≤ 1) (hq : q < 65536) :
    (65536 * q + r + b + 0x7FFF) / 65536 % 65536 =
      (q + (r + b + 0x7FFF) / 65536) % 65536 := by
  omega

/-- Below half: the low 16 bits are dropped (any input, since this case cannot wrap). -/
theorem round_down (i : Nat) (hi : i < 4294967296) (hl : i % 65536 < 0x8000) :
    bf16FromF32 i = i / 65536 := by
  obtain ⟨q, r, hr, rfl⟩ := split i
  have hq : (65536 * q + r) / 65536 = q := by omega
  have hr' : (65536 * q + r) % 65536 = r := by omega
  rw [hr'] at hl
  unfold bf16FromF32
  rw [hq, Nat.mod_eq_of_lt (by omega : 65536 * q + r + q % 2 + 0x7FFF < 4294967296)]
  have hb : q % 2 ≤ 1 := by omega
  rw [core q r (q % 2) hb (by omega)]
  have : (r + q % 2 + 0x7FFF) / 65536 = 0 := by omega
  rw [this]; omega

/-- Above half: the top half is incremented. (The all-ones top half is a NaN and wraps.) -/
theorem round_up (i : Nat) (hi : i < 0xFFFF0000) (hl : 0x8000 < i % 65536) :
    bf16FromF32 i = i / 65536 + 1 := by
  obtain ⟨q, r, hr, rfl⟩ := split i
  have hq : (65536 * q + r) / 65536 = q := by omega
  have hr' : (65536 * q + r) % 65536 = r := by omega
  rw [hr'] at hl
  unfold bf16FromF32
  rw [hq, Nat.mod_eq_of_lt (by omega : 65536 * q + r + q % 2 + 0x7FFF < 4294967296)]
  rw [core q r (q % 2) (by omega) (by omega)]
  have : (r + q % 2 + 0x7FFF) / 65536 = 1 := by omega
  rw [this]; omega

/-- An exact tie rounds to the even neighbour. -/
theorem round_tie_even (i : Nat) (hi : i < 0xFFFF0000) (hl : i % 65536 = 0x8000) :
    bf16FromF32 i = i / 65536 + i / 65536 % 2 ∧ bf16FromF32 i % 2 = 0 := by
  obtain ⟨q, r, hr, rfl⟩ := split i
  have hq : (65536 * q + r) / 65536 = q := by omega
  have hr' : (65536 * q + r) % 65536 = r := by omega
  rw [hr'] at hl
  subst hl
  unfold bf16FromF32
  rw [hq, Nat.mod_eq_of_lt (by omega : 65536 * q + 0x8000 + q % 2 + 0x7FFF < 4294967296)]
  rw [core q 0x8000 (q % 2) (by omega) (by omega)]
  have : (0x8000 + q % 2 + 0x7FFF) / 65536 = q % 2 := by omega
  rw [this]; omega

-- § round trip

/-- Narrowing after widening is the identity on every bf16 bit pattern. -/
theorem roundtrip (h : Nat) (hh : h < 65536) : bf16FromF32 (f32FromBf16 h) = h := by
  rw [widen_eq]
  have hlow : h * 65536 % 65536 < 0x8000 := by omega
  have hi : h * 65536 < 4294967296 := by omega
  rw [round_down _ hi hlow]
  omega

-- § the bit-15 mistake

/-- Away from ties the two variants agree when bit 15 is clear... -/
theorem bit15_agrees_below (i : Nat) (hi : i < 4294967296) (hl : i % 65536 < 0x8000) :
    bf16FromF32Bit15 i = bf16FromF32 i := by
  rw [round_down i hi hl]
  obtain ⟨q, r, hr, rfl⟩ := split i
  have hq : (65536 * q + r) / 65536 = q := by omega
  have hr' : (65536 * q + r) % 65536 = r := by omega
  rw [hr'] at hl
  rw [hq]
  unfold bf16FromF32Bit15
  have hb : (65536 * q + r) / 32768 % 2 = 0 := by omega
  rw [hb, Nat.mod_eq_of_lt (by omega : 65536 * q + r + 0 + 0x7FFF < 4294967296)]
  rw [core q r 0 (by omega) (by omega)]
  have : (r + 0 + 0x7FFF) / 65536 = 0 := by omega
  rw [this]; omega

/-- ...but on an exact tie with an even top half the bit-15 variant rounds the wrong way. -/
theorem bit15_wrong_on_tie : bf16FromF32 0x3F808000 = 0x3F80 ∧ bf16FromF32Bit15 0x3F808000 = 0x3F81 := by
  decide

end Strata.Bf16Bits
