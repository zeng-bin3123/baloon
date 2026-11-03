/-!
# MODULE: f16_conversion
Bit-level contracts of the hand-written fp32 -> fp16 (round-to-nearest-even) and fp16 -> fp32 conversions.

## Source semantics
The normal regime builds `sign | exp<<10 | man>>13` and rounds with a 13-bit sticky remainder; a carry may
bump the exponent (up to inf) but never reaches the sign. The subnormal regime shifts by `14 - exp`
(14..24 for exp in -10..0) and the shifted mantissa fits in 10 bits. The exact decoder normalizes
subnormals by shifting until bit 10 is set (at most 10 shifts), giving exponent field `113 - s`.

## Form template identifiers
- `h0`, `roundNormal` <- the normal-regime pack and its `++h` rounding
- `norm` <- the `while ((m & 0x400u) == 0) { m <<= 1; ++s; }` loop (guard stated as `m < 1024` for m < 2048)
-/

namespace Strata.F16Conversion

-- § fp32 -> fp16 normal regime

/-- Packed sign/exponent/truncated-mantissa before rounding. -/
def h0 (s e m : Nat) : Nat := s + e * 1024 + m / 8192

/-- Round-to-nearest-even on the 13 dropped bits. -/
def roundNormal (s e m : Nat) : Nat :=
  if m % 8192 > 4096 ∨ (m % 8192 = 4096 ∧ h0 s e m % 2 = 1) then h0 s e m + 1 else h0 s e m

/-- Rounding stays below the infinity pattern plus sign (a carry can at most reach inf, never the sign bit). -/
theorem roundNormal_le (s e m : Nat) (hs : s = 0 ∨ s = 32768) (he1 : 1 ≤ e) (he2 : e ≤ 30)
    (hm : m < 8388608) : roundNormal s e m ≤ s + 31744 := by
  unfold roundNormal h0
  repeat' split
  all_goals omega

/-- A tie with an even truncated mantissa rounds down (to even). -/
theorem roundNormal_tie_even (s e m : Nat) (hr : m % 8192 = 4096) (hev : h0 s e m % 2 = 0) :
    roundNormal s e m = h0 s e m := by
  unfold roundNormal
  split
  · omega
  · rfl

/-- A tie with an odd truncated mantissa rounds up (to even). -/
theorem roundNormal_tie_odd (s e m : Nat) (hr : m % 8192 = 4096) (hod : h0 s e m % 2 = 1) :
    roundNormal s e m = h0 s e m + 1 := by
  unfold roundNormal
  split
  · rfl
  · omega

/-- Strictly above the half-way point always rounds up. -/
theorem roundNormal_above_half (s e m : Nat) (hr : m % 8192 > 4096) :
    roundNormal s e m = h0 s e m + 1 := by
  unfold roundNormal
  split
  · rfl
  · omega

/-- Below the half-way point always rounds down. -/
theorem roundNormal_below_half (s e m : Nat) (hr : m % 8192 < 4096) :
    roundNormal s e m = h0 s e m := by
  unfold roundNormal
  split
  · omega
  · rfl

-- § fp32 -> fp16 subnormal regime

/-- For `-10 ≤ exp ≤ 0` the shift `14 - exp` lies in `[14, 24]`, a legal 32-bit shift. -/
theorem subnormal_shift_range (e : Int) (h1 : -10 ≤ e) (h2 : e ≤ 0) : 14 ≤ 14 - e ∧ 14 - e ≤ 24 := by
  omega

/-- The shifted 24-bit mantissa fits in 10 bits, so the `& 0x3FF` mask loses nothing. -/
theorem subnormal_quotient_lt (man sh : Nat) (hm : man < 16777216) (hsh : 14 ≤ sh) :
    man / 2 ^ sh < 1024 := by
  have h2 : 2 ^ 14 ≤ 2 ^ sh := Nat.pow_le_pow_right (by decide) hsh
  have h3 := Nat.mul_le_mul_left 1024 h2
  simp only [Nat.reducePow, Nat.reduceMul] at h3
  apply Nat.div_lt_of_lt_mul
  omega

/-- Exponents below -10 flush to a signed zero (the largest kept shift is 24, below 32). -/
theorem subnormal_cutoff (e : Int) (h : e < -10) : 14 - e > 24 := by omega

-- § fp16 -> fp32 exact decode

/-- The subnormal normalization loop (`m < 1024` is the bit-10-clear test for `m < 2048`). -/
def norm : Nat → Nat → Nat → Nat × Nat
  | m, s, 0 => (m, s)
  | m, s, f + 1 => if m < 1024 then norm (2 * m) (s + 1) f else (m, s)

/-- With enough fuel the normalized mantissa has bit 10 set and stays below 2048. -/
theorem norm_range : ∀ f m s, 0 < m → m < 2048 → 1024 ≤ m * 2 ^ f →
    1024 ≤ (norm m s f).1 ∧ (norm m s f).1 < 2048 := by
  intro f
  induction f with
  | zero =>
    intro m s hm1 hm2 hm3
    simp only [norm]
    simp at hm3
    omega
  | succ f ih =>
    intro m s hm1 hm2 hm3
    unfold norm
    split
    · rename_i hlt
      have e : 2 * m * 2 ^ f = m * 2 ^ (f + 1) := by
        rw [Nat.pow_succ]
        simp only [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]
      exact ih (2 * m) (s + 1) (by omega) (by omega) (by omega)
    · rename_i hge
      exact ⟨by omega, by omega⟩

/-- The shift count grows by at most one per iteration, so a 10-bit subnormal needs at most 10 shifts. -/
theorem norm_shift_le : ∀ f m s, (norm m s f).2 ≤ s + f := by
  intro f
  induction f with
  | zero =>
    intro m s
    simp [norm]
  | succ f ih =>
    intro m s
    unfold norm
    split
    · have := ih (2 * m) (s + 1)
      omega
    · omega

/-- The f32 exponent field `113 - s` is a legal normal exponent for `1 ≤ s ≤ 10`. -/
theorem subnormal_exponent_field (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 10) : 1 ≤ 113 - s ∧ 113 - s ≤ 254 := by
  omega

/-- A normal fp16 exponent field `ex ∈ [1,30]` maps to an f32 exponent in `[113, 142]`. -/
theorem normal_exponent_field (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by omega

end Strata.F16Conversion
