/-!
# MODULE: half_precision_bit_conversion
SOURCES: include/strata/kernels/f16_bits.hpp, include/strata/kernels/bf16_bits.hpp
Bit-exact f32 <-> fp16 and f32 <-> bf16 conversions used by the CPU/GPU kernels and their tests.
Every shift and mask is modelled with `/` and `%` by powers of two over `Nat`, which is exactly
what the C++ does on unsigned operands.

## Source semantics
* fp16 bits split as sign (bit 15), exponent (bits 10..14), mantissa (bits 0..9); the split is lossless.
* `f16_from_f32` rounds to nearest, ties to even, from the 13 discarded mantissa bits; a mantissa carry
  may bump the exponent field.  An f32 inf/NaN (raw exponent 255) stays inf/NaN; a FINITE value whose
  rebiased exponent is >= 31 saturates to exactly +-inf (mantissa field 0, never a NaN).
* `f32_from_f16` is exact; a subnormal fp16 is normalised by shifting until bit 10 is set, which needs
  at most 10 shifts and gives the f32 exponent field `113 - s` (always a normal f32).
* `bf16_from_f32` adds `((i >> 16) & 1) + 0x7FFF` and truncates: round to nearest, ties to even on the
  lowest KEPT bit (bit 16).  `f32_from_bf16` is a 16-bit left shift and exact, so round trips are exact.
* Observed: the bf16 encoder has no NaN special case (ggml's has one), so the NaN 0x7F800001 encodes as
  the infinity 0x7F80, and the NaN 0xFFFFFFFF wraps to +0.

## Form template identifiers
f16_from_f32 -> encodeHalf, f32_from_f16 -> decodeHalf, bf16_from_f32 -> bf16Encode,
f32_from_bf16 -> bf16Decode, rem/0x1000 tie test -> rneBump, subnormal shift loop -> normShift
-/

namespace Strata.HalfBits

-- § fp16 field layout

/-- Sign bit of an fp16 word (`h >> 15`). -/
def f16Sign (h : Nat) : Nat := h / 32768 % 2
/-- Exponent field of an fp16 word (`(h >> 10) & 0x1F`). -/
def f16Exp (h : Nat) : Nat := h / 1024 % 32
/-- Mantissa field of an fp16 word (`h & 0x3FF`). -/
def f16Man (h : Nat) : Nat := h % 1024

/-- The three fp16 fields recompose the original 16-bit word: the split is lossless. -/
theorem f16_fields_recompose (h : Nat) (hh : h < 65536) :
    h = f16Sign h * 32768 + f16Exp h * 1024 + f16Man h := by
  unfold f16Sign f16Exp f16Man
  omega

/-- Each fp16 field is within its declared width. -/
theorem f16_fields_bounded (h : Nat) :
    f16Sign h < 2 ∧ f16Exp h < 32 ∧ f16Man h < 1024 := by
  unfold f16Sign f16Exp f16Man
  omega

-- § round to nearest, ties to even

/-- The rounding increment of `f16_from_f32`: bump when the remainder is above half, or exactly
half and the kept value is odd. -/
def rneBump (q r half : Nat) : Nat :=
  if r > half ∨ (r = half ∧ q % 2 = 1) then 1 else 0

/-- Normal-path rounding of a 23-bit mantissa to 10 bits: keep `man >> 13`, round by `man & 0x1FFF`. -/
def roundMan13 (man : Nat) : Nat := man / 8192 + rneBump (man / 8192) (man % 8192) 4096

/-- The rounded mantissa is within half a unit (4096 of 8192) of the exact value. -/
theorem roundMan13_error (man : Nat) :
    roundMan13 man * 8192 ≤ man + 4096 ∧ man ≤ roundMan13 man * 8192 + 4096 := by
  unfold roundMan13 rneBump
  split <;> omega

/-- An exact tie rounds to an even result. -/
theorem roundMan13_tie_even (man : Nat) (ht : man % 8192 = 4096) :
    roundMan13 man % 2 = 0 := by
  unfold roundMan13 rneBump
  split <;> omega

/-- Below half rounds down, above half rounds up. -/
theorem roundMan13_direction (man : Nat) :
    (man % 8192 < 4096 → roundMan13 man = man / 8192) ∧
    (man % 8192 > 4096 → roundMan13 man = man / 8192 + 1) := by
  unfold roundMan13 rneBump
  constructor <;> intro h <;> split <;> omega

/-- Generic rounding by a divisor `k = 2*half` (the subnormal path, where the rounding point moves). -/
def roundDiv (x half : Nat) : Nat := x / (2 * half) + rneBump (x / (2 * half)) (x % (2 * half)) half

/-- The subnormal rounding is also within half a unit, for every shift amount. -/
theorem roundDiv_error (x half : Nat) (hp : 0 < half) :
    roundDiv x half * (2 * half) ≤ x + half ∧ x ≤ roundDiv x half * (2 * half) + half := by
  have hdm := Nat.div_add_mod x (2 * half)
  have hlt : x % (2 * half) < 2 * half := Nat.mod_lt _ (by omega)
  unfold roundDiv rneBump
  generalize x / (2 * half) = q at *
  generalize x % (2 * half) = r at *
  have hc : q * (2 * half) = 2 * half * q := Nat.mul_comm _ _
  split
  · rw [Nat.add_mul, Nat.one_mul, hc]
    omega
  · rw [Nat.add_mul, Nat.zero_mul, hc]
    omega

/-- A mantissa carry out of the 10-bit field moves into the exponent: incrementing a word whose
mantissa field is all ones zeroes the mantissa and adds one to the exponent. -/
theorem carry_bumps_exponent (s e : Nat) (hs : s < 2) (he : e < 31) :
    let h := s * 32768 + e * 1024 + 1023
    f16Man (h + 1) = 0 ∧ f16Exp (h + 1) = e + 1 ∧ f16Sign (h + 1) = s := by
  unfold f16Man f16Exp f16Sign
  omega

-- § overflow and special values

/-- The special-value branches of `f16_from_f32`, given the sign word (0 or 0x8000), raw f32
exponent, f32 mantissa and rebiased exponent.  `none` means "fall through to the normal path". -/
def encodeSpecial (sign rawexp man : Nat) (exp : Int) : Option Nat :=
  if rawexp = 255 then some (sign + 0x7C00 + (if man ≠ 0 then 0x200 else 0))
  else if exp ≥ 31 then some (sign + 0x7C00)
  else none

/-- A finite overflowing value saturates to infinity: mantissa field zero, exponent field 31. -/
theorem finite_overflow_is_inf (sign rawexp man : Nat) (exp : Int)
    (hs : sign = 0 ∨ sign = 32768) (hfin : rawexp ≠ 255) (hov : exp ≥ 31) :
    ∃ out, encodeSpecial sign rawexp man exp = some out ∧ f16Man out = 0 ∧ f16Exp out = 31 := by
  refine ⟨sign + 0x7C00, ?_, ?_, ?_⟩
  · simp [encodeSpecial, hfin, hov]
  · unfold f16Man; omega
  · unfold f16Exp; omega

/-- An f32 NaN stays a (quiet) NaN: exponent field 31 and mantissa field non-zero. -/
theorem nan_stays_nan (sign man : Nat) (exp : Int) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    ∃ out, encodeSpecial sign 255 man exp = some out ∧ f16Exp out = 31 ∧ f16Man out ≠ 0 := by
  refine ⟨sign + 0x7C00 + 0x200, ?_, ?_, ?_⟩
  · simp [encodeSpecial, hm]
  · unfold f16Exp; omega
  · unfold f16Man; omega

/-- The "conflated" encoder the header replaced (`exp >= 31 -> 0x7C00 | (man ? 0x200 : 0)`). -/
def conflatedOverflow (sign man : Nat) : Nat := sign + 0x7C00 + (if man ≠ 0 then 0x200 else 0)

/-- The replaced encoder turns a finite overflow with non-zero mantissa into a NaN; the fix does not. -/
theorem conflated_gives_nan (sign man : Nat) (hs : sign = 0 ∨ sign = 32768) (hm : man ≠ 0) :
    f16Man (conflatedOverflow sign man) ≠ 0 := by
  unfold f16Man conflatedOverflow
  simp [hm]
  omega

-- § exact fp16 -> f32 decode of subnormals

/-- The normalisation loop of `f32_from_f16`: number of left shifts until bit 10 is set. -/
def normShift : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, m => if m ≥ 1024 then 0 else normShift fuel (2 * m) + 1

/-- The loop invariant: after `normShift` shifts the mantissa lies in `[1024, 2048)`. -/
theorem normShift_spec (fuel m : Nat) (hm : m < 2048) (hreach : 1024 ≤ m * 2 ^ fuel) :
    1024 ≤ m * 2 ^ normShift fuel m ∧ m * 2 ^ normShift fuel m < 2048 ∧ normShift fuel m ≤ fuel := by
  induction fuel generalizing m with
  | zero => simp [normShift] at *; omega
  | succ f ih =>
    by_cases hge : m ≥ 1024
    · simp [normShift, hge]; omega
    · have hm2 : 2 * m < 2048 := by omega
      have hr : 1024 ≤ 2 * m * 2 ^ f := by
        rw [Nat.pow_succ] at hreach
        rw [Nat.mul_comm 2 m, Nat.mul_assoc]
        rw [Nat.mul_comm 2 (2 ^ f)]
        exact hreach
      obtain ⟨h1, h2, h3⟩ := ih (2 * m) hm2 hr
      have heq : m * 2 ^ (normShift f (2 * m) + 1) = 2 * m * 2 ^ normShift f (2 * m) := by
        rw [Nat.pow_succ, Nat.mul_comm (2 ^ _) 2, ← Nat.mul_assoc, Nat.mul_comm m 2]
      simp only [normShift, hge, if_false]
      rw [heq]
      omega

/-- Every non-zero fp16 subnormal mantissa normalises in at most 10 shifts. -/
theorem subnormal_normalises (m : Nat) (h1 : 1 ≤ m) (h2 : m < 1024) :
    1024 ≤ m * 2 ^ normShift 10 m ∧ m * 2 ^ normShift 10 m < 2048 ∧ normShift 10 m ≤ 10 := by
  apply normShift_spec 10 m (by omega)
  have : (2:Nat) ^ 10 = 1024 := by decide
  rw [this]
  omega

/-- The f32 exponent field written for a subnormal (`113 - s`) is a normal exponent (in `[103, 113]`).
`dequant.hpp`'s variant, which starts at `127 - 15 + 1` and decrements per shift, writes the same field. -/
theorem subnormal_exponent_normal (m : Nat) (h1 : 1 ≤ m) (h2 : m < 1024) :
    103 ≤ 113 - normShift 10 m ∧ 113 - normShift 10 m ≤ 113 ∧ 127 - 15 + 1 = 113 := by
  have := (subnormal_normalises m h1 h2).2.2
  omega

/-- Normal fp16 exponents rebias into the normal f32 range `[113, 142]`. -/
theorem normal_exponent_rebias (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    113 ≤ ex - 15 + 127 ∧ ex - 15 + 127 ≤ 142 := by omega

-- § bf16

/-- `bf16_from_f32` on the raw f32 bits `i < 2^32`, including the 32-bit wrap of the addition. -/
def bf16Encode (i : Nat) : Nat := (i + i / 65536 % 2 + 32767) % 4294967296 / 65536

/-- `f32_from_bf16`: the bf16 word becomes the top half of the f32 word. -/
def bf16Decode (h : Nat) : Nat := h * 65536

/-- Decoding then encoding is the identity on every bf16 word. -/
theorem bf16_roundtrip (h : Nat) (hh : h < 65536) : bf16Encode (bf16Decode h) = h := by
  unfold bf16Encode bf16Decode
  have hq : h * 65536 / 65536 = h := by omega
  rw [hq]
  have hb : h * 65536 + h % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hb]
  omega

/-- Below half of the dropped 16 bits truncates; above half rounds up (absent wrap). -/
theorem bf16_round_direction (h r : Nat) (hh : h < 65535) (hr : r < 65536) :
    (r < 32768 → bf16Encode (h * 65536 + r) = h) ∧
    (r > 32768 → bf16Encode (h * 65536 + r) = h + 1) := by
  unfold bf16Encode
  have hq : (h * 65536 + r) / 65536 = h := by omega
  rw [hq]
  have hb : h * 65536 + r + h % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hb]
  constructor <;> intro _ <;> omega

/-- An exact tie rounds to the even neighbour, because the `& 1` is on bit 16. -/
theorem bf16_tie_to_even (h : Nat) (hh : h < 65535) :
    bf16Encode (h * 65536 + 32768) = h + h % 2 ∧ (h + h % 2) % 2 = 0 := by
  unfold bf16Encode
  have hq : (h * 65536 + 32768) / 65536 = h := by omega
  rw [hq]
  have hb : h * 65536 + 32768 + h % 2 + 32767 < 4294967296 := by omega
  rw [Nat.mod_eq_of_lt hb]
  omega

/-- The plausible mistake (`& 1` on bit 15) differs from the real rule on a tie. -/
def bf16EncodeBit15 (i : Nat) : Nat := (i + i / 32768 % 2 + 32767) % 4294967296 / 65536

/-- Witness: on the tie 0x3F808000 the bit-15 variant rounds the even 0x3F80 up to 0x3F81. -/
theorem bit15_variant_wrong_on_tie :
    bf16Encode 0x3F808000 = 0x3F80 ∧ bf16EncodeBit15 0x3F808000 = 0x3F81 := by decide

/-- Observation: with no NaN branch, the f32 NaN 0x7F800001 encodes as the bf16 infinity 0x7F80. -/
theorem bf16_nan_becomes_inf : bf16Encode 0x7F800001 = 0x7F80 := by decide

/-- Observation: the negative NaN 0xFFFFFFFF wraps the 32-bit addition and encodes as +0. -/
theorem bf16_neg_nan_wraps_to_zero : bf16Encode 0xFFFFFFFF = 0 := by decide

end Strata.HalfBits
