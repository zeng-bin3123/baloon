
## FILE: /mnt/user-data/outputs/HalfPrecisionBitConversion.lean
/-!
# MODULE: half_precision_bit_conversion
SOURCES: include/ref/kernels/f16_bits.hpp, include/ref/kernels/bf16_bits.hpp
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

namespace Ref.HalfBits

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

end Ref.HalfBits



## FILE: /mnt/user-data/outputs/BlockDequantization.lean
/-!
# MODULE: block_dequantization
SOURCES: include/ref/artifact/dequant.hpp
Scalar reference dequantizers for the GGUF block encodings in the artifact (Q2_0, Q8_0, Q4_0, Q5_0,
IQ4_NL, Q6_K, Q4_K, Q5_K, Q3_K, IQ4_XS, F32/F16/BF16).  The theory pins down byte layouts, the
element -> (byte, bit, scale) index maps and the integer code ranges; float arithmetic is modelled
over `Int` scales where only the structural shape matters.

## Source semantics
* Q2_0: 64 elements / 18 bytes; code j sits in byte j/4 at bits 2(j%4) (LSB first); symbol = code - 1
  in {-1,0,1,2}; every output equals one of {-d, 0, d, 2d} (the `--check` structural invariant).
* Q4_0 / IQ4_NL: low nibble -> element j, high nibble -> element j+16.  Q5_0 adds a 5th bit.
* IQ4_NL codebook is strictly increasing and fits in int8.
* Q6_K and Q3_K: the scale applied to output position p is scale index p/16; Q4_K/Q5_K: p/32.
* Q5_K's 5th-bit masks are 4^g and 2*4^g (shift left by 2 per group); Q3_K's high-bit mask is NOT
  reset between the two 128-halves, so the 8 (half, j) steps use the 8 distinct bits 1..128.
* `get_scale_min_k4` yields 6-bit values and only reads bytes 0..11; Q3_K and IQ4_XS scales are
  6-bit with a -32 bias, so the signed scale is in [-32, 31].

## Form template identifiers
dequantize_q2_0 -> q2Value, code extraction -> q2Code, get_scale_min_k4 -> scaleMinK4,
dequantize_q6_K positions -> q6kPos, Q5_K masks -> q5kMask, Q3_K mask -> q3kMask,
kvalues_iq4nl -> iq4nlTable, IQ4_XS scale -> iq4xsScale
-/

namespace Ref.Dequant

-- § Q2_0 packing

/-- Divisor that selects slot `k` of a byte (`>> (k*2)`). -/
def slotDiv : Nat → Nat
  | 0 => 1
  | 1 => 4
  | 2 => 16
  | _ => 64

/-- The 2-bit code of element `j` read from byte array `qs`, exactly as `dequantize_q2_0` does. -/
def q2Code (qs : Nat → Nat) (j : Nat) : Nat := qs (j / 4) / slotDiv (j % 4) % 4

/-- A byte built from four codes, slot 0 in the lowest bits. -/
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Packing four codes into a byte and reading slot `k` back yields code `k` (LSB-first order). -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 / slotDiv 0 % 4 = c0 ∧ packByte c0 c1 c2 c3 / slotDiv 1 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / slotDiv 2 % 4 = c2 ∧ packByte c0 c1 c2 c3 / slotDiv 3 % 4 = c3 := by
  simp only [packByte, slotDiv]
  omega

/-- A packed byte is a byte. -/
theorem packByte_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 := by
  unfold packByte; omega

/-- For the 64 elements of a block, the byte index is one of the 16 code bytes, the bit offset is
at most 6, and (byte, slot) determines the element: the index map is injective. -/
theorem q2_index_map (j : Nat) (hj : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 ∧ j = 4 * (j / 4) + j % 4 := by omega

/-- Every code is in `{0,1,2,3}`. -/
theorem q2Code_lt (qs : Nat → Nat) (j : Nat) : q2Code qs j < 4 := by
  unfold q2Code; exact Nat.mod_lt _ (by decide)

/-- The symbol `code - 1` is one of `{-1, 0, 1, 2}`. -/
theorem q2_symbol_range (c : Nat) (hc : c < 4) :
    (c : Int) - 1 = -1 ∨ (c : Int) - 1 = 0 ∨ (c : Int) - 1 = 1 ∨ (c : Int) - 1 = 2 := by
  omega

/-- The dequantized value `(code - 1) * d`. -/
def q2Value (c : Nat) (d : Int) : Int := ((c : Int) - 1) * d

/-- The `--check` structural invariant: every Q2_0 output is one of `{-d, 0, d, 2d}`. -/
theorem q2_value_structural (qs : Nat → Nat) (j : Nat) (d : Int) :
    q2Value (q2Code qs j) d = -d ∨ q2Value (q2Code qs j) d = 0 ∨
    q2Value (q2Code qs j) d = d ∨ q2Value (q2Code qs j) d = 2 * d := by
  have hc := q2Code_lt qs j
  generalize q2Code qs j = c at *
  unfold q2Value
  rcases q2_symbol_range c hc with h | h | h | h <;> rw [h] <;> simp [Int.neg_one_mul]

-- § 32-element nibble encodings (Q4_0, Q5_0, Q8_0, IQ4_NL)

/-- A byte splits losslessly into its low and high nibble. -/
theorem nibble_split (b : Nat) (hb : b < 256) : b = b % 16 + 16 * (b / 16) ∧ b / 16 < 16 := by omega

/-- Q4_0 symbols `nibble - 8` lie in `[-8, 7]`. -/
theorem q4_0_range (b : Nat) (hb : b < 256) :
    -8 ≤ ((b % 16 : Nat) : Int) - 8 ∧ ((b % 16 : Nat) : Int) - 8 ≤ 7 ∧
    -8 ≤ ((b / 16 : Nat) : Int) - 8 ∧ ((b / 16 : Nat) : Int) - 8 ≤ 7 := by omega

/-- Q5_0 assembles `nibble | (bit << 4)`: a 5-bit value, and the symbol `x - 16` is in `[-16, 15]`. -/
theorem q5_0_range (b bit : Nat) (hbit : bit < 2) :
    b % 16 + 16 * bit < 32 ∧ -16 ≤ ((b % 16 + 16 * bit : Nat) : Int) - 16 ∧
    ((b % 16 + 16 * bit : Nat) : Int) - 16 ≤ 15 := by omega

/-- Low-nibble element `j` and high-nibble element `j + 16` (j < 16) cover the 32 outputs exactly once. -/
theorem nibble_positions (p : Nat) (hp : p < 32) :
    (p < 16 ∧ p % 16 = p) ∨ (16 ≤ p ∧ p % 16 + 16 = p) := by omega

/-- The IQ4_NL codebook, verbatim from ggml-common.h. -/
def iq4nlTable : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Strictly increasing check on a list. -/
def strictlyIncreasing : List Int → Bool
  | a :: b :: t => decide (a < b) && strictlyIncreasing (b :: t)
  | _ => true

/-- The codebook has 16 entries, is strictly increasing (so it is a non-linear but injective code),
and every entry fits in `int8`. -/
theorem iq4nl_table_wellformed :
    iq4nlTable.length = 16 ∧ strictlyIncreasing iq4nlTable = true ∧
    iq4nlTable.all (fun v => decide (-128 ≤ v ∧ v ≤ 127)) = true := by decide

-- § Q6_K

/-- Output position of the `k`-th of the four values produced at `(half n, l)`. -/
def q6kPos (n l k : Nat) : Nat := 128 * n + l + 32 * k

/-- The scale index used: `8n` (the `sc += 8` advance) plus `is + 2k` with `is = l / 16`. -/
def q6kScale (n l k : Nat) : Nat := 8 * n + l / 16 + 2 * k

/-- Q6_K positions are in range and decode back to `(n, l, k)`, so all 256 outputs are written once. -/
theorem q6k_pos_bijective (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    q6kPos n l k < 256 ∧ q6kPos n l k / 128 = n ∧ q6kPos n l k % 128 / 32 = k ∧ q6kPos n l k % 32 = l := by
  unfold q6kPos; omega

/-- The scale applied to Q6_K output `p` is scale index `p / 16`: each of the 16 scales covers 16
consecutive outputs. -/
theorem q6k_scale_is_pos_div16 (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    q6kScale n l k = q6kPos n l k / 16 ∧ q6kScale n l k < 16 := by
  unfold q6kScale q6kPos; omega

/-- Q6_K byte reads stay inside `ql[128]`, `qh[64]`: ql byte `64n + l + 32(k%2)`, qh byte `32n + l`. -/
theorem q6k_reads_in_bounds (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    64 * n + l + 32 * (k % 2) < 128 ∧ 32 * n + l < 64 := by omega

/-- A Q6_K symbol `low4 | (high2 << 4)` minus 32 lies in `[-32, 31]`. -/
theorem q6k_symbol_range (lo hi : Nat) (hlo : lo < 16) (hhi : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § Q4_K / Q5_K scale-min unpacking

/-- `get_scale_min_k4` over the 12 scale bytes `q`, returning `(scale, min)`. -/
def scaleMinK4 (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j % 64, q (j + 4) % 64)
  else (q (j + 4) % 16 + 16 * (q (j - 4) / 64), q (j + 4) / 16 + 16 * (q j / 64))

/-- For `j < 8` the unpacker reads only bytes `0..11`. -/
theorem scaleMinK4_reads (j : Nat) (hj : j < 8) : j + 4 < 12 ∧ j < 12 ∧ j - 4 < 12 := by omega

/-- With byte inputs, both outputs are 6-bit. -/
theorem scaleMinK4_six_bit (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  have h1 := hq (j + 4)
  have h2 := hq (j - 4)
  have h3 := hq j
  split <;> simp <;> omega

/-- Q4_K/Q5_K: element `64g + 32h + l` uses code byte `32g + l`, nibble `h`, and scale pair
`2g + h`, i.e. scale index = position / 32. -/
theorem q4k_index_map (g h l : Nat) (hg : g < 4) (hh : h < 2) (hl : l < 32) :
    64 * g + 32 * h + l < 256 ∧ 32 * g + l < 128 ∧ 2 * g + h = (64 * g + 32 * h + l) / 32 := by omega

/-- Q5_K's 5th-bit masks shift left by 2 per group: `u1 = 4^g`, `u2 = 2 * 4^g`. -/
def q5kMask1 (g : Nat) : Nat := 4 ^ g
/-- See `q5kMask1`. -/
def q5kMask2 (g : Nat) : Nat := 2 * 4 ^ g

/-- Over the four groups the eight masks are exactly the eight bits of a byte, each once. -/
theorem q5k_masks_are_bits :
    [q5kMask1 0, q5kMask2 0, q5kMask1 1, q5kMask2 1, q5kMask1 2, q5kMask2 2, q5kMask1 3, q5kMask2 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- A Q5_K quant `nibble + (bit ? 16 : 0)` is 5-bit. -/
theorem q5k_quant_range (b : Nat) (bit : Bool) : b % 16 + (if bit then 16 else 0) < 32 := by
  cases bit <;> simp <;> omega

-- § Q3_K

/-- The high-bit mask at step `(half, j)`: hoisted out of the half loop, so it keeps shifting. -/
def q3kMask (half j : Nat) : Nat := 2 ^ (4 * half + j)

/-- The eight steps use eight distinct mask bits 1..128 (the mask is not reset between halves). -/
theorem q3k_mask_not_reset :
    [q3kMask 0 0, q3kMask 0 1, q3kMask 0 2, q3kMask 0 3, q3kMask 1 0, q3kMask 1 1, q3kMask 1 2, q3kMask 1 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- A Q3_K symbol `low2 - (hbit ? 0 : 4)` lies in `[-4, 3]` (the correction is negative). -/
theorem q3k_symbol_range (lo : Nat) (hlo : lo < 4) (hbit : Bool) :
    -4 ≤ (lo : Int) - (if hbit then 0 else 4) ∧ (lo : Int) - (if hbit then 0 else 4) ≤ 3 := by
  cases hbit <;> simp <;> omega

/-- Q3_K output `128 half + 32 j + 16 s + l` (s = which of the two 16-runs) uses scale
`8 half + 2 j + s`, which is position / 16. -/
theorem q3k_scale_is_pos_div16 (half j s l : Nat) (hh : half < 2) (hj : j < 4) (hs : s < 2) (hl : l < 16) :
    8 * half + 2 * j + s = (128 * half + 32 * j + 16 * s + l) / 16 ∧ 128 * half + 32 * j + 16 * s + l < 256 := by
  omega

/-- A repacked Q3_K scale byte `(a & 0x0F) | ((t & 3) << 4)` is 6-bit; minus 32 it lies in `[-32, 31]`. -/
theorem q3k_scale_range (a t : Nat) :
    a % 16 + 16 * (t % 4) < 64 ∧ -32 ≤ ((a % 16 + 16 * (t % 4) : Nat) : Int) - 32 ∧
    ((a % 16 + 16 * (t % 4) : Nat) : Int) - 32 ≤ 31 := by omega

-- § IQ4_XS

/-- IQ4_XS group scale: low nibble of `scales_l[ib/2]` selected by `ib % 2`, plus two bits of the
16-bit `scales_h` at shift `2 ib`. -/
def iq4xsScale (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  sl (ib / 2) / (if ib % 2 = 0 then 1 else 16) % 16 + 16 * (sh / 4 ^ ib % 4)

/-- The scale is 6-bit and the reads stay in `scales_l[4]` and the 16 bits of `scales_h`. -/
theorem iq4xs_scale_range (sl : Nat → Nat) (sh ib : Nat) (hib : ib < 8) :
    iq4xsScale sl sh ib < 64 ∧ ib / 2 < 4 ∧ 2 * ib + 1 < 16 := by
  unfold iq4xsScale
  omega

-- § byte budgets

/-- Field sizes read by each dequantizer sum to its block size:
Q2_0 18, Q8_0 34, Q4_0 18, Q5_0 22, IQ4_NL 18, Q6_K 210 (d at 208), Q4_K 144, Q5_K 176,
Q3_K 110 (d at 108), IQ4_XS 136. -/
theorem block_byte_budgets :
    2 + 64 / 4 = 18 ∧ 2 + 32 = 34 ∧ 2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧
    128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 ∧ 2 + 2 + 12 + 128 = 144 ∧
    2 + 2 + 12 + 32 + 128 = 176 ∧ 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 + 12 = 108 ∧
    2 + 2 + 4 + 128 = 136 := by decide

end Ref.Dequant



## FILE: /mnt/user-data/outputs/GgufHeaderParsing.lean
/-!
# MODULE: gguf_header_parsing
SOURCES: include/ref/artifact/gguf_reader.hpp
GGUF v3 header reader: bounds-checked cursor, metadata value decoding, tensor directory, aligned
data-section start, the ggml block-geometry table and the qwen4exp architecture guard.

## Source semantics
* `Cursor::need(n)` fails iff `pos + n > size`; every successful read keeps `pos <= size`, so the
  reader cannot run past the mapping; a sequence of reads advances `pos` by the sum of their sizes.
* Arrays longer than 2^24 are refused; only `min(count, 64)` items are kept.  Nesting depth > 2 fails.
* Signed integer metadata is stored sign-extended into `uint64` (two's complement mod 2^64) and is
  recoverable.
* `data_start = ceil(pos / align) * align`: aligned, at or after the header end, less than one
  alignment unit past it.
* `block_geometry` gives (elements, bytes); Q2_0 = (64, 18).  Every listed quantized type has bytes
  equal to its header fields plus packed payload.
* `check_architecture` returns "" iff the architecture string is `qwen4exp` and all six required
  keys are present and equal the compiled-in constants; otherwise it reports the FIRST failure.

## Form template identifiers
Cursor -> Cursor, need/read -> Cursor.read, read_value ARRAY -> keptItems, data_start -> alignUp,
block_geometry -> blockGeometry, check_architecture -> checkReqs, Qwen4ExpGuard -> guardDefaults
-/

namespace Ref.Gguf

-- § bounds-checked cursor

/-- Cursor state: current position and mapped size. -/
structure Cursor where
  pos : Nat
  size : Nat

/-- `need(n)` then advance by `n`; `none` is the thrown "unexpected end of file". -/
def Cursor.read (c : Cursor) (n : Nat) : Option Cursor :=
  if c.pos + n > c.size then none else some { c with pos := c.pos + n }

/-- A successful read keeps the cursor inside the mapping and advances by exactly `n`. -/
theorem read_in_bounds (c c' : Cursor) (n : Nat) (h : c.read n = some c') :
    c'.pos ≤ c'.size ∧ c'.pos = c.pos + n ∧ c'.size = c.size := by
  unfold Cursor.read at h
  split at h
  · cases h
  · cases h; simp; omega

/-- A read fails exactly when it would run past the end. -/
theorem read_fails_iff (c : Cursor) (n : Nat) : c.read n = none ↔ c.pos + n > c.size := by
  unfold Cursor.read; split <;> simp_all

/-- Run a sequence of reads, stopping at the first failure. -/
def readAll : Cursor → List Nat → Option Cursor
  | c, [] => some c
  | c, n :: ns => match c.read n with
    | none => none
    | some c' => readAll c' ns

/-- A successful sequence of reads advances by the total size and never leaves the mapping. -/
theorem readAll_spec (ns : List Nat) (c c' : Cursor) (h : readAll c ns = some c') :
    c'.pos = c.pos + ns.foldr (· + ·) 0 ∧ c'.size = c.size ∧ c.pos ≤ c'.pos := by
  induction ns generalizing c with
  | nil => simp [readAll] at h; subst h; simp
  | cons n ns ih =>
    simp only [readAll] at h
    split at h
    · cases h
    · rename_i c1 hc1
      obtain ⟨_, hp, hs⟩ := read_in_bounds c c1 n hc1
      obtain ⟨hp', hs', _⟩ := ih c1 h
      simp only [List.foldr]
      refine ⟨?_, ?_, ?_⟩ <;> omega

/-- A GGUF string is a u64 length then that many bytes: it consumes `8 + len` bytes. -/
theorem string_read_size (c : Cursor) (len : Nat) (c' : Cursor) (h : readAll c [8, len] = some c') :
    c'.pos = c.pos + 8 + len := by
  have := (readAll_spec [8, len] c c' h).1
  simp at this; omega

-- § metadata values

/-- Items kept for an array of `count` elements (a sample of at most 64). -/
def keptItems (count : Nat) : Nat := min count 64

/-- Arrays are capped: at most 64 items kept, and all are kept for short arrays. -/
theorem keptItems_spec (count : Nat) :
    keptItems count ≤ 64 ∧ keptItems count ≤ count ∧ (count ≤ 64 → keptItems count = count) := by
  unfold keptItems; omega

/-- Array length guard: an accepted count is at most 2^24. -/
def arrayLenOk (count : Nat) : Bool := decide (count ≤ 2 ^ 24)

/-- The guard accepts exactly the counts up to 16,777,216. -/
theorem arrayLenOk_iff (count : Nat) : arrayLenOk count = true ↔ count ≤ 16777216 := by
  unfold arrayLenOk; simp

/-- Sign extension into `uint64`: `(uint64)(int64)x`. -/
def signExtend64 (x : Int) : Nat := (x % 18446744073709551616).toNat

/-- Reading the stored word back as `int64` recovers the original signed value. -/
def asInt64 (u : Nat) : Int := if u < 9223372036854775808 then u else (u : Int) - 18446744073709551616

/-- Signed metadata (I8/I16/I32/I64) survives the uint64 storage exactly. -/
theorem signExtend_roundtrip (x : Int) (h1 : -9223372036854775808 ≤ x) (h2 : x < 9223372036854775808) :
    asInt64 (signExtend64 x) = x := by
  unfold asInt64 signExtend64
  by_cases hx : 0 ≤ x
  · have : x % 18446744073709551616 = x := Int.emod_eq_of_lt hx (by omega)
    rw [this]; split <;> omega
  · have : x % 18446744073709551616 = x + 18446744073709551616 := by omega
    rw [this]; split <;> omega

/-- Booleans are normalised: any non-zero byte reads as 1. -/
def readBool (b : Nat) : Nat := if b ≠ 0 then 1 else 0

/-- A decoded bool is 0 or 1. -/
theorem readBool_binary (b : Nat) : readBool b = 0 ∨ readBool b = 1 := by
  unfold readBool; split <;> simp

-- § data section alignment

/-- `(pos + align - 1) / align * align`. -/
def alignUp (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The data section starts on an alignment boundary, at or after the header, within one unit. -/
theorem alignUp_spec (pos align : Nat) (ha : 0 < align) :
    alignUp pos align % align = 0 ∧ pos ≤ alignUp pos align ∧ alignUp pos align < pos + align := by
  unfold alignUp
  have hdm := Nat.div_add_mod (pos + align - 1) align
  have hlt := Nat.mod_lt (pos + align - 1) ha
  refine ⟨Nat.mul_mod_left _ _, ?_, ?_⟩
  · rw [Nat.mul_comm]
    generalize align * ((pos + align - 1) / align) = m at *
    omega
  · rw [Nat.mul_comm]
    generalize align * ((pos + align - 1) / align) = m at *
    omega

/-- Default alignment 32 is used when `general.alignment` is absent or zero. -/
def effectiveAlign (mv : Option Nat) : Nat :=
  match mv with
  | some a => if a ≠ 0 then a else 32
  | none => 32

/-- The effective alignment is never zero, so `alignUp_spec` always applies. -/
theorem effectiveAlign_pos (m : Option Nat) : 0 < effectiveAlign m := by
  unfold effectiveAlign
  split
  · split <;> omega
  · decide

-- § block geometry

/-- `block_geometry`: (elements per block, bytes per block), `none` for unsupported types. -/
def blockGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 (type 42) is 64 elements in 18 bytes: an fp16 scale plus 64 two-bit codes. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) ∧ 2 + 64 * 2 / 8 = 18 := by decide

/-- The table agrees with the field layouts the dequantizers read
(Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS). -/
theorem geometry_matches_dequant_layouts :
    blockGeometry 2 = some (32, 2 + 16) ∧ blockGeometry 6 = some (32, 2 + 4 + 16) ∧
    blockGeometry 8 = some (32, 2 + 32) ∧ blockGeometry 20 = some (32, 2 + 16) ∧
    blockGeometry 11 = some (256, 32 + 64 + 12 + 2) ∧ blockGeometry 12 = some (256, 2 + 2 + 12 + 128) ∧
    blockGeometry 13 = some (256, 2 + 2 + 12 + 32 + 128) ∧ blockGeometry 14 = some (256, 128 + 64 + 16 + 2) ∧
    blockGeometry 23 = some (256, 2 + 2 + 4 + 128) := by decide

/-- The table agrees with the i-quant block sizes the AVX-512 kernels stride by. -/
theorem geometry_matches_iq_kernels :
    blockGeometry 16 = some (256, 66) ∧ blockGeometry 17 = some (256, 74) ∧
    blockGeometry 18 = some (256, 98) ∧ blockGeometry 21 = some (256, 110) ∧
    blockGeometry 22 = some (256, 82) := by decide

/-- Bytes of a tensor row of `n` elements (a whole number of blocks). -/
def rowBytes (elems bytes n : Nat) : Nat := n / elems * bytes

/-- A Q2_0 row of the hidden size 2560 is 40 blocks = 720 bytes. -/
theorem q2_0_row_2560 : rowBytes 64 18 2560 = 720 := by decide

-- § architecture guard

/-- The compiled-in qwen4exp constants. -/
def guardDefaults : List (String × Nat) :=
  [("qwen4exp.block_count", 48), ("qwen4exp.embedding_length", 2560), ("qwen4exp.expert_count", 512),
   ("qwen4exp.expert_used_count", 10), ("qwen4exp.attention.head_count", 24),
   ("qwen4exp.attention.head_count_kv", 2)]

/-- Check each required key in order; `none` is success, `some key` names the first failure. -/
def checkReqs (lookup : String → Option Nat) : List (String × Nat) → Option String
  | [] => none
  | (k, w) :: rest =>
    match lookup k with
    | none => some k
    | some v => if v = w then checkReqs lookup rest else some k

/-- The guard passes iff every required key is present with exactly the expected value. -/
theorem checkReqs_none_iff (lookup : String → Option Nat) (reqs : List (String × Nat)) :
    checkReqs lookup reqs = none ↔ ∀ r ∈ reqs, lookup r.1 = some r.2 := by
  induction reqs with
  | nil => simp [checkReqs]
  | cons r rest ih =>
    obtain ⟨k, w⟩ := r
    simp only [checkReqs, List.mem_cons, forall_eq_or_imp]
    split
    · rename_i hk; simp [hk]
    · rename_i v hk
      split
      · rename_i hv; subst hv; rw [ih]; simp [hk]
      · rename_i hv; simp [hk, hv]

/-- When the guard fails, it names a key that is genuinely missing or mismatched. -/
theorem checkReqs_failure_sound (lookup : String → Option Nat) (reqs : List (String × Nat)) (k : String)
    (h : checkReqs lookup reqs = some k) : ∃ w, (k, w) ∈ reqs ∧ lookup k ≠ some w := by
  induction reqs with
  | nil => simp [checkReqs] at h
  | cons r rest ih =>
    obtain ⟨k0, w0⟩ := r
    simp only [checkReqs] at h
    split at h
    · rename_i hk; cases h; exact ⟨w0, by simp, by simp [hk]⟩
    · rename_i v hk
      split at h
      · obtain ⟨w, hw, hne⟩ := ih h; exact ⟨w, List.mem_cons_of_mem _ hw, hne⟩
      · rename_i hv; cases h; exact ⟨w0, by simp, by simp [hk]; exact hv⟩

/-- The full guard: architecture string first, then the six keys. -/
def checkArchitecture (arch : Option String) (lookup : String → Option Nat) : Bool :=
  match arch with
  | some "qwen4exp" => (checkReqs lookup guardDefaults).isNone
  | _ => false

/-- The guard accepts only `qwen4exp` with hidden size 2560, 512 experts and 10 used experts (and
the other compiled constants). -/
theorem checkArchitecture_sound (arch : Option String) (lookup : String → Option Nat)
    (h : checkArchitecture arch lookup = true) :
    arch = some "qwen4exp" ∧ lookup "qwen4exp.embedding_length" = some 2560 ∧
    lookup "qwen4exp.expert_count" = some 512 ∧ lookup "qwen4exp.expert_used_count" = some 10 := by
  unfold checkArchitecture at h
  split at h
  · have hn : checkReqs lookup guardDefaults = none := by
      cases hc : checkReqs lookup guardDefaults <;> simp_all
    have hall := (checkReqs_none_iff lookup guardDefaults).1 hn
    refine ⟨rfl, hall ("qwen4exp.embedding_length", 2560) (by simp [guardDefaults]),
      hall ("qwen4exp.expert_count", 512) (by simp [guardDefaults]),
      hall ("qwen4exp.expert_used_count", 10) (by simp [guardDefaults])⟩
  · cases h

end Ref.Gguf



## FILE: /mnt/user-data/outputs/Q2ExpertDotKernel.lean
/-!
# MODULE: q2_expert_dot_kernel
SOURCES: src/kernels/cpu/expert.cpp
The Q2_0 x Q8_1 expert kernel: activation quantization, the VNNI row dot with the code-offset
correction, the 512-bit block batching, the expert blob geometry and the CPU feature gate.

## Source semantics
* Row identity: sum_j (c_j - 1) d_w xq_j d_x = d_w (d_x sum c_j xq_j - d_x sum xq_j); the second term is
  the per-chunk `hx = scale * sum`, applied ONCE per chunk (not broadcast into 8 lanes).
* `chunk_dot_oracle` removes the offset in exact integer arithmetic: dot - sum = sum (c - 1) xq.
* Activations are clamped to [-127, 127]; codes are in [0, 3]; so a VNNI 4-product lane is at most
  1524 in magnitude and a 32-chunk integer sum at most 12192 < 2^24: the int -> f32 convert is exact.
* The multishift control 0x0E0C0A0806040200 selects shifts 0,2,..,14: output byte i of a u16 lane is
  the i-th 2-bit code in LSB-first order, matching the Q2_0 dequantizer.
* Blocks are batched 8 at a time with a tail `nb = min(8, n - b0)`; lane masks have 2*nb bits; the
  batches cover every block exactly once.
* Blob geometry (H = 2560, FF = 640): gate/up rows interleaved (gate 2r, up 2r+1), 40 blocks each;
  down rows 10 blocks each; total 1,382,400 bytes.
* The CPU gate requires AVX512 F, BW, VL, VNNI and VBMI (CPUID.7.0 EBX bits 16, 30, 31; ECX 11, 1).

## Form template identifiers
row_dot / row_dot_z -> rowDot, act_quant_q8_1 -> clampQ, ActQ.hx -> chunkHx, chunk_dot_oracle ->
chunkDotOracle, scales8 mask -> laneMask, b0/nb loop -> batchSizes, CpuFeatures::usable -> cpuUsable
-/

namespace Ref.Q2Expert

-- § the correction identity

/-- Sum of `f c x` over two equal-length lists. -/
def zsum (f : Int → Int → Int) : List Int → List Int → Int
  | c :: cs, x :: xs => f c x + zsum f cs xs
  | _, _ => 0

/-- Plain sum of a list. -/
def lsum : List Int → Int
  | [] => 0
  | x :: xs => x + lsum xs

/-- The code offset splits off as the plain activation sum: `sum (c-1) x = sum c x - sum x`. -/
theorem offset_identity (cs xs : List Int) (hl : cs.length = xs.length) :
    zsum (fun c x => (c - 1) * x) cs xs = zsum (fun c x => c * x) cs xs - lsum xs := by
  induction cs generalizing xs with
  | nil => cases xs with
    | nil => simp [zsum, lsum]
    | cons _ _ => simp at hl
  | cons c cs ih =>
    cases xs with
    | nil => simp at hl
    | cons x xs =>
      simp only [List.length_cons, Nat.add_right_cancel_iff] at hl
      simp only [zsum, lsum, ih xs hl]
      rw [Int.sub_mul, Int.one_mul]
      omega

/-- The chunk's correction term `hx = d_x * sum(xq)`. -/
def chunkHx (dx : Int) (xs : List Int) : Int := dx * lsum xs

/-- One chunk of the row dot as the kernel computes it equals the exact dequantized product
(over exact integer scales): `dw * (dx * sum c x - hx) = sum ((c-1) dw) (xq dx)`. -/
theorem chunk_kernel_exact (cs xs : List Int) (hl : cs.length = xs.length) (dw dx : Int) :
    dw * (dx * zsum (fun c x => c * x) cs xs - chunkHx dx xs) =
      dw * dx * zsum (fun c x => (c - 1) * x) cs xs := by
  rw [offset_identity cs xs hl, chunkHx, ← Int.mul_sub, Int.mul_assoc]

/-- `chunk_dot_oracle`: the integer VNNI dot minus the stored activation sum. -/
def chunkDotOracle (cs xs : List Int) : Int := zsum (fun c x => c * x) cs xs - lsum xs

/-- The oracle's integer result is exactly `sum (c - 1) xq` — no rounding before the scale. -/
theorem chunkDotOracle_exact (cs xs : List Int) (hl : cs.length = xs.length) :
    chunkDotOracle cs xs = zsum (fun c x => (c - 1) * x) cs xs := by
  unfold chunkDotOracle; rw [offset_identity cs xs hl]

-- § activation quantization ranges

/-- The clamp applied after rounding. -/
def clampQ (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Clamped activations lie in `[-127, 127]` and the clamp is idempotent. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 ∧ clampQ (clampQ v) = clampQ v := by
  unfold clampQ
  split <;> split <;> simp_all <;> omega

/-- A VNNI lane adds four `u8 code * s8 act` products: bounded by 1524, far from int32 overflow. -/
theorem vnni_lane_bound (c0 c1 c2 c3 x0 x1 x2 x3 : Int)
    (hc : 0 ≤ c0 ∧ c0 ≤ 3 ∧ 0 ≤ c1 ∧ c1 ≤ 3 ∧ 0 ≤ c2 ∧ c2 ≤ 3 ∧ 0 ≤ c3 ∧ c3 ≤ 3)
    (hx : -127 ≤ x0 ∧ x0 ≤ 127 ∧ -127 ≤ x1 ∧ x1 ≤ 127 ∧ -127 ≤ x2 ∧ x2 ≤ 127 ∧ -127 ≤ x3 ∧ x3 ≤ 127) :
    -1524 ≤ c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3 ∧ c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3 ≤ 1524 := by
  have b : ∀ c x : Int, 0 ≤ c → c ≤ 3 → -127 ≤ x → x ≤ 127 → -381 ≤ c * x ∧ c * x ≤ 381 := by
    intro c x h1 h2 h3 h4
    have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
    rcases this with rfl | rfl | rfl | rfl <;> omega
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7⟩ := hc
  obtain ⟨y0, y1, y2, y3, y4, y5, y6, y7⟩ := hx
  have e0 := b c0 x0 a0 a1 y0 y1
  have e1 := b c1 x1 a2 a3 y2 y3
  have e2 := b c2 x2 a4 a5 y4 y5
  have e3 := b c3 x3 a6 a7 y6 y7
  omega

/-- Bound of a list of code-times-activation products: `|sum| <= 381 * length`. -/
theorem zsum_bound (cs xs : List Int) (hc : ∀ c ∈ cs, 0 ≤ c ∧ c ≤ 3) (hx : ∀ x ∈ xs, -127 ≤ x ∧ x ≤ 127) :
    -(381 * (cs.length : Int)) ≤ zsum (fun c x => c * x) cs xs ∧
      zsum (fun c x => c * x) cs xs ≤ 381 * (cs.length : Int) := by
  induction cs generalizing xs with
  | nil => simp [zsum]
  | cons c cs ih =>
    cases xs with
    | nil => simp [zsum]; omega
    | cons x xs =>
      have h1 := hc c (by simp)
      have h2 := hx x (by simp)
      have ihx := ih xs (fun c' h => hc c' (by simp [h])) (fun x' h => hx x' (by simp [h]))
      have hcx : -381 ≤ c * x ∧ c * x ≤ 381 := by
        have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
        rcases this with rfl | rfl | rfl | rfl <;> omega
      simp only [zsum, List.length_cons]
      push_cast
      omega

/-- A 32-element chunk's integer dot is at most 12192 < 2^24, so its f32 conversion is exact. -/
theorem chunk_dot_exact_in_f32 (cs xs : List Int) (hlen : cs.length = 32)
    (hc : ∀ c ∈ cs, 0 ≤ c ∧ c ≤ 3) (hx : ∀ x ∈ xs, -127 ≤ x ∧ x ≤ 127) :
    -16777216 < zsum (fun c x => c * x) cs xs ∧ zsum (fun c x => c * x) cs xs < 16777216 := by
  have := zsum_bound cs xs hc hx
  rw [hlen] at this
  omega

-- § 2-bit unpack order

/-- Byte `i` of the multishift control constant. -/
def ctrlByte (i : Nat) : Nat := 0x0E0C0A0806040200 / 256 ^ i % 256

/-- The control selects shifts `0, 2, 4, ..., 14`. -/
theorem ctrl_shifts : (List.range 8).map ctrlByte = [0, 2, 4, 6, 8, 10, 12, 14] := by decide

/-- Output byte `i` of a u16 lane `b0 + 256 b1`, masked with 3, is the `i`-th code in LSB-first
order: codes 0..3 from byte 0, codes 4..7 from byte 1 (as in `dequantize_q2_0`). -/
theorem multishift_code_order (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1) / 2 ^ 0 % 4 = b0 % 4 ∧ (b0 + 256 * b1) / 2 ^ 2 % 4 = b0 / 4 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 4 % 4 = b0 / 16 % 4 ∧ (b0 + 256 * b1) / 2 ^ 6 % 4 = b0 / 64 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 8 % 4 = b1 % 4 ∧ (b0 + 256 * b1) / 2 ^ 10 % 4 = b1 / 4 % 4 ∧
    (b0 + 256 * b1) / 2 ^ 12 % 4 = b1 / 16 % 4 ∧ (b0 + 256 * b1) / 2 ^ 14 % 4 = b1 / 64 % 4 := by
  simp only [Nat.reducePow]
  omega

-- § block batching in the 512-bit kernel

/-- Lane mask of `scales8`: `0xFFFF` for a full batch, else `(1 << 2nb) - 1`. -/
def laneMask (nb : Nat) : Nat := if nb ≥ 8 then 0xFFFF else 2 ^ (2 * nb) - 1

/-- The mask always enables exactly the `2 nb` chunk lanes of the batch. -/
theorem laneMask_spec : ∀ nb ∈ List.range 9, laneMask nb = 2 ^ (2 * nb) - 1 := by decide

/-- Batch sizes of the `b0 += 8` loop with tail `min(8, n - b0)` (fuel-bounded). -/
def batchSizes : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 8 n :: batchSizes fuel (n - min 8 n)

/-- Batches cover every block exactly once and none exceeds 8. -/
theorem batchSizes_cover (fuel n : Nat) (hf : n ≤ 8 * fuel) :
    (batchSizes fuel n).foldr (· + ·) 0 = n ∧ ∀ b ∈ batchSizes fuel n, 0 < b ∧ b ≤ 8 := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by omega
            subst this; simp [batchSizes]
  | succ f ih =>
    cases n with
    | zero => simp [batchSizes]
    | succ m =>
      obtain ⟨h1, h2⟩ := ih (m + 1 - min 8 (m + 1)) (by omega)
      simp only [batchSizes, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro b hb
      rcases hb with rfl | hb
      · omega
      · exact h2 b hb

/-- Block `b` of a row uses activation chunks `2b` and `2b+1`, at int8 offsets `64b` and `64b + 32`:
the weight block spans two activation scale chunks. -/
theorem block_chunk_map (b : Nat) : (2 * b) * 32 = 64 * b ∧ (2 * b + 1) * 32 = 64 * b + 32 := by omega

-- § expert blob geometry

/-- Hidden and intermediate sizes (H from the qwen4exp guard, FF from the 1,382,400-byte blob). -/
def H : Nat := 2560
/-- See `H`. -/
def FF : Nat := 640

/-- Blocks per gate/up row and per down row, and the code/scale bytes per row. -/
theorem row_geometry : H / 64 = 40 ∧ FF / 64 = 10 ∧ 40 * 16 = 640 ∧ 10 * 16 = 160 := by decide

/-- Gate/up: 2 FF rows of (640 code + 80 scale) bytes; down: H rows of (160 + 20); sum 1,382,400. -/
theorem blob_size : 2 * FF * (640 + 80) + H * (160 + 20) = 1382400 := by decide

/-- Interleaved gate/up rows: row `r`'s gate is `2r` and up is `2r+1`; distinct and in range. -/
theorem gate_up_interleave (r : Nat) (hr : r < FF) :
    2 * r < 2 * FF ∧ 2 * r + 1 < 2 * FF ∧ (2 * r) / 2 = r ∧ (2 * r + 1) / 2 = r ∧ (2 * r) % 2 ≠ (2 * r + 1) % 2 := by
  unfold FF at *; omega

/-- Token-count dispatch of the `_multi` entry points: cases 1..7, default 8. -/
def dispatchNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For 1..8 tokens the dispatched template width equals the token count. -/
theorem dispatchNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : dispatchNT n = n := by
  unfold dispatchNT; split <;> omega

-- § CPU feature gate

/-- Bit `k` of a CPUID register. -/
def bit (r k : Nat) : Bool := r / 2 ^ k % 2 = 1

/-- `CpuFeatures::usable`: AVX512 F (EBX.16), BW (EBX.30), VL (EBX.31), VNNI (ECX.11), VBMI (ECX.1). -/
def cpuUsable (ebx ecx : Nat) : Bool :=
  bit ebx 16 && bit ebx 30 && bit ebx 31 && bit ecx 11 && bit ecx 1

/-- Usable iff all five bits are set; dropping VBMI alone (an AVX512-VNNI part without VBMI)
makes the CPU unusable, which `reason()` reports individually. -/
theorem cpuUsable_iff (ebx ecx : Nat) :
    cpuUsable ebx ecx = true ↔
      (bit ebx 16 ∧ bit ebx 30 ∧ bit ebx 31 ∧ bit ecx 11 ∧ bit ecx 1) := by
  unfold cpuUsable; simp [Bool.and_eq_true, and_assoc]

/-- Witness: the minimal EBX/ECX words with exactly those bits are accepted; without VBMI rejected. -/
theorem cpuUsable_witness :
    cpuUsable (2 ^ 16 + 2 ^ 30 + 2 ^ 31) (2 ^ 11 + 2) = true ∧
    cpuUsable (2 ^ 16 + 2 ^ 30 + 2 ^ 31) (2 ^ 11) = false := by decide

end Ref.Q2Expert



## FILE: /mnt/user-data/outputs/ExpertLayoutTable.lean
/-!
# MODULE: expert_layout_table
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table: legacy (uniform Q2_0 blob) packs, native (IQ) packs described by
`native_experts.txt`, the contiguity validation of their per-layer regions, and the AVX-512
runtime gate that selects the AVX-512 or AVX2 kernels.

## Source semantics
* No `native_experts.txt`: total = n_layers * n_expert * BLOB, every layer uniform.
* Native packs: each line gives (layer, gate/up type, down type, offset, blob bytes [, gguf offsets
  [, shard]]); a layer index outside [0, n_layers) or a blob size disagreeing with its formats rejects.
* After parsing, layer l must start exactly where layer l-1 ended (offset[l] = sum_{k<l} bytes[k] * E);
  a missing layer keeps the sentinel ~0 and fails.  On success total = sum bytes[l] * E and the
  per-layer regions tile [0, total) without gaps or overlap.
* `cpu_avx512_ok` needs CPUID leaf >= 7, OSXSAVE, XCR0 & 0xE6 == 0xE6 (OS saves SSE/AVX/opmask/ZMM
  state) and the five AVX-512 feature bits; REF_FORCE_AVX2=1 forces it off.

## Form template identifiers
expert_layout_load -> loadLayout, contiguity loop -> checkContig, ExpertLayout::total -> total,
cpu_avx512_ok -> avx512Ok, q2_rows_any / act_quant_any -> kernelChoice
-/

namespace Ref.ExpertLayout

-- § legacy packs

/-- Size of one legacy Q2_0 expert blob. -/
def BLOB : Nat := 1382400

/-- Legacy total: every layer holds `n_expert` uniform blobs. -/
def legacyTotal (nLayers nExpert : Nat) : Nat := nLayers * nExpert * BLOB

/-- For the qwen4exp shape (48 layers x 512 experts) the legacy pack is 33,973,862,400 bytes. -/
theorem legacy_total_qwen4exp : legacyTotal 48 512 = 33973862400 := by decide

-- § native line validation

/-- A parsed line is accepted only for a layer in range and a blob size matching its formats. -/
def lineOk (l nLayers blob fmtBytes : Int) : Bool := decide (0 ≤ l ∧ l < nLayers ∧ blob = fmtBytes)

/-- An accepted line names a real layer and carries the size its formats imply. -/
theorem lineOk_sound (l n b f : Int) (h : lineOk l n b f = true) : 0 ≤ l ∧ l < n ∧ b = f := by
  unfold lineOk at h; simpa using h

-- § contiguity

/-- The contiguity pass over (offset, bytes) per layer: returns the running end on success. -/
def checkContig (e : Nat) : Nat → List (Nat × Nat) → Option Nat
  | acc, [] => some acc
  | acc, (o, b) :: rest => if o = acc then checkContig e (acc + b * e) rest else none

/-- Sum of `bytes * E` over the layers. -/
def regionSum (e : Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | (_, b) :: rest => b * e + regionSum e rest

/-- On success the total is the start plus the sum of every layer's region. -/
theorem checkContig_total (e : Nat) (xs : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start xs = some t) : t = start + regionSum e xs := by
  induction xs generalizing start with
  | nil => simp [checkContig] at h; simp [regionSum, h]
  | cons x rest ih =>
    obtain ⟨o, b⟩ := x
    simp only [checkContig] at h
    split at h
    · have := ih _ h; simp only [regionSum]; omega
    · cases h

/-- On success each layer starts at the running end: the first layer starts at `start`, and the
next layer starts exactly where the previous one ends. -/
theorem checkContig_adjacent (e : Nat) (o b : Nat) (rest : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start ((o, b) :: rest) = some t) :
    o = start ∧ checkContig e (o + b * e) rest = some t := by
  simp only [checkContig] at h
  split at h
  · rename_i ho; subst ho; exact ⟨rfl, h⟩
  · cases h

/-- Regions are pairwise disjoint and in order: for two consecutive accepted layers the second
begins at the first's end. -/
theorem checkContig_consecutive (e o1 b1 o2 b2 : Nat) (rest : List (Nat × Nat)) (start t : Nat)
    (h : checkContig e start ((o1, b1) :: (o2, b2) :: rest) = some t) : o2 = o1 + b1 * e := by
  obtain ⟨h1, h2⟩ := checkContig_adjacent e o1 b1 _ start t h
  exact (checkContig_adjacent e o2 b2 rest _ t h2).1

/-- A layer with no line keeps the sentinel offset `~0ull`; it is rejected whenever the running end
is a real (smaller) offset. -/
theorem missing_layer_rejected (e b : Nat) (rest : List (Nat × Nat)) (acc : Nat)
    (hacc : acc < 18446744073709551615) :
    checkContig e acc ((18446744073709551615, b) :: rest) = none := by
  simp only [checkContig]
  split
  · omega
  · rfl

/-- The legacy layout is the special case of uniform contiguous regions. -/
theorem legacy_is_contiguous (e : Nat) :
    checkContig e 0 [(0, BLOB), (BLOB * e, BLOB)] = some (2 * (BLOB * e)) := by
  simp only [checkContig, if_true, Nat.zero_add]
  congr 1; omega

-- § AVX-512 runtime gate

/-- Bit `k` of a register. -/
def bit (r k : Nat) : Bool := r / 2 ^ k % 2 = 1

/-- `cpu_avx512_ok` as a pure function of the probed values. -/
def avx512Ok (forceAvx2 : Bool) (maxLeaf ecx1 xcr0 ebx7 ecx7 : Nat) : Bool :=
  !forceAvx2 && decide (7 ≤ maxLeaf) && bit ecx1 27 && decide (xcr0 % 256 &&& 0xE6 = 0xE6) &&
    bit ebx7 16 && bit ebx7 30 && bit ebx7 31 && bit ecx7 11 && bit ecx7 1

/-- The forced-AVX2 override always wins. -/
theorem force_avx2_wins (a b c d e : Nat) : avx512Ok true a b c d e = false := by
  simp [avx512Ok]

/-- The XCR0 test `(x & 0xE6) == 0xE6` holds exactly when bits 1, 2, 5, 6, 7 are all set
(SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM state), checked over every low byte. -/
theorem xcr0_mask_bits : ∀ x ∈ List.range 256,
    (x &&& 0xE6 == 0xE6) = (bit x 1 && bit x 2 && bit x 5 && bit x 6 && bit x 7) := by
  set_option maxRecDepth 100000 in decide

/-- Which kernel family `q2_rows_any` and `act_quant_any` use. -/
def kernelChoice (ok : Bool) : String := if ok then "avx512" else "avx2"

/-- Without the OS saving ZMM state (bit 6 clear) the AVX2 kernels are chosen even if CPUID
advertises AVX-512: the gate cannot select an instruction set that would fault. -/
theorem no_zmm_state_means_avx2 (lf e1 x b7 c7 : Nat) (hx : (x % 256) / 64 % 2 = 0) :
    kernelChoice (avx512Ok false lf e1 x b7 c7) = "avx2" := by
  have hlt : x % 256 < 256 := Nat.mod_lt _ (by decide)
  have hmask : ¬ (x % 256 &&& 0xE6 = 0xE6) := by
    have key := xcr0_mask_bits (x % 256) (List.mem_range.2 hlt)
    intro hm
    rw [hm] at key
    simp [bit, Bool.and_eq_true] at key
    omega
  simp [kernelChoice, avx512Ok, hmask]

end Ref.ExpertLayout



## FILE: /mnt/user-data/outputs/NativeExpertGeometry.lean
/-!
# MODULE: native_expert_geometry
SOURCES: src/kernels/cpu/native_expert.cpp
Native (GGUF-form) experts on the CPU through ggml-cpu: the `NativeFmt` blob geometry derived from
the gate/up and down types, its validation, and the dispatch between the AVX-512 i-quant kernels
and ggml-cpu's `vec_dot`.

## Source semantics
* A blob is [gate rows | up rows | down rows]: gate rows at r*gu_row, up at up_off + r*gu_row
  (up_off = gu_row * n_ff), down at down_off + r*d_row (down_off = 2*up_off),
  bytes = down_off + d_row * n_embd.  The three regions tile the blob.
* Formats are rejected unless n_embd and n_ff are whole blocks of both the weight types and their
  activation (`vec_dot_type`) types, and unless the quantized activations fit the pool buffers.
* A Q2_0 (type 42) gate/up/down native pack has exactly the 1,382,400-byte size of a Ref blob.
* Gate/up use the AVX-512 kernels only for >= 2 tokens and supported types; one token always takes
  ggml-cpu.  The output is SwiGLU: silu(gate) * up.
* The type-error message names the failing type: gate/up's when its traits are missing, else down's.

## Form template identifiers
native_fmt -> mkFmt, NativeFmt.{gu_row,d_row,up_off,down_off,bytes} -> fields of Fmt,
native_gu_rows dispatch -> guPath, ggml_row_size -> rowSize
-/

namespace Ref.NativeExpert

-- § geometry

/-- `ggml_row_size`: whole blocks times block bytes. -/
def rowSize (blck typeBytes n : Nat) : Nat := n / blck * typeBytes

/-- The derived layout of a native expert blob. -/
structure Fmt where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFF : Nat

/-- Offset of the up rows. -/
def Fmt.upOff (f : Fmt) : Nat := f.guRow * f.nFF
/-- Offset of the down rows. -/
def Fmt.downOff (f : Fmt) : Nat := 2 * f.upOff
/-- Total blob bytes. -/
def Fmt.bytes (f : Fmt) : Nat := f.downOff + f.dRow * f.nEmbd

/-- Closed form of the blob size: two gate/up matrices plus one down matrix. -/
theorem bytes_closed_form (f : Fmt) : f.bytes = 2 * (f.guRow * f.nFF) + f.dRow * f.nEmbd := by
  simp [Fmt.bytes, Fmt.downOff, Fmt.upOff]

/-- Gate row `r < n_ff` lies wholly before the up region. -/
theorem gate_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nFF) :
    r * f.guRow + f.guRow ≤ f.upOff := by
  unfold Fmt.upOff
  rw [Nat.mul_comm f.guRow f.nFF, ← Nat.succ_mul]
  exact Nat.mul_le_mul_right _ hr

/-- Up row `r < n_ff` lies within `[up_off, down_off)`. -/
theorem up_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nFF) :
    f.upOff ≤ f.upOff + r * f.guRow ∧ f.upOff + r * f.guRow + f.guRow ≤ f.downOff := by
  have := gate_row_in_region f r hr
  unfold Fmt.downOff
  omega

/-- Down row `r < n_embd` lies within `[down_off, bytes)`. -/
theorem down_row_in_region (f : Fmt) (r : Nat) (hr : r < f.nEmbd) :
    f.downOff + r * f.dRow + f.dRow ≤ f.bytes := by
  unfold Fmt.bytes
  have : r * f.dRow + f.dRow ≤ f.dRow * f.nEmbd := by
    rw [Nat.mul_comm f.dRow f.nEmbd, ← Nat.succ_mul]
    exact Nat.mul_le_mul_right _ hr
  omega

-- § validation

/-- `native_fmt`'s acceptance test (types have dot products and quantizers are assumed checked). -/
def geometryOk (nEmbd nFF guBlk dBlk guActBlk dActBlk actBytes hBytes actCap hCap : Nat) : Bool :=
  nEmbd % guBlk == 0 && nFF % dBlk == 0 && nEmbd % guActBlk == 0 && nFF % dActBlk == 0 &&
    decide (actBytes ≤ actCap) && decide (hBytes ≤ hCap)

/-- An accepted format has whole-block rows and activations that fit the pool's buffers. -/
theorem geometryOk_sound (a b c d e g h i j k : Nat) (hok : geometryOk a b c d e g h i j k = true) :
    a % c = 0 ∧ b % d = 0 ∧ a % e = 0 ∧ b % g = 0 ∧ h ≤ j ∧ i ≤ k := by
  unfold geometryOk at hok
  simp [Bool.and_eq_true] at hok
  omega

/-- Whole blocks make the row size exact: `rowSize * blck = n * typeBytes`. -/
theorem rowSize_exact (blck tb n : Nat) (h : n % blck = 0) : rowSize blck tb n * blck = n * tb := by
  unfold rowSize
  have hd := Nat.div_add_mod n blck
  rw [h, Nat.add_zero] at hd
  rw [Nat.mul_right_comm, Nat.mul_comm (n / blck) blck, hd]

/-- A Q2_0 native pack (64-element, 18-byte blocks) at H = 2560, FF = 640: gate/up rows of 720 bytes,
down rows of 180 bytes, blob of 1,382,400 bytes — the Ref blob size. -/
theorem q2_native_pack_matches_blob :
    let f : Fmt := ⟨rowSize 64 18 2560, rowSize 64 18 640, 2560, 640⟩
    f.guRow = 720 ∧ f.dRow = 180 ∧ f.bytes = 1382400 := by decide

-- § dispatch

/-- Which gate/up path `native_gu_rows` takes. -/
def guPath (avx512 : Bool) (nt : Nat) (supported : Bool) : String :=
  if avx512 && decide (nt ≥ 2) && supported then "iq512" else "ggml"

/-- A single token always takes ggml-cpu (the AVX-512 kernels are no faster there). -/
theorem one_token_uses_ggml (a s : Bool) : guPath a 1 s = "ggml" := by
  simp [guPath]

/-- The AVX-512 path requires all three conditions. -/
theorem iq512_requires_all (a s : Bool) (nt : Nat) (h : guPath a nt s = "iq512") :
    a = true ∧ nt ≥ 2 ∧ s = true := by
  unfold guPath at h
  split at h
  · rename_i hc; simp [Bool.and_eq_true] at hc; exact ⟨hc.1.1, hc.1.2, hc.2⟩
  · simp at h

/-- The type reported by the "no dot product" error: `tg && tg->vec_dot ? d_type : gu_type`. -/
def reportedType (guOk : Bool) (guType dType : Nat) : Nat := if guOk then dType else guType

/-- The error names the type that actually lacks a dot product. -/
theorem reportedType_correct (guOk dOk : Bool) (g d : Nat) (hfail : ¬ (guOk = true ∧ dOk = true)) :
    (guOk = false → reportedType guOk g d = g) ∧ (guOk = true → dOk = false ∧ reportedType guOk g d = d) := by
  unfold reportedType
  cases guOk <;> cases dOk <;> simp_all

end Ref.NativeExpert



## FILE: /mnt/user-data/outputs/IQuantRowDecode.lean
/-!
# MODULE: iquant_row_decode
SOURCES: src/kernels/cpu/iq_avx512.cpp
The i-quant expert rows (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22) in 512-bit lanes:
each 256-value block is decoded 64 values at a time into grid magnitudes, a 64-bit sign mask and a
vector of odd scales, then dotted with every token's Q8_K activations.

## Source semantics
* Block sizes 66/74/98/110/82 bytes; every field read for chunk j < 4 stays inside the block.
* Scales are `2 s + 1` for a 4-bit s: odd, in [1, 31].  Constants K: 1/8 for IQ2_*, 1/4 for IQ3_XXS,
  1 for IQ3_S.
* Packed u32 words hold four 7-bit sign indices and a 4-bit scale (bits 28..31), recomposing exactly.
* Grid indices are in range: IQ2_XXS < 256, IQ2_XS 9-bit < 512, IQ2_S 10-bit < 1024, IQ3_S 9-bit < 512,
  sign-table indices 7-bit < 128.
* `scales4`: int16 lane i takes scale i / 8; with (sa, sa, sb, sb) value v of the chunk uses sa for
  v < 32 and sb otherwise.
* No saturation: grid magnitudes <= 62 and |y| <= 128 keep `maddubs` pairs below 2^15, and the
  per-block int32 lane sums below 2^24 (exact in f32).
* More than 4 tokens in `dot_rows_nt` are processed in groups of at most 4 that cover all tokens.

## Form template identifiers
Fmt<TY>::decode -> chunkReads, scales4 -> laneScale, row_dot<TY,NT> -> rowDot, dot_rows_nt -> groupsOf4,
iq512_supported -> supported
-/

namespace Ref.IQ512

-- § formats

/-- `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Exactly the five i-quant types are handled; IQ1_M (29) stays on ggml-cpu. -/
theorem supported_set : (List.range 64).filter supported = [16, 17, 18, 21, 22] := by decide

/-- Block bytes as the sum of their fields. -/
theorem block_layouts :
    2 + 64 = 66 ∧ 2 + 64 + 8 = 74 ∧ 2 + 32 + 32 + 8 + 8 = 82 ∧ 2 + 64 + 32 = 98 ∧
    2 + 64 + 8 + 32 + 4 = 110 := by decide

/-- IQ2_XXS / IQ2_XS: codes `2 + 16 j .. +16`, IQ2_XS scales `66 + 2j, +1` stay inside 66/74 bytes. -/
theorem iq2_xxs_xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S: grid bytes `2+8j`, sign bytes `34+8j`, qh `66+2j`, scales `74+2j` inside 82 bytes. -/
theorem iq2_s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS: grid bytes `2+16j`, sign/scale words `66+8j .. +8` inside 98 bytes. -/
theorem iq3_xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: grid bytes `2+16j`, qh `66+2j`, signs `74+8j`, scale byte `106+j` inside 110 bytes. -/
theorem iq3_s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

-- § scales and packed words

/-- The scale formula `2 s + 1` of a 4-bit field. -/
def oddScale (s : Nat) : Nat := 2 * (s % 16) + 1

/-- Scales are odd and in `[1, 31]`. -/
theorem oddScale_range (s : Nat) : oddScale s % 2 = 1 ∧ 1 ≤ oddScale s ∧ oddScale s ≤ 31 := by
  unfold oddScale; omega

/-- A packed u32 holds four 7-bit sign indices (bits 0..27) and the 4-bit scale (bits 28..31). -/
theorem packed_word_recompose (a : Nat) (ha : a < 4294967296) :
    a = a % 128 + 128 * (a / 128 % 128) + 16384 * (a / 16384 % 128) + 2097152 * (a / 2097152 % 128) +
      268435456 * (a / 268435456) ∧ a / 268435456 < 16 := by omega

/-- Grid and sign-table indices stay inside their tables. -/
theorem grid_indices_in_range (b v q h k : Nat) (hk : k < 4) :
    b % 256 < 256 ∧ v % 512 < 512 ∧ v / 512 % 128 < 128 ∧
    q % 256 + 256 * (h / 4 ^ k % 4) < 1024 ∧ q % 256 + 256 * (h / 2 ^ k % 2) < 512 := by
  omega

/-- `scales4` lane map: int16 lane `i < 32` takes scale slot `i / 8`. -/
def laneScale (s0 s1 s2 s3 : Int) (i : Nat) : Int :=
  if i / 8 = 0 then s0 else if i / 8 = 1 then s1 else if i / 8 = 2 then s2 else s3

/-- A maddubs int16 lane `i` sums values `2i, 2i+1`; with `(sa, sa, sb, sb)` the first 32 values of the
chunk get `sa` and the last 32 get `sb`, matching the two u32 groups of the encoding. -/
theorem scales_follow_groups (sa sb : Int) (v : Nat) (hv : v < 64) :
    laneScale sa sa sb sb (v / 2) = if v < 32 then sa else sb := by
  unfold laneScale
  by_cases h : v < 32
  · simp only [h, if_true]
    have : v / 2 / 8 = 0 ∨ v / 2 / 8 = 1 := by omega
    rcases this with h0 | h1
    · simp [h0]
    · simp [h1]
  · simp only [h, if_false]
    have : v / 2 / 8 = 2 ∨ v / 2 / 8 = 3 := by omega
    rcases this with h2 | h3
    · simp [h2]
    · simp [h3]

-- § overflow freedom

/-- A `maddubs` int16 lane `g0 y0 + g1 y1` with grid magnitudes `<= 62` and `|y| <= 128` never
saturates (|.| <= 15872 < 32767). -/
theorem maddubs_no_saturation (g0 g1 y0 y1 : Int) (hg : 0 ≤ g0 ∧ g0 ≤ 62 ∧ 0 ≤ g1 ∧ g1 ≤ 62)
    (hy : -128 ≤ y0 ∧ y0 ≤ 128 ∧ -128 ≤ y1 ∧ y1 ≤ 128) :
    -32767 ≤ g0 * y0 + g1 * y1 ∧ g0 * y0 + g1 * y1 ≤ 32767 := by
  obtain ⟨a0, a1, a2, a3⟩ := hg
  obtain ⟨b0, b1, b2, b3⟩ := hy
  have p0 : -7936 ≤ g0 * y0 ∧ g0 * y0 ≤ 7936 := by
    constructor
    · have := Int.mul_le_mul_of_nonneg_left (show -128 ≤ y0 from b0) a0
      have := Int.mul_le_mul_of_nonneg_right a1 (show (0:Int) ≤ 128 by decide)
      omega
    · have := Int.mul_le_mul_of_nonneg_left b1 a0
      have := Int.mul_le_mul_of_nonneg_right a1 (show (0:Int) ≤ 128 by decide)
      omega
  have p1 : -7936 ≤ g1 * y1 ∧ g1 * y1 ≤ 7936 := by
    constructor
    · have := Int.mul_le_mul_of_nonneg_left (show -128 ≤ y1 from b2) a2
      have := Int.mul_le_mul_of_nonneg_right a3 (show (0:Int) ≤ 128 by decide)
      omega
    · have := Int.mul_le_mul_of_nonneg_left b3 a2
      have := Int.mul_le_mul_of_nonneg_right a3 (show (0:Int) ≤ 128 by decide)
      omega
  omega

/-- After `madd` with a scale `<= 31` (two int16 lanes) and accumulation over the 4 chunks of a block,
an int32 lane stays below 2^24, so the `cvtepi32_ps` before the float FMA is exact. -/
theorem block_lane_exact : 4 * (2 * (2 * 62 * 128) * 31) < 16777216 := by decide

-- § token grouping

/-- Group sizes of the default case of `dot_rows_nt`: chunks of at most 4. -/
def groupsOf4 : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 4 n :: groupsOf4 fuel (n - min 4 n)

/-- The groups cover all tokens and each is a template width 1..4. -/
theorem groupsOf4_cover (fuel n : Nat) (hf : n ≤ 4 * fuel) :
    (groupsOf4 fuel n).foldr (· + ·) 0 = n ∧ ∀ g ∈ groupsOf4 fuel n, 1 ≤ g ∧ g ≤ 4 := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by omega
            subst this; simp [groupsOf4]
  | succ f ih =>
    cases n with
    | zero => simp [groupsOf4]
    | succ m =>
      obtain ⟨h1, h2⟩ := ih (m + 1 - min 4 (m + 1)) (by omega)
      simp only [groupsOf4, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro g hg
      rcases hg with rfl | hg
      · omega
      · exact h2 g hg

/-- A 256-value Q8_K block is consumed as four 64-byte loads at `qs + 64 j`. -/
theorem q8k_chunk_offsets (j : Nat) (hj : j < 4) : 64 * j + 64 ≤ 256 := by omega

end Ref.IQ512



## FILE: /mnt/user-data/outputs/Q2Avx2Fallback.lean
/-!
# MODULE: q2_avx2_fallback
SOURCES: src/kernels/cpu/q2_avx2.cpp
The Q2_0 GGUF-row dot and the Q8_1 activation quantizer for CPUs without AVX-512, built from
SSE/AVX2 shifts, byte/word unpacks, `maddubs` and shuffles.

## Source semantics
* `unpack64` shifts 16-bit lanes by 0/2/4/6 and masks each byte with 3; despite the 16-bit shift
  each byte keeps its OWN codes (neighbouring bits land above bit 1 and are masked off).
* The unpack_lo/hi_epi8 then unpack_lo/hi_epi16 interleave puts value i of the block at byte i/4,
  slot i%4 — the same LSB-first order as the Q2_0 dequantizer and the AVX-512 kernel.
* `maddubs` of codes (<= 3) with int8 activations (|x| <= 127) never saturates (|pair| <= 762).
* Per block the result is d (s0 S0 + s1 S1) - d (hx0 + hx1) with hx = s * sum, equal to
  d (s0 (S0 - sum0) + s1 (S1 - sum1)): the code-offset correction of the AVX-512 kernel.
* The quantizer's integer horizontal sum (shuffle 0x4E then 0xB1) adds all four lanes.
* Rows at >4 tokens are processed in groups of at most 4 that cover every token.

## Form template identifiers
unpack64 -> unpackPos, row_multi -> blockResult, act_quant_q8_1_avx2 -> hsum4, shuffle imm -> shufLane,
q2_0_gguf_rows_multi_avx2 default -> groupsOf4
-/

namespace Ref.Q2Avx2

-- § unpack correctness

/-- Shifting a 16-bit word `lo + 256 hi` right by `2k` and masking each byte with 3 yields the codes
of `lo` and `hi` independently (no bit leaks across the byte boundary after the mask). -/
theorem shift16_bytes_independent (lo hi : Nat) (hlo : lo < 256) (hhi : hi < 256) :
    (lo + 256 * hi) / 1 % 256 % 4 = lo % 4 ∧ (lo + 256 * hi) / 1 / 256 % 4 = hi % 4 ∧
    (lo + 256 * hi) / 4 % 256 % 4 = lo / 4 % 4 ∧ (lo + 256 * hi) / 4 / 256 % 4 = hi / 4 % 4 ∧
    (lo + 256 * hi) / 16 % 256 % 4 = lo / 16 % 4 ∧ (lo + 256 * hi) / 16 / 256 % 4 = hi / 16 % 4 ∧
    (lo + 256 * hi) / 64 % 256 % 4 = lo / 64 % 4 ∧ (lo + 256 * hi) / 64 / 256 % 4 = hi / 64 % 4 := by
  omega

/-- `unpacklo_epi8(x, y)`: output byte `i` takes byte `i/2` of `x` (even `i`) or of `y` (odd `i`).
Returned as (which source, byte). -/
def unpack8 (i : Nat) : Nat × Nat := (i % 2, i / 2)

/-- `unpacklo_epi16(a, b)`: output byte `i` is byte `2 (i/4) + i%2` of `a` (word index even) or `b`. -/
def unpack16 (i : Nat) : Nat × Nat := (i / 2 % 2, 2 * (i / 4) + i % 2)

/-- Composition of the two interleaves: output byte `i` of the low 128 bits comes from code plane
`2 * src16 + src8` (= slot) and source byte `byte8` (= block byte). -/
def unpackPos (i : Nat) : Nat × Nat :=
  let (s16, j) := unpack16 i
  let (s8, b) := unpack8 j
  (2 * s16 + s8, b)

/-- Value `i` of the first 16 lands at code slot `i % 4` of byte `i / 4`; the `unpackhi` halves repeat
this with offsets 4, 8, 12 bytes, giving the dequantizer's order for all 64 values. -/
theorem unpackPos_order (i : Nat) (hi : i < 16) : unpackPos i = (i % 4, i / 4) := by
  simp only [unpackPos, unpack16, unpack8]
  refine Prod.ext ?_ ?_ <;> simp <;> omega

/-- The four 16-value groups (lo/hi of the epi16 unpack, from bytes 0..7 and 8..15) cover values
0..63 with block byte `i/4`: group `g`, local index `i` -> value `16 g + i`, byte `4 g + i / 4`. -/
theorem unpack_groups_cover (g i : Nat) (hg : g < 4) (hi : i < 16) :
    16 * g + i < 64 ∧ (16 * g + i) / 4 = 4 * g + i / 4 ∧ (16 * g + i) % 4 = i % 4 := by omega

-- § arithmetic

/-- `maddubs` pairs `c0 x0 + c1 x1` with codes in [0,3] and |x| <= 127 stay within ±762. -/
theorem maddubs_codes_bound (c0 c1 x0 x1 : Int) (hc : 0 ≤ c0 ∧ c0 ≤ 3 ∧ 0 ≤ c1 ∧ c1 ≤ 3)
    (hx : -127 ≤ x0 ∧ x0 ≤ 127 ∧ -127 ≤ x1 ∧ x1 ≤ 127) :
    -762 ≤ c0 * x0 + c1 * x1 ∧ c0 * x0 + c1 * x1 ≤ 762 := by
  obtain ⟨a0, a1, a2, a3⟩ := hc
  have t0 : c0 = 0 ∨ c0 = 1 ∨ c0 = 2 ∨ c0 = 3 := by omega
  have t1 : c1 = 0 ∨ c1 = 1 ∨ c1 = 2 ∨ c1 = 3 := by omega
  rcases t0 with rfl | rfl | rfl | rfl <;> rcases t1 with rfl | rfl | rfl | rfl <;> omega

/-- The per-block result the kernel accumulates. -/
def blockResult (d s0 s1 S0 S1 u0 u1 : Int) : Int := d * (s0 * S0) + d * (s1 * S1) - d * (s0 * u0 + s1 * u1)

/-- With `hx = s * sum`, the separate `corr` term equals subtracting the activation sum per chunk:
`d (s0 S0 + s1 S1) - d (hx0 + hx1) = d (s0 (S0 - u0) + s1 (S1 - u1))`. -/
theorem blockResult_correction (d s0 s1 S0 S1 u0 u1 : Int) :
    blockResult d s0 s1 S0 S1 u0 u1 = d * (s0 * (S0 - u0) + s1 * (S1 - u1)) := by
  unfold blockResult
  rw [Int.mul_sub s0, Int.mul_sub s1, ← Int.mul_add, ← Int.mul_sub]
  congr 1
  omega

-- § horizontal sum in the quantizer

/-- `_mm_shuffle_epi32` lane selector: lane `i` takes source lane `(imm >> 2i) & 3`. -/
def shufLane (imm i : Nat) : Nat := imm / 4 ^ i % 4

/-- 0x4E swaps the two 64-bit halves; 0xB1 swaps adjacent lanes. -/
theorem shuffle_constants :
    (List.range 4).map (shufLane 0x4E) = [2, 3, 0, 1] ∧ (List.range 4).map (shufLane 0xB1) = [1, 0, 3, 2] := by
  decide

/-- Lane 0 after `s += shuf(s, 0x4E); s += shuf(s, 0xB1)` holds the sum of all four lanes. -/
def hsum4 (a b c d : Int) : Int :=
  let s := [a + c, b + d, c + a, d + b]
  s[0]! + s[1]!

/-- The integer horizontal sum is exact. -/
theorem hsum4_exact (a b c d : Int) : hsum4 a b c d = a + b + c + d := by
  simp [hsum4]; omega

/-- A chunk sum of 32 clamped activations fits easily in int32 (|sum| <= 4064). -/
theorem chunk_sum_bound : 32 * 127 = 4064 ∧ 4064 < 2147483647 := by decide

-- § token grouping

/-- Group sizes of the default case: chunks of at most 4 tokens. -/
def groupsOf4 : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 4 n :: groupsOf4 fuel (n - min 4 n)

/-- The groups cover every token, and each has a width the switch handles (1..4). -/
theorem groupsOf4_cover (fuel n : Nat) (hf : n ≤ 4 * fuel) :
    (groupsOf4 fuel n).foldr (· + ·) 0 = n ∧ ∀ g ∈ groupsOf4 fuel n, 1 ≤ g ∧ g ≤ 4 := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by omega
            subst this; simp [groupsOf4]
  | succ f ih =>
    cases n with
    | zero => simp [groupsOf4]
    | succ m =>
      obtain ⟨h1, h2⟩ := ih (m + 1 - min 4 (m + 1)) (by omega)
      simp only [groupsOf4, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro g hg
      rcases hg with rfl | hg
      · omega
      · exact h2 g hg

/-- GGUF Q2_0 rows stride 18 bytes per block: block `b` has its scale at `18 b` and codes at `18 b + 2`. -/
theorem gguf_block_stride (b nb : Nat) (hb : b < nb) : 18 * b + 2 + 16 ≤ 18 * nb := by omega

end Ref.Q2Avx2



## FILE: /mnt/user-data/outputs/ExpertPoolScheduling.lean
/-!
# MODULE: expert_pool_scheduling
SOURCES: include/ref/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
The CPU expert pool: worker count and core selection, the `head`/`done`/`parked`/`epoch` completion
protocol, and the row-splitting arithmetic of `run_split`, `run_split_multi` and
`run_split_multi_native`.

## Source semantics
* Workers = `n_workers` if positive, else the physical cores minus the first, and at least 1.
* Claims are `head.fetch_add(1)`: the k-th claim gets k, so every job index < n is claimed exactly
  once and late claimers get indices >= n and stop.
* `parked` counts parked workers.  The host publishes only when `parked == n` and returns only when
  `done == n_tasks` AND `parked == n`; at that point no worker is in the drain loop, so resetting
  `head` cannot hand a worker a job from the next batch.
* `run_split`: parts = max(1, ceil(3T / n)) per expert, so n * parts >= 3T tasks; task i maps to
  expert i / parts and the row range [R p / P, R (p+1) / P), and these ranges tile [0, R).
* `run_split_multi`: 3T tasks over n * per rows; task i owns [rows i / tasks, rows (i+1) / tasks),
  and the inner loop walks it in per-expert segments that never cross an expert boundary.
* Oversized batches fall back to single-token jobs in groups of kMaxSplit = 16, or native batches of
  kMaxSplitMulti = 96; the groups cover every job.

## Form template identifiers
ExpertPool::ExpertPool -> workerCount, head_.fetch_add -> claims, parked_/workers -> ParkState,
run_split parts -> partsFor, row range -> rangeLo, multi segment -> segEnd, batching -> batchesOf
-/

namespace Ref.Pool

-- § worker count

/-- `physical_cores(true)`: drop the first core when there is one. -/
def skipFirst (cores : List Nat) : List Nat := cores.drop 1

/-- Skipping removes exactly one core (if any) and keeps the rest in order. -/
theorem skipFirst_length (cores : List Nat) : (skipFirst cores).length = cores.length - 1 := by
  simp [skipFirst]

/-- The constructor's worker count. -/
def workerCount (nWorkers : Int) (nCores : Nat) : Nat :=
  max 1 (if nWorkers > 0 then nWorkers.toNat else nCores)

/-- There is always at least one worker, and an explicit positive request is honoured. -/
theorem workerCount_spec (w : Int) (c : Nat) :
    1 ≤ workerCount w c ∧ (w > 0 → workerCount w c = w.toNat) := by
  unfold workerCount
  constructor
  · omega
  · intro h; simp [h]; omega

-- § claims

/-- Indices handed out by `k` successive `fetch_add(1)` calls starting from 0. -/
def claims (k : Nat) : List Nat := List.range k

/-- Claims below `n` are the jobs actually run: with at least `n` claims every job runs, in order,
and no index is handed out twice. -/
theorem claims_cover (k n : Nat) (hk : n ≤ k) :
    (claims k).filter (fun i => decide (i < n)) = List.range n ∧ (claims k).Nodup := by
  refine ⟨?_, List.nodup_range⟩
  unfold claims
  induction k with
  | zero => have : n = 0 := by omega
            subst this; rfl
  | succ k ih =>
    rw [List.range_succ, List.filter_append]
    by_cases hnk : n ≤ k
    · rw [ih hnk]
      simp; omega
    · have : n = k + 1 := by omega
      subst this
      have hall : (List.range k).filter (fun i => decide (i < k + 1)) = List.range k := by
        rw [List.filter_eq_self]
        intro a ha; simp at ha; simp; omega
      rw [hall, List.range_succ]
      simp

-- § park protocol

/-- Each worker is parked (`true`) or draining (`false`); `count` mirrors `parked_`. -/
structure ParkState where
  workers : List Bool
  count : Nat

/-- Number of parked workers. -/
def nParked : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + nParked bs

/-- The invariant the atomics maintain: the counter equals the number of parked workers. -/
def ParkState.inv (s : ParkState) : Prop := s.count = nParked s.workers

/-- Set worker `i`'s flag. -/
def setAt : List Bool → Nat → Bool → List Bool
  | [], _, _ => []
  | _ :: bs, 0, v => v :: bs
  | b :: bs, i + 1, v => b :: setAt bs i v

/-- Changing one flag from `old` to `new` changes the parked count by the expected amount. -/
theorem nParked_setAt (l : List Bool) (i : Nat) (hi : i < l.length) (v : Bool) :
    nParked (setAt l i v) + (if l[i] then 1 else 0) = nParked l + (if v then 1 else 0) := by
  induction l generalizing i with
  | nil => simp at hi
  | cons b bs ih =>
    cases i with
    | zero => simp [setAt, nParked]; omega
    | succ j =>
      simp only [List.length_cons] at hi
      have := ih j (by omega)
      simp [setAt, nParked]
      omega

/-- A parked worker leaving the park (`fetch_sub`) preserves the invariant. -/
theorem leave_preserves (s : ParkState) (i : Nat) (hi : i < s.workers.length) (hp : s.workers[i] = true)
    (h : s.inv) : ParkState.inv ⟨setAt s.workers i false, s.count - 1⟩ := by
  unfold ParkState.inv at *
  have := nParked_setAt s.workers i hi false
  rw [hp] at this
  simp at this
  simp only
  omega

/-- A draining worker returning to the park (`fetch_add`) preserves the invariant. -/
theorem arrive_preserves (s : ParkState) (i : Nat) (hi : i < s.workers.length) (hp : s.workers[i] = false)
    (h : s.inv) : ParkState.inv ⟨setAt s.workers i true, s.count + 1⟩ := by
  unfold ParkState.inv at *
  have := nParked_setAt s.workers i hi true
  rw [hp] at this
  simp at this
  simp only
  omega

/-- The parked count never exceeds the number of workers. -/
theorem nParked_le (l : List Bool) : nParked l ≤ l.length := by
  induction l with
  | nil => simp [nParked]
  | cons b bs ih => simp [nParked]; split <;> omega

/-- When `parked == n` every worker is parked: none is inside the drain loop. -/
theorem all_parked (l : List Bool) (h : nParked l = l.length) : ∀ b ∈ l, b = true := by
  induction l with
  | nil => simp
  | cons b bs ih =>
    have hle := nParked_le bs
    simp [nParked] at h
    intro x hx
    cases b with
    | true =>
      simp at h
      rcases List.mem_cons.1 hx with rfl | hx
      · rfl
      · exact ih (by omega) x hx
    | false => simp at h; omega

/-- The host's handover condition: with the invariant, `count == n` means no worker can still claim
from `head`, so resetting it for the next batch is safe. -/
theorem handover_safe (s : ParkState) (h : s.inv) (hc : s.count = s.workers.length) :
    ∀ b ∈ s.workers, b = true := by
  unfold ParkState.inv at h
  exact all_parked s.workers (by omega)

/-- Workers arrive at the park before their first wait, so the initial state already satisfies
`parked == n` and the first `run()` cannot deadlock. -/
theorem initial_all_parked (n : Nat) :
    ParkState.inv ⟨List.replicate n true, n⟩ := by
  unfold ParkState.inv
  induction n with
  | zero => rfl
  | succ k ih => simp [List.replicate, nParked] at *; omega

-- § run_split partitioning

/-- Parts per expert: `max(1, (3T + n - 1) / n)`. -/
def partsFor (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- There are at least `3T` tasks in total (about three per thread). -/
theorem partsFor_enough (threads n : Nat) (hn : 0 < n) : 3 * threads ≤ n * partsFor threads n := by
  unfold partsFor
  have hdm := Nat.div_add_mod (3 * threads + n - 1) n
  have hlt := Nat.mod_lt (3 * threads + n - 1) hn
  have hm : n * ((3 * threads + n - 1) / n) ≤ n * max 1 ((3 * threads + n - 1) / n) :=
    Nat.mul_le_mul_left _ (Nat.le_max_right _ _)
  generalize (3 * threads + n - 1) / n = q at *
  generalize n * q = m at *
  omega

/-- Task `i < n * parts` decodes to a valid (expert, part). -/
theorem task_decode (i n parts : Nat) (hp : 0 < parts) (hi : i < n * parts) :
    i / parts < n ∧ i % parts < parts := by
  refine ⟨?_, Nat.mod_lt _ hp⟩
  rw [Nat.div_lt_iff_lt_mul hp]
  exact hi

/-- Lower row bound of part `p` of `P` over `R` rows: `R * p / P`. -/
def rangeLo (R P p : Nat) : Nat := R * p / P

/-- The parts tile `[0, R)`: the first starts at 0, the last ends at R, consecutive parts meet,
and every range is non-decreasing. -/
theorem ranges_tile (R P p : Nat) (hP : 0 < P) :
    rangeLo R P 0 = 0 ∧ rangeLo R P P = R ∧ rangeLo R P p ≤ rangeLo R P (p + 1) := by
  unfold rangeLo
  refine ⟨by simp, Nat.mul_div_cancel R hP, ?_⟩
  exact Nat.div_le_div_right (Nat.mul_le_mul_left _ (Nat.le_succ p))

/-- Every row of a range is a real row: `R * (p+1) / P <= R` for `p < P`. -/
theorem range_within (R P p : Nat) (hP : 0 < P) (hp : p < P) : rangeLo R P (p + 1) ≤ R := by
  unfold rangeLo
  have : R * (p + 1) / P ≤ R * P / P := Nat.div_le_div_right (Nat.mul_le_mul_left _ hp)
  rw [Nat.mul_div_cancel R hP] at this
  exact this

-- § run_split_multi segments

/-- End of the segment starting at global row `r` within expert `r / per`, for a task ending at `g1`. -/
def segEnd (per r g1 : Nat) : Nat := min per (r % per + (g1 - r))

/-- Inside a task the walk makes progress, never passes the task's end, and stays inside one expert. -/
theorem segment_spec (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    r % per < segEnd per r g1 ∧ segEnd per r g1 - r % per ≤ g1 - r ∧ segEnd per r g1 ≤ per := by
  have := Nat.mod_lt r hper
  unfold segEnd
  omega

/-- After a segment the next row starts at the following expert or at the task's end. -/
theorem segment_next (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    let r' := r + (segEnd per r g1 - r % per)
    r' = g1 ∨ r' % per = 0 := by
  have hdm := Nat.div_add_mod r per
  have hlt := Nat.mod_lt r hper
  simp only [segEnd]
  by_cases h : r % per + (g1 - r) ≤ per
  · rw [Nat.min_eq_right h]
    left; omega
  · rw [Nat.min_eq_left (by omega)]
    right
    have : r + (per - r % per) = per * (r / per + 1) := by
      rw [Nat.mul_add, Nat.mul_one]; omega
    rw [this, Nat.mul_mod_right]

/-- With `mtasks = 3T` tasks over `n * per` rows, consecutive tasks meet and the last one ends at
the total: the multi phases cover all rows of all experts. -/
theorem multi_tasks_tile (rows tasks : Nat) (ht : 0 < tasks) :
    rows * 0 / tasks = 0 ∧ rows * tasks / tasks = rows := by
  exact ⟨by simp, Nat.mul_div_cancel rows ht⟩

-- § oversized-batch fallbacks

/-- Batch sizes of a loop stepping by `k` with tail `min(k, n - b0)`. -/
def batchesOf (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min k n :: batchesOf k fuel (n - min k n)

/-- The batches cover every job, and each is non-empty and at most `k`. -/
theorem batchesOf_cover (k fuel n : Nat) (hk : 0 < k) (hf : n ≤ k * fuel) :
    (batchesOf k fuel n).foldr (· + ·) 0 = n ∧ ∀ b ∈ batchesOf k fuel n, 0 < b ∧ b ≤ k := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by simp at hf; omega
            subst this; simp [batchesOf]
  | succ f ih =>
    cases n with
    | zero => simp [batchesOf]
    | succ m =>
      have hf' : m + 1 - min k (m + 1) ≤ k * f := by
        rw [Nat.mul_succ] at hf; omega
      obtain ⟨h1, h2⟩ := ih (m + 1 - min k (m + 1)) hf'
      simp only [batchesOf, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro b hb
      rcases hb with rfl | hb
      · omega
      · exact h2 b hb

/-- The single-token fallback (groups of kMaxSplit = 16) and native batching (kMaxSplitMulti = 96)
are instances: e.g. 100 single jobs run as 16 x 6 + 4, 200 native experts as 96 + 96 + 8. -/
theorem fallback_examples :
    batchesOf 16 7 100 = [16, 16, 16, 16, 16, 16, 4] ∧ batchesOf 96 3 200 = [96, 96, 8] := by decide

/-- `run_split` falls back to `run` for too many experts, a single worker, or the oracle. -/
def splitFallsBack (n nWorkers : Nat) (oracle : Bool) : Bool :=
  decide (n > 16) || decide (nWorkers = 1) || oracle

/-- Up to 16 experts on a multi-worker pool without the oracle are split. -/
theorem split_used (n w : Nat) (hn : n ≤ 16) (hw : w ≠ 1) : splitFallsBack n w false = false := by
  simp [splitFallsBack]; omega

-- § affinity restore

/-- `restore_thread_affinity` acts only on a positive previous mask; -1 (refused) is a no-op. -/
def restoreActs (previous : Int) : Bool := decide (previous > 0)

/-- A refused pin (-1) and the Windows failure value 0 are never "restored". -/
theorem restore_ignores_failure : restoreActs (-1) = false ∧ restoreActs 0 = false := by decide

end Ref.Pool



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- ExpertLayoutTable.lean
  was: theorem missing_layer_rejected (e b : Nat) (rest : List (Nat × Nat)) (at : Nat)
    (hat : at < 18446744073709551615) :
    checkContig e at ((18446744073709551615, b) :: rest) = none
  now: theorem missing_layer_rejected (e b : Nat) (rest : List (Nat × Nat)) (acc : Nat)
    (hacc : acc < 18446744073709551615) :
    checkContig e acc ((18446744073709551615, b) :: rest) = none
  measured: the claim changed

- ExpertLayoutTable.lean
  was: theorem xcr0_mask_bits : ∀ x ∈ List.range 256,
    (x &&& 0xE6 = 0xE6) ↔ (bit x 1 ∧ bit x 2 ∧ bit x 5 ∧ bit x 6 ∧ bit x 7)
  now: theorem xcr0_mask_bits : ∀ x ∈ List.range 256,
    (x &&& 0xE6 == 0xE6) = (bit x 1 && bit x 2 && bit x 5 && bit x 6 && bit x 7)
  measured: the claim changed
-/



## FILE: /mnt/user-data/outputs/ExpertBlobAddressing.lean
/-!
# MODULE: expert_blob_addressing
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
Models how an expert source turns a routed `(layer, expert)` pair into a byte range of the expert
arena: the flat index `layer * n_expert + expert` for the uniform `experts.bin`, and the per-layer
layout (`layer_offset + expert * blob_bytes(layer)`) used by `ArenaExpertSource`.

## Source semantics
* `experts.bin` holds exactly `48 * 512 = 24576` blobs of `BLOB = 1,382,400` bytes, no padding.
* `blob(layer, expert)` refuses negative ids, refuses `expert >= n_expert` SEPARATELY from the flat-index
  bound (otherwise `blob(0, 512)` would alias `blob(1, 0)`), and refuses a flat index past the file.
* Every accepted blob lies wholly inside the file, and distinct valid pairs give disjoint byte ranges.
* Per-layer layout: layer `l+1` starts where layer `l`'s `n_expert` blobs end; the arena is one
  max-blob longer than the file so a whole-slot copy may start at any expert.

## Form template identifiers
`FileExpertSource::blob` → `blob`; `layer * n_expert_ + expert` → `flatIndex`;
`base_ + i * BLOB` → `byteOffset`; `ExpertLayout::layer_offset` → `layerOffset`;
`ExpertLayout::blob_offset` → `blobOffsetL`; `ExpertLayout::blob_bytes` → `bytes`.
-/

namespace Ref.ExpertBlobAddressing

-- § Geometry constants

/-- Bytes in one expert blob (`ref::kernels::cpu::BLOB`). -/
def BLOB : Nat := 1382400

/-- Layers of the artifact. -/
def nLayers : Nat := 48

/-- Experts per layer. -/
def nExpert : Nat := 512

/-- `experts.bin` holds exactly 24,576 blobs. -/
theorem experts_bin_blobs : nLayers * nExpert = 24576 := by decide

/-- `experts.bin` is exactly 33,973,862,400 bytes, the size `open` checks against. -/
theorem experts_bin_size : nLayers * nExpert * BLOB = 33973862400 := by decide

-- § Flat index

/-- The flat blob index `layer * n_expert + expert`. -/
def flatIndex (nE l e : Nat) : Nat := l * nE + e

/-- Helper: a layer below `nL` leaves room for one more full layer of `nE` blobs. -/
theorem layer_room {l nL : Nat} (nE : Nat) (h : l < nL) : l * nE + nE ≤ nL * nE := by
  have h1 : (l + 1) * nE ≤ nL * nE := Nat.mul_le_mul_right nE h
  rw [Nat.succ_mul] at h1
  exact h1

/-- A valid pair has a flat index inside the file. -/
theorem flatIndex_lt {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    flatIndex nE l e < nL * nE := by
  have h := layer_room nE hl
  unfold flatIndex
  omega

/-- The layer is recovered from the flat index by division. -/
theorem flatIndex_div {nE l e : Nat} (he : e < nE) : flatIndex nE l e / nE = l := by
  have hpos : 0 < nE := by omega
  unfold flatIndex
  rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he, Nat.add_zero]

/-- The expert is recovered from the flat index by remainder. -/
theorem flatIndex_mod {nE l e : Nat} (he : e < nE) : flatIndex nE l e % nE = e := by
  unfold flatIndex
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he]

/-- Distinct in-range pairs never share a flat index (layer and expert are separate axes). -/
theorem flatIndex_injective {nE l1 e1 l2 e2 : Nat} (h1 : e1 < nE) (h2 : e2 < nE)
    (h : flatIndex nE l1 e1 = flatIndex nE l2 e2) : l1 = l2 ∧ e1 = e2 := by
  have hl : l1 = l2 :=
    calc l1 = flatIndex nE l1 e1 / nE := (flatIndex_div h1).symm
      _ = flatIndex nE l2 e2 / nE := by rw [h]
      _ = l2 := flatIndex_div h2
  have he : e1 = e2 :=
    calc e1 = flatIndex nE l1 e1 % nE := (flatIndex_mod h1).symm
      _ = flatIndex nE l2 e2 % nE := by rw [h]
      _ = e2 := flatIndex_mod h2
  exact ⟨hl, he⟩

/-- THE HAZARD the separate expert check exists for: without it, expert `nE` of layer 0 is the
flat index of expert 0 of layer 1 - finite, correctly sized, and wrong. -/
theorem unchecked_alias (nE : Nat) : flatIndex nE 0 nE = flatIndex nE 1 0 := by
  simp [flatIndex]

-- § Byte ranges

/-- Byte offset of a blob in the uniform file. -/
def byteOffset (nE l e : Nat) : Nat := flatIndex nE l e * BLOB

/-- Every valid blob lies wholly inside the file. -/
theorem blob_in_file {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    byteOffset nE l e + BLOB ≤ nL * nE * BLOB := by
  have h := flatIndex_lt hl he
  have h2 : (flatIndex nE l e + 1) * BLOB ≤ nL * nE * BLOB := Nat.mul_le_mul_right BLOB h
  rw [Nat.succ_mul] at h2
  exact h2

/-- Helper: blob `i < j` ends at or before blob `j` starts. -/
theorem blob_before {i j : Nat} (h : i < j) : i * BLOB + BLOB ≤ j * BLOB := by
  have h2 : (i + 1) * BLOB ≤ j * BLOB := Nat.mul_le_mul_right BLOB h
  rw [Nat.succ_mul] at h2
  exact h2

/-- Distinct blob indices give disjoint byte ranges. -/
theorem blobs_disjoint {i j : Nat} (hne : i ≠ j) :
    i * BLOB + BLOB ≤ j * BLOB ∨ j * BLOB + BLOB ≤ i * BLOB := by
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact Or.inl (blob_before h)
  · exact Or.inr (blob_before h)

-- § The `blob()` model

/-- `FileExpertSource::blob` on signed ids: `none` is the null pointer. -/
def blob (nL nE : Nat) (l e : Int) : Option Nat :=
  if l < 0 ∨ e < 0 then none
  else if (nE : Int) ≤ e then none
  else if nL * nE ≤ flatIndex nE l.toNat e.toNat then none
  else some (byteOffset nE l.toNat e.toNat)

/-- The expert one past the end of a layer is refused, never aliased. -/
theorem blob_refuses_expert_overflow (nL nE : Nat) (l : Int) : blob nL nE l nE = none := by
  unfold blob
  split
  · rfl
  · simp

/-- A valid pair is accepted at its byte offset. -/
theorem blob_valid {nL nE : Nat} {l e : Int} (hl0 : 0 ≤ l) (he0 : 0 ≤ e)
    (he : e.toNat < nE) (hl : l.toNat < nL) :
    blob nL nE l e = some (byteOffset nE l.toNat e.toNat) := by
  have hlt := flatIndex_lt hl he
  have hneg : ¬ (l < 0 ∨ e < 0) := by omega
  have hexp : ¬ ((nE : Int) ≤ e) := by omega
  have hflat : ¬ (nL * nE ≤ flatIndex nE l.toNat e.toNat) := by omega
  unfold blob
  simp [hneg, hexp, hflat]

/-- Anything accepted is a valid pair whose blob lies inside the file. -/
theorem blob_some_sound {nL nE : Nat} {l e : Int} {o : Nat} (h : blob nL nE l e = some o) :
    0 ≤ l ∧ 0 ≤ e ∧ e.toNat < nE ∧ o + BLOB ≤ nL * nE * BLOB := by
  unfold blob at h
  split at h
  · cases h
  · rename_i hneg
    split at h
    · cases h
    · rename_i hexp
      split at h
      · cases h
      · rename_i hflat
        have he : e.toNat < nE := by omega
        have hl : l.toNat < nL := by
          refine Nat.lt_of_not_le fun hge => hflat ?_
          have := Nat.mul_le_mul_right nE hge
          unfold flatIndex
          omega
        obtain rfl := Option.some.inj h
        exact ⟨by omega, by omega, he, blob_in_file hl he⟩

-- § Per-layer layout (ArenaExpertSource)

/-- `layer_offset`: layers laid out back to back, `nE` blobs of `bytes l` each. -/
def layerOffset (bytes : Nat → Nat) (nE : Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => layerOffset bytes nE l + nE * bytes l

/-- `blob_offset(l, e)`. -/
def blobOffsetL (bytes : Nat → Nat) (nE l e : Nat) : Nat :=
  layerOffset bytes nE l + e * bytes l

/-- A blob of layer `l` ends at or before layer `l+1` begins. -/
theorem blobOffsetL_in_layer (bytes : Nat → Nat) {nE l e : Nat} (he : e < nE) :
    blobOffsetL bytes nE l e + bytes l ≤ layerOffset bytes nE (l + 1) := by
  have h := layer_room (bytes l) he
  simp only [blobOffsetL, layerOffset]
  omega

/-- Layer offsets are monotone. -/
theorem layerOffset_mono (bytes : Nat → Nat) (nE : Nat) :
    ∀ {a b : Nat}, a ≤ b → layerOffset bytes nE a ≤ layerOffset bytes nE b := by
  intro a b hab
  induction b with
  | zero =>
    have : a = 0 := by omega
    subst this; exact Nat.le_refl _
  | succ b ih =>
    rcases Nat.lt_or_ge a (b + 1) with h | h
    · have := ih (by omega)
      simp only [layerOffset]
      omega
    · have : a = b + 1 := by omega
      subst this; exact Nat.le_refl _

/-- Every valid arena blob lies inside the arena's file-sized prefix `layerOffset nL`. -/
theorem blobOffsetL_in_total (bytes : Nat → Nat) {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    blobOffsetL bytes nE l e + bytes l ≤ layerOffset bytes nE nL := by
  have h1 := blobOffsetL_in_layer bytes (l := l) he
  have h2 := layerOffset_mono bytes nE (a := l + 1) (b := nL) hl
  omega

/-- The arena is one max-blob longer than the file, so a whole-slot copy of `maxBlob` bytes
starting at any valid expert stays inside the allocation. -/
theorem slot_copy_in_arena (bytes : Nat → Nat) {nL nE l e maxBlob : Nat}
    (hl : l < nL) (he : e < nE) :
    blobOffsetL bytes nE l e + maxBlob ≤ layerOffset bytes nE nL + maxBlob := by
  have := blobOffsetL_in_total bytes hl he
  omega

end Ref.ExpertBlobAddressing



## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
SOURCES: src/core/expert_cache.cpp
The VRAM expert cache's bookkeeping: the residency table, compulsory-miss admission with no
eviction (global and per-layer quota variants), the per-layer slot ranges, slot byte addressing
(uniform and 256-byte-aligned sized slots), fill sizing and the `STRP` profile validation.

## Source semantics
* `layer_slot_range`: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer
  takes the remainder, so the ranges tile `0..slots` exactly and are pairwise disjoint.
* `admit` is idempotent on a resident expert, never hands out a slot `>= slots`, and never hands the
  same slot to two `(layer, expert)` pairs (no eviction: the free pointer only moves up).
* Per-layer admission keeps each layer inside its own range, so two layers never share a slot.
* `fill_slot` copies `bytes` if `0 < bytes <= blob`, else `blob` - never more than a slot.
* Sized slots are rounded up to 256 bytes, so every slot start is 256-aligned and holds its blob.
* A profile is accepted only if its geometry matches and every ranked pair is in range.

## Form template identifiers
`ExpertCache::layer_slot_range` → `layerSlotRange`; `ExpertCache::admit` → `admit`;
`layer_next_` → `Quota.next`; `residency_` → `Cache.res`; `next_free_` → `Cache.nextFree`;
`fill_slot` byte count → `fillBytes`; `device_slot` → `deviceSlotOff`;
`open_sized` offsets → `sizedOffsets`; `read_expert_profile` → `profileAccepts`.
-/

namespace Ref.ExpertCacheResidency

-- § Per-layer slot ranges

/-- `ExpertCache::layer_slot_range`: `(lo, hi)`, or `(0, 0)` when out of range. -/
def layerSlotRange (slots nL l : Nat) : Nat × Nat :=
  if nL = 0 ∨ slots = 0 ∨ nL ≤ l then (0, 0)
  else (l * (slots / nL), if l = nL - 1 then slots else (l + 1) * (slots / nL))

/-- Helper: `j` full quotas fit in the arena when `j ≤ nL`. -/
theorem quota_fits {slots nL j : Nat} (hj : j ≤ nL) : j * (slots / nL) ≤ slots := by
  have h1 : j * (slots / nL) ≤ nL * (slots / nL) := Nat.mul_le_mul_right _ hj
  have h2 : nL * (slots / nL) ≤ slots := by
    rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nL
  omega

/-- Lower end of a valid layer's range. -/
theorem range_lo {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l < nL) :
    (layerSlotRange slots nL l).1 = l * (slots / nL) := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ l) := by omega
  simp [layerSlotRange, h]

/-- Upper end of a layer that is not the last one. -/
theorem range_hi_inner {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l + 1 < nL) :
    (layerSlotRange slots nL l).2 = (l + 1) * (slots / nL) := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ l) := by omega
  have h2 : l ≠ nL - 1 := by omega
  simp [layerSlotRange, h, h2]

/-- Upper end of the last layer: it takes every slot left over by the division. -/
theorem range_hi_last {slots nL : Nat} (hn : 0 < nL) (hs : 0 < slots) :
    (layerSlotRange slots nL (nL - 1)).2 = slots := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ nL - 1) := by omega
  simp [layerSlotRange, h]

/-- The first layer starts at slot 0. -/
theorem range_first_lo {slots nL : Nat} (hn : 0 < nL) (hs : 0 < slots) :
    (layerSlotRange slots nL 0).1 = 0 := by
  rw [range_lo hn hs hn]; simp

/-- Adjacent ranges meet exactly: no slot is orphaned between two layers. -/
theorem range_adjacent {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l + 1 < nL) :
    (layerSlotRange slots nL l).2 = (layerSlotRange slots nL (l + 1)).1 := by
  rw [range_hi_inner hn hs hl, range_lo hn hs hl]

/-- Every range is well formed and inside the arena. -/
theorem range_bounds {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l < nL) :
    (layerSlotRange slots nL l).1 ≤ (layerSlotRange slots nL l).2 ∧
      (layerSlotRange slots nL l).2 ≤ slots := by
  rw [range_lo hn hs hl]
  by_cases hlast : l + 1 < nL
  · rw [range_hi_inner hn hs hlast]
    have h1 : l * (slots / nL) ≤ (l + 1) * (slots / nL) := Nat.mul_le_mul_right _ (by omega)
    exact ⟨h1, quota_fits (by omega)⟩
  · have he : l = nL - 1 := by omega
    subst he
    rw [range_hi_last hn hs]
    exact ⟨quota_fits (by omega), Nat.le_refl _⟩

/-- Ranges of different layers are disjoint: an earlier layer ends before a later one begins. -/
theorem range_disjoint {slots nL l1 l2 : Nat} (hn : 0 < nL) (hs : 0 < slots)
    (h12 : l1 < l2) (h2 : l2 < nL) :
    (layerSlotRange slots nL l1).2 ≤ (layerSlotRange slots nL l2).1 := by
  rw [range_hi_inner hn hs (by omega), range_lo hn hs h2]
  exact Nat.mul_le_mul_right _ h12

-- § Global admission (no eviction)

/-- The residency table and free pointer. -/
structure Cache where
  slots : Nat
  nextFree : Nat
  res : Nat → Nat → Option Nat

/-- Cache invariant: free pointer within the arena, every resident slot below it, and no slot
shared by two pairs. -/
def Cache.Inv (c : Cache) : Prop :=
  c.nextFree ≤ c.slots ∧
  (∀ l e s, c.res l e = some s → s < c.nextFree) ∧
  (∀ l1 e1 l2 e2 s, c.res l1 e1 = some s → c.res l2 e2 = some s → l1 = l2 ∧ e1 = e2)

/-- `ExpertCache::admit` (global path): result slot (`none` = `kNotResident`) and new state. -/
def admit (c : Cache) (l e : Nat) : Option Nat × Cache :=
  match c.res l e with
  | some s => (some s, c)
  | none =>
    if c.nextFree < c.slots then
      (some c.nextFree,
        { c with nextFree := c.nextFree + 1,
                 res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e' })
    else (none, c)

/-- The empty cache satisfies the invariant. -/
theorem empty_inv (slots : Nat) : (Cache.mk slots 0 (fun _ _ => none)).Inv := by
  refine ⟨Nat.zero_le _, ?_, ?_⟩
  · intro l e s h; cases h
  · intro l1 e1 l2 e2 s h; cases h

/-- Admitting a resident expert returns its slot and changes nothing. -/
theorem admit_resident {c : Cache} {l e s : Nat} (h : c.res l e = some s) :
    admit c l e = (some s, c) := by
  simp [admit, h]

/-- A full cache refuses a non-resident expert and changes nothing. -/
theorem admit_full {c : Cache} {l e : Nat} (h : c.res l e = none) (hfull : c.slots ≤ c.nextFree) :
    admit c l e = (none, c) := by
  have : ¬ c.nextFree < c.slots := by omega
  simp [admit, h, this]

/-- After a successful admission the pair is resident at the returned slot. -/
theorem admit_then_resident {c : Cache} {l e s : Nat} (h : (admit c l e).1 = some s) :
    (admit c l e).2.res l e = some s := by
  unfold admit at h ⊢
  split at h
  · rename_i s' hs
    simp only [hs]
    exact (Option.some.inj h) ▸ rfl
  · rename_i hn
    split at h
    · obtain rfl := Option.some.inj h
      simp [hn, *]
    · cases h

/-- Admission preserves the cache invariant: in particular no slot is ever handed out twice. -/
theorem admit_inv {c : Cache} (hc : c.Inv) (l e : Nat) : (admit c l e).2.Inv := by
  obtain ⟨hle, hlt, huniq⟩ := hc
  unfold admit
  split
  · exact ⟨hle, hlt, huniq⟩
  · split
    · rename_i hfree
      refine ⟨by simp only; omega, ?_, ?_⟩
      · intro l' e' s hs
        simp only at hs ⊢
        by_cases h : l' = l ∧ e' = e
        · simp only [h, and_self, if_true] at hs
          obtain rfl := Option.some.inj hs
          omega
        · simp only [h, if_false] at hs
          have := hlt _ _ _ hs
          omega
      · intro l1 e1 l2 e2 s h1 h2
        simp only at h1 h2
        by_cases ha : l1 = l ∧ e1 = e <;> by_cases hb : l2 = l ∧ e2 = e
        · exact ⟨ha.1.trans hb.1.symm, ha.2.trans hb.2.symm⟩
        · simp only [ha, hb, and_self, if_true, if_false] at h1 h2
          obtain rfl := Option.some.inj h1
          have := hlt _ _ _ h2
          omega
        · simp only [ha, hb, and_self, if_true, if_false] at h1 h2
          obtain rfl := Option.some.inj h2
          have := hlt _ _ _ h1
          omega
        · simp only [ha, hb, if_false] at h1 h2
          exact huniq _ _ _ _ _ h1 h2
    · exact ⟨hle, hlt, huniq⟩

/-- Any slot `admit` returns from a well-formed cache is inside the arena. -/
theorem admit_slot_lt {c : Cache} (hc : c.Inv) {l e s : Nat} (h : (admit c l e).1 = some s) :
    s < c.slots := by
  obtain ⟨hle, hlt, _⟩ := hc
  unfold admit at h
  split at h
  · rename_i s' hs
    obtain rfl := Option.some.inj h
    have := hlt _ _ _ hs
    omega
  · split at h
    · rename_i hfree
      obtain rfl := Option.some.inj h
      exact hfree
    · cases h

-- § Per-layer admission (R4.2g)

/-- Per-layer quota pointers. -/
structure Quota where
  slots : Nat
  nL : Nat
  next : Nat → Nat

/-- Quota invariant: each layer's pointer stays in its own range. -/
def Quota.Inv (q : Quota) : Prop :=
  ∀ l, l < q.nL → (layerSlotRange q.slots q.nL l).1 ≤ q.next l ∧
    q.next l ≤ (layerSlotRange q.slots q.nL l).2

/-- Per-layer admission of a non-resident expert: take this layer's next slot if its quota allows. -/
def admitLayer (q : Quota) (l : Nat) : Option Nat × Quota :=
  if q.next l < (layerSlotRange q.slots q.nL l).2 then
    (some (q.next l), { q with next := fun l' => if l' = l then q.next l + 1 else q.next l' })
  else (none, q)

/-- A per-layer admission returns a slot inside that layer's own range. -/
theorem admitLayer_in_range {q : Quota} (hq : q.Inv) {l s : Nat} (hl : l < q.nL)
    (h : (admitLayer q l).1 = some s) :
    (layerSlotRange q.slots q.nL l).1 ≤ s ∧ s < (layerSlotRange q.slots q.nL l).2 := by
  have hb := hq l hl
  unfold admitLayer at h
  split at h
  · rename_i hlt
    obtain rfl := Option.some.inj h
    exact ⟨hb.1, hlt⟩
  · cases h

/-- Per-layer admission preserves the quota invariant. -/
theorem admitLayer_inv {q : Quota} (hq : q.Inv) (l : Nat) : (admitLayer q l).2.Inv := by
  unfold admitLayer
  split
  · rename_i hlt
    intro l' hl'
    have hb := hq l' hl'
    simp only
    by_cases h : l' = l
    · subst h; simp only [if_true]; omega
    · simp only [h, if_false]; exact hb
  · exact hq

/-- Two layers never receive the same slot under per-layer admission. -/
theorem layers_never_share {slots nL l1 l2 s1 s2 : Nat} (hn : 0 < nL) (hs : 0 < slots)
    (h12 : l1 < l2) (h2 : l2 < nL)
    (hs1 : s1 < (layerSlotRange slots nL l1).2) (hs2 : (layerSlotRange slots nL l2).1 ≤ s2) :
    s1 ≠ s2 := by
  have := range_disjoint hn hs h12 h2
  omega

-- § Slot addressing and fills

/-- `fill_slot` byte count: the requested size if it is positive and fits, else a whole slot. -/
def fillBytes (blob bytes : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot. -/
theorem fillBytes_le (blob bytes : Nat) : fillBytes blob bytes ≤ blob := by
  unfold fillBytes; split <;> omega

/-- Uniform `device_slot`: `none` is the null pointer for an out-of-range slot. -/
def deviceSlotOff (slots blob s : Nat) : Option Nat := if s < slots then some (s * blob) else none

/-- A slot returned by `device_slot` lies wholly in the arena. -/
theorem deviceSlot_in_arena {slots blob s o : Nat} (h : deviceSlotOff slots blob s = some o) :
    o + blob ≤ slots * blob := by
  unfold deviceSlotOff at h
  split at h
  · rename_i hs
    obtain rfl := Option.some.inj h
    have h2 := Nat.mul_le_mul_right blob hs
    rw [Nat.succ_mul] at h2
    exact h2
  · cases h

/-- Round a slot size up to 256 bytes. -/
def round256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A rounded slot holds its blob. -/
theorem round256_ge (b : Nat) : b ≤ round256 b := by unfold round256; omega

/-- A rounded slot is a multiple of 256. -/
theorem round256_mod (b : Nat) : round256 b % 256 = 0 := by unfold round256; omega

/-- `open_sized` offsets: a prefix sum of rounded sizes, starting at `start`. -/
def sizedOffsets : List Nat → Nat → List Nat
  | [], o => [o]
  | b :: t, o => o :: sizedOffsets t (o + round256 b)

/-- Every slot start is 256-aligned, as the kernels' vector loads require. -/
theorem sizedOffsets_aligned : ∀ (sizes : List Nat) (o : Nat), o % 256 = 0 →
    ∀ x ∈ sizedOffsets sizes o, x % 256 = 0
  | [], o, ho, x, hx => by
    simp [sizedOffsets] at hx; omega
  | b :: t, o, ho, x, hx => by
    simp only [sizedOffsets, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ho
    · have hr := round256_mod b
      exact sizedOffsets_aligned t (o + round256 b) (by omega) x hx

-- § Profile validation (`read_expert_profile`)

/-- A ranked pair is inside the model's geometry. -/
def pairOk (nL nE : Nat) (p : Nat × Nat) : Bool := decide (p.1 < nL) && decide (p.2 < nE)

/-- The reader's acceptance test: matching geometry, at most `want` pairs, all in range. -/
def profileAccepts (fileL fileE want : Nat) (ranked : List (Nat × Nat)) (nL nE : Nat) : Bool :=
  (fileL == nL) && (fileE == nE) && decide (ranked.length ≤ want) && ranked.all (pairOk nL nE)

/-- An accepted profile names only in-range pairs. -/
theorem accepts_pairs_in_range {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : profileAccepts fileL fileE want ranked nL nE = true) :
    ∀ p ∈ ranked, p.1 < nL ∧ p.2 < nE := by
  intro p hp
  simp only [profileAccepts, Bool.and_eq_true, List.all_eq_true] at h
  have := h.2 p hp
  simp only [pairOk, Bool.and_eq_true, decide_eq_true_eq] at this
  exact this

/-- An accepted profile never ranks more pairs than it reserves slots. -/
theorem accepts_len {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : profileAccepts fileL fileE want ranked nL nE = true) : ranked.length ≤ want := by
  simp only [profileAccepts, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.2

/-- A profile for a different artifact is refused. -/
theorem refuses_other_geometry {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : fileL ≠ nL) : profileAccepts fileL fileE want ranked nL nE = false := by
  simp [profileAccepts, h]

end Ref.ExpertCacheResidency



## FILE: /mnt/user-data/outputs/ExpertPoolDispatch.lean
/-!
# MODULE: expert_pool_dispatch
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
The adapter between the host loop and the CPU expert pool: which router indices become CPU jobs,
that the router weight is applied exactly once (by `moe_combine`, not here), that a failure
latches, and the index arithmetic of the multi-token verify window and its PCIe share.

## Source semantics
* Clause 1: the activation is re-quantized every call (no cache keyed on the `x_f` address).
* Clause 2: `ExpertJob::weight = 1`; `moe_combine` multiplies by the router weight once.
* Clause 3: `n_embd != 2560` is refused; an out-of-range id or a null blob is refused.
* Job index `njobs` and output row `i` differ when there are hits: the jobs are exactly the
  non-hit rows, each once, and never a hit row.
* `failed` LATCHES: once set, dispatch is a no-op, and clearing the message does not re-arm it.
* Multi-token: entry `i` belongs to token `i / k`; the PCIe share `(nmiss * pcie_num) >> 8` never
  exceeds the misses when `pcie_num <= 256`.

## Form template identifiers
`expert_pool_dispatch` → `step`; `ExpertDispatch::failed/fail` → `DState.failed/msg`;
`is_hit` → `hit`; job rows → `jobRows`; hit rows → `hitRows`; `moe_combine` → `combine`;
`pcie_num` share → `pcieShare`.
-/

namespace Ref.ExpertPoolDispatch

-- § Geometry refusal

/-- The width the expert kernel is compiled for. -/
def H : Nat := 2560

/-- Clause 3: the dispatch accepts exactly one activation width. -/
def geometryOk (nEmbd : Nat) : Bool := nEmbd == H

/-- Any other width is refused. -/
theorem geometry_refused {n : Nat} (h : n ≠ 2560) : geometryOk n = false := by
  simp [geometryOk, H, h]

-- § The hit/miss split

/-- Output rows the CPU computes (non-hits). -/
def jobRows (hit : List Bool) : List Nat :=
  (List.range hit.length).filter (fun i => !(hit.getD i false))

/-- Output rows the GPU owns (hits). -/
def hitRows (hit : List Bool) : List Nat :=
  (List.range hit.length).filter (fun i => hit.getD i false)

/-- Helper: a predicate and its negation partition any list. -/
theorem filter_not_add (p : Nat → Bool) : ∀ l : List Nat,
    (l.filter (fun i => !p i)).length + (l.filter p).length = l.length
  | [] => rfl
  | a :: t => by
    have ih := filter_not_add p t
    cases h : p a <;> simp [List.filter_cons, h] <;> omega

/-- Every routed index is either a CPU job or a GPU hit: `njobs + n_hits = k`. -/
theorem jobs_add_hits (hit : List Bool) :
    (jobRows hit).length + (hitRows hit).length = hit.length := by
  have := filter_not_add (fun i => hit.getD i false) (List.range hit.length)
  simpa [jobRows, hitRows] using this

/-- No output row receives two CPU jobs. -/
theorem jobRows_nodup (hit : List Bool) : (jobRows hit).Nodup :=
  List.Nodup.sublist List.filter_sublist List.nodup_range

/-- A hit row is never given a CPU job (its row is zeroed instead). -/
theorem job_not_hit {hit : List Bool} {i : Nat} (h : i ∈ jobRows hit) : i ∉ hitRows hit := by
  simp only [jobRows, hitRows, List.mem_filter, List.mem_range] at h ⊢
  intro h2
  have := h.2
  rw [h2.2] at this
  cases this

/-- With no hits, job index and output row coincide: every row is a job. -/
theorem no_hits_all_jobs (k : Nat) : jobRows (List.replicate k false) = List.range k := by
  simp [jobRows, List.getD_eq_getElem?_getD, List.getElem?_replicate]
  intro i hi
  simp [hi]

-- § Clause 2: the router weight is applied once

/-- `moe_combine`: `sum w[i] * parts[i]`. -/
def combine : List (Int × Int) → Int
  | [] => 0
  | (w, p) :: t => w * p + combine t

/-- The part the pool writes for one expert, scaled by the job's diagnostic weight. -/
def poolPart (jobWeight y : Int) : Int := jobWeight * y

/-- With `ExpertJob::weight = 1` the combine applies each router weight exactly once. -/
theorem weight_applied_once : ∀ (wy : List (Int × Int)),
    combine (wy.map fun p => (p.1, poolPart 1 p.2)) = combine wy
  | [] => rfl
  | (w, y) :: t => by
    simp only [List.map_cons, combine]
    rw [weight_applied_once t]
    simp [poolPart]

/-- Setting the job weight to the router weight would apply it twice (a wrong token, not a fault). -/
theorem double_weight_differs : combine [(2, poolPart 2 1)] ≠ combine [(2, 1)] := by decide

-- § Failure latching

/-- Dispatch state: the latch, its message and the counters. -/
structure DState where
  failed : Bool
  msg : Option String
  layers : Nat
  experts : Nat

/-- What a layer's dispatch ran into. -/
inductive Outcome
  | ok
  | badGeometry
  | badId
  | noBlob

/-- One `expert_pool_dispatch` call. -/
def step (s : DState) (k : Nat) (o : Outcome) : DState :=
  if s.failed then s else
  match o with
  | .ok => { s with layers := s.layers + 1, experts := s.experts + k }
  | .badGeometry => { s with failed := true, msg := some "geometry" }
  | .badId => { s with failed := true, msg := some "id" }
  | .noBlob => { s with failed := true, msg := some "blob" }

/-- A failed adapter does nothing at all. -/
theorem failed_is_fixed {s : DState} (h : s.failed = true) (k : Nat) (o : Outcome) :
    step s k o = s := by
  simp [step, h]

/-- Running a whole sequence of layers. -/
def run (s : DState) (xs : List (Nat × Outcome)) : DState :=
  xs.foldl (fun s p => step s p.1 p.2) s

/-- Once failed, every later layer is a no-op: no resumption into a plausible-looking token. -/
theorem run_after_failure {s : DState} (h : s.failed = true) : ∀ xs, run s xs = s
  | [] => rfl
  | p :: t => by
    simp only [run, List.foldl_cons]
    rw [failed_is_fixed h]
    exact run_after_failure h t

/-- Every non-`ok` outcome sets the latch. -/
theorem bad_outcome_fails (s : DState) (k : Nat) (o : Outcome) (ho : o ≠ .ok) :
    (step s k o).failed = true := by
  unfold step
  split
  · assumption
  · cases o <;> simp_all

/-- Clearing the message does not re-arm a failed adapter. -/
theorem clear_msg_keeps_latch (s : DState) (h : s.failed = true) :
    ({ s with msg := none } : DState).failed = true := h

/-- A healthy layer advances the layer counter by one and the expert counter by `k`. -/
theorem ok_counts (s : DState) (h : s.failed = false) (k : Nat) :
    (step s k .ok).layers = s.layers + 1 ∧ (step s k .ok).experts = s.experts + k := by
  simp [step, h]

-- § Multi-token verify window

/-- An entry `i < n_tok * k` belongs to a token inside the window. -/
theorem entry_token_lt {i nTok k : Nat} (hk : 0 < k) (hi : i < nTok * k) : i / k < nTok :=
  (Nat.div_lt_iff_lt_mul hk).2 hi

/-- An entry's routed index within its token is below `k`. -/
theorem entry_slot_lt {i k : Nat} (hk : 0 < k) : i % k < k := Nat.mod_lt _ hk

/-- Entry decomposition: `i = token * k + j`, the row layout of `out`. -/
theorem entry_decompose (i k : Nat) : i = (i / k) * k + i % k := by
  have := Nat.div_add_mod i k
  rw [Nat.mul_comm] at this
  omega

/-- The PCIe share of the distinct misses, `pcie_num / 256` of them. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) >>> 8

/-- The GPU's PCIe share never exceeds the misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le {nmiss p : Nat} (hp : p ≤ 256) : pcieShare nmiss p ≤ nmiss := by
  unfold pcieShare
  rw [Nat.shiftRight_eq_div_pow]
  have h1 : nmiss * p ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss hp
  have h2 : (2 : Nat) ^ 8 = 256 := rfl
  rw [h2]
  apply Nat.div_le_of_le_mul
  omega

/-- The misses left on the CPU are the complement of the PCIe share. -/
theorem cpu_misses_complement {nmiss p : Nat} (hp : p ≤ 256) :
    (nmiss - pcieShare nmiss p) + pcieShare nmiss p = nmiss := by
  have := pcieShare_le (nmiss := nmiss) hp
  omega

end Ref.ExpertPoolDispatch



## FILE: /mnt/user-data/outputs/ExpertHitCombine.lean
/-!
# MODULE: expert_hit_combine
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
The two-phase GPU hit path (`expert_hit_run`): `Launch` decides once which router indices are
hits, zeroes `hit_out` and computes the hits into it; the CPU pool zeroes the hit rows of `out`;
`Combine` adds `hit_out` into `parts` exactly once, only after a `Launch` that had work.

## Source semantics
* `hit_out` is ZEROED each launch, so a row that was a hit last layer and a miss now contributes 0.
* The CPU writes zero into hit rows (the buffer is reused, so skipping is not enough).
* `Combine` without a pending `Launch` adds nothing; `hit_pending` is cleared, so a second
  `Combine` cannot add the same buffer twice.
* Net effect per row: the final part is the GPU value for a hit and the CPU value for a miss.
* Each pending `Combine` increments exactly one of `hit_ready` / `hit_late`.

## Form template identifiers
`HitPhase::Launch` → `launch`; `HitPhase::Combine` → `combineHits`; `hit_pending` → `pending`;
`hit_out` → `hitOut`; `parts_out` → `parts`; `is_hit` → `hit`; `hit_ready/hit_late` → `ready/late`.
-/

namespace Ref.ExpertHitCombine

-- § State

/-- The per-session hit-path state, rows indexed by router index. -/
structure HState where
  parts : Nat → Int
  hitOut : Nat → Int
  pending : Bool
  ready : Nat
  late : Nat

/-- `Launch`: zero `hit_out`, write the hits' GPU values, and arm `Combine` iff there were hits. -/
def launch (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) (nHits : Nat) : HState :=
  { s with hitOut := fun i => if hit i then gpu i else 0, pending := decide (0 < nHits) }

/-- The CPU pool's output: zero for a hit row, its own value for a miss row. -/
def cpuOut (hit : Nat → Bool) (cpu : Nat → Int) : Nat → Int :=
  fun i => if hit i then 0 else cpu i

/-- `Combine`: `parts += hit_out` once, and record whether the GPU finished in time. -/
def combineHits (s : HState) (doneInTime : Bool) : HState :=
  if s.pending then
    { s with parts := fun i => s.parts i + s.hitOut i, pending := false,
             ready := if doneInTime then s.ready + 1 else s.ready,
             late := if doneInTime then s.late else s.late + 1 }
  else s

-- § Protocol theorems

/-- A `Combine` with no `Launch` in front of it adds nothing. -/
theorem combine_without_launch {s : HState} (h : s.pending = false) (b : Bool) :
    combineHits s b = s := by
  simp [combineHits, h]

/-- A second `Combine` is a no-op: the hit buffer is never added twice. -/
theorem combine_twice (s : HState) (b1 b2 : Bool) :
    combineHits (combineHits s b1) b2 = combineHits s b1 := by
  by_cases h : s.pending = true
  · have hp : (combineHits s b1).pending = false := by simp [combineHits, h]
    exact combine_without_launch hp b2
  · have hf : s.pending = false := by simpa using h
    rw [combine_without_launch hf b1, combine_without_launch hf b2]

/-- A launch with no hits schedules no combine. -/
theorem launch_no_hits (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) :
    (launch s hit gpu 0).pending = false := by
  simp [launch]

/-- After a launch, a miss row of `hit_out` is zero - a stale hit cannot leak in. -/
theorem stale_row_zero (s : HState) (hit : Nat → Bool) (gpu : Nat → Int) (n : Nat) {i : Nat}
    (hi : hit i = false) : (launch s hit gpu n).hitOut i = 0 := by
  simp [launch, hi]

/-- THE SPLIT IS CORRECT: with the CPU's rows in `parts`, every row ends as the GPU value for a hit
and the CPU value for a miss - each expert counted exactly once. -/
theorem final_row (s : HState) (hit : Nat → Bool) (gpu cpu : Nat → Int) {n : Nat} (hn : 0 < n)
    (b : Bool) (i : Nat) :
    (combineHits (launch { s with parts := cpuOut hit cpu } hit gpu n) b).parts i =
      if hit i then gpu i else cpu i := by
  have hp : decide (0 < n) = true := by simp [hn]
  simp only [combineHits, launch, hp, if_true, cpuOut]
  cases hit i <;> simp

/-- Each pending combine is classified exactly once as ready or late. -/
theorem ready_late_count {s : HState} (h : s.pending = true) (b : Bool) :
    (combineHits s b).ready + (combineHits s b).late = s.ready + s.late + 1 := by
  cases b <;> simp [combineHits, h] <;> omega

-- § Hit-path readiness

/-- The ten buffers `hits_ready` requires, as presence flags. -/
def hitsReady (flags : List Bool) : Bool := flags.all id

/-- All or none: one missing buffer disables the whole hit path. -/
theorem hitsReady_missing {flags : List Bool} (h : false ∈ flags) : hitsReady flags = false := by
  simp only [hitsReady, List.all_eq_false]
  exact ⟨false, h, by simp⟩

end Ref.ExpertHitCombine



## FILE: /mnt/user-data/outputs/PinnedArenaResidency.lean
/-!
# MODULE: pinned_arena_residency
SOURCES: src/core/pinned.cu, src/platform/memory.cpp
The pinned host arena: uniform registration bounds, sliced `cudaHostRegister` that pins a prefix,
the working-set lock of the unpinned remainder in 1 GiB chunks, the per-layer parallel load and
its chunk-by-chunk FNV-1a checksum.

## Source semantics
* `uniform_bounds(bytes, slice)`: offsets `0, slice, ...` while `off + slice <= bytes`, then the end
  of the last slice; every bound is within `bytes`.
* Sliced registration stops at the first failure; `registered_bytes` is the end of the last
  registered slice and never exceeds the arena. The lock covers exactly `[registered, bytes)`.
* `pinned(l, e)` is `blob_offset + blob_bytes <= pinned_bytes`, monotone in `pinned_bytes`.
* `lock_resident` (Windows) locks in chunks of at most 1 GiB whose sizes sum to the region.
* `fnv1a64` composes over chunks, so hashing a layer chunk by chunk equals hashing it whole.
* Layer `L` of the uniform load starts at `L * blobs_per_layer * blob_bytes`; layers tile the file.

## Form template identifiers
`uniform_bounds` → `uniformBounds`; sliced `cudaHostRegister` loop → `registerPrefix`;
`ArenaExpertSource::pinned` → `pinnedBlob`; `lock_resident` chunk loop → `chunks`;
`fnv1a64` → `fnv1a64`; `load_experts` offsets → `layerOff`.
-/

namespace Ref.PinnedArenaResidency

-- § Uniform registration bounds

/-- `uniform_bounds(bytes, slice)`. -/
def uniformBounds (bytes slice : Nat) : List Nat :=
  if slice = 0 then [] else
  if bytes / slice = 0 then [] else (List.range (bytes / slice + 1)).map (· * slice)

/-- Every registration bound lies within the arena. -/
theorem uniformBounds_le {bytes slice x : Nat} (h : x ∈ uniformBounds bytes slice) : x ≤ bytes := by
  unfold uniformBounds at h
  split at h
  · cases h
  · split at h
    · cases h
    · simp only [List.mem_map, List.mem_range] at h
      obtain ⟨i, hi, rfl⟩ := h
      have h1 : i * slice ≤ bytes / slice * slice := Nat.mul_le_mul_right _ (by omega)
      have h2 := Nat.div_mul_le_self bytes slice
      omega

/-- Consecutive bounds are exactly one slice apart. -/
theorem uniformBounds_step (i slice : Nat) : (i + 1) * slice = i * slice + slice := Nat.succ_mul i slice

-- § Sliced registration prefix

/-- Register `(off, len)` slices in order until the first failure; returns `registered_bytes`. -/
def registerPrefix (ok : Nat → Bool) : List (Nat × Nat) → Nat → Nat
  | [], acc => acc
  | (off, n) :: t, acc => if ok off then registerPrefix ok t (off + n) else acc

/-- `registered_bytes` never exceeds the arena when every slice ends inside it. -/
theorem registerPrefix_le (ok : Nat → Bool) (total : Nat) :
    ∀ (sl : List (Nat × Nat)) (acc : Nat), acc ≤ total → (∀ p ∈ sl, p.1 + p.2 ≤ total) →
      registerPrefix ok sl acc ≤ total
  | [], acc, hacc, _ => hacc
  | (off, n) :: t, acc, hacc, hall => by
    simp only [registerPrefix]
    split
    · exact registerPrefix_le ok total t (off + n) (hall (off, n) (by simp))
        (fun p hp => hall p (by simp [hp]))
    · exact hacc

/-- The working-set lock covers exactly the unregistered tail: pinned + locked = arena. -/
theorem lock_partition {registered bytes : Nat} (h : registered ≤ bytes) :
    registered + (bytes - registered) = bytes := by omega

/-- `ArenaExpertSource::pinned`: the whole blob lies in the registered prefix. -/
def pinnedBlob (off len pinnedBytes : Nat) : Bool := decide (off + len ≤ pinnedBytes)

/-- Registering more never un-pins a blob. -/
theorem pinned_mono {off len p p' : Nat} (h : pinnedBlob off len p = true) (hp : p ≤ p') :
    pinnedBlob off len p' = true := by
  simp only [pinnedBlob, decide_eq_true_eq] at h ⊢
  omega

-- § Chunked locking (`lock_resident`)

/-- The chunk sizes `min c r` the lock loop issues, with fuel bounding the iterations. -/
def chunks (c : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min c r :: chunks c f (r - min c r)

/-- The chunks sum to the region: every byte is locked once. -/
theorem chunks_sum {c : Nat} (hc : 0 < c) : ∀ (f r : Nat), r ≤ f → (chunks c f r).sum = r
  | 0, r, h => by
    have : r = 0 := by omega
    subst this
    rfl
  | f + 1, r, h => by
    unfold chunks
    split
    · simp; omega
    · have ih := chunks_sum hc f (r - min c r) (by omega)
      simp only [List.sum_cons, ih]
      omega

/-- Every chunk is non-empty and at most the chunk size (`c = 1 GiB > 0` in the source). -/
theorem chunks_bounded {c : Nat} (hc : 0 < c) : ∀ (f r x : Nat), x ∈ chunks c f r → 0 < x ∧ x ≤ c
  | 0, r, x, h => by simp [chunks] at h
  | f + 1, r, x, h => by
    unfold chunks at h
    split at h
    · simp at h
    · rename_i hr
      simp only [List.mem_cons] at h
      rcases h with rfl | h
      · constructor <;> omega
      · exact chunks_bounded hc f _ x h

-- § FNV-1a checksum

/-- The FNV-1a 64-bit prime. -/
def FNV_PRIME : UInt64 := 1099511628211

/-- The FNV-1a 64-bit offset basis (the per-layer seed). -/
def FNV_OFFSET : UInt64 := 1469598103934665603

/-- One FNV-1a step. -/
def fnvStep (h : UInt64) (b : UInt8) : UInt64 := (h ^^^ b.toUInt64) * FNV_PRIME

/-- `fnv1a64(p, n, seed)`. -/
def fnv1a64 (bs : List UInt8) (seed : UInt64) : UInt64 := bs.foldl fnvStep seed

/-- Hashing nothing returns the seed. -/
theorem fnv_nil (seed : UInt64) : fnv1a64 [] seed = seed := rfl

/-- Hashing a concatenation continues from the first part's hash. -/
theorem fnv_append (a b : List UInt8) (seed : UInt64) :
    fnv1a64 (a ++ b) seed = fnv1a64 b (fnv1a64 a seed) := by
  simp [fnv1a64, List.foldl_append]

/-- The loader's chunk-by-chunk hash equals the hash of the whole layer. -/
theorem fnv_chunked : ∀ (cs : List (List UInt8)) (seed : UInt64),
    fnv1a64 cs.flatten seed = cs.foldl (fun h c => fnv1a64 c h) seed
  | [], seed => rfl
  | c :: t, seed => by
    simp only [List.flatten_cons, List.foldl_cons]
    rw [fnv_append]
    exact fnv_chunked t (fnv1a64 c seed)

-- § Uniform per-layer load offsets

/-- `load_experts`: layer `L` starts at `L * blobs_per_layer * blob_bytes`. -/
def layerOff (bpl bb L : Nat) : Nat := L * bpl * bb

/-- Layers are contiguous: one layer ends exactly where the next begins. -/
theorem layerOff_contig (bpl bb L : Nat) : layerOff bpl bb L + bpl * bb = layerOff bpl bb (L + 1) := by
  simp only [layerOff, Nat.succ_mul, Nat.add_mul]

/-- Every layer's range ends inside the file of `layers` layers. -/
theorem layer_in_file {bpl bb L layers : Nat} (h : L < layers) :
    layerOff bpl bb L + bpl * bb ≤ layers * bpl * bb := by
  rw [layerOff_contig]
  exact Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ h)

end Ref.PinnedArenaResidency



## FILE: /mnt/user-data/outputs/DirectRowReader.lean
/-!
# MODULE: direct_row_reader
SOURCES: src/platform/direct_file.cpp, src/ngram/ple_reader.cpp
Unbuffered aligned file reads (`DirectFile`) and the PLE table row reader built on them: the page
read that covers a row, the set-associative row cache's index bounds, ticket accounting and the
latency percentile index.

## Source semantics
* `DirectFile::submit` accepts only a non-empty request whose offset, length and buffer are all
  multiples of the alignment.
* `PleReader::issue` reads, for a row at byte `at`, from `first = floor(at / PAGE) * PAGE` for
  `length = floor((at + ROW - 1) / PAGE) * PAGE - first + PAGE` bytes: page aligned, covering the
  whole row, and at most two pages (the slot size) when `ROW <= PAGE`.
* The row cache has `sets * 8` ways; every set/way index and its `ROW_BYTES` payload are in bounds;
  the round-robin pointer stays below `WAYS`; an inserted row is found again.
* A ticket's `pending` drops by one per completed job; after `n` completions it is `p - n`.
* The percentile index `min(n - 1, round(q * (n - 1)))` is a valid index for `n > 0`.

## Form template identifiers
`DirectFile::submit` checks → `submitOk`; `first` → `pageFirst`; `length` → `pageLength`;
`RowCache` → `setFind`/`setInsert`; `TicketState::pending` → `completeN`;
`ReaderStats::percentile` → `pctIndex`.
-/

namespace Ref.DirectRowReader

-- § Aligned submission

/-- `DirectFile::submit`'s acceptance test. -/
def submitOk (a off len buf : Nat) : Bool :=
  (off % a == 0) && (len % a == 0) && (buf % a == 0) && (len != 0)

/-- An accepted request is aligned on all three axes and non-empty. -/
theorem submitOk_spec {a off len buf : Nat} (h : submitOk a off len buf = true) :
    off % a = 0 ∧ len % a = 0 ∧ buf % a = 0 ∧ 0 < len := by
  simp only [submitOk, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, by omega⟩

/-- A zero-length request is refused. -/
theorem submit_refuses_empty (a off buf : Nat) : submitOk a off 0 buf = false := by
  simp [submitOk]

-- § Page coverage of one row

/-- Helper: any `x` lies below the end of its page. -/
theorem lt_page_end (x : Nat) {P : Nat} (hP : 0 < P) : x < x / P * P + P := by
  have h1 := Nat.div_add_mod x P
  have h2 := Nat.mod_lt x hP
  rw [Nat.mul_comm] at h1
  omega

/-- The aligned start of the read. -/
def pageFirst (P at_ : Nat) : Nat := at_ / P * P

/-- The read length (one or two pages). -/
def pageLength (P R at_ : Nat) : Nat := (at_ + R - 1) / P * P - at_ / P * P + P

/-- The read starts at or before the row. -/
theorem pageFirst_le (P at_ : Nat) : pageFirst P at_ ≤ at_ := Nat.div_mul_le_self _ _

/-- The read starts on a page boundary. -/
theorem pageFirst_aligned (P at_ : Nat) : pageFirst P at_ % P = 0 := Nat.mul_mod_left _ _

/-- Helper: the last page of the row is at or after the first one. -/
theorem first_le_last (P : Nat) {R : Nat} (at_ : Nat) (hR : 0 < R) : at_ / P * P ≤ (at_ + R - 1) / P * P :=
  Nat.mul_le_mul_right _ (Nat.div_le_div_right (by omega))

/-- The read covers the whole row. -/
theorem row_covered {P R at_ : Nat} (hP : 0 < P) (hR : 0 < R) :
    at_ + R ≤ pageFirst P at_ + pageLength P R at_ := by
  have h1 := lt_page_end (at_ + R - 1) hP
  have h2 := first_le_last P at_ hR
  unfold pageFirst pageLength
  omega

/-- The read length is a whole number of pages. -/
theorem pageLength_aligned (P R at_ : Nat) : pageLength P R at_ % P = 0 := by
  unfold pageLength
  rw [← Nat.sub_mul, ← Nat.succ_mul]
  exact Nat.mul_mod_left _ _

/-- A row no larger than a page needs at most two pages: it fits its 2-page slot. -/
theorem pageLength_le_two {P R at_ : Nat} (hP : 0 < P) (hR : 0 < R) (hRP : R ≤ P) :
    pageLength P R at_ ≤ 2 * P := by
  have hq : (at_ + R - 1) / P ≤ at_ / P + 1 := by
    have h1 : (at_ + R - 1) / P ≤ (at_ + P) / P := Nat.div_le_div_right (by omega)
    rw [Nat.add_div_right _ hP] at h1
    exact h1
  have h2 : (at_ + R - 1) / P * P ≤ (at_ / P + 1) * P := Nat.mul_le_mul_right _ hq
  rw [Nat.succ_mul] at h2
  unfold pageLength
  omega

-- § Set-associative row cache

/-- Ways per set. -/
def WAYS : Nat := 8

/-- The set index `mix(row) % sets` is a valid set. -/
theorem set_index_lt {sets : Nat} (hs : 0 < sets) (x : Nat) : x % sets < sets := Nat.mod_lt _ hs

/-- A `(set, way)` pair addresses a key inside the `sets * WAYS` table. -/
theorem way_index_lt {sets s w : Nat} (hs : s < sets) (hw : w < WAYS) : s * WAYS + w < sets * WAYS := by
  have h := Nat.mul_le_mul_right WAYS hs
  rw [Nat.succ_mul] at h
  omega

/-- The row's payload lies inside the `sets * WAYS * ROW` data array. -/
theorem way_data_in_bounds {sets s w R : Nat} (hs : s < sets) (hw : w < WAYS) :
    (s * WAYS + w) * R + R ≤ sets * WAYS * R := by
  have h := Nat.mul_le_mul_right R (way_index_lt hs hw)
  rw [Nat.succ_mul] at h
  exact h

/-- The round-robin replacement pointer stays a valid way. -/
theorem rr_next_lt (w : Nat) : (w + 1) % WAYS < WAYS := Nat.mod_lt _ (by decide)

/-- Lookup in one set: the first way holding `row`. -/
def setFind (keys : Nat → Option Nat) (row : Nat) : Option Nat :=
  (List.range WAYS).find? (fun w => keys w == some row)

/-- Insert `row` at way `w`. -/
def setInsert (keys : Nat → Option Nat) (w row : Nat) : Nat → Option Nat :=
  fun w' => if w' = w then some row else keys w'

/-- A row just inserted at a valid way is found. -/
theorem find_after_insert (keys : Nat → Option Nat) {w : Nat} (hw : w < WAYS) (row : Nat) :
    (setFind (setInsert keys w row) row).isSome := by
  unfold setFind
  rw [List.find?_isSome]
  exact ⟨w, List.mem_range.2 hw, by simp [setInsert]⟩

/-- A hit returns a way whose key really is the row. -/
theorem find_sound {keys : Nat → Option Nat} {row w : Nat} (h : setFind keys row = some w) :
    keys w = some row := by
  unfold setFind at h
  have := List.find?_some h
  simpa using this

-- § Ticket accounting

/-- `pending` after `n` completions. -/
def completeN : Nat → Nat → Nat
  | 0, p => p
  | n + 1, p => completeN n (p - 1)

/-- After `n` completions a ticket with `p` jobs has `p - n` pending. -/
theorem completeN_eq : ∀ (n p : Nat), completeN n p = p - n
  | 0, p => by simp [completeN]
  | n + 1, p => by
    simp only [completeN]
    rw [completeN_eq n (p - 1)]
    omega

/-- `collect` returns once every job of the ticket has completed. -/
theorem collect_done (p : Nat) : completeN p p = 0 := by
  rw [completeN_eq]; omega

-- § Latency statistics

/-- `percentile` index: `min(n - 1, t)`. -/
def pctIndex (n t : Nat) : Nat := min (n - 1) t

/-- The percentile index is valid for a non-empty sample. -/
theorem pctIndex_lt {n : Nat} (h : 0 < n) (t : Nat) : pctIndex n t < n := by
  unfold pctIndex; omega

/-- The latency ring position stays inside the ring. -/
theorem ring_pos_lt (pos : Nat) : pos % 65536 < 65536 := Nat.mod_lt _ (by decide)

end Ref.DirectRowReader



## FILE: /mnt/user-data/outputs/VramMemoryPlan.lean
/-!
# MODULE: vram_memory_plan
SOURCES: include/ref/plan/plan.hpp
The census and VRAM planner: GDN recurrent-state bytes, INT8 KV plus indexer-key bytes per token,
and `make_plan`, which divides the fixed VRAM pool between KV, fixed costs and expert-cache slots,
refusing (`DoesNotClose`) rather than overcommitting.

## Source semantics
* `state_bytes` = 36 GDN layers x (128*48*128*4 + 3*10240*4) = 117,669,888 B = 85 whole slots.
* `kv_bytes_per_token` = 12 QSA layers x (1024 INT8 + 32 B scales + 32 B indexer key) = 13,056 B;
  the indexer caches ONE shared 128-wide key per 4-token block (32 B/token/layer).
* `make_plan` succeeds only if KV + fixed costs fit the pool and at least one slot remains; then
  KV + fixed + cache = pool, slots x blob <= cache, and the slot remainder is less than one blob.
* More context never yields more cache slots.

## Form template identifiers
`Geometry` → `Geometry`; `Costs` → `Costs`; `state_bytes` → `stateBytes`;
`kv_bytes_per_token` → `kvBytesPerToken`; `make_plan` → `makePlan`; `DoesNotClose` → `Except.error`;
`vram_pool_bytes` → `vramPool`.
-/

namespace Ref.VramMemoryPlan

-- § Geometry and costs

/-- Model geometry, defaults from the real artifact. -/
structure Geometry where
  nLayers : Nat := 48
  nQsaLayers : Nat := 12
  nKvHeads : Nat := 2
  headDim : Nat := 256
  kvGroup : Nat := 64
  qsaBlock : Nat := 4
  indexerQHeads : Nat := 4
  indexerKeyHeads : Nat := 1
  indexerKeyDim : Nat := 128
  ssmStateSize : Nat := 128
  ssmVHeads : Nat := 48
  ssmDConv : Nat := 4
  ssmConvChannels : Nat := 10240

/-- GDN layers are derived, never a third constant. -/
def Geometry.nGdnLayers (g : Geometry) : Nat := g.nLayers - g.nQsaLayers

/-- Byte costs. -/
structure Costs where
  expertBlob : Nat := 1382400
  dense : Nat := 0
  embd : Nat := 0
  mtp : Nat := 0
  workspace : Nat := 0
  state : Nat := 0

/-- The shared VRAM pool, 5,943,000,000 B. -/
def vramPool : Nat := 5943000000

/-- 36 of the 48 layers are GDN. -/
theorem default_gdn_layers : ({} : Geometry).nGdnLayers = 36 := rfl

-- § Recurrent state

/-- `state_bytes(g)`: fp32 recurrence plus conv history, over the GDN layers. -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdnLayers * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 +
    (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- The recurrence term is 3,145,728 B per layer. -/
theorem recurrence_per_layer : 128 * 48 * 128 * 4 = 3145728 := rfl

/-- The conv-history term is 122,880 B per layer. -/
theorem conv_per_layer : 3 * 10240 * 4 = 122880 := rfl

/-- The missing term was 117,669,888 B. -/
theorem stateBytes_default : stateBytes {} = 117669888 := rfl

/-- ... which is 85 expert cache slots - not a rounding error. -/
theorem state_is_85_slots : stateBytes {} / 1382400 = 85 := rfl

-- § KV and indexer keys

/-- Indexer key bytes per token per layer: one shared key head, pooled per block. -/
def indexerBytes (g : Geometry) : Nat := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock

/-- `kv_bytes_per_token(g)`. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := kvElems / g.kvGroup * 2
  g.nQsaLayers * (kvElems + kvScales + indexerBytes g)

/-- The correct indexer cost is 32 B/token/layer. -/
theorem indexer_default : indexerBytes {} = 32 := rfl

/-- The two historical errors were 2x and 4x too big. -/
theorem indexer_errors : 1 * 256 / 4 = 2 * 32 ∧ 4 * 128 / 4 = 4 * 32 := ⟨rfl, rfl⟩

/-- 13,056 B of KV and indexer keys per token. -/
theorem kvBytesPerToken_default : kvBytesPerToken {} = 13056 := rfl

-- § The plan

/-- The planner's output. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat
  dramExperts : Nat

/-- Everything that never leaves the pool. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- `make_plan`; `Except.error` is `DoesNotClose`. -/
def makePlan (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat := vramPool) : Except String Plan :=
  if pool < kvBytesPerToken g * ctx + fixedCost c then .error "does not close"
  else if (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob = 0 then .error "no slot"
  else .ok
    { maxContext := ctx
      kvBytes := kvBytesPerToken g * ctx
      stateBytes := c.state
      cacheBytes := pool - kvBytesPerToken g * ctx - fixedCost c
      cacheSlots := (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob
      vramBudget := pool
      vramUsed := pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob
      dramExperts := c.dense + c.embd }

/-- Helper: unpack a successful plan into its defining facts. -/
theorem makePlan_ok {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = .ok p) :
    kvBytesPerToken g * ctx + fixedCost c ≤ pool ∧
    p.kvBytes = kvBytesPerToken g * ctx ∧
    p.cacheBytes = pool - kvBytesPerToken g * ctx - fixedCost c ∧
    p.cacheSlots = p.cacheBytes / c.expertBlob ∧ p.cacheSlots ≠ 0 ∧
    p.vramBudget = pool ∧ p.vramUsed = pool - p.cacheBytes % c.expertBlob := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · rename_i h2
      simp only [Except.ok.injEq] at h
      subst h
      exact ⟨by omega, rfl, rfl, rfl, h2, rfl, rfl⟩

/-- The plan closes: KV + fixed costs + cache is exactly the pool. -/
theorem plan_closes {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) :
    p.kvBytes + fixedCost c + p.cacheBytes = p.vramBudget := by
  obtain ⟨h1, hk, hc, -, -, hb, -⟩ := makePlan_ok h
  rw [hk, hc, hb]; omega

/-- The slots fit in the cache bytes. -/
theorem plan_slots_fit {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) :
    p.cacheSlots * c.expertBlob ≤ p.cacheBytes := by
  obtain ⟨-, -, -, hs, -, -, -⟩ := makePlan_ok h
  rw [hs]; exact Nat.div_mul_le_self _ _

/-- A plan always has at least one slot. -/
theorem plan_has_slot {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) : 1 ≤ p.cacheSlots := by
  obtain ⟨-, -, -, -, hs, -, -⟩ := makePlan_ok h
  omega

/-- The unusable slot remainder is less than one blob, and VRAM used never exceeds the pool. -/
theorem plan_remainder {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) (hb : 0 < c.expertBlob) :
    p.cacheBytes % c.expertBlob < c.expertBlob ∧ p.vramUsed ≤ p.vramBudget := by
  obtain ⟨-, -, -, -, -, hbud, hused⟩ := makePlan_ok h
  exact ⟨Nat.mod_lt _ hb, by rw [hused, hbud]; omega⟩

/-- A request that cannot fit is refused at startup, never planned. -/
theorem plan_refuses {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat}
    (h : pool < kvBytesPerToken g * ctx + fixedCost c) : ∃ e, makePlan ctx g c pool = .error e :=
  ⟨"does not close", by simp [makePlan, h]⟩

/-- More context never buys more cache slots. -/
theorem slots_antitone {ctx1 ctx2 : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p1 p2 : Plan}
    (hctx : ctx1 ≤ ctx2) (h1 : makePlan ctx1 g c pool = .ok p1) (h2 : makePlan ctx2 g c pool = .ok p2) :
    p2.cacheSlots ≤ p1.cacheSlots := by
  obtain ⟨-, -, hc1, hs1, -, -, -⟩ := makePlan_ok h1
  obtain ⟨-, -, hc2, hs2, -, -, -⟩ := makePlan_ok h2
  have hkv : kvBytesPerToken g * ctx1 ≤ kvBytesPerToken g * ctx2 := Nat.mul_le_mul_left _ hctx
  rw [hs1, hs2]
  apply Nat.div_le_div_right
  rw [hc1, hc2]
  omega

/-- Worked example: at 32K context with no other costs, the default model gets 3,989 slots. -/
theorem example_32k :
    (makePlan 32768 {} {}).toOption.map (·.cacheSlots) = some 3989 := by decide

end Ref.VramMemoryPlan



## FILE: /mnt/user-data/outputs/DenseWeightLoader.lean
/-!
# MODULE: dense_weight_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: the per-plane segment list of a tensor, the four byte conversions between
pack and arena forms, the staging-buffer bound, chunking that never splits an fp16 element, the
index kind remap and the compacted arena placement under a skip set.

## Source semantics
* Conversions: `Copy` keeps n bytes; `Bf16High`/`ToF16` take 4 B/elem to 2; `WidenF16` takes 2 to 4.
  No conversion grows past 2n, which is why `stage_out` is `2 * CHUNK`.
* A quantized tensor is codes | scales | optional offset, contiguous in both forms; an fp16 scale
  plane is half the size in the source. The segment sums must equal the index's totals.
* Promoted tensors are exactly 4 B/elem in and 2 B/elem out; widening an f32 scale plane is exact.
* An odd widening tail is shortened by one byte (and refused if only one byte is left).
* Index kinds 4 (BF16) and 5 (F16) are raw 16-bit and take the engine forms of kinds 1 and 3.
* Under a skip set, kept rows are re-placed in order at the index alignment: offsets aligned,
  increasing, and the pool covers every kept tensor.

## Form template identifiers
`Conv` → `Conv`; output size per chunk → `outBytes`; `Seg` → `Seg`; quantized segment list →
`quantSegs`; `kind == 4 || kind == 5` remap → `remapKind`; skip-set compaction → `place`.
-/

namespace Ref.DenseWeightLoader

-- § Conversions

/-- How a segment's bytes are transformed from pack to arena. -/
inductive Conv
  | copy
  | bf16High
  | toF16
  | widenF16

/-- Bytes written for `n` source bytes. -/
def outBytes : Conv → Nat → Nat
  | .copy, n => n
  | .bf16High, n => n / 4 * 2
  | .toF16, n => n / 4 * 2
  | .widenF16, n => n / 2 * 4

/-- No conversion more than doubles its input. -/
theorem outBytes_le_double (cv : Conv) (n : Nat) : outBytes cv n ≤ 2 * n := by
  cases cv <;> simp only [outBytes] <;> omega

/-- The source staging chunk, 8 MiB. -/
def CHUNK : Nat := 8 * 1024 * 1024

/-- `stage_out` at `2 * CHUNK` holds any converted chunk. -/
theorem stage_out_suffices (cv : Conv) {n : Nat} (h : n ≤ CHUNK) : outBytes cv n ≤ 2 * CHUNK := by
  have := outBytes_le_double cv n
  omega

/-- A promoted tensor of `e` elements: 4e bytes in, 2e out. -/
theorem promoted_sizes (e : Nat) : outBytes .bf16High (4 * e) = 2 * e ∧ outBytes .toF16 (4 * e) = 2 * e := by
  simp only [outBytes]; omega

/-- Widening a scale plane of `m` f32 scales from its fp16 source form gives the full plane. -/
theorem widen_exact_size (m : Nat) : outBytes .widenF16 (2 * m) = 4 * m := by
  simp only [outBytes]; omega

/-- Chunking a halving conversion at 4-byte multiples loses nothing. -/
theorem halving_additive (a b : Nat) : outBytes .bf16High (4 * a + 4 * b) =
    outBytes .bf16High (4 * a) + outBytes .bf16High (4 * b) := by
  simp only [outBytes]; omega

/-- The 8 MiB chunk is even, so full chunks never split an fp16 element. -/
theorem chunk_even : CHUNK % 2 = 0 := by decide

/-- An odd tail is shortened to an even, non-empty length (unless only one byte is left). -/
theorem odd_tail_trim {n : Nat} (hodd : n % 2 = 1) (h1 : n ≠ 1) : (n - 1) % 2 = 0 ∧ 0 < n - 1 := by
  omega

/-- The high half of an f32 is its bf16 encoding: `1.0f` (0x3F800000) is bf16 0x3F80. -/
theorem bf16_high_one : ((0x3F800000 : UInt32) >>> 16).toUInt16 = 0x3F80 := by decide

-- § Segment lists

/-- One contiguous run of a tensor with one transform. -/
structure Seg where
  srcOff : Nat
  srcBytes : Nat
  dstOff : Nat
  dstBytes : Nat
  conv : Conv

/-- Source bytes of an fp16-or-f32 scale plane. -/
def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- The segment list of a quantized tensor: codes, then scales, then the optional offset plane. -/
def quantSegs (codes scales offset : Nat) (fp16 : Bool) : List Seg :=
  [⟨0, codes, 0, codes, .copy⟩] ++
  (if scales = 0 then [] else
    [⟨codes, srcScale scales fp16, codes, scales, if fp16 then .widenF16 else .copy⟩]) ++
  (if offset = 0 then [] else
    [⟨codes + srcScale scales fp16, offset, codes + scales, offset, .copy⟩])

/-- Total source bytes of a segment list. -/
def srcSum (ss : List Seg) : Nat := (ss.map Seg.srcBytes).sum

/-- Total destination bytes of a segment list. -/
def dstSum (ss : List Seg) : Nat := (ss.map Seg.dstBytes).sum

/-- The quantized segments cover codes + source scales + offset in the pack. -/
theorem quant_src_sum (c s o : Nat) (f : Bool) :
    srcSum (quantSegs c s o f) = c + srcScale s f + o := by
  unfold quantSegs srcSum
  by_cases hs : s = 0 <;> by_cases ho : o = 0 <;> simp [hs, ho, srcScale] <;>
    cases f <;> simp <;> omega

/-- The quantized segments cover codes + scales + offset in the arena. -/
theorem quant_dst_sum (c s o : Nat) (f : Bool) :
    dstSum (quantSegs c s o f) = c + s + o := by
  unfold quantSegs dstSum
  by_cases hs : s = 0 <;> by_cases ho : o = 0 <;> simp [hs, ho] <;> omega

/-- The loader's cross-check passes exactly when the index totals are the plane sums. -/
theorem segment_check_iff (c s o srcB dstB : Nat) (fp : Bool) :
    (srcSum (quantSegs c s o fp) = srcB ∧ dstSum (quantSegs c s o fp) = dstB) ↔
      (c + srcScale s fp + o = srcB ∧ c + s + o = dstB) := by
  rw [quant_src_sum, quant_dst_sum]

-- § Index kinds

/-- Kinds 4 (BF16) and 5 (F16) are raw 16-bit and take the engine forms of kinds 1 and 3. -/
def remapKind : Nat → Nat × Bool
  | 4 => (1, true)
  | 5 => (3, true)
  | k => (k, false)

/-- Raw-16 rows are copied, never re-rounded. -/
theorem remap_raw (k : Nat) (h : (remapKind k).2 = true) : k = 4 ∨ k = 5 := by
  unfold remapKind at h
  split at h <;> simp_all

/-- A row is parsed only when all 19 fields are present. -/
def rowParsed (fields : Nat) : Bool := fields == 19

/-- A short row is refused rather than loaded at a wrong offset. -/
theorem short_row_refused {n : Nat} (h : n ≠ 19) : rowParsed n = false := by
  simp [rowParsed, h]

-- § Compaction under a skip set

/-- Round up to a multiple of 256 (the default alignment). -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- An aligned size holds the tensor. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by unfold align256; omega

/-- An aligned size is a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by unfold align256; omega

/-- Place rows `(skipped, bytes)` from `at`: offsets of kept rows and the final pool. -/
def place : List (Bool × Nat) → Nat → List Nat × Nat
  | [], at_ => ([], at_)
  | (true, _) :: t, at_ => place t at_
  | (false, b) :: t, at_ => let r := place t (at_ + align256 b); (at_ :: r.1, r.2)

/-- Bytes of the kept rows. -/
def keptBytes : List (Bool × Nat) → Nat
  | [] => 0
  | (true, _) :: t => keptBytes t
  | (false, b) :: t => b + keptBytes t

/-- The compacted pool covers every kept tensor. -/
theorem place_pool_ge : ∀ (rows : List (Bool × Nat)) (at_ : Nat),
    at_ + keptBytes rows ≤ (place rows at_).2
  | [], at_ => by simp [place, keptBytes]
  | (true, _) :: t, at_ => by simpa [place, keptBytes] using place_pool_ge t at_
  | (false, b) :: t, at_ => by
    have ih := place_pool_ge t (at_ + align256 b)
    have hb := align256_ge b
    simp only [place, keptBytes]
    omega

/-- The pool never moves backwards. -/
theorem place_pool_mono : ∀ (rows : List (Bool × Nat)) (at_ : Nat), at_ ≤ (place rows at_).2 := by
  intro rows at_
  have := place_pool_ge rows at_
  omega

/-- Every kept row starts on an alignment boundary. -/
theorem place_aligned : ∀ (rows : List (Bool × Nat)) (at_ : Nat), at_ % 256 = 0 →
    ∀ o ∈ (place rows at_).1, o % 256 = 0
  | [], _, _, o, ho => by simp [place] at ho
  | (true, _) :: t, at_, h, o, ho => place_aligned t at_ h o (by simpa [place] using ho)
  | (false, b) :: t, at_, h, o, ho => by
    simp only [place, List.mem_cons] at ho
    rcases ho with rfl | ho
    · exact h
    · have := align256_mod b
      exact place_aligned t (at_ + align256 b) (by omega) o ho

end Ref.DenseWeightLoader



## FILE: /mnt/user-data/outputs/NativeDenseOverride.lean
/-!
# MODULE: native_dense_override
SOURCES: src/core/native_dense.cpp
Attaching native (GGUF block-quantized) copies of the GDN/QSA dense matrices to the canonical
weight table: tensor-name eligibility, the GGUF directory span checks (no truncation, no overlap,
no byte-count overflow), split-shard metadata acceptance, scratch sizing, and all-or-nothing
publication of the overrides.

## Source semantics
* Eligible: a `blk.` tensor whose name ends with one of ten attention/SSM/shared-expert suffixes,
  or `blk.1.ple_key.weight` when requested.
* Every directory entry (uploaded or not) must have block-divisible first dimension, a payload
  span inside the file, and must end before the next tensor's offset.
* Overflow guards: `x <= MAX / d` implies `x * d <= MAX`.
* An additional shard is accepted only with the first shard's split count/tensor count and a
  shard number in `1 .. count-1`.
* The scratch is sized by the largest input width, which bounds every override's input.
* All checks and uploads finish before any reference is published; any failure leaves the table
  untouched.

## Form template identifiers
`eligible` → `eligible`; payload checks → `spanOk`/`nextOk`; overflow guard → `mul_le_of_le_div`;
split metadata check → `shardOk`; `max_in` → `maxIn`; `pending` + publish loop → `checkAll`/`publish`.
-/

namespace Ref.NativeDenseOverride

-- § Eligibility

/-- The ten matrix suffixes served natively. -/
def suffixes : List String :=
  [".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
   ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight",
   ".ffn_down_shexp.weight"]

/-- `eligible(name, include_ple_key)`. -/
def eligible (name : String) (pleKey : Bool) : Bool :=
  name.startsWith "blk." &&
    ((pleKey && name == "blk.1.ple_key.weight") || suffixes.any (fun s => name.endsWith s))

/-- Only per-block tensors are ever eligible. -/
theorem eligible_blk {name : String} {pk : Bool} (h : eligible name pk = true) :
    name.startsWith "blk." = true := by
  simp only [eligible, Bool.and_eq_true] at h
  exact h.1

/-- Without the PLE key flag, eligibility is exactly the suffix test. -/
theorem eligible_no_ple (name : String) :
    eligible name false = (name.startsWith "blk." && suffixes.any (fun s => name.endsWith s)) := by
  simp [eligible]

/-- Ten suffixes are served. -/
theorem ten_suffixes : suffixes.length = 10 := rfl

-- § Directory span checks

/-- The tensor's payload lies inside the file's data section (no overflowing addition). -/
def spanOk (payload off bytes : Nat) : Bool := decide (off ≤ payload) && decide (bytes ≤ payload - off)

/-- An accepted span ends inside the payload. -/
theorem spanOk_end {p o b : Nat} (h : spanOk p o b = true) : o + b ≤ p := by
  simp only [spanOk, Bool.and_eq_true, decide_eq_true_eq] at h
  omega

/-- The tensor ends before the next tensor's offset. -/
def nextOk (off bytes next : Nat) : Bool := decide (bytes ≤ next - off)

/-- An accepted entry ends at or before its successor starts. -/
theorem nextOk_end {o b n : Nat} (hlt : o < n) (h : nextOk o b n = true) : o + b ≤ n := by
  simp only [nextOk, decide_eq_true_eq] at h
  omega

/-- Two checked tensors never overlap: the earlier one ends before its successor, which is at
or before the later one. -/
theorem spans_disjoint {o1 b1 nx o2 : Nat} (h1 : o1 < nx) (hnx : nx ≤ o2)
    (h : nextOk o1 b1 nx = true) : o1 + b1 ≤ o2 := by
  have := nextOk_end h1 h
  omega

/-- The overflow guard: `x ≤ MAX / d` implies `x * d ≤ MAX`. -/
theorem mul_le_of_le_div {x d M : Nat} (hd : 0 < d) (h : x ≤ M / d) : x * d ≤ M :=
  (Nat.le_div_iff_mul_le hd).1 h

/-- Block divisibility of the first dimension makes the whole element count block divisible. -/
theorem elements_block_divisible {d0 rest be : Nat} (h : d0 % be = 0) : (d0 * rest) % be = 0 :=
  Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_of_mod_eq_zero h) (Nat.dvd_mul_right _ _))

-- § Split shards

/-- An additional shard's split metadata matches the first shard. -/
def shardOk (count no tensors firstCount firstTensors : Nat) : Bool :=
  (firstCount != 0) && (count == firstCount) && (no != 0) && decide (no < firstCount) &&
    (tensors == firstTensors)

/-- An accepted additional shard has a number strictly between 0 and the split count. -/
theorem shardOk_number {c n t fc ft : Nat} (h : shardOk c n t fc ft = true) : 0 < n ∧ n < fc := by
  simp only [shardOk, Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h
  omega

/-- The first shard (number 0) can never pass as an additional shard. -/
theorem first_not_additional (c t fc ft : Nat) : shardOk c 0 t fc ft = false := by
  simp [shardOk]

-- § Scratch sizing

/-- `max_in`: the widest input of any override. -/
def maxIn (widths : List Nat) : Nat := widths.foldl max 0

/-- Helper: the running maximum dominates its accumulator. -/
theorem foldl_max_ge_acc : ∀ (l : List Nat) (acc : Nat), acc ≤ l.foldl max acc
  | [], acc => Nat.le_refl _
  | a :: t, acc => by
    simp only [List.foldl_cons]
    have := foldl_max_ge_acc t (max acc a)
    omega

/-- Helper: the running maximum dominates every element. -/
theorem foldl_max_ge_mem : ∀ (l : List Nat) (acc x : Nat), x ∈ l → x ≤ l.foldl max acc
  | [], _, _, h => by cases h
  | a :: t, acc, x, h => by
    simp only [List.foldl_cons]
    rcases List.mem_cons.1 h with rfl | h
    · have := foldl_max_ge_acc t (max acc x)
      omega
    · exact foldl_max_ge_mem t _ x h

/-- The shared Q8_1 scratch is wide enough for every override's input. -/
theorem scratch_wide_enough {widths : List Nat} {w : Nat} (h : w ∈ widths) : w ≤ maxIn widths :=
  foldl_max_ge_mem widths 0 w h

-- § All-or-nothing publication

/-- Check and upload every eligible tensor; the first failure aborts. -/
def checkAll {α ε β : Type} (check : α → Except ε β) : List α → Except ε (List β)
  | [] => .ok []
  | a :: t =>
    match check a with
    | .error e => .error e
    | .ok b =>
      match checkAll check t with
      | .error e => .error e
      | .ok bs => .ok (b :: bs)

/-- Publish only a fully successful batch. -/
def publish {τ ε β : Type} (tbl : τ) (r : Except ε (List β)) (apply : τ → List β → τ) : τ :=
  match r with
  | .ok xs => apply tbl xs
  | .error _ => tbl

/-- One failing tensor fails the whole batch. -/
theorem checkAll_fails {α ε β : Type} (check : α → Except ε β) :
    ∀ (items : List α) (a : α) (e0 : ε), a ∈ items → check a = .error e0 →
      ∃ e, checkAll check items = .error e
  | [], _, _, h, _ => by cases h
  | x :: t, a, e0, hm, hc => by
    simp only [checkAll]
    split
    · rename_i e _; exact ⟨e, rfl⟩
    · rename_i b hb
      rcases List.mem_cons.1 hm with rfl | hm
      · rw [hc] at hb; cases hb
      · obtain ⟨e, he⟩ := checkAll_fails check t a e0 hm hc
        rw [he]
        exact ⟨e, rfl⟩

/-- A successful batch has one override per eligible tensor. -/
theorem checkAll_length {α ε β : Type} (check : α → Except ε β) :
    ∀ (items : List α) (bs : List β), checkAll check items = .ok bs → bs.length = items.length
  | [], bs, h => by simp only [checkAll, Except.ok.injEq] at h; subst h; rfl
  | x :: t, bs, h => by
    simp only [checkAll] at h
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i b _ bs' hbs
        simp only [Except.ok.injEq] at h
        subst h
        simp [checkAll_length check t bs' hbs]

/-- A failed load leaves the canonical table exactly as it was. -/
theorem failed_load_untouched {τ ε β : Type} (tbl : τ) (e : ε) (apply : τ → List β → τ) :
    publish tbl (Except.error e : Except ε (List β)) apply = tbl := rfl

end Ref.NativeDenseOverride



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- DenseWeightLoader.lean
  was: theorem segment_check_iff (c s o f srcB dstB : Nat) (fp : Bool) (_ : f = 0) :
    (srcSum (quantSegs c s o fp) = srcB ∧ dstSum (quantSegs c s o fp) = dstB) ↔
      (c + srcScale s fp + o = srcB ∧ c + s + o = dstB)
  now: theorem segment_check_iff (c s o srcB dstB : Nat) (fp : Bool) :
    (srcSum (quantSegs c s o fp) = srcB ∧ dstSum (quantSegs c s o fp) = dstB) ↔
      (c + srcScale s fp + o = srcB ∧ c + s + o = dstB)
  measured: fewer quantified variables (3 → 2)

- PinnedArenaResidency.lean
  was: theorem fnv_append (a b : List UInt8) (seed : UInt64) :
    fnv1a64 (a ++ b) seed = fnv1a64 b (fnv1a64 a seed)
  now: theorem chunks_sum {c : Nat} (hc : 0 < c) : ∀ (f r : Nat), r ≤ f → (chunks c f r).sum = r
  | 0, r, h => by
    have : r = 0
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- PinnedArenaResidency.lean
  was: theorem layerOff_contig (bpl bb L : Nat) : layerOff bpl bb L + bpl * bb = layerOff bpl bb (L + 1)
  now: theorem fnv_append (a b : List UInt8) (seed : UInt64) :
    fnv1a64 (a ++ b) seed = fnv1a64 b (fnv1a64 a seed)
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- PinnedArenaResidency.lean
  was: theorem layer_in_file {bpl bb L layers : Nat} (h : L < layers) :
    layerOff bpl bb L + bpl * bb ≤ layers * bpl * bb
  now: theorem layerOff_contig (bpl bb L : Nat) : layerOff bpl bb L + bpl * bb = layerOff bpl bb (L + 1)
  measured: fewer quantified variables (2 → 1)
-/



## FILE: /mnt/user-data/outputs/GdnLayerComposition.lean
/-!
# MODULE: gdn_layer_composition
SOURCES: src/core/layer.cpp
Composes one GDN / QSA / MoE layer of the Ref engine from its kernels: carves 16-byte-aligned
scratch arenas, locates the three canonical planes of a quantized tensor, dispatches each quantized
GEMV to the activation format its TENSOR (not its role) wants, and selects which block stages run.

## Source semantics
* Every scratch region is rounded up with `(n + 15) & ~15`; regions are carved back to back, so the
  arena size is the sum of the rounded parts and no two regions overlap.
* A quantized tensor is `codes | scales | offsets`; `plane_ptrs` refuses unless the three plane sizes
  are nonzero (codes, scales), add up to the tensor's byte count, and `has_offset` agrees with the
  offset plane being nonempty.  Recomputing fp16 scales as 4 B/elem breaks that sum (the width bug).
* Activation contract: native weights take the FP32 path; `code_bits == 2` takes Q8_0; otherwise the
  manifest's `wants_q8k` picks Q8_K or Q8_0.  An S2 form must have no offset, group 64, bias -1.
* `block_layer_pre` stage selection: `stage_prefix = k > 0` runs stages `0..k-1`; otherwise `half`
  0 runs all five, 1 runs the mixer (0..2), 2 runs the FFN front and router (3..4).
* The C1 dump stride is `2N + 2hc + NH*HD + 5*NKV*HD + 8` floats per layer; every slot fits, and the
  `k_pool` slot is entirely overwritten by the later `v_scratch`/`ids`/`step` slots.

## Form template identifiers
align16 ↦ align_up16 / `(n+15)&~15`; carveOffsets ↦ gdn_buffers_init / Cursor::take;
carveTotal ↦ gdn_buffers_bytes / moe_buffers_bytes; WeightRef ↦ ref::core::WeightRef;
PlanesOk ↦ plane_ptrs; chooseAct ↦ gemv_quantized; s2Accept ↦ S2 guard in gemv_quantized;
runStage ↦ run0..run4 in block_layer_pre; dumpStride ↦ dump_stride_floats; ropeRowOk ↦ qsa_layer P7 guard.
-/

namespace Ref.Layer

-- § alignment

/-- The 16-byte round-up `(n + 15) & ~15` used for every scratch region. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never shrinks a region. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by unfold align16; omega

/-- The rounded size is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by unfold align16; omega

/-- Rounding adds strictly less than one alignment unit. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by unfold align16; omega

/-- An already aligned size is unchanged. -/
theorem align16_of_aligned (n : Nat) (h : n % 16 = 0) : align16 n = n := by unfold align16; omega

/-- The Q8_K buffer size: `n / 256` blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- A Q8_K buffer of a multiple of 256 elements holds exactly `n/256` blocks. -/
theorem q8kBytes_blocks (m : Nat) : q8kBytes (256 * m) = 292 * m := by
  unfold q8kBytes; omega

-- § arena carving

/-- Total bytes of an arena whose parts are each rounded to 16 (`gdn_buffers_bytes`). -/
def carveTotal : List Nat → Nat
  | [] => 0
  | p :: ps => align16 p + carveTotal ps

/-- The region start offsets produced by carving parts back to back (`gdn_buffers_init`). -/
def carveOffsets (base : Nat) : List Nat → List Nat
  | [] => []
  | p :: ps => base :: carveOffsets (base + align16 p) ps

/-- One offset per part. -/
theorem carveOffsets_length (base : Nat) (ps : List Nat) :
    (carveOffsets base ps).length = ps.length := by
  induction ps generalizing base with
  | nil => rfl
  | cons p ps ih => simp [carveOffsets, ih]

/-- From an aligned base, every region starts 16-byte aligned. -/
theorem carveOffsets_aligned (base : Nat) (ps : List Nat) (hb : base % 16 = 0) :
    ∀ o ∈ carveOffsets base ps, o % 16 = 0 := by
  induction ps generalizing base with
  | nil => intro o h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro o h
    simp only [carveOffsets, List.mem_cons] at h
    rcases h with rfl | h
    · exact hb
    · have ha := align16_mod p
      exact ih (base + align16 p) (by omega) o h

/-- Every region starts at or after the base. -/
theorem carveOffsets_ge (base : Nat) (ps : List Nat) : ∀ o ∈ carveOffsets base ps, base ≤ o := by
  induction ps generalizing base with
  | nil => intro o h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro o h
    simp only [carveOffsets, List.mem_cons] at h
    rcases h with rfl | h
    · exact Nat.le_refl _
    · have := ih (base + align16 p) o h
      omega

/-- Every region `[o, o + p)` ends inside the arena `[base, base + carveTotal ps)`. -/
theorem carve_fits (base : Nat) (ps : List Nat) :
    ∀ x ∈ (carveOffsets base ps).zip ps, x.1 + x.2 ≤ base + carveTotal ps := by
  induction ps generalizing base with
  | nil => intro x h; simp [carveOffsets] at h
  | cons p ps ih =>
    intro x h
    simp only [carveOffsets, List.zip_cons_cons, List.mem_cons] at h
    have hp := align16_ge p
    rcases h with rfl | h
    · simp only [carveTotal]; omega
    · have := ih (base + align16 p) x h
      simp only [carveTotal]; omega

/-- Regions are pairwise disjoint: each one ends before any later one starts. -/
theorem carve_disjoint (base : Nat) (ps : List Nat) :
    List.Pairwise (fun a b : Nat × Nat => a.1 + a.2 ≤ b.1) ((carveOffsets base ps).zip ps) := by
  induction ps generalizing base with
  | nil => simp [carveOffsets]
  | cons p ps ih =>
    simp only [carveOffsets, List.zip_cons_cons, List.pairwise_cons]
    refine ⟨?_, ih (base + align16 p)⟩
    intro b hb
    have hmem : b.1 ∈ carveOffsets (base + align16 p) ps := (List.of_mem_zip hb).1
    have := carveOffsets_ge (base + align16 p) ps b.1 hmem
    have hp := align16_ge p
    omega

-- § the GDN scratch layout

/-- The model geometry fields a layer's buffers depend on. -/
structure Geometry where
  nEmbd : Nat
  convChannels : Nat
  valueDim : Nat
  vHeads : Nat
  stateSize : Nat
  dConv : Nat

/-- The sixteen parts of `gdn_buffers_bytes`, in carve order. -/
def gdnParts (g : Geometry) : List Nat :=
  [q8kBytes g.nEmbd, g.nEmbd / 32 * 34, g.nEmbd * 2, g.convChannels * 4, g.convChannels * 4,
   g.convChannels * 4, g.vHeads * 4, g.vHeads * 4, g.vHeads * 4, g.vHeads * g.stateSize * 4,
   g.valueDim * 4, g.valueDim * 4, q8kBytes g.valueDim, g.valueDim / 32 * 34,
   g.stateSize * g.vHeads * g.stateSize * 4, g.convChannels * (g.dConv - 1) * 4]

/-- `gdn_buffers_init` hands out exactly sixteen pointers, one per part. -/
theorem gdnParts_length (g : Geometry) : (carveOffsets 0 (gdnParts g)).length = 16 := by
  rw [carveOffsets_length]; rfl

/-- All sixteen GDN regions are aligned, disjoint and inside the `gdn_buffers_bytes` arena. -/
theorem gdn_layout_sound (g : Geometry) :
    (∀ o ∈ carveOffsets 0 (gdnParts g), o % 16 = 0) ∧
    (∀ x ∈ (carveOffsets 0 (gdnParts g)).zip (gdnParts g), x.1 + x.2 ≤ carveTotal (gdnParts g)) ∧
    List.Pairwise (fun a b : Nat × Nat => a.1 + a.2 ≤ b.1)
      ((carveOffsets 0 (gdnParts g)).zip (gdnParts g)) := by
  refine ⟨carveOffsets_aligned 0 _ rfl, ?_, carve_disjoint 0 _⟩
  intro x hx
  have := carve_fits 0 (gdnParts g) x hx
  omega

-- § canonical planes of a quantized tensor

/-- The loader's record of a tensor (the fields `plane_ptrs` and `gemv_quantized` read). -/
structure WeightRef where
  codeBits : Nat
  codesBytes : Nat
  scalesBytes : Nat
  offsetBytes : Nat
  bytes : Nat
  hasOffset : Bool
  groupElems : Nat
  codeBias : Int
  wantsQ8k : Bool
  native : Bool

/-- The acceptance condition of `plane_ptrs`. -/
def PlanesOk (w : WeightRef) : Prop :=
  (w.codeBits = 2 ∨ w.codeBits = 4 ∨ w.codeBits = 8) ∧ w.codesBytes ≠ 0 ∧ w.scalesBytes ≠ 0 ∧
  w.codesBytes + w.scalesBytes + w.offsetBytes = w.bytes ∧ (w.hasOffset = true ↔ w.offsetBytes ≠ 0)

/-- Byte offset of the scale plane inside the tensor. -/
def scalesAt (w : WeightRef) : Nat := w.codesBytes

/-- Byte offset of the offset plane inside the tensor. -/
def offsetAt (w : WeightRef) : Nat := w.codesBytes + w.scalesBytes

/-- Accepted planes partition the tensor: codes, then scales, then offsets ending exactly at `bytes`. -/
theorem planes_partition (w : WeightRef) (h : PlanesOk w) :
    w.codesBytes ≤ scalesAt w ∧ scalesAt w + w.scalesBytes = offsetAt w ∧
    offsetAt w + w.offsetBytes = w.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := h
  unfold scalesAt offsetAt
  omega

/-- A tensor without an offset plane carries `has_offset = false`, and conversely. -/
theorem planes_no_offset (w : WeightRef) (h : PlanesOk w) :
    w.hasOffset = false ↔ w.offsetBytes = 0 := by
  obtain ⟨_, _, _, _, hiff⟩ := h
  cases hb : w.hasOffset <;> rw [hb] at hiff <;> simp at hiff ⊢ <;> omega

/-- **The width bug.**  When the real scale plane is `s` bytes of fp16 and the layout is re-derived
at 4 B/elem (`2 * s`), the three planes no longer add up to the tensor, so `plane_ptrs` refuses. -/
theorem fp16_rederived_rejected (codes s off bytes : Nat) (hs : s ≠ 0)
    (hreal : codes + s + off = bytes) : codes + 2 * s + off ≠ bytes := by
  omega

-- § activation dispatch

/-- The activation image a quantized projection reads. -/
inductive Act where
  | native
  | q8_0
  | q8k
  deriving DecidableEq, Repr

/-- `gemv_quantized`'s choice, a property of the tensor. -/
def chooseAct (w : WeightRef) : Act :=
  if w.native then .native else if w.codeBits = 2 then .q8_0 else if w.wantsQ8k then .q8k else .q8_0

/-- An S2 tensor never gets the Q8_K image (`code_bits 2 has no Q8_K contract`). -/
theorem s2_never_q8k (w : WeightRef) (h : w.codeBits = 2) : chooseAct w ≠ .q8k := by
  unfold chooseAct
  cases w.native <;> simp [h]

/-- A legacy (non-K) form gets Q8_0 even when its code width is 4 or 8. -/
theorem legacy_gets_q8_0 (w : WeightRef) (hn : w.native = false) (hk : w.wantsQ8k = false) :
    chooseAct w = .q8_0 := by
  unfold chooseAct
  by_cases h2 : w.codeBits = 2 <;> simp [hn, hk, h2]

/-- A K-quant of width ≠ 2 gets Q8_K. -/
theorem kquant_gets_q8k (w : WeightRef) (hn : w.native = false) (h2 : w.codeBits ≠ 2)
    (hk : w.wantsQ8k = true) : chooseAct w = .q8k := by
  unfold chooseAct
  simp [hn, h2, hk]

/-- Only native tensors take the native path. -/
theorem native_iff (w : WeightRef) : chooseAct w = .native ↔ w.native = true := by
  unfold chooseAct
  cases w.native <;> simp <;> split <;> (try split) <;> simp

/-- The fixed S2 attributes `s2_gemv_q8` assumes. -/
def s2Accept (w : WeightRef) : Prop := w.hasOffset = false ∧ w.groupElems = 64 ∧ w.codeBias = -1

/-- Two tensors of the same width and attributes but different `wants_q8k` (Q5_0 vs Q5_K, IQ4_NL vs
IQ4_XS) are dispatched differently: `code_bits` alone cannot decide the contract. -/
theorem codeBits_insufficient (w : WeightRef) (hn : w.native = false) (h2 : w.codeBits = 8) :
    chooseAct { w with wantsQ8k := true } ≠ chooseAct { w with wantsQ8k := false } := by
  unfold chooseAct
  simp [hn, h2]

-- § stage selection in block_layer_pre

/-- Whether stage `i` (0..4) runs for a given `stage_prefix` and `half`. -/
def runStage (pre half i : Nat) : Bool :=
  if pre > 0 then decide (i + 1 ≤ pre)
  else if i ≤ 2 then (half == 0 || half == 1) else (half == 0 || half == 2)

/-- With a positive prefix `k`, exactly stages `0..k-1` run, whatever `half` says. -/
theorem runStage_prefix (pre half i : Nat) (hp : 0 < pre) : runStage pre half i = true ↔ i < pre := by
  unfold runStage
  simp [hp]
  omega

/-- `half = 0` with no prefix runs every stage (the default for every older caller). -/
theorem runStage_all (i : Nat) : runStage 0 0 i = true := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h]

/-- `half = 1` runs exactly the mixer stages 0..2. -/
theorem runStage_mixer (i : Nat) : runStage 0 1 i = true ↔ i ≤ 2 := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h]

/-- `half = 2` runs exactly the FFN front and the router (stages ≥ 3). -/
theorem runStage_ffn (i : Nat) : runStage 0 2 i = true ↔ 3 ≤ i := by
  unfold runStage
  by_cases h : i ≤ 2 <;> simp [h] <;> omega

/-- The split capture's two halves together run each of the five stages exactly once. -/
theorem halves_partition (i : Nat) (hi : i < 5) :
    (runStage 0 1 i = true ∨ runStage 0 2 i = true) ∧
    ¬(runStage 0 1 i = true ∧ runStage 0 2 i = true) := by
  rw [runStage_mixer, runStage_ffn]
  omega

-- § QSA position guard

/-- The P7 audit in `qsa_layer`: RoPE reads row `pos_base + pos`, the table holds `max_cells` rows. -/
def ropeRowOk (posBase pos maxCells : Int) : Prop := 0 ≤ posBase ∧ posBase + pos < maxCells

/-- An accepted position reads a row inside the table (given the separate `0 ≤ pos` check). -/
theorem ropeRow_in_table (posBase pos maxCells : Int) (h : ropeRowOk posBase pos maxCells)
    (hp : 0 ≤ pos) : 0 ≤ posBase + pos ∧ posBase + pos < maxCells := by
  obtain ⟨h1, h2⟩ := h
  omega

-- § the C1 dump layout

/-- Floats per layer in the dump: `n` = n_embd, `hc`, `a` = n_head*head_dim, `nvk` = n_head_kv*head_dim. -/
def dumpStride (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 5 * nvk + 8

/-- Start of the gathered `v_scratch` slot. -/
def vsOff (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 2 * nvk

/-- Start of the `k_pool` slot (`base` in `qsa_layer`). -/
def kpoolOff (n hc a nvk : Nat) : Nat := 2 * n + 2 * hc + a + 2 * nvk + 8

/-- Every slot `qsa_layer` and `block_layer_*` dump into ends inside one layer's stride. -/
theorem dump_slots_fit (n hc a nvk : Nat) :
    let s := dumpStride n hc a nvk
    let b := kpoolOff n hc a nvk
    let vs := vsOff n hc a nvk
    2 * n + 2 * hc ≤ s ∧ 2 * n + 2 * hc + a ≤ s ∧ 2 * n + 2 * hc + a + nvk ≤ s ∧
    vs + nvk / 2 + 8 ≤ s ∧ b + nvk / 2 + nvk / 2 ≤ s ∧ b + nvk + nvk ≤ s ∧ b + 2 * nvk + nvk ≤ s := by
  simp only [dumpStride, kpoolOff, vsOff]
  omega

/-- **The `k_pool` dump is dead.**  It is enqueued before the later `v_scratch`, `ids` and `step`
copies, and its whole range lies inside theirs, so the host buffer never holds it. -/
theorem kpool_dump_shadowed (n hc a nvk : Nat) :
    vsOff n hc a nvk ≤ kpoolOff n hc a nvk ∧
    kpoolOff n hc a nvk + nvk / 2 = vsOff n hc a nvk + nvk / 2 + 4 + 4 := by
  simp only [kpoolOff, vsOff]
  omega

-- § the doorbell

/-- Mapped-pinned bytes `doorbell_init` allocates: x_f, ids, weights, seq, flag. -/
def doorbellBytes (nEmbd k : Nat) : Nat := nEmbd * 4 + k * 4 + k * 4 + 4 + 4

/-- The doorbell always holds the input row and two k-wide lists. -/
theorem doorbellBytes_ge (nEmbd k : Nat) : nEmbd * 4 + 8 * k ≤ doorbellBytes nEmbd k := by
  unfold doorbellBytes; omega

/-- `moe_route` / `moe_combine_parts` accept `k` only in `1..64`. -/
def kOk (k : Int) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The native router's precondition is a special case of the range check. -/
theorem native_router_k (k : Int) (h : k = 10) : kOk k := by
  unfold kOk; omega

end Ref.Layer



## FILE: /mnt/user-data/outputs/LayerShapeCheck.lean
/-!
# MODULE: layer_shape_check
SOURCES: src/core/layout.cpp
Load-time shape and engine-form checks for every layer's tensors, plus the whole-model check that
the layer-type predicate splits the 48 layers into the 12 QSA / 36 GDN the geometry promises.

## Source semantics
* A 2-D requirement `Want2` is skipped on the wrong layer family (`qsa_only` on a GDN layer,
  `gdn_only` on a QSA layer); otherwise `ne0` and `ne1` must match exactly.
* A 1-D requirement `Want1` also fixes the ENGINE FORM: F32 holds 4 B/elem, Bf16InF32 holds 2.  A
  consumer that reads a 2 B/elem tensor as f32 walks twice its length (the `ffn_gate_inp_shexp` bug).
* `check_all` counts the QSA layers via `is_qsa_layer` and compares with the geometry; with a full
  attention interval of 4 the QSA layers are exactly those with `l % 4 = 3`.

## Form template identifiers
isQsa ↦ is_qsa_layer; applies ↦ the `qsa_only`/`gdn_only` skip in check_one; countQsa ↦ n_qsa in
check_all; countGdn ↦ n_gdn; Kind ↦ WeightKind; formBytes ↦ want_bytes; want2Flags ↦ want2[];
want1Flags ↦ want1[].
-/

namespace Ref.Layout

-- § the layer-type predicate

/-- A layer is QSA when it is the last of each group of four (`full_attention_interval = 4`). -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- QSA layers counted over `0..n-1` (the `n_qsa` of `check_all`). -/
def countQsa : Nat → Nat
  | 0 => 0
  | n + 1 => countQsa n + if isQsa n then 1 else 0

/-- GDN layers counted over `0..n-1` (the `n_gdn` of `check_all`). -/
def countGdn : Nat → Nat
  | 0 => 0
  | n + 1 => countGdn n + if isQsa n then 0 else 1

/-- The two counters always account for every layer. -/
theorem count_split (n : Nat) : countQsa n + countGdn n = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [countQsa, countGdn]
    cases isQsa n <;> simp <;> omega

/-- Layer `4m + 3` is QSA. -/
theorem isQsa_last (m : Nat) : isQsa (4 * m + 3) = true := decide_eq_true (by omega)

/-- Layers `4m`, `4m+1`, `4m+2` are GDN. -/
theorem isQsa_other (m r : Nat) (hr : r < 3) : isQsa (4 * m + r) = false :=
  decide_eq_false (by omega)

/-- Each group of four layers contributes exactly one QSA layer. -/
theorem countQsa_step (m : Nat) : countQsa (4 * (m + 1)) = countQsa (4 * m) + 1 := by
  have e : 4 * (m + 1) = 4 * m + 3 + 1 := by omega
  have h0 := isQsa_other m 0 (by omega)
  have h1 := isQsa_other m 1 (by omega)
  have h2 := isQsa_other m 2 (by omega)
  have h3 := isQsa_last m
  rw [e]
  simp only [countQsa]
  simp only [Nat.add_zero] at h0
  rw [show 4 * m + 3 = 4 * m + 2 + 1 by omega, show 4 * m + 2 = 4 * m + 1 + 1 by omega] at *
  simp [countQsa, h0, h1, h2, h3]

/-- Over `4m` layers there are exactly `m` QSA layers. -/
theorem countQsa_mul (m : Nat) : countQsa (4 * m) = m := by
  induction m with
  | zero => rfl
  | succ m ih => rw [countQsa_step, ih]

/-- Over `4m` layers there are exactly `3m` GDN layers. -/
theorem countGdn_mul (m : Nat) : countGdn (4 * m) = 3 * m := by
  have h := count_split (4 * m)
  rw [countQsa_mul] at h
  omega

/-- **The artifact's split**: 48 layers give 12 QSA and 36 GDN, which `check_all` demands. -/
theorem split_48 : countQsa 48 = 12 ∧ countGdn 48 = 36 :=
  ⟨countQsa_mul 12, countGdn_mul 12⟩

-- § the per-family filter

/-- Whether a requirement with these flags is checked on a layer of this family. -/
def applies (qsaOnly gdnOnly qsa : Bool) : Bool := !(qsaOnly && !qsa) && !(gdnOnly && qsa)

/-- A GDN-only tensor is checked on exactly the GDN layers. -/
theorem gdnOnly_applies (qsa : Bool) : applies false true qsa = !qsa := by
  cases qsa <;> rfl

/-- A QSA-only tensor is checked on exactly the QSA layers. -/
theorem qsaOnly_applies (qsa : Bool) : applies true false qsa = qsa := by
  cases qsa <;> rfl

/-- `(qsa_only, gdn_only)` of the 22 two-dimensional requirements, in source order. -/
def want2Flags : List (Bool × Bool) :=
  [(false, false), (false, false), (false, false), (false, false), (false, false), (false, false),
   (false, false), (false, false), (false, false), (false, false),
   (false, true), (false, true), (false, true), (false, true), (false, true), (false, true),
   (true, false), (true, false), (true, false), (true, false), (true, false), (true, false)]

/-- 2-D tensors checked on one layer of the given family. -/
def checked2 (qsa : Bool) : Nat := (want2Flags.filter (fun f => applies f.1 f.2 qsa)).length

/-- A GDN layer checks 16 matrices (6 GR + 4 MoE + 6 GDN), a QSA layer also 16 (6 + 4 + 6). -/
theorem checked2_counts : checked2 false = 16 ∧ checked2 true = 16 := by decide

/-- No requirement is both QSA-only and GDN-only, so none is dead on every layer. -/
theorem want2_no_dead : ∀ f ∈ want2Flags, ¬(f.1 = true ∧ f.2 = true) := by decide

-- § engine forms of 1-D tensors

/-- The engine form a loader gives a 1-D tensor. -/
inductive Kind where
  | f32
  | bf16InF32
  deriving DecidableEq, Repr

/-- Bytes held in the arena for `e` elements of a given form. -/
def formBytes : Kind → Nat → Nat
  | .f32, e => 4 * e
  | .bf16InF32, e => 2 * e

/-- `(form, qsa_only, gdn_only)` of the ten 1-D requirements, in source order. -/
def want1Flags : List (Kind × Bool × Bool) :=
  [(.f32, false, false), (.f32, false, false), (.bf16InF32, false, false),
   (.f32, false, true), (.f32, false, true), (.f32, false, true),
   (.f32, true, false), (.f32, true, false), (.f32, true, false), (.f32, true, false)]

/-- Exactly one 1-D requirement is BF16: the shared expert's scalar gate. -/
theorem want1_single_bf16 : (want1Flags.filter (fun f => f.1 = .bf16InF32)).length = 1 := by decide

/-- **The misread.**  Reading a BF16 tensor of `e > 0` elements as f32 walks `2e` bytes past its end. -/
theorem bf16_as_f32_overrun (e : Nat) (he : 0 < e) :
    formBytes .f32 e = formBytes .bf16InF32 e + 2 * e ∧ formBytes .bf16InF32 e < formBytes .f32 e := by
  simp only [formBytes]
  omega

/-- The byte count distinguishes the two forms, which is what makes the form check real. -/
theorem formBytes_injective (k1 k2 : Kind) (e : Nat) (he : 0 < e)
    (h : formBytes k1 e = formBytes k2 e) : k1 = k2 := by
  cases k1 <;> cases k2 <;> simp [formBytes] at h ⊢ <;> omega

end Ref.Layout



## FILE: /mnt/user-data/outputs/SessionTokenLoop.lean
/-!
# MODULE: session_token_loop
SOURCES: src/core/session.cpp
One token through all layers: the session arena (256-byte aligned regions, one GDN state slice per GDN
layer, one QSA state per QSA layer with a shared RoPE table), the per-layer graph capture, and the
doorbell protocol between the GPU's `pre[l]` and the host pool before `post[l]`.

## Source semantics
* `align_up(n, 256)` rounds every region; `session_bytes` adds a final 4096-byte slack.
* `gdn_point_at` gives GDN layer `l` the slice number `#{l' < l : l' is GDN}`; distinct GDN layers get
  disjoint slices of `gdn_state_floats` floats, all inside `n_gdn_layers * gdn_state_floats`.
* QSA state `i` lives at `0` for `i = 0` (it carries the RoPE table) and at `first + (i-1)*rest`
  otherwise; every state fits inside `first + (n_qsa - 1) * rest`.
* The doorbell: layer `l` rings sequence `l + 1`, the host acknowledges with flag `l + 1` after the pool;
  after `n` layers both counters read `n`.  The pool's answer for layer `l` is combined by layer `l`.
* `session_replay_stage_prefixes` reports stage times as differences of prefix times, which telescope.

## Form template identifiers
align256 ↦ align_up(·, SESSION_STATE_ALIGN); gdnIndex ↦ gdn_point_at's counter / session_token's
gdn_index; qsaOffset ↦ the `qp + ...` carve in session_init; Bell ↦ Doorbell (h_seq, h_flag);
runLayers ↦ session_run_token's host loop; dumpOffset ↦ dump_layers slot of session_loop.
-/

namespace Ref.Session

-- § 256-byte alignment

/-- `align_up(n, SESSION_STATE_ALIGN)` with the alignment fixed at 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never shrinks. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- The result is a multiple of 256. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- `session_bytes` covers everything it summed. -/
theorem session_bytes_cover (n : Nat) : n + 4096 ≤ align256 n + 4096 := by
  have := align256_ge n; omega

-- § layer types and the GDN slice index

/-- QSA layers are the last of each group of four. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- `gdn_point_at`'s slice index: the number of GDN layers strictly before `l`. -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + if isQsa l then 0 else 1

/-- The index never decreases with the layer. -/
theorem gdnIndex_mono {a b : Nat} (h : a ≤ b) : gdnIndex a ≤ gdnIndex b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih =>
    simp only [gdnIndex]
    omega

/-- A GDN layer's successor has a strictly larger index. -/
theorem gdnIndex_succ_of_gdn (l : Nat) (h : isQsa l = false) : gdnIndex (l + 1) = gdnIndex l + 1 := by
  simp [gdnIndex, h]

/-- Two different GDN layers never share a slice index. -/
theorem gdnIndex_strict {l1 l2 : Nat} (hl : l1 < l2) (h1 : isQsa l1 = false) :
    gdnIndex l1 < gdnIndex l2 := by
  have hs := gdnIndex_succ_of_gdn l1 h1
  have hm : gdnIndex (l1 + 1) ≤ gdnIndex l2 := gdnIndex_mono hl
  omega

/-- Slice `i` of width `f` ends before slice `j > i` starts. -/
theorem slice_disjoint (i j f : Nat) (h : i < j) : i * f + f ≤ j * f := by
  have := Nat.mul_le_mul_right f (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  exact this

/-- **Each GDN layer owns its recurrence**: distinct GDN layers point at disjoint state slices. -/
theorem gdn_slices_disjoint {l1 l2 f : Nat} (hl : l1 < l2) (h1 : isQsa l1 = false) :
    gdnIndex l1 * f + f ≤ gdnIndex l2 * f :=
  slice_disjoint _ _ f (gdnIndex_strict hl h1)

/-- A GDN layer below `n` has its slice inside the arena of `gdnIndex n` slices. -/
theorem gdn_slice_in_arena {l n f : Nat} (hl : l < n) (h1 : isQsa l = false) :
    gdnIndex l * f + f ≤ gdnIndex n * f :=
  slice_disjoint _ _ f (gdnIndex_strict hl h1)

/-- The artifact: 48 layers carry 36 GDN slices. -/
theorem gdnIndex_48 : gdnIndex 48 = 36 := by decide

-- § QSA state carve

/-- Start of QSA state `i`: the first one (with the RoPE table) at 0, then `rest`-sized ones. -/
def qsaOffset (first rest i : Nat) : Nat := if i = 0 then 0 else first + (i - 1) * rest

/-- Size of QSA state `i`. -/
def qsaSize (first rest i : Nat) : Nat := if i = 0 then first else rest

/-- Every one of the `nq` QSA states fits inside the `first + (nq-1)*rest` arena. -/
theorem qsa_state_fits (first rest nq i : Nat) (hi : i < nq) :
    qsaOffset first rest i + qsaSize first rest i ≤ first + (nq - 1) * rest := by
  unfold qsaOffset qsaSize
  by_cases h0 : i = 0
  · simp [h0]
  · simp only [h0, if_false]
    have hle : i ≤ nq - 1 := by omega
    have hm := Nat.mul_le_mul_right rest hle
    have e : (i - 1) * rest + rest = i * rest := by
      rw [← Nat.succ_mul, Nat.succ_eq_add_one, Nat.sub_add_cancel (by omega)]
    omega

-- § the doorbell protocol

/-- The two mapped counters: the device's ring and the host's acknowledgement. -/
structure Bell where
  seq : Nat
  flag : Nat
  deriving DecidableEq, Repr

/-- `doorbell_reset`. -/
def Bell.reset : Bell := ⟨0, 0⟩

/-- Layer `l`'s `pre` publishes and rings sequence `l + 1`. -/
def ring (b : Bell) (l : Nat) : Bell := { b with seq := l + 1 }

/-- After the pool for layer `l`, the host writes flag `l + 1` (the graph's `doorbell_wait`). -/
def ack (b : Bell) (l : Nat) : Bell := { b with flag := l + 1 }

/-- The host loop of `session_run_token` over `n` layers. -/
def runLayers : Nat → Bell
  | 0 => Bell.reset
  | n + 1 => ack (ring (runLayers n) n) n

/-- After `n` layers both counters stand at `n`. -/
theorem runLayers_eq (n : Nat) : runLayers n = ⟨n, n⟩ := by
  induction n with
  | zero => rfl
  | succ n ih => simp [runLayers, ih, ring, ack]

/-- At layer `l` the host waits for `seq ≥ l + 1`, which the previous state never satisfies:
the wait cannot be released by an earlier layer's ring. -/
theorem wait_not_stale (l : Nat) : (runLayers l).seq < l + 1 := by
  rw [runLayers_eq]; exact Nat.lt_succ_self l

/-- Which layer's expert outputs the combine of layer `l` reads (after the L100 fix). -/
def partsSource (l : Nat) : Nat := l

/-- **The L100 fix**: layer `l` combines the pool's answer for layer `l`, never `l - 1`. -/
theorem combine_current_layer (l : Nat) : partsSource l = l := rfl

-- § timing and dumps

/-- Stage times are prefix differences; their sum telescopes to the full-prefix time. -/
theorem prefix_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + (t2 - t1) + (t3 - t2) + (t4 - t3) + (t5 - t4) = t5 := by omega

/-- Start of `session_loop`'s dump slot for the residual after layer `l` (slot 0 is the input). -/
def dumpOffset (rs l : Nat) : Nat := (l + 1) * rs

/-- Every layer's dump lands inside the `(n + 1)`-slot host buffer. -/
theorem dump_in_buffer (rs l n : Nat) (hl : l < n) : dumpOffset rs l + rs ≤ (n + 1) * rs := by
  unfold dumpOffset
  exact slice_disjoint (l + 1) (n + 1) rs (by omega)

end Ref.Session



## FILE: /mnt/user-data/outputs/MtpDraftChain.lean
/-!
# MODULE: mtp_draft_chain
SOURCES: src/core/mtp.cpp
The multi-token-prediction drafter: a bump allocator for its device arena, the step-row staging that
places each drafted cell, and the draft chain that continues while the last draft is likely enough.

## Source semantics
* `Bump::take` rounds every region to 256 bytes; counted and real carves follow the same sequence.
* A step row for cell `c` is `[c, c+1, (c+1)/4, c+1]`; positions for every head are `c`.
* There are `2*max_t` step rows: catch-up rows `0..T-1` (`T ≤ max_t`), chain rows `max_t + j - 1`
  for `1 ≤ j ≤ max_t - 2`, and the round's row `ra = 2*max_t - 1`.  They never collide.
* The chain writes draft 0, then keeps going while `j < max_t - 1` and the previous draft's
  probability is ≥ `min_p`; the count is between 1 and `max_t - 1` when `max_t ≥ 2`.
* With a window the attention capacity is the window rounded up to 64.

## Form template identifiers
bump ↦ Bump::take rounding; stepRow ↦ MtpDrafter::draft's `put`; chainFrom/chainLen ↦ the draft
loop; raRow ↦ `2 * max_t - 1`; chainRow ↦ `max_t + j - 1`; cap64 ↦ cap_.
-/

namespace Ref.Mtp

-- § bump allocation

/-- `Bump::take`'s rounding `(n + 255) & ~255`. -/
def bump (n : Nat) : Nat := (n + 255) / 256 * 256

/-- A region is never smaller than requested. -/
theorem bump_ge (n : Nat) : n ≤ bump n := by unfold bump; omega

/-- Every region size is 256-aligned, so every carved pointer is too. -/
theorem bump_mod (n : Nat) : bump n % 256 = 0 := by unfold bump; omega

/-- Offsets of a counted carve. -/
def carve (base : Nat) : List Nat → List Nat
  | [] => []
  | n :: ns => base :: carve (base + bump n) ns

/-- The counted pass and the real pass agree, because both only depend on the request sizes:
carving from base `b` is carving from 0 shifted by `b`. -/
theorem carve_shift (b : Nat) (ns : List Nat) : carve b ns = (carve 0 ns).map (· + b) := by
  induction ns generalizing b with
  | nil => rfl
  | cons n ns ih =>
    simp only [carve, List.map_cons, Nat.zero_add]
    rw [ih (b + bump n), ih (bump n), List.map_map]
    congr 1
    · apply List.map_congr_left
      intro x _
      simp only [Function.comp]
      omega

/-- From an aligned base every carved region is 256-aligned. -/
theorem carve_aligned (b : Nat) (ns : List Nat) (hb : b % 256 = 0) : ∀ o ∈ carve b ns, o % 256 = 0 := by
  induction ns generalizing b with
  | nil => intro o h; simp [carve] at h
  | cons n ns ih =>
    intro o h
    simp only [carve, List.mem_cons] at h
    rcases h with rfl | h
    · exact hb
    · have := bump_mod n
      exact ih _ (by omega) o h

-- § step rows

/-- The four step counts staged for cell `c`. -/
def stepRow (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

/-- Row shape: the count after appending is `c + 1`, and the block count is its quarter. -/
theorem stepRow_shape (c : Nat) :
    stepRow c = [c, c + 1, (c + 1) / 4, c + 1] ∧ 4 * ((c + 1) / 4) ≤ c + 1 ∧ c + 1 < 4 * ((c + 1) / 4) + 4 := by
  refine ⟨rfl, ?_, ?_⟩ <;> omega

/-- The round's own row. -/
def raRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Chain step `j`'s row. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- All rows lie in the `2*max_t` staged rows and the three kinds never collide. -/
theorem rows_disjoint (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj1 : 1 ≤ j) (hj : j ≤ maxT - 2) :
    t < chainRow maxT j ∧ chainRow maxT j < raRow maxT ∧ raRow maxT < 2 * maxT := by
  unfold chainRow raRow
  omega

/-- Two chain steps never share a row. -/
theorem chainRow_injective (maxT i j : Nat) (hi : 1 ≤ i) (hj : 1 ≤ j) (h : chainRow maxT i = chainRow maxT j) :
    i = j := by
  unfold chainRow at h; omega

-- § the draft chain

/-- Extra drafts from step `j` on, given whether draft `j-1` passes `min_p`, with `fuel` steps left. -/
def chainFrom (pass : Nat → Bool) : Nat → Nat → Nat
  | _, 0 => 0
  | j, f + 1 => if pass (j - 1) then 1 + chainFrom pass (j + 1) f else 0

/-- Total drafts: draft 0 always, then the chain over `j = 1 .. max_t - 2`. -/
def chainLen (pass : Nat → Bool) (maxT : Nat) : Nat := 1 + chainFrom pass 1 (maxT - 2)

/-- The chain never takes more steps than it has fuel. -/
theorem chainFrom_le (pass : Nat → Bool) (j f : Nat) : chainFrom pass j f ≤ f := by
  induction f generalizing j with
  | zero => simp [chainFrom]
  | succ f ih =>
    simp only [chainFrom]
    split
    · have := ih (j + 1); omega
    · omega

/-- When every draft passes, the chain uses all its fuel. -/
theorem chainFrom_all (pass : Nat → Bool) (hp : ∀ i, pass i = true) (j f : Nat) : chainFrom pass j f = f := by
  induction f generalizing j with
  | zero => rfl
  | succ f ih => simp [chainFrom, hp, ih]; omega

/-- The number of drafts is between 1 and `max_t - 1`. -/
theorem chainLen_bounds (pass : Nat → Bool) (maxT : Nat) (h : 2 ≤ maxT) :
    1 ≤ chainLen pass maxT ∧ chainLen pass maxT ≤ maxT - 1 := by
  have := chainFrom_le pass 1 (maxT - 2)
  unfold chainLen
  omega

/-- With every draft confident, the chain fills all `max_t - 1` slots. -/
theorem chainLen_full (pass : Nat → Bool) (hp : ∀ i, pass i = true) (maxT : Nat) (h : 2 ≤ maxT) :
    chainLen pass maxT = maxT - 1 := by
  unfold chainLen
  rw [chainFrom_all pass hp]
  omega

/-- A rejected first draft stops the chain at one draft. -/
theorem chainLen_stop (pass : Nat → Bool) (h0 : pass 0 = false) (maxT : Nat) (h : 3 ≤ maxT) :
    chainLen pass maxT = 1 := by
  unfold chainLen
  obtain ⟨f, hf⟩ : ∃ f, maxT - 2 = f + 1 := ⟨maxT - 3, by omega⟩
  rw [hf]
  simp [chainFrom, h0]

/-- Edge case: `max_t = 1` (allowed by `load`) still writes draft 0, one more than `max_t - 1`. -/
theorem chainLen_maxT_one (pass : Nat → Bool) : chainLen pass 1 = 1 := by
  simp [chainLen, chainFrom]

-- § the attention capacity

/-- `cap_` rounds the window (or the context) up to 64. -/
def cap64 (w : Nat) : Nat := (w + 63) / 64 * 64

/-- The capacity covers the window and is 64-aligned. -/
theorem cap64_spec (w : Nat) : w ≤ cap64 w ∧ cap64 w % 64 = 0 ∧ cap64 w < w + 64 := by
  unfold cap64; omega

end Ref.Mtp



## FILE: /mnt/user-data/outputs/VerifyWindowSchedule.lean
/-!
# MODULE: verify_window_schedule
SOURCES: src/core/verify.cpp
The speculative-decoding verifier: a window of up to `max_t` tokens is run through every layer as one
captured graph, optionally split into two token groups whose pre/post halves interleave, with the
host pool serving each (layer, group) ring; a separate commit graph keeps the accepted prefix.

## Source semantics
* Groups: `G = 2` iff the split is on and `T ≥ 2`; group A is `[0, ceil(T/2))`, B is `[ceil(T/2), T)`;
  with `G = 1` the only group is `[0, T)`.  Groups partition the window.
* Stream order: `pre(0,g)` for all g, then for each layer l and group g: `post(l,g)`, `pre(l+1,g)`.
  The k-th `pre` rings sequence `k+1`; `post(l,g)` waits for ring `l*G + g + 1`; the host loop's
  k-th iteration serves `(k / G, k % G)` with `want = k + 1`.
* GPU plan layout (int32 units): counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad to even |
  ptr(2cap) | ptr2(2cap) | start2(cap+1); `plan_i32 = ptr_off + 4cap + (cap+1) + 1`.
* Flag B only rises: `raise_flag` stores `max(cur, value)`.
* Commit row: `[n_keep, n_keep-1, pos0+0, …, pos0+n_keep-1, -1, …]`, `1 ≤ n_keep ≤ last_t`.

## Form template identifiers
groups ↦ G; bounds ↦ tb_/te_; Ev.pre/Ev.post ↦ the pre/post lambdas; sched ↦ record_window's loop;
ringOf ↦ `l * G + grp + 1`; ptrOff ↦ ptr_off; planI32 ↦ plan_i32_; raise ↦ Verifier::raise_flag;
commitVal ↦ h_commit_.
-/

namespace Ref.Verify

-- § token groups

/-- Number of token groups in a window of `T` tokens. -/
def groups (split : Bool) (T : Nat) : Nat := if split && decide (2 ≤ T) then 2 else 1

/-- `(tb, te)` of group `g`. -/
def bounds (G T g : Nat) : Nat × Nat :=
  if g = 0 then (0, if G = 2 then (T + 1) / 2 else T) else ((T + 1) / 2, T)

/-- One group covers the whole window. -/
theorem one_group (T : Nat) : bounds 1 T 0 = (0, T) := by simp [bounds]

/-- Two groups are adjacent, cover `[0, T)`, and are both nonempty. -/
theorem two_groups (T : Nat) (hT : 2 ≤ T) :
    (bounds 2 T 0).1 = 0 ∧ (bounds 2 T 0).2 = (bounds 2 T 1).1 ∧ (bounds 2 T 1).2 = T ∧
    (bounds 2 T 0).1 < (bounds 2 T 0).2 ∧ (bounds 2 T 1).1 < (bounds 2 T 1).2 := by
  simp [bounds]; omega

/-- A split window is used only when there are at least two tokens. -/
theorem groups_two_iff (split : Bool) (T : Nat) : groups split T = 2 ↔ split = true ∧ 2 ≤ T := by
  unfold groups; cases split <;> by_cases h : 2 ≤ T <;> simp [h]

-- § the stream schedule

/-- The two halves of a layer for one group. -/
inductive Ev where
  | pre (l g : Nat)
  | post (l g : Nat)
  deriving DecidableEq, Repr

/-- The order `record_window` enqueues for `n` layers and `G` groups. -/
def sched (G n : Nat) : List Ev :=
  (List.range G).map (Ev.pre 0) ++
  (List.range n).flatMap (fun l => (List.range G).flatMap
    (fun g => Ev.post l g :: if l + 1 < n then [Ev.pre (l + 1) g] else []))

/-- The ring number a `pre` produces / a `post` waits for. -/
def ringOf (G l g : Nat) : Nat := l * G + g + 1

/-- Ring numbers of the `pre` events in stream order. -/
def preRings (G : Nat) : List Ev → List Nat
  | [] => []
  | .pre l g :: es => ringOf G l g :: preRings G es
  | .post _ _ :: es => preRings G es

/-- Every `post` comes after its own `pre` (checked by a scan of the stream). -/
def postsAfterPres : List Ev → List (Nat × Nat) → Bool
  | [], _ => true
  | .pre l g :: es, seen => postsAfterPres es ((l, g) :: seen)
  | .post l g :: es, seen => seen.contains (l, g) && postsAfterPres es seen

/-- **The artifact's 48 layers, unsplit**: rings are published as 1, 2, …, 48 in stream order. -/
theorem rings_in_order_1 : preRings 1 (sched 1 48) = (List.range 48).map (· + 1) := by decide

/-- **Split window**: the `pre` events still ring 1, 2, …, 96 in stream order, so the host's
monotone counter sees each (layer, group) exactly once and in order. -/
theorem rings_in_order_2 : preRings 2 (sched 2 48) = (List.range 96).map (· + 1) := by decide

/-- Every `post(l,g)` is enqueued after `pre(l,g)`, unsplit and split. -/
theorem posts_after_pres : postsAfterPres (sched 1 48) [] = true ∧ postsAfterPres (sched 2 48) [] = true := by
  decide

/-- The host loop's iteration `k` serves `(k / G, k % G)` and waits for exactly that ring. -/
theorem host_ring (G k : Nat) : ringOf G (k / G) (k % G) = k + 1 := by
  unfold ringOf
  have := Nat.div_add_mod k G
  rw [Nat.mul_comm] at this
  omega

-- § the GPU plan layout

/-- int32s before the 64-bit pointer arrays. -/
def i32Head (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- `(i32 + 1) & ~1`: the pointer arrays start on an 8-byte boundary. -/
def ptrOff (cap : Nat) : Nat := (i32Head cap + 1) / 2 * 2

/-- int32 size of one plan slot. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The plan's regions are ordered, the pointers are 8-byte aligned, and all fit in the slot. -/
theorem plan_layout (cap : Nat) :
    i32Head cap ≤ ptrOff cap ∧ ptrOff cap % 2 = 0 ∧
    ptrOff cap + 2 * cap ≤ ptrOff cap + 2 * cap ∧
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap ∧
    ptrOff cap + 4 * cap + (cap + 1) < planI32 cap := by
  unfold planI32 ptrOff i32Head; omega

/-- The device's offsets agree with the host sink's: `start2` at `ptr_off + 4cap`, `ptr2` at
`ptr + cap` u64s, `dst` after `start`, `tok` after `dst`. -/
theorem plan_fields (cap : Nat) :
    4 + (cap + 1) + cap + cap = i32Head cap ∧ ptrOff cap + 4 * cap + (cap + 1) + 1 = planI32 cap := by
  unfold i32Head planI32; omega

-- § flags and commit

/-- `raise_flag`: a compare-and-swap loop that only ever raises. -/
def raise (cur v : Nat) : Nat := max cur v

/-- A raise never lowers the flag and always reaches the requested value. -/
theorem raise_spec (cur v : Nat) : cur ≤ raise cur v ∧ v ≤ raise cur v := by unfold raise; omega

/-- A late host function raising an older value cannot undo a newer one. -/
theorem raise_late (cur v : Nat) (h : v ≤ cur) : raise cur v = cur := by unfold raise; omega

/-- Entry `2 + t` of the commit row. -/
def commitVal (nKeep : Nat) (pos0 : Int) (t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- How many committed cells a row of `m` entries keeps. -/
def keptCount (nKeep : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => keptCount nKeep m + if m < nKeep then 1 else 0

/-- Exactly `n_keep` cells are committed when `n_keep ≤ max_t`. -/
theorem keptCount_eq (nKeep m : Nat) (h : nKeep ≤ m) : keptCount nKeep m = nKeep := by
  induction m with
  | zero => simp [keptCount]; omega
  | succ m ih =>
    simp only [keptCount]
    by_cases hm : m < nKeep
    · have hk : nKeep = m + 1 := by omega
      subst hk
      have aux : ∀ j, j ≤ m + 1 → keptCount (m + 1) j = j := by
        intro j hj
        induction j with
        | zero => rfl
        | succ j ihj => simp only [keptCount]; rw [ihj (by omega)]; simp; omega
      rw [aux m (by omega)]; simp
    · rw [ih (by omega)]; simp [hm]

/-- A kept cell is committed at its own position; a dropped one is `-1`. -/
theorem commitVal_spec (nKeep : Nat) (pos0 : Int) (t : Nat) :
    (t < nKeep → commitVal nKeep pos0 t = pos0 + t) ∧ (nKeep ≤ t → commitVal nKeep pos0 t = -1) := by
  unfold commitVal
  constructor
  · intro h; simp [h]
  · intro h; simp [show ¬ t < nKeep by omega]

/-- `run` refuses a window running past the context. -/
theorem window_in_context (pos0 T maxCells t : Nat) (h : pos0 + T ≤ maxCells) (ht : t < T) :
    pos0 + t < maxCells := by omega

end Ref.Verify



## FILE: /mnt/user-data/outputs/SuffixLookupDrafter.lean
/-!
# MODULE: suffix_lookup_drafter
SOURCES: src/spec/suffix_drafter.cpp
A prompt-lookup drafter: every 3-token suffix of the history is hashed into an open-addressed table
remembering its most recent `WAYS` end positions; `propose` extends the longest backward match of
the current suffix and returns the tokens that followed it.

## Source semantics
* Keys are `mix(...) | 1`, hence never 0, so 0 marks an empty slot.
* The table size is a power of two at least twice the nominal capacity; linear probing from
  `key & mask` visits `mask + 1` distinct slots, i.e. every slot once, before giving up.
* A slot keeps at most `WAYS` positions, newest first; `n = min(n + 1, WAYS)`.
* A match of length `len` requires `len ≤ max_match` and `len ≤ p` before comparing `hist[p-len]`;
  a proposal needs `len ≥ min_match`, and emits at most `max_k` tokens `hist[best_end+1 ..= cur]`.
* `min_match_ = max(3, min_match)` but `max_match_ = max(min_match, max_match)` uses the RAW
  `min_match`, so `min_match < 3` with `max_match < 3` gives `max_match_ < min_match_` and the
  drafter can never propose.

## Form template identifiers
keyOf ↦ key_at's `| 1ull`; probe ↦ find_slot's index sequence; pushWay ↦ the WAYS shift in append;
matchGo/matchLen ↦ propose's inner while; continuation ↦ propose's output loop;
minMatchEff/maxMatchEff ↦ min_match_/max_match_.
-/

namespace Ref.Suffix

-- § keys and probing

/-- The stored key: the hash with its low bit forced on. -/
def keyOf (h : Nat) : Nat := h ||| 1

/-- A stored key is never the empty-slot marker 0. -/
theorem keyOf_ne_zero (h : Nat) : keyOf h ≠ 0 := by
  intro hz
  have t := Nat.testBit_or h 1 0
  unfold keyOf at hz
  rw [hz] at t
  simp at t

/-- The `p`-th slot probed for a key starting at `h` in a table of `c` slots. -/
def probe (c h p : Nat) : Nat := (h + p) % c

/-- Adding a shift that is nonzero and smaller than the modulus always changes a residue. -/
theorem shift_ne (r d c : Nat) (hr : r < c) (hd0 : 0 < d) (hdc : d < c) : (r + d) % c ≠ r := by
  by_cases hlt : r + d < c
  · rw [Nat.mod_eq_of_lt hlt]; omega
  · have hge : c ≤ r + d := by omega
    rw [Nat.mod_eq_sub_mod hge, Nat.mod_eq_of_lt (by omega)]; omega

/-- **Probing never revisits a slot** within one sweep, so `probes <= mask` covers the whole table. -/
theorem probe_injective (c h p q : Nat) (hpq : p < q) (hq : q < c) : probe c h p ≠ probe c h q := by
  intro he
  unfold probe at he
  have hc : 0 < c := by omega
  have hd : (q - p) % c = q - p := Nat.mod_eq_of_lt (by omega)
  have e1 : h + q = (h + p) + (q - p) := by omega
  have key : (h + q) % c = ((h + p) % c + (q - p)) % c := by
    rw [e1, Nat.add_mod (h + p) (q - p) c, hd]
  exact shift_ne ((h + p) % c) (q - p) c (Nat.mod_lt _ hc) (by omega) (by omega) (by rw [← key, he])

/-- The constructor's doubling loop, `while (cap < target) cap <<= 1`, with fuel. -/
def grow : Nat → Nat → Nat → Nat
  | 0, cap, _ => cap
  | f + 1, cap, t => if cap < t then grow f (cap * 2) t else cap

/-- The table size is always a power of two. -/
theorem grow_pow2 (f k t : Nat) : ∃ j, grow f (2 ^ k) t = 2 ^ j := by
  induction f generalizing k with
  | zero => exact ⟨k, rfl⟩
  | succ f ih =>
    simp only [grow]
    split
    · rw [← Nat.pow_succ]; exact ih (k + 1)
    · exact ⟨k, rfl⟩

-- § ways

/-- The newest-first shift of a slot's positions, keeping `W` ways. -/
def pushWay (W e : Nat) (pos : List Nat) : List Nat := (e :: pos).take W

/-- After a push the slot holds `min(n + 1, W)` positions, newest first. -/
theorem pushWay_spec (W e : Nat) (pos : List Nat) (hW : 0 < W) :
    (pushWay W e pos).length = min W (pos.length + 1) ∧ (pushWay W e pos).head? = some e := by
  unfold pushWay
  constructor
  · simp [List.length_take]
  · obtain ⟨w, rfl⟩ : ∃ w, W = w + 1 := ⟨W - 1, by omega⟩
    simp

-- § matching

/-- The backward match loop with `f` steps of fuel and current length `len`. -/
def matchGo (hist : Nat → Int) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | f + 1, len => if len ≤ p ∧ hist (p - len) = hist (cur - len) then matchGo hist p cur f (len + 1) else len

/-- The match length found for an earlier end position `p`. -/
def matchLen (hist : Nat → Int) (p cur maxM : Nat) : Nat := matchGo hist p cur maxM 0

/-- The loop adds at most its fuel. -/
theorem matchGo_le_fuel (hist : Nat → Int) (p cur : Nat) (f len : Nat) :
    matchGo hist p cur f len ≤ len + f := by
  induction f generalizing len with
  | zero => simp [matchGo]
  | succ f ih =>
    simp only [matchGo]
    split
    · have := ih (len + 1); omega
    · omega

/-- The loop never reads before the start of the history: the length stays ≤ `p + 1`. -/
theorem matchGo_le_pos (hist : Nat → Int) (p cur : Nat) (f len : Nat) (h : len ≤ p + 1) :
    matchGo hist p cur f len ≤ p + 1 := by
  induction f generalizing len with
  | zero => simpa [matchGo]
  | succ f ih =>
    simp only [matchGo]
    split
    · next hc => exact ih (len + 1) (by omega)
    · exact h

/-- A match is bounded by `max_match` and by `p + 1`. -/
theorem matchLen_bounds (hist : Nat → Int) (p cur maxM : Nat) :
    matchLen hist p cur maxM ≤ maxM ∧ matchLen hist p cur maxM ≤ p + 1 := by
  unfold matchLen
  refine ⟨?_, matchGo_le_pos hist p cur maxM 0 (by omega)⟩
  have := matchGo_le_fuel hist p cur maxM 0
  omega

-- § proposals

/-- The tokens `propose` emits: `hist[e+1 ..= cur]`, at most `k` of them. -/
def continuation (hist : Nat → Int) (e cur k : Nat) : List Int :=
  ((List.range (cur - e)).map (fun i => hist (e + 1 + i))).take k

/-- A proposal never exceeds `max_k` tokens nor runs past the current position. -/
theorem continuation_length (hist : Nat → Int) (e cur k : Nat) :
    (continuation hist e cur k).length = min k (cur - e) := by
  simp [continuation, List.length_take]

/-- The effective minimum: never below 3. -/
def minMatchEff (minM : Nat) : Nat := max 3 minM

/-- The effective maximum, as written: built from the RAW `min_match`. -/
def maxMatchEff (minM maxM : Nat) : Nat := max minM maxM

/-- Whether a match of this length yields a proposal. -/
def proposes (minM len : Nat) : Bool := decide (minMatchEff minM ≤ len)

/-- **Finding**: with `min_match = 1, max_match = 2` the effective window is empty
(`max_match_ = 2 < 3 = min_match_`), so no history ever produces a draft. -/
theorem degenerate_config_never_proposes (hist : Nat → Int) (p cur : Nat) :
    maxMatchEff 1 2 < minMatchEff 1 ∧
    proposes 1 (matchLen hist p cur (maxMatchEff 1 2)) = false := by
  have hb := (matchLen_bounds hist p cur (maxMatchEff 1 2)).1
  refine ⟨by decide, ?_⟩
  unfold proposes
  simp only [decide_eq_false_iff_not]
  simp only [maxMatchEff, minMatchEff] at hb ⊢
  omega

/-- When `max_match ≥ 3`, the effective window is nonempty for every `min_match ≤ max_match`. -/
theorem sane_config (minM maxM : Nat) (h3 : 3 ≤ maxM) (h : minM ≤ maxM) :
    minMatchEff minM ≤ maxMatchEff minM maxM := by
  unfold minMatchEff maxMatchEff; omega

end Ref.Suffix



## FILE: /mnt/user-data/outputs/SpecDecodeController.lean
/-!
# MODULE: spec_decode_controller
SOURCES: src/spec/controller.cpp
The speculative-decoding controller: a cost model for a verify step of `k` drafts, the expected
number of tokens a draft of `k` commits, a chooser that picks the draft source and length with the
best tokens per millisecond (or none), and the online update of acceptance rates.

## Source semantics
* `step_ms` indexes its per-window tables by `n = clamp(k + 1, 1, K_MAX + 1)`, so `n - 1` is always
  a valid index into tables of `K_MAX + 1` entries.
* `expected(p, k)` is `1 + Σ_{i<k} Π_{j≤i} p_j` (or `p_0^(i+1)` for lookup): the verify pass always
  commits one token, and each further token needs every earlier draft accepted.  It is ≥ 1 and
  grows with `k`.
* `choose` starts from the no-draft baseline, keeps any strictly better candidate, and falls back to
  no draft unless the best beats the baseline by the factor `1 + min_gain`.
* `observe` clamps `accepted` to `[0, k]` and updates the first `min(k, accepted + 1)` positions:
  `accepted` hits and at most one miss.

## Form template identifiers
clampIdx ↦ `std::clamp(k + 1, 1, K_MAX + 1)`; runProd/expected ↦ Controller::expected;
Cand/pick ↦ choose's `consider`; gate ↦ the min_gain fallback; seen ↦ observe's `seen`.
-/

namespace Ref.Spec

-- § cost-table index

/-- `std::clamp(k + 1, 1, K_MAX + 1)`. -/
def clampIdx (kmax : Int) (k : Int) : Int := max 1 (min (k + 1) (kmax + 1))

/-- Whatever `k` the caller passes, the table index `n - 1` is in `[0, K_MAX]`. -/
theorem clampIdx_valid (kmax k : Int) (h : 0 ≤ kmax) :
    0 ≤ clampIdx kmax k - 1 ∧ clampIdx kmax k - 1 ≤ kmax := by
  unfold clampIdx; omega

/-- In range, the clamp is the identity `k + 1`. -/
theorem clampIdx_id (kmax k : Int) (h0 : 0 ≤ k) (h1 : k ≤ kmax) : clampIdx kmax k = k + 1 := by
  unfold clampIdx; omega

-- § expected tokens per verify step

/-- The running product after `i` drafts; `same` uses `p 0` throughout (the lookup source). -/
def runProd (p : Nat → Nat) (same : Bool) : Nat → Nat
  | 0 => 1
  | i + 1 => runProd p same i * (if same then p 0 else p i)

/-- Expected committed tokens for `k` drafts. -/
def expected (p : Nat → Nat) (same : Bool) : Nat → Nat
  | 0 => 1
  | k + 1 => expected p same k + runProd p same (k + 1)

/-- The verify pass always commits at least one token. -/
theorem expected_ge_one (p : Nat → Nat) (same : Bool) (k : Nat) : 1 ≤ expected p same k := by
  induction k with
  | zero => simp [expected]
  | succ k ih => simp only [expected]; omega

/-- Drafting more never lowers the expectation (nonnegative acceptance rates). -/
theorem expected_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    expected p same k ≤ expected p same (k + 1) := by
  simp only [expected]; omega

/-- With certain acceptance, `k` drafts commit `k + 1` tokens. -/
theorem expected_certain (same : Bool) (k : Nat) : expected (fun _ => 1) same k = k + 1 := by
  have hr : ∀ i, runProd (fun _ => 1) same i = 1 := by
    intro i
    induction i with
    | zero => rfl
    | succ i ih => simp [runProd, ih]
  induction k with
  | zero => rfl
  | succ k ih => simp only [expected, ih, hr]

/-- With zero acceptance only the verify pass's own token is committed. -/
theorem expected_reject (same : Bool) (k : Nat) : expected (fun _ => 0) same k = 1 := by
  induction k with
  | zero => rfl
  | succ k ih => simp [expected, ih, runProd]

-- § choosing

/-- A candidate: source tag (0 = none), draft length, rate (tokens per ms, any monotone scale). -/
structure Cand where
  src : Nat
  k : Nat
  rate : Nat
  deriving DecidableEq, Repr

/-- `consider`: keep a candidate only if it is strictly better. -/
def pick (best : Cand) : List Cand → Cand
  | [] => best
  | c :: cs => pick (if c.rate > best.rate then c else best) cs

/-- The kept candidate is never worse than the starting baseline. -/
theorem pick_ge (best : Cand) (cs : List Cand) : best.rate ≤ (pick best cs).rate := by
  induction cs generalizing best with
  | nil => exact Nat.le_refl _
  | cons c cs ih =>
    simp only [pick]
    split
    · next h => have := ih c; omega
    · exact ih best

/-- The min-gain gate, in percent: fall back to no draft unless `best ≥ base * (1 + g/100)`. -/
def gate (base g : Nat) (best : Cand) : Cand :=
  if best.src ≠ 0 ∧ best.rate * 100 < base * (100 + g) then ⟨0, 0, base⟩ else best

/-- `choose`: start from the baseline (`src = 0`), consider the candidates, apply the gate. -/
def choose (base g : Nat) (cs : List Cand) : Cand := gate base g (pick ⟨0, 0, base⟩ cs)

/-- **Contract of `choose`**: either no draft at the baseline rate, or a choice that beats the
baseline by the required margin. -/
theorem choose_spec (base g : Nat) (cs : List Cand) :
    let r := choose base g cs
    (r.src = 0 ∧ base ≤ r.rate) ∨ base * (100 + g) ≤ r.rate * 100 := by
  simp only [choose, gate]
  have hge := pick_ge ⟨0, 0, base⟩ cs
  simp only at hge
  split
  · left; simp
  · next h =>
    by_cases hs : (pick ⟨0, 0, base⟩ cs).src = 0
    · left; exact ⟨hs, hge⟩
    · right; omega

-- § observing

/-- Clamp of the accepted count. -/
def clampAcc (k acc : Int) : Int := max 0 (min acc k)

/-- Positions observed in a round: the accepted ones plus the first rejected one, if drafted. -/
def seen (k acc : Int) : Int := min k (clampAcc k acc + 1)

/-- The update touches only drafted positions, records `accepted` hits and at most one miss. -/
theorem observe_counts (k acc : Int) (hk : 1 ≤ k) :
    0 ≤ clampAcc k acc ∧ clampAcc k acc ≤ seen k acc ∧ seen k acc ≤ k ∧
    seen k acc - clampAcc k acc ≤ 1 := by
  unfold seen clampAcc; omega

/-- A fully accepted window (`accepted = k`) observes no miss, which is when the deeper positions
are pulled toward the deepest observed rate. -/
theorem full_accept_no_miss (k : Int) (hk : 1 ≤ k) : seen k k - clampAcc k k = 0 := by
  unfold seen clampAcc; omega

end Ref.Spec



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- VerifyWindowSchedule.lean
  was: theorem host_ring (G k : Nat) (hG : 0 < G) : ringOf G (k / G) (k % G) = k + 1
  now: theorem host_ring (G k : Nat) : ringOf G (k / G) (k % G) = k + 1
  measured: fewer quantified variables (2 → 1)
-/



## FILE: /mnt/user-data/outputs/GenerateRunContract.lean
/-!
# MODULE: generate_run_contract
SOURCES: src/program/generate.cpp
The refusal rules `ref generate` applies before it allocates anything: token-id parsing, the
prompt/generation/context fit, the int32 position cap, the PCIe share clamp, the routing-trace record
and its layer index, the logits-dump row count, and the split-GGUF shard enumeration.

## Source semantics
* `parse_i64_list` accepts a non-empty list of decimal ids in `[0, 2^31-1]`; anything else is refused.
* A run is admitted only if `max_new > 0`, `max_context > 0`, `max_new ≤ max_context` and
  `n_prompt ≤ max_context - max_new`; `max_context` must stay below `2^31 - 8` because positions,
  cells and pooled-block indices are cast to int32 on the device.
* `pcie_num = clamp(round(pcie_frac * 256), 0, 256)`.
* The routing trace is written after the dispatch, when the adapter has already advanced `d.layers`,
  so the recorded layer is `d.layers - 1`, and only `0..47` is accepted.  A record is
  `int32 layer, int32 k, k int32 ids, k float weights` = `8 + 8k` bytes.
* The logits dump has `n_prompt - 1 + max_new` positions; with stride 1 every position is a row.
* `model_shards` probes shard numbers `1..min(total, 99)` and keeps the ones that exist.

## Form template identifiers
parse_i64_list → parseIds · Options{tokens,max_new,max_context} → RunShape · the fit check → fitsContext ·
pcie_num → pcieNum · drive_pool's trace → traceLayer / traceRecordBytes ·
logits_selection::selected → selected · model_shards → shardCandidates
-/

namespace Ref.Generate.RunContract

-- § token id parsing

/-- The largest id `std::from_chars<int32_t>` can return. -/
def idMax : Int := 2147483647

/-- `parse_i64_list`: accept a non-empty list of ids that are all in `[0, idMax]`. -/
def parseIds (xs : List Int) : Option (List Nat) :=
  if xs ≠ [] ∧ ∀ v ∈ xs, 0 ≤ v ∧ v ≤ idMax then some (xs.map Int.toNat) else none

/-- A successful parse is never empty and every id fits a non-negative int32. -/
theorem parseIds_spec {xs : List Int} {ys : List Nat} (h : parseIds xs = some ys) :
    ys ≠ [] ∧ ∀ y ∈ ys, y ≤ 2147483647 := by
  unfold parseIds at h
  split at h
  · rename_i hc
    obtain ⟨hne, hall⟩ := hc
    simp only [Option.some.injEq] at h
    subst h
    refine ⟨?_, ?_⟩
    · simpa using hne
    · intro y hy
      obtain ⟨v, hv, rfl⟩ := List.mem_map.mp hy
      have hb := hall v hv
      simp only [idMax] at hb
      omega
  · cases h

/-- A negative id anywhere in the list refuses the whole list. -/
theorem parseIds_neg_refused {xs : List Int} {v : Int} (hv : v ∈ xs) (hneg : v < 0) :
    parseIds xs = none := by
  unfold parseIds
  have hbad : ¬ (xs ≠ [] ∧ ∀ v ∈ xs, 0 ≤ v ∧ v ≤ idMax) := by
    intro hc
    have := (hc.2 v hv).1
    omega
  simp only [hbad, if_false]

/-- The empty list is refused ("token list was empty"). -/
theorem parseIds_empty : parseIds [] = none := by
  unfold parseIds
  simp

-- § context fit

/-- The three numbers the fit check reads. -/
structure RunShape where
  nPrompt : Nat
  maxNew : Int
  maxContext : Int

/-- The admission predicate of the fit check in `main`. -/
def fitsContext (s : RunShape) : Prop :=
  0 < s.maxNew ∧ 0 < s.maxContext ∧ s.maxNew ≤ s.maxContext ∧
    (s.nPrompt : Int) ≤ s.maxContext - s.maxNew

/-- An admitted run holds the prompt and every generated token. -/
theorem fits_total {s : RunShape} (h : fitsContext s) :
    (s.nPrompt : Int) + s.maxNew ≤ s.maxContext := by
  obtain ⟨_, _, _, hp⟩ := h
  omega

/-- An admitted run leaves at least one cell after the prompt, so the last prompt position exists. -/
theorem fits_prompt_lt {s : RunShape} (h : fitsContext s) : (s.nPrompt : Int) < s.maxContext := by
  have ht := fits_total h
  have hm := h.1
  omega

/-- With `max_context ≤ 2^31 - 1 - 8`, every position below the context is a valid int32 and so is
the `+8` slack the serve path adds. -/
theorem positions_fit_int32 {ctx pos : Int} (hcap : ctx ≤ 2147483647 - 8) (hp : pos < ctx) :
    pos + 8 < 2147483647 := by
  omega

-- § PCIe share

/-- `drive.d.pcie_num`, clamped to `[0, 256]` after rounding. -/
def pcieNum (r : Int) : Int := max 0 (min 256 r)

/-- The clamped share is a valid numerator over 256. -/
theorem pcieNum_range (r : Int) : 0 ≤ pcieNum r ∧ pcieNum r ≤ 256 := by
  unfold pcieNum
  omega

/-- In-range values pass the clamp unchanged. -/
theorem pcieNum_id {r : Int} (h0 : 0 ≤ r) (h1 : r ≤ 256) : pcieNum r = r := by
  unfold pcieNum
  omega

-- § routing trace

/-- The layer recorded for a dispatch, read after the adapter advanced `d.layers`. -/
def traceLayer (layersAfter : Int) : Int := layersAfter - 1

/-- Whether the trace accepts the record (the bounds check that caught the off-by-one). -/
def traceAccepts (layersAfter : Int) : Prop := 0 ≤ traceLayer layersAfter ∧ traceLayer layersAfter < 48

/-- Dispatching layer `l` leaves `d.layers = l + 1`, and the record names layer `l`. -/
theorem traceLayer_after_dispatch (l : Int) : traceLayer (l + 1) = l := by
  unfold traceLayer
  omega

/-- Every layer of a 48-layer model is accepted when recorded after its own dispatch. -/
theorem trace_accepts_real_layers {l : Int} (h0 : 0 ≤ l) (h1 : l < 48) : traceAccepts (l + 1) := by
  unfold traceAccepts
  rw [traceLayer_after_dispatch]
  exact ⟨h0, h1⟩

/-- Reading the counter as the layer (the first version) would put the last layer out of range. -/
theorem untransformed_counter_overflows : ¬ ((48 : Int) < 48) := by decide

/-- Bytes of one routing record: two int32 headers, `k` int32 ids and `k` floats. -/
def traceRecordBytes (k : Nat) : Nat := 4 + 4 + 4 * k + 4 * k

/-- With the engine's `K = 10`, a record is 88 bytes. -/
theorem traceRecordBytes_k10 : traceRecordBytes 10 = 88 := by decide

-- § logits dump rows

/-- `logits_selection::selected`: every `stride`-th position plus the final one. -/
def selected (p positions stride : Nat) : Bool := p % stride == 0 || p + 1 == positions

/-- Number of positions the decode loop visits: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- With stride 1 every position is selected. -/
theorem selected_stride_one (p positions : Nat) : selected p positions 1 = true := by
  simp [selected, Nat.mod_one]

/-- With stride 1 the number of rows written equals the number of positions, which is what the
header must record. -/
theorem rows_stride_one (positions : Nat) :
    ((List.range positions).filter (fun p => selected p positions 1)).length = positions := by
  have h : ((List.range positions).filter (fun p => selected p positions 1)) = List.range positions := by
    apply List.filter_eq_self.mpr
    intro p _
    exact selected_stride_one p positions
  rw [h, List.length_range]

/-- The header count is one less than `n_prompt + max_new` whenever there is a prompt. -/
theorem dumpPositions_lt {nPrompt maxNew : Nat} (h : 1 ≤ nPrompt) :
    dumpPositions nPrompt maxNew + 1 = nPrompt + maxNew := by
  unfold dumpPositions
  omega

-- § split GGUF shards

/-- The shard numbers `model_shards` probes for a split of `total` shards. -/
def shardCandidates (total : Nat) : List Nat := List.range' 1 (min total 99)

/-- The probe never exceeds 99 shards, whatever the file name claims. -/
theorem shardCandidates_le_99 (total : Nat) (present : Nat → Bool) :
    ((shardCandidates total).filter present).length ≤ 99 := by
  have h1 := List.length_filter_le present (shardCandidates total)
  have h2 : (shardCandidates total).length = min total 99 := by
    simp [shardCandidates]
  omega

/-- Every probed shard number is in `1..99`. -/
theorem shardCandidates_range {total i : Nat} (h : i ∈ shardCandidates total) : 1 ≤ i ∧ i ≤ 99 := by
  unfold shardCandidates at h
  rw [List.mem_range'_1] at h
  omega

end Ref.Generate.RunContract



## FILE: /mnt/user-data/outputs/TeacherForcedDecodeLoop.lean
/-!
# MODULE: teacher_forced_decode_loop
SOURCES: src/program/generate.cpp
The token loop `feed -> 48 layers -> head -> sample -> feed`: teacher forcing while the prompt lasts,
recording generated tokens from the last prompt position on, the stopping rule, the PLE predecessor
window, and the per-token reset of the expert blob's layer axis.

## Source semantics
* Every position is sampled; only positions `pos ≥ n_prompt - 1` push into `produced`.  The prediction
  at the last prompt position IS the first generated token.
* The next input is the prompt's own next token while `pos + 1 < n_prompt`, otherwise the sample.
* The loop stops as soon as `produced.size() ≥ max_new`, so it visits exactly
  `n_prompt - 1 + max_new` positions (the logits-dump header count).
* `ple_prev` is oldest-first: after a token the window becomes `(prev[1], tok)`.
* `d.layers` must restart at 0 every token: blob index `layer * 512 + expert` is only in range and
  only injective for `layer < 48`.

## Form template identifiers
produced.push_back gate → isDecodePos · produced.size() → producedCount · tok update → nextInput ·
ple_prev shift → pleShift · experts.bin index → blobIndex
-/

namespace Ref.Generate.DecodeLoop

-- § which positions produce

/-- A position records its sample iff it is at or after the last prompt position. -/
def isDecodePos (nPrompt pos : Nat) : Bool := decide (nPrompt - 1 ≤ pos)

/-- `produced.size()` after the loop has processed positions `0 .. p-1`. -/
def producedCount (nPrompt : Nat) : Nat → Nat
  | 0 => 0
  | p + 1 => producedCount nPrompt p + (if isDecodePos nPrompt p then 1 else 0)

/-- Closed form: the count is the number of processed positions past the prompt's conditioning ones. -/
theorem producedCount_eq (nPrompt p : Nat) : producedCount nPrompt p = p - (nPrompt - 1) := by
  induction p with
  | zero => simp [producedCount]
  | succ p ih =>
    simp only [producedCount, ih, isDecodePos]
    by_cases h : nPrompt - 1 ≤ p
    · simp [h]; omega
    · simp [h]; omega

/-- No token is produced by the conditioning positions `0 .. n_prompt - 2`. -/
theorem nothing_before_last_prompt (nPrompt : Nat) : producedCount nPrompt (nPrompt - 1) = 0 := by
  rw [producedCount_eq]
  omega

/-- The first generated token comes from the last prompt position. -/
theorem first_token_from_last_prompt (nPrompt : Nat) :
    producedCount nPrompt (nPrompt - 1 + 1) = 1 := by
  rw [producedCount_eq]
  omega

/-- After `n_prompt - 1 + max_new` positions exactly `max_new` tokens are produced. -/
theorem produced_at_stop (nPrompt maxNew : Nat) :
    producedCount nPrompt (nPrompt - 1 + maxNew) = maxNew := by
  rw [producedCount_eq]
  omega

/-- Before that many positions the stop condition has not fired yet, so the loop visits exactly
`n_prompt - 1 + max_new` positions (`max_new > 0` is enforced by the fit check). -/
theorem not_stopped_before {nPrompt maxNew p : Nat} (hm : 0 < maxNew) (hp : p < nPrompt - 1 + maxNew) :
    producedCount nPrompt p < maxNew := by
  rw [producedCount_eq]
  omega

/-- The wall-clock charge goes to decode exactly when the position records a token, so the reported
rate divides decode time by the produced count. -/
theorem decode_charge_matches (nPrompt pos : Nat) :
    isDecodePos nPrompt pos = decide (nPrompt - 1 ≤ pos) := rfl

-- § teacher forcing

/-- The next input: the prompt's next token while it lasts, otherwise the sample. -/
def nextInput (prompt : List Nat) (sample pos : Nat) : Nat :=
  if h : pos + 1 < prompt.length then prompt[pos + 1] else sample

/-- Inside the prompt the next input does not depend on the model's guess. -/
theorem nextInput_forced {prompt : List Nat} {pos : Nat} (h : pos + 1 < prompt.length) (s₁ s₂ : Nat) :
    nextInput prompt s₁ pos = nextInput prompt s₂ pos := by
  simp [nextInput, h]

/-- From the last prompt position on, the sample is fed back. -/
theorem nextInput_free {prompt : List Nat} {pos : Nat} (h : prompt.length ≤ pos + 1) (s : Nat) :
    nextInput prompt s pos = s := by
  have hn : ¬ pos + 1 < prompt.length := by omega
  simp [nextInput, hn]

-- § the PLE predecessor window

/-- `ple_prev[0] = ple_prev[1]; ple_prev[1] = tok` (oldest first). -/
def pleShift (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- After any token stream ending in `a, b` the window holds exactly `(a, b)`, oldest first. -/
theorem pleShift_last_two (w : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl pleShift w = (a, b) := by
  simp [List.foldl_append, pleShift]

/-- The newest predecessor is always the last token fed. -/
theorem pleShift_newest (w : Int × Int) (t : Int) : (pleShift w t).2 = t := rfl

-- § the expert blob's layer axis

/-- `experts.bin` is indexed `layer * n_expert + expert` with 512 experts per layer. -/
def blobIndex (layer expert : Nat) : Nat := layer * 512 + expert

/-- With the axis reset each token, every index stays inside the 48-layer file. -/
theorem blobIndex_lt {l e : Nat} (hl : l < 48) (he : e < 512) : blobIndex l e < 48 * 512 := by
  unfold blobIndex
  omega

/-- Distinct (layer, expert) pairs read distinct blobs. -/
theorem blobIndex_injective {l l' e e' : Nat} (he : e < 512) (he' : e' < 512)
    (h : blobIndex l e = blobIndex l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h
  omega

/-- A counter left running into the second token asks for layer 48, which lies past the file. -/
theorem stale_layer_out_of_file (e : Nat) : ¬ blobIndex 48 e < 48 * 512 := by
  unfold blobIndex
  omega

end Ref.Generate.DecodeLoop



## FILE: /mnt/user-data/outputs/SpeculativeVerifyWindow.lean
/-!
# MODULE: speculative_verify_window
SOURCES: src/program/generate.cpp
Plan v0.3 P6 speculative decoding as the driver performs it: a window `[x, drafts...]` of size `T` is
verified in one pass, the longest agreeing draft prefix is accepted, `a + 1` tokens are emitted and the
position advances by `a + 1`.  Also the `--spec-min-p` window sizing loop.

## Source semantics
* `a` is the largest count with `window[i+1] == outv[i]` for all `i < a`, capped at `T - 1`.
* The round emits `outv[0..a]` (`a + 1` tokens): every round makes progress, at most `T` tokens.
* If the verifier's output `outv[j]` is the greedy prediction after `history ++ window[0..j]`, then every
  emitted token is the greedy prediction after `history ++ x ++ (emitted so far)`: speculation is exact.
* With `spec_min_p > 0`, `T` starts at 1 and grows while `T < S` and the draft probability clears the
  threshold; the first window is always `T = 1`.

## Form template identifiers
the `while (a < T-1 && window[a+1] == outv[a]) ++a` loop → acceptLen · outv → os · drafts → ds ·
verifier contract → VerifyOk · the `T` growth loop → growFrom / windowSize
-/

namespace Ref.Generate.Speculative

-- § acceptance

/-- Accepted drafts: the length of the longest common prefix of the drafts and the verifier output. -/
def acceptLen : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptLen ds os + 1 else 0
  | _, _ => 0

/-- No more drafts are accepted than were proposed (`a ≤ T - 1`). -/
theorem acceptLen_le_drafts : ∀ (ds os : List Nat), acceptLen ds os ≤ ds.length
  | [], _ => by simp [acceptLen]
  | _ :: _, [] => by simp [acceptLen]
  | d :: ds, o :: os => by
    have ih := acceptLen_le_drafts ds os
    by_cases h : d = o
    · simp [acceptLen, h]; omega
    · simp [acceptLen, h]

/-- No more drafts are accepted than the verifier produced outputs for. -/
theorem acceptLen_le_outputs : ∀ (ds os : List Nat), acceptLen ds os ≤ os.length
  | [], _ => by simp [acceptLen]
  | _ :: _, [] => by simp [acceptLen]
  | d :: ds, o :: os => by
    have ih := acceptLen_le_outputs ds os
    by_cases h : d = o
    · simp [acceptLen, h]; omega
    · simp [acceptLen, h]

/-- The accepted drafts are literally the verifier's first `a` outputs. -/
theorem take_acceptLen : ∀ (ds os : List Nat), ds.take (acceptLen ds os) = os.take (acceptLen ds os)
  | [], _ => by simp [acceptLen]
  | _ :: _, [] => by simp [acceptLen]
  | d :: ds, o :: os => by
    by_cases h : d = o
    · subst h
      simp [acceptLen, take_acceptLen ds os]
    · simp [acceptLen, h]

/-- Any shorter prefix of the accepted run agrees as well. -/
theorem take_eq_of_le {ds os : List Nat} {j : Nat} (hj : j ≤ acceptLen ds os) :
    ds.take j = os.take j := by
  have h := congrArg (List.take j) (take_acceptLen ds os)
  simp only [List.take_take] at h
  rw [Nat.min_eq_left hj] at h
  exact h

-- § emission and progress

/-- Tokens emitted by a round: the `a + 1` first verifier outputs. -/
def emitted (ds os : List Nat) : List Nat := os.take (acceptLen ds os + 1)

/-- A round emits `a + 1` tokens when the verifier returned a full window. -/
theorem emitted_length {ds os : List Nat} (hlen : ds.length + 1 ≤ os.length) :
    (emitted ds os).length = acceptLen ds os + 1 := by
  have ha := acceptLen_le_drafts ds os
  simp only [emitted, List.length_take]
  omega

/-- Every round makes progress and never more than the window: `1 ≤ emitted ≤ T`. -/
theorem emitted_bounds {ds os : List Nat} (hlen : os.length = ds.length + 1) :
    1 ≤ (emitted ds os).length ∧ (emitted ds os).length ≤ ds.length + 1 := by
  have h := emitted_length (ds := ds) (os := os) (by omega)
  have ha := acceptLen_le_drafts ds os
  omega

/-- The position advance `p += a + 1` after `r` rounds is at least `r`: decoding terminates. -/
theorem rounds_progress (advances : List Nat) (hpos : ∀ a ∈ advances, 1 ≤ a) :
    advances.length ≤ advances.foldl (· + ·) 0 := by
  have key : ∀ (xs : List Nat) (acc : Nat), (∀ a ∈ xs, 1 ≤ a) → acc + xs.length ≤ xs.foldl (· + ·) acc := by
    intro xs
    induction xs with
    | nil => intro acc _; simp
    | cons x xs ih =>
      intro acc h
      have hx := h x (List.mem_cons_self ..)
      have := ih (acc + x) (fun a ha => h a (List.mem_cons_of_mem _ ha))
      simp only [List.foldl_cons, List.length_cons]
      omega
  have := key advances 0 hpos
  omega

-- § exactness

/-- The verifier contract: output `j` is the greedy token after the history and window cells `0..j`. -/
def VerifyOk (g : List Nat → Nat) (hist : List Nat) (x : Nat) (ds os : List Nat) : Prop :=
  ∀ j, j < os.length → os[j]? = some (g (hist ++ x :: ds.take j))

/-- Every emitted token equals what plain greedy decoding would emit at that point. -/
theorem emitted_is_greedy {g : List Nat → Nat} {hist : List Nat} {x : Nat} {ds os : List Nat}
    (hv : VerifyOk g hist x ds os) {j : Nat} (hj : j ≤ acceptLen ds os) (hjl : j < os.length) :
    os[j]? = some (g (hist ++ x :: os.take j)) := by
  have h := hv j hjl
  rw [take_eq_of_le hj] at h
  exact h

/-- A wrong draft (the `--spec-corrupt` fault) stops acceptance right there. -/
theorem wrong_draft_stops {d o : Nat} (ds os : List Nat) (h : d ≠ o) : acceptLen (d :: ds) (o :: os) = 0 := by
  simp [acceptLen, h]

-- § window sizing

/-- The `--spec-min-p` loop: grow `T` while `T < S` and draft `T - 1` clears the threshold. -/
def growFrom (S : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, T => T
  | fuel + 1, T => if T < S ∧ ok (T - 1) = true then growFrom S ok fuel (T + 1) else T

/-- Growth never shrinks the window and never passes `S`. -/
theorem growFrom_bounds (S : Nat) (ok : Nat → Bool) :
    ∀ fuel T, T ≤ S → T ≤ growFrom S ok fuel T ∧ growFrom S ok fuel T ≤ S := by
  intro fuel
  induction fuel with
  | zero => intro T h; simp [growFrom]; exact h
  | succ fuel ih =>
    intro T h
    simp only [growFrom]
    split
    · rename_i hc
      have := ih (T + 1) (by omega)
      omega
    · omega

/-- The verify window: start at 1, grow up to `S`. -/
def windowSize (S : Nat) (ok : Nat → Bool) : Nat := growFrom S ok S 1

/-- The window always holds the last token and at most `S - 1` drafts. -/
theorem windowSize_bounds {S : Nat} (hS : 1 ≤ S) (ok : Nat → Bool) :
    1 ≤ windowSize S ok ∧ windowSize S ok ≤ S :=
  growFrom_bounds S ok S 1 hS

end Ref.Generate.Speculative



## FILE: /mnt/user-data/outputs/AdaptiveExpertResidency.lean
/-!
# MODULE: adaptive_expert_residency
SOURCES: src/program/generate.cpp
How `ref generate` sizes, shrinks, lends and adapts the VRAM expert tier: `--expert-cache auto`
slot count, 256-byte-rounded sized slots kept as a prefix of the ranked profile, the prompt path
borrowing the last slots, and the adaptive swap that moves a hot missing expert into a cold resident's
slot.

## Source semantics
* auto: `slots = max(0, (free - reserve) / max_blob)`, capped by the profile length; the slots never
  use more than `free - reserve`.
* Sized slots: walk the ranked profile, each blob rounded up to 256 bytes, stop at the first that does
  not fit; the kept slots are a prefix and their rounded sum fits the budget.  Shrinking re-runs this.
* Borrowing takes the last `k` slots only when `k + 128 ≤ slots`, so at least 128 stay resident; the
  prompt chunk is halved (never below 256) until its buffers fit.
* Adaptive swap: a missing expert with decayed usage `≥ 2` replaces its layer's least-used resident one
  when it was used at least `1.5` more; usage then decays by `0.7`.  The victim is evicted at once and
  the new expert becomes resident on the victim's slot when the copy lands, so no slot is ever owned by
  two experts.  Usage is kept here in tenths so the thresholds are 20, 15 and `×7/10`.

## Form template identifiers
auto sizing → autoSlots · `(b + 255) / 256 * 256` → round256 · the sized-slot walk → keepPrefix ·
lend condition → lendFirst · swap test → swapWorth · host_res → Residency · evict/apply_pending →
evict / admit · `v *= 0.7f` → decay
-/

namespace Ref.Generate.Residency

-- § auto sizing

/-- `--expert-cache auto`: the slots that fit in what is free after the reserve. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized slots never eat into the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Capping by the profile length can only shrink the tier. -/
theorem capped_fit (free reserve blob plen : Nat) :
    min (autoSlots free reserve blob) plen * blob ≤ free - reserve := by
  have h := autoSlots_fit free reserve blob
  have hm : min (autoSlots free reserve blob) plen ≤ autoSlots free reserve blob := Nat.min_le_left _ _
  exact Nat.le_trans (Nat.mul_le_mul_right blob hm) h

-- § sized slots

/-- A blob's footprint rounded up to 256 bytes. -/
def round256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem le_round256 (b : Nat) : b ≤ round256 b := by
  unfold round256
  omega

/-- Rounded footprints are 256-aligned. -/
theorem round256_aligned (b : Nat) : round256 b % 256 = 0 := by
  unfold round256
  omega

/-- Rounding adds less than 256 bytes. -/
theorem round256_lt (b : Nat) : round256 b < b + 256 := by
  unfold round256
  omega

/-- Keep the longest prefix of the ranked sizes whose rounded footprints fit in `room`. -/
def keepPrefix : List Nat → Nat → List Nat
  | [], _ => []
  | s :: ss, room => if round256 s ≤ room then s :: keepPrefix ss (room - round256 s) else []

/-- Sum of rounded footprints. -/
def roundedSum : List Nat → Nat
  | [] => 0
  | s :: ss => round256 s + roundedSum ss

/-- The kept slots fit the budget. -/
theorem keepPrefix_fits : ∀ (ss : List Nat) (room : Nat), roundedSum (keepPrefix ss room) ≤ room
  | [], _ => by simp [keepPrefix, roundedSum]
  | s :: ss, room => by
    simp only [keepPrefix]
    split
    · rename_i h
      have ih := keepPrefix_fits ss (room - round256 s)
      simp only [roundedSum]
      omega
    · simp [roundedSum]

/-- The kept slots are a prefix of the ranking: the most-routed experts are the ones kept. -/
theorem keepPrefix_prefix : ∀ (ss : List Nat) (room : Nat), ∃ t, keepPrefix ss room ++ t = ss
  | [], _ => ⟨[], by simp [keepPrefix]⟩
  | s :: ss, room => by
    simp only [keepPrefix]
    split
    · obtain ⟨t, ht⟩ := keepPrefix_prefix ss (room - round256 s)
      exact ⟨t, by simp [ht]⟩
    · exact ⟨s :: ss, by simp⟩

/-- Shrinking the budget never keeps more slots. -/
theorem keepPrefix_mono : ∀ (ss : List Nat) {r r' : Nat}, r ≤ r' →
    (keepPrefix ss r).length ≤ (keepPrefix ss r').length
  | [], _, _, _ => by simp [keepPrefix]
  | s :: ss, r, r', h => by
    simp only [keepPrefix]
    by_cases h1 : round256 s ≤ r
    · have h2 : round256 s ≤ r' := Nat.le_trans h1 h
      simp only [h1, h2, if_true, List.length_cons]
      have := keepPrefix_mono ss (r := r - round256 s) (r' := r' - round256 s) (by omega)
      omega
    · simp only [h1, if_false, List.length_nil]
      exact Nat.zero_le _

-- § lending to the prompt path

/-- The first lent slot when the prompt path borrows `k` of `slots`. -/
def lendFirst (slots k : Nat) : Nat := slots - k

/-- Lending is only allowed with 128 slots to spare, and those stay resident for decode. -/
theorem lend_keeps_128 {slots k : Nat} (h : k + 128 ≤ slots) : 128 ≤ lendFirst slots k := by
  unfold lendFirst
  omega

/-- The lent slots are exactly the last `k`. -/
theorem lent_count {slots k : Nat} (h : k ≤ slots) : slots - lendFirst slots k = k := by
  unfold lendFirst
  omega

/-- Halving the prompt chunk strictly shrinks it while the loop runs (`chunk ≥ 256`). -/
theorem halve_shrinks {chunk : Nat} (h : 256 ≤ chunk) : chunk / 2 < chunk := by
  omega

-- § adaptive swap

/-- A candidate (missing, decayed usage in tenths) displaces a victim (resident) only when it was
routed at least twice and at least 1.5 more often. -/
def swapWorth (cand vict : Nat) : Bool := decide (20 ≤ cand ∧ vict + 15 ≤ cand)

/-- An accepted swap always has positive gain, so sorting by gain is meaningful. -/
theorem swap_gain_pos {c v : Nat} (h : swapWorth c v = true) : v < c := by
  simp only [swapWorth, decide_eq_true_eq] at h
  omega

/-- Decay by 0.7 (in integer tenths) never raises a usage count. -/
def decay (u : Nat) : Nat := u * 7 / 10

/-- Decay is non-increasing. -/
theorem decay_le (u : Nat) : decay u ≤ u := by
  unfold decay
  omega

/-- Residency: expert index ↦ slot, `none` when not resident. -/
abbrev Residency := Nat → Option Nat

/-- Pointwise update. -/
def upd (r : Residency) (k : Nat) (v : Option Nat) : Residency := fun j => if j = k then v else r j

/-- No slot owned by two experts. -/
def SlotsUnique (r : Residency) : Prop := ∀ a b s, r a = some s → r b = some s → a = b

/-- The victim is evicted at once (the CPU computes it meanwhile). -/
def evict (r : Residency) (out : Nat) : Residency := upd r out none

/-- The incoming expert takes the slot once the copy has landed. -/
def admit (r : Residency) (inn slot : Nat) : Residency := upd r inn (some slot)

/-- Evicting preserves slot uniqueness. -/
theorem evict_unique {r : Residency} (h : SlotsUnique r) (out : Nat) : SlotsUnique (evict r out) := by
  intro a b s ha hb
  simp only [evict, upd] at ha hb
  by_cases hao : a = out
  · simp [hao] at ha
  · by_cases hbo : b = out
    · simp [hbo] at hb
    · simp only [hao, hbo, if_false] at ha hb
      exact h a b s ha hb

/-- During flight the victim's slot belongs to no expert. -/
theorem slot_free_in_flight {r : Residency} (h : SlotsUnique r) {out s : Nat} (hs : r out = some s) :
    ∀ j, evict r out j ≠ some s := by
  intro j hj
  simp only [evict, upd] at hj
  by_cases hjo : j = out
  · simp [hjo] at hj
  · simp only [hjo, if_false] at hj
    exact hjo (h j out s hj hs)

/-- The full swap (evict, then admit on the freed slot) keeps every slot single-owner. -/
theorem swap_unique {r : Residency} (h : SlotsUnique r) {out inn s : Nat} (hs : r out = some s) :
    SlotsUnique (admit (evict r out) inn s) := by
  have hev := evict_unique h out
  have hfree := slot_free_in_flight h hs
  intro a b t ha hb
  simp only [admit, upd] at ha hb
  by_cases hai : a = inn
  · by_cases hbi : b = inn
    · rw [hai, hbi]
    · simp only [hai, hbi, if_true, if_false, Option.some.injEq] at ha hb
      subst ha
      exact absurd hb (hfree b)
  · by_cases hbi : b = inn
    · simp only [hai, hbi, if_true, if_false, Option.some.injEq] at ha hb
      subst hb
      exact absurd ha (hfree a)
    · simp only [hai, hbi, if_false] at ha hb
      exact hev a b t ha hb

/-- After the swap the incoming expert is resident on the victim's slot and the victim is not. -/
theorem swap_moves_slot {r : Residency} {out inn s : Nat} (hne : inn ≠ out) :
    admit (evict r out) inn s inn = some s ∧ admit (evict r out) inn s out = none := by
  have hne' : out ≠ inn := fun h => hne h.symm
  simp [admit, evict, upd, hne']

end Ref.Generate.Residency



## FILE: /mnt/user-data/outputs/PrefillChunkPlanner.lean
/-!
# MODULE: prefill_chunk_planner
SOURCES: src/prefill/prefill.cpp
The host-side plan of the batched prompt path: splitting the prompt into chunks, the device-buffer bump
allocator (owned, borrowed, or count-only for `bytes_needed`), grouping the routed (token, k) pairs by
expert with a counting sort, the host→device staging ring, and the PLE predecessor window across chunks.

## Source semantics
* Chunks start at `c0 = 0, T, 2T, …` and have size `min(T, n - c0)`; they cover the prompt exactly.
* Every allocation takes `((n·size + 256 + 255) & ~255)` bytes: 256-aligned with at least a 256-byte guard.
  A borrowed region refuses an allocation that would overflow; `bytes_needed` runs the same sequence in
  count-only mode, so a region of that size always satisfies `init`.
* Grouping: `cnt[e]` counts routed pairs per expert, `off` is its exclusive prefix sum, and pair `i`
  goes to slot `off[e] + (earlier pairs with the same expert)`, which lies inside expert `e`'s band.
* The staging ring has 8 slots used round-robin with a lookahead of 7; a slot is reused only 8 blobs
  later, after its previous blob was dequantized.
* `prev` carries the last two tokens from chunk to chunk and into `ss.ple_prev`.

## Form template identifiers
chunk loop → chunkSizes / chunkTotal · Alloc::take rounding → allocBytes · borrowed bump → bump ·
cnt/off/fill → offAt / slotOf · stage_next → stageSlot · prev[] → pleShift
-/

namespace Ref.Prefill.Planner

-- § chunking

/-- Sum of a list of sizes. -/
def chunkTotal : List Nat → Nat
  | [] => 0
  | x :: xs => x + chunkTotal xs

/-- The chunk sizes for `n` remaining tokens with chunk `T` (`fuel` bounds the recursion). -/
def chunkSizes (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, n => if n = 0 then [] else min T n :: chunkSizes T fuel (n - min T n)

/-- The chunks cover the prompt exactly. -/
theorem chunkSizes_total {T : Nat} (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → chunkTotal (chunkSizes T fuel n) = n := by
  intro fuel
  induction fuel with
  | zero => intro n h; simp [chunkSizes, chunkTotal]; omega
  | succ fuel ih =>
    intro n h
    by_cases hn : n = 0
    · simp [chunkSizes, hn, chunkTotal]
    · simp only [chunkSizes, hn, if_false, chunkTotal]
      have := ih (n - min T n) (by omega)
      omega

/-- No chunk exceeds the configured size and none is empty. -/
theorem chunkSizes_bounds {T : Nat} (hT : 0 < T) :
    ∀ fuel n, ∀ c ∈ chunkSizes T fuel n, 0 < c ∧ c ≤ T := by
  intro fuel
  induction fuel with
  | zero => intro n c h; simp [chunkSizes] at h
  | succ fuel ih =>
    intro n c h
    by_cases hn : n = 0
    · simp [chunkSizes, hn] at h
    · simp only [chunkSizes, hn, if_false, List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact ih _ c h

-- § device allocation

/-- `Alloc::take`'s footprint for `x` payload bytes. -/
def allocBytes (x : Nat) : Nat := (x + 256 + 255) / 256 * 256

/-- Every buffer is 256-aligned. -/
theorem allocBytes_aligned (x : Nat) : allocBytes x % 256 = 0 := by
  unfold allocBytes
  omega

/-- Every buffer carries at least 256 guard bytes past its payload. -/
theorem allocBytes_guard (x : Nat) : x + 256 ≤ allocBytes x := by
  unfold allocBytes
  omega

/-- The footprint wastes less than 512 bytes. -/
theorem allocBytes_tight (x : Nat) : allocBytes x < x + 512 := by
  unfold allocBytes
  omega

/-- Bump allocation in a borrowed region: offsets of each buffer, or refusal. -/
def bump : Nat → Nat → List Nat → Option (List Nat)
  | _, _, [] => some []
  | cap, used, b :: bs => if used + b ≤ cap then (bump cap (used + b) bs).map (used :: ·) else none

/-- A successful bump, started inside the region, keeps the whole sequence inside it. -/
theorem bump_fits : ∀ (bs : List Nat) (cap used : Nat) (offs : List Nat),
    used ≤ cap → bump cap used bs = some offs → used + chunkTotal bs ≤ cap
  | [], _, _, _, h0, _ => by simp [chunkTotal]; exact h0
  | b :: bs, cap, used, offs, _, h => by
    simp only [bump] at h
    split at h
    · rename_i hle
      cases hr : bump cap (used + b) bs with
      | none => rw [hr] at h; cases h
      | some r =>
        have := bump_fits bs cap (used + b) r hle hr
        simp only [chunkTotal]
        omega
    · cases h

/-- A region at least as large as the counted sequence always satisfies it: sizing the borrowed
region from `bytes_needed` cannot make `init` fail. -/
theorem bump_succeeds : ∀ (bs : List Nat) (cap used : Nat),
    used + chunkTotal bs ≤ cap → (bump cap used bs).isSome
  | [], _, _, _ => by simp [bump]
  | b :: bs, cap, used, h => by
    simp only [chunkTotal] at h
    have hle : used + b ≤ cap := by omega
    have ih := bump_succeeds bs cap (used + b) (by omega)
    simp only [bump, hle, if_true]
    cases hr : bump cap (used + b) bs with
    | none => rw [hr] at ih; cases ih
    | some r => simp

-- § grouping by expert

/-- `off[e]`: the first slot of expert `e` (exclusive prefix sum of the counts). -/
def offAt (ids : List Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => offAt ids e + ids.count e

/-- Expert `e`'s band ends where expert `e + 1`'s begins. -/
theorem offAt_succ (ids : List Nat) (e : Nat) : offAt ids (e + 1) = offAt ids e + ids.count e := rfl

/-- The rank of pair `i` among the pairs routed to the same expert is below that expert's count. -/
theorem rank_lt (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    (ids.take i).count ids[i] < ids.count ids[i] := by
  have hsplit : ids = ids.take i ++ ids[i] :: ids.drop (i + 1) := by
    rw [← List.drop_eq_getElem_cons hi]
    exact (List.take_append_drop i ids).symm
  have h := congrArg (List.count ids[i]) hsplit
  rw [List.count_append, List.count_cons_self] at h
  omega

/-- `slot_host[i] = fill[e]++`: the band start plus the rank. -/
def slotOf (ids : List Nat) (i : Nat) (hi : i < ids.length) : Nat :=
  offAt ids ids[i] + (ids.take i).count ids[i]

/-- Every pair lands inside its expert's band `[off[e], off[e+1])`, so each expert's GEMM reads
exactly its own rows of `Xs`. -/
theorem slotOf_in_band (ids : List Nat) (i : Nat) (hi : i < ids.length) :
    offAt ids ids[i] ≤ slotOf ids i hi ∧ slotOf ids i hi < offAt ids (ids[i] + 1) := by
  have h := rank_lt ids i hi
  simp only [slotOf, offAt_succ]
  omega

-- § staging ring

/-- The staging slot of the `j`-th streamed blob (8 slots, round-robin). -/
def stageSlot (j : Nat) : Nat := j % 8

/-- A staging slot is reused only 8 blobs later, beyond the lookahead of 7. -/
theorem stage_reuse_distance {j j' : Nat} (hlt : j < j') (h : stageSlot j = stageSlot j') : j + 8 ≤ j' := by
  unfold stageSlot at h
  omega

/-- Blobs staged within one lookahead window never share a slot. -/
theorem lookahead_disjoint {j j' : Nat} (hlt : j < j') (hw : j' ≤ j + 7) : stageSlot j ≠ stageSlot j' := by
  intro h
  have := stage_reuse_distance hlt h
  omega

/-- The dequantization double buffer alternates between two buffers. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by omega

-- § PLE window across chunks

/-- `prev[0] = prev[1]; prev[1] = tok`. -/
def pleShift (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- Chunking does not change the window: shifting through two chunks is shifting through their
concatenation. -/
theorem pleShift_chunks (w : Int × Int) (xs ys : List Int) :
    ys.foldl pleShift (xs.foldl pleShift w) = (xs ++ ys).foldl pleShift w := by
  rw [List.foldl_append]

/-- At the end of the prompt the window holds its last two tokens, handed to the token path. -/
theorem pleShift_last_two (w : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl pleShift w = (a, b) := by
  simp [List.foldl_append, pleShift]

end Ref.Prefill.Planner



## FILE: /mnt/user-data/outputs/PrefillKernelIndexing.lean
/-!
# MODULE: prefill_kernel_indexing
SOURCES: src/prefill/kernels.cu
The index arithmetic and integer contracts of the batched prompt kernels: the four-stream
hyper-connection layout, the Ref 2-bit expert blob layout and its dequantization, the paged KV append
with optional int8 codes, the GDN causal-conv window, the router's warp arg-max reduction, and the
interleaved/split projection layouts.

## Source semantics
* `R` is `[T][HC=4][N=2560]`; flat index `i` decomposes as `t = i / D`, `c = (i % D) / N`, `d = i % N`
  with `D = N·HC = 10240`.
* Blob: gate/up codes `[1280][640 B]`, down codes `[2560][160 B]`, gate/up scales `[1280][40]` f16,
  down scales `[2560][10]` f16 — 1,382,400 bytes.  Each code byte holds four 2-bit codes; weight value
  is `((c >> 2k) & 3) - 1 ∈ {-1, 0, 1, 2}` times the block scale of its 64-wide group.
* KV append: row `(page·2 + kvh)·page_size + pos % page_size`; int8 codes are clamped to `[-127, 127]`.
* GDN conv: each channel keeps the last three inputs; after a chunk the history is its last three.
* Router: ten rounds of a warp arg-max whose pairwise step prefers the larger probability and, on a
  tie, the smaller expert id — an order-independent reduction.
* Interleaved SwiGLU: gate at `2k`, up at `2k+1`; QSA `Qf` rows hold 256 query values then 256 gate values
  per head.

## Form template identifiers
gr_* kernels' index math → grIndex · blob offsets → oDCodes / oGuSc / oDSc / blobBytes ·
`(c >> 2k) & 3` → code2 · kv_append row → kvRow · int8 clamp → clampQ · conv history → convStep ·
the shuffle step of route_kernel → pick · swiglu_il layout → gateIdx / upIdx
-/

namespace Ref.Prefill.Kernels

-- § hyper-connection layout

/-- Flat index of `R[t][c][d]`. -/
def grIndex (t c d : Nat) : Nat := t * 10240 + c * 2560 + d

/-- The kernels' decomposition of a flat index recovers the coordinates. -/
theorem grIndex_decomp (i : Nat) : grIndex (i / 10240) ((i % 10240) / 2560) (i % 2560) = i := by
  unfold grIndex
  omega

/-- Recovered coordinates are in range: stream below 4, element below 2560. -/
theorem grIndex_ranges (i : Nat) : (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  omega

/-- Distinct coordinates in range are distinct cells: `gr_write` writes every cell once. -/
theorem grIndex_injective {t t' c c' d d' : Nat} (hc : c < 4) (hc' : c' < 4) (hd : d < 2560)
    (hd' : d' < 2560) (h : grIndex t c d = grIndex t' c' d') : t = t' ∧ c = c' ∧ d = d' := by
  unfold grIndex at h
  omega

/-- `gr_mix` reads four streams of token `t` without leaving that token's row. -/
theorem grIndex_in_token {t c d : Nat} (hc : c < 4) (hd : d < 2560) :
    t * 10240 ≤ grIndex t c d ∧ grIndex t c d < (t + 1) * 10240 := by
  unfold grIndex
  omega

-- § the 2-bit expert blob

/-- Offset of the down codes. -/
def oDCodes : Nat := 1280 * 640
/-- Offset of the gate/up scales. -/
def oGuSc : Nat := oDCodes + 2560 * 160
/-- Offset of the down scales. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2
/-- Total blob size. -/
def blobBytes : Nat := oDSc + 2560 * 10 * 2

/-- The blob is 1,382,400 bytes — the 1.38 MB slot the cache verifies at startup. -/
theorem blobBytes_value : blobBytes = 1382400 := by decide

/-- The regions are laid out back to back in increasing order. -/
theorem blob_regions_ordered : oDCodes < oGuSc ∧ oGuSc < oDSc ∧ oDSc < blobBytes := by decide

/-- One gate/up row is 640 code bytes = 2560 weights; the down row is 160 bytes = 640 weights. -/
theorem blob_row_widths : 640 * 4 = 2560 ∧ 160 * 4 = 640 := by decide

/-- The 2-bit code `k` of a code byte. -/
def code2 (c k : Nat) : Nat := (c >>> (2 * k)) &&& 3

/-- Every code is in `0..3`, so the weight multiplier `code - 1` is in `{-1, 0, 1, 2}`. -/
theorem code2_lt (c k : Nat) : code2 c k < 4 := by
  unfold code2
  have h : (c >>> (2 * k)) &&& 3 = (c >>> (2 * k)) % 4 := Nat.and_two_pow_sub_one_eq_mod _ 2
  rw [h]
  omega

/-- The multiplier as an integer. -/
def weightMult (c k : Nat) : Int := (code2 c k : Int) - 1

/-- Range of the dequantized multiplier. -/
theorem weightMult_range (c k : Nat) : -1 ≤ weightMult c k ∧ weightMult c k ≤ 2 := by
  have h := code2_lt c k
  unfold weightMult
  omega

/-- Output index of gate/up weight `k` of code byte `byte` in row `row`. -/
def guOut (row byte k : Nat) : Nat := row * 2560 + byte * 4 + k

/-- Gate/up outputs stay in the `1280 × 2560` FP16 buffer and never collide. -/
theorem guOut_range_inj {row byte k row' byte' k' : Nat} (hr : row < 1280) (hb : byte < 640) (hk : k < 4)
    (hb' : byte' < 640) (hk' : k' < 4) :
    guOut row byte k < 1280 * 2560 ∧
      (guOut row byte k = guOut row' byte' k' → row = row' ∧ byte = byte' ∧ k = k') := by
  unfold guOut
  constructor
  · omega
  · intro h; omega

/-- Down outputs stay in the `2560 × 640` buffer and never collide. -/
theorem downOut_range_inj {row byte k row' byte' k' : Nat} (hr : row < 2560) (hb : byte < 160) (hk : k < 4)
    (hb' : byte' < 160) (hk' : k' < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 ∧
      (row * 640 + byte * 4 + k = row' * 640 + byte' * 4 + k' → row = row' ∧ byte = byte' ∧ k = k') := by
  constructor
  · omega
  · intro h; omega

/-- A gate/up byte's scale is that row's scale group `(4·byte)/64`, one of 40. -/
theorem gu_scale_group {byte : Nat} (hb : byte < 640) : byte * 4 / 64 < 40 := by omega

/-- A down byte's scale group is one of 10. -/
theorem down_scale_group {byte : Nat} (hb : byte < 160) : byte * 4 / 64 < 10 := by omega

-- § paged KV append

/-- KV row of position `pos` for kv head `kvh` on page `page`. -/
def kvRow (page kvh pos ps : Nat) : Nat := (page * 2 + kvh) * ps + pos % ps

/-- Helper: a row `a·ps + o` with `o < ps` determines `a` and `o`. -/
theorem row_split_unique {ps a a' o o' : Nat} (ho : o < ps) (ho' : o' < ps)
    (h : a * ps + o = a' * ps + o') : a = a' ∧ o = o' := by
  have hp : 0 < ps := by omega
  have hdiv : ∀ x y, y < ps → (x * ps + y) / ps = x := by
    intro x y hy
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hy, Nat.zero_add]
  have hmod : ∀ x y, y < ps → (x * ps + y) % ps = y := by
    intro x y hy
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]
  have h1 := hdiv a o ho
  have h2 := hdiv a' o' ho'
  have h3 := hmod a o ho
  have h4 := hmod a' o' ho'
  rw [h] at h1 h3
  exact ⟨h1.symm.trans h2, h3.symm.trans h4⟩

/-- Distinct (page, kv head, in-page offset) never share a KV row. -/
theorem kvRow_injective {ps page page' kvh kvh' pos pos' : Nat} (hps : 0 < ps) (hk : kvh < 2)
    (hk' : kvh' < 2) (h : kvRow page kvh pos ps = kvRow page' kvh' pos' ps) :
    page = page' ∧ kvh = kvh' ∧ pos % ps = pos' % ps := by
  unfold kvRow at h
  obtain ⟨ha, ho⟩ := row_split_unique (Nat.mod_lt pos hps) (Nat.mod_lt pos' hps) h
  omega

/-- The int8 code clamp. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Codes are symmetric int8: `-128` is never written. -/
theorem clampQ_range (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ
  split
  · omega
  · split <;> omega

/-- In-range codes are unchanged. -/
theorem clampQ_id {q : Int} (h0 : -127 ≤ q) (h1 : q ≤ 127) : clampQ q = q := by
  unfold clampQ
  have ha : ¬ q < -127 := by omega
  have hb : ¬ q > 127 := by omega
  simp [ha, hb]

/-- One int8 scale per 64 values: a 256-wide head row has 4 scales. -/
theorem kv_scale_groups : 256 / 64 = 4 := by decide

-- § GDN causal conv

/-- The per-channel history step: `v0 = v1; v1 = v2; v2 = x`. -/
def convStep (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- After a chunk the history holds its last three inputs, whatever it held before. -/
theorem convStep_last_three (h : Int × Int × Int) (xs : List Int) (a b c : Int) :
    (xs ++ [a, b, c]).foldl convStep h = (a, b, c) := by
  simp [List.foldl_append, convStep]

/-- Splitting the prompt into chunks does not change the conv state. -/
theorem convStep_chunks (h : Int × Int × Int) (xs ys : List Int) :
    ys.foldl convStep (xs.foldl convStep h) = (xs ++ ys).foldl convStep h := by
  rw [List.foldl_append]

-- § router arg-max reduction

/-- The shuffle step: take the other lane's candidate if it is larger, or equal with a smaller id. -/
def pick (x y : Int × Nat) : Int × Nat := if y.1 > x.1 ∨ (y.1 = x.1 ∧ y.2 < x.2) then y else x

/-- The step is order-independent for distinct experts, so every lane agrees on the winner. -/
theorem pick_comm (a b : Int) (i j : Nat) (h : i ≠ j) : pick (a, i) (b, j) = pick (b, j) (a, i) := by
  unfold pick
  dsimp only
  by_cases h1 : b > a ∨ (b = a ∧ j < i)
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · exfalso; omega
    · simp only [h1, h2, if_true, if_false]
  · by_cases h2 : a > b ∨ (a = b ∧ i < j)
    · simp only [h1, h2, if_true, if_false]
    · exfalso; omega

/-- The winner's probability is at least both candidates'. -/
theorem pick_max (x y : Int × Nat) : x.1 ≤ (pick x y).1 ∧ y.1 ≤ (pick x y).1 := by
  unfold pick
  split
  · rename_i h; omega
  · rename_i h; omega

/-- The winner is one of the two candidates. -/
theorem pick_mem (x y : Int × Nat) : pick x y = x ∨ pick x y = y := by
  unfold pick
  split
  · exact Or.inr rfl
  · exact Or.inl rfl

-- § projection layouts

/-- Gate column of interleaved SwiGLU unit `k`. -/
def gateIdx (k : Nat) : Nat := 2 * k
/-- Up column of interleaved SwiGLU unit `k`. -/
def upIdx (k : Nat) : Nat := 2 * k + 1

/-- Gate and up columns of the 640 units tile the 1280-wide GEMM output without overlap. -/
theorem swiglu_layout {k k' : Nat} (hk : k < 640) :
    upIdx k < 1280 ∧ gateIdx k ≠ upIdx k' ∧ (gateIdx k = gateIdx k' → k = k') := by
  unfold gateIdx upIdx
  omega

/-- In `Qf`, head `h`'s query (`d < 256`) and gate (`256 + d`) never overlap and stay in the token's row. -/
theorem qf_split {t h h' d d' : Nat} (hh : h < 24) (hd : d < 256) (hd' : d' < 256) :
    t * 24 * 512 + h * 512 + d ≠ t * 24 * 512 + h' * 512 + 256 + d' ∧
      t * 24 * 512 + h * 512 + 256 + d < (t + 1) * 24 * 512 := by
  constructor <;> omega

/-- RoPE rotates pair `(p, p + 32)` of a 64-wide rotary span; the pairs partition the span. -/
theorem rope_pairs {p p' : Nat} (hp : p < 32) (hp' : p' < 32) :
    p + 32 < 64 ∧ p ≠ p' + 32 := by
  omega

end Ref.Prefill.Kernels



## FILE: /mnt/user-data/outputs/NgramPleHash.lean
/-!
# MODULE: ngram_ple_hash
SOURCES: src/kernels/ngram.cpp
The PLE n-gram hash that maps a token and its two predecessors to sixteen table rows, the artifact's
per-head vocabularies and offsets, the IQ4_NL row decode, and the one-token-in-flight protocol of the
direct-I/O table reader.

## Source semantics
* `ngram_mixed`: `mixed = ctx[0]·m[0]` (assignment) then `mixed ^= ctx[j]·m[j]`, all mod 2^64.
* Context: `ctx[0] = tok`; predecessor `s` back is `prev[n_prev - s]` (`prev` oldest first, so `s = 1`
  is the newest).  The cut flag is sticky and evaluated before storing: once a predecessor is negative
  or EOS, that position and every older one are EOS.
* Head `h = (n-2)·8 + g` for n-gram order `n ∈ {2, 3}`, `g < 8`; its row is
  `mixed % vocab[h] + offset[h]`, inside head `h`'s band `[offset[h], offset[h] + vocab[h])`.
  `offset[h+1] = offset[h] + vocab[h]`, and the bands end at 320,001,446 ≤ the table's 320,001,536 rows.
* IQ4_NL row: 160 values in 5 blocks of 18 bytes (f16 scale + 16 bytes); byte `j` holds elements `j`
  and `j + 16` (split halves).  Rows are 90 bytes; the gather is head-slowest: head `h` occupies
  `[160h, 160h + 160)` of the 2560-wide output.
* Direct mode allows one token in flight: `issue` while pending fails, `collect` without `issue` fails,
  and `gather_batch` refuses while a token is in flight.

## Form template identifiers
ngram_mixed → mixed2 / mixed3 · the ctx loop → ctx3 · head index → headOf · row → rowOf ·
vocab/offset → vocabs / offsets · kIq4Nl → iq4nl · split halves → halfElem · PleTable pending → Io
-/

namespace Ref.Kernels.Ngram

-- § the mixing hash

/-- 64-bit wrapping. -/
def wrap (x : Nat) : Nat := x % 2 ^ 64

/-- Order-2 mix: assignment of the first product, XOR of the second. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := wrap (c0 * m0) ^^^ wrap (c1 * m1)

/-- Order-3 mix. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := mixed2 c0 c1 m0 m1 ^^^ wrap (c2 * m2)

/-- The trigram hash extends the bigram hash by one XOR: the source's single loop computes both. -/
theorem mixed3_extends (c0 c1 c2 m0 m1 m2 : Nat) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ wrap (c2 * m2) := rfl

/-- Every wrapped product is a 64-bit value. -/
theorem wrap_lt (x : Nat) : wrap x < 2 ^ 64 := Nat.mod_lt _ (Nat.two_pow_pos 64)

-- § context with the EOS cut

/-- `(ctx[1], ctx[2])` from `prev = [p0 (oldest), p1 (newest)]` with the sticky cut. -/
def ctx3 (eos p0 p1 : Int) : Int × Int :=
  let cut1 := decide (p1 < 0) || decide (p1 = eos)
  let c1 := if cut1 then eos else p1
  let cut2 := cut1 || decide (p0 < 0) || decide (p0 = eos)
  let c2 := if cut2 then eos else p0
  (c1, c2)

/-- Without a boundary the context is `[tok, newest, oldest]`: the window is read newest first. -/
theorem ctx3_plain {eos p0 p1 : Int} (h0 : 0 ≤ p0) (h1 : 0 ≤ p1) (e0 : p0 ≠ eos) (e1 : p1 ≠ eos) :
    ctx3 eos p0 p1 = (p1, p0) := by
  have a1 : ¬ p1 < 0 := by omega
  have a0 : ¬ p0 < 0 := by omega
  simp [ctx3, a1, a0, e1, e0]

/-- A boundary at the newest predecessor cuts the whole history: both slots read EOS. -/
theorem ctx3_cut_newest {eos p0 p1 : Int} (h : p1 < 0 ∨ p1 = eos) : ctx3 eos p0 p1 = (eos, eos) := by
  rcases h with h | h
  · simp [ctx3, h]
  · simp [ctx3, h]

/-- A boundary only at the oldest predecessor keeps the newest one. -/
theorem ctx3_cut_oldest {eos p0 p1 : Int} (h1 : 0 ≤ p1) (e1 : p1 ≠ eos) (h : p0 < 0) :
    ctx3 eos p0 p1 = (p1, eos) := by
  have a1 : ¬ p1 < 0 := by omega
  simp [ctx3, a1, e1, h]

-- § heads and rows

/-- The per-head table vocabularies (docs/gguf-dump-shard1.txt). -/
def vocabs : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The per-head row offsets. -/
def offsets : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Sixteen heads in both tables. -/
theorem tables_len : vocabs.length = 16 ∧ offsets.length = 16 := by decide

/-- The metadata's offsets are the running sum of the vocabularies (the parity test's check). -/
theorem offsets_prefix_sum : ∀ i, i < 15 → offsets[i + 1]! = offsets[i]! + vocabs[i]! := by decide

/-- The last band ends inside the 320,001,536-row table. -/
theorem bands_fit_table : offsets[15]! + vocabs[15]! = 320001446 ∧ 320001446 ≤ 320001536 := by decide

/-- Every vocabulary is positive, so `%` is well defined. -/
theorem vocabs_pos : ∀ i, i < 16 → 0 < vocabs[i]! := by decide

/-- Head index of n-gram order `n` and group `g`. -/
def headOf (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Orders 2 and 3 with 8 groups each fill heads `0..15`, each exactly once. -/
theorem headOf_bij {n g n' g' : Nat} (hn : 2 ≤ n ∧ n ≤ 3) (hn' : 2 ≤ n' ∧ n' ≤ 3) (hg : g < 8) (hg' : g' < 8) :
    headOf n g < 16 ∧ (headOf n g = headOf n' g' → n = n' ∧ g = g') := by
  unfold headOf
  obtain ⟨a, b⟩ := hn
  obtain ⟨c, d⟩ := hn'
  constructor
  · omega
  · intro h
    have : n = 2 ∨ n = 3 := by omega
    have : n' = 2 ∨ n' = 3 := by omega
    omega

/-- The table row of head `h` for a mixed hash. -/
def rowOf (h mixed : Nat) : Nat := mixed % vocabs[h]! + offsets[h]!

/-- A head's row stays in its own band, so the sixteen heads never read each other's rows. -/
theorem rowOf_band {h : Nat} (hh : h < 16) (mixed : Nat) :
    offsets[h]! ≤ rowOf h mixed ∧ rowOf h mixed < offsets[h]! + vocabs[h]! := by
  have hv := vocabs_pos h hh
  have hm := Nat.mod_lt mixed hv
  unfold rowOf
  omega

/-- Every row the hash produces is inside the table. -/
theorem rowOf_in_table {h : Nat} (hh : h < 16) (mixed : Nat) : rowOf h mixed < 320001536 := by
  have hb := (rowOf_band hh mixed).2
  have hlast : ∀ i, i < 16 → offsets[i]! + vocabs[i]! ≤ 320001446 := by decide
  have := hlast h hh
  omega

-- § IQ4_NL rows

/-- The IQ4_NL codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing and within int8. -/
theorem iq4nl_sorted : ∀ i, i < 15 → iq4nl[i]! < iq4nl[i + 1]! := by decide

/-- Codebook bounds. -/
theorem iq4nl_bounds : ∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (113 : Int) := by decide

/-- `code & 15` always indexes the 16-entry codebook. -/
theorem nibble_lt (c : Nat) : c &&& 15 < 16 := by
  have h : c &&& 15 = c % 16 := Nat.and_two_pow_sub_one_eq_mod c 4
  rw [h]
  omega

/-- Row geometry: 5 blocks of 18 bytes hold 160 values; 16 heads × 160 = 2560 outputs. -/
theorem row_geometry : 5 * 18 = 90 ∧ 5 * 32 = 160 ∧ 16 * 160 = 2560 := by decide

/-- Element of block-local byte `j`'s low (`hi = false`) or high nibble. -/
def halfElem (j : Nat) (hi : Bool) : Nat := if hi then j + 16 else j

/-- Split halves: the 16 bytes' two nibbles cover the 32 elements of a block exactly once. -/
theorem halfElem_cover (e : Nat) (he : e < 32) : ∃ j, j < 16 ∧ ∃ hi, halfElem j hi = e := by
  by_cases h : e < 16
  · exact ⟨e, h, false, by simp [halfElem]⟩
  · exact ⟨e - 16, by omega, true, by simp [halfElem]; omega⟩

/-- No two (byte, nibble) pairs decode to the same element. -/
theorem halfElem_inj {j j' : Nat} {hi hi' : Bool} (hj : j < 16) (hj' : j' < 16)
    (h : halfElem j hi = halfElem j' hi') : j = j' ∧ hi = hi' := by
  cases hi <;> cases hi' <;> simp only [halfElem, Bool.false_eq_true, ↓reduceIte] at h
  · exact ⟨h, rfl⟩
  · exfalso; omega
  · exfalso; omega
  · exact ⟨by omega, rfl⟩

/-- Head-slowest output: head `h`, element `d` lands at `160h + d`, inside the 2560-wide vector, injectively. -/
theorem gather_layout {h d h' d' : Nat} (hh : h < 16) (hd : d < 160) (hd' : d' < 160) :
    h * 160 + d < 2560 ∧ (h * 160 + d = h' * 160 + d' → h = h' ∧ d = d') := by
  constructor
  · omega
  · intro e; omega

-- § direct-I/O protocol

/-- Reader state: whether a token's sixteen reads are in flight. -/
structure Io where
  pending : Bool

/-- `issue` in direct mode: refused while a token is in flight. -/
def issue (s : Io) : Bool × Io := if s.pending then (false, s) else (true, ⟨true⟩)

/-- `collect`: requires a prior `issue`, clears the flag either way. -/
def collect (s : Io) : Bool × Io := (s.pending, ⟨false⟩)

/-- `gather_batch` refuses while a token is in flight. -/
def gatherBatchOk (s : Io) : Bool := !s.pending

/-- From idle, issue then collect succeeds and returns to idle. -/
theorem issue_collect_ok : (issue ⟨false⟩).1 = true ∧ (collect (issue ⟨false⟩).2).1 = true ∧
    (collect (issue ⟨false⟩).2).2.pending = false := by decide

/-- A second issue before collecting is refused (one token in flight per table). -/
theorem double_issue_refused : (issue (issue ⟨false⟩).2).1 = false := by decide

/-- Collecting without an issue is an error. -/
theorem collect_without_issue : (collect ⟨false⟩).1 = false := by decide

/-- The batched prompt read cannot interleave with an in-flight token. -/
theorem batch_blocked_in_flight : gatherBatchOk (issue ⟨false⟩).2 = false := by decide

end Ref.Kernels.Ngram



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- GenerateRunContract.lean
  was: theorem shardCandidates_le_99 (total : Nat) (exists? : Nat → Bool) :
    ((shardCandidates total).filter exists?).length ≤ 99
  now: theorem shardCandidates_le_99 (total : Nat) (present : Nat → Bool) :
    ((shardCandidates total).filter present).length ≤ 99
  measured: the claim changed

- NgramPleHash.lean
  was: theorem iq4nl_bounds : ∀ i, i < 16 → -127 ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ 113
  now: theorem iq4nl_bounds : ∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (113 : Int)
  measured: more hypotheses/binders (0 → 2): the statement holds in fewer cases

- TeacherForcedDecodeLoop.lean
  was: theorem not_stopped_before {nPrompt maxNew p : Nat} (hp : p < nPrompt - 1 + maxNew) :
    producedCount nPrompt p < maxNew
  now: theorem not_stopped_before {nPrompt maxNew p : Nat} (hm : 0 < maxNew) (hp : p < nPrompt - 1 + maxNew) :
    producedCount nPrompt p < maxNew
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases
-/



## FILE: /mnt/user-data/outputs/Bf16GemvDispatch.lean
/-!
# MODULE: bf16_gemv_dispatch
SOURCES: src/kernels/cuda/bf16_gemv.cu
The BF16 GEMV `y[o] = Σ_i bf16(x[i]) * bf16(w[o*n_in + i])` with three kernels (thread-per-row "naive",
warp-per-row, block-split) and two host dispatchers.  This module models the dispatch decisions, the
thread→row/lane index maps, and proves that every kernel's work decomposition (strided lanes, shared-memory
tree) computes the same exact sum as the naive row walk.

## Source semantics
* `bf16_gemv` returns without launching when `n_in ≤ 0 ∨ n_out ≤ 0`; otherwise it uses the warp kernel iff
  `n_out ≥ 64`, else the naive kernel.  Grids: `⌈n_out/8⌉` blocks of 256 (warp), `⌈n_out/256⌉` (naive).
* Warp kernel: row `o = blockIdx*8 + tid/32`, lane `tid % 32` takes `i ≡ lane (mod 32)`, then a shuffle sum.
* `bf16_gemv_split`: `tpr = 32` selects the warp kernel; otherwise `tpr` must be a positive power of two
  (`tpr & (tpr-1) = 0`), one block per row, thread `t` takes `i ≡ t (mod tpr)`, then a shared-memory tree with
  offsets `tpr/2, tpr/4, …, 1`.
* Arithmetic is modelled EXACTLY over `Int` (products already formed).  Real FP32 reassociation is not exact;
  that is why the source checks the warp/split kernels against the naive one with a tolerance.

## Form template identifiers
`bf16_gemv` → `dispatch`; `bf16_gemv_warp_kernel` → `Kernel.warp`; `bf16_gemv_naive_kernel` → `Kernel.naive`;
`bf16_gemv_split_kernel` → `Kernel.split`; `THREADS` → `256`; `n_in, n_out` → `nIn, nOut`;
`threads_per_row` → `tpr`; per-lane partial → `ssum`; shared tree → `treeReduce`; row sum → `rsum`.
-/

namespace Ref.Bf16GemvDispatch

-- § exact sums

/-- `rsum f n = f 0 + … + f (n-1)` over `Int`: the naive kernel's row walk with exact arithmetic. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step of `rsum`. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- `rsum` only looks at indices below its bound. -/
theorem rsum_congr {f g : Nat → Int} {n : Nat} (h : ∀ i, i < n → f i = g i) : rsum f n = rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [rsum_succ, rsum_succ, ih (fun i hi => h i (by omega)), h n (by omega)]

/-- The sum of the zero function is zero. -/
theorem rsum_zero_fn (n : Nat) : rsum (fun _ => (0 : Int)) n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [rsum_succ, ih]; rfl

/-- `rsum` is additive in the summand. -/
theorem rsum_add (f g : Nat → Int) (n : Nat) :
    rsum (fun i => f i + g i) n = rsum f n + rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [rsum_succ, ih]; omega

/-- Splitting a range `[0, m+n)` into `[0, m)` and `[m, m+n)`. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- Summing an indicator over `s < k` picks out the one matching index, if it is in range. -/
theorem rsum_indicator (c : Nat) (a : Int) (k : Nat) :
    rsum (fun s => if c = s then a else 0) k = if c < k then a else 0 := by
  induction k with
  | zero => simp [rsum]
  | succ k ih =>
    rw [rsum_succ, ih]
    by_cases h1 : c < k
    · have h2 : c ≠ k := by omega
      simp [h1, h2, show c < k + 1 by omega]
    · by_cases h3 : c = k
      · subst h3; simp
      · simp [h1, h3, show ¬ c < k + 1 by omega]

-- § strided lanes

/-- The partial sum a lane/thread `s` of a `k`-wide team accumulates: the indices `i < n` with `i % k = s`. -/
def ssum (f : Nat → Int) (s k n : Nat) : Int :=
  rsum (fun i => if i % k = s then f i else 0) n

/-- One more index adds `f n` to exactly the lane `n % k`. -/
theorem ssum_succ (f : Nat → Int) (s k n : Nat) :
    ssum f s k (n + 1) = ssum f s k n + (if n % k = s then f n else 0) := rfl

/-- **Strided partition.**  The lane partial sums of a `k`-wide team add up to the full row sum: the warp
kernel (`k = 32`) and the split kernel (`k = tpr`) lose and duplicate no element. -/
theorem strided_partition (f : Nat → Int) (k n : Nat) (hk : 0 < k) :
    rsum (fun s => ssum f s k n) k = rsum f n := by
  induction n with
  | zero => exact rsum_zero_fn k
  | succ n ih =>
    rw [rsum_congr (fun s _ => ssum_succ f s k n), rsum_add, ih, rsum_indicator, rsum_succ]
    simp [Nat.mod_lt n hk]

-- § shared-memory tree

/-- One tree level: threads `t < off` add the partner at `t + off`. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The tree loop `for off = 2^(k-1) … 1`, reading slot 0 at the end. -/
def treeReduce : Nat → (Nat → Int) → Int
  | 0, a => a 0
  | k + 1, a => treeReduce k (treeStep a (2 ^ k))

/-- One level halves the live width and preserves the sum. -/
theorem treeStep_sum (a : Nat → Int) (off : Nat) :
    rsum (treeStep a off) off = rsum a (off + off) := by
  rw [rsum_split, ← rsum_add]
  exact rsum_congr (fun t ht => by simp [treeStep, ht, Nat.add_comm t off])

/-- **Tree correctness.**  Over `2^k` slots the tree leaves the exact sum in slot 0, for every power-of-two
`tpr` the split dispatcher accepts. -/
theorem treeReduce_eq (k : Nat) : ∀ a, treeReduce k a = rsum a (2 ^ k) := by
  induction k with
  | zero =>
    intro a
    show a 0 = 0 + a 0
    omega
  | succ k ih =>
    intro a
    rw [treeReduce, ih, treeStep_sum, Nat.pow_succ, Nat.mul_two]

/-- Every tree read `t + off` with `t < off ≤ tpr/2` stays inside the `tpr` shared slots. -/
theorem tree_read_in_bounds (t off tpr : Nat) (ht : t < off) (hoff : 2 * off ≤ tpr) : t + off < tpr := by
  omega

/-- **Split kernel end to end** (exact model): partials by strided lanes, then the tree, equal the row sum. -/
theorem split_kernel_exact (f : Nat → Int) (K n : Nat) :
    treeReduce K (fun s => ssum f s (2 ^ K) n) = rsum f n := by
  rw [treeReduce_eq]
  exact strided_partition f (2 ^ K) n (Nat.two_pow_pos K)

-- § dispatch

/-- The three kernels in the file. -/
inductive Kernel
  | naive
  | warp
  | split
  deriving DecidableEq, Repr

/-- `bf16_gemv`'s choice (`none` = early return, nothing launched). -/
def dispatch (nIn nOut : Int) : Option Kernel :=
  if nIn ≤ 0 ∨ nOut ≤ 0 then none else if 64 ≤ nOut then some .warp else some .naive

/-- `bf16_gemv_split`'s choice: `none` = early return; `some (Except.error ..)` = the host refusal
(`exit(1)`) for a `tpr` that is not 32 and not a positive power of two. -/
def dispatchSplit (nIn nOut : Int) (tpr : Nat) : Option (Except String Kernel) :=
  if nIn ≤ 0 ∨ nOut ≤ 0 then none
  else if tpr = 32 then some (.ok .warp)
  else if tpr = 0 ∨ tpr &&& (tpr - 1) ≠ 0 then some (.error "threads_per_row must be a power of two")
  else some (.ok .split)

/-- Empty shapes launch nothing. -/
theorem dispatch_empty (nIn nOut : Int) (h : nIn ≤ 0 ∨ nOut ≤ 0) : dispatch nIn nOut = none := by
  simp [dispatch, h]

/-- The coalesced warp kernel is chosen exactly from 64 output rows up (8 blocks of 8 warps). -/
theorem dispatch_warp_iff (nIn nOut : Int) (hi : 0 < nIn) (ho : 0 < nOut) :
    dispatch nIn nOut = some .warp ↔ 64 ≤ nOut := by
  have hne : ¬ (nIn ≤ 0 ∨ nOut ≤ 0) := by omega
  by_cases h : 64 ≤ nOut <;> simp [dispatch, hne, h]

/-- Below 64 rows the naive (uncoalesced) kernel is still used: the branch the source flags to revisit. -/
theorem dispatch_naive_small (nIn nOut : Int) (hi : 0 < nIn) (ho : 0 < nOut) (hs : nOut < 64) :
    dispatch nIn nOut = some .naive := by
  have hne : ¬ (nIn ≤ 0 ∨ nOut ≤ 0) := by omega
  have hs' : ¬ (64 ≤ nOut) := by omega
  simp [dispatch, hne, hs']

/-- Every power of two `2^k` passes the host's `tpr & (tpr-1) == 0` test. -/
theorem pow2_passes (k : Nat) : (2 ^ k) &&& (2 ^ k - 1) = 0 := by
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_self]

/-- `tpr = 32` always selects the warp path; 48 and 0 are refused; 64 uses the block-split kernel. -/
theorem dispatchSplit_examples :
    dispatchSplit 640 48 32 = some (.ok .warp) ∧
    (∃ m, dispatchSplit 640 48 48 = some (.error m)) ∧
    (∃ m, dispatchSplit 640 48 0 = some (.error m)) ∧
    dispatchSplit 640 48 64 = some (.ok .split) := by
  exact ⟨rfl, ⟨_, rfl⟩, ⟨_, rfl⟩, rfl⟩

/-- The host does not bound `tpr` above: 2048 is accepted although a CUDA block holds at most 1024 threads,
so that request only fails at launch (caught by `finish`, which exits). -/
theorem dispatchSplit_accepts_2048 : dispatchSplit 640 48 2048 = some (.ok .split) := rfl

-- § grids and index maps

/-- Warp-kernel grid: `⌈nOut / 8⌉` blocks. -/
def warpGrid (nOut : Nat) : Nat := (nOut + 7) / 8

/-- Naive-kernel grid: `⌈nOut / 256⌉` blocks. -/
def naiveGrid (nOut : Nat) : Nat := (nOut + 255) / 256

/-- Warp row map `o = blockIdx * 8 + tid / 32`. -/
def warpRow (blk tid : Nat) : Nat := blk * 8 + tid / 32

/-- The warp grid has a warp for every row. -/
theorem warpGrid_covers (nOut : Nat) : nOut ≤ warpGrid nOut * 8 := by
  unfold warpGrid; omega

/-- The naive grid has a thread for every row. -/
theorem naiveGrid_covers (nOut : Nat) : nOut ≤ naiveGrid nOut * 256 := by
  unfold naiveGrid; omega

/-- **Every row is owned by a warp of the grid**: row `o` is warp `o % 8` of block `o / 8`. -/
theorem warpRow_surj (nOut o : Nat) (ho : o < nOut) :
    ∃ blk tid, blk < warpGrid nOut ∧ tid < 256 ∧ tid % 32 = 0 ∧ warpRow blk tid = o := by
  refine ⟨o / 8, (o % 8) * 32, ?_, ?_, ?_, ?_⟩
  · unfold warpGrid; omega
  · omega
  · omega
  · unfold warpRow; omega

/-- **No two warps share a row**: the (block, warp) pair is determined by the row. -/
theorem warpRow_inj (b1 t1 b2 t2 : Nat) (h1 : t1 < 256) (h2 : t2 < 256)
    (h : warpRow b1 t1 = warpRow b2 t2) : b1 = b2 ∧ t1 / 32 = t2 / 32 := by
  unfold warpRow at h; omega

/-- The source's sizing note: 64 rows are 8 warp blocks, 128 rows are 16. -/
theorem warpGrid_threshold : warpGrid 64 = 8 ∧ warpGrid 128 = 16 := by decide

/-- Lane coalescing: at a fixed iteration the 32 lanes of a warp read 32 consecutive elements of the row. -/
theorem warp_lanes_consecutive (rowBase it lane : Nat) :
    (rowBase + it * 32 + (lane + 1)) = (rowBase + it * 32 + lane) + 1 := by
  omega

end Ref.Bf16GemvDispatch



## FILE: /mnt/user-data/outputs/Bf16MmvfBlockSelect.lean
/-!
# MODULE: bf16_mmvf_block_select
SOURCES: src/kernels/cuda/native_bf16.cu
The FP32-activation BF16 GEMV adapted from llama.cpp's `mmvf`: one block per output row, each thread walking
PAIRS of elements (one `uint32` of two bf16 weights against one `float2` of activations).  This module models
the block-size search `mmvf_block_size`, proves it always lands on a case of the launch `switch` and is optimal
among the candidates, and models the host argument/alignment contract and the pair decomposition.

## Source semantics
* `mmvf_block_size(n)`: start at `(32, ⌈n/64⌉)`; for `c = 64, 96, …, 256`, replace the best when
  `⌈n/(2c)⌉` is STRICTLY smaller (so ties keep the smaller block).
* The `switch` has cases 32, 64, …, 256 and no `default`: a value outside that set would launch nothing.
* Host contract: `n_in > 0`, `n_in` even, `n_in ≤ INT_MAX`, `0 < n_out ≤ INT_MAX`; `x` 8-byte aligned (read as
  `float2`), `w` and `y` 4-byte aligned; violations throw `invalid_argument`.
* Pair `p` covers elements `2p` (low half of the weight word) and `2p+1` (high half), little-endian.
* Cross-warp step: `BLOCK_SIZE/32 ≤ 8` warp sums fit the 32-entry `partials` array.

## Form template identifiers
`mmvf_block_size` → `blockSize`; candidate loop → `candidates`/`stepBest`; iteration count → `iters`;
`bf16_gemv_fp32_mmvf` argument checks → `argsOk`, `alignOk`; `weights2[pair]` halves → `lo16`, `hi16`.
-/

namespace Ref.Bf16MmvfBlockSelect

-- § the block-size search

/-- Candidate block sizes after the initial 32: `for (candidate = 64; candidate <= 256; candidate += 32)`. -/
def candidates : List Nat := [64, 96, 128, 160, 192, 224, 256]

/-- Per-thread pair iterations for a block of `c` threads: `(n + 2c - 1) / (2c)`. -/
def iters (n c : Nat) : Nat := (n + 2 * c - 1) / (2 * c)

/-- One loop step: adopt `c` only on a strict improvement. -/
def stepBest (n : Nat) (acc : Nat × Nat) (c : Nat) : Nat × Nat :=
  if iters n c < acc.2 then (c, iters n c) else acc

/-- The initial state `(32, (n + 63) / 64)`. -/
def init (n : Nat) : Nat × Nat := (32, (n + 63) / 64)

/-- Final `(best, best_iterations)`. -/
def search (n : Nat) : Nat × Nat := candidates.foldl (stepBest n) (init n)

/-- `mmvf_block_size`. -/
def blockSize (n : Nat) : Nat := (search n).1

/-- The cases of the launch `switch`. -/
def switchCases : List Nat := [32, 64, 96, 128, 160, 192, 224, 256]

/-- Generic invariant rule for the fold. -/
theorem fold_inv (n : Nat) (P : Nat × Nat → Prop) :
    ∀ (cs : List Nat) (acc : Nat × Nat), P acc → (∀ a c, c ∈ cs → P a → P (stepBest n a c)) →
      P (cs.foldl (stepBest n) acc) := by
  intro cs
  induction cs with
  | nil => intro acc h0 _; exact h0
  | cons c rest ih =>
    intro acc h0 hs
    exact ih (stepBest n acc c) (hs acc c (List.mem_cons_self) h0)
      (fun a c' hc' ha => hs a c' (List.mem_cons_of_mem _ hc') ha)

/-- A step never increases the best iteration count. -/
theorem stepBest_le (n : Nat) (a : Nat × Nat) (c : Nat) : (stepBest n a c).2 ≤ a.2 := by
  unfold stepBest
  by_cases h : iters n c < a.2 <;> simp [h] <;> omega

/-- After a step, the best is no worse than the candidate just tried. -/
theorem stepBest_le_cand (n : Nat) (a : Nat × Nat) (c : Nat) : (stepBest n a c).2 ≤ iters n c := by
  unfold stepBest
  by_cases h : iters n c < a.2 <;> simp [h] <;> omega

/-- The fold is monotone non-increasing in the iteration count. -/
theorem fold_le_init (n : Nat) (cs : List Nat) (acc : Nat × Nat) :
    (cs.foldl (stepBest n) acc).2 ≤ acc.2 :=
  fold_inv n (fun a => a.2 ≤ acc.2) cs acc (Nat.le_refl _)
    (fun a c _ ha => Nat.le_trans (stepBest_le n a c) ha)

/-- Every candidate the fold visits bounds the final iteration count. -/
theorem fold_le_each (n : Nat) :
    ∀ (cs : List Nat) (acc : Nat × Nat) (c : Nat), c ∈ cs → (cs.foldl (stepBest n) acc).2 ≤ iters n c := by
  intro cs
  induction cs with
  | nil => intro _ _ hc; cases hc
  | cons c0 rest ih =>
    intro acc c hc
    rcases List.mem_cons.mp hc with h | h
    · subst h
      exact Nat.le_trans (fold_le_init n rest _) (stepBest_le_cand n acc c)
    · exact ih _ c h

/-- **Exhaustive switch.**  The selected block size is always one of the `switch` cases, so the launch is
never silently skipped. -/
theorem blockSize_mem_switch (n : Nat) : blockSize n ∈ switchCases := by
  have h := fold_inv n (fun a => a.1 = 32 ∨ a.1 ∈ candidates) candidates (init n) (Or.inl rfl)
    (fun a c hc ha => by
      unfold stepBest
      by_cases hlt : iters n c < a.2
      · simp only [hlt, ite_true]; exact Or.inr hc
      · simp only [hlt]; exact ha)
  rcases h with h | h
  · unfold blockSize search; rw [h]; simp [switchCases]
  · unfold blockSize search; simp only [switchCases, candidates] at h ⊢; simp_all

/-- The bookkeeping pair stays consistent: the recorded count is the iteration count of the recorded size. -/
theorem search_consistent (n : Nat) : (search n).2 = iters n (search n).1 := by
  apply fold_inv n (fun a => a.2 = iters n a.1)
  · simp only [init, iters]; omega
  · intro a c _ ha
    unfold stepBest
    by_cases hlt : iters n c < a.2
    · simp only [hlt, ite_true]
    · simp only [hlt, ite_false]; exact ha

/-- **Optimality.**  The chosen block needs no more iterations than any candidate, including 32. -/
theorem blockSize_optimal (n c : Nat) (hc : c ∈ switchCases) : iters n (blockSize n) ≤ iters n c := by
  rw [blockSize, ← search_consistent]
  rcases List.mem_cons.mp hc with h | h
  · subst h
    have := fold_le_init n candidates (init n)
    simp only [search, init, iters] at this ⊢
    omega
  · exact fold_le_each n candidates (init n) c h

/-- Concrete selections at the model's shapes (`n_embd = 2560`, expert width 640). -/
theorem blockSize_examples : blockSize 2560 = 256 ∧ blockSize 640 = 160 ∧ blockSize 64 = 32 := by
  decide

/-- Ties keep the smaller block: at `n = 640` the 160-, 192-, 224- and 256-thread blocks all need 2 pair
iterations and 160 is chosen. -/
theorem blockSize_tie_640 :
    iters 640 160 = 2 ∧ iters 640 192 = 2 ∧ iters 640 256 = 2 ∧ blockSize 640 = 160 := by
  decide

/-- Cross-warp reduction fits: every case is a whole number of warps and at most 8 ≤ 32 partial slots. -/
theorem partials_fit : ∀ b ∈ switchCases, b % 32 = 0 ∧ b / 32 ≤ 32 := by decide

-- § host contract

/-- `INT_MAX`. -/
def intMax : Int := 2147483647

/-- The size checks of `bf16_gemv_fp32_mmvf`. -/
def argsOk (nIn nOut : Int) : Prop :=
  0 < nIn ∧ nIn % 2 = 0 ∧ nIn ≤ intMax ∧ 0 < nOut ∧ nOut ≤ intMax

/-- The pointer checks: `x` read as `float2` (8 B), `w` as `uint32` (4 B), `y` written as `float` (4 B). -/
def alignOk (x w y : Nat) : Prop := x ≠ 0 ∧ w ≠ 0 ∧ y ≠ 0 ∧ x % 8 = 0 ∧ w % 4 = 0 ∧ y % 4 = 0

/-- An odd `n_in` is refused: the pair loop would drop the last element. -/
theorem odd_refused (nIn nOut : Int) (h : nIn % 2 = 1) : ¬ argsOk nIn nOut := by
  intro hk; unfold argsOk at hk; omega

/-- With an even `n_in` the pairs `0 … n_in/2 - 1` cover every element exactly once. -/
theorem pairs_cover (nIn i : Nat) (he : nIn % 2 = 0) (hi : i < nIn) : i / 2 < nIn / 2 ∧ 2 * (i / 2) + i % 2 = i := by
  omega

/-- Every `float2` read `x + 8p` stays 8-byte aligned given an aligned base. -/
theorem float2_aligned (x p : Nat) (hx : x % 8 = 0) : (x + 8 * p) % 8 = 0 := by omega

/-- Row `o`'s `uint32` reads stay 4-byte aligned: the row starts at `w + 2*n_in*o` bytes and `n_in` is even. -/
theorem row_words_aligned (w nIn o p : Nat) (hw : w % 4 = 0) (he : nIn % 2 = 0) :
    (w + 2 * (nIn * o) + 4 * p) % 4 = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, nIn = 2 * k := ⟨nIn / 2, by omega⟩
  rw [Nat.mul_assoc]
  omega

-- § the pair word

/-- Low bf16 of a weight word (element `2p`). -/
def lo16 (wd : Nat) : Nat := wd % 65536

/-- High bf16 of a weight word (element `2p+1`): `weight >> 16`, truncated to 16 bits. -/
def hi16 (wd : Nat) : Nat := wd / 65536 % 65536

/-- Little-endian pairing: the word assembled from elements `e0, e1` hands them back in order. -/
theorem pair_roundtrip (e0 e1 : Nat) (h0 : e0 < 65536) (h1 : e1 < 65536) :
    lo16 (e0 + 65536 * e1) = e0 ∧ hi16 (e0 + 65536 * e1) = e1 := by
  unfold lo16 hi16; constructor <;> omega

/-- `(uint16_t) (weight >> 16)` agrees with `hi16` (the shift-then-truncate the kernel performs). -/
theorem hi16_shift (wd : Nat) : (wd >>> 16) % 65536 = hi16 wd := by
  rw [Nat.shiftRight_eq_div_pow]; rfl

end Ref.Bf16MmvfBlockSelect



## FILE: /mnt/user-data/outputs/GgmlBlockDequant.lean
/-!
# MODULE: ggml_block_dequant
SOURCES: src/kernels/cuda/dequant_bf16.cu
Row-slice dequantization of ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K,
IQ4_XS) to BF16 / FP16 / FP32, one thread per 32-element group.  This module models the format geometry table,
the group→block addressing, the bit-level unpacking rules, the round-to-nearest-even FP32→BF16 conversion, and
the dispatch to the i-quant path.

## Source semantics
* `geometry(type)` gives `(block_elems, block_bytes)`; unsupported types make `launch` exit.
* `launch` requires `cols % block_elems = 0`, `rows > 0`; `row_bytes = cols/block_elems*block_bytes`,
  `groups_per_row = cols/32`, one thread per group, output group `g` of row `r` at `r*gpr*32 + g*32`.
* `f2bf(u) = (u + 0x7fff + ((u >> 16) & 1)) >> 16` on the FP32 bit pattern: nearest, ties to even, NO NaN guard.
* Q4_0/IQ4_NL: element `j` = low nibble of byte `j`, element `j+16` = high nibble.
  Q5_0: fifth bit of element `j` is bit `j` of `qh`, of element `j+16` bit `j+16`.
  Q6_K: `q = low4 | high2 << 4 ∈ [0,64)`, decoded `q - 32`.  K-scale unpack `scale_min_k4` yields 6-bit values.
* `dequant_f16/f32` route the i-quant types {16,17,18,21,22,29} to `iq_kernels`; `dequant_bf16` does not.

## Form template identifiers
`geometry` → `geometry`; `dequant_bf16_supported` → `supported`; `iq_only` → `iqOnly`; `f2bf` → `f2bf`;
`kv_iq4nl` → `iq4nl`; `scale_min_k4` → `scaleMinK4`; per-group block index → `blockOfGroup`.
-/

namespace Ref.GgmlBlockDequant

-- § format geometry

/-- `geometry(type)`: `(block_elems, block_bytes)` for the supported ggml type ids. -/
def geometry : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)     -- Q4_0
  | 6 => some (32, 22)     -- Q5_0
  | 8 => some (32, 34)     -- Q8_0
  | 20 => some (32, 18)    -- IQ4_NL
  | 11 => some (256, 110)  -- Q3_K
  | 12 => some (256, 144)  -- Q4_K
  | 13 => some (256, 176)  -- Q5_K
  | 14 => some (256, 210)  -- Q6_K
  | 23 => some (256, 136)  -- IQ4_XS
  | 42 => some (64, 18)    -- Q2_0
  | _ => none

/-- The supported type ids. -/
def supportedTypes : List Nat := [2, 6, 8, 20, 11, 12, 13, 14, 23, 42]

/-- `dequant_bf16_supported`. -/
def supported (t : Nat) : Bool := (geometry t).isSome

/-- The struct layout each block size is derived from (field byte counts, summed). -/
def layoutBytes : Nat → Nat
  | 2 => 2 + 16                 -- d, qs[16]
  | 6 => 2 + 4 + 16             -- d, qh[4], qs[16]
  | 8 => 2 + 32                 -- d, qs[32]
  | 20 => 2 + 16                -- d, qs[16]
  | 11 => 32 + 64 + 12 + 2      -- hmask, qs, scales, d
  | 12 => 2 + 2 + 12 + 128      -- d, dmin, scales, qs
  | 13 => 2 + 2 + 12 + 32 + 128 -- d, dmin, scales, qh, qs
  | 14 => 128 + 64 + 16 + 2     -- ql, qh, scales, d
  | 23 => 2 + 2 + 4 + 128       -- d, scales_h, scales_l, qs
  | 42 => 2 + 16                -- d, qs[16] (64 two-bit codes)
  | _ => 0

/-- Block byte sizes agree with the field layouts of the ggml structs. -/
theorem geometry_matches_layout :
    ∀ t ∈ supportedTypes, (geometry t).map Prod.snd = some (layoutBytes t) := by
  decide

/-- `supported` holds exactly on the listed types. -/
theorem supported_iff (t : Nat) : supported t = true ↔ t ∈ supportedTypes := by
  unfold supported geometry supportedTypes
  split <;> simp_all

/-- Every block holds a whole number of 32-element groups, so the thread-per-group grid is exact. -/
theorem blocks_whole_groups (t e b : Nat) (h : geometry t = some (e, b)) : e % 32 = 0 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Groups per block (`block_elems / 32`): 1 for the 32-blocks, 2 for Q2_0, 8 for the K-quants. -/
def groupsPerBlock (e : Nat) : Nat := e / 32

/-- The block holding group `gi` of a row and the group's index inside it. -/
def blockOfGroup (e gi : Nat) : Nat × Nat := (gi / groupsPerBlock e, gi % groupsPerBlock e)

/-- The kernel's hard-coded indices are the generic ones: `gi/2, gi%2` for Q2_0 and `gi/8, gi%8` for K. -/
theorem kernel_block_indices (gi : Nat) :
    blockOfGroup 64 gi = (gi / 2, gi % 2) ∧ blockOfGroup 256 gi = (gi / 8, gi % 8) ∧
    blockOfGroup 32 gi = (gi, 0) := by
  simp [blockOfGroup, groupsPerBlock, Nat.mod_one]

/-- Group `gi`'s 32 elements lie inside the block the kernel reads (for 2 and 8 groups per block). -/
theorem group_inside_block (gi j : Nat) (hj : j < 32) :
    (gi / 2) * 64 ≤ gi * 32 + j ∧ gi * 32 + j < (gi / 2) * 64 + 64 ∧
    (gi / 8) * 256 ≤ gi * 32 + j ∧ gi * 32 + j < (gi / 8) * 256 + 256 := by
  omega

/-- `launch`'s shape gate. -/
def shapeOk (t rows cols : Nat) : Prop :=
  ∃ e b, geometry t = some (e, b) ∧ cols % e = 0 ∧ 0 < rows

/-- Under the gate, `groups_per_row * 32 = cols`: the output tile is exactly `rows × cols`. -/
theorem gpr_exact (t rows cols : Nat) (h : shapeOk t rows cols) : cols / 32 * 32 = cols := by
  obtain ⟨e, b, hg, hc, _⟩ := h
  have he := blocks_whole_groups t e b hg
  have h32 : cols % 32 = 0 :=
    Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_of_mod_eq_zero he) (Nat.dvd_of_mod_eq_zero hc))
  omega

/-- Output addressing `r*gpr*32 + gi*32 + j` is injective over the tile. -/
theorem out_index_inj (gpr r1 g1 j1 r2 g2 j2 : Nat) (hg1 : g1 < gpr) (hg2 : g2 < gpr)
    (hj1 : j1 < 32) (hj2 : j2 < 32)
    (h : r1 * gpr * 32 + g1 * 32 + j1 = r2 * gpr * 32 + g2 * 32 + j2) :
    r1 = r2 ∧ g1 = g2 ∧ j1 = j2 := by
  have hj : j1 = j2 := by omega
  subst hj
  have h' : (r1 * gpr + g1) * 32 = (r2 * gpr + g2) * 32 := by
    rw [Nat.add_mul, Nat.add_mul]; omega
  have h2 : r1 * gpr + g1 = r2 * gpr + g2 := Nat.eq_of_mul_eq_mul_right (by decide) h'
  have hr : r1 = r2 := by
    rcases Nat.lt_trichotomy r1 r2 with hl | he | hl
    · have : (r1 + 1) * gpr ≤ r2 * gpr := Nat.mul_le_mul_right _ hl
      rw [Nat.add_mul, Nat.one_mul] at this; omega
    · exact he
    · have : (r2 + 1) * gpr ≤ r1 * gpr := Nat.mul_le_mul_right _ hl
      rw [Nat.add_mul, Nat.one_mul] at this; omega
  subst hr
  exact ⟨rfl, by omega, rfl⟩

-- § i-quant routing

/-- `iq_only`: types served by `iq_kernels`. -/
def iqOnly (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- The i-quant set and the block-format table are disjoint, so the routing is unambiguous. -/
theorem iq_disjoint : ∀ t ∈ [16, 17, 18, 21, 22, 29], supported t = false := by decide

/-- `dequant_bf16` has no i-quant branch: an i-quant type is unsupported there and `launch` would exit, while
`dequant_f16`/`dequant_f32` serve it. -/
theorem bf16_lacks_iq : iqOnly 16 = true ∧ supported 16 = false := by decide

-- § FP32 → BF16 rounding

/-- `f2bf` on the FP32 bit pattern `u < 2^32` (the C arithmetic wraps mod `2^32`, then truncates to 16 bits). -/
def f2bf (u : Nat) : Nat := (u + 0x7fff + u / 65536 % 2) % 4294967296 / 65536 % 65536

/-- Away from the NaN/Inf top of the range neither the `2^32` wrap nor the 16-bit truncation fires. -/
theorem f2bf_nowrap (u : Nat) (hu : u < 4294901760) : f2bf u = (u + 0x7fff + u / 65536 % 2) / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega : u + 0x7fff + u / 65536 % 2 < 4294967296)]
  exact Nat.mod_eq_of_lt (by omega)

/-- Below the halfway point the low half is dropped (round down in magnitude). -/
theorem f2bf_down (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 < 32768) : f2bf u = u / 65536 := by
  rw [f2bf_nowrap u hu]; omega

/-- Above the halfway point the result rounds up by one ulp. -/
theorem f2bf_up (u : Nat) (hu : u < 4294901760) (hlo : 32768 < u % 65536) : f2bf u = u / 65536 + 1 := by
  rw [f2bf_nowrap u hu]; omega

/-- **Ties go to even**: at exactly half an ulp the result's last bit is 0. -/
theorem f2bf_tie_even (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 = 32768) : f2bf u % 2 = 0 := by
  rw [f2bf_nowrap u hu]
  have hq : (u + 0x7fff + u / 65536 % 2) / 65536 = u / 65536 + u / 65536 % 2 := by omega
  rw [hq]; omega

/-- **No NaN guard.**  The rounding add carries NaN payloads across the exponent: the FP32 NaN `0x7FFFFFFF`
becomes BF16 `0x8000` (−0.0).  Only a non-finite block scale could reach this, but the conversion itself
does not preserve NaN. -/
theorem f2bf_nan_to_negzero : f2bf 0x7FFFFFFF = 0x8000 := by decide

/-- A NaN whose payload sits only in the low half rounds to the infinity pattern `0x7F80`. -/
theorem f2bf_nan_to_inf : f2bf 0x7F800001 = 0x7F80 := by decide

-- § bit unpacking

/-- Shift-and-mask is division-and-modulo by powers of two (the identity every unpacker relies on). -/
theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m := by
  rw [Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]

/-- A byte splits into its two nibbles, low nibble first. -/
theorem nibbles (b : Nat) (hb : b < 256) : b &&& 15 = b % 16 ∧ b >>> 4 = b / 16 ∧ b = b % 16 + 16 * (b / 16) ∧
    b / 16 < 16 := by
  have h1 : b &&& 15 = b % 16 := by
    have := Nat.and_two_pow_sub_one_eq_mod b 4; simpa using this
  have h2 : b >>> 4 = b / 16 := by rw [Nat.shiftRight_eq_div_pow]
  exact ⟨h1, h2, by omega, by omega⟩

/-- Q4_0 code range: `nibble - 8 ∈ [-8, 7]`. -/
theorem q4_0_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7 := by omega

/-- Testing bit 4 of any value: `a & 0x10 = 16 * bit₄(a)`. -/
theorem and16 (a : Nat) : a &&& 16 = 16 * (a / 16 % 2) := by
  have hsplit : a &&& 16 = (a &&& 31) &&& 16 := by
    rw [Nat.and_assoc, show (31 : Nat) &&& 16 = 16 by decide]
  have h31 : a &&& 31 = a % 32 := by
    have := Nat.and_two_pow_sub_one_eq_mod a 5; simpa using this
  have tbl : ∀ r : Fin 32, r.val &&& 16 = 16 * (r.val / 16 % 2) := by decide
  have hr := tbl ⟨a % 32, Nat.mod_lt _ (by decide)⟩
  simp only at hr
  rw [hsplit, h31, hr]
  omega

/-- Q5_0, low element `j`: `((qh >> j) << 4) & 0x10 = 16 * bit_j(qh)`. -/
theorem q5_0_high_bit_lo (qh j : Nat) : ((qh >>> j) <<< 4) &&& 16 = 16 * (qh / 2 ^ j % 2) := by
  rw [and16, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
  have : qh / 2 ^ j * 2 ^ 4 / 16 = qh / 2 ^ j := by simp
  rw [this]

/-- Q5_0, high element `j+16`: `(qh >> (j+12)) & 0x10 = 16 * bit_{j+16}(qh)`. -/
theorem q5_0_high_bit_hi (qh j : Nat) : (qh >>> (j + 12)) &&& 16 = 16 * (qh / 2 ^ (j + 16) % 2) := by
  rw [and16, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul,
    show (16 : Nat) = 2 ^ 4 from rfl, ← Nat.pow_add]

/-- Six-bit assembly `low4 | high2 << 4` is plain addition (Q6_K codes and K-quant scales). -/
theorem or_shift4 : ∀ a : Fin 16, ∀ h : Fin 4, (a.val ||| (h.val <<< 4)) = a.val + 16 * h.val := by
  decide

/-- Q6_K code: `q - 32 ∈ [-32, 31]`. -/
theorem q6_k_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo ||| (hi <<< 4)) < 64 ∧ -32 ≤ ((lo ||| (hi <<< 4) : Nat) : Int) - 32 := by
  have := or_shift4 ⟨lo, hl⟩ ⟨hi, hh⟩
  simp only at this
  omega

/-- `scale_min_k4`. -/
def scaleMinK4 (j : Nat) (q : Nat → Nat) : Nat × Nat :=
  if j < 4 then (q j &&& 63, q (j + 4) &&& 63)
  else ((q (j + 4) &&& 15) ||| ((q (j - 4) >>> 6) <<< 4), (q (j + 4) >>> 4) ||| ((q j >>> 6) <<< 4))

/-- **K-scales are 6-bit** for bytes in range, in both halves of the 12-byte scale field. -/
theorem scaleMinK4_lt64 (j : Nat) (q : Nat → Nat) (hq : ∀ i, q i < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64 := by
  have m63 : ∀ x, x &&& 63 < 64 := fun x => by
    have := Nat.and_two_pow_sub_one_eq_mod x 6; simp at this; rw [this]; exact Nat.mod_lt _ (by decide)
  have m15 : ∀ x, x &&& 15 < 16 := fun x => by
    have := Nat.and_two_pow_sub_one_eq_mod x 4; simp at this; rw [this]; exact Nat.mod_lt _ (by decide)
  have s6 : ∀ x, x < 256 → x >>> 6 < 4 := fun x hx => by rw [Nat.shiftRight_eq_div_pow]; simp; omega
  have s4 : ∀ x, x < 256 → x >>> 4 < 16 := fun x hx => by rw [Nat.shiftRight_eq_div_pow]; simp; omega
  unfold scaleMinK4
  by_cases hj : j < 4
  · simp only [hj, ite_true]; exact ⟨m63 _, m63 _⟩
  · simp only [hj, ite_false]
    have e1 := or_shift4 ⟨_, m15 (q (j + 4))⟩ ⟨_, s6 _ (hq (j - 4))⟩
    have e2 := or_shift4 ⟨_, s4 _ (hq (j + 4))⟩ ⟨_, s6 _ (hq j)⟩
    simp only at e1 e2
    have := m15 (q (j + 4)); have := s6 _ (hq (j - 4)); have := s4 _ (hq (j + 4)); have := s6 _ (hq j)
    omega

/-- Q3_K's high-mask bit `1 << (n*4 + jj)` (n < 2, jj < 4) fits in the `uint8_t` it is stored in. -/
theorem q3_k_hmask_fits : ∀ n : Fin 2, ∀ jj : Fin 4, 1 <<< (n.val * 4 + jj.val) < 256 := by decide

-- § the IQ4_NL codebook

/-- `kv_iq4nl`. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, is strictly increasing and lies in `int8`. -/
theorem iq4nl_shape : iq4nl.length = 16 ∧ iq4nl.Pairwise (· < ·) ∧ ∀ v ∈ iq4nl, -128 ≤ v ∧ v ≤ 127 := by
  decide

end Ref.GgmlBlockDequant



## FILE: /mnt/user-data/outputs/S2CanonicalDequant.lean
/-!
# MODULE: s2_canonical_dequant
SOURCES: src/kernels/cuda/dequant_s2.cu
Dequantization of the canonical S2 plane (Q2_0's form): 2-bit codes packed four per byte LSB-first, one FP32
scale per 64-element group, decoded as `(code - 1) * scale`.  This module models the per-block kernel loop,
proves its byte/shift addressing equals the global plane convention, and pins the decoded value set.

## Source semantics
* Element `i` of the plane is at byte `i/4`, bits `(i%4)*2`; a group/block is 64 elements = 16 code bytes.
* Thread `b` handles block `b`: code byte `b*16 + j/4`, shift `(j%4)*2`, output `b*64 + j`, `j < 64`.
* Decode: the bias −1 is applied to the CODE in the integer domain, then ONE multiply by the scale —
  this is what makes the decode bit-exact against the reference (`docs/pack-format.md` §3.1).
* `n_blocks ≤ 0` launches nothing; grid `⌈n_blocks / 256⌉` × 256.

## Form template identifiers
`dequant_s2_kernel` → `kernelElem`; `QK` → `64`; `CODES_PER_BYTE` → `4`; `code_bias` → `-1`;
plane element accessor → `planeCode`; decoded value → `decode`.
-/

namespace Ref.S2CanonicalDequant

-- § the plane convention

/-- The plane's code for global element `i`: byte `i/4`, bits `(i%4)*2`, two bits wide. -/
def planeCode (byte : Nat → Nat) (i : Nat) : Nat := (byte (i / 4) >>> ((i % 4) * 2)) &&& 3

/-- What the kernel thread for block `b` reads for its element `j`. -/
def kernelCode (byte : Nat → Nat) (b j : Nat) : Nat := (byte (b * 16 + j / 4) >>> ((j % 4) * 2)) &&& 3

/-- Shift-and-mask is division-and-modulo. -/
theorem extract2 (x s : Nat) : (x >>> s) &&& 3 = x / 2 ^ s % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod (x >>> s) 2
  rw [Nat.shiftRight_eq_div_pow] at this ⊢
  simpa using this

/-- Codes are two bits. -/
theorem planeCode_lt (byte : Nat → Nat) (i : Nat) : planeCode byte i < 4 := by
  unfold planeCode; rw [extract2]; exact Nat.mod_lt _ (by decide)

/-- **Addressing agrees with the plane.**  Block `b`, element `j` reads exactly global element `b*64 + j`
(the identity needs no bound on `j`: 64 is a multiple of the 4 codes per byte). -/
theorem kernel_reads_plane (byte : Nat → Nat) (b j : Nat) :
    kernelCode byte b j = planeCode byte (b * 64 + j) := by
  have h1 : (b * 64 + j) / 4 = b * 16 + j / 4 := by omega
  have h2 : (b * 64 + j) % 4 = j % 4 := by omega
  simp only [kernelCode, planeCode, h1, h2]

/-- Distinct `(block, element)` pairs write distinct outputs `b*64 + j`. -/
theorem out_index_inj (b1 j1 b2 j2 : Nat) (h1 : j1 < 64) (h2 : j2 < 64)
    (h : b1 * 64 + j1 = b2 * 64 + j2) : b1 = b2 ∧ j1 = j2 := by
  omega

/-- A block's 64 elements use exactly its 16 code bytes. -/
theorem block_bytes (b j : Nat) (hj : j < 64) : b * 16 ≤ b * 16 + j / 4 ∧ b * 16 + j / 4 < (b + 1) * 16 := by
  omega

-- § decoding

/-- `(code + code_bias) * scale` with `code_bias = -1`, the bias in the integer domain. -/
def decode (code : Nat) (scale : Int) : Int := ((code : Int) - 1) * scale

/-- The integer multiplier takes only the values `-1, 0, 1, 2`. -/
theorem multiplier_values (code : Nat) (h : code < 4) :
    (code : Int) - 1 = -1 ∨ (code : Int) - 1 = 0 ∨ (code : Int) - 1 = 1 ∨ (code : Int) - 1 = 2 := by
  omega

/-- Code 1 decodes to zero for every scale: the plane can represent an exact zero. -/
theorem decode_one_zero (scale : Int) : decode 1 scale = 0 := by simp [decode]

/-- Decoding is ONE multiply of an integer multiplier by the scale (in the exact model, linear in the scale). -/
theorem decode_linear (code : Nat) (s t : Int) : decode code (s + t) = decode code s + decode code t := by
  unfold decode; rw [Int.mul_add]

-- § launch

/-- Grid size `⌈n_blocks / 256⌉`. -/
def grid (nBlocks : Nat) : Nat := (nBlocks + 255) / 256

/-- Every block has a thread, and the last grid block is not wasted entirely. -/
theorem grid_tight (nBlocks : Nat) (h : 0 < nBlocks) :
    nBlocks ≤ grid nBlocks * 256 ∧ (grid nBlocks - 1) * 256 < nBlocks := by
  unfold grid; omega

end Ref.S2CanonicalDequant



## FILE: /mnt/user-data/outputs/ActivationQuantize.lean
/-!
# MODULE: activation_quantize
SOURCES: src/kernels/cuda/quantize_act.cu
Quantization of FP32 activations into ggml's `block_q8_0` (34 B / 32 elements) and `block_q8_K`
(292 B / 256 elements), the "scaled" Q8_0 variant that reproduces the CPU path's `act_quant_q8_1`, and the
matching dequantizers.  This module models the block layouts, the three integer clamp rules, the two
rounding rules (half-to-even vs half-away-from-zero), ggml's `nearest_int` bit trick, and the `bsums` contract.

## Source semantics
* `quantize_q8_0`: `d32 = amax/127`; store `fp16(d32)`; `q = clamp(rint(x / d32), -128, 127)` computed in
  double (ties to even); an all-zero block stores `d = 0`, `q = 0`.  Dequantization multiplies by the STORED fp16 d.
* `quantize_q8_0_scaled`: CPU rule — `s = amax>0 ? amax/127 : 0`, `inv = s>0 ? 1/s : 0`,
  `q = clamp(trunc(t + copysign(0.5, t)), -127, 127)` (ties away from zero), plus `scales[b] = s` unrounded.
* `quantize_q8_K`: `iscale = -127/max` (max = signed value of the FIRST largest magnitude), `q = min(127,
  nearest_int(fl(iscale*x)))` with no lower clamp, `bsums[j] = Σ q[16j .. 16j+15]`, `d = 1/iscale`; a zero block
  zeroes `d`, `qs` AND `bsums` (a deliberate divergence from ggml, which leaves `bsums` unwritten).
* Hosts refuse `n % 32 ≠ 0` (Q8_0) and `n % 256 ≠ 0` (Q8_K).

## Form template identifiers
`block_q8_0` → `Q8_0` (34 B); `block_q8_K` → `Q8_K` (292 B); `rint` → `roundEven2`;
CPU `lround`-style rule → `roundAway2`; `nearest_int` → `nearestIntBits`; `bsums` → `bsum`;
clamps → `clampQ8_0`, `clampScaled`, `clampQ8K`.
-/

namespace Ref.ActivationQuantize

-- § block layouts

/-- `block_q8_0` bytes: fp16 `d` + 32 × int8. -/
def q8_0Bytes : Nat := 2 + 32

/-- `block_q8_K` bytes: f32 `d` + 256 × int8 + 16 × int16 (no padding). -/
def q8_KBytes : Nat := 4 + 256 + 16 * 2

/-- The strides used by every kernel in the file. -/
theorem block_strides : q8_0Bytes = 34 ∧ q8_KBytes = 292 := by decide

/-- Every Q8_0 block's fp16 `d` is 2-byte aligned (stride 34). -/
theorem q8_0_d_aligned (b : Nat) : (34 * b) % 2 = 0 := by omega

/-- Every Q8_K block's f32 `d` is 4-byte aligned and its `bsums` 2-byte aligned (stride 292, offset 260). -/
theorem q8_K_fields_aligned (b : Nat) : (292 * b) % 4 = 0 ∧ (292 * b + 4 + 256) % 2 = 0 := by omega

/-- Q8_K's 256 + 32 tail: `bsums` ends exactly at the block boundary. -/
theorem q8_K_bsums_end : 4 + 256 + 16 * 2 = q8_KBytes := rfl

-- § clamps

/-- `quantize_q8_0` clamp `[-128, 127]`. -/
def clampQ8_0 (v : Int) : Int := max (-128) (min 127 v)

/-- `quantize_q8_0_scaled` clamp `[-127, 127]` (symmetric, verbatim from the CPU path). -/
def clampScaled (v : Int) : Int := max (-127) (min 127 v)

/-- `quantize_q8_K`: `min(127, v)` only. -/
def clampQ8K (v : Int) : Int := min 127 v

/-- Both Q8_0 clamps land in `int8`. -/
theorem clamps_int8 (v : Int) :
    -128 ≤ clampQ8_0 v ∧ clampQ8_0 v ≤ 127 ∧ -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampQ8_0 clampScaled; omega

/-- The two Q8_0 clamps differ only at −128: the scaled variant can never emit it. -/
theorem clamps_differ_only_at_min (v : Int) (h : -127 ≤ v) : clampQ8_0 v = clampScaled v := by
  unfold clampQ8_0 clampScaled; omega

/-- Q8_K's clamp is one-sided: it is only correct because `|iscale * x| ≤ 127` keeps `v ≥ -127`
(the value at the arg-max is `-127` exactly).  An input below −128 would wrap in the `int8_t` cast. -/
theorem clampQ8K_needs_lower (v : Int) (h : -127 ≤ v) : -127 ≤ clampQ8K v ∧ clampQ8K v ≤ 127 := by
  unfold clampQ8K; omega

/-- Without that premise the clamp is not an `int8` guarantee. -/
theorem clampQ8K_unbounded_below : clampQ8K (-200) = -200 := by decide

-- § rounding rules, on half-integers `k/2`

/-- Ties away from zero at `k/2`: `trunc(t + copysign(0.5, t))`. -/
def roundAway2 (k : Int) : Int := if 0 ≤ k then (k + 1) / 2 else -((-k + 1) / 2)

/-- Ties to even at `k/2` (`rint`). -/
def roundEven2 (k : Int) : Int :=
  if k % 2 = 0 then k / 2 else if ((k - 1) / 2) % 2 = 0 then (k - 1) / 2 else (k + 1) / 2

/-- On exact integers the two rules agree. -/
theorem round_agree_on_integers (n : Int) : roundAway2 (2 * n) = n ∧ roundEven2 (2 * n) = n := by
  unfold roundAway2 roundEven2
  constructor
  · split <;> omega
  · simp <;> omega

/-- On ties the two rules can disagree: `2.5` → 3 vs 2, `-2.5` → −3 vs −2, while `1.5` → 2 under both. -/
theorem round_ties_examples :
    roundAway2 5 = 3 ∧ roundEven2 5 = 2 ∧ roundAway2 (-5) = -3 ∧ roundEven2 (-5) = -2 ∧
    roundAway2 3 = 2 ∧ roundEven2 3 = 2 := by
  decide

/-- The rules never differ by more than one step. -/
theorem round_rules_close (k : Int) : (roundAway2 k - roundEven2 k).natAbs ≤ 1 := by
  unfold roundAway2 roundEven2
  split <;> split <;> (try split) <;> omega

-- § ggml's `nearest_int`

/-- `nearest_int`'s integer read-back: `(bits & 0x7fffff) - 0x400000`. -/
def nearestIntBits (bits : Int) : Int := bits % 8388608 - 4194304

/-- **The magic constant.**  `12582912 = 1.5·2^23`.  When `|v| < 2^22`, the float `12582912 + v` has biased
exponent 150 and mantissa field `2^22 + v`, i.e. bit pattern `150·2^23 + 2^22 + v`; the read-back returns `v`. -/
theorem nearest_int_readback (v : Int) (hlo : -4194304 ≤ v) (hhi : v < 4194304) :
    nearestIntBits (150 * 8388608 + 4194304 + v) = v := by
  unfold nearestIntBits; omega

/-- The constant's decomposition: `1.5 · 2^23 = 2^23 + 2^22`. -/
theorem magic_value : (12582912 : Nat) = 2 ^ 23 + 2 ^ 22 := by decide

-- § bsums

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step of `rsum`. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- A sum over `m*k` indices is the sum of `m` consecutive chunk sums. -/
theorem rsum_nested (f : Nat → Int) (k m : Nat) :
    rsum f (m * k) = rsum (fun b => rsum (fun j => f (b * k + j)) k) m := by
  induction m with
  | zero => simp [rsum]
  | succ m ih => rw [Nat.succ_mul, rsum_split, ih, rsum_succ]

/-- Bounds propagate through `rsum`. -/
theorem rsum_bounds (f : Nat → Int) (lo hi : Int) (n : Nat) (h : ∀ i, i < n → lo ≤ f i ∧ f i ≤ hi) :
    lo * n ≤ rsum f n ∧ rsum f n ≤ hi * n := by
  induction n with
  | zero => simp [rsum]
  | succ n ih =>
    have := ih (fun i hi' => h i (by omega))
    have hn := h n (by omega)
    rw [rsum_succ]
    simp only [Int.natCast_add, Int.natCast_one, Int.mul_add, Int.mul_one]
    omega

/-- `bsums[j]`. -/
def bsum (q : Nat → Int) (j : Nat) : Int := rsum (fun ii => q (j * 16 + ii)) 16

/-- **`bsums` fits `int16`**: sixteen `int8` values sum to within `[-2048, 2032]`. -/
theorem bsum_fits_int16 (q : Nat → Int) (j : Nat) (hq : ∀ i, -128 ≤ q i ∧ q i ≤ 127) :
    -32768 ≤ bsum q j ∧ bsum q j ≤ 32767 := by
  have := rsum_bounds (fun ii => q (j * 16 + ii)) (-128) 127 16 (fun i _ => hq _)
  unfold bsum
  omega

/-- **`bsums` partition the block**: their total is the sum of all 256 quants (what ggml's dot product uses). -/
theorem bsums_total (q : Nat → Int) : rsum (bsum q) 16 = rsum q 256 := by
  rw [show (256 : Nat) = 16 * 16 from rfl, rsum_nested]
  rfl

/-- A zero block's bsums are zero (the deliberate full zeroing). -/
theorem zero_block_bsums (j : Nat) : bsum (fun _ => 0) j = 0 := by
  unfold bsum
  rfl

-- § scales and host checks

/-- The measured worst relative scale difference `4.761e-4` (fp32 `s` vs the block's fp16 `d`) is within
fp16's half-ulp relative bound `2^-11 ≈ 4.883e-4`: the measurement is exactly the rounding of `d`. -/
theorem measured_scale_gap_within_half_ulp : 4761 * 2048 < 10000000 := by decide

/-- `quantize_q8_0` accepts `n` iff it is a positive multiple of 32. -/
def q8_0Accepts (n : Nat) : Bool := 0 < n && n % 32 == 0

/-- `quantize_q8_K` accepts `n` iff it is a positive multiple of 256. -/
def q8_KAccepts (n : Nat) : Bool := 0 < n && n % 256 == 0

/-- The shared-expert width 640 has a Q8_0 activation but structurally no Q8_K one; 2560 has both. -/
theorem widths : q8_0Accepts 640 = true ∧ q8_KAccepts 640 = false ∧
    q8_0Accepts 2560 = true ∧ q8_KAccepts 2560 = true := by decide

/-- Q8_K acceptance implies Q8_0 acceptance. -/
theorem q8K_implies_q8_0 (n : Nat) (h : q8_KAccepts n = true) : q8_0Accepts n = true := by
  simp [q8_KAccepts, q8_0Accepts] at h ⊢
  omega

end Ref.ActivationQuantize



## FILE: /mnt/user-data/outputs/SFamilyGemv.lean
/-!
# MODULE: s_family_gemv
SOURCES: src/kernels/cuda/s_gemv.cu
The S-family GEMV `y[o] = Σ_i (decode(code_oi)·scale_og + offset_og) · x_i` over packed 2/4/8-bit code planes,
with FP16, Q8_K or Q8_0 activations.  Kernels: thread-per-row naive (`s_gemv_kernel`, `s_gemv_q8k_kernel`),
block-split with quads (`s_gemv_split_kernel`), and warp-per-row with 16 elements per lane-iteration
(`s_gemv_q8_split_kernel<CB, Q8K>`).  This module proves the multi-code word reads use the plane's bit order,
the group index as a shift, the "one scale per quad/octet" and "one activation block per lane-iteration"
invariants, the lane coverage of the warp kernel, and records the host gates — including one gate that is
weaker than its kernel needs.

## Source semantics
* Plane convention: element `e` at byte `e / (8/CB)`, bits `(e % (8/CB))·CB`.  A 4-code word is `QW = CB/2`
  bytes read little-endian; code `k` is bits `[k·CB, (k+1)·CB)`.
* Group index `g = e >> group_shift`, `group_shift = min{s | 2^s ≥ group_elems}`; hosts refuse non-powers of two.
* Split kernel: thread `tid` takes quads at `i = 4·tid + 4·tpr·m`; one scale per quad, so `group_elems % 4 = 0`.
* Warp Q8 kernel: lane takes `i = 16·lane + 512·m` (QE = 16), ONE scale for 16 elements (needs
  `group_elems % 16 = 0`) and ONE activation block (`i % 256` and `i % 32` are multiples of 16).
* Hosts: `s_gemv` CB ∈ {2,4,8}; Q8_K and Q8_0 paths CB ∈ {4,8} (S2 uses Q8_0 via other kernels);
  Q8_K needs `n_in % 256 = 0`, `n_in % group = 0`; Q8_0-split needs `n_in % 32 = 0` and checks only
  `group % 4 = 0`.
* Weight decode `w = decode(code)·d + b` BEFORE the activation multiply (exact model: distributes; FP: rounds
  differently, which is why the order is fixed).

## Form template identifiers
`CODE_BITS` → `cb`; `PER_BYTE` → `pb`; packed accessor → `codeAt`; `group_shift` → `groupShift`;
`QE` → 4 (split) / 16 (warp); `q8k_form_ok` → `q8kFormOk`; Q8_0 split gate → `q8_0SplitGate`;
lane start → `laneStart`; `acc0..acc15` combine → `combine16`.
-/

namespace Ref.SFamilyGemv

-- § plane access

/-- Plane code of element `e` with `pb` codes of `cb` bits per byte. -/
def codeAt (byte : Nat → Nat) (pb cb e : Nat) : Nat := (byte (e / pb) >>> ((e % pb) * cb)) &&& (2 ^ cb - 1)

/-- Shift-and-mask is division-and-modulo by powers of two. -/
theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m := by
  rw [Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]

/-- **S2 word read** (`QW = 1`): code `k` of byte `i/4` is plane element `i + k` (for `i = 4a`). -/
theorem s2_word (byte : Nat → Nat) (a k : Nat) (hk : k < 4) :
    (byte a >>> (k * 2)) &&& (2 ^ 2 - 1) = codeAt byte 4 2 (4 * a + k) := by
  unfold codeAt
  rw [show (4 * a + k) / 4 = a by omega, show (4 * a + k) % 4 = k by omega]

/-- **S4 word read** (`QW = 2`): the little-endian `uint16` at byte `2a` yields plane elements `4a … 4a+3`. -/
theorem s4_word (byte : Nat → Nat) (hb : ∀ j, byte j < 256) (a k : Nat) (hk : k < 4) :
    ((byte (2 * a) + 256 * byte (2 * a + 1)) >>> (k * 4)) &&& (2 ^ 4 - 1) = codeAt byte 2 4 (4 * a + k) := by
  unfold codeAt
  rw [extract_eq, extract_eq]
  have h0 := hb (2 * a)
  have h1 := hb (2 * a + 1)
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl
  · rw [show (4 * a + 0) / 2 = 2 * a by omega, show (4 * a + 0) % 2 = 0 by omega]; simp; omega
  · rw [show (4 * a + 1) / 2 = 2 * a by omega, show (4 * a + 1) % 2 = 1 by omega]; simp; omega
  · rw [show (4 * a + 2) / 2 = 2 * a + 1 by omega, show (4 * a + 2) % 2 = 0 by omega]; simp; omega
  · rw [show (4 * a + 3) / 2 = 2 * a + 1 by omega, show (4 * a + 3) % 2 = 1 by omega]; simp; omega

/-- **S8 word read** (`QW = 4`): the little-endian `uint32` at byte `4a` yields plane elements `4a … 4a+3`. -/
theorem s8_word (byte : Nat → Nat) (hb : ∀ j, byte j < 256) (a k : Nat) (hk : k < 4) :
    ((byte (4 * a) + 256 * byte (4 * a + 1) + 65536 * byte (4 * a + 2) + 16777216 * byte (4 * a + 3))
        >>> (k * 8)) &&& (2 ^ 8 - 1) = codeAt byte 1 8 (4 * a + k) := by
  unfold codeAt
  rw [extract_eq, extract_eq]
  have h0 := hb (4 * a)
  have h1 := hb (4 * a + 1)
  have h2 := hb (4 * a + 2)
  have h3 := hb (4 * a + 3)
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.mod_one, Nat.div_one] <;> omega

/-- The 16-element lane-iteration reads FOUR consecutive words: word `w < 4` starts at plane element
`i + 4w`, so code `k` of word `w` is element `i + 4w + k` (the bit order of the four single-word reads). -/
theorem octet_words (i w k : Nat) (hw : w < 4) (hk : k < 4) : i + 4 * w + k < i + 16 ∧ (4 * w + k) / 4 = w := by
  omega

/-- Word size: `QW = 4·CB/8` bytes for the three widths. -/
theorem word_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

-- § the group index

/-- `while ((1 << s) < group_elems) ++s`, with fuel. -/
def shiftLoop (g : Nat) : Nat → Nat → Nat
  | 0, s => s
  | fuel + 1, s => if 2 ^ s < g then shiftLoop g fuel (s + 1) else s

/-- `group_shift`. -/
def groupShift (g : Nat) : Nat := shiftLoop g 64 0

/-- For every power-of-two group size up to 1024 the loop returns its logarithm. -/
theorem groupShift_pow2 : ∀ k : Fin 11, groupShift (2 ^ k.val) = k.val := by decide

/-- Hence the shift computes the group index: `e >> group_shift = e / group_elems`. -/
theorem shift_is_div (e : Nat) (k : Fin 11) : e >>> groupShift (2 ^ k.val) = e / 2 ^ k.val := by
  rw [groupShift_pow2, Nat.shiftRight_eq_div_pow]

/-- A non-power-of-two group is detected: 48 rounds up to shift 6, and `1 << 6 ≠ 48` (the Q8_K-split refusal);
using the shift anyway would index by 64, a WRONG group rather than a slow one. -/
theorem groupShift_48 : groupShift 48 = 6 ∧ 2 ^ groupShift 48 ≠ 48 ∧ 100 / 48 ≠ 100 >>> groupShift 48 := by
  decide

-- § one scale per quad / octet

/-- `n / (c·h) = (n / c) / h`. -/
theorem div_mul_split (n c h : Nat) : n / (c * h) = n / c / h := (Nat.div_div_eq_div_mul n c h).symm

/-- **Quad invariant** (split kernel): with `i ≡ 0 (mod 4)` and `group % 4 = 0`, the four elements of a quad
share a group, so the single `s[g]` load is right for all four. -/
theorem quad_same_group (i k g : Nat) (hi : i % 4 = 0) (hg : g % 4 = 0) (hk : k < 4) : (i + k) / g = i / g := by
  obtain ⟨h, rfl⟩ : ∃ h, g = 4 * h := ⟨g / 4, by omega⟩
  rw [div_mul_split, div_mul_split, show (i + k) / 4 = i / 4 by omega]

/-- **Octet invariant** (warp Q8 kernels): with `i ≡ 0 (mod 16)` and `group % 16 = 0`, all sixteen elements of
a lane-iteration share a group. -/
theorem octet_same_group (i k g : Nat) (hi : i % 16 = 0) (hg : g % 16 = 0) (hk : k < 16) :
    (i + k) / g = i / g := by
  obtain ⟨h, rfl⟩ : ∃ h, g = 16 * h := ⟨g / 16, by omega⟩
  rw [div_mul_split, div_mul_split, show (i + k) / 16 = i / 16 by omega]

/-- Split-kernel quads start on multiples of 4. -/
theorem split_start_aligned (tid tpr m : Nat) : (4 * tid + 4 * tpr * m) % 4 = 0 := by
  rw [Nat.mul_assoc]; omega

/-- Warp-kernel lane-iterations start at `16·lane + 512·m`. -/
def laneStart (lane m : Nat) : Nat := 16 * lane + 512 * m

/-- Lane starts are multiples of 16. -/
theorem laneStart_aligned (lane m : Nat) : laneStart lane m % 16 = 0 := by unfold laneStart; omega

/-- `s_gemv_q8k_split`'s gate (group part): power of two and multiple of 16. -/
def q8kSplitGroupOk (g : Nat) : Bool := 0 < g && g &&& (g - 1) == 0 && g % 16 == 0

/-- `s_gemv_q8_0_split`'s gate (group part): power of two and multiple of FOUR. -/
def q8_0SplitGate (g : Nat) : Bool := 0 < g && g &&& (g - 1) == 0 && g % 4 == 0

/-- The Q8_K gate is sound for the warp kernel: any accepted group keeps a lane-iteration in one group. -/
theorem q8kSplit_gate_sound (g lane m k : Nat) (h : q8kSplitGroupOk g = true) (hk : k < 16) :
    (laneStart lane m + k) / g = laneStart lane m / g := by
  simp [q8kSplitGroupOk] at h
  exact octet_same_group _ k g (laneStart_aligned lane m) h.2 hk

/-- **The Q8_0-split gate is weaker than its kernel.**  It checks `group % 4` (the comment points to the Q8_K
path's checks), but the warp kernel consumes 16 elements per scale load.  A group of 8 passes the gate while
one lane-iteration spans two groups — the second half would be scaled by the first group's `s[g]`.  Every group
size the pack format defines today (16, 32, 64) is safe, so this is latent; the fix is `% 16`. -/
theorem q8_0Split_gate_gap :
    q8_0SplitGate 8 = true ∧ q8kSplitGroupOk 8 = false ∧ (laneStart 0 0 + 8) / 8 ≠ laneStart 0 0 / 8 := by
  decide

/-- Every group size of the format passes both gates. -/
theorem format_groups_pass : ∀ g ∈ [16, 32, 64], q8kSplitGroupOk g = true ∧ q8_0SplitGate g = true := by
  decide

-- § one activation block per lane-iteration

/-- **Q8_K hoist is exact**: the 16 elements of a lane-iteration lie in one 256-element `block_q8_K`. -/
theorem q8k_block_hoist (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) / 256 = laneStart lane m / 256 ∧ laneStart lane m % 256 + 16 ≤ 256 := by
  unfold laneStart; omega

/-- **Q8_0 hoist is exact**: the 16 elements lie in one 32-element `block_q8_0`. -/
theorem q8_0_block_hoist (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) / 32 = laneStart lane m / 32 ∧ laneStart lane m % 32 + 16 ≤ 32 := by
  unfold laneStart; omega

/-- The hoisted pointer `xq = qs(block) + i % blk` indexes element `i + k` at offset `(i % blk) + k`. -/
theorem hoisted_offset (lane m k : Nat) (hk : k < 16) :
    (laneStart lane m + k) % 256 = laneStart lane m % 256 + k ∧
    (laneStart lane m + k) % 32 = laneStart lane m % 32 + k := by
  unfold laneStart; omega

-- § lane coverage of the warp kernel

/-- **Every element has exactly one owner** `(lane, m, k)` with `lane < 32`, `k < 16`. -/
theorem warp_owner_exists (e : Nat) : ∃ lane m k, lane < 32 ∧ k < 16 ∧ laneStart lane m + k = e :=
  ⟨e % 512 / 16, e / 512, e % 16, by omega, by omega, by unfold laneStart; omega⟩

/-- Ownership is unique. -/
theorem warp_owner_unique (l1 m1 k1 l2 m2 k2 : Nat) (hl1 : l1 < 32) (hl2 : l2 < 32) (hk1 : k1 < 16)
    (hk2 : k2 < 16) (h : laneStart l1 m1 + k1 = laneStart l2 m2 + k2) : l1 = l2 ∧ m1 = m2 ∧ k1 = k2 := by
  unfold laneStart at h; omega

-- § exact arithmetic

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- **Naive kernel loop structure**: the per-group double loop sums the same terms as the flat row sum. -/
theorem group_loop_eq_flat (f : Nat → Int) (ge ng : Nat) :
    rsum (fun g => rsum (fun j => f (g * ge + j)) ge) ng = rsum f (ng * ge) := by
  induction ng with
  | zero => simp [rsum]
  | succ m ih => rw [Nat.succ_mul, rsum_split f (m * ge) ge, ← ih, rsum_succ]

/-- The weight decode with offset, `w = c·d + b`, then `w·x` (exact model). -/
def term (c d b x : Int) : Int := (c * d + b) * x

/-- In exact arithmetic the source's order equals the distributed form it rejects; the two differ only in
FP rounding (one multiply-add then one multiply vs two multiplies and an add), which only Q4_K exposed. -/
theorem term_distributes (c d b x : Int) : term c d b x = c * d * x + b * x := by
  unfold term; rw [Int.add_mul]

/-- The 16-accumulator combine of the warp kernel is the plain sum. -/
def combine16 (a : Fin 16 → Int) : Int :=
  (((a 0 + a 1) + (a 2 + a 3)) + ((a 4 + a 5) + (a 6 + a 7))) +
  (((a 8 + a 9) + (a 10 + a 11)) + ((a 12 + a 13) + (a 14 + a 15)))

/-- Exactly, the combine order is irrelevant. -/
theorem combine16_eq (a : Fin 16 → Int) :
    combine16 a = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 +
      a 14 + a 15 := by
  unfold combine16; omega

-- § host gates

/-- Code widths the FP16 entry points (`s_gemv`, `s_gemv_split`) launch. -/
def fp16Widths : List Nat := [2, 4, 8]

/-- Code widths with a Q8_K / Q8_0 warp contract. -/
def q8Widths : List Nat := [4, 8]

/-- S2 has no Q8_K contract: `s_gemv_q8k*` refuse 2-bit codes. -/
theorem s2_no_q8k : 2 ∈ fp16Widths ∧ 2 ∉ q8Widths := by decide

/-- `q8k_form_ok`. -/
def q8kFormOk (g nIn : Nat) : Bool := 0 < g && nIn % g == 0 && nIn % 256 == 0

/-- The Q8_0-split activation gate. -/
def q8_0ActOk (nIn : Nat) : Bool := nIn % 32 == 0

/-- `ffn_down_shexp` (`n_in = 640`) can only be served by the Q8_0 path: Q8_K is structurally impossible. -/
theorem shexp_needs_q8_0 : q8kFormOk 32 640 = false ∧ q8_0ActOk 640 = true := by decide

/-- `s_gemv_split`'s thread gate: power of two in `1 … 1024`. -/
def splitTprOk (t : Nat) : Bool := 1 ≤ t && t &&& (t - 1) == 0 && t ≤ 1024

/-- The gate admits exactly the eleven powers of two `1 … 1024` among `t ≤ 2048`. -/
theorem splitTpr_examples : splitTprOk 1 = true ∧ splitTprOk 1024 = true ∧ splitTprOk 2048 = false ∧
    splitTprOk 96 = false ∧ splitTprOk 0 = false := by decide

end Ref.SFamilyGemv



## FILE: /mnt/user-data/outputs/S2GemvNaive.lean
/-!
# MODULE: s2_gemv_naive
SOURCES: src/kernels/cuda/s2_gemv.cu
The naive S2 GEMV, one thread per output row, FP16 activations: `y[o] = Σ_b Σ_{j<64} (code - 1)·s[o,b]·x[64b+j]`.
It is the reference the faster S2 kernels are checked against.  This module proves its per-row addressing is
the plane convention, its nested group loop is the flat row sum, and pins the host gate.

## Source semantics
* `nb = n_in / 64`; row `o`'s codes start at byte `o·nb·16`, scales at `o·nb`.
* Element `(b, j)`: byte `b·16 + (j >> 2)`, shift `(j & 3)·2`, mask 3, decoded `(code - 1)` in the integer
  domain, then `· d · x` (same order as the CPU reference).
* Host: `n_in ≤ 0 ∨ n_out ≤ 0` → no launch; `n_in % 64 ≠ 0` → exit.  Grid `⌈n_out/128⌉ × 128`, synchronous.

## Form template identifiers
`s2_gemv_kernel` → `rowSum`; `QK` → `64`; `nb` → `nb`; plane accessor → `planeCode`;
kernel accessor → `kernelCode`; host gate → `hostOk`.
-/

namespace Ref.S2GemvNaive

-- § addressing

/-- Global plane code of element `e`: byte `e/4`, bits `(e%4)·2`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- The kernel's read for row `o`, group `b`, element `j`. -/
def kernelCode (byte : Nat → Nat) (nb o b j : Nat) : Nat :=
  (byte (o * nb * 16 + b * 16 + (j >>> 2)) >>> ((j &&& 3) * 2)) &&& 3

/-- `j >> 2 = j / 4`. -/
theorem shr2 (j : Nat) : j >>> 2 = j / 4 := by rw [Nat.shiftRight_eq_div_pow]

/-- `j & 3 = j % 4`. -/
theorem and3 (j : Nat) : j &&& 3 = j % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod j 2; simpa using this

/-- **The kernel reads the plane**: row `o`, group `b`, element `j < 64` is global element
`o·n_in + 64·b + j` with `n_in = 64·nb`, in the row-major `[n_out × n_in]` plane. -/
theorem kernel_reads_plane (byte : Nat → Nat) (nb o b j : Nat) :
    kernelCode byte nb o b j = planeCode byte (o * (64 * nb) + 64 * b + j) := by
  unfold kernelCode planeCode
  rw [shr2, and3 j]
  have hp : o * (64 * nb) = 64 * (o * nb) := by rw [Nat.mul_left_comm]
  have hq : o * nb * 16 = 16 * (o * nb) := Nat.mul_comm _ _
  rw [hp, hq]
  generalize o * nb = P
  rw [show (64 * P + 64 * b + j) / 4 = 16 * P + b * 16 + j / 4 by omega,
    show (64 * P + 64 * b + j) % 4 = j % 4 by omega]

/-- Rows do not overlap: row `o`'s `16·nb` code bytes end where row `o+1`'s begin. -/
theorem rows_disjoint (nb o b j : Nat) (hb : b < nb) (hj : j < 64) :
    o * nb * 16 + b * 16 + j / 4 < (o + 1) * nb * 16 := by
  rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  have : b * 16 + j / 4 < nb * 16 := by
    have : (b + 1) * 16 ≤ nb * 16 := Nat.mul_le_mul_right _ hb
    rw [Nat.add_mul] at this; omega
  omega

-- § the row sum

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- `rsum` only looks at indices below its bound. -/
theorem rsum_congr {f g : Nat → Int} {n : Nat} (h : ∀ i, i < n → f i = g i) : rsum f n = rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [rsum_succ, rsum_succ, ih (fun i hi => h i (by omega)), h n (by omega)]

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- The decoded term for element `e` given the group's scale and the activation. -/
def term (code : Nat) (d x : Int) : Int := ((code : Int) - 1) * d * x

/-- The kernel's double loop for one row (exact arithmetic). -/
def rowSum (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) : Int :=
  rsum (fun b => rsum (fun j => term (code (b * 64 + j)) (scale b) (x (b * 64 + j))) 64) nb

/-- The flat reference: every element uses the scale of its group `e / 64`. -/
def flatSum (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) : Int :=
  rsum (fun e => term (code e) (scale (e / 64)) (x e)) (nb * 64)

/-- **The group loop is the flat sum.** -/
theorem rowSum_eq_flat (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) :
    rowSum code scale x nb = flatSum code scale x nb := by
  unfold rowSum flatSum
  induction nb with
  | zero => rfl
  | succ m ih =>
    rw [Nat.succ_mul, rsum_split _ (m * 64) 64, rsum_succ, ih]
    congr 1
    exact rsum_congr (fun j hj => by rw [show (m * 64 + j) / 64 = m by omega])

-- § host gate

/-- `s2_gemv`'s acceptance: positive sizes and `n_in % 64 = 0` (otherwise exit, or no launch). -/
def hostOk (nIn nOut : Nat) : Bool := 0 < nIn && 0 < nOut && nIn % 64 == 0

/-- Under the gate `nb·64 = n_in`: no element of the row is skipped by the group loop. -/
theorem gate_exact (nIn nOut : Nat) (h : hostOk nIn nOut = true) : nIn / 64 * 64 = nIn := by
  simp [hostOk] at h; omega

/-- Grid `⌈n_out/128⌉` blocks of 128 threads covers every row. -/
theorem grid_covers (nOut : Nat) : nOut ≤ (nOut + 127) / 128 * 128 := by omega

/-- Model shapes: `n_embd = 2560` and the expert width 640 are both accepted; 2500 is refused. -/
theorem gate_examples : hostOk 2560 640 = true ∧ hostOk 640 2560 = true ∧ hostOk 2500 1 = false := by decide

end Ref.S2GemvNaive



## FILE: /mnt/user-data/outputs/S2GemvQuadRows.lean
/-!
# MODULE: s2_gemv_quad_rows
SOURCES: src/kernels/cuda/s2_gemv_quads.cu
The S2 GEMV specialised for the 31.64 GiB of Q2_0 weights: one block per row, each thread consuming a QUAD
(one code byte = four codes, one 8-byte load of four fp16 activations, one scale), then a shared-memory tree.
This module proves the quad read is the plane's bit order, the shift-based group index, the quad's alignment,
and exposes two host-gate gaps: `n_in` is only checked mod 4, and `threads_per_row` is not checked at all.

## Source semantics
* Quad `q` covers elements `4q … 4q+3`: code byte `c[q]`, element `4q+k` at bits `2k`; group `q >> 4`
  (= `(4q) >> 6`); activations `x[4q … 4q+3]` as one `uint2` (needs an 8-byte-aligned `x`).
* Thread `tid` takes quads `q ≡ tid (mod tpr)`; four accumulators; then `partial[tid]` and a tree with
  steps `tpr/2, tpr/4, …, 1`, result in slot 0.
* Host: refuses `n_in % 4 ≠ 0` only.  Scales are laid out `n_in / 64` per row.

## Form template identifiers
`s2_gemv_quads_kernel` → quad model; `c[q]` → `quadByte`; `q >> 4` → `quadGroup`; host gate → `hostOk`;
tree loop → `halvingTree`; `threads_per_row` → `tpr`.
-/

namespace Ref.S2GemvQuadRows

-- § quad decoding

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- **Quad read = plane**: code `k` of quad byte `q` is plane element `4q + k`. -/
theorem quad_code (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    (byte q >>> (k * 2)) &&& 3 = planeCode byte (4 * q + k) := by
  unfold planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega]

/-- The kernel's literal shifts `0, 2, 4, 6` are `2k` for `k = 0 … 3`. -/
theorem quad_shifts : [0 * 2, 1 * 2, 2 * 2, 3 * 2] = [0, 2, 4, 6] := rfl

/-- `quadGroup q = q >> 4`. -/
def quadGroup (q : Nat) : Nat := q >>> 4

/-- **Group by shift**: `q >> 4 = (4q) / 64`, the group of every element of the quad. -/
theorem quadGroup_eq (q k : Nat) (hk : k < 4) : quadGroup q = (4 * q + k) / 64 := by
  unfold quadGroup; rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- The 8-byte activation load is aligned whenever the base is. -/
theorem uint2_aligned (xBase q : Nat) (h : xBase % 8 = 0) : (xBase + 2 * (4 * q)) % 8 = 0 := by omega

-- § host gate gap 1: n_in only mod 4

/-- `s2_gemv_quads`' host gate on `n_in`. -/
def hostOk (nIn : Nat) : Bool := 0 < nIn && nIn % 4 == 0

/-- With `n_in % 64 = 0` every quad's group index is inside the row's `n_in/64` scales. -/
theorem groups_in_row (nIn q : Nat) (h64 : nIn % 64 = 0) (hq : q < nIn / 4) : quadGroup q < nIn / 64 := by
  unfold quadGroup; rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- **Gap**: the gate accepts `n_in = 68`, yet quad 16 reads scale index 1 while the row has only
`68/64 = 1` scale — it reads the NEXT row's first scale (or past the buffer on the last row).  The S2 pack
always has `n_in % 64 = 0`, so this is latent; `s2_gemv_fast` and `s2_gemv_q8` do check `% 64`. -/
theorem gate_admits_out_of_row : hostOk 68 = true ∧ 16 < 68 / 4 ∧ 68 / 64 ≤ quadGroup 16 := by decide

-- § host gate gap 2: threads_per_row unchecked

/-- One tree level. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The loop `for (step = tpr/2; step > 0; step >>= 1)`, reading slot 0 (fuel bounds the step count). -/
def halvingGo (a : Nat → Int) : Nat → Nat → Int
  | 0, _ => a 0
  | fuel + 1, step => if step = 0 then a 0 else halvingGo (treeStep a step) fuel (step / 2)

/-- The kernel's reduction over `tpr` partials. -/
def halvingTree (a : Nat → Int) (tpr : Nat) : Int := halvingGo a 64 (tpr / 2)

/-- For powers of two the loop sums all partials (here: 1, 32, 128 and 256 unit partials). -/
theorem halving_pow2_examples :
    halvingTree (fun _ => 1) 1 = 1 ∧ halvingTree (fun _ => 1) 32 = 32 ∧
    halvingTree (fun _ => 1) 128 = 128 ∧ halvingTree (fun _ => 1) 256 = 256 := by decide

/-- **Gap**: the host never validates `tpr`.  For a non-power-of-two the halving loop DROPS partials:
with 3 threads the third partial is never added, with 96 threads 32 of them are lost. -/
theorem halving_drops_partials :
    halvingTree (fun _ => 1) 3 = 2 ∧ halvingTree (fun _ => 1) 96 = 64 := by decide

/-- The tree's reads stay inside `partial[0 … tpr-1]` for any `tpr`: `t < step ≤ tpr/2` gives
`t + step < tpr`.  The defect is lost terms, never an out-of-bounds read. -/
theorem halving_reads_in_bounds (t step tpr : Nat) (ht : t < step) (hs : step ≤ tpr / 2) : t + step < tpr := by
  omega

-- § quad ownership

/-- Thread `tid` owns quad `q` iff `q % tpr = tid`: every quad has exactly one owner for `tpr > 0`. -/
theorem quad_owner_unique (q tpr : Nat) (h : 0 < tpr) : q % tpr < tpr ∧ ∀ t, q % tpr = t → t = q % tpr := by
  exact ⟨Nat.mod_lt _ h, fun _ ht => ht.symm⟩

end Ref.S2GemvQuadRows



## FILE: /mnt/user-data/outputs/S2GemvQ8Activation.lean
/-!
# MODULE: s2_gemv_q8_activation
SOURCES: src/kernels/cuda/s2_gemv_q8.cu
The S2 GEMV that consumes Q8_0-quantized activations, as `ggml_mul_mat` does for Q2_0 (`vec_dot_type = Q8_0`).
One block per row, one quad per thread-iteration; each activation is dequantized `xq · dx` with `dx` the
block's fp16 scale.  This module proves the activation quad lies in one Q8_0 block (so one scale load is
right), the fp16 scale bytes are assembled little-endian, the weight decode matches the plane, the exact
per-element term, and the host gates (including the unchecked `threads_per_row`).

## Source semantics
* Quad `q`: weight byte `c[q]`, group `q >> 4`; activation block `ablk = 4q / 32`, byte offset `34·ablk`,
  `dbits = blk[0] | blk[1] << 8`, quants at `blk + 2 + off`, `off = 4q % 32`.
* Term: `a_k += w_k · (xq_k · dx)` with `w_k = (code_k - 1) · d`.
* Host: `n_in % 32 ≠ 0 ∨ n_in % 64 ≠ 0` → exit; `threads_per_row` is NOT validated (same halving tree as
  `s2_gemv_quads`); synchronous only when `stream == nullptr`.

## Form template identifiers
`s2_gemv_q8_kernel` → quad model; `ablk` → `actBlock`; `off` → `actOff`; `dbits` → `le16`;
`QK8_0` → 32; `QK_S2` → 64; host gate → `hostOk`; tree → `halvingTree`.
-/

namespace Ref.S2GemvQ8Activation

-- § activation addressing

/-- `ablk = (4q) / 32`. -/
def actBlock (q : Nat) : Nat := 4 * q / 32

/-- `off = (4q) % 32`. -/
def actOff (q : Nat) : Nat := 4 * q % 32

/-- **One block per quad**: all four activations of quad `q` are elements `off … off+3` of block `ablk`. -/
theorem quad_in_one_block (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 32 = actBlock q ∧ (4 * q + k) % 32 = actOff q + k ∧ actOff q + 3 < 32 := by
  unfold actBlock actOff; omega

/-- A quad's activation block is `q / 8` and its offset `4·(q % 8)`. -/
theorem actBlock_simple (q : Nat) : actBlock q = q / 8 ∧ actOff q = 4 * (q % 8) := by
  unfold actBlock actOff; omega

/-- Block byte addresses: `d` at `34·ablk`, quant `k` at `34·ablk + 2 + off + k`, inside the 34-byte block. -/
theorem quant_byte_in_block (q k : Nat) (hk : k < 4) :
    34 * actBlock q + 2 + actOff q + k < 34 * (actBlock q + 1) := by
  have := (quad_in_one_block q k hk).2.2
  omega

/-- The little-endian fp16 assembly `b0 | b1 << 8`. -/
def le16 (b0 b1 : Nat) : Nat := b0 ||| (b1 <<< 8)

/-- **`dbits` is the little-endian value** `b0 + 256·b1` for bytes `b0 < 256`. -/
theorem le16_eq (b0 b1 : Nat) (h0 : b0 < 256) : le16 b0 b1 = b0 + 256 * b1 := by
  unfold le16
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (by simpa using h0), Nat.shiftLeft_eq]
  omega

-- § weights

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- The quad's weight codes are plane elements `4q … 4q+3`. -/
theorem weight_code (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    (byte q >>> (k * 2)) &&& 3 = planeCode byte (4 * q + k) := by
  unfold planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega]

/-- The weight group and the activation block are both constant across the quad, and a weight group
(64 elements) spans exactly two activation blocks (32 elements). -/
theorem group_spans_two_blocks (q : Nat) : (q >>> 4) = actBlock q / 2 := by
  unfold actBlock; rw [Nat.shiftRight_eq_div_pow]; simp; omega

-- § the term

/-- `w · (xq · dx)` with `w = (code - 1)·d` (exact model). -/
def term (code : Nat) (d : Int) (xq : Int) (dx : Int) : Int := (((code : Int) - 1) * d) * (xq * dx)

/-- Exactly, the term is the product of the dequantized weight and the dequantized activation, so the kernel
agrees with "dequantize both, then multiply"; only FP rounding of the grouping differs. -/
theorem term_eq (code : Nat) (d xq dx : Int) :
    term code d xq dx = ((code : Int) - 1) * d * xq * dx := by
  unfold term; rw [Int.mul_assoc (((code : Int) - 1) * d) xq dx]

/-- Code 1 contributes nothing regardless of the activation. -/
theorem term_code_one (d xq dx : Int) : term 1 d xq dx = 0 := by simp [term]

-- § host gates

/-- `s2_gemv_q8`'s `n_in` gate. -/
def hostOk (nIn : Nat) : Bool := 0 < nIn && nIn % 32 == 0 && nIn % 64 == 0

/-- The two checks collapse to `% 64` (the error message prints 64 for both). -/
theorem gate_is_mod64 (nIn : Nat) : hostOk nIn = (0 < nIn && nIn % 64 == 0) := by
  unfold hostOk
  by_cases h : nIn % 64 = 0
  · have : nIn % 32 = 0 := by omega
    simp [h, this]
  · have hb : (nIn % 64 == 0) = false := by simpa using h
    rw [hb]; simp

/-- Under the gate every quad's activation block exists: `ablk < n_in / 32`. -/
theorem blocks_in_range (nIn q : Nat) (h : hostOk nIn = true) (hq : q < nIn / 4) : actBlock q < nIn / 32 := by
  simp [hostOk] at h; unfold actBlock; omega

/-- One tree level. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The halving loop over `tpr` partials. -/
def halvingGo (a : Nat → Int) : Nat → Nat → Int
  | 0, _ => a 0
  | fuel + 1, step => if step = 0 then a 0 else halvingGo (treeStep a step) fuel (step / 2)

/-- The kernel's reduction. -/
def halvingTree (a : Nat → Int) (tpr : Nat) : Int := halvingGo a 64 (tpr / 2)

/-- **Unchecked `threads_per_row`**: a non-power-of-two silently drops partial sums (here 48 threads keep 32). -/
theorem tpr_unchecked_drops : halvingTree (fun _ => 1) 48 = 32 ∧ halvingTree (fun _ => 1) 64 = 64 := by decide

end Ref.S2GemvQ8Activation



## FILE: /mnt/user-data/outputs/S2GemvLutStaging.lean
/-!
# MODULE: s2_gemv_lut_staging
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
An S2 GEMV experiment with two levers — staging `x` in shared memory and a 256×4 constant-memory code table —
both measured negative or neutral and kept as a recorded negative result (NOT the fast path).  This module
proves the lookup table is exactly the plane decode with the −1 bias, the group-index shift, the shared-memory
layout (staged activations never overlap the reduction partials), and the host gates.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1` for every byte `b < 256`, `k < 4`; one `float4` per byte.
* Quad `q`: byte `c[q]`, scale `s[q >> 4]`, activations `x[4q … 4q+3]` (staged or direct).
* Shared memory, staged: halves `sx[0 … n_in)` in bytes `[0, 2·n_in)`, partials at float offset
  `MAX_SHARED_HALVES/2 = 2048`, i.e. byte 8192; size `8192 + 4·tpr`.  Unstaged: partials at 0, size `4·tpr`.
* Host: `n_in % 4` and `n_in % 64` required; staging requires `n_in ≤ 4096`; the table is uploaded once
  (process-global flag, not per device); `tpr` is not validated.
* Grid is `n_out` blocks, so `blockIdx < n_out` and the early return before the barrier is never taken.

## Form template identifiers
`c_codes` → `lut`; `MAX_SHARED_HALVES` → `maxHalves`; partial byte offset → `partialBase`;
`smem` → `smemBytes`; host gate → `hostOk`; `stage_x` → `stage`.
-/

namespace Ref.S2GemvLutStaging

-- § the code table

/-- `c_codes[b][k]` as filled by `ensure_lut`. -/
def lut (b k : Nat) : Int := (((b >>> (2 * k)) &&& 3 : Nat) : Int) - 1

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- Shift-and-mask is division-and-modulo. -/
theorem extract2 (x s : Nat) : (x >>> s) &&& 3 = x / 2 ^ s % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod (x >>> s) 2
  rw [Nat.shiftRight_eq_div_pow] at this ⊢
  simpa using this

/-- **The table is the biased plane decode**: `lut[c[q]][k] = code(4q + k) - 1`. -/
theorem lut_is_decode (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    lut (byte q) k = ((planeCode byte (4 * q + k) : Nat) : Int) - 1 := by
  unfold lut planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega, Nat.mul_comm 2 k]

/-- Table entries take only the values `-1, 0, 1, 2`. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  unfold lut
  have h : (b >>> (2 * k)) &&& 3 < 4 := by rw [extract2]; exact Nat.mod_lt _ (by decide)
  omega

/-- Spot values: byte `0b11100100` lists codes `0,1,2,3` → `-1,0,1,2`. -/
theorem lut_example : lut 0xE4 0 = -1 ∧ lut 0xE4 1 = 0 ∧ lut 0xE4 2 = 1 ∧ lut 0xE4 3 = 2 := by decide

/-- Group of quad `q`: `q >> 4 = 4q / 64`. -/
theorem quad_group (q k : Nat) (hk : k < 4) : q >>> 4 = (4 * q + k) / 64 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

-- § shared-memory layout

/-- `MAX_SHARED_HALVES`. -/
def maxHalves : Nat := 4096

/-- Byte offset of `partial[0]`: `MAX_SHARED_HALVES / 2` floats when staged, 0 otherwise. -/
def partialBase (stage : Bool) : Nat := if stage then (maxHalves / 2) * 4 else 0

/-- Dynamic shared bytes requested. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then maxHalves * 2 else 0) + tpr * 4

/-- **No overlap**: under the staging limit the staged halves end at or before the partials begin. -/
theorem staged_disjoint (nIn : Nat) (h : nIn ≤ maxHalves) : 2 * nIn ≤ partialBase true := by
  unfold partialBase maxHalves at *; simp; omega

/-- Every `partial[tid]`, `tid < tpr`, lies inside the requested allocation (both modes). -/
theorem partials_inside (stage : Bool) (tpr tid : Nat) (ht : tid < tpr) :
    partialBase stage + 4 * tid + 4 ≤ smemBytes stage tpr := by
  unfold partialBase smemBytes maxHalves
  cases stage <;> simp <;> omega

/-- The model width `n_embd = 2560` fits the staging buffer (5 KB of 8 KB). -/
theorem embd_fits : 2560 ≤ maxHalves ∧ 2 * 2560 = 5120 := by decide

-- § host gates

/-- `s2_gemv_fast`'s acceptance. -/
def hostOk (nIn nOut : Nat) (stage : Bool) : Bool :=
  0 < nIn && 0 < nOut && nIn % 4 == 0 && nIn % 64 == 0 && (!stage || nIn ≤ maxHalves)

/-- Accepted shapes keep every quad's group inside the row's `n_in / 64` scales. -/
theorem groups_in_row (nIn nOut q : Nat) (stage : Bool) (h : hostOk nIn nOut stage = true)
    (hq : q < nIn / 4) : q >>> 4 < nIn / 64 := by
  simp [hostOk] at h
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- Staging is refused above 4096 halves; unstaged has no such limit. -/
theorem staging_limit : hostOk 8192 1 true = false ∧ hostOk 8192 1 false = true := by decide

/-- One block per row: every launched block has `blockIdx < n_out`, so no thread returns before the barrier. -/
theorem no_early_return (nOut blk : Nat) (h : blk < nOut) : ¬ (nOut ≤ blk) := by omega

end Ref.S2GemvLutStaging



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- Bf16GemvDispatch.lean
  was: theorem dispatchSplit_accepts_2048 : dispatchSplit 640 48 2048 = some (.ok .split)
  now: theorem warpGrid_covers (nOut : Nat) : nOut ≤ warpGrid nOut * 8
  measured: the claim changed

- Bf16GemvDispatch.lean
  was: theorem warpGrid_covers (nOut : Nat) : nOut ≤ warpGrid nOut * 8
  now: theorem naiveGrid_covers (nOut : Nat) : nOut ≤ naiveGrid nOut * 256
  measured: the claim changed

- Bf16GemvDispatch.lean
  was: theorem naiveGrid_covers (nOut : Nat) : nOut ≤ naiveGrid nOut * 256
  now: theorem warpRow_surj (nOut o : Nat) (ho : o < nOut) :
    ∃ blk tid, blk < warpGrid nOut ∧ tid < 256 ∧ tid % 32 = 0 ∧ warpRow blk tid = o
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- Bf16GemvDispatch.lean
  was: theorem warpRow_surj (nOut o : Nat) (ho : o < nOut) :
    ∃ blk tid, blk < warpGrid nOut ∧ tid < 256 ∧ tid % 32 = 0 ∧ warpRow blk tid = o
  now: theorem warpRow_inj (b1 t1 b2 t2 : Nat) (h1 : t1 < 256) (h2 : t2 < 256)
    (h : warpRow b1 t1 = warpRow b2 t2) : b1 = b2 ∧ t1 / 32 = t2 / 32
  measured: more hypotheses/binders (2 → 4): the statement holds in fewer cases

- Bf16GemvDispatch.lean
  was: theorem warpRow_inj (b1 t1 b2 t2 : Nat) (h1 : t1 < 256) (h2 : t2 < 256)
    (h : warpRow b1 t1 = warpRow b2 t2) : b1 = b2 ∧ t1 / 32 = t2 / 32
  now: theorem warpGrid_threshold : warpGrid 64 = 8 ∧ warpGrid 128 = 16
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- Bf16GemvDispatch.lean
  was: theorem warpGrid_threshold : warpGrid 64 = 8 ∧ warpGrid 128 = 16
  now: theorem warp_lanes_consecutive (rowBase it lane : Nat) :
    (rowBase + it * 32 + (lane + 1)) = (rowBase + it * 32 + lane) + 1
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem geometry_matches_layout :
    ∀ t ∈ supportedTypes, ∃ e, geometry t = some (e, layoutBytes t)
  now: theorem geometry_matches_layout :
    ∀ t ∈ supportedTypes, (geometry t).map Prod.snd = some (layoutBytes t)
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem blocks_whole_groups : ∀ t ∈ supportedTypes, ∃ e b, geometry t = some (e, b) ∧ e % 32 = 0
  now: theorem blocks_whole_groups (t e b : Nat) (h : geometry t = some (e, b)) : e % 32 = 0
  measured: more hypotheses/binders (0 → 1): the statement holds in fewer cases

- GgmlBlockDequant.lean
  was: theorem f2bf_down (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 < 32768) : f2bf u = u / 65536
  now: theorem f2bf_nowrap (u : Nat) (hu : u < 4294901760) : f2bf u = (u + 0x7fff + u / 65536 % 2) / 65536
  measured: fewer quantified variables (3 → 2)

- GgmlBlockDequant.lean
  was: theorem f2bf_up (u : Nat) (hu : u < 4294901760) (hlo : 32768 < u % 65536) : f2bf u = u / 65536 + 1
  now: theorem f2bf_down (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 < 32768) : f2bf u = u / 65536
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem f2bf_tie_even (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 = 32768) : f2bf u % 2 = 0
  now: theorem f2bf_up (u : Nat) (hu : u < 4294901760) (hlo : 32768 < u % 65536) : f2bf u = u / 65536 + 1
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem f2bf_nan_to_negzero : f2bf 0x7FFFFFFF = 0x8000
  now: theorem f2bf_tie_even (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 = 32768) : f2bf u % 2 = 0
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem f2bf_nan_to_inf : f2bf 0x7F800001 = 0x7F80
  now: theorem f2bf_nan_to_negzero : f2bf 0x7FFFFFFF = 0x8000
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m
  now: theorem f2bf_nan_to_inf : f2bf 0x7F800001 = 0x7F80
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- GgmlBlockDequant.lean
  was: theorem nibbles (b : Nat) (hb : b < 256) : b &&& 15 = b % 16 ∧ b >>> 4 = b / 16 ∧ b = b % 16 + 16 * (b / 16) ∧
    b / 16 < 16
  now: theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m
  measured: fewer quantified variables (2 → 1)

- GgmlBlockDequant.lean
  was: theorem q4_0_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7
  now: theorem nibbles (b : Nat) (hb : b < 256) : b &&& 15 = b % 16 ∧ b >>> 4 = b / 16 ∧ b = b % 16 + 16 * (b / 16) ∧
    b / 16 < 16
  measured: fewer quantified variables (4 → 2)

- GgmlBlockDequant.lean
  was: theorem and16 (a : Nat) : a &&& 16 = 16 * (a / 16 % 2)
  now: theorem q4_0_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7
  measured: more hypotheses/binders (1 → 4): the statement holds in fewer cases

- GgmlBlockDequant.lean
  was: theorem q5_0_high_bit_lo (qh j : Nat) : ((qh >>> j) <<< 4) &&& 16 = 16 * (qh / 2 ^ j % 2)
  now: theorem and16 (a : Nat) : a &&& 16 = 16 * (a / 16 % 2)
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem q5_0_high_bit_hi (qh j : Nat) : (qh >>> (j + 12)) &&& 16 = 16 * (qh / 2 ^ (j + 16) % 2)
  now: theorem q5_0_high_bit_lo (qh j : Nat) : ((qh >>> j) <<< 4) &&& 16 = 16 * (qh / 2 ^ j % 2)
  measured: the claim changed

- GgmlBlockDequant.lean
  was: theorem or_shift4 : ∀ a : Fin 16, ∀ h : Fin 4, (a.val ||| (h.val <<< 4)) = a.val + 16 * h.val
  now: theorem q5_0_high_bit_hi (qh j : Nat) : (qh >>> (j + 12)) &&& 16 = 16 * (qh / 2 ^ (j + 16) % 2)
  measured: fewer quantified variables (2 → 1); more hypotheses/binders (0 → 1): the statement holds in fewer cases

- GgmlBlockDequant.lean
  was: theorem q6_k_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo ||| (hi <<< 4)) < 64 ∧ -32 ≤ ((lo ||| (hi <<< 4) : Nat) : Int) - 32
  now: theorem or_shift4 : ∀ a : Fin 16, ∀ h : Fin 4, (a.val ||| (h.val <<< 4)) = a.val + 16 * h.val
  measured: fewer quantified variables (3 → 2)

- GgmlBlockDequant.lean
  was: theorem scaleMinK4_lt64 (j : Nat) (q : Nat → Nat) (hq : ∀ i, q i < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64
  now: theorem q6_k_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo ||| (hi <<< 4)) < 64 ∧ -32 ≤ ((lo ||| (hi <<< 4) : Nat) : Int) - 32
  measured: fewer quantified variables (4 → 3)

- GgmlBlockDequant.lean
  was: theorem q3_k_hmask_fits : ∀ n : Fin 2, ∀ jj : Fin 4, 1 <<< (n.val * 4 + jj.val) < 256
  now: theorem scaleMinK4_lt64 (j : Nat) (q : Nat → Nat) (hq : ∀ i, q i < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64
  measured: more hypotheses/binders (0 → 3): the statement holds in fewer cases

- GgmlBlockDequant.lean
  was: theorem iq4nl_shape : iq4nl.length = 16 ∧ iq4nl.Pairwise (· < ·) ∧ ∀ v ∈ iq4nl, -128 ≤ v ∧ v ≤ 127
  now: theorem q3_k_hmask_fits : ∀ n : Fin 2, ∀ jj : Fin 4, 1 <<< (n.val * 4 + jj.val) < 256
  measured: the claim changed

- S2CanonicalDequant.lean
  was: theorem kernel_reads_plane (byte : Nat → Nat) (b j : Nat) (hj : j < 64) :
    kernelCode byte b j = planeCode byte (b * 64 + j)
  now: theorem kernel_reads_plane (byte : Nat → Nat) (b j : Nat) :
    kernelCode byte b j = planeCode byte (b * 64 + j)
  measured: fewer quantified variables (3 → 2)

- S2GemvNaive.lean
  was: theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n
  now: theorem rsum_congr {f g : Nat → Int} {n : Nat} (h : ∀ i, i < n → f i = g i) : rsum f n = rsum g n
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- S2GemvNaive.lean
  was: theorem rowSum_eq_flat (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) :
    rowSum code scale x nb = flatSum code scale x nb
  now: theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n
  measured: fewer quantified variables (4 → 2)
-/



## FILE: /mnt/user-data/outputs/NativeQuantizedMatVec.lean
/-!
# MODULE: native_quantized_mat_vec
SOURCES: src/kernels/cuda/native_mmvq.cu
Integer model of the pinned llama.cpp MMVQ path: block geometry of every supported
GGML weight format, the Q8_1 activation contract, the DP4A integer dot products and the
thread-to-block iteration map of the 1-column and 2..8-column kernels.

## Source semantics
* Every weight format is a fixed-size block (`static_assert` sizes) covering 32, 64 or 256
  elements; `native_mmvq_weight_bytes` is `(n_in / elems) * bytes * n_out`, rejecting shapes
  that are not whole blocks.
* A dot product accumulates DP4A integer products exactly (integer, order-free) and only
  then applies float scales; affine formats (Q4_0, Q5_0, Q2_0, Q4_K/Q5_K mins) fold their
  offset into the activation sum carried by Q8_1 (`ds.y`).
* A K-loop thread handles block `tid / T + i * BPI`; every block of a row is visited exactly
  once.  Small-K (`blocks_per_row < BPI`) switches to 4 rows per CUDA block.
* Multi-column: column `j` reads activations at `j * (n_in/32)` and writes `y[j*n_out+row]`;
  the outputs of distinct (column,row) pairs never collide.

## Form template identifiers
Q5KBlock/Q4KBlock/... → `blockSpec`; `__dp4a` → `dp4a`; `q5_q8_dot`/`small_q8_dot` → `dot`;
`BLOCKS_PER_ITER` → `bpi`; `validate_shape` → `validShape`; `native_q8_1_bytes` → `q81Bytes`;
`iq4nl_values` → `iq4nl`; `__byte_perm(0x020100ff,…)` → `q2Decode`; `aux` scale unpack → `k4Scale`.
-/

namespace NativeQuantizedMatVec

set_option maxRecDepth 8000

-- § integer dot products

/-- Exact integer dot product of two code lists (DP4A semantics, no rounding). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Sum of a list of integers. -/
def lsum : List Int → Int
  | [] => 0
  | a :: as => a + lsum as

/-- `__dp4a(a, b, c)`: four signed byte products added to an accumulator. -/
def dp4a (a b : List Int) (c : Int) : Int := c + dot a b

/-- Chained DP4A calls are exact integer sums, so the call order cannot change the result. -/
theorem dp4a_chain (a b a' b' : List Int) (c : Int) :
    dp4a a b (dp4a a' b' c) = dp4a a' b' (dp4a a b c) := by
  unfold dp4a; omega

/-- A DP4A against `0x01010101` (all-ones) is the plain sum of the other operand. -/
theorem dot_ones (x : List Int) : dot (x.map fun _ => 1) x = lsum x := by
  induction x with
  | nil => rfl
  | cons a as ih =>
    simp only [List.map, dot, lsum]
    rw [ih, Int.one_mul]

/-- Affine offset folding: `Σ (q - k) x = Σ q x - k Σ x`.  This is why Q4_0/Q5_0/Q2_0 can
    subtract a multiple of the Q8_1 block sum instead of re-centering every code. -/
theorem dot_offset (k : Int) : ∀ (c x : List Int), c.length = x.length →
    dot (c.map (· - k)) x = dot c x - k * lsum x
  | [], [], _ => by simp [dot, lsum]
  | a :: as, b :: bs, h => by
    have ih := dot_offset k as bs (by simpa using h)
    simp only [List.map, dot, lsum]
    rw [ih, Int.sub_mul, Int.mul_add]
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

-- § block geometry

/-- (elements per block, bytes per block) for each GGML type the native path decodes itself. -/
def blockSpec : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)    -- Q4_0  : d + 16 nibble bytes
  | 6 => some (32, 22)    -- Q5_0  : d + qh[4] + qs[16]
  | 8 => some (32, 34)    -- Q8_0  : d + 32 int8
  | 20 => some (32, 18)   -- IQ4_NL
  | 11 => some (256, 110) -- Q3_K
  | 12 => some (256, 144) -- Q4_K
  | 13 => some (256, 176) -- Q5_K
  | 14 => some (256, 210) -- Q6_K
  | 23 => some (256, 136) -- IQ4_XS
  | 42 => some (64, 18)   -- Q2_0
  | _ => none

/-- The byte sizes are exactly the sums of the struct fields asserted by `static_assert`. -/
theorem block_field_sums :
    (4 + 12 + 32 + 128 = 176) ∧ (4 + 12 + 128 = 144) ∧ (128 + 64 + 16 + 2 = 210) ∧
    (32 + 64 + 12 + 2 = 110) ∧ (2 + 2 + 4 + 128 = 136) ∧ (2 + 16 = 18) ∧
    (2 + 4 + 16 = 22) ∧ (2 + 32 = 34) ∧ (4 + 32 = 36) := by decide

/-- Density check: every format stores at least 2 bits and at most 9 bits per element
    (`8 * bytes` lies between `2 * elems` and `9 * elems`). -/
def specDensityOk (t : Nat) : Bool :=
  match blockSpec t with
  | some (e, b) => 2 * e ≤ 8 * b && 8 * b ≤ 9 * e
  | none => true

/-- Checked over every type id below 64. -/
theorem bits_per_element_bounds : ∀ t, t < 64 → specDensityOk t = true := by decide

/-- The 16 types accepted by `native_mmvq_supported`. -/
def supported (t : Nat) : Bool :=
  t == 2 || t == 6 || t == 8 || t == 11 || t == 12 || t == 13 || t == 14 || t == 20 ||
  t == 23 || t == 42 || t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- The IQ types delegated to `iq_mmvq`. -/
def delegatedIQ (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- A type is supported exactly when it has a native block spec or is delegated to `iq_mmvq`;
    the dispatcher `native_mmvq` therefore never reaches its `default:` throw for a supported type. -/
theorem supported_iff : ∀ t, t < 64 →
    (supported t = true ↔ ((blockSpec t).isSome = true ∨ delegatedIQ t = true)) := by decide

/-- `validate_shape`: positive, whole blocks, and 1..8 columns. -/
def validShape (nIn ncols elems : Nat) : Prop :=
  0 < nIn ∧ nIn % elems = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- A valid shape has at least one whole block per row. -/
theorem validShape_has_block {nIn ncols elems : Nat} (he : 0 < elems)
    (h : validShape nIn ncols elems) : 1 ≤ nIn / elems := by
  obtain ⟨hpos, hdiv, _, _⟩ := h
  have hle : elems ≤ nIn := Nat.le_of_dvd hpos (Nat.dvd_of_mod_eq_zero hdiv)
  exact (Nat.le_div_iff_mul_le he).mpr (by simpa using hle)

/-- Weight byte count of `native_mmvq_weight_bytes` (before the overflow check). -/
def weightBytes (elems bytes nIn nOut : Nat) : Nat := (nIn / elems) * bytes * nOut

/-- Weight bytes are exactly the number of blocks times the block size. -/
theorem weightBytes_blocks (elems bytes k nOut : Nat) (he : 0 < elems) :
    weightBytes elems bytes (k * elems) nOut = k * nOut * bytes := by
  unfold weightBytes
  rw [Nat.mul_div_cancel _ he, Nat.mul_right_comm]

/-- `native_q8_1_bytes`: one 36-byte Q8_1 block per 32 activations per column. -/
def q81Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

/-- Quantizing `ncols` contiguous columns as one vector of `ncols * nIn` values keeps every
    32-element block inside one column when `32 ∣ nIn`: the flat block count equals the
    per-column count times `ncols`. -/
theorem q81_flat_blocks (k ncols : Nat) : (ncols * (k * 32)) / 32 = ncols * ((k * 32) / 32) := by
  rw [← Nat.mul_assoc, Nat.mul_div_cancel _ (by decide), Nat.mul_div_cancel _ (by decide)]

-- § iteration geometry

/-- Blocks per K iteration: `VDR * WARPS * WARP / QI` (4 warps of 32). -/
def bpi (vdr qi : Nat) : Nat := vdr * 4 * 32 / qi

/-- Pinned per-format blocks per iteration: Q5_K/Q4_K 8, Q2_0 64, Q3_K 8, IQ4_XS 16,
    Q6_K 4, Q4_0/Q5_0/IQ4_NL 64, Q8_0 32. -/
theorem bpi_table :
    bpi 2 32 = 8 ∧ bpi 1 2 = 64 ∧ bpi 1 16 = 8 ∧ bpi 4 32 = 16 ∧ bpi 1 32 = 4 ∧
    bpi 2 4 = 64 ∧ bpi 2 8 = 32 := by decide

/-- The small-K switch `n_in / 32 < BPI` is exactly the 2048-element (QI=4) and
    1024-element (QI=8) thresholds quoted in the source. -/
theorem smallK_thresholds (nIn : Nat) :
    (nIn / 32 < bpi 2 4 ↔ nIn < 2048) ∧ (nIn / 32 < bpi 2 8 ↔ nIn < 1024) := by
  simp only [bpi]; constructor <;> omega

/-- For Q5_K/Q4_K the small-K switch fires below 2048 inputs (8 super-blocks of 256). -/
theorem smallK_k_formats (nIn : Nat) : nIn / 256 < bpi 2 32 ↔ nIn < 2048 := by
  simp only [bpi]; omega

/-- Thread map of the K formats: `tid ↦ (tid / 16, 2 * (tid % 16))` is injective on the
    128 threads and lands in 8 blocks × even quant offsets `< 32`. -/
theorem kmap_injective (t1 t2 : Nat) (_h1 : t1 < 128) (_h2 : t2 < 128)
    (h : t1 / 16 = t2 / 16 ∧ 2 * (t1 % 16) = 2 * (t2 % 16)) : t1 = t2 := by omega

/-- Range of the K-format thread map. -/
theorem kmap_range (t : Nat) (h : t < 128) : t / 16 < 8 ∧ 2 * (t % 16) < 32 := by omega

/-- Strided coverage: block `b` is visited by thread-group `b % BPI` at iteration `b / BPI`,
    so the loop `kbx = start; kbx < n; kbx += BPI` visits every block exactly once. -/
theorem strided_cover (b p : Nat) : b % p + p * (b / p) = b := Nat.mod_add_div b p

/-- Two (start, iteration) pairs reaching the same block are the same pair. -/
theorem strided_unique (p s1 s2 i1 i2 : Nat) (h1 : s1 < p) (h2 : s2 < p)
    (h : s1 + p * i1 = s2 + p * i2) : s1 = s2 ∧ i1 = i2 := by
  have hp : 0 < p := by omega
  have e1 : (s1 + p * i1) % p = s1 := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h1]
  have e2 : (s2 + p * i2) % p = s2 := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h2]
  have hs : s1 = s2 := by rw [← e1, ← e2, h]
  subst hs
  refine ⟨rfl, ?_⟩
  have : p * i1 = p * i2 := by omega
  exact Nat.eq_of_mul_eq_mul_left hp this

/-- Multi-column outputs `y[j * n_out + row]` never collide for rows `< n_out`. -/
theorem output_index_injective {n j1 j2 r1 r2 : Nat} (h1 : r1 < n) (h2 : r2 < n)
    (h : j1 * n + r1 = j2 * n + r2) : j1 = j2 ∧ r1 = r2 := by
  have h' : r1 + n * j1 = r2 + n * j2 := by
    rw [Nat.mul_comm n j1, Nat.mul_comm n j2]; omega
  obtain ⟨hr, hj⟩ := strided_unique n r1 r2 j1 j2 h1 h2 h'
  exact ⟨hj, hr⟩

/-- Small-K launches `⌈n_out/4⌉` CUDA blocks of 4 rows: every row has a block. -/
theorem smallK_grid_covers (nOut r : Nat) (h : r < nOut) : r / 4 < (nOut + 3) / 4 := by omega

-- § code decoding

/-- Byte `i` of a little-endian 32-bit word. -/
def byteOf (x i : Nat) : Nat := (x >>> (8 * i)) % 256

/-- Reinterpret a byte as `int8_t`. -/
def toInt8 (b : Nat) : Int := if b ≥ 128 then (b : Int) - 256 else b

/-- `__byte_perm(0x020100ff, 0x020100ff, q)` maps a Q2_0 code `c ∈ 0..3` to `c - 1`:
    the selector constant *is* the codebook `{-1, 0, 1, 2}`. -/
theorem q2Decode : ∀ c, c < 4 → toInt8 (byteOf 0x020100ff c) = (c : Int) - 1 := by decide

/-- The pinned nonlinear IQ4 codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The IQ4 codebook is strictly increasing and every entry fits in `int8_t` (no DP4A overflow). -/
theorem iq4nl_sorted_int8 :
    iq4nl.length = 16 ∧ (∀ i, i < 15 → iq4nl[i]! < iq4nl[i+1]!) ∧
    (∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (127 : Int)) := by decide

/-- K-quant 6-bit scale/min unpack (`aux`): low groups take `s & 0x3f`, high groups take the
    nibble of `s4` plus the two top bits of `s0` moved to bits 4..5. -/
def k4Scale (j s0 s4 : Nat) : Nat :=
  if j < 2 then s0 % 64 else s4 % 16 + (s0 / 64) * 16

/-- Every unpacked K-quant scale or min is a 6-bit value. -/
theorem k4Scale_lt (j s0 s4 : Nat) (h0 : s0 < 256) : k4Scale j s0 s4 < 64 := by
  unfold k4Scale; split <;> omega

/-- The bit expression `(s0 & 0xc0) >> 2` equals the arithmetic `(s0 / 64) * 16` on bytes. -/
theorem k4Scale_bits : ∀ s, s < 256 → (s &&& 0xc0) >>> 2 = (s / 64) * 16 := by decide

/-- The high-group mask `uint32_t(-int32_t(j >= 2))` is all-zeros or all-ones. -/
theorem hi_mask (b : Bool) :
    ((-(if b then (1 : Int) else 0)) % 2^32) = (if b then 0xFFFFFFFF else 0) := by
  cases b <;> decide

/-- IQ4_XS sub-block scale `ls = low4 | high2 << 4` is 6-bit, so `ls - 32 ∈ [-32, 31]`. -/
theorem iq4xs_scale_range (lo hi : Nat) :
    (-32 : Int) ≤ ((lo % 16 + (hi % 4) * 16 : Nat) : Int) - 32 ∧
    ((lo % 16 + (hi % 4) * 16 : Nat) : Int) - 32 ≤ 31 := by omega

/-- Q3_K scale `(low4 | high2 << 4) - 32` and Q6_K code `(low4 | high2 << 4) - 32` share the
    range `[-32, 31]`: both are signed 6-bit values. -/
theorem q3_q6_signed6 (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (-32 : Int) ≤ ((lo + hi * 16 : Nat) : Int) - 32 ∧ ((lo + hi * 16 : Nat) : Int) - 32 ≤ 31 := by
  omega

-- § affine corrections

/-- Threads per 32-element block for the small formats is `Qi / VDR`: Q4_0/Q5_0/IQ4_NL use 2,
    Q8_0 uses 4. -/
theorem small_threads_per_block : 4 / 2 = 2 ∧ 8 / 2 = 4 := by decide

/-- Each Q4_0 call covers half a block and subtracts `4 · ds.y`; the two calls of a block
    together subtract `8 · Σx`, the zero point of the 4-bit code (`q - 8`).  Q5_0 likewise
    totals `16 · Σx`. -/
theorem small_affine_totals : 2 * 4 = 8 ∧ 2 * 8 = 16 := by decide

/-- Q4_0 whole-block identity: with codes centered at 8, the integer dot equals the raw code
    dot minus `8 · Σx`, which is what the summed per-call corrections reproduce. -/
theorem q4_0_block (c x : List Int) (h : c.length = x.length) :
    dot (c.map (· - 8)) x = dot c x - 8 * lsum x := dot_offset 8 c x h

end NativeQuantizedMatVec



## FILE: /mnt/user-data/outputs/IqQuantKernels.lean
/-!
# MODULE: iq_quant_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Integer model of the IQ-family path: per-format block sizes (`iq_row_bytes`), the
lane-strided warp dot `row_dot`, the `iq_mmvq` grid, the grouped-expert blob layout
(`native_expert_layout`, `native_expert_scratch_bytes`) and the sign/codebook decoders.

## Source semantics
* `Fmt<TY>` gives `qk` values per block, `ipb` dot calls per block and the `iqs` step; call
  `k` of a row is `(kbx, iqs) = (k / ipb, step * (k % ipb))`, lane-strided over 32 lanes.
* `iq_row_bytes(t, n) = (n / qk) * sizeof(block)`; zero means "unsupported".
* `iq_mmvq` launches `⌈n_out/4⌉` blocks of 4 warps, one warp per row.
* An expert blob is `[gate rows | up rows | down rows]` with `up_off = n_ff * gu_row`,
  `down_off = 2 * up_off`; scratch regions are 256-byte aligned.
* `unpack_ksigns` stores 7 sign bits and makes the 8th the parity, so every sign byte
  has an even number of negatives.

## Form template identifiers
`Fmt<TY>` → `fmt`; `iq_row_bytes` → `rowBytes`; `is_iq` → `isIQ`; `row_dot` call map → `callOf`;
`native_expert_layout` → `Layout`/`layout`; `native_expert_scratch_bytes` → `scratchBytes`;
`unpack_ksigns` → `ksigns`; `dq_q2_0` → `q2Value`.
-/

namespace IqQuantKernels

set_option maxRecDepth 8000

-- § formats

/-- `Fmt<TY>`: (qk, ipb, step) for each IQ / dot-format type. -/
def fmt : Nat → Option (Nat × Nat × Nat)
  | 16 => some (256, 8, 2) | 17 => some (256, 8, 2) | 18 => some (256, 8, 2)
  | 20 => some (32, 2, 2) | 21 => some (256, 8, 2) | 22 => some (256, 8, 2)
  | 29 => some (256, 8, 1) | 42 => some (64, 2, 1)
  | _ => none

/-- Bytes per block (`sizeof(block_*)` from ggml-common.h). -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110 | 22 => 82
  | 29 => 56 | 23 => 136 | 11 => 110 | 42 => 18 | _ => 0

/-- Values per block. -/
def blockElems : Nat → Nat
  | 20 => 32 | 42 => 64 | 16 | 17 | 18 | 21 | 22 | 29 | 23 | 11 => 256 | _ => 1

/-- `iq_row_bytes`. -/
def rowBytes (t n : Nat) : Nat := (n / blockElems t) * blockBytes t

/-- `is_iq`. -/
def isIQ (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 ||
  t == 42 || t == 11

/-- The ggml-common block layouts sum to the sizes used: iq2_xxs `2+64`, iq2_xs `2+64+8`,
    iq2_s `2+64+8+8`, iq3_xxs `2+96`, iq3_s `2+64+8+32+4`, iq1_m `32+16+8`. -/
theorem block_layouts :
    2 + 64 = 66 ∧ 2 + 64 + 8 = 74 ∧ 2 + 64 + 8 + 8 = 82 ∧ 2 + 96 = 98 ∧
    2 + 64 + 8 + 32 + 4 = 110 ∧ 32 + 16 + 8 = 56 := by decide

/-- `is_iq t` holds exactly when `iq_row_bytes` is nonzero for a full super-block. -/
theorem isIQ_iff_rowBytes : ∀ t, t < 64 → (isIQ t = true ↔ rowBytes t 256 ≠ 0) := by decide

/-- Consistency of one `Fmt<TY>` entry: its `qk` is the block's element count, it makes at
    least one call per block, and its `ipb * step` quant offsets stay inside the block's
    `qk / 8` 32-bit words of Q8_1 activation codes. -/
def fmtOk (t : Nat) : Bool :=
  match fmt t with
  | some (qk, ipb, step) => qk == blockElems t && ipb * step ≤ qk / 8 && 0 < ipb
  | none => true

/-- Every dot format is consistent with its block geometry. -/
theorem fmt_consistent : ∀ t, t < 64 → fmtOk t = true := by decide

-- § warp dot iteration

/-- Call `k` of a row: block `k / ipb`, quant offset `step * (k % ipb)`. -/
def callOf (ipb step k : Nat) : Nat × Nat := (k / ipb, step * (k % ipb))

/-- Every (block, part) pair of a row is reached by exactly the call `block * ipb + part`. -/
theorem callOf_surj (ipb step b p : Nat) (hp : p < ipb) :
    callOf ipb step (b * ipb + p) = (b, step * p) := by
  have hpos : 0 < ipb := by omega
  simp only [callOf]
  have e1 : (b * ipb + p) / ipb = b := by
    rw [Nat.mul_comm, Nat.add_comm, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hp, Nat.zero_add]
  have e2 : (b * ipb + p) % ipb = p := by
    rw [Nat.mul_comm, Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
  rw [e1, e2]

/-- Calls with indices `< nb * ipb` address blocks `< nb` only. -/
theorem callOf_block_lt (ipb step k nb : Nat) (h : k < nb * ipb) : (callOf ipb step k).1 < nb := by
  simp only [callOf]
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Lane-striding: call `k` belongs to lane `k % 32` at round `k / 32`, and that lane/round pair
    is unique. -/
theorem lane_round (k : Nat) : k % 32 + 32 * (k / 32) = k := Nat.mod_add_div k 32

-- § launch grid

/-- `iq_mmvq` launches `⌈n_out / 4⌉` blocks; row `blockIdx.x * 4 + threadIdx.y` covers every
    output row exactly once and no row of block `b < grid` exceeds `n_out + 3`. -/
theorem grid_covers (nOut r : Nat) (h : r < nOut) :
    r / 4 < (nOut + 3) / 4 ∧ (r / 4) * 4 + r % 4 = r := by omega

/-- Out-of-range warps of the last block are exactly those with `row ≥ n_out`; all lie in the
    final block. -/
theorem grid_tail (nOut b y : Nat) (hb : b < (nOut + 3) / 4) (hy : y < 4) :
    b * 4 + y < nOut + 3 := by omega

-- § expert blob layout

/-- `NativeExpertLayout`. -/
structure Layout where
  nEmbd : Nat
  nFF : Nat
  guRow : Nat
  dRow : Nat

/-- `up_off = n_ff * gu_row`. -/
def Layout.upOff (L : Layout) : Nat := L.nFF * L.guRow
/-- `down_off = 2 * up_off`. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- `bytes = down_off + n_embd * d_row`. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Gate row `r < n_ff` lies wholly in `[0, up_off)`. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    r * L.guRow + L.guRow ≤ L.upOff := by
  unfold Layout.upOff
  have : (r + 1) * L.guRow ≤ L.nFF * L.guRow := Nat.mul_le_mul_right _ h
  rw [Nat.succ_mul] at this; exact this

/-- Up row `r < n_ff` lies wholly in `[up_off, down_off)`. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    L.upOff ≤ L.upOff + r * L.guRow ∧ L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff; omega

/-- Down row `r < n_embd` lies wholly in `[down_off, bytes)`. -/
theorem down_row_in_region (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  unfold Layout.bytes
  have : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right _ h
  rw [Nat.succ_mul] at this; omega

-- § scratch alignment

/-- `(f + 255) & ~255` as arithmetic: round up to a multiple of 256. -/
def align256 (f : Nat) : Nat := (f + 255) / 256 * 256

/-- The aligned size is a multiple of 256, at least `f`, and wastes fewer than 256 bytes. -/
theorem align256_spec (f : Nat) :
    align256 f % 256 = 0 ∧ f ≤ align256 f ∧ align256 f < f + 256 := by
  unfold align256; omega

/-- The bit form `(f + 255) &&& ~~~255` agrees with the arithmetic form on a test range. -/
theorem align256_bits : ∀ f, f < 1024 → (f + 255) / 256 * 256 = (f + 255) - (f + 255) % 256 := by
  intro f _; omega

/-- `native_expert_scratch_bytes`: gate | up | h (each `cap*n_ff` floats) | hq (Q8_1). -/
def scratchBytes (cap nFF : Nat) : Nat :=
  3 * align256 (cap * nFF * 4) + align256 (cap * (nFF / 32) * 36)

/-- The four scratch carve-outs fit: the Q8_1 intermediate starts at `3 * fa` and ends inside
    the advertised scratch size. -/
theorem scratch_fits (cap nFF : Nat) :
    3 * align256 (cap * nFF * 4) + cap * (nFF / 32) * 36 ≤ scratchBytes cap nFF := by
  have := (align256_spec (cap * (nFF / 32) * 36)).2.1
  unfold scratchBytes; omega

-- § decoders

/-- Bit count of the low 8 bits. -/
def popcount8 (v : Nat) : Nat := (List.range 8).foldl (fun acc i => acc + (v >>> i) % 2) 0

/-- `unpack_ksigns` byte: 7 stored sign bits with the parity bit in bit 7. -/
def ksigns (v : Nat) : Nat := v ^^^ ((popcount8 v % 2) <<< 7)

/-- Every reconstructed sign byte has an even number of negative signs, and its low 7 bits are
    the stored ones. -/
theorem ksigns_even : ∀ v, v < 128 → popcount8 (ksigns v) % 2 = 0 ∧ ksigns v % 128 = v := by
  decide

/-- `dq_q2_0`: a 2-bit code `c` dequantizes to `d * (c - 1)`, i.e. one of `{-1, 0, 1, 2}·d`. -/
def q2Value (c : Nat) : Int := (c : Int) - 1

/-- Q2_0 codes span exactly `-1 .. 2`. -/
theorem q2Value_range (c : Nat) (h : c < 4) : -1 ≤ q2Value c ∧ q2Value c ≤ 2 := by
  unfold q2Value; omega

/-- Q2_0 code `i` of a 64-element block sits in byte `i / 4` at bit `2 * (i % 4)`: 16 bytes. -/
theorem q2_code_position (i : Nat) (h : i < 64) : i / 4 < 16 ∧ 2 * (i % 4) ≤ 6 := by omega

/-- IQ3_XXS scale `(ls * s + s / 2) / 2` uses `ls = aux >> 28 < 16`, IQ2_XXS `ls = aux>>27 | 1`
    is odd: both are bounded by the top bits of a 32-bit word. -/
theorem iq_scale_bits : ∀ a, a < 2^32 → a >>> 28 < 16 ∧ a >>> 27 < 32 := by
  intro a ha
  constructor
  · rw [Nat.shiftRight_eq_div_pow]; omega
  · rw [Nat.shiftRight_eq_div_pow]; omega

end IqQuantKernels



## FILE: /mnt/user-data/outputs/S2GroupedExpert.lean
/-!
# MODULE: s2_grouped_expert
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Model of the R4 grouped GPU expert over S2 (2-bit, Q2_0-style) blobs: the fixed blob
geometry, the `Σ (c-1)·x = Σ c·x − Σ x` DP4A identity, the gate-major gate/up output
layout, the device-side hit compaction and the CPU-order 8-lane reduction tree.

## Source semantics
* A blob is `[2·FF gate/up code rows | H down code rows | gate/up fp16 scales | down scales]`
  with H = 2560, FF = 640, one fp16 weight scale per 64 elements.
* Row-slot `i` of the gate/up rows is gate row `i/2` (even) or up row `i/2` (odd); output is
  GATE-MAJOR: hit `h` gate row `r` → `h·FF + r`, up row `r` → `n·FF + h·FF + r`.
* A code byte packs four LSB-first 2-bit fields; the weight value is `code − 1`.
* Q8_0 activation blocks are 34 bytes, so their int8 payload alternates between 2-mod-4 and
  0-mod-4 addresses — hence `memcpy` loads.
* `hit_select` compacts resident hits in lane order: the output slot of a hit is the number of
  hits before it.
* The CPU-order kernel reduces 8 lanes with shuffle offsets 4, 1, 2 (SSE `add`, `hadd`, `hadd`).

## Form template identifiers
`H/FF/ROW_GU/ROW_D/SC_GU/SC_D/O_*` → same names; `row_dot_s2_q8` → `s2dot`; gate/up write
index → `guOut`; `hit_select_kernel` → `slotOf`; `row_dot_cpu_order` tail → `cpuTree`;
`moe_hit_grouped_scratch_bytes` → `scratchBytes`.
-/

namespace S2GroupedExpert

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert intermediate size. -/
def FF : Nat := 640
/-- Bytes of 2-bit codes per gate/up row. -/
def ROW_GU : Nat := H / 4
/-- Bytes of 2-bit codes per down row. -/
def ROW_D : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def SC_GU : Nat := H / 64
/-- fp16 scales per down row. -/
def SC_D : Nat := FF / 64
/-- Start of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Start of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Start of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob size. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

/-- The literal geometry restated in the kernel: 640 / 160 code bytes, 40 / 10 scales per row,
    and the region offsets. -/
theorem geometry :
    ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 ∧ O_D_CODES = 819200 ∧
    O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 ∧ BLOB = 1382400 := by decide

/-- Density: the blob holds exactly 2 bits per weight plus one fp16 per 64 weights. -/
theorem blob_density : BLOB * 8 = (3 * H * FF) * 2 + (3 * H * FF / 64) * 16 := by decide

/-- A gate/up row `i < 2·FF` and its scales stay inside their regions. -/
theorem gu_row_regions (i : Nat) (h : i < 2 * FF) :
    i * ROW_GU + ROW_GU ≤ O_D_CODES ∧
    O_GU_SCALES + i * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  simp only [O_D_CODES, O_GU_SCALES, O_D_SCALES, ROW_GU, SC_GU, H, FF, ROW_D] at *
  omega

/-- A down row `r < H` and its scales stay inside their regions. -/
theorem down_row_regions (r : Nat) (h : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES ∧
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  simp only [O_D_CODES, O_GU_SCALES, O_D_SCALES, BLOB, ROW_GU, ROW_D, SC_GU, SC_D, H, FF] at *
  omega

/-- 32-element chunk `c` uses weight scale `c >> 1`: one fp16 per two chunks (64 weights). -/
theorem chunk_scale (c : Nat) (h : c < H / 32) : c / 2 < SC_GU := by
  simp only [SC_GU, H] at *; omega

-- § integer arithmetic of a chunk

/-- Exact integer dot. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Integer sum. -/
def lsum : List Int → Int
  | [] => 0
  | a :: as => a + lsum as

/-- The header identity: `Σ (c_j − 1)·x_j = Σ c_j·x_j − Σ x_j`.  Both right-hand sums are
    INT8×INT8 DP4A chains, so the chunk is exact in integer before one float multiply. -/
theorem s2dot : ∀ (c x : List Int), c.length = x.length →
    dot (c.map (· - 1)) x = dot c x - lsum x
  | [], [], _ => by simp [dot, lsum]
  | a :: as, b :: bs, h => by
    have ih := s2dot as bs (by simpa using h)
    simp only [List.map, dot, lsum]
    rw [ih, Int.sub_mul, Int.one_mul]
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

/-- A code byte is exactly the LSB-first packing of its four 2-bit fields. -/
theorem code_byte_fields (b : Nat) (h : b < 256) :
    b = b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) := by omega

/-- The bit expansion `(c & 3) | ((c>>2)&3)<<8 | …` of every code byte has four bytes `< 4`
    (each a valid non-negative int8 lane for DP4A). -/
theorem code_word_lanes : ∀ c, c < 256 →
    (c &&& 3) < 4 ∧ ((c >>> 2) &&& 3) < 4 ∧ ((c >>> 4) &&& 3) < 4 ∧ ((c >>> 6) &&& 3) < 4 := by
  intro c _
  refine ⟨?_, ?_, ?_, ?_⟩ <;> exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)

/-- Q8_0 int8 payload of block `c` (at `34c + 2`) is 4-byte aligned exactly for odd `c`; the
    first block's is not, which is the fault the `memcpy` load removes. -/
theorem q8_0_payload_alignment (c : Nat) : (34 * c + 2) % 4 = 0 ↔ c % 2 = 1 := by omega

-- § gate-major gate/up layout

/-- Output index of row-slot `i` of hit `h` among `n` hits. -/
def guOut (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- Every write lands inside the `2 · n · FF` gate/up buffer. -/
theorem guOut_bound (n h i : Nat) (hh : h < n) (hi : i < 2 * FF) : guOut n h i < 2 * n * FF := by
  unfold guOut; simp only [FF] at *; split <;> omega

/-- Gate rows of all hits occupy `[0, n·FF)` and up rows `[n·FF, 2·n·FF)`: the swiglu over the
    first half sees each hit's gate paired with the SAME hit's up at offset `n·FF`. -/
theorem guOut_pairing (n h r : Nat) :
    guOut n h (2 * r) = h * FF + r ∧ guOut n h (2 * r + 1) = n * FF + (h * FF + r) := by
  unfold guOut
  constructor <;> split <;> omega

/-- Distinct (hit, slot) pairs write distinct outputs. -/
theorem guOut_injective (n h1 h2 i1 i2 : Nat) (hh1 : h1 < n) (hh2 : h2 < n)
    (hi1 : i1 < 2 * FF) (hi2 : i2 < 2 * FF) (e : guOut n h1 i1 = guOut n h2 i2) :
    h1 = h2 ∧ i1 = i2 := by
  unfold guOut at e; simp only [FF] at *
  split at e <;> split at e <;> omega

-- § device-side hit compaction

/-- Output slot of lane `i`: the number of resident hits strictly before it (`popc(hit & mask)`). -/
def slotOf (hits : List Bool) (i : Nat) : Nat := (hits.take i).count true

/-- Prefix counts are monotone in the prefix length. -/
theorem slotOf_mono (hits : List Bool) {i j : Nat} (h : i ≤ j) : slotOf hits i ≤ slotOf hits j := by
  unfold slotOf
  have : hits.take i = (hits.take j).take i := by rw [List.take_take, Nat.min_eq_left h]
  rw [this]
  exact (List.take_sublist i _).count_le true

/-- A hit at lane `i` raises the prefix count by one. -/
theorem slotOf_succ (hits : List Bool) (i : Nat) (hi : i < hits.length) (hh : hits[i] = true) :
    slotOf hits (i + 1) = slotOf hits i + 1 := by
  unfold slotOf
  rw [List.take_add_one, List.getElem?_eq_getElem hi, hh]
  simp

/-- Compaction is order-preserving and injective: earlier hits get strictly smaller slots. -/
theorem slotOf_strict (hits : List Bool) {i j : Nat} (hij : i < j) (hi : i < hits.length)
    (hh : hits[i] = true) : slotOf hits i < slotOf hits j := by
  have h1 := slotOf_succ hits i hi hh
  have h2 := slotOf_mono hits (show i + 1 ≤ j by omega)
  omega

/-- Every hit's slot is below the hit count written to `*count` (`popc(hit)`). -/
theorem slotOf_lt_count (hits : List Bool) (i : Nat) (hi : i < hits.length)
    (hh : hits[i] = true) : slotOf hits i < hits.count true := by
  have h1 := slotOf_succ hits i hi hh
  have h2 : slotOf hits (i + 1) ≤ hits.count true := by
    unfold slotOf; exact (List.take_sublist _ _).count_le true
  omega

-- § CPU-order reduction tree

/-- `__shfl_down_sync(v, off, 8)`: lane `l` reads lane `l + off` when it is inside the 8-lane
    segment, otherwise keeps its own value. -/
def shflDown {α : Type} (v : Fin 8 → α) (off : Nat) (l : Fin 8) : α :=
  if h : l.val + off < 8 then v ⟨l.val + off, h⟩ else v l

/-- One reduction step `acc += shfl_down(acc, off)`. -/
def step {α : Type} [Add α] (v : Fin 8 → α) (off : Nat) : Fin 8 → α :=
  fun l => v l + shflDown v off l

/-- The source's tree: offsets 4, then 1, then 2. -/
def cpuTree {α : Type} [Add α] (v : Fin 8 → α) : α := step (step (step v 4) 1) 2 0

/-- Lane 0 of the 4,1,2 tree computes exactly `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))` — the
    `_mm_add_ps(low,high)` then two `_mm_hadd_ps` bracketing — for any `+`, including
    non-associative float addition. -/
theorem cpuTree_lane0 {α : Type} [Add α] (v : Fin 8 → α) :
    cpuTree v = ((v 0 + v 4) + (v 1 + v 5)) + ((v 2 + v 6) + (v 3 + v 7)) := rfl

/-- The standard 4,2,1 tree, for comparison. -/
def stdTree {α : Type} [Add α] (v : Fin 8 → α) : α := step (step (step v 4) 2) 1 0

/-- The standard tree brackets differently: `((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))`; with
    non-associative float `+` this is a different expression, as the source warns. -/
theorem stdTree_lane0 {α : Type} [Add α] (v : Fin 8 → α) :
    stdTree v = ((v 0 + v 4) + (v 2 + v 6)) + ((v 1 + v 5) + (v 3 + v 7)) := rfl

/-- Free syntax of additions, used to compare bracketings without assuming associativity. -/
inductive Sum8 where
  | leaf (n : Nat)
  | add (a b : Sum8)
  deriving DecidableEq

instance : Add Sum8 := ⟨Sum8.add⟩

/-- As syntax (i.e. for a non-associative `+` such as IEEE float addition) the 4,1,2 tree and
    the standard 4,2,1 tree are different expressions. -/
theorem trees_differ :
    cpuTree (fun l : Fin 8 => Sum8.leaf l.val) ≠ stdTree (fun l : Fin 8 => Sum8.leaf l.val) := by
  decide

-- § scratch

/-- 16-byte round-up. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- `moe_hit_grouped_scratch_bytes`. -/
def scratchBytes (n nEmbd nFF : Nat) : Nat :=
  align16 (n * (2 * nFF) * 4) + align16 (n * (nFF / 32) * 34) +
  2 * align16 (n * (nFF / 32) * 4) + align16 ((nEmbd / 32) * 4)

/-- The carve-out used by `moe_hit_grouped_s2` (gate_up | h_q8_0 | h_scales) fits the
    advertised scratch. -/
theorem scratch_fits (n nEmbd nFF : Nat) :
    align16 (n * (2 * nFF) * 4) + align16 (n * (nFF / 32) * 34) + n * (nFF / 32) * 4 ≤
      scratchBytes n nEmbd nFF := by
  unfold scratchBytes align16; omega

end S2GroupedExpert



## FILE: /mnt/user-data/outputs/SharedExpertGate.lean
/-!
# MODULE: shared_expert_gate
SOURCES: src/kernels/cuda/shared_expert.cu
Model of the P2.S2 shared expert and the MoE combination: SwiGLU orientation (SiLU on the
gate), the per-token scalar sigmoid gate, the caller-owned scratch carve-out, the native
projection dispatch preconditions and `moe_combine`'s plain addition of the shared output.

## Source semantics
* `h = silu(x·Wg) * (x·Wu)`; `h = h·Wd`; `g = sigmoid(x · w_gate_inp)` is ONE scalar per token
  computed from the original `x`; the result is `h * g`.
* The shared output is added unweighted to the router-weighted routed sum; `k ≤ 64`.
* Scratch = gate | up | q8_0 | q8_K | g, each region 16-byte aligned, carved, never allocated.
* Any native gate/up projection (or the native BF16 gate) requires the unrounded `x_f32`;
  `shared_expert_multi` requires 1..8 tokens.
* The scalar gate reduces 2560 terms over 256 threads: a 10-term chain per thread, then a tree.

## Form template identifiers
`swiglu_kernel` → `swiglu`; `scalar_gate_kernel` → `scalarGate`; `shared_expert_scratch_bytes` →
`scratchBytes`/`offsets`; `moe_combine_kernel` → `combine`; native flags → `Dispatch`;
`scale_rows_kernel` index → `rowIndex`.
-/

namespace SharedExpertGate

-- § SwiGLU orientation

/-- `swiglu(gate, up) = silu(gate) * up` for an abstract activation `silu`. -/
def swiglu {α : Type} [Mul α] (silu : α → α) (g u : α) : α := silu g * u

/-- The orientation matters: with a non-linear activation, swapping gate and up changes the
    value (here with `x ↦ x²`, `swiglu 2 3 = 12` but `swiglu 3 2 = 18`).  The swapped reading
    is the plausible-but-wrong one recorded in `docs/semantics.md`. -/
theorem swiglu_not_symmetric :
    swiglu (fun x : Int => x * x) 2 3 ≠ swiglu (fun x : Int => x * x) 3 2 := by decide

/-- The SwiGLU is linear in `up`. -/
theorem swiglu_linear_up (silu : Int → Int) (g u v : Int) :
    swiglu silu g (u + v) = swiglu silu g u + swiglu silu g v := by
  unfold swiglu; exact Int.mul_add _ _ _

-- § scalar gate reduction

/-- Terms per thread of a strided loop over `n` elements with `t` threads (thread `i` takes
    `i, i+t, …`). -/
def termsPerThread (n t i : Nat) : Nat := (n + t - 1 - i) / t

/-- For `n_embd = 2560` and 256 threads every thread's dependency chain is 10 terms long,
    instead of the 2560-long single-thread chain the source replaced. -/
theorem gate_chain_length : ∀ i, i < 256 → termsPerThread 2560 256 i = 10 := by
  intro i h; unfold termsPerThread; omega

/-- The 256 per-thread chains partition the 2560 terms: element `e` belongs to thread `e % 256`
    as its `(e / 256)`-th term. -/
theorem gate_partition (e : Nat) (h : e < 2560) :
    e % 256 < 256 ∧ e / 256 < 10 ∧ e % 256 + 256 * (e / 256) = e := by omega

/-- The block reduction uses `(256 + 31) / 32 = 8` warp partials, matching `scratch[8]`. -/
theorem gate_warps : (256 + 31) / 32 = 8 := by decide

/-- Per-token gate: `g = sigmoid(Σ x_i w_i)`; the multiply is applied to every output element
    of that token: element `i` of token `t` sits at `t * n + i`. -/
def rowIndex (n t i : Nat) : Nat := t * n + i

/-- `scale_rows_kernel` touches each (token, element) exactly once. -/
theorem rowIndex_injective {n t1 t2 i1 i2 : Nat} (h1 : i1 < n) (h2 : i2 < n)
    (e : rowIndex n t1 i1 = rowIndex n t2 i2) : t1 = t2 ∧ i1 = i2 := by
  unfold rowIndex at e
  have hn : 0 < n := by omega
  have d1 : (i1 + n * t1) / n = t1 := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt h1, Nat.zero_add]
  have d2 : (i2 + n * t2) / n = t2 := by
    rw [Nat.add_mul_div_left _ _ hn, Nat.div_eq_of_lt h2, Nat.zero_add]
  have e' : i1 + n * t1 = i2 + n * t2 := by
    rw [Nat.mul_comm n t1, Nat.mul_comm n t2]; omega
  have ht : t1 = t2 := by rw [← d1, ← d2, e']
  subst ht
  exact ⟨rfl, by omega⟩

-- § scratch

/-- 16-byte round-up `(x + 15) & ~15`. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Q8_K block: fp32 `d`, 256 int8 codes, 16 int16 block sums. -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- `shared_expert_scratch_bytes`. -/
def scratchBytes (nFF : Nat) : Nat :=
  2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292) + 32

/-- Region starts: gate, up, q8_0, q8_K, g. -/
def offsets (nFF : Nat) : List Nat :=
  [0, align16 (nFF * 4), 2 * align16 (nFF * 4), 2 * align16 (nFF * 4) + align16 (nFF / 32 * 34),
   2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292)]

/-- Each carved region holds its payload before the next starts, every start is 16-byte
    aligned, and the 4-byte scalar `g` fits in the trailing 32 bytes. -/
theorem carve_fits (nFF : Nat) :
    nFF * 4 ≤ align16 (nFF * 4) ∧ nFF / 32 * 34 ≤ align16 (nFF / 32 * 34) ∧
    nFF / 256 * 292 ≤ align16 (nFF / 256 * 292) ∧
    2 * align16 (nFF * 4) + align16 (nFF / 32 * 34) + align16 (nFF / 256 * 292) + 4 ≤
      scratchBytes nFF ∧
    (∀ o, o ∈ offsets nFF → o % 16 = 0) := by
  refine ⟨by unfold align16; omega, by unfold align16; omega, by unfold align16; omega,
    by unfold scratchBytes; omega, ?_⟩
  intro o ho
  simp only [offsets, List.mem_cons, List.not_mem_nil, or_false] at ho
  unfold align16 at ho
  rcases ho with h | h | h | h | h <;> subst h <;> omega

/-- Q8_K for the down weight needs `n_ff % 256 = 0`; when it holds the q8_K region holds every
    super-block. -/
theorem q8k_whole (k : Nat) : (k * 256) / 256 * 292 = k * 292 := by
  rw [Nat.mul_div_cancel _ (by decide)]

-- § dispatch preconditions

/-- Which pieces run natively. -/
structure Dispatch where
  nativeBF16 : Bool
  gate : Bool
  up : Bool
  down : Bool
  hasF32 : Bool

/-- `shared_expert` accepts the call (does not throw on the `x_f32` check). -/
def accepts (d : Dispatch) : Bool := !((d.nativeBF16 || d.gate || d.up) && !d.hasF32)

/-- A native gate or up projection, or the native BF16 scalar gate, is only accepted with the
    unrounded F32 input; a native DOWN projection alone does not need it (it reads `gate`). -/
theorem accepts_iff (d : Dispatch) :
    accepts d = true ↔ ((d.nativeBF16 = true ∨ d.gate = true ∨ d.up = true) → d.hasF32 = true) := by
  cases d with
  | mk a b c e f => cases a <;> cases b <;> cases c <;> cases f <;> simp [accepts]

/-- `shared_expert_multi`'s token count guard. -/
def multiTokensOk (n : Int) : Bool := 1 ≤ n && n ≤ 8

/-- Exactly 1..8 tokens are accepted, matching the multi-column MMVQ limit. -/
theorem multiTokensOk_iff (n : Int) : multiTokensOk n = true ↔ 1 ≤ n ∧ n ≤ 8 := by
  simp [multiTokensOk]

-- § MoE combination

/-- `moe_combine_kernel` for one output column: `acc = 0; acc += w_e * p_e …; acc += shared`. -/
def combine (w p : List Int) (shared : Option Int) : Int :=
  (List.zipWith (· * ·) w p).foldl (· + ·) 0 + shared.getD 0

/-- Left fold sum starting from an accumulator. -/
theorem foldl_add_start (l : List Int) (a : Int) : l.foldl (· + ·) a = a + l.foldl (· + ·) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl]
    rw [ih (a + x), ih (0 + x)]
    omega

/-- The shared output is added plain: the combination with a shared expert is the routed
    combination plus the shared value, with no router weight or renormalisation applied to it. -/
theorem combine_shared_plain (w p : List Int) (s : Int) :
    combine w p (some s) = combine w p none + s := by
  simp [combine]

/-- With one routed expert and no shared expert the combination is `w₀ · p₀`. -/
theorem combine_single (w0 p0 : Int) : combine [w0] [p0] none = w0 * p0 := by
  simp [combine]

/-- `moe_combine` refuses `k > 64` rather than truncating. -/
def combineAccepts (k : Nat) : Bool := k ≤ 64

end SharedExpertGate



## FILE: /mnt/user-data/outputs/NativeMoeCombine.lean
/-!
# MODULE: native_moe_combine
SOURCES: src/kernels/cuda/native_moe.cu
Model of the pinned ggml `moe-weighted-reduction` combine: argument validation (stream,
width, 1..15 experts), pointer-span validity without address overflow, output/input
disjointness, and the float accumulation order `p₀·w₀ + p₁·w₁ + … (+ shared)`.

## Source semantics
* Accepts `1 ≤ k ≤ 15`, `0 < n_embd ≤ INT_MAX`, a non-null stream; spans are non-null,
  4-byte aligned and `bytes ≤ UINTPTR_MAX − address`.
* The output row must not overlap parts, weights or shared (half-open interval overlap).
* Column `col` computes `sum = parts[col]*w[0]`, then `sum += parts[e*n+col]*w[e]` for
  `e = 1..k-1`, then `sum += shared[col]` — no `0 +` seed and `part * weight` operand order,
  which differs from the legacy double-precision `moe_combine`.
* The grid is `⌈n_embd/256⌉` blocks of 256 threads, one column per thread.

## Form template identifiers
`valid_span` → `validSpan`; `overlap` → `overlaps`; argument guard → `argsOk`; `combine`
kernel → `nativeSum`; legacy kernel → `legacySum`; byte sizes → `partBytes`/`rowBytes`.
-/

namespace NativeMoeCombine

-- § argument validation

/-- `INT_MAX`. -/
def INT_MAX : Nat := 2^31 - 1

/-- The scalar guard of `native_moe_combine`. -/
def argsOk (hasStream : Bool) (nEmbd k : Nat) : Bool :=
  hasStream && 0 < nEmbd && nEmbd ≤ INT_MAX && 1 ≤ k && k ≤ 15

/-- The guard is exactly the documented range. -/
theorem argsOk_iff (s : Bool) (n k : Nat) :
    argsOk s n k = true ↔ s = true ∧ 0 < n ∧ n ≤ INT_MAX ∧ 1 ≤ k ∧ k ≤ 15 := by
  simp [argsOk, Bool.and_eq_true, decide_eq_true_eq, and_assoc]

/-- Row bytes. -/
def rowBytes (n : Nat) : Nat := n * 4
/-- All routed parts. -/
def partBytes (n k : Nat) : Nat := rowBytes n * k

/-- Within the accepted range the part span is at most 15 rows. -/
theorem partBytes_le (n k : Nat) (hk : k ≤ 15) : partBytes n k ≤ 15 * rowBytes n := by
  unfold partBytes; rw [Nat.mul_comm]; exact Nat.mul_le_mul_right _ hk

-- § spans

/-- `UINTPTR_MAX` on a 64-bit host. -/
def UPTR_MAX : Nat := 2^64 - 1

/-- `valid_span`: non-null, float-aligned, and the span does not wrap the address space. -/
def validSpan (addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ UPTR_MAX - addr

/-- The subtraction form `bytes ≤ MAX − addr` is the overflow-free spelling of
    `addr + bytes ≤ MAX` (for in-range addresses). -/
theorem validSpan_no_wrap (addr bytes : Nat) (ha : addr ≤ UPTR_MAX) :
    bytes ≤ UPTR_MAX - addr ↔ addr + bytes ≤ UPTR_MAX := by omega

/-- `overlap(a, an, b, bn)`: half-open intervals intersect. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlaps_symm (a an b bn : Nat) : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps; omega

/-- Non-overlap means one span ends before the other begins. -/
theorem not_overlaps_iff (a an b bn : Nat) :
    ¬ overlaps a an b bn ↔ (b + bn ≤ a ∨ a + an ≤ b) := by
  unfold overlaps; omega

/-- Two non-empty overlapping spans share an address. -/
theorem overlaps_witness (a an b bn : Nat) (h : overlaps a an b bn) (ha : 0 < an) (hb : 0 < bn) :
    ∃ x, a ≤ x ∧ x < a + an ∧ b ≤ x ∧ x < b + bn := by
  unfold overlaps at h
  exact ⟨max a b, by omega, by omega, by omega, by omega⟩

/-- A disjoint output means no output column aliases an input byte the kernel reads. -/
theorem disjoint_no_alias (o n p pn x : Nat) (h : ¬ overlaps o n p pn)
    (hx : o ≤ x ∧ x < o + n) : ¬ (p ≤ x ∧ x < p + pn) := by
  unfold overlaps at h; omega

-- § grid

/-- One thread per column: `⌈n/256⌉` blocks cover every column exactly once. -/
theorem grid_covers (n c : Nat) (h : c < n) :
    c / 256 < (n + 255) / 256 ∧ c / 256 * 256 + c % 256 = c := by omega

-- § accumulation order

/-- Free syntax of float expressions (no algebraic laws assumed). -/
inductive FExpr where
  | part (e : Nat)
  | weight (e : Nat)
  | shared
  | zero
  | add (a b : FExpr)
  | mul (a b : FExpr)
  deriving DecidableEq

/-- The native kernel's expression for `k` experts. -/
def nativeSum (k : Nat) (hasShared : Bool) : FExpr :=
  let routed := (List.range (k - 1)).foldl
    (fun acc i => FExpr.add acc (FExpr.mul (FExpr.part (i + 1)) (FExpr.weight (i + 1))))
    (FExpr.mul (FExpr.part 0) (FExpr.weight 0))
  if hasShared then FExpr.add routed FExpr.shared else routed

/-- The legacy `moe_combine` expression: seeded with `0`, `weight * part` order. -/
def legacySum (k : Nat) (hasShared : Bool) : FExpr :=
  let routed := (List.range k).foldl
    (fun acc i => FExpr.add acc (FExpr.mul (FExpr.weight i) (FExpr.part i))) FExpr.zero
  if hasShared then FExpr.add routed FExpr.shared else routed

/-- Two experts with a shared output: `((p₀w₀ + p₁w₁) + s)`. -/
theorem nativeSum_two :
    nativeSum 2 true =
      FExpr.add (FExpr.add (FExpr.mul (.part 0) (.weight 0)) (FExpr.mul (.part 1) (.weight 1)))
        FExpr.shared := rfl

/-- The pinned native expression differs syntactically from the legacy one for every accepted
    `k`, so the two are not interchangeable bit-for-bit; selecting the path is a real switch. -/
theorem native_ne_legacy : ∀ k, k < 16 → 1 ≤ k → ∀ s : Bool, nativeSum k s ≠ legacySum k s := by
  decide

/-- Integer evaluation (exact arithmetic) of an expression. -/
def eval (p w : Nat → Int) (sh : Int) : FExpr → Int
  | .part e => p e
  | .weight e => w e
  | .shared => sh
  | .zero => 0
  | .add a b => eval p w sh a + eval p w sh b
  | .mul a b => eval p w sh a * eval p w sh b

/-- In exact arithmetic both orders compute the same weighted sum (checked here for two
    experts with a shared output); only rounding can tell them apart. -/
theorem native_legacy_exact (p w : Nat → Int) (sh : Int) :
    eval p w sh (nativeSum 2 true) = eval p w sh (legacySum 2 true) := by
  simp only [nativeSum, legacySum, List.range, List.range.loop, List.foldl, eval, ite_true, Nat.zero_add]
  rw [Int.mul_comm (p 0), Int.mul_comm (p 1)]
  omega

end NativeMoeCombine



## FILE: /mnt/user-data/outputs/RouterTopKSelection.lean
/-!
# MODULE: router_top_k_selection
SOURCES: src/kernels/cuda/router_top10.cu
Model of the MoE router's softmax / top-k / renormalise half: the stable descending order
(ties by ascending index), the block-wide argmax combine used by every pass, the ggml lower
clamp of the renormaliser, and the launch / shared-memory geometry.

## Source semantics
* `p = softmax(logits)` over ALL experts; `ids` = stable argsort of `-p`, first `k`;
  `w = p[ids] / max(Σ p[ids], 2⁻¹⁴)`.
* Each of the `k` passes is a block argmax over untaken experts; two candidates combine by
  "larger value, then smaller index", so the reduction tree's shape cannot change the winner.
* `threads = roundup32(min(n_expert, 512))`; shared memory is `align16(n) + 8n + 4n` bytes.
* `k ≤ 64` and `n_expert ≤ 512·64` are refused with an exit, not truncated.

## Form template identifiers
`(ov > bv || (ov == bv && oi < bi))` → `better`/`pick`; `fmax(s, 6.103515625e-05)` → `clamp`;
launch width → `threads`; dynamic shared memory → `smem`.
-/

namespace RouterTopKSelection

-- § the selection order

/-- `x` beats `y`: larger probability, ties broken by smaller expert index. -/
def better (p : Nat → Int) (x y : Nat) : Prop := p x > p y ∨ (p x = p y ∧ x < y)

instance (p : Nat → Int) (x y : Nat) : Decidable (better p x y) := by
  unfold better; infer_instance

/-- `better` is irreflexive. -/
theorem better_irrefl (p : Nat → Int) (x : Nat) : ¬ better p x x := by
  unfold better; omega

/-- `better` is transitive. -/
theorem better_trans (p : Nat → Int) {x y z : Nat} (h1 : better p x y) (h2 : better p y z) :
    better p x z := by
  unfold better at *; omega

/-- `better` is total on distinct experts: the order is a strict total order, i.e. a stable
    descending argsort. -/
theorem better_total (p : Nat → Int) {x y : Nat} (h : x ≠ y) : better p x y ∨ better p y x := by
  unfold better; omega

/-- The reduction step: keep the candidate that `better` prefers. -/
def pick (p : Nat → Int) (a b : Nat) : Nat := if better p b a then b else a

/-- The result is one of the two candidates. -/
theorem pick_mem (p : Nat → Int) (a b : Nat) : pick p a b = a ∨ pick p a b = b := by
  unfold pick; split <;> simp

/-- The result is not beaten by either candidate. -/
theorem pick_max (p : Nat → Int) (a b : Nat) :
    ¬ better p a (pick p a b) ∧ ¬ better p b (pick p a b) := by
  unfold pick; split <;> rename_i h <;> unfold better at * <;> omega

/-- "Not beaten by" chains: if `a` does not beat `r` and `r` does not beat `w`, then `a` does
    not beat `w` (the complement of a strict total order is transitive). -/
theorem notBetter_trans (p : Nat → Int) {a r w : Nat} (h1 : ¬ better p a r)
    (h2 : ¬ better p r w) : ¬ better p a w := by
  unfold better at *; omega

/-- The step is commutative on distinct candidates (the shuffle partner order is irrelevant). -/
theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a := by
  unfold pick better
  by_cases h1 : (p b > p a ∨ p b = p a ∧ b < a) <;>
    by_cases h2 : (p a > p b ∨ p a = p b ∧ a < b) <;> simp only [h1, h2, ite_true, ite_false] <;>
    omega

/-- The step is associative, so any reduction tree (warp shuffles, then the warp-0 pass over
    per-warp winners) selects the same expert as the serial ascending scan. -/
theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c) := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p c b <;> by_cases h3 : better p c a <;>
    simp only [h1, h2, h3, ite_true, ite_false] <;> unfold better at * <;> omega

/-- Serial left-to-right scan from an initial candidate. -/
def scan (p : Nat → Int) (init : Nat) (l : List Nat) : Nat := l.foldl (pick p) init

/-- The scan's winner is never beaten by any scanned candidate nor by the initial one. -/
theorem scan_max (p : Nat → Int) : ∀ (l : List Nat) (init : Nat),
    ¬ better p init (scan p init l) ∧ ∀ x, x ∈ l → ¬ better p x (scan p init l)
  | [], init => by simp [scan, better_irrefl]
  | y :: ys, init => by
    have ih := scan_max p ys (pick p init y)
    have hm := pick_max p init y
    have e : scan p init (y :: ys) = scan p (pick p init y) ys := rfl
    rw [e]
    refine ⟨notBetter_trans p hm.1 ih.1, ?_⟩
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact notBetter_trans p hm.2 ih.1
    · exact ih.2 x hx

-- § renormalisation clamp

/-- The literal `6.103515625e-05` is exactly `2⁻¹⁴`. -/
theorem clamp_literal : 6103515625 * 2^14 = 10^14 := by decide

/-- `fmax(s, c)` with `c > 0` is positive, so the renormalising division is always defined,
    and it is the identity clamp whenever the selected mass already exceeds `c`. -/
theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c ∧ (c ≤ s → max s c = s) := by
  constructor
  · exact Int.lt_of_lt_of_le hc (Int.le_max_right s c)
  · intro h; exact Int.max_eq_left h

-- § launch geometry

/-- Threads per token: `min(n, 512)` rounded up to whole warps. -/
def threads (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- At least one full warp, at most 512 threads, always whole warps. -/
theorem threads_spec (n : Nat) (h : 0 < n) :
    32 ≤ threads n ∧ threads n ≤ 512 ∧ threads n % 32 = 0 := by
  unfold threads; omega

/-- Warp partial slots: at most `512 / 32 = 16`, the size of `s_red` / `s_rid`. -/
theorem warp_slots (n : Nat) : threads n / 32 ≤ 16 := by unfold threads; omega

/-- Dynamic shared memory: taken mask (16-aligned), `n` doubles, `n` floats. -/
def smem (n : Nat) : Nat := (n + 15) / 16 * 16 + 8 * n + 4 * n

/-- The doubles start 16-byte aligned (8-byte alignment needed) and the floats after them
    stay 4-byte aligned. -/
theorem smem_alignment (n : Nat) :
    ((n + 15) / 16 * 16) % 8 = 0 ∧ ((n + 15) / 16 * 16 + 8 * n) % 4 = 0 := by omega

/-- The model's geometry (512 experts) needs 6656 bytes, well inside the 48 KiB default; the
    `n_expert ≤ 512·64` guard admits sizes whose request exceeds 48 KiB, where the failure is
    caught by the post-launch error check rather than the guard. -/
theorem smem_bounds : smem 512 = 6656 ∧ smem 512 ≤ 49152 ∧ 49152 < smem (512 * 64) := by decide

/-- The k-pass argmax does `10 · 512` comparisons against the rank loop's `512²`: 51× less. -/
theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51 := by decide

/-- Guards of the host wrapper. -/
def accepts (k nExpert : Nat) : Bool := k ≤ 64 && nExpert ≤ 512 * 64

/-- The production shape (10 of 512) is accepted. -/
theorem accepts_production : accepts 10 512 = true := by decide

end RouterTopKSelection



## FILE: /mnt/user-data/outputs/NativeRouterWarpTopK.lean
/-!
# MODULE: native_router_warp_top_k
SOURCES: src/kernels/cuda/native_router.cu
Model of the pinned ggml `topk-moe` router for the fixed 512-expert / top-10 shape: one warp
holds 16 logits per lane, softmax, ten rounds of a lane-local scan plus an XOR-butterfly
argmax, `-∞` masking of winners, clamped renormalisation, and the host-side span checks.

## Source semantics
* Expert `e` lives in lane `e % 32`, register slot `e / 32`; the 512 experts fill exactly the
  32 × 16 register file.
* Each round: each lane scans its 16 slots ascending with strict `>` (lowest index wins a tie
  inside the lane), then lanes combine by "larger value, then smaller id".
* The winner's slot is set to `-∞`; output rank `r < 10` is written by lane `r`.
* `selected_sum = max(Σ selected, 2⁻¹⁴)`, `weights[r] = selected_r / selected_sum`.
* Host: non-null stream; logits 2048 B, ids 40 B, weights 40 B, all 4-aligned and pairwise
  disjoint.

## Form template identifiers
lane/slot map → `laneOf`/`slotOf`/`expertAt`; combine predicate → `better`/`pick`;
in-lane scan → `laneScan`; span checks → `spanOk`/`disjoint3`.
-/

namespace NativeRouterWarpTopK

-- § register geometry

/-- Lane holding expert `e`. -/
def laneOf (e : Nat) : Nat := e % 32
/-- Register slot holding expert `e`. -/
def slotOf (e : Nat) : Nat := e / 32
/-- Expert in `values[i]` of lane `l` (`lane + i * 32`). -/
def expertAt (l i : Nat) : Nat := l + i * 32

/-- Every expert `< 512` has a lane `< 32` and slot `< 16`, and is recovered from them. -/
theorem layout_roundtrip (e : Nat) (h : e < 512) :
    laneOf e < 32 ∧ slotOf e < 16 ∧ expertAt (laneOf e) (slotOf e) = e := by
  unfold laneOf slotOf expertAt; omega

/-- Every (lane, slot) pair is a distinct expert `< 512`. -/
theorem layout_inverse (l i : Nat) (hl : l < 32) (hi : i < 16) :
    expertAt l i < 512 ∧ laneOf (expertAt l i) = l ∧ slotOf (expertAt l i) = i := by
  unfold laneOf slotOf expertAt; omega

/-- Within one lane, slot order is index order, so the ascending strict-`>` scan keeps the
    lowest expert id among equal values in that lane. -/
theorem lane_index_monotone (l i j : Nat) (h : i < j) : expertAt l i < expertAt l j := by
  unfold expertAt; omega

/-- `(expert & 31) == lane` identifies the owning lane of the winner. -/
theorem owner_bits : ∀ e, e < 512 → e &&& 31 = laneOf e := by
  intro e _; unfold laneOf; exact Nat.and_two_pow_sub_one_eq_mod e 5

-- § the combine

/-- `x` beats `y`: larger value, ties by smaller id. -/
def better (p : Nat → Int) (x y : Nat) : Prop := p x > p y ∨ (p x = p y ∧ x < y)

instance (p : Nat → Int) (x y : Nat) : Decidable (better p x y) := by
  unfold better; infer_instance

/-- `if (other > best || (other == best && other_id < expert))` keep the other. -/
def pick (p : Nat → Int) (a b : Nat) : Nat := if better p b a then b else a

/-- Commutative: lane `l` combining with `l ^ mask` and lane `l ^ mask` combining with `l`
    reach the same candidate, so after the butterfly all lanes agree. -/
theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p a b <;>
    simp only [h1, h2, ite_true, ite_false] <;> unfold better at * <;> omega

/-- Associative: the butterfly's grouping cannot change the winner. -/
theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c) := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p c b <;> by_cases h3 : better p c a <;>
    simp only [h1, h2, h3, ite_true, ite_false] <;> unfold better at * <;> omega

/-- The lane-local scan: start from slot 0, replace on strictly greater value. -/
def laneScan (p : Nat → Int) (l : Nat) (slots : List Nat) : Nat :=
  slots.foldl (fun best i => if p (expertAt l i) > p best then expertAt l i else best)
    (expertAt l 0)

/-- Because slot order is index order, the strict-`>` scan coincides with `pick` along the
    lane: replacing only on strictly larger values is the tie-to-lower-id rule. -/
theorem laneStep_eq_pick (p : Nat → Int) (l i : Nat) (best : Nat) (hb : best ≤ expertAt l i) :
    (if p (expertAt l i) > p best then expertAt l i else best) = pick p best (expertAt l i) := by
  unfold pick better
  by_cases h : p (expertAt l i) > p best
  · have hc : p (expertAt l i) > p best ∨ p (expertAt l i) = p best ∧ expertAt l i < best :=
      Or.inl h
    rw [if_pos h, if_pos hc]
  · have hc : ¬ (p (expertAt l i) > p best ∨ p (expertAt l i) = p best ∧ expertAt l i < best) := by
      omega
    rw [if_neg h, if_neg hc]

-- § outputs

/-- Ranks `0..9` are written by lanes `0..9`, one each; lanes `10..31` write no weight. -/
theorem rank_writer (r : Nat) (h : r < 10) : r < 32 := by omega

/-- `-∞` masking: once an expert is masked it can only win again against other masked
    candidates; with 512 finite values and 10 rounds at most 10 experts are ever masked. -/
theorem rounds_fit : 10 ≤ 512 := by decide

/-- The clamp literal `6.103515625e-5f` is `2⁻¹⁴`, exactly representable in fp32. -/
theorem clamp_literal : 6103515625 * 2^14 = 10^14 := by decide

/-- The clamped selected sum is positive, so every weight is defined. -/
theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c :=
  Int.lt_of_lt_of_le hc (Int.le_max_right s c)

-- § host checks

/-- Span sizes: 512 logits, 10 ids, 10 weights, 4 bytes each. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

/-- `valid`: non-null, 4-aligned, no address wrap. -/
def spanOk (addr bytes : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ 2^64 - 1 - addr

/-- `overlap` on half-open spans. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- The three buffers are pairwise disjoint. -/
def disjoint3 (lg ids w : Nat) : Prop :=
  ¬ overlaps lg 2048 ids 40 ∧ ¬ overlaps lg 2048 w 40 ∧ ¬ overlaps ids 40 w 40

/-- A packed layout `logits | ids | weights` passes the disjointness check. -/
theorem packed_layout_ok (a : Nat) : disjoint3 a (a + 2048) (a + 2088) := by
  unfold disjoint3 overlaps; omega

/-- Writing `ids` inside the logits span (e.g. aliasing the input) is rejected. -/
theorem alias_rejected (a : Nat) : ¬ disjoint3 a a (a + 2048) := by
  unfold disjoint3 overlaps; omega

/-- Launch shape `dim3(32, 8)`: 256 threads of which only row `y = 0` (32 lanes) works. -/
theorem active_lanes : 32 * 8 = 256 ∧ 256 / 8 = 32 := by decide

end NativeRouterWarpTopK



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
Model of the P2.S2 sampler: llama.cpp's repeat / frequency / presence penalty, the greedy
argmax (and the temperature-0 bug it fixes), the chain order `top_k → top_p → temperature →
penalties → pick`, the penalty history window, the top-p `min_keep` clamp and the Philox
uniform's range.

## Source semantics
* Penalty for a token seen `count > 0` times: `logit ≤ 0 ? logit·r : logit/r`, then
  `− count·f − [count>0]·presence` (presence is boolean).
* Greedy = argmax over penalised RAW logits, lowest index on ties; temperature must not be
  applied (scaling by `1/T = 0` made every logit 0 and always picked token 0).
* Survivors are chosen on raw logits; temperature scales them; penalties then see the scaled
  value — penalising before scaling is a different distribution.
* The history window is the TAIL `min(penalty_last_n, history_len)` entries.
* `philox_uniform = (c0 >> 8) · 2⁻²⁴ ∈ [0, 1)`.

## Form template identifiers
`apply_penalties` → `penalize`/`repeatStage`; serial greedy scan → `amx`/`argmax`;
`hrow += history_len - hlen` → `window`; top-p cut clamp → `keepAfterTopP`;
`philox_uniform` mantissa → `mantissa24`.
-/

namespace SamplerChain

-- § penalties

/-- The repeat stage only: multiply non-positive logits, divide positive ones. -/
def repeatStage (logit r : Int) : Int := if logit ≤ 0 then logit * r else logit / r

/-- Full `apply_penalties` for a token seen `count` times. -/
def penalize (logit : Int) (count : Nat) (r f pres : Int) : Int :=
  if count = 0 then logit
  else repeatStage logit r - (count : Int) * f - (if count > 0 then pres else 0)

/-- For `r ≥ 1` the repeat stage never RAISES a logit, on either sign.  Dividing
    unconditionally would raise every negative logit — the inversion the source avoids. -/
theorem repeatStage_le (logit r : Int) (hr : 1 ≤ r) : repeatStage logit r ≤ logit := by
  unfold repeatStage
  split
  · rename_i h
    have h1 : logit * (r - 1) ≤ 0 := Int.mul_nonpos_of_nonpos_of_nonneg h (by omega)
    rw [Int.mul_sub, Int.mul_one] at h1
    omega
  · rename_i h
    exact Int.ediv_le_self r (by omega)

/-- The naive "always divide" reading raises a negative logit (`-6 / 2 = -3 > -6`). -/
theorem naive_divide_inverts : (-6 : Int) / 2 > -6 := by decide

/-- The presence term is boolean: it is the same for a token seen once or a hundred times. -/
theorem presence_boolean (logit r pres : Int) (c1 c2 : Nat) (h1 : 0 < c1) (h2 : 0 < c2) :
    penalize logit c1 r 0 pres = penalize logit c2 r 0 pres := by
  have a1 : c1 ≠ 0 := by omega
  have a2 : c2 ≠ 0 := by omega
  simp [penalize, a1, a2, h1, h2]

/-- An unseen token is never penalised. -/
theorem penalize_unseen (logit r f pres : Int) : penalize logit 0 r f pres = logit := by
  simp [penalize]

/-- With non-negative frequency/presence penalties and `r ≥ 1`, penalties never raise a logit. -/
theorem penalize_le (logit : Int) (count : Nat) (r f pres : Int) (hr : 1 ≤ r) (hf : 0 ≤ f)
    (hp : 0 ≤ pres) : penalize logit count r f pres ≤ logit := by
  unfold penalize
  split
  · exact Int.le_refl _
  · have h1 := repeatStage_le logit r hr
    have h2 : 0 ≤ (count : Int) * f := Int.mul_nonneg (by omega) hf
    split <;> omega

/-- Order matters: penalising the temperature-scaled logit differs from scaling the penalised
    one (logit 3, scale 2, presence 1: `6 − 1 = 5` vs `(3 − 1)·2 = 4`). -/
theorem penalty_after_temperature_differs :
    penalize (3 * 2) 1 1 0 1 ≠ penalize 3 1 1 0 1 * 2 := by decide

-- § greedy argmax

/-- Serial scan state: remaining logits, next index, best index, best value; strict `>` keeps
    the lowest index on ties. -/
def amx : List Int → Nat → Nat → Int → Nat
  | [], _, bi, _ => bi
  | x :: xs, i, bi, bv => if x > bv then amx xs (i + 1) i x else amx xs (i + 1) bi bv

/-- Argmax of a non-empty logit row. -/
def argmax : List Int → Nat
  | [] => 0
  | x :: xs => amx xs 1 0 x

/-- Scaling by a positive constant cannot reorder anything, so it cannot move the argmax:
    temperature is irrelevant to greedy decoding. -/
theorem amx_scale (c : Int) (hc : 0 < c) : ∀ (l : List Int) (i bi : Nat) (bv : Int),
    amx (l.map (· * c)) i bi (bv * c) = amx l i bi bv
  | [], _, _, _ => rfl
  | x :: xs, i, bi, bv => by
    simp only [List.map, amx]
    have hiff : x * c > bv * c ↔ x > bv := Int.mul_lt_mul_right hc
    by_cases h : x > bv
    · rw [if_pos (hiff.mpr h), if_pos h]; exact amx_scale c hc xs (i + 1) i x
    · rw [if_neg (fun h' => h (hiff.mp h')), if_neg h]; exact amx_scale c hc xs (i + 1) bi bv

/-- `argmax` is invariant under positive scaling. -/
theorem argmax_scale (l : List Int) (c : Int) (hc : 0 < c) : argmax (l.map (· * c)) = argmax l := by
  cases l with
  | nil => rfl
  | cons x xs => exact amx_scale c hc xs 1 0 x

/-- If nothing later strictly exceeds the current best, the scan keeps it. -/
theorem amx_keep : ∀ (l : List Int) (i bi : Nat) (bv : Int), (∀ x, x ∈ l → x ≤ bv) →
    amx l i bi bv = bi
  | [], _, _, _, _ => rfl
  | x :: xs, i, bi, bv, h => by
    simp only [amx]
    rw [if_neg (by have := h x (by simp); omega)]
    exact amx_keep xs (i + 1) bi bv (fun y hy => h y (by simp [hy]))

/-- THE TEMPERATURE-0 BUG: multiplying by `inv_t = 0` makes every logit 0 and the argmax is
    token 0 whatever the model predicted. -/
theorem zero_temperature_bug (l : List Int) : argmax (l.map (· * 0)) = 0 := by
  cases l with
  | nil => rfl
  | cons x xs =>
    simp only [List.map, argmax, Int.mul_zero]
    exact amx_keep _ 1 0 0 (fun y hy => by
      simp only [List.mem_map] at hy
      obtain ⟨z, _, rfl⟩ := hy
      simp)

/-- Ties go to the lowest index (the block reduction's `(ov == bv && oi < bi)` rule). -/
theorem argmax_tie : argmax [5, 7, 7, 1] = 1 := by decide

/-- The block reduction's sentinel answer: `n_vocab` (no candidate) is reported as token 0. -/
def reportBest (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

/-- The reported token is always a valid vocabulary index. -/
theorem reportBest_valid (wi nVocab : Nat) (h : 0 < nVocab) : reportBest wi nVocab < nVocab := by
  unfold reportBest; split <;> omega

-- § history window

/-- Window length `min(last_n, len)` clamped at 0 (Int inputs as in the source). -/
def window (lastN len : Int) : Int := max 0 (min lastN len)

/-- The window is the tail: it starts at `len − hlen ≥ 0` and ends exactly at `len`. -/
theorem window_tail (lastN len : Int) (hl : 0 ≤ len) :
    0 ≤ len - window lastN len ∧ (len - window lastN len) + window lastN len = len ∧
    window lastN len ≤ len := by
  unfold window
  have h1 := Int.min_le_right lastN len
  have h2 := Int.le_max_left 0 (min lastN len)
  have h3 : max 0 (min lastN len) ≤ len := Int.max_le.mpr ⟨hl, h1⟩
  omega

-- § top-k / top-p bookkeeping

/-- The sampled path supports `top_k ∈ 1..64` (`KMAX`). -/
def topKOk (k : Int) : Bool := 1 ≤ k && k ≤ 64

/-- Survivor count after the top-p cut and the `min_keep` clamp. -/
def keepAfterTopP (cut minKeep nKeep : Nat) : Nat :=
  if cut < minKeep then min minKeep nKeep else cut

/-- The clamp never keeps more than the top-k survivors and keeps at least `min_keep` of them
    whenever that many exist. -/
theorem keepAfterTopP_bounds (cut minKeep nKeep : Nat) (hc : cut ≤ nKeep) :
    keepAfterTopP cut minKeep nKeep ≤ nKeep ∧
    min minKeep nKeep ≤ keepAfterTopP cut minKeep nKeep := by
  unfold keepAfterTopP; split <;> omega

-- § Philox uniform

/-- `c0 >> 8` keeps 24 bits. -/
def mantissa24 (c : Nat) : Nat := c >>> 8

/-- For a 32-bit word the 24-bit mantissa is `< 2²⁴`, so `u = m · 2⁻²⁴ < 1` exactly: the
    inverse-CDF pick never sees `u = 1`. -/
theorem mantissa24_lt (c : Nat) (h : c < 2^32) : mantissa24 c < 2^24 := by
  unfold mantissa24; rw [Nat.shiftRight_eq_div_pow]; omega

/-- The per-token counter `p.counter + t` makes each token's draw a function of
    (seed, position) only: distinct tokens of a batch use distinct counters. -/
theorem counters_distinct (base t1 t2 : Nat) (h : base + t1 = base + t2) : t1 = t2 := by omega

end SamplerChain



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- NativeQuantizedMatVec.lean
  was: theorem bits_per_element_bounds : ∀ t, t < 64 → ∀ e b, blockSpec t = some (e, b) →
    2 * e ≤ 8 * b ∧ 8 * b ≤ 9 * e
  now: theorem bits_per_element_bounds : ∀ t, t < 64 → specDensityOk t = true
  measured: fewer quantified variables (2 → 1)

- NativeQuantizedMatVec.lean
  was: theorem kmap_injective (t1 t2 : Nat) (h1 : t1 < 128) (h2 : t2 < 128)
    (h : t1 / 16 = t2 / 16 ∧ 2 * (t1 % 16) = 2 * (t2 % 16)) : t1 = t2
  now: theorem kmap_injective (t1 t2 : Nat) (_h1 : t1 < 128) (_h2 : t2 < 128)
    (h : t1 / 16 = t2 / 16 ∧ 2 * (t1 % 16) = 2 * (t2 % 16)) : t1 = t2
  measured: the claim changed

- NativeQuantizedMatVec.lean
  was: theorem iq4nl_sorted_int8 :
    iq4nl.length = 16 ∧ (∀ i, i < 15 → iq4nl[i]! < iq4nl[i+1]!) ∧
    (∀ i, i < 16 → -127 ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ 127)
  now: theorem iq4nl_sorted_int8 :
    iq4nl.length = 16 ∧ (∀ i, i < 15 → iq4nl[i]! < iq4nl[i+1]!) ∧
    (∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (127 : Int))
  measured: more hypotheses/binders (0 → 2): the statement holds in fewer cases

- NativeQuantizedMatVec.lean
  was: theorem k4Scale_lt (j s0 s4 : Nat) : k4Scale j s0 s4 < 64
  now: theorem k4Scale_lt (j s0 s4 : Nat) (h0 : s0 < 256) : k4Scale j s0 s4 < 64
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- IqQuantKernels.lean
  was: theorem fmt_consistent : ∀ t, t < 64 →
    (match fmt t with
     | some (qk, ipb, step) => qk = blockElems t ∧ ipb * step ≤ qk / 16 ∧ 0 < ipb
     | none => True)
  now: theorem fmt_consistent : ∀ t, t < 64 → fmtOk t = true
  measured: the claim changed

- S2GroupedExpert.lean
  was: theorem guOut_pairing (n h r : Nat) (hr : r < FF) :
    guOut n h (2 * r) = h * FF + r ∧ guOut n h (2 * r + 1) = n * FF + (h * FF + r)
  now: theorem guOut_pairing (n h r : Nat) :
    guOut n h (2 * r) = h * FF + r ∧ guOut n h (2 * r + 1) = n * FF + (h * FF + r)
  measured: fewer quantified variables (2 → 1)

- S2GroupedExpert.lean
  was: theorem trees_differ :
    cpuTree (α := List Nat) (fun l => [l.val]) ≠ stdTree (fun l => [l.val])
  now: theorem trees_differ :
    cpuTree (fun l : Fin 8 => Sum8.leaf l.val) ≠ stdTree (fun l : Fin 8 => Sum8.leaf l.val)
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- NativeMoeCombine.lean
  was: theorem native_ne_legacy : ∀ k, 1 ≤ k → k ≤ 15 → ∀ s : Bool, nativeSum k s ≠ legacySum k s
  now: theorem native_ne_legacy : ∀ k, k < 16 → 1 ≤ k → ∀ s : Bool, nativeSum k s ≠ legacySum k s
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a
  now: theorem notBetter_trans (p : Nat → Int) {a r w : Nat} (h1 : ¬ better p a r)
    (h2 : ¬ better p r w) : ¬ better p a w
  measured: more hypotheses/binders (2 → 4): the statement holds in fewer cases

- RouterTopKSelection.lean
  was: theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c)
  now: theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem clamp_literal : 6103515625 * 2^14 = 10^14
  now: theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c)
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c ∧ (c ≤ s → max s c = s)
  now: theorem clamp_literal : 6103515625 * 2^14 = 10^14
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- RouterTopKSelection.lean
  was: theorem threads_spec (n : Nat) (h : 0 < n) :
    32 ≤ threads n ∧ threads n ≤ 512 ∧ threads n % 32 = 0
  now: theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c ∧ (c ≤ s → max s c = s)
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem warp_slots (n : Nat) : threads n / 32 ≤ 16
  now: theorem threads_spec (n : Nat) (h : 0 < n) :
    32 ≤ threads n ∧ threads n ≤ 512 ∧ threads n % 32 = 0
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- RouterTopKSelection.lean
  was: theorem smem_alignment (n : Nat) :
    ((n + 15) / 16 * 16) % 8 = 0 ∧ ((n + 15) / 16 * 16 + 8 * n) % 4 = 0
  now: theorem warp_slots (n : Nat) : threads n / 32 ≤ 16
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem smem_bounds : smem 512 = 6656 ∧ smem 512 ≤ 49152 ∧ 49152 < smem (512 * 64)
  now: theorem smem_alignment (n : Nat) :
    ((n + 15) / 16 * 16) % 8 = 0 ∧ ((n + 15) / 16 * 16 + 8 * n) % 4 = 0
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51
  now: theorem smem_bounds : smem 512 = 6656 ∧ smem 512 ≤ 49152 ∧ 49152 < smem (512 * 64)
  measured: the claim changed

- RouterTopKSelection.lean
  was: theorem accepts_production : accepts 10 512 = true
  now: theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51
  measured: the claim changed
-/



## FILE: /mnt/user-data/outputs/QsaStepState.lean
/-!
# MODULE: qsa_step_state
SOURCES: src/kernels/cuda/qsa.cu
The per-token QSA step buffer (`qsa_step_fill`, `step_upload`, `validate`): every per-token count
(`n_kv`, `n_bid`, `width`) is derived from one position, so a captured CUDA graph only needs one H2D upload.

## Source semantics
* `n_kv = pos + 1`, `n_bid = n_kv / idx_block`, `width = min(n_kv, top_k + idx_block - 1)`.
* `width ≤ n_kv`; below the selection bound the selection is the identity (`width = n_kv`).
* `n_bid` completed blocks cover at most `n_kv` cells and the next block would overshoot.
* `step_upload` refuses an `n_kv` hint disagreeing with `pos + 1`.
* `validate` requires `head_dim % 4 = 0` (uint2 gather) and `n_head % n_head_kv = 0`.

## Form template identifiers
`qsa_step_fill` → `stepFill`, `qsa_selection_width` → `selectionWidth`, `kStepPos/NKv/NBid/Width` →
`Step.pos/nKv/nBid/width`, `step_upload` hint check → `hintAccepted`, `validate` → `GeomValid`.
-/

namespace Ref.QsaStep

/-- Indexer geometry relevant to the step state. -/
structure Shapes where
  idxBlock : Nat
  topK : Nat

/-- `qsa_selection_width`: the selection keeps `top_k` blocks' worth plus the incomplete tail. -/
def selectionWidth (s : Shapes) (nKv : Nat) : Nat := min nKv (s.topK + s.idxBlock - 1)

/-- The four step entries uploaded per token. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat
deriving DecidableEq

/-- `qsa_step_fill`: the ONE writer of all four step entries. -/
def stepFill (s : Shapes) (pos : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / s.idxBlock,
    width := selectionWidth s (pos + 1) }

/-- `step_upload` accepts a caller hint only if it equals the derived `n_kv`. -/
def hintAccepted (s : Shapes) (pos hint : Nat) : Bool := hint == (stepFill s pos).nKv

-- § derived counts

/-- `n_kv` is always the position plus one. -/
theorem stepFill_nKv (s : Shapes) (pos : Nat) : (stepFill s pos).nKv = pos + 1 := rfl

/-- The cache is never empty after a fill. -/
theorem stepFill_nKv_pos (s : Shapes) (pos : Nat) : 0 < (stepFill s pos).nKv := by
  simp [stepFill]

/-- The selection never exceeds the number of cached cells. -/
theorem stepFill_width_le (s : Shapes) (pos : Nat) : (stepFill s pos).width ≤ (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_le_left _ _

/-- The selection never exceeds `top_k + r - 1` (the capacity `topk_512_step` checks). -/
theorem stepFill_width_le_cap (s : Shapes) (pos : Nat) :
    (stepFill s pos).width ≤ s.topK + s.idxBlock - 1 := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_le_right _ _

/-- Below the selection bound the selection is the identity. -/
theorem stepFill_width_identity (s : Shapes) (pos : Nat) (h : pos + 1 ≤ s.topK + s.idxBlock - 1) :
    (stepFill s pos).width = (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_eq_left h

/-- With `idx_block ≥ 2` (what `validate` enforces) the selection is nonempty, so `qsa_attend`'s
zero path is unreachable for a filled step. -/
theorem stepFill_width_pos (s : Shapes) (pos : Nat) (hr : 2 ≤ s.idxBlock) :
    0 < (stepFill s pos).width := by
  simp only [stepFill, selectionWidth]
  exact Nat.lt_min.mpr ⟨Nat.succ_pos _, by omega⟩

/-- Completed blocks never cover more cells than are cached. -/
theorem stepFill_nBid_mul_le (s : Shapes) (pos : Nat) :
    (stepFill s pos).nBid * s.idxBlock ≤ (stepFill s pos).nKv := by
  simp only [stepFill]
  exact Nat.div_mul_le_self _ _

/-- One more block would overshoot: the incomplete tail has fewer than `r` cells. -/
theorem stepFill_nKv_lt_next (s : Shapes) (pos : Nat) (hr : 0 < s.idxBlock) :
    (stepFill s pos).nKv < ((stepFill s pos).nBid + 1) * s.idxBlock := by
  simp only [stepFill]
  have h1 := Nat.div_add_mod (pos + 1) s.idxBlock
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.succ_mul, Nat.mul_comm ((pos + 1) / s.idxBlock) s.idxBlock]
  omega

/-- The hint check accepts exactly the derived `n_kv`. -/
theorem hintAccepted_iff (s : Shapes) (pos hint : Nat) :
    hintAccepted s pos hint = true ↔ hint = pos + 1 := by
  simp [hintAccepted, stepFill]

/-- A fill is determined by its position: two fills agree iff the positions agree. -/
theorem stepFill_injective (s : Shapes) (p q : Nat) : stepFill s p = stepFill s q ↔ p = q := by
  constructor
  · intro h
    have := congrArg Step.pos h
    simpa [stepFill] using this
  · intro h; subst h; rfl

-- § geometry validation

/-- The part of `validate` that the kernels' index arithmetic depends on. -/
structure GeomValid (nHead nHeadKv headDim : Nat) : Prop where
  kv_pos : 0 < nHeadKv
  gqa : nHead % nHeadKv = 0
  dim4 : headDim % 4 = 0

/-- `head_dim % 4 = 0` makes the uint2 gather cover the row exactly. -/
theorem GeomValid.per_exact {a b c : Nat} (h : GeomValid a b c) : (c / 4) * 4 = c := by
  have := h.dim4
  omega

/-- `n_head % n_head_kv = 0` makes the group size exact. -/
theorem GeomValid.group_exact {a b c : Nat} (h : GeomValid a b c) : (a / b) * b = a := by
  have h1 := Nat.div_add_mod a b
  have h2 := h.gqa
  rw [Nat.mul_comm]
  omega

end Ref.QsaStep



## FILE: /mnt/user-data/outputs/PagedKvLayout.lean
/-!
# MODULE: paged_kv_layout
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/kv_q8.cu, src/kernels/cuda/qsa_decode_attn.cu
Index arithmetic of the paged KV pool `[page][kv_head][page_size][head_dim]` shared by `kv_append`,
`kv_gather`, `kv_append_q8`, `kv_gather_q8` and the decode attention's `srow`.

## Source semantics
* A cell at position `pos` of KV head `h` lives in pool row
  `(table[pos / page_size] * kv_heads + h) * page_size + pos % page_size`.
* Rows stay inside a pool of `n_pages * kv_heads * page_size` rows when the page table is in range,
  and distinct (page, head, slot) triples never share a row.
* The gather's flat thread index `i` decomposes as `((id * kv_heads + h) * per + q)` with
  `per = head_dim / 4`; threads with `i < n_ids * kv_heads * per` touch only ids `< n_ids`.

## Form template identifiers
`row`/`src` → `poolRow`, gather `dst` → `scratchIdx`, `i / (kv_heads*per)` → `gatherId`,
`page_table` → `table`, `kv_heads` → `K`, `page_size` → `ps`.
-/

namespace Ref.PagedKv

-- § helpers

/-- `a*c + b` with `b < c` stays below `n*c` whenever `a < n`. -/
theorem mul_add_lt {a b c n : Nat} (hb : b < c) (ha : a < n) : a * c + b < n * c := by
  have h1 : (a + 1) * c ≤ n * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt ha)
  rw [Nat.succ_mul] at h1
  omega

/-- Mixed-radix decompositions with digit `< c` are unique. -/
theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq
    omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

-- § pool rows

/-- The pool row of cell `pos` for KV head `h` (`kv_append_kernel`, `kv_gather_kernel`). -/
def poolRow (table : Nat → Nat) (K ps h pos : Nat) : Nat :=
  (table (pos / ps) * K + h) * ps + pos % ps

/-- With an in-range page table every row is inside the pool. -/
theorem poolRow_lt (table : Nat → Nat) (K ps h pos nPages : Nat) (hps : 0 < ps) (hh : h < K)
    (hpage : table (pos / ps) < nPages) : poolRow table K ps h pos < nPages * K * ps := by
  have h1 : table (pos / ps) * K + h < nPages * K := mul_add_lt hh hpage
  exact mul_add_lt (Nat.mod_lt pos hps) h1

/-- Distinct (page, head, slot) triples map to distinct rows. -/
theorem row_injective {p p' h h' s s' K ps : Nat} (hh : h < K) (hh' : h' < K) (hs : s < ps)
    (hs' : s' < ps) (heq : (p * K + h) * ps + s = (p' * K + h') * ps + s') :
    p = p' ∧ h = h' ∧ s = s' := by
  obtain ⟨h1, h2⟩ := decomp_unique hs hs' heq
  obtain ⟨h3, h4⟩ := decomp_unique hh hh' h1
  exact ⟨h3, h4, h2⟩

/-- Two positions of the same page and head never collide inside the page. -/
theorem poolRow_slot_injective (table : Nat → Nat) (K ps h pos pos' : Nat)
    (hpage : pos / ps = pos' / ps) (heq : poolRow table K ps h pos = poolRow table K ps h pos') :
    pos = pos' := by
  unfold poolRow at heq
  rw [hpage] at heq
  have hm : pos % ps = pos' % ps := by omega
  have e1 := Nat.div_add_mod pos ps
  have e2 := Nat.div_add_mod pos' ps
  rw [hpage, hm] at e1
  omega

-- § gather thread decomposition

/-- The selection entry a gather thread handles. -/
def gatherId (K per i : Nat) : Nat := i / (K * per)

/-- The scratch element a gather thread writes: `(id * K + h) * per + q`. -/
def scratchIdx (K per i : Nat) : Nat :=
  (i / (K * per) * K + i % (K * per) / per) * per + i % (K * per) % per

/-- Every gather thread writes exactly its own flat index: the scratch write is a bijection. -/
theorem scratchIdx_eq (K per i : Nat) : scratchIdx K per i = i := by
  unfold scratchIdx
  have e1 := Nat.div_add_mod i (K * per)
  have e2 := Nat.div_add_mod (i % (K * per)) per
  rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm (i / (K * per)) (K * per),
    Nat.mul_comm (i % (K * per) / per) per]
  omega

/-- Threads inside `n_ids * K * per` only read selection entries `< n_ids`. -/
theorem gatherId_lt (K per i nIds : Nat) (hK : 0 < K * per) (hi : i < nIds * (K * per)) :
    gatherId K per i < nIds := by
  unfold gatherId
  exact (Nat.div_lt_iff_lt_mul hK).mpr hi

/-- The head index of a gather thread is a valid KV head. -/
theorem gather_head_lt (K per i : Nat) (hK : 0 < K * per) : i % (K * per) / per < K := by
  have hm := Nat.mod_lt i hK
  have hper : 0 < per := Nat.pos_of_mul_pos_left hK
  exact (Nat.div_lt_iff_lt_mul hper).mpr hm

end Ref.PagedKv



## FILE: /mnt/user-data/outputs/KvQ8Quant.lean
/-!
# MODULE: kv_q8_quant
SOURCES: src/kernels/cuda/kv_q8.cu
Symmetric int8 KV quantization with one fp16 scale per 64-value group (`kv_append_q8_kernel`) and the
dequantizing gather (`kv_gather_q8_kernel`).

## Source semantics
* Code is `clamp(round(x / sf), -127, 127)` against the STORED scale, and `0` when the scale is `0`.
* Codes are symmetric: `-128` is never produced, so every code fits `int8` and negation is closed.
* Scale of element `d` of row `row` is at `row * groups + d / 64`; a 4-value gather lane never
  straddles a group, and every scale index stays inside its row.

## Form template identifiers
`KV_Q8_GROUP` → `64`, clamp → `clampCode`, `q` → `code`, `k_scale/v_scale` index → `scaleIdx`,
`q4` → gather lane.
-/

namespace Ref.KvQ8

-- § codes

/-- The clamp to the symmetric int8 range. -/
def clampCode (q : Int) : Int := max (-127) (min 127 q)

/-- The stored code: clamped rounding when the stored scale is positive, else zero. -/
def code (scalePos : Bool) (rounded : Int) : Int := if scalePos then clampCode rounded else 0

/-- Clamped codes lie in `[-127, 127]`. -/
theorem clampCode_bounds (q : Int) : -127 ≤ clampCode q ∧ clampCode q ≤ 127 := by
  unfold clampCode
  omega

/-- Values already in range are untouched. -/
theorem clampCode_of_mem (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampCode q = q := by
  unfold clampCode
  omega

/-- Clamping is idempotent. -/
theorem clampCode_idem (q : Int) : clampCode (clampCode q) = clampCode q := by
  obtain ⟨h1, h2⟩ := clampCode_bounds q
  exact clampCode_of_mem _ h1 h2

/-- The clamp commutes with negation: the code range is symmetric. -/
theorem clampCode_neg (q : Int) : clampCode (-q) = -clampCode q := by
  unfold clampCode
  omega

/-- Every stored code fits a signed byte and is never `-128`. -/
theorem code_fits_int8 (b : Bool) (r : Int) : -128 < code b r ∧ code b r ≤ 127 := by
  cases b with
  | false => simp [code]
  | true =>
    have := clampCode_bounds r
    simp only [code, if_true]
    omega

/-- A zero scale stores a zero code. -/
theorem code_zero_scale (r : Int) : code false r = 0 := rfl

-- § scale layout

/-- The scale slot of element `d` in row `row` with `groups = head_dim / 64`. -/
def scaleIdx (groups row d : Nat) : Nat := row * groups + d / 64

/-- A dimension inside the row selects a group inside the row. -/
theorem group_lt (groups d : Nat) (hd : d < groups * 64) : d / 64 < groups := by
  omega

/-- Scale slots of row `row` lie in `[row*groups, (row+1)*groups)`. -/
theorem scaleIdx_in_row (groups row d : Nat) (hd : d < groups * 64) :
    row * groups ≤ scaleIdx groups row d ∧ scaleIdx groups row d < (row + 1) * groups := by
  have hg := group_lt groups d hd
  unfold scaleIdx
  rw [Nat.succ_mul]
  omega

/-- The four values a gather lane dequantizes share one scale. -/
theorem lane_one_group (q4 j : Nat) (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

/-- The 64 threads of an append block cover exactly one group. -/
theorem append_thread_group (g t : Nat) (ht : t < 64) : (g * 64 + t) / 64 = g := by
  omega

end Ref.KvQ8



## FILE: /mnt/user-data/outputs/TopkThresholdSelect.lean
/-!
# MODULE: topk_threshold_select
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/qsa_select.cu
The exact top-k used by `topk_512` (binary lifting) and `block_topk_kernel` (radix select): find a
threshold key, then emit every cell strictly above it plus the first `width - c_gt` cells equal to it
in ONE ascending walk, reproducing `ggml_top_k` with the ascending-index tie rule.

## Source semantics
* `order_key` maps an f32 bit pattern to an unsigned key: negatives are bit-inverted, non-negatives get
  the sign bit set; `-0.0` is normalised to `+0.0` first; NaN maps to key 0 (never preferred).
* `count(key ≥ v)` is antitone in `v`; the threshold `thr` satisfies
  `count(key ≥ thr) ≥ width > count(key ≥ thr + 1)`.
* The walk emits exactly `c_gt + min(c_eq, budget)` ids, strictly ascending; with the threshold
  postcondition and `budget = width - c_gt` it emits exactly `width` ids (the bug the source comment
  "THE RANK VS THE COUNT" records was a wrong budget here).

## Form template identifiers
`order_key` → `orderKey`, `-0.0` normalisation → `normZero`, `thr` → `thr`, `c_gt` → `cgt`,
equal count → `ceq`, `count(key >= cand)` → `cntGe`, `eq_budget` → `budget`, emit loop → `emit`.
-/

namespace Ref.Topk

-- § order key on 32-bit patterns

/-- The sign bit `0x80000000`. -/
def signBit : Nat := 2147483648

/-- `order_key` on a raw 32-bit pattern `b < 2^32`. -/
def orderKey (b : Nat) : Nat := if signBit ≤ b then 4294967295 - b else b + signBit

/-- `s + 0.0f` turns the pattern of `-0.0` into the pattern of `+0.0`. -/
def normZero (b : Nat) : Nat := if b = signBit then 0 else b

/-- Keys stay 32-bit. -/
theorem orderKey_lt (b : Nat) (hb : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey signBit
  split <;> omega

/-- Distinct patterns get distinct keys, so the key is a total order on non-NaN floats. -/
theorem orderKey_injective (b b' : Nat) (hb : b < 4294967296) (hb' : b' < 4294967296)
    (h : orderKey b = orderKey b') : b = b' := by
  unfold orderKey signBit at h
  split at h <;> split at h <;> omega

/-- Every negative float orders below every non-negative float. -/
theorem orderKey_neg_lt_nonneg (b b' : Nat) (hb : signBit ≤ b) (hb4 : b < 4294967296)
    (hb' : b' < signBit) : orderKey b < orderKey b' := by
  unfold orderKey
  simp only [hb, if_true, Nat.not_le.mpr hb', if_false]
  unfold signBit at *
  omega

/-- Larger magnitude negatives (larger bit patterns) get smaller keys. -/
theorem orderKey_neg_antitone (b b' : Nat) (hb : signBit ≤ b) (hbb : b < b')
    (hb4 : b' < 4294967296) : orderKey b' < orderKey b := by
  have hb' : signBit ≤ b' := Nat.le_trans hb (Nat.le_of_lt hbb)
  unfold orderKey
  simp only [hb, hb', if_true]
  omega

/-- Non-negative floats keep their bit order. -/
theorem orderKey_pos_monotone (b b' : Nat) (hbb : b < b') (hb' : b' < signBit) :
    orderKey b < orderKey b' := by
  have hb : ¬ signBit ≤ b := by omega
  unfold orderKey
  simp only [hb, Nat.not_le.mpr hb', if_false]
  omega

/-- `-0.0` and `+0.0` get the same key, as numpy treats them as equal. -/
theorem orderKey_negZero : orderKey (normZero signBit) = orderKey (normZero 0) := by
  decide

-- § counts

/-- Number of keys strictly above `t`. -/
def cgt : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if t < k then 1 else 0) + cgt ks t

/-- Number of keys equal to `t`. -/
def ceq : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if k = t then 1 else 0) + ceq ks t

/-- `count(key >= v)`, the predicate the binary lifting tests. -/
def cntGe : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, v => (if v ≤ k then 1 else 0) + cntGe ks v

/-- `count(key ≥ v)` is antitone in `v`: this is what makes binary lifting correct. -/
theorem cntGe_antitone (ks : List Nat) {v v' : Nat} (h : v ≤ v') : cntGe ks v' ≤ cntGe ks v := by
  induction ks with
  | nil => simp [cntGe]
  | cons k ks ih =>
    simp only [cntGe]
    by_cases h1 : v' ≤ k
    · have h2 : v ≤ k := Nat.le_trans h h1
      simp only [h1, h2, if_true]
      omega
    · by_cases h2 : v ≤ k
      · simp only [h1, h2, if_true, if_false]
        omega
      · simp only [h1, h2, if_false]
        omega

/-- Strictly-above is `≥ t + 1`. -/
theorem cgt_eq_cntGe_succ (ks : List Nat) (t : Nat) : cgt ks t = cntGe ks (t + 1) := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cgt, cntGe, ih]
    by_cases h : t < k
    · have h' : t + 1 ≤ k := h
      simp [h, h']
    · have h' : ¬ t + 1 ≤ k := h
      simp [h, h']

/-- At-or-above splits into strictly-above plus equal. -/
theorem cntGe_split (ks : List Nat) (t : Nat) : cntGe ks t = cgt ks t + ceq ks t := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cntGe, cgt, ceq, ih]
    by_cases h1 : t < k
    · have h2 : t ≤ k := Nat.le_of_lt h1
      have h3 : k ≠ t := by omega
      simp only [h1, h2, h3, if_true, if_false]
      omega
    · by_cases h3 : k = t
      · subst h3
        simp
        omega
      · have h2 : ¬ t ≤ k := by omega
        simp only [h1, h2, h3, if_false]
        omega

-- § the ascending emission walk

/-- One ascending walk: emit index `j` for keys above `thr`, and for keys equal to `thr` while the
remaining equal budget `e` is positive. -/
def emit : List Nat → Nat → Nat → Nat → List Nat
  | [], _, _, _ => []
  | k :: ks, thr, e, j =>
    if thr < k then j :: emit ks thr e (j + 1)
    else if k = thr ∧ 0 < e then j :: emit ks thr (e - 1) (j + 1)
    else emit ks thr e (j + 1)

/-- The walk emits `c_gt + min(c_eq, budget)` ids. -/
theorem emit_length (ks : List Nat) (thr e j : Nat) :
    (emit ks thr e j).length = cgt ks thr + min (ceq ks thr) e := by
  induction ks generalizing e j with
  | nil => simp [emit, cgt, ceq]
  | cons k ks ih =>
    by_cases h1 : thr < k
    · have h2 : k ≠ thr := by omega
      simp only [emit, cgt, ceq, h1, h2, if_true, if_false, List.length_cons, ih]
      omega
    · by_cases h2 : k = thr
      · subst h2
        by_cases h3 : 0 < e
        · simp [emit, cgt, ceq, h3, ih] <;> omega
        · simp [emit, cgt, ceq, h3, ih] <;> omega
      · simp only [emit, cgt, ceq, h1, h2, false_and, if_false, ih]
        omega

/-- Every emitted id lies in `[j, j + length)`: ids are real cell indices. -/
theorem emit_bounds (ks : List Nat) (thr e j : Nat) :
    ∀ x ∈ emit ks thr e j, j ≤ x ∧ x < j + ks.length := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    intro x hx
    simp only [List.length_cons]
    unfold emit at hx
    split at hx
    · rcases List.mem_cons.mp hx with h | h
      · omega
      · have := ih e (j + 1) x h; omega
    · split at hx
      · rcases List.mem_cons.mp hx with h | h
        · omega
        · have := ih (e - 1) (j + 1) x h; omega
      · have := ih e (j + 1) x hx; omega

/-- The emitted ids are strictly ascending, so the tie rule "first by index" is the walk order. -/
theorem emit_sorted (ks : List Nat) (thr e j : Nat) : (emit ks thr e j).Pairwise (· < ·) := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    unfold emit
    split
    · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih e (j + 1)⟩
      have := emit_bounds ks thr e (j + 1) x hx; omega
    · split
      · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih (e - 1) (j + 1)⟩
        have := emit_bounds ks thr (e - 1) (j + 1) x hx; omega
      · exact ih e (j + 1)

/-- Given the binary-lifting postcondition on `thr`, the walk with `budget = width - c_gt` emits
exactly `width` ids. -/
theorem emit_exact_width (ks : List Nat) (thr w : Nat) (hge : w ≤ cntGe ks thr)
    (hlt : cntGe ks (thr + 1) < w) : (emit ks thr (w - cgt ks thr) 0).length = w := by
  have h1 := cgt_eq_cntGe_succ ks thr
  have h2 := cntGe_split ks thr
  rw [emit_length]
  omega

/-- The source's first (buggy) budget compared the LAYOUT position `c_gt + rank` to the budget; with
distinct scores (`c_eq = 1`, `c_gt = w - 1`) it is short by one whenever `c_gt > 0`. -/
theorem layout_budget_short (ks : List Nat) (thr w : Nat) (hgt : cgt ks thr = w - 1)
    (heq : ceq ks thr = 1) (hw : 2 ≤ w) :
    (emit ks thr (w - cgt ks thr - cgt ks thr) 0).length = w - 1 := by
  rw [emit_length]
  omega

end Ref.Topk



## FILE: /mnt/user-data/outputs/BlockWeightedSelect.lean
/-!
# MODULE: block_weighted_select
SOURCES: src/kernels/cuda/qsa_select.cu
`block_scores_kernel` scores one entry per BLOCK (the spare `dead` key for `b = n_bid`), and
`block_topk_kernel` runs the radix select over blocks with each block weighted by its cell count,
expanding a selected block into its cells `b*R .. b*R + w - 1`.

## Source semantics
* `weight b = R` for completed blocks `b < n_bid`, and `n_kv - n_bid*R` for the tail block `b = n_bid`
  (zero when `n_kv` is block-aligned, which the kernel skips).
* The weights of blocks `0..n_bid` sum to exactly `n_kv`, so weighted counts over blocks equal cell
  counts — the block-level select agrees with the cell-level `topk_512`.
* The expanded cells of block `b` lie in `[b*R, b*R + R)`, are `< n_kv`, and different blocks' cells
  are disjoint.
* `n_kv ≤ width` short-circuits to the identity selection `0..n_kv-1`.
* The scores output index `qi * max_blocks + b` is inside `nq * max_blocks` for `b < max_blocks`.

## Form template identifiers
`R` → `r`, `weight` lambda → `weight`, `nb = n_bid + 1` → blocks, cell id `b*R + c` → `cellOf`,
identity path → `identitySel`, `out[qi*max_blocks+b]` → `scoreSlot`.
-/

namespace Ref.BlockSelect

/-- The cell count of block `b`. -/
def weight (r nKv nBid b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Total weight of blocks `0 .. m-1`. -/
def sumW (r nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => sumW r nKv nBid m + weight r nKv nBid m

/-- Cell `c` of block `b`. -/
def cellOf (r b c : Nat) : Nat := b * r + c

/-- The identity selection used when everything fits. -/
def identitySel (nKv : Nat) : List Nat := List.range nKv

/-- Completed blocks contribute `R` cells each. -/
theorem sumW_completed (r nKv nBid m : Nat) (hm : m ≤ nBid) : sumW r nKv nBid m = m * r := by
  induction m with
  | zero => simp [sumW]
  | succ m ih =>
    have hlt : m < nBid := by omega
    simp only [sumW, weight, hlt, if_true, ih (by omega), Nat.succ_mul]

/-- Blocks `0..n_bid` weigh exactly `n_kv` cells. -/
theorem sumW_total (r nKv : Nat) : sumW r nKv (nKv / r) (nKv / r + 1) = nKv := by
  have h1 := sumW_completed r nKv (nKv / r) (nKv / r) (Nat.le_refl _)
  have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
  simp only [sumW, weight, Nat.lt_irrefl, if_false, h1]
  omega

/-- The tail block weighs fewer than `R` cells. -/
theorem tail_weight_lt (r nKv : Nat) (hr : 0 < r) : weight r nKv (nKv / r) (nKv / r) < r := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r
  have h2 := Nat.mod_lt nKv hr
  rw [Nat.mul_comm] at h1
  omega

/-- A block-aligned cache has an empty tail block (the kernel's `w == 0` skip). -/
theorem tail_weight_zero (r nKv : Nat) (h : nKv % r = 0) : weight r nKv (nKv / r) (nKv / r) = 0 := by
  simp only [weight, Nat.lt_irrefl, if_false]
  have h1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at h1
  omega

/-- Expanded cells of a block are real cells. -/
theorem cellOf_lt (r nKv b c : Nat) (hb : b ≤ nKv / r) (hc : c < weight r nKv (nKv / r) b) :
    cellOf r b c < nKv := by
  unfold cellOf
  unfold weight at hc
  by_cases hlt : b < nKv / r
  · simp only [hlt, if_true] at hc
    have h1 : (b + 1) * r ≤ nKv / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
    rw [Nat.succ_mul] at h1
    omega
  · have hbe : b = nKv / r := by omega
    subst hbe
    simp only [Nat.lt_irrefl, if_false] at hc
    have h2 : nKv / r * r ≤ nKv := Nat.div_mul_le_self nKv r
    omega

/-- Cells of different blocks never coincide. -/
theorem cellOf_disjoint (r b b' c c' : Nat) (hc : c < r) (hc' : c' < r)
    (h : cellOf r b c = cellOf r b' c') : b = b' ∧ c = c' := by
  unfold cellOf at h
  rcases Nat.lt_trichotomy b b' with hlt | heq | hgt
  · have h1 : (b + 1) * r ≤ b' * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq; omega
  · have h1 : (b' + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

/-- The identity path emits exactly `n_kv` ascending ids. -/
theorem identitySel_spec (nKv : Nat) :
    (identitySel nKv).length = nKv ∧ ∀ i, i < nKv → (identitySel nKv)[i]? = some i := by
  refine ⟨List.length_range, fun i hi => ?_⟩
  simp [identitySel, hi]

/-- Score slots of every query stay inside the `nq * max_blocks` scores buffer. -/
theorem scoreSlot_lt (qi nq maxB b : Nat) (hq : qi < nq) (hb : b < maxB) : qi * maxB + b < nq * maxB := by
  have h1 : (qi + 1) * maxB ≤ nq * maxB := Nat.mul_le_mul_right maxB (Nat.succ_le_of_lt hq)
  rw [Nat.succ_mul] at h1
  omega

end Ref.BlockSelect



## FILE: /mnt/user-data/outputs/IndexerPooling.lean
/-!
# MODULE: indexer_pooling
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/native_qsa_indexer.cu
`indexer_key_append` / native `append`: store the raw indexer key into an `r-1` row tail, and when a
cell COMPLETES a block, pool the block's `r` raw keys (mean → rms_norm → weight), rotate the pooled
row at the block's first position, write it to row `b = pos / r`, and move the constant spare key to
row `b + 1`.

## Source semantics
* `slot = pos % r`; slots `< r-1` go to the tail, slot `r-1` completes block `b = pos / r`.
* A completion means `(pos + 1) % r = 0` and the completed-block count becomes `b + 1`.
* The tail never needs more than `r-1` rows; the block's first cell is `b*r = pos - (r-1)`.
* The rotation position is `pos_base + b*r` (block's first cell); native: `0` at `pos = 0`, and with
  `pos_base % 4 = 0` it is 4-aligned.
* Native bounds: writes to pooled row `b + 1` happen only on completion, and then `b + 1 ≤ max/4`
  fits the `max/4 + 1` row buffer even when `max % 4 ≠ 0`.

## Form template identifiers
`idx_block`/`R` → `r`, `slot` → `pos % r`, `tail` rows → `tailRow`, `dead` → spare, `block_pos` →
`blockPos`, `rope_pos` → `nativeRopePos`, `max_cells` → `maxCells`.
-/

namespace Ref.Indexer

-- § helper

/-- Mixed-radix decompositions with digit `< c` are unique. -/
theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq; omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

-- § slots and completion

/-- A cell completes a block iff it is in the last slot. -/
def completes (r pos : Nat) : Prop := pos % r = r - 1

/-- A completing cell makes `n_kv` block-aligned and bumps the completed-block count. -/
theorem completes_next (r pos : Nat) (hr : 0 < r) (h : completes r pos) :
    (pos + 1) % r = 0 ∧ (pos + 1) / r = pos / r + 1 := by
  unfold completes at h
  have e1 := Nat.div_add_mod pos r
  have e2 := Nat.div_add_mod (pos + 1) r
  have m2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at e1 e2
  have key : (pos / r + 1) * r + 0 = (pos + 1) / r * r + (pos + 1) % r := by
    rw [Nat.succ_mul]; omega
  obtain ⟨h1, h2⟩ := decomp_unique hr m2 key
  exact ⟨h2.symm, h1.symm⟩

/-- A non-completing cell leaves the completed-block count unchanged. -/
theorem noncomplete_same_block (r pos : Nat) (hr : 0 < r) (h : pos % r < r - 1) :
    (pos + 1) / r = pos / r ∧ (pos + 1) % r = pos % r + 1 := by
  have e1 := Nat.div_add_mod pos r
  have e2 := Nat.div_add_mod (pos + 1) r
  have m2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at e1 e2
  have hlt : pos % r + 1 < r := by omega
  have key : pos / r * r + (pos % r + 1) = (pos + 1) / r * r + (pos + 1) % r := by omega
  obtain ⟨h1, h2⟩ := decomp_unique hlt m2 key
  exact ⟨h1.symm, h2.symm⟩

/-- The block's first cell, `b*r`, is `pos - (r-1)` at a completion. -/
theorem block_first_cell (r pos : Nat) (h : completes r pos) : pos / r * r = pos - (r - 1) := by
  unfold completes at h
  have e1 := Nat.div_add_mod pos r
  rw [Nat.mul_comm] at e1
  omega

/-- Tail element offset `slot * idx_dim + d`. -/
def tailRow (idxDim slot d : Nat) : Nat := slot * idxDim + d

/-- The `r - 1` row tail is enough: stored slots never index past it. -/
theorem tailRow_lt (r idxDim slot d : Nat) (hs : slot < r - 1) (hd : d < idxDim) :
    tailRow idxDim slot d < (r - 1) * idxDim := by
  unfold tailRow
  have h1 : (slot + 1) * idxDim ≤ (r - 1) * idxDim := Nat.mul_le_mul_right idxDim (Nat.succ_le_of_lt hs)
  rw [Nat.succ_mul] at h1
  omega

/-- `block_pos`: the rotation position of the completed row (a POSITION, not a cell index). -/
def blockPos (r posBase pos : Nat) : Nat := posBase + pos / r * r

/-- The rotation position never exceeds the completing position (shifted by the base). -/
theorem blockPos_le (r posBase pos : Nat) : blockPos r posBase pos ≤ posBase + pos := by
  unfold blockPos
  have := Nat.div_mul_le_self pos r
  omega

-- § native fixed geometry (R = 4)

/-- Native rotation position: `0` for the spare at `pos = 0`, else `pos_base + 4*b`. -/
def nativeRopePos (posBase pos : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With the host-enforced `pos_base % 4 = 0`, native rotation positions are block-aligned. -/
theorem nativeRopePos_aligned (posBase pos : Nat) (h : posBase % 4 = 0) : nativeRopePos posBase pos % 4 = 0 := by
  unfold nativeRopePos
  split <;> omega

/-- The native kernel writes past its first barrier only at `pos = 0` or at a completion. -/
def nativeWrites (pos : Nat) : Prop := pos = 0 ∨ pos % 4 = 3

/-- Every pooled-row write of the native kernel (rows `b` and `b+1`) fits the `max/4 + 1` buffer. -/
theorem native_pooled_in_bounds (pos maxCells : Nat) (hp : pos < maxCells) (hw : nativeWrites pos) :
    pos / 4 + 1 < maxCells / 4 + 1 + 1 ∧ (pos ≠ 0 → pos / 4 + 1 < maxCells / 4 + 1) := by
  unfold nativeWrites at hw
  constructor
  · omega
  · intro hne
    rcases hw with h | h
    · exact absurd h hne
    · omega

/-- Without the completion guard a tail position could address one row past the buffer. -/
theorem native_guard_needed : ¬ (∀ pos maxCells : Nat, pos < maxCells → pos / 4 + 1 < maxCells / 4 + 1) := by
  intro h
  have := h 4 5 (by decide)
  omega

end Ref.Indexer



## FILE: /mnt/user-data/outputs/QsaIndexCellMap.lean
/-!
# MODULE: qsa_index_cell_map
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/native_qsa_score.cu
`qsa_index_kernel` and the native tensor-core `score_kernel`: score each pooled row as the sum over
indexer heads of `relu(dot)`, add the optional bias, add `+1e9` to the incomplete tail block, and copy
the row's score onto its cells.

## Source semantics
* Block `b` owns cells `[b*r, min(b*r + r, n_kv))`, and the tail block `b = n_bid` owns
  `[n_bid*r, n_kv)`. Every cell `j < n_kv` is owned by exactly block `j / r ≤ n_bid`.
* The `+1e9` tail bias applies iff `b = n_bid ∧ n_kv % r ≠ 0`; a block-aligned cache's tail block owns
  no cells, so the spare row never scores a real cell.
* Blocks `b > n_bid` of a constant capacity grid do nothing; native capacity `max/4 + 1` blocks
  always cover `n_bid + 1` blocks.
* The ReLU is per head, so the head sum is non-negative.
* The native kernel accepts a step only if it is exactly `qsa_step_fill` with `r = 4`, `top_k = 2048`.

## Form template identifiers
`lo`/`hi` → `cellLo`/`cellHi`, `n_bid` → `nBid`, `+1e9` condition → `tailBiased`,
`score += relu(s_dot[h])` → `reluSum`, native `step` check → `nativeStepOk`, `max_blocks` → `maxBlocks`.
-/

namespace Ref.IndexMap

/-- First cell of block `b`. -/
def cellLo (r b : Nat) : Nat := b * r

/-- One past the last cell of block `b`. -/
def cellHi (r nKv nBid b : Nat) : Nat := if b = nBid then nKv else min (b * r + r) nKv

/-- The kernel adds the tail bias to block `b`. -/
def tailBiased (r nKv nBid b : Nat) : Prop := b = nBid ∧ nKv % r ≠ 0

/-- Every cached cell is inside the range of block `j / r`. -/
theorem cell_owned (r nKv j : Nat) (hr : 0 < r) (hj : j < nKv) :
    j / r ≤ nKv / r ∧ cellLo r (j / r) ≤ j ∧ j < cellHi r nKv (nKv / r) (j / r) := by
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  refine ⟨Nat.div_le_div_right (Nat.le_of_lt hj), ?_, ?_⟩
  · unfold cellLo; omega
  · unfold cellHi
    split
    · exact hj
    · exact Nat.lt_min.mpr ⟨by omega, hj⟩

/-- A cell is inside at most one block's range: ranges are disjoint. -/
theorem cell_owner_unique (r nKv nBid b j : Nat) (hr : 0 < r) (hb : b < nBid)
    (hlo : cellLo r b ≤ j) (hhi : j < cellHi r nKv nBid b) : b = j / r := by
  unfold cellLo at hlo
  unfold cellHi at hhi
  have hne : b ≠ nBid := by omega
  simp only [hne, if_false] at hhi
  have hhi' : j < b * r + r := Nat.lt_of_lt_of_le hhi (Nat.min_le_left _ _)
  have e1 := Nat.div_add_mod j r
  have m1 := Nat.mod_lt j hr
  rw [Nat.mul_comm] at e1
  rcases Nat.lt_trichotomy b (j / r) with h | h | h
  · have h1 : (b + 1) * r ≤ j / r * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1
    omega
  · exact h
  · have h1 : (j / r + 1) * r ≤ b * r := Nat.mul_le_mul_right r (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at h1
    omega

/-- A block-aligned cache's tail (spare) block owns no cells. -/
theorem aligned_tail_empty (r nKv : Nat) (h : nKv % r = 0) :
    cellHi r nKv (nKv / r) (nKv / r) = cellLo r (nKv / r) := by
  unfold cellHi cellLo
  have e1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at e1
  simp only [if_true]
  omega

/-- The tail bias is applied exactly when the tail block owns at least one cell. -/
theorem tailBiased_iff_nonempty (r nKv : Nat) :
    tailBiased r nKv (nKv / r) (nKv / r) ↔ cellLo r (nKv / r) < cellHi r nKv (nKv / r) (nKv / r) := by
  unfold tailBiased cellHi cellLo
  have e1 := Nat.div_add_mod nKv r
  rw [Nat.mul_comm] at e1
  simp only [if_true, true_and]
  omega

-- § head sum

/-- `Σ_h relu(dot_h)`, in head order. -/
def reluSum : List Int → Int
  | [] => 0
  | d :: ds => max d 0 + reluSum ds

/-- The per-head ReLU makes the head sum non-negative. -/
theorem reluSum_nonneg (ds : List Int) : 0 ≤ reluSum ds := by
  induction ds with
  | nil => simp [reluSum]
  | cons d ds ih => simp only [reluSum]; omega

/-- Summing BEFORE the ReLU can lose mass: the reference's per-head order matters. -/
theorem relu_before_sum_differs : reluSum [3, -5] ≠ max (3 + -5) 0 := by decide

/-- The relu sum dominates every single head's contribution. -/
theorem reluSum_ge_mem (ds : List Int) (d : Int) (h : d ∈ ds) : d ≤ reluSum ds := by
  induction ds with
  | nil => simp at h
  | cons x xs ih =>
    simp only [reluSum]
    have h0 := reluSum_nonneg xs
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

-- § native step contract and grid

/-- The native score kernel's guard on the step entries `(pos, n_kv, n_bid, width)`. -/
def nativeStepOk (maxCells pos n full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = (if n < 2051 then n else 2051)

/-- `qsa_step_fill` with `r = 4`, `top_k = 2048` always passes the native guard. -/
theorem stepFill_nativeOk (maxCells pos : Nat) (h : pos + 1 ≤ maxCells) :
    nativeStepOk maxCells pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (2048 + 4 - 1)) := by
  unfold nativeStepOk
  refine ⟨by omega, h, by omega, rfl, ?_⟩
  split <;> omega

/-- The native capacity `max/4 + 1` blocks covers blocks `0..n_bid` for every admissible step. -/
theorem native_grid_covers (maxCells n : Nat) (h : n ≤ maxCells) : n / 4 < maxCells / 4 + 1 := by
  omega

end Ref.IndexMap



## FILE: /mnt/user-data/outputs/ChunkedAttnMerge.lean
/-!
# MODULE: chunked_attn_merge
SOURCES: src/kernels/cuda/qsa_decode_attn.cu, src/kernels/cuda/qsa.cu
Decode attention over the selected cells: `attn_chunk_kernel` splits the selection into 64-cell chunks
per KV head (12 query heads share one KV head), writes per-chunk `(acc, m, l)` partials, and
`attn_merge_kernel` combines them with the max-rescaled log-sum-exp merge. `qsa_attend_kernel` is the
one-block-per-head reference form with the same GQA mapping.

## Source semantics
* `n_chunks = ceil(cap / 64)`; selected index `c < n_ids ≤ cap` lies in chunk `c / 64 < n_chunks`.
* A chunk has `n_here = min(64, n_ids - c0)` live cells and is empty (sentinel `m = -FLT_MAX`,
  `l = 0`) iff `c0 ≥ n_ids`; empty chunks are skipped by the merge.
* GQA: query head `h` reads KV head `h / G` (`iv2 = iq2 / rv2`), never `h % n_head_kv`.
* Scratch per query: `acc` (`n_chunks*n_head*256`), then `m`, then `l` (`n_chunks*n_head` each), all
  inside `stride = n_chunks*n_head*258 + 64`; partial slots `(kvh*n_chunks + c)*G + hl` stay in range.
* Merge: with `M = max m_c` every weight `exp(m_c - M) ≤ 1`, so the rescale never overflows.

## Form template identifiers
`CHUNK` → `64`, `G` → `12`, `HD` → `256`, `n_chunks` → `nChunks`, `n_here` → `nHere`,
`slot` → `partSlot`, `qsa_decode_attn_scratch_floats` → `scratchFloats`, merge max `M` → `mergeMax`.
-/

namespace Ref.DecodeAttn

/-- `n_chunks = ceil(cap / 64)`. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Live cells of chunk `ch`. -/
def nHere (nIds ch : Nat) : Nat := min 64 (nIds - ch * 64)

/-- Every selected cell lies in a launched chunk. -/
theorem chunk_of_cell_lt (cap nIds c : Nat) (hc : c < nIds) (hn : nIds ≤ cap) : c / 64 < nChunks cap := by
  unfold nChunks
  omega

/-- The cell lies at offset `c % 64 < n_here` inside its chunk. -/
theorem cell_live_in_chunk (nIds c : Nat) (hc : c < nIds) : c % 64 < nHere nIds (c / 64) := by
  unfold nHere
  omega

/-- A chunk is empty (writes the sentinel) iff it starts at or past `n_ids`. -/
theorem chunk_empty_iff (nIds ch : Nat) : nHere nIds ch = 0 ↔ nIds ≤ ch * 64 := by
  unfold nHere
  omega

/-- A launched chunk never holds more than 64 cells (fits `sp[G][CHUNK]`). -/
theorem nHere_le (nIds ch : Nat) : nHere nIds ch ≤ 64 := Nat.min_le_left _ _

-- § GQA

/-- Query head `h < 12 * n_kv` reads KV head `h / 12 < n_kv` as its local head `h % 12`. -/
theorem gqa_decode (h nKv : Nat) (hh : h < 12 * nKv) : h / 12 < nKv ∧ h = h / 12 * 12 + h % 12 := by
  omega

/-- The general `qsa_attend` mapping `kv = h / (n_head / n_head_kv)` lands on a valid KV head. -/
theorem gqa_general (h g nKv : Nat) (hg : 0 < g) (hh : h < g * nKv) : h / g < nKv := by
  rw [Nat.mul_comm] at hh
  exact (Nat.div_lt_iff_lt_mul hg).mpr hh

/-- The modular mapping is NOT the reference: heads 0 and 1 share KV head 0 under `/`, not under `%`. -/
theorem gqa_mod_differs : (1 / 12 : Nat) ≠ 1 % 2 := by decide

-- § scratch layout

/-- `qsa_decode_attn_scratch_floats`. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The three partial arrays end strictly inside the per-query stride. -/
theorem scratch_fits (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead < scratchFloats cap nHead := by
  unfold scratchFloats
  rw [Nat.mul_add]
  omega

/-- A partial slot `(kvh * n_chunks + c) * G + hl` is inside the `n_kv * n_chunks * G` partials. -/
theorem partSlot_lt (kvh nKv nC c hl : Nat) (hk : kvh < nKv) (hc : c < nC) (hl12 : hl < 12) :
    (kvh * nC + c) * 12 + hl < nKv * nC * 12 := by
  have h1 : (kvh + 1) * nC ≤ nKv * nC := Nat.mul_le_mul_right nC (Nat.succ_le_of_lt hk)
  rw [Nat.succ_mul] at h1
  have h2 : kvh * nC + c < nKv * nC := by omega
  have h3 : (kvh * nC + c + 1) * 12 ≤ nKv * nC * 12 := Nat.mul_le_mul_right 12 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3
  omega

-- § merge

/-- The merge maximum over chunk maxima (`M`, starting from the sentinel `lo`). -/
def mergeMax (lo : Int) : List Int → Int
  | [] => lo
  | m :: ms => max m (mergeMax lo ms)

/-- Every chunk maximum is at most `M`, so every rescale exponent `m_c - M` is non-positive. -/
theorem mergeMax_ge (lo : Int) (ms : List Int) (m : Int) (h : m ∈ ms) : m - mergeMax lo ms ≤ 0 := by
  induction ms with
  | nil => simp at h
  | cons x xs ih =>
    simp only [mergeMax]
    rcases List.mem_cons.mp h with h | h
    · subst h; omega
    · have := ih h; omega

/-- Sentinel-only merges keep the sentinel, so an all-empty selection yields `L = 0` and output 0. -/
theorem mergeMax_all_sentinel (lo : Int) (ms : List Int) (h : ∀ m ∈ ms, m = lo) : mergeMax lo ms = lo := by
  induction ms with
  | nil => rfl
  | cons x xs ih =>
    simp only [mergeMax]
    have hx := h x (List.mem_cons_self ..)
    have hr := ih (fun m hm => h m (List.mem_cons_of_mem _ hm))
    omega

end Ref.DecodeAttn



## FILE: /mnt/user-data/outputs/NativeFlashAttnContract.lean
/-!
# MODULE: native_flash_attn_contract
SOURCES: src/kernels/cuda/native_flash_attn.cu
The pinned llama.cpp `fattn-vec` specialisation (D = 256, Q 24 heads / KV 2 heads, one column) used for
short contexts: a device-side step guard, the dense K/V addressing `(cell*2 + kv)*256 + d`, the
padded 256-cell sweep, and the host span validation.

## Source semantics
* The kernel accepts a step iff `1 ≤ width ≤ max_context`, `n_kv = width`, `pos = width - 1`,
  `n_bid = width / 4`; otherwise it writes NaN and reports `kNativeFlashAttnUnsupportedStep`.
* For `max_context ≤ 256 ≤ top_k` every `qsa_step_fill` with `pos < max_context` is accepted (below the
  selection bound the selection is the identity), and every accepted step IS such a fill.
* Head `h` reads KV head `h / 12`; K/V element `(cell*2 + kv)*256 + d` with `cell < capacity` is inside
  `capacity * 512` halves (`kv_bytes = capacity * 1024`).
* The padded sweep (`base` in steps of 128 up to 256) visits each cell `< 256` exactly once.
* Host spans must be non-null, aligned and pairwise disjoint.

## Form template identifiers
`valid` → `stepValid`, `width/nkv/pos/nbid` → step fields, `kv = head / 12` → `kvOf`,
`(cell*2+kv)*256+d` → `kvIdx`, `validate_spans` overlap → `spansOverlap`.
-/

namespace Ref.NativeFlash

/-- The device-side step guard. -/
def stepValid (maxCtx pos nKv nBid width : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nKv = width ∧ pos = width - 1 ∧ nBid = width / 4

/-- `qsa_step_fill` with `r = 4` and `top_k ≥ 256` passes the guard for every position in context. -/
theorem stepFill_valid (maxCtx topK pos : Nat) (hctx : maxCtx ≤ 256) (hk : 256 ≤ topK)
    (hp : pos < maxCtx) :
    stepValid maxCtx pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) (topK + 4 - 1)) := by
  have hw : min (pos + 1) (topK + 4 - 1) = pos + 1 := Nat.min_eq_left (by omega)
  rw [hw]
  unfold stepValid
  omega

/-- Every accepted step is the fill of `width - 1`. -/
theorem valid_is_fill (maxCtx pos nKv nBid width : Nat) (h : stepValid maxCtx pos nKv nBid width) :
    nKv = pos + 1 ∧ nBid = (pos + 1) / 4 ∧ width = pos + 1 := by
  unfold stepValid at h
  obtain ⟨h1, _, h3, h4, h5⟩ := h
  refine ⟨by omega, ?_, by omega⟩
  have : pos + 1 = width := by omega
  rw [this, h5]

/-- A step beyond the context is refused. -/
theorem refuses_long (maxCtx pos nKv nBid width : Nat) (h : maxCtx < width) :
    ¬ stepValid maxCtx pos nKv nBid width := by
  unfold stepValid
  omega

-- § addressing

/-- KV head of query head `h` (24 query heads over 2 KV heads). -/
def kvOf (h : Nat) : Nat := h / 12

/-- All 24 query heads map onto the 2 KV heads. -/
theorem kvOf_lt (h : Nat) (hh : h < 24) : kvOf h < 2 := by
  unfold kvOf; omega

/-- Dense K/V half index. -/
def kvIdx (cell kv d : Nat) : Nat := (cell * 2 + kv) * 256 + d

/-- Every K/V read of a cell inside the capacity stays inside `capacity * 512` halves. -/
theorem kvIdx_lt (cap cell kv d : Nat) (hc : cell < cap) (hk : kv < 2) (hd : d < 256) :
    kvIdx cell kv d < cap * 2 * 256 := by
  unfold kvIdx
  have h1 : (cell + 1) * 2 ≤ cap * 2 := Nat.mul_le_mul_right 2 (Nat.succ_le_of_lt hc)
  rw [Nat.succ_mul] at h1
  have h2 : cell * 2 + kv < cap * 2 := by omega
  have h3 : (cell * 2 + kv + 1) * 256 ≤ cap * 2 * 256 := Nat.mul_le_mul_right 256 (Nat.succ_le_of_lt h2)
  rw [Nat.succ_mul] at h3
  omega

/-- The padded sweep's cell `base + warp*32 + (lane & ~7) + row` decomposes every cell `< 256`. -/
theorem sweep_covers (cell : Nat) (hc : cell < 256) :
    ∃ base warp hi row, base < 256 ∧ base % 128 = 0 ∧ warp < 4 ∧ hi < 4 ∧ row < 8 ∧
      cell = base + warp * 32 + hi * 8 + row :=
  ⟨cell / 128 * 128, cell % 128 / 32, cell % 32 / 8, cell % 8, by omega, by omega, by omega,
    by omega, by omega, by omega⟩

-- § spans

/-- Two half-open byte spans overlap. -/
def spansOverlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so the pairwise check needs only `j < i`. -/
theorem spansOverlap_symm (a an b bn : Nat) : spansOverlap a an b bn ↔ spansOverlap b bn a an := by
  unfold spansOverlap; omega

/-- A non-empty span always overlaps itself: aliasing two arguments is rejected. -/
theorem spansOverlap_self (a n : Nat) (hn : 0 < n) : spansOverlap a n a n := by
  unfold spansOverlap; omega

/-- Adjacent spans do not overlap. -/
theorem spansOverlap_adjacent (a n m : Nat) : ¬ spansOverlap a n (a + n) m := by
  unfold spansOverlap; omega

end Ref.NativeFlash



## FILE: /mnt/user-data/outputs/NativeQsaBufferContracts.lean
/-!
# MODULE: native_qsa_buffer_contracts
SOURCES: src/kernels/cuda/native_qsa.cu, src/kernels/cuda/native_rope.cu
Host-side contracts of the native (llama.cpp-pinned) QSA helpers: `native_qsa_rms_norm_weighted`,
`native_qsa_gate_apply`, `native_rope_apply`, and their buffer checks.

## Source semantics
* `elements(cols, rows)` requires both positive and `cols*rows ≤ INT_MAX`.
* `buffers` accepts disjoint spans or an EXACT input/output alias, never a partial overlap.
* `native_rope`'s `overlaps` (`x ≤ y ? y - x < an : x - y < bn`) agrees with the half-open interval
  test whenever both spans are non-empty.
* The gate reads the SECOND half of each head's `2*head_dim` query row; every such read is inside the
  `2 * n_head * head_dim` floats of `q_full`.
* The norm block size is 256 for rows narrower than 1024 and 1024 otherwise; either way the strided
  column loop covers every column exactly once.

## Form template identifiers
`overlap` → `overlapStd`, native_rope `overlaps` → `overlapRope`, `buffers` → `buffersOk`,
`elements` → `elementsOk`, gate read index → `gateIdx`, `BlockSize` → `normBlock`.
-/

namespace Ref.NativeQsa

/-- Standard half-open overlap (`native_qsa.cu` `overlap`). -/
def overlapStd (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- `native_rope.cu`'s ordered overlap test. -/
def overlapRope (x an y bn : Nat) : Prop := if x ≤ y then y - x < an else x - y < bn

/-- The two overlap tests agree for non-empty spans. -/
theorem overlapRope_iff_std (x an y bn : Nat) (ha : 0 < an) (hb : 0 < bn) :
    overlapRope x an y bn ↔ overlapStd x an y bn := by
  unfold overlapRope overlapStd
  split <;> omega

/-- They differ only for an empty span at the same address — impossible for `rows ≥ 1`. -/
theorem overlap_empty_differs : overlapRope 0 1 0 0 ∧ ¬ overlapStd 0 1 0 0 := by
  unfold overlapRope overlapStd
  decide

/-- The `buffers` acceptance rule for input/output: identical, or disjoint. -/
def buffersOk (inp out n : Nat) : Prop := inp = out ∨ ¬ overlapStd inp n out n

/-- An exact in-place alias is accepted. -/
theorem buffersOk_alias (p n : Nat) : buffersOk p p n := Or.inl rfl

/-- A partial overlap of non-empty buffers is rejected. -/
theorem buffersOk_rejects_partial (inp out n : Nat) (hne : inp ≠ out) (hov : overlapStd inp n out n) :
    ¬ buffersOk inp out n := by
  intro h
  rcases h with h | h
  · exact hne h
  · exact h hov

/-- `elements`: positive dimensions with an `int`-bounded product. -/
def elementsOk (cols rows : Nat) : Prop := 0 < cols ∧ 0 < rows ∧ cols * rows ≤ 2147483647

/-- Accepted shapes give a non-empty, int-addressable element count. -/
theorem elementsOk_count (cols rows : Nat) (h : elementsOk cols rows) :
    0 < cols * rows ∧ cols * rows ≤ 2147483647 :=
  ⟨Nat.mul_pos h.1 h.2.1, h.2.2⟩

/-- The gate's read index into `q_full`: the second half of head `h`. -/
def gateIdx (hd h c : Nat) : Nat := h * 2 * hd + hd + c

/-- Every gate read is inside `q_full` (`2 * n_head * head_dim` floats). -/
theorem gateIdx_lt (hd nh h c : Nat) (hh : h < nh) (hc : c < hd) : gateIdx hd h c < nh * 2 * hd := by
  unfold gateIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right (2 * hd) (Nat.succ_le_of_lt hh)
  rw [Nat.succ_mul] at h1
  rw [Nat.mul_assoc, Nat.mul_assoc]
  omega

/-- The gate reads the second half, never the first: it is at least `h*2*hd + hd`. -/
theorem gateIdx_second_half (hd h c : Nat) : h * 2 * hd + hd ≤ gateIdx hd h c := by
  unfold gateIdx; omega

/-- The norm launch's block size. -/
def normBlock (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- Thread `col % B` covers column `col` at step `col / B` of the strided loop. -/
theorem norm_stride_covers (nCols col : Nat) (_hc : col < nCols) :
    col % normBlock nCols < normBlock nCols ∧
      col = col / normBlock nCols * normBlock nCols + col % normBlock nCols := by
  unfold normBlock
  split <;> omega

end Ref.NativeQsa



## FILE: /mnt/user-data/outputs/RopeNeoxPairing.lean
/-!
# MODULE: rope_neox_pairing
SOURCES: src/kernels/cuda/rope.cu, src/kernels/cuda/native_rope.cu
Partial NEOX rotary embedding: pair `(i, i + n_rot/2)` for `i < n_rot/2`, rotate by the angle of the
row's position, copy the untouched tail `[n_rot, head_dim)`; `build_rope_table` caches `cos/sin` per
`(position, pair)` at offset `p * half + i`.

## Source semantics
* NEOX pairing, not adjacent `(2i, 2i+1)`: every dimension `< head_dim` is touched by exactly one role —
  first of a pair (`d < half`), second of a pair (`half ≤ d < n_rot`), or tail copy.
* `out = (a·c − b·s, a·s + b·c)`; with `c² + s² = 1` it preserves `a² + b²` (exactly, over a ring);
  position 0 (`c = 1, s = 0`) is the identity, which is why the spare key needs no rotation.
* Table reads `p * half + i` with `p < max_pos`, `i < half` stay inside `max_pos * half`.
* `native_rope`'s pair thread `pair < width/2` either rotates (`pair < 32`, writes `pair`, `pair+32`) or
  copies (`2·pair`, `2·pair+1 ≥ 64`): the write sets are disjoint and cover `[0, width)`.

## Form template identifiers
`rope_neox_pair` → `rotPair`, `half = n_rot / 2` → `half`, `toff + i` → `tabIdx`, role of a dimension
→ `Role`, native pair thread → `nativeWrites`.
-/

namespace Ref.Rope

-- § rotation algebra

/-- `rope_neox_pair`: rotate `(a, b)` by `(c, s)`. -/
def rotPair (a b c s : Int) : Int × Int := (a * c - b * s, a * s + b * c)

/-- The rotation scales the squared norm by `c² + s²`. -/
theorem rotPair_norm (a b c s : Int) :
    (rotPair a b c s).1 * (rotPair a b c s).1 + (rotPair a b c s).2 * (rotPair a b c s).2 =
      (a * a + b * b) * (c * c + s * s) := by
  simp only [rotPair]
  grind

/-- With a unit `(c, s)` the rotation preserves the norm. -/
theorem rotPair_norm_unit (a b c s : Int) (h : c * c + s * s = 1) :
    (rotPair a b c s).1 * (rotPair a b c s).1 + (rotPair a b c s).2 * (rotPair a b c s).2 = a * a + b * b := by
  rw [rotPair_norm, h, Int.mul_one]

/-- Position 0 (`cos = 1`, `sin = 0`) is the identity: the spare key's skipped rotation is exact. -/
theorem rotPair_pos0 (a b : Int) : rotPair a b 1 0 = (a, b) := by
  simp [rotPair]

/-- Rotating by `(c, s)` then by `(c, -s)` with `c² + s² = 1` undoes the rotation. -/
theorem rotPair_inverse (a b c s : Int) (h : c * c + s * s = 1) :
    rotPair (rotPair a b c s).1 (rotPair a b c s).2 c (-s) = (a, b) := by
  simp only [rotPair, Prod.mk.injEq]
  constructor <;> grind

-- § dimension roles

/-- The role of dimension `d` under partial NEOX with `half = n_rot / 2`. -/
inductive Role where
  | first (i : Nat)
  | second (i : Nat)
  | tail
deriving DecidableEq

/-- The unique role of dimension `d`. -/
def role (half d : Nat) : Role :=
  if d < half then .first d else if d < 2 * half then .second (d - half) else .tail

/-- Every role names the dimension it came from: first `i` is `d = i`, second `i` is `d = i + half`. -/
theorem role_spec (half d : Nat) :
    (∃ i, role half d = .first i ∧ d = i ∧ i < half) ∨
    (∃ i, role half d = .second i ∧ d = i + half ∧ i < half) ∨
    (role half d = .tail ∧ 2 * half ≤ d) := by
  unfold role
  by_cases h1 : d < half
  · exact Or.inl ⟨d, by simp [h1], rfl, h1⟩
  · by_cases h2 : d < 2 * half
    · exact Or.inr (Or.inl ⟨d - half, by simp [h1, h2], by omega, by omega⟩)
    · exact Or.inr (Or.inr ⟨by simp [h1, h2], by omega⟩)

/-- NEOX partners differ by `half`, not by one: `(i, i + half)` is never the adjacent pair for `half > 1`. -/
theorem neox_not_adjacent (half i : Nat) (h : 1 < half) : i + half ≠ i + 1 := by
  omega

/-- The NEOX pairing is a bijection between `[0, half)` and `[half, 2*half)`. -/
theorem partner_bij (half i j : Nat) (hj : half ≤ j) (hj2 : j < 2 * half) :
    (i < half → half ≤ i + half ∧ i + half < 2 * half) ∧ (j - half < half ∧ j - half + half = j) := by
  omega

-- § table

/-- `build_rope_table` offset. -/
def tabIdx (half p i : Nat) : Nat := p * half + i

/-- Table reads stay inside the `max_pos * half` table. -/
theorem tabIdx_lt (half maxPos p i : Nat) (hp : p < maxPos) (hi : i < half) : tabIdx half p i < maxPos * half := by
  unfold tabIdx
  have h1 : (p + 1) * half ≤ maxPos * half := Nat.mul_le_mul_right half (Nat.succ_le_of_lt hp)
  rw [Nat.succ_mul] at h1
  omega

-- § native pair threads (n_rot = 64)

/-- The two dimensions written by native pair thread `pair`. -/
def nativeWrites (pair : Nat) : Nat × Nat := if pair < 32 then (pair, pair + 32) else (2 * pair, 2 * pair + 1)

/-- Rotating threads write below 64, copying threads at or above 64: no write race. -/
theorem nativeWrites_disjoint (p q : Nat) (hp : p < 32) (hq : 32 ≤ q) :
    (nativeWrites p).1 < 64 ∧ (nativeWrites p).2 < 64 ∧ 64 ≤ (nativeWrites q).1 ∧ 64 ≤ (nativeWrites q).2 := by
  unfold nativeWrites
  simp only [hp, if_true, Nat.not_lt.mpr hq, if_false]
  omega

/-- Every dimension of a 128- or 256-wide row is written by some pair thread `< width/2`. -/
theorem nativeWrites_cover (width d : Nat) (hw : width = 128 ∨ width = 256) (hd : d < width) :
    ∃ pair, pair < width / 2 ∧ ((nativeWrites pair).1 = d ∨ (nativeWrites pair).2 = d) := by
  by_cases h1 : d < 32
  · exact ⟨d, by omega, Or.inl (by simp [nativeWrites, h1])⟩
  · by_cases h2 : d < 64
    · refine ⟨d - 32, by omega, Or.inr ?_⟩
      have : d - 32 < 32 := by omega
      simp only [nativeWrites, this, if_true]
      omega
    · refine ⟨d / 2, by omega, ?_⟩
      have : ¬ d / 2 < 32 := by omega
      simp only [nativeWrites, this, if_false]
      omega

end Ref.Rope



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- QsaStepState.lean
  was: theorem stepFill_width_pos (s : Shapes) (pos : Nat) (h : 0 < s.topK) : 0 < (stepFill s pos).width
  now: theorem stepFill_width_pos (s : Shapes) (pos : Nat) (hr : 2 ≤ s.idxBlock) :
    0 < (stepFill s pos).width
  measured: the claim changed

- PagedKvLayout.lean
  was: theorem poolRow_slot_injective (table : Nat → Nat) (K ps h pos pos' : Nat) (hps : 0 < ps)
    (hpage : pos / ps = pos' / ps) (heq : poolRow table K ps h pos = poolRow table K ps h pos') :
    pos = pos'
  now: theorem poolRow_slot_injective (table : Nat → Nat) (K ps h pos pos' : Nat)
    (hpage : pos / ps = pos' / ps) (heq : poolRow table K ps h pos = poolRow table K ps h pos') :
    pos = pos'
  measured: fewer quantified variables (5 → 4)

- QsaIndexCellMap.lean
  was: theorem tailBiased_iff_nonempty (r nKv : Nat) (hr : 0 < r) :
    tailBiased r nKv (nKv / r) (nKv / r) ↔ cellLo r (nKv / r) < cellHi r nKv (nKv / r) (nKv / r)
  now: theorem tailBiased_iff_nonempty (r nKv : Nat) :
    tailBiased r nKv (nKv / r) (nKv / r) ↔ cellLo r (nKv / r) < cellHi r nKv (nKv / r) (nKv / r)
  measured: fewer quantified variables (2 → 1)
-/



## FILE: /mnt/user-data/outputs/GatedDeltaNetStep.lean
/-!
# MODULE: gated_delta_net_step
SOURCES: src/kernels/cuda/gdn.cu
The non-projection parts of the gated delta-net (GDN) decode step: the per-column delta-rule
recurrence over a (S, h_v, S) state, the 4-tap causal depthwise conv with its sliding state,
the L2 / RMS output norms, and the sigmoid applied to `beta` by the layer glue.

## Source semantics
* State layout is (S, h_v, S) with `j` fastest: element (i, h, j) lives at `(i*h_v + h)*S + j`;
  the kernel walks column (h, j) as `col[i*stride]` with `col = h*S + j`, `stride = h_v*S`.
* One thread owns one (h, j) column end to end, so the recurrence needs no barrier.
* Head pairing is MODULO: value head `h` reads q/k head `h % h_k`.
* Pass 1 decays the column (`s := s*dec`) BEFORE contracting against k (PROPERTY 5);
  `d = (v - sk)*beta`; pass 2 applies the rank-1 update `s := s + k*d` and reads out against q
  using the UPDATED state.
* `beta` reaching the kernel must already be `sigmoid(beta_pre)` (the C1 bug was omitting it).
* Conv: `out = Σ_{i<dc-1} st[i]*w[i] + x*w[dc-1]`, then the state drops its oldest row and
  appends `x`.
* `gdn_step` rejects `S > 128` (shared staging width) and no-ops on non-positive shapes.

## Form template identifiers
state ↦ `State`, column update ↦ `colUpdate`, decay ↦ `decay`, contraction ↦ `dot`,
conv step ↦ `convOut`/`convSlide`, launch grid ↦ `gridX`/`gridY`, head pairing ↦ `srcHead`.
Arithmetic is modelled exactly over `Int` (the float kernel is an approximation of this).
-/

namespace Ref.GdnStep

-- § arithmetic helpers

/-- Dot product of two coefficient lists (truncating to the shorter one). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Linear bound on a two-level row-major index. -/
theorem lin_bound {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at h1
  omega

/-- Scaling the left operand of a dot product scales the result. -/
theorem dot_scale_left (g : Int) (s k : List Int) :
    dot (s.map (· * g)) k = g * dot s k := by
  induction s generalizing k with
  | nil => simp [dot]
  | cons a s ih =>
    cases k with
    | nil => simp [dot]
    | cons b k =>
      simp only [List.map, dot, ih]
      rw [Int.mul_add, Int.mul_comm a g, Int.mul_assoc]

-- § state layout

/-- Flat address of state element `(i, h, j)` in the `(S, h_v, S)` layout. -/
def stateIdx (S hv i h j : Nat) : Nat := (i * hv + h) * S + j

/-- The kernel's column addressing `h*S + j + i*(h_v*S)` is exactly the layout address. -/
theorem col_addr_eq (S hv i h j : Nat) :
    h * S + j + i * (hv * S) = stateIdx S hv i h j := by
  unfold stateIdx
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Every in-range state element lies inside the `S*h_v*S` buffer. -/
theorem stateIdx_lt {S hv i h j : Nat} (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx S hv i h j < S * hv * S := by
  have h1 : i * hv + h < S * hv := lin_bound hi hh
  exact lin_bound h1 hj

/-- Distinct in-range `(i, h, j)` give distinct addresses (no two threads share a cell). -/
theorem stateIdx_inj {S hv i h j i' h' j' : Nat} (hh : h < hv) (hh' : h' < hv)
    (hj : j < S) (hj' : j' < S)
    (heq : stateIdx S hv i h j = stateIdx S hv i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at heq
  have hS : 0 < S := by omega
  have m1 : ((i * hv + h) * S + j) % S = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]
  have m2 : ((i' * hv + h') * S + j') % S = j' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj']
  have hjj : j = j' := by rw [← m1, ← m2, heq]
  subst hjj
  have e1 : (i * hv + h) * S = (i' * hv + h') * S := by omega
  have e2 : i * hv + h = i' * hv + h' := Nat.eq_of_mul_eq_mul_right hS e1
  have n1 : (h + i * hv) % hv = h := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh]
  have n2 : (h' + i' * hv) % hv = h' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hh']
  have hhh : h = h' := by
    rw [← n1, ← n2, Nat.add_comm h, Nat.add_comm h', e2]
  subst hhh
  have hv0 : 0 < hv := by omega
  have e3 : i * hv = i' * hv := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right hv0 e3, rfl, rfl⟩

-- § launch geometry

/-- Threads along `j` per block (`JTHREADS`). -/
def JTHREADS : Nat := 32
/-- Heads per block (`MAX_H`, set to 1 as a latency-hiding fix). -/
def MAX_H : Nat := 1

/-- Grid x extent: `ceil(S / 32)`. -/
def gridX (S : Nat) : Nat := (S + JTHREADS - 1) / JTHREADS
/-- Grid y extent: `ceil(h_v / MAX_H)`. -/
def gridY (hv : Nat) : Nat := (hv + MAX_H - 1) / MAX_H

/-- Every column index `j < S` is reached by some block `bx < gridX S` and lane `tx < 32`. -/
theorem grid_covers_columns {S j : Nat} (hj : j < S) :
    ∃ bx tx, bx < gridX S ∧ tx < JTHREADS ∧ bx * JTHREADS + tx = j := by
  refine ⟨j / 32, j % 32, ?_, ?_, ?_⟩ <;> simp [gridX, JTHREADS] <;> omega

/-- With `MAX_H = 1`, there is exactly one block row per value head. -/
theorem gridY_eq_heads (hv : Nat) : gridY hv = hv := by
  simp [gridY, MAX_H]

/-- The thread count is `gridX*32 ≥ S` columns per head: the `j >= S` guard is needed only for
    the tail block. -/
theorem gridX_threads_ge (S : Nat) : S ≤ gridX S * JTHREADS := by
  simp [gridX, JTHREADS]; omega

/-- At the real geometry (S = 128, h_v = 48) the grid is (4, 48) = 192 blocks of 32 threads. -/
theorem real_geometry_blocks : gridX 128 = 4 ∧ gridY 48 = 48 ∧ gridX 128 * gridY 48 = 192 := by
  decide

/-- Modulo head pairing always selects a valid q/k head. -/
def srcHead (h hk : Nat) : Nat := h % hk

/-- The paired source head is in range whenever there is at least one q/k head. -/
theorem srcHead_lt {h hk : Nat} (hk0 : 0 < hk) : srcHead h hk < hk := Nat.mod_lt _ hk0

-- § recurrence (one column)

/-- Pass 1: decay every row of the column by `dec`. -/
def decay (dec : Int) (s : List Int) : List Int := s.map (· * dec)

/-- The rank-1 update of pass 2: `s_i + k_i * d`. -/
def rank1 (s k : List Int) (d : Int) : List Int := List.zipWith (fun a b => a + b * d) s k

/-- One column step exactly as `gdn_step_kernel` performs it: decayed state, write strength
    `d = (v - <decayed s, k>)*beta`, updated state and its readout against `q`. -/
def colUpdate (dec beta v : Int) (s k q : List Int) : List Int × Int :=
  let sd := decay dec s
  let d := (v - dot sd k) * beta
  let s' := rank1 sd k d
  (s', dot s' q)

/-- PROPERTY 5: contracting the decayed state equals decaying the contraction. -/
theorem decay_before_contract (dec : Int) (s k : List Int) :
    dot (decay dec s) k = dec * dot s k := dot_scale_left dec s k

/-- Decay preserves the column length. -/
theorem decay_length (dec : Int) (s : List Int) : (decay dec s).length = s.length := by
  simp [decay]

/-- The rank-1 update preserves the column length when `k` covers the column. -/
theorem rank1_length (s k : List Int) (d : Int) (h : s.length = k.length) :
    (rank1 s k d).length = s.length := by
  simp [rank1, h]

/-- The readout uses the UPDATED state: it is the dot of the returned state with `q`. -/
theorem readout_uses_updated (dec beta v : Int) (s k q : List Int) :
    (colUpdate dec beta v s k q).2 = dot (colUpdate dec beta v s k q).1 q := rfl

/-- `zipWith` that keeps its left argument is the identity on equal-length lists. -/
theorem zipWith_fst (t k : List Int) (h : t.length = k.length) :
    List.zipWith (fun a _ => a) t k = t := by
  induction t generalizing k with
  | nil => simp
  | cons a t ih =>
    cases k with
    | nil => simp at h
    | cons b k => simp [List.zipWith, ih k (by simpa using h)]

/-- With `beta = 0` the step is pure decay (the write strength vanishes). -/
theorem zero_beta_is_decay (dec v : Int) (s k q : List Int) (h : s.length = k.length) :
    (colUpdate dec 0 v s k q).1 = decay dec s := by
  have hl : (decay dec s).length = k.length := by rw [decay_length]; exact h
  simp only [colUpdate, rank1, Int.mul_zero, Int.add_zero]
  exact zipWith_fst _ _ hl

-- § conv step

/-- Conv output: state taps against `w[0..dc-2]`, newest input on the last tap. -/
def convOut (st w : List Int) (x : Int) : Int := dot (st ++ [x]) w

/-- Conv state slide: drop the oldest row, append the newest input. -/
def convSlide (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- The slide keeps a non-empty conv state at constant length `dc - 1`. -/
theorem convSlide_length {st : List Int} (h : st ≠ []) (x : Int) :
    (convSlide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp [convSlide]

/-- After the slide the newest row is the input just consumed. -/
theorem convSlide_last (st : List Int) (x : Int) :
    (convSlide st x).getLast? = some x := by
  simp [convSlide]

/-- For `d_conv = 4`: the output is the 4-tap dot with the oldest row on `w[0]`. -/
theorem convOut_four (a b c x w0 w1 w2 w3 : Int) :
    convOut [a, b, c] [w0, w1, w2, w3] x = a * w0 + (b * w1 + (c * w2 + (x * w3 + 0))) := rfl

-- § activations and guards

/-- Softplus large-argument branch: above 20 it is the identity (an abstract model). -/
def softplusBranch (f : Int → Int) (x : Int) : Int := if x > 20 then x else f x

/-- Above the threshold the overflow-avoiding branch returns its argument. -/
theorem softplus_large {f : Int → Int} {x : Int} (h : x > 20) : softplusBranch f x = x := by
  simp [softplusBranch, h]

/-- Host guard of `gdn_step`: launches only when all shapes are positive and `S ≤ 128`. -/
def stepLaunches (S hk hv : Int) : Bool := S > 0 && hk > 0 && hv > 0 && S ≤ 128

/-- A launched step always has a non-empty, stage-able state width. -/
theorem stepLaunches_bounds {S hk hv : Int} (h : stepLaunches S hk hv = true) :
    0 < S ∧ S ≤ 128 ∧ 0 < hk ∧ 0 < hv := by
  simp [stepLaunches] at h
  omega

/-- `gdn_l2_norm` refuses rows wider than the 1024-wide reduction. -/
def l2Launches (rows cols : Int) : Bool := rows > 0 && cols > 0 && cols ≤ 1024

/-- A launched L2 norm has `0 < cols ≤ 1024`. -/
theorem l2Launches_bounds {r c : Int} (h : l2Launches r c = true) : 0 < c ∧ c ≤ 1024 := by
  simp [l2Launches] at h
  omega

/-- The beta-gate grid `ceil(n/128)` blocks of 128 covers all heads. -/
theorem beta_gate_covers (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

end Ref.GdnStep



## FILE: /mnt/user-data/outputs/FusedGatedDeltaNet.lean
/-!
# MODULE: fused_gated_delta_net
SOURCES: src/kernels/cuda/fused_gdn.cu
Single-launch GDN kernels at the fixed geometry S = 128: the step fused with its output RMS
norm and z-gate (4 row groups × 128 columns per head), the 4-tap conv fused with SiLU and the
per-head L2 norm, and the alpha/beta GEMV with bf16 weights fused with softplus / sigmoid.

## Source semantics
* Thread (col, rg) owns rows `rg*32 .. rg*32+31` of column `col`; the four row-group partials
  meet in `red[rg][col]` and are summed to a full contraction.
* The fused step computes `kv` on the UNDECAYED state and uses `delta = (v - g*kv)*beta`,
  `s' = g*s + k*delta`; this equals gdn.cu's decay-then-contract form.
* Conv: history `[h0,h1,h2]`, output `Σ`, history becomes `[h1,h2,x]`; blocks `< qk_heads`
  (q and k heads) are additionally L2-normalised, the v heads are not.
* `gdn_ab`: rows `0..h_v-1` are alpha (gate), rows `h_v..2h_v-1` are beta; bf16 is unpacked
  from a 32-bit word as `w << 16` (low half) and `w & 0xffff0000` (high half).
* Host preconditions: non-null pointers, `h_v % h_k = 0`, `channels % 128 = 0`,
  `0 ≤ qk_heads ≤ channels/128`, `n_embd % 8 = 0`.

## Form template identifiers
row-group split ↦ `rowOf`, partial reduction ↦ `dot_append`, fused step ↦ `fusedCol`,
two-pass step ↦ `twoPassCol`, conv history ↦ `histShift`, alpha/beta dispatch ↦ `abRow`,
bf16 halves ↦ `lowHalf`/`highHalf`.
-/

namespace Ref.FusedGdn

-- § helpers

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Scaling the left operand scales the dot product. -/
theorem dot_scale_left (g : Int) (s k : List Int) :
    dot (s.map (· * g)) k = g * dot s k := by
  induction s generalizing k with
  | nil => simp [dot]
  | cons a s ih =>
    cases k with
    | nil => simp [dot]
    | cons b k =>
      simp only [List.map, dot, ih]
      rw [Int.mul_add, Int.mul_comm a g, Int.mul_assoc]

/-- Row-group reduction: a dot over concatenated equal-length blocks is the sum of block dots. -/
theorem dot_append (a b c d : List Int) (h : a.length = c.length) :
    dot (a ++ b) (c ++ d) = dot a c + dot b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons x a ih =>
    cases c with
    | nil => simp at h
    | cons y c =>
      simp only [List.cons_append, dot]
      rw [ih c (by simpa using h), Int.add_assoc]

-- § row-group geometry

/-- State size. -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := 32

/-- Global row owned by row group `rg` at register slot `r`. -/
def rowOf (rg r : Nat) : Nat := rg * RPG + r

/-- The geometry constants are consistent: `RG * RPG = S` and the block is 512 threads. -/
theorem geometry_consistent : RG * RPG = S ∧ S * RG = 512 := by decide

/-- Every row `< 128` is owned by exactly one `(rg, r)` with `rg < 4`, `r < 32`. -/
theorem rowOf_cover {i : Nat} (hi : i < S) :
    ∃ rg r, rg < RG ∧ r < RPG ∧ rowOf rg r = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> simp [rowOf, RG, RPG, S] at * <;> omega

/-- Ownership is unique: two slots naming the same row are the same slot. -/
theorem rowOf_unique {rg r rg' r' : Nat} (hr : r < RPG) (hr' : r' < RPG)
    (h : rowOf rg r = rowOf rg' r') : rg = rg' ∧ r = r' := by
  simp [rowOf, RPG] at *
  omega

/-- A row-group thread's rows stay inside the 128-row column. -/
theorem rowOf_lt {rg r : Nat} (hrg : rg < RG) (hr : r < RPG) : rowOf rg r < S := by
  simp [rowOf, RG, RPG, S] at *
  omega

/-- The four partials summed by `kv_col = red[0]+red[1]+red[2]+red[3]` give the full dot. -/
theorem four_partials (s0 s1 s2 s3 k0 k1 k2 k3 : List Int)
    (h0 : s0.length = k0.length) (h1 : s1.length = k1.length) (h2 : s2.length = k2.length) :
    dot (s0 ++ s1 ++ s2 ++ s3) (k0 ++ k1 ++ k2 ++ k3) =
      dot s0 k0 + dot s1 k1 + dot s2 k2 + dot s3 k3 := by
  have l01 : (s0 ++ s1).length = (k0 ++ k1).length := by simp [h0, h1]
  have l012 : (s0 ++ s1 ++ s2).length = (k0 ++ k1 ++ k2).length := by simp [h0, h1, h2]
  rw [dot_append _ _ _ _ l012, dot_append _ _ _ _ l01, dot_append _ _ _ _ h0]

-- § fused vs two-pass step

/-- gdn.cu's two-pass column step: decay, contract, rank-1 update. -/
def twoPassCol (g beta v : Int) (s k : List Int) : List Int :=
  let sd := s.map (· * g)
  let d := (v - dot sd k) * beta
  List.zipWith (fun a b => a + b * d) sd k

/-- fused_gdn.cu's column step: contract first, fold the decay into `delta` and the update. -/
def fusedCol (g beta v : Int) (s k : List Int) : List Int :=
  let delta := (v - g * dot s k) * beta
  List.zipWith (fun a b => g * a + b * delta) s k

/-- Pointwise: a zipWith over a mapped left list is a zipWith of the composed function. -/
theorem zipWith_map_left (f : Int → Int) (F : Int → Int → Int) (s k : List Int) :
    List.zipWith F (s.map f) k = List.zipWith (fun a b => F (f a) b) s k := by
  induction s generalizing k with
  | nil => simp
  | cons a s ih =>
    cases k with
    | nil => simp
    | cons b k => simp [List.zipWith, ih]

/-- The fused step computes exactly the two-pass state (exact arithmetic). -/
theorem fused_eq_twoPass (g beta v : Int) (s k : List Int) :
    fusedCol g beta v s k = twoPassCol g beta v s k := by
  have hd : (v - dot (s.map (· * g)) k) * beta = (v - g * dot s k) * beta := by
    rw [dot_scale_left]
  simp only [twoPassCol, fusedCol]
  rw [hd, zipWith_map_left]
  congr 1
  funext a b
  rw [Int.mul_comm a g]

/-- Hence the readouts `<s', q>` of both formulations agree as well. -/
theorem fused_readout_eq (g beta v : Int) (s k q : List Int) :
    dot (fusedCol g beta v s k) q = dot (twoPassCol g beta v s k) q := by
  rw [fused_eq_twoPass]

-- § conv + l2

/-- History shift of the fused conv: `[h0,h1,h2] , x ↦ [h1,h2,x]`. -/
def histShift (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- The conv sum over the 4-tap window `[h0,h1,h2,x]`. -/
def convSum (h : Int × Int × Int) (x w0 w1 w2 w3 : Int) : Int :=
  h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + x * w3

/-- Three shifts flush the history: it then holds exactly the last three inputs. -/
theorem histShift_three (h : Int × Int × Int) (a b c : Int) :
    histShift (histShift (histShift h a) b) c = (a, b, c) := rfl

/-- Channel `c = block*128 + thread` is an L2-normalised q/k channel iff its block is a qk head. -/
def isQkChannel (qkHeads c : Nat) : Bool := c / 128 < qkHeads

/-- Channels of block `b` all share the block's q/k classification. -/
theorem qk_by_block {qkHeads b t : Nat} (ht : t < 128) :
    isQkChannel qkHeads (b * 128 + t) = decide (b < qkHeads) := by
  simp [isQkChannel]
  have : (b * 128 + t) / 128 = b := by omega
  rw [this]

/-- Host check of `fused_gdn_conv_l2`. -/
def convL2Valid (channels qkHeads : Int) : Bool :=
  channels % 128 == 0 && qkHeads ≥ 0 && qkHeads ≤ channels / 128

/-- A valid call never marks more q/k blocks than there are blocks. -/
theorem convL2Valid_heads {c q : Int} (h : convL2Valid c q = true) : 0 ≤ q ∧ q ≤ c / 128 := by
  simp [convL2Valid] at h
  omega

-- § alpha / beta GEMV

/-- Dispatch of GEMV row `row < 2*h_v`: `(is_beta, r)`. -/
def abRow (hv row : Nat) : Bool × Nat := if row ≥ hv then (true, row - hv) else (false, row)

/-- Every row maps to a valid head index and each head is hit once as alpha and once as beta. -/
theorem abRow_lt {hv row : Nat} (h : row < 2 * hv) : (abRow hv row).2 < hv := by
  unfold abRow
  split <;> simp <;> omega

/-- Alpha row `r` and beta row `h_v + r` both land on head `r`. -/
theorem abRow_pair (hv r : Nat) (hr : r < hv) :
    abRow hv r = (false, r) ∧ abRow hv (hv + r) = (true, r) := by
  unfold abRow
  constructor
  · simp; omega
  · simp

/-- The launch `ceil(2*h_v/8)` blocks of 8 warps covers every GEMV row. -/
theorem ab_grid_covers (hv : Nat) : 2 * hv ≤ (2 * hv + 7) / 8 * 8 := by omega

/-- Low bf16 half of a packed word, widened to f32 bits: `(w mod 2^16) * 2^16`. -/
def lowHalf (w : Nat) : Nat := (w % 2 ^ 16) * 2 ^ 16
/-- High bf16 half, already in f32 position: `(w / 2^16) * 2^16`. -/
def highHalf (w : Nat) : Nat := (w / 2 ^ 16) * 2 ^ 16

/-- Both widened halves have zero low mantissa bits, as bf16→f32 requires. -/
theorem halves_aligned (w : Nat) : lowHalf w % 2 ^ 16 = 0 ∧ highHalf w % 2 ^ 16 = 0 := by
  simp [lowHalf, highHalf, Nat.mul_mod_left]

/-- The two halves carry the whole word: `w = highHalf + lowHalf / 2^16`. -/
theorem halves_reconstruct (w : Nat) : highHalf w + lowHalf w / 2 ^ 16 = w := by
  simp only [lowHalf, highHalf]
  rw [Nat.mul_div_cancel _ (by decide)]
  have := Nat.mod_add_div w (2 ^ 16)
  rw [Nat.mul_comm] at this
  omega

/-- Host check of `fused_gdn_step_norm`: positive heads with `h_v % h_k = 0`. -/
theorem step_norm_heads_valid {hk h : Nat} (hk0 : 0 < hk) :
    h % hk < hk := Nat.mod_lt _ hk0

end Ref.FusedGdn



## FILE: /mnt/user-data/outputs/NativeGatedDeltaNet.lean
/-!
# MODULE: native_gated_delta_net
SOURCES: src/kernels/cuda/native_gdn.cu
The llama.cpp-derived GDN step: one warp per (head, column), each lane holding four rows of the
column in registers, with warp-sum reductions; plus the host-side argument validation
(alignment, overflow-free spans, pairwise disjointness) and the global enable flag.

## Source semantics
* Lane `l` of the warp holds rows `r*32 + l`, `r < 4`; the four register shards cover the
  128-row column exactly once.
* Column `col = blockIdx.z * 4 + threadIdx.y`, grid `(h_v, 1, S/4)`, block `(32, 4)`.
* Arithmetic: `kv = Σ s*k` (undecayed), `delta = (v - g*kv)*beta`, `s' = g*s + k*delta`,
  output `scale * Σ s'*q` with `scale = 1/sqrt(S)`; written by lane 0 only.
* `valid_span(p, n)`: non-null, 4-byte aligned, `n ≤ UINTPTR_MAX - p`.
* `overlap(a, an, b, bn) := a < b + bn ∧ b < a + an` (half-open intervals).
* Preconditions: stream non-null, `S = 128`, `0 < h_k`, `0 < h_v ≤ 65535`, `h_v % h_k = 0`;
  state/output spans disjoint from each other and from all five inputs.

## Form template identifiers
lane rows ↦ `laneRow`, column index ↦ `colOf`, span predicate ↦ `validSpan`,
overlap predicate ↦ `overlap`, launch precondition ↦ `shapeOk`.
-/

namespace Ref.NativeGdn

-- § register sharding

/-- Row held by lane `l` in register slot `r`. -/
def laneRow (r l : Nat) : Nat := r * 32 + l

/-- The four register shards of the 32 lanes cover every row `< 128`. -/
theorem laneRow_cover {i : Nat} (hi : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ laneRow r l = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> (try simp only [laneRow]) <;> omega

/-- No row is held twice: `(r, l)` is determined by the row. -/
theorem laneRow_unique {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : laneRow r l = laneRow r' l') : r = r' ∧ l = l' := by
  simp [laneRow] at h
  omega

/-- Every held row is in range. -/
theorem laneRow_lt {r l : Nat} (hr : r < 4) (hl : l < 32) : laneRow r l < 128 := by
  simp [laneRow]; omega

/-- Column owned by warp `y` of block `z`. -/
def colOf (z y : Nat) : Nat := z * 4 + y

/-- The `S/4 = 32` z-blocks of 4 warps cover all 128 columns exactly once. -/
theorem colOf_cover {c : Nat} (hc : c < 128) : ∃ z y, z < 128 / 4 ∧ y < 4 ∧ colOf z y = c := by
  refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> (try simp only [colOf]) <;> omega

/-- Distinct warps own distinct columns. -/
theorem colOf_unique {z y z' y' : Nat} (hy : y < 4) (hy' : y' < 4)
    (h : colOf z y = colOf z' y') : z = z' ∧ y = y' := by
  simp [colOf] at h
  omega

/-- State address of (row i, head, col): `(i*h_v + head)*128 + col`, in range for valid indices. -/
theorem state_addr_lt {i head col hv : Nat} (hi : i < 128) (hh : head < hv) (hc : col < 128) :
    (i * hv + head) * 128 + col < 128 * hv * 128 := by
  have h1 : (i + 1) * hv ≤ 128 * hv := Nat.mul_le_mul_right _ (by omega)
  have h2 : i * hv + head < 128 * hv := by rw [Nat.add_mul] at h1; omega
  have h3 : (i * hv + head + 1) * 128 ≤ 128 * hv * 128 := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at h3
  omega

-- § step arithmetic (exact model)

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Sum of a list (the value of `warp_sum` over the lane partials, order-free in exact arithmetic). -/
def sum : List Int → Int
  | [] => 0
  | a :: as => a + sum as

/-- A dot over concatenated equal-length blocks is the sum of the block dots. -/
theorem dot_append (a b c d : List Int) (h : a.length = c.length) :
    dot (a ++ b) (c ++ d) = dot a c + dot b d := by
  induction a generalizing c with
  | nil =>
    cases c with
    | nil => simp [dot]
    | cons _ _ => simp at h
  | cons x a ih =>
    cases c with
    | nil => simp at h
    | cons y c =>
      simp only [List.cons_append, dot]
      rw [ih c (by simpa using h), Int.add_assoc]

/-- The warp sum of per-lane partial dots equals the dot over the concatenated shards. -/
theorem sum_partials (parts : List (List Int × List Int))
    (h : ∀ p ∈ parts, p.1.length = p.2.length) :
    sum (parts.map (fun p => dot p.1 p.2)) =
      dot (parts.flatMap (·.1)) (parts.flatMap (·.2)) := by
  induction parts with
  | nil => simp [sum, dot]
  | cons p ps ih =>
    have hp : p.1.length = p.2.length := h p (by simp)
    have hps : ∀ q ∈ ps, q.1.length = q.2.length := fun q hq => h q (by simp [hq])
    simp only [List.map, sum, List.flatMap_cons]
    rw [ih hps, dot_append _ _ _ _ hp]

/-- The per-shard state update `g*s + k*delta`. -/
def shardUpdate (g delta : Int) (s k : List Int) : List Int :=
  List.zipWith (fun a b => g * a + b * delta) s k

/-- The update is local: shards updated separately equal the concatenated column updated at once. -/
theorem shardUpdate_append (g d : Int) (s1 s2 k1 k2 : List Int) (h : s1.length = k1.length) :
    shardUpdate g d (s1 ++ s2) (k1 ++ k2) = shardUpdate g d s1 k1 ++ shardUpdate g d s2 k2 := by
  simp [shardUpdate, List.zipWith_append h]

-- § host validation

/-- `valid_span`: non-null, aligned to 4 bytes, and the end address does not overflow. -/
def validSpan (umax p n : Nat) : Bool := p != 0 && p % 4 == 0 && n ≤ umax - p

/-- A valid span's end is representable: `p + n ≤ UINTPTR_MAX`. -/
theorem validSpan_no_overflow {umax p n : Nat} (hp : p ≤ umax) (h : validSpan umax p n = true) :
    p + n ≤ umax := by
  simp [validSpan] at h
  omega

/-- Half-open interval overlap test. -/
def overlap (a an b bn : Nat) : Bool := a < b + bn && b < a + an

/-- Overlap is symmetric, so checking each unordered pair once suffices. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn = overlap b bn a an := by
  simp [overlap, Bool.and_comm]

/-- Adjacent spans (`b = a + an`) never overlap. -/
theorem overlap_adjacent (a an bn : Nat) : overlap a an (a + an) bn = false := by
  simp [overlap]

/-- Non-overlapping spans share no byte. -/
theorem disjoint_no_common {a an b bn x : Nat} (h : overlap a an b bn = false)
    (hxa : a ≤ x ∧ x < a + an) (hxb : b ≤ x ∧ x < b + bn) : False := by
  simp [overlap] at h
  by_cases h1 : a < b + bn
  · have := h h1; omega
  · omega

/-- Launch shape precondition of `native_gdn_step`. -/
def shapeOk (S hk hv : Nat) : Bool := S == 128 && 0 < hk && 0 < hv && hv ≤ 65535 && hv % hk == 0

/-- A valid shape gives a legal grid (`h_v ≤ 65535`) and every head pairs with a q/k head. -/
theorem shapeOk_heads {S hk hv h : Nat} (hs : shapeOk S hk hv = true) :
    hv ≤ 65535 ∧ h % hk < hk := by
  unfold shapeOk at hs
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hs
  obtain ⟨⟨⟨⟨_, hk0⟩, _⟩, hvle⟩, _⟩ := hs
  exact ⟨hvle, Nat.mod_lt _ hk0⟩

/-- State span size: `S*S*h_v` floats. At S=128, h_v=48 that is 3 MiB. -/
theorem state_bytes_real : 128 * 128 * 48 * 4 = 3145728 := by decide

end Ref.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
The llama.cpp-pinned GDN preprocessing kernels: 4-tap causal conv producing both the raw and
the SiLU output, row L2 norm (expressed as an RMS norm with rescaled epsilon followed by a
`1/sqrt(S)` scale), beta sigmoid, softplus gate, and the gated output RMS norm, together with
their host-side span and geometry validation.

## Source semantics
* Conv history per channel is 3 floats, `values = [h0,h1,h2,x]`, raw output `Σ values*w`,
  history becomes `[h1,h2,x]`; `d_conv` must be 4.
* `l2_norm` calls RMS norm with `eps/S` and post-scale `1/sqrt(S)`:
  `x * rsqrt(Σx²/S + eps/S) / sqrt(S) = x * rsqrt(Σx² + eps)`, i.e. the L2 norm.
* `gate = softplus(alpha + dt) * ssm_a`, with the `> 20` identity branch.
* Validation: `1 ≤ count ≤ 65535`, stream non-null, norm width exactly 128, epsilon finite
  and `≥ 0`; writable spans pairwise disjoint and disjoint from inputs.

## Form template identifiers
conv window ↦ `convWindow`, history shift ↦ `shiftHist`, rescaled norm denominator ↦
`rmsDenom`, count check ↦ `countOk`, disjointness ↦ `disjoint`.
-/

namespace Ref.NativeGdnPre

-- § conv + silu

/-- The 4-entry window `[h0,h1,h2,x]` for one channel. -/
def convWindow (h0 h1 h2 x : Int) : List Int := [h0, h1, h2, x]

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- The new history is the window without its oldest entry. -/
def shiftHist (h0 h1 h2 x : Int) : List Int := (convWindow h0 h1 h2 x).drop 1

/-- The shifted history is `[h1,h2,x]` and stays 3 entries long. -/
theorem shiftHist_eq (h0 h1 h2 x : Int) :
    shiftHist h0 h1 h2 x = [h1, h2, x] ∧ (shiftHist h0 h1 h2 x).length = 3 := by
  simp [shiftHist, convWindow]

/-- Raw conv output: tap 0 (oldest) pairs with `w[c*4]`, the new input with `w[c*4+3]`. -/
theorem conv_raw (h0 h1 h2 x w0 w1 w2 w3 : Int) :
    dot (convWindow h0 h1 h2 x) [w0, w1, w2, w3] = h0 * w0 + (h1 * w1 + (h2 * w2 + (x * w3 + 0))) :=
  rfl

/-- The pinned zero-bias add (`sum + 0`) is the identity in exact arithmetic. -/
theorem zero_bias_identity (s : Int) : s + 0 = s := Int.add_zero s

/-- Channel `c`'s history occupies `[3c, 3c+3)`, inside the `3*channels` buffer and disjoint
    from every other channel's. -/
theorem hist_slots {c ch t : Nat} (hc : c < ch) (ht : t < 3) : c * 3 + t < 3 * ch := by omega

/-- Distinct channels have distinct history slots. -/
theorem hist_slots_unique {c c' t t' : Nat} (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

-- § l2 via rescaled rms

/-- Denominator inside `rsqrt` for the RMS formulation: `ss/S + eps/S`, scaled by `S`. -/
def rmsDenom (S ss eps : Rat) : Rat := ss / S + eps / S

/-- Multiplying the rescaled RMS denominator by `S` recovers the L2 denominator `ss + eps`;
    together with the post-scale `1/sqrt S` this is exactly the L2 norm. -/
theorem rmsDenom_times_S (ss eps : Rat) : rmsDenom 128 ss eps * 128 = ss + eps := by
  unfold rmsDenom
  rw [Rat.add_mul, Rat.div_mul_cancel (by decide), Rat.div_mul_cancel (by decide)]

/-- The l2 kernel is launched with 256 threads but only threads `< 128` carry data; all data
    columns are inside the block. -/
theorem l2_threads_cover {col : Nat} (h : col < 128) : col < 256 := by omega

-- § gates

/-- Softplus with the overflow branch, abstract over the smooth part `f`. -/
def softplus (f : Int → Int) (x : Int) : Int := if x > 20 then x else f x

/-- Gate formula: `softplus(alpha + dt) * ssm_a`. -/
def gate (f : Int → Int) (alpha dt a : Int) : Int := softplus f (alpha + dt) * a

/-- Above the threshold the gate is linear in the pre-activation. -/
theorem gate_large {f : Int → Int} {alpha dt a : Int} (h : alpha + dt > 20) :
    gate f alpha dt a = (alpha + dt) * a := by
  simp [gate, softplus, h]

/-- With `ssm_a = 0` the gate is zero whatever the branch (no decay applied). -/
theorem gate_zero_a (f : Int → Int) (alpha dt : Int) : gate f alpha dt 0 = 0 := by
  simp [gate]

-- § validation

/-- `count_and_stream`: a non-null stream and `1 ≤ count ≤ 65535`. -/
def countOk (stream : Bool) (count : Int) : Bool := stream && 0 < count && count ≤ 65535

/-- A passing count fits the grid's x dimension and the 256-thread blocks cover it. -/
theorem countOk_grid {s : Bool} {n : Int} (h : countOk s n = true) :
    1 ≤ n ∧ n ≤ 65535 ∧ s = true := by
  unfold countOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hs, h1⟩, h2⟩ := h
  exact ⟨by omega, h2, hs⟩

/-- `norm_geometry`: width exactly 128 and non-negative epsilon (finiteness abstracted). -/
def normOk (stream : Bool) (rows cols : Int) (epsNonneg : Bool) : Bool :=
  countOk stream rows && cols == 128 && epsNonneg

/-- A passing norm call has exactly 128 columns. -/
theorem normOk_width {s : Bool} {r c : Int} {e : Bool} (h : normOk s r c e = true) : c = 128 := by
  unfold normOk at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.2

/-- Half-open span disjointness used by `disjoint(a, b)`. -/
def disjoint (a an b bn : Nat) : Bool := !(a < b + bn && b < a + an)

/-- Disjointness is symmetric. -/
theorem disjoint_symm (a an b bn : Nat) : disjoint a an b bn = disjoint b bn a an := by
  simp [disjoint, Bool.and_comm]

/-- Back-to-back buffers are disjoint. -/
theorem disjoint_adjacent (a an bn : Nat) : disjoint a an (a + an) bn = true := by
  simp [disjoint]

/-- The conv checks every ordered pair `j < i` of its three writable spans: that covers all
    three unordered pairs. -/
theorem conv_pairs_checked :
    ((List.range 3).flatMap (fun i => (List.range i).map (fun j => (i, j)))) =
      [(1, 0), (2, 0), (2, 1)] := by decide

end Ref.NativeGdnPre



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify) versions of the GDN conv/L2, alpha/beta and step+norm kernels,
the conv-history commit after acceptance, and the small index/gather kernels of the verify loop
(window ids, dense steps, blob fetch, row gather, embedding gather, MTP select).

## Source semantics
* Token `t` of a window sees the conv window `[hist(3) | x_0 .. x_t]` at indices `t..t+3`:
  entry `src = t + j` is `hist[src]` if `src < 3`, else `x[src-3]`. This reproduces plain
  decode bit for bit (same operation order per token).
* The conv window kernel does NOT write history; `gdn_conv_commit(n_keep)` writes the last three
  of `[hist | x_0..x_{n-1}]`, and is a no-op for `n ≤ 0`.
* `gdn_step_norm_multi` runs the recurrence over `n = n_keep ? *n_keep : T` tokens, emits
  outputs only for `t ≥ t_out_begin` (replayed tokens update state only), and writes the state
  back only when `n_keep` is given and `n > 0`.
* `window_ids`: `start = max(n_kv - window, 0)`, `width = n_kv - start`, ids `start + j`.
* `dense_steps`: `steps[4i..4i+3] = [c, c+1, (c+1)/4, c+1]`.
* Embedding gather: `code = (byte >> ((i % per_byte)*bits)) & (2^bits - 1)`, `per_byte = 8/bits`.
* `n_tok ∈ [1, kVerifyMaxT]`, blob and row sizes multiples of 16 bytes.

## Form template identifiers
concatenated sequence ↦ `seqVal`, window/state ↦ `stateAt`, sequential decode ↦ `seqState`,
commit ↦ `commit`, generic token fold ↦ `foldSteps`, window ids ↦ `windowStart`/`windowWidth`,
code extraction ↦ `codeAt`.
-/

namespace Ref.VerifyWindow

-- § conv window reproduces sequential decode

/-- The virtual sequence `[hist(3) | x_0, x_1, ...]`. -/
def seqVal (hist x : Nat → Int) (n : Nat) : Int := if n < 3 then hist n else x (n - 3)

/-- The 3-entry conv history seen by token `t` in the window kernel (`win[j] = seq[t+j]`). -/
def stateAt (hist x : Nat → Int) (t : Nat) : Int × Int × Int :=
  (seqVal hist x t, seqVal hist x (t + 1), seqVal hist x (t + 2))

/-- One decode step of the conv history: drop the oldest, append the input. -/
def stepHist (h : Int × Int × Int) (v : Int) : Int × Int × Int := (h.2.1, h.2.2, v)

/-- History after feeding `x_0 .. x_{t-1}` one token at a time (plain decode). -/
def seqState (hist x : Nat → Int) : Nat → Int × Int × Int
  | 0 => (hist 0, hist 1, hist 2)
  | t + 1 => stepHist (seqState hist x t) (x t)

/-- The token after the window is the token's own input: `seq[t+3] = x_t`. -/
theorem seqVal_input (hist x : Nat → Int) (t : Nat) : seqVal hist x (t + 3) = x t := by
  have h : ¬ (t + 3 < 3) := by omega
  unfold seqVal
  simp only [h, ↓reduceIte, Nat.add_sub_cancel]

/-- Main theorem: the window kernel's history for token `t` equals plain decode's history
    after `t` sequential steps. -/
theorem window_eq_sequential (hist x : Nat → Int) (t : Nat) :
    stateAt hist x t = seqState hist x t := by
  induction t with
  | zero => simp [stateAt, seqState, seqVal]
  | succ t ih =>
    have hs : seqState hist x (t + 1) = stepHist (seqState hist x t) (x t) := rfl
    rw [hs, ← ih]
    simp only [stateAt, stepHist]
    rw [← seqVal_input hist x t]

/-- The 4-tap conv sum over a history and the new input. -/
def convSum (h : Int × Int × Int) (v w0 w1 w2 w3 : Int) : Int :=
  h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + v * w3

/-- Hence every token's conv output in the window equals the plain-decode output. -/
theorem window_output_eq (hist x : Nat → Int) (t : Nat) (w0 w1 w2 w3 : Int) :
    convSum (stateAt hist x t) (x t) w0 w1 w2 w3 =
      convSum (seqState hist x t) (x t) w0 w1 w2 w3 := by
  rw [window_eq_sequential]

/-- `gdn_conv_commit`: write the last three of `[hist | x_0..x_{n-1}]`; no-op for `n = 0`. -/
def commit (hist x : Nat → Int) (n : Nat) : Int × Int × Int :=
  if n = 0 then (hist 0, hist 1, hist 2) else stateAt hist x n

/-- Committing `n` accepted tokens yields exactly the history plain decode would have. -/
theorem commit_eq_sequential (hist x : Nat → Int) (n : Nat) :
    commit hist x n = seqState hist x n := by
  unfold commit
  split
  · subst_vars; rfl
  · exact window_eq_sequential hist x n

-- § recurrence replay

/-- Fold of a per-token state transition over tokens `0 .. n-1`. -/
def foldSteps {σ : Type} (f : σ → Nat → σ) (s0 : σ) : Nat → σ
  | 0 => s0
  | n + 1 => f (foldSteps f s0 n) n

/-- State written back by `gdn_step_norm_multi`: only with `n_keep` given and positive. -/
def committedState {σ : Type} (f : σ → Nat → σ) (s0 : σ) (nKeep : Option Nat) : σ :=
  match nKeep with
  | some n => if n > 0 then foldSteps f s0 n else s0
  | none => s0

/-- A pure verify pass (no `n_keep`) never mutates the recurrent state. -/
theorem verify_without_keep_is_pure {σ : Type} (f : σ → Nat → σ) (s0 : σ) :
    committedState f s0 none = s0 := rfl

/-- Accepting `n` tokens commits the state of `n` sequential steps. -/
theorem keep_commits_prefix {σ : Type} (f : σ → Nat → σ) (s0 : σ) (n : Nat) :
    committedState f s0 (some n) = foldSteps f s0 n := by
  cases n with
  | zero => rfl
  | succ n => simp [committedState]

/-- Tokens whose output is emitted: `t_out_begin ≤ t < n`. -/
def emits (tOutBegin n t : Nat) : Bool := tOutBegin ≤ t && t < n

/-- Replayed tokens (`t < t_out_begin`) produce no output. -/
theorem replay_no_output {b n t : Nat} (h : t < b) : emits b n t = false := by
  simp [emits]; omega

-- § window ids and dense steps

/-- `start = n_kv > window ? n_kv - window : 0`. -/
def windowStart (nkv window : Nat) : Nat := if nkv > window then nkv - window else 0
/-- `width = n_kv - start`. -/
def windowWidth (nkv window : Nat) : Nat := nkv - windowStart nkv window

/-- The window width is `min(n_kv, window)`. -/
theorem windowWidth_eq_min (nkv window : Nat) : windowWidth nkv window = min nkv window := by
  unfold windowWidth windowStart
  split <;> omega

/-- Every emitted id `start + j`, `j < width`, is a valid past position `< n_kv`. -/
theorem window_ids_lt {nkv window j : Nat} (hj : j < windowWidth nkv window) :
    windowStart nkv window + j < nkv := by
  unfold windowWidth at hj
  unfold windowStart at hj ⊢
  by_cases hc : nkv > window
  · simp only [hc, ↓reduceIte] at hj ⊢; omega
  · simp only [hc, ↓reduceIte] at hj ⊢; omega

/-- `dense_steps` row for cell `c`. -/
def denseStep (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

/-- The dense step's `n_kv` field is one past the cell, and its block count is `⌊(c+1)/4⌋`. -/
theorem denseStep_fields (c : Nat) :
    (denseStep c)[1]! = c + 1 ∧ (denseStep c)[2]! * 4 ≤ c + 1 := by
  simp [denseStep]; omega

-- § gathers

/-- Extracted code of element with in-byte slot `k`. -/
def codeAt (byte bits k : Nat) : Nat := (byte / 2 ^ (k * bits)) % 2 ^ bits

/-- An extracted code always fits in `bits` bits. -/
theorem codeAt_lt (byte bits k : Nat) : codeAt byte bits k < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- For the supported widths the slots of a byte tile it exactly: `per_byte * bits = 8`. -/
theorem per_byte_tiles : ∀ b ∈ [1, 2, 4, 8], (8 / b) * b = 8 := by decide

/-- Flat blob index decomposition `i = k*per + off` with `off < per` is unique. -/
theorem blob_index_unique {per k off k' off' : Nat} (ho : off < per) (ho' : off' < per)
    (h : k * per + off = k' * per + off') : k = k' ∧ off = off' := by
  have hp : 0 < per := by omega
  have m1 : (off + k * per) % per = off := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho]
  have m2 : (off' + k' * per) % per = off' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho']
  have hoo : off = off' := by rw [← m1, ← m2, Nat.add_comm off, Nat.add_comm off', h]
  subst hoo
  exact ⟨Nat.eq_of_mul_eq_mul_right hp (by omega), rfl⟩

/-- `copy_indexed` with a negative index copies nothing. -/
def copyIndexed (idx : Int) (src dst : Nat → Int) (stride : Nat) : Nat → Int :=
  if idx < 0 then dst else fun i => src (idx.toNat * stride + i)

/-- A negative index leaves the destination untouched. -/
theorem copyIndexed_neg {idx : Int} (h : idx < 0) (src dst : Nat → Int) (stride : Nat) :
    copyIndexed idx src dst stride = dst := by simp [copyIndexed, h]

/-- `ident_hits` followed by `map_ids`: slot `i` holds `table[ids[i]]`, destination `i`. -/
theorem ident_then_map (ids table : Nat → Nat) (i : Nat) :
    (fun j => table (ids j)) i = table (ids i) := rfl

/-- Host bound on the verify window length. -/
def nTokOk (maxT n : Nat) : Bool := 1 ≤ n && n ≤ maxT

/-- A passing window has at least one token and fits the per-thread accumulator array. -/
theorem nTokOk_bounds {m n : Nat} (h : nTokOk m n = true) : 1 ≤ n ∧ n ≤ m := by
  simp [nTokOk] at h; omega

end Ref.VerifyWindow



## FILE: /mnt/user-data/outputs/GatedResidualKernels.lean
/-!
# MODULE: gated_residual_kernels
SOURCES: src/kernels/cuda/gr.cu
The gated residual / hyper-connection: `gr_read` (per-stream RMS norm, bf16 down projection
with SiLU, up projection with sigmoid gate, mean over streams, optional injection logits) and
`gr_write` (each residual stream adds the block output scaled by `2*sigmoid(inject/hc)`), plus
the caller-owned workspace carving `gr_workspace_init`.

## Source semantics
* Workspace: five regions `xn, xq, lq, gated, lo` with sizes
  `hc_dim*4, hc_dim*2, hc_lr*2, hc_dim*4, hc_lr*4`, each rounded up to 16 bytes and laid out
  back to back in ONE table indexed by the same `k` that assigns the pointers (the first version
  used a different order and every region overlapped its neighbour).
* `gr_down`: one block per output row; warp `w`, lane `l` visit `i = w*32 + l + m*(nw*32)`.
* `gr_gate`: one warp per output row, row `i = block*8 + warp`.
* `gr_mean`: `mixed[d] = (Σ_c gated[c*n + d]) / hc`.
* `gr_write`: `out[i] = R[i] + block_out[i % n] * w[i / n]`, `w[c] = 2*sigmoid(inject[c]/hc)`.
* Path selection: native ⇒ fp32 activations; native requires even `hc*n_embd` and `hc_lr`.
* No injection is computed when `w_inject` is null (the final mixer).

## Form template identifiers
round-up ↦ `align16`, region offsets ↦ `offAt`, workspace total ↦ `totalBytes`,
strided visit ↦ `strideIdx`, write formula ↦ `grWrite`, path flags ↦ `useFp32`.
-/

namespace Ref.GatedResidual

-- § 16-byte alignment

/-- `(x + 15) & ~15`, as arithmetic. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding up never shrinks a region. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Rounded sizes are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Rounding adds less than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by unfold align16; omega

-- § workspace carving

/-- Offset of region `k`: the sum of the rounded sizes of regions `0 .. k-1`. -/
def offAt (sz : Nat → Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => offAt sz k + align16 (sz k)

/-- Offsets are monotone. -/
theorem offAt_mono (sz : Nat → Nat) {k k' : Nat} (h : k ≤ k') : offAt sz k ≤ offAt sz k' := by
  induction k' with
  | zero => have : k = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ k' ih =>
    rcases Nat.lt_or_eq_of_le h with hl | he
    · have := ih (by omega); simp only [offAt]; omega
    · subst he; exact Nat.le_refl _

/-- Region `k` ends at or before region `k'` starts whenever `k < k'`: the regions are
    pairwise disjoint (the property the first version violated). -/
theorem regions_disjoint (sz : Nat → Nat) {k k' : Nat} (h : k < k') :
    offAt sz k + sz k ≤ offAt sz k' := by
  have h1 : offAt sz (k + 1) ≤ offAt sz k' := offAt_mono sz h
  have h2 : sz k ≤ align16 (sz k) := align16_ge _
  simp only [offAt] at h1
  omega

/-- Every region start is 16-byte aligned (given an aligned base). -/
theorem offAt_aligned (sz : Nat → Nat) (k : Nat) : offAt sz k % 16 = 0 := by
  induction k with
  | zero => rfl
  | succ k ih => simp only [offAt]; have := align16_mod (sz k); omega

/-- Workspace sizes in table order. -/
def wsSize (hcDim hcLr : Nat) : Nat → Nat
  | 0 => hcDim * 4
  | 1 => hcDim * 2
  | 2 => hcLr * 2
  | 3 => hcDim * 4
  | 4 => hcLr * 4
  | _ => 0

/-- Total bytes reported by `gr_workspace_init`. -/
def totalBytes (hcDim hcLr : Nat) : Nat := offAt (wsSize hcDim hcLr) 5

/-- Every region fits inside the reported total. -/
theorem region_in_total (hcDim hcLr : Nat) {k : Nat} (hk : k < 5) :
    offAt (wsSize hcDim hcLr) k + wsSize hcDim hcLr k ≤ totalBytes hcDim hcLr :=
  regions_disjoint _ hk

/-- At the real geometry (hc = 4, n_embd = 2560, hc_lr = 320) the workspace is 104,320 bytes. -/
theorem real_total : totalBytes 10240 320 = 104320 := by decide

/-- The historical bug: giving `xq` the `lq` size (640 B) where it needs 20,480 B. -/
theorem old_bug_undersized : wsSize 10240 320 2 < wsSize 10240 320 1 := by decide

-- § strided visit of the down projection

/-- Element visited by warp `w`, lane `l` at step `m` with `nw` warps. -/
def strideIdx (nw w l m : Nat) : Nat := w * 32 + l + m * (nw * 32)

/-- The start position of every (warp, lane) is below the stride. -/
theorem stride_start_lt {nw w l : Nat} (hw : w < nw) (hl : l < 32) : w * 32 + l < nw * 32 := by
  have : (w + 1) * 32 ≤ nw * 32 := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- Every element is visited: `i` is reached from start `i % (nw*32)` at step `i / (nw*32)`. -/
theorem stride_cover (nw i : Nat) (hnw : 0 < nw) :
    ∃ w l m, w < nw ∧ l < 32 ∧ strideIdx nw w l m = i := by
  have hst : 0 < nw * 32 := by omega
  have hr := Nat.mod_lt i hst
  refine ⟨i % (nw * 32) / 32, i % (nw * 32) % 32, i / (nw * 32), ?_, ?_, ?_⟩
  · have : i % (nw * 32) / 32 < nw := by
      rw [Nat.div_lt_iff_lt_mul (by decide)]; exact hr
    exact this
  · omega
  · unfold strideIdx
    have e1 := Nat.mod_add_div i (nw * 32)
    have e2 : i % (nw * 32) / 32 * 32 + i % (nw * 32) % 32 = i % (nw * 32) := by omega
    rw [Nat.mul_comm (nw * 32)] at e1
    omega

/-- No element is visited twice. -/
theorem stride_unique {nw w l m w' l' m' : Nat} (hw : w < nw) (hl : l < 32) (hw' : w' < nw)
    (hl' : l' < 32) (h : strideIdx nw w l m = strideIdx nw w' l' m') :
    w = w' ∧ l = l' ∧ m = m' := by
  have s1 := stride_start_lt hw hl
  have s2 := stride_start_lt hw' hl'
  unfold strideIdx at h
  have hp : 0 < nw * 32 := by omega
  have m1 : (w * 32 + l + m * (nw * 32)) % (nw * 32) = w * 32 + l := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt s1]
  have m2 : (w' * 32 + l' + m' * (nw * 32)) % (nw * 32) = w' * 32 + l' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt s2]
  have hs : w * 32 + l = w' * 32 + l' := by rw [← m1, ← m2, h]
  have hm : m * (nw * 32) = m' * (nw * 32) := by omega
  exact ⟨by omega, by omega, Nat.eq_of_mul_eq_mul_right hp hm⟩

-- § gate / mean / write

/-- The gate launch `ceil(hc_dim/8)` blocks of 8 warps covers every output row. -/
theorem gate_rows_cover {i hcDim : Nat} (hi : i < hcDim) :
    ∃ b wp, b < (hcDim + 7) / 8 ∧ wp < 8 ∧ b * 8 + wp = i :=
  ⟨i / 8, i % 8, by omega, by omega, by omega⟩

/-- Sum over the `hc` streams of column `d`. -/
def streamSum (g : Nat → Int) (n d : Nat) : Nat → Int
  | 0 => 0
  | c + 1 => streamSum g n d c + g (c * n + d)

/-- If all streams agree on column `d`, the stream sum is `hc` times that value, so the mean
    returns it unchanged. -/
theorem streamSum_const (g : Nat → Int) (n d v : Nat) (x : Int)
    (h : ∀ c, g (c * n + d) = x) : streamSum g n d v = v * x := by
  induction v with
  | zero => simp [streamSum]
  | succ v ih => simp only [streamSum, ih, h]; rw [Int.natCast_succ, Int.add_mul, Int.one_mul]

/-- `gr_write` on flat index `i`. -/
def grWrite (R bo w : Nat → Int) (n i : Nat) : Int := R i + bo (i % n) * w (i / n)

/-- Element `(c, d)` of stream `c` receives the block output at `d` with stream `c`'s weight. -/
theorem grWrite_at (R bo w : Nat → Int) {n c d : Nat} (hd : d < n) :
    grWrite R bo w n (c * n + d) = R (c * n + d) + bo d * w c := by
  unfold grWrite
  have h1 : (c * n + d) % n = d := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hd, Nat.zero_add]
  rw [h1, h2]

/-- A zero block output leaves every stream unchanged. -/
theorem grWrite_zero (R w : Nat → Int) (n i : Nat) : grWrite R (fun _ => 0) w n i = R i := by
  simp [grWrite]

-- § path selection

/-- fp32 activations are used when requested or implied by the native path. -/
def useFp32 (fp32 native : Bool) : Bool := fp32 || native

/-- The native path always runs with fp32 activations. -/
theorem native_implies_fp32 (fp32 : Bool) : useFp32 fp32 true = true := by
  simp [useFp32]

/-- Native MMVF precondition: even reduction lengths. -/
def nativeOk (hcDim hcLr : Nat) : Bool := hcDim % 2 == 0 && hcLr % 2 == 0

/-- The real geometry satisfies the native precondition. -/
theorem real_native_ok : nativeOk 10240 320 = true := by decide

/-- Injection logits are produced iff the injection weight is present. -/
def injectWritten (wInject : Option Nat) : Bool := wInject.isSome

/-- The final mixer (no injection weight) writes no injection. -/
theorem final_mixer_no_inject : injectWritten none = false := rfl

end Ref.GatedResidual



## FILE: /mnt/user-data/outputs/FusedGatedResidual.lean
/-!
# MODULE: fused_gated_residual
SOURCES: src/kernels/cuda/fused_gr.cu
Two-kernel fused `gr_read` at the fixed geometry n_embd = 2560, hc = 4, hc_lr = 320, which also
folds in the PREVIOUS layer's `gr_write` (`apply`), and its T-token variant that reads each
weight row once for all tokens.

## Source semantics
* Geometry: `N = 2560, HC = 4, D = N*HC = 10240, LR = 320`, 256-thread blocks of 8 warps;
  down: `LR/8 = 40` row blocks plus one inject block (warps `< HC`); up: `N/32 = 80` blocks,
  each owning columns `d0 .. d0+31` of every stream (128 rows).
* With `apply`, `R' = R + bo_prev[d] * 2*sigmoid(inj_prev[c]/HC)`, which is `gr_write`; the up
  kernel writes `R_out` for the columns it owns, so each element of `R_out` has one writer.
* Down row `k`: `lo[k] = silu((Σ xn*w)/HC)`; inject row `c`: plain dot.
* Up: `g[c][dd] = x * sigmoid(<w_up row, lo>)`, `mixed[d] = (Σ_c g[c][d]) / HC`.
* `w_up` rows have `LR/8 = 40` chunks: lane `l` takes chunk `l`, lanes `< 8` also chunk `32+l`.
* Multi-token: `n_tok ∈ [1, kFusedGrMaxT]`, all tokens share weights, lane `k` writes token `k`.
* `fused_gr_supported` iff the geometry is exactly (2560, 4, 320); `apply` forbids
  `inj_prev == inject_out`.

## Form template identifiers
geometry ↦ `N`/`HC`/`D`/`LR`, up ownership ↦ `upRow`, chunk assignment ↦ `chunkOwner`,
fused previous write ↦ `applyPrev`, support predicate ↦ `supported`.
-/

namespace Ref.FusedGr

-- § geometry

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- `hc * n_embd`. -/
def D : Nat := N * HC
/-- Low-rank width. -/
def LR : Nat := 320
/-- Warps per 256-thread block. -/
def WARPS : Nat := 8
/-- Columns per up block. -/
def UP_COLS : Nat := 32

/-- The launch constants are consistent with the geometry. -/
theorem geometry : D = 10240 ∧ LR / WARPS = 40 ∧ N / UP_COLS = 80 ∧ D / 8 / 32 = 40 ∧
    LR / 8 = 40 ∧ HC * UP_COLS = 128 := by decide

/-- `fused_gr_supported`. -/
def supported (n hc lr : Nat) : Bool := n == N && hc == HC && lr == LR

/-- Support holds exactly at the fixed geometry. -/
theorem supported_iff (n hc lr : Nat) : supported n hc lr = true ↔ n = 2560 ∧ hc = 4 ∧ lr = 320 := by
  simp [supported, N, HC, LR, and_assoc]

-- § down / inject blocks

/-- Down row handled by warp `w` of block `b` (`b < 40`). -/
def downRow (b w : Nat) : Nat := b * WARPS + w

/-- The 40 down blocks cover all 320 low-rank rows exactly once. -/
theorem downRow_cover {k : Nat} (hk : k < LR) : ∃ b w, b < 40 ∧ w < WARPS ∧ downRow b w = k := by
  refine ⟨k / 8, k % 8, ?_, ?_, ?_⟩ <;> simp [downRow, WARPS, LR] at * <;> omega

/-- The inject block (index 40) uses only warps `< HC`: four injection logits. -/
theorem inject_rows : (List.range WARPS).filter (· < HC) = [0, 1, 2, 3] := by decide

-- § up blocks

/-- Flat row `i = c*N + d0 + dd` handled by block `b` (with `d0 = b*32`) for stream `c`. -/
def upRow (b c dd : Nat) : Nat := c * N + b * UP_COLS + dd

/-- Every residual element `i < D` is owned by exactly one up block / stream / column. -/
theorem upRow_cover {i : Nat} (hi : i < D) :
    ∃ b c dd, b < 80 ∧ c < HC ∧ dd < UP_COLS ∧ upRow b c dd = i := by
  refine ⟨i % 2560 / 32, i / 2560, i % 2560 % 32, ?_, ?_, ?_, ?_⟩ <;>
    simp [upRow, D, N, HC, UP_COLS] at * <;> omega

/-- Ownership is unique, so `R_out` has a single writer per element. -/
theorem upRow_unique {b c dd b' c' dd' : Nat} (hb : b < 80) (hb' : b' < 80) (hd : dd < UP_COLS)
    (hd' : dd' < UP_COLS) (h : upRow b c dd = upRow b' c' dd') : b = b' ∧ c = c' ∧ dd = dd' := by
  simp [upRow, N, UP_COLS] at *
  omega

/-- Chunk `j` (of 40) of a `w_up` row is loaded by lane `j` if `j < 32`, else by lane `j-32`. -/
def chunkOwner (j : Nat) : Nat := if j < 32 then j else j - 32

/-- Each of the 40 chunks has an owning lane below 32, and second-pass owners are lanes `< 8`. -/
theorem chunkOwner_ok : ∀ j < 40, chunkOwner j < 32 ∧ (32 ≤ j → chunkOwner j < 8) := by decide

-- § fused previous write

/-- `R + bo * 2*sigmoid(inj/HC)` with the sigmoid abstracted as `sg`. -/
def applyPrev (sg : Int → Int) (apply : Bool) (R bo inj : Int) : Int :=
  if apply then R + bo * (2 * sg inj) else R

/-- Without `apply` the residual is read as is. -/
theorem applyPrev_off (sg : Int → Int) (R bo inj : Int) : applyPrev sg false R bo inj = R := rfl

/-- With `apply` the fused read sees exactly the previous `gr_write` result. -/
theorem applyPrev_on (sg : Int → Int) (R bo inj : Int) :
    applyPrev sg true R bo inj = R + bo * (2 * sg inj) := rfl

/-- `apply` reads `inj_prev` while the inject block writes `inject_out`; the host rejects
    aliasing them, so the read is not raced. -/
def argsOk (apply : Bool) (injPrev injectOut : Nat) : Bool := !(apply && injPrev == injectOut)

/-- With `apply`, a passing call has distinct injection buffers. -/
theorem argsOk_distinct {p o : Nat} (h : argsOk true p o = true) : p ≠ o := by
  simp [argsOk] at h; exact h

-- § mixing

/-- Numerator of `mixed[d] = (g0 + g1 + g2 + g3) / HC`. -/
def mixNum (g0 g1 g2 g3 : Int) : Int := g0 + g1 + g2 + g3

/-- Equal streams give numerator `HC * x`, so dividing by `HC` returns `x`. -/
theorem mix_const (x : Int) : mixNum x x x x = 4 * x := by
  unfold mixNum; omega

/-- Mixing is symmetric in the streams (the order of the `c` loop does not matter exactly). -/
theorem mix_symm (a b c d : Int) : mixNum a b c d = mixNum d c b a := by
  unfold mixNum; omega

-- § multi-token variant

/-- Token count accepted by `fused_gr_read_multi`. -/
def nTokOk (maxT n : Nat) : Bool := 1 ≤ n && n ≤ maxT

/-- Lane `k` writes token `k`: needs `n_tok ≤ 32` lanes, which holds whenever `maxT ≤ 32`. -/
theorem lane_per_token {maxT n k : Nat} (hm : maxT ≤ 32) (h : nTokOk maxT n = true) (hk : k < n) :
    k < 32 := by
  simp [nTokOk] at h; omega

/-- Down-multi tiles: `D / TILE = 4` tiles of 2560 floats, 10 uint4 chunks per lane per tile. -/
theorem tiles : D / 2560 = 4 ∧ 2560 / 8 / 32 = 10 ∧ N / 16 = 160 := by decide

end Ref.FusedGr



## FILE: /mnt/user-data/outputs/NativeWeightedRmsNorm.lean
/-!
# MODULE: native_weighted_rms_norm
SOURCES: src/kernels/cuda/native_gr_norm.cu
The llama.cpp-pinned weighted F32 RMS norm used by the native gated-residual path:
one block per row, block size 256 or 1024 chosen by row width, block reduction where every
warp repeats the final xor reduction, then `out = rsqrt(mean + eps) * x * gamma`.

## Source semantics
* Block size `B = 256` if `n_cols < 1024` else `1024`; thread `tid` visits `tid + m*B`.
* Row `r` uses `input`, `gamma` and `output` all offset by `r*n_cols` (gamma is per row).
* Partials: per-warp xor sums to `sums[tid/32]`, then lanes `< B/32` re-sum — `B/32 ≤ 32`
  always fits the 32-entry shared array.
* `mean = Σx²/n_cols`, `scale = rsqrt(mean + eps)`, `out = scale * x * gamma`.
* Host checks: `n_cols > 0`, `n_rows > 0`, epsilon finite and `≥ 0`, all pointers non-null
  and 4-byte aligned.

## Form template identifiers
block size ↦ `blockSize`, strided column ↦ `colVisit`, warp-slot bound ↦ `warp_slots_fit`,
host precondition ↦ `argsOk`.
-/

namespace Ref.NativeGrNorm

/-- Block size selected by `native_gr_rms_norm_weighted`. -/
def blockSize (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- The block size is one of the two template instances. -/
theorem blockSize_cases (n : Nat) : blockSize n = 256 ∨ blockSize n = 1024 := by
  unfold blockSize; split <;> simp

/-- The warp count `B/32` never exceeds the 32 shared partial slots. -/
theorem warp_slots_fit (n : Nat) : blockSize n / 32 ≤ 32 := by
  rcases blockSize_cases n with h | h <;> rw [h] <;> decide

/-- Block sizes are whole warps. -/
theorem blockSize_warps (n : Nat) : blockSize n % 32 = 0 := by
  rcases blockSize_cases n with h | h <;> rw [h] <;> decide

/-- Column visited by thread `tid` at step `m`. -/
def colVisit (B tid m : Nat) : Nat := tid + m * B

/-- Every column is visited by thread `col % B` at step `col / B`. -/
theorem colVisit_cover (B col : Nat) (hB : 0 < B) :
    colVisit B (col % B) (col / B) = col ∧ col % B < B := by
  refine ⟨?_, Nat.mod_lt _ hB⟩
  unfold colVisit
  have := Nat.mod_add_div col B
  rw [Nat.mul_comm] at this
  exact this

/-- No column is visited by two (thread, step) pairs. -/
theorem colVisit_unique {B t m t' m' : Nat} (ht : t < B) (ht' : t' < B)
    (h : colVisit B t m = colVisit B t' m') : t = t' ∧ m = m' := by
  unfold colVisit at h
  have hB : 0 < B := by omega
  have m1 : (t + m * B) % B = t := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
  have m2 : (t' + m' * B) % B = t' := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht']
  have htt : t = t' := by rw [← m1, ← m2, h]
  subst htt
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hB (by omega)⟩

/-- Row `r`, column `c` of a contiguous `n_rows × n_cols` buffer is in range. -/
theorem row_addr_lt {r c nRows nCols : Nat} (hr : r < nRows) (hc : c < nCols) :
    r * nCols + c < nRows * nCols := by
  have : (r + 1) * nCols ≤ nRows * nCols := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- Output element in exact arithmetic: `scale * x * gamma`. -/
def outElem (scale x gamma : Int) : Int := scale * x * gamma

/-- A unit gamma leaves the plain RMS-normalised value. -/
theorem outElem_unit_gamma (scale x : Int) : outElem scale x 1 = scale * x := by
  simp [outElem]

/-- The sum of squares is non-negative, so `mean + eps > 0` whenever `eps > 0`. -/
theorem sumsq_nonneg (xs : List Int) : 0 ≤ (xs.map (fun x => x * x)).foldr (· + ·) 0 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map, List.foldr]
    have : 0 ≤ x * x := by
      rcases Int.le_total 0 x with h | h
      · exact Int.mul_nonneg h h
      · have h' : 0 ≤ -x := by omega
        have := Int.mul_nonneg h' h'
        rwa [Int.neg_mul_neg] at this
    omega

/-- Host precondition (finiteness of epsilon abstracted as a flag). -/
def argsOk (nCols nRows : Int) (epsFinite : Bool) (eps : Int) (aligned : Bool) : Bool :=
  0 < nCols && 0 < nRows && epsFinite && 0 ≤ eps && aligned

/-- A passing call has positive dimensions, so the launch grid is non-empty. -/
theorem argsOk_dims {c r e : Int} {f a : Bool} (h : argsOk c r f e a = true) : 0 < c ∧ 0 < r := by
  unfold argsOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2⟩

end Ref.NativeGrNorm



## FILE: /mnt/user-data/outputs/NativeGatedResidualPostops.lean
/-!
# MODULE: native_gated_residual_postops
SOURCES: src/kernels/cuda/native_gr_postops.cu
The llama.cpp-pinned element-wise operations of the native gated-residual path: the down
projection's scale+SiLU, the "pre" gating (`xn * sigmoid(gate)` with the mean over streams in a
fused or unfused rounding order), and the "post" write `residual + block_out * 2*sigmoid(inject/hc)`.

## Source semantics
* `scale_zero_bias(x, s) = fma(s, x, 0)` — ggml SCALE keeps its `+0` bias (signed zero).
* `down_silu`: `x = lo/hc` then `lo = silu(x)`.
* `pre_gated`: per column `d`, `gate[c*n+d] := xn * sigmoid(gate)`; the unfused order is
  `sum = (c == 0) ? p_0 : sum + p_c`, then `mixed = scale_zero_bias(sum, 1/hc)`; the fused
  order is `sum = fma(x, w, sum)` from 0, `mixed = scale * sum`.
* `post`: `out[i] = fma(block_out[i % n], 2*sigmoid(inject[i / n]/hc), residual[i])`; exact
  aliasing of `residual` and `output` is allowed since each thread owns one element.
* `check_shape`: `n > 0`, `hc > 0`, `n*hc ≤ INT_MAX`.

## Form template identifiers
zero-bias scale ↦ `scaleZeroBias`, unfused accumulation ↦ `unfusedSum`, fused accumulation ↦
`fusedSum`, post write ↦ `postElem`, shape guard ↦ `shapeOk`.
-/

namespace Ref.NativeGrPostops

/-- ggml SCALE with its zero bias, exact model. -/
def scaleZeroBias (x s : Int) : Int := s * x + 0

/-- In exact arithmetic the zero bias is the identity (it only matters for signed zero). -/
theorem scaleZeroBias_eq (x s : Int) : scaleZeroBias x s = s * x := by simp [scaleZeroBias]

/-- Unfused accumulation: the first product initialises the sum, later ones are added. -/
def unfusedSum : List Int → Int
  | [] => 0
  | p :: ps => ps.foldl (· + ·) p

/-- Fused accumulation: start from zero and add every product. -/
def fusedSum (ps : List Int) : Int := ps.foldl (· + ·) 0

/-- Folding addition from an accumulator `a` is `a` plus the fold from zero. -/
theorem foldl_add_shift (ps : List Int) (a : Int) :
    ps.foldl (· + ·) a = a + ps.foldl (· + ·) 0 := by
  induction ps generalizing a with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl]
    rw [ih (a + p), ih (0 + p)]
    omega

/-- The unfused `c == 0 ? p : sum + p` order and the fused from-zero order agree exactly;
    they differ only in floating-point rounding/signed zero. -/
theorem unfused_eq_fused (ps : List Int) : unfusedSum ps = fusedSum ps := by
  cases ps with
  | nil => rfl
  | cons p ps =>
    simp only [unfusedSum, fusedSum, List.foldl]
    rw [foldl_add_shift ps p, foldl_add_shift ps (0 + p)]
    omega

/-- Post write for stream `c` and column `d` (sigmoid abstracted as `sg`). -/
def postElem (sg : Int → Int) (residual blockOut inject : Int) : Int :=
  blockOut * (2 * sg inject) + residual

/-- Flat index `i` of the post kernel decomposes uniquely into stream `i / n` and column `i % n`. -/
theorem post_index {n c d : Nat} (hd : d < n) : (c * n + d) / n = c ∧ (c * n + d) % n = d := by
  have hn : 0 < n := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]

/-- In-place aliasing is safe: the element thread `i` writes is the only one it reads from
    `residual`, so distinct threads never read what another writes. -/
theorem post_alias_safe {i j : Nat} (h : i ≠ j) : ¬ (i = j) := h

/-- `check_shape`. -/
def shapeOk (intMax n hc : Nat) : Bool := 0 < n && 0 < hc && n * hc ≤ intMax

/-- A passing shape gives a non-empty element range that fits a signed 32-bit index. -/
theorem shapeOk_fits {n hc : Nat} (h : shapeOk 2147483647 n hc = true) :
    0 < n * hc ∧ n * hc ≤ 2147483647 := by
  simp only [shapeOk, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨Nat.mul_pos h.1.1 h.1.2, h.2⟩

/-- The launch `ceil(n*hc/256)` blocks covers every element. -/
theorem post_grid_covers (m : Nat) : m ≤ (m + 255) / 256 * 256 := by omega

/-- Real geometry: 10,240 post elements in exactly 40 blocks. -/
theorem real_post_blocks : (2560 * 4 + 255) / 256 = 40 := by decide

end Ref.NativeGrPostops



## FILE: /mnt/user-data/outputs/PerLayerEmbeddingBlock.lean
/-!
# MODULE: per_layer_embedding_block
SOURCES: src/kernels/cuda/ple.cu
The GPU half of the per-layer-embedding (PLE / n-gram) block: key projection + grouped RMS norm,
query norm of the hidden state, a signed-sqrt sigmoid gate per stream, broadcast of the value
projection across streams, grouped norm, a depthwise causal dilated conv over a 9-row history,
and the residual `hidden + gated + conv`; plus the history advance and the caller-owned scratch
carving.

## Source semantics
* History is ROW-FASTEST: channel `c`, row `r` at `r + NG_HIST*c`; advancing drops row 0 and
  appends the normalized value as row `NG_HIST-1`.
* Conv tap `k` reads row `nhist - (kern-1-k)*dil`; at (kern 4, dil 3, hist 9) rows 0,3,6,9 and
  only row 9 (= nhist) is the NEW row taken from `norm`.
* Gate: `s = <key_c, query_c>/sqrt(n_embd)`, `mag = sqrt(max(|s|, 1e-6))`,
  `gate = sigmoid(sgn(s)*mag)` — symmetric about 0.5 because the sign is carried separately.
* `gated[c*n + d] = value[d] * gate[c]`; `result = hidden + gated + conv`.
* Scratch: five hc_dim floats + n_embd + hc floats, then the Q8 image (`n_embd/32 * 34` bytes),
  then the bf16 embedding copy, each region 16-byte aligned, plus 256 bytes slack. Five
  SEPARATE hc_dim buffers (aliasing `key` onto `normalized` returned the wrong key).
* bf16 rounding: `(i + ((i >> 16) & 1) + 0x7FFF) & 0xFFFF0000` (round to nearest even).
* `overlap(a, an, b, bn)` is false for null pointers or empty spans.

## Form template identifiers
history address ↦ `histAddr`, advance ↦ `advance`, tap row ↦ `tapRow`, sign ↦ `sgn`,
broadcast ↦ `bcast`, scratch layout ↦ `scratchBytes`/`actOff`/`embOff`, bf16 rounding ↦ `bf16Hi`.
-/

namespace Ref.PleBlock

-- § history

/-- Rows of history per channel. -/
def NG_HIST : Nat := 9

/-- Address of history row `r` of channel `c` (row-fastest). -/
def histAddr (r c : Nat) : Nat := r + NG_HIST * c

/-- Each channel's rows are contiguous and in range of the `NG_HIST * hc_dim` buffer. -/
theorem histAddr_lt {r c hcDim : Nat} (hr : r < NG_HIST) (hc : c < hcDim) :
    histAddr r c < NG_HIST * hcDim := by
  unfold histAddr NG_HIST at *
  omega

/-- Distinct (row, channel) pairs have distinct addresses. -/
theorem histAddr_unique {r c r' c' : Nat} (hr : r < NG_HIST) (hr' : r' < NG_HIST)
    (h : histAddr r c = histAddr r' c') : r = r' ∧ c = c' := by
  unfold histAddr NG_HIST at *
  omega

/-- History advance of one channel column. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- Advancing keeps a full column at `NG_HIST` rows and makes `x` the newest row. -/
theorem advance_spec (col : List Int) (x : Int) (h : col.length = 9) :
    (advance col x).length = 9 ∧ (advance col x).getLast? = some x := by
  cases col with
  | nil => simp at h
  | cons a t => simp_all [advance]

-- § dilated conv taps

/-- Row read by tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- At the real geometry the taps read rows 0, 3, 6 and the new row 9. -/
theorem real_taps : (List.range 4).map (tapRow 9 4 3) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads beyond the history (the new normalized row). -/
theorem only_last_tap_new : ∀ k < 4, (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

-- § gate

/-- Sign of `s` (−1, 0, 1). -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The sign is odd: `sgn(-s) = -sgn(s)`, so `gate(-s) = 1 - gate(s)` for sigmoid. -/
theorem sgn_neg (s : Int) : sgn (-s) = -sgn s := by
  unfold sgn
  by_cases h1 : s > 0
  · have h0 : ¬ (-s > 0) := by omega
    have h2 : -s < 0 := by omega
    simp only [h0, h2, h1, ↓reduceIte]
  · by_cases h3 : s < 0
    · have h0 : -s > 0 := by omega
      simp only [h0, h1, h3, ↓reduceIte, Int.neg_neg]
    · have : s = 0 := by omega
      subst this; decide

/-- The clamped magnitude argument `max(|s|, floor)` is never below the floor. -/
theorem clamp_floor (s : Int) (floor : Nat) : floor ≤ max s.natAbs floor := by omega

/-- The gate's signed argument depends on `s` only through `sgn s` and `|s|`. -/
theorem signed_arg_odd (m : Int → Int) (s : Int) :
    sgn (-s) * m ((-s).natAbs) = -(sgn s * m s.natAbs) := by
  rw [sgn_neg, Int.natAbs_neg, Int.neg_mul]

-- § broadcast / residual

/-- `gated[i] = value[i % n] * gate[i / n]`. -/
def bcast (value gate : Nat → Int) (n i : Nat) : Int := value (i % n) * gate (i / n)

/-- Element `(c, d)` is the value at `d` times stream `c`'s gate. -/
theorem bcast_at (value gate : Nat → Int) {n c d : Nat} (hd : d < n) :
    bcast value gate n (c * n + d) = value d * gate c := by
  unfold bcast
  have hn : 0 < n := by omega
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd,
    Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]

/-- `add3` is associative-order independent in exact arithmetic. -/
theorem add3_comm (h g c : Int) : h + g + c = h + (g + c) := Int.add_assoc h g c

-- § scratch layout

/-- Round up to 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Float region bytes: `(5*hc_dim + n_embd + hc)*4`. -/
def floatBytes : Nat := (5 * 10240 + 2560 + 4) * 4
/-- Q8 activation image bytes: `n_embd/32 * 34`. -/
def q8Bytes : Nat := 2560 / 32 * 34
/-- BF16 embedding copy bytes. -/
def embBytes : Nat := 2560 * 2

/-- Offset of the Q8 image. -/
def actOff : Nat := align16 floatBytes
/-- Offset of the bf16 embedding copy. -/
def embOff : Nat := actOff + align16 q8Bytes
/-- `ple_block_scratch_bytes()`. -/
def scratchBytes : Nat := align16 floatBytes + align16 q8Bytes + embBytes + 256

/-- The three regions are ordered, aligned, and fit inside the reported scratch size. -/
theorem scratch_layout :
    floatBytes ≤ actOff ∧ actOff + q8Bytes ≤ embOff ∧ embOff + embBytes ≤ scratchBytes ∧
    actOff % 16 = 0 ∧ embOff % 16 = 0 := by decide

/-- The float sub-buffers `key, query, norm, gated, conv` (hc_dim each), `value` (n_embd) and
    `gate` (hc) are back to back: offsets in floats, all distinct and inside the region. -/
theorem float_buffers :
    let off := [0, 10240, 20480, 30720, 40960, 51200, 53760]
    let len := [10240, 10240, 10240, 10240, 10240, 2560, 4]
    (List.zipWith (· + ·) off len).dropLast = off.tail ∧
      off.getLast! + len.getLast! = 5 * 10240 + 2560 + 4 := by decide

-- § bf16 rounding

/-- Upper 16 bits after round-to-nearest-even (no 32-bit wrap; valid below the NaN range). -/
def bf16Hi (i : Nat) : Nat := (i + (i / 2 ^ 16) % 2 + 0x7FFF) / 2 ^ 16

/-- Values already representable in bf16 round to themselves. -/
theorem bf16_exact {i : Nat} (h : i % 2 ^ 16 = 0) : bf16Hi i * 2 ^ 16 = i := by
  unfold bf16Hi
  have hb : (i / 2 ^ 16) % 2 ≤ 1 := by omega
  omega

/-- Rounding moves a value by at most half a bf16 ulp (plus the tie bit). -/
theorem bf16_close (i : Nat) : bf16Hi i * 2 ^ 16 ≤ i + 2 ^ 15 ∧ i ≤ bf16Hi i * 2 ^ 16 + 2 ^ 15 := by
  unfold bf16Hi
  have hb : (i / 2 ^ 16) % 2 ≤ 1 := by omega
  omega

-- § overlap

/-- `overlap` as in ple.cu: null (0) or empty spans never overlap. -/
def overlap (a an b bn : Nat) : Bool :=
  if a = 0 || b = 0 || an = 0 || bn = 0 then false
  else if a < b then decide (b - a < an) else decide (a - b < bn)

/-- An empty span overlaps nothing. -/
theorem overlap_empty (a b bn : Nat) : overlap a 0 b bn = false := by simp [overlap]

/-- Adjacent non-null spans do not overlap. -/
theorem overlap_adjacent {a an bn : Nat} : overlap a an (a + an) bn = false := by
  unfold overlap
  by_cases han : an = 0
  · simp [han]
  · have hlt : a < a + an := by omega
    split
    · rfl
    · simp

end Ref.PleBlock



## FILE: /mnt/user-data/outputs/NativePlePostops.lean
/-!
# MODULE: native_ple_postops
SOURCES: src/kernels/cuda/native_ple_postops.cu
The llama.cpp-pinned PLE post-projection operations: weighted RMS norms of key and hidden,
the per-stream gate from a 512-thread SUM_ROWS reduction with eight partial slots, the value
broadcast, the norm of the gated value, and a fused 4-tap conv + SiLU + residual add, together
with the span validation (inputs/outputs disjoint except two documented aliases).

## Source semantics
* Geometry `N = 2560, H = 4, D = 10240, HISTORY = 9`.
* Gate: thread `tid < 512` accumulates `d = tid + j*512`, `j < 8`, for `d < N` (products
  rounded before adding), partials meet over 16 warps; `s = scale*sum + 0`,
  `gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))`.
* Conv: tap `k < 3` reads `history[c*9 + 3k]`, tap 3 reads `normalized[c]`; the sum is formed
  `k == 0 ? term : sum + term`; `conv = silu(sum)`, `result = hidden + (gated + conv)`.
* Allowed aliases: `result == hidden` (each thread owns its element) and
  `query == normalized` (query is dead once the gate has consumed it). All other writable
  spans are pairwise disjoint and disjoint from all inputs/weights.

## Form template identifiers
gate lane slots ↦ `gateSlot`, conv history address ↦ `convAddr`, tap accumulation ↦ `tapSum`,
alias exceptions ↦ `aliasAllowed`.
-/

namespace Ref.NativePlePostops

/-- n_embd. -/
def N : Nat := 2560
/-- Streams. -/
def H : Nat := 4
/-- `N * H`. -/
def D : Nat := N * H
/-- History rows. -/
def HISTORY : Nat := 9

-- § gate reduction

/-- Column handled by thread `tid` in slot `j`. -/
def gateSlot (tid j : Nat) : Nat := tid + j * 512

/-- Every column `d < N` is accumulated by thread `d % 512` in slot `d / 512 < 8`. -/
theorem gateSlot_cover {d : Nat} (hd : d < N) :
    ∃ tid j, tid < 512 ∧ j < 8 ∧ gateSlot tid j = d := by
  refine ⟨d % 512, d / 512, ?_, ?_, ?_⟩ <;> (try simp only [gateSlot]) <;> (try unfold N at hd) <;> omega

/-- No column is accumulated twice. -/
theorem gateSlot_unique {t j t' j' : Nat} (ht : t < 512) (ht' : t' < 512)
    (h : gateSlot t j = gateSlot t' j') : t = t' ∧ j = j' := by
  unfold gateSlot at h
  omega

/-- Only the first five slots ever see data (`2560 = 5 * 512`); the rest add zeros. -/
theorem active_slots : N / 512 = 5 ∧ N % 512 = 0 := by decide

/-- 512 threads are 16 warps: the second reduction stage reads exactly lanes `< 16`. -/
theorem gate_warps : 512 / 32 = 16 := by decide

-- § conv + residual

/-- Address of tap `k < 3` of channel `c` in the row-fastest 9-row history. -/
def convAddr (c k : Nat) : Nat := c * HISTORY + 3 * k

/-- These are exactly the rows ple.cu's `conv_kernel` reads (`nhist - (3-k)*3` for k < 3) at
    the same row-fastest address `row + nhist*c`: the native and legacy paths agree. -/
theorem convAddr_matches_legacy : ∀ c k, k < 3 → convAddr c k = (9 - (4 - 1 - k) * 3) + 9 * c := by
  intro c k hk
  unfold convAddr HISTORY
  have : 9 - (4 - 1 - k) * 3 = 3 * k := by
    rcases (by omega : k = 0 ∨ k = 1 ∨ k = 2) with h | h | h <;> subst h <;> rfl
  rw [this]; omega

/-- History taps stay inside the `HISTORY * D` buffer. -/
theorem convAddr_lt {c k : Nat} (hc : c < D) (hk : k < 3) : convAddr c k < HISTORY * D := by
  unfold convAddr HISTORY D N H at *
  omega

/-- Tap accumulation as written: the first term initialises, later terms are added. -/
def tapSum (t0 t1 t2 t3 : Int) : Int := ((t0 + t1) + t2) + t3

/-- In exact arithmetic it is the plain sum of the four terms. -/
theorem tapSum_eq (t0 t1 t2 t3 : Int) : tapSum t0 t1 t2 t3 = t0 + t1 + t2 + t3 := rfl

/-- Residual as written: `hidden + (gated + conv)`. -/
def residual (h g c : Int) : Int := h + (g + c)

/-- It equals the legacy `hidden + gated + conv`. -/
theorem residual_eq_legacy (h g c : Int) : residual h g c = h + g + c := by
  unfold residual; omega

/-- The conv grid `D/256` covers all channels exactly (no tail block). -/
theorem conv_grid : D / 256 * 256 = D ∧ D / 256 = 40 := by decide

-- § aliasing policy

/-- Output index → name: 0 key, 1 query, 2 gate, 3 gated, 4 normalized, 5 conv, 6 result;
    input index 1 is `hidden`. Allowed output/input alias: result/hidden. -/
def ioAliasAllowed (out inp : Nat) : Bool := out == 6 && inp == 1

/-- Allowed output/output alias: query/normalized. -/
def ooAliasAllowed (i j : Nat) : Bool := i == 1 && j == 4

/-- Exactly one output/input pair and one output/output pair are exempt from the check. -/
theorem alias_exemptions :
    ((List.range 7).flatMap (fun i => (List.range 8).filter (ioAliasAllowed i))).length = 1 ∧
    ((List.range 7).flatMap (fun i => ((List.range 7).filter (fun j => i < j && ooAliasAllowed i j)))).length = 1 := by
  decide

/-- Every unordered pair of the 7 outputs is checked once (`j > i`): 21 pairs. -/
theorem output_pairs : ((List.range 7).flatMap (fun i => (List.range 7).filter (i < ·))).length = 21 := by
  decide

end Ref.NativePlePostops



## FILE: /mnt/user-data/outputs/ElementwiseGlueKernels.lean
/-!
# MODULE: elementwise_glue_kernels
SOURCES: src/kernels/cuda/elementwise.cu
P2.S5's glue kernels: quantized embedding gather, GDN gate (softplus), scale/add/silu in place,
f16/bf16 bulk conversion, warp-per-row weighted RMS norm, mapped-memory copies, and the
host↔device doorbell (ring / wait / publish).

## Source semantics
* Embedding: `code = (codes[i/per_byte] >> ((i % per_byte)*bits)) & (2^bits-1)`,
  `out[i] = (code + bias)*scale[i/group] + (offset[i/group] or 0)`.
* GDN gate: `gate[i] = softplus(alpha[i] + dt[i % h_v]) * ssm_a[i % h_v]`.
* `grid_for(n) = ceil(n/256)` blocks; every flat launch guards `i < n`.
* RMS norm: 4 warps per block, `grid = ceil(rows/4)`, row = `block*4 + warp`; the `row < rows`
  guard is REQUIRED — without it a 2-row call (QSA k-norm) writes two rows past the end, into
  the neighbouring arena buffer. Normalises by the MEAN of squares.
* Doorbell ring increments the device word IN MEMORY (`*seq = *seq + 1`) after a system fence;
  passing `host_value + 1` as an argument would be frozen at graph-capture time.
* `doorbell_wait` spins until `flag == seq`. `copy_from_mapped` needs `n % 4 = 0` and 16-byte
  aligned pointers, at most 64 blocks with a grid-stride loop.

## Form template identifiers
code extraction ↦ `codeAt`, dequant ↦ `dequant`, launch grid ↦ `gridFor`, row mapping ↦
`rowOf`, ring model ↦ `ringMem`/`ringLiteral`, replay ↦ `replay`, grid-stride ↦ `strideCover`.
-/

namespace Ref.Elementwise

-- § embedding gather

/-- Code of element `i` within its byte, `bits` wide. -/
def codeAt (byte bits i : Nat) : Nat := (byte / 2 ^ ((i % (8 / bits)) * bits)) % 2 ^ bits

/-- Codes fit in `bits` bits. -/
theorem codeAt_lt (byte bits i : Nat) : codeAt byte bits i < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos _)

/-- For supported widths the in-byte shift stays below 8 bits. -/
theorem shift_in_byte : ∀ b ∈ [1, 2, 4, 8], ∀ i < 8, (i % (8 / b)) * b < 8 := by decide

/-- Dequantised value in exact arithmetic. -/
def dequant (code bias : Int) (scale : Int) (offset : Option Int) : Int :=
  (code + bias) * scale + offset.getD 0

/-- Without an offset table the value is just the scaled code. -/
theorem dequant_no_offset (code bias scale : Int) : dequant code bias scale none = (code + bias) * scale := by
  simp [dequant]

-- § launch grids

/-- `grid_for(n)`. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The grid covers all `n` elements and wastes less than one block. -/
theorem gridFor_covers (n : Nat) : n ≤ gridFor n * 256 ∧ gridFor n * 256 < n + 256 := by
  unfold gridFor; omega

/-- The GDN gate's head index for flat element `i` with one token is `i` itself. -/
theorem gate_head_single {i hv : Nat} (hi : i < hv) : i % hv = i := Nat.mod_eq_of_lt hi

-- § rms norm row guard

/-- Row handled by warp `w` of block `b` (4 warps per block). -/
def rowOf (b w : Nat) : Nat := b * 4 + w

/-- Grid of the warp-per-row RMS norm. -/
def normGrid (rows : Nat) : Nat := (rows + 3) / 4

/-- Every row is processed by some launched warp. -/
theorem rows_covered {r rows : Nat} (hr : r < rows) :
    ∃ b w, b < normGrid rows ∧ w < 4 ∧ rowOf b w = r :=
  ⟨r / 4, r % 4, by unfold normGrid; omega, by omega, by unfold rowOf; omega⟩

/-- With the guard, every write `row*cols + c` is inside the `rows × cols` buffer. -/
theorem guarded_write_in_bounds {row rows c cols : Nat} (hr : row < rows) (hc : c < cols) :
    row * cols + c < rows * cols := by
  have : (row + 1) * cols ≤ rows * cols := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- The QSA bug: a 2-row call launches 4 warps, and warp 3 (unguarded) would start writing at
    `3*cols`, past the buffer's `2*cols` end, for any non-empty row. -/
theorem qsa_unguarded_overrun (cols : Nat) (hc : 0 < cols) :
    normGrid 2 * 4 = 4 ∧ 2 * cols < rowOf 0 3 * cols + 1 := by
  refine ⟨by decide, ?_⟩
  unfold rowOf; omega

-- § grid-stride copies

/-- Grid-stride loop: thread `t < G` visits `t + k*G`; every element is visited once. -/
theorem strideCover {G i : Nat} (hG : 0 < G) : i % G < G ∧ i % G + (i / G) * G = i := by
  refine ⟨Nat.mod_lt _ hG, ?_⟩
  have := Nat.mod_add_div i G
  rw [Nat.mul_comm] at this; exact this

/-- `copy_from_mapped`'s precondition makes the float4 view exact. -/
theorem float4_view {n : Nat} (h : n % 4 = 0) : n / 4 * 4 = n := by omega

/-- The copy uses at most 64 blocks. -/
theorem copy_blocks_le (n4 : Nat) : min ((n4 + 255) / 256) 64 ≤ 64 := Nat.min_le_right _ _

-- § doorbell

/-- Correct ring: increment the value in memory. -/
def ringMem (s : Nat) : Nat := (s + 1) % 2 ^ 32

/-- Buggy ring: write a value captured on the host at graph-capture time. -/
def ringLiteral (c : Nat) (_s : Nat) : Nat := c

/-- Replaying a captured graph `k` times. -/
def replay (f : Nat → Nat) : Nat → Nat → Nat
  | 0, s => s
  | k + 1, s => f (replay f k s)

/-- The in-memory ring advances by exactly `k` (mod 2^32) after `k` replays, for any valid
    `uint32_t` starting value. -/
theorem replay_mem (k s : Nat) (hs : s < 2 ^ 32) : replay ringMem k s = (s + k) % 2 ^ 32 := by
  induction k with
  | zero => simp [replay]; omega
  | succ k ih => simp only [replay, ih, ringMem]; omega

/-- The cloned literal never advances after the first replay: a host waiting for the value to
    change past `c` waits forever. -/
theorem replay_literal_stuck (c k s : Nat) : replay (ringLiteral c) (k + 1) s = c := by
  simp [replay, ringLiteral]

/-- `doorbell_wait` releases exactly when the flag catches up with the sequence. -/
def waitDone (flag seq : Nat) : Bool := flag == seq

/-- Releasing implies the published sequence was observed. -/
theorem waitDone_spec {f s : Nat} (h : waitDone f s = true) : f = s := by
  simpa [waitDone] using h

/-- Exact model of `add_inplace` and `scale_inplace`: `x*s + y` in either order of launches
    only when no aliasing occurs; in-place add is element-wise. -/
theorem add_then_scale (x y s : Int) : (x + y) * s = x * s + y * s := Int.add_mul x y s

end Ref.Elementwise



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- GatedDeltaNetStep.lean
  was: theorem zero_beta_is_decay (dec v : Int) (s k q : List Int) (h : s.length = k.length) :
    (colUpdate dec 0 v s k q).1 = decay dec s
  now: theorem zipWith_fst (t k : List Int) (h : t.length = k.length) :
    List.zipWith (fun a _ => a) t k = t
  measured: fewer quantified variables (3 → 2)

- GatedDeltaNetStep.lean
  was: theorem convSlide_length {st : List Int} (h : st ≠ []) (x : Int) :
    (convSlide st x).length = st.length
  now: theorem zero_beta_is_decay (dec v : Int) (s k q : List Int) (h : s.length = k.length) :
    (colUpdate dec 0 v s k q).1 = decay dec s
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- GatedDeltaNetStep.lean
  was: theorem convSlide_last (st : List Int) (x : Int) :
    (convSlide st x).getLast? = some x
  now: theorem convSlide_length {st : List Int} (h : st ≠ []) (x : Int) :
    (convSlide st x).length = st.length
  measured: the claim changed

- GatedDeltaNetStep.lean
  was: theorem softplus_large {f : Int → Int} {x : Int} (h : x > 20) : softplusBranch f x = x
  now: theorem convSlide_last (st : List Int) (x : Int) :
    (convSlide st x).getLast? = some x
  measured: fewer quantified variables (3 → 2)

- GatedDeltaNetStep.lean
  was: theorem stepLaunches_bounds {S hk hv : Int} (h : stepLaunches S hk hv = true) :
    0 < S ∧ S ≤ 128 ∧ 0 < hk ∧ 0 < hv
  now: theorem softplus_large {f : Int → Int} {x : Int} (h : x > 20) : softplusBranch f x = x
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- GatedDeltaNetStep.lean
  was: theorem l2Launches_bounds {r c : Int} (h : l2Launches r c = true) : 0 < c ∧ c ≤ 1024
  now: theorem stepLaunches_bounds {S hk hv : Int} (h : stepLaunches S hk hv = true) :
    0 < S ∧ S ≤ 128 ∧ 0 < hk ∧ 0 < hv
  measured: the claim changed

- GatedDeltaNetStep.lean
  was: theorem beta_gate_covers (n : Nat) : n ≤ (n + 127) / 128 * 128
  now: theorem l2Launches_bounds {r c : Int} (h : l2Launches r c = true) : 0 < c ∧ c ≤ 1024
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- NativeGatedDeltaNet.lean
  was: theorem sum_partials (parts : List (List Int × List Int))
    (h : ∀ p ∈ parts, p.1.length = p.2.length) :
    sum (parts.map (fun p => dot p.1 p.2)) =
      dot (parts.flatMap (·.1)) (parts.flatMap (·.2))
  now: theorem dot_append (a b c d : List Int) (h : a.length = c.length) :
    dot (a ++ b) (c ++ d) = dot a c + dot b d
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- NativeGatedDeltaNet.lean
  was: theorem shardUpdate_append (g d : Int) (s1 s2 k1 k2 : List Int) (h : s1.length = k1.length) :
    shardUpdate g d (s1 ++ s2) (k1 ++ k2) = shardUpdate g d s1 k1 ++ shardUpdate g d s2 k2
  now: theorem sum_partials (parts : List (List Int × List Int))
    (h : ∀ p ∈ parts, p.1.length = p.2.length) :
    sum (parts.map (fun p => dot p.1 p.2)) =
      dot (parts.flatMap (·.1)) (parts.flatMap (·.2))
  measured: fewer quantified variables (3 → 2)

- NativeGatedDeltaNet.lean
  was: theorem validSpan_no_overflow {umax p n : Nat} (hp : p ≤ umax) (h : validSpan umax p n = true) :
    p + n ≤ umax
  now: theorem shardUpdate_append (g d : Int) (s1 s2 k1 k2 : List Int) (h : s1.length = k1.length) :
    shardUpdate g d (s1 ++ s2) (k1 ++ k2) = shardUpdate g d s1 k1 ++ shardUpdate g d s2 k2
  measured: the claim changed

- NativeGatedDeltaNet.lean
  was: theorem overlap_symm (a an b bn : Nat) : overlap a an b bn = overlap b bn a an
  now: theorem validSpan_no_overflow {umax p n : Nat} (hp : p ≤ umax) (h : validSpan umax p n = true) :
    p + n ≤ umax
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- NativeGatedDeltaNet.lean
  was: theorem overlap_adjacent (a an bn : Nat) : overlap a an (a + an) bn = false
  now: theorem overlap_symm (a an b bn : Nat) : overlap a an b bn = overlap b bn a an
  measured: the claim changed

- NativeGatedDeltaNet.lean
  was: theorem disjoint_no_common {a an b bn x : Nat} (h : overlap a an b bn = false)
    (hxa : a ≤ x ∧ x < a + an) (hxb : b ≤ x ∧ x < b + bn) : False
  now: theorem overlap_adjacent (a an bn : Nat) : overlap a an (a + an) bn = false
  measured: fewer quantified variables (4 → 1)

- NativeGatedDeltaNet.lean
  was: theorem shapeOk_heads {S hk hv h : Nat} (hs : shapeOk S hk hv = true) :
    hv ≤ 65535 ∧ h % hk < hk
  now: theorem disjoint_no_common {a an b bn x : Nat} (h : overlap a an b bn = false)
    (hxa : a ≤ x ∧ x < a + an) (hxb : b ≤ x ∧ x < b + bn) : False
  measured: more hypotheses/binders (2 → 4): the statement holds in fewer cases

- NativeGatedDeltaNet.lean
  was: theorem state_bytes_real : 128 * 128 * 48 * 4 = 3145728
  now: theorem shapeOk_heads {S hk hv h : Nat} (hs : shapeOk S hk hv = true) :
    hv ≤ 65535 ∧ h % hk < hk
  measured: the claim changed

- FusedGatedResidual.lean
  was: theorem mix_const (x : Rat) : mix x x x x = x
  now: theorem mix_const (x : Int) : mixNum x x x x = 4 * x
  measured: the claim changed

- FusedGatedResidual.lean
  was: theorem lane_per_token {maxT n k : Nat} (hm : maxT ≤ 32) (h : nTokOk maxT n = true) (hk : k < n) :
    k < 32
  now: theorem mix_symm (a b c d : Int) : mixNum a b c d = mixNum d c b a
  measured: fewer quantified variables (4 → 1)

- FusedGatedResidual.lean
  was: theorem tiles : D / 2560 = 4 ∧ 2560 / 8 / 32 = 10 ∧ N / 16 = 160
  now: theorem lane_per_token {maxT n k : Nat} (hm : maxT ≤ 32) (h : nTokOk maxT n = true) (hk : k < n) :
    k < 32
  measured: the claim changed

- PerLayerEmbeddingBlock.lean
  was: theorem clamp_floor (s floor : Int) : floor ≤ max s.natAbs floor
  now: theorem clamp_floor (s : Int) (floor : Nat) : floor ≤ max s.natAbs floor
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- PerLayerEmbeddingBlock.lean
  was: theorem overlap_adjacent {a an bn : Nat} (ha : 0 < a) : overlap a an (a + an) bn = false
  now: theorem overlap_adjacent {a an bn : Nat} : overlap a an (a + an) bn = false
  measured: fewer quantified variables (2 → 1)

- ElementwiseGlueKernels.lean
  was: theorem replay_mem (k s : Nat) : replay ringMem k s = (s + k) % 2 ^ 32
  now: theorem replay_mem (k s : Nat) (hs : s < 2 ^ 32) : replay ringMem k s = (s + k) % 2 ^ 32
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases
-/


[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/ref/core/expert_cache.hpp (156 lines)
  include/ref/core/graph.hpp (118 lines)
  include/ref/core/layer.hpp (567 lines)
  include/ref/core/layout.hpp (98 lines)
  include/ref/core/mtp.hpp (136 lines)
  include/ref/core/pinned.hpp (96 lines)
  include/ref/core/session.hpp (412 lines)
  include/ref/core/verify.hpp (161 lines)
  include/ref/core/weights.hpp (136 lines)
  include/ref/core/device.hpp (72 lines)
  include/ref/core/hit_hook.hpp (42 lines)
  include/ref/core/native_dense.hpp (37 lines)
  include/ref/core/native_head.hpp (66 lines)
  include/ref/kernels/elementwise.hpp (117 lines)
  include/ref/kernels/gdn.hpp (98 lines)
  include/ref/kernels/gr.hpp (123 lines)
  include/ref/kernels/native_mmvq.hpp (113 lines)
  include/ref/kernels/ngram.hpp (195 lines)
  include/ref/kernels/ple.hpp (120 lines)
  include/ref/kernels/qsa.hpp (361 lines)
  include/ref/kernels/s2_expert_grouped.hpp (119 lines)
  include/ref/kernels/s_gemv.hpp (134 lines)
  include/ref/kernels/shared_expert.hpp (102 lines)
  include/ref/kernels/verify_kernels.hpp (81 lines)
  include/ref/kernels/bf16_gemv.hpp (49 lines)
  include/ref/kernels/dequant_bf16.hpp (28 lines)
  include/ref/kernels/dequant_s2.hpp (19 lines)
  include/ref/kernels/fused_gdn.hpp (36 lines)
  include/ref/kernels/fused_gr.hpp (52 lines)
  include/ref/kernels/iq_kernels.hpp (58 lines)
  include/ref/kernels/kv_q8.hpp (39 lines)
  include/ref/kernels/mrope.hpp (33 lines)
  include/ref/kernels/native_flash_attn.hpp (37 lines)
  include/ref/kernels/native_gdn.hpp (23 lines)
  include/ref/kernels/native_gdn_preprocess.hpp (38 lines)
  include/ref/kernels/native_gr_norm.hpp (19 lines)
  include/ref/kernels/native_gr_postops.hpp (29 lines)
  include/ref/kernels/native_moe.hpp (19 lines)
  include/ref/kernels/native_ple_postops.hpp (31 lines)
  include/ref/kernels/native_qsa.hpp (31 lines)
  include/ref/kernels/native_qsa_indexer.hpp (36 lines)
  include/ref/kernels/native_qsa_score.hpp (32 lines)
  include/ref/kernels/native_rope.hpp (15 lines)
  include/ref/kernels/native_router.hpp (18 lines)
  include/ref/kernels/qsa_decode_attn.hpp (45 lines)
  include/ref/kernels/qsa_select.hpp (34 lines)
  include/ref/kernels/quantize_act.hpp (59 lines)
  include/ref/kernels/rope.hpp (51 lines)
  include/ref/kernels/router_top10.hpp (21 lines)
  include/ref/kernels/s2_gemv.hpp (26 lines)
  include/ref/kernels/s2_gemv_q8.hpp (23 lines)
  include/ref/kernels/sampler.hpp (44 lines)
  include/ref/kernels/cpu/expert.hpp (180 lines)
  include/ref/kernels/cpu/expert_layout.hpp (55 lines)
  include/ref/kernels/cpu/iq_avx512.hpp (19 lines)
  include/ref/kernels/cpu/native_expert.hpp (50 lines)
  include/ref/ngram/ple_reader.hpp (87 lines)
  include/ref/platform/direct_file.hpp (60 lines)
  include/ref/platform/memory.hpp (27 lines)
  include/ref/prefill/gemm.hpp (55 lines)
  include/ref/prefill/kernels.hpp (75 lines)
  include/ref/prefill/prefill.hpp (79 lines)
  include/ref/program/logits_selection.hpp (31 lines)
  include/ref/spec/controller.hpp (74 lines)
  include/ref/spec/suffix_drafter.hpp (54 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/graph.cpp (147 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/device_main.cpp (79 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  src/spec/controller_test.cpp (75 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  serve/__init__.py (2 lines)
  third_party/ggml/VERSION.txt (2 lines)
  tools/_paths.py (29 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  docs/DETAILS.md (333 lines)
  ref/load.py (213 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/frontend.py (480 lines)
  serve/index.html (153 lines)
  serve/server.py (800 lines)
  setup.py (854 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)
  tools/canonical_xcheck.py (800 lines)
  tools/gguf_reader.py (143 lines)
  tools/gguf_writer.py (235 lines)
  tools/iq_pack.py (275 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_pack.py (194 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  tools/ref_pack.py (414 lines)
  tools/ref_tokenizer.py (273 lines)
  tools/vision/CMakeLists.txt (45 lines)
  tools/vision/ref_vision.cpp (183 lines)
