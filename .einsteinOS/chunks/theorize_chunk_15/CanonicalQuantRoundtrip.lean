/-!
# MODULE: canonical_quant_roundtrip
Integer-level model of the ggml block-quant → canonical-plane mappings (Q2_0, Q3_K, Q4_0, Q4_K, Q5_0,
Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, floats) checked by `canonical_xcheck.py`. Only the exact integer
parts are modelled (code ranges, bias, nibble/bit layout, scale unpacking, index arithmetic); the
f32 multiply is left abstract.

## Source semantics
* Every code plane is UNSIGNED; a signed/affine codebook is an integer bias applied to the code
  BEFORE the single f32 multiply (`-1` Q2_0, `-4` Q3_K, `-8` Q4_0, `-16` Q5_0, `-32` Q6_K, `-128` Q8_0).
* Nibble split: low nibbles fill the first half of a group, high nibbles the second (not interleaved).
* `get_scale_min_k4` and the Q3_K / IQ4_XS scale unpackers recombine low bits and high bits from
  separate bytes; all results are 6-bit.
* Every stored code is below `2^bits` of its form; every group size divides its block size.

## Form template identifiers
* `decodeAffine`        ← `from_s2 / from_s4_* / from_s8_*` (integer bias then scale)
* `scaleMinK4`          ← `get_scale_min_k4`
* `q3Scale`             ← `unpack_q3_k_scales`
* `lsOf`                ← IQ4_XS `ls` reconstruction
* `Row` / `table`       ← `MAPPINGS` registry
* `kvIq4nl`             ← `KV_IQ4NL`
-/

namespace CanonicalQuant

-- § affine decode with integer bias

/-- Affine decode in the integer domain: `(code + bias) * scale`. -/
def decodeAffine (code : Nat) (bias scale : Int) : Int := ((code : Int) + bias) * scale

/-- Q2_0 codebook is `{-d, 0, d, 2d}`: the `-1` offset is part of the codebook. -/
theorem q2_0_value_set (c : Nat) (d : Int) (h : c < 4) :
    decodeAffine c (-1) d = -d ∨ decodeAffine c (-1) d = 0 ∨
    decodeAffine c (-1) d = d ∨ decodeAffine c (-1) d = 2 * d := by
  unfold decodeAffine
  interval_cases c <;> omega

/-- Decoding Q2_0 codes as unsigned `code * d` would differ from the real decode for every code. -/
theorem q2_0_unsigned_misread (c : Nat) (d : Int) (hd : d ≠ 0) :
    decodeAffine c (-1) d ≠ (c : Int) * d := by
  unfold decodeAffine
  intro h
  have h2 : ((c : Int) + -1) * d = (c : Int) * d + (-1) * d := Int.add_mul _ _ _
  rw [h2] at h
  have : (-1 : Int) * d = 0 := by omega
  omega

/-- Q4_0 bias: a 4-bit code decodes to an integer in `-8..7`. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) + -8 ∧ (c : Int) + -8 ≤ 7 := by omega

/-- Q3_K bias: a code in `0..7` decodes to an integer in `-4..3`. -/
theorem q3_k_range (c : Nat) (h : c < 8) : -4 ≤ (c : Int) + -4 ∧ (c : Int) + -4 ≤ 3 := by omega

/-- Q5_0 bias: a 5-bit code decodes to an integer in `-16..15`. -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ (c : Int) + -16 ∧ (c : Int) + -16 ≤ 15 := by omega

/-- Q6_K bias: a 6-bit code decodes to an integer in `-32..31`. -/
theorem q6_k_range (c : Nat) (h : c < 64) : -32 ≤ (c : Int) + -32 ∧ (c : Int) + -32 ≤ 31 := by omega

/-- Q8_0: storing the signed `int8` code `+128` and subtracting 128 recovers it exactly. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    0 ≤ s + 128 ∧ s + 128 < 256 ∧ (s + 128) - 128 = s := by omega

/-- Folding a bias into the value domain is an integer identity (which is why the float version needs care). -/
theorem bias_fold_int (c b a : Int) : (c + b) * a = c * a + b * a := Int.add_mul c b a

-- § nibble and bit layout

/-- Low nibble of a byte. -/
def lowNib (b : Nat) : Nat := b % 16

/-- High nibble of a byte. -/
def highNib (b : Nat) : Nat := b / 16

/-- A byte is recomposed from its two nibbles. -/
theorem nibble_recompose (b : Nat) (h : b < 256) : lowNib b + 16 * highNib b = b := by
  unfold lowNib highNib; omega

/-- The low nibble is a 4-bit code. -/
theorem lowNib_lt (b : Nat) : lowNib b < 16 := by unfold lowNib; omega

/-- The high nibble of a byte is a 4-bit code. -/
theorem highNib_lt (b : Nat) (h : b < 256) : highNib b < 16 := by unfold highNib; omega

/-- Q4_0/IQ4_NL layout: element `e` of a 32-block lives at byte `e % 16`, half `e / 16` (no interleave). -/
theorem q4_position_decomposition (e : Nat) (h : e < 32) :
    e % 16 + 16 * (e / 16) = e ∧ e % 16 < 16 ∧ e / 16 < 2 := by omega

/-- Distinct elements of a 32-block occupy distinct (byte, half) slots. -/
theorem q4_position_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h : a % 16 = b % 16 ∧ a / 16 = b / 16) : a = b := by omega

/-- Pack four 2-bit codes into a byte, LSB first (Q2_0's convention). -/
def pack4x2 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Unpack the `t`-th 2-bit code of a byte. -/
def unpack2 (b t : Nat) : Nat := (b / 4 ^ t) % 4

/-- A packed Q2_0 byte fits in 8 bits. -/
theorem pack4x2_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4x2 c0 c1 c2 c3 < 256 := by unfold pack4x2; omega

/-- Round trip of the first 2-bit slot. -/
theorem unpack2_pack_0 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 0 = c0 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the second 2-bit slot. -/
theorem unpack2_pack_1 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 1 = c1 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the third 2-bit slot. -/
theorem unpack2_pack_2 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 2 = c2 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the fourth 2-bit slot. -/
theorem unpack2_pack_3 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 3 = c3 := by
  simp [unpack2, pack4x2] <;> omega

/-- Any 2-bit slot of a byte is a valid 2-bit code. -/
theorem unpack2_lt (b t : Nat) : unpack2 b t < 4 := by
  unfold unpack2; exact Nat.mod_lt _ (by decide)

-- § get_scale_min_k4

/-- `get_scale_min_k4`: 6-bit (scale, min) of group `j` from the 12 packed bytes. -/
def scaleMinK4 (s : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (s j % 64, s (j + 4) % 64)
  else (s (j + 4) % 16 + (s (j - 4) / 64) * 16, s (j + 4) / 16 + (s j / 64) * 16)

/-- Both outputs of `get_scale_min_k4` are 6-bit for byte inputs. -/
theorem scaleMinK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    (scaleMinK4 s j).1 < 64 ∧ (scaleMinK4 s j).2 < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  by_cases h : j < 4
  · simp [scaleMinK4, h] <;> omega
  · simp [scaleMinK4, h] <;> omega

/-- For the first four groups only `s[j]` and `s[j+4]` are read (low 6 bits). -/
theorem scaleMinK4_low (s : Nat → Nat) (j : Nat) (h : j < 4) :
    scaleMinK4 s j = (s j % 64, s (j + 4) % 64) := by simp [scaleMinK4, h]

-- § Q3_K scales and element order

/-- Select the source byte for group-quarter `k` and take the low or high nibble. -/
def nibSel (a b k : Nat) : Nat :=
  if k < 2 then (if k % 2 = 0 then a else b) % 16 else (if k % 2 = 0 then a else b) / 16

/-- The selected nibble is 4-bit for byte inputs. -/
theorem nibSel_lt (a b k : Nat) (ha : a < 256) (hb : b < 256) : nibSel a b k < 16 := by
  unfold nibSel
  split_ifs <;> omega

/-- `unpack_q3_k_scales` for group `g < 16`: nibble from `s[0..7]`, two high bits from `s[8..11]`. -/
def q3Scale (s : Nat → Nat) (g : Nat) : Nat :=
  nibSel (s (g % 4)) (s (4 + g % 4)) (g / 4) + ((s (8 + g % 4) / 4 ^ (g / 4)) % 4) * 16

/-- Every unpacked Q3_K scale is 6-bit. -/
theorem q3Scale_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (g : Nat) (hg : g < 16) :
    q3Scale s g < 64 := by
  have h1 := hs (g % 4) (by omega)
  have h2 := hs (4 + g % 4) (by omega)
  have h3 := nibSel_lt (s (g % 4)) (s (4 + g % 4)) (g / 4) h1 h2
  have h4 := Nat.mod_lt (s (8 + g % 4) / 4 ^ (g / 4)) (by decide : 4 > 0)
  unfold q3Scale
  omega

/-- Q3_K scale minus 32 lies in `-32..31`. -/
theorem q3Scale_bias_range (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (g : Nat) (hg : g < 16) :
    -32 ≤ (q3Scale s g : Int) - 32 ∧ (q3Scale s g : Int) - 32 ≤ 31 := by
  have := q3Scale_lt s hs g hg
  omega

/-- Q3_K element order `128*half + 32*j + 16*sub + l` equals `16*g + l` with `g = 8*half + 2*j + sub`. -/
theorem q3_element_order (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    128 * half + 32 * j + 16 * sub + l = 16 * (8 * half + 2 * j + sub) + l ∧
    8 * half + 2 * j + sub < 16 := by omega

/-- The hmask bit index `4*half + j` never resets between halves: it is injective in `(half, j)`. -/
theorem q3_hmask_bit_injective (h1 h2 j1 j2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4) (b2 : j2 < 4)
    (h : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by omega

/-- Q3_K stored code `(2 bits) + 4*(hm bit)` stays in `0..7`. -/
theorem q3_code_lt (c2 m : Nat) (h : c2 < 4) (hm : m < 2) : c2 + 4 * m < 8 := by omega

-- § Q4_K / Q5_K

/-- Q4_K/Q5_K: element `64*j + 32*h + l` belongs to group `2*j + h`. -/
theorem q4k_group_of_element (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    (64 * j + 32 * h + l) / 32 = 2 * j + h ∧ 2 * j + h < 8 := by omega

/-- Q5_K high-bit masks `u1 = 4^j` take values 1,4,16,64. -/
theorem q5k_u1_values : (List.range 4).map (fun j => 4 ^ j) = [1, 4, 16, 64] := by decide

/-- Q5_K high-bit masks `u2 = 2*4^j` take values 2,8,32,128. -/
theorem q5k_u2_values : (List.range 4).map (fun j => 2 * 4 ^ j) = [2, 8, 32, 128] := by decide

/-- The eight Q5_K masks are pairwise distinct (each of the 8 bits of a `qh` byte is used once). -/
theorem q5k_masks_nodup : ([1, 4, 16, 64] ++ [2, 8, 32, 128]).Nodup := by decide

/-- A Q5_K / Q5_0 5-bit code `nibble + 16*bit` stays below 32. -/
theorem five_bit_code_lt (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : nib + 16 * bit < 32 := by omega

-- § Q6_K

/-- A Q6_K 6-bit code `nibble + 16*(2 bits)` stays below 64. -/
theorem q6k_code_lt (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) : nib + 16 * hi < 64 := by omega

/-- Q6_K quads land at stride 32: the scale group of element `128*half + 32*q + l` is `8*half + 2*q + l/16`. -/
theorem q6k_group_of_element (half q l : Nat) (hh : half < 2) (hq : q < 4) (hl : l < 32) :
    (128 * half + 32 * q + l) / 16 = 8 * half + 2 * q + l / 16 ∧ 8 * half + 2 * q + l / 16 < 16 := by omega

-- § IQ4_XS and the IQ4 codebook

/-- IQ4_XS 6-bit scale: nibble of `scales_l[ib/2]` plus two bits of `scales_h`. -/
def lsOf (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  (sl (ib / 2) / 16 ^ (ib % 2)) % 16 + ((sh / 4 ^ ib) % 4) * 16

/-- IQ4_XS `ls` is 6-bit. -/
theorem lsOf_lt (sl : Nat → Nat) (sh ib : Nat) : lsOf sl sh ib < 64 := by
  have h1 := Nat.mod_lt (sl (ib / 2) / 16 ^ (ib % 2)) (by decide : 16 > 0)
  have h2 := Nat.mod_lt (sh / 4 ^ ib) (by decide : 4 > 0)
  unfold lsOf
  omega

/-- The IQ4_NL codebook `kvalues_iq4nl`. -/
def kvIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries, one per 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing, hence non-affine but monotone. -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (· < ·) := by decide

/-- The codebook is not an affine function of the code: second differences are not all zero. -/
theorem kvIq4nl_nonaffine : (13 : Int) - 1 ≠ 1 - (-10) := by decide

-- § registry well-formedness

/-- One row of the `MAPPINGS` registry (numeric fields only). -/
structure Row where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  bits : Nat
  maxCode : Nat

/-- The registry as transcribed from `MAPPINGS`. -/
def table : List Row :=
  [ ⟨"Q2_0", 18, 64, 64, 2, 3⟩, ⟨"Q3_K", 110, 256, 16, 4, 7⟩, ⟨"Q4_0", 18, 32, 32, 4, 15⟩,
    ⟨"Q4_K", 144, 256, 32, 4, 15⟩, ⟨"Q5_0", 22, 32, 32, 8, 31⟩, ⟨"Q5_K", 176, 256, 32, 8, 31⟩,
    ⟨"Q6_K", 210, 256, 16, 8, 63⟩, ⟨"Q8_0", 34, 32, 32, 8, 255⟩, ⟨"IQ4_NL", 18, 32, 32, 4, 15⟩,
    ⟨"IQ4_XS", 136, 256, 32, 4, 15⟩ ]

/-- Every group size divides its block, and every maximal code fits its code plane. -/
theorem table_wf :
    table.all (fun r => r.blockElems % r.groupElems == 0 && decide (r.maxCode < 2 ^ r.bits)) = true := by
  decide

/-- The registry has ten quantized rows. -/
theorem table_length : table.length = 10 := by decide

/-- Block byte sizes are the sums of their fields (scale/min/high-bits/codes). -/
theorem block_byte_sums :
    18 = 2 + 16 ∧ 110 = 32 + 64 + 12 + 2 ∧ 144 = 2 + 2 + 12 + 128 ∧ 22 = 2 + 4 + 16 ∧
    176 = 2 + 2 + 12 + 32 + 128 ∧ 210 = 128 + 64 + 16 + 2 ∧ 34 = 2 + 32 ∧ 136 = 2 + 2 + 4 + 128 := by
  decide

/-- Code-plane widths divide a byte evenly, so `8 / bits` codes pack exactly per byte. -/
theorem bits_divide_byte : [2, 4, 8].all (fun b => (8 / b) * b == 8) = true := by decide

end CanonicalQuant
