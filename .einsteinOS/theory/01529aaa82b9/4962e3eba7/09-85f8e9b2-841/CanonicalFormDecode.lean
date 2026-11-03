/-!
# MODULE: canonical_form_decode
Arithmetic content of `tools/canonical_xcheck.py`: each ggml quantized block type is mapped to a
canonical form (codes plane + scale plane [+ offset plane]) whose decode must equal the ggml decode
bit for bit.  This module states the integer-level invariants that make those mappings well formed.

## Source semantics
* A canonical decode is `(code + bias) * scale` : ONE integer bias before ONE multiply (never folded
  into the value domain).  Q2_0 bias -1, Q3_K -4, Q4_0 -8, Q5_0 -16, Q6_K -32, Q8_0 -128 (code stored +128).
* Q4_K / Q5_K decode `code * scale + offset` with 6-bit scale/min fields unpacked by `get_scale_min_k4`.
* IQ4_NL / IQ4_XS use the non-affine 16-entry table `kvalues_iq4nl`.
* Nibble layout: low nibbles fill the first half of a group, high nibbles the second half.
* Q3_K hmask bit `4*half + j` never resets; Q5_K masks `u1/u2` walk 1,4,16,64 / 2,8,32,128.
* Every code plane is unsigned and fits the declared form width (S2/S4/S8).

## Form template identifiers
* decodeBias        <- from_s2 / from_s4_* / from_s8_* (affine decode)
* pack4             <- Q2_0 byte layout (4 codes per byte, LSB first)
* scaleK4 / minK4   <- get_scale_min_k4 (ggml-quants.c L880)
* kvIq4nl           <- KV_IQ4NL codebook
* Mapping/mappings  <- MAPPINGS registry (namespace entries: form, group_elems, code_bias, has_offset)
-/

namespace Strata.CanonicalForm

-- § Integer-bias affine decode

/-- Canonical affine decode: add the integer bias to the code, then multiply by the scale once. -/
def decodeBias (bias : Int) (code : Nat) (scale : Int) : Int :=
  ((code : Int) + bias) * scale

/-- Q2_0 codes 0..3 decode to symbols in {-1,0,1,2}. -/
theorem q2_0_symbol_range (c : Nat) (h : c < 4) :
    -1 ≤ (c : Int) + (-1) ∧ (c : Int) + (-1) ≤ 2 := by
  omega

/-- The four Q2_0 levels are exactly -d, 0, d, 2d (the `- 1` offset is part of the codebook). -/
theorem q2_0_decode_levels (d : Int) :
    decodeBias (-1) 0 d = -d ∧ decodeBias (-1) 1 d = 0 ∧
    decodeBias (-1) 2 d = d ∧ decodeBias (-1) 3 d = 2 * d := by
  unfold decodeBias
  omega

/-- Treating Q2_0 codes as unsigned `code * d` would shift every nonzero level by one step. -/
theorem q2_0_unsigned_reading_differs (d : Int) (hd : d ≠ 0) :
    decodeBias (-1) 1 d ≠ decodeBias 0 1 d := by
  unfold decodeBias
  intro h
  apply hd
  omega

/-- Q4_0: a 4-bit code with bias -8 covers -8..7. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q8_0: signed int8 stored as unsigned `+128`; the decode `code - 128` recovers the int8. -/
theorem q8_0_bias_roundtrip (i : Int) (h1 : -128 ≤ i) (h2 : i ≤ 127) :
    (i + 128).toNat < 256 ∧ ((i + 128).toNat : Int) - 128 = i := by
  omega

-- § Nibble and bit-plane layouts

/-- A byte splits into a low and a high nibble that recombine exactly. -/
theorem nibble_split (b : Nat) (h : b < 256) :
    b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- Four 2-bit codes in one byte, LSB first (byte j/4 of a Q2_0 block). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed quad always fits one byte. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- Unpacking with `(byte >> 2t) & 3` returns the original codes. -/
theorem unpack4_pack4 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
    pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 = c3 := by
  unfold pack4
  omega

/-- Packing the unpacked quads of any byte gives the byte back (no source bit is dropped). -/
theorem pack4_unpack4 (b : Nat) (h : b < 256) :
    pack4 (b % 4) (b / 4 % 4) (b / 16 % 4) (b / 64) = b := by
  unfold pack4
  omega

/-- Element j of a Q2_0 block lives in byte j/4 at bit offset 2*(j%4). -/
theorem q2_0_element_index (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ j % 4 < 4 ∧ 4 * (j / 4) + j % 4 = j ∧ (j % 4) * 2 < 8 := by
  omega

/-- Q2_0 block bytes: 2-byte scale plus 64/4 code bytes. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by decide

-- § Q3_K

/-- Q3_K: group index, element base and hmask bit for (half, j, sub). -/
theorem q3_k_group_layout (half j sub : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) :
    8 * half + 2 * j + sub < 16 ∧
    16 * (8 * half + 2 * j + sub) = 128 * half + 32 * j + 16 * sub ∧
    4 * half + j < 8 ∧ half * 32 + sub * 16 + 15 < 64 := by
  omega

/-- Q3_K stored code: 2 low bits plus the hmask bit worth 4 stays within 0..7. -/
theorem q3_k_code_range (c2 m : Nat) (h2 : c2 < 4) (hm : m < 2) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) - 4 ∧ ((c2 + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

/-- Joining a 4-bit low part with two high bits always gives a 6-bit scale. -/
theorem join6_lt (lo hi : Nat) : lo % 16 + 16 * (hi % 4) < 64 := by
  omega

/-- Q3_K scale group `4+j`'s two high bits come from bits 2-3 of `s[8+j]`. -/
theorem q3_k_scale_6bit (s lo : Nat) : lo % 16 + 16 * ((s / 4) % 4) < 64 := by
  omega

/-- Q3_K scale minus 32 lies in -32..31. -/
theorem q3_k_scale_signed (v : Nat) (h : v < 64) : -32 ≤ (v : Int) - 32 ∧ (v : Int) - 32 ≤ 31 := by
  omega

-- § Q4_K / Q5_K scale and min unpacking

/-- `get_scale_min_k4`, scale half, over a byte accessor `s`. -/
def scaleK4 (s : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then s j % 64 else s (j + 4) % 16 + 16 * (s (j - 4) / 64)

/-- `get_scale_min_k4`, min half. -/
def minK4 (s : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then s (j + 4) % 64 else s (j + 4) / 16 + 16 * (s j / 64)

/-- Scales of all 8 groups are 6-bit when the 12 scale bytes are bytes. -/
theorem scaleK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    scaleK4 s j < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  unfold scaleK4
  split_ifs <;> omega

/-- Mins of all 8 groups are 6-bit when the 12 scale bytes are bytes. -/
theorem minK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    minK4 s j < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  unfold minK4
  split_ifs <;> omega

/-- Reading only `s[j]` is wrong for the high four groups: they differ from the low-group formula. -/
theorem scaleK4_high_uses_top_bits (s : Nat → Nat) (j : Nat) (hj : 4 ≤ j) :
    scaleK4 s j = s (j + 4) % 16 + 16 * (s (j - 4) / 64) := by
  unfold scaleK4
  rw [if_neg (by omega)]

/-- Q4_K: step j covers elements 64j..64j+63 as groups 2j (low nibbles) and 2j+1 (high nibbles). -/
theorem q4_k_step_layout (j : Nat) (hj : j < 4) :
    2 * j + 1 < 8 ∧ j * 64 + 32 + 32 ≤ 256 := by
  omega

/-- Q5_K bit masks used for the 5th code bit. -/
def u1 (j : Nat) : Nat := 2 ^ (2 * j)

/-- Q5_K second mask, one bit above `u1`. -/
def u2 (j : Nat) : Nat := 2 * 2 ^ (2 * j)

/-- The masks walk 1,4,16,64 and 2,8,32,128. -/
theorem q5_k_masks :
    [u1 0, u1 1, u1 2, u1 3] = [1, 4, 16, 64] ∧ [u2 0, u2 1, u2 2, u2 3] = [2, 8, 32, 128] := by
  decide

/-- The eight masks are pairwise distinct across `u1` and `u2` and cover all 8 bits. -/
theorem q5_k_masks_disjoint : ∀ j : Fin 4, ∀ k : Fin 4, u1 j.val ≠ u2 k.val := by
  decide

/-- Sum of the eight masks is 255: every bit of `qh[l]` is used exactly once. -/
theorem q5_k_masks_cover :
    u1 0 + u1 1 + u1 2 + u1 3 + u2 0 + u2 1 + u2 2 + u2 3 = 255 := by
  decide

/-- A Q5 code (4 bits plus the 16-valued fifth bit) is below 32. -/
theorem q5_code_range (nib b : Nat) (hn : nib < 16) (hb : b < 2) : nib + 16 * b < 32 := by
  omega

/-- Q5_0: the source's `(qh >> (j+12)) & 0x10` is bit `j+16` of `qh`. -/
theorem q5_0_high_bit_shift (qh j : Nat) :
    ((qh / 2 ^ (j + 12)) / 16) % 2 = (qh / 2 ^ (j + 16)) % 2 := by
  rw [Nat.div_div_eq_div_mul]
  have h : 2 ^ (j + 12) * 16 = 2 ^ (j + 16) := by
    ring
  rw [h]

-- § Q6_K

/-- Q6_K code: 4 low bits plus 2 high bits is below 64, bias -32 gives -32..31. -/
theorem q6_k_code_range (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) :
    nib + 16 * hi < 64 ∧ -32 ≤ ((nib + 16 * hi : Nat) : Int) - 32 ∧
    ((nib + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q6_K quads land at strides of 32, so the scale group is plain consecutive 16-element groups. -/
theorem q6_k_quad_layout (half q l : Nat) (hh : half < 2) (hq : q < 4) (hl : l < 32) :
    128 * half + 32 * q + l < 256 ∧
    (128 * half + 32 * q + l) / 16 = 8 * half + 2 * q + l / 16 ∧
    8 * half + 2 * q + l / 16 < 16 := by
  omega

-- § IQ4 codebook (non-affine)

/-- `kvalues_iq4nl`. -/
def kvIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvIq4nl_strictly_increasing :
    ∀ i : Fin 15, kvIq4nl.getD i.val 0 < kvIq4nl.getD (i.val + 1) 0 := by
  decide

/-- It is not equally spaced, hence not an integer bias on the code. -/
theorem kvIq4nl_not_affine :
    kvIq4nl.getD 1 0 - kvIq4nl.getD 0 0 ≠ kvIq4nl.getD 2 0 - kvIq4nl.getD 1 0 := by
  decide

/-- IQ4_XS: scale `ls - 32` for a 6-bit `ls` lies in -32..31. -/
theorem iq4_xs_scale_range (a b : Nat) :
    (a % 16 + 16 * (b % 4) : Nat) < 64 ∧ -32 ≤ ((a % 16 + 16 * (b % 4) : Nat) : Int) - 32 ∧
    ((a % 16 + 16 * (b % 4) : Nat) : Int) - 32 ≤ 31 := by
  omega

-- § Block geometry

/-- Byte layouts of the K-quant and IQ blocks. -/
theorem block_byte_layouts :
    2 + 2 + 12 + 128 = 144 ∧ 32 + 64 + 12 + 2 = 110 ∧ 4 + 12 + 32 + 128 = 176 ∧
    128 + 64 + 16 + 2 = 210 ∧ 2 + 2 + 4 + 128 = 136 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧
    2 + 16 = 18 := by
  decide

/-- Registry row of the canonical mapping table. -/
structure Mapping where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBits : Nat
  maxCode : Nat
  codeBias : Int
  hasOffset : Bool

/-- The ten quantized rows of MAPPINGS (the three float rows are pass-through). -/
def mappings : List Mapping :=
  [ ⟨"Q2_0", 18, 64, 64, 2, 3, -1, false⟩,
    ⟨"Q3_K", 110, 256, 16, 4, 7, -4, false⟩,
    ⟨"Q4_0", 18, 32, 32, 4, 15, -8, false⟩,
    ⟨"Q4_K", 144, 256, 32, 4, 15, 0, true⟩,
    ⟨"Q5_0", 22, 32, 32, 8, 31, -16, false⟩,
    ⟨"Q5_K", 176, 256, 32, 8, 31, 0, true⟩,
    ⟨"Q6_K", 210, 256, 16, 8, 63, -32, false⟩,
    ⟨"Q8_0", 34, 32, 32, 8, 255, -128, false⟩,
    ⟨"IQ4_NL", 18, 32, 32, 4, 15, 0, false⟩,
    ⟨"IQ4_XS", 136, 256, 32, 4, 15, 0, false⟩ ]

/-- Groups tile blocks exactly. -/
theorem mappings_groups_tile :
    mappings.all (fun m => m.blockElems % m.groupElems == 0) = true := by
  decide

/-- Every code fits its declared plane width. -/
theorem mappings_codes_fit :
    mappings.all (fun m => decide (m.maxCode < 2 ^ m.codeBits)) = true := by
  decide

/-- Only the two K-quants with mins carry an offset plane. -/
theorem mappings_offset_rows :
    (mappings.filter (fun m => m.hasOffset)).map (fun m => m.name) = ["Q4_K", "Q5_K"] := by
  decide

/-- Group counts per 256-element block. -/
theorem mappings_group_counts : 256 / 32 = 8 ∧ 256 / 16 = 16 ∧ 64 / 64 = 1 := by
  decide

end Strata.CanonicalForm
