/-!
# MODULE: canonical_form_codebook_decode
Integer-level contracts of `tools/canonical_xcheck.py`: each GGML quant type is re-expressed as an unsigned
code plane + scale (+ offset) plane, and the decode is `(code + bias) * scale (+ offset)`.  The bias lives in
the INTEGER domain, so the value set of each type is a fixed finite integer range.

## Source semantics
* Q2_0 value = (code - 1) * d, code in 0..3 (symbols -1,0,1,2); Q3_K bias -4; Q4_0 bias -8; Q5_0 -16;
  Q6_K -32; Q8_0 stored as int8 + 128 and decoded with -128.
* IQ4 types use a 16-entry non-affine table (no bias).
* Nibble split: low nibbles fill the first half of a block, high nibbles the second.
* `get_scale_min_k4` yields two 6-bit values per group from 12 packed bytes.
* Q3_K / Q5_K / Q6_K element order is fixed by loop nests whose index maps must be exact bijections.

## Form template identifiers
code_bias            -> code_bias (registry field)
block_bytes          -> block_bytes
block_elems          -> block_elems
group_elems          -> group_elems
codeValue            -> decode of one code, before the scale multiply
scaleMin             -> get_scale_min_k4
-/

namespace Strata.CanonicalFormCodebookDecode

-- § affine codebooks

/-- Integer value of a stored code under a registry `code_bias` (added to the code). -/
def codeValue (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Q2_0 codebook is {-1,0,1,2}. -/
theorem q2_0_range (c : Nat) (h : c < 4) : -1 ≤ codeValue c (-1) ∧ codeValue c (-1) ≤ 2 := by
  unfold codeValue; omega

/-- Q3_K codebook is -4..3 (codes 0..7). -/
theorem q3_k_range (c : Nat) (h : c < 8) : -4 ≤ codeValue c (-4) ∧ codeValue c (-4) ≤ 3 := by
  unfold codeValue; omega

/-- Q4_0 codebook is -8..7 (codes 0..15). -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ codeValue c (-8) ∧ codeValue c (-8) ≤ 7 := by
  unfold codeValue; omega

/-- Q5_0 codebook is -16..15 (codes 0..31). -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ codeValue c (-16) ∧ codeValue c (-16) ≤ 15 := by
  unfold codeValue; omega

/-- Q6_K codebook is -32..31 (codes 0..63). -/
theorem q6_k_range (c : Nat) (h : c < 64) : -32 ≤ codeValue c (-32) ∧ codeValue c (-32) ≤ 31 := by
  unfold codeValue; omega

/-- Q8_0 codebook is -128..127 (codes 0..255). -/
theorem q8_0_range (c : Nat) (h : c < 256) :
    -128 ≤ codeValue c (-128) ∧ codeValue c (-128) ≤ 127 := by
  unfold codeValue; omega

/-- A code plane is a faithful representation: distinct codes decode to distinct integers. -/
theorem codeValue_injective (a b : Nat) (bias : Int) (h : codeValue a bias = codeValue b bias) :
    a = b := by
  unfold codeValue at h; omega

/-- Storing an `int8` quant as `q + 128` and decoding with bias `-128` returns `q` exactly. -/
theorem q8_0_store_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    codeValue (q + 128).toNat (-128) = q := by
  unfold codeValue; omega

/-- The stored Q8_0 code is a valid unsigned byte. -/
theorem q8_0_store_lt (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) : (q + 128).toNat < 256 := by
  omega

-- § source block geometry

/-- Byte size of each source block, computed from its fields. -/
theorem block_sizes :
    2 + 64 / 4 = 18 ∧ 2 + 32 / 2 = 18 ∧ 2 + 32 = 34 ∧ 2 + 4 + 32 / 2 = 22 ∧
    4 + 12 + 128 = 144 ∧ 4 + 12 + 32 + 128 = 176 ∧ 128 + 64 + 16 + 2 = 210 ∧
    32 + 64 + 12 + 2 = 110 ∧ 2 + 2 + 4 + 128 = 136 := by
  decide

/-- Registry row: (block_bytes, block_elems, group_elems). -/
structure Geom where
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat

/-- The 13-type mapping registry's geometry. -/
def registry : List Geom :=
  [⟨18, 64, 64⟩, ⟨110, 256, 16⟩, ⟨18, 32, 32⟩, ⟨144, 256, 32⟩, ⟨22, 32, 32⟩, ⟨176, 256, 32⟩,
   ⟨210, 256, 16⟩, ⟨34, 32, 32⟩, ⟨18, 32, 32⟩, ⟨136, 256, 32⟩, ⟨4, 1, 1⟩, ⟨2, 1, 1⟩, ⟨2, 1, 1⟩]

/-- Every group lies inside one block: `group_elems` divides `block_elems`. -/
theorem group_divides_block : ∀ g ∈ registry, g.blockElems % g.groupElems = 0 := by
  decide

/-- The registry has 13 types. -/
theorem registry_size : registry.length = 13 := by decide

-- § nibble handling

/-- Low and high nibble of a packed byte recover the two codes. -/
theorem nibble_roundtrip (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by
  omega

/-- A packed byte of two nibbles is a byte. -/
theorem nibble_pack_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by
  omega

-- § get_scale_min_k4

/-- `get_scale_min_k4` over a byte-valued array: (scale, min) of group `j`. -/
def scaleMin (s : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (s j % 64, s (j + 4) % 64)
  else (s (j + 4) % 16 + (s (j - 4) / 64) * 16, s (j + 4) / 16 + (s j / 64) * 16)

/-- Both outputs are 6-bit values whenever the scale bytes are bytes. -/
theorem scaleMin_lt (s : Nat → Nat) (hs : ∀ i, s i < 256) (j : Nat) :
    (scaleMin s j).1 < 64 ∧ (scaleMin s j).2 < 64 := by
  have h1 := hs j
  have h2 := hs (j + 4)
  have h3 := hs (j - 4)
  unfold scaleMin
  split <;> (dsimp only; omega)

/-- Groups 0..3 read only the low 6 bits of their own and the paired byte. -/
theorem scaleMin_low (s : Nat → Nat) (j : Nat) (hj : j < 4) :
    scaleMin s j = (s j % 64, s (j + 4) % 64) := by
  unfold scaleMin; simp [hj]

-- § Q4_K / Q5_K group layout

/-- Canonical group `g = 2j + h` covers elements `32 g .. 32 g + 31`: the `is+h` pair of the source loop. -/
theorem q4k_group_base (j h : Nat) (hh : h < 2) : j * 64 + h * 32 = 32 * (2 * j + h) := by
  omega

/-- The eight Q5_K high-bit masks `u1, u2` over the four 64-step iterations are exactly the eight bits. -/
theorem q5k_masks :
    (List.range 4).flatMap (fun j => [4 ^ j, 2 * 4 ^ j]) = [1, 2, 4, 8, 16, 32, 64, 128] := by
  decide

-- § Q3_K index maps

/-- Q3_K group index `8*half + 2*j + sub` enumerates 0..15 in order. -/
theorem q3k_group_enum :
    (List.range 2).flatMap (fun h => (List.range 4).flatMap (fun j =>
      (List.range 2).map (fun s => 8 * h + 2 * j + s))) = List.range 16 := by
  decide

/-- The Q3_K hmask bit `4*half + j` (the mask never resets) enumerates 0..7. -/
theorem q3k_mask_bit_enum :
    (List.range 2).flatMap (fun h => (List.range 4).map (fun j => 4 * h + j)) = List.range 8 := by
  decide

/-- The two halves use disjoint hmask bit ranges. -/
theorem q3k_halves_disjoint (j j' : Nat) (hj : j < 4) (hj' : j' < 4) : 4 * 0 + j ≠ 4 * 1 + j' := by
  omega

-- § Q6_K index map

/-- Q6_K element order `128*half + 32*q + l` enumerates 0..255 in order. -/
theorem q6k_element_enum :
    (List.range 2).flatMap (fun h => (List.range 4).flatMap (fun q =>
      (List.range 32).map (fun l => 128 * h + 32 * q + l))) = List.range 256 := by
  decide +kernel

/-- The source scale index `is + 2q` (with `is = l/16`) is the plain 16-element group of the element. -/
theorem q6k_scale_index (q l : Nat) (hl : l < 32) : (q * 32 + l) / 16 = 2 * q + l / 16 := by
  omega

-- § IQ4_XS scale

/-- The 6-bit `ls` of IQ4_XS: a nibble of `scales_l` plus two high bits from `scales_h`. -/
def ls6 (slByte par hi2 : Nat) : Nat :=
  (if par = 0 then slByte % 16 else slByte / 16) + hi2 * 16

/-- `ls` fits in six bits. -/
theorem ls6_lt (slByte par hi2 : Nat) (hb : slByte < 256) (hh : hi2 < 4) : ls6 slByte par hi2 < 64 := by
  unfold ls6
  split <;> omega

end Strata.CanonicalFormCodebookDecode
