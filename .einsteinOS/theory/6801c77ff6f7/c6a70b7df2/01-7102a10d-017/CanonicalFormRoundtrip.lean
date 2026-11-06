/-!
# MODULE: canonical_form_roundtrip
SOURCES: tools/canonical_xcheck.py
Prototype of the P1.T3 test: every GGUF quant type is mapped to a canonical plane form (codes, scales,
optional offsets) and must decode bit-exactly. Modelled here as the integer index/bias bookkeeping that makes
the mapping lossless.

## Source semantics
* Codes are unsigned; the source bias (-1, -4, -8, -16, -32, -128) is an INTEGER subtraction before the multiply.
* Nibble types place low nibbles in the first half of a group and high nibbles in the second.
* Q3_K/Q6_K/Q4_K/Q5_K element orders are fixed index formulas; the registry lists block bytes, elements, group size.

## Form template identifiers
to_s2/from_s2 -> split_planes_2bit ; to_s4_q4_k -> canon_q4k ; to_s4_q3_k -> canon_q3k ;
to_s8_q6_K -> canon_q6k ; to_s8_q8_0 -> canon_q8_0 ; MAPPINGS -> canonical_registry
-/
set_option maxRecDepth 100000

namespace Strata.Canonical

-- § integer-bias codebooks
/-- Q8_0: storing the int8 code +128 yields an unsigned code and decoding subtracts 128 exactly. -/
theorem q8_0_bias_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    0 ≤ q + 128 ∧ q + 128 ≤ 255 ∧ (q + 128) - 128 = q := by
  omega

/-- Q4_0: unsigned 4-bit code minus 8 spans exactly [-8, 7]. -/
theorem q4_0_bias_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q3_K: the stored code (2 bits + 4*hbit) lies in 0..7. -/
theorem q3k_code_range (c2 hb : Nat) (h2 : c2 < 4) (hh : hb < 2) : c2 + 4 * hb < 8 := by
  omega

/-- Q3_K: subtracting the integer bias 4 reproduces ggml's "- (hmask bit ? 0 : 4)". -/
theorem q3k_bias_matches_ggml (c2 hb : Nat) (h2 : c2 < 4) (hh : hb < 2) :
    ((c2 + 4 * hb : Nat) : Int) - 4 = (c2 : Int) - (if hb = 1 then 0 else 4) := by
  split <;> omega

/-- Q5_0 and Q6_K bias ranges: 5-bit minus 16 and 6-bit minus 32. -/
theorem q5_q6_bias_range (c d : Nat) (hc : c < 32) (hd : d < 64) :
    -16 ≤ (c : Int) - 16 ∧ (c : Int) - 16 ≤ 15 ∧ -32 ≤ (d : Int) - 32 ∧ (d : Int) - 32 ≤ 31 := by
  omega

/-- Q2_0: unsigned 2-bit code minus 1 spans [-1, 2]. -/
theorem q2_0_bias_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

-- § Q2_0 -> S2 plane split
/-- Element j = 4b+t lives in byte b at bit position 2t, so j/4 = b and j%4 = t. -/
theorem s2_plane_index (b t : Nat) (ht : t < 4) : (4 * b + t) / 4 = b ∧ (4 * b + t) % 4 = t := by
  omega

/-- Each 18-byte Q2_0 block yields 64 one-code-per-element entries and one scale. -/
theorem s2_block_counts : (18 - 2) * 4 = 64 := by decide

-- § Q4_K / Q5_K element order
/-- Element index 64j + 32h + l (j<4, h<2, l<32) is a bijection onto [0,256). -/
theorem k45_index_bijective (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    64 * j + 32 * h + l < 256 ∧ (64 * j + 32 * h + l) / 64 = j ∧
    (64 * j + 32 * h + l) / 32 % 2 = h ∧ (64 * j + 32 * h + l) % 32 = l := by
  omega

/-- Every element index below 256 decomposes into (j, h, l). -/
theorem k45_index_surjective (i : Nat) (hi : i < 256) :
    64 * (i / 64) + 32 * (i / 32 % 2) + i % 32 = i := by
  omega

/-- Group index of element 64j+32h+l is 2j+h, i.e. 8 groups of 32. -/
theorem k45_group_index (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    (64 * j + 32 * h + l) / 32 = 2 * j + h ∧ 2 * j + h < 8 := by
  omega

/-- The Q5_K mask bit 2j+h is unique and below 8 for each (j,h). -/
theorem q5k_mask_bits_unique (j h j' h' : Nat) (hj : j < 4) (hh : h < 2) (hj' : j' < 4) (hh' : h' < 2)
    (heq : 2 * j + h = 2 * j' + h') : j = j' ∧ h = h' := by
  omega

/-- Q5_K code = nibble + 16*bit stays below 32. -/
theorem q5k_code_range (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : nib + 16 * bit < 32 := by
  omega

-- § Q3_K element order
/-- Q3_K element order 128*half + 32*j + 16*sub + l equals 16*g + l with g = 8*half + 2*j + sub < 16. -/
theorem q3k_group_order (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    128 * half + 32 * j + 16 * sub + l = 16 * (8 * half + 2 * j + sub) + l ∧
    8 * half + 2 * j + sub < 16 := by
  omega

/-- The qs source offset half*32 + sub*16 + l stays inside the 64 qs bytes. -/
theorem q3k_qs_index (half sub l : Nat) (hh : half < 2) (hs : sub < 2) (hl : l < 16) :
    half * 32 + sub * 16 + l < 64 := by
  omega

/-- The hmask index sub*16 + l is the same for both halves and stays inside 32 bytes. -/
theorem q3k_hmask_index (sub l : Nat) (hs : sub < 2) (hl : l < 16) : sub * 16 + l < 32 := by
  omega

-- § Q6_K element order
/-- Q6_K element 128h + 32q + l belongs to scale group 8h + 2q + l/16, matching ggml's sc index. -/
theorem q6k_group_of_element (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    (128 * h + 32 * q + l) / 16 = 8 * h + 2 * q + l / 16 ∧ 8 * h + 2 * q + l / 16 < 16 := by
  omega

/-- Q6_K element positions 128h + 32q + l cover [0,256) exactly once. -/
theorem q6k_index_bijective (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    128 * h + 32 * q + l < 256 ∧ (128 * h + 32 * q + l) / 128 = h ∧
    (128 * h + 32 * q + l) / 32 % 4 = q ∧ (128 * h + 32 * q + l) % 32 = l := by
  omega

-- § registry
/-- Registry rows: (block_bytes, block_elems, group_elems, max_code, form_levels). -/
def registry : List (Nat × Nat × Nat × Nat × Nat) :=
  [(18, 64, 64, 3, 4),      -- Q2_0 S2
   (110, 256, 16, 7, 16),   -- Q3_K S4
   (18, 32, 32, 15, 16),    -- Q4_0 S4
   (144, 256, 32, 15, 16),  -- Q4_K S4
   (22, 32, 32, 31, 256),   -- Q5_0 S8
   (176, 256, 32, 31, 256), -- Q5_K S8
   (210, 256, 16, 63, 256), -- Q6_K S8
   (34, 32, 32, 255, 256),  -- Q8_0 S8
   (18, 32, 32, 15, 16),    -- IQ4_NL S4
   (136, 256, 32, 15, 16)]  -- IQ4_XS S4

/-- Every registry group size divides its block size. -/
theorem registry_group_divides :
    registry.all (fun r => r.2.1 % r.2.2.1 == 0) = true := by decide

/-- Every maximum code fits the code plane of its canonical form. -/
theorem registry_codes_fit :
    registry.all (fun r => decide (r.2.2.2.1 < r.2.2.2.2)) = true := by decide

/-- There are ten quantized mappings in the registry. -/
theorem registry_size : registry.length = 10 := by decide

/-- Q4_K scales/offsets are (n,8): 256 elements in groups of 32. -/
theorem q4k_groups_per_block : 256 / 32 = 8 := by decide

end Strata.Canonical
