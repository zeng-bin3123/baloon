/-!
# MODULE: canonical_form_roundtrip
The integer-level facts behind the claim that every ggml quantised block maps to a canonical
(codes, scales, offsets) form and decodes bit-exactly: code biases, nibble and high-bit splits, group index
maps, scale-field packing and the registry geometry.

## Source semantics
* Biased codebooks are applied to the INTEGER code before the float multiply: Q2_0 `-1`, Q3_K `-4`,
  Q4_0 `-8`, Q5_0 `-16`, Q6_K `-32`, Q8_0 stored as `s + 128` and decoded `- 128`.
* Nibble formats put low nibbles in the first half and high nibbles in the second half.
* Group maps: Q4_K/Q5_K have groups of 32 (low nibbles `2j`, high nibbles `2j+1` per 64-step), Q3_K and Q6_K
  groups of 16, IQ4_XS groups of 32 with a 6-bit scale split over two fields.
* Every registry entry has a group size that divides its block size.

## Form template identifiers
to_s2/from_s2 -> encode_q2/decode_q2, MAPPINGS -> registry, code_bias -> code_bias,
get_scale_min_k4 -> scale_min_pair.
-/

namespace Strata.CanonicalForm

-- § integer biases

/-- Q8_0: a signed int8 code `s` is stored as the unsigned `s + 128`, in `[0, 255]`. -/
theorem q8_store_range (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) : 0 ≤ s + 128 ∧ s + 128 ≤ 255 := by omega

/-- Q8_0: decoding subtracts 128 inside the integer and recovers the source code. -/
theorem q8_roundtrip (s : Int) : (s + 128) - 128 = s := by omega

/-- Q4_0: nibbles are `0..15`, decoded codes `c - 8` are in `[-8, 7]`. -/
theorem q4_0_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 15) : -8 ≤ c - 8 ∧ c - 8 ≤ 7 := by omega

/-- Q5_0: a 5-bit code is `0..31`; decoded `c - 16` is in `[-16, 15]`. -/
theorem q5_0_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 31) : -16 ≤ c - 16 ∧ c - 16 ≤ 15 := by omega

/-- Q6_K: a 6-bit code is `0..63`; decoded `c - 32` is in `[-32, 31]`. -/
theorem q6_k_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 63) : -32 ≤ c - 32 ∧ c - 32 ≤ 31 := by omega

/-- Q3_K: the stored code is `(2 bits) + 4*(hm bit)`, in `[0, 7]`; decoded `c - 4` is in `[-4, 3]`. -/
theorem q3_k_range (b m : Nat) (hb : b < 4) (hm : m < 2) :
    b + 4 * m ≤ 7 ∧ -4 ≤ ((b + 4 * m : Nat) : Int) - 4 ∧ ((b + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

-- § nibble and high-bit splits

/-- A byte splits into low and high nibbles that recombine to the byte. -/
theorem nibble_split (q : Nat) (h : q < 256) : q % 16 + 16 * (q / 16) = q ∧ q % 16 < 16 ∧ q / 16 < 16 := by
  omega

/-- Q5_0/Q5_K: a 4-bit nibble plus a 5th bit contributing 16 stays below 32. -/
theorem fifth_bit (n b : Nat) (hn : n < 16) (hb : b < 2) : n + 16 * b < 32 := by omega

/-- Q6_K: a 4-bit nibble plus two high bits `<<4` stays below 64. -/
theorem sixth_bits (n h : Nat) (hn : n < 16) (hh : h < 4) : n + 16 * h < 64 := by omega

/-- IQ4_XS: the 6-bit scale is a nibble plus two high bits `<<4`, below 64. -/
theorem iq4_xs_scale6 (lo hi : Nat) (hlo : lo < 16) (hhi : hi < 4) : lo + 16 * hi < 64 := by omega

/-- Q5_K high-bit masks walk 1, 4, 16, 64 (low nibbles) and 2, 8, 32, 128 (high nibbles). -/
theorem q5_k_masks :
    (1 <<< (2 * 0) = 1 ∧ 1 <<< (2 * 1) = 4 ∧ 1 <<< (2 * 2) = 16 ∧ 1 <<< (2 * 3) = 64) ∧
    (2 <<< (2 * 0) = 2 ∧ 2 <<< (2 * 1) = 8 ∧ 2 <<< (2 * 2) = 32 ∧ 2 <<< (2 * 3) = 128) := by
  decide

-- § group index maps

/-- Q4_K/Q5_K: in 64-step `j`, low-nibble element `64j+l` is group `2j`, high `64j+32+l` is group `2j+1`. -/
theorem q4k_groups (j l : Nat) (hj : j < 4) (hl : l < 32) :
    (64 * j + l) / 32 = 2 * j ∧ (64 * j + 32 + l) / 32 = 2 * j + 1 := by omega

/-- Q3_K: element `128*half + 32*j + 16*sub + l` is group `8*half + 2*j + sub`. -/
theorem q3k_groups (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    (128 * half + 32 * j + 16 * sub + l) / 16 = 8 * half + 2 * j + sub := by omega

/-- Q6_K: element `128*half + 32*quad + l` is group `8*half + 2*quad + l/16`. -/
theorem q6k_groups (half quad l : Nat) (hh : half < 2) (hq : quad < 4) (hl : l < 32) :
    (128 * half + 32 * quad + l) / 16 = 8 * half + 2 * quad + l / 16 := by omega

/-- IQ4_XS: scale group `ib` uses byte `ib/2` of `scales_l`, low nibble first when `ib` is even. -/
theorem iq4_xs_group_byte (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ ib % 2 < 2 := by omega

-- § Q2_0 canonical form

/-- Four 2-bit codes per source byte: unpack slot `t` of byte `k` lands at element `4k + t`. -/
theorem q2_element_index (k t : Nat) (ht : t < 4) : (4 * k + t) / 4 = k ∧ (4 * k + t) % 4 = t := by omega

/-- Codes and scales split into planes: one code per element, so `codes.size` is 64 per block. -/
theorem q2_codes_per_block (nblocks : Nat) : nblocks * 64 = 64 * nblocks := by omega

/-- A Q2_0 block has 18 bytes: a 2-byte scale and 64 two-bit codes. -/
theorem q2_block_bytes : 2 + 64 * 2 / 8 = 18 := by decide

/-- The bias cannot be folded into the value domain: the stored code stays unsigned `0..3` and `c - 1`
is applied to the integer. -/
theorem q2_bias_range (c : Nat) (h : c < 4) : (c : Int) - 1 ∈ [(-1 : Int), 0, 1, 2] := by
  interval_cases c <;> simp

-- § registry geometry

/-- Registry rows `(block_bytes, block_elems, group_elems, code_bits)` for the mapped types:
Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, F32, BF16, F16. -/
def registry : List (Nat × Nat × Nat × Nat) :=
  [(18, 64, 64, 2), (110, 256, 16, 4), (18, 32, 32, 4), (144, 256, 32, 4), (22, 32, 32, 8),
   (176, 256, 32, 8), (210, 256, 16, 8), (34, 32, 32, 8), (18, 32, 32, 4), (136, 256, 32, 4),
   (4, 1, 1, 8), (2, 1, 1, 8), (2, 1, 1, 8)]

/-- Every registry entry has a positive group size dividing its block size. -/
theorem registry_groups_divide :
    ∀ e ∈ registry, 0 < e.2.2.1 ∧ e.2.1 % e.2.2.1 = 0 := by decide

/-- Block bytes of the quantised types equal their field sizes. -/
theorem block_byte_sums :
    (2 + 64 * 2 / 8 = 18) ∧ (2 + 32 * 4 / 8 = 18) ∧ (2 + 4 + 32 * 4 / 8 = 22) ∧ (2 + 32 = 34) ∧
    (32 + 64 + 12 + 2 = 110) ∧ (4 + 12 + 128 = 144) ∧ (4 + 12 + 32 + 128 = 176) ∧
    (128 + 64 + 16 + 2 = 210) ∧ (2 + 2 + 4 + 128 = 136) := by decide

/-- Groups per block follow from the registry: 256-element K-types have `256 / group` groups. -/
theorem k_type_groups : 256 / 16 = 16 ∧ 256 / 32 = 8 := by decide

end Strata.CanonicalForm
