/-!
# MODULE: canonical_quant_roundtrip
SOURCES: tools/canonical_xcheck.py
Bit-exact canonicalisation of GGUF quantisation blocks (Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0,
IQ4_NL, IQ4_XS and the float types) into the S2/S4/S8/P16/P32 canonical forms: one code per element, one
scale (and optionally one offset) per group, with the integer code bias applied BEFORE the single multiply.

## Source semantics
* Every decode is `(code + code_bias) * scale (+ offset)`; the bias is an INTEGER subtraction on the code,
  never a value-space offset (that would not be bit exact).
* Block geometry (bytes, elements, group size) is fixed per source type and the registry is consistent:
  group size divides block size; only the K-quants with a min carry a per-group offset.
* Sub-byte unpacking is a bijection between the packed byte and its codes (nibbles, 2-bit quads, the 5th
  bit of Q5_0, the `hmask` bit of Q3_K, the `qh` pairs of Q6_K).
* IQ4 uses a non-affine 16-entry codebook, strictly increasing.
* `--split I/N` partitions tensors by `index % N == I`.

## Form template identifiers
- MAPPINGS (registry)              → `mappings`
- Mapping.code_bias                → `Mapping.codeBias`
- Mapping.block_bytes / block_elems / group_elems → `blockBytes` / `blockElems` / `groupElems`
- KV_IQ4NL                         → `kvIq4nl`
- to_s2 / from_s2 (codes in separate plane) → `pack4` / `pack4_unpack`
-/

namespace Strata.CanonicalQuant

-- § mapping registry

/-- One row of the `MAPPINGS` registry (geometry, integer code bias, offset flag). -/
structure Mapping where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBias : Int
  hasOffset : Bool

/-- The registry of source types, transcribed from the `MAPPINGS` dict. -/
def mappings : List Mapping :=
  [⟨"Q2_0", 18, 64, 64, -1, false⟩, ⟨"Q3_K", 110, 256, 16, -4, false⟩, ⟨"Q4_0", 18, 32, 32, -8, false⟩,
   ⟨"Q4_K", 144, 256, 32, 0, true⟩, ⟨"Q5_0", 22, 32, 32, -16, false⟩, ⟨"Q5_K", 176, 256, 32, 0, true⟩,
   ⟨"Q6_K", 210, 256, 16, -32, false⟩, ⟨"Q8_0", 34, 32, 32, -128, false⟩, ⟨"IQ4_NL", 18, 32, 32, 0, false⟩,
   ⟨"IQ4_XS", 136, 256, 32, 0, false⟩, ⟨"F32", 4, 1, 1, 0, false⟩, ⟨"BF16", 2, 1, 1, 0, false⟩,
   ⟨"F16", 2, 1, 1, 0, false⟩]

/-- All 13 source types of the shard are mapped. -/
theorem mappings_length : mappings.length = 13 := by
  decide

/-- The group size always divides the block size, so groups tile a block exactly. -/
theorem group_divides_block : ∀ m ∈ mappings, m.blockElems % m.groupElems = 0 := by
  decide

/-- Only the two K-quants with a per-group min (Q4_K, Q5_K) carry an offset plane. -/
theorem offset_only_k_quants :
    ∀ m ∈ mappings, m.hasOffset = true → m.name = "Q4_K" ∨ m.name = "Q5_K" := by
  decide

/-- Block byte counts equal the sum of their fields (scales, mins, high bits, codes). -/
theorem block_byte_layouts :
    (2 + 64 / 4 = 18) ∧ (2 + 32 / 2 = 18) ∧ (2 + 32 = 34) ∧ (2 + 4 + 32 / 2 = 22) ∧
    (2 + 2 + 12 + 256 / 2 = 144) ∧ (256 / 8 + 256 / 4 + 12 + 2 = 110) ∧
    (2 + 2 + 12 + 256 / 8 + 256 / 2 = 176) ∧ (256 / 2 + 256 / 4 + 16 + 2 = 210) ∧
    (2 + 2 + 4 + 256 / 2 = 136) := by
  decide

-- § integer code bias ranges

/-- Q2_0: codes 0..3 decode to the integers -1..2 (the "- 1" is an offset in the codebook). -/
theorem q2_0_symbol_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Q4_0: codes 0..15 decode to -8..7. -/
theorem q4_0_symbol_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q5_0: a nibble plus the 5th bit is a 5-bit code and decodes to -16..15. -/
theorem q5_0_code_range (lo hb : Nat) (hl : lo < 16) (hh : hb < 2) :
    lo + 16 * hb < 32 ∧ -16 ≤ ((lo + 16 * hb : Nat) : Int) - 16 ∧
      ((lo + 16 * hb : Nat) : Int) - 16 ≤ 15 := by
  omega

/-- Q3_K: two bits plus the hmask bit stay in 0..7 and decode to -4..3. -/
theorem q3k_code_range (c2 m : Nat) (h2 : c2 < 4) (hm : m < 2) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) - 4 ∧ ((c2 + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

/-- Q6_K: a nibble plus two `qh` bits is a 6-bit code and decodes to -32..31. -/
theorem q6k_code_range (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) :
    nib + 16 * hi < 64 ∧ -32 ≤ ((nib + 16 * hi : Nat) : Int) - 32 ∧
      ((nib + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q8_0: storing the int8 code as `+128` keeps it unsigned, and the decode's `-128` undoes it exactly. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    0 ≤ s + 128 ∧ s + 128 ≤ 255 ∧ (s + 128) - 128 = s := by
  omega

-- § sub-byte packing

/-- Pack four 2-bit codes into a byte, LSB first (Q2_0's own convention). -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed Q2_0 byte fits in 8 bits. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4
  omega

/-- `codes[:, t::4] = (byte >> 2t) & 3` recovers each of the four codes. -/
theorem pack4_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
      pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold pack4
  omega

/-- Two nibbles per byte: low nibble first half of the block, high nibble second half. -/
theorem nibble_roundtrip (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi ∧ lo + 16 * hi < 256 := by
  omega

/-- The Q5_0 source reaches bit `j+16` of `qh` via `(qh >> (j+12)) & 0x10`; this is the same bit. -/
theorem q5_0_high_bit_shift (qh j : Nat) :
    (qh / 2 ^ (j + 12)) % 32 / 16 = (qh / 2 ^ (j + 16)) % 2 := by
  have h1 : 2 ^ (j + 16) = 2 ^ (j + 12) * 16 := by
    rw [show j + 16 = (j + 12) + 4 by omega, Nat.pow_add]
  have h2 : qh / 2 ^ (j + 16) = (qh / 2 ^ (j + 12)) / 16 := by
    rw [h1, Nat.div_div_eq_div_mul]
  rw [h2]
  generalize qh / 2 ^ (j + 12) = x
  omega

/-- A 6-bit scale from a low nibble and two high bits stays below 64. -/
theorem scale6_lt (lo t : Nat) : lo % 16 + 16 * (t % 4) < 64 := by
  omega

/-- `get_scale_min_k4` for `j >= 4`: low 4 bits of one byte plus the top 2 bits of another stay 6-bit. -/
theorem scale_min_k4_hi_lt (a b : Nat) (hb : b < 256) : a % 16 + 16 * (b / 64) < 64 := by
  omega

-- § index maps are injective (no element read twice, none skipped)

/-- Q3_K hmask bit `4*half + j` is distinct for every (half, j) and stays in a byte. -/
theorem q3k_mask_bit_injective (h1 h2 j1 j2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4) (b2 : j2 < 4)
    (h : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by
  omega

/-- Q3_K group index `8*half + 2*j + sub` enumerates 16 distinct groups. -/
theorem q3k_group_injective (h1 h2 j1 j2 s1 s2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4)
    (b2 : j2 < 4) (c1 : s1 < 2) (c2 : s2 < 2) (h : 8 * h1 + 2 * j1 + s1 = 8 * h2 + 2 * j2 + s2) :
    h1 = h2 ∧ j1 = j2 ∧ s1 = s2 := by
  omega

/-- Q6_K quads land at strides of 32: `128*half + 32*quad + l` is injective on its domain. -/
theorem q6k_quad_injective (h1 h2 q1 q2 l1 l2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : q1 < 4)
    (b2 : q2 < 4) (c1 : l1 < 32) (c2 : l2 < 32) (h : 128 * h1 + 32 * q1 + l1 = 128 * h2 + 32 * q2 + l2) :
    h1 = h2 ∧ q1 = q2 ∧ l1 = l2 := by
  omega

/-- Q5_K masks `u1 = 1 << 2j`, `u2 = 2 << 2j` use bit `2j+k`, all distinct across the four steps. -/
theorem q5k_mask_bit_injective (j1 j2 k1 k2 : Nat) (a1 : j1 < 4) (a2 : j2 < 4) (b1 : k1 < 2) (b2 : k2 < 2)
    (h : 2 * j1 + k1 = 2 * j2 + k2) : j1 = j2 ∧ k1 = k2 := by
  omega

/-- Q4_K: four 64-element steps of two groups give 8 groups of 32 = 256 elements. -/
theorem q4k_group_count : 4 * 2 = 8 ∧ 8 * 32 = 256 := by
  decide

/-- Q2_0 canonical codes plane states its own element count: `codes.size` determines the block count. -/
theorem q2_0_codes_size (nb : Nat) : nb * 64 / 64 = nb := by
  omega

-- § the IQ4 codebook

/-- `kvalues_iq4nl`. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries (a 4-bit index). -/
theorem kv_length : kvIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kv_strictly_increasing : kvIq4nl.Pairwise (· < ·) := by
  decide

/-- It is NOT affine: consecutive differences differ, so it cannot be expressed as a code bias. -/
theorem kv_not_affine :
    (kvIq4nl.getD 1 0 - kvIq4nl.getD 0 0) ≠ (kvIq4nl.getD 2 0 - kvIq4nl.getD 1 0) := by
  decide

-- § work splitting and code widths

/-- `--split I/N`: each tensor index lands in exactly one process slot. -/
theorem split_partition (idx sn si si' : Nat) (hs : 0 < sn) (h1 : idx % sn = si) (h2 : idx % sn = si') :
    idx % sn < sn ∧ si = si' := by
  have hlt : idx % sn < sn := Nat.mod_lt idx hs
  refine ⟨hlt, ?_⟩
  omega

/-- Codes per byte times bits per code is 8 for every canonical form width. -/
theorem code_bits_fill_byte : ∀ b ∈ [2, 4, 8], b * (8 / b) = 8 := by
  decide

end Strata.CanonicalQuant
