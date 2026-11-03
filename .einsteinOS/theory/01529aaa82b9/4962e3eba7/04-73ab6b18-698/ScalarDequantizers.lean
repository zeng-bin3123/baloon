/-!
# MODULE: scalar_dequantizers
Scalar reference dequantizers (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS): block
geometry, bit-field addressing and value ranges, modelled over Nat/Int.

## Source semantics
Q2_0: 64 elements per 18-byte block, code c in {0..3} gives symbol c-1, four codes per byte LSB-first.
Q4_0/IQ4_NL: low nibble is element j, high nibble element j+16. Q5_0 adds one high bit per element.
K-quants use 256-element super-blocks; Q4_K/Q5_K use d,dmin and 6-bit scale/min, Q3_K and IQ4_XS
use signed 6-bit scales with a -32 bias. Q5_K selects its fifth bit with a mask shifting left by 2
per 64-element group.

## Form template identifiers
q2Code -> unpack_code ; q2Pack -> pack_codes ; kv -> codebook ; blk_* -> block_size_bytes
-/

set_option maxRecDepth 100000

namespace Strata.Dequant

-- § Q2_0

/-- Code extraction for element k (0..3) of a packed byte. -/
def q2Code (b k : Nat) : Nat := (b >>> (k * 2)) &&& 3

/-- Packing four 2-bit codes into a byte, LSB-first. -/
def q2Pack (c0 c1 c2 c3 : Nat) : Nat := c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6)

/-- Unpacking recovers each of the four packed codes: no bit-order ambiguity. -/
theorem q2_roundtrip : ∀ c0 < 4, ∀ c1 < 4, ∀ c2 < 4, ∀ c3 < 4,
    q2Code (q2Pack c0 c1 c2 c3) 0 = c0 ∧ q2Code (q2Pack c0 c1 c2 c3) 1 = c1 ∧
    q2Code (q2Pack c0 c1 c2 c3) 2 = c2 ∧ q2Code (q2Pack c0 c1 c2 c3) 3 = c3 := by
  decide +kernel

/-- Every extracted code is a 2-bit value. -/
theorem q2_code_lt4 : ∀ b < 256, ∀ k < 4, q2Code b k < 4 := by decide

/-- The symbol of a code: code - 1. -/
def q2Sym (c : Nat) : Int := (c : Int) - 1

/-- Symbols lie in {-1,0,1,2}. -/
theorem q2_sym_range (c : Nat) (h : c < 4) : -1 ≤ q2Sym c ∧ q2Sym c ≤ 2 := by
  unfold q2Sym; omega

/-- Byte index and bit offset stay inside the 16-byte payload. -/
theorem q2_index (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by omega

/-- (byte, slot) addressing is a bijection onto element index. -/
theorem q2_cover (j : Nat) : (j / 4) * 4 + j % 4 = j := by omega

/-- Block layout: 2 byte scale + 16 payload bytes = 18; 16 bytes * 4 codes = 64 elements. -/
theorem q2_block : 2 + 16 = 18 ∧ 16 * 4 = 64 := ⟨rfl, rfl⟩

-- § Q4_0 / Q5_0 / Q8_0

/-- Every element 0..31 is the low nibble of byte j (i = j) or the high nibble (i = j + 16). -/
theorem q4_cover (i : Nat) (hi : i < 32) : ∃ j, j < 16 ∧ (i = j ∨ i = j + 16) :=
  ⟨i % 16, by omega, by omega⟩

/-- Q4_0 values after the -8 bias lie in [-8,7]. -/
theorem q4_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7 := by omega

/-- Q5_0 layout: scale 2, qh 4, qs 16 bytes. -/
theorem q5_0_block : 2 + 4 + 16 = 22 := rfl

/-- Q5_0 five-bit values after the -16 bias lie in [-16,15]. -/
theorem q5_0_range (nib b : Nat) (hn : nib < 16) (hb : b < 2) :
    -16 ≤ ((nib + 16 * b : Nat) : Int) - 16 ∧ ((nib + 16 * b : Nat) : Int) - 16 ≤ 15 := by omega

/-- Q5_0 high bits come from qh bit j (low half) and bit j+16 (high half): all 32 distinct. -/
theorem q5_0_bits (j : Nat) (hj : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j ≠ j + 16 := by omega

/-- Q8_0 layout: scale 2 + 32 int8. -/
theorem q8_0_block : 2 + 32 = 34 := rfl

-- § IQ4_NL codebook

/-- `kvalues_iq4nl`, verbatim. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries: one per nibble. -/
theorem kv_len : kv.length = 16 := rfl

/-- Strictly increasing, so the codebook is a monotone non-linear grid. -/
theorem kv_sorted : kv.Pairwise (· < ·) := by decide

/-- All entries lie in [-127,113]; no entry is offset by -8. -/
theorem kv_bounds : ∀ x ∈ kv, -127 ≤ x ∧ x ≤ 113 := by decide

/-- IQ4_NL block: 2 + 16 = 18 bytes for 32 elements. -/
theorem iq4nl_block : 2 + 16 = 18 := rfl

-- § K-quant layouts

/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2 = 210, with d at offset 208. -/
theorem q6k_block : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 := ⟨rfl, rfl⟩

/-- Q6_K scale index is_ + {0,2,4,6} plus 8 per half never exceeds 15. -/
theorem q6k_scale_idx (h l q : Nat) (hh : h < 2) (hl : l < 32) (hq : q < 4) :
    8 * h + l / 16 + 2 * q < 16 := by omega

/-- Q6_K output positions of one half cover 128 slots. -/
theorem q6k_out_idx (h l q : Nat) (hh : h < 2) (hl : l < 32) (hq : q < 4) :
    128 * h + l + 32 * q < 256 := by omega

/-- Q4_K: d, dmin, scales 12, qs 128 = 144, qs at offset 16. -/
theorem q4k_block : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := ⟨rfl, rfl⟩

/-- Q5_K: d, dmin, scales 12, qh 32, qs 128 = 176, qh at 16, qs at 48. -/
theorem q5k_block : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := ⟨rfl, rfl⟩

/-- High 2 bits of a byte fit in 0..3. -/
theorem shr6_lt4 : ∀ b < 256, b >>> 6 < 4 := by decide

/-- The reassembled 6-bit scale (low nibble + 2 high bits shifted 4) is below 64. -/
theorem scale6_lt64 : ∀ x < 256, ∀ y < 4, ((x &&& 15) ||| (y <<< 4)) < 64 := by decide +kernel

/-- Q5_K masks u1 = 1,4,16,64. -/
theorem q5k_u1 : ∀ k < 4, (1 <<< (2 * k)) = [1, 4, 16, 64].getD k 0 := by decide

/-- u2 is always twice u1, so the two masks of a group select adjacent bits. -/
theorem q5k_u2 : ∀ k < 4, (2 <<< (2 * k)) = 2 * (1 <<< (2 * k)) := by decide

/-- The eight masks are exactly the eight bits of a qh byte: sum 255. -/
theorem q5k_masks_cover : (1 + 4 + 16 + 64) + (2 + 8 + 32 + 128) = 255 := rfl

/-- A fixed mask or a shift by one would produce different masks at group 1. -/
theorem q5k_not_shift1 : (1 <<< (2 * 1)) ≠ (1 <<< 1) := by decide

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2 = 110, d at offset 108. -/
theorem q3k_block : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 + 12 = 108 := ⟨rfl, rfl⟩

/-- Q3_K hoisted mask: at (half n, step j) the mask is 1 << (4n+j), below 256. -/
theorem q3k_mask_lt (n j : Nat) (hn : n < 2) (hj : j < 4) : (1 <<< (4 * n + j)) < 256 := by
  have : 4 * n + j < 8 := by omega
  have h8 : ∀ k < 8, (1 <<< k) < 256 := by decide
  exact h8 _ this

/-- Resetting the mask in the second half would give a different (wrong) mask. -/
theorem q3k_mask_hoist : (1 <<< (4 * 1 + 0)) ≠ (1 <<< 0) := by decide

/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128 = 136; eight groups of 16 bytes. -/
theorem iq4xs_block : 2 + 2 + 4 + 128 = 136 ∧ 8 * 16 = 128 := ⟨rfl, rfl⟩

/-- scales_l index and shifts stay in range; the two high bits stay inside the 16-bit field. -/
theorem iq4xs_index (ib : Nat) (h : ib < 8) :
    ib / 2 < 4 ∧ 4 * (ib % 2) ≤ 4 ∧ 2 * ib + 2 ≤ 16 := by omega

/-- The 6-bit group scale ls lies in 0..63, so ls - 32 lies in [-32,31]. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § fp16 exponent mapping

/-- Normal fp16 exponent maps to f32 exponent ex + 112, always a normal f32 exponent. -/
theorem fp16_exp_map (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    (ex : Int) - 15 + 127 = (ex : Int) + 112 ∧ 1 ≤ ex + 112 ∧ ex + 112 ≤ 254 := by omega

/-- The mantissa shift of 13 stays below 2^23. -/
theorem fp16_man_shift (man : Nat) (h : man < 1024) : man * 8192 < 8388608 := by omega

end Strata.Dequant
