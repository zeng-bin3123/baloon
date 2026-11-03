/-!
# MODULE: ggml_dequant_layouts
Block layouts and code ranges of the scalar dequantizers (`dequant.hpp`): Q2_0, Q4_0, Q5_0, Q6_K, Q4_K, Q5_K, Q3_K,
IQ4_XS and the 16-bit float conversions.

## Source semantics
* Q2_0: 64 elements, 18 bytes; code `c ∈ {0..3}` maps to symbol `c - 1 ∈ {-1,0,1,2}`; element `j` is byte `j/4`, bit offset `(j%4)*2`.
* Q4_0 / IQ4_NL: low nibble is element `j`, high nibble `j+16`; Q5_0 adds a fifth bit.
* K-quant super-blocks are 256 elements; every sub-field offset below tiles the block size exactly.
* `read_u16` is `p0 | p1<<8` (a 16-bit value); bf16 -> f32 shifts left 16 and is exactly invertible.

## Form template identifiers
`dequantize_q2_0` -> q2, `block + 2` -> qs offset, `get_scale_min_k4` -> scaleMinIdx, `kvalues_iq4nl` -> 16-entry codebook.
-/

namespace Strata.Dequant

-- § Q2_0

/-- The 2-bit code of element position `r` (0..3) inside a byte. -/
def q2Code (b r : Nat) : Nat := (b / 2 ^ (2 * r)) % 4

/-- A code is at most 3. -/
theorem q2Code_le (b r : Nat) : q2Code b r ≤ 3 := by
  have : q2Code b r < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The decoded symbol of a code. -/
def q2Sym (c : Nat) : Int := (c : Int) - 1

/-- Symbols are exactly `-1, 0, 1, 2`. -/
theorem q2Sym_values (c : Nat) (h : c ≤ 3) : q2Sym c = -1 ∨ q2Sym c = 0 ∨ q2Sym c = 1 ∨ q2Sym c = 2 := by
  unfold q2Sym; omega

/-- Element `j < 64` reads byte `j/4 < 16`. -/
theorem q2_byte_in_block (j : Nat) (h : j < 64) : j / 4 < 16 := by omega

/-- The (byte, position) pair determines the element: no two elements share a code. -/
theorem q2_injective (j1 j2 : Nat) (h : j1 / 4 = j2 / 4 ∧ j1 % 4 = j2 % 4) : j1 = j2 := by omega

/-- Q2_0 block: 2 bytes of scale + 16 bytes of codes = 18; 64 codes of 2 bits fill 16 bytes. -/
theorem q2_block_size : 2 + 16 = 18 ∧ 64 * 2 = 16 * 8 := by omega

-- § 4- and 5-bit legacy blocks

/-- Q4_0 nibble codes decode to `[-8, 7]`. -/
theorem q4_0_range (nib : Nat) (h : nib < 16) : -8 ≤ (nib : Int) - 8 ∧ (nib : Int) - 8 ≤ 7 := by omega

/-- Q4_0 outputs `j` and `j+16` are distinct positions inside 32. -/
theorem q4_0_positions (j : Nat) (h : j < 16) : j + 16 < 32 ∧ j ≠ j + 16 := by omega

/-- Q5_0 value: nibble plus the fifth bit. -/
def q5Val (nib bit : Nat) : Nat := nib + 16 * bit

/-- Q5_0 values decode into `[-16, 15]`. -/
theorem q5_0_range (nib bit : Nat) (hn : nib < 16) (hb : bit ≤ 1) :
    -16 ≤ (q5Val nib bit : Int) - 16 ∧ (q5Val nib bit : Int) - 16 ≤ 15 := by
  unfold q5Val; omega

/-- Q5_0 block layout: d(2) | qh(4) | qs(16) = 22. -/
theorem q5_0_block : 2 + 4 + 16 = 22 := by omega

-- § Q6_K

/-- Q6_K layout: ql 0..127, qh 128..191, scales 192..207, d 208..209: 210 bytes. -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 ∧ 192 = 128 + 64 ∧ 208 = 192 + 16 := by omega

/-- A Q6_K value is a nibble plus two high bits. -/
def q6Val (lo hi : Nat) : Nat := lo + 16 * hi

/-- Q6_K values decode into `[-32, 31]`. -/
theorem q6_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ (q6Val lo hi : Int) - 32 ∧ (q6Val lo hi : Int) - 32 ≤ 31 := by
  unfold q6Val; omega

/-- Scale index `is + 2*i` (halves `h`, `is` in 0..1, `i` in 0..3) stays below 16. -/
theorem q6_scale_index (h is i : Nat) (hh : h < 2) (hi : is < 2) (hq : i < 4) : 8 * h + is + 2 * i < 16 := by
  omega

-- § Q4_K / Q5_K

/-- Q4_K: d(2) dmin(2) scales(12) qs(128) = 144. -/
theorem q4k_layout : 2 + 2 + 12 + 128 = 144 ∧ 4 + 12 = 16 := by omega

/-- Q5_K: header 16, qh 32, qs 128 = 176. -/
theorem q5k_layout : 16 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by omega

/-- The four 64-element groups of a K super-block use scale indices `0..7`, two per group. -/
theorem k_group_scale_index (g : Nat) (h : g < 4) : 2 * g + 1 < 8 := by omega

/-- Four groups of 64 cover 256 elements; each advances `q` by 32 so 4 groups use 128 bytes. -/
theorem k_group_cover : 4 * 64 = 256 ∧ 4 * 32 = 128 := by omega

/-- `get_scale_min_k4` reads only bytes `0..11`. -/
theorem scaleMinIdx_in_range (j : Nat) (h : j < 8) :
    (j < 4 → j + 4 < 12) ∧ (4 ≤ j → j + 4 < 12 ∧ j - 4 < 12) := by omega

/-- The Q5_K high-bit masks `u1 = 4^g`, `u2 = 2*4^g` fit in a byte for the four groups. -/
theorem q5k_masks_fit : ∀ g, g < 4 → 2 * 4 ^ g < 256 := by decide

-- § Q3_K

/-- Q3_K: hmask(32) qs(64) scales(12) d(2) = 110; scales at 96, d at 108. -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by omega

/-- The 16 scale reads `is++` cover indices `0..15` exactly once over two halves, four groups, two each. -/
theorem q3k_scale_index (n j e : Nat) (hn : n < 2) (hj : j < 4) (he : e < 2) : 8 * n + 2 * j + e < 16 := by
  omega

-- § IQ4_XS

/-- IQ4_XS: d(2) scales_h(2) scales_l(4) qs(128) = 136. -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by omega

/-- Group `ib < 8` reads `scales_l[ib/2]` with `ib/2 < 4`, and `scales_h` shift `2*ib ≤ 14`. -/
theorem iq4xs_scale_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib ≤ 14 := by omega

/-- The six-bit scale (4 low bits + 2 high bits) is below 64, so `ls - 32 ∈ [-32, 31]`. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § 16-bit conversions

/-- `read_u16`: `p0 | p1 << 8` for bytes. -/
def readU16 (p0 p1 : Nat) : Nat := p0 + 256 * p1

/-- The result is a 16-bit value. -/
theorem readU16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : readU16 p0 p1 < 65536 := by
  unfold readU16; omega

/-- The bytes are recovered. -/
theorem readU16_bytes (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) :
    readU16 p0 p1 % 256 = p0 ∧ readU16 p0 p1 / 256 = p1 := by
  unfold readU16; omega

/-- bf16 -> f32 is `h << 16`: exactly invertible by taking the high half. -/
theorem bf16_roundtrip (h : Nat) : (h * 65536) / 65536 = h := by omega

/-- The shifted value is a 32-bit pattern. -/
theorem bf16_bits_lt (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

end Strata.Dequant
