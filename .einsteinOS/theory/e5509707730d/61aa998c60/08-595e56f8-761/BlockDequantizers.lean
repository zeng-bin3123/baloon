/-!
# MODULE: block_dequantizers
SOURCES: include/strata/artifact/dequant.hpp
Scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q6_K, Q4_K, Q5_K, Q3_K, IQ4_XS) modelled
at the level of integer codes: bit/nibble extraction, index coverage, code ranges and block sizes.

## Source semantics
* Each output is `scale * integer_code`; the integer code ranges are fixed by the bit layout.
* Q2_0: 4 codes per byte, LSB first, code {0..3} -> symbol {-1,0,1,2}.
* Q4_0/IQ4_NL: low nibble is element j, high nibble element j+16.
* Q5_0: a 5th bit from a 32-bit mask; code - 16.
* K-quants: 256-element super-blocks with 6-bit scales unpacked from 12 bytes.
* Block byte sizes add up exactly (no gaps).

## Form template identifiers
- `dequantize_q2_0` -> `ternary2_decode`; `get_scale_min_k4` -> `scale_min_unpack`;
  `kvalues_iq4nl` -> `nonlinear_codebook`.
-/

namespace Strata.BlockDequantizers

-- § block sizes

/-- Q2_0: 2-byte scale + 16 code bytes = 18 bytes holding 16 * 4 = 64 elements. -/
theorem q2_0_sizes : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  decide

/-- Q8_0: 2-byte scale + 32 int8 = 34 bytes. -/
theorem q8_0_size : 2 + 32 = 34 := by
  decide

/-- Q4_0 and IQ4_NL: 2-byte scale + 16 nibble bytes = 18 bytes for 32 elements. -/
theorem q4_0_size : 2 + 16 = 18 ∧ 16 * 2 = 32 := by
  decide

/-- Q5_0: scale 2 + qh 4 + qs 16 = 22 bytes. -/
theorem q5_0_size : 2 + 4 + 16 = 22 := by
  decide

/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2 = 210 bytes (d is last at offset 208). -/
theorem q6_k_size : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 := by
  decide

/-- Q4_K: d 2 + dmin 2 + scales 12 + qs 128 = 144 bytes (qs at offset 16). -/
theorem q4_k_size : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := by
  decide

/-- Q5_K: d 2 + dmin 2 + scales 12 + qh 32 + qs 128 = 176 bytes (qh at 16, qs at 48). -/
theorem q5_k_size : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by
  decide

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2 = 110 bytes (scales at 96, d at 108). -/
theorem q3_k_size : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by
  decide

/-- IQ4_XS: d 2 + scales_h 2 + scales_l 4 + qs 128 = 136 bytes (qs at offset 8). -/
theorem iq4_xs_size : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by
  decide

-- § Q2_0

/-- The 2-bit code of element `k` (0..3) inside a byte. -/
def q2Code (byte k : Nat) : Nat := byte / 4 ^ k % 4

/-- The symbol of a code: code - 1, so {0,1,2,3} -> {-1,0,1,2}. -/
def q2Sym (byte k : Nat) : Int := (q2Code byte k : Int) - 1

/-- Every Q2_0 symbol is one of -1, 0, 1, 2. -/
theorem q2_sym_range (byte k : Nat) : -1 ≤ q2Sym byte k ∧ q2Sym byte k ≤ 2 := by
  have h : q2Code byte k < 4 := by
    unfold q2Code
    exact Nat.mod_lt _ (by decide)
  unfold q2Sym
  omega

/-- A code byte is exactly its four 2-bit fields (no bits are lost or double-counted). -/
theorem q2_byte_decomposition (b : Nat) (hb : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- The element index maps bijectively onto (byte j/4, slot j%4). -/
theorem q2_index_split (j : Nat) : 4 * (j / 4) + j % 4 = j := by
  omega

/-- All 64 elements of a block address one of the 16 code bytes. -/
theorem q2_byte_index_lt (j : Nat) (hj : j < 64) : j / 4 < 16 := by
  omega

/-- The bit offset (j%4)*2 is at most 6, so a 2-bit field never straddles bytes. -/
theorem q2_bit_offset_le (j : Nat) : (j % 4) * 2 ≤ 6 := by
  omega

-- § nibble formats: Q4_0, IQ4_NL, Q5_0

/-- Q4_0 value for a nibble `n < 16`: n - 8. -/
def q4Val (n : Nat) : Int := (n : Int) - 8

/-- Q4_0 values lie in [-8, 7]. -/
theorem q4_range (n : Nat) (h : n < 16) : -8 ≤ q4Val n ∧ q4Val n ≤ 7 := by
  unfold q4Val
  omega

/-- A byte is exactly its low nibble plus 16 times its high nibble. -/
theorem nibble_decomposition (q : Nat) : q % 16 + 16 * (q / 16) = q := by
  omega

/-- For a byte, the high nibble is below 16, so it indexes a 16-entry table. -/
theorem high_nibble_lt (q : Nat) (hq : q < 256) : q / 16 < 16 := by
  omega

/-- Elements j (low nibble) and j+16 (high nibble) are distinct and cover 0..31 for j < 16. -/
theorem q4_halves_cover (i : Nat) (hi : i < 32) :
    (i < 16 ∧ i = i) ∨ (16 ≤ i ∧ ∃ j, j < 16 ∧ i = j + 16) := by
  by_cases h : i < 16
  · exact Or.inl ⟨h, rfl⟩
  · exact Or.inr ⟨by omega, i - 16, by omega, by omega⟩

/-- Q5_0 value: 5-bit code minus 16. -/
def q5Val (nib bit : Nat) : Int := ((nib + 16 * bit : Nat) : Int) - 16

/-- Q5_0 values lie in [-16, 15] for a nibble and a single high bit. -/
theorem q5_range (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : -16 ≤ q5Val nib bit ∧ q5Val nib bit ≤ 15 := by
  unfold q5Val
  omega

/-- The i-th bit of a 32-bit mask is 0 or 1. -/
theorem mask_bit_lt (h i : Nat) : h / 2 ^ i % 2 < 2 := by
  exact Nat.mod_lt _ (by decide)

-- § IQ4_NL codebook

/-- The IQ4_NL non-linear codebook, verbatim from ggml. -/
def kvaluesIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per nibble value. -/
theorem iq4nl_length : kvaluesIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing, so the 4-bit code is order preserving. -/
theorem iq4nl_sorted : kvaluesIq4nl.Pairwise (· < ·) := by
  decide

/-- Every codebook entry is within [-127, 113]. -/
theorem iq4nl_bounds : ∀ v ∈ kvaluesIq4nl, -127 ≤ v ∧ v ≤ 113 := by
  decide

/-- The codebook is asymmetric: its largest magnitude is 127 on the negative side. -/
theorem iq4nl_asymmetric : kvaluesIq4nl.head? = some (-127) ∧ kvaluesIq4nl.getLast? = some 113 := by
  decide

-- § Q6_K

/-- Q6_K 6-bit code: 4 low bits from ql, 2 high bits from qh, minus 32. -/
def q6Val (lo4 hi2 : Nat) : Int := ((lo4 % 16 + 16 * (hi2 % 4) : Nat) : Int) - 32

/-- Q6_K codes lie in [-32, 31]. -/
theorem q6_range (lo hi : Nat) : -32 ≤ q6Val lo hi ∧ q6Val lo hi ≤ 31 := by
  unfold q6Val
  omega

/-- The four output positions l, l+32, l+64, l+96 (l < 32) cover each index of a 128-half exactly once. -/
theorem q6_positions_cover (i : Nat) (hi : i < 128) : i = i % 32 + 32 * (i / 32) ∧ i / 32 < 4 := by
  omega

/-- Scale indices used in a Q6_K half stay in 0..15 over both halves (n = 0, 1). -/
theorem q6_scale_index_lt (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k ≤ 6) :
    8 * n + l / 16 + k < 16 := by
  omega

-- § 6-bit scale/min unpacking (Q4_K, Q5_K)

/-- `get_scale_min_k4` for j ≥ 4: low 4 bits of q[j+4] plus the top 2 bits of q[j-4] shifted by 4. -/
def scaleHigh (qj4 qjm4 : Nat) : Nat := qj4 % 16 + 16 * (qjm4 / 64)

/-- The unpacked 6-bit scale for j ≥ 4 is below 64 when the source byte is below 256. -/
theorem scale_high_lt (a b : Nat) (hb : b < 256) : scaleHigh a b < 64 := by
  unfold scaleHigh
  omega

/-- The unpacked 6-bit scale for j < 4 (`q[j] & 63`) is below 64. -/
theorem scale_low_lt (a : Nat) : a % 64 < 64 := by
  exact Nat.mod_lt _ (by decide)

/-- Q4_K / Q5_K loop: group scale indices are is and is+1 with is in {0,2,4,6}, always below 8. -/
theorem k_scale_index_lt (g : Nat) (hg : g < 4) : 2 * g + 1 < 8 := by
  omega

/-- Q4_K covers 256 elements as 4 iterations of 64. -/
theorem q4_k_coverage : 4 * 64 = 256 := by
  decide

-- § Q5_K high-bit masks

/-- Mask of the 5th bit for the low-nibble half of group `g`: 1, 4, 16, 64. -/
def maskLo (g : Nat) : Nat := 4 ^ g

/-- Mask of the 5th bit for the high-nibble half of group `g`: 2, 8, 32, 128. -/
def maskHi (g : Nat) : Nat := 2 * 4 ^ g

/-- All eight masks are distinct single bits of a byte (each group/half pair has its own bit). -/
theorem q5_masks_distinct :
    ([maskLo 0, maskLo 1, maskLo 2, maskLo 3, maskHi 0, maskHi 1, maskHi 2, maskHi 3] : List Nat).Nodup := by
  decide

/-- The eight masks together are exactly 1,2,4,...,128 (every bit of qh is used once). -/
theorem q5_masks_all_bits :
    ([maskLo 0, maskHi 0, maskLo 1, maskHi 1, maskLo 2, maskHi 2, maskLo 3, maskHi 3] : List Nat) =
      [1, 2, 4, 8, 16, 32, 64, 128] := by
  decide

/-- Q5_K code is a nibble plus 16 when the mask bit is set: at most 31. -/
theorem q5_k_code_le (nib : Nat) (b : Bool) (h : nib < 16) : nib + (if b then 16 else 0) ≤ 31 := by
  cases b <;> simp <;> omega

-- § Q3_K

/-- Q3_K element code: 2-bit low part minus 4 when the high-mask bit is clear. -/
def q3Val (c : Nat) (hiSet : Bool) : Int := (c : Int) - (if hiSet then 0 else 4)

/-- Q3_K element codes lie in [-4, 3]. -/
theorem q3_range (c : Nat) (b : Bool) (h : c < 4) : -4 ≤ q3Val c b ∧ q3Val c b ≤ 3 := by
  cases b <;> simp [q3Val] <;> omega

/-- Q3_K repacked scale byte: low 4 bits plus 2 high bits shifted by 4 is a 6-bit value; minus 32 gives [-32, 31]. -/
theorem q3_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q3_K draws 16 scales: 2 halves x 4 shifts x 2 sub-blocks. -/
theorem q3_k_scale_count : 2 * 4 * 2 = 16 := by
  decide

-- § IQ4_XS

/-- The 6-bit scale of group `ib`: a nibble of scales_l plus 2 bits of scales_h at shift 2*ib. -/
def iq4xsScale (sl : Nat) (ib : Nat) (sh : Nat) : Nat :=
  (sl / 16 ^ (ib % 2)) % 16 + 16 * ((sh / 4 ^ ib) % 4)

/-- The IQ4_XS scale is below 64, so `ls - 32` lies in [-32, 31]. -/
theorem iq4xs_scale_lt (sl ib sh : Nat) : iq4xsScale sl ib sh < 64 := by
  unfold iq4xsScale
  have h1 : (sl / 16 ^ (ib % 2)) % 16 < 16 := Nat.mod_lt _ (by decide)
  have h2 : (sh / 4 ^ ib) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The scales_l byte index ib/2 is below 4 for the 8 groups. -/
theorem iq4xs_scales_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

/-- Eight 32-element groups cover 256 elements, consuming 8 * 16 = 128 qs bytes. -/
theorem iq4xs_coverage : 8 * 32 = 256 ∧ 8 * 16 = 128 := by
  decide

end Strata.BlockDequantizers
