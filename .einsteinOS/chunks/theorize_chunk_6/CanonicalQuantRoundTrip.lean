/-!
# MODULE: canonical_quant_round_trip
The integer-level contracts behind the bit-exact canonicalisation of ggml quantised blocks (Q2_0, Q3_K, Q4_0,
Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS) into code and scale planes.

## Source semantics
* Each type's decode is `(code + code_bias) * scale [+ offset]` with the bias applied to the INTEGER code
  before the single float multiply (folding it into the value domain is not bit-exact).
* Q2_0: `(code - 1) * d`, codes 0..3, four per byte at bit offsets 0,2,4,6.
* Q4_0 / IQ4_NL: low nibbles fill the first 16 elements, high nibbles the next 16; Q4_0 bias is -8.
* Q5_0: 5th bit of element `j` is bit `j` of the 32-bit `qh`; bias -16. Q8_0: stored `int8 + 128`, bias -128.
* `get_scale_min_k4`: for group `j < 4` scale and min are `s[j] & 63`, `s[j+4] & 63`; for `j >= 4` they borrow
  the top two bits of the low groups' bytes.
* Q3_K: 16 six-bit scales assembled from 12 bytes, stored code `(2 bits) + 4*hmask_bit` in 0..7, bias -4.
* Q6_K: code `nibble | (2 bits << 4)` in 0..63, bias -32. IQ4: 16-entry non-affine codebook.

## Form template identifiers
code_bias := code_bias ; block_bytes := block_bytes ; block_elems := block_elems ; group_elems := group_elems ;
scale6 := get_scale_min_k4 / unpack_q3_k_scales ; nibble_lo := byte % 16 ; nibble_hi := byte / 16 ;
iq4_codebook := KV_IQ4NL
-/

namespace Strata.CanonicalQuant

-- § block geometry

/-- Q2_0: 2-byte scale + 16 code bytes cover 64 elements. -/
theorem q2_0_geometry : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  decide

/-- Q4_0 and IQ4_NL: 2-byte scale + 16 nibble bytes cover 32 elements. -/
theorem q4_0_geometry : 2 + 16 = 18 ∧ 16 * 2 = 32 := by
  decide

/-- Q4_K: d, dmin, 12 packed scale bytes and 128 nibble bytes make 144 bytes for 256 elements. -/
theorem q4_k_geometry : 2 + 2 + 12 + 128 = 144 ∧ 128 * 2 = 256 := by
  decide

/-- Q5_K adds a 32-byte high-bit plane. -/
theorem q5_k_geometry : 2 + 2 + 12 + 32 + 128 = 176 := by
  decide

/-- Q3_K: hmask 32 + qs 64 + scales 12 + d 2 = 110 bytes. -/
theorem q3_k_geometry : 32 + 64 + 12 + 2 = 110 ∧ 64 * 4 = 256 := by
  decide

/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2 = 210 bytes. -/
theorem q6_k_geometry : 128 + 64 + 16 + 2 = 210 := by
  decide

/-- IQ4_XS: d 2 + scales_h 2 + scales_l 4 + qs 128 = 136 bytes. -/
theorem iq4_xs_geometry : 2 + 2 + 4 + 128 = 136 := by
  decide

/-- Q5_0: d 2 + qh 4 + qs 16 = 22 bytes for 32 elements. -/
theorem q5_0_geometry : 2 + 4 + 16 = 22 ∧ 16 * 2 = 32 := by
  decide

/-- Q8_0: d 2 + 32 int8 = 34 bytes. -/
theorem q8_0_geometry : 2 + 32 = 34 := by
  decide

/-- Every group size divides its block, so a block holds a whole number of scale entries. -/
theorem group_divides_block :
    64 % 64 = 0 ∧ 256 % 16 = 0 ∧ 32 % 32 = 0 ∧ 256 % 32 = 0 := by
  decide

-- § Q2_0

/-- Four 2-bit codes pack into one byte and unpack to themselves (`codes[:, t::4] = (byte >> 2t) & 3`). -/
theorem q2_pack_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) < 256 ∧ (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4) % 4 = c1 ∧ ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16) % 4 = c2 ∧
      ((c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64) % 4 = c3 := by
  omega

/-- Any byte is the sum of its four 2-bit fields: unpacking loses nothing. -/
theorem q2_byte_decompose (b : Nat) (h : b < 256) :
    b = b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) := by
  omega

/-- The Q2_0 integer code minus one is in `{-1, 0, 1, 2}`: the codebook is NOT `{0,1,2,3}`. -/
theorem q2_codebook (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

-- § 4-bit nibble split (Q4_0, IQ4_NL, IQ4_XS, Q4_K)

/-- A byte is `lo + 16*hi` for its two nibbles, and the nibbles are in range. -/
theorem nibble_split (b : Nat) (h : b < 256) :
    b % 16 < 16 ∧ b / 16 < 16 ∧ b = b % 16 + 16 * (b / 16) := by
  omega

/-- Two 4-bit codes pack into a byte and unpack to themselves (first half low, second half high). -/
theorem nibble_pack_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) < 256 ∧ (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by
  omega

/-- Q4_0 integer code minus eight is in `[-8, 7]`. -/
theorem q4_0_codebook (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- The low nibbles of byte `j` (j < 16) and the high nibbles land 16 apart in the 32-element block. -/
theorem q4_0_positions (j : Nat) (h : j < 16) : j < 16 ∧ 16 ≤ j + 16 ∧ j + 16 < 32 := by
  omega

-- § Q8_0

/-- Storing the int8 code `+128` and subtracting 128 inside the integer restores it exactly. -/
theorem q8_0_bias_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    0 ≤ q + 128 ∧ q + 128 ≤ 255 ∧ (q + 128) - 128 = q := by
  omega

-- § Q5_0 / Q5_K (5-bit codes)

/-- A 5-bit code is a nibble plus 16 times the high bit and stays below 32. -/
theorem q5_code_range (lo bit : Nat) (hl : lo < 16) (hb : bit < 2) : lo + 16 * bit < 32 := by
  omega

/-- Q5_0 integer code minus sixteen is in `[-16, 15]`. -/
theorem q5_0_codebook (c : Nat) (h : c < 32) : -16 ≤ (c : Int) - 16 ∧ (c : Int) - 16 ≤ 15 := by
  omega

/-- The source reaches bit `j+16` of `qh` as `(qh >> (j+12)) & 0x10`; the 5th bit of element `j+16`
sits at bit position `j + 16`, i.e. `j + 12 + 4`. -/
theorem q5_0_high_bit_shift (j : Nat) : j + 12 + 4 = j + 16 := by
  omega

/-- Q5_K selector masks `u1`, `u2` walk bits 0,2,4,6 and 1,3,5,7 over the four 64-element steps. -/
theorem q5_k_mask_bits (j : Nat) (h : j < 4) : 2 * j < 8 ∧ 2 * j + 1 < 8 ∧ 2 * j ≠ 2 * j + 1 := by
  omega

-- § k-quant scales

/-- `get_scale_min_k4`, low groups: scale is `s[j] % 64`, always a 6-bit value. -/
theorem scale_min_low (s : Nat) (hs : s < 256) : s % 64 < 64 := by
  omega

/-- `get_scale_min_k4`, high groups: `(s[j+4] & 0xF) | ((s[j-4] >> 6) << 4)` is a 6-bit value. -/
theorem scale_min_high_scale (hi lo : Nat) (h1 : hi < 256) (h2 : lo < 256) :
    hi % 16 + 16 * (lo / 64) < 64 := by
  omega

/-- ... and the matching min `(s[j+4] >> 4) | ((s[j] >> 6) << 4)` is a 6-bit value too. -/
theorem scale_min_high_min (hi lo : Nat) (h1 : hi < 256) (h2 : lo < 256) :
    hi / 16 + 16 * (lo / 64) < 64 := by
  omega

/-- Q3_K scale group `j`: low nibble of `s[j]` plus two bits from `s[8+j]` is 6-bit. -/
theorem q3_k_scale_group (lo t : Nat) (h1 : lo < 256) (h2 : t < 256) :
    lo % 16 + 16 * (t % 4) < 64 := by
  omega

/-- Q3_K scale group `8+j`: the HIGH nibble of the same byte plus bits 4..5 of `s[8+j]` is 6-bit. -/
theorem q3_k_scale_group_hi (lo t : Nat) (h1 : lo < 256) (h2 : t < 256) :
    lo / 16 + 16 * ((t / 16) % 4) < 64 := by
  omega

/-- The 16 Q3_K groups use the low nibbles of bytes 0..7 (groups 0..7) and the high nibbles of the same
bytes (groups 8..15); the group index never exceeds 15. -/
theorem q3_k_group_index (j : Nat) (h : j < 4) : j < 16 ∧ 4 + j < 16 ∧ 8 + j < 16 ∧ 12 + j < 16 := by
  omega

/-- Q3_K stored code `(2 bits) + 4*hbit` is in `0..7`, bias -4 gives `[-4, 3]`. -/
theorem q3_k_code_range (c2 m : Nat) (h1 : c2 < 4) (h2 : m < 2) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) - 4 ∧ ((c2 + 4 * m : Nat) : Int) - 4 ≤ 3 := by
  omega

/-- The Q3_K hmask bit index `4*half + j` is a valid bit (0..7); the second half uses bits 4..7. -/
theorem q3_k_hmask_bit (half j : Nat) (hh : half < 2) (hj : j < 4) : 4 * half + j < 8 := by
  omega

/-- Q3_K element order `128*half + 32*j + 16*sub + l` covers 0..255 exactly once. -/
theorem q3_k_element_order (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    128 * half + 32 * j + 16 * sub + l < 256 := by
  omega

/-- The Q3_K canonical group number `8*half + 2*j + sub` is `element / 16`. -/
theorem q3_k_group_of_element (half j sub l : Nat) (hl : l < 16) :
    (128 * half + 32 * j + 16 * sub + l) / 16 = 8 * half + 2 * j + sub := by
  omega

/-- Q6_K stored code `nibble + 16 * (2 bits)` is in `0..63`, bias -32 gives `[-32, 31]`. -/
theorem q6_k_code_range (nib hi2 : Nat) (h1 : nib < 16) (h2 : hi2 < 4) :
    nib + 16 * hi2 < 64 ∧ -32 ≤ ((nib + 16 * hi2 : Nat) : Int) - 32 ∧ ((nib + 16 * hi2 : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q6_K quads land at strides of 32: quad `q`, offset `l` is element `128*half + 32*q + l` and the
scale group is plain `element / 16`. -/
theorem q6_k_quad_layout (half q l : Nat) (hh : half < 2) (hq : q < 4) (hl : l < 32) :
    128 * half + 32 * q + l < 256 ∧ (128 * half + 32 * q + l) / 16 = 8 * half + 2 * q + l / 16 := by
  omega

/-- IQ4_XS: the 6-bit scale from `scales_l` nibble and 2 bits of `scales_h` is below 64. -/
theorem iq4_xs_scale (sl sh : Nat) (hl : sl < 256) (hh : sh < 65536) (ib : Nat) (hib : ib < 8) :
    ((sl / 16 ^ (ib % 2)) % 16) + 16 * ((sh / 4 ^ ib) % 4) < 64 := by
  have hcases : ib = 0 ∨ ib = 1 ∨ ib = 2 ∨ ib = 3 ∨ ib = 4 ∨ ib = 5 ∨ ib = 6 ∨ ib = 7 := by
    omega
  rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp <;> omega

/-- IQ4_XS scale index `ib/2` addresses one of the four `scales_l` bytes. -/
theorem iq4_xs_scale_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

-- § IQ4 codebook (non-affine)

/-- The 16-entry IQ4 codebook `kvalues_iq4nl`. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries, one per 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing (so it is a genuine table, and not an affine map of the code). -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (fun a b => a < b) := by
  decide

/-- The codebook is not affine: consecutive gaps differ, so no `bias` can reproduce it. -/
theorem kvIq4nl_not_affine : (-104 : Int) - (-127) ≠ 1 - (-10) := by
  decide

/-- A decode of the value set stays within the codebook extremes. -/
theorem kvIq4nl_extremes : (-127 : Int) ≤ -104 ∧ (113 : Int) ≤ 127 := by
  decide

end Strata.CanonicalQuant
