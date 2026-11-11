/-!
# MODULE: k_quant_scale_unpacking
SOURCES: include/strata/artifact/dequant.hpp, tools/canonical_xcheck.py
Integer model of the super-block (256-element) dequantizers Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS: the
packed 6-bit scale/min unpacking, the walking bit masks, the group/element order, and the code
ranges that decide which canonical form (S4/S8) each type fits.

## Source semantics
* `get_scale_min_k4(j)`: groups 0..3 take 6 bits of q[j] / q[j+4]; groups 4..7 take a nibble of
  q[j+4] and borrow the top two bits of q[j-4] / q[j]. Every scale and min is < 64.
* Q4_K/Q5_K: each 64-element step emits group `is` (low nibbles) then `is+1` (high nibbles), so
  element e belongs to group e/32.
* Q5_K: `qh` never advances; the 5th bit comes from masks u1 = 1,4,16,64 and u2 = 2,8,32,128, i.e.
  all eight bits of each qh byte, each used once. Codes are ≤ 31.
* Q6_K: four quads from one qh byte land at strides of 32; the scale index `is + 2*quad` with
  is = l/16 equals (position in half)/16. Codes are < 64, biased by -32.
* Q3_K: 16 six-bit scales unpacked from 12 bytes (biased by -32); the hmask bit for (half, j) is
  4*half + j because `m` is never reset; stored code = 2 bits + 4*(hmask bit) ∈ 0..7, decoded as
  code - 4, i.e. the low bits minus 4 when the mask bit is clear.
* IQ4_XS: ls = low nibble of scales_l[ib/2] | 2 bits of scales_h at 2*ib, shifted by 4; ls < 64.

## Form template identifiers
get_scale_min_k4 → `scaleMinK4`; unpack_q3_k_scales → `q3Scale`; C++ kmask1/kmask2 repack →
`q3RepackWord`; Q5_K u1/u2 → `q5U1`/`q5U2`; Q6_K is/quad → `q6ScaleIndex`; IQ4_XS ls → `iq4xsLs`;
the 6-bit assembly `lo | (hi << 4)` → `pack6`.
-/

namespace Strata.KQuant

-- § helpers

/-- A disjoint OR is addition: `a | (b << 4) = a + 16 b` when `a < 16`. -/
theorem or_shl4 (a b : Nat) (ha : a < 16) : a ||| (b <<< 4) = a + b * 16 := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt ha, Nat.shiftLeft_eq]; omega

/-- A nibble mask is `% 16`. -/
theorem and15 (x : Nat) : x &&& 15 = x % 16 := Nat.and_two_pow_sub_one_eq_mod x 4

/-- A 2-bit mask is `% 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := Nat.and_two_pow_sub_one_eq_mod x 2

/-- A 6-bit mask is `% 64`. -/
theorem and63 (x : Nat) : x &&& 63 = x % 64 := Nat.and_two_pow_sub_one_eq_mod x 6

/-- Assemble a 6-bit value from a low nibble and a 2-bit high part. -/
def pack6 (lo hi : Nat) : Nat := (lo &&& 15) ||| ((hi &&& 3) <<< 4)

/-- `pack6` is `lo % 16 + 16 * (hi % 4)`, hence always below 64. -/
theorem pack6_eq (lo hi : Nat) : pack6 lo hi = lo % 16 + (hi % 4) * 16 ∧ pack6 lo hi < 64 := by
  unfold pack6
  rw [and15, and3, or_shl4 _ _ (Nat.mod_lt _ (by decide))]
  omega

-- § get_scale_min_k4 (Q4_K, Q5_K)

/-- `get_scale_min_k4(j, q)` returning (scale, min). -/
def scaleMinK4 (j : Nat) (q : Nat → Nat) : Nat × Nat :=
  if j < 4 then (q j &&& 63, q (j + 4) &&& 63)
  else ((q (j + 4) &&& 15) ||| ((q (j - 4) >>> 6) <<< 4),
        (q (j + 4) >>> 4) ||| ((q j >>> 6) <<< 4))

/-- For byte inputs, both halves of every group are 6-bit values. -/
theorem scaleMinK4_lt (j : Nat) (q : Nat → Nat) (hq : ∀ k, q k < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64 := by
  unfold scaleMinK4
  split
  · simp only [and63]; exact ⟨Nat.mod_lt _ (by decide), Nat.mod_lt _ (by decide)⟩
  · have h1 := hq (j + 4); have h2 := hq (j - 4); have h3 := hq j
    simp only [Nat.shiftRight_eq_div_pow]
    refine ⟨?_, ?_⟩
    · rw [and15, or_shl4 _ _ (Nat.mod_lt _ (by decide))]; omega
    · rw [or_shl4 _ _ (by omega)]; omega

/-- Reading only `q[j]` for the high groups is wrong: with q = 0xC5 everywhere, group 4's
scale is 0x35 = 53, not the 6 low bits 0x05. -/
theorem high_group_borrows_bits : (scaleMinK4 4 (fun _ => 0xC5)).1 = 53 := by decide

-- § Q4_K / Q5_K element order

/-- Element `e` of a Q4_K/Q5_K super-block: step `e/64`, nibble half `(e%64)/32`, group `e/32`. -/
theorem k4_group_of_element (e : Nat) :
    2 * (e / 64) + (e % 64) / 32 = e / 32 := by omega

/-- Element `e` reads code byte `32*(e/64) + e%32`, which stays inside the 128-byte `qs`. -/
theorem k4_code_byte_lt (e : Nat) (he : e < 256) : 32 * (e / 64) + e % 32 < 128 := by omega

/-- Distinct elements of a 64-element step use distinct (byte, nibble) pairs. -/
theorem k4_address_injective (e f : Nat)
    (hb : 32 * (e / 64) + e % 32 = 32 * (f / 64) + f % 32)
    (hn : (e % 64) / 32 = (f % 64) / 32) : e = f := by omega

-- § Q5_K masks

/-- `u1` at step `j` after `j` shifts by two. -/
def q5U1 (j : Nat) : Nat := 1 <<< (2 * j)
/-- `u2` at step `j` after `j` shifts by two. -/
def q5U2 (j : Nat) : Nat := 2 <<< (2 * j)

/-- The masks are 1,4,16,64 and 2,8,32,128: together, each of the eight bits exactly once. -/
theorem q5_masks_cover_byte :
    [q5U1 0, q5U2 0, q5U1 1, q5U2 1, q5U1 2, q5U2 2, q5U1 3, q5U2 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- A shift-by-one mask would reuse a bit: step 1's u1 would equal step 0's u2. -/
theorem q5_shift_by_one_collides : (1 <<< 1 : Nat) = 2 <<< 0 := by decide

/-- Q5_K codes are `nibble + 16*bit`, at most 31, so they fit the S8 form. -/
theorem q5_code_le (b h : Nat) (hh : h ≤ 1) : b % 16 + 16 * h ≤ 31 := by omega

-- § Q6_K

/-- Scale index for row offset `l` (0..31) and quad `qd` (0..3): `l/16 + 2*qd`. -/
def q6ScaleIndex (l qd : Nat) : Nat := l / 16 + 2 * qd

/-- The quad layout at strides of 32 makes the scale index plain consecutive 16-element groups. -/
theorem q6_scale_is_group (l qd : Nat) (hl : l < 32) :
    q6ScaleIndex l qd = (l + 32 * qd) / 16 := by
  unfold q6ScaleIndex; omega

/-- Within a half the scale index stays below 8, so a half consumes exactly `sc[0..7]`. -/
theorem q6_scale_lt (l qd : Nat) (hl : l < 32) (hq : qd < 4) : q6ScaleIndex l qd < 8 := by
  unfold q6ScaleIndex; omega

/-- A Q6_K code `(nibble) | ((2 bits) << 4)` is below 64, so `code - 32 ∈ [-32, 31]`. -/
theorem q6_code_range (nib hi : Nat) :
    pack6 nib hi < 64 ∧ -32 ≤ (pack6 nib hi : Int) - 32 ∧ (pack6 nib hi : Int) - 32 ≤ 31 := by
  have h := (pack6_eq nib hi).2
  omega

-- § Q3_K

/-- Python `unpack_q3_k_scales`: group `g` (0..15) from the 12 scale bytes `s`. -/
def q3Scale (s : Nat → Nat) (g : Nat) : Nat :=
  let j := g % 4
  let t := s (8 + j)
  match g / 4 with
  | 0 => pack6 (s j) t
  | 1 => pack6 (s (4 + j)) (t >>> 2)
  | 2 => pack6 (s j >>> 4) (t >>> 4)
  | _ => pack6 (s (4 + j) >>> 4) (t >>> 6)

/-- Every unpacked Q3_K scale is 6 bits; after the `- 32` bias it is in `[-32, 31]`. -/
theorem q3Scale_range (s : Nat → Nat) (g : Nat) :
    q3Scale s g < 64 ∧ -32 ≤ (q3Scale s g : Int) - 32 ∧ (q3Scale s g : Int) - 32 ≤ 31 := by
  have h : q3Scale s g < 64 := by
    unfold q3Scale
    split <;> exact (pack6_eq _ _).2
  omega

/-- Little-endian 32-bit word from four bytes. -/
def word (b0 b1 b2 b3 : Nat) : Nat := b0 + 256 * b1 + 65536 * b2 + 16777216 * b3

/-- Byte `k` of a 32-bit word. -/
def byteOf (w k : Nat) : Nat := w / 256 ^ k % 256

/-- The C++ repack of `aux[0..3]` with `kmask1 = 0x03030303`, `kmask2 = 0x0f0f0f0f`. -/
def q3RepackWord (a0 a1 tmp : Nat) (k : Nat) : Nat :=
  let k1 := 0x03030303
  let k2 := 0x0f0f0f0f
  match k with
  | 0 => (a0 &&& k2) ||| (((tmp >>> 0) &&& k1) <<< 4)
  | 1 => (a1 &&& k2) ||| (((tmp >>> 2) &&& k1) <<< 4)
  | 2 => ((a0 >>> 4) &&& k2) ||| (((tmp >>> 4) &&& k1) <<< 4)
  | _ => ((a1 >>> 4) &&& k2) ||| (((tmp >>> 6) &&& k1) <<< 4)

/-- Read the 16 C++ scales (aux viewed as bytes) for a 12-byte scale array given as a list. -/
def q3CppScales (s : List Nat) : List Nat :=
  let b := fun i => s.getD i 0
  let a0 := word (b 0) (b 1) (b 2) (b 3)
  let a1 := word (b 4) (b 5) (b 6) (b 7)
  let tmp := word (b 8) (b 9) (b 10) (b 11)
  (List.range 16).map fun g => byteOf (q3RepackWord a0 a1 tmp (g / 4)) (g % 4)

/-- The Python per-byte formula on the same list. -/
def q3PyScales (s : List Nat) : List Nat :=
  (List.range 16).map fun g => q3Scale (fun i => s.getD i 0) g

/-- C++ word-level repack and Python byte-level unpack agree on test vectors that exercise every
bit of the 12 scale bytes (all ones, a ramp, and alternating patterns). -/
theorem q3_repack_agrees_vectors :
    q3CppScales (List.replicate 12 0xFF) = q3PyScales (List.replicate 12 0xFF) ∧
    q3CppScales [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x1B, 0xE4, 0x6C, 0x93] =
      q3PyScales [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE, 0x1B, 0xE4, 0x6C, 0x93] ∧
    q3CppScales [0xA5, 0x5A, 0xA5, 0x5A, 0x5A, 0xA5, 0x5A, 0xA5, 0xC3, 0x3C, 0x96, 0x69] =
      q3PyScales [0xA5, 0x5A, 0xA5, 0x5A, 0x5A, 0xA5, 0x5A, 0xA5, 0xC3, 0x3C, 0x96, 0x69] := by
  decide

/-- The hmask selector after `4*half + j` left shifts of `m = 1` (it is never reset). -/
def q3Mask (half j : Nat) : Nat := 1 <<< (4 * half + j)

/-- The two halves use disjoint mask bits: 0..3 in the first, 4..7 in the second. -/
theorem q3_mask_bits :
    [q3Mask 0 0, q3Mask 0 1, q3Mask 0 2, q3Mask 0 3, q3Mask 1 0, q3Mask 1 1, q3Mask 1 2, q3Mask 1 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- Resetting `m` per half would make the second half re-read bit 0. -/
theorem q3_reset_mask_wrong : (1 <<< 0 : Nat) ≠ q3Mask 1 0 := by decide

/-- Element order of `*y++`: group `8h + 2j + sub` holds elements `128h + 32j + 16sub + l`. -/
theorem q3_group_of_element (h j sub l : Nat) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    (128 * h + 32 * j + 16 * sub + l) / 16 = 8 * h + 2 * j + sub := by omega

/-- The stored code `c2 + 4*bit` is in 0..7, and `code - 4` is exactly the C++
`c2 - (bit ? 0 : 4)`: the bias is an integer bias on the code. -/
theorem q3_code_bias (c2 bit : Nat) (hc : c2 < 4) (hb : bit ≤ 1) :
    c2 + 4 * bit < 8 ∧
      ((c2 + 4 * bit : Nat) : Int) - 4 = (c2 : Int) - (if bit = 1 then 0 else 4) := by
  constructor
  · omega
  · split <;> omega

-- § IQ4_XS

/-- IQ4_XS group scale: nibble `ib%2` of `scales_l[ib/2]` plus 2 bits of `scales_h` at `2*ib`. -/
def iq4xsLs (scalesL : Nat → Nat) (scalesH ib : Nat) : Nat :=
  ((scalesL (ib / 2) >>> (4 * (ib % 2))) &&& 15) ||| (((scalesH >>> (2 * ib)) &&& 3) <<< 4)

/-- `ls` is 6 bits, so `ls - 32 ∈ [-32, 31]`. -/
theorem iq4xs_ls_range (sl : Nat → Nat) (sh ib : Nat) :
    iq4xsLs sl sh ib < 64 ∧ -32 ≤ (iq4xsLs sl sh ib : Int) - 32 := by
  have h : iq4xsLs sl sh ib < 64 := by
    unfold iq4xsLs
    rw [and15, and3, or_shl4 _ _ (Nat.mod_lt _ (by decide))]
    omega
  omega

/-- Reading `scales_l` alone is wrong once `ls ≥ 16`: with scales_h = 0xFFFF the scale of group 0
is 0x30 + nibble, not the nibble. -/
theorem iq4xs_needs_high_bits : iq4xsLs (fun _ => 0x21) 0xFFFF 0 = 0x31 := by decide

end Strata.KQuant
