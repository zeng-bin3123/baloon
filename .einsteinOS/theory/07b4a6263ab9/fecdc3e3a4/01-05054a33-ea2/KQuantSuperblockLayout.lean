/-!
# MODULE: kquant_superblock_layout
SOURCES: include/strata/artifact/dequant.hpp, tools/canonical_xcheck.py
The 256-element super-block encodings Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS: their byte layouts, the
6-bit packed scale unpackers (`get_scale_min_k4`, Q3_K's 12-byte repack, IQ4_XS's split `ls`), the
moving bit masks, and the element-to-group maps the canonical forms rely on.

## Source semantics
* Super-block sizes: Q3_K 110, Q4_K 144, Q5_K 176, Q6_K 210, IQ4_XS 136 bytes per 256 elements.
* `get_scale_min_k4(j)`: for `j < 4` take `q[j] & 63`, `q[j+4] & 63`; otherwise build 6 bits from
  a low nibble of `q[j+4]` and the top two bits of `q[j-4]` / `q[j]`. Results are always `< 64`.
* Q3_K scales are signed 6-bit with `- 32`; its low 2-bit code is corrected by `-4` when the hmask
  bit is CLEAR. The canonical code `c2 + 4*hbit` with bias `-4` gives the same integer.
* Q3_K's hmask bit `m` is declared outside the half loop, so group `(half, j)` uses bit
  `4*half + j`; Q5_K's masks walk `u1 = 1,4,16,64` and `u2 = 2,8,32,128`.
* Q6_K: element `128h + 32q + l` uses scale `8h + 2q + l/16`, which is just `element / 16`.
* Q4_K/Q5_K: element `64j + 32s + l` belongs to group `2j + s = element / 32`.

## Form template identifiers
`get_scale_min_k4` → `scaleMinK4`, Q3_K repacked scale → `q3kScale`, IQ4_XS `ls` → `iq4xsLs`,
Q3_K canonical code → `q3kCanonCode`, hmask selector → `q3kMaskBit`, Q5_K masks → `q5kU1`/`q5kU2`,
Q6_K code → `q6kCode`.
-/

namespace Strata.KQuant

-- § super-block sizes

/-- Q3_K: hmask[32] + qs[64] + scales[12] + fp16 d. -/
def q3kBytes : Nat := 32 + 64 + 12 + 2
/-- Q4_K: fp16 d + fp16 dmin + scales[12] + qs[128]. -/
def q4kBytes : Nat := 2 + 2 + 12 + 128
/-- Q5_K: fp16 d + fp16 dmin + scales[12] + qh[32] + qs[128]. -/
def q5kBytes : Nat := 2 + 2 + 12 + 32 + 128
/-- Q6_K: ql[128] + qh[64] + scales[16] + fp16 d (last). -/
def q6kBytes : Nat := 128 + 64 + 16 + 2
/-- IQ4_XS: fp16 d + uint16 scales_h + scales_l[4] + qs[128]. -/
def iq4xsBytes : Nat := 2 + 2 + 4 + 128

/-- The super-block sizes agree with `block_geometry`. -/
theorem superblock_bytes :
    q3kBytes = 110 ∧ q4kBytes = 144 ∧ q5kBytes = 176 ∧ q6kBytes = 210 ∧ iq4xsBytes = 136 := by
  decide

/-- Field offsets used by the C++: Q6_K's `d` at 208, Q3_K's `d` at 108 and scales at 96, Q5_K's
`qs` at 48. -/
theorem field_offsets : 128 + 64 + 16 = 208 ∧ 32 + 64 + 12 = 108 ∧ 32 + 64 = 96 ∧ 2 + 2 + 12 + 32 = 48 := by
  decide

-- § bit helpers

/-- A low-six-bits mask is below 64. -/
theorem and63_lt (x : Nat) : x &&& 63 < 64 := by
  have h : x &&& 63 ≤ 63 := Nat.and_le_right
  omega

/-- Combining a value below 16 with a 2-bit value shifted to bit 4 yields a 6-bit value. -/
theorem six_bit_combine {lo hi : Nat} (hlo : lo < 16) (hhi : hi < 4) : lo ||| (hi <<< 4) < 64 := by
  have h1 : lo < 2 ^ 6 := by omega
  have h2 : hi <<< 4 < 2 ^ 6 := by rw [Nat.shiftLeft_eq]; omega
  exact Nat.or_lt_two_pow h1 h2

/-- The top two bits of a byte are below 4. -/
theorem top2_lt {b : Nat} (hb : b < 256) : b >>> 6 < 4 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by decide)]; omega

/-- The high nibble of a byte is below 16. -/
theorem hi_nib_lt {b : Nat} (hb : b < 256) : b >>> 4 < 16 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.div_lt_iff_lt_mul (by decide)]; omega

/-- The low nibble of anything is below 16. -/
theorem lo_nib_lt (x : Nat) : x &&& 15 < 16 := by
  have h : x &&& 15 ≤ 15 := Nat.and_le_right
  omega

/-- Two bits of anything are below 4. -/
theorem two_bits_lt (x : Nat) : x &&& 3 < 4 := by
  have h : x &&& 3 ≤ 3 := Nat.and_le_right
  omega

-- § get_scale_min_k4

/-- `get_scale_min_k4(j, q)` returning `(scale, min)` from the 12 packed bytes `q`. -/
def scaleMinK4 (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j &&& 63, q (j + 4) &&& 63)
  else ((q (j + 4) &&& 15) ||| ((q (j - 4) >>> 6) <<< 4),
        (q (j + 4) >>> 4) ||| ((q j >>> 6) <<< 4))

/-- For byte-valued inputs both the scale and the min of every group are 6-bit. -/
theorem scaleMinK4_lt_64 (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  by_cases hj : j < 4
  · simp only [hj, if_true]
    exact ⟨and63_lt _, and63_lt _⟩
  · simp only [hj, if_false]
    exact ⟨six_bit_combine (lo_nib_lt _) (top2_lt (hq _)),
           six_bit_combine (hi_nib_lt (hq _)) (top2_lt (hq _))⟩

/-- In the low half the scale is the byte's own low 6 bits: reading `q[j]` alone is correct only
there, which is the trap the source comment names. -/
theorem scaleMinK4_low (q : Nat → Nat) {j : Nat} (hj : j < 4) : (scaleMinK4 q j).1 = q j &&& 63 := by
  unfold scaleMinK4; simp [hj]

-- § Q3_K

/-- Q3_K scale of group `g` (0..15) from the 12 scale bytes `s`, as `unpack_q3_k_scales` writes it. -/
def q3kScale (s : Nat → Nat) (g : Nat) : Nat :=
  let j := g % 4
  let t := s (8 + j)
  match g / 4 with
  | 0 => (s j &&& 15) ||| ((t &&& 3) <<< 4)
  | 1 => (s (4 + j) &&& 15) ||| (((t >>> 2) &&& 3) <<< 4)
  | 2 => ((s j >>> 4) &&& 15) ||| (((t >>> 4) &&& 3) <<< 4)
  | _ => ((s (4 + j) >>> 4) &&& 15) ||| (((t >>> 6) &&& 3) <<< 4)

/-- Every repacked Q3_K scale is 6-bit, for any scale bytes. -/
theorem q3kScale_lt_64 (s : Nat → Nat) (g : Nat) : q3kScale s g < 64 := by
  unfold q3kScale
  split
  · exact six_bit_combine (lo_nib_lt _) (two_bits_lt _)
  · exact six_bit_combine (lo_nib_lt _) (two_bits_lt _)
  · exact six_bit_combine (lo_nib_lt _) (two_bits_lt _)
  · exact six_bit_combine (lo_nib_lt _) (two_bits_lt _)

/-- The signed Q3_K scale `sc - 32` lies in `-32..31`. -/
theorem q3k_signed_scale (s : Nat → Nat) (g : Nat) :
    -32 ≤ (q3kScale s g : Int) - 32 ∧ (q3kScale s g : Int) - 32 ≤ 31 := by
  have := q3kScale_lt_64 s g
  omega

/-- Q3_K source symbol: `c2 - (hbit ? 0 : 4)`. -/
def q3kSourceSymbol (c2 : Nat) (hbit : Bool) : Int := (c2 : Int) - (if hbit then 0 else 4)
/-- Q3_K canonical code: `c2 + 4 * hbit`, stored unsigned. -/
def q3kCanonCode (c2 : Nat) (hbit : Bool) : Nat := c2 + (if hbit then 4 else 0)

/-- The canonical code with integer bias `-4` decodes to the source symbol for every input. -/
theorem q3k_canonical_bias (c2 : Nat) (hbit : Bool) :
    (q3kCanonCode c2 hbit : Int) - 4 = q3kSourceSymbol c2 hbit := by
  unfold q3kCanonCode q3kSourceSymbol
  cases hbit <;> simp <;> omega

/-- Canonical Q3_K codes fit in 3 bits (`0..7`), inside a 4-bit S4 plane. -/
theorem q3kCanonCode_lt {c2 : Nat} (hc : c2 < 4) (hbit : Bool) : q3kCanonCode c2 hbit < 8 := by
  unfold q3kCanonCode; cases hbit <;> simp <;> omega

/-- The hmask bit used by group `(half, j)`: `m` is never reset, so it is bit `4*half + j`. -/
def q3kMaskBit (half j : Nat) : Nat := 4 * half + j

/-- The eight `(half, j)` pairs use eight distinct hmask bits (the second half uses 4..7, so it does
NOT re-read the first half's bits). -/
theorem q3kMaskBit_injective {h1 j1 h2 j2 : Nat} (hh1 : h1 < 2) (hh2 : h2 < 2) (hj1 : j1 < 4)
    (hj2 : j2 < 4) (e : q3kMaskBit h1 j1 = q3kMaskBit h2 j2) : h1 = h2 ∧ j1 = j2 := by
  unfold q3kMaskBit at e; omega

/-- Q3_K element order `128*half + 32*j + 16*sub + l` belongs to group `8*half + 2*j + sub`, which
is `element / 16`: consecutive 16-element groups. -/
theorem q3k_group_of_element {half j sub l : Nat} (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    (128 * half + 32 * j + 16 * sub + l) / 16 = 8 * half + 2 * j + sub := by
  omega

-- § Q4_K / Q5_K

/-- Q4_K/Q5_K element `64*j + 32*s + l` belongs to group `2*j + s = element / 32`: 8 groups of 32. -/
theorem q4k_group_of_element {j s l : Nat} (hs : s < 2) (hl : l < 32) :
    (64 * j + 32 * s + l) / 32 = 2 * j + s := by
  omega

/-- Q5_K low-nibble mask for step `j`: `1 << 2j`. -/
def q5kU1 (j : Nat) : Nat := 1 <<< (2 * j)
/-- Q5_K high-nibble mask for step `j`: `2 << 2j`. -/
def q5kU2 (j : Nat) : Nat := 2 <<< (2 * j)

/-- The masks walk 1,4,16,64 and 2,8,32,128 - shifting by 2, not by 1. -/
theorem q5k_mask_values :
    [q5kU1 0, q5kU1 1, q5kU1 2, q5kU1 3] = [1, 4, 16, 64] ∧
    [q5kU2 0, q5kU2 1, q5kU2 2, q5kU2 3] = [2, 8, 32, 128] := by decide

/-- Across the four steps the eight masks are disjoint and together cover all 8 bits of `qh`. -/
theorem q5k_masks_partition_byte :
    q5kU1 0 ||| q5kU2 0 ||| q5kU1 1 ||| q5kU2 1 ||| q5kU1 2 ||| q5kU2 2 ||| q5kU1 3 ||| q5kU2 3 = 255 ∧
    q5kU1 0 + q5kU2 0 + q5kU1 1 + q5kU2 1 + q5kU1 2 + q5kU2 2 + q5kU1 3 + q5kU2 3 = 255 := by decide

/-- The Q5_K code `nibble + (bit ? 16 : 0)` is a 5-bit value. -/
theorem q5k_code_lt {n : Nat} (hn : n < 16) (b : Bool) : n + (if b then 16 else 0) < 32 := by
  cases b <;> simp <;> omega

-- § Q6_K

/-- Q6_K 6-bit code from a nibble and two high bits. -/
def q6kCode (nib hi2 : Nat) : Nat := nib ||| (hi2 <<< 4)

/-- Q6_K codes are 6-bit and their symbols `code - 32` cover `-32..31`. -/
theorem q6k_symbol_range {n h : Nat} (hn : n < 16) (hh : h < 4) :
    q6kCode n h < 64 ∧ -32 ≤ (q6kCode n h : Int) - 32 ∧ (q6kCode n h : Int) - 32 ≤ 31 := by
  have := six_bit_combine hn hh
  unfold q6kCode; omega

/-- Q6_K scale index: element `128*h + 32*q + l` (quad `q`, `l < 32`) uses `sc[8*h + 2*q + l/16]`,
which equals `element / 16`, i.e. plain consecutive 16-element groups. -/
theorem q6k_scale_index {h q l : Nat} (hq : q < 4) (hl : l < 32) :
    (128 * h + 32 * q + l) / 16 = 8 * h + l / 16 + 2 * q := by
  omega

/-- Within one half the four quads land at strides of 32 and never collide. -/
theorem q6k_quads_disjoint {q1 l1 q2 l2 : Nat} (hl1 : l1 < 32) (hl2 : l2 < 32)
    (e : 32 * q1 + l1 = 32 * q2 + l2) : q1 = q2 ∧ l1 = l2 := by
  omega

-- § IQ4_XS

/-- IQ4_XS 6-bit group scale: low nibble of `scales_l[ib/2]` (by `ib % 2`) and two bits of
`scales_h` at `2*ib`. -/
def iq4xsLs (scalesL : Nat → Nat) (scalesH ib : Nat) : Nat :=
  ((scalesL (ib / 2) >>> (4 * (ib % 2))) &&& 15) ||| (((scalesH >>> (2 * ib)) &&& 3) <<< 4)

/-- Every IQ4_XS group scale is 6-bit and its signed form `ls - 32` lies in `-32..31`. -/
theorem iq4xsLs_range (scalesL : Nat → Nat) (scalesH ib : Nat) :
    iq4xsLs scalesL scalesH ib < 64 ∧
    -32 ≤ (iq4xsLs scalesL scalesH ib : Int) - 32 ∧ (iq4xsLs scalesL scalesH ib : Int) - 32 ≤ 31 := by
  have := six_bit_combine (lo_nib_lt (scalesL (ib / 2) >>> (4 * (ib % 2))))
    (two_bits_lt (scalesH >>> (2 * ib)))
  unfold iq4xsLs; omega

/-- The 8 groups take two `scales_h` bits each at offsets 0,2,...,14: the 16-bit field is fully used
and no two groups share a bit. -/
theorem iq4xs_scalesH_offsets {a b : Nat} (ha : a < 8) (hb : b < 8) (e : 2 * a = 2 * b) : a = b := by
  omega

end Strata.KQuant
