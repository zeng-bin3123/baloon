/-!
# MODULE: block_dequantization
SOURCES: include/strata/artifact/dequant.hpp
Scalar reference dequantizers for the GGUF block encodings in the artifact (Q2_0, Q8_0, Q4_0, Q5_0,
IQ4_NL, Q6_K, Q4_K, Q5_K, Q3_K, IQ4_XS, F32/F16/BF16).  The theory pins down byte layouts, the
element -> (byte, bit, scale) index maps and the integer code ranges; float arithmetic is modelled
over `Int` scales where only the structural shape matters.

## Source semantics
* Q2_0: 64 elements / 18 bytes; code j sits in byte j/4 at bits 2(j%4) (LSB first); symbol = code - 1
  in {-1,0,1,2}; every output equals one of {-d, 0, d, 2d} (the `--check` structural invariant).
* Q4_0 / IQ4_NL: low nibble -> element j, high nibble -> element j+16.  Q5_0 adds a 5th bit.
* IQ4_NL codebook is strictly increasing and fits in int8.
* Q6_K and Q3_K: the scale applied to output position p is scale index p/16; Q4_K/Q5_K: p/32.
* Q5_K's 5th-bit masks are 4^g and 2*4^g (shift left by 2 per group); Q3_K's high-bit mask is NOT
  reset between the two 128-halves, so the 8 (half, j) steps use the 8 distinct bits 1..128.
* `get_scale_min_k4` yields 6-bit values and only reads bytes 0..11; Q3_K and IQ4_XS scales are
  6-bit with a -32 bias, so the signed scale is in [-32, 31].

## Form template identifiers
dequantize_q2_0 -> q2Value, code extraction -> q2Code, get_scale_min_k4 -> scaleMinK4,
dequantize_q6_K positions -> q6kPos, Q5_K masks -> q5kMask, Q3_K mask -> q3kMask,
kvalues_iq4nl -> iq4nlTable, IQ4_XS scale -> iq4xsScale
-/

namespace Strata.Dequant

-- § Q2_0 packing

/-- Divisor that selects slot `k` of a byte (`>> (k*2)`). -/
def slotDiv : Nat → Nat
  | 0 => 1
  | 1 => 4
  | 2 => 16
  | _ => 64

/-- The 2-bit code of element `j` read from byte array `qs`, exactly as `dequantize_q2_0` does. -/
def q2Code (qs : Nat → Nat) (j : Nat) : Nat := qs (j / 4) / slotDiv (j % 4) % 4

/-- A byte built from four codes, slot 0 in the lowest bits. -/
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Packing four codes into a byte and reading slot `k` back yields code `k` (LSB-first order). -/
theorem unpack_pack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 / slotDiv 0 % 4 = c0 ∧ packByte c0 c1 c2 c3 / slotDiv 1 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / slotDiv 2 % 4 = c2 ∧ packByte c0 c1 c2 c3 / slotDiv 3 % 4 = c3 := by
  simp only [packByte, slotDiv]
  omega

/-- A packed byte is a byte. -/
theorem packByte_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 := by
  unfold packByte; omega

/-- For the 64 elements of a block, the byte index is one of the 16 code bytes, the bit offset is
at most 6, and (byte, slot) determines the element: the index map is injective. -/
theorem q2_index_map (j : Nat) (hj : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 ∧ j = 4 * (j / 4) + j % 4 := by omega

/-- Every code is in `{0,1,2,3}`. -/
theorem q2Code_lt (qs : Nat → Nat) (j : Nat) : q2Code qs j < 4 := by
  unfold q2Code; exact Nat.mod_lt _ (by decide)

/-- The symbol `code - 1` is one of `{-1, 0, 1, 2}`. -/
theorem q2_symbol_range (c : Nat) (hc : c < 4) :
    (c : Int) - 1 = -1 ∨ (c : Int) - 1 = 0 ∨ (c : Int) - 1 = 1 ∨ (c : Int) - 1 = 2 := by
  omega

/-- The dequantized value `(code - 1) * d`. -/
def q2Value (c : Nat) (d : Int) : Int := ((c : Int) - 1) * d

/-- The `--check` structural invariant: every Q2_0 output is one of `{-d, 0, d, 2d}`. -/
theorem q2_value_structural (qs : Nat → Nat) (j : Nat) (d : Int) :
    q2Value (q2Code qs j) d = -d ∨ q2Value (q2Code qs j) d = 0 ∨
    q2Value (q2Code qs j) d = d ∨ q2Value (q2Code qs j) d = 2 * d := by
  have hc := q2Code_lt qs j
  generalize q2Code qs j = c at *
  unfold q2Value
  rcases q2_symbol_range c hc with h | h | h | h <;> rw [h] <;> simp [Int.neg_one_mul]

-- § 32-element nibble encodings (Q4_0, Q5_0, Q8_0, IQ4_NL)

/-- A byte splits losslessly into its low and high nibble. -/
theorem nibble_split (b : Nat) (hb : b < 256) : b = b % 16 + 16 * (b / 16) ∧ b / 16 < 16 := by omega

/-- Q4_0 symbols `nibble - 8` lie in `[-8, 7]`. -/
theorem q4_0_range (b : Nat) (hb : b < 256) :
    -8 ≤ ((b % 16 : Nat) : Int) - 8 ∧ ((b % 16 : Nat) : Int) - 8 ≤ 7 ∧
    -8 ≤ ((b / 16 : Nat) : Int) - 8 ∧ ((b / 16 : Nat) : Int) - 8 ≤ 7 := by omega

/-- Q5_0 assembles `nibble | (bit << 4)`: a 5-bit value, and the symbol `x - 16` is in `[-16, 15]`. -/
theorem q5_0_range (b bit : Nat) (hbit : bit < 2) :
    b % 16 + 16 * bit < 32 ∧ -16 ≤ ((b % 16 + 16 * bit : Nat) : Int) - 16 ∧
    ((b % 16 + 16 * bit : Nat) : Int) - 16 ≤ 15 := by omega

/-- Low-nibble element `j` and high-nibble element `j + 16` (j < 16) cover the 32 outputs exactly once. -/
theorem nibble_positions (p : Nat) (hp : p < 32) :
    (p < 16 ∧ p % 16 = p) ∨ (16 ≤ p ∧ p % 16 + 16 = p) := by omega

/-- The IQ4_NL codebook, verbatim from ggml-common.h. -/
def iq4nlTable : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Strictly increasing check on a list. -/
def strictlyIncreasing : List Int → Bool
  | a :: b :: t => decide (a < b) && strictlyIncreasing (b :: t)
  | _ => true

/-- The codebook has 16 entries, is strictly increasing (so it is a non-linear but injective code),
and every entry fits in `int8`. -/
theorem iq4nl_table_wellformed :
    iq4nlTable.length = 16 ∧ strictlyIncreasing iq4nlTable = true ∧
    iq4nlTable.all (fun v => decide (-128 ≤ v ∧ v ≤ 127)) = true := by decide

-- § Q6_K

/-- Output position of the `k`-th of the four values produced at `(half n, l)`. -/
def q6kPos (n l k : Nat) : Nat := 128 * n + l + 32 * k

/-- The scale index used: `8n` (the `sc += 8` advance) plus `is + 2k` with `is = l / 16`. -/
def q6kScale (n l k : Nat) : Nat := 8 * n + l / 16 + 2 * k

/-- Q6_K positions are in range and decode back to `(n, l, k)`, so all 256 outputs are written once. -/
theorem q6k_pos_bijective (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    q6kPos n l k < 256 ∧ q6kPos n l k / 128 = n ∧ q6kPos n l k % 128 / 32 = k ∧ q6kPos n l k % 32 = l := by
  unfold q6kPos; omega

/-- The scale applied to Q6_K output `p` is scale index `p / 16`: each of the 16 scales covers 16
consecutive outputs. -/
theorem q6k_scale_is_pos_div16 (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    q6kScale n l k = q6kPos n l k / 16 ∧ q6kScale n l k < 16 := by
  unfold q6kScale q6kPos; omega

/-- Q6_K byte reads stay inside `ql[128]`, `qh[64]`: ql byte `64n + l + 32(k%2)`, qh byte `32n + l`. -/
theorem q6k_reads_in_bounds (n l k : Nat) (hn : n < 2) (hl : l < 32) (hk : k < 4) :
    64 * n + l + 32 * (k % 2) < 128 ∧ 32 * n + l < 64 := by omega

/-- A Q6_K symbol `low4 | (high2 << 4)` minus 32 lies in `[-32, 31]`. -/
theorem q6k_symbol_range (lo hi : Nat) (hlo : lo < 16) (hhi : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by omega

-- § Q4_K / Q5_K scale-min unpacking

/-- `get_scale_min_k4` over the 12 scale bytes `q`, returning `(scale, min)`. -/
def scaleMinK4 (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j % 64, q (j + 4) % 64)
  else (q (j + 4) % 16 + 16 * (q (j - 4) / 64), q (j + 4) / 16 + 16 * (q j / 64))

/-- For `j < 8` the unpacker reads only bytes `0..11`. -/
theorem scaleMinK4_reads (j : Nat) (hj : j < 8) : j + 4 < 12 ∧ j < 12 ∧ j - 4 < 12 := by omega

/-- With byte inputs, both outputs are 6-bit. -/
theorem scaleMinK4_six_bit (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  have h1 := hq (j + 4)
  have h2 := hq (j - 4)
  have h3 := hq j
  split <;> simp <;> omega

/-- Q4_K/Q5_K: element `64g + 32h + l` uses code byte `32g + l`, nibble `h`, and scale pair
`2g + h`, i.e. scale index = position / 32. -/
theorem q4k_index_map (g h l : Nat) (hg : g < 4) (hh : h < 2) (hl : l < 32) :
    64 * g + 32 * h + l < 256 ∧ 32 * g + l < 128 ∧ 2 * g + h = (64 * g + 32 * h + l) / 32 := by omega

/-- Q5_K's 5th-bit masks shift left by 2 per group: `u1 = 4^g`, `u2 = 2 * 4^g`. -/
def q5kMask1 (g : Nat) : Nat := 4 ^ g
/-- See `q5kMask1`. -/
def q5kMask2 (g : Nat) : Nat := 2 * 4 ^ g

/-- Over the four groups the eight masks are exactly the eight bits of a byte, each once. -/
theorem q5k_masks_are_bits :
    [q5kMask1 0, q5kMask2 0, q5kMask1 1, q5kMask2 1, q5kMask1 2, q5kMask2 2, q5kMask1 3, q5kMask2 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- A Q5_K quant `nibble + (bit ? 16 : 0)` is 5-bit. -/
theorem q5k_quant_range (b : Nat) (bit : Bool) : b % 16 + (if bit then 16 else 0) < 32 := by
  cases bit <;> simp <;> omega

-- § Q3_K

/-- The high-bit mask at step `(half, j)`: hoisted out of the half loop, so it keeps shifting. -/
def q3kMask (half j : Nat) : Nat := 2 ^ (4 * half + j)

/-- The eight steps use eight distinct mask bits 1..128 (the mask is not reset between halves). -/
theorem q3k_mask_not_reset :
    [q3kMask 0 0, q3kMask 0 1, q3kMask 0 2, q3kMask 0 3, q3kMask 1 0, q3kMask 1 1, q3kMask 1 2, q3kMask 1 3] =
      [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- A Q3_K symbol `low2 - (hbit ? 0 : 4)` lies in `[-4, 3]` (the correction is negative). -/
theorem q3k_symbol_range (lo : Nat) (hlo : lo < 4) (hbit : Bool) :
    -4 ≤ (lo : Int) - (if hbit then 0 else 4) ∧ (lo : Int) - (if hbit then 0 else 4) ≤ 3 := by
  cases hbit <;> simp <;> omega

/-- Q3_K output `128 half + 32 j + 16 s + l` (s = which of the two 16-runs) uses scale
`8 half + 2 j + s`, which is position / 16. -/
theorem q3k_scale_is_pos_div16 (half j s l : Nat) (hh : half < 2) (hj : j < 4) (hs : s < 2) (hl : l < 16) :
    8 * half + 2 * j + s = (128 * half + 32 * j + 16 * s + l) / 16 ∧ 128 * half + 32 * j + 16 * s + l < 256 := by
  omega

/-- A repacked Q3_K scale byte `(a & 0x0F) | ((t & 3) << 4)` is 6-bit; minus 32 it lies in `[-32, 31]`. -/
theorem q3k_scale_range (a t : Nat) :
    a % 16 + 16 * (t % 4) < 64 ∧ -32 ≤ ((a % 16 + 16 * (t % 4) : Nat) : Int) - 32 ∧
    ((a % 16 + 16 * (t % 4) : Nat) : Int) - 32 ≤ 31 := by omega

-- § IQ4_XS

/-- IQ4_XS group scale: low nibble of `scales_l[ib/2]` selected by `ib % 2`, plus two bits of the
16-bit `scales_h` at shift `2 ib`. -/
def iq4xsScale (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  sl (ib / 2) / (if ib % 2 = 0 then 1 else 16) % 16 + 16 * (sh / 4 ^ ib % 4)

/-- The scale is 6-bit and the reads stay in `scales_l[4]` and the 16 bits of `scales_h`. -/
theorem iq4xs_scale_range (sl : Nat → Nat) (sh ib : Nat) (hib : ib < 8) :
    iq4xsScale sl sh ib < 64 ∧ ib / 2 < 4 ∧ 2 * ib + 1 < 16 := by
  unfold iq4xsScale
  omega

-- § byte budgets

/-- Field sizes read by each dequantizer sum to its block size:
Q2_0 18, Q8_0 34, Q4_0 18, Q5_0 22, IQ4_NL 18, Q6_K 210 (d at 208), Q4_K 144, Q5_K 176,
Q3_K 110 (d at 108), IQ4_XS 136. -/
theorem block_byte_budgets :
    2 + 64 / 4 = 18 ∧ 2 + 32 = 34 ∧ 2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧
    128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 ∧ 2 + 2 + 12 + 128 = 144 ∧
    2 + 2 + 12 + 32 + 128 = 176 ∧ 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 + 12 = 108 ∧
    2 + 2 + 4 + 128 = 136 := by decide

end Strata.Dequant
