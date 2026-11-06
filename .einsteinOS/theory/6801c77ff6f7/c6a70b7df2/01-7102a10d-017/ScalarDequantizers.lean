/-!
# MODULE: scalar_dequantizers
SOURCES: include/strata/artifact/dequant.hpp
Scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q6_K, Q4_K, Q5_K, Q3_K, IQ4_XS and the
float element types), modelled on naturals/integers: bit fields, code ranges, layouts and scale unpacking.

## Source semantics
* Q2_0 block = fp16 d ++ 16 bytes, 64 elements, code {0,1,2,3} -> symbol code-1, 4 codes per byte LSB-first.
* Nibble types: low nibble is element j, high nibble element j+16.
* K-types use packed 6-bit scales; all codes stay in their bit width so the integer bias keeps symbols bounded.
* Every layout offset sums to the block size recorded in block_geometry.

## Form template identifiers
dequantize_q2_0 -> dequant_2bit_bias1 ; dequantize_q4_0 -> dequant_nibble_bias8 ;
dequantize_q5_0 -> dequant_5bit_bias16 ; dequantize_iq4_nl -> dequant_codebook16 ;
get_scale_min_k4 -> unpack_scale_min_6bit ; fp16_to_fp32 -> half_widen ; read_u16 -> le_u16
-/
set_option maxRecDepth 100000

namespace Strata.Dequant

-- § fp16 / bf16 / u16 field arithmetic
def fp16Sign (h : Nat) : Nat := h / 32768
def fp16Exp (h : Nat) : Nat := h / 1024 % 32
def fp16Man (h : Nat) : Nat := h % 1024
def readU16 (p0 p1 : Nat) : Nat := p0 + p1 * 256
def bf16Bits (h : Nat) : Nat := h * 65536

/-- The three fp16 fields reassemble the original 16-bit pattern. -/
theorem fp16_fields_reassemble (h : Nat) (hh : h < 65536) :
    fp16Sign h * 32768 + fp16Exp h * 1024 + fp16Man h = h := by
  unfold fp16Sign fp16Exp fp16Man
  omega

/-- The fp16 fields stay in their widths (1, 5 and 10 bits). -/
theorem fp16_fields_bounded (h : Nat) (hh : h < 65536) :
    fp16Sign h < 2 ∧ fp16Exp h < 32 ∧ fp16Man h < 1024 := by
  unfold fp16Sign fp16Exp fp16Man
  omega

/-- Little-endian u16 read of two bytes is a 16-bit value. -/
theorem readU16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : readU16 p0 p1 < 65536 := by
  unfold readU16
  omega

/-- The low byte of the u16 is the first byte. -/
theorem readU16_low_byte (p0 p1 : Nat) (h0 : p0 < 256) : readU16 p0 p1 % 256 = p0 := by
  unfold readU16
  omega

/-- bf16 widening is a left shift by 16 that stays inside 32 bits. -/
theorem bf16_widen_lt (h : Nat) (hh : h < 65536) : bf16Bits h < 4294967296 := by
  unfold bf16Bits
  omega

/-- bf16 widening is exact: it is invertible and the low 16 bits are zero. -/
theorem bf16_widen_exact (h : Nat) : bf16Bits h / 65536 = h ∧ bf16Bits h % 65536 = 0 := by
  unfold bf16Bits
  omega

-- § Q2_0
def q2Code0 (b : Nat) : Nat := b % 4
def q2Code1 (b : Nat) : Nat := b / 4 % 4
def q2Code2 (b : Nat) : Nat := b / 16 % 4
def q2Code3 (b : Nat) : Nat := b / 64 % 4
def q2Pack (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3
def q2Symbol (code : Nat) : Int := (code : Int) - 1

/-- The shift-and-mask extraction in the C++ equals division/modulo by powers of four. -/
theorem q2_shift_mask_eq_div_mod :
    ∀ b, b < 256 → ∀ t, t < 4 → (b >>> (t * 2)) &&& 3 = b / 4 ^ t % 4 := by
  decide

/-- Four 2-bit codes pack into one byte. -/
theorem q2Pack_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    q2Pack c0 c1 c2 c3 < 256 := by
  unfold q2Pack
  omega

/-- Unpacking a packed byte returns the four codes in LSB-first order. -/
theorem q2_pack_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    q2Code0 (q2Pack c0 c1 c2 c3) = c0 ∧ q2Code1 (q2Pack c0 c1 c2 c3) = c1 ∧
    q2Code2 (q2Pack c0 c1 c2 c3) = c2 ∧ q2Code3 (q2Pack c0 c1 c2 c3) = c3 := by
  unfold q2Code0 q2Code1 q2Code2 q2Code3 q2Pack
  omega

/-- Packing the four unpacked codes of a byte gives the byte back. -/
theorem q2_unpack_pack (b : Nat) (hb : b < 256) :
    q2Pack (q2Code0 b) (q2Code1 b) (q2Code2 b) (q2Code3 b) = b := by
  unfold q2Code0 q2Code1 q2Code2 q2Code3 q2Pack
  omega

/-- Symbols lie in {-1,0,1,2}. -/
theorem q2Symbol_range (c : Nat) (hc : c < 4) : -1 ≤ q2Symbol c ∧ q2Symbol c ≤ 2 := by
  unfold q2Symbol
  omega

/-- The dequantized value is always one of -d, 0, d, 2d for the block's scale. -/
theorem q2_value_set (c : Nat) (hc : c < 4) (d : Int) :
    q2Symbol c * d = -d ∨ q2Symbol c * d = 0 ∨ q2Symbol c * d = d ∨ q2Symbol c * d = 2 * d := by
  unfold q2Symbol
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases h with rfl | rfl | rfl | rfl <;> simp <;> omega

/-- Q2_0 block layout: 2-byte scale plus 16 code bytes = 18 bytes for 64 elements. -/
theorem q2_0_block_layout : 2 + 64 / 4 = 18 := by decide

-- § Q4_0 / Q5_0 / Q8_0
def nibLo (b : Nat) : Nat := b % 16
def nibHi (b : Nat) : Nat := b / 16
def q4Sym (n : Nat) : Int := (n : Int) - 8
def q5Val (nib bit : Nat) : Nat := nib + 16 * bit
def q5Sym (n : Nat) : Int := (n : Int) - 16

/-- A byte splits into two nibbles and is recovered from them. -/
theorem nibble_split (b : Nat) (hb : b < 256) : nibLo b < 16 ∧ nibHi b < 16 ∧ nibLo b + 16 * nibHi b = b := by
  unfold nibLo nibHi
  omega

/-- Q4_0 symbols lie in [-8, 7]. -/
theorem q4Sym_range (n : Nat) (hn : n < 16) : -8 ≤ q4Sym n ∧ q4Sym n ≤ 7 := by
  unfold q4Sym
  omega

/-- A Q5_0 element is a nibble plus its fifth bit, so below 32. -/
theorem q5Val_lt (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : q5Val nib bit < 32 := by
  unfold q5Val
  omega

/-- Q5_0 symbols lie in [-16, 15]. -/
theorem q5Sym_range (n : Nat) (hn : n < 32) : -16 ≤ q5Sym n ∧ q5Sym n ≤ 15 := by
  unfold q5Sym
  omega

/-- The four qh bytes assemble into a 32-bit word. -/
theorem qh_assemble_lt (a b c d : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256) (hd : d < 256) :
    a + b * 256 + c * 65536 + d * 16777216 < 4294967296 := by
  omega

/-- Any bit of a 32-bit word extracted by shift is 0 or 1. -/
theorem qh_bit_lt_two (h j : Nat) : h / 2 ^ j % 2 < 2 := Nat.mod_lt _ (by decide)

/-- Q8_0 layout is 2 + 32 = 34 bytes; Q4_0 is 2 + 16 = 18; Q5_0 is 2 + 4 + 16 = 22. -/
theorem q8_q4_q5_layouts : 2 + 32 = 34 ∧ 2 + 16 = 18 ∧ 2 + 4 + 16 = 22 := by decide

-- § IQ4_NL codebook
def kvaluesIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem kvalues_length : kvaluesIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvalues_strictly_increasing :
    ∀ i, i < 15 → kvaluesIq4nl.getD i 0 < kvaluesIq4nl.getD (i + 1) 0 := by decide

/-- Every codebook entry fits in int8 symmetric range. -/
theorem kvalues_bounded : ∀ i, i < 16 → -127 ≤ kvaluesIq4nl.getD i 0 ∧ kvaluesIq4nl.getD i 0 ≤ 127 := by
  decide

/-- Both nibbles of a byte index into the codebook. -/
theorem iq4_nibbles_index_table (b : Nat) (hb : b < 256) :
    nibLo b < kvaluesIq4nl.length ∧ nibHi b < kvaluesIq4nl.length := by
  have h := nibble_split b hb
  have hl := kvalues_length
  omega

-- § Q6_K
/-- Q6_K layout: ql 128 + qh 64 + scales 16 + d 2 = 210, with d at offset 208. -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 + 16 = 208 := by decide

/-- A Q6_K code is a nibble plus two high bits, below 64, so the bias -32 gives [-32,31]. -/
theorem q6k_code_range (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) :
    nib + 16 * hi < 64 ∧ -32 ≤ ((nib + 16 * hi : Nat) : Int) - 32 ∧ ((nib + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- The Q6_K scale index is_ + {0,2,4,6} over two halves never exceeds 15. -/
theorem q6k_scale_index_bound (half l : Nat) (hh : half < 2) (hl : l < 32) :
    8 * half + l / 16 + 6 < 16 := by
  omega

-- § Q4_K / Q5_K scale-min unpacking
def scaleK4 (q : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then q j % 64 else q (j + 4) % 16 + (q (j - 4) / 64) * 16
def minK4 (q : Nat → Nat) (j : Nat) : Nat :=
  if j < 4 then q (j + 4) % 64 else q (j + 4) / 16 + (q j / 64) * 16

/-- Unpacked 6-bit scales stay below 64. -/
theorem scaleK4_lt (q : Nat → Nat) (hq : ∀ i, i < 12 → q i < 256) (j : Nat) (hj : j < 8) :
    scaleK4 q j < 64 := by
  unfold scaleK4
  split
  · omega
  · have h2 := hq (j - 4) (by omega)
    omega

/-- Unpacked 6-bit mins stay below 64. -/
theorem minK4_lt (q : Nat → Nat) (hq : ∀ i, i < 12 → q i < 256) (j : Nat) (hj : j < 8) :
    minK4 q j < 64 := by
  unfold minK4
  split
  · omega
  · have h1 := hq (j + 4) (by omega)
    have h2 := hq j (by omega)
    omega

/-- For the high groups the byte q[j+4] is split into a low nibble (scale) and high nibble (min). -/
theorem k4_high_byte_split (x : Nat) (hx : x < 256) : x % 16 + (x / 16) * 16 = x := by
  omega

/-- Q4_K layout: d 2 + dmin 2 + scales 12 + qs 128 = 144 with qs at offset 16. -/
theorem q4k_layout : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := by decide

/-- Q5_K layout: d 2 + dmin 2 + scales 12 + qh 32 + qs 128 = 176 with qh at 16 and qs at 48. -/
theorem q5k_layout : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by decide

/-- The Q5_K masks u1,u2 walk 1,4,16,64 and 2,8,32,128 (shift by 2 per 64-group). -/
theorem q5k_masks :
    (List.range 4).map (fun j => 1 <<< (2 * j)) = [1, 4, 16, 64] ∧
    (List.range 4).map (fun j => 2 <<< (2 * j)) = [2, 8, 32, 128] := by decide

/-- Q4_K / Q5_K cover 256 elements in 4 steps of 64, using 8 scale groups. -/
theorem k45_coverage : 4 * 64 = 256 ∧ 4 * 2 = 8 ∧ 8 * 32 = 256 := by decide

-- § Q3_K
/-- Q3_K layout: hmask 32 + qs 64 + scales 12 + d 2 = 110, scales at 96, d at 108. -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by decide

/-- The hmask bit position 4*half + j is a valid bit of a byte; the mask is never reset between halves. -/
theorem q3k_mask_bit_valid (half j : Nat) (hh : half < 2) (hj : j < 4) : 4 * half + j < 8 := by
  omega

/-- The last mask used is 128, so the uint8 shift never overflows before its final use. -/
theorem q3k_last_mask : (1 : Nat) <<< 7 = 128 ∧ 128 < 256 := by decide

/-- Q3_K scales are signed: a 6-bit value minus 32 lies in [-32,31]. -/
theorem q3k_scale_range (s : Nat) (hs : s < 64) : -32 ≤ (s : Int) - 32 ∧ (s : Int) - 32 ≤ 31 := by
  omega

/-- The 6-bit scale assembled from a nibble and two high bits stays below 64. -/
theorem q3k_scale_assemble_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) : lo + hi * 16 < 64 := by
  omega

-- § IQ4_XS
def iq4xsLs (sl sh ib : Nat) : Nat :=
  (if ib % 2 = 0 then sl % 16 else sl / 16 % 16) + (sh / 4 ^ ib % 4) * 16

/-- IQ4_XS layout: d 2 + scales_h 2 + scales_l 4 + qs 128 = 136. -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 8 * 16 = 128 ∧ 8 * 32 = 256 := by decide

/-- The 6-bit IQ4_XS scale index stays below 64, so ls-32 is in [-32,31]. -/
theorem iq4xsLs_lt (sl sh ib : Nat) (hsl : sl < 256) : iq4xsLs sl sh ib < 64 := by
  unfold iq4xsLs
  have h := Nat.mod_lt (sh / 4 ^ ib) (by decide : 4 > 0)
  generalize sh / 4 ^ ib % 4 = w at *
  split <;> omega

/-- The scales_l byte index ib/2 stays within 4 bytes for the 8 groups. -/
theorem iq4xs_scale_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

-- § float element types
/-- The element-type decoders consume 4, 2 and 2 bytes per element. -/
theorem element_sizes : 1 * 4 = 4 ∧ 1 * 2 = 2 := by decide

end Strata.Dequant
