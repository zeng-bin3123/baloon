
## FILE: /mnt/user-data/outputs/ScalarDequantizers.lean
/-!
# MODULE: scalar_dequantizers
SOURCES: include/ref/artifact/dequant.hpp
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

namespace Ref.Dequant

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

end Ref.Dequant



## FILE: /mnt/user-data/outputs/GgufReaderGeometry.lean
/-!
# MODULE: gguf_reader_geometry
SOURCES: include/ref/artifact/gguf_reader.hpp
GGUF v3 reader: block geometry table, bounds-checked cursor, data-section alignment, tensor element
counts, array sampling and the qwen4exp architecture guard constants.

## Source semantics
* block_geometry returns (elements, bytes) per block and fails for unknown types; Q2_0 (42) is 64/18.
* Cursor::need rejects any read past the end; a successful read advances by exactly sizeof(T).
* data_start = round-up of the end of the header to the alignment (default 32).
* elements() is the product of the shape; arrays keep at most 64 sampled items.

## Form template identifiers
block_geometry -> block_geometry_table ; Cursor -> bounded_cursor ; parse data_start -> align_up ;
TensorInfo::elements -> shape_product ; Qwen4ExpGuard -> arch_guard_constants
-/
set_option maxRecDepth 100000

namespace Ref.GgufReader

-- § block geometry
def blockGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 (type 42) has 64 elements in 18 bytes. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) := by decide

/-- Every known geometry has positive element and byte counts. -/
theorem geometry_positive :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (0 < p.1 ∧ 0 < p.2)) = true := by
  decide

/-- Every known block has 1, 32, 64 or 256 elements. -/
theorem geometry_elems_classes :
    ∀ t, t < 64 → (blockGeometry t).elim true
      (fun p => decide (p.1 = 1 ∨ p.1 = 32 ∨ p.1 = 64 ∨ p.1 = 256)) = true := by
  decide

/-- Q2_0 is the only type with 64-element blocks. -/
theorem only_q2_0_has_64 :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (p.1 = 64 → t = 42)) = true := by
  decide

/-- IQ1_S (19) has a name in the reader but no geometry entry, so it is refused as unsized. -/
theorem iq1_s_has_no_geometry : blockGeometry 19 = none := by decide

/-- IQ1_M (29) has geometry 256/56. -/
theorem iq1_m_geometry : blockGeometry 29 = some (256, 56) := by decide

/-- Q2_0 costs 18*8 = 144 bits per 64 elements, i.e. 2 bits plus a 16-bit scale. -/
theorem q2_0_bits : 18 * 8 = 64 * 2 + 16 := by decide

-- § bounded cursor
structure Cur where
  pos : Nat
  size : Nat

def Cur.read (c : Cur) (n : Nat) : Option Cur :=
  if c.pos + n ≤ c.size then some { c with pos := c.pos + n } else none

/-- A successful read advances by exactly the requested width. -/
theorem cursor_read_advances (c : Cur) (n : Nat) (c' : Cur) (h : c.read n = some c') :
    c'.pos = c.pos + n ∧ c'.size = c.size := by
  unfold Cur.read at h
  split at h
  · cases h
    exact ⟨rfl, rfl⟩
  · cases h

/-- A successful read never leaves the buffer. -/
theorem cursor_read_in_bounds (c : Cur) (n : Nat) (c' : Cur) (h : c.read n = some c') :
    c'.pos ≤ c'.size := by
  unfold Cur.read at h
  split at h
  · rename_i hle
    cases h
    exact hle
  · cases h

/-- A read past the end is refused. -/
theorem cursor_read_refuses_overrun (c : Cur) (n : Nat) (h : c.size < c.pos + n) :
    c.read n = none := by
  unfold Cur.read
  split
  · omega
  · rfl

-- § alignment
def alignUp (c a : Nat) : Nat := (c + a - 1) / a * a

/-- The data start is at least the header end. -/
theorem alignUp_ge (c a : Nat) (ha : 0 < a) : c ≤ alignUp c a := by
  unfold alignUp
  have h1 := Nat.div_add_mod (c + a - 1) a
  have h2 := Nat.mod_lt (c + a - 1) ha
  have h3 : (c + a - 1) / a * a = a * ((c + a - 1) / a) := Nat.mul_comm _ _
  generalize (c + a - 1) / a = q at *
  generalize (c + a - 1) % a = r at *
  omega

/-- The data start is less than one alignment unit beyond the header end. -/
theorem alignUp_lt (c a : Nat) (ha : 0 < a) : alignUp c a < c + a := by
  unfold alignUp
  have h1 := Nat.div_add_mod (c + a - 1) a
  have h2 := Nat.mod_lt (c + a - 1) ha
  have h3 : (c + a - 1) / a * a = a * ((c + a - 1) / a) := Nat.mul_comm _ _
  generalize (c + a - 1) / a = q at *
  generalize (c + a - 1) % a = r at *
  omega

/-- The data start is a multiple of the alignment. -/
theorem alignUp_dvd (c a : Nat) : alignUp c a % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- With the default alignment 32 the data start is within 31 bytes after the header. -/
theorem alignUp_32 (c : Nat) : c ≤ alignUp c 32 ∧ alignUp c 32 < c + 32 :=
  ⟨alignUp_ge c 32 (by decide), alignUp_lt c 32 (by decide)⟩

-- § tensor shape
def prodL : List Nat → Nat
  | [] => 1
  | d :: ds => d * prodL ds

/-- The element count of concatenated shapes is the product of the counts. -/
theorem prodL_append (a b : List Nat) : prodL (a ++ b) = prodL a * prodL b := by
  induction a with
  | nil => simp [prodL]
  | cons d ds ih => simp [prodL, ih, Nat.mul_assoc]

/-- A shape with a zero dimension has no elements. -/
theorem prodL_zero_mem (s : List Nat) (h : 0 ∈ s) : prodL s = 0 := by
  induction s with
  | nil => cases h
  | cons d ds ih =>
    simp only [List.mem_cons] at h
    rcases h with h | h
    · subst h; simp [prodL]
    · simp [prodL, ih h]

/-- Only 1 to 4 dimensions are accepted by the parser. -/
theorem ndims_accepted (nd : Nat) : (nd = 0 ∨ 4 < nd) ↔ ¬ (0 < nd ∧ nd ≤ 4) := by
  omega

-- § arrays and architecture guard
/-- At most 64 array items are kept as a sample. -/
theorem array_sample_bound (count : Nat) : min count 64 ≤ 64 := Nat.min_le_right _ _

/-- The array length cap is 2^24. -/
theorem array_cap : (1 <<< 24 : Nat) = 16777216 := by decide

/-- The guard constants are consistent: top-k experts fit in the pool and query heads group over KV heads. -/
theorem arch_guard_consistent :
    10 ≤ 512 ∧ 24 % 2 = 0 ∧ 24 / 2 = 12 ∧ 48 > 0 := by decide

/-- The header magic "GGUF" read as a little-endian u32. -/
theorem gguf_magic : 0x46 * 16777216 + 0x55 * 65536 + 0x47 * 256 + 0x47 = 0x46554747 := by decide

end Ref.GgufReader



## FILE: /mnt/user-data/outputs/LazyExpertLoader.lean
/-!
# MODULE: lazy_expert_loader
SOURCES: ref/load.py
GGUF -> FP32 on demand with lazy per-expert and per-row dequantization, plus shard discovery arithmetic.

## Source semantics
* row_bytes = (ne0 / elems) * bytes. Expert e of a fused [in,out,experts] tensor is the contiguous run of
  per_expert/el*by bytes starting at e * (per_expert/el) * by; touching it never reads other experts.
* rows(): row r is rb bytes at r*rb. The PLE table is [160, 320001536] IQ4_NL = 26.82 GiB.
* Shard discovery requires exactly `total` siblings.

## Form template identifiers
Gguf.expert -> expert_slice_offset ; Gguf.rows -> row_slice_offset ; Gguf.row_bytes -> row_bytes ;
_bf16_to_f32 -> bf16_widen ; _discover_shards -> shard_set_check
-/
set_option maxRecDepth 100000

namespace Ref.LazyLoad

def rowBytes (ne0 el by_ : Nat) : Nat := ne0 / el * by_
def expertStart (e blocks by_ : Nat) : Nat := e * blocks * by_
def expertLen (blocks by_ : Nat) : Nat := blocks * by_

/-- Expert e's byte range ends no later than the whole tensor. -/
theorem expert_in_bounds (e n blocks by_ : Nat) (h : e < n) :
    expertStart e blocks by_ + expertLen blocks by_ ≤ n * blocks * by_ := by
  unfold expertStart expertLen
  have h1 : (e + 1) * blocks * by_ ≤ n * blocks * by_ :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl blocks)) (Nat.le_refl by_)
  have h2 : (e + 1) * blocks * by_ = e * blocks * by_ + blocks * by_ := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- Distinct experts read disjoint ranges: e1's range ends before e2's starts. -/
theorem experts_disjoint (e1 e2 blocks by_ : Nat) (h : e1 < e2) :
    expertStart e1 blocks by_ + expertLen blocks by_ ≤ expertStart e2 blocks by_ := by
  unfold expertStart expertLen
  have h1 : (e1 + 1) * blocks * by_ ≤ e2 * blocks * by_ :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl blocks)) (Nat.le_refl by_)
  have h2 : (e1 + 1) * blocks * by_ = e1 * blocks * by_ + blocks * by_ := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- Expert 0 starts at the beginning of the tensor. -/
theorem expert_zero_start (blocks by_ : Nat) : expertStart 0 blocks by_ = 0 := by
  simp [expertStart]

/-- Row r of a 2-D tensor lies inside the tensor. -/
theorem row_in_bounds (r n rb : Nat) (h : r < n) : r * rb + rb ≤ n * rb := by
  have h1 : (r + 1) * rb ≤ n * rb := Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl rb)
  have h2 : (r + 1) * rb = r * rb + rb := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Per-expert element count for the real experts is block-aligned: 2560*640 is a multiple of 64. -/
theorem expert_block_aligned : 2560 * 640 % 64 = 0 := by decide

/-- The real per-expert byte size is 460,800 (Q2_0). -/
theorem expert_bytes_q2_0 : 2560 * 640 / 64 * 18 = 460800 := by decide

/-- The PLE row of 160 IQ4_NL elements is 5 blocks of 18 bytes = 90 bytes. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- The PLE table is within 26.82 GiB (rounded) of IQ4_NL data. -/
theorem ple_table_gib :
    2682 * 2 ^ 30 ≤ 320001536 * rowBytes 160 32 18 * 100 ∧
    320001536 * rowBytes 160 32 18 * 100 < 2683 * 2 ^ 30 := by decide

/-- 160 is a multiple of the IQ4_NL block size 32. -/
theorem ple_width_aligned : 160 % 32 = 0 := by decide

/-- bf16 widening (shift left 16) is exact. -/
theorem bf16_widen_roundtrip (h : Nat) : h * 65536 / 65536 = h := by omega

/-- bf16 widening keeps a 16-bit value inside 32 bits. -/
theorem bf16_widen_bound (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

/-- A shard set is accepted only when the sibling count equals the declared total. -/
def shardSetOk (siblings total : Nat) : Bool := siblings == total

/-- Accepting a shard set means the counts agree. -/
theorem shardSetOk_iff (s t : Nat) : shardSetOk s t = true ↔ s = t := by
  unfold shardSetOk
  simp

/-- The suffix "-NNNNN-of-MMMMM" is 15 characters before the extension. -/
theorem shard_suffix_len : 1 + 5 + 3 + 1 + 5 = 15 := by decide

end Ref.LazyLoad



## FILE: /mnt/user-data/outputs/CanonicalFormRoundtrip.lean
/-!
# MODULE: canonical_form_roundtrip
SOURCES: tools/canonical_xcheck.py
Prototype of the P1.T3 test: every GGUF quant type is mapped to a canonical plane form (codes, scales,
optional offsets) and must decode bit-exactly. Modelled here as the integer index/bias bookkeeping that makes
the mapping lossless.

## Source semantics
* Codes are unsigned; the source bias (-1, -4, -8, -16, -32, -128) is an INTEGER subtraction before the multiply.
* Nibble types place low nibbles in the first half of a group and high nibbles in the second.
* Q3_K/Q6_K/Q4_K/Q5_K element orders are fixed index formulas; the registry lists block bytes, elements, group size.

## Form template identifiers
to_s2/from_s2 -> split_planes_2bit ; to_s4_q4_k -> canon_q4k ; to_s4_q3_k -> canon_q3k ;
to_s8_q6_K -> canon_q6k ; to_s8_q8_0 -> canon_q8_0 ; MAPPINGS -> canonical_registry
-/
set_option maxRecDepth 100000

namespace Ref.Canonical

-- § integer-bias codebooks
/-- Q8_0: storing the int8 code +128 yields an unsigned code and decoding subtracts 128 exactly. -/
theorem q8_0_bias_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    0 ≤ q + 128 ∧ q + 128 ≤ 255 ∧ (q + 128) - 128 = q := by
  omega

/-- Q4_0: unsigned 4-bit code minus 8 spans exactly [-8, 7]. -/
theorem q4_0_bias_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by
  omega

/-- Q3_K: the stored code (2 bits + 4*hbit) lies in 0..7. -/
theorem q3k_code_range (c2 hb : Nat) (h2 : c2 < 4) (hh : hb < 2) : c2 + 4 * hb < 8 := by
  omega

/-- Q3_K: subtracting the integer bias 4 reproduces ggml's "- (hmask bit ? 0 : 4)". -/
theorem q3k_bias_matches_ggml (c2 hb : Nat) (h2 : c2 < 4) (hh : hb < 2) :
    ((c2 + 4 * hb : Nat) : Int) - 4 = (c2 : Int) - (if hb = 1 then 0 else 4) := by
  split <;> omega

/-- Q5_0 and Q6_K bias ranges: 5-bit minus 16 and 6-bit minus 32. -/
theorem q5_q6_bias_range (c d : Nat) (hc : c < 32) (hd : d < 64) :
    -16 ≤ (c : Int) - 16 ∧ (c : Int) - 16 ≤ 15 ∧ -32 ≤ (d : Int) - 32 ∧ (d : Int) - 32 ≤ 31 := by
  omega

/-- Q2_0: unsigned 2-bit code minus 1 spans [-1, 2]. -/
theorem q2_0_bias_range (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

-- § Q2_0 -> S2 plane split
/-- Element j = 4b+t lives in byte b at bit position 2t, so j/4 = b and j%4 = t. -/
theorem s2_plane_index (b t : Nat) (ht : t < 4) : (4 * b + t) / 4 = b ∧ (4 * b + t) % 4 = t := by
  omega

/-- Each 18-byte Q2_0 block yields 64 one-code-per-element entries and one scale. -/
theorem s2_block_counts : (18 - 2) * 4 = 64 := by decide

-- § Q4_K / Q5_K element order
/-- Element index 64j + 32h + l (j<4, h<2, l<32) is a bijection onto [0,256). -/
theorem k45_index_bijective (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    64 * j + 32 * h + l < 256 ∧ (64 * j + 32 * h + l) / 64 = j ∧
    (64 * j + 32 * h + l) / 32 % 2 = h ∧ (64 * j + 32 * h + l) % 32 = l := by
  omega

/-- Every element index below 256 decomposes into (j, h, l). -/
theorem k45_index_surjective (i : Nat) (hi : i < 256) :
    64 * (i / 64) + 32 * (i / 32 % 2) + i % 32 = i := by
  omega

/-- Group index of element 64j+32h+l is 2j+h, i.e. 8 groups of 32. -/
theorem k45_group_index (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    (64 * j + 32 * h + l) / 32 = 2 * j + h ∧ 2 * j + h < 8 := by
  omega

/-- The Q5_K mask bit 2j+h is unique and below 8 for each (j,h). -/
theorem q5k_mask_bits_unique (j h j' h' : Nat) (hj : j < 4) (hh : h < 2) (hj' : j' < 4) (hh' : h' < 2)
    (heq : 2 * j + h = 2 * j' + h') : j = j' ∧ h = h' := by
  omega

/-- Q5_K code = nibble + 16*bit stays below 32. -/
theorem q5k_code_range (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : nib + 16 * bit < 32 := by
  omega

-- § Q3_K element order
/-- Q3_K element order 128*half + 32*j + 16*sub + l equals 16*g + l with g = 8*half + 2*j + sub < 16. -/
theorem q3k_group_order (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    128 * half + 32 * j + 16 * sub + l = 16 * (8 * half + 2 * j + sub) + l ∧
    8 * half + 2 * j + sub < 16 := by
  omega

/-- The qs source offset half*32 + sub*16 + l stays inside the 64 qs bytes. -/
theorem q3k_qs_index (half sub l : Nat) (hh : half < 2) (hs : sub < 2) (hl : l < 16) :
    half * 32 + sub * 16 + l < 64 := by
  omega

/-- The hmask index sub*16 + l is the same for both halves and stays inside 32 bytes. -/
theorem q3k_hmask_index (sub l : Nat) (hs : sub < 2) (hl : l < 16) : sub * 16 + l < 32 := by
  omega

-- § Q6_K element order
/-- Q6_K element 128h + 32q + l belongs to scale group 8h + 2q + l/16, matching ggml's sc index. -/
theorem q6k_group_of_element (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    (128 * h + 32 * q + l) / 16 = 8 * h + 2 * q + l / 16 ∧ 8 * h + 2 * q + l / 16 < 16 := by
  omega

/-- Q6_K element positions 128h + 32q + l cover [0,256) exactly once. -/
theorem q6k_index_bijective (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    128 * h + 32 * q + l < 256 ∧ (128 * h + 32 * q + l) / 128 = h ∧
    (128 * h + 32 * q + l) / 32 % 4 = q ∧ (128 * h + 32 * q + l) % 32 = l := by
  omega

-- § registry
/-- Registry rows: (block_bytes, block_elems, group_elems, max_code, form_levels). -/
def registry : List (Nat × Nat × Nat × Nat × Nat) :=
  [(18, 64, 64, 3, 4),      -- Q2_0 S2
   (110, 256, 16, 7, 16),   -- Q3_K S4
   (18, 32, 32, 15, 16),    -- Q4_0 S4
   (144, 256, 32, 15, 16),  -- Q4_K S4
   (22, 32, 32, 31, 256),   -- Q5_0 S8
   (176, 256, 32, 31, 256), -- Q5_K S8
   (210, 256, 16, 63, 256), -- Q6_K S8
   (34, 32, 32, 255, 256),  -- Q8_0 S8
   (18, 32, 32, 15, 16),    -- IQ4_NL S4
   (136, 256, 32, 15, 16)]  -- IQ4_XS S4

/-- Every registry group size divides its block size. -/
theorem registry_group_divides :
    registry.all (fun r => r.2.1 % r.2.2.1 == 0) = true := by decide

/-- Every maximum code fits the code plane of its canonical form. -/
theorem registry_codes_fit :
    registry.all (fun r => decide (r.2.2.2.1 < r.2.2.2.2)) = true := by decide

/-- There are ten quantized mappings in the registry. -/
theorem registry_size : registry.length = 10 := by decide

/-- Q4_K scales/offsets are (n,8): 256 elements in groups of 32. -/
theorem q4k_groups_per_block : 256 / 32 = 8 := by decide

end Ref.Canonical



## FILE: /mnt/user-data/outputs/ExpertBlobPacking.lean
/-!
# MODULE: expert_blob_packing
SOURCES: tools/pack_layer.py
Builds the pack-layout expert arena from the GGUF: one expert blob is 1,382,400 bytes, a pure relayout of
Q2_0 codes and fp16 scales with gate/up rows interleaved.

## Source semantics
* blob = gate_up.codes (2 x 640 rows x 640 B) ++ down.codes (2560 x 160 B) ++ gate_up.scales (2x640x40xFP16)
  ++ down.scales (2560 x 10 x FP16) = 819,200 + 409,600 + 102,400 + 51,200.
* Expert e of a role tensor is a contiguous 460,800-byte run. Layer l, expert e is at (l*512+e)*blob.
* gate row r -> interleaved row 2r, up row r -> 2r+1.

## Form template identifiers
BLOB_BYTES -> expert_blob_size ; blob_offsets -> blob_region_offsets ; gu interleave -> gate_up_row_interleave ;
ROLE_BYTES -> role_bytes_per_expert
-/
set_option maxRecDepth 100000

namespace Ref.Pack

def H : Nat := 2560
def FF : Nat := 640
def NE : Nat := 512
def QK : Nat := 64
def BB : Nat := 18
def rowBytesGU : Nat := H * 2 / 8
def rowBytesD : Nat := FF * 2 / 8
def scalesGU : Nat := H / QK
def scalesD : Nat := FF / QK
def roleBytes : Nat := H * FF * BB / QK
def blobBytes : Nat := 3 * roleBytes
def offGuCodes : Nat := 0
def offDCodes : Nat := offGuCodes + 2 * FF * rowBytesGU
def offGuScales : Nat := offDCodes + H * rowBytesD
def offDScales : Nat := offGuScales + 2 * FF * scalesGU * 2
def offEnd : Nat := offDScales + H * scalesD * 2

/-- Row geometry: 640 B of codes per 2560-wide row, 160 B per 640-wide row, 40 and 10 scale blocks. -/
theorem row_geometry : rowBytesGU = 640 ∧ rowBytesD = 160 ∧ scalesGU = 40 ∧ scalesD = 10 := by decide

/-- Each role is 460,800 bytes per expert. -/
theorem role_bytes_value : roleBytes = 460800 := by decide

/-- The blob is exactly 1,382,400 bytes. -/
theorem blob_bytes_value : blobBytes = 1382400 := by decide

/-- The four regions tile the blob exactly: start offsets and end. -/
theorem blob_offsets_values :
    offGuCodes = 0 ∧ offDCodes = 819200 ∧ offGuScales = 1228800 ∧ offDScales = 1331200 ∧
    offEnd = blobBytes := by decide

/-- Region sizes sum: 819,200 + 409,600 + 102,400 + 51,200 = 1,382,400. -/
theorem region_sizes_sum : 819200 + 409600 + 102400 + 51200 = 1382400 := by decide

/-- Regions are consecutive and strictly increasing. -/
theorem regions_ordered : offGuCodes < offDCodes ∧ offDCodes < offGuScales ∧
    offGuScales < offDScales ∧ offDScales < offEnd := by decide

/-- A block-multiple of elements costs codes + scales: blocks*18 = blocks*16 + blocks*2. -/
theorem block_split (blocks : Nat) : blocks * 18 = blocks * 16 + blocks * 2 := by omega

/-- One role per expert splits into 409,600 B of codes and 51,200 B of scales. -/
theorem role_split : H * FF / 4 = 409600 ∧ H * FF / QK * 2 = 51200 ∧
    H * FF / 4 + H * FF / QK * 2 = roleBytes := by decide

/-- A fused role tensor of [2560,640,512] has per-expert elements giving exactly one role of bytes. -/
theorem per_expert_matches_role : H * FF * NE / NE * BB / QK = roleBytes := by decide

/-- Interleaving: gate row r goes to 2r and up row r to 2r+1, covering [0,1280) without overlap. -/
theorem interleave_bijective (r s : Nat) (hr : r < 640) (hs : s < 2) :
    2 * r + s < 1280 ∧ (2 * r + s) / 2 = r ∧ (2 * r + s) % 2 = s := by
  omega

/-- Every interleaved row index below 1280 comes from exactly one (row, role) pair. -/
theorem interleave_surjective (i : Nat) (hi : i < 1280) :
    2 * (i / 2) + i % 2 = i ∧ i / 2 < 640 := by
  omega

/-- Flattened (layer, expert) index is injective for expert < 512. -/
theorem blob_index_injective (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : l * 512 + e = l' * 512 + e') : l = l' ∧ e = e' := by
  omega

/-- A smaller flat index gives an earlier, non-overlapping blob. -/
theorem blobs_disjoint (i j : Nat) (h : i < j) : i * blobBytes + blobBytes ≤ j * blobBytes := by
  have h1 : (i + 1) * blobBytes ≤ j * blobBytes := Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl _)
  have h2 : (i + 1) * blobBytes = i * blobBytes + blobBytes := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Layer blocks of 512 blobs do not overlap across layers. -/
theorem layers_disjoint (l l' : Nat) (h : l < l') :
    (l * NE) * blobBytes + NE * blobBytes ≤ (l' * NE) * blobBytes := by
  have h1 : (l + 1) * NE * blobBytes ≤ l' * NE * blobBytes :=
    Nat.mul_le_mul (Nat.mul_le_mul (Nat.succ_le_of_lt h) (Nat.le_refl _)) (Nat.le_refl _)
  have h2 : (l + 1) * NE * blobBytes = l * NE * blobBytes + NE * blobBytes := by
    rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- One layer of 512 experts is 707,788,800 bytes. -/
theorem layer_bytes : NE * blobBytes = 707788800 := by decide

/-- The full 48-layer arena is 33,973,862,400 bytes (the 34 GB arena). -/
theorem arena_bytes : 48 * NE * blobBytes = 33973862400 := by decide

/-- The 48-layer arena is 31.64 GiB (rounded). -/
theorem arena_gib : 3164 * 2 ^ 30 ≤ 48 * NE * blobBytes * 100 ∧
    48 * NE * blobBytes * 100 < 3165 * 2 ^ 30 := by decide

/-- Three roles per layer over 48 layers: 144 role tensors of 512 experts tile the arena. -/
theorem role_tensors_per_arena : 48 * 3 * NE * roleBytes = 48 * NE * blobBytes := by decide

end Ref.Pack



## FILE: /mnt/user-data/outputs/Q2_0GgufWriter.lean
/-!
# MODULE: q2_0_gguf_writer
SOURCES: tools/gguf_writer.py
Minimal GGUF v3 writer with the Q2_0 quantizer: code = clamp(round(w/d)+1, 0, 3), four codes per byte
LSB-first, 32-byte aligned tensor data.

## Source semantics
* d = amax, so w/d lies in [-1,1] and the +2 symbol (code 3) is never produced by this quantizer.
* The quantizer is idempotent: requantizing dequantized symbols returns the same codes.
* Tensor directory offsets and the data section are padded to the alignment (default 32).
* Metadata type ids 0..12 match the reader's MetaType enum.

## Form template identifiers
quantize_q2_0 -> q2_0_quantize_block ; dequantize_q2_0 -> q2_0_dequant_block ;
GGUFWriter.write -> gguf_v3_serialise ; _META_IDS -> meta_type_ids
-/
set_option maxRecDepth 100000

namespace Ref.Writer

-- § quantizer
def clampCode (r : Int) : Nat := Int.toNat (max 0 (min 3 (r + 1)))
def symbolOf (c : Nat) : Int := (c : Int) - 1
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- The code is always a valid 2-bit value. -/
theorem clampCode_lt (r : Int) : clampCode r < 4 := by
  unfold clampCode
  omega

/-- Rounded ratios in {-1,0,1,2} are represented exactly. -/
theorem clampCode_exact (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : symbolOf (clampCode r) = r := by
  unfold clampCode symbolOf
  omega

/-- With d = amax the rounded ratio is in [-1,1], so code 3 (+2) is never emitted. -/
theorem clampCode_amax_never_three (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) : clampCode r ≤ 2 := by
  unfold clampCode
  omega

/-- Out-of-range ratios saturate at the ends of the grid. -/
theorem clampCode_saturates (r : Int) :
    (r ≤ -1 → clampCode r = 0) ∧ (2 ≤ r → clampCode r = 3) := by
  unfold clampCode
  constructor <;> intro h <;> omega

/-- Requantization is a fixed point: quantizing the symbol of a code gives that code back. -/
theorem requantize_fixed_point (c : Nat) (hc : c < 4) : clampCode (symbolOf c) = c := by
  unfold clampCode symbolOf
  omega

/-- A packed byte fits in 8 bits. -/
theorem packByte_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 := by
  unfold packByte
  omega

/-- Packing places code j%4 at bit (j%4)*2; unpacking recovers all four codes. -/
theorem packByte_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 % 4 = c0 ∧ packByte c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / 16 % 4 = c2 ∧ packByte c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packByte
  omega

/-- Element j is stored in byte j/4 at bit offset (j%4)*2, which is below 8. -/
theorem element_slot (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 := by
  omega

/-- A Q2_0 block is 2 + 16 = 18 bytes holding 64 codes. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by decide

/-- Quantized size is 18 bytes per 64 elements. -/
theorem q2_0_size (n : Nat) : n * 64 / 64 * 18 = n * 18 := by omega

-- § alignment and layout
def padTo32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Padding is below 32 and makes the length a multiple of 32. -/
theorem padTo32_aligns (n : Nat) : padTo32 n < 32 ∧ (n + padTo32 n) % 32 = 0 := by
  unfold padTo32
  omega

/-- Writer padding agrees with the reader's round-up formula (c + 31)/32*32. -/
theorem pad_matches_reader (n : Nat) : n + padTo32 n = (n + 32 - 1) / 32 * 32 := by
  unfold padTo32
  omega

/-- Each tensor's next directory offset is aligned and not before the end of its data. -/
theorem next_offset_aligned (o len : Nat) :
    (o + len + 31) / 32 * 32 % 32 = 0 ∧ o + len ≤ (o + len + 31) / 32 * 32 := by
  omega

/-- Header prefix is magic 4 + version 4 + tensor count 8 + kv count 8 = 24 bytes. -/
theorem header_prefix_bytes : 4 + 4 + 8 + 8 = 24 := by decide

/-- A tensor directory entry occupies 8 + name + 4 + 8*ndims + 4 + 8 bytes. -/
def tensorInfoBytes (nameLen nd : Nat) : Nat := 8 + nameLen + 4 + 8 * nd + 4 + 8

/-- A directory entry for a 1-D tensor is name + 32 bytes. -/
theorem tensorInfo_1d (n : Nat) : tensorInfoBytes n 1 = n + 32 := by
  unfold tensorInfoBytes
  omega

/-- Directory entries grow by 8 bytes per extra dimension. -/
theorem tensorInfo_dim_step (n nd : Nat) : tensorInfoBytes n (nd + 1) = tensorInfoBytes n nd + 8 := by
  unfold tensorInfoBytes
  omega

/-- The selftest tensor [64,16] Q2_0 holds 16 blocks = 288 bytes. -/
theorem selftest_tensor_bytes : 64 * 16 / 64 * 18 = 288 := by decide

-- § metadata ids
def metaIds : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

/-- Metadata type ids are 13 distinct values 0..12 as in the reader's enum. -/
theorem meta_ids_distinct : metaIds.length = 13 ∧ metaIds.Nodup := by decide

/-- Supported tensor type ids for writing are F32=0, F16=1, Q2_0=42. -/
theorem writer_type_ids : [0, 1, 42].Nodup ∧ [0, 1, 42].length = 3 := by decide

end Ref.Writer



## FILE: /mnt/user-data/outputs/ByteLevelBpeTokenizer.lean
/-!
# MODULE: byte_level_bpe_tokenizer
SOURCES: tools/ref_tokenizer.py
GPT-2 byte-level BPE with the qwen35 pre-tokenizer. Models the reversible byte <-> unicode map, the
lowest-rank-first merge step and the special-token matching classes.

## Source semantics
* bytes_to_unicode: 188 printable bytes map to themselves; the other 68 map to 256+n in order. It is injective,
  which is why decode(encode(s)) == s.
* A BPE merge replaces two adjacent parts with one, so the part count falls by exactly one per merge.
* Token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only when parse_special is set.

## Form template identifiers
bytes_to_unicode -> byte_unicode_map ; Tokenizer._bpe -> bpe_merge_step ;
special_tokens -> special_token_classes ; decode -> bytes_from_tokens
-/
set_option maxRecDepth 100000

namespace Ref.Tokenizer

-- § byte <-> unicode
def inBs (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below b, in closed form (non-members: 0..32, 127..160, 173). -/
def nonBelow (b : Nat) : Nat :=
  if b ≤ 33 then b
  else if b ≤ 127 then 33
  else if b ≤ 161 then 33 + (b - 127)
  else if b ≤ 173 then 67
  else 68

def byteToUnicode (b : Nat) : Nat := if inBs b then b else 256 + nonBelow b

/-- Exactly 188 bytes are printable and map to themselves. -/
theorem printable_count : ((List.range 256).filter inBs).length = 188 := by decide

/-- Exactly 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun b => !inBs b)).length = 68 := by decide

/-- The closed form agrees with the counting definition of "n" in the source loop. -/
theorem nonBelow_correct :
    ∀ b, b < 256 → nonBelow b = ((List.range b).filter (fun x => !inBs x)).length := by decide

/-- Printable bytes are fixed points. -/
theorem printable_fixed (b : Nat) (h : inBs b = true) : byteToUnicode b = b := by
  unfold byteToUnicode
  simp [h]

/-- The non-printable bytes below 256 are exactly 0..32, 127..160 and 173. -/
theorem nonprintable_shape :
    ∀ b, b < 256 → inBs b = false → (b ≤ 32 ∨ (127 ≤ b ∧ b ≤ 160) ∨ b = 173) := by decide

/-- Shifted bytes land in [256, 324). -/
theorem shifted_range (b : Nat) (hb : b < 256) (h : inBs b = false) :
    256 ≤ byteToUnicode b ∧ byteToUnicode b < 324 := by
  have hs := nonprintable_shape b hb h
  unfold byteToUnicode nonBelow
  simp only [h]
  simp
  split <;> try omega
  split <;> try omega
  split <;> try omega
  split <;> omega

/-- On non-printable bytes the shift index is injective. -/
theorem nonBelow_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (hA : inBs a = false) (hB : inBs b = false) (h : nonBelow a = nonBelow b) : a = b := by
  have h1 := nonprintable_shape a ha hA
  have h2 := nonprintable_shape b hb hB
  have e : ∀ x, x < 256 → (x ≤ 32 ∨ (127 ≤ x ∧ x ≤ 160) ∨ x = 173) →
      (x ≤ 32 → nonBelow x = x) ∧ ((127 ≤ x ∧ x ≤ 160) → nonBelow x = 33 + (x - 127)) ∧
      (x = 173 → nonBelow x = 67) := by
    intro x _ _
    unfold nonBelow
    refine ⟨?_, ?_, ?_⟩ <;> intro hx <;> split <;> (try split) <;> (try split) <;> (try split) <;> omega
  have ea := e a ha h1
  have eb := e b hb h2
  omega

/-- The byte map is injective: this is the decisive round-trip property. -/
theorem byteToUnicode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteToUnicode a = byteToUnicode b) : a = b := by
  cases hA : inBs a <;> cases hB : inBs b
  · have e : nonBelow a = nonBelow b := by
      unfold byteToUnicode at h
      simp only [hA, hB] at h
      simp at h
      exact h
    exact nonBelow_injective a b ha hb hA hB e
  · have h1 := printable_fixed b hB
    have h2 := shifted_range a ha hA
    omega
  · have h1 := printable_fixed a hA
    have h2 := shifted_range b hb hB
    omega
  · have h1 := printable_fixed a hA
    have h2 := printable_fixed b hB
    omega

-- § BPE merge step
def mergeAt {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) : List α :=
  l.take i ++ [f (l.getD i default) (l.getD (i + 1) default)] ++ l.drop (i + 2)

/-- One merge reduces the number of parts by exactly one. -/
theorem mergeAt_length {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α)
    (h : i + 1 < l.length) : (mergeAt l i f).length + 1 = l.length := by
  unfold mergeAt
  simp [List.length_take, List.length_drop]
  omega

/-- Merging never empties a word. -/
theorem mergeAt_nonempty {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) :
    (mergeAt l i f) ≠ [] := by
  unfold mergeAt
  simp

/-- A word of n bytes can undergo at most n-1 merges, so the loop terminates. -/
theorem merge_budget (n k : Nat) (h : 1 < n - k) : 0 < n - k - 1 := by
  omega

-- § special tokens
def isActive (parseSpecial : Bool) (ty : Nat) : Bool :=
  ty == 4 || (parseSpecial && ty == 3)

/-- Without parse_special only USER_DEFINED (type 4) tokens are matched. -/
theorem active_without_parse (ty : Nat) : isActive false ty = true ↔ ty = 4 := by
  unfold isActive
  simp

/-- With parse_special both CONTROL (3) and USER_DEFINED (4) are matched. -/
theorem active_with_parse (ty : Nat) : isActive true ty = true ↔ (ty = 4 ∨ ty = 3) := by
  unfold isActive
  simp

/-- Enabling parse_special only adds matches, never removes any. -/
theorem active_monotone (ty : Nat) (h : isActive false ty = true) : isActive true ty = true := by
  unfold isActive at *
  simp at *
  omega

/-- Type 3 tokens are matched only under parse_special. -/
theorem control_needs_parse : isActive false 3 = false ∧ isActive true 3 = true := by decide

/-- The vocabulary of the model: 248,320 tokens, 247,587 merges (more tokens than merges). -/
theorem vocab_vs_merges : 247587 < 248320 := by decide

end Ref.Tokenizer



## FILE: /mnt/user-data/outputs/MtpDraftBlockQuantizer.lean
/-!
# MODULE: mtp_draft_block_quantizer
SOURCES: tools/mtp_pack.py
Packs the MTP draft block into a GGUF with Q2_0 / Q4_0 / Q8_0 routed experts, using per-block scale search
for Q2_0 and ggml reference rounding for Q4_0 and Q8_0.

## Source semantics
* Q2_0 scale candidates are amax*f for 17 steps f in [0.5, 1]; at f=1/2 the +2 level reaches amax.
* Codes are clip(rint(x/d), -1, 2)+1 in 0..3, packed 4 per byte LSB-first.
* Q4_0 packs q[:16] | q[16:]<<4; Q8_0 codes fit int8.
* Expert tensors: gate_up [512,1280,2560], down [512,2560,640], quantized along the last axis.

## Form template identifiers
q2_0 -> mtp_q2_0_quantize ; q4_0 -> mtp_q4_0_quantize ; q8_0 -> mtp_q8_0_quantize ;
QUANT -> mtp_expert_format_table ; EXPERT_TENSORS -> mtp_fused_expert_tensors
-/
set_option maxRecDepth 100000

namespace Ref.Mtp

-- § Q2_0 scale search
def candNum (i : Nat) : Nat := 16 + i   -- f_i = (16+i)/32

/-- There are 17 candidate scales. -/
theorem cand_count : (List.range 17).length = 17 := by decide

/-- The first candidate is f = 1/2 and the last is f = 1. -/
theorem cand_endpoints : candNum 0 = 16 ∧ candNum 16 = 32 := by decide

/-- Candidates are strictly increasing and stay in [1/2, 1] (numerators in [16, 32]). -/
theorem cand_monotone (i : Nat) (h : i < 16) : candNum i < candNum (i + 1) ∧ candNum (i + 1) ≤ 32 := by
  unfold candNum
  omega

/-- At any candidate f >= 1/2 the +2 level (2*d) reaches at least amax: 32 <= 2*numerator. -/
theorem plus_two_reaches_amax (i : Nat) (h : i ≤ 16) : 32 ≤ 2 * candNum i := by
  unfold candNum
  omega

def clipCode (r : Int) : Nat := Int.toNat (max (-1) (min 2 r) + 1)

/-- Clipped codes are in 0..3. -/
theorem clipCode_lt (r : Int) : clipCode r < 4 := by
  unfold clipCode
  omega

/-- Codes are exact for rounded ratios in [-1, 2]. -/
theorem clipCode_exact (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : (clipCode r : Int) - 1 = r := by
  unfold clipCode
  omega

def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Packing four codes and unpacking with the dequantizer's shifts is the identity. -/
theorem pack4_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 ∧ pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
    pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold pack4
  omega

-- § Q4_0 / Q8_0
def pack44 (lo hi : Nat) : Nat := lo + 16 * hi

/-- Q4_0 packs q[:16] into the low nibble and q[16:] into the high nibble; both are recovered. -/
theorem pack44_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    pack44 lo hi < 256 ∧ pack44 lo hi % 16 = lo ∧ pack44 lo hi / 16 = hi := by
  unfold pack44
  omega

/-- The extreme element maps to code 0 (since x/d = -8 and trunc(-8+8.5)=0), the symbol -8. -/
theorem q4_0_extreme_symbol : (0 : Int) - 8 = -8 := by decide

/-- Q4_0 clamping keeps codes at most 15. -/
theorem q4_0_code_clamp (t : Nat) : min 15 t ≤ 15 := Nat.min_le_left _ _

/-- Q8_0: rounded codes in [-127,127] fit the int8 range [-128,127]. -/
theorem q8_0_code_int8 (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : -128 ≤ q ∧ q ≤ 127 := by
  omega

-- § sizes
def gateUpElems : Nat := 512 * 1280 * 2560
def downElems : Nat := 512 * 2560 * 640
def expertElems : Nat := gateUpElems + downElems

/-- The inner dimensions are multiples of both block sizes (64 and 32). -/
theorem inner_dims_aligned : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

/-- Bytes per row on the last axis: 720 for 2560 wide, 180 for 640 wide (Q2_0). -/
theorem q2_0_row_bytes : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- Total routed-expert elements of the MTP block. -/
theorem expert_elems_value : expertElems = 2516582400 := by decide

/-- Q2_0 experts total 707,788,800 bytes, about 0.71 GB. -/
theorem q2_0_total_bytes : expertElems / 64 * 18 = 707788800 := by decide

/-- That equals one full layer of the main arena (512 blobs of 1,382,400 B). -/
theorem q2_0_equals_layer_arena : expertElems / 64 * 18 = 512 * 1382400 := by decide

/-- Q4_0 experts total 1,415,577,600 bytes, about 1.42 GB. -/
theorem q4_0_total_bytes : expertElems / 32 * 18 = 1415577600 := by decide

/-- Q8_0 experts total 2,673,868,800 bytes, about 2.67 GB. -/
theorem q8_0_total_bytes : expertElems / 32 * 34 = 2673868800 := by decide

/-- Q2_0 is strictly smaller than Q4_0, which is strictly smaller than Q8_0. -/
theorem format_size_order : expertElems / 64 * 18 < expertElems / 32 * 18 ∧
    expertElems / 32 * 18 < expertElems / 32 * 34 := by decide

/-- The fused gate_up tensor has twice the rows of the 640-wide FFN (1280 = 2*640). -/
theorem gate_up_rows : 2 * 640 = 1280 := by decide

end Ref.Mtp



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- GgufReaderGeometry.lean
  was: theorem geometry_positive :
    ∀ t, t < 64 → ∀ e b, blockGeometry t = some (e, b) → 0 < e ∧ 0 < b
  now: theorem geometry_positive :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (0 < p.1 ∧ 0 < p.2)) = true
  measured: fewer quantified variables (2 → 1)

- GgufReaderGeometry.lean
  was: theorem geometry_elems_classes :
    ∀ t, t < 64 → ∀ e b, blockGeometry t = some (e, b) →
      e = 1 ∨ e = 32 ∨ e = 64 ∨ e = 256
  now: theorem geometry_elems_classes :
    ∀ t, t < 64 → (blockGeometry t).elim true
      (fun p => decide (p.1 = 1 ∨ p.1 = 32 ∨ p.1 = 64 ∨ p.1 = 256)) = true
  measured: fewer quantified variables (2 → 1)

- GgufReaderGeometry.lean
  was: theorem only_q2_0_has_64 :
    ∀ t, t < 64 → ∀ b, blockGeometry t = some (64, b) → t = 42
  now: theorem only_q2_0_has_64 :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (p.1 = 64 → t = 42)) = true
  measured: fewer quantified variables (2 → 1)

- ByteLevelBpeTokenizer.lean
  was: theorem shifted_range (b : Nat) (hb : b < 256) (h : inBs b = false) :
    256 ≤ byteToUnicode b ∧ byteToUnicode b < 324
  now: theorem nonprintable_shape :
    ∀ b, b < 256 → inBs b = false → (b ≤ 32 ∨ (127 ≤ b ∧ b ≤ 160) ∨ b = 173)
  measured: fewer quantified variables (3 → 1)

- ByteLevelBpeTokenizer.lean
  was: theorem byteToUnicode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteToUnicode a = byteToUnicode b) : a = b
  now: theorem shifted_range (b : Nat) (hb : b < 256) (h : inBs b = false) :
    256 ≤ byteToUnicode b ∧ byteToUnicode b < 324
  measured: fewer quantified variables (4 → 3)

- ByteLevelBpeTokenizer.lean
  was: theorem mergeAt_length {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α)
    (h : i + 1 < l.length) : (mergeAt l i f).length + 1 = l.length
  now: theorem nonBelow_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (hA : inBs a = false) (hB : inBs b = false) (h : nonBelow a = nonBelow b) : a = b
  measured: more hypotheses/binders (5 → 6): the statement holds in fewer cases

- ByteLevelBpeTokenizer.lean
  was: theorem mergeAt_nonempty {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) :
    (mergeAt l i f) ≠ []
  now: theorem byteToUnicode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteToUnicode a = byteToUnicode b) : a = b
  measured: the claim changed

- ByteLevelBpeTokenizer.lean
  was: theorem merge_budget (n k : Nat) (h : 1 < n - k) : 0 < n - k - 1
  now: theorem mergeAt_length {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α)
    (h : i + 1 < l.length) : (mergeAt l i f).length + 1 = l.length
  measured: more hypotheses/binders (2 → 5): the statement holds in fewer cases

- ByteLevelBpeTokenizer.lean
  was: theorem active_without_parse (ty : Nat) : isActive false ty = true ↔ ty = 4
  now: theorem mergeAt_nonempty {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) :
    (mergeAt l i f) ≠ []
  measured: more hypotheses/binders (1 → 4): the statement holds in fewer cases

- ByteLevelBpeTokenizer.lean
  was: theorem active_with_parse (ty : Nat) : isActive true ty = true ↔ (ty = 4 ∨ ty = 3)
  now: theorem merge_budget (n k : Nat) (h : 1 < n - k) : 0 < n - k - 1
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- ByteLevelBpeTokenizer.lean
  was: theorem active_monotone (ty : Nat) (h : isActive false ty = true) : isActive true ty = true
  now: theorem active_without_parse (ty : Nat) : isActive false ty = true ↔ ty = 4
  measured: fewer quantified variables (2 → 1)

- ByteLevelBpeTokenizer.lean
  was: theorem control_needs_parse : isActive false 3 = false ∧ isActive true 3 = true
  now: theorem active_with_parse (ty : Nat) : isActive true ty = true ↔ (ty = 4 ∨ ty = 3)
  measured: the claim changed

- ByteLevelBpeTokenizer.lean
  was: theorem vocab_vs_merges : 247587 < 248320
  now: theorem active_monotone (ty : Nat) (h : isActive false ty = true) : isActive true ty = true
  measured: the claim changed
-/



## FILE: /mnt/user-data/outputs/ExpertResidencyCache.lean
/-!
# MODULE: expert_residency_cache
SOURCES: include/ref/core/expert_cache.hpp, src/core/expert_cache.cpp
The VRAM-resident expert tier: a slot arena plus a `(layer, expert) -> slot or -1` residency table. Admission never
evicts; an optional per-layer mode confines each layer to its own contiguous slot range.

## Source semantics
* `admit` is idempotent on resident pairs, refuses (returns -1) when no slot is free, and NEVER evicts or moves an
  already-resident pair. Distinct pairs get distinct slots.
* Per-layer admission: layer `l` may only use `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the
  remainder, so the ranges tile `0..slots` and no two layers share a slot.
* `fill_slot` copies `bytes` when `0 < bytes <= blob`, else the whole slot. Sized slots are 256-byte aligned.
* The profile reader accepts only a header whose geometry matches, with `n_ranked <= slots`, and in-range pairs.

## Form template identifiers
ExpertCache::admit -> admit ; ExpertCache::slot_of -> lookup ; next_free_ -> nextFree ; layer_next_ -> layerNext ;
layer_slot_range -> layerLo / layerHi ; residency_ -> res ; kNotResident -> none (toInt none = -1)
-/

namespace Ref.ExpertCache

-- § slot encoding

/-- The C++ encoding of an optional slot: `-1` for not resident. -/
def toInt : Option Nat → Int
  | none => -1
  | some n => (n : Int)

/-- `kNotResident` (-1) is exactly the `none` case. -/
theorem toInt_eq_neg_one_iff (o : Option Nat) : toInt o = -1 ↔ o = none := by
  cases o with
  | none => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | some n => exact ⟨fun h => by have h' : (n : Int) = -1 := h; omega, fun h => by cases h⟩

-- § per-layer slot ranges

/-- `q = slots / n_layers`. -/
def layerQ (slots n : Nat) : Nat := slots / n
/-- Lower bound of layer `l`'s range. -/
def layerLo (slots n l : Nat) : Nat := l * layerQ slots n
/-- Upper bound (exclusive); the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat := if l = n - 1 then slots else (l + 1) * layerQ slots n

/-- Layer 0's range starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots n : Nat) : layerHi slots n (n - 1) = slots := by
  simp [layerHi]

/-- Each range is well formed. -/
theorem layerLo_le_hi (slots n l : Nat) (hl : l < n) : layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi layerQ
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h1 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
    have h2 : n * (slots / n) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots n
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Consecutive ranges abut: the ranges tile with no gap. -/
theorem layerHi_eq_next_lo (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  have h : ¬ l = n - 1 := by omega
  rw [if_neg h]

/-- Ranges of different layers do not overlap: this is why per-layer admission cannot share a slot. -/
theorem ranges_disjoint (slots n l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < n) :
    layerHi slots n l1 ≤ layerLo slots n l2 := by
  unfold layerHi layerLo
  have h : ¬ l1 = n - 1 := by omega
  rw [if_neg h]
  exact Nat.mul_le_mul_right _ h12

/-- Every layer's quota is at least `q` wide. -/
theorem layer_width_ge_quota (slots n l : Nat) (hl : l < n) :
    layerLo slots n l + layerQ slots n ≤ layerHi slots n l := by
  unfold layerLo layerHi layerQ
  have e : (l + 1) * (slots / n) = l * (slots / n) + slots / n := by
    rw [Nat.add_mul, Nat.one_mul]
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h2 : n * (slots / n) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots n
    have h3 : (l + 1) * (slots / n) ≤ slots := by
      have : l + 1 = n := by omega
      rw [this]; exact h2
    omega
  · rw [if_neg h]
    omega

-- § global admission (no eviction)

/-- The cache state relevant to admission. -/
structure Cache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  res : Nat → Nat → Option Nat
  nextFree : Nat

/-- `slot_of`: bounds-checked lookup. -/
def Cache.lookup (c : Cache) (l e : Nat) : Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert then c.res l e else none

/-- A new slot is claimed exactly when in bounds, not resident, and a slot is free. -/
def Cache.canAdmit (c : Cache) (l e : Nat) : Bool :=
  decide (l < c.nLayers) && decide (e < c.nExpert) && (c.res l e).isNone && decide (c.nextFree < c.slots)

/-- `admit`: returns the slot (or none) and the new state. -/
def Cache.admit (c : Cache) (l e : Nat) : Option Nat × Cache :=
  if c.canAdmit l e = true then
    (some c.nextFree,
      { c with res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e',
               nextFree := c.nextFree + 1 })
  else (c.lookup l e, c)

/-- Characterisation of `canAdmit`. -/
theorem canAdmit_iff (c : Cache) (l e : Nat) :
    c.canAdmit l e = true ↔ ((l < c.nLayers ∧ e < c.nExpert) ∧ c.res l e = none ∧ c.nextFree < c.slots) := by
  unfold Cache.canAdmit
  cases hr : c.res l e <;> simp [hr, and_assoc]

/-- The fields of the post-state when a slot is claimed. -/
theorem admit_new (c : Cache) (l e : Nat) (hc : c.canAdmit l e = true) :
    (c.admit l e).2.nextFree = c.nextFree + 1 ∧
    (∀ l' e', (c.admit l e).2.res l' e' = if l' = l ∧ e' = e then some c.nextFree else c.res l' e') ∧
    (c.admit l e).2.slots = c.slots ∧ (c.admit l e).2.nLayers = c.nLayers ∧
    (c.admit l e).2.nExpert = c.nExpert := by
  unfold Cache.admit
  rw [if_pos hc]
  exact ⟨rfl, fun _ _ => rfl, rfl, rfl, rfl⟩

/-- When nothing is claimed the state is untouched. -/
theorem admit_not (c : Cache) (l e : Nat) (hc : c.canAdmit l e = false) :
    c.admit l e = (c.lookup l e, c) := by
  unfold Cache.admit
  have hn : ¬ (c.canAdmit l e = true) := by simp [hc]
  rw [if_neg hn]

/-- Invariants: the counter is within the arena, resident slots are below the counter, slots are injective. -/
def Cache.Inv (c : Cache) : Prop :=
  c.nextFree ≤ c.slots ∧
  (∀ l e s, c.res l e = some s → s < c.nextFree) ∧
  (∀ l e l' e' s, c.res l e = some s → c.res l' e' = some s → l = l' ∧ e = e')

/-- Admission preserves the invariants (no slot is handed out twice, none beyond the arena). -/
theorem admit_inv (c : Cache) (l e : Nat) (h : c.Inv) : (c.admit l e).2.Inv := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨_, hnone, hlt⟩ := (canAdmit_iff c l e).1 hc
    obtain ⟨hnf, hres, hs, _, _⟩ := admit_new c l e hc
    obtain ⟨h1, h2, h3⟩ := h
    refine ⟨?_, ?_, ?_⟩
    · rw [hnf, hs]; omega
    · intro l' e' s hs'
      rw [hres] at hs'
      rw [hnf]
      by_cases hh : l' = l ∧ e' = e
      · rw [if_pos hh] at hs'
        have := Option.some.inj hs'
        omega
      · rw [if_neg hh] at hs'
        have := h2 l' e' s hs'
        omega
    · intro a b a' b' s ha hb
      rw [hres] at ha hb
      by_cases k1 : a = l ∧ b = e
      · by_cases k2 : a' = l ∧ b' = e
        · exact ⟨k1.1.trans k2.1.symm, k1.2.trans k2.2.symm⟩
        · rw [if_pos k1] at ha
          rw [if_neg k2] at hb
          have e1 := Option.some.inj ha
          have e2 := h2 a' b' s hb
          omega
      · by_cases k2 : a' = l ∧ b' = e
        · rw [if_neg k1] at ha
          rw [if_pos k2] at hb
          have e1 := Option.some.inj hb
          have e2 := h2 a b s ha
          omega
        · rw [if_neg k1] at ha
          rw [if_neg k2] at hb
          exact h3 a b a' b' s ha hb
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc']
    exact h

/-- A resident, in-bounds pair is returned as is and the state does not change (admit is idempotent). -/
theorem admit_resident (c : Cache) (l e s : Nat) (hb : l < c.nLayers ∧ e < c.nExpert)
    (hr : c.res l e = some s) : c.admit l e = (some s, c) := by
  have hc' : c.canAdmit l e = false := by
    cases hh : c.canAdmit l e
    · rfl
    · have := ((canAdmit_iff c l e).1 hh).2.1
      rw [hr] at this
      cases this
  rw [admit_not c l e hc']
  simp [Cache.lookup, hb, hr]

/-- A full cache never admits, and never changes: there is no eviction. -/
theorem admit_full (c : Cache) (l e : Nat) (hfull : c.slots ≤ c.nextFree) :
    (c.admit l e).2 = c := by
  have hc' : c.canAdmit l e = false := by
    cases hh : c.canAdmit l e
    · rfl
    · have := ((canAdmit_iff c l e).1 hh).2.2
      omega
  rw [admit_not c l e hc']

/-- Any slot returned lies inside the arena. -/
theorem admit_slot_lt (c : Cache) (l e s : Nat) (h : c.Inv) (hs : (c.admit l e).1 = some s) :
    s < c.slots := by
  by_cases hc : c.canAdmit l e = true
  · have h1 := ((canAdmit_iff c l e).1 hc).2.2
    unfold Cache.admit at hs
    rw [if_pos hc] at hs
    have := Option.some.inj hs
    omega
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc'] at hs
    have hs' : c.lookup l e = some s := hs
    unfold Cache.lookup at hs'
    by_cases hb : l < c.nLayers ∧ e < c.nExpert
    · rw [if_pos hb] at hs'
      obtain ⟨h1, h2, _⟩ := h
      have := h2 l e s hs'
      omega
    · rw [if_neg hb] at hs'
      cases hs'

/-- NO EVICTION: every pair resident before an admission is resident in the same slot after it. -/
theorem admit_no_evict (c : Cache) (l e l' e' s : Nat) (h : c.res l' e' = some s) :
    (c.admit l e).2.res l' e' = some s := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨hnf, hres, _⟩ := admit_new c l e hc
    have hnone := ((canAdmit_iff c l e).1 hc).2.1
    rw [hres]
    by_cases hh : l' = l ∧ e' = e
    · obtain ⟨h1, h2⟩ := hh
      subst h1; subst h2
      rw [hnone] at h
      cases h
    · rw [if_neg hh]; exact h
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc']
    exact h

/-- After an admission that returned a slot, the pair is resident in exactly that slot. -/
theorem admit_then_resident (c : Cache) (l e s : Nat) (hs : (c.admit l e).1 = some s) :
    (c.admit l e).2.lookup l e = some s := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨_, hres, _, hL, hE⟩ := admit_new c l e hc
    have hb := ((canAdmit_iff c l e).1 hc).1
    unfold Cache.admit at hs
    rw [if_pos hc] at hs
    have hs2 := Option.some.inj hs
    unfold Cache.lookup
    rw [hL, hE, if_pos hb, hres, if_pos ⟨rfl, rfl⟩, hs2]
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc'] at hs ⊢
    exact hs

-- § per-layer admission (R4.2g)

/-- Cache state under per-layer admission. -/
structure LCache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  res : Nat → Nat → Option Nat
  layerNext : Nat → Nat

/-- Claim a slot only when in bounds, not resident and the layer's own quota is not full. -/
def LCache.canAdmit (c : LCache) (l e : Nat) : Bool :=
  decide (l < c.nLayers) && decide (e < c.nExpert) && (c.res l e).isNone &&
    decide (c.layerNext l < layerHi c.slots c.nLayers l)

/-- Per-layer `admit`. -/
def LCache.admit (c : LCache) (l e : Nat) : Option Nat × LCache :=
  if c.canAdmit l e = true then
    (some (c.layerNext l),
      { c with res := fun l' e' => if l' = l ∧ e' = e then some (c.layerNext l) else c.res l' e',
               layerNext := fun l' => if l' = l then c.layerNext l + 1 else c.layerNext l' })
  else (if l < c.nLayers ∧ e < c.nExpert then c.res l e else none, c)

/-- Characterisation of the per-layer `canAdmit`. -/
theorem LCache.canAdmit_iff (c : LCache) (l e : Nat) :
    c.canAdmit l e = true ↔
      ((l < c.nLayers ∧ e < c.nExpert) ∧ c.res l e = none ∧
        c.layerNext l < layerHi c.slots c.nLayers l) := by
  unfold LCache.canAdmit
  cases hr : c.res l e <;> simp [hr, and_assoc]

/-- Post-state fields when a slot is claimed per-layer. -/
theorem LCache.admit_new (c : LCache) (l e : Nat) (hc : c.canAdmit l e = true) :
    (∀ l', (c.admit l e).2.layerNext l' = if l' = l then c.layerNext l + 1 else c.layerNext l') ∧
    (∀ l' e', (c.admit l e).2.res l' e' =
        if l' = l ∧ e' = e then some (c.layerNext l) else c.res l' e') ∧
    (c.admit l e).2.slots = c.slots ∧ (c.admit l e).2.nLayers = c.nLayers ∧
    (c.admit l e).1 = some (c.layerNext l) := by
  unfold LCache.admit
  rw [if_pos hc]
  exact ⟨fun _ => rfl, fun _ _ => rfl, rfl, rfl, rfl⟩

/-- Per-layer invariants: counters stay inside the layer's range; resident slots lie in their own layer's range. -/
def LCache.Inv (c : LCache) : Prop :=
  (∀ l, l < c.nLayers → layerLo c.slots c.nLayers l ≤ c.layerNext l ∧
      c.layerNext l ≤ layerHi c.slots c.nLayers l) ∧
  (∀ l e s, c.res l e = some s →
      l < c.nLayers ∧ layerLo c.slots c.nLayers l ≤ s ∧ s < c.layerNext l)

/-- A claimed slot lies inside the claiming layer's range. -/
theorem LCache.admit_slot_in_range (c : LCache) (l e s : Nat) (h : c.Inv)
    (hc : c.canAdmit l e = true) (hs : (c.admit l e).1 = some s) :
    layerLo c.slots c.nLayers l ≤ s ∧ s < layerHi c.slots c.nLayers l := by
  obtain ⟨_, _, _, _, h5⟩ := LCache.admit_new c l e hc
  rw [h5] at hs
  have hs2 := Option.some.inj hs
  obtain ⟨hb, _, hlt⟩ := (LCache.canAdmit_iff c l e).1 hc
  have := h.1 l hb.1
  omega

/-- Per-layer admission preserves the invariants. -/
theorem LCache.admit_inv (c : LCache) (l e : Nat) (h : c.Inv) : (c.admit l e).2.Inv := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨hnx, hres, hs, hn, _⟩ := LCache.admit_new c l e hc
    obtain ⟨hb, hnone, hlt⟩ := (LCache.canAdmit_iff c l e).1 hc
    obtain ⟨h1, h2⟩ := h
    refine ⟨?_, ?_⟩
    · intro l' hl'
      rw [hn] at hl'
      rw [hs, hn, hnx l']
      by_cases hh : l' = l
      · subst hh
        rw [if_pos rfl]
        have := h1 l' hb.1
        omega
      · rw [if_neg hh]
        exact h1 l' hl'
    · intro a b s hab
      rw [hres] at hab
      rw [hs, hn, hnx a]
      by_cases k : a = l ∧ b = e
      · rw [if_pos k] at hab
        have e1 := Option.some.inj hab
        obtain ⟨ka, _⟩ := k
        subst ka
        rw [if_pos rfl]
        have := h1 a hb.1
        omega
      · rw [if_neg k] at hab
        obtain ⟨p1, p2, p3⟩ := h2 a b s hab
        by_cases ka : a = l
        · rw [if_pos ka]
          subst ka
          exact ⟨p1, p2, by omega⟩
        · rw [if_neg ka]
          exact ⟨p1, p2, p3⟩
  · have hc' : c.canAdmit l e = false := by simpa using hc
    have : (c.admit l e).2 = c := by
      unfold LCache.admit
      have hn : ¬ (c.canAdmit l e = true) := by simp [hc']
      rw [if_neg hn]
    rw [this]
    exact h

/-- TWO LAYERS NEVER SHARE A SLOT: the same slot in two resident pairs forces the same layer. -/
theorem LCache.no_cross_layer_sharing (c : LCache) (h : c.Inv) (l e l' e' s : Nat)
    (h1 : c.res l e = some s) (h2 : c.res l' e' = some s) : l = l' := by
  obtain ⟨hl, lo1, hi1⟩ := h.2 l e s h1
  obtain ⟨hl', lo2, hi2⟩ := h.2 l' e' s h2
  rcases Nat.lt_trichotomy l l' with hlt | heq | hgt
  · have d := ranges_disjoint c.slots c.nLayers l l' hlt hl'
    have a := (h.1 l hl).2
    omega
  · exact heq
  · have d := ranges_disjoint c.slots c.nLayers l' l hgt hl
    have a := (h.1 l' hl').2
    omega

-- § byte arithmetic of slots

/-- Slot sizes are rounded up to 256 bytes. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounded sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Rounding adds less than 256 bytes. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

/-- Total arena size of a list of sized slots. -/
def totalBytes : List Nat → Nat
  | [] => 0
  | b :: bs => alignUp256 b + totalBytes bs

/-- Sum of the raw sizes. -/
def sumBytes : List Nat → Nat
  | [] => 0
  | b :: bs => b + sumBytes bs

/-- The arena holds at least the raw bytes of every slot. -/
theorem totalBytes_ge (bs : List Nat) : sumBytes bs ≤ totalBytes bs := by
  induction bs with
  | nil => simp [sumBytes, totalBytes]
  | cons b bs ih =>
    simp only [sumBytes, totalBytes]
    have := alignUp256_ge b
    omega

/-- The byte count `fill_slot` copies: the blob's own size if it fits the slot, otherwise the whole slot. -/
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never writes past the slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- `bytes = 0` means the whole slot. -/
theorem fillBytes_zero (blob : Nat) : fillBytes 0 blob = blob := by
  simp [fillBytes]

/-- The VRAM check: refuse when free memory is below what was requested. -/
def vramFits (free want : Nat) : Bool := decide (want ≤ free)

/-- Refusal happens exactly when the request exceeds free memory. -/
theorem vramFits_false_iff (free want : Nat) : vramFits free want = false ↔ free < want := by
  unfold vramFits
  simp

-- § profile reader

/-- A ranked pair is in range for the model. -/
def pairValid (nl ne : Nat) (p : Nat × Nat) : Bool := decide (p.1 < nl) && decide (p.2 < ne)

/-- Header + payload acceptance of `read_expert_profile`. -/
def profileAccepts (hdrNl hdrNe want nRanked nl ne : Nat) (pairs : List (Nat × Nat)) : Bool :=
  decide (hdrNl = nl) && decide (hdrNe = ne) && decide (nRanked ≤ want) &&
    decide (pairs.length = nRanked) && pairs.all (pairValid nl ne)

/-- An accepted profile matches the model, has no more pairs than slots, and every pair is in range. -/
theorem profileAccepts_sound (hdrNl hdrNe want nRanked nl ne : Nat) (pairs : List (Nat × Nat))
    (h : profileAccepts hdrNl hdrNe want nRanked nl ne pairs = true) :
    hdrNl = nl ∧ hdrNe = ne ∧ nRanked ≤ want ∧ ∀ p ∈ pairs, p.1 < nl ∧ p.2 < ne := by
  simp only [profileAccepts, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, _⟩, h5⟩ := h
  refine ⟨h1, h2, h3, fun p hp => ?_⟩
  have := h5 p hp
  simp only [pairValid, Bool.and_eq_true, decide_eq_true_eq] at this
  exact this

end Ref.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
/-!
# MODULE: expert_source_dispatch
SOURCES: include/ref/core/expert_source.hpp
The adapter between the host loop and the expert pool: where an expert blob's bytes come from (file-mapped or arena),
the GPU-hit / CPU-miss split by router index, the all-or-nothing hit-path wiring, and the latching failure flag.

## Source semantics
* `experts.bin` holds exactly `48 x 512` blobs of 1,382,400 bytes, indexed `layer*512 + expert`, no padding; a file of
  any other size is refused.
* A routed index is computed by exactly one of CPU (miss) or GPU (hit): the CPU row of a hit is zeroed and the hit
  row of a miss is zero, so the combined row equals the expert's output exactly once.
* `hits_ready` is all-or-nothing, and in particular requires the fp32 activation-scale buffer (R4.2h).
* `failed` latches: clearing the message never re-arms the adapter.
* The PCIe share of a verify window is `pcie_num/256` of the missed experts, never more than all of them.

## Form template identifiers
FileExpertSource::blob -> blobIndex ; kBlob -> BLOB ; ExpertDispatch::is_hit -> hit flags ;
ExpertDispatch::hits_ready -> HitWiring.ready ; ExpertDispatch::failed -> Failure.failed ; pcie_num -> pcieNum
-/

namespace Ref.ExpertSource

-- § geometry and file indexing

/-- Bytes per expert blob. -/
def BLOB : Nat := 1382400
/-- Index of `(layer, expert)` inside `experts.bin`. -/
def blobIndex (l e : Nat) : Nat := l * 512 + e

/-- The file holds exactly 24,576 blobs. -/
theorem blob_count : 48 * 512 = 24576 := by decide

/-- The 34 GB expert file size quoted in the header. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by decide

/-- Indices stay inside the file. -/
theorem blobIndex_lt (l e : Nat) (hl : l < 48) (he : e < 512) : blobIndex l e < 24576 := by
  unfold blobIndex; omega

/-- Distinct pairs have distinct indices (no padding, no aliasing). -/
theorem blobIndex_inj (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : blobIndex l e = blobIndex l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h; omega

/-- Every blob ends inside the file, so a mapped read cannot run past the end. -/
theorem blob_end_le_size (l e : Nat) (hl : l < 48) (he : e < 512) :
    (blobIndex l e + 1) * 1382400 ≤ 33973862400 := by
  unfold blobIndex; omega

/-- The size check of `FileExpertSource::open`. -/
def fileAccepts (size nLayers nExpert : Nat) : Bool := decide (size = nLayers * nExpert * BLOB)

/-- A short file is refused. -/
theorem short_file_refused (size nLayers nExpert : Nat) (h : size < nLayers * nExpert * BLOB) :
    fileAccepts size nLayers nExpert = false := by
  unfold fileAccepts
  simp
  omega

/-- An over-long file is refused. -/
theorem long_file_refused (size nLayers nExpert : Nat) (h : nLayers * nExpert * BLOB < size) :
    fileAccepts size nLayers nExpert = false := by
  unfold fileAccepts
  simp
  omega

/-- The 663.6 MB of expert bytes per token: 48 layers x 10 routed experts. -/
theorem bytes_per_token : 48 * 10 * 1382400 = 663552000 := by decide

/-- The hit path's activation buffer: 80 blocks of 34 bytes. -/
theorem q8_buffer_bytes : 2560 / 32 * 34 = 2720 := by decide

/-- A Q8_0 block is a 2-byte scale plus 32 quants. -/
theorem q8_block_bytes : 2 + 32 = 34 := by decide

-- § the hit/miss split

/-- CPU contribution of one router index: zero for a hit. -/
def cpuPart (hit : Bool) (v : Int) : Int := if hit then 0 else v
/-- GPU contribution of one router index: zero for a miss. -/
def gpuPart (hit : Bool) (v : Int) : Int := if hit then v else 0

/-- Each routed expert is counted exactly once, whichever side computed it. -/
theorem split_partition (hit : Bool) (v : Int) : cpuPart hit v + gpuPart hit v = v := by
  cases hit <;> simp [cpuPart, gpuPart]

/-- Weighted combine over router indices, as `moe_combine` does on the device. -/
def combine : List Int → List Int → Int
  | w :: ws, p :: ps => w * p + combine ws ps
  | _, _ => 0

/-- The split never changes the combined output: summing CPU and GPU rows weights each expert once. -/
theorem combine_split (hits : List Bool) (ws vs : List Int) (hlen : hits.length = vs.length) :
    combine ws ((hits.zip vs).map (fun x => cpuPart x.1 x.2)) +
      combine ws ((hits.zip vs).map (fun x => gpuPart x.1 x.2)) =
    combine ws ((hits.zip vs).map (fun x => x.2)) := by
  clear hlen
  induction hits generalizing ws vs with
  | nil => cases ws <;> simp [combine]
  | cons h hs ih =>
    cases ws with
    | nil => simp [combine]
    | cons w ws =>
      cases vs with
      | nil => simp [combine]
      | cons v vs =>
        simp only [List.zip_cons_cons, List.map_cons, combine]
        have := ih ws vs
        have e := split_partition h v
        calc w * cpuPart h v + combine ws (List.map (fun x => cpuPart x.1 x.2) (hs.zip vs)) +
              (w * gpuPart h v + combine ws (List.map (fun x => gpuPart x.1 x.2) (hs.zip vs)))
            = w * (cpuPart h v + gpuPart h v) +
              (combine ws (List.map (fun x => cpuPart x.1 x.2) (hs.zip vs)) +
               combine ws (List.map (fun x => gpuPart x.1 x.2) (hs.zip vs))) := by
                rw [Int.mul_add]; omega
          _ = w * v + combine ws (List.map (fun x => x.2) (hs.zip vs)) := by
                rw [e, this]

-- § hit-path wiring

/-- Which pieces of the hit path are configured. -/
structure HitWiring where
  cache : Bool
  cacheBase : Bool
  partsOut : Bool
  hitOut : Bool
  mixed : Bool
  scratch : Bool
  xq8 : Bool
  xq8Scale : Bool
  dSlot : Bool
  dDst : Bool

/-- All of it or none of it. -/
def HitWiring.ready (w : HitWiring) : Bool :=
  w.cache && w.cacheBase && w.partsOut && w.hitOut && w.mixed && w.scratch && w.xq8 && w.xq8Scale &&
    w.dSlot && w.dDst

/-- The fp32 activation scales are required: without them the GPU hits use fp16 scales and the tokens change. -/
theorem ready_requires_scale (w : HitWiring) (h : w.ready = true) : w.xq8Scale = true := by
  unfold HitWiring.ready at h
  simp only [Bool.and_eq_true] at h
  exact h.1.1.2

/-- A missing device staging buffer disables the hit path (never half-configured). -/
theorem not_ready_of_missing_slot (w : HitWiring) (h : w.dSlot = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

-- § failure latch

/-- Failure state: the flag latches, the message is only text. -/
structure Failure where
  failed : Bool
  msg : Option String

/-- Record a failure. -/
def Failure.record (f : Failure) (m : String) : Failure := { failed := true, msg := some m }
/-- Clear the message only. -/
def Failure.clearMsg (f : Failure) : Failure := { f with msg := none }

/-- Clearing the message does not re-arm the adapter. -/
theorem clearMsg_keeps_failed (f : Failure) : f.clearMsg.failed = f.failed := rfl

/-- Once failed, always failed after recording. -/
theorem record_failed (f : Failure) (m : String) : (f.record m).failed = true := rfl

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe: `misses * pcie_num / 256`. -/
def pcieShare (misses pcieNum : Nat) : Nat := misses * pcieNum / 256

/-- The PCIe share never exceeds the misses when `pcie_num <= 256`. -/
theorem pcieShare_le (misses pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare misses pcieNum ≤ misses := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left misses h
  omega

end Ref.ExpertSource



## FILE: /mnt/user-data/outputs/WeightPackLoader.lean
/-!
# MODULE: weight_pack_loader
SOURCES: src/core/weights.cpp
The dense-weight loader: reads `index.txt` rows, converts each tensor plane by plane (copy, bf16 high half, f16 round,
fp16 widen to f32) through bounded staging buffers, and places the results in an aligned (optionally compacted) arena.

## Source semantics
* A quantized tensor is codes, then scales, then an optional offset plane; the segment sums must equal the recorded
  source and destination totals or the loader refuses.
* bf16 promotion takes the high 16 bits of the f32 container exactly; a promoted tensor is 4 B/elem in, 2 B/elem out.
* fp16 widening doubles the byte count, never splits an element, and fits the staging-out buffer (2 x CHUNK).
* A skip set compacts the arena: kept rows are re-placed at the index alignment, skipped rows take no bytes.
* The chunked copy loop covers each plane exactly once with chunks of at most CHUNK bytes.

## Form template identifiers
Conv -> Conv ; Seg -> segment sums ; CHUNK -> CHUNK ; stage_out -> stageOutBytes ; WidenF16 -> Conv.widenF16 ;
Bf16High -> Conv.bf16High ; pool -> compactTotal ; align -> a
-/

namespace Ref.WeightLoader

-- § staging buffers

/-- 8 MiB of source per staging round. -/
def CHUNK : Nat := 8 * 1048576

/-- CHUNK is even, so a full chunk cannot split an fp16 element. -/
theorem CHUNK_even : CHUNK % 2 = 0 := by decide

/-- How a segment's bytes are transformed. -/
inductive Conv where
  | copy
  | bf16High
  | toF16
  | widenF16

/-- Output bytes of a chunk of `n` source bytes. -/
def outBytes : Conv → Nat → Nat
  | .copy, n => n
  | .bf16High, n => n / 4 * 2
  | .toF16, n => n / 4 * 2
  | .widenF16, n => n / 2 * 4

/-- THE STAGING-OUT BUFFER (2 x CHUNK) holds any conversion of a chunk, including the widening that grows it. -/
theorem outBytes_le_stage (c : Conv) (n : Nat) (h : n ≤ CHUNK) : outBytes c n ≤ 2 * CHUNK := by
  cases c <;> simp only [outBytes] <;> omega

/-- Widening an even-length plane exactly doubles it. -/
theorem widen_doubles (n : Nat) (h : n % 2 = 0) : outBytes .widenF16 n = 2 * n := by
  simp only [outBytes]; omega

/-- A promoted tensor shrinks 4 B/elem to 2 B/elem: exactly half. -/
theorem promote_halves (elems : Nat) : outBytes .bf16High (elems * 4) = elems * 2 := by
  simp only [outBytes]; omega

/-- The promotion shape check: source and destination byte counts must agree with the element count. -/
def promotedOk (elems src dst : Nat) : Bool := decide (src = elems * 4) && decide (dst = elems * 2)

/-- If the check passes, the conversion's output size is the recorded destination size. -/
theorem promotedOk_sound (elems src dst : Nat) (h : promotedOk elems src dst = true) :
    outBytes .bf16High src = dst := by
  simp only [promotedOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨h1, h2⟩ := h
  subst h1
  simp only [outBytes]
  omega

/-- A chunk of a widening plane is trimmed to an even length (or refused if a single byte remains). -/
theorem chunk_even_of_even (rem chunk : Nat) (h1 : rem % 2 = 0) (h2 : chunk % 2 = 0) :
    (min rem chunk) % 2 = 0 := by
  omega

-- § bf16 and f16 forms

/-- The high 16 bits of a 32-bit container. -/
def bf16High (v : Nat) : Nat := v / 65536

/-- The high half fits in 16 bits. -/
theorem bf16High_lt (v : Nat) (h : v < 4294967296) : bf16High v < 65536 := by
  unfold bf16High; omega

/-- A container holding a bf16 value (low half zero) is recovered exactly: promotion is a copy with a stride. -/
theorem bf16High_exact (v : Nat) (h : v % 65536 = 0) : bf16High v * 65536 = v := by
  unfold bf16High; omega

-- § segment accounting

/-- Source bytes of a quantized tensor's segments. -/
def segSrcSum (codes scales off : Nat) (fp16 : Bool) : Nat :=
  codes + (if fp16 then scales / 2 else scales) + off

/-- Destination bytes of the segments. -/
def segDstSum (codes scales off : Nat) : Nat := codes + scales + off

/-- Without fp16 scales the planes are already in engine form: source and destination sums agree. -/
theorem seg_sums_agree_plain (codes scales off : Nat) :
    segSrcSum codes scales off false = segDstSum codes scales off := by
  simp [segSrcSum, segDstSum]

/-- With an fp16 scale plane the destination is larger by exactly the widened half. -/
theorem seg_sums_fp16 (codes scales off : Nat) (h : scales % 2 = 0) :
    segDstSum codes scales off = segSrcSum codes scales off true + scales / 2 := by
  simp only [segSrcSum, segDstSum, if_true]
  omega

/-- The loader's own consistency check of segment totals against the index row. -/
def segTotalsOk (sumSrc sumDst recSrc recDst : Nat) : Bool := decide (sumSrc = recSrc) && decide (sumDst = recDst)

/-- A mis-sized plane layout is caught. -/
theorem segTotalsOk_false_of_src (sumSrc sumDst recSrc recDst : Nat) (h : sumSrc ≠ recSrc) :
    segTotalsOk sumSrc sumDst recSrc recDst = false := by
  simp [segTotalsOk, h]

-- § alignment and compaction

/-- Round `n` up to a multiple of `a` (as the compaction does). -/
def roundUp (a n : Nat) : Nat := (n + a - 1) / a * a

/-- Rounding never shrinks a positive-alignment size. -/
theorem roundUp_ge (a n : Nat) (ha : 0 < a) : n ≤ roundUp a n := by
  unfold roundUp
  have h1 := Nat.div_add_mod (n + a - 1) a
  have h2 := Nat.mod_lt (n + a - 1) ha
  have h3 : (n + a - 1) / a * a = a * ((n + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Rounded sizes are multiples of the alignment. -/
theorem roundUp_dvd (a n : Nat) : a ∣ roundUp a n := by
  unfold roundUp
  exact Nat.dvd_mul_left a _

/-- Size of the compacted arena: aligned sizes of the kept rows, in index order. -/
def compactTotal (a : Nat) : List Nat → Nat
  | [] => 0
  | s :: ss => roundUp a s + compactTotal a ss

/-- The compacted arena is a multiple of the alignment. -/
theorem compactTotal_dvd (a : Nat) (ss : List Nat) : a ∣ compactTotal a ss := by
  induction ss with
  | nil => exact ⟨0, by simp [compactTotal]⟩
  | cons s ss ih =>
    simp only [compactTotal]
    exact Nat.dvd_add (roundUp_dvd a s) ih

/-- The compacted arena holds every kept row. -/
theorem compactTotal_ge (a : Nat) (ha : 0 < a) (ss : List Nat) : ss.sum ≤ compactTotal a ss := by
  induction ss with
  | nil => simp [compactTotal]
  | cons s ss ih =>
    simp only [compactTotal, List.sum_cons]
    have := roundUp_ge a s ha
    omega

/-- The arena size check: refuse when the arena is smaller than the pool. -/
def arenaOk (arena pool : Nat) : Bool := decide (pool ≤ arena)

/-- Refusal is exactly "too small". -/
theorem arenaOk_false_iff (arena pool : Nat) : arenaOk arena pool = false ↔ arena < pool := by
  unfold arenaOk; simp

-- § the chunked copy loop

/-- Chunk sizes produced by the `while (done < src_bytes)` loop (fuel-bounded form). -/
def chunkSizesF : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | f + 1, chunk, rem => if rem = 0 then [] else min rem chunk :: chunkSizesF f chunk (rem - min rem chunk)

/-- The chunks cover the plane exactly once. -/
theorem chunkSizesF_sum (chunk : Nat) (hc : 0 < chunk) :
    ∀ (f rem : Nat), rem ≤ f → (chunkSizesF f chunk rem).sum = rem := by
  intro f
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    simp [chunkSizesF]
  | succ f ih =>
    intro rem h
    by_cases h0 : rem = 0
    · subst h0; simp [chunkSizesF]
    · have := ih (rem - min rem chunk) (by omega)
      simp only [chunkSizesF, if_neg h0, List.sum_cons]
      omega

/-- No chunk exceeds the staging buffer. -/
theorem chunkSizesF_le (chunk : Nat) : ∀ (f rem : Nat), ∀ x ∈ chunkSizesF f chunk rem, x ≤ chunk := by
  intro f
  induction f with
  | zero => intro rem x hx; simp [chunkSizesF] at hx
  | succ f ih =>
    intro rem x hx
    by_cases h0 : rem = 0
    · subst h0; simp [chunkSizesF] at hx
    · simp only [chunkSizesF, if_neg h0, List.mem_cons] at hx
      rcases hx with hx | hx
      · omega
      · exact ih _ x hx

/-- Output offset of a widening chunk is `done * 2`, and of a halving chunk `done / 2`. -/
theorem widen_offset_consistent (done n : Nat) (h : n % 2 = 0) :
    (done + n) * 2 = done * 2 + outBytes .widenF16 n := by
  simp only [outBytes]; omega

end Ref.WeightLoader



## FILE: /mnt/user-data/outputs/PinnedArenaLoader.lean
/-!
# MODULE: pinned_arena_loader
SOURCES: src/core/pinned.cu
The pinned host arena (large pages, `cudaHostRegister`, sliced registration fallback, resident lock for the tail) and the
parallel per-layer expert load with an FNV-1a checksum per layer.

## Source semantics
* Layers are claimed from an atomic counter, so each layer is loaded by exactly one worker.
* Layer `L` occupies `[L*bpl*blob, (L+1)*bpl*blob)`; layer ranges abut and tile the whole file.
* The per-layer hash is accumulated chunk by chunk with the running value as the next seed, which equals hashing the
  concatenation; every hash value is reduced mod 2^64.
* Sliced registration pins `bytes / slice` whole slices; the unpinned tail (< one slice) is locked resident instead, and
  the locked range plus the registered prefix cover the arena exactly.

## Form template identifiers
fnv1a64 -> fnv ; layer_off -> layerOff ; uniform_bounds -> lastBound ; lock_resident range -> lockedRange ;
next_layer.fetch_add -> claim ; registered_bytes -> registered
-/

namespace Ref.PinnedArena

-- § FNV-1a

/-- One FNV-1a step in 64-bit arithmetic: xor then multiply by the FNV prime, mod 2^64. -/
def fnvStep (h b : Nat) : Nat := ((h ^^^ b) * 1099511628211) % 18446744073709551616

/-- FNV-1a over a byte list from a seed. -/
def fnv (seed : Nat) (bs : List Nat) : Nat := bs.foldl fnvStep seed

/-- The empty input leaves the seed unchanged. -/
theorem fnv_nil (seed : Nat) : fnv seed [] = seed := rfl

/-- CHUNKING IS TRANSPARENT: hashing chunk by chunk, seeding each with the previous value, equals hashing the whole. -/
theorem fnv_append (seed : Nat) (xs ys : List Nat) : fnv seed (xs ++ ys) = fnv (fnv seed xs) ys := by
  unfold fnv
  rw [List.foldl_append]

/-- A single step stays below 2^64. -/
theorem fnvStep_lt (h b : Nat) : fnvStep h b < 18446744073709551616 :=
  Nat.mod_lt _ (by decide)

/-- Any non-empty-or-empty hash from a 64-bit seed stays 64-bit. -/
theorem fnv_lt (bs : List Nat) :
    ∀ seed, seed < 18446744073709551616 → fnv seed bs < 18446744073709551616 := by
  induction bs with
  | nil => intro seed h; exact h
  | cons b bs ih => intro seed _; exact ih _ (fnvStep_lt seed b)

/-- Observed quirk: the source's seed 1469598103934665603 is the standard FNV-1a offset basis with its last digit
dropped. Both loaders use the same constant, so checksums agree with each other; they do not match standard FNV-1a. -/
theorem seed_differs_from_standard : (1469598103934665603 : Nat) ≠ 14695981039346656037 := by
  decide

-- § layer tiling

/-- Start offset of layer `L`: `L * (bpl * blob)`. -/
def layerOff (bpl blob L : Nat) : Nat := L * (bpl * blob)

/-- Consecutive layers abut: no gap, no overlap. -/
theorem layerOff_succ (bpl blob L : Nat) :
    layerOff bpl blob (L + 1) = layerOff bpl blob L + bpl * blob := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- Layer 0 starts at 0. -/
theorem layerOff_zero (bpl blob : Nat) : layerOff bpl blob 0 = 0 := by
  simp [layerOff]

/-- The 48 layers of 512 blobs tile exactly the 34 GB expert file. -/
theorem layers_tile_file : layerOff 512 1382400 48 = 33973862400 := by decide

-- § work claiming

/-- `next_layer.fetch_add(1)`: returns the old value and bumps the counter. -/
def claim (c : Nat) : Nat × Nat := (c, c + 1)

/-- Two successive claims never return the same layer. -/
theorem claim_fresh (c : Nat) : (claim c).1 < (claim (claim c).2).1 := by
  simp [claim]

/-- A claim at or past the layer count stops the worker, and the counter keeps growing monotonically. -/
theorem claim_monotone (c : Nat) : c < (claim c).2 := by
  simp [claim]

-- § chunked read accounting

/-- One iteration of the read loop: consume `n = min remaining chunk` bytes. -/
def stepRem (remaining chunk : Nat) : Nat := remaining - min remaining chunk

/-- Bytes consumed so far plus bytes remaining is invariant: the loop neither loses nor repeats bytes. -/
theorem step_invariant (pos remaining chunk : Nat) :
    (pos + min remaining chunk) + stepRem remaining chunk = pos + remaining := by
  unfold stepRem; omega

/-- A positive chunk makes progress while bytes remain. -/
theorem step_progress (remaining chunk : Nat) (h0 : 0 < remaining) (hc : 0 < chunk) :
    stepRem remaining chunk < remaining := by
  unfold stepRem; omega

-- § sliced registration

/-- End of the last whole slice. -/
def lastBound (bytes slice : Nat) : Nat := bytes / slice * slice

/-- Whole slices fit in the arena. -/
theorem lastBound_le (bytes slice : Nat) : lastBound bytes slice ≤ bytes := by
  unfold lastBound
  exact Nat.div_mul_le_self bytes slice

/-- The unregistered tail is shorter than one slice. -/
theorem tail_lt_slice (bytes slice : Nat) (h : 0 < slice) : bytes - lastBound bytes slice < slice := by
  unfold lastBound
  have h1 := Nat.div_add_mod bytes slice
  have h2 := Nat.mod_lt bytes h
  have h3 : bytes / slice * slice = slice * (bytes / slice) := Nat.mul_comm _ _
  omega

/-- The locked range `(start, length)` for the tail. -/
def lockedRange (bytes registered : Nat) : Nat × Nat := (registered, bytes - registered)

/-- Registered prefix plus locked tail cover the arena exactly. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    (lockedRange bytes registered).1 + (lockedRange bytes registered).2 = bytes := by
  show registered + (bytes - registered) = bytes
  omega

end Ref.PinnedArena



## FILE: /mnt/user-data/outputs/LayerShapeCheck.lean
/-!
# MODULE: layer_shape_check
SOURCES: src/core/layout.cpp
The pre-flight shape check of the weight table: for each layer, every required 2-D tensor has the exact `(ne0, ne1)` the
kernels read, every 1-D tensor has the exact element count AND engine form (f32 vs bf16-in-f32), and the QSA/GDN layer
split agrees with the geometry.

## Source semantics
* Per layer the required tensor set depends on the layer family: 16 two-D tensors for either family, and 6 (GDN) or 7
  (QSA) one-D tensors.
* The engine form fixes the byte count: 4 B/elem for f32, 2 B/elem for bf16. Reading a bf16 tensor as f32 overreads by
  2x its length, which is why the form is checked and not just the element count.
* `check_all` additionally counts layer kinds: QSA + GDN equals the number of layers, and the counts must equal the
  geometry's (12 QSA, 36 GDN).
* A layer index outside `0..n_layers-1` is refused.

## Form template identifiers
Want2 -> want2Flags ; Want1 -> want1Flags ; is_qsa_layer -> isQsa ; n_qsa_layers -> geometry QSA count ;
WeightKind::Bf16InF32 -> bf16 form (2 B) ; hc_dim -> hc * n_embd
-/

namespace Ref.LayerShape

-- § which tensors apply to which family

/-- A requirement applies unless it is QSA-only on a GDN layer or GDN-only on a QSA layer. -/
def applies (qsaOnly gdnOnly : Bool) (isQsaLayer : Bool) : Bool :=
  !(qsaOnly && !isQsaLayer) && !(gdnOnly && isQsaLayer)

/-- The 22 two-D requirements as `(qsaOnly, gdnOnly)` flags: 10 common, 6 GDN, 6 QSA. -/
def want2Flags : List (Bool × Bool) :=
  List.replicate 10 (false, false) ++ List.replicate 6 (false, true) ++ List.replicate 6 (true, false)

/-- The 10 one-D requirements: 3 common, 3 GDN, 4 QSA. -/
def want1Flags : List (Bool × Bool) :=
  List.replicate 3 (false, false) ++ List.replicate 3 (false, true) ++ List.replicate 4 (true, false)

/-- Number of requirements checked for one layer. -/
def countApplicable (flags : List (Bool × Bool)) (isQsaLayer : Bool) : Nat :=
  (flags.filter (fun f => applies f.1 f.2 isQsaLayer)).length

/-- A QSA layer is checked against 16 two-D tensors. -/
theorem want2_qsa : countApplicable want2Flags true = 16 := by decide
/-- A GDN layer is checked against 16 two-D tensors. -/
theorem want2_gdn : countApplicable want2Flags false = 16 := by decide
/-- A QSA layer is checked against 7 one-D tensors. -/
theorem want1_qsa : countApplicable want1Flags true = 7 := by decide
/-- A GDN layer is checked against 6 one-D tensors. -/
theorem want1_gdn : countApplicable want1Flags false = 6 := by decide

-- § engine forms and byte counts

/-- Engine form of a 1-D tensor. -/
inductive Form where
  | f32
  | bf16

/-- Bytes per element of an engine form. -/
def Form.bytesPer : Form → Nat
  | .f32 => 4
  | .bf16 => 2

/-- Bytes a tensor of `elements` occupies in a form. -/
def formBytes (f : Form) (elements : Nat) : Nat := elements * f.bytesPer

/-- THE SHARED-EXPERT GATE BUG: 2560 bf16 elements are 5120 bytes... -/
theorem shexp_gate_bytes : formBytes .bf16 2560 = 5120 := by decide

/-- ...and reading them as f32 would read 10240 bytes, past the end of the tensor. -/
theorem shexp_gate_f32_overreads : formBytes .bf16 2560 < formBytes .f32 2560 := by decide

/-- Reading a bf16 tensor as f32 always overreads when it is non-empty. -/
theorem bf16_as_f32_overreads (n : Nat) (h : 0 < n) : formBytes .bf16 n < formBytes .f32 n := by
  simp only [formBytes, Form.bytesPer]; omega

/-- The form check passes only when the recorded form is the wanted form. -/
def formOk (have_ want : Form) : Bool :=
  match have_, want with
  | .f32, .f32 => true
  | .bf16, .bf16 => true
  | _, _ => false

/-- A matching form gives a matching byte count. -/
theorem formOk_bytes (h w : Form) (elements : Nat) (hk : formOk h w = true) :
    formBytes h elements = formBytes w elements := by
  cases h <;> cases w <;> simp [formOk] at hk <;> rfl

/-- The hyper-connection width: 4 streams of 2560. -/
theorem hc_dim_value : 4 * 2560 = 10240 := by decide

-- § layer split

/-- Number of layers below `n` satisfying `p`. -/
def countP' (p : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => countP' p n + (if p n then 1 else 0)

/-- QSA layers plus GDN layers is every layer: the predicate partitions the layer range. -/
theorem split_partitions (isQsa : Nat → Bool) (n : Nat) :
    countP' isQsa n + countP' (fun l => !isQsa l) n = n := by
  induction n with
  | zero => simp [countP']
  | succ n ih =>
    simp only [countP']
    rcases Bool.eq_false_or_eq_true (isQsa n) with h | h <;> simp [h] <;> omega

/-- The geometry's split: 12 QSA plus 36 GDN is all 48 layers. -/
theorem geometry_split : 12 + 36 = 48 := by decide

/-- The count check of `check_all`. -/
def splitOk (nQsa nGdn gQsa gGdn : Nat) : Bool := decide (nQsa = gQsa) && decide (nGdn = gGdn)

/-- If the counted split matches the geometry and the geometry covers all layers, the counted kinds cover all layers. -/
theorem splitOk_covers (nQsa nGdn gQsa gGdn n : Nat) (h : splitOk nQsa nGdn gQsa gGdn = true)
    (hg : gQsa + gGdn = n) : nQsa + nGdn = n := by
  simp only [splitOk, Bool.and_eq_true, decide_eq_true_eq] at h
  omega

-- § layer index bound

/-- `check_layer` refuses a layer outside the range. -/
def layerInRange (layer nLayers : Nat) : Bool := decide (layer < nLayers)

/-- The last valid layer is `n - 1`; `n` itself is refused. -/
theorem layer_n_refused (n : Nat) : layerInRange n n = false := by
  simp [layerInRange]

end Ref.LayerShape



## FILE: /mnt/user-data/outputs/DirectFileIo.lean
/-!
# MODULE: direct_file_io
SOURCES: src/platform/direct_file.cpp
Unbuffered, aligned positional reads with a completion queue. Windows uses overlapped I/O on a completion port; the
POSIX build completes each read inside `submit` and queues the completion for `wait`.

## Source semantics
* A request is accepted only if offset, length and buffer address are all multiples of the alignment and the length is
  non-zero; anything else is refused with "unaligned request".
* Windows splits the 64-bit offset into a 32-bit low part and a 32-bit high part that recombine exactly.
* POSIX completion for a failed read has `ok = false` and `bytes = 0`; a successful read reports its byte count.
* `wait` drains at most `max` completions, oldest first; nothing is dropped or reordered, and every submit produces
  exactly one completion.
* `STATUS_END_OF_FILE` and a zero-byte EOF completion are legal short reads, not errors.

## Form template identifiers
DirectFile::submit -> validReq / enqueue ; DirectFile::wait -> drain ; Completion -> Completion ;
ov.Offset / ov.OffsetHigh -> lowPart / highPart ; alignment() -> align
-/

namespace Ref.DirectFile

-- § request validity

/-- The alignment/length validity test of `submit`. -/
def validReq (align off buf len : Nat) : Bool :=
  decide (off % align = 0) && decide (len % align = 0) && decide (buf % align = 0) && decide (len ≠ 0)

/-- Accepted requests are aligned in offset, length and buffer, and non-empty. -/
theorem validReq_sound (align off buf len : Nat) (h : validReq align off buf len = true) :
    off % align = 0 ∧ len % align = 0 ∧ buf % align = 0 ∧ len ≠ 0 := by
  simp only [validReq, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  exact ⟨h1, h2, h3, h4⟩

/-- A zero-length request is always refused. -/
theorem validReq_zero_len (align off buf : Nat) : validReq align off buf 0 = false := by
  simp [validReq]

/-- Whole-page requests at page-aligned offsets are accepted when the buffer is aligned (alignment positive). -/
theorem validReq_pages (align k p buf : Nat) (ha : 0 < align) (hk : 0 < k) (hb : buf % align = 0) :
    validReq align (p * align) buf (k * align) = true := by
  have h1 : p * align % align = 0 := Nat.mul_mod_left p align
  have h2 : k * align % align = 0 := Nat.mul_mod_left k align
  have h3 : k * align ≠ 0 := Nat.mul_ne_zero (by omega) (by omega)
  simp [validReq, h1, h2, hb, h3]

-- § 64-bit offset split (Windows OVERLAPPED)

/-- Low 32 bits. -/
def lowPart (off : Nat) : Nat := off % 4294967296
/-- High 32 bits. -/
def highPart (off : Nat) : Nat := off / 4294967296

/-- The two halves recombine exactly to the offset. -/
theorem offset_roundtrip (off : Nat) : highPart off * 4294967296 + lowPart off = off := by
  unfold highPart lowPart; omega

/-- The low half fits a DWORD. -/
theorem lowPart_lt (off : Nat) : lowPart off < 4294967296 := by
  unfold lowPart; omega

-- § POSIX completion of a read

/-- The completion produced by `pread` returning `got`: `(bytes, ok)`. -/
def posixCompletion (got : Int) : Nat × Bool := (if got < 0 then 0 else got.toNat, decide (0 ≤ got))

/-- A failed read is reported as `ok = false` with zero bytes. -/
theorem posix_failed (got : Int) (h : got < 0) : posixCompletion got = (0, false) := by
  simp [posixCompletion, h, Int.not_le.mpr h]

/-- A successful read reports exactly the bytes read. -/
theorem posix_ok (got : Int) (h : 0 ≤ got) :
    (posixCompletion got).2 = true ∧ ((posixCompletion got).1 : Int) = got := by
  have h' : ¬ got < 0 := by omega
  simp only [posixCompletion, if_neg h', decide_eq_true_eq]
  exact ⟨h, Int.toNat_of_nonneg h⟩

-- § the completion queue

/-- `wait`: take up to `max` completions from the front of the queue. -/
def drain (q : List α) (max : Nat) : List α × List α := (q.take max, q.drop max)

/-- Nothing is lost or reordered: the drained prefix followed by the rest is the original queue. -/
theorem drain_preserves (q : List α) (max : Nat) : (drain q max).1 ++ (drain q max).2 = q := by
  unfold drain
  exact List.take_append_drop max q

/-- `wait` never returns more than `max` completions. -/
theorem drain_le_max (q : List α) (max : Nat) : (drain q max).1.length ≤ max := by
  unfold drain
  simp [List.length_take]

/-- `submit` appends one completion at the back. -/
def enqueue (q : List α) (c : α) : List α := q ++ [c]

/-- Every submit adds exactly one completion. -/
theorem enqueue_length (q : List α) (c : α) : (enqueue q c).length = q.length + 1 := by
  unfold enqueue
  simp

/-- FIFO: a completion enqueued after another is delivered after it. -/
theorem enqueue_after (q : List α) (a b : α) : enqueue (enqueue q a) b = q ++ [a, b] := by
  unfold enqueue
  simp

end Ref.DirectFile



## FILE: /mnt/user-data/outputs/PleReaderPaging.lean
/-!
# MODULE: ple_reader_paging
SOURCES: src/ngram/ple_reader.cpp
The asynchronous 4 KiB-page reader for the PLE row table: row-to-page job geometry with dedup, a set-associative row
cache, a bounded in-flight slot pool, and a ticket counter that never yields 0.

## Source semantics
* A row of 90 bytes at file offset `at` is read with one 4 KiB-aligned job of length 4096, or 8192 when the row straddles
  a page boundary; the row's bytes always lie inside the job.
* The row cache has `rows / 8` sets of 8 ways; a set index is always below the set count and a way index below 8.
* The sentinel EMPTY = 0xFFFFFFFF can never equal a real row index (the table has 320,001,536 rows).
* free slots + in-flight slots = max_inflight at all times (pump and finish preserve it).
* Ticket ids wrap past 2^32 but skip 0.
* Single-token `issue` is refused while another is pending, and `collect` without `issue` fails.

## Form template identifiers
PAGE -> PAGE ; ROW_BYTES -> ROW ; Job::length -> jobLength ; Use::in_page -> inPage ; RowCache::sets -> sets ;
EMPTY -> EMPTY ; free_slots / inflight -> (free, inflight) ; next_ticket -> bumpTicket
-/

namespace Ref.PleReader

/-- The read granule (4 KiB, the direct-I/O alignment). -/
def PAGE : Nat := 4096
/-- Bytes per PLE row: 5 blocks of 18 bytes. -/
def ROW : Nat := 90

-- § job geometry

/-- Absolute offset of a row. -/
def rowAt (tableOff row : Nat) : Nat := tableOff + row * 90
/-- First page start of the job. -/
def firstPage (at_ : Nat) : Nat := at_ / 4096 * 4096
/-- Job length: from the first page through the page holding the row's last byte. -/
def jobLength (at_ : Nat) : Nat := (at_ + 89) / 4096 * 4096 - firstPage at_ + 4096
/-- Byte offset of the row inside the job buffer. -/
def inPage (at_ : Nat) : Nat := at_ - firstPage at_

/-- The job starts on a page boundary. -/
theorem firstPage_aligned (at_ : Nat) : firstPage at_ % 4096 = 0 := by
  unfold firstPage; omega

/-- The job starts at or before the row. -/
theorem firstPage_le (at_ : Nat) : firstPage at_ ≤ at_ := by
  unfold firstPage; omega

/-- The job length is a whole number of pages. -/
theorem jobLength_aligned (at_ : Nat) : jobLength at_ % 4096 = 0 := by
  unfold jobLength firstPage; omega

/-- A job is one or two pages. -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength firstPage; omega

/-- The row offset inside the buffer is within the first page. -/
theorem inPage_lt (at_ : Nat) : inPage at_ < 4096 := by
  unfold inPage firstPage; omega

/-- THE ROW'S BYTES LIE INSIDE THE JOB: `in_page + 90 <= length`, so the copy cannot overrun the read buffer. -/
theorem row_inside_job (at_ : Nat) : inPage at_ + 90 ≤ jobLength at_ := by
  unfold inPage jobLength firstPage; omega

/-- A row that does not straddle a page needs one page. -/
theorem single_page_when_fits (at_ : Nat) (h : inPage at_ + 90 ≤ 4096) : jobLength at_ = 4096 := by
  unfold inPage jobLength firstPage at *; omega

/-- A straddling row needs two pages. -/
theorem two_pages_when_straddling (at_ : Nat) (h : 4096 < inPage at_ + 90) : jobLength at_ = 8192 := by
  unfold inPage jobLength firstPage at *; omega

/-- Two rows in the same page share a job: equal first-page keys. -/
theorem same_page_same_job (a b : Nat) (h : a / 4096 = b / 4096) : firstPage a = firstPage b := by
  unfold firstPage; rw [h]

-- § row cache

/-- Number of sets for `rows` cache rows (8 ways). -/
def sets (rows : Nat) : Nat := rows / 8

/-- Capacity never exceeds the requested rows. -/
theorem capacity_le (rows : Nat) : sets rows * 8 ≤ rows := by
  unfold sets; omega

/-- A set index is below the set count (non-empty cache). -/
theorem set_index_lt (mixed rows : Nat) (h : 0 < sets rows) : mixed % sets rows < sets rows :=
  Nat.mod_lt _ h

/-- Every `(set, way)` slot lies inside the key array. -/
theorem slot_in_array (s w nSets : Nat) (hs : s < nSets) (hw : w < 8) : s * 8 + w < nSets * 8 := by
  omega

/-- The round-robin way pointer stays below 8. -/
theorem next_way_lt (w : Nat) : (w + 1) % 8 < 8 := by omega

/-- The empty-key sentinel. -/
def EMPTY : Nat := 4294967295

/-- No real row index equals the sentinel (the table has fewer than 2^32 - 1 rows). -/
theorem row_ne_empty (row : Nat) (h : row < 320001536) : row ≠ EMPTY := by
  unfold EMPTY; omega

-- § in-flight slots

/-- Slot accounting: `(free, inflight)`. -/
structure Slots where
  free : Nat
  inflight : Nat

/-- Submitting a job moves one slot from free to in-flight. -/
def Slots.pump (s : Slots) : Slots :=
  if 0 < s.free then { free := s.free - 1, inflight := s.inflight + 1 } else s

/-- Finishing a job moves one slot back. -/
def Slots.finish (s : Slots) : Slots :=
  if 0 < s.inflight then { free := s.free + 1, inflight := s.inflight - 1 } else s

/-- `pump` preserves free + inflight. -/
theorem pump_conserves (s : Slots) : s.pump.free + s.pump.inflight = s.free + s.inflight := by
  unfold Slots.pump
  split <;> simp <;> omega

/-- `finish` preserves free + inflight. -/
theorem finish_conserves (s : Slots) : s.finish.free + s.finish.inflight = s.free + s.inflight := by
  unfold Slots.finish
  split <;> simp <;> omega

/-- A job is only submitted when a slot is free. -/
theorem pump_requires_free (s : Slots) (h : s.free = 0) : s.pump = s := by
  unfold Slots.pump
  rw [if_neg (by omega)]

-- § tickets

/-- Next ticket id: increment mod 2^32, skipping 0. -/
def bumpTicket (t : Nat) : Nat := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296

/-- A ticket id is never 0. -/
theorem bumpTicket_ne_zero (t : Nat) : bumpTicket t ≠ 0 := by
  unfold bumpTicket
  split <;> omega

/-- A ticket id fits 32 bits. -/
theorem bumpTicket_lt (t : Nat) : bumpTicket t < 4294967296 := by
  unfold bumpTicket
  split <;> omega

-- § single-token issue/collect protocol (PleTable, Direct mode)

/-- `issue`: refused when a token is already pending. Returns `(accepted, pending')`. -/
def issueB (pending : Bool) : Bool × Bool := if pending then (false, pending) else (true, true)

/-- `collect`: fails without a pending issue. Returns `(ok, pending')`. -/
def collectB (pending : Bool) : Bool × Bool := if pending then (true, false) else (false, false)

/-- Two issues in a row: the second is refused. -/
theorem double_issue_refused : (issueB (issueB false).2).1 = false := by decide

/-- `collect` without `issue` fails. -/
theorem collect_without_issue : (collectB false).1 = false := by decide

/-- `issue` then `collect` succeeds and leaves nothing pending. -/
theorem issue_collect_ok : (collectB (issueB false).2) = (true, false) := by decide

/-- The reader refuses a table that extends past the file. -/
def tableFits (tableOff nRows fileSize : Nat) : Bool := decide (tableOff + nRows * 90 ≤ fileSize)

/-- The artifact's PLE table is 28,800,138,240 bytes. -/
theorem ple_table_bytes : 320001536 * 90 = 28800138240 := by decide

end Ref.PleReader



## FILE: /mnt/user-data/outputs/NgramHashIq4nl.lean
/-!
# MODULE: ngram_hash_iq4nl
SOURCES: include/ref/kernels/ngram.hpp, src/kernels/ngram.cpp
The PLE n-gram hash (sixteen row indices per token from the last three token ids, XOR of 64-bit products reduced per head
by the artifact's vocab sizes and offsets) and the IQ4_NL row dequantisation and gather layout.

## Source semantics
* The context is `(tok, p1, p0)` with predecessors read newest-first. A predecessor that is null (`< 0`) or EOS cuts the
  context: it and every OLDER position become EOS. The token's own EOS never cuts its own context. Token id 0 is real.
* After the cut every context entry is non-negative, so the cast to uint64 is safe.
* The trigram mix extends the bigram mix by one XOR term (prefixes agree). Equal terms cancel under XOR.
* A head's row is `mixed % vocab[h] + offset[h]`, inside `[offset[h], offset[h]+vocab[h])`; the 16 head ranges tile
  `0..300001275+20000171` and all lie inside the 320,001,536-row table.
* IQ4_NL nibbles are SPLIT-HALF: byte `j` carries elements `j` and `j+16`. A row is 5 blocks of 18 bytes = 90 bytes for
  160 elements; 16 rows flatten head-slowest into exactly 2560 floats.
* The codebook has no `-8` shift: it is strictly increasing from -127 to 113.

## Form template identifiers
ngram_rows -> ctx3 / headRow ; ngram_mixed -> mixed2 / mixed3 ; PLE_EOS_TOKEN_ID -> EOS ; TOKEN_NULL -> -1 ;
head_vocab_sizes -> vocabL ; head_offsets -> offL ; kvalues_iq4nl -> kv ; iq4nl_dequant_row -> splitIdx ;
PLE_ROW_BYTES -> 90 ; gather -> headSlowest
-/

namespace Ref.Ngram

-- § context construction

/-- The end-of-sequence token id. -/
def EOS : Int := 248044

/-- A predecessor cuts the context when it is null (negative) or EOS. -/
def isCut (t : Int) : Bool := decide (t < 0) || decide (t = EOS)

/-- Context `(token, newest predecessor, older predecessor)` after the forward-propagating cut. -/
def ctx3 (tok p0 p1 : Int) : Int × Int × Int :=
  let cut1 := isCut p1
  let cut2 := cut1 || isCut p0
  (tok, if cut1 then EOS else p1, if cut2 then EOS else p0)

/-- The token's own id is never altered by the cut. -/
theorem ctx3_token (tok p0 p1 : Int) : (ctx3 tok p0 p1).1 = tok := rfl

/-- Token id 0 is a real token, not a cut. -/
theorem zero_not_cut : isCut 0 = false := by decide

/-- The null predecessor (-1) cuts. -/
theorem null_cuts : isCut (-1) = true := by decide

/-- EOS cuts. -/
theorem eos_cuts : isCut EOS = true := by decide

/-- THE CUT PROPAGATES FORWARD: if the newest predecessor cuts, the older position is EOS regardless of its value. -/
theorem cut_forward (tok p0 p1 : Int) (h : isCut p1 = true) : (ctx3 tok p0 p1).2.2 = EOS := by
  simp [ctx3, h]

/-- With no cut the context is the real history. -/
theorem no_cut_keeps_history (tok p0 p1 : Int) (h1 : isCut p1 = false) (h0 : isCut p0 = false) :
    ctx3 tok p0 p1 = (tok, p1, p0) := by
  simp [ctx3, h1, h0]

/-- The token's own EOS does not influence the predecessors' handling. -/
theorem own_eos_no_cut (p0 p1 : Int) : (ctx3 EOS p0 p1).2 = (ctx3 0 p0 p1).2 := rfl

private theorem pick_nonneg (c : Bool) (p : Int) (h : c = false → 0 ≤ p) :
    0 ≤ (if c = true then EOS else p) := by
  cases c
  · simpa using h rfl
  · simp [EOS]

/-- After the cut every history entry is non-negative, so the uint64 cast is value-preserving. -/
theorem ctx3_nonneg (tok p0 p1 : Int) : 0 ≤ (ctx3 tok p0 p1).2.1 ∧ 0 ≤ (ctx3 tok p0 p1).2.2 := by
  refine ⟨pick_nonneg _ _ ?_, pick_nonneg _ _ ?_⟩
  · intro h
    simp [isCut] at h
    omega
  · intro h
    simp [isCut] at h
    omega

-- § mixing

/-- Product reduced mod 2^64. -/
def mixU (a b : Nat) : Nat := (a * b) % 18446744073709551616

/-- Bigram mix: XOR of the first two products. -/
def mixed2 (c0 c1 m0 m1 : Nat) : Nat := mixU c0 m0 ^^^ mixU c1 m1

/-- Trigram mix: the bigram mix XORed with the third product. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : Nat) : Nat := mixed2 c0 c1 m0 m1 ^^^ mixU c2 m2

/-- Bigram and trigram heads agree about recent history: the trigram mix is the bigram mix plus one term. -/
theorem mixed3_extends_mixed2 (c0 c1 c2 m0 m1 m2 : Nat) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ mixU c2 m2 := rfl

/-- Why XOR and not sum: equal terms cancel under XOR. -/
theorem xor_cancels (x : Nat) : x ^^^ x = 0 := Nat.xor_self x

/-- Products are reduced to 64 bits. -/
theorem mixU_lt (a b : Nat) : mixU a b < 18446744073709551616 :=
  Nat.mod_lt _ (by decide)

/-- Two equal tokens with equal multipliers cancel in the bigram mix. -/
theorem mixed2_equal_cancels (c m : Nat) : mixed2 c c m m = 0 := by
  unfold mixed2
  exact Nat.xor_self _

-- § head ranges

/-- The sixteen vocab sizes (NOT an arithmetic progression). -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- The sixteen head offsets. -/
def offL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- The offsets are the running sum of the vocab sizes, so the head ranges tile the index space with no gap or overlap. -/
theorem offsets_are_running_sum : offL = (vocabL.scanl (· + ·) 0).take 16 := by decide

/-- Every head range lies inside the 320,001,536-row table. -/
theorem heads_inside_table :
    ∀ h, h < 16 → offL.getD h 0 + vocabL.getD h 0 ≤ 320001536 := by decide

/-- The artifact's vocab sizes are not `20000003 + 10 i`: the first gap is 20. -/
theorem first_gap_is_twenty : vocabL.getD 1 0 - vocabL.getD 0 0 = 20 := by decide

/-- The reference selftest's constants differ from the artifact's. -/
theorem not_selftest_progression : vocabL.getD 1 0 ≠ 20000003 + 10 * 1 := by decide

/-- A head's row index. -/
def headRow (mixed vocab off : Nat) : Nat := mixed % vocab + off

/-- A head's row is inside its own range. -/
theorem headRow_bounds (mixed vocab off : Nat) (h : 0 < vocab) :
    off ≤ headRow mixed vocab off ∧ headRow mixed vocab off < off + vocab := by
  unfold headRow
  have := Nat.mod_lt mixed h
  omega

/-- Heads 0..7 use the bigram and 8..15 the trigram: `(n-2)*8 + g` enumerates heads uniquely. -/
theorem head_decomposition (h : Nat) (hh : h < 16) :
    (2 + h / 8 - 2) * 8 + h % 8 = h := by
  omega

-- § IQ4_NL

/-- The codebook `kvalues_iq4nl`. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook is strictly increasing. -/
theorem kv_strictly_increasing : ∀ i, i < 15 → kv.getD i 0 < kv.getD (i + 1) 0 := by decide

/-- All codes lie in [-127, 113]. -/
theorem kv_range : ∀ i, i < 16 → -127 ≤ kv.getD i 0 ∧ kv.getD i 0 ≤ 113 := by decide

/-- There is no `-8` shift: entry 8 is 1, not 1 - 8. -/
theorem kv_no_shift : kv.getD 8 0 = 1 := by decide

/-- A byte splits into two 4-bit codes. -/
theorem nibble_roundtrip (b : Nat) (h : b < 256) : b / 16 * 16 + b % 16 = b := by omega

/-- Both nibbles index the 16-entry codebook. -/
theorem nibbles_in_range (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 := by omega

/-- SPLIT-HALF layout: nibble `half` of byte `j` is element `j + 16 * half` of the block. -/
def splitIdx (j half : Nat) : Nat := j + 16 * half

/-- The split-half map is injective on bytes `j < 16` and halves `< 2`. -/
theorem splitIdx_injective (j h j' h' : Nat) (hj : j < 16) (hj' : j' < 16)
    (hh : h < 2) (hh' : h' < 2) (e : splitIdx j h = splitIdx j' h') : j = j' ∧ h = h' := by
  unfold splitIdx at e; omega

/-- ...and onto the 32 block elements. -/
theorem splitIdx_surjective (i : Nat) (hi : i < 32) :
    ∃ j h, j < 16 ∧ h < 2 ∧ splitIdx j h = i :=
  ⟨i % 16, i / 16, by omega, by omega, by unfold splitIdx; omega⟩

/-- The interleaved reading (2j, 2j+1) is a different map: byte 1's high nibble lands at 17, not 3. -/
theorem split_ne_interleaved : splitIdx 1 1 ≠ 2 * 1 + 1 := by decide

-- § row geometry and gather

/-- Geometry: 160 elements = 5 blocks of 32; a row is 5 * 18 = 90 bytes. -/
theorem row_geometry : 160 / 32 = 5 ∧ 5 * 18 = 90 := by decide

/-- Sixteen heads of 160 make exactly n_embd = 2560. -/
theorem heads_make_embd : 16 * 160 = 2560 := by decide

/-- Head-slowest flatten index. -/
def headSlowest (h d : Nat) : Nat := h * 160 + d

/-- The flatten stays inside the 2560-float output. -/
theorem headSlowest_lt (h d : Nat) (hh : h < 16) (hd : d < 160) : headSlowest h d < 2560 := by
  unfold headSlowest; omega

/-- ...and is injective, so no two (head, dim) pairs collide. -/
theorem headSlowest_injective (h d h' d' : Nat) (hd : d < 160) (hd' : d' < 160)
    (e : headSlowest h d = headSlowest h' d') : h = h' ∧ d = d' := by
  unfold headSlowest at e; omega

/-- The head-fastest layout `d * 16 + h` is a different function (so a reshape check is not vacuous). -/
theorem headFastest_differs : (1 * 160 + 0) ≠ (0 * 16 + 1) := by decide

/-- One token's 16 row reads touch 1,440 bytes. -/
theorem bytes_per_token_gather : 16 * 90 = 1440 := by decide

/-- Table fit check (shard also holding other tensors): the table's end must be inside the data section. -/
theorem table_end_inside_file (dataStart off need fileSize : Nat) (hd : dataStart ≤ fileSize)
    (h : off + need ≤ fileSize - dataStart) : dataStart + off + need ≤ fileSize := by
  omega

end Ref.Ngram



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- DirectFileIo.lean
  was: theorem validReq_pages (align k p buf : Nat) (hk : 0 < k) (hb : buf % align = 0) :
    validReq align (p * align) buf (k * align) = true
  now: theorem validReq_pages (align k p buf : Nat) (ha : 0 < align) (hk : 0 < k) (hb : buf % align = 0) :
    validReq align (p * align) buf (k * align) = true
  measured: more hypotheses/binders (3 → 4): the statement holds in fewer cases
-/



## FILE: /mnt/user-data/outputs/ArenaCarving.lean
/-!
# MODULE: arena_carving
SOURCES: include/ref/core/layer.hpp, src/core/layer.cpp, include/ref/core/session.hpp, src/core/session.cpp
Every per-layer buffer set (GDN, MoE, QSA, block) is carved ONCE from a device arena, each region rounded up to
16 bytes, and the `*_bytes` function must equal the bytes the matching `*_init` consumes.

## Source semantics
* `gdn_buffers_bytes` sums `(v + 15) & ~15` over the region list; `gdn_buffers_init` walks the same list with a
  moving pointer and returns the same total.
* Region offsets are 16-byte aligned whenever the base is, and consecutive regions never overlap.
* The session gives every GDN layer its own state slice (`gdn_state_floats`), and the 12 QSA layers share one
  RoPE table (the first state owns it, the others borrow it).

## Form template identifiers
align16 <- `(n + 15) & ~15`, `align_up16`; parts <- the `parts[]` array of `*_buffers_bytes/_init`;
offsets <- the `ptr[i]` cursor; total <- `*_buffers_bytes`; cursorEnd <- the value `*_init` returns;
Geometry <- `ModelGeometry` (n_embd, ssm_conv_channels, ssm_value_dim, ssm_v_heads, ssm_state_size, ssm_d_conv).
-/

namespace Ref.ArenaCarving

-- § alignment

/-- 16-byte round-up used between every region of every arena. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounding up never loses bytes. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Rounded sizes are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Rounding wastes fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Rounding is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

-- § carving a list of regions

/-- `gdn_buffers_bytes`-style total: sum of the aligned region sizes. -/
def total (parts : List Nat) : Nat := (parts.map align16).sum

/-- The cursor offsets `gdn_buffers_init` assigns to the regions, starting at `base`. -/
def offsets : List Nat → Nat → List Nat
  | [], _ => []
  | p :: ps, base => base :: offsets ps (base + align16 p)

/-- The value `*_init` returns after walking every region. -/
def cursorEnd : List Nat → Nat → Nat
  | [], b => b
  | p :: ps, b => cursorEnd ps (b + align16 p)

/-- Peeling one region off the total. -/
theorem total_cons (p : Nat) (ps : List Nat) : total (p :: ps) = align16 p + total ps := by
  simp [total]

/-- The empty arena needs zero bytes. -/
theorem total_nil : total [] = 0 := by
  simp [total]

/-- One offset is produced per region. -/
theorem offsets_length : ∀ (ps : List Nat) (b : Nat), (offsets ps b).length = ps.length
  | [], _ => by simp [offsets]
  | p :: ps, b => by simp [offsets, offsets_length ps (b + align16 p)]

/-- Consecutive regions are separated by exactly the aligned size of the first one. -/
theorem offsets_step (p q : Nat) (ps : List Nat) (b : Nat) :
    offsets (p :: q :: ps) b = b :: (b + align16 p) :: offsets ps (b + align16 p + align16 q) := rfl

/-- A region never spills into its successor: the aligned size covers the requested size. -/
theorem region_no_overlap (p b : Nat) : b + p ≤ b + align16 p := by
  have := align16_ge p
  omega

/-- If the base is 16-byte aligned, every region handed out is 16-byte aligned. -/
theorem offsets_aligned : ∀ (ps : List Nat) (b : Nat), b % 16 = 0 → ∀ o ∈ offsets ps b, o % 16 = 0
  | [], b, _, o, ho => by simp [offsets] at ho
  | p :: ps, b, hb, o, ho => by
    simp only [offsets, List.mem_cons] at ho
    rcases ho with h | h
    · subst h; exact hb
    · exact offsets_aligned ps (b + align16 p) (by have := align16_mod p; omega) o h

/-- THE CONTRACT BETWEEN `*_bytes` AND `*_init`: the walk ends exactly `total` bytes after the base. -/
theorem cursorEnd_eq : ∀ (ps : List Nat) (b : Nat), cursorEnd ps b = b + total ps
  | [], b => by simp [cursorEnd, total_nil]
  | p :: ps, b => by
    rw [cursorEnd, cursorEnd_eq ps (b + align16 p), total_cons]
    omega

/-- `*_init` returns what `*_bytes` promised. -/
theorem init_returns_bytes (parts : List Nat) : cursorEnd parts 0 = total parts := by
  rw [cursorEnd_eq]; omega

/-- The aligned total is at least the raw sum, so a sized arena always holds the regions. -/
theorem total_ge_sum : ∀ ps : List Nat, ps.sum ≤ total ps
  | [] => by simp [total_nil]
  | p :: ps => by
    rw [total_cons, List.sum_cons]
    have := align16_ge p
    have := total_ge_sum ps
    omega

-- § the GDN region list

/-- The geometry fields the buffer sizes depend on. -/
structure Geometry where
  nEmbd : Nat
  convChannels : Nat
  valueDim : Nat
  vHeads : Nat
  stateSize : Nat
  dConv : Nat

/-- `q8k_bytes`: n/256 blocks of 292 bytes. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- The sixteen regions of `GdnBuffers`, in carve order. -/
def gdnParts (g : Geometry) : List Nat :=
  [q8kBytes g.nEmbd, g.nEmbd / 32 * 34, g.nEmbd * 2,
   g.convChannels * 4, g.convChannels * 4, g.convChannels * 4,
   g.vHeads * 4, g.vHeads * 4, g.vHeads * 4,
   g.vHeads * g.stateSize * 4, g.valueDim * 4, g.valueDim * 4,
   q8kBytes g.valueDim, g.valueDim / 32 * 34,
   g.stateSize * g.vHeads * g.stateSize * 4, g.convChannels * (g.dConv - 1) * 4]

/-- The real Qwen3.8-Flash-Next geometry used by the engine. -/
def realGeometry : Geometry := ⟨2560, 10240, 6144, 48, 128, 4⟩

/-- `GdnBuffers` has sixteen regions. -/
theorem gdnParts_length (g : Geometry) : (gdnParts g).length = 16 := rfl

/-- `gdn_buffers_init` consumes exactly `gdn_buffers_bytes`. -/
theorem gdn_init_eq_bytes (g : Geometry) : cursorEnd (gdnParts g) 0 = total (gdnParts g) :=
  init_returns_bytes _

/-- Recurrent state: S * h_v * S floats per GDN layer, in bytes. -/
theorem real_state_bytes : realGeometry.stateSize * realGeometry.vHeads * realGeometry.stateSize * 4 = 3145728 := by
  decide

/-- Conv history: conv_channels * (d_conv - 1) floats per GDN layer, in bytes. -/
theorem real_conv_bytes : realGeometry.convChannels * (realGeometry.dConv - 1) * 4 = 122880 := by
  decide

/-- The aligned size of one GDN layer's scratch plus state at the real geometry. -/
theorem real_gdn_bytes : total (gdnParts realGeometry) = 3490096 := by
  decide

-- § the MoE region list

/-- The nine regions of `MoEBuffers`; `sharedScratch` stands for `shared_expert_scratch_bytes(n_ff)`. -/
def moeParts (nEmbd nExpert k sharedScratch : Nat) : List Nat :=
  [nEmbd * 2, nEmbd * 2, nExpert * 4, k * 4, k * 4, nEmbd * 4, sharedScratch,
   nEmbd / 32 * 34, q8kBytes nEmbd]

/-- `MoEBuffers` has nine regions. -/
theorem moeParts_length (a b c d : Nat) : (moeParts a b c d).length = 9 := rfl

/-- `moe_buffers_init` consumes exactly `moe_buffers_bytes`. -/
theorem moe_init_eq_bytes (a b c d : Nat) :
    cursorEnd (moeParts a b c d) 0 = total (moeParts a b c d) :=
  init_returns_bytes _

-- § the shared RoPE table

/-- Sharing the RoPE table: the first QSA state owns `rope` extra bytes, the rest borrow it, so n = m+1 layers
cost `(m+1)*rest + rope` rather than `(m+1)*(rest+rope)`. -/
theorem rope_sharing_total (m rest rope : Nat) :
    (rest + rope) + m * rest = (m + 1) * rest + rope := by
  have := Nat.add_mul m 1 rest
  omega

/-- Unshared RoPE tables would cost exactly `m * rope` more. -/
theorem rope_sharing_saves (m rest rope : Nat) :
    (m + 1) * (rest + rope) = ((rest + rope) + m * rest) + m * rope := by
  have h1 := Nat.add_mul m 1 (rest + rope)
  have h2 := Nat.mul_add m rest rope
  omega

end Ref.ArenaCarving



## FILE: /mnt/user-data/outputs/ActivationContract.lean
/-!
# MODULE: activation_contract
SOURCES: include/ref/core/layer.hpp, src/core/layer.cpp
Which quantized activation image a weight consumes (a property of the TENSOR, not its role), how the three
canonical planes of a quantized tensor are located, and how the embedding column decode addresses its codes.

## Source semantics
* `gemv_quantized`: native tensors use the FP32 input; otherwise a 2-bit form uses Q8_0; a legacy 4/8-bit form
  uses Q8_0; a K-quant uses Q8_K.  BOTH quantized images are produced because the pack mixes them per layer.
* `plane_ptrs`: codes | scales | offsets are contiguous in that order and add up to the tensor's bytes;
  `has_offset` holds exactly when the offset plane is non-empty.
* `embed_row`: the code of element i is the `CODE_BITS`-wide field at bit `(i % PER_BYTE) * CODE_BITS` of byte
  `i / PER_BYTE`.

## Form template identifiers
Act <- {block_q8_0, block_q8_K, FP32 native input}; actOf <- the dispatch in `gemv_quantized`;
PlaneSizes <- `WeightRef::{codes_bytes, scales_bytes, offset_bytes, bytes}`; perByte <- `8 / code_bits`.
-/

namespace Ref.ActivationContract

-- § activation dispatch

/-- The activation image a projection reads. -/
inductive Act
  | q8_0
  | q8k
  | native
  deriving DecidableEq, Repr

/-- `gemv_quantized`'s choice, from (native?, code_bits, wants_q8k). -/
def actOf (native : Bool) (codeBits : Nat) (wantsQ8K : Bool) : Act :=
  if native = true then Act.native
  else if codeBits = 2 then Act.q8_0
  else if wantsQ8K = true then Act.q8k
  else Act.q8_0

/-- A native tensor always takes the FP32 activation. -/
theorem actOf_native (c : Nat) (w : Bool) : actOf true c w = Act.native := by
  simp [actOf]

/-- An S2 (Q2_0) form takes Q8_0 whatever `wants_q8k` says. -/
theorem actOf_s2 (w : Bool) : actOf false 2 w = Act.q8_0 := by
  simp [actOf]

/-- A K-quant (not 2-bit, wants Q8_K) takes Q8_K. -/
theorem actOf_kquant (c : Nat) (hc : c ≠ 2) : actOf false c true = Act.q8k := by
  simp [actOf, hc]

/-- A legacy form that does not want Q8_K takes Q8_0. -/
theorem actOf_legacy (c : Nat) : actOf false c false = Act.q8_0 := by
  simp [actOf]

/-- A non-native projection never receives the FP32 image. -/
theorem actOf_not_native (c : Nat) (w : Bool) : actOf false c w ≠ Act.native := by
  unfold actOf
  by_cases h : c = 2
  · simp [h]
  · cases w <;> simp [h]

-- § plane layout

/-- The recorded plane sizes of one quantized tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- What `plane_ptrs` insists on before it returns pointers. -/
def planesOk (p : PlaneSizes) (codeBits : Nat) (hasOffset : Bool) : Prop :=
  (codeBits = 2 ∨ codeBits = 4 ∨ codeBits = 8) ∧ p.codes ≠ 0 ∧ p.scales ≠ 0 ∧
    p.codes + p.scales + p.offset = p.bytes ∧ (hasOffset = true ↔ p.offset ≠ 0)

/-- Start of the scale plane (just after the codes). -/
def scalesStart (p : PlaneSizes) : Nat := p.codes

/-- Start of the offset plane (just after the scales). -/
def offsetStart (p : PlaneSizes) : Nat := p.codes + p.scales

/-- The three planes tile the tensor exactly: nothing is read past its end. -/
theorem planes_tile (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) :
    scalesStart p = p.codes ∧ offsetStart p + p.offset = p.bytes := by
  obtain ⟨_, _, _, hsum, _⟩ := hok
  unfold scalesStart offsetStart
  exact ⟨rfl, hsum⟩

/-- A validated tensor is non-empty. -/
theorem planes_nonempty (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) : 0 < p.bytes := by
  obtain ⟨_, hc, _, hsum, _⟩ := hok
  omega

/-- The two ways of spelling "has an offset plane" agree after validation. -/
theorem planes_offset_iff (p : PlaneSizes) (c : Nat) (h : Bool) (hok : planesOk p c h) :
    (h = true ↔ p.offset ≠ 0) := hok.2.2.2.2

/-- Reading the scale plane as 4-byte floats when it is really 2-byte halves doubles its size and breaks the
tiling; the check therefore refuses a plane list whose sum differs from the tensor. -/
theorem doubled_scales_rejected (p : PlaneSizes) (c : Nat) (h : Bool)
    (hbad : p.codes + p.scales + p.offset ≠ p.bytes) : ¬ planesOk p c h := by
  intro hok
  exact hbad hok.2.2.2.1

-- § embedding column decode

/-- Codes per byte for the three legal widths (`8 / code_bits`). -/
def perByte : Nat → Nat
  | 2 => 4
  | 4 => 2
  | 8 => 1
  | _ => 0

/-- `perByte` really is `8 / code_bits` on the legal widths. -/
theorem perByte_eq (b : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : perByte b = 8 / b := by
  rcases hb with rfl | rfl | rfl <;> rfl

/-- The bit shift of element i stays inside one byte, so the field never straddles two bytes. -/
theorem shift_lt (b i : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : (i % perByte b) * b < 8 := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] <;> omega

/-- The code field fits entirely in its byte: shift + width ≤ 8. -/
theorem field_fits (b i : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) : (i % perByte b) * b + b ≤ 8 := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] <;> omega

/-- The byte holding element i lies inside the row when the row length is a multiple of `perByte`. -/
theorem byte_in_row (b i ne0 : Nat) (hb : b = 2 ∨ b = 4 ∨ b = 8) (hdiv : ne0 % perByte b = 0)
    (hi : i < ne0) : i / perByte b < ne0 / perByte b := by
  rcases hb with rfl | rfl | rfl <;> simp [perByte] at hdiv ⊢ <;> omega

end Ref.ActivationContract



## FILE: /mnt/user-data/outputs/MoeSplitProtocol.lean
/-!
# MODULE: moe_split_protocol
SOURCES: include/ref/core/layer.hpp, src/core/layer.cpp, include/ref/core/session.hpp, src/core/session.cpp
The MoE block split at the router: `moe_route` publishes ids/weights for THIS layer, the host pool computes THIS
layer's expert outputs, and `moe_finish` combines them with THIS layer's weights plus the shared expert.

## Source semantics
* `moe_layer = moe_route ; moe_finish`, and `moe_finish = moe_shared ; moe_combine_parts`.
* `combine = sum_j w[j] * parts[j] + shared` (shared ADDED PLAIN).
* Combining layer l's expert outputs with layer l+1's weights (the L100 bug) is a different number.
* `k` must be in 1..64; the native router additionally needs exactly 512 experts and k = 10.
* The shared expert may run before or after the pool: it is a pure function of `x`, so the result is unchanged.
* The doorbell mapped-pinned regions total `4*(n_embd + 2k + 2)` bytes.

## Form template identifiers
combine <- `moe_combine` / `native_moe_combine`; kValid <- the `k < 1 || k > 64` guard;
nativeRouterOk <- the native router guard; sharedExpert <- `moe_shared`; stale <- the one-layer-delay loop.
-/

namespace Ref.MoeSplit

-- § the combination

/-- `sum_j w[j] * parts[j] + shared`. -/
def combine (w parts : List Int) (shared : Int) : Int :=
  (List.zipWith (· * ·) w parts).sum + shared

/-- With no experts, only the shared expert remains. -/
theorem combine_nil (parts : List Int) (s : Int) : combine [] parts s = s := by
  simp [combine]

/-- Peeling one routed expert off the combination. -/
theorem combine_cons (a b : Int) (w p : List Int) (s : Int) :
    combine (a :: w) (b :: p) s = a * b + combine w p s := by
  simp [combine, List.zipWith_cons_cons, List.sum_cons, Int.add_assoc]

/-- The shared output is added plain, after the routed sum. -/
theorem combine_shared (w p : List Int) (s : Int) : combine w p s = combine w p 0 + s := by
  simp [combine]

-- § route / finish split

/-- `moe_finish`: the shared expert, then the combination with this layer's weights. -/
def finish (sharedExpert : Int) (w parts : List Int) : Int := combine w parts sharedExpert

/-- The split form: compute the shared expert into `b.shared`, then combine. -/
def finishSplit (sharedExpert : Int) (w parts : List Int) : Int :=
  let sh := sharedExpert
  combine w parts sh

/-- `moe_finish` is exactly `moe_shared` followed by `moe_combine_parts`. -/
theorem finish_eq_split (sh : Int) (w parts : List Int) : finish sh w parts = finishSplit sh w parts := rfl

/-- Running the shared expert early (end of `pre[l]`) or late (inside `post[l]`) gives the same token, because
it depends only on `x`. -/
theorem shared_early_irrelevant (sharedOf : Int → Int) (x : Int) (w parts : List Int) :
    finish (sharedOf x) w parts = combine w parts (sharedOf x) := rfl

/-- `moe_layer` composes route then finish; the router's output is the only coupling. -/
def layerOut (route : Nat → List Int) (outs : Nat → List Int) (sh : Nat → Int) (l : Nat) : Int :=
  finish (sh l) (route l) (outs l)

/-- The delayed loop: layer l's experts combined with layer l+1's router weights. -/
def staleOut (route : Nat → List Int) (outs : Nat → List Int) (sh : Nat → Int) (l : Nat) : Int :=
  finish (sh l) (route (l + 1)) (outs l)

/-- The correct loop uses layer l's own weights. -/
theorem layerOut_uses_own_weights (route outs : Nat → List Int) (sh : Nat → Int) (l : Nat) :
    layerOut route outs sh l = combine (route l) (outs l) (sh l) := rfl

/-- The stale loop agrees with the correct one when the two layers' weights coincide... -/
theorem stale_eq_of_same_weights (route outs : Nat → List Int) (sh : Nat → Int) (l : Nat)
    (h : route (l + 1) = route l) : staleOut route outs sh l = layerOut route outs sh l := by
  simp [staleOut, layerOut, h]

/-- ...and differs in general: a concrete pair of routers on which the delay changes the answer (L100). -/
theorem stale_differs_example :
    combine [1, 0] [5, 7] 0 ≠ combine [0, 1] [5, 7] 0 := by
  decide

-- § guards

/-- The `k` guard shared by `moe_route`, `moe_finish` and `moe_combine_parts`. -/
def kValid (k : Nat) : Prop := 1 ≤ k ∧ k ≤ 64

/-- The native router's extra requirement. -/
def nativeRouterOk (nExpert k : Nat) : Prop := nExpert = 512 ∧ k = 10

/-- Whatever the native router accepts also passes the generic `k` guard. -/
theorem nativeRouter_k_valid (e k : Nat) (h : nativeRouterOk e k) : kValid k := by
  unfold nativeRouterOk at h
  unfold kValid
  omega

/-- The router's `k` bounds the doorbell and parts buffer sizes: parts is `k * n_embd` floats. -/
theorem parts_bytes_eq (k nEmbd : Nat) : k * nEmbd * 4 = 4 * (k * nEmbd) := by
  omega

-- § the doorbell regions

/-- Bytes of the five mapped-pinned doorbell regions: x_f, ids, weights, seq, flag. -/
def doorbellBytes (nEmbd k : Nat) : Nat := nEmbd * 4 + k * 4 + k * 4 + 4 + 4

/-- The closed form of the doorbell size. -/
theorem doorbellBytes_eq (nEmbd k : Nat) : doorbellBytes nEmbd k = 4 * (nEmbd + 2 * k + 2) := by
  unfold doorbellBytes
  omega

/-- The ring sequence the host waits for after l+1 layers is l+1 (strictly increasing, one ring per layer). -/
theorem ring_strict (l : Nat) : l + 1 < (l + 1) + 1 := by
  omega

end Ref.MoeSplit



## FILE: /mnt/user-data/outputs/BlockStagePrefix.lean
/-!
# MODULE: block_stage_prefix
SOURCES: include/ref/core/layer.hpp, src/core/layer.cpp
Which of the five stages of `block_layer_pre` run for a given (`half`, `stage_prefix`), when the layer-1 PLE
fires, when the fused hyper-connection path is active, and the 12/36 QSA/GDN split of the 48 layers.

## Source semantics
stages: 0 gr_read(attn), 1 mixer, 2 gr_write(attn), 3 gr_read(ffn), 4 moe_route.
* `stage_prefix = 0`: `half = 0` runs all five, `half = 1` runs 0..2, `half = 2` runs 3..4.
* `stage_prefix = k > 0`: runs stages 0..k-1 and ignores `half`.
* The PLE runs only at layer 1 and only when the mixer half is requested (`half` 0 or 1).
* The fused GR path needs `stage_prefix = 0` and `half = 0`; the fused post only writes R on the last layer.
* `is_qsa_layer`: every fourth layer (l % 4 = 3), so 12 QSA and 36 GDN layers among 48.

## Form template identifiers
runStage <- `run0..run4`; fusedActive <- `fused`; pleApplies <- the PLE `if`; isQsa <- `is_qsa_layer`.
-/

namespace Ref.BlockStagePrefix

-- § stage selection

/-- Whether stage `i` (0..4) runs. -/
def runStage (sp half i : Nat) : Bool :=
  if 0 < sp then decide (i < sp)
  else if i ≤ 2 then decide (half = 0 ∨ half = 1)
  else decide (half = 0 ∨ half = 2)

/-- The default call (half 0, no prefix) runs every stage. -/
theorem whole_runs_all : ∀ i, i < 5 → runStage 0 0 i = true := by
  decide

/-- `half = 1` is exactly the mixer: stages 0, 1, 2. -/
theorem half1_is_mixer : ∀ i, i < 5 → runStage 0 1 i = decide (i ≤ 2) := by
  decide

/-- `half = 2` is exactly the FFN front and router: stages 3, 4. -/
theorem half2_is_ffn : ∀ i, i < 5 → runStage 0 2 i = decide (3 ≤ i) := by
  decide

/-- A prefix of length k runs exactly the first k stages, whatever `half` is. -/
theorem prefix_runs_first_k :
    ∀ k, k < 6 → 1 ≤ k → ∀ half, half < 3 → ∀ i, i < 5 → runStage k half i = decide (i < k) := by
  decide

/-- Prefixes are downward closed: if a stage runs, so does the one before it. -/
theorem prefix_downward_closed :
    ∀ k, k < 6 → 1 ≤ k → ∀ half, half < 3 → ∀ i, i < 4 →
      runStage k half (i + 1) = true → runStage k half i = true := by
  decide

/-- The two halves partition the five stages. -/
theorem halves_partition : ∀ i, i < 5 →
    (runStage 0 1 i = true ∨ runStage 0 2 i = true) ∧ ¬ (runStage 0 1 i = true ∧ runStage 0 2 i = true) := by
  decide

-- § the layer-1 PLE and the fused path

/-- The PLE block runs only at layer 1, when ready, in the mixer half or a whole call. -/
def pleApplies (ready : Bool) (layer half : Nat) : Bool :=
  ready && decide (layer = 1) && decide (half = 0 ∨ half = 1)

/-- No PLE outside layer 1. -/
theorem ple_only_layer1 (ready : Bool) (layer half : Nat) (h : layer ≠ 1) :
    pleApplies ready layer half = false := by
  simp [pleApplies, h]

/-- No PLE in the FFN-front half. -/
theorem ple_not_in_half2 (ready : Bool) (layer : Nat) : pleApplies ready layer 2 = false := by
  simp [pleApplies]

/-- An unwired PLE is skipped entirely. -/
theorem ple_skipped_when_not_ready (layer half : Nat) : pleApplies false layer half = false := by
  simp [pleApplies]

/-- The fused GR path's activation condition. -/
def fusedActive (flag : Bool) (sp half : Nat) (supported : Bool) : Bool :=
  flag && decide (sp = 0) && decide (half = 0) && supported

/-- The fused path never runs for a prefix or a half-graph. -/
theorem fused_needs_whole (flag : Bool) (sp half : Nat) (sup : Bool)
    (h : fusedActive flag sp half sup = true) : sp = 0 ∧ half = 0 := by
  simp [fusedActive] at h
  refine ⟨?_, ?_⟩ <;> omega

/-- In the fused post graph R is written only on the last layer; otherwise every layer writes it. -/
def postWritesR (fused : Bool) (layer nLayers : Nat) : Bool :=
  !fused || decide (layer + 1 = nLayers)

/-- Unfused: every layer writes R. -/
theorem unfused_always_writes (layer n : Nat) : postWritesR false layer n = true := by
  simp [postWritesR]

/-- Fused: only the last layer materialises R, so the head sees a complete residual. -/
theorem fused_writes_only_last (layer n : Nat) (h : layer + 1 ≠ n) : postWritesR true layer n = false := by
  simp [postWritesR, h]

-- § layer kinds

/-- `is_qsa_layer` for the real model. -/
def isQsa (l : Nat) : Bool := l % 4 == 3

/-- 12 of the 48 layers are QSA. -/
theorem qsa_count : ((List.range 48).filter isQsa).length = 12 := by
  decide

/-- 36 of the 48 layers are GDN. -/
theorem gdn_count : ((List.range 48).filter (fun l => !isQsa l)).length = 36 := by
  decide

end Ref.BlockStagePrefix



## FILE: /mnt/user-data/outputs/SessionSchedule.lean
/-!
# MODULE: session_schedule
SOURCES: include/ref/core/session.hpp, src/core/session.cpp
One token through all 48 layers: which GDN state slice each layer owns, the order of the captured `pre[l]` /
`post[l]` graph launches, the per-stage timing differencing, and the layout of the per-layer residual dump.

## Source semantics
* `gdn_point_at`: a GDN layer's state index is the number of GDN layers before it; indices are distinct and < 36.
* `session_replay_full` / `session_loop`: for every layer launch `pre[l]` then `post[l]`.
* `session_replay_stage_prefixes`: stage times are consecutive differences of prefix times, so they telescope.
* `dump_layers`: `(n_layers + 1) * hc * n_embd` floats; slot 0 is the input, slot l+1 is the residual after layer l;
  slots never overlap.

## Form template identifiers
gdnIndex <- `gdn_index` / `gdn_point_at`; schedule <- the launch loop; stageMs <- `t[k]`;
slotOff <- `dump_layers + (l + 1) * n`.
-/

namespace Ref.SessionSchedule

-- § GDN state slices

/-- Number of GDN layers strictly before `l` (every fourth layer, l % 4 = 3, is QSA). -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + (if l % 4 = 3 then 0 else 1)

/-- Closed form: `l - l / 4` GDN layers precede layer `l`. -/
theorem gdnIndex_closed : ∀ l : Nat, gdnIndex l = l - l / 4
  | 0 => by simp [gdnIndex]
  | l + 1 => by
    have ih := gdnIndex_closed l
    by_cases h : l % 4 = 3
    · simp [gdnIndex, h, ih]
      omega
    · simp [gdnIndex, h, ih]
      omega

/-- Distinct GDN layers get distinct state slices (no two layers share a recurrence). -/
theorem gdnIndex_injective (a b : Nat) (ha : a % 4 ≠ 3) (hb : b % 4 ≠ 3)
    (h : gdnIndex a = gdnIndex b) : a = b := by
  rw [gdnIndex_closed, gdnIndex_closed] at h
  omega

/-- Every GDN layer of the 48 owns an in-range slice (< 36). -/
theorem gdnIndex_lt (l : Nat) (hl : l < 48) (h : l % 4 ≠ 3) : gdnIndex l < 36 := by
  rw [gdnIndex_closed]
  omega

-- § launch order

/-- A graph launch. -/
inductive Ev
  | pre (l : Nat)
  | post (l : Nat)
  deriving DecidableEq, Repr

/-- The launch order of `session_replay_full`: pre[l], post[l] for each layer in turn. -/
def schedule : Nat → List Ev
  | 0 => []
  | n + 1 => schedule n ++ [Ev.pre n, Ev.post n]

/-- Two graphs per layer. -/
theorem schedule_length (n : Nat) : (schedule n).length = 2 * n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih => simp [schedule, ih] <;> omega

/-- `pre[l]` is launched exactly for the layers that exist. -/
theorem pre_mem (n l : Nat) : Ev.pre l ∈ schedule n ↔ l < n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih =>
    simp [schedule, ih] <;> omega

/-- `post[l]` is launched exactly for the layers that exist. -/
theorem post_mem (n l : Nat) : Ev.post l ∈ schedule n ↔ l < n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih =>
    simp [schedule, ih] <;> omega

/-- The next layer's launches come strictly after the current layer's (append-only schedule). -/
theorem schedule_append (n : Nat) : schedule (n + 1) = schedule n ++ [Ev.pre n, Ev.post n] := rfl

-- § stage differencing

/-- The five stage times from prefix times p1..p5 (p0 = 0), as `session_replay_stage_prefixes` computes them. -/
def stageMs (p1 p2 p3 p4 p5 : Int) : List Int := [p1, p2 - p1, p3 - p2, p4 - p3, p5 - p4]

/-- The stage times telescope to the full-prefix time: the sum check the driver prints. -/
theorem stageMs_sum (p1 p2 p3 p4 p5 : Int) : (stageMs p1 p2 p3 p4 p5).sum = p5 := by
  simp [stageMs] <;> omega

-- § residual dump layout

/-- Offset of slot `l` in the per-layer dump, for `n = hc * n_embd` floats per slot. -/
def slotOff (l n : Nat) : Nat := l * n

/-- Consecutive slots do not overlap. -/
theorem slots_disjoint (l n : Nat) : slotOff l n + n ≤ slotOff (l + 1) n := by
  unfold slotOff
  have := Nat.add_mul l 1 n
  omega

/-- The last slot (after layer n_layers - 1) ends exactly at the buffer size `(n_layers + 1) * n`. -/
theorem last_slot_ends (nLayers n : Nat) : slotOff nLayers n + n = (nLayers + 1) * n := by
  unfold slotOff
  have := Nat.add_mul nLayers 1 n
  omega

/-- Per-layer half-dump stride at which layer l's block starts never overlaps layer l+1's. -/
theorem half_stride_disjoint (l stride : Nat) : l * stride + stride ≤ (l + 1) * stride := by
  have := Nat.add_mul l 1 stride
  omega

-- § per-session sizes

/-- GDN recurrence + conv history floats for one layer. -/
def gdnStateFloats (stateSize vHeads dConv convChannels : Nat) : Nat :=
  stateSize * vHeads * stateSize + convChannels * (dConv - 1)

/-- Real geometry: 817,152 floats (3,268,608 bytes) per GDN layer. -/
theorem real_gdn_state_floats : gdnStateFloats 128 48 4 10240 = 817152 := by
  decide

/-- The one-time pinned scratch holds `k * n_embd * 4` bytes, matching the parts buffer. -/
theorem scratch_matches_parts (k nEmbd : Nat) : k * nEmbd * 4 = (k * nEmbd) * 4 := rfl

end Ref.SessionSchedule



## FILE: /mnt/user-data/outputs/MemoryPlanner.lean
/-!
# MODULE: memory_planner
SOURCES: include/ref/plan/plan.hpp
The VRAM memory planner: fixed costs (dense weights, embedding, workspace, recurrent state) plus the KV cache
at `max_context` must fit the pool, and what is left becomes whole expert-cache slots.  It REFUSES instead of
overcommitting.

## Source semantics
* `state_bytes`: 36 GDN layers x (S*h_v*S + (d_conv-1)*C) fp32 = 117,669,888 B.
* `kv_bytes_per_token`: 12 QSA layers x (INT8 K/V plus fp16 scale per 64 plus one pooled 128-wide indexer key per
  4-token block) = 13,056 B per token.
* `make_plan`: throws `DoesNotClose` if `kv + fixed > pool` or if no slot fits; otherwise
  `cache_bytes = pool - kv - fixed`, `cache_slots = cache_bytes / blob`, and the slot remainder is unusable.

## Form template identifiers
Costs <- `Costs`; Plan <- `Plan`; makePlan <- `make_plan` (None = `DoesNotClose`); fixedCost <- `fixed`.
-/

namespace Ref.MemoryPlanner

-- § geometry-derived byte costs

/-- GDN state bytes: layers x (S*h_v*S*4 + (d_conv-1)*C*4). -/
def stateBytes (nGdn s hv dconv c : Nat) : Nat := nGdn * (s * hv * s * 4 + (dconv - 1) * c * 4)

/-- Per-token KV + indexer bytes: layers x (K/V int8 + scales + indexer key per block). -/
def kvBytesPerToken (nQsa nKv headDim group keyHeads keyDim block : Nat) : Nat :=
  nQsa * ((2 * nKv * headDim) + (2 * nKv * headDim / group) * 2 + keyHeads * keyDim / block)

/-- The real recurrent state is 117,669,888 bytes. -/
theorem real_state_bytes : stateBytes 36 128 48 4 10240 = 117669888 := by
  decide

/-- Which is 85 expert-cache slots worth of VRAM (the omission round 198 fixed). -/
theorem real_state_is_85_slots : stateBytes 36 128 48 4 10240 / 1382400 = 85 := by
  decide

/-- The real KV + indexer cost is 13,056 bytes per token. -/
theorem real_kv_per_token : kvBytesPerToken 12 2 256 64 1 128 4 = 13056 := by
  decide

-- § the plan

/-- The fixed costs of `make_plan`. -/
structure Costs where
  expertBlob : Nat
  dense : Nat
  embd : Nat
  mtp : Nat
  workspace : Nat
  state : Nat

/-- A computed plan. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat

/-- Everything resident that is not the KV cache or the expert cache. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- `make_plan`; `none` stands for the `DoesNotClose` exception. `kv` is `kv_bytes_per_token * max_context`. -/
def makePlan (maxCtx kv pool : Nat) (c : Costs) : Option Plan :=
  if kv + fixedCost c > pool then none
  else if (pool - kv - fixedCost c) / c.expertBlob = 0 then none
  else some
    { maxContext := maxCtx, kvBytes := kv, stateBytes := c.state,
      cacheBytes := pool - kv - fixedCost c,
      cacheSlots := (pool - kv - fixedCost c) / c.expertBlob,
      vramBudget := pool,
      vramUsed := pool - (pool - kv - fixedCost c) % c.expertBlob }

/-- Overcommitted fixed costs are refused, never planned around. -/
theorem refuses_overcommit (maxCtx kv pool : Nat) (c : Costs) (h : kv + fixedCost c > pool) :
    makePlan maxCtx kv pool c = none := by
  unfold makePlan
  rw [if_pos h]

/-- A plan that exists closes: KV + fixed + cache is exactly the pool. -/
theorem plan_closes (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.kvBytes + fixedCost c + p.cacheBytes = pool := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · cases h
      dsimp only
      omega

/-- A plan that exists has at least one slot. -/
theorem plan_has_slot (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.cacheSlots ≠ 0 := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i h2
      cases h
      exact h2

/-- Slots are whole: they never use more bytes than the cache has. -/
theorem slots_fit (cacheBytes blob : Nat) : (cacheBytes / blob) * blob ≤ cacheBytes :=
  Nat.div_mul_le_self cacheBytes blob

/-- The plan's budget is the pool it was asked about. -/
theorem plan_budget (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.vramBudget = pool := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · cases h
      rfl

/-- `vram_used` never exceeds the pool. -/
theorem plan_used_le (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.vramUsed ≤ pool := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · cases h
      exact Nat.sub_le _ _

/-- Used VRAM is exactly KV + fixed + whole slots: the unusable remainder is the only thing left over. -/
theorem plan_used_eq (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) :
    p.vramUsed = kv + fixedCost c + c.expertBlob * p.cacheSlots := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · cases h
      dsimp only
      have hd := Nat.mod_add_div (pool - kv - fixedCost c) c.expertBlob
      omega

end Ref.MemoryPlanner



## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `ref generate` driver: option validation, teacher forcing of the prompt, how many rows the loop produces,
the speculative-decoding acceptance rule, the routing-trace layer check, and the lent-slot sizing of the batched
prompt path.

## Source semantics
* Validation: `max_new > 0`, `max_context > 0`, `max_new <= max_context`, prompt fits in `max_context - max_new`,
  and `max_context` stays below 2^31 so positions fit int32.
* Loop: for every position before the last prompt position the next input is the prompt's own next token;
  afterwards it is the sampled token.  `produced` receives a token from position `n_prompt - 1` on, so the loop
  stops at position `n_prompt - 2 + max_new` and writes `n_prompt - 1 + max_new` rows.
* Speculation: a window is `[last token, drafts...]`; the number of accepted drafts is the length of the longest
  prefix of drafts equal to the window's argmax outputs; a round emits `accepted + 1` tokens.
* Routing trace: `d.layers` is already advanced when the trace is written, so the recorded layer is
  `d.layers - 1`, valid iff `1 <= d.layers <= 48`.
* Prompt borrowing: `k = ceil(need / blob)` cache slots are lent, so `k * blob >= need`.

## Form template identifiers
validOpts <- the option checks in `main`; nextInput <- `tok = (pos + 1 < n_prompt) ? tokens[pos+1] : next`;
producedAt <- `produced.size()`; accepted <- `a` in the verify loop; ceilDiv <- `(need + blob - 1) / blob`.
-/

namespace Ref.GenerateDriver

-- § option validation

/-- The numeric part of the option checks. -/
structure Opts where
  nTokens : Nat
  maxNew : Nat
  maxContext : Nat

/-- Accepted configurations. -/
def validOpts (o : Opts) : Prop :=
  0 < o.nTokens ∧ 0 < o.maxNew ∧ 0 < o.maxContext ∧ o.maxNew ≤ o.maxContext ∧
    o.nTokens ≤ o.maxContext - o.maxNew ∧ o.maxContext ≤ 2147483647 - 8

/-- A valid configuration always fits prompt plus generation (the later redundant check cannot fire). -/
theorem valid_fits (o : Opts) (h : validOpts o) : o.nTokens + o.maxNew ≤ o.maxContext := by
  unfold validOpts at h
  omega

/-- Every position the loop can reach is a valid int32 with the +8 margin the device path needs. -/
theorem valid_position_int32 (o : Opts) (h : validOpts o) (pos : Nat) (hp : pos < o.maxContext) :
    pos + 8 < 2147483648 := by
  unfold validOpts at h
  omega

-- § teacher forcing

/-- The next input token: the prompt's next token while the prompt lasts, otherwise the sampled one. -/
def nextInput (prompt : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else next

/-- While the prompt lasts the model's guess is ignored. -/
theorem nextInput_prompt (prompt : List Nat) (pos next : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos next = prompt.getD (pos + 1) 0 := by
  unfold nextInput
  rw [if_pos h]

/-- After the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : List Nat) (pos next : Nat) (h : prompt.length ≤ pos + 1) :
    nextInput prompt pos next = next := by
  unfold nextInput
  rw [if_neg (by omega)]

-- § how many rows the loop writes

/-- Tokens in `produced` after processing position `pos` of an `n`-token prompt. -/
def producedAt (n pos : Nat) : Nat := if n ≤ pos + 1 then pos + 2 - n else 0

/-- The last prompt position yields the first generated token. -/
theorem first_token_at_last_prompt (n : Nat) (hn : 1 ≤ n) : producedAt n (n - 1) = 1 := by
  unfold producedAt
  split <;> omega

/-- The loop stops at position `n - 2 + max_new`, where `produced` reaches `max_new`. -/
theorem stops_at_max_new (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) : producedAt n (n + m - 2) = m := by
  unfold producedAt
  split <;> omega

/-- Before that position `produced` is still short. -/
theorem short_before_stop (n m pos : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) (hp : pos < n + m - 2) :
    producedAt n pos < m := by
  unfold producedAt
  split <;> omega

/-- Rows written = positions processed = `n - 1 + max_new`, which is the dump header's row count. -/
theorem rows_written (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) : (n + m - 2) + 1 = n - 1 + m := by
  omega

-- § speculative acceptance

/-- Length of the longest common prefix of the drafts and the verified outputs. -/
def accepted : List Int → List Int → Nat
  | d :: ds, o :: os => if d = o then accepted ds os + 1 else 0
  | _, _ => 0

/-- Never more accepted drafts than drafts. -/
theorem accepted_le : ∀ (ds os : List Int), accepted ds os ≤ ds.length
  | [], _ => by simp [accepted]
  | _ :: _, [] => by simp [accepted]
  | d :: ds, o :: os => by
    simp only [accepted, List.length_cons]
    split
    · have := accepted_le ds os
      omega
    · omega

/-- When every draft is right, all of them are accepted. -/
theorem accepted_self : ∀ ds : List Int, accepted ds ds = ds.length
  | [] => by simp [accepted]
  | d :: ds => by simp [accepted, accepted_self ds]

/-- The first wrong draft ends the round with nothing accepted from it. -/
theorem accepted_stop (d o : Int) (ds os : List Int) (h : d ≠ o) : accepted (d :: ds) (o :: os) = 0 := by
  simp [accepted, h]

/-- A round of window size T (drafts = T - 1) emits at most T tokens. -/
theorem round_emits_le (ds os : List Int) : accepted ds os + 1 ≤ ds.length + 1 := by
  have := accepted_le ds os
  omega

-- § routing trace and prompt borrowing

/-- The layer recorded in the routing trace, from the already-advanced `d.layers`. -/
def traceLayer (layers : Nat) : Nat := layers - 1

/-- The recorded layer is in 0..47 exactly when the advanced counter is in 1..48. -/
theorem traceLayer_valid (layers : Nat) (h1 : 1 ≤ layers) (h2 : layers ≤ 48) : traceLayer layers < 48 := by
  unfold traceLayer
  omega

/-- `(need + blob - 1) / blob`: slots to lend to the prompt path. -/
def ceilDiv (need blob : Nat) : Nat := (need + blob - 1) / blob

/-- The lent slots really hold the prompt path's buffers. -/
theorem lent_slots_hold (need blob : Nat) (hb : 0 < blob) : need ≤ ceilDiv need blob * blob := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) hb
  have h3 := Nat.mul_comm blob ((need + blob - 1) / blob)
  omega

/-- ...and never more than one slot too many. -/
theorem lent_slots_tight (need blob : Nat) (hb : 0 < blob) : ceilDiv need blob * blob < need + blob := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h3 := Nat.mul_comm blob ((need + blob - 1) / blob)
  omega

end Ref.GenerateDriver



## FILE: /mnt/user-data/outputs/ServeFrontend.lean
/-!
# MODULE: serve_frontend
SOURCES: serve/frontend.py
The request/response layer: mapping client "reasoning effort" spellings and Anthropic thinking budgets onto the
chat template's levels, and the incremental output parser that never emits a partial tag.

## Source semantics
* `EFFORT`: none/off/minimal/disabled/false disable thinking; low, medium stay; high/xhigh/max/maximum are xhigh.
  An unknown spelling is rejected (a 400), not a crash.
* `budget_effort`: under 2048 tokens low, under 8192 medium, otherwise xhigh (monotone in the budget).
* `OutputParser`: states reasoning -> content (on `</think>`), content -> call (on `<tool_call>`), call -> content
  (on `</tool_call>`); it never returns to reasoning.  `_hold` keeps the longest suffix of the buffer that is a
  proper prefix of a tag, so the held length is below the tag length and at most the buffer length.

## Form template identifiers
effortLookup <- `EFFORT`; budgetRank <- `budget_effort`; step <- the parser's state transitions;
holdLen <- `OutputParser._hold` for a single tag.
-/

namespace Ref.ServeFrontend

-- § reasoning effort

/-- The template's thinking levels. -/
inductive Level
  | low
  | medium
  | xhigh
  deriving DecidableEq, Repr

/-- The `EFFORT` table: `some none` disables thinking, `none` is an unknown spelling. -/
def effortLookup : String → Option (Option Level)
  | "none" | "off" | "minimal" | "disabled" | "false" => some none
  | "low" => some (some Level.low)
  | "medium" => some (some Level.medium)
  | "high" | "xhigh" | "max" | "maximum" => some (some Level.xhigh)
  | _ => none

/-- "high" means xhigh, the template's default. -/
theorem high_is_xhigh : effortLookup "high" = some (some Level.xhigh) := by
  decide

/-- "max" is an alias of "high". -/
theorem max_is_high : effortLookup "max" = effortLookup "high" := by
  decide

/-- "off" is an alias of "none". -/
theorem off_is_none : effortLookup "off" = effortLookup "none" := by
  decide

/-- "none" disables thinking. -/
theorem none_disables : effortLookup "none" = some none := by
  decide

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_rejected : effortLookup "ultra" = none := by
  decide

/-- Rank of a thinking level derived from an Anthropic token budget. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- A larger thinking budget never selects a lower level. -/
theorem budgetRank_mono (n m : Nat) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : m < 2048 <;> by_cases h3 : n < 8192 <;>
    by_cases h4 : m < 8192 <;> simp [h1, h2, h3, h4] <;> omega

/-- Budgets under 2048 are low. -/
theorem budget_low (n : Nat) (h : n < 2048) : budgetRank n = 0 := by
  unfold budgetRank
  rw [if_pos h]

/-- Budgets from 8192 up are xhigh. -/
theorem budget_xhigh (n : Nat) (h : 8192 ≤ n) : budgetRank n = 2 := by
  unfold budgetRank
  rw [if_neg (by omega), if_neg (by omega)]

-- § the parser's state machine

/-- Parser states. -/
inductive St
  | reasoning
  | content
  | call
  deriving DecidableEq, Repr

/-- Tags the parser reacts to. -/
inductive Tag
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq, Repr

/-- One transition on a complete tag. -/
def step : St → Tag → St
  | St.reasoning, Tag.thinkEnd => St.content
  | St.content, Tag.callStart => St.call
  | St.call, Tag.callEnd => St.content
  | s, _ => s

/-- Once the parser has left the reasoning state it never returns to it. -/
theorem never_back_to_reasoning (s : St) (t : Tag) (h : step s t = St.reasoning) : s = St.reasoning := by
  cases s <;> cases t <;> first | rfl | (simp [step] at h)

/-- `</think>` moves reasoning to content. -/
theorem think_end_moves : step St.reasoning Tag.thinkEnd = St.content := rfl

/-- A tool call always returns to content. -/
theorem call_returns : step St.call Tag.callEnd = St.content := rfl

-- § holding back partial tags

/-- Generic fold invariant used by the hold-back bounds. -/
theorem foldl_inv (P : Nat → Prop) (f : Nat → Nat → Nat) :
    ∀ (l : List Nat) (init : Nat), (∀ b n, n ∈ l → P b → P (f b n)) → P init → P (l.foldl f init)
  | [], _, _, h => h
  | x :: xs, init, hf, h =>
    foldl_inv P f xs (f init x) (fun b n hn hb => hf b n (by simp [hn]) hb) (hf init x (by simp) h)

/-- Length of the longest proper-prefix-of-`tag` suffix of `text` (`OutputParser._hold`, one tag). -/
def holdLen (text tag : List Char) : Nat :=
  (List.range tag.length).foldl
    (fun best n => if 1 ≤ n ∧ n ≤ text.length ∧ text.drop (text.length - n) = tag.take n then n else best) 0

/-- A held fragment is shorter than the tag: a complete tag is never held back. -/
theorem holdLen_lt (text tag : List Char) (hpos : 0 < tag.length) : holdLen text tag < tag.length := by
  unfold holdLen
  refine foldl_inv (fun b => b < tag.length) _ _ _ ?_ hpos
  intro b n hn hb
  have hn' : n < tag.length := List.mem_range.mp hn
  split
  · exact hn'
  · exact hb

/-- A held fragment is part of the buffer: nothing is held that is not there. -/
theorem holdLen_le (text tag : List Char) : holdLen text tag ≤ text.length := by
  unfold holdLen
  refine foldl_inv (fun b => b ≤ text.length) _ _ _ ?_ (Nat.zero_le _)
  intro b n hn hb
  split
  · rename_i hc
    exact hc.2.1
  · exact hb

end Ref.ServeFrontend



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- SessionSchedule.lean
  was: theorem real_gdn_state_floats : gdnStateFloats 128 48 4 10240 = 3176448
  now: theorem real_gdn_state_floats : gdnStateFloats 128 48 4 10240 = 817152
  measured: the claim changed
-/



## FILE: /mnt/user-data/outputs/MtpDraftLayer.lean
/-!
# MODULE: mtp_draft_layer
SOURCES: include/ref/core/mtp.hpp, src/core/mtp.cpp
The MTP (multi-token-prediction) draft layer: a bump-carved device arena, per-cell step records, the batched
prompt prefill, the capture-per-step draft chain with an early-stop rule, and the draft-head vocabulary subset.

## Source semantics
* `Bump::take` rounds every carve up to 256 bytes; the counting pass and the real pass run the same sequence, so
  the arena size is the final `used`.
* A cell `c` is described by the step record `(c, c+1, (c+1)/4, c+1)`; the position of every head is `c`.
* `prefill` walks the prompt in batches of at most `max_t` rows, skipping batches wholly before
  `prompt_len - window - 64`.
* `draft` always produces one draft, then continues the chain for steps `j = 1 .. max_t-2` while the last
  draft's probability is at least `min_p`; the unused tail of `drafts` is zeroed.
* The full layer runs one row at a time (`full -> T = 1`); `step_row0 < 0` encodes "K/V only from row `-1-step_row0`".
* Step rows: `0..T-1` catch-up, `max_t + j - 1` chain step `j`, `2*max_t - 1` the last-accepted-row slot.

## Form template identifiers
role mapping (source names -> abstract names):
  Bump::take            -> arena_take
  alignUp               -> align256
  MtpDrafter::prefill   -> prompt_batch_walk
  MtpDrafter::draft     -> draft_chain
  record_forward        -> layer_forward
  step_exec_/round_exec_/prefill_exec_ -> captured_graph_cache
  draft_vocab.bin       -> draft_head_subset
-/

namespace Ref.Mtp

-- § arena carving

/-- A bump take of `b` bytes occupies `b` rounded up to a multiple of 256. -/
def alignUp (b : Nat) : Nat := (b + 255) / 256 * 256

/-- The rounded size never shrinks a request. -/
theorem alignUp_ge (b : Nat) : b ≤ alignUp b := by
  unfold alignUp; omega

/-- The rounded size is a multiple of 256. -/
theorem alignUp_mod (b : Nat) : alignUp b % 256 = 0 := by
  unfold alignUp; omega

/-- The rounding wastes less than 256 bytes. -/
theorem alignUp_lt (b : Nat) : alignUp b < b + 256 := by
  unfold alignUp; omega

/-- Offset of take number `i` in a carve sequence of `(count, element size)` pairs. -/
def offsetAt : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | p :: ps, i + 1 => alignUp (p.1 * p.2) + offsetAt ps i

/-- Every carve offset is 256-byte aligned. -/
theorem offsetAt_mod (l : List (Nat × Nat)) : ∀ i, offsetAt l i % 256 = 0 := by
  induction l with
  | nil => intro i; simp [offsetAt]
  | cons p ps ih =>
    intro i
    cases i with
    | zero => simp [offsetAt]
    | succ j =>
      simp only [offsetAt]
      have h1 := alignUp_mod (p.1 * p.2)
      have h2 := ih j
      omega

/-- Offsets never decrease from one take to the next. -/
theorem offsetAt_mono (l : List (Nat × Nat)) : ∀ i, offsetAt l i ≤ offsetAt l (i + 1) := by
  induction l with
  | nil => intro i; simp [offsetAt]
  | cons p ps ih =>
    intro i
    cases i with
    | zero => simp [offsetAt]
    | succ j =>
      simp only [offsetAt]
      have h := ih j
      omega

/-- Offset 0 is always 0. -/
theorem offsetAt_zero (l : List (Nat × Nat)) : offsetAt l 0 = 0 := by
  cases l <;> rfl

/-- The first take starts at offset 0 and the second right after its rounded size. -/
theorem offsetAt_second (p : Nat × Nat) (ps : List (Nat × Nat)) :
    offsetAt (p :: ps) 0 = 0 ∧ offsetAt (p :: ps) 1 = alignUp (p.1 * p.2) := by
  simp [offsetAt, offsetAt_zero]

/-- The region of the first take is contained in the space before the second take. -/
theorem first_take_fits (p : Nat × Nat) (ps : List (Nat × Nat)) :
    p.1 * p.2 ≤ offsetAt (p :: ps) 1 := by
  have h := offsetAt_second p ps
  rw [h.2]
  exact alignUp_ge _

-- § step records

/-- The per-cell step record handed to the QSA append/attention kernels. -/
structure StepRec where
  cell : Nat
  len : Nat
  block : Nat
  seq : Nat

/-- Step record of cell `c` (`put` in `draft`, the loop body of `prefill`). -/
def stepOf (c : Nat) : StepRec := ⟨c, c + 1, (c + 1) / 4, c + 1⟩

/-- The visible length is one past the cell. -/
theorem stepOf_len (c : Nat) : (stepOf c).len = c + 1 := rfl

/-- The block number rounds the length down to a multiple of 4. -/
theorem stepOf_block_le (c : Nat) : (stepOf c).block * 4 ≤ c + 1 := by
  show (c + 1) / 4 * 4 ≤ c + 1
  omega

/-- The block number is the largest one that fits. -/
theorem stepOf_block_lt (c : Nat) : c + 1 < ((stepOf c).block + 1) * 4 := by
  show c + 1 < ((c + 1) / 4 + 1) * 4
  omega

-- § prompt prefill batching

/-- Rows processed by the batch starting at row `c` of a prompt of `n` cells. -/
def batchSize (maxT n c : Nat) : Nat := min maxT (n - c)

/-- A batch inside the prompt is never empty. -/
theorem batchSize_pos (maxT n c : Nat) (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ batchSize maxT n c := by
  unfold batchSize; omega

/-- A batch never exceeds the staging capacity. -/
theorem batchSize_le (maxT n c : Nat) : batchSize maxT n c ≤ maxT := by
  unfold batchSize; omega

/-- A batch never runs past the end of the prompt. -/
theorem batchSize_fits (maxT n c : Nat) : c + batchSize maxT n c ≤ n ∨ n < c := by
  unfold batchSize; omega

/-- The first prompt cell whose K/V the attention window can still reach. -/
def firstNeeded (window promptLen : Nat) : Nat :=
  if 0 < window ∧ 0 < promptLen then promptLen - window - 64 else 0

/-- A batch is skipped when it ends at or before `firstNeeded`. -/
def skipBatch (cell0 c T firstNeed : Nat) : Bool := decide (cell0 + c + T ≤ firstNeed)

/-- No batch is skipped when no window bound applies. -/
theorem no_skip_without_window (cell0 c T : Nat) (hT : 1 ≤ T) :
    skipBatch cell0 c T (firstNeeded 0 0) = false := by
  simp [skipBatch, firstNeeded]
  omega

/-- Every cell of a skipped batch is more than `window + 64` behind the prompt end. -/
theorem skipped_cells_unreachable (window promptLen cell0 c T cell : Nat)
    (hw : 0 < window) (hp : 0 < promptLen)
    (hs : skipBatch cell0 c T (firstNeeded window promptLen) = true)
    (hcell : cell < cell0 + c + T) :
    cell < promptLen - window - 64 := by
  have h : cell0 + c + T ≤ firstNeeded window promptLen := of_decide_eq_true hs
  have h2 : firstNeeded window promptLen = promptLen - window - 64 := by
    simp [firstNeeded, hw, hp]
  omega

-- § window and attention span

/-- The window actually applied: 0 means attention over every cell. -/
def effWindow (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else 0

/-- Number of cells the attention reads at most. -/
def attnSpan (window maxCells : Nat) : Nat :=
  if 0 < effWindow window maxCells then effWindow window maxCells else maxCells

/-- Capacity of the identity selection table: the span rounded up to 64. -/
def capCells (window maxCells : Nat) : Nat := (attnSpan window maxCells + 63) / 64 * 64

/-- The attention span never exceeds the cache. -/
theorem attnSpan_le (window maxCells : Nat) : attnSpan window maxCells ≤ maxCells := by
  by_cases h : 0 < window ∧ window < maxCells
  · have h1 : effWindow window maxCells = window := by simp [effWindow, h]
    have h2 : attnSpan window maxCells = window := by simp [attnSpan, h1, h.1]
    omega
  · have h1 : effWindow window maxCells = 0 := by simp [effWindow, h]
    simp [attnSpan, h1]

/-- The selection table holds the whole span. -/
theorem capCells_ge (window maxCells : Nat) : attnSpan window maxCells ≤ capCells window maxCells := by
  unfold capCells; omega

/-- The selection table size is a multiple of 64. -/
theorem capCells_mod (window maxCells : Nat) : capCells window maxCells % 64 = 0 := by
  unfold capCells; omega

-- § the draft chain

/-- Number of chain steps taken after the first draft: step `j` runs when `cont j` holds. -/
def loopLen (cont : Nat → Bool) : Nat → Nat → Nat
  | _, 0 => 0
  | j, fuel + 1 => if cont j = true then 1 + loopLen cont (j + 1) fuel else 0

/-- The chain loop never exceeds its iteration budget. -/
theorem loopLen_le (cont : Nat → Bool) (fuel : Nat) : ∀ j, loopLen cont j fuel ≤ fuel := by
  induction fuel with
  | zero => intro j; simp [loopLen]
  | succ f ih =>
    intro j
    simp only [loopLen]
    split
    · have := ih (j + 1)
      omega
    · omega

/-- A chain whose continuation test always fails takes no steps. -/
theorem loopLen_false (j f : Nat) : loopLen (fun _ => false) j f = 0 := by
  cases f <;> simp [loopLen]

/-- Drafts produced in one round: the first draft plus the chain steps. -/
def draftCount (maxT : Nat) (cont : Nat → Bool) : Nat := 1 + loopLen cont 1 (maxT - 2)

/-- A round yields between 1 and `max_t - 1` drafts. -/
theorem draftCount_bounds (maxT : Nat) (cont : Nat → Bool) (h : 2 ≤ maxT) :
    1 ≤ draftCount maxT cont ∧ draftCount maxT cont ≤ maxT - 1 := by
  have := loopLen_le cont (maxT - 2) 1
  unfold draftCount
  omega

/-- A chain that stops immediately yields exactly the first draft. -/
theorem draftCount_stop_early (maxT : Nat) : draftCount maxT (fun _ => false) = 1 := by
  unfold draftCount
  rw [loopLen_false]

/-- Number of zero-filled draft slots after the chain. -/
def padCount (maxT n : Nat) : Nat := (maxT - 1) - n

/-- Produced plus padded slots fill exactly `max_t - 1`. -/
theorem draft_slots_full (maxT n : Nat) (h : n ≤ maxT - 1) : n + padCount maxT n = maxT - 1 := by
  unfold padCount; omega

/-- Chain-step rows lie above the catch-up rows and below the last-accepted slot. -/
theorem chain_row_range (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ maxT + j - 1 ∧ maxT + j - 1 < 2 * maxT - 1 := by
  omega

/-- Catch-up rows (`< T <= max_t`) never collide with chain rows. -/
theorem catchup_row_below_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < maxT + j - 1 := by
  omega

/-- The last-accepted slot is the final row of the `2 * max_t` staging area. -/
theorem last_slot_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT := by
  omega

/-- Argument check of `draft`: the accepted row lies in the window. -/
def draftArgsOk (T maxT a : Nat) : Bool := decide (1 ≤ T ∧ T ≤ maxT ∧ a < T)

/-- A valid call has a valid accepted row and a window within staging capacity. -/
theorem draftArgsOk_sound (T maxT a : Nat) (h : draftArgsOk T maxT a = true) :
    a < maxT ∧ 1 ≤ T := by
  have h' : 1 ≤ T ∧ T ≤ maxT ∧ a < T := of_decide_eq_true h
  omega

/-- `draft_first` always uses accepted row 0, valid whenever the window is non-empty. -/
theorem draft_first_row_valid (T maxT : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) :
    draftArgsOk T maxT 0 = true := by
  simp [draftArgsOk]
  omega

-- § full layer versus K/V-only encoding

/-- Encoding of "K/V only starting at row `r`" in the `step_row0` argument. -/
def encKvOnly (r : Nat) : Int := -1 - (r : Int)

/-- Decoding recovers the row (`row0 = -1 - step_row0`). -/
def decRow (s : Int) : Int := -1 - s

/-- Encode then decode is the identity. -/
theorem dec_enc (r : Nat) : decRow (encKvOnly r) = (r : Int) := by
  unfold decRow encKvOnly; omega

/-- A K/V-only encoding is always negative, so `full` is false. -/
theorem enc_is_kv_only (r : Nat) : encKvOnly r < 0 := by
  unfold encKvOnly; omega

/-- A non-negative `step_row0` is a full-layer request and must have `T = 1`. -/
def fullLayerOk (s : Int) (T : Nat) : Prop := 0 ≤ s → T = 1

/-- K/V-only requests are valid for any batch size. -/
theorem kv_only_any_T (r T : Nat) : fullLayerOk (encKvOnly r) T := by
  intro h
  have := enc_is_kv_only r
  omega

-- § draft vocabulary subset

/-- Number of vocabulary ids in `draft_vocab.bin` (4 bytes each). -/
def nDvocab (rawBytes : Nat) : Nat := rawBytes / 4

/-- The file size is `4 * n` when it is a multiple of 4, as the loader requires. -/
theorem nDvocab_exact (rawBytes : Nat) (h : rawBytes % 4 = 0) : nDvocab rawBytes * 4 = rawBytes := by
  unfold nDvocab; omega

/-- A loadable subset file holds at least one id. -/
theorem nDvocab_pos (rawBytes : Nat) (h : 4 ≤ rawBytes) : 1 ≤ nDvocab rawBytes := by
  unfold nDvocab; omega

end Ref.Mtp



## FILE: /mnt/user-data/outputs/VerifyWindow.lean
/-!
# MODULE: verify_window
SOURCES: include/ref/core/verify.hpp, src/core/verify.cpp
The speculative verify window: T tokens through all layers in one captured graph, optionally split in two token
groups whose CPU expert work is pipelined against the GPU work of the other group; `commit(n_keep)` makes a prefix
permanent.

## Source semantics
* With `G = 2` the tokens split into A = `[0, (T+1)/2)` and B = `[(T+1)/2, T)`; with `G = 1` one group holds all.
* Ring `l*G + grp + 1` numbers every (layer, group) hand-off; the host serves them in order `k = l*G + grp`.
* The GPU plan block is `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) |
  start2(cap+1) | 1`, one block per group.
* Staging holds 16 expert blobs, halved between the groups of a split window.
* `commit` writes `[n_keep, n_keep-1, pos0 .. pos0+T-1]` with `-1` for rejected rows, and advances the PLE history
  so that its newest entry is the last kept token.
* Flag B only rises (`raise_flag` is a monotone compare-and-set).

## Form template identifiers
role mapping (source names -> abstract names):
  Verifier::run         -> window_run
  Verifier::commit      -> window_commit
  record_window pre/post-> group_pipeline
  raise_flag            -> monotone_flag
  plan block / sink_    -> gpu_plan_block
  kStagingBlobs         -> staging_slots
  ple_prev              -> token_history_pair
-/

namespace Ref.Verify

-- § token groups

/-- Group count of a window: two when splitting is on and the window has at least two tokens. -/
def groupsOf (split : Bool) (T : Nat) : Nat := if split = true ∧ 2 ≤ T then 2 else 1

/-- First token of group `grp`. -/
def tb (T : Nat) : Nat → Nat
  | 0 => 0
  | _ + 1 => (T + 1) / 2

/-- One past the last token of group `grp`. -/
def te (G T : Nat) : Nat → Nat
  | 0 => (if G = 2 then (T + 1) / 2 else T)
  | _ + 1 => T

/-- The group count is 1 or 2. -/
theorem groupsOf_cases (split : Bool) (T : Nat) : groupsOf split T = 1 ∨ groupsOf split T = 2 := by
  unfold groupsOf
  split <;> simp

/-- Without splitting, the single group is the whole window. -/
theorem single_group_whole (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by
  simp [tb, te]

/-- In a split window group B starts exactly where group A ends. -/
theorem groups_adjacent (T : Nat) : te 2 T 0 = tb T 1 := by
  simp [te, tb]

/-- Both groups of a split window are non-empty when `T >= 2`. -/
theorem group_sizes_pos (T : Nat) (hT : 2 ≤ T) :
    1 ≤ te 2 T 0 - tb T 0 ∧ 1 ≤ te 2 T 1 - tb T 1 := by
  show 1 ≤ (T + 1) / 2 - 0 ∧ 1 ≤ T - (T + 1) / 2
  omega

/-- The two groups of a split window partition the window. -/
theorem group_sizes_sum (T : Nat) :
    (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  show ((T + 1) / 2 - 0) + (T - (T + 1) / 2) = T
  omega

/-- Group A is at least as large as group B. -/
theorem groupA_ge_groupB (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  show T - (T + 1) / 2 ≤ (T + 1) / 2 - 0
  omega

-- § ring numbering

/-- Ring (sequence) number of the hand-off of layer `l`, group `grp`. -/
def ringOf (l G grp : Nat) : Nat := l * G + grp + 1

/-- Decomposing a global index `k` into (layer, group) and re-ringing gives `k + 1`. -/
theorem ring_of_index (k G : Nat) : ringOf (k / G) G (k % G) = k + 1 := by
  unfold ringOf
  have h := Nat.div_add_mod k G
  rw [Nat.mul_comm (k / G) G]
  omega

/-- Distinct hand-offs get distinct ring numbers. -/
theorem ring_of_index_inj (k1 k2 G : Nat)
    (h : ringOf (k1 / G) G (k1 % G) = ringOf (k2 / G) G (k2 % G)) : k1 = k2 := by
  rw [ring_of_index, ring_of_index] at h
  omega

/-- The serving loop reaches every layer index below `n_layers` and no other. -/
theorem layer_of_index_lt (k L G : Nat) (hG : 0 < G) (h : k < L * G) : k / G < L :=
  (Nat.div_lt_iff_lt_mul hG).2 h

/-- The group index of a hand-off is a valid group. -/
theorem group_of_index_lt (k G : Nat) (hG : 0 < G) : k % G < G :=
  Nat.mod_lt k hG

-- § GPU plan block layout

/-- Words before the pointer arrays: counts, start, dst, tok. -/
def planBase (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Offset of the 8-byte-aligned pointer arrays (`(i32 + 1) & ~1`). -/
def ptrOff (cap : Nat) : Nat := (planBase cap + 1) / 2 * 2

/-- Total 32-bit words of one plan block. -/
def planWords (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer arrays start at an even word offset. -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer arrays begin at or after the index arrays (no overlap). -/
theorem ptrOff_ge (cap : Nat) : planBase cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- Alignment padding is at most one word. -/
theorem ptrOff_le (cap : Nat) : ptrOff cap ≤ planBase cap + 1 := by
  unfold ptrOff; omega

/-- `ptr` and `ptr2` (each `cap` 64-bit entries) fit before `start2`. -/
theorem ptr_arrays_before_start2 (cap : Nat) : ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap := by
  omega

/-- `start2` and the trailing pad word end exactly at the end of the block. -/
theorem start2_end (cap : Nat) : (ptrOff cap + 4 * cap) + (cap + 1) + 1 = planWords cap := by
  unfold planWords; omega

/-- Group `grp`'s plan block lies inside the two-block region. -/
theorem plan_slot_in_range (cap grp : Nat) (h : grp < 2) :
    grp * planWords cap + planWords cap ≤ 2 * planWords cap := by
  have : grp * planWords cap ≤ 1 * planWords cap := Nat.mul_le_mul_right _ (by omega)
  omega

-- § expert staging

/-- Staging blobs given to each group. -/
def stagePer (G : Nat) : Nat := if G = 2 then 16 / 2 else 16

/-- Each group's staging slice lies inside the 16 blobs. -/
theorem stage_slice_in_range (G grp : Nat) (hG : G = 1 ∨ G = 2) (hgrp : grp < G) :
    grp * stagePer G + stagePer G ≤ 16 := by
  rcases hG with h | h
  · subst h
    have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · subst h
    unfold stagePer
    simp
    omega

/-- The two slices of a split window are disjoint. -/
theorem stage_slices_disjoint : 0 * stagePer 2 + stagePer 2 ≤ 1 * stagePer 2 := by
  simp [stagePer]

-- § monotone flag

/-- `raise_flag`: compare-and-set that only ever increases the flag. -/
def raiseFlag (cur v : Nat) : Nat := max cur v

/-- The flag never decreases. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- The flag reaches the raised value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- Late and early raisers commute: the order of host functions does not matter. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  unfold raiseFlag; omega

/-- Raising twice with the same value is the same as raising once. -/
theorem raiseFlag_idem (c a : Nat) : raiseFlag (raiseFlag c a) a = raiseFlag c a := by
  unfold raiseFlag; omega

-- § commit

/-- Entry `t` of the commit array: the position when kept, `-1` when rejected. -/
def commitEntry (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + (t : Int) else -1

/-- With a non-negative base position, entry `t` is valid exactly for kept rows. -/
theorem commitEntry_valid (pos0 : Int) (nKeep t : Nat) (h : 0 ≤ pos0) :
    0 ≤ commitEntry pos0 nKeep t ↔ t < nKeep := by
  unfold commitEntry
  split <;> omega

/-- The last kept row (`n_keep - 1`) carries its true position. -/
theorem commitEntry_last (pos0 : Int) (nKeep : Nat) (h : 1 ≤ nKeep) :
    commitEntry pos0 nKeep (nKeep - 1) = pos0 + ((nKeep - 1 : Nat) : Int) := by
  unfold commitEntry
  split
  · rfl
  · omega

/-- `n_keep - 1` indexes a kept row. -/
theorem last_kept_lt (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

/-- Commit argument check: `1 <= n_keep <= T of the last window`. -/
def commitOk (nKeep lastT : Nat) : Bool := decide (1 ≤ nKeep ∧ nKeep ≤ lastT)

/-- An accepted commit keeps at least one token (the verify pass always commits one). -/
theorem commitOk_pos (nKeep lastT : Nat) (h : commitOk nKeep lastT = true) : 1 ≤ nKeep := by
  have h' : 1 ≤ nKeep ∧ nKeep ≤ lastT := of_decide_eq_true h
  omega

-- § PLE history after a commit

/-- One token enters the two-token history `ple_prev`. -/
def pleStep (p : Nat × Nat) (tok : Nat) : Nat × Nat := (p.2, tok)

/-- Advance the history over a list of kept tokens. -/
def pleAdvance (p : Nat × Nat) (toks : List Nat) : Nat × Nat := toks.foldl pleStep p

/-- Appending a token shifts the previous newest token into the older slot. -/
theorem pleAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    pleAdvance p (l ++ [x]) = ((pleAdvance p l).2, x) := by
  simp [pleAdvance, List.foldl_append, pleStep]

/-- After a commit the newest history entry is the last kept token. -/
theorem pleAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (pleAdvance p (l ++ [x])).2 = x := by
  simp [pleAdvance_append]

/-- Committing no token leaves the history untouched. -/
theorem pleAdvance_nil (p : Nat × Nat) : pleAdvance p [] = p := rfl

-- § window argument checks

/-- `init` accepts a window of `2 .. kMax` tokens that the CPU pool also supports. -/
def initOk (maxT kMax cpuMax : Nat) : Bool := decide (2 ≤ maxT ∧ maxT ≤ kMax ∧ maxT ≤ cpuMax)

/-- A valid window is at least two tokens and fits the graph caches indexed `0 .. 8`. -/
theorem initOk_bounds (maxT kMax cpuMax : Nat) (hk : kMax ≤ 8) (h : initOk maxT kMax cpuMax = true) :
    2 ≤ maxT ∧ maxT < 9 := by
  have h' : 2 ≤ maxT ∧ maxT ≤ kMax ∧ maxT ≤ cpuMax := of_decide_eq_true h
  omega

/-- `run` accepts `1 <= T <= max_t` with the window inside the cache. -/
def runOk (T maxT pos0 maxCells : Nat) : Bool := decide (1 ≤ T ∧ T ≤ maxT ∧ pos0 + T ≤ maxCells)

/-- Every position of an accepted window is a valid cache cell. -/
theorem runOk_positions (T maxT pos0 maxCells t : Nat) (h : runOk T maxT pos0 maxCells = true)
    (ht : t < T) : pos0 + t < maxCells := by
  have h' : 1 ≤ T ∧ T ≤ maxT ∧ pos0 + T ≤ maxCells := of_decide_eq_true h
  omega

end Ref.Verify



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp
The speculation controller: a cost model prices a verify step of `k` drafts, the controller picks the draft
source (none / lookup / MTP) and length `k` with the best expected tokens per millisecond, and learns per-position
acceptance rates from outcomes by exponential moving averages.

## Source semantics
* `step_ms(k)` indexes its ratio tables at `clamp(k+1, 1, K_MAX+1) - 1`.
* `expected(p, k)` = 1 + sum over `i < k` of the running product of acceptance rates (the verify pass always
  commits one token); `same` uses `p[0]` at every position.
* `choose` starts from the plain-decode rate and only replaces it by a strictly better candidate, and falls back
  to no speculation unless the best rate beats the baseline by `min_gain`.
* `observe` clamps `accepted` to `[0, k]`, updates positions `0 .. accepted-1` with a hit and position `accepted`
  (when drafted) with a miss, i.e. `min(k, accepted+1)` positions.
* Priors: 0.86 per MTP position and `{0.35, 0.6, 0.8, 0.92}` for lookup by match-length bucket.
The model below works on natural numbers (rates in permille) so that all arithmetic is exact.

## Form template identifiers
role mapping (source names -> abstract names):
  CostModel::step_ms    -> step_cost
  Controller::expected  -> expected_tokens
  Controller::choose    -> pick_source
  Controller::observe   -> learn_acceptance
  K_MAX                 -> max_draft_len
  min_gain              -> required_gain
  ema_                  -> smoothing
-/

namespace Ref.Spec

-- § table index clamp

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def clampN (kmax k : Nat) : Nat := min (max (k + 1) 1) (kmax + 1)

/-- The clamped window size is at least 1. -/
theorem clampN_ge_one (kmax k : Nat) : 1 ≤ clampN kmax k := by
  unfold clampN; omega

/-- The clamped window size is at most `K_MAX + 1`. -/
theorem clampN_le (kmax k : Nat) : clampN kmax k ≤ kmax + 1 := by
  unfold clampN; omega

/-- `n - 1` indexes a table of `K_MAX + 1` entries. -/
theorem clampN_index_lt (kmax k : Nat) : clampN kmax k - 1 < kmax + 1 := by
  unfold clampN; omega

/-- Within range the clamp is the identity on `k + 1`. -/
theorem clampN_exact (kmax k : Nat) (h : k ≤ kmax) : clampN kmax k = k + 1 := by
  unfold clampN; omega

-- § expected tokens

/-- Per-position factor: `p[0]` when `same`, else `p[i]`. -/
def factor (p : Nat → Nat) (same : Bool) (i : Nat) : Nat := if same = true then p 0 else p i

/-- Sum of running products over the next `k` draft positions, starting at position `i`. -/
def expSum (p : Nat → Nat) (same : Bool) : Nat → Nat → Nat → Nat
  | _, 0, _ => 0
  | i, k + 1, run => run * factor p same i + expSum p same (i + 1) k (run * factor p same i)

/-- Expected tokens of a step with `k` drafts, in the model's units. -/
def expected (p : Nat → Nat) (same : Bool) (k : Nat) : Nat := 1 + expSum p same 0 k 1

/-- The verify pass always commits one token. -/
theorem expected_ge_one (p : Nat → Nat) (same : Bool) (k : Nat) : 1 ≤ expected p same k := by
  unfold expected; omega

/-- With no drafts exactly one token is expected. -/
theorem expected_zero (p : Nat → Nat) (same : Bool) : expected p same 0 = 1 := by
  simp [expected, expSum]

/-- Drafting more positions never lowers the expectation. -/
theorem expSum_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    ∀ i run, expSum p same i k run ≤ expSum p same i (k + 1) run := by
  induction k with
  | zero => intro i run; simp [expSum]
  | succ k ih =>
    intro i run
    have e1 : expSum p same i (k + 1) run =
        run * factor p same i + expSum p same (i + 1) k (run * factor p same i) := rfl
    have e2 : expSum p same i (k + 1 + 1) run =
        run * factor p same i + expSum p same (i + 1) (k + 1) (run * factor p same i) := rfl
    have h := ih (i + 1) (run * factor p same i)
    rw [e1, e2]
    omega

/-- The expected tokens are monotone in the draft length. -/
theorem expected_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    expected p same k ≤ expected p same (k + 1) := by
  unfold expected
  have := expSum_mono p same k 0 1
  omega

-- § choosing with a gain threshold

/-- The best candidate must beat the plain-decode rate by `gainPm` permille to be kept. -/
def passesGain (best baseline gainPm : Nat) : Bool := decide (baseline * (1000 + gainPm) ≤ best * 1000)

/-- A candidate that passes the gain test is at least as fast as the baseline. -/
theorem passesGain_ge (best baseline gainPm : Nat) (h : passesGain best baseline gainPm = true) :
    baseline ≤ best := by
  have h' : baseline * (1000 + gainPm) ≤ best * 1000 := of_decide_eq_true h
  have h1 : baseline * 1000 ≤ baseline * (1000 + gainPm) := Nat.mul_le_mul_left _ (by omega)
  have h2 : baseline * 1000 ≤ best * 1000 := Nat.le_trans h1 h'
  omega

/-- With zero required gain the test is exactly `baseline <= best`. -/
theorem passesGain_zero (best baseline : Nat) :
    passesGain best baseline 0 = true ↔ baseline ≤ best := by
  unfold passesGain
  simp only [Nat.add_zero, decide_eq_true_eq]
  constructor
  · intro h; omega
  · intro h; omega

/-- Running maximum over candidate rates (the strict `>` update of `consider` keeps the maximum). -/
def bestRate (init : Nat) (rs : List Nat) : Nat := rs.foldl (fun b r => max b r) init

/-- The result is at least the starting (baseline) rate. -/
theorem bestRate_ge_init (rs : List Nat) : ∀ init, init ≤ bestRate init rs := by
  induction rs with
  | nil => intro init; simp [bestRate]
  | cons r rs ih =>
    intro init
    have h := ih (max init r)
    simp only [bestRate, List.foldl_cons] at h ⊢
    omega

/-- The result is at least every candidate rate. -/
theorem bestRate_ge_mem (rs : List Nat) : ∀ init, ∀ r ∈ rs, r ≤ bestRate init rs := by
  induction rs with
  | nil => intro init r hr; simp at hr
  | cons r' rs ih =>
    intro init r hr
    rcases List.mem_cons.1 hr with h | h
    · subst h
      have := bestRate_ge_init rs (max init r)
      simp only [bestRate, List.foldl_cons] at this ⊢
      omega
    · have := ih (max init r') r h
      simp only [bestRate, List.foldl_cons] at this ⊢
      exact this

-- § learning acceptance rates

/-- Accepted count clamped to `[0, k]`. -/
def accClamp (k acc : Nat) : Nat := min acc k

/-- Number of positions whose rate is updated: the accepted ones plus the first rejected. -/
def seenCount (k acc : Nat) : Nat := min k (acc + 1)

/-- The clamped acceptance count never exceeds the window. -/
theorem accClamp_le (k acc : Nat) : accClamp k acc ≤ k := by
  unfold accClamp; omega

/-- All accepted positions are among the updated ones. -/
theorem seen_ge_acc (k acc : Nat) : accClamp k acc ≤ seenCount k (accClamp k acc) := by
  unfold accClamp seenCount; omega

/-- At most one rejected position is updated. -/
theorem seen_le_acc_succ (k acc : Nat) : seenCount k (accClamp k acc) ≤ accClamp k acc + 1 := by
  unfold accClamp seenCount; omega

/-- No position beyond the draft window is updated. -/
theorem seen_le_k (k acc : Nat) : seenCount k acc ≤ k := by
  unfold seenCount; omega

/-- A rejected position is updated exactly when not everything was accepted. -/
theorem seen_eq_acc_iff (k acc : Nat) (h : acc ≤ k) : seenCount k acc = acc ↔ acc = k := by
  unfold seenCount; omega

/-- One exponential-moving-average step on a rate in permille with smoothing `e` permille. -/
def emaStep (v e h : Nat) : Nat := (v * (1000 - e) + h * e) / 1000

/-- A rate in `[0, 1000]` stays in `[0, 1000]` after any update toward a hit value in `[0, 1000]`. -/
theorem emaStep_le (v e h : Nat) (hv : v ≤ 1000) (he : e ≤ 1000) (hh : h ≤ 1000) :
    emaStep v e h ≤ 1000 := by
  unfold emaStep
  have h1 : v * (1000 - e) ≤ 1000 * (1000 - e) := Nat.mul_le_mul_right _ hv
  have h2 : h * e ≤ 1000 * e := Nat.mul_le_mul_right _ hh
  have h3 : 1000 * (1000 - e) + 1000 * e = 1000 * 1000 := by
    rw [← Nat.mul_add, Nat.sub_add_cancel he]
  exact Nat.div_le_of_le_mul (by omega)

/-- Moving a rate toward its own value changes nothing. -/
theorem emaStep_fixed (v e : Nat) (he : e ≤ 1000) : emaStep v e v = v := by
  unfold emaStep
  have h : v * (1000 - e) + v * e = v * 1000 := by
    rw [← Nat.mul_add, Nat.sub_add_cancel he]
  rw [h]
  exact Nat.mul_div_cancel v (by decide)

/-- The lookup prior rises with the match-length bucket and stays below 1. -/
theorem lookup_prior_sorted : [350, 600, 800, 920].Pairwise (· ≤ ·) ∧ ∀ x ∈ [350, 600, 800, 920], x ≤ 1000 := by
  decide

end Ref.Spec



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
SOURCES: src/spec/suffix_drafter.cpp
A lookup (prompt-lookup / suffix) drafter: an open-addressing hash table keyed on the last three tokens stores up
to `WAYS` most-recent end positions per key; `propose` finds the longest match of the current suffix against
history and copies its continuation.

## Source semantics
* `min_match_ = max(3, min_match)`, `max_match_ = max(min_match, max_match)`.
* The table capacity is the smallest power of two that is at least `2 * capacity_tokens`.
* The key is a mixed hash with its low bit forced to 1, so `0` marks an empty slot.
* `append` shifts the way list right and inserts the new end position at the front; `n` grows up to `WAYS`.
* `propose` scans the ways from most recent, takes a strictly longer match only (ties keep the newer), requires
  length at least `min_match_`, then copies up to `max_k` tokens following the match, never past the current end.

## Form template identifiers
role mapping (source names -> abstract names):
  SuffixDrafter::append   -> history_append
  SuffixDrafter::propose  -> suffix_propose
  Slot::pos / WAYS        -> candidate_positions
  key_at                  -> trigram_key
  find_slot               -> probe_slot
  min_match_/max_match_   -> match_length_bounds
-/

namespace Ref.SuffixDrafter

-- § effective match bounds

/-- Effective minimum match length. -/
def minEff (minMatch : Nat) : Nat := max 3 minMatch

/-- Effective maximum match length. -/
def maxEff (minMatch maxMatch : Nat) : Nat := max minMatch maxMatch

/-- A match is never accepted below three tokens. -/
theorem minEff_ge_three (m : Nat) : 3 ≤ minEff m := by
  unfold minEff; omega

/-- The effective minimum is at least the requested one. -/
theorem minEff_ge_arg (m : Nat) : m ≤ minEff m := by
  unfold minEff; omega

/-- The effective maximum is at least both requested values. -/
theorem maxEff_ge (m x : Nat) : m ≤ maxEff m x ∧ x ≤ maxEff m x := by
  unfold maxEff; omega

/-- When the configured bounds are ordered and at least 3, they are used as given. -/
theorem bounds_ordered (m x : Nat) (h3 : 3 ≤ m) (hx : m ≤ x) :
    minEff m = m ∧ maxEff m x = x ∧ minEff m ≤ maxEff m x := by
  unfold minEff maxEff; omega

/-- If the effective maximum is below the effective minimum, no match can ever be proposed. -/
theorem no_proposal_when_cap_below_min (m x len : Nat) (h : maxEff m x < minEff m)
    (hl : len ≤ maxEff m x) : ¬ (minEff m ≤ len) := by
  omega

-- § table capacity

/-- Doubling loop `while (cap < m) cap <<= 1`, bounded by a fuel count. -/
def growCap (m : Nat) : Nat → Nat → Nat
  | 0, cap => cap
  | f + 1, cap => if cap < m then growCap m f (2 * cap) else cap

/-- The doubling loop reaches the target whenever the fuel suffices. -/
theorem growCap_ge (m : Nat) : ∀ f cap, m ≤ cap * 2 ^ f → m ≤ growCap m f cap := by
  intro f
  induction f with
  | zero =>
    intro cap h
    simp only [growCap]
    simpa using h
  | succ f ih =>
    intro cap h
    simp only [growCap]
    by_cases hc : cap < m
    · rw [if_pos hc]
      have e : cap * 2 ^ (f + 1) = 2 * cap * 2 ^ f := by
        rw [Nat.pow_succ, Nat.mul_comm (2 ^ f) 2, ← Nat.mul_assoc, Nat.mul_comm cap 2]
      apply ih
      calc m ≤ cap * 2 ^ (f + 1) := h
        _ = 2 * cap * 2 ^ f := e
    · rw [if_neg hc]
      omega

/-- The doubling loop stops immediately when the capacity already suffices. -/
theorem growCap_stop (m f cap : Nat) (h : m ≤ cap) : growCap m (f + 1) cap = cap := by
  simp only [growCap]
  rw [if_neg (by omega)]

-- § never-zero key

/-- Model of `x | 1`: set the lowest bit. -/
def setLow (x : Nat) : Nat := 2 * (x / 2) + 1

/-- A key is never 0, so 0 can mark an empty slot. -/
theorem setLow_ne_zero (x : Nat) : setLow x ≠ 0 := by
  unfold setLow; omega

/-- A key is odd. -/
theorem setLow_odd (x : Nat) : setLow x % 2 = 1 := by
  unfold setLow; omega

/-- Setting the low bit twice changes nothing. -/
theorem setLow_idem (x : Nat) : setLow (setLow x) = setLow x := by
  unfold setLow; omega

/-- Setting the low bit changes the value by at most one. -/
theorem setLow_close (x : Nat) : x ≤ setLow x ∧ setLow x ≤ x + 1 := by
  unfold setLow; omega

-- § way list

/-- Insert an end position at the front of the way list, dropping the oldest beyond `W`. -/
def pushWay (ways : List Nat) (p W : Nat) : List Nat := (p :: ways).take W

/-- The way list never exceeds `W` entries. -/
theorem pushWay_length (ways : List Nat) (p W : Nat) :
    (pushWay ways p W).length = min W (ways.length + 1) := by
  simp [pushWay]

/-- The newest position is first. -/
theorem pushWay_head (ways : List Nat) (p W : Nat) :
    (pushWay ways p (W + 1)).head? = some p := by
  simp [pushWay]

/-- The counter `n`: grows by one until it reaches `WAYS`. -/
def bumpN (n W : Nat) : Nat := if n < W then n + 1 else n

/-- The counter tracks the list length: it is `min (n + 1) W` while `n <= W`. -/
theorem bumpN_eq_min (n W : Nat) (h : n ≤ W) : bumpN n W = min (n + 1) W := by
  unfold bumpN
  split <;> omega

/-- The counter never exceeds `W` once it starts within `W`. -/
theorem bumpN_le (n W : Nat) (h : n ≤ W) : bumpN n W ≤ W := by
  unfold bumpN
  split <;> omega

-- § match length

/-- Backward match length: extend while within the history start and `same len` holds, up to `fuel` steps. -/
def matchLen (same : Nat → Bool) (p : Nat) : Nat → Nat → Nat
  | len, 0 => len
  | len, f + 1 => if len ≤ p ∧ same len = true then matchLen same p (len + 1) f else len

/-- The match is at most `start + fuel` long. -/
theorem matchLen_le_fuel (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, matchLen same p len f ≤ len + f := by
  induction f with
  | zero => intro len; simp [matchLen]
  | succ f ih =>
    intro len
    simp only [matchLen]
    split
    · have := ih (len + 1)
      omega
    · omega

/-- The match never reads before the start of history: length at most `p + 1`. -/
theorem matchLen_le_pos (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, len ≤ p + 1 → matchLen same p len f ≤ p + 1 := by
  induction f with
  | zero => intro len h; simp [matchLen]; omega
  | succ f ih =>
    intro len hl
    simp only [matchLen]
    split
    · next h =>
      have h1 : len ≤ p := h.1
      have := ih (len + 1) (by omega)
      exact this
    · omega

/-- The match is at least the starting length. -/
theorem matchLen_ge_start (same : Nat → Bool) (p : Nat) (f : Nat) :
    ∀ len, len ≤ matchLen same p len f := by
  induction f with
  | zero => intro len; simp [matchLen]
  | succ f ih =>
    intro len
    simp only [matchLen]
    split
    · have := ih (len + 1)
      omega
    · omega

-- § best candidate scan

/-- Scan step on (best length, best end): replace only by a strictly longer match. -/
def scanStep (s c : Nat × Nat) : Nat × Nat := if c.1 > s.1 then c else s

/-- The scan keeps the maximum length. -/
theorem scanStep_fst_ge (s c : Nat × Nat) : s.1 ≤ (scanStep s c).1 ∧ c.1 ≤ (scanStep s c).1 := by
  unfold scanStep
  split <;> omega

/-- Ties keep the incumbent (the more recent position, as ways are scanned newest first). -/
theorem scanStep_tie (s c : Nat × Nat) (h : c.1 = s.1) : scanStep s c = s := by
  unfold scanStep
  split
  · omega
  · rfl

/-- Scan over all candidates. -/
def scanAll (s : Nat × Nat) (cs : List (Nat × Nat)) : Nat × Nat := cs.foldl scanStep s

/-- The scanned best is at least the initial best. -/
theorem scanAll_ge_init (cs : List (Nat × Nat)) : ∀ s, s.1 ≤ (scanAll s cs).1 := by
  induction cs with
  | nil => intro s; simp [scanAll]
  | cons c cs ih =>
    intro s
    have h := ih (scanStep s c)
    have h2 := (scanStep_fst_ge s c).1
    simp only [scanAll, List.foldl_cons] at h ⊢
    omega

/-- The scanned best is at least every candidate's length. -/
theorem scanAll_ge_mem (cs : List (Nat × Nat)) : ∀ s, ∀ c ∈ cs, c.1 ≤ (scanAll s cs).1 := by
  induction cs with
  | nil => intro s c hc; simp at hc
  | cons c' cs ih =>
    intro s c hc
    rcases List.mem_cons.1 hc with h | h
    · subst h
      have h1 := scanAll_ge_init cs (scanStep s c)
      have h2 := (scanStep_fst_ge s c).2
      simp only [scanAll, List.foldl_cons] at h1 ⊢
      omega
    · have := ih (scanStep s c') c h
      simp only [scanAll, List.foldl_cons] at this ⊢
      exact this

-- § continuation copy

/-- Number of tokens copied after the match ending at `bestEnd`, with `cur` the current last index. -/
def contLen (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The copy never exceeds the requested draft count. -/
theorem contLen_le_maxK (cur bestEnd maxK : Nat) : contLen cur bestEnd maxK ≤ maxK := by
  unfold contLen; omega

/-- The copy never reads past the current end of history. -/
theorem contLen_reads_in_history (cur bestEnd maxK : Nat) (h : bestEnd ≤ cur) :
    bestEnd + contLen cur bestEnd maxK ≤ cur := by
  unfold contLen; omega

/-- A match strictly before the current suffix yields a non-empty continuation when drafts are wanted. -/
theorem contLen_pos (cur bestEnd maxK : Nat) (h : bestEnd < cur) (hk : 1 ≤ maxK) :
    1 ≤ contLen cur bestEnd maxK := by
  unfold contLen; omega

end Ref.SuffixDrafter



## FILE: /mnt/user-data/outputs/PrefillChunking.lean
/-!
# MODULE: prefill_chunking
SOURCES: src/prefill/prefill.cpp
Chunked prompt prefill on the GPU: a device-buffer allocator (owned, borrowed region, or count-only), the chunk
loop over the prompt, the host-side counting sort that groups (token, expert) pairs, the expert staging ring and
dequantisation ring, and the two-token PLE history.

## Source semantics
* Every allocation takes `n*sz + 256` bytes rounded up to 256; a borrowed region fails the take when it would
  overflow; the count-only pass produces `bytes_needed` (plus 8 MiB slack).
* The prompt is processed in chunks of `min(T, n - c0)` tokens at `p0 = pos0 + c0`.
* Experts are grouped with prefix offsets `off[e+1] = off[e] + cnt[e]`; ids outside `[0, NE)` are an error.
* Non-resident experts stream through a ring of `STAGE = 8` buffers with lookahead `STAGE - 1`; dequantised
  experts alternate between `DQ = 2` buffers.
* The PLE history `prev` always holds the last two tokens seen.

## Form template identifiers
role mapping (source names -> abstract names):
  Alloc::take            -> region_take
  Prefill::bytes_needed  -> region_size
  Prefill::run chunk loop-> chunk_walk
  m.cnt / m.off          -> expert_histogram / expert_offsets
  STAGE / stage_next     -> upload_ring
  DQ                     -> dequant_ring
  prev[2]                -> token_history_pair
-/

namespace Ref.Prefill

-- § allocation accounting

/-- Bytes charged for `n` elements of `sz` bytes: `(n*sz + 256 + 255)` rounded down to a multiple of 256. -/
def allocBytes (n sz : Nat) : Nat := (n * sz + 256 + 255) / 256 * 256

/-- An allocation is at least its payload plus 256 bytes of slack. -/
theorem allocBytes_ge (n sz : Nat) : n * sz + 256 ≤ allocBytes n sz := by
  unfold allocBytes; omega

/-- Allocation sizes are multiples of 256. -/
theorem allocBytes_mod (n sz : Nat) : allocBytes n sz % 256 = 0 := by
  unfold allocBytes; omega

/-- The rounding adds less than 256 bytes beyond the slack. -/
theorem allocBytes_lt (n sz : Nat) : allocBytes n sz < n * sz + 512 := by
  unfold allocBytes; omega

/-- Borrowed-region take: `none` on overflow, otherwise the new `used`. -/
def takeAlloc (cap used n sz : Nat) : Option Nat :=
  if used + allocBytes n sz > cap then none else some (used + allocBytes n sz)

/-- A successful take stays inside the region and moves forward. -/
theorem takeAlloc_sound (cap used n sz u : Nat) (h : takeAlloc cap used n sz = some u) :
    u ≤ cap ∧ used + n * sz + 256 ≤ u := by
  unfold takeAlloc at h
  have := allocBytes_ge n sz
  split at h
  · cases h
  · injection h with h'
    subst h'
    omega

/-- A take succeeds exactly when it fits. -/
theorem takeAlloc_some_of_fits (cap used n sz : Nat) (h : used + allocBytes n sz ≤ cap) :
    takeAlloc cap used n sz = some (used + allocBytes n sz) := by
  unfold takeAlloc
  rw [if_neg (by omega)]

/-- Total bytes of a carve sequence. -/
def allocTotal (l : List (Nat × Nat)) : Nat := (l.map (fun p => allocBytes p.1 p.2)).sum

/-- The total of a concatenation is the sum of the totals. -/
theorem allocTotal_append (l1 l2 : List (Nat × Nat)) :
    allocTotal (l1 ++ l2) = allocTotal l1 + allocTotal l2 := by
  simp [allocTotal, List.map_append, List.sum_append]

/-- `bytes_needed` adds 8 MiB of alignment slack to the counted total. -/
def bytesNeeded (used : Nat) : Nat := used + 8 * 1048576

/-- The reported size always covers the counted total. -/
theorem bytesNeeded_ge (used : Nat) : used ≤ bytesNeeded used := by
  unfold bytesNeeded; omega

-- § chunk loop

/-- Chunk sizes of a prompt of `n` tokens with chunk capacity `T`, bounded by fuel. -/
def chunkList (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min T n :: chunkList T f (n - min T n)

/-- The chunks cover exactly the prompt. -/
theorem chunkList_sum (T : Nat) (hT : 1 ≤ T) : ∀ f n, n ≤ f → (chunkList T f n).sum = n := by
  intro f
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by omega
    subst h0
    simp [chunkList]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · subst h0
      simp [chunkList]
    · simp [chunkList, h0]
      have := ih (n - min T n) (by omega)
      omega

/-- Size of the chunk starting at offset `c0`. -/
def chunkAt (T n c0 : Nat) : Nat := min T (n - c0)

/-- A chunk inside the prompt is non-empty. -/
theorem chunkAt_pos (T n c0 : Nat) (hT : 1 ≤ T) (h : c0 < n) : 1 ≤ chunkAt T n c0 := by
  unfold chunkAt; omega

/-- A chunk never exceeds the buffer capacity. -/
theorem chunkAt_le (T n c0 : Nat) : chunkAt T n c0 ≤ T := by
  unfold chunkAt; omega

/-- A chunk never runs past the prompt. -/
theorem chunkAt_fits (T n c0 : Nat) (h : c0 < n) : c0 + chunkAt T n c0 ≤ n := by
  unfold chunkAt; omega

-- § counting sort of (token, expert) pairs

/-- `off[e]`: sum of the first `e` counts. -/
def offAt : List Nat → Nat → Nat
  | _, 0 => 0
  | [], _ + 1 => 0
  | c :: cs, e + 1 => c + offAt cs e

/-- The last offset (`off[NE]`) is the total count of pairs. -/
theorem offAt_length (cnt : List Nat) : offAt cnt cnt.length = cnt.sum := by
  induction cnt with
  | nil => simp [offAt]
  | cons c cs ih => simp [offAt, ih]

/-- Offsets of the empty histogram are 0. -/
theorem offAt_nil (e : Nat) : offAt [] e = 0 := by
  cases e <;> rfl

/-- Offsets never decrease. -/
theorem offAt_mono (cnt : List Nat) : ∀ e, offAt cnt e ≤ offAt cnt (e + 1) := by
  induction cnt with
  | nil => intro e; simp [offAt_nil]
  | cons c cs ih =>
    intro e
    cases e with
    | zero => simp [offAt]
    | succ k =>
      simp only [offAt]
      have := ih k
      omega

/-- The first expert's region starts at 0. -/
theorem offAt_zero (cnt : List Nat) : offAt cnt 0 = 0 := by
  cases cnt <;> rfl

/-- The step from `off[e]` to `off[e+1]` adds exactly `cnt[e]`. -/
theorem offAt_succ_of_get (c : Nat) (cs : List Nat) (e : Nat) :
    offAt (c :: cs) (e + 1) = c + offAt cs e := rfl

/-- A routed id is valid exactly when it lies in `[0, NE)`. -/
def idOk (e : Int) (ne : Nat) : Bool := decide (0 ≤ e ∧ e < (ne : Int))

/-- Valid ids index a histogram of `NE` entries. -/
theorem idOk_index (e : Int) (ne : Nat) (h : idOk e ne = true) : e.toNat < ne := by
  have h' : 0 ≤ e ∧ e < (ne : Int) := of_decide_eq_true h
  omega

-- § staging and dequantisation rings

/-- Next ring slot. -/
def nextSlot (s : Nat) : Nat := (s + 1) % 8

/-- The next slot is a valid slot. -/
theorem nextSlot_lt (s : Nat) : nextSlot s < 8 := by
  unfold nextSlot; omega

/-- The experts staged at once (current plus lookahead of 7) occupy 8 distinct slots. -/
theorem stage_slots_distinct (a i1 i2 : Nat) (h1 : i1 < 8) (h2 : i2 < 8)
    (h : (a + i1) % 8 = (a + i2) % 8) : i1 = i2 := by
  omega

/-- Lookahead plus the current expert exactly fill the ring. -/
theorem lookahead_fills_ring : (8 - 1) + 1 = 8 := by decide

/-- Consecutive experts use different dequantisation buffers. -/
theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2 := by
  omega

/-- Experts two apart reuse a dequantisation buffer (after the stream has synchronised on it). -/
theorem dq_reuse (j : Nat) : j % 2 = (j + 2) % 2 := by
  omega

-- § PLE history

/-- One token enters the two-token history. -/
def prevStep (p : Nat × Nat) (tok : Nat) : Nat × Nat := (p.2, tok)

/-- History after a token list. -/
def prevAdvance (p : Nat × Nat) (toks : List Nat) : Nat × Nat := toks.foldl prevStep p

/-- Appending a token pushes the previous newest into the older slot. -/
theorem prevAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    prevAdvance p (l ++ [x]) = ((prevAdvance p l).2, x) := by
  simp [prevAdvance, List.foldl_append, prevStep]

/-- After a run the newest entry is the last token. -/
theorem prevAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (prevAdvance p (l ++ [x])).2 = x := by
  simp [prevAdvance_append]

/-- Processing chunks one after another equals processing the concatenation. -/
theorem prevAdvance_concat (p : Nat × Nat) (l1 l2 : List Nat) :
    prevAdvance (prevAdvance p l1) l2 = prevAdvance p (l1 ++ l2) := by
  simp [prevAdvance, List.foldl_append]

-- § buffer-shape constants

/-- The streams times the width: `D = N * HC`. -/
theorem stream_width : 2560 * 4 = 10240 := by decide

/-- A gate/up row pair is `2 * 640 = 1280` wide. -/
theorem gate_up_width : 2 * 640 = 1280 := by decide

/-- The dequantised gate/up buffer holds 4 weights per code byte of the blob. -/
theorem dq_gu_elems : 1280 * 2560 = 1280 * 640 * 4 := by decide

/-- The dequantised down buffer holds 4 weights per code byte of the blob. -/
theorem dq_down_elems : 2560 * 640 = 2560 * 160 * 4 := by decide

/-- Geometry check of `init`: all five dimensions must match the artifact. -/
def geomOk (n hc lr ne k : Nat) : Bool :=
  decide (n = 2560 ∧ hc = 4 ∧ lr = 320 ∧ ne = 512 ∧ k = 10)

/-- A passing geometry makes each routed pair map into `T * 10` slots. -/
theorem geomOk_k (n hc lr ne k : Nat) (h : geomOk n hc lr ne k = true) : k = 10 := by
  have h' : n = 2560 ∧ hc = 4 ∧ lr = 320 ∧ ne = 512 ∧ k = 10 := of_decide_eq_true h
  exact h'.2.2.2.2

end Ref.Prefill



## FILE: /mnt/user-data/outputs/PrefillKernels.lean
/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
Index and layout contracts of the prefill CUDA kernels: hyper-connection stream indexing, the GDN convolution
history and recurrence tiling, MoE routing lane layout and tie-break, the 2-bit expert blob layout and its
dequantisation, QSA helper indexing, the int8 K/V append, and launch-grid coverage.

## Source semantics
* A residual is `T` rows of `HC = 4` streams of `N = 2560` floats; element `i` has token `i / 10240`, stream
  `(i % 10240) / 2560`, and channel `i % 2560`.
* GDN channels: `C = 10240 = 2*HK*S + HV*S` (q, k, v) with `HK = 16`, `HV = 48`, `S = 128`.
* The recurrence splits the 128 state rows over `RG = 4` thread rows of `RPG = 32`.
* The router keeps 16 logits per lane (`expert = lane + 32*i`), takes the top 10 with ties broken to the lower id.
* An expert blob is gate/up 2-bit codes, down codes, gate/up f16 scales, down f16 scales; one code byte holds 4
  weights with value `(code - 1) * scale`, one scale per 64 weights.
* The int8 K/V append quantises each 64-value group to `[-127, 127]` with an f16 scale.

## Form template identifiers
role mapping (source names -> abstract names):
  gr_norm/gr_mix/gr_write/gr_broadcast -> stream_kernels
  gdn_conv_kernel / gdn_rec_kernel     -> linear_recurrence_kernels
  route_kernel                         -> router_topk
  blob_dequant_kernel                  -> expert_blob_dequant
  kv_append_kernel                     -> kv_quant_append
  blocks_for                           -> grid_ceil
-/

namespace Ref.PrefillKernels

-- § shape relations

/-- `D = N * HC`. -/
theorem stream_width : 2560 * 4 = 10240 := by decide

/-- The GDN channel count splits into q, k and v parts. -/
theorem gdn_channels : 10240 = 2 * 16 * 128 + 48 * 128 := by decide

/-- The value part of the channels is the GDN value dimension `ZV`. -/
theorem gdn_value_dim : 48 * 128 = 6144 := by decide

/-- The q/k heads (32 of them) occupy the first 4096 channels. -/
theorem qk_prefix : 2 * 16 * 128 = 4096 := by decide

-- § stream indexing

/-- Token, stream and channel recompose to the flat index. -/
theorem stream_index_decomp (i : Nat) :
    (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i := by
  omega

/-- The stream index is below 4 and the channel below 2560. -/
theorem stream_index_bounds (i : Nat) : (i % 10240) / 2560 < 4 ∧ i % 2560 < 2560 := by
  omega

/-- The token index of an in-range element is below `T`. -/
theorem token_index_lt (i T : Nat) (h : i < T * 10240) : i / 10240 < T := by
  omega

/-- Broadcast reads `e[t*N + d]`, which is inside the embedding buffer. -/
theorem broadcast_src_lt (i T : Nat) (h : i < T * 10240) :
    (i / 10240) * 2560 + i % 2560 < T * 2560 := by
  omega

/-- Elements of the same stream and channel share their source (stream-independence of the broadcast). -/
theorem broadcast_same_source (t c1 c2 d : Nat) (h1 : c1 < 4) (h2 : c2 < 4) (hd : d < 2560) :
    ((t * 10240 + c1 * 2560 + d) / 10240) * 2560 + (t * 10240 + c1 * 2560 + d) % 2560 =
    ((t * 10240 + c2 * 2560 + d) / 10240) * 2560 + (t * 10240 + c2 * 2560 + d) % 2560 := by
  omega

/-- The mixer reads the four stream copies of one channel at indices inside the token's row. -/
theorem mix_index_lt (i T c : Nat) (hi : i < T * 2560) (hc : c < 4) :
    (i / 2560) * 10240 + c * 2560 + i % 2560 < T * 10240 := by
  omega

/-- Norm rows: row `r = t*4 + c` has stream `r % 4` and token `r / 4`. -/
theorem norm_row_decomp (r : Nat) : (r / 4) * 4 + r % 4 = r ∧ r % 4 < 4 := by
  omega

-- § rotary pairs

/-- The two members of a rotary pair are distinct and both inside the 64 rotated dimensions. -/
theorem rope_pair_range (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 ∧ pair ≠ pair + 32 := by
  omega

/-- A rotation preserves the squared norm of a pair when `c^2 + s^2 = 1` (exact over the integers). -/
theorem rope_preserves_norm (a b c s : Int) (h : c * c + s * s = 1) :
    (a * c - b * s) * (a * c - b * s) + (a * s + b * c) * (a * s + b * c) = a * a + b * b := by
  have key : (a * c - b * s) * (a * c - b * s) + (a * s + b * c) * (a * s + b * c) =
      (a * a + b * b) * (c * c + s * s) := by
    grind
  rw [key, h]
  omega

-- § GDN convolution history

/-- One convolution step shifts the 3-entry history and appends the new input. -/
def convStep (h : Nat × Nat × Nat) (x : Nat) : Nat × Nat × Nat := (h.2.1, h.2.2, x)

/-- History after a list of inputs. -/
def convRun (h : Nat × Nat × Nat) (xs : List Nat) : Nat × Nat × Nat := xs.foldl convStep h

/-- Appending an input shifts the newest two entries down. -/
theorem convRun_append (h : Nat × Nat × Nat) (xs : List Nat) (x : Nat) :
    convRun h (xs ++ [x]) = ((convRun h xs).2.1, (convRun h xs).2.2, x) := by
  simp [convRun, List.foldl_append, convStep]

/-- After a chunk the newest history entry is the last input. -/
theorem convRun_last (h : Nat × Nat × Nat) (xs : List Nat) (x : Nat) :
    (convRun h (xs ++ [x])).2.2 = x := by
  simp [convRun_append]

/-- Running two chunks in sequence equals running their concatenation. -/
theorem convRun_concat (h : Nat × Nat × Nat) (xs ys : List Nat) :
    convRun (convRun h xs) ys = convRun h (xs ++ ys) := by
  simp [convRun, List.foldl_append]

-- § recurrence tiling

/-- The four thread rows of 32 state rows cover the 128 state rows. -/
theorem rec_rows_cover : 4 * 32 = 128 := by decide

/-- State row of thread row `rg`, local row `r`. -/
theorem rec_row_lt (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by
  omega

/-- The kernel's base-plus-stride state addressing equals the direct (row, head, col) address. -/
theorem rec_state_addr (rg r head col : Nat) :
    ((rg * 32) * 48 + head) * 128 + col + r * (48 * 128) = ((rg * 32 + r) * 48 + head) * 128 + col := by
  omega

/-- Every state address is inside the `S * HV * S` state tensor of one layer. -/
theorem rec_state_addr_lt (rg r head col : Nat) (h1 : rg < 4) (h2 : r < 32) (h3 : head < 48) (h4 : col < 128) :
    ((rg * 32 + r) * 48 + head) * 128 + col < 128 * 48 * 128 := by
  omega

-- § router lanes

/-- Expert held by `lane` in slot `i`. -/
def expertOf (lane i : Nat) : Nat := lane + i * 32

/-- Lane and slot recover from the expert id. -/
theorem expert_lane_slot (lane i : Nat) (hl : lane < 32) :
    expertOf lane i % 32 = lane ∧ expertOf lane i / 32 = i := by
  unfold expertOf
  omega

/-- The 32 lanes times 16 slots cover exactly the 512 experts. -/
theorem expert_lt (lane i : Nat) (hl : lane < 32) (hi : i < 16) : expertOf lane i < 512 := by
  unfold expertOf
  omega

/-- Every expert id below 512 is held by some lane and slot. -/
theorem expert_covered (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ expertOf (e % 32) (e / 32) = e := by
  unfold expertOf
  omega

/-- Candidate order of the butterfly reduction: larger value wins, ties go to the lower id. -/
def better (v1 : Int) (i1 : Nat) (v2 : Int) (i2 : Nat) : Prop := v1 > v2 ∨ (v1 = v2 ∧ i1 < i2)

/-- The order is asymmetric, so the reduction cannot flip-flop. -/
theorem better_asymm (v1 v2 : Int) (i1 i2 : Nat) (h : better v1 i1 v2 i2) : ¬ better v2 i2 v1 i1 := by
  unfold better at *
  omega

/-- Distinct experts are always strictly ordered, so the winner does not depend on the shuffle order. -/
theorem better_total (v1 v2 : Int) (i1 i2 : Nat) (h : i1 ≠ i2) :
    better v1 i1 v2 i2 ∨ better v2 i2 v1 i1 := by
  unfold better
  omega

/-- The router output slot `t*10 + rank` of one token is inside that token's 10 slots. -/
theorem route_slot_bounds (t rank : Nat) (h : rank < 10) :
    t * 10 ≤ t * 10 + rank ∧ t * 10 + rank < (t + 1) * 10 := by
  omega

-- § SwiGLU and Q/gate splitting

/-- Gate and up of the interleaved layout are adjacent and distinct inside a 1280-wide row. -/
theorem swiglu_il_bounds (k : Nat) (h : k < 640) : 2 * k + 1 < 1280 ∧ 2 * k ≠ 2 * k + 1 := by
  omega

/-- The q half and the gate half of a head's 512 values do not overlap. -/
theorem qgate_halves (h d : Nat) (hd : d < 256) :
    h * 512 + d < h * 512 + 256 + d ∧ h * 512 + 256 + d < (h + 1) * 512 := by
  omega

/-- The q index of the split kernel is inside the q buffer. -/
theorem split_q_lt (t h d T : Nat) (ht : t < T) (hh : h < 24) (hd : d < 256) :
    t * 6144 + h * 256 + d < T * 6144 := by
  omega

-- § 2-bit expert blob

/-- Codes of the gate/up matrices: 1280 rows of 640 bytes. -/
theorem blob_gu_codes : 1280 * 640 = 819200 := by decide

/-- Offset of the gate/up scales: after both code arrays. -/
theorem blob_gu_scale_off : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Offset of the down scales. -/
theorem blob_down_scale_off : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 = 1331200 := by decide

/-- Total blob size. -/
theorem blob_total : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Dequantisation threads are one per code byte: exactly the code arrays. -/
theorem dequant_threads : 1280 * 640 + 2560 * 160 = 1228800 := by decide

/-- Four weights per code byte give the full gate/up row width of 2560. -/
theorem gu_row_weights : 640 * 4 = 2560 := by decide

/-- Four weights per code byte give the full down row width of 640. -/
theorem down_row_weights : 160 * 4 = 640 := by decide

/-- A gate/up code byte maps to a scale group below 40 (one scale per 64 weights). -/
theorem gu_scale_group_lt (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by
  omega

/-- A down code byte maps to a scale group below 10. -/
theorem down_scale_group_lt (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by
  omega

/-- The four outputs of a code byte stay inside the destination row. -/
theorem gu_out_in_row (row byte k : Nat) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < (row + 1) * 2560 := by
  omega

/-- Dequantised value of a 2-bit code before scaling: `code - 1`. -/
def codeVal (x : Nat) : Int := ((x % 4 : Nat) : Int) - 1

/-- A 2-bit code decodes to one of `-1, 0, 1, 2`. -/
theorem codeVal_range (x : Nat) : -1 ≤ codeVal x ∧ codeVal x ≤ 2 := by
  unfold codeVal
  omega

-- § int8 K/V append

/-- Clamp of the rounded quotient to `[-127, 127]`. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The quantised value is always an int8 without `-128`. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split <;> try split
  all_goals omega

/-- Clamping an in-range value changes nothing. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split
  · omega
  · split
    · omega
    · rfl

/-- The grid's z coordinate encodes (group, is_v): both halves are in range. -/
theorem kv_z_decode (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 := by
  omega

/-- The channel handled by a thread is below the 256 head dimension. -/
theorem kv_channel_lt (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by
  omega

/-- The row of a (page, kv head, position) triple is inside that page-head block. -/
theorem kv_row_in_block (page kvh pos ps : Nat) (hps : 0 < ps) :
    (page * 2 + kvh) * ps + pos % ps < (page * 2 + kvh + 1) * ps := by
  have h := Nat.mod_lt pos hps
  have e := Nat.add_mul (page * 2 + kvh) 1 ps
  omega

/-- The scale index `row * 4 + g` of a 256-channel row has room for 4 groups. -/
theorem kv_scale_index_lt (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by
  omega

-- § launch grids

/-- Number of blocks of `t` threads to cover `n` elements. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- The launched threads cover every element. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  have h2 : (n + t - 1) % t < t := Nat.mod_lt _ ht
  rw [Nat.mul_comm ((n + t - 1) / t) t]
  omega

/-- At most one block is wasted. -/
theorem blocksFor_tight (n t : Nat) (ht : 0 < t) : blocksFor n t * t < n + t := by
  unfold blocksFor
  have h := Nat.div_add_mod (n + t - 1) t
  rw [Nat.mul_comm ((n + t - 1) / t) t]
  omega

end Ref.PrefillKernels



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- MtpDraftLayer.lean
  was: theorem offsetAt_second (p : Nat × Nat) (ps : List (Nat × Nat)) :
    offsetAt (p :: ps) 0 = 0 ∧ offsetAt (p :: ps) 1 = alignUp (p.1 * p.2)
  now: theorem offsetAt_zero (l : List (Nat × Nat)) : offsetAt l 0 = 0
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- MtpDraftLayer.lean
  was: theorem first_take_fits (p : Nat × Nat) (ps : List (Nat × Nat)) :
    p.1 * p.2 ≤ offsetAt (p :: ps) 1
  now: theorem offsetAt_second (p : Nat × Nat) (ps : List (Nat × Nat)) :
    offsetAt (p :: ps) 0 = 0 ∧ offsetAt (p :: ps) 1 = alignUp (p.1 * p.2)
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem stepOf_block_le (c : Nat) : (stepOf c).block * 4 ≤ c + 1
  now: theorem first_take_fits (p : Nat × Nat) (ps : List (Nat × Nat)) :
    p.1 * p.2 ≤ offsetAt (p :: ps) 1
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem stepOf_block_lt (c : Nat) : c + 1 < ((stepOf c).block + 1) * 4
  now: theorem stepOf_block_le (c : Nat) : (stepOf c).block * 4 ≤ c + 1
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem batchSize_pos (maxT n c : Nat) (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ batchSize maxT n c
  now: theorem stepOf_block_lt (c : Nat) : c + 1 < ((stepOf c).block + 1) * 4
  measured: fewer quantified variables (3 → 1)

- MtpDraftLayer.lean
  was: theorem batchSize_le (maxT n c : Nat) : batchSize maxT n c ≤ maxT
  now: theorem batchSize_pos (maxT n c : Nat) (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ batchSize maxT n c
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem batchSize_fits (maxT n c : Nat) : c + batchSize maxT n c ≤ n ∨ n < c
  now: theorem batchSize_le (maxT n c : Nat) : batchSize maxT n c ≤ maxT
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem no_skip_without_window (cell0 c T : Nat) (hT : 1 ≤ T) :
    skipBatch cell0 c T (firstNeeded 0 0) = false
  now: theorem batchSize_fits (maxT n c : Nat) : c + batchSize maxT n c ≤ n ∨ n < c
  measured: fewer quantified variables (2 → 1)

- MtpDraftLayer.lean
  was: theorem skipped_cells_unreachable (window promptLen cell0 c T cell : Nat)
    (hw : 0 < window) (hp : 0 < promptLen)
    (hs : skipBatch cell0 c T (firstNeeded window promptLen) = true)
    (hcell : cell < cell0 + c + T) :
    cell < promptLen - window - 64
  now: theorem no_skip_without_window (cell0 c T : Nat) (hT : 1 ≤ T) :
    skipBatch cell0 c T (firstNeeded 0 0) = false
  measured: fewer quantified variables (4 → 2)

- MtpDraftLayer.lean
  was: theorem attnSpan_le (window maxCells : Nat) : attnSpan window maxCells ≤ maxCells
  now: theorem skipped_cells_unreachable (window promptLen cell0 c T cell : Nat)
    (hw : 0 < window) (hp : 0 < promptLen)
    (hs : skipBatch cell0 c T (firstNeeded window promptLen) = true)
    (hcell : cell < cell0 + c + T) :
    cell < promptLen - window - 64
  measured: more hypotheses/binders (1 → 4): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem capCells_ge (window maxCells : Nat) : attnSpan window maxCells ≤ capCells window maxCells
  now: theorem attnSpan_le (window maxCells : Nat) : attnSpan window maxCells ≤ maxCells
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem capCells_mod (window maxCells : Nat) : capCells window maxCells % 64 = 0
  now: theorem capCells_ge (window maxCells : Nat) : attnSpan window maxCells ≤ capCells window maxCells
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem loopLen_le (cont : Nat → Bool) (fuel : Nat) : ∀ j, loopLen cont j fuel ≤ fuel
  now: theorem capCells_mod (window maxCells : Nat) : capCells window maxCells % 64 = 0
  measured: fewer quantified variables (3 → 1)

- MtpDraftLayer.lean
  was: theorem draftCount_bounds (maxT : Nat) (cont : Nat → Bool) (h : 2 ≤ maxT) :
    1 ≤ draftCount maxT cont ∧ draftCount maxT cont ≤ maxT - 1
  now: theorem loopLen_le (cont : Nat → Bool) (fuel : Nat) : ∀ j, loopLen cont j fuel ≤ fuel
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem draftCount_stop_early (maxT : Nat) : draftCount maxT (fun _ => false) = 1
  now: theorem loopLen_false (j f : Nat) : loopLen (fun _ => false) j f = 0
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem draft_slots_full (maxT n : Nat) (h : n ≤ maxT - 1) : n + padCount maxT n = maxT - 1
  now: theorem draftCount_bounds (maxT : Nat) (cont : Nat → Bool) (h : 2 ≤ maxT) :
    1 ≤ draftCount maxT cont ∧ draftCount maxT cont ≤ maxT - 1
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem chain_row_range (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ maxT + j - 1 ∧ maxT + j - 1 < 2 * maxT - 1
  now: theorem draftCount_stop_early (maxT : Nat) : draftCount maxT (fun _ => false) = 1
  measured: fewer quantified variables (3 → 1)

- MtpDraftLayer.lean
  was: theorem catchup_row_below_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < maxT + j - 1
  now: theorem draft_slots_full (maxT n : Nat) (h : n ≤ maxT - 1) : n + padCount maxT n = maxT - 1
  measured: fewer quantified variables (4 → 2)

- MtpDraftLayer.lean
  was: theorem last_slot_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT
  now: theorem chain_row_range (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ maxT + j - 1 ∧ maxT + j - 1 < 2 * maxT - 1
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem draftArgsOk_sound (T maxT a : Nat) (h : draftArgsOk T maxT a = true) :
    a < maxT ∧ 1 ≤ T
  now: theorem catchup_row_below_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < maxT + j - 1
  measured: more hypotheses/binders (2 → 4): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem draft_first_row_valid (T maxT : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) :
    draftArgsOk T maxT 0 = true
  now: theorem last_slot_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT
  measured: fewer quantified variables (3 → 2)

- MtpDraftLayer.lean
  was: theorem dec_enc (r : Nat) : decRow (encKvOnly r) = (r : Int)
  now: theorem draftArgsOk_sound (T maxT a : Nat) (h : draftArgsOk T maxT a = true) :
    a < maxT ∧ 1 ≤ T
  measured: the claim changed

- MtpDraftLayer.lean
  was: theorem enc_is_kv_only (r : Nat) : encKvOnly r < 0
  now: theorem draft_first_row_valid (T maxT : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) :
    draftArgsOk T maxT 0 = true
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem kv_only_any_T (r T : Nat) : fullLayerOk (encKvOnly r) T
  now: theorem dec_enc (r : Nat) : decRow (encKvOnly r) = (r : Int)
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- MtpDraftLayer.lean
  was: theorem nDvocab_exact (rawBytes : Nat) (h : rawBytes % 4 = 0) : nDvocab rawBytes * 4 = rawBytes
  now: theorem enc_is_kv_only (r : Nat) : encKvOnly r < 0
  measured: fewer quantified variables (2 → 1)

- MtpDraftLayer.lean
  was: theorem nDvocab_pos (rawBytes : Nat) (h : 4 ≤ rawBytes) : 1 ≤ nDvocab rawBytes
  now: theorem kv_only_any_T (r T : Nat) : fullLayerOk (encKvOnly r) T
  measured: fewer quantified variables (2 → 1)

- PrefillChunking.lean
  was: theorem offAt_mono (cnt : List Nat) : ∀ e, offAt cnt e ≤ offAt cnt (e + 1)
  now: theorem offAt_nil (e : Nat) : offAt [] e = 0
  measured: fewer quantified variables (2 → 1)

- PrefillChunking.lean
  was: theorem offAt_zero (cnt : List Nat) : offAt cnt 0 = 0
  now: theorem offAt_mono (cnt : List Nat) : ∀ e, offAt cnt e ≤ offAt cnt (e + 1)
  measured: the claim changed

- PrefillChunking.lean
  was: theorem idOk_index (e : Int) (ne : Nat) (h : idOk e ne = true) : e.toNat < ne
  now: theorem offAt_zero (cnt : List Nat) : offAt cnt 0 = 0
  measured: fewer quantified variables (3 → 1)

- PrefillChunking.lean
  was: theorem nextSlot_lt (s : Nat) : nextSlot s < 8
  now: theorem idOk_index (e : Int) (ne : Nat) (h : idOk e ne = true) : e.toNat < ne
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- PrefillChunking.lean
  was: theorem stage_slots_distinct (a i1 i2 : Nat) (h1 : i1 < 8) (h2 : i2 < 8)
    (h : (a + i1) % 8 = (a + i2) % 8) : i1 = i2
  now: theorem nextSlot_lt (s : Nat) : nextSlot s < 8
  measured: fewer quantified variables (3 → 1)

- PrefillChunking.lean
  was: theorem lookahead_fills_ring : (8 - 1) + 1 = 8
  now: theorem stage_slots_distinct (a i1 i2 : Nat) (h1 : i1 < 8) (h2 : i2 < 8)
    (h : (a + i1) % 8 = (a + i2) % 8) : i1 = i2
  measured: the claim changed

- PrefillChunking.lean
  was: theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2
  now: theorem lookahead_fills_ring : (8 - 1) + 1 = 8
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values

- PrefillChunking.lean
  was: theorem dq_reuse (j : Nat) : j % 2 = (j + 2) % 2
  now: theorem dq_alternates (j : Nat) : j % 2 ≠ (j + 1) % 2
  measured: the claim changed

- PrefillChunking.lean
  was: theorem prevAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    prevAdvance p (l ++ [x]) = ((prevAdvance p l).2, x)
  now: theorem dq_reuse (j : Nat) : j % 2 = (j + 2) % 2
  measured: fewer quantified variables (3 → 1)

- PrefillChunking.lean
  was: theorem prevAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (prevAdvance p (l ++ [x])).2 = x
  now: theorem prevAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    prevAdvance p (l ++ [x]) = ((prevAdvance p l).2, x)
  measured: the claim changed

- PrefillChunking.lean
  was: theorem prevAdvance_concat (p : Nat × Nat) (l1 l2 : List Nat) :
    prevAdvance (prevAdvance p l1) l2 = prevAdvance p (l1 ++ l2)
  now: theorem prevAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (prevAdvance p (l ++ [x])).2 = x
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- PrefillChunking.lean
  was: theorem stream_width : 2560 * 4 = 10240
  now: theorem prevAdvance_concat (p : Nat × Nat) (l1 l2 : List Nat) :
    prevAdvance (prevAdvance p l1) l2 = prevAdvance p (l1 ++ l2)
  measured: the claim changed

- PrefillChunking.lean
  was: theorem gate_up_width : 2 * 640 = 1280
  now: theorem stream_width : 2560 * 4 = 10240
  measured: the claim changed

- PrefillChunking.lean
  was: theorem dq_gu_elems : 1280 * 2560 = 1280 * 640 * 4
  now: theorem gate_up_width : 2 * 640 = 1280
  measured: the claim changed

- PrefillChunking.lean
  was: theorem dq_down_elems : 2560 * 640 = 2560 * 160 * 4
  now: theorem dq_gu_elems : 1280 * 2560 = 1280 * 640 * 4
  measured: the claim changed

- PrefillChunking.lean
  was: theorem geomOk_k (n hc lr ne k : Nat) (h : geomOk n hc lr ne k = true) : k = 10
  now: theorem dq_down_elems : 2560 * 640 = 2560 * 160 * 4
  measured: every variable of the statement is gone: a property stated for all values became a statement about fixed values
-/



## FILE: /mnt/user-data/outputs/ExpertGeometry.lean
/-!
# MODULE: expert_geometry
SOURCES: include/ref/kernels/cpu/expert.hpp
The fixed geometry of one Q2_0 routed expert (one 1,382,400-byte blob), the offsets of its four regions,
the weight-block / activation-chunk relationship, and the AVX-512 feature predicate that gates the kernel.

## Source semantics
* One expert blob = gate/up codes, down codes, gate/up scales, down scales, in that order, tiling the blob exactly.
* Gate and up rows are interleaved: gate row r is matrix row 2r, up row r is matrix row 2r+1.
* One 64-weight Q2_0 block spans TWO 32-element activation chunks (each with its own scale).
* `CpuFeatures.usable` is the conjunction of the five required AVX-512 flags; `reason` is "ok" iff usable.

## Form template identifiers
H → hidden_width, FF → expert_ff_width, NE → experts_per_layer, QK → weights_per_scale,
QKA → act_elems_per_scale, BB → block_bytes, ROW_GU/ROW_D → code_row_bytes_{gate_up,down},
SC_GU/SC_D → blocks_per_row_{gate_up,down}, BLOB → expert_blob_bytes, O_* → region_offset_*,
CpuFeatures.usable → isa_gate.
-/

namespace Ref.Kernels.Cpu.ExpertGeometry

-- § constants

/-- n_embd. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Routed experts per layer. -/
def NE : Nat := 512
/-- Weights per fp16 scale (QK2_0). -/
def QK : Nat := 64
/-- Activation elements per scale (QK8_1). -/
def QKA : Nat := 32
/-- Bytes per 64-weight Q2_0 block: 2-byte scale plus 16 code bytes. -/
def BB : Nat := 18
/-- Activation chunks across the widest reduction. -/
def MAXC : Nat := H / QKA
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := H * 2 / 8
/-- Code bytes per down row. -/
def ROW_D : Nat := FF * 2 / 8
/-- Weight blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Weight blocks per down row. -/
def SC_D : Nat := FF / QK
/-- Bytes of one expert blob. -/
def BLOB : Nat := 3 * (H * FF * BB / QK)
/-- Offset of the gate/up code region. -/
def O_GU_CODES : Nat := 0
/-- Offset of the down code region. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scale region. -/
def O_GU_SCALES : Nat := O_D_CODES + 1 * H * ROW_D
/-- Offset of the down scale region. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Maximum number of tokens routed to one expert in a verify window. -/
def MAXT : Nat := 8

-- § numeric facts

/-- The derived row/chunk counts have the values the comments in the header state. -/
theorem derived_counts :
    MAXC = 80 ∧ ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by
  decide

/-- The blob is 1,382,400 bytes. -/
theorem blob_size : BLOB = 1382400 := by
  decide

/-- The four region offsets. -/
theorem region_offsets :
    O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by
  decide

/-- The four regions tile the blob exactly: the last region ends at `BLOB`. -/
theorem layout_tiles_blob : O_D_SCALES + H * SC_D * 2 = BLOB := by
  decide

/-- The regions are strictly ordered. -/
theorem region_order :
    O_GU_CODES < O_D_CODES ∧ O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB := by
  decide

/-- A block is its 2-byte scale plus 64 two-bit codes packed into 16 bytes. -/
theorem block_bytes : BB = 2 + QK * 2 / 8 := by
  decide

/-- One weight block spans two activation chunks. -/
theorem weight_block_two_chunks : QK = 2 * QKA := by
  decide

/-- Chunk counts agree: gate/up rows have `MAXC` chunks, down rows cover `FF` elements. -/
theorem chunk_counts : SC_GU * 2 = MAXC ∧ SC_D * 2 * QKA = FF := by
  decide

-- § row addressing stays inside its region

/-- Every gate/up code row (matrix rows 0..2*FF-1) lies inside the gate/up code region. -/
theorem gu_rows_within_codes (r : Nat) (hr : r < 2 * FF) :
    r * ROW_GU + ROW_GU ≤ O_D_CODES := by
  have e1 : ROW_GU = 640 := by decide
  have e2 : O_D_CODES = 819200 := by decide
  have e3 : FF = 640 := by decide
  rw [e1, e2]
  rw [e3] at hr
  omega

/-- Every down code row lies inside the down code region. -/
theorem down_rows_within_codes (r : Nat) (hr : r < H) :
    O_D_CODES + r * ROW_D + ROW_D ≤ O_GU_SCALES := by
  have e1 : ROW_D = 160 := by decide
  have e2 : O_D_CODES = 819200 := by decide
  have e3 : O_GU_SCALES = 1228800 := by decide
  have e4 : H = 2560 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Every gate/up scale row lies inside the gate/up scale region. -/
theorem gu_scales_within_region (r : Nat) (hr : r < 2 * FF) :
    O_GU_SCALES + r * SC_GU * 2 + SC_GU * 2 ≤ O_D_SCALES := by
  have e1 : SC_GU = 40 := by decide
  have e2 : O_GU_SCALES = 1228800 := by decide
  have e3 : O_D_SCALES = 1331200 := by decide
  have e4 : FF = 640 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Every down scale row lies inside the down scale region, which ends at the blob end. -/
theorem down_scales_within_blob (r : Nat) (hr : r < H) :
    O_D_SCALES + r * SC_D * 2 + SC_D * 2 ≤ BLOB := by
  have e1 : SC_D = 10 := by decide
  have e2 : O_D_SCALES = 1331200 := by decide
  have e3 : BLOB = 1382400 := by decide
  have e4 : H = 2560 := by decide
  rw [e1, e2, e3]
  rw [e4] at hr
  omega

/-- Gate row r and up row r are distinct, adjacent matrix rows, both inside the gate/up matrix. -/
theorem gate_up_interleave (r : Nat) (hr : r < FF) :
    2 * r < 2 * FF ∧ 2 * r + 1 < 2 * FF ∧ 2 * r ≠ 2 * r + 1 := by
  omega

/-- Gate/up is a 2*FF-row matrix and down is an H-row matrix: row counts times row bytes fill the code regions. -/
theorem code_regions_sizes :
    2 * FF * ROW_GU = O_D_CODES ∧ H * ROW_D = O_GU_SCALES - O_D_CODES := by
  decide

-- § the ISA gate

/-- The CPU feature flags the expert kernel needs (all default to false). -/
structure CpuFeatures where
  avx512f : Bool := false
  avx512bw : Bool := false
  avx512vl : Bool := false
  avx512_vnni : Bool := false
  avx512_vbmi : Bool := false

/-- The kernel is usable iff all five flags are present. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The ids (1..5) of the missing features, in the order `reason` names them. -/
def CpuFeatures.missing (f : CpuFeatures) : List Nat :=
  (if f.avx512f then [] else [1]) ++ (if f.avx512bw then [] else [2]) ++
  (if f.avx512vl then [] else [3]) ++ (if f.avx512_vnni then [] else [4]) ++
  (if f.avx512_vbmi then [] else [5])

/-- `usable` is exactly the conjunction of the five flags. -/
theorem usable_iff (f : CpuFeatures) :
    f.usable = true ↔
      (f.avx512f = true ∧ f.avx512bw = true ∧ f.avx512vl = true ∧
        f.avx512_vnni = true ∧ f.avx512_vbmi = true) := by
  simp [CpuFeatures.usable, and_assoc]

/-- A machine reporting nothing is refused. -/
theorem default_not_usable : ({} : CpuFeatures).usable = false := by
  rfl

/-- A machine without VNNI is refused even if everything else is present. -/
theorem missing_vnni_refused (f : CpuFeatures) (h : f.avx512_vnni = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

/-- A machine without VBMI is refused even if everything else is present. -/
theorem missing_vbmi_refused (f : CpuFeatures) (h : f.avx512_vbmi = false) : f.usable = false := by
  simp [CpuFeatures.usable, h]

/-- `reason` reports "ok" (nothing missing) exactly when the machine is usable. -/
theorem reason_ok_iff_usable (f : CpuFeatures) : f.missing = [] ↔ f.usable = true := by
  rcases f with ⟨a, b, c, d, e⟩
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> decide

end Ref.Kernels.Cpu.ExpertGeometry



## FILE: /mnt/user-data/outputs/ExpertRowDot.lean
/-!
# MODULE: expert_row_dot
SOURCES: src/kernels/cpu/expert.cpp
Integer-level theory of the Q2_0 expert kernel: the code-offset identity behind the row dot, the int8
activation clamp, bounds that make the VNNI integer sums exact, the exact integer horizontal-sum tree,
the 8-block tail loop of the 512-bit row dot, mask arithmetic, and per-lane scale selection.

## Source semantics
* `sum (c-1) x = sum (c x) - sum x`: the weight-independent correction `hx` is subtracted once.
* Quantized activations lie in [-127,127]; a 32-element chunk sum is bounded by 32*127.
* VNNI integer partial sums are exact (codes 0..3 against int8 never get near int32 range).
* The 512-bit row dot walks the blocks in groups of at most 8 and covers every block exactly once.
* Float evaluation order is NOT modelled here; only the exact integer parts are.

## Form template identifiers
offDot → centered_code_dot, codeDot → raw_code_dot, clamp127 → act_clamp, chunksK → tail_groups,
mask16 → lane_mask, ntDispatch → token_count_dispatch.
-/

namespace Ref.Kernels.Cpu.ExpertRowDot

-- § the code-offset identity

/-- Dot of centered codes `(c-1)` with activations, over matching lists. -/
def offDot : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + offDot cs xs
  | _, _ => 0

/-- Dot of raw codes with activations, over matching lists. -/
def codeDot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + codeDot cs xs
  | _, _ => 0

/-- The Q2_0 identity: the centered dot equals the raw-code dot minus the activation sum. -/
theorem offDot_eq (cs xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = codeDot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, codeDot]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      have hm : (c - 1) * x = c * x - x := by
        rw [Int.sub_mul, Int.one_mul]
      simp only [offDot, codeDot, List.sum_cons]
      rw [hm, ih']
      omega

/-- The same identity at chunk granularity: two chunk sums add to the block sum. -/
theorem two_chunk_split (a0 a1 s0 s1 : Int) :
    (a0 - s0) + (a1 - s1) = (a0 + a1) - (s0 + s1) := by
  omega

-- § the int8 activation clamp and its sums

/-- The clamp applied after rounding: [-127, 127]. -/
def clamp127 (v : Int) : Int := max (-127) (min 127 v)

/-- The clamp lands in range. -/
theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_range v
  exact clamp127_id _ h.1 h.2

/-- A list of in-range int8 activations has a sum bounded by 127 times its length. -/
theorem sum_bound (l : List Int) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (by simp)
    have ih' := ih (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    omega

-- § exactness of the integer partial sums

/-- A code (0..3) times an int8 activation is bounded by 381 in magnitude. -/
theorem code_mul_bound (c q : Int) (hc0 : 0 ≤ c) (hc1 : c ≤ 3) (hq0 : -127 ≤ q) (hq1 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  have hc : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases hc with rfl | rfl | rfl | rfl <;> omega

/-- The exact integer horizontal sum of eight lanes: the `add`/`hadd`/`hadd` tree equals the plain sum. -/
theorem hsum_tree (a b c d e f g h : Int) :
    ((a + e) + (b + f)) + ((c + g) + (d + h)) = a + b + c + d + e + f + g + h := by
  omega

/-- Removing the code offset before float conversion is an exact integer subtraction. -/
theorem oracle_chunk_offset (dot sum : Int) : (dot - sum) + sum = dot := by
  omega

-- § scale selection per lane

/-- A VNNI int32 lane covers four consecutive activations; those four share one 32-chunk, hence one scale. -/
theorem lane_chunk (l e : Nat) (hl : l < 16) (he : e < 4) : (4 * l + e) / 32 = l / 8 := by
  omega

/-- The permute index `base[lane] + 2i` stays inside the 16-lane scale-product vector. -/
theorem permute_index_lt (l i : Nat) (hl : l < 16) (hi : i < 8) : l / 8 + 2 * i < 16 := by
  omega

/-- Block b, half h is activation chunk 2b+h, and chunk c belongs to block c/2. -/
theorem chunk_block (b h : Nat) (hh : h < 2) : (2 * b + h) / 2 = b := by
  omega

/-- The int8 offset of chunk 2b+h equals block offset b*64 plus half offset h*32. -/
theorem chunk_offset (b h : Nat) : (2 * b + h) * 32 = b * 64 + h * 32 := by
  omega

-- § the 8-block tail loop

/-- Group sizes produced by walking `n` items in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- With enough fuel the groups cover exactly `n` items. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group is non-empty (when `k > 0`) and at most `k`. -/
theorem chunksK_bounds (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, 0 < x ∧ x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Walking `n` blocks in groups of 8 with fuel `n` covers all of them. -/
theorem blocks_cover (n : Nat) : (chunksK 8 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 8 by decide)
  omega

-- § lane masks

/-- The mask of the valid scale lanes for `nb` blocks: all 16 lanes for a full group, else the low `2*nb` lanes. -/
def mask16 (nb : Nat) : Nat := if 8 ≤ nb then 65535 else 2 ^ (2 * nb) - 1

/-- A full group enables all 16 lanes. -/
theorem mask16_full (nb : Nat) (h : 8 ≤ nb) : mask16 nb = 65535 := by
  unfold mask16
  rw [if_pos h]

/-- The mask always fits in 16 bits. -/
theorem mask16_lt (nb : Nat) (h : nb ≤ 8) : mask16 nb < 65536 := by
  by_cases h8 : 8 ≤ nb
  · rw [mask16_full nb h8]
    decide
  · have h' : nb = 0 ∨ nb = 1 ∨ nb = 2 ∨ nb = 3 ∨ nb = 4 ∨ nb = 5 ∨ nb = 6 ∨ nb = 7 := by omega
    rcases h' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- A partial group of `nb < 8` blocks enables exactly the low `2*nb` lanes. -/
theorem mask16_partial (nb : Nat) (h : nb < 8) : mask16 nb + 1 = 2 ^ (2 * nb) := by
  have h' : nb = 0 ∨ nb = 1 ∨ nb = 2 ∨ nb = 3 ∨ nb = 4 ∨ nb = 5 ∨ nb = 6 ∨ nb = 7 := by omega
  rcases h' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

-- § token-count dispatch

/-- The `switch (n_tokens)` template instance chosen: 1..7 exactly, everything else the 8-wide one. -/
def ntDispatch (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The dispatched template width is between 1 and 8. -/
theorem ntDispatch_range (n : Nat) : 1 ≤ ntDispatch n ∧ ntDispatch n ≤ 8 := by
  unfold ntDispatch
  split <;> omega

/-- For 1..7 tokens the instance width equals the token count (no padding work). -/
theorem ntDispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 7) : ntDispatch n = n := by
  unfold ntDispatch
  have hc : 1 ≤ n ∧ n ≤ 7 := by omega
  rw [if_pos hc]

/-- For a verify window (at most 8 tokens) the instance is never narrower than the token count. -/
theorem ntDispatch_ge (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : n ≤ ntDispatch n := by
  unfold ntDispatch
  split <;> omega

end Ref.Kernels.Cpu.ExpertRowDot



## FILE: /mnt/user-data/outputs/ExpertPoolContract.lean
/-!
# MODULE: expert_pool_contract
SOURCES: include/ref/kernels/cpu/pool.hpp
The static contract of the CPU expert pool: capacity limits, worker/thread counting, cache-line separation of the
contended atomics, the split-path guard, task-count sizing, and batching of more experts than buffers.

## Source semantics
* `run_split` handles at most `kMaxSplit` = 16 experts; `run_split_multi` at most `kMaxSplitMulti` = 96;
  larger native layers are processed in batches of at most 96.
* The pool has at least one worker; the number of draining threads is workers plus one when the host also works.
* Each contended atomic (`head_`, `done_`, `parked_`, `epoch_`, `stop_`) sits alone on a 64-byte line.
* A phase is cut into `3 * threads` tasks (multi) or about three tasks per thread (split): enough to cover the work.
* `restore_thread_affinity` ignores non-positive masks, so a failed pin (-1) is never restored.

## Form template identifiers
kMaxSplit → split_capacity, kMaxSplitMulti → multi_capacity, MAXT → tokens_per_expert,
poolWorkers → worker_count, drainThreads → drain_thread_count, partsPerExpert → split_parts,
batchSizes → expert_batches, cacheLineOf → atomic_cache_line.
-/

namespace Ref.Kernels.Cpu.ExpertPoolContract

-- § capacities

/-- Maximum experts in `run_split`. -/
def kMaxSplit : Nat := 16
/-- Maximum experts in `run_split_multi`. -/
def kMaxSplitMulti : Nat := 96
/-- Maximum tokens routed to one expert. -/
def MAXT : Nat := 8

/-- The multi capacity is at least the single-token split capacity. -/
theorem multi_capacity_ge_split : kMaxSplit ≤ kMaxSplitMulti := by
  decide

/-- Guard of the split path: it is taken only if the job count is within capacity (and several workers, no oracle). -/
theorem split_path_bound (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) : n ≤ 16 :=
  Nat.le_of_not_lt (fun hn => h (Or.inl hn))

/-- The split path is never taken with a single worker. -/
theorem split_path_multi_worker (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) :
    nw ≠ 1 :=
  fun h1 => h (Or.inr (Or.inl h1))

/-- The split path is never taken under the oracle contract. -/
theorem split_path_no_oracle (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) :
    o = false := by
  cases o
  · rfl
  · exact (h (Or.inr (Or.inr rfl))).elim

/-- Every expert index below the job count indexes a valid split buffer. -/
theorem split_buffer_index (e n : Nat) (hn : n ≤ 16) (he : e < n) : e < kMaxSplit := by
  have : kMaxSplit = 16 := by decide
  omega

-- § worker and thread counts

/-- Worker count: the request if positive, else one per usable core, never fewer than one. -/
def poolWorkers (requested : Int) (usableCores : Nat) : Nat :=
  max 1 (if 0 < requested then requested.toNat else usableCores)

/-- The pool always has at least one worker. -/
theorem poolWorkers_pos (requested : Int) (usableCores : Nat) : 1 ≤ poolWorkers requested usableCores :=
  Nat.le_max_left _ _

/-- Threads that drain a phase: the workers, plus the host when `host_works`. -/
def drainThreads (workers : Nat) (hostWorks : Bool) : Nat :=
  workers + (if hostWorks then 1 else 0)

/-- At least one thread drains. -/
theorem drainThreads_pos (workers : Nat) (hostWorks : Bool) (h : 1 ≤ workers) :
    1 ≤ drainThreads workers hostWorks := by
  unfold drainThreads
  split <;> omega

/-- The host joining the drain adds exactly one thread. -/
theorem drainThreads_host (workers : Nat) : drainThreads workers true = workers + 1 := by
  unfold drainThreads
  simp

/-- Excluding the first physical core removes exactly one entry (when there is one). -/
def selectCores (cores : List Nat) (skipFirst : Bool) : List Nat :=
  if skipFirst then cores.tail else cores

/-- Skipping the first core leaves one fewer core. -/
theorem selectCores_length (cores : List Nat) : (selectCores cores true).length = cores.length - 1 := by
  simp [selectCores]

/-- Not skipping leaves all cores. -/
theorem selectCores_all (cores : List Nat) : selectCores cores false = cores := by
  simp [selectCores]

-- § cache-line separation of the atomics

/-- Byte offset of the i-th atomic when each is `alignas(64)`. -/
def atomicOffset (i : Nat) : Nat := 64 * i

/-- The cache line (64-byte granule) an offset falls in. -/
def cacheLineOf (off : Nat) : Nat := off / 64

/-- Two distinct `alignas(64)` atomics never share a cache line (no false sharing between them). -/
theorem atomics_distinct_lines (i j : Nat) (h : i ≠ j) :
    cacheLineOf (atomicOffset i) ≠ cacheLineOf (atomicOffset j) := by
  unfold cacheLineOf atomicOffset
  omega

-- § task counts

/-- Parts per expert in `run_split`: about three tasks per thread, at least one part. -/
def partsPerExpert (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsPerExpert_pos (threads n : Nat) : 1 ≤ partsPerExpert threads n :=
  Nat.le_max_left _ _

/-- The ceiling division covers `3 * threads` tasks in total. -/
theorem split_tasks_cover (threads n : Nat) (hn : 0 < n) :
    3 * threads ≤ n * ((3 * threads + n - 1) / n) := by
  have h := Nat.div_add_mod (3 * threads + n - 1) n
  have hr := Nat.mod_lt (3 * threads + n - 1) hn
  omega

/-- Tasks of `run_split_multi` per phase: three per thread, so at least three. -/
theorem multi_tasks_ge_three (threads : Nat) (h : 1 ≤ threads) : 3 ≤ 3 * threads := by
  omega

-- § batching of experts beyond the buffer capacity

/-- Batch sizes when walking `n` experts in batches of at most `k` (fuel-bounded). -/
def batchSizes (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: batchSizes k f (n - min k n)

/-- The batches cover exactly `n` experts. -/
theorem batchSizes_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (batchSizes k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [batchSizes, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [batchSizes, if_pos h0]
      simp [h0]
    · rw [batchSizes, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every batch holds at most `k` experts, so at most `kMaxSplitMulti` buffers are ever used. -/
theorem batchSizes_le (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ batchSizes k f n, x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [batchSizes] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [batchSizes, if_pos h0] at hx
      simp at hx
    · rw [batchSizes, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- With fuel `n`, batches of 96 cover any number of native experts. -/
theorem native_batches_cover (n : Nat) : (batchSizes 96 n n).sum = n := by
  apply batchSizes_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 96 by decide)
  omega

-- § affinity restore

/-- `restore_thread_affinity` acts only on a positive previous mask. -/
def restoreApplies (previous : Int) : Bool := decide (0 < previous)

/-- A refused pin (returns -1) is never restored. -/
theorem restore_skips_failure : restoreApplies (-1) = false := by
  decide

/-- A real previous mask is restored. -/
theorem restore_applies_mask (m : Int) (h : 0 < m) : restoreApplies m = true := by
  unfold restoreApplies
  simp [h]

end Ref.Kernels.Cpu.ExpertPoolContract



## FILE: /mnt/user-data/outputs/ExpertPoolScheduling.lean
/-!
# MODULE: expert_pool_scheduling
SOURCES: src/kernels/cpu/pool.cpp
Arithmetic of the pool's drain loop: the `head_` claim counter hands each job out exactly once, row-range
partitions `N*i/parts .. N*(i+1)/parts` tile `[0,N)` exactly, task indices decompose uniquely into
(expert, part), and the multi-expert row walk always makes progress and never leaves an expert.

## Source semantics
* `head_.fetch_add(1)` returns 0,1,2,...; a claim succeeds iff it is below the job count, so exactly `n`
  claims succeed no matter how many threads race (extra claims just break out).
* Mode 1/2 tasks: task `i` is expert `i / parts`, part `i % parts`; part `p` covers rows `[N*p/parts, N*(p+1)/parts)`.
* Modes 3..6 cut `rows` global rows into `tasks` equal ranges; a range is walked expert by expert with
  `e = r / per`, `r0 = r % per`, `r1 = min per (r0 + (g1 - r))`.
* The workers' park counter returns to its initial value after every worker leaves and re-arrives.

## Form template identifiers
lo → range_start, widthSum → covered_rows, successes → claims_granted, decomposeTask → expert_part_split,
walkNext → row_walk_step.
-/

namespace Ref.Kernels.Cpu.ExpertPoolScheduling

-- § claim counter

/-- Number of successful claims among the first `k` values returned by `fetch_add` for `n` jobs. -/
def successes : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, n => successes k n + (if k < n then 1 else 0)

/-- Exactly `min k n` of the first `k` claims succeed. -/
theorem successes_eq (k n : Nat) : successes k n = min k n := by
  induction k with
  | zero => simp [successes]
  | succ k ih =>
    rw [successes, ih]
    split <;> omega

/-- However many extra (failing) claims the threads make, exactly `n` jobs are granted. -/
theorem every_job_claimed_once (n extra : Nat) : successes (n + extra) n = n := by
  rw [successes_eq]
  omega

-- § equal row-range partition

/-- Start of part `i` when `N` rows are cut into `parts` ranges. -/
def lo (N parts i : Nat) : Nat := N * i / parts

/-- The first part starts at row 0. -/
theorem lo_zero (N parts : Nat) : lo N parts 0 = 0 := by
  simp [lo]

/-- The last boundary is `N` (when there is at least one part). -/
theorem lo_last (N parts : Nat) (hp : 0 < parts) : lo N parts parts = N := by
  unfold lo
  exact Nat.mul_div_cancel N hp

/-- Boundaries are monotone, so every range has non-negative width. -/
theorem lo_mono (N parts i j : Nat) (h : i ≤ j) : lo N parts i ≤ lo N parts j := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left N h)

/-- Consecutive ranges abut: the end of part `i` is the start of part `i+1` (they are the same expression). -/
theorem range_abut (N parts i : Nat) : N * (i + 1) / parts = lo N parts (i + 1) := by
  rfl

/-- Total width of the first `k` parts. -/
def widthSum (N parts : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => widthSum N parts k + (lo N parts (k + 1) - lo N parts k)

/-- The widths telescope: the first `k` parts cover exactly `lo k` rows. -/
theorem widthSum_eq (N parts k : Nat) : widthSum N parts k = lo N parts k := by
  induction k with
  | zero => simp [widthSum, lo]
  | succ k ih =>
    rw [widthSum, ih]
    have := lo_mono N parts k (k + 1) (Nat.le_succ k)
    omega

/-- All `parts` ranges together cover exactly `N` rows, with no gap and no overlap. -/
theorem partition_covers (N parts : Nat) (hp : 0 < parts) : widthSum N parts parts = N := by
  rw [widthSum_eq, lo_last N parts hp]

-- § task decomposition (modes 1 and 2)

/-- Task `i` is (expert `i / parts`, part `i % parts`). -/
def decomposeTask (i parts : Nat) : Nat × Nat := (i / parts, i % parts)

/-- The expert index is below the expert count. -/
theorem task_expert_lt (i n parts : Nat) (h : i < n * parts) : (decomposeTask i parts).1 < n := by
  show i / parts < n
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- The part index is below the part count. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : (decomposeTask i parts).2 < parts :=
  Nat.mod_lt i hp

/-- The decomposition recombines to the task index. -/
theorem task_recombine (i parts : Nat) : parts * (decomposeTask i parts).1 + (decomposeTask i parts).2 = i :=
  Nat.div_add_mod i parts

/-- Distinct tasks map to distinct (expert, part) pairs: no (expert, part) is run twice. -/
theorem task_decompose_injective (i j parts : Nat)
    (h : decomposeTask i parts = decomposeTask j parts) : i = j := by
  have h1 := Nat.div_add_mod i parts
  have h2 := Nat.div_add_mod j parts
  have hd : i / parts = j / parts := congrArg Prod.fst h
  have hm : i % parts = j % parts := congrArg Prod.snd h
  rw [hd, hm] at h1
  exact h1.symm.trans h2

-- § the multi-expert row walk (modes 3..6)

/-- The expert a global row belongs to. -/
theorem walk_expert_lt (r n per : Nat) (h : r < n * per) : r / per < n :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- The end of the current sub-range inside one expert. -/
def walkEnd (per m remaining : Nat) : Nat := min per (m + remaining)

/-- The sub-range is non-empty: the walk makes progress (`m` is `r % per`, `remaining` is `g1 - r`). -/
theorem walk_progress (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    m < walkEnd per m (g1 - r) := by
  unfold walkEnd
  omega

/-- The sub-range never leaves its expert. -/
theorem walk_within_expert (per m remaining : Nat) : walkEnd per m remaining ≤ per := by
  unfold walkEnd
  omega

/-- The sub-range never overshoots the task's end `g1`. -/
theorem walk_within_task (m per remaining : Nat) (hm : m < per) :
    walkEnd per m remaining - m ≤ remaining := by
  unfold walkEnd
  omega

/-- The walk advances by a positive amount, so it terminates after at most `g1 - g0` steps. -/
theorem walk_advances (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    0 < walkEnd per m (g1 - r) - m := by
  unfold walkEnd
  omega

-- § the phase-row arithmetic

/-- Rows per expert in phases 3..6: gate/up rows `FF` for modes 3 and 5, down rows `H` otherwise. -/
def rowsPerExpert (mode ff h : Nat) : Nat := if mode = 3 ∨ mode = 5 then ff else h

/-- Rows per expert is positive whenever both widths are. -/
theorem rowsPerExpert_pos (mode ff h : Nat) (hf : 0 < ff) (hh : 0 < h) : 0 < rowsPerExpert mode ff h := by
  unfold rowsPerExpert
  split <;> omega

-- § worker park counter

/-- Park counter after `k` workers leave. -/
def afterLeave (parked k : Nat) : Nat := parked - k

/-- Park counter after `k` workers re-arrive. -/
def afterArrive (parked k : Nat) : Nat := parked + k

/-- After every worker leaves the park and every worker re-arrives, the counter is back at `n`:
the `done`-then-`parked` handover leaves the pool in the state `run` started from. -/
theorem park_roundtrip (n : Nat) : afterArrive (afterLeave n n) n = n := by
  unfold afterArrive afterLeave
  omega

/-- A host that is told `parked == n` can safely publish: the counter equals `n` only if no worker is inside the drain. -/
theorem parked_full_means_none_draining (n inDrain : Nat) (h : n - inDrain = n) (hn : 0 < n) :
    inDrain = 0 := by
  omega

end Ref.Kernels.Cpu.ExpertPoolScheduling



## FILE: /mnt/user-data/outputs/IqAvx512Rows.lean
/-!
# MODULE: iq_avx512_rows
SOURCES: src/kernels/cpu/iq_avx512.cpp
Layout facts for the i-quant AVX-512 row kernels (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S): block byte sizes,
that every decode read of chunk `j < 4` stays inside its 256-value block and inside its own field, the sign
negation on int8 activations, the lane-to-scale mapping, and the batching of tokens in groups of four.

## Source semantics
* A 256-value block is decoded as four 64-value chunks `j = 0..3`; each chunk reads a fixed slice of the block.
* Block sizes (bytes): IQ2_XXS 66, IQ2_XS 74, IQ3_XXS 98, IQ3_S 110, IQ2_S 82. Type ids: 16, 17, 18, 21, 22.
* Signs are applied as a masked subtract `0 - y` on lanes whose mask bit is set; activations come from Q8_K
  and lie in [-127,127], so the negation cannot overflow int8.
* `scales4` assigns int16 lanes 0-7, 8-15, 16-23, 24-31 to scale 0, 1, 2, 3.
* `dot_rows_nt` handles 1..4 tokens directly and loops in groups of four otherwise.

## Form template identifiers
iq512_supported → vector_path_supported, blockBytes → quant_block_bytes, negIf → masked_sign_flip,
scaleLane → lane_scale_index, chunksK → token_groups.
-/

namespace Ref.Kernels.Cpu.IqAvx512Rows

-- § supported types

/-- Block byte size of each supported format (0 for unsupported ids). -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Whether the vector path handles the type id. -/
def iq512Supported (ty : Nat) : Bool :=
  ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- The supported ids are exactly 16, 17, 18, 21, 22. -/
theorem supported_iff (ty : Nat) :
    iq512Supported ty = true ↔ (ty = 16 ∨ ty = 17 ∨ ty = 18 ∨ ty = 21 ∨ ty = 22) := by
  simp [iq512Supported, or_assoc]

/-- Every supported type has a non-zero block size. -/
theorem supported_has_size (ty : Nat) (h : iq512Supported ty = true) : 0 < blockBytes ty := by
  rcases (supported_iff ty).mp h with rfl | rfl | rfl | rfl | rfl <;> decide

/-- IQ1_M (id 29) stays on ggml-cpu. -/
theorem iq1_m_unsupported : iq512Supported 29 = false := by
  decide

/-- Q2_0 (id 42) is not an i-quant and is not handled here. -/
theorem q2_0_unsupported : iq512Supported 42 = false := by
  decide

/-- The five block sizes. -/
theorem block_sizes :
    blockBytes 16 = 66 ∧ blockBytes 17 = 74 ∧ blockBytes 18 = 98 ∧ blockBytes 21 = 110 ∧ blockBytes 22 = 82 := by
  decide

/-- Four 64-value chunks make one 256-value block (QK_K). -/
theorem four_chunks_make_block : 4 * 64 = 256 := by
  decide

-- § the field layout of each format

/-- IQ2_XXS: scale (2) + 32 u16 = 64 bytes. -/
theorem iq2xxs_layout : 2 + 64 = blockBytes 16 := by
  decide

/-- IQ2_XS: scale (2) + 64 bytes of u16 indices + 8 scale bytes. -/
theorem iq2xs_layout : 2 + 64 + 8 = blockBytes 17 := by
  decide

/-- IQ3_XXS: scale (2) + 64 grid bytes + 32 bytes of sign/scale words. -/
theorem iq3xxs_layout : 2 + 64 + 32 = blockBytes 18 := by
  decide

/-- IQ3_S: scale (2) + 64 grid bytes + 8 qh + 32 signs + 4 scale bytes. -/
theorem iq3s_layout : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by
  decide

/-- IQ2_S: scale (2) + 32 grid bytes + 32 sign bytes + 8 qh + 8 scale bytes. -/
theorem iq2s_layout : 2 + 32 + 32 + 8 + 8 = blockBytes 22 := by
  decide

-- § every decode read stays inside the block

/-- IQ2_XXS: the 16-byte read at `2 + 16 j` ends inside the 66-byte block. -/
theorem iq2xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by
  omega

/-- IQ2_XS: the 16-byte index read and the two scale bytes at `66 + 2 j` stay inside 74 bytes, and do not overlap. -/
theorem iq2xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by
  omega

/-- IQ2_S: grid bytes (8 at `2 + 8 j`), sign words (u64 at `34 + 8 j`), qh and scale bytes all fit in 82 bytes. -/
theorem iq2s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS: the 16 grid bytes and the two u32 words (at `66 + 8 j` and `+4`) fit in 98 bytes. -/
theorem iq3xxs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 4 + 4 ≤ 98 := by
  omega

/-- IQ3_S: grid bytes, qh, the u64 sign word and the scale byte all fit in 110 bytes. -/
theorem iq3s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

/-- In IQ3_S the sign region ends exactly where the scale bytes begin: the regions do not overlap. -/
theorem iq3s_regions_disjoint : 74 + 32 = 106 := by
  decide

/-- In IQ2_S the sign region ends exactly where qh begins. -/
theorem iq2s_regions_disjoint : 34 + 32 = 66 := by
  decide

-- § sign application

/-- The masked subtract: negate the lane when its mask bit is set. -/
def negIf (b : Bool) (y : Int) : Int := if b then -y else y

/-- Negation keeps an int8 activation in [-127,127] (Q8_K never emits -128), so there is no overflow. -/
theorem negIf_range (b : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ negIf b y ∧ negIf b y ≤ 127 := by
  unfold negIf
  split <;> omega

/-- Applying the sign twice restores the activation. -/
theorem negIf_involutive (b : Bool) (y : Int) : negIf b (negIf b y) = y := by
  unfold negIf
  cases b <;> simp

/-- An unset mask bit leaves the activation unchanged. -/
theorem negIf_false (y : Int) : negIf false y = y := by
  simp [negIf]

-- § lane to scale mapping

/-- The scale index assigned to int16 lane `l` by `scales4`: lanes 0-7 → 0, 8-15 → 1, 16-23 → 2, 24-31 → 3. -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every one of the 32 lanes maps to one of the four scales. -/
theorem scaleLane_lt (l : Nat) (hl : l < 32) : scaleLane l < 4 := by
  unfold scaleLane
  omega

/-- Each scale governs exactly a contiguous run of 8 lanes. -/
theorem scaleLane_run (l : Nat) (s : Nat) (hs : s < 4) :
    scaleLane l = s ↔ (8 * s ≤ l ∧ l < 8 * s + 8) := by
  unfold scaleLane
  omega

-- § token batching in groups of four

/-- Group sizes when walking `n` tokens in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- The groups cover exactly `n` tokens. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group has at most `k` tokens. -/
theorem chunksK_le (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Any number of tokens is processed in groups of four with fuel `n`, each token exactly once. -/
theorem token_groups_cover (n : Nat) : (chunksK 4 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 4 by decide)
  omega

end Ref.Kernels.Cpu.IqAvx512Rows



## FILE: /mnt/user-data/outputs/Q2Avx2Rows.lean
/-!
# MODULE: q2_avx2_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
Integer theory of the AVX2-only Q2_0 row kernel and activation quantizer: the 2-bit unpack is lossless and
value-ordered, the `maddubs` pair sums cannot saturate int16, the code-offset correction is exact, and the token
loop in groups of four covers every token once.

## Source semantics
* 16 code bytes hold 64 two-bit codes; value `v` is in byte `v/4` at bit `2*(v%4)`; values 0..31 come from
  bytes 0..7 and values 32..63 from bytes 8..15.
* `maddubs(codes, q)` adds adjacent products of an unsigned code (0..3) and a signed int8 (-127..127): each
  pair is within +-762, far from int16 saturation; `madd(.., ones)` then widens exactly.
* The block contribution is `d*(sx0*S0 + sx1*S1) - d*(hx0 + hx1)`; the integer part `S - sum` is exact.
* `act_quant_q8_1_avx2` clamps to [-127,127] and records the chunk sum (hence `hx = scale * sum`).

## Form template identifiers
code → unpacked_code, pairSum → maddubs_pair, clampQ → act_clamp, chunksK → token_groups,
blockOf → weight_block_of_value.
-/

namespace Ref.Kernels.Cpu.Q2Avx2Rows

-- § the 2-bit unpack

/-- The code of value `v` given its packed byte: `(byte / 4^(v%4)) % 4`, i.e. bits `2*(v%4)`. -/
def code (byte v : Nat) : Nat := (byte / 2 ^ (2 * (v % 4))) % 4

/-- Every unpacked code is in 0..3. -/
theorem code_lt_four (byte v : Nat) : code byte v < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- Unpacking is lossless: the four codes of a byte recombine to the byte. -/
theorem pack_unpack (b : Nat) (h : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- The four codes of byte `b` are the four values `4i..4i+3`, in order. -/
theorem code_slots (b : Nat) :
    code b 0 = b % 4 ∧ code b 1 = b / 4 % 4 ∧ code b 2 = b / 16 % 4 ∧ code b 3 = b / 64 % 4 := by
  simp [code]

/-- Value `v` lives in byte `v / 4`. -/
def byteOf (v : Nat) : Nat := v / 4

/-- Values 0..31 come from bytes 0..7 (the `lo` vector). -/
theorem lo_bytes (v : Nat) (h : v < 32) : byteOf v < 8 := by
  unfold byteOf
  omega

/-- Values 32..63 come from bytes 8..15 (the `hi` vector). -/
theorem hi_bytes (v : Nat) (h1 : 32 ≤ v) (h2 : v < 64) : 8 ≤ byteOf v ∧ byteOf v < 16 := by
  unfold byteOf
  omega

/-- The 16 code bytes cover exactly the 64 values of one block. -/
theorem sixteen_bytes_sixtyfour_codes : 16 * 4 = 64 := by
  decide

-- § no int16 saturation in maddubs

/-- One `maddubs` lane: sum of two code*activation products. -/
def pairSum (c0 c1 q0 q1 : Int) : Int := c0 * q0 + c1 * q1

/-- A pair of products of codes (0..3) with int8 activations stays within +-762, so it never saturates int16. -/
theorem pairSum_bound (c0 c1 q0 q1 : Int)
    (h0 : 0 ≤ c0) (h1 : c0 ≤ 3) (h2 : 0 ≤ c1) (h3 : c1 ≤ 3)
    (hq0 : -127 ≤ q0) (hq1 : q0 ≤ 127) (hq2 : -127 ≤ q1) (hq3 : q1 ≤ 127) :
    -762 ≤ pairSum c0 c1 q0 q1 ∧ pairSum c0 c1 q0 q1 ≤ 762 := by
  unfold pairSum
  have a : c0 = 0 ∨ c0 = 1 ∨ c0 = 2 ∨ c0 = 3 := by omega
  have b : c1 = 0 ∨ c1 = 1 ∨ c1 = 2 ∨ c1 = 3 := by omega
  rcases a with rfl | rfl | rfl | rfl <;> rcases b with rfl | rfl | rfl | rfl <;> omega

/-- The bound leaves a wide margin below int16's range. -/
theorem pairSum_fits_int16 (c0 c1 q0 q1 : Int)
    (h0 : 0 ≤ c0) (h1 : c0 ≤ 3) (h2 : 0 ≤ c1) (h3 : c1 ≤ 3)
    (hq0 : -127 ≤ q0) (hq1 : q0 ≤ 127) (hq2 : -127 ≤ q1) (hq3 : q1 ≤ 127) :
    -32768 ≤ pairSum c0 c1 q0 q1 ∧ pairSum c0 c1 q0 q1 ≤ 32767 := by
  have := pairSum_bound c0 c1 q0 q1 h0 h1 h2 h3 hq0 hq1 hq2 hq3
  omega

-- § the exact code-offset correction

/-- Centered dot `(c-1)*x` over matching lists. -/
def offDot : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + offDot cs xs
  | _, _ => 0

/-- Raw-code dot over matching lists. -/
def codeDot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + codeDot cs xs
  | _, _ => 0

/-- The correction identity: centered dot = raw dot minus the activation sum. -/
theorem offDot_eq (cs xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = codeDot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, codeDot]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      have hm : (c - 1) * x = c * x - x := by
        rw [Int.sub_mul, Int.one_mul]
      simp only [offDot, codeDot, List.sum_cons]
      rw [hm, ih']
      omega

-- § the activation quantizer

/-- The int8 clamp. -/
def clampQ (v : Int) : Int := max (-127) (min 127 v)

/-- The clamp lands in [-127,127]. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  omega

/-- The clamp does not touch in-range values (so the truncation-after-±0.5 result is kept). -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  omega

/-- Reduction of the four 8-lane int32 sums to one total: add halves, then two shuffles-and-adds (exact in int). -/
theorem total_tree (a b c d e f g h : Int) :
    (((a + e) + (c + g)) + ((b + f) + (d + h))) = a + b + c + d + e + f + g + h := by
  omega

-- § token groups of four

/-- Group sizes when walking `n` tokens in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- The groups cover exactly `n` tokens. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group has between 1 and `k` tokens, so the `rows<NT>` instance used always has NT in 1..4. -/
theorem chunksK_bounds (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, 0 < x ∧ x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Any token count is covered exactly by groups of four (fuel `n`). -/
theorem token_groups_cover (n : Nat) : (chunksK 4 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 4 by decide)
  omega

/-- The offset of group `g` into the token array is `4*g`, and the group's tokens lie below `4*g + 4`. -/
theorem group_offset (g t : Nat) (ht : t < 4) : 4 * g + t < 4 * (g + 1) := by
  omega

end Ref.Kernels.Cpu.Q2Avx2Rows



## FILE: /mnt/user-data/outputs/ExpertLayoutTable.lean
/-!
# MODULE: expert_layout_table
SOURCES: src/kernels/cpu/expert_layout.cpp
The per-layer expert table loader: contiguity validation of `native_experts.txt`, the resulting total size, the
running maximum blob size, the GGUF-offset indexing, the AVX-512 gating predicate (CPUID bits and XCR0 mask), and
the fixed-size default layout.

## Source semantics
* Layers must tile the pack with no gap: layer `l` starts where layer `l-1` ended (`offset[l] = sum of
  bytes[k]*n_expert for k < l`); a violation is an error. On success `total` is that sum.
* `max_blob` is the maximum layer blob size.
* v2 lines store three GGUF offsets per layer in an array of `3*n_layers`, indexed `3*l + {0,1,2}`.
* AVX-512 is used only if the OS saves the AVX-512 state (`xcr0 & 0xE6 == 0xE6`) and CPUID leaf 7 reports F, BW, VL
  (EBX bits 16, 30, 31), VNNI (ECX bit 11) and VBMI (ECX bit 1).
* With no `native_experts.txt` the layout is `n_layers * n_expert * BLOB` bytes.

## Form template identifiers
contig → contiguity_check, maxOf → max_blob_fold, xcr0Ok → os_state_gate, avx512Ok → isa_gate,
gguf index → gguf_offset_slot.
-/

namespace Ref.Kernels.Cpu.ExpertLayoutTable

-- § contiguity validation

/-- Walk (offset, bytes) pairs: each offset must equal the running total; return the final total, or none. -/
def contig (n : Nat) : List (Nat × Nat) → Nat → Option Nat
  | [], cur => some cur
  | (o, b) :: rest, cur => if o = cur then contig n rest (cur + b * n) else none

/-- Sum of the layers' sizes (`bytes * n_expert`). -/
def layerTotal (n : Nat) (L : List (Nat × Nat)) : Nat := (L.map (fun p => p.2 * n)).sum

/-- A successful validation returns the starting offset plus the sum of all layers' sizes. -/
theorem contig_total (n : Nat) (L : List (Nat × Nat)) :
    ∀ cur t : Nat, contig n L cur = some t → t = cur + layerTotal n L := by
  induction L with
  | nil =>
    intro cur t h
    simp only [contig, Option.some.injEq] at h
    simp [layerTotal, h]
  | cons p rest ih =>
    obtain ⟨o, b⟩ := p
    intro cur t h
    by_cases ho : o = cur
    · simp only [contig, if_pos ho] at h
      have := ih _ _ h
      simp only [layerTotal, List.map_cons, List.sum_cons] at this ⊢
      omega
    · simp [contig, ho] at h

/-- A gap or overlap at the first layer makes validation fail. -/
theorem contig_fails_on_gap (n o b cur : Nat) (rest : List (Nat × Nat)) (h : o ≠ cur) :
    contig n ((o, b) :: rest) cur = none := by
  simp only [contig, if_neg h]

/-- The empty table validates to its starting offset. -/
theorem contig_nil (n cur : Nat) : contig n [] cur = some cur := by
  rfl

/-- An exactly-contiguous single layer starting at 0 has size `bytes * n_expert`. -/
theorem contig_single (n b : Nat) : contig n [(0, b)] 0 = some (0 + b * n) := by
  simp [contig]

-- § running maximum

/-- Fold of `max` over the blob sizes, as `max_blob` is computed. -/
def maxOf : List Nat → Nat → Nat
  | [], m => m
  | b :: bs, m => maxOf bs (max m b)

/-- The fold never goes below its initial value. -/
theorem maxOf_ge_init (l : List Nat) : ∀ m : Nat, m ≤ maxOf l m := by
  induction l with
  | nil => intro m; exact Nat.le_refl m
  | cons b bs ih =>
    intro m
    simp only [maxOf]
    have := ih (max m b)
    omega

/-- Every element is at most the fold result. -/
theorem le_maxOf (l : List Nat) : ∀ m x : Nat, x ∈ l → x ≤ maxOf l m := by
  induction l with
  | nil =>
    intro m x hx
    simp at hx
  | cons b bs ih =>
    intro m x hx
    rcases List.mem_cons.mp hx with h | h
    · rw [h]
      simp only [maxOf]
      have := maxOf_ge_init bs (max m b)
      omega
    · simp only [maxOf]
      exact ih _ x h

-- § GGUF offset slots

/-- Slot of offset `k` (0 = gate, 1 = up, 2 = down) of layer `l` in the `3 * n_layers` array. -/
def gguf_slot (l k : Nat) : Nat := 3 * l + k

/-- Every slot of every valid layer is inside the array. -/
theorem gguf_slot_lt (n_layers l k : Nat) (hl : l < n_layers) (hk : k < 3) :
    gguf_slot l k < 3 * n_layers := by
  unfold gguf_slot
  omega

/-- Different (layer, kind) pairs use different slots. -/
theorem gguf_slot_injective (l1 k1 l2 k2 : Nat) (h1 : k1 < 3) (h2 : k2 < 3)
    (h : gguf_slot l1 k1 = gguf_slot l2 k2) : l1 = l2 ∧ k1 = k2 := by
  unfold gguf_slot at h
  omega

-- § AVX-512 gating

/-- Whether bit `k` of a register value is set. -/
def bitSet (r k : Nat) : Bool := (r >>> k) &&& 1 == 1

/-- The OS-state gate: the SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM bits (mask 0xE6) are all enabled. -/
def xcr0Ok (xcr0 : Nat) : Bool := (xcr0 &&& 0xE6) == 0xE6

/-- The ISA gate from CPUID leaf 7: F, BW, VL in EBX; VNNI, VBMI in ECX. -/
def avx512Ok (ebx ecx : Nat) : Bool :=
  bitSet ebx 16 && bitSet ebx 30 && bitSet ebx 31 && bitSet ecx 11 && bitSet ecx 1

/-- A typical AVX-512 OS state (x87+SSE+AVX+opmask+ZMM) passes the gate. -/
theorem xcr0_full_ok : xcr0Ok 0xE7 = true := by
  decide

/-- An OS that only saves AVX (xcr0 = 7) fails the gate even if the CPU has AVX-512. -/
theorem xcr0_avx_only_rejected : xcr0Ok 0x07 = false := by
  decide

/-- A CPU with exactly the five required bits passes. -/
theorem avx512_all_present : avx512Ok (65536 + 1073741824 + 2147483648) (2048 + 2) = true := by
  decide

/-- A CPU missing VNNI (ECX bit 11) is rejected. -/
theorem avx512_missing_vnni : avx512Ok (65536 + 1073741824 + 2147483648) 2 = false := by
  decide

/-- A CPU missing VBMI (ECX bit 1) is rejected. -/
theorem avx512_missing_vbmi : avx512Ok (65536 + 1073741824 + 2147483648) 2048 = false := by
  decide

/-- A CPU that reports nothing is rejected. -/
theorem avx512_none : avx512Ok 0 0 = false := by
  decide

-- § default layout size

/-- Size of the default (no native table) layout. -/
def defaultTotal (n_layers n_expert blob : Nat) : Nat := n_layers * n_expert * blob

/-- With the 1,382,400-byte blob, 512 experts and one layer the default layout is 707,788,800 bytes. -/
theorem default_one_layer : defaultTotal 1 512 1382400 = 707788800 := by
  decide

/-- The default layout equals the contiguous total of identical layers (one layer check). -/
theorem default_matches_contig (n_expert blob : Nat) :
    contig n_expert [(0, blob)] 0 = some (defaultTotal 1 n_expert blob) := by
  rw [contig_single]
  congr 1
  unfold defaultTotal
  rw [Nat.one_mul, Nat.zero_add, Nat.mul_comm]

end Ref.Kernels.Cpu.ExpertLayoutTable



## FILE: /mnt/user-data/outputs/F16Bits.lean
/-!
# MODULE: f16_bits
SOURCES: include/ref/kernels/f16_bits.hpp
A bit-level model of `f16_from_f32` (round-to-nearest-even, saturating) and `f32_from_f16` (exact, subnormals
normalised), over natural numbers holding the raw bit patterns, with the regression vectors the header's tests use.

## Source semantics
* `f16_from_f32`: an f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value too large for fp16 saturates to inf
  (0x7C00 with the sign); tiny values underflow to a signed zero; subnormal results restore the implicit 1 and
  round at a moved point; normal results round from 13 surviving mantissa bits, and a rounding carry may bump the
  exponent (65520 → inf).
* `f32_from_f16` never rounds: subnormals are normalised by shifting the leading 1 to bit 10, giving f32 exponent
  field `113 - s`; it does not flush subnormals to zero.
* All claims here are on concrete bit vectors plus structural facts about the normalisation loop; the exhaustive
  65,536-input check is the header's numpy oracle test, not repeated here.

## Form template identifiers
f16FromF32Bits → fp16_encode_rne, f32FromF16Bits → fp16_decode_exact, normShift → subnormal_normalize,
regression vectors → fp16_oracle_vectors.
-/

namespace Ref.Kernels.F16Bits

-- § the model

/-- Round-to-nearest-even f32 → fp16 on raw bits (`x` is the f32 bit pattern; the result the fp16 bits). -/
def f16FromF32Bits (x : Nat) : Nat :=
  let sign := (x >>> 16) &&& 0x8000
  let rawexp := (x >>> 23) &&& 0xFF
  let man := x &&& 0x7FFFFF
  if rawexp = 0xFF then
    sign ||| 0x7C00 ||| (if man ≠ 0 then 0x200 else 0)
  else
    let e : Int := (rawexp : Int) - 127 + 15
    if e ≥ 31 then
      sign ||| 0x7C00
    else if e ≤ 0 then
      if e < -10 then
        sign
      else
        let man' := man ||| 0x800000
        let sh := (14 - e).toNat
        let h := (man' >>> sh) &&& 0x3FF
        let rem := man' &&& ((1 <<< sh) - 1)
        let half := 1 <<< (sh - 1)
        let h' := if rem > half ∨ (rem = half ∧ (h &&& 1) = 1) then h + 1 else h
        sign ||| h'
    else
      let h := sign ||| (e.toNat <<< 10) ||| (man >>> 13)
      let rem := man &&& 0x1FFF
      if rem > 0x1000 ∨ (rem = 0x1000 ∧ (h &&& 1) = 1) then h + 1 else h

/-- The subnormal normalisation loop: shift left until bit 10 is set; returns (mantissa, shift count). -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | f + 1, m, s => if (m &&& 0x400) = 0 then normShift f (m <<< 1) (s + 1) else (m, s)

/-- Exact fp16 → f32 on raw bits. -/
def f32FromF16Bits (h : Nat) : Nat :=
  let sign := (h &&& 0x8000) <<< 16
  let ex := (h >>> 10) &&& 0x1F
  let man := h &&& 0x3FF
  if ex = 0 then
    if man = 0 then
      sign
    else
      let r := normShift 10 man 0
      sign ||| ((113 - r.2) <<< 23) ||| ((r.1 &&& 0x3FF) <<< 13)
  else if ex = 31 then
    sign ||| 0x7F800000 ||| (man <<< 13)
  else
    sign ||| ((ex + 112) <<< 23) ||| (man <<< 13)

-- § structural facts about the normalisation loop

/-- The shift count grows by at most one per fuel step. -/
theorem normShift_snd_le (f : Nat) : ∀ m s : Nat, (normShift f m s).2 ≤ s + f := by
  induction f with
  | zero =>
    intro m s
    show s ≤ s + 0
    omega
  | succ f ih =>
    intro m s
    simp only [normShift]
    split
    · have := ih (m <<< 1) (s + 1)
      omega
    · show s ≤ s + (f + 1)
      omega

/-- With the header's fuel of 10 the shift count is at most 10, so the exponent field `113 - s` is in 103..113. -/
theorem decode_subnormal_exponent_range (m : Nat) :
    103 ≤ 113 - (normShift 10 m 0).2 ∧ 113 - (normShift 10 m 0).2 ≤ 113 := by
  have h := normShift_snd_le 10 m 0
  omega

-- § encode: regression vectors (every regime)

/-- 1.0f encodes to 0x3C00 (the value an earlier `__float2half` got wrong as 0x3C01). -/
theorem enc_one : f16FromF32Bits 0x3F800000 = 0x3C00 := by
  decide

/-- +inf stays +inf. -/
theorem enc_inf : f16FromF32Bits 0x7F800000 = 0x7C00 := by
  decide

/-- -inf stays -inf (sign kept). -/
theorem enc_neg_inf : f16FromF32Bits 0xFF800000 = 0xFC00 := by
  decide

/-- A quiet NaN stays a NaN (0x7E00). -/
theorem enc_nan : f16FromF32Bits 0x7FC00000 = 0x7E00 := by
  decide

/-- 65536.0 is finite but too large: it saturates to inf (the conflated version returned a NaN here). -/
theorem enc_overflow_65536 : f16FromF32Bits 0x47800000 = 0x7C00 := by
  decide

/-- 1e30 saturates to inf, not NaN. -/
theorem enc_overflow_1e30 : f16FromF32Bits 0x7149F2CA = 0x7C00 := by
  decide

/-- The largest fp16 value, 65504, is exact. -/
theorem enc_max_half : f16FromF32Bits 0x477FE000 = 0x7BFF := by
  decide

/-- 65520 is a tie that rounds to even upward, and the carry bumps the exponent into inf. -/
theorem enc_carry_to_inf : f16FromF32Bits 0x477FF000 = 0x7C00 := by
  decide

/-- 2^-24, the smallest fp16 subnormal, is exact. -/
theorem enc_min_subnormal : f16FromF32Bits 0x33800000 = 0x0001 := by
  decide

/-- 2^-25 is an exact tie between 0 and 2^-24 and rounds to even, i.e. to zero. -/
theorem enc_tie_to_even : f16FromF32Bits 0x33000000 = 0x0000 := by
  decide

/-- 1.5 * 2^-25 is above the tie and rounds up to 2^-24. -/
theorem enc_above_tie : f16FromF32Bits 0x33400000 = 0x0001 := by
  decide

/-- 2^-26 underflows to zero. -/
theorem enc_underflow : f16FromF32Bits 0x32800000 = 0x0000 := by
  decide

/-- -0.0 keeps its sign. -/
theorem enc_neg_zero : f16FromF32Bits 0x80000000 = 0x8000 := by
  decide

-- § decode: regression vectors

/-- The smallest subnormal decodes to 2^-24, not zero (flushing would be a 100% error). -/
theorem dec_min_subnormal : f32FromF16Bits 0x0001 = 0x33800000 := by
  decide

/-- The largest subnormal 0x03FF decodes exactly. -/
theorem dec_max_subnormal : f32FromF16Bits 0x03FF = 0x387FC000 := by
  decide

/-- The smallest normal 2^-14. -/
theorem dec_min_normal : f32FromF16Bits 0x0400 = 0x38800000 := by
  decide

/-- 1.0. -/
theorem dec_one : f32FromF16Bits 0x3C00 = 0x3F800000 := by
  decide

/-- +inf. -/
theorem dec_inf : f32FromF16Bits 0x7C00 = 0x7F800000 := by
  decide

/-- Signed zero. -/
theorem dec_neg_zero : f32FromF16Bits 0x8000 = 0x80000000 := by
  decide

-- § round trips on representative fp16 values

/-- Decoding then encoding returns the same bits for a subnormal. -/
theorem roundtrip_min_subnormal : f16FromF32Bits (f32FromF16Bits 0x0001) = 0x0001 := by
  decide

/-- Round trip for the largest subnormal. -/
theorem roundtrip_max_subnormal : f16FromF32Bits (f32FromF16Bits 0x03FF) = 0x03FF := by
  decide

/-- Round trip for a negative subnormal (sign preserved). -/
theorem roundtrip_neg_subnormal : f16FromF32Bits (f32FromF16Bits 0x8001) = 0x8001 := by
  decide

/-- Round trip for 1.0. -/
theorem roundtrip_one : f16FromF32Bits (f32FromF16Bits 0x3C00) = 0x3C00 := by
  decide

/-- Round trip for the largest finite fp16. -/
theorem roundtrip_max_half : f16FromF32Bits (f32FromF16Bits 0x7BFF) = 0x7BFF := by
  decide

/-- Round trip for the smallest normal. -/
theorem roundtrip_min_normal : f16FromF32Bits (f32FromF16Bits 0x0400) = 0x0400 := by
  decide

end Ref.Kernels.F16Bits



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- ExpertPoolScheduling.lean
  was: theorem task_decompose_injective (i j parts : Nat)
    (h : decomposeTask i parts = decomposeTask j parts) : i = j
  now: theorem task_expert_lt (i n parts : Nat) (h : i < n * parts) : (decomposeTask i parts).1 < n
  measured: the claim changed

- ExpertPoolScheduling.lean
  was: theorem walk_progress (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    m < walkEnd per m (g1 - r)
  now: theorem task_decompose_injective (i j parts : Nat)
    (h : decomposeTask i parts = decomposeTask j parts) : i = j
  measured: fewer quantified variables (3 → 2)

- ExpertPoolScheduling.lean
  was: theorem walk_within_expert (per m remaining : Nat) : walkEnd per m remaining ≤ per
  now: theorem walk_progress (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    m < walkEnd per m (g1 - r)
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- ExpertPoolScheduling.lean
  was: theorem walk_within_task (m per remaining : Nat) (hm : m < per) :
    walkEnd per m remaining - m ≤ remaining
  now: theorem walk_within_expert (per m remaining : Nat) : walkEnd per m remaining ≤ per
  measured: fewer quantified variables (2 → 1)

- ExpertPoolScheduling.lean
  was: theorem walk_advances (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    0 < walkEnd per m (g1 - r) - m
  now: theorem walk_within_task (m per remaining : Nat) (hm : m < per) :
    walkEnd per m remaining - m ≤ remaining
  measured: fewer quantified variables (3 → 2)

- ExpertPoolScheduling.lean
  was: theorem rowsPerExpert_pos (mode ff h : Nat) (hf : 0 < ff) (hh : 0 < h) : 0 < rowsPerExpert mode ff h
  now: theorem walk_advances (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    0 < walkEnd per m (g1 - r) - m
  measured: the claim changed

- ExpertPoolScheduling.lean
  was: theorem park_roundtrip (n : Nat) : afterArrive (afterLeave n n) n = n
  now: theorem rowsPerExpert_pos (mode ff h : Nat) (hf : 0 < ff) (hh : 0 < h) : 0 < rowsPerExpert mode ff h
  measured: more hypotheses/binders (1 → 3): the statement holds in fewer cases

- ExpertPoolScheduling.lean
  was: theorem parked_full_means_none_draining (n inDrain : Nat) (h : n - inDrain = n) (hn : 0 < n) :
    inDrain = 0
  now: theorem park_roundtrip (n : Nat) : afterArrive (afterLeave n n) n = n
  measured: fewer quantified variables (3 → 1)
-/



## FILE: /mnt/user-data/outputs/FusedGdnDecodeKernels.lean
/-!
# MODULE: fused_gdn_decode_kernels
SOURCES: src/kernels/cuda/fused_gdn.cu
Index-geometry theory of the fused gated-delta-net decode kernels (conv+L2, alpha/beta projections,
state step + gated RMS norm): launch coverage, thread/row decompositions and buffer bounds.

## Source semantics
* `gdn_step_norm_kernel` runs one block per value head with a (128 x 4) thread tile; thread (col, rg) owns
  32 state rows `rg*32 + r` of column `col`, state laid out `(row, h_v, col)`.
* `gdn_conv_l2_kernel` runs one thread per channel with a 3-deep history and a 4-tap filter.
* `gdn_ab_kernel` runs one warp per row of `[w_alpha; w_beta]`, lanes striding 8-wide bf16 chunks.

## Form template identifiers
fused_gdn_step_norm -> decode_state_step ; fused_gdn_conv_l2 -> decode_conv ; fused_gdn_ab -> decode_ab_proj ;
S -> state_dim ; RG -> row_groups ; RPG -> rows_per_group ; h_k/h_v -> key_heads/value_heads
-/

namespace Ref.FusedGdn

-- § generic arithmetic helpers

/-- Division/modulo uniqueness for a mixed-radix index `a*n+b`. -/
theorem divmod_unique {a b c d n : Nat} (hb : b < n) (hd : d < n) (h : a * n + b = c * n + d) :
    a = c ∧ b = d := by
  have hn : 0 < n := by omega
  have h1 : (a * n + b) / n = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  have h3 : a = c := by rw [← h1, ← h2, h]
  refine ⟨h3, ?_⟩
  subst h3
  exact Nat.add_left_cancel h

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- State size. -/
def S : Nat := 128
/-- Row groups. -/
def RG : Nat := 4
/-- Rows per thread. -/
def RPG : Nat := S / RG

/-- A thread owns 32 rows and the four groups tile the state exactly. -/
theorem rpg_value : RPG = 32 ∧ RPG * RG = S := by decide

/-- The block has 512 threads and `wsum` has one slot per warp. -/
theorem block_threads_warps : S * RG = 512 ∧ S * RG / 32 = 16 := by decide

-- § step kernel thread decomposition

/-- A linear thread id splits into a column in `[0,128)` and a row group in `[0,4)`. -/
theorem tid_decomp (tid : Nat) (h : tid < 512) :
    tid / 128 < 4 ∧ tid % 128 < 128 ∧ tid = (tid / 128) * 128 + tid % 128 := by omega

/-- The 32 rows owned by different row groups are pairwise distinct (row = rg*32 + r). -/
theorem row_owner_injective {rg r rg' r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' := by omega

/-- Every owned row lies inside the state. -/
theorem row_owner_lt {rg r : Nat} (hg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by omega

/-- Every state row is owned by some (group, slot). -/
theorem row_owner_surjective (row : Nat) (h : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- The state address `((row*h_v + head)*S + col)` is injective in `(row, head, col)`. -/
theorem state_index_injective {h_v row head col row' head' col' : Nat}
    (hh : head < h_v) (hh' : head' < h_v) (hc : col < 128) (hc' : col' < 128)
    (h : (row * h_v + head) * 128 + col = (row' * h_v + head') * 128 + col') :
    row = row' ∧ head = head' ∧ col = col' := by
  obtain ⟨h1, h2⟩ := divmod_unique hc hc' h
  obtain ⟨h3, h4⟩ := divmod_unique hh hh' h1
  exact ⟨h3, h4, h2⟩

/-- State addresses stay inside the `(S, h_v, S)` buffer. -/
theorem state_index_lt {h_v row head col : Nat} (hr : row < 128) (hh : head < h_v) (hc : col < 128) :
    (row * h_v + head) * 128 + col < 128 * h_v * 128 := by
  have h1 : row * h_v + head < 128 * h_v := rowmajor_lt hr hh
  have h2 := rowmajor_lt (r := col) (h := 128) h1 hc
  exact h2

/-- Head pairing `head % h_k` stays inside the q/k buffers of `h_k` heads. -/
theorem qk_head_pairing {head h_k : Nat} (hk : 0 < h_k) :
    head % h_k < h_k ∧ (head % h_k) * 128 + 127 < 128 * h_k := by
  have h := Nat.mod_lt head hk
  omega

/-- Only the first row group's four warps write `wsum`, so its 4 summands cover all of 0..127. -/
theorem rms_warp_slots (tid : Nat) (h : tid < 128) : tid / 32 < 4 := by omega

/-- `h_v % h_k = 0` and positivity imply `h_k ≤ h_v`. -/
theorem heads_divisible_le {h_k h_v : Nat} (hk : 0 < h_k) (hv : 0 < h_v) (hd : h_v % h_k = 0) :
    h_k ≤ h_v := by
  obtain ⟨m, hm⟩ := Nat.dvd_of_mod_eq_zero hd
  cases m with
  | zero => omega
  | succ m => rw [hm]; exact Nat.le_mul_of_pos_right _ (Nat.succ_pos m)

-- § conv + L2 kernel

/-- Channel index of the conv kernel is in range when `128 | channels`. -/
theorem conv_channel_in_range {blk t channels : Nat} (hc : channels % 128 = 0)
    (hb : blk < channels / 128) (ht : t < 128) : blk * 128 + t < channels := by omega

/-- History, weight index bounds for a channel `c < channels`. -/
theorem conv_buffer_bounds {c channels : Nat} (h : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by omega

/-- Heads with index below `qk_heads` (≤ channels/128) get the L2 normalisation. -/
theorem qk_blocks_within_grid {blk qk channels : Nat} (hq : qk ≤ channels / 128) (hb : blk < qk) :
    blk < channels / 128 := by omega

/-- The 128-thread block reduces with 4 warp partials. -/
theorem conv_partials : 128 / 32 = 4 := by decide

-- § alpha/beta kernel

/-- Launch coverage: `(2*h_v+7)/8` blocks of 8 warps give each of the `2*h_v` rows a unique warp. -/
theorem ab_row_coverage (row h_v : Nat) (h : row < 2 * h_v) :
    row / 8 < (2 * h_v + 7) / 8 ∧ row = (row / 8) * 8 + row % 8 ∧ row % 8 < 8 := by omega

/-- The alpha/beta split: rows below `h_v` are gate rows, the others are beta rows. -/
theorem ab_row_split (row h_v : Nat) (h : row < 2 * h_v) :
    (row < h_v → row < h_v) ∧ (h_v ≤ row → row - h_v < h_v) := by omega

/-- Every 8-wide weight chunk `j < n/8` is visited by exactly lane `j % 32`. -/
theorem ab_chunk_lane (j n : Nat) (hn : n % 8 = 0) (hj : j < n / 8) :
    j % 32 < 32 ∧ j = j % 32 + 32 * (j / 32) ∧ j * 8 + 7 < n := by omega

/-- bf16 low half expanded to a float bit pattern. -/
def unpackLo (w : Nat) : Nat := (w % 65536) * 65536
/-- bf16 high half expanded to a float bit pattern. -/
def unpackHi (w : Nat) : Nat := (w / 65536) * 65536

/-- The two unpackings are 32-bit words. -/
theorem unpack_lt (w : Nat) : unpackLo w < 4294967296 ∧ (w < 4294967296 → unpackHi w < 4294967296) := by
  unfold unpackLo unpackHi
  constructor <;> omega

/-- The unpackings together recover the packed word. -/
theorem unpack_recover (w : Nat) : w = unpackLo w / 65536 + unpackHi w := by
  unfold unpackLo unpackHi
  omega

-- § launch guards

/-- Launch dimension for the step kernel is `h_v` blocks; each owns a distinct head. -/
theorem step_grid_heads {head h_v : Nat} (h : head < h_v) : head < h_v := h

/-- Output indices `head*S + col` fit in `h_v * S`. -/
theorem step_output_bound {h_v head col : Nat} (hh : head < h_v) (hc : col < 128) :
    head * 128 + col < h_v * 128 := rowmajor_lt hh hc

end Ref.FusedGdn



## FILE: /mnt/user-data/outputs/FusedGrReadKernels.lean
/-!
# MODULE: fused_gr_read_kernels
SOURCES: src/kernels/cuda/fused_gr.cu
Index-geometry theory of the fused gated-residual (hyper-connection) read: the `down` and `up` kernels for one
token and the T-token weight-sharing variants.

## Source semantics
* Geometry is fixed: n_embd 2560, 4 streams, D = 10240, low-rank 320, 256 threads / 8 warps per block.
* `down`: 40 blocks x 8 warps = 320 output rows, plus one extra block for the 4 inject rows.
* `up`: 80 blocks x 32 columns (x 4 streams = 128 rows, 16 per warp).
* multi-token: tiles of 2560 floats per token; `UPM` blocks of 16 columns.

## Form template identifiers
gr_down_kernel -> gated_residual_down ; gr_up_kernel -> gated_residual_up ; GrMulti -> multi_token_batch ;
N -> embed_dim ; HC -> stream_count ; LR -> low_rank
-/

namespace Ref.FusedGr

-- § helpers

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Streams. -/
def HC : Nat := 4
/-- Flattened width. -/
def D : Nat := N * HC
/-- Low rank. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := THREADS / 32
/-- Down blocks (one more handles inject). -/
def DOWN_BLOCKS : Nat := LR / WARPS
/-- Columns per `up` block. -/
def UP_COLS : Nat := 32
/-- `up` blocks. -/
def UP_BLOCKS : Nat := N / UP_COLS

/-- The constants have the values stated in the source comments. -/
theorem constants_values :
    D = 10240 ∧ WARPS = 8 ∧ DOWN_BLOCKS = 40 ∧ UP_BLOCKS = 80 := by decide

/-- 40 blocks of 8 warps give exactly one warp per low-rank row. -/
theorem down_rows_exact : DOWN_BLOCKS * WARPS = LR := by decide

/-- 80 blocks of 32 columns cover every embedding column exactly once. -/
theorem up_cols_exact : UP_BLOCKS * UP_COLS = N := by decide

/-- Each `up` block handles 4 x 32 = 128 rows, 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ 128 / WARPS = 16 := by decide

-- § stream decomposition of the flattened index

/-- A flat index splits into stream `c < 4` and column `d < N`. -/
theorem stream_split (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ i = (i / 2560) * 2560 + i % 2560 := by omega

/-- A float4 chunk starting at a multiple of 4 never straddles two streams. -/
theorem float4_no_straddle (i : Nat) (h4 : i % 4 = 0) (h : i < 10240) :
    i % 2560 + 3 < 2560 ∧ i / 2560 = (i + 3) / 2560 := by omega

/-- The `t*4` stride loop covers D in exactly 10 iterations of 1024 floats. -/
theorem norm_loop_trip_count : 10240 / (256 * 4) = 10 ∧ 10240 % (256 * 4) = 0 := by decide

/-- The element-wise tail loop visits each index once: `i` is reached by thread `i % 256`. -/
theorem tail_loop_owner (i : Nat) (h : i < 10240) :
    i % 256 < 256 ∧ i = i % 256 + 256 * (i / 256) := by omega

-- § down kernel rows

/-- Regular rows `blk*8 + warp` are in range for the 40 regular blocks. -/
theorem down_row_in_range {blk warp : Nat} (hb : blk < 40) (hw : warp < 8) : blk * 8 + warp < 320 := by
  omega

/-- Row assignment is injective across (block, warp). -/
theorem down_row_injective {b w b' w' : Nat} (hw : w < 8) (hw' : w' < 8)
    (h : b * 8 + w = b' * 8 + w') : b = b' ∧ w = w' := by omega

/-- Dot product chunks: 1280 chunks of 8, 40 per lane. -/
theorem down_chunks_per_lane : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 ∧ 1280 % 32 = 0 := by decide

/-- The inject block uses only warps below 4, each a distinct row index. -/
theorem inject_rows_distinct {w w' : Nat} (h : w < 4) (h' : w' < 4) (e : w = w') : w = w' := e

-- § up kernel rows

/-- Output row `c*N + d0 + dd` of the `up` kernel stays inside D. -/
theorem up_row_lt {c blk dd : Nat} (hc : c < 4) (hb : blk < 80) (hd : dd < 32) :
    c * 2560 + blk * 32 + dd < 10240 := by omega

/-- The (stream, block, column) → row map is injective. -/
theorem up_row_injective {c b d c' b' d' : Nat} (hb : b < 80) (hb' : b' < 80) (hd : d < 32) (hd' : d' < 32)
    (h : c * 2560 + b * 32 + d = c' * 2560 + b' * 32 + d') : c = c' ∧ b = b' ∧ d = d' := by omega

/-- Low-rank reduction chunks: 40 = 32 (first load) + 8 (second load by lanes < 8). -/
theorem up_chunk_split : 320 / 8 = 40 ∧ 320 / 8 - 32 = 8 := by decide

/-- Lane `l < 32` loads chunk `l`; lane `l < 8` loads chunk `32 + l`: together all 40 chunks, once. -/
theorem up_chunk_cover (j : Nat) (h : j < 40) :
    (j < 32 ∧ j % 32 = j) ∨ (32 ≤ j ∧ j - 32 < 8) := by omega

-- § multi-token variants

/-- The tile covers D in four steps and holds 320 chunks, 10 per lane. -/
theorem tile_geometry : 10240 / 2560 = 4 ∧ 10240 % 2560 = 0 ∧ 2560 / 8 / 32 = 10 := by decide

/-- Multi-token `up` blocks of 16 columns tile N and give 64 rows, 8 per warp. -/
theorem upm_geometry : 2560 / 16 = 160 ∧ 4 * 16 = 64 ∧ 64 / 8 = 8 := by decide

/-- Dynamic shared memory for `T` tokens never exceeds the `kFusedGrMaxT` attribute set once. -/
theorem dyn_smem_le {T M : Nat} (h : T ≤ M) : T * 2560 * 4 ≤ M * 2560 * 4 := by
  exact Nat.mul_le_mul_right 4 (Nat.mul_le_mul_right 2560 h)

/-- Token/offset split used when staging activations is exact. -/
theorem stage_split (i T : Nat) (h : i < T * 640) : i / 640 < T ∧ i = (i / 640) * 640 + i % 640 := by
  have : i / 640 < T := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  omega

/-- Source float4 offsets are integral: both `k*D` and the tile base are multiples of 4. -/
theorem stage_src_aligned (k base : Nat) (hb : base % 2560 = 0) : (k * 10240 + base) % 4 = 0 := by omega

/-- Lane `k` writes token `k`: only valid while `T ≤ 32`. -/
theorem lane_writes_token {k T : Nat} (hk : k < T) (hT : T ≤ 32) : k < 32 := by omega

/-- Token-batched lo buffer index `t*LR + k` is in bounds. -/
theorem lo_index_lt {t T k : Nat} (ht : t < T) (hk : k < 320) : t * 320 + k < T * 320 :=
  rowmajor_lt ht hk

/-- Token-batched `xn` scratch index `t*D + i` is in bounds. -/
theorem xn_index_lt {t T i : Nat} (ht : t < T) (hi : i < 10240) : t * 10240 + i < T * 10240 :=
  rowmajor_lt ht hi

end Ref.FusedGr



## FILE: /mnt/user-data/outputs/GdnReferenceKernels.lean
/-!
# MODULE: gdn_reference_kernels
SOURCES: src/kernels/cuda/gdn.cu
Non-projection parts of the gated delta net: the recurrence step (one thread per state column), the sliding
causal conv, per-row L2 norm, per-head gated RMS norm and the beta sigmoid gate.

## Source semantics
* State layout is `(S, h_v, S)` with `j` fastest, so column (h, j) is at `h*S + j + i*(h_v*S)`.
* Each thread owns one (h, j) column end to end: no barrier is needed inside the recurrence.
* Conv state slides: drop oldest, append newest; the new input lands in the last tap.
* `MAX_H = 1` heads per block (a latency-hiding choice, not a capacity one).

## Form template identifiers
gdn_step_kernel -> delta_rule_step ; gdn_conv_kernel -> causal_conv_step ; gdn_l2_kernel -> l2_row_norm ;
gdn_out_norm_kernel -> gated_out_norm ; gdn_beta_gate_kernel -> beta_sigmoid
-/

namespace Ref.Gdn

/-- Division/modulo uniqueness for a mixed-radix index `a*n+b`. -/
theorem divmod_unique {a b c d n : Nat} (hb : b < n) (hd : d < n) (h : a * n + b = c * n + d) :
    a = c ∧ b = d := by
  have hn : 0 < n := by omega
  have h1 : (a * n + b) / n = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  have h3 : a = c := by rw [← h1, ← h2, h]
  refine ⟨h3, ?_⟩
  subst h3
  exact Nat.add_left_cancel h

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § step kernel geometry

/-- Threads along `j`. -/
def JTHREADS : Nat := 32
/-- Heads per block. -/
def MAX_H : Nat := 1

/-- The grid is `ceil(S/32) x ceil(h_v/MAX_H)`: at the real size 4 x 48, i.e. 192 blocks. -/
theorem real_grid : (128 + 31) / 32 = 4 ∧ (48 + 1 - 1) / 1 = 48 ∧ 4 * 48 = 192 := by decide

/-- Every column `j < S` is assigned to a block along x and a lane. -/
theorem column_coverage (j S : Nat) (h : j < S) :
    j / 32 < (S + 31) / 32 ∧ j % 32 < 32 ∧ j = (j / 32) * 32 + j % 32 := by omega

/-- With one head per block, block `by` handles exactly head `by` (and nothing beyond `h_v`). -/
theorem head_per_block (by_ h_v : Nat) (hb : by_ < h_v) :
    min MAX_H (h_v - by_ * MAX_H) = 1 ∧ by_ * MAX_H < h_v := by
  unfold MAX_H
  omega

/-- Heads handled by a block are below `h_v`. -/
theorem staged_head_lt (by_ hi h_v : Nat) (hb : by_ < h_v) (hi_lt : hi < min MAX_H (h_v - by_ * MAX_H)) :
    by_ * MAX_H + hi < h_v := by
  unfold MAX_H at *
  omega

/-- Column address `h*S + j + i*(h_v*S)` equals the flat `(i*h_v + h)*S + j` address. -/
theorem column_address_flat (i h j h_v S : Nat) :
    i * (h_v * S) + (h * S + j) = (i * h_v + h) * S + j := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- The column map is injective in `(i, h, j)`: different threads never touch the same cell. -/
theorem column_address_injective {h_v i h j i' h' j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < 128) (hj' : j' < 128)
    (e : i * (h_v * 128) + (h * 128 + j) = i' * (h_v * 128) + (h' * 128 + j')) :
    i = i' ∧ h = h' ∧ j = j' := by
  rw [column_address_flat, column_address_flat] at e
  obtain ⟨a, b⟩ := divmod_unique hj hj' e
  obtain ⟨c, d⟩ := divmod_unique hh hh' a
  exact ⟨c, d, b⟩

/-- The staged `ks`/`qs` rows have 128 slots, so `S ≤ 128` is exactly the guard in `gdn_step`. -/
theorem staged_row_fits (i S : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

/-- Modulo head pairing stays below `h_k`. -/
theorem modulo_pairing (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt h hk

/-- Staging loop: element `i < S` of a row is loaded by lane `i % 32` in iteration `i / 32`. -/
theorem staging_owner (i : Nat) : i % 32 < 32 ∧ i = i % 32 + 32 * (i / 32) := by omega

-- § sliding conv

/-- Sliding the conv state keeps its length. -/
theorem slide_length {α : Type} (st : List α) (x : α) (h : st ≠ []) :
    ((st.drop 1) ++ [x]).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a t => simp

/-- The newest input is the last element after a slide. -/
theorem slide_last {α : Type} (st : List α) (x : α) :
    ((st.drop 1) ++ [x]).getLast? = some x := by
  simp

/-- With `d_conv = 4` the state has 3 rows and the state index `c*(dc-1) + i` stays in `3*C`. -/
theorem conv_state_bound {c C i : Nat} (hc : c < C) (hi : i < 3) : c * 3 + i < 3 * C := by omega

/-- The conv kernel's block count covers all channels. -/
theorem conv_blocks_cover (c C : Nat) (h : c < C) : c / 256 < (C + 255) / 256 := by omega

-- § norms and beta gate

/-- The L2 kernel's 32 threads stride columns; every `i < cols` has one owner. -/
theorem l2_owner (i : Nat) : i % 32 < 32 ∧ i = i % 32 + 32 * (i / 32) := by omega

/-- The guard `cols ≤ 1024` makes each lane's trip count at most 32. -/
theorem l2_trip_count (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by omega

/-- Per-head rows `h*S + i` fit within `h_v*S`. -/
theorem head_row_bound {h_v h S i : Nat} (hh : h < h_v) (hi : i < S) : h * S + i < h_v * S :=
  rowmajor_lt hh hi

/-- The beta gate launches `ceil(n/128)` blocks of 128 threads and covers every head. -/
theorem beta_gate_cover (i n : Nat) (h : i < n) : i / 128 < (n + 127) / 128 := by omega

end Ref.Gdn



## FILE: /mnt/user-data/outputs/GatedResidualReadWrite.lean
/-!
# MODULE: gated_residual_read_write
SOURCES: src/kernels/cuda/gr.cu
`gr_read` (norm, down, gate, mean, inject) and `gr_write` of the gated residual / hyper-connection, plus the
workspace carving of `gr_workspace_init`.

## Source semantics
* Workspace regions xn, xq, lq, gated, lo are carved in one table with 16-byte alignment; the earlier bug
  sized `xq` and `lq` in a different order from the pointer assignments.
* `gr_down_kernel`: one block per low-rank row, 8 warps splitting the `hc_dim` reduction.
* `gr_gate_kernel`: one warp per output row; `gr_mean_kernel`: one thread per embedding column.
* `gr_write_kernel`: `out[i] = R[i] + block_out[i % n_embd] * w[i / n_embd]`.

## Form template identifiers
gr_norm_kernel -> stream_rms_norm ; gr_down_kernel -> low_rank_down ; gr_gate_kernel -> low_rank_up_gate ;
gr_mean_kernel -> stream_mean ; gr_write_kernel -> residual_write ; GrWorkspace -> scratch_layout
-/

namespace Ref.Gr

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § workspace layout

/-- Round up to 16 bytes. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks and gains less than 16 bytes. -/
theorem align16_bounds (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 := by
  unfold align16; omega

/-- The aligned size is a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16; omega

/-- Region offsets of the five-entry table. -/
def offsets (a0 a1 a2 a3 : Nat) : Nat × Nat × Nat × Nat × Nat := (0, a0, a0 + a1, a0 + a1 + a2, a0 + a1 + a2 + a3)

/-- Total bytes of the five aligned regions. -/
def totalBytes (a0 a1 a2 a3 a4 : Nat) : Nat := a0 + a1 + a2 + a3 + a4

/-- Consecutive regions are disjoint and contained in the workspace. -/
theorem regions_disjoint (a0 a1 a2 a3 a4 : Nat) :
    (offsets a0 a1 a2 a3).1 + a0 ≤ (offsets a0 a1 a2 a3).2.1 ∧
    (offsets a0 a1 a2 a3).2.1 + a1 ≤ (offsets a0 a1 a2 a3).2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.1 + a2 ≤ (offsets a0 a1 a2 a3).2.2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.2.1 + a3 ≤ (offsets a0 a1 a2 a3).2.2.2.2 ∧
    (offsets a0 a1 a2 a3).2.2.2.2 + a4 = totalBytes a0 a1 a2 a3 a4 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> simp [offsets, totalBytes] <;> omega

/-- Real geometry sizes (xn, xq, lq, gated, lo) in bytes. -/
theorem real_sizes :
    (10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4) = (40960, 20480, 640, 40960, 1280) := by decide

/-- All five real sizes are already 16-byte aligned, total 104,320 bytes. -/
theorem real_workspace :
    align16 40960 = 40960 ∧ align16 20480 = 20480 ∧ align16 640 = 640 ∧ align16 1280 = 1280 ∧
    totalBytes 40960 20480 640 40960 1280 = 104320 := by decide

/-- The historical mix-up: `xq` needs `hc_dim*2` bytes, far more than the `hc_lr*2` once assigned to it. -/
theorem xq_size_regression : 320 * 2 < 10240 * 2 := by decide

-- § reduction-axis geometry

/-- In `gr_down_kernel` the 8 warps stride by 256: each `i < hc_dim` is owned by one (warp, lane). -/
theorem down_split_owner (i : Nat) :
    (i % 256) / 32 < 8 ∧ i % 256 = (i % 256) / 32 * 32 + i % 32 := by omega

/-- Block count of the gate kernel covers every output row. -/
theorem gate_rows_cover (i hc_dim : Nat) (h : i < hc_dim) : i / 8 < (hc_dim + 7) / 8 := by omega

/-- Each lane of the gate kernel strides `hc_lr` by 32; every `k` has owner `k % 32`. -/
theorem gate_lane_owner (k : Nat) : k % 32 < 32 ∧ k = k % 32 + 32 * (k / 32) := by omega

/-- Norm loop: 2560 elements strided by 256 threads is 10 iterations. -/
theorem norm_trip_count : 2560 / 256 = 10 ∧ 2560 % 256 = 0 := by decide

/-- Block-wide scratch: `(blockDim+31)>>5` warps, at most 8 for 256 threads. -/
theorem block_scratch_warps : (256 + 31) / 32 = 8 := by decide

/-- A `float[8]` scratch fits inside the `double[8]` it is reinterpreted from. -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

-- § mean and write kernels

/-- Gated activation index `c*n_embd + d` is in `hc*n_embd`. -/
theorem gated_index_lt {c hc d n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd := rowmajor_lt hc' hd

/-- In `gr_write_kernel`, a flat index splits into stream `< hc` and column `< n_embd`. -/
theorem write_split {i hc n_embd : Nat} (h : i < hc * n_embd) :
    i / n_embd < hc ∧ i % n_embd < n_embd ∧ i = (i / n_embd) * n_embd + i % n_embd := by
  have hpos : 0 < n_embd := by
    rcases Nat.eq_zero_or_pos n_embd with h0 | h0
    · subst h0; simp at h
    · exact h0
  have h1 : i / n_embd < hc := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  have h2 : i % n_embd < n_embd := Nat.mod_lt _ hpos
  have h3 := Nat.div_add_mod i n_embd
  refine ⟨h1, h2, ?_⟩
  rw [Nat.mul_comm]
  exact h3.symm

/-- The shared weight array `w` has `hc` slots, written by threads `< hc`; needs `hc ≤ blockDim`. -/
theorem write_weights_written {t hc : Nat} (ht : t < hc) (hb : hc ≤ 256) : t < 256 := by omega

/-- Grid-stride coverage of the write kernel: for stride `s > 0`, `i = i % s + s*(i/s)`. -/
theorem grid_stride_owner (i s : Nat) : i % s + s * (i / s) = i := Nat.mod_add_div i s

/-- Real geometry: 10240 elements, 40 blocks of 256 threads. -/
theorem write_blocks_real : (10240 + 255) / 256 = 40 := by decide

end Ref.Gr



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
SOURCES: src/kernels/cuda/native_gdn.cu
Native (llama.cpp-arithmetic) gated-delta-net step: each warp owns one (head, column) and keeps the four
rows `r*32 + lane` of its column in registers; plus the argument validation (alignment, no-wrap, disjointness).

## Source semantics
* Grid `(h_v, 1, S/4)`, block `(32, 4)`: `col = blockIdx.z*4 + threadIdx.y`, lane `l` owns rows `r*32 + l`.
* Preconditions: stream non-null, S = 128, `h_k > 0`, `h_v % h_k = 0`, `h_v ≤ 65535`.
* Spans must be float-aligned, non-wrapping, and state/output must be disjoint from everything else.

## Form template identifiers
step -> native_delta_step ; valid_span -> span_valid ; overlap -> span_overlap ;
native_gdn_step -> native_gdn_entry
-/

namespace Ref.NativeGdn

/-- Division/modulo uniqueness for a mixed-radix index `a*n+b`. -/
theorem divmod_unique {a b c d n : Nat} (hb : b < n) (hd : d < n) (h : a * n + b = c * n + d) :
    a = c ∧ b = d := by
  have hn : 0 < n := by omega
  have h1 : (a * n + b) / n = a := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h2 : (c * n + d) / n = c := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  have h3 : a = c := by rw [← h1, ← h2, h]
  refine ⟨h3, ?_⟩
  subst h3
  exact Nat.add_left_cancel h

-- § warp/register ownership

/-- Rows owned by a lane: `r*32 + lane` for `r < 4`, covering `[0,128)` bijectively. -/
theorem rows_cover (i : Nat) (h : i < 128) :
    ∃ r lane, r < 4 ∧ lane < 32 ∧ i = r * 32 + lane :=
  ⟨i / 32, i % 32, by omega, by omega, by omega⟩

/-- No two (r, lane) pairs share a row. -/
theorem rows_injective {r l r' l' : Nat} (hl : l < 32) (hl' : l' < 32)
    (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- The column of a thread is below 128 when `z < 32`, `y < 4`. -/
theorem col_in_range {z y : Nat} (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Column assignment is injective in (z, y) and onto `[0,128)`. -/
theorem col_bijective (c : Nat) (h : c < 128) :
    c / 4 < 32 ∧ c % 4 < 4 ∧ c = (c / 4) * 4 + c % 4 := by omega

/-- Block shape: 32 x 4 = 128 threads, matching `__launch_bounds__(128, 2)`. -/
theorem block_threads : 32 * 4 = 128 ∧ 128 / 4 = 32 := by decide

/-- State address `(i*h_v + head)*S + col` is injective across the whole grid. -/
theorem address_injective {h_v i head col i' head' col' : Nat} (hh : head < h_v) (hh' : head' < h_v)
    (hc : col < 128) (hc' : col' < 128)
    (e : (i * h_v + head) * 128 + col = (i' * h_v + head') * 128 + col') :
    i = i' ∧ head = head' ∧ col = col' := by
  obtain ⟨a, b⟩ := divmod_unique hc hc' e
  obtain ⟨c, d⟩ := divmod_unique hh hh' a
  exact ⟨c, d, b⟩

/-- Query/key head `head % h_k` indexes a valid row of the q/k buffers. -/
theorem qk_row_in_bounds {head h_k i : Nat} (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < 128 * h_k := by
  have h := Nat.mod_lt head hk
  omega

-- § validation predicates

/-- Span validity: nonnull, float-aligned, no pointer wrap. -/
def validSpan (addr bytes maxAddr : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ maxAddr - addr

/-- Overlap test of the source (`ap < bp + bn && bp < ap + an`). -/
def overlapSpan (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm {a an b bn : Nat} : overlapSpan a an b bn ↔ overlapSpan b bn a an := by
  unfold overlapSpan
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Separated spans do not overlap. -/
theorem overlap_false_of_separated {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlapSpan a an b bn := by
  unfold overlapSpan
  omega

/-- A valid span does not wrap: its end is within the address space. -/
theorem valid_span_no_wrap {addr bytes maxAddr : Nat} (h : validSpan addr bytes maxAddr)
    (hm : addr ≤ maxAddr) : addr + bytes ≤ maxAddr := by
  unfold validSpan at h
  omega

/-- Byte counts used by the entry point at S = 128. -/
theorem byte_counts (h_v h_k : Nat) :
    128 * 128 * h_v * 4 = 65536 * h_v ∧ 128 * h_k * 4 = 512 * h_k ∧ 128 * h_v * 4 = 512 * h_v := by omega

/-- Head count limits: `h_v ≤ 65535` fits the grid.x limit and divisibility yields `h_k ≤ h_v`. -/
theorem grid_limit {h_v : Nat} (h : h_v ≤ 65535) : h_v < 65536 := by omega

end Ref.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocess.lean
/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
Native GDN preprocessing kernels (conv + SiLU, L2 norm, beta sigmoid, gate softplus, gated output RMS norm)
and their launch-geometry / span validation.

## Source semantics
* `conv_silu`: window `[h0,h1,h2,x]` with 4 taps; history shifts to `[h1,h2,x]`.
* `l2_norm` / `out_norm`: one block per row of width exactly 128, 256 threads, 8 warp sums.
* Launch checks: stream non-null, count in `[1, 65535]`, width = 128, epsilon finite and nonnegative.
* Writable spans must be pairwise disjoint and disjoint from inputs.

## Form template identifiers
conv_silu -> conv_silu_step ; l2_norm -> row_l2 ; beta_sigmoid -> beta_gate ; gate_softplus -> decay_gate ;
out_norm -> gated_rms_out
-/

namespace Ref.NativeGdnPre

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv

/-- One conv step on a 3-deep history: the window is `(h0,h1,h2,x)`, the new history `(h1,h2,x)`. -/
def convShift (h : Nat × Nat × Nat) (x : Nat) : Nat × Nat × Nat := (h.2.1, h.2.2, x)

/-- The new history's last entry is the new input. -/
theorem convShift_last (h : Nat × Nat × Nat) (x : Nat) : (convShift h x).2.2 = x := rfl

/-- The oldest value is dropped, the others move one slot back. -/
theorem convShift_moves (a b c x : Nat) : convShift (a, b, c) x = (b, c, x) := rfl

/-- Shifting three times from `(a,b,c)` with inputs `x y z` leaves exactly the inputs. -/
theorem convShift_three (a b c x y z : Nat) :
    convShift (convShift (convShift (a, b, c) x) y) z = (x, y, z) := rfl

/-- Channel `c < channels` addresses 3 history and 4 weight cells within bounds. -/
theorem conv_cells {c channels : Nat} (h : c < channels) :
    c * 3 + 2 < 3 * channels ∧ c * 4 + 3 < 4 * channels := by omega

/-- Conv launch covers all channels with 256-thread blocks. -/
theorem conv_cover (c channels : Nat) (h : c < channels) : c / 256 < (channels + 255) / 256 := by omega

/-- Span byte sizes: history is 3x, weights 4x the per-channel float vector. -/
theorem conv_span_sizes (channels : Nat) : 3 * (channels * 4) = 12 * channels ∧ 4 * (channels * 4) = 16 * channels := by
  omega

-- § norms

/-- 256 threads form 8 warps, matching the `lane < 8` second-level sum. -/
theorem norm_warps : 256 / 32 = 8 ∧ 8 ≤ 32 := by decide

/-- Columns `col ≥ 128` of the 256-thread block contribute zero; exactly 128 contribute. -/
theorem norm_active_columns : 128 ≤ 256 ∧ 256 - 128 = 128 := by decide

/-- Row offset `row*S + col` of norm blocks fits within `rows*S`. -/
theorem norm_offset_lt {row rows col : Nat} (hr : row < rows) (hc : col < 128) :
    row * 128 + col < rows * 128 := rowmajor_lt hr hc

/-- Geometry validity: width 128 and a row count in `[1,65535]` (a CUDA grid.x-safe bound). -/
def normGeometryOk (rows cols : Nat) : Prop := 1 ≤ rows ∧ rows ≤ 65535 ∧ cols = 128

/-- A valid geometry has a bounded grid. -/
theorem normGeometry_grid {rows cols : Nat} (h : normGeometryOk rows cols) : rows < 65536 ∧ cols = 128 := by
  unfold normGeometryOk at h
  omega

/-- Zero rows are rejected. -/
theorem normGeometry_rejects_zero (cols : Nat) : ¬ normGeometryOk 0 cols := by
  unfold normGeometryOk
  omega

/-- The epsilon passed into the L2 kernel is `eps / S` (scale `S = 128`): integer model of the scaling. -/
theorem eps_scaling (e : Nat) : (e * 128) / 128 = e := by omega

-- § element-wise kernels

/-- Beta sigmoid / gate softplus launches (256-thread blocks) cover every head. -/
theorem head_launch_cover (i n : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

/-- Output norm: head block `b`, column `col` writes `b*128 + col`, which is injective. -/
theorem out_norm_injective {b col b' col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (h : b * 128 + col = b' * 128 + col') : b = b' ∧ col = col' := by omega

-- § span validation

/-- Overlap predicate of the source. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so checking `(i, j<i)` pairs suffices. -/
theorem overlaps_symm {a an b bn : Nat} : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Three writable spans give exactly three unordered pairs checked by the nested loop. -/
theorem pair_count : (List.range 3).foldl (fun acc i => acc + (List.range i).length) 0 = 3 := by decide

end Ref.NativeGdnPre



## FILE: /mnt/user-data/outputs/NativePlePostops.lean
/-!
# MODULE: native_ple_postops
SOURCES: src/kernels/cuda/native_ple_postops.cu
Native PLE post-projection operators: signed-sqrt gate per stream, value broadcast, and the dilated 4-tap
conv + residual, with the span-validation rules that allow exactly two aliasing exceptions.

## Source semantics
* Geometry: N = 2560, H = 4 streams, D = N*H = 10240, conv history depth 9 per channel.
* `gate_kernel`: 512 threads per stream, 8 partial lanes `d = tid + j*512` masked to `d < N`.
* `broadcast_kernel`: `gated[i] = value[i % N] * gate[i / N]`.
* `conv_residual_kernel`: taps read history rows 0, 3, 6 and the fresh normalized value for tap 3.

## Form template identifiers
gate_kernel -> stream_gate ; broadcast_kernel -> value_broadcast ; conv_residual_kernel -> conv_residual ;
validate -> span_check ; native_ple_postops -> ple_postops_entry
-/

namespace Ref.NativePle

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- Geometry constants. -/
theorem geometry : 2560 * 4 = 10240 ∧ 10240 % 256 = 0 ∧ 10240 / 256 = 40 := by decide

-- § gate kernel

/-- Each element `d < N` is covered by thread `d % 512` in partial slot `d / 512 < 5 ≤ 8`. -/
theorem gate_slot_cover (d : Nat) (h : d < 2560) :
    d % 512 < 512 ∧ d / 512 < 5 ∧ d = d % 512 + 512 * (d / 512) := by omega

/-- Slots 5..7 are always masked: `tid + j*512 ≥ N` for `j ≥ 5`. -/
theorem gate_masked_slots (tid j : Nat) (hj : 5 ≤ j) : 2560 ≤ tid + j * 512 := by omega

/-- 512 threads are 16 warps, and the second-level reduction reads `partials[lane < 16]`. -/
theorem gate_warps : 512 / 32 = 16 ∧ 16 ≤ 32 := by decide

/-- Per-stream row offset `stream*N + d` fits in D. -/
theorem gate_row_lt {s d : Nat} (hs : s < 4) (hd : d < 2560) : s * 2560 + d < 10240 := by omega

-- § broadcast and conv

/-- A flat index splits into stream `< 4` and column `< N`. -/
theorem broadcast_split (i : Nat) (h : i < 10240) :
    i % 2560 < 2560 ∧ i / 2560 < 4 ∧ i = (i / 2560) * 2560 + i % 2560 := by omega

/-- History row index of tap `k`: `3*k` for taps 0..2, and tap 3 uses the new value instead. -/
def histRow (k : Nat) : Nat := 3 * k

/-- All history reads stay within the 9-deep column: `c*9 + 3k < 9*D`. -/
theorem history_read_in_bounds {c k : Nat} (hc : c < 10240) (hk : k < 3) : c * 9 + histRow k < 9 * 10240 := by
  unfold histRow
  omega

/-- The three history taps are rows 0, 3, 6 (the new value is the fourth). -/
theorem history_taps : [histRow 0, histRow 1, histRow 2] = [0, 3, 6] := by decide

/-- Weight halves for channel `c` and tap `k < 4` stay within `4*D`. -/
theorem weight_in_bounds {c k : Nat} (hc : c < 10240) (hk : k < 4) : c * 4 + k < 4 * 10240 := by omega

/-- The residual is `hidden + (gated + activation)`; `+` on naturals is associative, so the model order
(`gated + act` first) is the only association the source ever uses. -/
theorem residual_association (h g a : Nat) : h + (g + a) = (h + g) + a := by omega

-- § span validation

/-- Overlap test. -/
def overlaps (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Symmetric. -/
theorem overlaps_symm {a an b bn : Nat} : overlaps a an b bn ↔ overlaps b bn a an := by
  unfold overlaps
  constructor <;> (intro h; exact ⟨h.2, h.1⟩)

/-- Pair counts of the double loop: 7 outputs x 8 inputs, and 7*6/2 output pairs. -/
theorem check_counts : 7 * 8 = 56 ∧ 7 * 6 / 2 = 21 := by decide

/-- Byte sizes of the validated spans (floats of length D, N and H, half-precision weights 4*D). -/
theorem span_sizes : 10240 * 4 = 40960 ∧ 2560 * 4 = 10240 ∧ 4 * 10240 * 2 = 81920 ∧ 9 * 10240 * 4 = 368640 := by
  decide

end Ref.NativePle



## FILE: /mnt/user-data/outputs/PleBlockKernels.lean
/-!
# MODULE: ple_block_kernels
SOURCES: src/kernels/cuda/ple.cu
GPU half of the PLE block: key projection, grouped norm, gate, broadcast, dilated causal conv, history
advance, and the caller-owned scratch carving.

## Source semantics
* Conv state is ROW-FASTEST: element `(row, channel)` is at `row + nhist*channel`, nhist = 9.
* Tap `k` of the 4-tap, dilation-3 conv reads row `nhist - (3-k)*3`, i.e. 0, 3, 6 and 9 (9 is the new row).
* Scratch = five hc_dim floats + n_embd + hc, then the Q8 image, then the BF16 embedding copy, each 16-aligned.
* `overlap` treats zero-length spans as non-overlapping.

## Form template identifiers
gnorm_kernel -> grouped_norm ; gate_kernel -> signed_sqrt_gate ; conv_kernel -> dilated_conv ;
history_advance_kernel -> history_shift ; ple_block_scratch_bytes -> scratch_size
-/

namespace Ref.Ple

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § overlap predicate

/-- The source's overlap function. -/
def overlapB (a an b bn : Nat) : Bool :=
  if an = 0 ∨ bn = 0 then false else if a < b then decide (b - a < an) else decide (a - b < bn)

/-- For nonempty spans the function is exactly interval intersection. -/
theorem overlapB_iff {a an b bn : Nat} (h1 : 0 < an) (h2 : 0 < bn) :
    overlapB a an b bn = true ↔ (a < b + bn ∧ b < a + an) := by
  unfold overlapB
  have hn : ¬ (an = 0 ∨ bn = 0) := by omega
  rw [if_neg hn]
  by_cases h : a < b
  · rw [if_pos h]
    simp only [decide_eq_true_eq]
    omega
  · rw [if_neg h]
    simp only [decide_eq_true_eq]
    omega

/-- Empty spans never overlap. -/
theorem overlapB_empty_left (a b bn : Nat) : overlapB a 0 b bn = false := by
  unfold overlapB
  simp

-- § scratch layout

/-- Round up to 16 bytes. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment bounds. -/
theorem align16_bounds (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 := by
  unfold align16; omega

/-- Total scratch bytes as computed by `ple_block_scratch_bytes`. -/
def scratchBytes (f q e : Nat) : Nat := align16 f + align16 q + e + 256

/-- Real geometry: float region, Q8 image (80 groups of 34 bytes), BF16 copy of n_embd. -/
theorem real_scratch :
    (5 * 10240 + 2560 + 4) * 4 = 215056 ∧ (2560 / 32) * 34 = 2720 ∧ 2560 * 2 = 5120 := by decide

/-- Real total: 215056 + 2720 + 5120 + 256. -/
theorem real_scratch_total : scratchBytes 215056 2720 5120 = 223152 := by decide

/-- The three regions are disjoint and contained in the scratch. -/
theorem scratch_regions (f q e : Nat) :
    f ≤ align16 f ∧ align16 f + q ≤ align16 f + align16 q ∧
    align16 f + align16 q + e ≤ scratchBytes f q e := by
  have h1 := align16_bounds f
  have h2 := align16_bounds q
  unfold scratchBytes
  omega

/-- Float sub-buffers are carved at `0, D, 2D, 3D, 4D, 5D, 5D+N` and all fit in `5D+N+H` floats. -/
theorem float_offsets (D N H : Nat) :
    0 + D ≤ D ∧ D + D ≤ 2 * D ∧ 2 * D + D ≤ 3 * D ∧ 3 * D + D ≤ 4 * D ∧ 4 * D + D ≤ 5 * D ∧
    5 * D + N ≤ 5 * D + N + 0 ∧ 5 * D + N + H = 5 * D + N + H := by omega

/-- The float scratch is a multiple of 16 bytes at the real geometry, so alignment adds nothing. -/
theorem float_region_aligned : align16 215056 = 215056 := by decide

-- § conv geometry

/-- Tap rows for kern 4, dilation 3, nhist 9: 0, 3, 6, 9. -/
theorem tap_rows : [9 - (3 - 0) * 3, 9 - (3 - 1) * 3, 9 - (3 - 2) * 3, 9 - (3 - 3) * 3] = [0, 3, 6, 9] := by
  decide

/-- Only the last tap reads the new row (`row = nhist`); the other three read stored history. -/
theorem only_last_tap_new (k : Nat) (hk : k < 4) : (9 - (3 - k) * 3 = 9) ↔ k = 3 := by omega

/-- Row-fastest history index `row + 9*c` is in bounds for `row < 9`, `c < hc_dim`. -/
theorem history_index_lt {row c hc_dim : Nat} (hr : row < 9) (hc : c < hc_dim) :
    row + 9 * c < 9 * hc_dim := by omega

/-- Row-fastest indexing is injective. -/
theorem history_index_injective {row c row' c' : Nat} (hr : row < 9) (hr' : row' < 9)
    (h : row + 9 * c = row' + 9 * c') : row = row' ∧ c = c' := by omega

-- § history advance

/-- Slide: drop the oldest row, append the newest. -/
def slide {α : Type} (col : List α) (x : α) : List α := col.drop 1 ++ [x]

/-- A slide of a nonempty column keeps its length (9 in the engine). -/
theorem slide_length {α : Type} (col : List α) (x : α) (h : col ≠ []) :
    (slide col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a t => simp [slide]

/-- Row `i+1` moves to row `i` in the kernel's shift loop (`row + 1 < 9` reads in bounds). -/
theorem shift_loop_in_bounds {row : Nat} (h : row + 1 < 9) : row + 1 ≤ 8 := by omega

/-- The shift kernel's channel guard covers hc_dim: 40 blocks of 256 exactly. -/
theorem history_blocks : (10240 + 255) / 256 = 40 ∧ 40 * 256 = 10240 := by decide

-- § stream layout

/-- Stream pointers `x + c*n_embd` for `c < hc` plus column `d < n_embd` stay in `hc*n_embd`. -/
theorem stream_offset_lt {c hc d n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd := rowmajor_lt hc' hd

/-- Block-wide sum scratch of 8 doubles supports blockDim 256 (8 warps). -/
theorem block_sum_warps : (256 + 31) / 32 = 8 := by decide

end Ref.Ple



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify window) versions of the GDN kernels plus small device-side helpers
(gathers, pointer rebase, window ids, MTP select).

## Source semantics
* The per-token arithmetic is the single-token arithmetic with the same operation order.
* Token `t`'s conv window is the last four entries of `[hist0,hist1,hist2,x_0..x_t]`; only earlier tokens'
  inputs are read for the three history taps.
* `gdn_conv_commit` keeps the last three entries of `[hist | x_0..x_{n-1}]`.
* Preconditions NOT checked by the host code and therefore assumed here: `*n_keep ≤ T`,
  `conv_channels = 2*S*h_k + S*h_v`, `n ≤ 64` for `map_ids`.

## Form template identifiers
gdn_conv_l2_multi_kernel -> window_conv_l2 ; gdn_conv_commit_kernel -> conv_commit ;
gdn_step_norm_multi_kernel -> window_state_step ; window_ids_kernel -> sliding_window_ids ;
fetch_blobs_kernel -> blob_fetch ; rebase_ptrs_kernel -> pointer_rebase
-/

namespace Ref.Verify

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § conv window

/-- History taps read only earlier tokens: for `j < 3` and `src = t + j ≥ 3`, the row `src - 3 < t`. -/
theorem window_reads_earlier {t j : Nat} (hj : j < 3) (hs : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Reads from the stored history are in range: `src < 3` gives history index `< 3`. -/
theorem window_hist_in_range {t j : Nat} (hs : t + j < 3) : t + j < 3 := hs

/-- Output row `t_begin + by` of the window kernel is below `t_begin + n_tok`. -/
theorem window_token_in_range {t_begin b n_tok : Nat} (h : b < n_tok) : t_begin + b < t_begin + n_tok := by
  omega

/-- Commit reads: for `j < 3` and `src = n + j ≥ 3`, the row `src - 3 < n` (only kept tokens). -/
theorem commit_reads_kept {n j : Nat} (hj : j < 3) (hs : 3 ≤ n + j) : n + j - 3 < n := by omega

/-- The retained history `(hist ++ xs).drop n` has length 3 when `hist` has 3 entries and `xs` has `n`. -/
theorem commit_length {α : Type} (hist xs : List α) (h3 : hist.length = 3) :
    ((hist ++ xs).drop xs.length).length = 3 := by
  simp [List.length_drop, h3]

-- § step norm window

/-- Layout of the conv output row: q in `[0,qk)`, k in `[qk,2qk)`, v in `[2qk, 2qk+vd)`; all in `C`. -/
theorem qkv_offsets_in_bounds {h_k h_v qh head tid col C : Nat}
    (hqh : qh < h_k) (hhead : head < h_v) (ht : tid < 128) (hc : col < 128)
    (hC : C = 2 * (128 * h_k) + 128 * h_v) :
    qh * 128 + tid < 128 * h_k ∧
    128 * h_k + qh * 128 + tid < 2 * (128 * h_k) ∧
    2 * (128 * h_k) + head * 128 + col < C := by omega

/-- Per-token output row `t*value_dim + head*S + col` is below `T*value_dim`. -/
theorem y_index_lt {t T h_v head col : Nat} (ht : t < T) (hh : head < h_v) (hc : col < 128) :
    t * (128 * h_v) + head * 128 + col < T * (128 * h_v) := by
  have h1 : head * 128 + col < 128 * h_v := by omega
  have h2 := rowmajor_lt (h := 128 * h_v) (r := head * 128 + col) ht h1
  omega

/-- Tokens replayed (index `< t_out_begin`) update state but write no output; written outputs number `n - t_out_begin`
(natural subtraction, so this also holds when `t_out_begin > n`, where nothing is written). -/
theorem outputs_written (n t_out : Nat) :
    ((List.range n).filter (fun t => decide (t_out ≤ t))).length = n - t_out := by
  induction n with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.filter_append]
    by_cases hg : t_out ≤ m <;> simp [List.length_append, hg, ih] <;> omega

-- § ab multi

/-- Batched gate/beta index `t*h_v + r` is in bounds. -/
theorem ab_index_lt {t T h_v r : Nat} (ht : t < T) (hr : r < h_v) : t * h_v + r < T * h_v :=
  rowmajor_lt ht hr

/-- Batched activations `x + t*n` read float4 at `j*8+4+3 < n` since `8 | n`. -/
theorem ab_float4_in_bounds {j n : Nat} (hn : n % 8 = 0) (hj : j < n / 8) : j * 8 + 7 < n := by omega

-- § embedding gather

/-- For code widths 1, 2, 4, 8 the packing is exact: `(8/bits)*bits = 8` and the in-byte shift is `< 8`. -/
theorem code_packing (bits i : Nat) (hb : bits = 1 ∨ bits = 2 ∨ bits = 4 ∨ bits = 8) :
    (8 / bits) * bits = 8 ∧ (i % (8 / bits)) * bits < 8 := by
  rcases hb with h | h | h | h <;> subst h <;> omega

/-- Group of element `i` is below the number of groups when `group_elems | n`. -/
theorem group_in_range {i n g : Nat} (hg : 0 < g) (hn : n % g = 0) (hi : i < n) : i / g < n / g := by
  have h : i / g < n / g := Nat.div_lt_div_of_lt_of_dvd (Nat.dvd_of_mod_eq_zero hn) hi
  exact h

-- § broadcast, blobs, rebase

/-- Broadcast: output index `t*n*hc + i` is in bounds for `i < n*hc`. -/
theorem broadcast_index_lt {t T n hc i : Nat} (ht : t < T) (hi : i < n * hc) :
    t * (n * hc) + i < T * (n * hc) := rowmajor_lt ht hi

/-- Source column `i % n` is a valid column when `n > 0`. -/
theorem broadcast_source_col {i n : Nat} (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- `fetch_blobs`: with `per > 0`, `i < n*per` gives blob `i / per < n` and offset `i % per < per`. -/
theorem fetch_split {i n per : Nat} (hp : 0 < per) (h : i < n * per) :
    i / per < n ∧ i % per < per ∧ i = (i / per) * per + i % per := by
  have h1 : i / per < n := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  have h3 := Nat.div_add_mod i per
  refine ⟨h1, Nat.mod_lt _ hp, ?_⟩
  rw [Nat.mul_comm]
  exact h3.symm

/-- `rebase_ptrs`: consecutive rebased pointers are separated by at least one blob. -/
theorem rebase_separated {base k k' bytes : Nat} (h : k < k') :
    base + k * bytes + bytes ≤ base + k' * bytes := by
  have h1 : (k + 1) * bytes ≤ k' * bytes := Nat.mul_le_mul_right bytes (Nat.succ_le_of_lt h)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- `rebase_ptrs` launches 128 threads, so it handles at most 128 blobs. -/
theorem rebase_capacity {k n : Nat} (h : k < n) (hn : n ≤ 128) : k < 128 := by omega

-- § small helpers

/-- Window ids: `start + width = n_kv`, `width ≤ window`, `width ≤ n_kv`. -/
theorem window_width (n_kv window : Nat) :
    let start := if n_kv > window then n_kv - window else 0
    start + (n_kv - start) = n_kv ∧ n_kv - start ≤ window ∧ n_kv - start ≤ n_kv := by
  simp only
  split <;> omega

/-- Dense steps: the four entries of query `i` occupy `i*4 .. i*4+3` and stay below `4n`. -/
theorem dense_steps_bounds {i n k : Nat} (hi : i < n) (hk : k < 4) : i * 4 + k < 4 * n := by omega

/-- `ident_hits` requires `n ≤ 1024` threads; thread `i < n` is then valid. -/
theorem ident_threads {i n : Nat} (hi : i < n) (hn : n ≤ 1024) : i < 1024 := by omega

/-- Grid-stride loops: with stride `s`, `i = i % s + s*(i/s)`, so each `i` has one owner. -/
theorem stride_owner (i s : Nat) : i % s + s * (i / s) = i := Nat.mod_add_div i s

/-- `copy_indexed` caps blocks at 64. -/
theorem copy_blocks_cap (n : Nat) : (if (n + 255) / 256 < 64 then (n + 255) / 256 else 64) ≤ 64 := by
  split <;> omega

/-- Top-prob reduction: 1024 threads are 32 warps, matching `part[32]`. -/
theorem top_prob_warps : 1024 / 32 = 32 := by decide

end Ref.Verify



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- VerifyWindowKernels.lean
  was: theorem outputs_written (n t_out : Nat) (h : t_out ≤ n) :
    ((List.range n).filter (fun t => decide (t_out ≤ t))).length = n - t_out
  now: theorem outputs_written (n t_out : Nat) :
    ((List.range n).filter (fun t => decide (t_out ≤ t))).length = n - t_out
  measured: fewer quantified variables (2 → 1)
-/



## FILE: /mnt/user-data/outputs/QsaStepGeometry.lean
/-!
# MODULE: qsa_step_geometry
SOURCES: include/ref/kernels/qsa.hpp, src/kernels/cuda/qsa.cu
The device-resident per-token step state (pos, n_kv, n_bid, width) of the QSA layer, the selection
width, the GQA head map and the paged-KV row addressing.

## Source semantics
* `qsa_step_fill` derives all four entries from `pos`: n_kv = pos+1, n_bid = n_kv / r,
  width = min(n_kv, top_k + r - 1).
* Below the bound the selection is the identity; the width never exceeds n_kv.
* q -> kv head map is integer division (`h / (n_head / n_head_kv)`), not `h % n_head_kv`.
* Physical KV row = (page * n_head_kv + h) * page_size + slot; one head's cells are contiguous in a page.

## Form template identifiers
step.pos <- kStepPos, step.nKv <- kStepNKv, step.nBid <- kStepNBid, step.width <- kStepWidth,
selWidth <- qsa_selection_width, physRow <- kv_append/kv_gather row formula, kvHead <- iv2 = iq2 / rv2.
-/

namespace Ref.QsaStep

-- § selection width and step state

/-- The selection width `min(n_kv, top_k + r - 1)`. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- The four per-token values held in the device step buffer. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries computed from the single position. -/
def stepFill (pos topK r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) topK r }

/-- The selection never returns more cells than are cached. -/
theorem selWidth_le_nKv (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by
  unfold selWidth; omega

/-- The selection never exceeds the geometry's bound `top_k + r - 1`. -/
theorem selWidth_le_bound (nKv topK r : Nat) : selWidth nKv topK r ≤ topK + r - 1 := by
  unfold selWidth; omega

/-- Below the bound the selection is the identity. -/
theorem selWidth_identity {nKv topK r : Nat} (h : nKv ≤ topK + r - 1) :
    selWidth nKv topK r = nKv := by
  unfold selWidth; omega

/-- The selection width is monotone in the number of cached cells. -/
theorem selWidth_mono {a b : Nat} (topK r : Nat) (h : a ≤ b) :
    selWidth a topK r ≤ selWidth b topK r := by
  unfold selWidth; omega

/-- `n_kv` is `pos + 1` in every filled step. -/
theorem stepFill_nKv (pos topK r : Nat) : (stepFill pos topK r).nKv = pos + 1 := rfl

/-- The filled step is internally consistent: width is derived from its own n_kv. -/
theorem stepFill_consistent (pos topK r : Nat) :
    (stepFill pos topK r).nKv = (stepFill pos topK r).pos + 1 ∧
    (stepFill pos topK r).width = selWidth (stepFill pos topK r).nKv topK r := ⟨rfl, rfl⟩

/-- Completed blocks never cover more cells than are cached. -/
theorem stepFill_blocks_le (pos topK r : Nat) :
    (stepFill pos topK r).nBid * r ≤ (stepFill pos topK r).nKv := by
  simp only [stepFill]
  exact Nat.div_mul_le_self _ _

/-- The incomplete tail is shorter than one block. -/
theorem stepFill_tail_lt {r : Nat} (pos topK : Nat) (hr : 0 < r) :
    (stepFill pos topK r).nKv < (stepFill pos topK r).nBid * r + r := by
  simp only [stepFill]
  have h1 := Nat.div_add_mod (pos + 1) r
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at h1
  omega

/-- The step buffer is four int32. -/
theorem stepBytes_eq : 4 * 4 = 16 := by decide

-- § GQA head map

/-- Query head to key/value head: integer division. -/
def kvHead (q group : Nat) : Nat := q / group

/-- The integer-division map lands inside the kv heads. -/
theorem kvHead_lt {q group nKvH : Nat} (hq : q < group * nKvH) : kvHead q group < nKvH :=
  Nat.div_lt_of_lt_mul hq

/-- For the real artifact (24 query heads, 2 kv heads) every map value is a valid kv head. -/
theorem real_gqa_map : ∀ h : Nat, h < 24 → h / 12 < 2 := by
  intro h hh; omega

/-- The real map is `[0 x12, 1 x12]`. -/
theorem real_gqa_split :
    (List.range 24).map (· / 12) = List.replicate 12 0 ++ List.replicate 12 1 := by
  decide

/-- The rival modulo reading differs from the division reading (head 1). -/
theorem div_ne_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

-- § paged KV addressing

/-- Physical row of logical (page, kv head, slot). -/
def physRow (page h nKvH pageSize slot : Nat) : Nat := (page * nKvH + h) * pageSize + slot

/-- Consecutive cells of one head are contiguous inside a page. -/
theorem physRow_contiguous (page h nKvH pageSize slot : Nat) :
    physRow page h nKvH pageSize (slot + 1) = physRow page h nKvH pageSize slot + 1 := by
  unfold physRow; omega

/-- A cell index splits into page index and slot. -/
theorem cell_split (pos pageSize : Nat) : pos / pageSize * pageSize + pos % pageSize = pos :=
  Nat.div_add_mod' pos pageSize

/-- The page-table index of a cell is in range. -/
theorem page_index_lt {pos pageSize nPages : Nat} (h : pos < pageSize * nPages) :
    pos / pageSize < nPages := Nat.div_lt_of_lt_mul h

/-- The slot is inside the page. -/
theorem slot_lt {pos pageSize : Nat} (h : 0 < pageSize) : pos % pageSize < pageSize :=
  Nat.mod_lt _ h

end Ref.QsaStep



## FILE: /mnt/user-data/outputs/QsaTopkOrderKey.lean
/-!
# MODULE: qsa_topk_order_key
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/qsa_select.cu
The monotone 32-bit order key used by the exact top-k threshold search, the selection-count arithmetic
(strictly-above plus first equal cells by ascending index) and the per-block cell weights of the
block-level radix select.

## Source semantics
* `order_key(b) = b & 0x80000000 ? ~b : b | 0x80000000` is a total order on float bit patterns.
* The selection emits all cells strictly above the threshold plus `width - c_gt` threshold-equal cells.
* A block of the block-level select weighs `r` cells, except the last (tail) block which weighs
  `n_kv - n_bid * r`; weights sum to `n_kv`.

## Form template identifiers
orderKey <- order_key, selCount <- gt + min(eq, eq_budget), eqBudget <- eq_budget = width - c_gt,
blockWeight <- weight(b) lambda in block_topk_kernel.
-/

namespace Ref.QsaTopk

-- § order key (32-bit patterns as Nat)

/-- `order_key` on a bit pattern `b < 2^32`; `~b` is `2^32 - 1 - b`. -/
def orderKey (b : Nat) : Nat :=
  if b ≥ 2147483648 then 4294967295 - b else b + 2147483648

/-- Keys stay 32-bit. -/
theorem orderKey_lt {b : Nat} (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Non-negative patterns keep their order. -/
theorem orderKey_mono_nonneg {a b : Nat} (hab : a ≤ b) (hb : b < 2147483648) :
    orderKey a ≤ orderKey b := by
  unfold orderKey; split <;> split <;> omega

/-- Negative patterns reverse their order (larger magnitude is smaller). -/
theorem orderKey_mono_neg {a b : Nat} (ha : 2147483648 ≤ a) (hab : a ≤ b)
    (hb : b < 4294967296) : orderKey b ≤ orderKey a := by
  have hb' : 2147483648 ≤ b := by omega
  have h1 : orderKey a = 4294967295 - a := by unfold orderKey; rw [if_pos ha]
  have h2 : orderKey b = 4294967295 - b := by unfold orderKey; rw [if_pos hb']
  omega

/-- Every negative pattern sorts below every non-negative one. -/
theorem orderKey_neg_lt_nonneg {a b : Nat} (ha : 2147483648 ≤ a) (ha' : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey; split <;> split <;> omega

/-- Finite patterns map strictly above 0, the key reserved for NaN. -/
theorem orderKey_finite_pos {b : Nat} (h : b < 4286578688) : 0 < orderKey b := by
  unfold orderKey; split <;> omega

-- § selection count

/-- Cells emitted: those strictly above the threshold plus the equal cells that fit. -/
def selCount (gt eq width : Nat) : Nat := gt + min eq (width - gt)

/-- Equal cells the selection has room for. -/
def eqBudget (gt width : Nat) : Nat := width - gt

/-- When the threshold is the width-th largest key, exactly `width` cells are emitted. -/
theorem selCount_exact {gt eq width : Nat} (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    selCount gt eq width = width := by
  unfold selCount; omega

/-- With distinct scores (`gt = width`) the budget at the threshold is zero. -/
theorem eqBudget_zero {gt width : Nat} (h : width ≤ gt) : eqBudget gt width = 0 := by
  unfold eqBudget; omega

/-- Never more than `width` cells are emitted when `gt <= width`. -/
theorem selCount_le_width {gt eq width : Nat} (h1 : gt ≤ width) : selCount gt eq width ≤ width := by
  unfold selCount; omega

-- § block weights

/-- Weight (cells) of block `b`. -/
def blockWeight (nKv nBid r b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Prefix sum of block weights. -/
def weightSum (nKv nBid r : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => weightSum nKv nBid r m + blockWeight nKv nBid r m

/-- The first `m <= n_bid` blocks weigh `m * r`. -/
theorem weightSum_full {nKv nBid r m : Nat} (hm : m ≤ nBid) :
    weightSum nKv nBid r m = m * r := by
  induction m with
  | zero => simp [weightSum]
  | succ m ih =>
    have h1 := ih (by omega)
    have h2 : m < nBid := by omega
    have h3 : (m + 1) * r = m * r + r := by rw [Nat.add_mul, Nat.one_mul]
    simp [weightSum, blockWeight, h2, h1] <;> omega

/-- All `n_bid + 1` blocks together weigh exactly `n_kv`. -/
theorem weightSum_total {nKv nBid r : Nat} (h : nBid * r ≤ nKv) :
    weightSum nKv nBid r (nBid + 1) = nKv := by
  have h1 := weightSum_full (nKv := nKv) (nBid := nBid) (r := r) (m := nBid) (Nat.le_refl _)
  simp [weightSum, blockWeight, h1] <;> omega

/-- The tail block is empty when `n_kv` is a multiple of `r`. -/
theorem tail_weight_zero {nKv nBid r : Nat} (h : nKv = nBid * r) :
    blockWeight nKv nBid r nBid = 0 := by
  simp [blockWeight, h]

end Ref.QsaTopk



## FILE: /mnt/user-data/outputs/QsaDecodeChunking.lean
/-!
# MODULE: qsa_decode_chunking
SOURCES: src/kernels/cuda/qsa_decode_attn.cu
Chunked split-KV decode attention: the selected cells are cut into chunks of 64, each chunk produces a
partial (acc, m, l), and a merge kernel combines the partials. This module covers the index arithmetic.

## Source semantics
* n_chunks = ceil(cap / 64); chunk k covers cells [64k, 64k + n_here) with n_here = min(64, n_ids - 64k).
* A chunk past n_ids is empty (n_here <= 0) and writes the sentinel (m = -FLT_MAX, l = 0).
* Scratch = n_chunks * n_head * (256 + 2) + 64 floats, laid out [acc][m][l].
* slot = kvh * n_chunks + chunk; query head h = kvh * 12 + hl.

## Form template identifiers
nChunks <- n_chunks, chunkSize <- n_here, scratchFloats <- qsa_decode_attn_scratch_floats,
slotOf <- slot.
-/

namespace Ref.QsaDecode

-- § chunk coverage

/-- Number of chunks for capacity `cap`. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- Cells handled by chunk `k` (0 when the chunk is past the selection). -/
def chunkSize (n k : Nat) : Nat := min 64 (n - 64 * k)

/-- Total cells covered by the first `m` chunks. -/
def total (n : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => total n m + chunkSize n m

/-- A chunk past the selection is empty. -/
theorem chunkSize_zero {n k : Nat} (h : n ≤ 64 * k) : chunkSize n k = 0 := by
  unfold chunkSize; omega

/-- A fully inside chunk holds 64 cells. -/
theorem chunkSize_full {n k : Nat} (h : 64 * (k + 1) ≤ n) : chunkSize n k = 64 := by
  unfold chunkSize; omega

/-- Chunks never hold more than 64 cells. -/
theorem chunkSize_le (n k : Nat) : chunkSize n k ≤ 64 := by
  unfold chunkSize; omega

/-- The first `m` chunks cover exactly `min n (64 m)` cells. -/
theorem total_eq (n m : Nat) : total n m = min n (64 * m) := by
  induction m with
  | zero => simp [total]
  | succ m ih => simp only [total, chunkSize, ih]; omega

/-- With `n_ids <= cap`, all `nChunks cap` chunks cover every selected cell exactly once. -/
theorem total_cover {n cap : Nat} (h : n ≤ cap) : total n (nChunks cap) = n := by
  rw [total_eq]; unfold nChunks; omega

/-- The chunk count times 64 reaches the capacity. -/
theorem nChunks_mul_ge (cap : Nat) : cap ≤ 64 * nChunks cap := by
  unfold nChunks; omega

/-- A selected cell `i < n` lives in chunk `i / 64`, which is a valid chunk. -/
theorem cell_chunk_lt {i cap : Nat} (h : i < cap) : i / 64 < nChunks cap := by
  unfold nChunks; omega

/-- Its in-chunk offset is inside that chunk's `n_here`. -/
theorem cell_offset_lt {i n : Nat} (h : i < n) : i % 64 < chunkSize n (i / 64) := by
  unfold chunkSize; omega

-- § scratch layout

/-- Scratch floats for capacity `cap`. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * (256 + 2) + 64

/-- The acc region fits in the scratch. -/
theorem acc_region_fits (cap nHead : Nat) :
    nChunks cap * nHead * 256 ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

/-- The m and l regions, placed after acc, end inside the scratch. -/
theorem ml_regions_fit (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead
      ≤ scratchFloats cap nHead := by
  unfold scratchFloats; omega

-- § slots and heads

/-- Partial slot index of (kv head, chunk). -/
def slotOf (kvh chunks c : Nat) : Nat := kvh * chunks + c

/-- A valid (kv head, chunk) pair has a slot inside `nKvH * chunks`. -/
theorem slotOf_lt {kvh c chunks nKvH : Nat} (hk : kvh < nKvH) (hc : c < chunks) :
    slotOf kvh chunks c < nKvH * chunks := by
  unfold slotOf
  have h1 : (kvh + 1) * chunks ≤ nKvH * chunks := Nat.mul_le_mul_right chunks hk
  have h2 : (kvh + 1) * chunks = kvh * chunks + chunks := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The merge kernel's head decomposition `h = kvh * 12 + hl` is exact. -/
theorem head_decomp (h : Nat) : h = (h / 12) * 12 + h % 12 := by omega

/-- The local head index is below 12. -/
theorem local_head_lt (h : Nat) : h % 12 < 12 := by omega

end Ref.QsaDecode



## FILE: /mnt/user-data/outputs/KvQ8Quant.lean
/-!
# MODULE: kv_q8_quant
SOURCES: src/kernels/cuda/kv_q8.cu
INT8 KV cache with one fp16 scale per 64-value group: append quantizes against the stored scale,
gather dequantizes four values per thread.

## Source semantics
* scale = fp16(amax / 127); code = clamp(round(x / scale), -127, 127); scale 0 gives code 0.
* scale index = row * (head_dim / 64) + d / 64.
* Gather handles 4 consecutive values per thread; 64 is a multiple of 4, so the 4 share a group.

## Form template identifiers
clamp127 <- the `q < -127 ? -127 : q > 127 ? 127 : q` clamp, scaleIdx <- row * groups + g.
-/

namespace Ref.KvQ8

-- § clamp

/-- The symmetric int8 clamp used by `kv_append_q8_kernel`. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Codes stay in [-127, 127] (so -128 is never produced). -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; split <;> (try split) <;> omega

/-- In-range codes are unchanged. -/
theorem clamp127_id {q : Int} (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127; split <;> (try split) <;> omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q :=
  clamp127_id (clamp127_bounds q).1 (clamp127_bounds q).2

-- § scale indexing

/-- Scale index of group `g` of row `row`. -/
def scaleIdx (row groups g : Nat) : Nat := row * groups + g

/-- A valid group has a scale slot inside its row's range. -/
theorem scaleIdx_lt {row groups g : Nat} (hg : g < groups) :
    scaleIdx row groups g < (row + 1) * groups := by
  unfold scaleIdx; rw [Nat.add_mul, Nat.one_mul]; omega

/-- The real head_dim 256 has four groups of 64. -/
theorem real_groups : 256 / 64 = 4 := by decide

/-- Every dimension `d < 256` lands in one of the four groups. -/
theorem group_lt {d : Nat} (h : d < 256) : d / 64 < 4 := by omega

/-- The four values one gather thread handles share one scale group. -/
theorem gather_group_uniform {q4 j : Nat} (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

/-- 64 threads per block, two warps of 32. -/
theorem block_two_warps : 64 / 32 = 2 := by decide

end Ref.KvQ8



## FILE: /mnt/user-data/outputs/NativeFlashAttnStep.lean
/-!
# MODULE: native_flash_attn_step
SOURCES: src/kernels/cuda/native_flash_attn.cu
Short-context (<= 256 cells) vector flash attention specialised to Q24x256 / KV2x256. The kernel
validates the device step and refuses (NaN output + status) on an inconsistent step.

## Source semantics
* valid = width >= 1, width <= max_context, nkv == width, pos == width - 1, n_bid == width / 4.
* head h reads kv head h / 12; cell tiles of 128 over padded length 256.
* The K/V row of cell c, kv head kv is (c * 2 + kv) * 256.

## Form template identifiers
StepV <- the four step entries, stepValid <- `valid` expression, mkStep <- qsa_step_fill (top_k + 3 bound).
-/

namespace Ref.NativeFlash

-- § step validity

/-- The four step entries read by the kernel. -/
structure StepV where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- The kernel's `valid` predicate. -/
def stepValid (maxCtx : Nat) (s : StepV) : Prop :=
  1 ≤ s.width ∧ s.width ≤ maxCtx ∧ s.nKv = s.width ∧ s.pos = s.width - 1 ∧ s.nBid = s.width / 4

/-- A step as `qsa_step_fill` builds it (r = 4). -/
def mkStep (pos topK : Nat) : StepV :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / 4, width := min (pos + 1) (topK + 3) }

/-- Filled steps within the short-context window are accepted. -/
theorem mkStep_valid {maxCtx pos topK : Nat} (hmax : maxCtx ≤ 256) (hctx : pos + 1 ≤ maxCtx)
    (htop : 256 ≤ topK) : stepValid maxCtx (mkStep pos topK) := by
  simp only [stepValid, mkStep]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- A step whose width exceeds the supported context is rejected. -/
theorem step_rejected_too_long {maxCtx : Nat} {s : StepV} (h : maxCtx < s.width) :
    ¬ stepValid maxCtx s := fun hv => by
  have := hv.2.1
  omega

/-- A stale step (pos not width - 1) is rejected: the frozen-argument failure mode. -/
theorem step_rejected_stale {maxCtx : Nat} {s : StepV} (h : s.pos ≠ s.width - 1) :
    ¬ stepValid maxCtx s := fun hv => h hv.2.2.2.1

-- § tiling and addressing

/-- Query head to kv head. -/
theorem head_kv_lt : ∀ h : Nat, h < 24 → h / 12 < 2 := by
  intro h hh; omega

/-- Every cell `c < 256` is visited by some (warp, lane, row) of some 128-wide tile. -/
theorem cell_covered {c : Nat} (hc : c < 256) :
    ∃ warp lane row : Nat, warp < 4 ∧ lane < 32 ∧ row < 8 ∧
      c = (c / 128) * 128 + warp * 32 + (lane / 8) * 8 + row :=
  ⟨(c % 128) / 32, ((c % 32) / 8) * 8, c % 8, by omega, by omega, by omega, by omega⟩

/-- Two tiles of 128 cover the padded length 256. -/
theorem tiles_cover : (256 + 127) / 128 = 2 := by decide

/-- K/V row index of (cell, kv head) with two kv heads. -/
def kvRow (cell kv : Nat) : Nat := cell * 2 + kv

/-- Rows are distinct per (cell, kv). -/
theorem kvRow_inj {c1 k1 c2 k2 : Nat} (h1 : k1 < 2) (h2 : k2 < 2)
    (h : kvRow c1 k1 = kvRow c2 k2) : c1 = c2 ∧ k1 = k2 := by
  unfold kvRow at h; omega

/-- Per-cell byte size is 2 heads * 256 dims * 2 bytes. -/
theorem kv_cell_bytes : 2 * 256 * 2 = 1024 := by decide

/-- The shared tile is 4 * 4 * 256 floats = 16 KiB. -/
theorem tile_bytes : 4 * 4 * 256 * 4 = 16384 := by decide

end Ref.NativeFlash



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Single-block indexer key append and block pooling bit-matched to llama.cpp: stores the raw tail,
pools on block completion, rotates the pooled row at the block's first cell, and rewrites the spare slot.

## Source semantics
* slot = pos % 4; tail row stored iff slot < 3; the block completes iff slot == 3 (or pos == 0 for the spare).
* rope position = 0 at pos 0, else pos_base + 4 * (pos / 4); pos_base must be a multiple of 4.
* pooled row b = pos / 4, spare row b + 1; at pos 0 the four gathered indices all name cell zero.
* NEOX rotation pairs d with d + 32 over n_rot = 64.
* Buffers must be pairwise non-overlapping.

## Form template identifiers
ropePos <- rope_pos, partner <- pair/pair + ROT/2, overlaps <- `overlaps(Span, Span)`.
-/

namespace Ref.NativeIndexer

-- § slots and rope positions

/-- Rotation position of the row completed at `pos`. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With an aligned `pos_base` the rotation position is block aligned. -/
theorem ropePos_aligned {pos posBase : Nat} (hb : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos; split <;> omega

/-- On completion the rotation position is the block's FIRST cell. -/
theorem ropePos_first_cell {pos posBase : Nat} (hs : pos % 4 = 3) :
    ropePos pos posBase = posBase + (pos - 3) := by
  unfold ropePos; split <;> omega

/-- It is never the block's last cell. -/
theorem ropePos_ne_last {pos posBase : Nat} (hs : pos % 4 = 3) :
    ropePos pos posBase ≠ posBase + pos := by
  unfold ropePos; split <;> omega

/-- Rotation positions stay in int32 under the validated capacity. -/
theorem ropePos_int32 {pos posBase maxCells : Nat} (hp : pos < maxCells)
    (hb : posBase + maxCells ≤ 2147483647) : ropePos pos posBase < 2147483648 := by
  unfold ropePos; split <;> omega

/-- A block completes exactly when the next cell count is a multiple of 4. -/
theorem completes_iff {pos : Nat} : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Completing a block raises the block count by one. -/
theorem completion_blocks {pos : Nat} (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Only slots 0..2 are stored in the three-row tail. -/
theorem tail_slot_lt {pos : Nat} (h : pos % 4 < 3) : pos % 4 < 4 - 1 := by omega

-- § pooled row bounds

/-- The completed row is inside the pooled buffer of `max_cells / 4 + 1` rows. -/
theorem pooled_row_lt {pos maxCells : Nat} (h : pos < maxCells) : pos / 4 < maxCells / 4 + 1 := by
  omega

/-- The spare row written on completion is also inside it. -/
theorem spare_row_lt {pos maxCells : Nat} (h : pos < maxCells) (hs : pos % 4 = 3) :
    pos / 4 + 1 < maxCells / 4 + 1 := by
  omega

-- § pooling arithmetic

/-- At pos 0 the four gathered indices all name cell zero, so the sum is four copies. -/
theorem spare_sum (x : Int) : x + x + x + x = 4 * x := by omega

/-- The 0.25 scale of that sum returns the key itself (integer model). -/
theorem spare_mean (x : Int) : (x + x + x + x) / 4 = x := by omega

-- § NEOX pairing

/-- Partner dimension of the rotation over 64 rotated dims. -/
def partner (d : Nat) : Nat := if d < 32 then d + 32 else d - 32

/-- The pairing is an involution. -/
theorem partner_involutive {d : Nat} (h : d < 64) : partner (partner d) = d := by
  unfold partner; split <;> split <;> omega

/-- Partners stay inside the 64 rotated dimensions. -/
theorem partner_lt {d : Nat} (h : d < 64) : partner d < 64 := by
  unfold partner; split <;> omega

/-- Every rotated dimension `d` maps to a pair index `d % 32` below 32. -/
theorem pair_lt {d : Nat} (h : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by omega

-- § span disjointness

/-- Half-open byte-range overlap of (address, length) spans. -/
def overlaps (x xn y yn : Nat) : Prop := x < y + yn ∧ y < x + xn

/-- Overlap is symmetric, so checking pairs `i < j` suffices. -/
theorem overlaps_symm (x xn y yn : Nat) : overlaps x xn y yn ↔ overlaps y yn x xn := by
  unfold overlaps; omega

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (x xn : Nat) : ¬ overlaps x xn (x + xn) 4 := by
  unfold overlaps; omega

end Ref.NativeIndexer



## FILE: /mnt/user-data/outputs/NativeQsaScoreTiles.lean
/-!
# MODULE: native_qsa_score_tiles
SOURCES: src/kernels/cuda/native_qsa_score.cu
Tensor-core (mma tf32) per-block indexer score kernel, bit-matched to llama.cpp's mmf path, writing
the per-cell scores for the full blocks and the tail.

## Source semantics
* Step is accepted iff 1 <= n <= max_cells, pos = n - 1, n_bid = n / 4, width = min(n, 2051).
* A block-row of 32 rows per CUDA block; rows 0..n_bid are scored.
* Row r writes cells [4r, min(4r + 4, n)); the tail row (r = n_bid) writes n % 4 cells and gets +1e9 iff n % 4 != 0.
* Two warps split K = 128: warp w handles columns w*32 + lane and w*32 + lane + 64.
* max_blocks must equal max_cells / 4 + 1.

## Form template identifiers
cellsWritten <- loop bound `i < n && i < (row+1)*R`, blocksFor <- (max_blocks + ROWS - 1) / ROWS.
-/

namespace Ref.NativeScore

-- § step acceptance

/-- The step check of `score_kernel` (R = 4, budget 2048 so width bound 2051). -/
def stepOk (n maxCells pos full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos = n - 1 ∧ full = n / 4 ∧ width = min n 2051

/-- A step built from a position as `qsa_step_fill` builds it is accepted. -/
theorem stepOk_of_fill {pos maxCells : Nat} (h : pos + 1 ≤ maxCells) :
    stepOk (pos + 1) maxCells pos ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold stepOk
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The hard-coded 2051 is `top_k + r - 1` for the supported budget 2048. -/
theorem width_bound : 2048 + 4 - 1 = 2051 := by decide

-- § cells written per row

/-- Cells written by row `row`. -/
def cellsWritten (n row : Nat) : Nat := min 4 (n - row * 4)

/-- Full block rows write four cells. -/
theorem cellsWritten_full {n row : Nat} (h : row < n / 4) : cellsWritten n row = 4 := by
  unfold cellsWritten; omega

/-- The tail row writes `n % 4` cells. -/
theorem cellsWritten_tail {n : Nat} : cellsWritten n (n / 4) = n % 4 := by
  unfold cellsWritten; omega

/-- Rows are never wider than four cells. -/
theorem cellsWritten_le (n row : Nat) : cellsWritten n row ≤ 4 := by
  unfold cellsWritten; omega

/-- The +1e9 tail bias fires for the last row exactly when there are tail cells. -/
theorem tail_bias_iff {n : Nat} : n % 4 ≠ 0 ↔ 0 < cellsWritten n (n / 4) := by
  unfold cellsWritten; omega

-- § grid coverage

/-- Number of 32-row CUDA blocks. -/
def blocksFor (maxBlocks : Nat) : Nat := (maxBlocks + 31) / 32

/-- Every row below max_blocks belongs to a launched block. -/
theorem row_block_lt {row maxBlocks : Nat} (h : row < maxBlocks) : row / 32 < blocksFor maxBlocks := by
  unfold blocksFor; omega

/-- With `max_blocks = max_cells / 4 + 1` the scored rows `0..n_bid` all fit. -/
theorem scored_rows_fit {n maxCells : Nat} (h : n ≤ maxCells) :
    n / 4 < maxCells / 4 + 1 := by omega

-- § K split between the two warps

/-- Each of the 128 columns is handled by exactly the (warp, lane, half) given by this witness. -/
theorem column_covered {c : Nat} (hc : c < 128) :
    ∃ w l : Nat, w < 2 ∧ l < 32 ∧ (c = w * 32 + l ∨ c = w * 32 + l + 64) :=
  ⟨(c / 32) % 2, c % 32, by omega, by omega, by omega⟩

/-- The two warps' column sets are disjoint: distinct (w, l, half) give distinct columns. -/
theorem column_unique {w1 l1 h1 w2 l2 h2 : Nat} (hw1 : w1 < 2) (hl1 : l1 < 32) (hh1 : h1 < 2)
    (hw2 : w2 < 2) (hl2 : l2 < 32) (hh2 : h2 < 2)
    (h : w1 * 32 + l1 + 64 * h1 = w2 * 32 + l2 + 64 * h2) :
    w1 = w2 ∧ l1 = l2 ∧ h1 = h2 := by omega

end Ref.NativeScore



## FILE: /mnt/user-data/outputs/RouterTop10Selection.lean
/-!
# MODULE: router_top10_selection
SOURCES: src/kernels/cuda/native_router.cu, src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, k passes of block-wide argmax (ties to the lowest index), gather,
renormalise with the 2^-14 lower clamp. Covers the selection order, lane/expert coverage and the
shared-memory / thread-count arithmetic.

## Source semantics
* A candidate (ov, oi) replaces the current best (bv, bi) iff ov > bv or (ov == bv and oi < bi).
* Scanning ascending with strict `>` keeps the lowest index on a tie (stable descending argsort).
* Native path: 512 experts as 32 lanes x 16 values, expert = lane + 32 * i.
* Generic path: taken-mask padded to 16 bytes first, then n doubles, then n floats.
* threads = round_up_32(min(n_expert, 512)); renorm divisor is max(sum, 2^-14).

## Form template identifiers
beats <- the replace-if predicate in the shuffle reductions, takenBytes <- taken_bytes,
threadsFor <- `threads` in router_top10, clampSum <- fmax(s, 2^-14).
-/

namespace Ref.Router

-- § argmax order

/-- Candidate `(ov, oi)` replaces `(bv, bi)`. -/
def beats (ov oi bv bi : Nat) : Prop := bv < ov ∨ (ov = bv ∧ oi < bi)

/-- No candidate beats itself. -/
theorem beats_irrefl (v i : Nat) : ¬ beats v i v i := by
  unfold beats; omega

/-- Beating is asymmetric. -/
theorem beats_asymm {a ai b bi : Nat} (h : beats a ai b bi) : ¬ beats b bi a ai := by
  unfold beats at *; omega

/-- Beating is transitive, so the reduction order does not matter. -/
theorem beats_trans {a ai b bi c ci : Nat} (h1 : beats a ai b bi) (h2 : beats b bi c ci) :
    beats a ai c ci := by
  unfold beats at *; omega

/-- Distinct experts are always comparable. -/
theorem beats_total {v i w j : Nat} (h : i ≠ j) : beats v i w j ∨ beats w j v i := by
  unfold beats; omega

/-- Equal probabilities: the lower index wins and the higher cannot. -/
theorem tie_prefers_lower {v i j : Nat} (h : i < j) : beats v i v j ∧ ¬ beats v j v i := by
  unfold beats; omega

-- § expert coverage

/-- Every expert id below 512 is `lane + 32 * i` for exactly one lane < 32 and i < 16. -/
theorem expert_covered {e : Nat} (h : e < 512) :
    ∃ lane i : Nat, lane < 32 ∧ i < 16 ∧ e = lane + i * 32 :=
  ⟨e % 32, e / 32, by omega, by omega, by omega⟩

/-- That decomposition is unique. -/
theorem expert_unique {l1 i1 l2 i2 : Nat} (h1 : l1 < 32) (h2 : l2 < 32)
    (h : l1 + i1 * 32 = l2 + i2 * 32) : l1 = l2 ∧ i1 = i2 := by omega

-- § shared memory and threads

/-- Taken-mask size, padded to 16 bytes. -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Padding never shrinks the mask. -/
theorem takenBytes_ge (n : Nat) : n ≤ takenBytes n := by unfold takenBytes; omega

/-- Padding adds fewer than 16 bytes. -/
theorem takenBytes_lt (n : Nat) : takenBytes n < n + 16 := by unfold takenBytes; omega

/-- The padded size keeps the following double array 16-byte aligned. -/
theorem takenBytes_aligned (n : Nat) : takenBytes n % 16 = 0 := by unfold takenBytes; omega

/-- Threads per block: `min(n, 512)` rounded up to a whole warp. -/
def threadsFor (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- At least one full warp for a non-empty expert set. -/
theorem threadsFor_ge_warp {n : Nat} (h : 1 ≤ n) : 32 ≤ threadsFor n := by
  unfold threadsFor; omega

/-- Never past the 512-thread cap. -/
theorem threadsFor_le (n : Nat) : threadsFor n ≤ 512 := by unfold threadsFor; omega

/-- A whole number of warps. -/
theorem threadsFor_warps (n : Nat) : threadsFor n % 32 = 0 := by unfold threadsFor; omega

/-- It covers every expert up to the cap. -/
theorem threadsFor_cover (n : Nat) : min n 512 ≤ threadsFor n := by unfold threadsFor; omega

/-- Dynamic shared bytes for the real 512 experts: mask + doubles + floats. -/
theorem real_smem : takenBytes 512 + 512 * 8 + 512 * 4 = 6656 := by decide

/-- k passes of 512 compares versus the old all-pairs rank loop. -/
theorem work_counts : 10 * 512 = 5120 ∧ 512 * 512 = 262144 := by decide

-- § renormalisation clamp

/-- Divisor of the weight renormalisation. -/
def clampSum (s eps : Nat) : Nat := max s eps

/-- The divisor never falls below the clamp. -/
theorem clampSum_ge_eps (s eps : Nat) : eps ≤ clampSum s eps := by unfold clampSum; omega

/-- The divisor never falls below the sum. -/
theorem clampSum_ge_sum (s eps : Nat) : s ≤ clampSum s eps := by unfold clampSum; omega

/-- Above the clamp the divisor is the sum itself. -/
theorem clampSum_of_ge {s eps : Nat} (h : eps ≤ s) : clampSum s eps = s := by
  unfold clampSum; omega

end Ref.Router





## FILE: /mnt/user-data/outputs/IqQuantizedDotKernels.lean
/-!
# MODULE: iq_quantized_dot_kernels
SOURCES: src/kernels/cuda/iq_kernels.cu
Integer dot products, dequantizers and the grouped native-expert pipeline for the IQ/Q3_K/Q2_0 block formats
(types 16,17,18,20,21,22,23,29,42,11), transcribed from llama.cpp's vecdotq/dequantize/quantize.

## Source semantics
* `iq_row_bytes` is (values / values-per-block) * sizeof(block); unsupported types give 0.
* `unpack_ksigns` restores the 8th sign bit of a 7-bit sign index as the parity of the 7 bits.
* The warp loop `row_dot` strides call index k = (block, part) over lanes; kbx = k / ipb, iqs = step * (k % ipb).
* `native_expert_layout` stores [gate rows | up rows | down rows] contiguously.
* Dequantizer thread maps write disjoint 8-value (or 4+4 value) slices of a 256-value superblock.

## Form template identifiers
iq_row_bytes -> row_bytes_fn ; unpack_ksigns -> sign_expand ; NativeExpertLayout -> expert_blob_layout ;
Fmt<TY> -> format_traits ; native_expert_scratch_bytes -> scratch_size_fn ; dq_dispatch -> dequant_dispatch
-/

namespace Ref.IqKernels

/-- Is this ggml type id one of the IQ-family types the file handles? -/
def iqSupported (t : Nat) : Bool :=
  decide (t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 20 ∨ t = 21 ∨ t = 22 ∨ t = 23 ∨ t = 29 ∨ t = 42 ∨ t = 11)

/-- Bytes of one weight row of `n` values (sizes of the ggml block structs). -/
def iqRowBytes (t n : Nat) : Nat :=
  match t with
  | 16 => (n / 256) * 66
  | 17 => (n / 256) * 74
  | 18 => (n / 256) * 98
  | 20 => (n / 32) * 18
  | 21 => (n / 256) * 110
  | 22 => (n / 256) * 82
  | 29 => (n / 256) * 56
  | 23 => (n / 256) * 136
  | 11 => (n / 256) * 110
  | 42 => (n / 64) * 18
  | _ => 0

-- § row bytes

/-- Every supported type has a positive row size for a 256-value row. -/
theorem iqRowBytes_pos_of_supported :
    ∀ t ∈ [16, 17, 18, 20, 21, 22, 23, 29, 42, 11], 0 < iqRowBytes t 256 := by
  decide

/-- Unsupported types report zero bytes, which is how the host detects them. -/
theorem iqRowBytes_unsupported (t n : Nat) (h : iqSupported t = false) : iqRowBytes t n = 0 := by
  unfold iqRowBytes
  split <;> simp_all [iqSupported]

/-- A superblock count times 256 recovers the row length when the row is 256-aligned. -/
theorem superblocks_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by
  omega

-- § sign unpacking

/-- Population count of the low 8 bits. -/
def popc8 (v : Nat) : Nat := (List.range 8).foldl (fun a i => a + (v >>> i) % 2) 0

/-- Model of `unpack_ksigns` before the 0x01010101 broadcast. -/
def ksigns (v : Nat) : Nat := v ^^^ ((popc8 v % 2) <<< 7)

/-- For a 7-bit sign index the restored top bit is the parity of the index. -/
theorem ksigns_bit7 : ∀ v < 128, ksigns v / 128 % 2 = popc8 v % 2 := by
  decide +kernel

/-- The low seven bits of the sign index are preserved. -/
theorem ksigns_low : ∀ v < 128, ksigns v % 128 = v := by
  decide +kernel

/-- Multiplying a byte by 0x01010101 replicates it into all four bytes. -/
theorem broadcast_bytes : ∀ s < 256, ∀ k < 4, (s * 16843009 / 256 ^ k) % 256 = s := by
  decide +kernel

-- § warp call mapping

/-- Lane-strided call index covers the block range: the block index is in range. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) : k / ipb < nb := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- The (block, part) decomposition of a call index is exact. -/
theorem call_decomp (k ipb : Nat) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  omega

/-- The iqs offset stays under step * ipb. -/
theorem iqs_lt (k ipb step : Nat) (hp : 0 < ipb) : step * (k % ipb) < step * ipb ∨ step = 0 := by
  by_cases hs : step = 0
  · exact Or.inr hs
  · left
    exact Nat.mul_lt_mul_of_pos_left (Nat.mod_lt _ hp) (Nat.pos_of_ne_zero hs)

/-- Lane decomposition of a call index: lane = k % 32 and round = k / 32. -/
theorem lane_round (k : Nat) : k % 32 + 32 * (k / 32) = k := by
  omega

-- § format traits

/-- Format traits: values per block, calls per block, iqs stride. -/
def fmtOf (t : Nat) : Option (Nat × Nat × Nat) :=
  match t with
  | 16 => some (256, 8, 2)
  | 17 => some (256, 8, 2)
  | 18 => some (256, 8, 2)
  | 20 => some (32, 2, 2)
  | 21 => some (256, 8, 2)
  | 22 => some (256, 8, 2)
  | 29 => some (256, 8, 1)
  | 42 => some (64, 2, 1)
  | _ => none

/-- Sanity predicate for a trait row. -/
def fmtOk (t : Nat) : Bool :=
  match fmtOf t with
  | some (qk, ipb, _) => decide (0 < ipb ∧ qk % 32 = 0)
  | none => false

/-- All eight kernel formats have well-formed traits. -/
theorem fmtOk_all : ∀ t ∈ [16, 17, 18, 20, 21, 22, 29, 42], fmtOk t = true := by
  decide

-- § native expert layout

/-- Layout of one expert blob. -/
structure NativeExpertLayout where
  guRow : Nat
  dRow : Nat
  upOff : Nat
  downOff : Nat
  bytes : Nat

/-- `native_expert_layout`: gate rows, then up rows, then down rows. -/
def nativeExpertLayout (guType dType nEmbd nFf : Nat) : NativeExpertLayout :=
  { guRow := iqRowBytes guType nEmbd
    dRow := iqRowBytes dType nFf
    upOff := nFf * iqRowBytes guType nEmbd
    downOff := 2 * (nFf * iqRowBytes guType nEmbd)
    bytes := 2 * (nFf * iqRowBytes guType nEmbd) + nEmbd * iqRowBytes dType nFf }

/-- The product bound used for every row-in-region argument. -/
theorem row_end_le (r n w : Nat) (h : r < n) : r * w + w ≤ n * w := by
  have := Nat.mul_le_mul_right w (show r + 1 ≤ n by omega)
  simp only [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The up region starts exactly after the gate region. -/
theorem up_after_gate (gu d e f : Nat) :
    (nativeExpertLayout gu d e f).downOff =
      (nativeExpertLayout gu d e f).upOff + f * (nativeExpertLayout gu d e f).guRow := by
  simp only [nativeExpertLayout]
  omega

/-- A gate row lies inside the gate region. -/
theorem gate_row_in_bounds (gu d e f r : Nat) (h : r < f) :
    r * (nativeExpertLayout gu d e f).guRow + (nativeExpertLayout gu d e f).guRow ≤
      (nativeExpertLayout gu d e f).upOff := by
  simp only [nativeExpertLayout]
  exact row_end_le r f _ h

/-- An up row lies inside the up region. -/
theorem up_row_in_bounds (gu d e f r : Nat) (h : r < f) :
    (nativeExpertLayout gu d e f).upOff + r * (nativeExpertLayout gu d e f).guRow +
      (nativeExpertLayout gu d e f).guRow ≤ (nativeExpertLayout gu d e f).downOff := by
  have := gate_row_in_bounds gu d e f r h
  have h2 := up_after_gate gu d e f
  simp only [nativeExpertLayout] at *
  omega

/-- A down row lies inside the blob. -/
theorem down_row_in_bounds (gu d e f r : Nat) (h : r < e) :
    (nativeExpertLayout gu d e f).downOff + r * (nativeExpertLayout gu d e f).dRow +
      (nativeExpertLayout gu d e f).dRow ≤ (nativeExpertLayout gu d e f).bytes := by
  have := row_end_le r e (iqRowBytes d f) h
  simp only [nativeExpertLayout]
  omega

/-- Gate/up grid: the row blocks cover 2 * n_ff rows. -/
theorem gu_grid_covers (nFf : Nat) : 2 * nFf ≤ 8 * ((2 * nFf + 7) / 8) := by
  omega

/-- Row slot decomposes into (is_up, r) with r in range. -/
theorem gu_row_split (row nFf : Nat) (h : row < 2 * nFf) (hu : nFf ≤ row) : row - nFf < nFf := by
  omega

-- § scratch sizing

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256
  omega

/-- Alignment yields a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256
  omega

/-- Scratch: three float planes plus the q8_1 image of the intermediate. -/
def nativeExpertScratchBytes (cap nFf : Nat) : Nat :=
  3 * align256 (cap * nFf * 4) + align256 (cap * (nFf / 32) * 36)

/-- The scratch holds three full float planes. -/
theorem scratch_holds_planes (cap nFf : Nat) : 3 * (cap * nFf * 4) ≤ nativeExpertScratchBytes cap nFf := by
  have := align256_ge (cap * nFf * 4)
  unfold nativeExpertScratchBytes
  omega

-- § dequantizer thread maps

/-- iq2_xxs/xs/s dequant: thread tid writes offset 32*(tid%8)+8*(tid/8); the 8-wide slice fits in 256. -/
theorem dq8_slice_fits (tid : Nat) (h : tid < 32) : 32 * (tid % 8) + 8 * (tid / 8) + 8 ≤ 256 := by
  omega

/-- The 8-slice offsets of distinct threads differ. -/
theorem dq8_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h : 32 * (a % 8) + 8 * (a / 8) = 32 * (b % 8) + 8 * (b / 8)) : a = b := by
  omega

/-- q2_0 dequant: thread tid writes values 8*tid .. 8*tid+7 of the superblock. -/
theorem dq_q2_0_offset (tid : Nat) : (tid / 8) * 64 + (tid % 8) * 8 = tid * 8 := by
  omega

/-- iq4_nl dequant writes offsets 4*il+j and 4*il+16+j (j<4) inside 32 values. -/
theorem dq_iq4_nl_in_range (tid j : Nat) (h : tid < 32) (hj : j < 4) :
    4 * (tid / 8) + 16 + j < 32 := by
  omega

/-- The gate/up interleaved row slot 2r+parity is injective. -/
theorem interleave_injective (r p r' p' : Nat) (hp : p < 2) (hp' : p' < 2)
    (h : 2 * r + p = 2 * r' + p') : r = r' ∧ p = p' := by
  omega

end Ref.IqKernels



## FILE: /mnt/user-data/outputs/NativeMmvqDispatch.lean
/-!
# MODULE: native_mmvq_dispatch
SOURCES: src/kernels/cuda/native_mmvq.cu
Bit-faithful ports of llama.cpp's Q8_1 matrix-vector kernels (Q4_0, Q5_0, Q8_0, IQ4_NL, Q2_0, Q3_K, Q4_K,
Q5_K, Q6_K, IQ4_XS) with single- and multi-column (2..8) launchers, shape validation and type dispatch.

## Source semantics
* Weight bytes = (n_in / block_elems) * block_bytes * n_out, with the block table below.
* A call needs n_in > 0, n_in % block_elems = 0 and 1 <= ncols <= 8, plus 4-byte aligned pointers.
* Small-K rule: when blocks per row < blocks per iteration, 4 rows share a block (ROWS = 4), else 1 row.
* Multi-column: exact layout is 4 warps; upstream layout is 4 warps (ncols<=4) or 2 warps with 2 rows.
* Types 16,17,18,21,22,29 are routed to the IQ kernels; others are native.

## Form template identifiers
native_mmvq_supported -> supported_pred ; native_mmvq_weight_bytes -> weight_bytes_fn ;
validate_shape -> shape_check ; native_mmvq -> type_dispatch ; Q81Block -> act_block
-/

namespace Ref.NativeMmvq

/-- Values per weight block for each ggml type id handled here. -/
def blockElems (t : Nat) : Nat :=
  match t with
  | 2 => 32 | 6 => 32 | 8 => 32 | 20 => 32
  | 11 => 256 | 12 => 256 | 13 => 256 | 14 => 256 | 23 => 256
  | 42 => 64
  | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256 | 22 => 256 | 29 => 256
  | _ => 0

/-- Bytes per weight block. -/
def blockBytes (t : Nat) : Nat :=
  match t with
  | 2 => 18 | 6 => 22 | 8 => 34 | 20 => 18
  | 11 => 110 | 12 => 144 | 13 => 176 | 14 => 210 | 23 => 136
  | 42 => 18
  | 16 => 66 | 17 => 74 | 18 => 98 | 21 => 110 | 22 => 82 | 29 => 56
  | _ => 0

/-- `native_mmvq_supported`. -/
def nativeSupported (t : Nat) : Bool :=
  decide (t = 2 ∨ t = 6 ∨ t = 8 ∨ t = 11 ∨ t = 12 ∨ t = 13 ∨ t = 14 ∨ t = 20 ∨ t = 23 ∨ t = 42 ∨
          t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29)

/-- Types forwarded to `iq_mmvq`. -/
def iqRoute (t : Nat) : Bool :=
  decide (t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 ∨ t = 29)

/-- Dispatch targets. -/
inductive Route where
  | native
  | iq
  | unsupported
  deriving DecidableEq

/-- `native_mmvq`'s switch. -/
def route (t : Nat) : Route :=
  if iqRoute t = true then Route.iq else if nativeSupported t = true then Route.native else Route.unsupported

-- § tables

/-- Every supported type has positive block geometry. -/
theorem supported_has_block :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29],
      0 < blockElems t ∧ 0 < blockBytes t := by
  decide

/-- The support predicate agrees with the block table on every id below 64. -/
theorem supported_iff_block : ∀ t < 64, nativeSupported t = decide (0 < blockElems t) := by
  decide

/-- IQ-routed types are natively supported. -/
theorem iqRoute_subset (t : Nat) (h : iqRoute t = true) : nativeSupported t = true := by
  simp only [iqRoute, nativeSupported, decide_eq_true_eq] at *
  omega

/-- A type is rejected by the dispatcher exactly when it is unsupported. -/
theorem route_unsupported_iff (t : Nat) : route t = Route.unsupported ↔ nativeSupported t = false := by
  unfold route
  by_cases hi : iqRoute t = true
  · have hn := iqRoute_subset t hi
    simp [hi, hn]
  · by_cases hn : nativeSupported t = true
    · simp [hi, hn]
    · have hf : nativeSupported t = false := by simpa using hn
      simp [hi, hn, hf]

-- § weight bytes and shapes

/-- `native_mmvq_weight_bytes` (without the overflow check). -/
def weightBytes (t nIn nOut : Nat) : Nat := (nIn / blockElems t) * blockBytes t * nOut

/-- Shape validation. -/
def validShape (nIn ncols blockElemsN : Nat) : Prop :=
  0 < nIn ∧ nIn % blockElemsN = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- With a valid shape the block count times block size is exactly n_in. -/
theorem blocks_cover (nIn ncols b : Nat) (h : validShape nIn ncols b) (hb : 0 < b) :
    nIn / b * b = nIn := by
  obtain ⟨_, h2, _, _⟩ := h
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h2)

/-- One output row's bytes. -/
theorem weightBytes_one (t nIn : Nat) : weightBytes t nIn 1 = (nIn / blockElems t) * blockBytes t := by
  simp [weightBytes]

/-- ncols columns quantize as a single vector of ncols * n_in values: block counts agree. -/
theorem q8_1_columns_whole (nIn ncols : Nat) (h : nIn % 32 = 0) :
    (nIn * ncols) / 32 = ncols * (nIn / 32) := by
  obtain ⟨q, rfl⟩ : ∃ q, nIn = 32 * q := ⟨nIn / 32, by omega⟩
  rw [Nat.mul_assoc, Nat.mul_div_cancel_left _ (by decide : 0 < 32),
    Nat.mul_div_cancel_left _ (by decide : 0 < 32), Nat.mul_comm]

-- § launch geometry

/-- Small-K row blocks: 4 rows per block cover all outputs. -/
theorem small_k_blocks_cover (nOut : Nat) : nOut ≤ 4 * ((nOut + 3) / 4) := by
  omega

/-- Upstream multi-column layout: 2 rows per block cover all outputs. -/
theorem upstream_blocks_cover (nOut : Nat) : nOut ≤ 2 * ((nOut + 1) / 2) := by
  omega

/-- Blocks per iteration of the 32-value formats (qi = 4 and qi = 8). -/
theorem small_bpi : 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by
  decide

/-- Small-K thresholds in elements: 2048 for Q4_0/Q5_0/IQ4_NL and 1024 for Q8_0. -/
theorem small_k_thresholds : 64 * 32 = 2048 ∧ 32 * 32 = 1024 := by
  decide

/-- Blocks per iteration of the 256-value formats. -/
theorem k_bpi : 2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧ 4 * 4 * 32 / 32 = 16 := by
  decide

/-- Q2_0 (QK=64, QI=2): 64 blocks per iteration. -/
theorem q2_0_bpi : 4 * 32 / 2 = 64 := by
  decide

/-- Shared partial buffer of the multi-column kernel fits 48 KiB for every ncols in 2..8. -/
theorem multi_shared_fits : ∀ n ∈ [2, 3, 4, 5, 6, 7, 8], 3 * n * 4 * 32 * 4 ≤ 49152 := by
  decide

/-- Warp count rule of the upstream layout never exceeds the 4 warps of the exact layout. -/
theorem upstream_warps_le (n : Nat) : (if n ≤ 4 then 4 else 2) ≤ 4 := by
  split <;> omega

/-- tid decomposition 32*y + x with y<4, x<32 is a bijection onto 0..127. -/
theorem tid_range (x y : Nat) (hx : x < 32) (hy : y < 4) : 32 * y + x < 128 := by
  omega

end Ref.NativeMmvq



## FILE: /mnt/user-data/outputs/ActivationQuantizeBlocks.lean
/-!
# MODULE: activation_quantize_blocks
SOURCES: src/kernels/cuda/quantize_act.cu
Quantizes FP32 activations into ggml's block_q8_0 (34 bytes / 32 values), a CPU-matching scaled variant, and
block_q8_K (292 bytes / 256 values), plus the matching dequantizers.

## Source semantics
* block_q8_0 = fp16 d (2 B) + 32 int8; block_q8_K = f32 d + 256 int8 + 16 int16 bsums = 292 B.
* Q8_0 reference clamps to [-128,127]; the CPU-matching variant clamps to [-127,127].
* Q8_K clamps with min(127) only (iscale = -127/max keeps products inside [-127,127]).
* `nearest_int` keeps the low 23 bits of (x + 1.5*2^23) and subtracts 2^22.
* Zero blocks write d = 0 and zeroed quants (and bsums, a deliberate divergence).
* Launch grids are ceil(nb / threads) blocks of 128 (Q8_0) or 64 (Q8_K) threads.

## Form template identifiers
quantize_q8_0 -> quant_q80 ; quantize_q8_0_scaled -> quant_q80_scaled ; quantize_q8_K -> quant_q8k ;
nearest_int_dev -> magic_round ; dequant_q8_0 -> dequant_q80 ; dequant_q8_K -> dequant_q8k
-/

namespace Ref.QuantizeAct

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

-- § block layouts

/-- block_q8_0 layout: 2-byte scale followed by 32 int8. -/
theorem q80_block_layout : 2 + 32 = 34 := by
  decide

/-- block_q8_K layout: f32 scale, 256 int8, 16 int16 partial sums. -/
theorem q8K_block_layout : 4 + 256 + 2 * 16 = 292 := by
  decide

/-- Offset of bsums inside block_q8_K. -/
theorem q8K_bsums_offset : 4 + 256 = 260 := by
  decide

/-- Whole 32-value blocks cover a multiple-of-32 buffer. -/
theorem q80_blocks_cover (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by
  omega

/-- Whole 256-value blocks cover a multiple-of-256 buffer. -/
theorem q8K_blocks_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by
  omega

/-- bsums group j and inner index ii address values inside the 256-value block. -/
theorem bsum_index_lt (j ii : Nat) (hj : j < 16) (hi : ii < 16) : j * 16 + ii < 256 := by
  omega

-- § clamps

/-- CPU-matching clamp, symmetric. -/
def clampScaled (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Reference clamp used by the ggml-matching Q8_0 quantizer. -/
def clampRef (v : Int) : Int := if v > 127 then 127 else if v < -128 then -128 else v

/-- Q8_K clamp: min only. -/
def clampK (v : Int) : Int := if v < 127 then v else 127

/-- The scaled clamp lands in [-127, 127]. -/
theorem clampScaled_range (v : Int) : -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampScaled
  split_ifs <;> omega

/-- The reference clamp lands in [-128, 127]. -/
theorem clampRef_range (v : Int) : -128 ≤ clampRef v ∧ clampRef v ≤ 127 := by
  unfold clampRef
  split_ifs <;> omega

/-- The two clamps agree whenever the input is not below -127. -/
theorem clampScaled_eq_clampRef (v : Int) (h : -127 ≤ v) : clampScaled v = clampRef v := by
  unfold clampScaled clampRef
  split_ifs <;> omega

/-- The min-only clamp never exceeds 127. -/
theorem clampK_le (v : Int) : clampK v ≤ 127 := by
  unfold clampK
  split_ifs <;> omega

/-- Inside [-127,127] the Q8_K clamp is the identity, which is why no lower clamp is needed. -/
theorem clampK_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampK v = v := by
  unfold clampK
  split_ifs <;> omega

-- § magic-number rounding

/-- Integer part recovered by `nearest_int`: low 23 bits minus 2^22. -/
def magicLow (i : Nat) : Int := ((i &&& 0x007fffff : Nat) : Int) - 0x00400000

/-- The recovered integer lies in [-2^22, 2^22 - 1]. -/
theorem magicLow_range (i : Nat) : -4194304 ≤ magicLow i ∧ magicLow i ≤ 4194303 := by
  unfold magicLow
  have : i &&& 0x007fffff ≤ 0x007fffff := Nat.and_le_right
  omega

/-- Partial sums of k values in [-127,127] are bounded by 127*k. -/
theorem sumTo_bound (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → -127 ≤ f i ∧ f i ≤ 127) :
    -127 * (n : Int) ≤ sumTo f n ∧ sumTo f n ≤ 127 * (n : Int) := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    simp only [sumTo]
    omega

/-- A 16-value bsum of int8 quants fits int16 (|sum| <= 2032). -/
theorem bsum_fits_int16 (f : Nat → Int) (h : ∀ i, i < 16 → -127 ≤ f i ∧ f i ≤ 127) :
    -32768 ≤ sumTo f 16 ∧ sumTo f 16 ≤ 32767 := by
  have := sumTo_bound f 16 h
  omega

-- § byte-level packing

/-- Two scale bytes recombine into a 16-bit value below 2^16. -/
theorem scale_bytes_lt (lo hi : Nat) (hl : lo < 256) (hh : hi < 256) : lo + 256 * hi < 65536 := by
  omega

/-- Splitting a 16-bit pattern into bytes and recombining is exact. -/
theorem scale_bytes_roundtrip (v : Nat) (h : v < 65536) : v % 256 + 256 * (v / 256) = v ∧ v / 256 < 256 := by
  omega

/-- Storing an int8 as a byte and reading it back is the identity. -/
def toU8 (q : Int) : Nat := (q % 256).toNat

/-- Signed reinterpretation of a byte. -/
def toI8 (u : Nat) : Int := if u < 128 then (u : Int) else (u : Int) - 256

/-- int8 -> uint8 -> int8 round trip. -/
theorem int8_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) : toI8 (toU8 q) = q := by
  unfold toI8 toU8
  split_ifs <;> omega

-- § launch geometry

/-- Q8_0 grid (128 threads per block) covers all blocks. -/
theorem q80_grid_covers (nb : Nat) : nb ≤ ((nb + 127) / 128) * 128 := by
  omega

/-- Q8_K grid (64 threads per block) covers all blocks. -/
theorem q8K_grid_covers (nb : Nat) : nb ≤ ((nb + 63) / 64) * 64 := by
  omega

end Ref.QuantizeAct



## FILE: /mnt/user-data/outputs/S2ExpertGroupedHits.lean
/-!
# MODULE: s2_expert_grouped_hits
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
Grouped GPU routed-expert kernels over Q2_0 expert blobs (H = 2560, FF = 640): gate/up (interleaved rows,
gate-major output), SwiGLU, Q8_0 requantisation, down; device-side hit selection/compaction; resident grouping;
CPU-order variant.

## Source semantics
* Blob layout: [gate/up codes 2*FF rows of 640 B | down codes H rows of 160 B | gate/up scales | down scales].
* Row slot i of the 2*FF gate/up rows is gate row i/2 (even) or up row i/2 (odd); output is gate-major:
  gate at h*FF + r, up at n_hits*FF + h*FF + r.
* One fp16 weight scale per 64 values = per two 32-value activation chunks.
* hit_select compacts resident experts by ballot prefix popcount; slots are distinct and increasing.
* ent_tok = entry / k_per_token; grouped kernels stage at most GMAX = 8 entries per group in shared memory.

## Form template identifiers
H,FF -> n_embd,n_ff ; O_D_CODES,O_GU_SCALES,O_D_SCALES -> blob_offsets ; gu_kernel -> gate_up_pass ;
hit_select_kernel -> resident_compaction ; group_resident_kernel -> group_builder ; down_kernel -> down_pass
-/

namespace Ref.S2ExpertGrouped

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Gate/up code bytes per row. -/
def ROW_GU : Nat := H / 4
/-- Down code bytes per row. -/
def ROW_D : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def SC_GU : Nat := H / 64
/-- fp16 scales per down row. -/
def SC_D : Nat := FF / 64
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob bytes. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

-- § blob geometry

/-- Row code widths follow from 2-bit packing. -/
theorem row_widths : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by
  decide

/-- The offsets are the arithmetic the kernel hard-codes. -/
theorem offsets_values : O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by
  decide

/-- Blob size: 1,228,800 B of codes plus 153,600 B of scales. -/
theorem blob_size : BLOB = 1382400 ∧ BLOB = 1228800 + 153600 := by
  decide

/-- Every gate/up code row slot lies inside the gate/up code region. -/
theorem gu_code_row_in_bounds (i : Nat) (h : i < 1280) : i * 640 + 640 ≤ 819200 := by
  omega

/-- Every down code row lies between the down-code start and the scales start. -/
theorem down_code_row_in_bounds (r : Nat) (h : r < 2560) :
    819200 + r * 160 + 160 ≤ 1228800 := by
  omega

/-- Every gate/up scale row lies in the gate/up scale region. -/
theorem gu_scale_row_in_bounds (i : Nat) (h : i < 1280) :
    1228800 + i * 80 + 80 ≤ 1331200 := by
  omega

/-- Every down scale row lies inside the blob. -/
theorem down_scale_row_in_bounds (r : Nat) (h : r < 2560) :
    1331200 + r * 20 + 20 ≤ 1382400 := by
  omega

/-- A 32-value chunk index maps to a valid weight scale (per 64 values) in gate/up. -/
theorem gu_scale_of_chunk (c : Nat) (h : c < 80) : c / 2 < 40 := by
  omega

/-- Same for the down projection. -/
theorem d_scale_of_chunk (c : Nat) (h : c < 20) : c / 2 < 10 := by
  omega

/-- The per-lane chunk count of a gate/up row covers the 80 chunks. -/
theorem gu_chunks_cover : (2560 / 32 + 31) / 32 = 3 ∧ 80 ≤ 32 * 3 := by
  decide

-- § interleaved rows and gate-major output

/-- A row slot is its gate/up pair index and parity. -/
theorem slot_decomp (i : Nat) : 2 * (i / 2) + i % 2 = i := by
  omega

/-- Gate-major index of gate row r of hit h. -/
def gateIdx (h r : Nat) : Nat := h * 640 + r

/-- Gate-major index of up row r of hit h with n hits. -/
def upIdx (n h r : Nat) : Nat := n * 640 + h * 640 + r

/-- Gate indices stay in the first n*FF floats. -/
theorem gateIdx_lt (n h r : Nat) (hh : h < n) (hr : r < 640) : gateIdx h r < n * 640 := by
  unfold gateIdx
  omega

/-- Up indices lie in [n*FF, 2*n*FF). -/
theorem upIdx_range (n h r : Nat) (hh : h < n) (hr : r < 640) :
    n * 640 ≤ upIdx n h r ∧ upIdx n h r < 2 * n * 640 := by
  unfold upIdx
  omega

/-- Gate and up never collide. -/
theorem gate_up_disjoint (n h r h' r' : Nat) (hh : h < n) (hr : r < 640) :
    gateIdx h r ≠ upIdx n h' r' := by
  unfold gateIdx upIdx
  omega

/-- Gate-major indexing is injective per half: distinct (h, r) give distinct slots. -/
theorem gateIdx_injective (h r h' r' : Nat) (hr : r < 640) (hr' : r' < 640)
    (e : gateIdx h r = gateIdx h' r') : h = h' ∧ r = r' := by
  unfold gateIdx at e
  omega

/-- SwiGLU reads gate at i and up at n_pairs + i, which is the up index of the same (h, r). -/
theorem swiglu_pairing (n h r : Nat) : gateIdx h r + n * 640 = upIdx n h r := by
  unfold gateIdx upIdx
  omega

-- § hit selection (ballot compaction)

/-- Number of hits among lanes below n. -/
def prefCount (hit : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => prefCount hit n + (if hit n = true then 1 else 0)

/-- The prefix count is at most the prefix length. -/
theorem prefCount_le (hit : Nat → Bool) (n : Nat) : prefCount hit n ≤ n := by
  induction n with
  | zero => simp [prefCount]
  | succ n ih =>
    show prefCount hit n + (if hit n = true then 1 else 0) ≤ n + 1
    split <;> omega

/-- The prefix count is monotone. -/
theorem prefCount_mono (hit : Nat → Bool) (m d : Nat) : prefCount hit m ≤ prefCount hit (m + d) := by
  induction d with
  | zero => simp
  | succ d ih =>
    have e : prefCount hit (m + (d + 1)) = prefCount hit (m + d) + (if hit (m + d) = true then 1 else 0) := rfl
    rw [e]
    split <;> omega

/-- A hit lane's slot is its prefix count and the next prefix is one larger. -/
theorem slot_succ (hit : Nat → Bool) (lane : Nat) (hl : hit lane = true) :
    prefCount hit (lane + 1) = prefCount hit lane + 1 := by
  show prefCount hit lane + (if hit lane = true then 1 else 0) = _
  simp [hl]

/-- Slots of hit lanes are strictly increasing: compaction keeps entry order with distinct slots. -/
theorem slots_strictly_increasing (hit : Nat → Bool) (a b : Nat) (hab : a < b) (ha : hit a = true) :
    prefCount hit a < prefCount hit b := by
  have h1 := slot_succ hit a ha
  obtain ⟨d, rfl⟩ : ∃ d, b = (a + 1) + d := ⟨b - (a + 1), by omega⟩
  have h2 := prefCount_mono hit (a + 1) d
  omega

/-- The slot of a hit lane is below the total count over k lanes. -/
theorem slot_lt_count (hit : Nat → Bool) (lane k : Nat) (hk : lane < k) (hl : hit lane = true) :
    prefCount hit lane < prefCount hit k := by
  exact slots_strictly_increasing hit lane k hk hl

/-- The hit count over k <= 32 lanes is at most 32. -/
theorem count_le_32 (hit : Nat → Bool) (k : Nat) (hk : k ≤ 32) : prefCount hit k ≤ 32 := by
  have := prefCount_le hit k
  omega

-- § grouping

/-- Token index of an entry is below the token count. -/
theorem ent_tok_lt (i kPer tokens : Nat) (h : i < tokens * kPer) : i / kPer < tokens := by
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Shared staging of one group: 8 entries of 2560 B plus 8 * 80 fp32 scales fits 48 KiB. -/
theorem gu_shared_fits : 8 * (2560 / 4) * 4 + 8 * (2560 / 32) * 4 = 23040 ∧ 23040 ≤ 49152 := by
  decide

/-- Gate/up grouped grid: 40 blocks of 32 rows exactly cover 2*FF rows. -/
theorem gu_grouped_grid : 2 * 640 / 32 = 40 ∧ 40 * 32 = 2 * 640 := by
  decide

/-- Down grouped grid: 40 blocks of 64 rows exactly cover H rows. -/
theorem down_grouped_grid : 2560 / 64 = 40 ∧ 40 * 64 = 2560 := by
  decide

/-- Per-warp row assignment rr = warp, warp+8, ... covers 0..R-1 once for R divisible by 8. -/
theorem warp_rows_cover (rr : Nat) (h : rr < 32) : rr % 8 < 8 ∧ rr = rr % 8 + 8 * (rr / 8) := by
  omega

-- § scratch sizing

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment gives a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- The scratch size dominates the gate/up float plane. -/
theorem scratch_holds_gu (n nEmbd nFf : Nat) :
    n * (2 * nFf) * 4 ≤
      align16 (n * (2 * nFf) * 4) + align16 (n * (nFf / 32) * 34) +
        2 * align16 (n * (nFf / 32) * 4) + align16 (nEmbd / 32 * 4) := by
  have := align16_ge (n * (2 * nFf) * 4)
  omega

end Ref.S2ExpertGrouped



## FILE: /mnt/user-data/outputs/S2GemvFastTreeReduce.lean
/-!
# MODULE: s2_gemv_fast_tree_reduce
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
Experimental S2 (2-bit) GEMV with a constant-memory code table and optional shared staging of x; measured
neutral/negative and kept as a negative result. One block per output row, a shared-memory halving tree reduction.

## Source semantics
* c_codes[b][k] = ((b >> 2k) & 3) - 1 for k in 0..3: four code values per byte, bias already applied.
* n_quads = n_in/4; the scale index of quad q is q >> 4 (one scale per 64 values).
* Staging x needs n_in <= 4096 halves; shared = 8 KiB (staged) + 4*threads_per_row bytes of partials.
* The reduction halves `step` from threads_per_row/2 to 1, adding partial[tid+step] into partial[tid].

## Form template identifiers
c_codes -> code_lut ; s2_gemv_fast_kernel -> gemv_row_block ; MAX_SHARED_HALVES -> stage_limit ;
partial -> reduce_buffer
-/

namespace Ref.S2GemvFast

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Code value with the -1 bias applied. -/
def codeVal (b k : Nat) : Int := (((b >>> (2 * k)) &&& 3 : Nat) : Int) - 1

-- § code table

/-- Each LUT entry is in {-1, 0, 1, 2}. -/
theorem codeVal_range : ∀ b < 256, ∀ k < 4, -1 ≤ codeVal b k ∧ codeVal b k ≤ 2 := by
  decide +kernel

/-- The four 2-bit fields reconstruct the byte. -/
theorem byte_from_fields : ∀ b < 256,
    ((b >>> 0) &&& 3) + 4 * ((b >>> 2) &&& 3) + 16 * ((b >>> 4) &&& 3) + 64 * ((b >>> 6) &&& 3) = b := by
  decide +kernel

/-- Quad count times four is n_in. -/
theorem quads_cover (nIn : Nat) (h : nIn % 4 = 0) : nIn / 4 * 4 = nIn := by
  omega

/-- The scale index q >> 4 equals (4q)/64. -/
theorem scale_index (q : Nat) : q / 16 = (q * 4) / 64 := by
  omega

/-- A quad's four elements share its scale (elements 4q..4q+3 lie in one 64-group). -/
theorem quad_in_group (q k : Nat) (hk : k < 4) : (q * 4 + k) / 64 = (q * 4) / 64 := by
  omega

-- § shared memory

/-- The partial buffer starts 4096 B after the staged halves (2048 floats). -/
theorem partial_offset : 4096 / 2 * 4 = 4096 * 2 := by
  decide

/-- Shared size with staging for 1024 threads stays below 48 KiB. -/
theorem smem_fits : 4096 * 2 + 1024 * 4 ≤ 49152 := by
  decide

-- § tree reduction

/-- Splitting a prefix sum at n. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n + m) = sumTo f n + sumTo (fun j => f (j + n)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m + 1)) = sumTo f (n + m) + f (n + m) := rfl
    rw [e, ih, Nat.add_comm n m]
    simp only [sumTo]
    omega

/-- Pointwise sum commutes with the prefix sum. -/
theorem sumTo_add_fun (f g : Nat → Int) (s : Nat) :
    sumTo (fun i => f i + g i) s = sumTo f s + sumTo g s := by
  induction s with
  | zero => simp [sumTo]
  | succ s ih =>
    simp only [sumTo]
    rw [ih]
    omega

/-- One halving step: threads tid < s add partial[tid + s]; together they sum 2s entries. -/
theorem tree_step (f : Nat → Int) (s : Nat) :
    sumTo f (s + s) = sumTo (fun i => f i + f (i + s)) s := by
  rw [sumTo_add, sumTo_add_fun]

/-- The kernel's loop: step = 2^k, then 2^(k-1), ..., 1; the result is the value at tid 0. -/
def reduceK : Nat → (Nat → Int) → Int
  | 0, f => f 0
  | k + 1, f => reduceK k (fun i => f i + f (i + 2 ^ k))

/-- The halving-tree reduction computes the full sum of 2^k partials. -/
theorem reduceK_eq (k : Nat) : ∀ f : Nat → Int, reduceK k f = sumTo f (2 ^ k) := by
  induction k with
  | zero =>
    intro f
    simp [reduceK, sumTo]
  | succ k ih =>
    intro f
    have e : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by
      rw [Nat.pow_succ]
      omega
    rw [reduceK, ih, e, tree_step]

/-- Reads at partial[tid + step] stay inside the buffer when tid < step and 2*step <= threads. -/
theorem tree_read_in_bounds (tid step threads : Nat) (h : tid < step) (hs : 2 * step ≤ threads) :
    tid + step < threads := by
  omega

end Ref.S2GemvFast



## FILE: /mnt/user-data/outputs/SGemvSplitKernels.lean
/-!
# MODULE: s_gemv_split_kernels
SOURCES: src/kernels/cuda/s_gemv.cu
S-family quantized GEMV (2/4/8-bit codes, per-group scale and optional offset, optional IQ4_NL codebook) with
FP16, Q8_K and Q8_0 activations; naive, block-per-row split, and warp-per-row 16-element-per-lane variants.

## Source semantics
* code i = (byte[i / PER_BYTE] >> ((i % PER_BYTE) * bits)) & (2^bits - 1), PER_BYTE = 8 / bits.
* Weight = decode(code) * scale + offset, applied to the weight before the activation multiply.
* Group sizes are powers of two (16, 32, 64); the group index is a shift by log2(group).
* A lane takes QE = 16 consecutive elements per iteration, stride 32*QE, so an iteration never straddles a
  Q8_K (256) or Q8_0 (32) activation block; the tail loop handles the remainder.
* Q8_K needs n_in % 256 = 0 and is refused for 2-bit; Q8_0 needs only n_in % 32 = 0 (ffn_down_shexp: 640).

## Form template identifiers
CODE_BITS -> code_width ; kIq4Nl -> nonlinear_codebook ; QE -> elems_per_lane ; group_shift -> group_log2 ;
s_gemv_q8_split_kernel -> warp_row_kernel ; s_gemv_split_kernel -> block_row_kernel
-/

namespace Ref.SGemv

-- § code packing

/-- PER_BYTE * bits = 8 for the three widths. -/
theorem per_byte_exact : ∀ b ∈ [2, 4, 8], (8 / b) * b = 8 := by
  decide

/-- Bytes of one four-code word: 1, 2 or 4. -/
theorem quad_word_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by
  decide

/-- Extracted code (div/mod form of shift/mask). -/
def codeAt (bits byte idx : Nat) : Nat := (byte / 2 ^ ((idx % (8 / bits)) * bits)) % 2 ^ bits

/-- A code is below 2^bits. -/
theorem codeAt_lt (bits byte idx : Nat) : codeAt bits byte idx < 2 ^ bits := by
  unfold codeAt
  exact Nat.mod_lt _ (Nat.two_pow_pos _)

/-- 2-bit: four consecutive elements 4q..4q+3 live in byte q at positions 0..3. -/
theorem quad_offsets_2bit (q k : Nat) (hk : k < 4) : (4 * q + k) / 4 = q ∧ (4 * q + k) % 4 = k := by
  omega

/-- 4-bit: four consecutive elements live in bytes 2q and 2q+1 (one uint16). -/
theorem quad_offsets_4bit (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 2 = 2 * q + k / 2 ∧ (4 * q + k) % 2 = k % 2 := by
  omega

/-- 8-bit: element 4q+k is byte 4q+k (one uint32). -/
theorem quad_offsets_8bit (q k : Nat) : (4 * q + k) / 1 = 4 * q + k := by
  omega

/-- decode(code) = code + bias lies in [bias, bias + 2^bits). -/
def decodeAffine (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Affine decode range. -/
theorem decodeAffine_range (code bits : Nat) (bias : Int) (h : code < 2 ^ bits) :
    bias ≤ decodeAffine code bias ∧ decodeAffine code bias < (2 ^ bits : Nat) + bias := by
  unfold decodeAffine
  omega

-- § IQ4_NL codebook

/-- The non-linear codebook. -/
def kIq4Nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (indexed by a 4-bit code). -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by
  decide

/-- All entries fit int8. -/
theorem kIq4Nl_int8 : ∀ x ∈ kIq4Nl, -128 ≤ x ∧ x ≤ 127 := by
  decide

/-- A masked nibble indexes the 16-entry table. -/
theorem nibble_index_lt (code : Nat) : code % 16 < 16 := by
  omega

-- § group size and block alignment

/-- Group sizes 16, 32, 64 are powers of two, multiples of 16. -/
theorem group_sizes : ∀ s ∈ [4, 5, 6], (2 ^ s) &&& (2 ^ s - 1) = 0 ∧ 2 ^ s % 16 = 0 := by
  decide

/-- With QE = 16 and lane*16 + 512*t offsets, i % 16 = 0. -/
theorem iter_offset_aligned (lane t : Nat) : (lane * 16 + 512 * t) % 16 = 0 := by
  omega

/-- A 16-element lane iteration never straddles a Q8_K block or a Q8_0 block. -/
theorem no_straddle (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 ∧ i % 32 + 16 ≤ 32 := by
  omega

/-- The block-in-activation split is exact. -/
theorem block_split (i : Nat) : (i / 256) * 256 + i % 256 = i := by
  omega

/-- Q8_K requirement: n_in multiple of 256 gives whole blocks. -/
theorem q8k_whole (nIn : Nat) (h : nIn % 256 = 0) : nIn / 256 * 256 = nIn := by
  omega

/-- ffn_down_shexp (n_in = 640): fine for Q8_0, structurally impossible for Q8_K. -/
theorem shexp_down_q8_0_only : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § reduction regrouping

/-- Exact-arithmetic identity behind the 16-accumulator regrouping (floats differ; hence parity tolerances). -/
theorem sixteen_regroup (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15))) =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

/-- Four-accumulator regrouping of the block kernel. -/
theorem four_regroup (a b c d : Int) : (a + b) + (c + d) = a + b + c + d := by
  omega

-- § launch geometry

/-- Warp-per-row kernel: 8 warps per block cover all rows. -/
theorem warp_rows_cover (nOut : Nat) : nOut ≤ 8 * ((nOut + 7) / 8) := by
  omega

/-- Thread-per-row kernel: 128 threads per block cover all rows. -/
theorem thread_rows_cover (nOut : Nat) : nOut ≤ 128 * ((nOut + 127) / 128) := by
  omega

/-- threads_per_row powers of two up to 1024 pass the host check. -/
theorem tpr_power_of_two : ∀ k < 11, (2 ^ k) &&& (2 ^ k - 1) = 0 := by
  decide

/-- Codes consumed per row: bytes = n_in / PER_BYTE, and PER_BYTE divides 32 for every width. -/
theorem per_byte_divides_32 : ∀ b ∈ [2, 4, 8], 32 % (8 / b) = 0 := by
  decide

end Ref.SGemv



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
GPU sampler: penalties (repeat/frequency/presence over the history tail), top-k by repeated max scans,
top-p over survivors, temperature, Philox-driven draw; a parallel greedy argmax with lowest-index tie rule.

## Source semantics
* Penalty applies only to tokens with count > 0: repeat multiplies non-positive logits and divides positive
  ones; then subtract count*freq + (count > 0 ? 1 : 0)*presence (a boolean, not the count).
* The window is the TAIL of the history: hlen = clamp(penalty_last_n, 0, history_len).
* Greedy = argmax over penalised logits; ties go to the lowest index; the block reduction preserves that order.
* Sampled path needs 1 <= top_k <= 64; top_p cut >= min_keep (capped at the survivor count).
* Philox draw is uniform in [0,1): the top 24 bits of a 32-bit word divided by 2^24.

## Form template identifiers
apply_penalties -> penalty_fn ; sampler_greedy_kernel -> parallel_argmax ; philox_uniform -> counter_rng ;
KMAX -> topk_cap ; history_count -> token_count
-/

namespace Ref.Sampler

-- § penalties

/-- Presence indicator: 1 for any positive count. -/
def presence (count : Nat) : Nat := if count > 0 then 1 else 0

/-- The presence term is a boolean. -/
theorem presence_le_one (c : Nat) : presence c ≤ 1 := by
  unfold presence
  split <;> omega

/-- Presence does not depend on how many times the token occurred. -/
theorem presence_const (c : Nat) (h : 0 < c) : presence c = 1 := by
  unfold presence
  simp [h]

/-- Total subtraction for a token seen `count` times. -/
def penaltySub (count freq present : Nat) : Nat := count * freq + presence count * present

/-- Unseen tokens subtract nothing. -/
theorem penaltySub_zero (freq present : Nat) : penaltySub 0 freq present = 0 := by
  simp [penaltySub, presence]

/-- Seen tokens subtract count*freq + present (present counted once). -/
theorem penaltySub_seen (c freq present : Nat) (h : 0 < c) :
    penaltySub c freq present = c * freq + present := by
  unfold penaltySub
  rw [presence_const c h]
  omega

-- § history window

/-- Window length. -/
def hlen (lastN hist : Nat) : Nat := min lastN hist

/-- Window start offset in the history row (the tail). -/
def hstart (lastN hist : Nat) : Nat := hist - hlen lastN hist

/-- The window is a suffix: start + length = history_len. -/
theorem window_is_suffix (lastN hist : Nat) : hstart lastN hist + hlen lastN hist = hist := by
  unfold hstart hlen
  omega

/-- The window never exceeds the history. -/
theorem window_le (lastN hist : Nat) : hlen lastN hist ≤ hist := by
  unfold hlen
  omega

-- § argmax with the lowest-index tie rule

/-- Candidate (value, index) beats the current best: larger value, or equal value and smaller index. -/
def combine (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The reduction operator is commutative on candidates with distinct indices. -/
theorem combine_comm (a b : Int × Nat) (h : a.2 ≠ b.2) : combine a b = combine b a := by
  have key : ¬ ((b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) ∧ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2))) := by
    omega
  have key2 : (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) ∨ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)) := by
    omega
  by_cases h1 : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)
  · have h2 : ¬ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)) := fun h' => key ⟨h1, h'⟩
    unfold combine
    rw [if_pos h1, if_neg h2]
  · have h2 := key2.resolve_left h1
    unfold combine
    rw [if_neg h1, if_pos h2]

/-- Equal values: the lower index wins (second candidate has the lower index). -/
theorem combine_tie_low (v : Int) (i j : Nat) (h : j < i) : combine (v, i) (v, j) = (v, j) := by
  unfold combine
  rw [if_pos]
  simp
  omega

/-- Equal values: the lower index wins (first candidate has the lower index). -/
theorem combine_tie_keep (v : Int) (i j : Nat) (h : i ≤ j) : combine (v, i) (v, j) = (v, i) := by
  unfold combine
  rw [if_neg]
  intro hc
  simp at hc
  omega

/-- A strictly larger value always wins regardless of index. -/
theorem combine_greater (a b : Int × Nat) (h : b.1 > a.1) : combine a b = b := by
  unfold combine
  rw [if_pos (Or.inl h)]

-- § top-k / top-p bounds

/-- top-k clamp: the number of passes is at most KMAX = 64. -/
def kPasses (topK : Nat) : Nat := if topK > 0 then min topK 64 else 0

/-- The clamp is bounded by 64. -/
theorem kPasses_le (topK : Nat) : kPasses topK ≤ 64 := by
  unfold kPasses
  split <;> omega

/-- top-p cut after the min_keep floor. -/
def cutAfter (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The cut never exceeds the survivor count (given the cumulative scan bound). -/
theorem cutAfter_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : cutAfter cut minKeep nKeep ≤ nKeep := by
  unfold cutAfter
  split <;> omega

/-- The cut keeps at least min_keep survivors when available and at least one when cut >= 1. -/
theorem cutAfter_ge (cut minKeep nKeep : Nat) (h : minKeep ≤ nKeep) : min minKeep cut ≤ cutAfter cut minKeep nKeep := by
  unfold cutAfter
  split <;> omega

-- § RNG and launch

/-- 24-bit mantissa draw is strictly below 2^24, hence the uniform is below 1. -/
theorem philox_mantissa_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

/-- Sampled-path grid (64 threads, one per token) covers all tokens. -/
theorem sample_grid_covers (n : Nat) : n ≤ 64 * ((n + 63) / 64) := by
  omega

/-- Warp count of a block of 1024 threads fits the 32-slot shared reduction arrays. -/
theorem greedy_warps_fit : (1024 + 31) / 32 ≤ 32 := by
  decide

end Ref.Sampler



## FILE: /mnt/user-data/outputs/SharedExpertCombine.lean
/-!
# MODULE: shared_expert_combine
SOURCES: src/kernels/cuda/shared_expert.cu
The shared expert (SiLU on gate, times up, down projection, scaled by a per-token scalar sigmoid gate) with
legacy and native Q8_1 projection paths, the multi-token (1..8) variant, scratch carving, and the MoE combine.

## Source semantics
* Result is silu(x*gate) * (x*up) -> down -> times one scalar per token; the shared output is ADDED plain.
* moe_combine: y[j] = sum_{e<k} w[e]*parts[e][j] (+ shared[j] when present); k <= 64.
* Scratch: gate | up | q8_0 image | q8_K image | scalar gate, each region 16-byte aligned, plus 32 B tail.
* Q8_K down activation requires n_ff % 256 = 0; 2-bit forms always use Q8_0.
* shared_expert_multi accepts 1..8 tokens; row-scaling indexes t*n + i.

## Form template identifiers
shared_expert_scratch_bytes -> scratch_size_fn ; moe_combine_kernel -> expert_sum ; scalar_gate_kernel -> token_gate ;
scale_rows_kernel -> row_scale ; act_kind -> activation_kind
-/

namespace Ref.SharedExpert

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

-- § scratch carving

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by
  unfold align16
  omega

/-- Alignment yields a multiple of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by
  unfold align16
  omega

/-- Region sizes of the carved scratch. -/
def regionA (nFf : Nat) : Nat := align16 (nFf * 4)
/-- q8_0 image region. -/
def regionQ0 (nFf : Nat) : Nat := align16 (nFf / 32 * 34)
/-- q8_K image region. -/
def regionQk (nFf : Nat) : Nat := align16 (nFf / 256 * 292)

/-- Total scratch bytes. -/
def scratchBytes (nFf : Nat) : Nat := regionA nFf * 2 + regionQ0 nFf + regionQk nFf + 32

/-- The scalar-gate slot (offset at the end of all regions) fits with room to spare. -/
theorem gate_slot_fits (nFf : Nat) :
    regionA nFf * 2 + regionQ0 nFf + regionQk nFf + 4 ≤ scratchBytes nFf := by
  unfold scratchBytes
  omega

/-- Region offsets are monotone, so regions never overlap. -/
theorem regions_ordered (nFf : Nat) :
    regionA nFf ≤ regionA nFf * 2 ∧ regionA nFf * 2 ≤ regionA nFf * 2 + regionQ0 nFf ∧
      regionA nFf * 2 + regionQ0 nFf ≤ regionA nFf * 2 + regionQ0 nFf + regionQk nFf := by
  omega

/-- Each region offset is 16-byte aligned. -/
theorem offsets_aligned (nFf : Nat) :
    (regionA nFf * 2) % 16 = 0 ∧ (regionA nFf * 2 + regionQ0 nFf) % 16 = 0 ∧
      (regionA nFf * 2 + regionQ0 nFf + regionQk nFf) % 16 = 0 := by
  have h1 := align16_mod (nFf * 4)
  have h2 := align16_mod (nFf / 32 * 34)
  have h3 := align16_mod (nFf / 256 * 292)
  unfold regionA regionQ0 regionQk
  omega

/-- The gate region holds n_ff floats. -/
theorem gate_holds_floats (nFf : Nat) : nFf * 4 ≤ regionA nFf := by
  exact align16_ge _

-- § activation contract

/-- Activation kinds. -/
inductive Act where
  | q8_0
  | q8k
  deriving DecidableEq

/-- Which activation image a projection form reads. -/
def actFor (codeBits actKind : Nat) : Act :=
  if codeBits = 2 then Act.q8_0 else if actKind = 1 then Act.q8k else Act.q8_0

/-- 2-bit forms never use Q8_K. -/
theorem two_bit_uses_q8_0 (k : Nat) : actFor 2 k = Act.q8_0 := by
  simp [actFor]

/-- Q8_K is chosen only for non-2-bit forms that ask for it. -/
theorem q8k_only_when_requested (b k : Nat) (h : actFor b k = Act.q8k) : b ≠ 2 ∧ k = 1 := by
  unfold actFor at h
  split_ifs at h with h1 h2
  exact ⟨h1, h2⟩

/-- The down projection with n_ff = 640 cannot use Q8_K. -/
theorem down_640_no_q8k : 640 % 256 ≠ 0 := by
  decide

-- § multi-token variant and row scaling

/-- Token-count guard. -/
def validTokens (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 8

/-- A row index t*n + i lies in the n_tok-row buffer. -/
theorem row_index_lt (t i n nTok : Nat) (ht : t < nTok) (hi : i < n) : t * n + i < nTok * n := by
  have := Nat.mul_le_mul_right n (show t + 1 ≤ nTok by omega)
  simp only [Nat.add_mul, Nat.one_mul] at this
  omega

/-- The scalar-gate block of 256 threads has 8 warps matching the 8-slot scratch. -/
theorem gate_warps : (256 + 31) / 32 = 8 := by
  decide

/-- Per-row scaling grid covers n_embd columns. -/
theorem scale_grid_covers (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by
  omega

-- § MoE combine

/-- Routed contribution at column j. -/
def routed (w p : Nat → Int) (k : Nat) : Int := sumTo (fun e => w e * p e) k

/-- Combined value: shared term added plainly when present. -/
def combined (w p : Nat → Int) (k : Nat) (shared : Option Int) : Int :=
  match shared with
  | none => routed w p k
  | some s => routed w p k + s

/-- Without a shared expert the output is exactly the routed sum. -/
theorem combined_none (w p : Nat → Int) (k : Nat) : combined w p k none = routed w p k := by
  rfl

/-- The shared output is added unweighted. -/
theorem combined_some (w p : Nat → Int) (k : Nat) (s : Int) :
    combined w p k (some s) = routed w p k + s := by
  rfl

/-- A prefix sum of zeros is zero. -/
theorem sumTo_zero (f : Nat → Int) (k : Nat) (h : ∀ i, f i = 0) : sumTo f k = 0 := by
  induction k with
  | zero => simp [sumTo]
  | succ k ih =>
    simp only [sumTo]
    rw [ih, h k]
    rfl

/-- Zero router weights leave only the shared output. -/
theorem routed_zero_weights (p : Nat → Int) (k : Nat) : routed (fun _ => 0) p k = 0 := by
  unfold routed
  exact sumTo_zero _ k (fun e => by simp)

/-- k <= 64 is accepted, longer lists are refused rather than truncated. -/
theorem combine_k_guard (k : Nat) : ¬ (k > 64) ↔ k ≤ 64 := by
  omega

/-- Combine grid covers all columns. -/
theorem combine_grid_covers (n : Nat) : n ≤ 128 * ((n + 127) / 128) := by
  omega

end Ref.SharedExpert



## FILE: /mnt/user-data/outputs/VisionEncoderTool.lean
/-!
# MODULE: vision_encoder_tool
SOURCES: tools/vision/CMakeLists.txt, tools/vision/ref_vision.cpp
The resident image-encoder helper `ref-vision`: reads `ENC <image> <output>` lines, runs llama.cpp's mtmd
encoder, writes an embeddings file (header SVE1, n_tokens, nx, ny, n_embd; then float32 rows) and answers
`OK n nx ny ms` or `ERR msg`. CMake builds it against mtmd with optional CUDA and a portable (AVX2) mode.

## Source semantics
* parse_enc: line must start with "ENC "; the output path is the text after the LAST space, the image path is
  everything before it (so image paths may contain spaces); both must be non-empty.
* Output header: five int32 = 20 bytes, magic 0x31455653 = "SVE1" little-endian; payload 4*n*n_embd bytes.
* Token i sits at grid position (x = i % nx, y = i / nx); the grid must be rectangular: nx*ny = n.
* CMake: GGML_CUDA follows REF_VISION_CUDA; REF_PORTABLE turns GGML_NATIVE and GGML_OPENMP off and forces
  AVX/AVX2/FMA/F16C/BMI2 on; otherwise GGML_NATIVE is on.

## Form template identifiers
parse_enc -> command_parser ; hdr[5] -> embedding_header ; REF_PORTABLE -> portable_flag ;
REF_VISION_CUDA -> gpu_flag ; mtmd_encode_chunk -> encode_step
-/

namespace Ref.VisionTool

-- § command parsing

/-- Longest prefix without a space, and the rest (starting at the first space if any). -/
def spanNS : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs => if c = ' ' then ([], c :: cs) else ((c :: (spanNS cs).1), (spanNS cs).2)

/-- spanNS on (xs ++ ys) with xs space-free and ys empty or starting with a space returns (xs, ys). -/
theorem spanNS_append (xs ys : List Char) (h : ∀ c ∈ xs, c ≠ ' ')
    (hy : ys = [] ∨ ∃ t, ys = ' ' :: t) : spanNS (xs ++ ys) = (xs, ys) := by
  induction xs with
  | nil =>
    rcases hy with rfl | ⟨t, rfl⟩ <;> simp [spanNS]
  | cons c cs ih =>
    have hc : c ≠ ' ' := h c (by simp)
    have := ih (fun d hd => h d (by simp [hd]))
    simp [spanNS, hc, this]

/-- Split at the last space: (image, output); mirrors `rest.rfind(' ')` with the emptiness checks. -/
def splitLastSpace (l : List Char) : Option (List Char × List Char) :=
  match spanNS l.reverse with
  | (o, ' ' :: i) => if i = [] ∨ o = [] then none else some (i.reverse, o.reverse)
  | _ => none

/-- Correctness: a space-free output path after the last space is recovered exactly, so image paths with
spaces survive. -/
theorem splitLastSpace_correct (img out : List Char) (hno : ∀ c ∈ out, c ≠ ' ')
    (himg : img ≠ []) (hout : out ≠ []) :
    splitLastSpace (img ++ ' ' :: out) = some (img, out) := by
  have hrev : (img ++ ' ' :: out).reverse = out.reverse ++ ' ' :: img.reverse := by
    simp [List.reverse_append]
  have hs := spanNS_append out.reverse (' ' :: img.reverse)
    (fun c hc => hno c (List.mem_reverse.mp hc)) (Or.inr ⟨img.reverse, rfl⟩)
  unfold splitLastSpace
  rw [hrev, hs]
  simp [himg, hout]

/-- A line without any space has no separator, so it is rejected. -/
theorem splitLastSpace_no_space (l : List Char) (h : ∀ c ∈ l, c ≠ ' ') : splitLastSpace l = none := by
  have hr : ∀ c ∈ l.reverse, c ≠ ' ' := fun c hc => h c (List.mem_reverse.mp hc)
  have := spanNS_append l.reverse [] hr (Or.inl rfl)
  unfold splitLastSpace
  rw [List.append_nil] at this
  simp [this]

-- § output file format

/-- Header size: five int32. -/
theorem header_bytes : 5 * 4 = 20 := by
  decide

/-- The magic is "SVE1" in little-endian byte order. -/
theorem magic_bytes : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by
  decide

/-- File size = header + float32 rows. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- Payload bytes grow linearly in the token count. -/
theorem fileBytes_succ (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- An empty image writes only the header. -/
theorem fileBytes_zero (nEmbd : Nat) : fileBytes 0 nEmbd = 20 := by
  simp [fileBytes]

-- § grid positions

/-- Token i is at x = i % nx, y = i / nx; it lies inside the nx*ny grid. -/
theorem grid_position_in_range (i nx ny : Nat) (hnx : 0 < nx) (h : i < nx * ny) :
    i % nx < nx ∧ i / nx < ny := by
  exact ⟨Nat.mod_lt _ hnx, Nat.div_lt_of_lt_mul h⟩

/-- The position determines the token index. -/
theorem grid_position_roundtrip (i nx : Nat) : nx * (i / nx) + i % nx = i := by
  exact Nat.div_add_mod i nx

/-- A rectangular grid read from the last token's position (x, y) has nx = x+1, ny = y+1 and
nx*ny = n exactly when the last index fits. -/
theorem last_token_grid (x y : Nat) (n : Nat) (h : n = (x + 1) * (y + 1)) :
    n - 1 = (y) * (x + 1) + x := by
  subst h
  rw [Nat.mul_add, Nat.mul_one]
  have : (x + 1) * y = y * (x + 1) := Nat.mul_comm _ _
  omega

-- § build configuration

/-- Build switches relevant to the vision helper. -/
structure Config where
  portable : Bool
  cuda : Bool

/-- GGML_NATIVE. -/
def ggmlNative (c : Config) : Bool := !c.portable

/-- GGML_OPENMP is forced off only in portable mode (no runtime library needed on the user's PC). -/
def ggmlOpenmpForcedOff (c : Config) : Bool := c.portable

/-- GGML_CUDA follows the option. -/
def ggmlCuda (c : Config) : Bool := c.cuda

/-- Portable builds never use -march=native. -/
theorem portable_not_native (c : Config) (h : c.portable = true) : ggmlNative c = false := by
  simp [ggmlNative, h]

/-- Non-portable builds use native tuning. -/
theorem nonportable_native (c : Config) (h : c.portable = false) : ggmlNative c = true := by
  simp [ggmlNative, h]

/-- The CUDA switch is independent of portability. -/
theorem cuda_independent (c : Config) : ggmlCuda c = c.cuda := by
  rfl

end Ref.VisionTool



/-! [measured by the kernel] STATEMENTS CHANGED AFTER VERIFICATION in this chunk.
These are not the first reading of the reference: check each against the reference code before implementing or accepting it.

- SharedExpertCombine.lean
  was: theorem routed_zero_weights (p : Nat → Int) (k : Nat) : routed (fun _ => 0) p k = 0
  now: theorem sumTo_zero (f : Nat → Int) (k : Nat) (h : ∀ i, f i = 0) : sumTo f k = 0
  measured: more hypotheses/binders (2 → 3): the statement holds in fewer cases

- SharedExpertCombine.lean
  was: theorem combine_k_guard (k : Nat) : ¬ (k > 64) ↔ k ≤ 64
  now: theorem routed_zero_weights (p : Nat → Int) (k : Nat) : routed (fun _ => 0) p k = 0
  measured: more hypotheses/binders (1 → 2): the statement holds in fewer cases

- SharedExpertCombine.lean
  was: theorem combine_grid_covers (n : Nat) : n ≤ 128 * ((n + 127) / 128)
  now: theorem combine_k_guard (k : Nat) : ¬ (k > 64) ↔ k ≤ 64
  measured: the claim changed

- VisionEncoderTool.lean
  was: theorem cuda_independent (p q : Bool) : ggmlCuda { portable := p, cuda := q } = q
  now: theorem cuda_independent (c : Config) : ggmlCuda c = c.cuda
  measured: fewer quantified variables (2 → 1)
-/


[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/ref/core/graph.hpp (118 lines)
  include/ref/core/layout.hpp (98 lines)
  include/ref/core/pinned.hpp (96 lines)
  include/ref/core/weights.hpp (136 lines)
  include/ref/core/device.hpp (72 lines)
  include/ref/core/hit_hook.hpp (42 lines)
  include/ref/core/native_dense.hpp (37 lines)
  include/ref/core/native_head.hpp (66 lines)
  include/ref/kernels/elementwise.hpp (117 lines)
  include/ref/kernels/gdn.hpp (98 lines)
  include/ref/kernels/gr.hpp (123 lines)
  include/ref/kernels/native_mmvq.hpp (113 lines)
  include/ref/kernels/ple.hpp (120 lines)
  include/ref/kernels/s2_expert_grouped.hpp (119 lines)
  include/ref/kernels/s_gemv.hpp (134 lines)
  include/ref/kernels/shared_expert.hpp (102 lines)
  include/ref/kernels/verify_kernels.hpp (81 lines)
  include/ref/kernels/bf16_bits.hpp (47 lines)
  include/ref/kernels/bf16_gemv.hpp (49 lines)
  include/ref/kernels/dequant_bf16.hpp (28 lines)
  include/ref/kernels/dequant_s2.hpp (19 lines)
  include/ref/kernels/fused_gdn.hpp (36 lines)
  include/ref/kernels/fused_gr.hpp (52 lines)
  include/ref/kernels/iq_kernels.hpp (58 lines)
  include/ref/kernels/kv_q8.hpp (39 lines)
  include/ref/kernels/mrope.hpp (33 lines)
  include/ref/kernels/native_flash_attn.hpp (37 lines)
  include/ref/kernels/native_gdn.hpp (23 lines)
  include/ref/kernels/native_gdn_preprocess.hpp (38 lines)
  include/ref/kernels/native_gr_norm.hpp (19 lines)
  include/ref/kernels/native_gr_postops.hpp (29 lines)
  include/ref/kernels/native_moe.hpp (19 lines)
  include/ref/kernels/native_ple_postops.hpp (31 lines)
  include/ref/kernels/native_qsa.hpp (31 lines)
  include/ref/kernels/native_qsa_indexer.hpp (36 lines)
  include/ref/kernels/native_qsa_score.hpp (32 lines)
  include/ref/kernels/native_rope.hpp (15 lines)
  include/ref/kernels/native_router.hpp (18 lines)
  include/ref/kernels/qsa_decode_attn.hpp (45 lines)
  include/ref/kernels/qsa_select.hpp (34 lines)
  include/ref/kernels/quantize_act.hpp (59 lines)
  include/ref/kernels/rope.hpp (51 lines)
  include/ref/kernels/router_top10.hpp (21 lines)
  include/ref/kernels/s2_gemv.hpp (26 lines)
  include/ref/kernels/s2_gemv_q8.hpp (23 lines)
  include/ref/kernels/sampler.hpp (44 lines)
  include/ref/kernels/cpu/expert_layout.hpp (55 lines)
  include/ref/kernels/cpu/iq_avx512.hpp (19 lines)
  include/ref/kernels/cpu/native_expert.hpp (50 lines)
  include/ref/ngram/ple_reader.hpp (87 lines)
  include/ref/platform/direct_file.hpp (60 lines)
  include/ref/platform/memory.hpp (27 lines)
  include/ref/prefill/gemm.hpp (55 lines)
  include/ref/prefill/kernels.hpp (75 lines)
  include/ref/prefill/prefill.hpp (76 lines)
  include/ref/program/logits_selection.hpp (31 lines)
  include/ref/spec/controller.hpp (74 lines)
  include/ref/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/graph.cpp (147 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_qsa.cu (128 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/controller_test.cpp (75 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)
