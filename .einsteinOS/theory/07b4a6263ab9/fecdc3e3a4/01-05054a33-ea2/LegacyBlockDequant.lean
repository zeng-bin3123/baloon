/-!
# MODULE: legacy_block_dequant
SOURCES: include/strata/artifact/dequant.hpp, tools/canonical_xcheck.py
The 32-element "legacy" encodings Q8_0, Q4_0, Q5_0 and IQ4_NL: nibble splitting, the 5th-bit
plane of Q5_0, the integer code biases, and the non-affine IQ4_NL codebook, together with the
canonical (S4/S8) re-encodings that store every code unsigned.

## Source semantics
* Q4_0/Q5_0/IQ4_NL put the LOW nibble of byte `j` at element `j` and the HIGH nibble at `j + 16`.
* Q4_0 decodes `(nibble - 8) * d`; Q5_0 decodes `((nibble | hbit << 4) - 16) * d`, where the 5th
  bit of element `j` is bit `j` of the little-endian 32-bit `qh`.
* ggml writes the high half's bit as `(qh >> (j + 12)) & 0x10`; that equals bit `j + 16` shifted to
  position 4, which is why the C++ port may write `(h >> (j + 16)) & 1` directly.
* Q8_0 is signed; the canonical S8 plane stores `qs + 128` and decodes `code - 128`.
* IQ4_NL indexes a 16-entry table that is NOT affine, so it cannot be a code bias.

## Form template identifiers
`qs[j] & 0x0F` → `lowNib`, `qs[j] >> 4` → `highNib`, Q4_0 symbol → `q4Symbol`, Q5_0 code →
`q5Code`, `kvalues_iq4nl` → `kvaluesIq4nl`, Q8_0 canonical code → `s8Code`, block byte sizes →
`q4_0Bytes`, `q5_0Bytes`, `q8_0Bytes`, `iq4nlBytes`.
-/

namespace Strata.Legacy

-- § block sizes

/-- Q4_0 block bytes: fp16 d + 16 nibble bytes. -/
def q4_0Bytes : Nat := 2 + 32 / 2
/-- Q5_0 block bytes: fp16 d + 4 bytes of high bits + 16 nibble bytes. -/
def q5_0Bytes : Nat := 2 + 32 / 8 + 32 / 2
/-- Q8_0 block bytes: fp16 d + 32 signed bytes. -/
def q8_0Bytes : Nat := 2 + 32
/-- IQ4_NL block bytes: same shape as Q4_0. -/
def iq4nlBytes : Nat := 2 + 32 / 2

/-- The four legacy block sizes match `block_geometry`: 18, 22, 34, 18 bytes per 32 elements. -/
theorem legacy_block_bytes :
    q4_0Bytes = 18 ∧ q5_0Bytes = 22 ∧ q8_0Bytes = 34 ∧ iq4nlBytes = 18 := by decide

-- § nibbles

/-- Low nibble of a byte (element `j`). -/
def lowNib (b : Nat) : Nat := b &&& 15
/-- High nibble of a byte (element `j + 16`). -/
def highNib (b : Nat) : Nat := b >>> 4

/-- Splitting a byte into its nibbles loses nothing: `b = low + 16 * high`. -/
theorem nibble_split : ∀ b, b < 256 → lowNib b + 16 * highNib b = b := by
  unfold lowNib highNib
  set_option maxRecDepth 20000 in decide

/-- The low nibble is below 16 for any input. -/
theorem lowNib_lt (b : Nat) : lowNib b < 16 := by
  have h : b &&& 15 ≤ 15 := Nat.and_le_right
  unfold lowNib; omega

/-- The high nibble of a byte is below 16. -/
theorem highNib_lt {b : Nat} (hb : b < 256) : highNib b < 16 := by
  unfold highNib; rw [Nat.shiftRight_eq_div_pow]
  have : b / 2 ^ 4 < 16 := by
    rw [Nat.div_lt_iff_lt_mul (by decide)]; omega
  exact this

/-- Element placement: low nibbles fill `0..15`, high nibbles fill `16..31`, never interleaved. -/
theorem nibble_positions {j : Nat} (hj : j < 16) : j < 16 ∧ 16 ≤ j + 16 ∧ j + 16 < 32 := by
  omega

-- § Q4_0

/-- Q4_0 symbol: the nibble minus 8, an integer bias before the multiply. -/
def q4Symbol (n : Nat) : Int := (n : Int) - 8

/-- Q4_0 symbols range over `-8..7`. -/
theorem q4Symbol_range {n : Nat} (hn : n < 16) : -8 ≤ q4Symbol n ∧ q4Symbol n ≤ 7 := by
  unfold q4Symbol; omega

-- § Q5_0

/-- The 5-bit Q5_0 code from a nibble and its high bit. -/
def q5Code (nib hbit : Nat) : Nat := nib ||| (hbit <<< 4)

/-- With a nibble below 16 and a single high bit, the 5-bit code is `nib + 16 * hbit`. -/
theorem q5Code_eq : ∀ n, n < 16 → ∀ h, h < 2 → q5Code n h = n + 16 * h := by
  unfold q5Code; decide

/-- Q5_0 codes fit in 5 bits and their symbols `code - 16` cover `-16..15`. -/
theorem q5_symbol_range {n h : Nat} (hn : n < 16) (hh : h < 2) :
    q5Code n h < 32 ∧ -16 ≤ (q5Code n h : Int) - 16 ∧ (q5Code n h : Int) - 16 ≤ 15 := by
  rw [q5Code_eq n hn h hh]; omega

/-- Masking with `0x10` keeps exactly bit 4: `x & 16 = ((x >> 4) & 1) << 4`. -/
theorem and16_eq_bit4 (x : Nat) : x &&& 16 = ((x >>> 4) &&& 1) <<< 4 := by
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_and, Nat.testBit_shiftLeft, Nat.testBit_shiftRight]
  by_cases h : i = 4
  · subst h
    have : (16 : Nat).testBit 4 = true := by decide
    simp [this]
  · have h16 : (16 : Nat).testBit i = false := by
      rw [show (16 : Nat) = 2 ^ 4 by rfl, Nat.testBit_two_pow]
      simp; omega
    by_cases hi : 4 ≤ i
    · have h1 : (1 : Nat).testBit (i - 4) ≠ true := by
        intro ht
        rw [Nat.testBit_one_eq_true_iff_self_eq_zero] at ht
        omega
      simp [h16, h1]
    · simp [h16]; omega

/-- ggml's `(qh >> (j + 12)) & 0x10` is the 5th bit of element `j + 16`, i.e. bit `j + 16` of `qh`
moved to position 4: the C++ form `((h >> (j + 16)) & 1) << 4` is the same number. -/
theorem q5_high_half_bit (qh j : Nat) :
    (qh >>> (j + 12)) &&& 16 = ((qh >>> (j + 16)) &&& 1) <<< 4 := by
  rw [and16_eq_bit4, ← Nat.shiftRight_add]

/-- The low-half form `((qh >> j) << 4) & 0x10` likewise isolates bit `j`. -/
theorem q5_low_half_bit (qh j : Nat) :
    ((qh >>> j) <<< 4) &&& 16 = ((qh >>> j) &&& 1) <<< 4 := by
  apply Nat.eq_of_testBit_eq
  intro i
  simp only [Nat.testBit_and, Nat.testBit_shiftLeft]
  by_cases h : i = 4
  · subst h
    have : (16 : Nat).testBit 4 = true := by decide
    have h1 : (1 : Nat).testBit 0 = true := by decide
    simp [this, h1]
  · have h16 : (16 : Nat).testBit i = false := by
      rw [show (16 : Nat) = 2 ^ 4 by rfl, Nat.testBit_two_pow]
      simp; omega
    by_cases hi : 4 ≤ i
    · have h1 : (1 : Nat).testBit (i - 4) ≠ true := by
        intro ht
        rw [Nat.testBit_one_eq_true_iff_self_eq_zero] at ht
        omega
      simp [h16, h1]
    · simp [h16]; omega

-- § Q8_0 canonical form

/-- The S8 code of a signed Q8_0 byte: `qs + 128`, stored unsigned. -/
def s8Code (v : Int) : Int := v + 128

/-- Every int8 maps into the unsigned byte range `0..255`. -/
theorem s8Code_range {v : Int} (h1 : -128 ≤ v) (h2 : v ≤ 127) : 0 ≤ s8Code v ∧ s8Code v ≤ 255 := by
  unfold s8Code; omega

/-- Decoding `code - 128` restores the signed value exactly. -/
theorem s8Code_roundtrip (v : Int) : s8Code v - 128 = v := by
  unfold s8Code; omega

/-- Reading the S8 plane SIGNED instead of with the bias puts every negative value 128 codes away
from where it belongs - the ambiguity the pack exposed. -/
theorem s8_signed_misread {v : Int} (h : v < 0) (h2 : -128 ≤ v) : s8Code v ≠ v ∧ s8Code v - v = 128 := by
  unfold s8Code; omega

-- § IQ4_NL codebook

/-- `kvalues_iq4nl`, copied verbatim from ggml-common.h. -/
def kvaluesIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries, one per nibble. -/
theorem kvalues_length : kvaluesIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing, so the code order is the value order. -/
theorem kvalues_strictly_increasing :
    ∀ i, i < 15 → kvaluesIq4nl.getD i 0 < kvaluesIq4nl.getD (i + 1) 0 := by decide

/-- The codebook is NOT affine: consecutive gaps are not constant (23 vs 21), so no code bias
`code + b` times a scale can express it, which is why the pack carries a `codebook` field. -/
theorem kvalues_not_affine :
    kvaluesIq4nl.getD 1 0 - kvaluesIq4nl.getD 0 0 ≠ kvaluesIq4nl.getD 2 0 - kvaluesIq4nl.getD 1 0 := by
  decide

/-- The extreme entries are -127 and 113: values fit in int8. -/
theorem kvalues_bounds : ∀ i, i < 16 → -127 ≤ kvaluesIq4nl.getD i 0 ∧ kvaluesIq4nl.getD i 0 ≤ 113 := by
  decide

end Strata.Legacy
