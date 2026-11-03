/-!
# MODULE: q2_gguf_writer
The minimal GGUF v3 writer and its Q2_0 quantiser: code clamp, packing order, fixed-point property,
block size, and alignment padding.

## Source semantics
* Q2_0: block of 64 elements, `d = amax`, `code = clamp(round(w/d) + 1, 0, 3)`, symbol `code - 1 ∈ {-1,0,1,2}`.
* Codes are packed LSB-first, four per byte: element `j` sits in byte `j/4` at bit `2*(j%4)`.
* Requantising dequantised symbols reproduces the same codes (fixed point).
* Every tensor and the data section start on an `alignment`-byte boundary via padding `(-len) % alignment`.

## Form template identifiers
quantize_q2_0 -> quantize_block, dequantize_q2_0 -> dequantize_block, GGUFWriter.write -> write_file,
DEFAULT_ALIGNMENT -> alignment.
-/

namespace Strata.Q2GgufWriter

-- § quantiser

/-- Code for a rounded symbol `s`: `clamp(s + 1, 0, 3)`. -/
def codeOfSymbol (s : Int) : Int := max 0 (min 3 (s + 1))

/-- Every produced code is in `[0, 3]`. -/
theorem code_range (s : Int) : 0 ≤ codeOfSymbol s ∧ codeOfSymbol s ≤ 3 := by
  unfold codeOfSymbol
  omega

/-- For symbols already in `{-1, 0, 1, 2}` the code is `s + 1` (no clamping happens). -/
theorem code_exact (s : Int) (h1 : -1 ≤ s) (h2 : s ≤ 2) : codeOfSymbol s = s + 1 := by
  unfold codeOfSymbol
  omega

/-- Fixed point: decoding a code (`c - 1`) and requantising returns the same code. -/
theorem requantize_fixed (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 3) : codeOfSymbol (c - 1) = c := by
  unfold codeOfSymbol
  omega

/-- Decoded symbols are in `[-1, 2]`. -/
theorem symbol_range (c : Int) (h1 : 0 ≤ c) (h2 : c ≤ 3) : -1 ≤ c - 1 ∧ c - 1 ≤ 2 := by omega

-- § packing

/-- Byte index of element `j`. -/
def byteIndex (j : Nat) : Nat := j / 4

/-- Bit offset of element `j` inside its byte. -/
def bitOffset (j : Nat) : Nat := (j % 4) * 2

/-- Element `j < 64` lands inside the block's 16 code bytes at an even bit offset below 8. -/
theorem pack_position (j : Nat) (hj : j < 64) : byteIndex j < 16 ∧ bitOffset j < 8 ∧ bitOffset j % 2 = 0 := by
  unfold byteIndex bitOffset
  omega

/-- Distinct elements never share a (byte, bit) slot. -/
theorem pack_injective (i j : Nat) (hi : i < 64) (hj : j < 64)
    (hb : byteIndex i = byteIndex j) (ho : bitOffset i = bitOffset j) : i = j := by
  unfold byteIndex bitOffset at *
  omega

/-- Four 2-bit codes pack into one byte and unpack to themselves (slot `t` is `byte / 4^t % 4`). -/
theorem pack_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by omega

/-- A block is 2 bytes of scale plus 16 bytes of codes. -/
theorem block_size : 2 + 16 = 18 := by decide

/-- Sixty-four codes need exactly sixteen bytes. -/
theorem codes_bytes : 64 / 4 = 16 := by decide

/-- The element count must be a whole number of blocks; blocks times 64 is the element count. -/
theorem blocks_cover (n : Nat) (h : n % 64 = 0) : (n / 64) * 64 = n := by omega

-- § alignment

/-- Padding needed to reach the next multiple of `a`. -/
def pad (a n : Nat) : Nat := (a - n % a) % a

/-- Adding the padding yields a multiple of the alignment. -/
theorem pad_aligns (a n : Nat) (ha : 0 < a) : (n + pad a n) % a = 0 := by
  unfold pad
  have h1 := Nat.div_add_mod n a
  have h2 := Nat.mod_lt n ha
  by_cases h0 : n % a = 0
  · rw [h0]
    simpa [Nat.mod_self] using h0
  · have h3 : (a - n % a) % a = a - n % a := Nat.mod_eq_of_lt (by omega)
    rw [h3]
    have h4 : n + (a - n % a) = a * (n / a) + a := by omega
    rw [h4]
    simp

/-- The padding is smaller than the alignment. -/
theorem pad_lt (a n : Nat) (ha : 0 < a) : pad a n < a := by
  unfold pad
  exact Nat.mod_lt _ ha

/-- With the default 32-byte alignment the padding is below 32. -/
theorem pad32_lt (n : Nat) : pad 32 n < 32 := pad_lt 32 n (by decide)

-- § header

/-- GGUF magic is the bytes `G G U F` little-endian: 0x46554747. -/
theorem magic_bytes :
    0x46554747 % 256 = 0x47 ∧ 0x46554747 / 256 % 256 = 0x47 ∧
    0x46554747 / 65536 % 256 = 0x55 ∧ 0x46554747 / 16777216 % 256 = 0x46 := by decide

end Strata.Q2GgufWriter
