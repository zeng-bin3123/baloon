/-!
# MODULE: iq_avx512_rows
Block layouts and field extraction of the i-quant AVX-512 expert rows (`iq_avx512.cpp`).

## Source semantics
Each 256-value super-block is four 64-value chunks. Every decode reads only bytes inside its block
(66, 74, 82, 98, 110 bytes for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S). Scales are `2*s+1` with a
4-bit `s`, so they are odd and at most 31. Four 16-bit scales pack into one 64-bit word and
unpack losslessly. The multi-token tile loop covers `nt` in tiles of at most 4.

## Form template identifiers
* `Fmt<TY>::bytes` → block bytes; `scales4` → `pack4`; `ksigns_iq2xs[x & 127]` → 7-bit field
* `dot_rows_nt` tile → `tileLen`
-/

namespace StrataIqAvx512

-- § block layouts

/-- Four 64-value chunks per 256-value super-block. -/
theorem chunks_per_block : 4 * 64 = 256 := by decide

/-- IQ2_XXS: 2-byte d + 32 u16 = 66 bytes, and chunk `j` stays inside. -/
theorem iq2xxs_in_block (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS: quants in `[2,66)`, scales at `66 + 2j + 1 < 74`. -/
theorem iq2xs_in_block (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S: grid bytes, sign bytes, high bits and scales all inside 82 bytes. -/
theorem iq2s_in_block (j : Nat) (h : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS: grid bytes and sign/scale words inside 98 bytes. -/
theorem iq3xxs_in_block (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: grid, high bits, signs and scales inside 110 bytes. -/
theorem iq3s_in_block (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

/-- Block byte sizes add up from their parts. -/
theorem block_sizes :
    2 + 32 * 2 = 66 ∧ 66 + 8 = 74 ∧ 2 + 32 + 32 + 8 + 8 = 82 ∧ 66 + 32 = 98 ∧ 2 + 64 + 8 + 32 + 4 = 110 := by
  decide

-- § scales and sign fields

/-- Low-nibble scale `2*(s%16)+1` is odd and at most 31. -/
theorem scale_lo (s : Nat) : (2 * (s % 16) + 1) % 2 = 1 ∧ 2 * (s % 16) + 1 ≤ 31 := by omega

/-- High-nibble scale of a byte is at most 31. -/
theorem scale_hi (s : Nat) (h : s < 256) : 2 * (s / 16) + 1 ≤ 31 := by omega

/-- Top-nibble scale of a 32-bit word is at most 31. -/
theorem scale_top (a : Nat) (h : a < 4294967296) : 2 * (a / 268435456) + 1 ≤ 31 := by omega

/-- A 32-bit word splits into four 7-bit sign indices and a 4-bit scale, losslessly. -/
theorem word_fields (a : Nat) (h : a < 4294967296) :
    a % 128 + 128 * ((a / 128) % 128) + 16384 * ((a / 16384) % 128) +
      2097152 * ((a / 2097152) % 128) + 268435456 * (a / 268435456) = a := by omega

/-- Packing four 16-bit scales into a 64-bit word and extracting them is the identity. -/
theorem pack4_roundtrip (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536)
    (h2 : s2 < 65536) (h3 : s3 < 65536) :
    let p := s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3
    p % 65536 = s0 ∧ (p / 65536) % 65536 = s1 ∧ (p / 4294967296) % 65536 = s2 ∧
      p / 281474976710656 = s3 := by
  intro p
  simp only [p]
  omega

/-- Packing eight sign bytes into a mask and extracting the low and high byte is the identity. -/
theorem mask_bytes (b0 b7 mid : Nat) (h0 : b0 < 256) (h7 : b7 < 256) (hm : mid < 281474976710656) :
    let m := b0 + 256 * mid + 72057594037927936 * b7
    m % 256 = b0 ∧ m / 72057594037927936 = b7 := by
  intro m
  simp only [m]
  omega

-- § tiling

/-- Tile length of the multi-token loop. -/
def tileLen (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each tile is non-empty and stays within `nt`. -/
theorem tile_progress (nt t0 : Nat) (h : t0 < nt) : 1 ≤ tileLen nt t0 ∧ t0 + tileLen nt t0 ≤ nt := by
  unfold tileLen; omega

/-- Supported i-quant types. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- IQ2_XS is supported. -/
theorem supported_iq2xs : supported 17 = true := by decide

/-- IQ1_M (29) stays on ggml-cpu. -/
theorem iq1m_unsupported : supported 29 = false := by decide

end StrataIqAvx512
