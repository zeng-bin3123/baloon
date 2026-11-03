/-!
# MODULE: q2_avx2_rows
Addressing and quantizer contracts of the AVX2-only Q2_0 rows (`q2_avx2.cpp`).

## Source semantics
A GGUF Q2_0 row is `nblocks` blocks of 18 bytes (2-byte scale, 16 code bytes). Value `i` of a block
is bits `2*(i%4)` of code byte `i/4`; the low vector holds values 0..31 (bytes 0..7) and the high
vector values 32..63. The activation uses 64 int8 values and two scales per block. Tokens are
processed in tiles of at most 4. The quantizer clamps codes to `[-127, 127]`.

## Form template identifiers
* `blk = row + b*18` → `blockOff`; `a[t]->q + b*64` → `actOff`; `tile` → `tileLen`
-/

namespace StrataQ2Avx2

-- § block addressing

/-- Offset of block `b` in a row. -/
def blockOff (b : Nat) : Nat := b * 18

/-- Block `b`'s scale and codes (18 bytes) end where block `b+1` starts. -/
theorem block_end (b : Nat) : blockOff b + 18 = blockOff (b + 1) := by
  unfold blockOff; omega

/-- Every block read stays within the row of `nblocks` blocks. -/
theorem block_in_row (b nblocks : Nat) (h : b < nblocks) : blockOff b + 18 ≤ nblocks * 18 := by
  unfold blockOff; omega

/-- The code bytes start 2 bytes after the scale and end at the block end. -/
theorem codes_span (b : Nat) : blockOff b + 2 + 16 = blockOff b + 18 := by omega

/-- Activation int8 offsets: both 32-value halves of block `b` stay inside `nblocks * 64`. -/
theorem act_in_range (b nblocks : Nat) (h : b < nblocks) : b * 64 + 32 + 32 ≤ nblocks * 64 := by
  omega

/-- The two chunk scales of block `b` are indices `2b` and `2b+1`, below `2*nblocks`. -/
theorem chunk_scale_idx (b nblocks : Nat) (h : b < nblocks) : 2 * b + 1 < 2 * nblocks := by omega

-- § code unpacking

/-- Value `i < 64` lives in byte `i/4` at bit `2*(i%4)`; the pair is a bijection. -/
theorem unpack_index (i : Nat) (h : i < 64) :
    i / 4 < 16 ∧ 2 * (i % 4) < 8 ∧ (i / 4) * 4 + i % 4 = i := by omega

/-- The low vector holds values below 32, which come from bytes below 8. -/
theorem unpack_low_half (i : Nat) (h : i < 32) : i / 4 < 8 := by omega

/-- The high vector holds values 32..63, which come from bytes 8..15. -/
theorem unpack_high_half (i : Nat) (h1 : 32 ≤ i) (h2 : i < 64) : 8 ≤ i / 4 ∧ i / 4 < 16 := by omega

/-- A 2-bit code minus the offset lies in `{-1, 0, 1, 2}`. -/
theorem code_range (c : Int) : -1 ≤ c % 4 - 1 ∧ c % 4 - 1 ≤ 2 := by omega

-- § tiling and quantizer

/-- Tile length of the multi-token loop. -/
def tileLen (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each tile is non-empty and stays within `nt`. -/
theorem tile_progress (nt t0 : Nat) (h : t0 < nt) : 1 ≤ tileLen nt t0 ∧ t0 + tileLen nt t0 ≤ nt := by
  unfold tileLen; omega

/-- A single tile covers `nt ≤ 4` entirely. -/
theorem tile_single (nt : Nat) (h : nt ≤ 4) (hp : 0 < nt) : tileLen nt 0 = nt := by
  unfold tileLen; omega

/-- Clamp of an activation code. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Quantized codes lie in `[-127, 127]`. -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; omega

/-- The quantizer's chunk count covers the input when the width is a multiple of 32. -/
theorem nchunks_cover (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by omega

end StrataQ2Avx2
