/-!
# MODULE: iquant_avx512_rows
Multi-token i-quant expert row dots (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) against Q8_K activations: the
per-format block layout, in-bounds decoding of every field, grid-index ranges, scale lane mapping,
no-saturation bounds of `maddubs`, and the token-batching loop.

## Source semantics
* Type ids 16, 17, 18, 21, 22 are supported; IQ1_M stays on ggml-cpu.
* A block holds 256 values; each of four 64-value chunks decodes grid magnitudes, a 64-bit sign mask and
  four scales (one per 16 values).
* Every byte read of the block layout lies inside the format's block size.
* `maddubs` pairs (grid magnitude * int8) cannot saturate the int16 lanes for these grids.

## Form template identifiers
Fmt<TY>::decode -> decode_chunk, row_dot -> row_dot, gu_rows -> gate_up_rows, dot_rows -> down_rows,
iq512_supported -> supported, QK_K -> super_block.
-/

namespace Strata.IQuantAvx512

-- § formats

/-- Values per super-block (`QK_K`). -/
def QK_K : Nat := 256

/-- Bytes per block for each supported ggml type id, zero when unsupported. -/
def bytesOf : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- The type ids handled by the AVX-512 path. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Supported types are exactly those with a non-zero block size (checked over all small ids). -/
theorem supported_iff_bytes : ∀ t : Fin 64, supported t.val = true ↔ bytesOf t.val ≠ 0 := by
  decide

/-- IQ1_M (type 29) is not handled here. -/
theorem iq1_m_unsupported : supported 29 = false := by decide

/-- IQ2_XXS: fp16 d plus 32 u16. -/
theorem iq2xxs_size : bytesOf 16 = 2 + 64 := by decide
/-- IQ2_XS: d, 32 u16, 8 scale bytes. -/
theorem iq2xs_size : bytesOf 17 = 2 + 64 + 8 := by decide
/-- IQ2_S: d, 64 qs bytes, 8 qh, 8 scales. -/
theorem iq2s_size : bytesOf 22 = 2 + 64 + 8 + 8 := by decide
/-- IQ3_XXS: d, 64 grid bytes, 8 u32 of signs and scales. -/
theorem iq3xxs_size : bytesOf 18 = 2 + 64 + 32 := by decide
/-- IQ3_S: d, qs, qh, signs, scales. -/
theorem iq3s_size : bytesOf 21 = 2 + 64 + 8 + 32 + 4 := by decide

-- § in-bounds decoding (chunk j of four)

/-- IQ2_XXS reads 16 bytes of `qs` for chunk `j` inside the block. -/
theorem iq2xxs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS reads its grid words and its two scale bytes inside the block. -/
theorem iq2xs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S reads grid bytes, a sign word, two qh bytes and two scale bytes inside the block. -/
theorem iq2s_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 7 < 66 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS reads 16 grid bytes and two sign/scale words inside the block. -/
theorem iq3xxs_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by
  omega

/-- IQ3_S reads grid bytes, qh, signs and scales inside the block. -/
theorem iq3s_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

/-- Four 64-value chunks tile a 256-value block and the activation reads stay inside it. -/
theorem chunks_tile (j : Nat) (hj : j < 4) : 4 * 64 = QK_K ∧ 64 * j + 64 ≤ QK_K := by
  unfold QK_K
  omega

-- § grid index ranges

/-- IQ2_S grid index (8 low bits plus a 2-bit extension) stays below the 1024-entry grid. -/
theorem iq2s_index (q h : Nat) (hq : q < 256) (hh : h < 4) : q + 256 * h < 1024 := by omega

/-- IQ3_S grid index (8 low bits plus 1 extension bit) stays below the 512-entry grid. -/
theorem iq3s_index (q b : Nat) (hq : q < 256) (hb : b < 2) : q + 256 * b < 512 := by omega

/-- IQ2_XS: a 16-bit word splits into a 9-bit grid index and a 7-bit sign index. -/
theorem iq2xs_split (v : Nat) (hv : v < 65536) : v % 512 < 512 ∧ v / 512 < 128 := by omega

-- § scales

/-- Scale lane mapping: int16 lane `l` (values `2l`, `2l+1`) uses scale `l/8`, i.e. one scale per 16 values. -/
theorem scale_lane_group (l : Nat) (hl : l < 32) : l / 8 < 4 ∧ (2 * l) / 16 = l / 8 ∧ (2 * l + 1) / 16 = l / 8 := by
  omega

/-- 32 int16 lanes carry 64 values. -/
theorem lanes_cover_chunk : 32 * 2 = 64 := by decide

/-- A 4-bit scale `s` becomes the odd multiplier `2s+1`, at most 31. -/
theorem odd_scale (s : Nat) (hs : s < 16) : 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by omega

-- § no int16 saturation in maddubs

/-- A pair of grid magnitudes (≤ 43, as in the 2-bit grids) times int8 activations fits int16. -/
theorem maddubs_no_saturation (g : Nat) (hg : g ≤ 43) : 2 * g * 127 ≤ 32767 := by omega

/-- Applying a sign mask only negates: the magnitude of the product is unchanged. -/
def applySign (neg : Bool) (y : Int) : Int := if neg then -y else y

/-- Signed activation stays in the int8 symmetric range. -/
theorem applySign_bound (neg : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ applySign neg y ∧ applySign neg y ≤ 127 := by
  unfold applySign
  split <;> omega

-- § row layout and batching

/-- Blocks per row for an activation width `n`. -/
def nBlocks (n : Nat) : Nat := n / QK_K

/-- Byte offset of row `r` in the gate (or, shifted by `up_off`, up) matrix. -/
def rowOffset (base rowBytes r : Nat) : Nat := base + r * rowBytes

/-- The next row starts exactly one row stride later. -/
theorem rowOffset_succ (base rowBytes r : Nat) :
    rowOffset base rowBytes (r + 1) = rowOffset base rowBytes r + rowBytes := by
  unfold rowOffset
  rw [Nat.succ_mul]
  omega

/-- Tokens handled by one `dot_rows` batch starting at token `t0`: at most four. -/
def batchSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Each batch is non-empty, at most four tokens, and stays inside the token list. -/
theorem batch_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ batchSize nt t0 ∧ batchSize nt t0 ≤ 4 ∧ t0 + batchSize nt t0 ≤ nt := by
  unfold batchSize
  split <;> omega

end Strata.IQuantAvx512
