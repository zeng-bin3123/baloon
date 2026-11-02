/-!
# MODULE: iquant_avx512_rows
512-bit multi-token dot products for the i-quant expert formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21) and IQ2_S(22) against Q8_K activations. Each 64-value chunk is decoded once (grid lookups, sign mask,
scales) and applied to every token.

## Source semantics
* Per-format block sizes are fixed; every byte offset the decoder touches stays inside its block for chunk j < 4.
* Sign application is a masked negate, an involution that preserves magnitude.
* Scales are `2*nibble + 1`, hence odd and at most 31.
* Grid indices are built from byte | high bits and stay inside the grid tables.
* Token tiling: 1..4 tokens map directly, more are processed in groups of 4; the gate/up variant clamps to 8.

## Form template identifiers
Fmt<TY>::decode -> chunk_decoder ; row_dot -> row_kernel ; iq512_gu_rows -> gate_up_dispatch ;
iq512_rows -> down_dispatch ; iq512_supported -> format_supported
-/

namespace Strata.Cpu.IquantAvx512

-- § format table

/-- Bytes per 256-value block, per ggml type id. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Formats handled by the AVX-512 path. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Every supported format has a positive block size. -/
theorem supported_blockBytes_pos (ty : Nat) (h : supported ty = true) : 0 < blockBytes ty := by
  unfold supported at h
  simp at h
  unfold blockBytes
  rcases h with ((((h | h) | h) | h) | h) <;> subst h <;> decide

/-- Unsupported types (IQ1_M, 20 etc.) stay on ggml-cpu. -/
theorem iq1_m_unsupported : supported 29 = false := by decide

/-- IQ2_XXS layout: fp16 scale plus 32 u16. -/
theorem layout_iq2xxs : 2 + 64 = blockBytes 16 := by decide

/-- IQ2_XS layout: scale, 32 u16, 8 scale bytes. -/
theorem layout_iq2xs : 2 + 64 + 8 = blockBytes 17 := by decide

/-- IQ3_XXS layout: scale, 64 grid bytes, 8 u32. -/
theorem layout_iq3xxs : 2 + 64 + 32 = blockBytes 18 := by decide

/-- IQ3_S layout: scale, qs, qh, signs, scales. -/
theorem layout_iq3s : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by decide

/-- IQ2_S layout: scale, qs(64), qh, scales. -/
theorem layout_iq2s : 2 + 64 + 8 + 8 = blockBytes 22 := by decide

-- § decoder byte offsets stay inside the block (chunk j < 4)

/-- IQ2_XXS reads 16 bytes at `2 + 16j`. -/
theorem xxs_in_block (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS scale bytes `66 + 2j (+1)`. -/
theorem xs_scale_in_block (j : Nat) (h : j < 4) : 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S sign qword at `2 + 32 + 8j` ends within qs. -/
theorem iq2s_sign_in_block (j : Nat) (h : j < 4) : 2 + 32 + 8 * j + 8 ≤ 66 := by omega

/-- IQ2_S grid bytes `2 + 8j + 7` are inside qs. -/
theorem iq2s_qs_in_block (j : Nat) (h : j < 4) : 2 + 8 * j + 7 < 34 := by omega

/-- IQ2_S scales `74 + 2j (+1)`. -/
theorem iq2s_scale_in_block (j : Nat) (h : j < 4) : 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS sign/scale words at `66 + 8j` stay inside. -/
theorem iq3xxs_words_in_block (j : Nat) (h : j < 4) : 66 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: qh, sign qword and scale byte offsets are inside the block. -/
theorem iq3s_offsets_in_block (j : Nat) (h : j < 4) :
    66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

-- § indices and scales

/-- Seven-bit sign index is below 128 (size of the ksigns table). -/
theorem sign_index_lt (a k : Nat) : ((a >>> (7 * k)) &&& 127) < 128 := by
  have h : ((a >>> (7 * k)) &&& 127) ≤ 127 := Nat.and_le_right
  omega

/-- A 4-bit scale nibble of a 32-bit word. -/
theorem nibble_lt (a : Nat) (h : a < 4294967296) : a >>> 28 < 16 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- Block scales `2*s+1` are odd. -/
theorem scale_odd (s : Nat) : (2 * s + 1) % 2 = 1 := by omega

/-- Block scales from a nibble are at most 31. -/
theorem scale_le (s : Nat) (h : s < 16) : 2 * s + 1 ≤ 31 := by omega

/-- IQ2_XS 9-bit grid index is below 512. -/
theorem iq2xs_grid_index_lt (v : Nat) : (v &&& 511) < 512 := by
  have h : (v &&& 511) ≤ 511 := Nat.and_le_right
  omega

/-- IQ2_XS 7-bit sign index from a 16-bit entry is below 128. -/
theorem iq2xs_sign_index_lt (v : Nat) (h : v < 65536) : v >>> 9 < 128 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- IQ2_S grid index (byte plus two high bits) is below 1024. -/
theorem iq2s_index_lt (q h : Nat) (hq : q < 256) : (q ||| ((h <<< 2) &&& 0x300)) < 1024 := by
  have h1 : (h <<< 2) &&& 0x300 ≤ 0x300 := Nat.and_le_right
  exact Nat.or_lt_two_pow (n := 10) (by omega) (by omega)

/-- IQ3_S grid index (byte plus one high bit) is below 512. -/
theorem iq3s_index_lt (q x kk : Nat) (hq : q < 256) : (q ||| (((x >>> kk) &&& 1) <<< 8)) < 512 := by
  have h1 : ((x >>> kk) &&& 1) ≤ 1 := Nat.and_le_right
  have h2 : (((x >>> kk) &&& 1) <<< 8) ≤ 256 := by
    rw [Nat.shiftLeft_eq]
    omega
  exact Nat.or_lt_two_pow (n := 9) (by omega) (by omega)

-- § sign application

/-- Masked negate: lane is negated when its sign bit is set. -/
def applySign (m : Bool) (y : Int) : Int := if m then 0 - y else y

/-- Applying the same sign twice restores the lane. -/
theorem applySign_invol (m : Bool) (y : Int) : applySign m (applySign m y) = y := by
  cases m <;> simp [applySign]

/-- Sign application preserves magnitude. -/
theorem applySign_abs (m : Bool) (y : Int) : (applySign m y).natAbs = y.natAbs := by
  cases m <;> simp [applySign]

/-- A weight times a signed activation is the signed product. -/
theorem applySign_mul (g y : Int) : g * applySign true y = -(g * y) := by
  simp [applySign, Int.mul_neg]

-- § token tiling

/-- Group size for the down-projection tiling over `nt` tokens starting at `t0`. -/
def tileSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each group is non-empty, at most 4, and stays inside the token range. -/
theorem tile_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ tileSize nt t0 ∧ tileSize nt t0 ≤ 4 ∧ t0 + tileSize nt t0 ≤ nt := by
  unfold tileSize
  omega

/-- The tiling loop advances and so terminates. -/
theorem tile_advance (nt t0 : Nat) (h : t0 < nt) : t0 < t0 + tileSize nt t0 := by
  unfold tileSize
  omega

/-- Gate/up dispatch clamps to 8 tokens. -/
def guTile (nt : Nat) : Nat := if nt ≥ 8 then 8 else nt

/-- The gate/up tile is at most 8. -/
theorem guTile_le (nt : Nat) : guTile nt ≤ 8 := by
  unfold guTile
  split <;> omega

-- § row geometry

/-- Blocks per row for an `n`-value row (QK_K = 256). -/
theorem blocks_per_row (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by omega

/-- Up row offset is past every gate row: row r < rows implies r*gu_row + gu_row ≤ rows*gu_row. -/
theorem gate_row_end (r rows g : Nat) (h : r < rows) : r * g + g ≤ rows * g := by
  have : (r + 1) * g ≤ rows * g := Nat.mul_le_mul_right g h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.Cpu.IquantAvx512
