/-!
# MODULE: s_gemv_family_kernels
SOURCES: src/kernels/cuda/s_gemv.cu
S-family GEMV: dequantize-on-the-fly of 2/4/8-bit packed codes (optional IQ4_NL codebook, per-group scale and
optional offset) against FP16, Q8_K or Q8_0 activations; naive, split-per-row and warp-per-row variants.

## Source semantics
- A weight is `decode(code) * scale + offset`; the offset is applied before the activation multiply.
- Codes are packed LSB-first, `8/bits` per byte; a row has `n_in/(8/bits)` code bytes and `n_in/group` scales.
- Q8_K activation block: 292 bytes per 256 elements (d:4, qs:256, bsums:32); Q8_0: 34 bytes per 32.
- The warp-per-row kernel takes 16 consecutive elements per lane per iteration with stride 32*16; all 16 lie
  in one activation block and one group. The group index is a shift because group sizes are powers of two.

## Form template identifiers
- SForm.code_bits -> bits ; SForm.group_elems -> group ; group_shift -> shift
- s_gemv_q8_split_kernel -> warp-per-row ; s_gemv_split_kernel -> block-per-row
-/

namespace Strata.SGemv

-- § packing
/-- Codes per byte for 2/4/8-bit widths. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- 2-bit codes: 4 per byte. -/
theorem perByte_2 : perByte 2 = 4 := by decide
/-- 4-bit codes: 2 per byte. -/
theorem perByte_4 : perByte 4 = 2 := by decide
/-- 8-bit codes: 1 per byte. -/
theorem perByte_8 : perByte 8 = 1 := by decide

/-- Bytes per four-code word (the QW constant): 1, 2 or 4 for S2, S4, S8. -/
theorem qw_values : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Extract the k-th `bits`-wide code of a packed word (little-endian field order). -/
def codeAt (v bits k : Nat) : Nat := (v / 2 ^ (k * bits)) % 2 ^ bits

/-- Every extracted code fits in `bits` bits. -/
theorem codeAt_lt (v bits k : Nat) : codeAt v bits k < 2 ^ bits := by
  unfold codeAt
  exact Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- Codes are bounded by 15 for 4-bit fields (index into the 16-entry codebook). -/
theorem codeAt_lt16 (v k : Nat) : codeAt v 4 k < 16 := by
  have := codeAt_lt v 4 k
  simpa using this

-- § activation block geometry
/-- Q8_K block bytes: scale (4) + 256 int8 + 16 int16 bsums (32). -/
theorem q8k_block_bytes : 4 + 256 + 16 * 2 = 292 := by decide
/-- Q8_0 block bytes: fp16 scale (2) + 32 int8. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- A 16-aligned start index keeps its 16 elements inside one Q8_K (256) block. -/
theorem lane_in_q8k_block (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by
  omega

/-- A 16-aligned start index keeps its 16 elements inside one Q8_0 (32) block. -/
theorem lane_in_q8_0_block (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by
  omega

/-- The lane stride 32*16 = 512 preserves 16-alignment. -/
theorem stride_preserves_alignment (i : Nat) (h : i % 16 = 0) : (i + 512) % 16 = 0 := by
  omega

/-- Lane start `lane*16` is 16-aligned. -/
theorem lane_start_aligned (lane : Nat) : (lane * 16) % 16 = 0 := by
  omega

/-- Every element has an owning (lane, iteration): `e/16 = lane + 32*iter`. -/
theorem element_ownership (e : Nat) : e / 16 = (e / 16) % 32 + 32 * (e / 512) := by
  omega

-- § group indexing
/-- Shift by `s` is division by `2^s` (group index as shift). -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := by
  exact Nat.shiftRight_eq_div_pow i s

/-- A 16-aligned run stays inside a group whose size is a multiple of 16. -/
theorem quad_in_group (a m : Nat) (hm : 0 < m) : 16 * a % (16 * m) + 16 ≤ 16 * m := by
  rw [Nat.mul_mod_mul_left]
  have := Nat.mod_lt a hm
  omega

/-- Row base offset in the codes array: row `o` starts at `o * codes_per_row`. -/
theorem row_offsets_disjoint (o r cpr : Nat) (h : o < r) (hr : 0 < cpr) : o * cpr + cpr ≤ r * cpr := by
  have h1 : (o + 1) * cpr ≤ r * cpr := Nat.mul_le_mul_right cpr h
  have h2 : (o + 1) * cpr = o * cpr + cpr := Nat.succ_mul o cpr
  omega

-- § reductions
/-- Shuffle-down steps 16+8+4+2+1 cover 31 other lanes of a 32-lane warp. -/
theorem shuffle_steps : 16 + 8 + 4 + 2 + 1 = 32 - 1 := by decide

/-- Warps per 256-thread block. -/
theorem warps_per_block_256 : 256 / 32 = 8 := by decide

/-- The grid covers all rows for one warp per row (8 per block). -/
theorem warp_grid_covers (nOut : Nat) : nOut ≤ ((nOut + 8 - 1) / 8) * 8 := by
  omega

/-- Quad kernel: `tid*4` starts are 4-aligned. -/
theorem quad_start_aligned (tid : Nat) : (tid * 4) % 4 = 0 := by
  omega

-- § non-multiple activation: ffn_down_shexp
/-- n_in = 640 is a multiple of 32 but not of 256, so Q8_K is structurally impossible there. -/
theorem q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- The 2.12x codebook measurement concerns a 16-entry table: 16 bytes = four 32-bit words. -/
theorem codebook_words : 16 / 4 = 4 := by decide

end Strata.SGemv
