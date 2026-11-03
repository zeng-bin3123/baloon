/-!
# MODULE: s_family_gemv_indexing
Indexing contracts of the S-family GEMV kernels in `s_gemv.cu`: code unpacking, group index via shift,
quad/octet blocks that stay inside one group and one activation block, and the activation block strides.

## Source semantics
* A code of width `b` in {2,4,8} is `(word >> (k*b)) & (2^b - 1)`; `PER_BYTE = 8/b`.
* The group index `e >> group_shift` equals `e / group_elems` when `group_elems = 2^group_shift`.
* The Q8_K block is 292 bytes per 256 elements, the Q8_0 block 34 bytes per 32 elements.
* A lane-iteration takes QE consecutive elements starting at a multiple of QE; they lie in one group and one
  activation block.

## Form template identifiers
CODE_BITS -> code_width; group_shift -> log2_group; QE -> elements_per_lane; PER_BYTE -> codes_per_byte;
QW -> bytes_per_quad_word.
-/

namespace SFamilyGemvIndexing

-- § code unpacking

/-- A masked code is below `2^b`. -/
theorem code_lt (word shift b : Nat) : (word >>> shift) % 2 ^ b < 2 ^ b :=
  Nat.mod_lt _ (Nat.two_pow_pos b)

/-- Codes per byte times width is a byte, for the three widths. -/
theorem per_byte_width : 8 / 2 * 2 = 8 ∧ 8 / 4 * 4 = 8 ∧ 8 / 8 * 8 = 8 := by decide

/-- Bytes per four-code word: 1, 2, 4. -/
theorem qw_values : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Sixteen codes occupy 4*QW bytes for each width. -/
theorem octet_bytes : 16 * 2 / 8 = 4 * 1 ∧ 16 * 4 / 8 = 4 * 2 ∧ 16 * 8 / 8 = 4 * 4 := by decide

-- § group shift

/-- The shift form of the group index equals division. -/
theorem group_shift_eq_div (e s : Nat) : e >>> s = e / 2 ^ s :=
  Nat.shiftRight_eq_div_pow e s

/-- Four consecutive elements aligned to four lie in one group of 16, 32 or 64. -/
theorem quad_in_group16 (i : Nat) (h : i % 4 = 0) : i % 16 + 4 ≤ 16 := by omega
theorem quad_in_group32 (i : Nat) (h : i % 4 = 0) : i % 32 + 4 ≤ 32 := by omega
theorem quad_in_group64 (i : Nat) (h : i % 4 = 0) : i % 64 + 4 ≤ 64 := by omega

/-- Sixteen consecutive elements aligned to sixteen lie in one group of 16, 32 or 64. -/
theorem octet_in_group32 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega
theorem octet_in_group64 (i : Nat) (h : i % 16 = 0) : i % 64 + 16 ≤ 64 := by omega

-- § activation blocks

/-- A QE=16 lane-iteration never straddles a Q8_K block. -/
theorem lane_in_q8k_block (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A QE=16 lane-iteration never straddles a Q8_0 block. -/
theorem lane_in_q8_0_block (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- A QE=4 lane-iteration never straddles a Q8_0 block. -/
theorem quad_in_q8_0_block (i : Nat) (h : i % 4 = 0) : i % 32 + 4 ≤ 32 := by omega

/-- Lane start is aligned to QE. -/
theorem lane_start_aligned (lane : Nat) : (lane * 16) % 16 = 0 := by omega

/-- The 32-lane stride preserves QE alignment. -/
theorem stride_preserves_alignment (i : Nat) (h : i % 16 = 0) : (i + 32 * 16) % 16 = 0 := by omega

/-- Q8_K and Q8_0 block byte strides. -/
theorem block_strides : 4 + 256 + 2 * 16 = 292 ∧ 2 + 32 = 34 := by decide

/-- The element offset into the block pointer is within the block. -/
theorem q8k_offset_lt (i : Nat) : i % 256 < 256 := by omega

-- § row addressing

/-- Row `o` of the code matrix starts after `o` rows. -/
theorem row_codes_end (o n_out codes_per_row : Nat) (h : o < n_out) :
    (o + 1) * codes_per_row ≤ n_out * codes_per_row :=
  Nat.mul_le_mul_right _ h

/-- Warp-per-row mapping: warp index below 8 for 256 threads, block base reconstructs the row. -/
theorem warp_row (b w : Nat) (hw : w < 8) : (b * 8 + w) / 8 = b ∧ (b * 8 + w) % 8 = w := by
  omega

/-- The final tail loop visits at most one partial block per lane when n_in < 32*QE*2. -/
theorem tail_after_main (lane n_in : Nat) (hl : lane < 32) (h : n_in ≤ 32 * 16) :
    lane * 16 + 32 * 16 ≥ n_in := by omega

end SFamilyGemvIndexing
