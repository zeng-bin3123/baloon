/-!
# MODULE: s_gemv_quad_indexing
Index arithmetic of the S-family GEMV kernels (`s_gemv.cu`): shift-for-division group index, per-lane
16-element octets that never straddle an activation block, and the code-word width per bit width.

## Source semantics
`group_elems` is a power of two, passed as `group_shift`; `e >> shift = e / 2^shift`. A lane takes 16
consecutive elements starting at a multiple of 16, so they lie in one Q8_K (256) or Q8_0 (32) block.
`QW = 4*CODE_BITS/8` bytes hold four codes.

## Form template identifiers
groupShift -> group_shift; octet base -> i; blkElems -> Q8K_BLOCK_ELEMS / Q8_0_BLOCK_ELEMS.
-/

namespace Strata.SGemvQuadIndexing

-- § shift replaces division

/-- Group index by shift equals group index by division. -/
theorem group_by_shift (i s : Nat) : i >>> s = i / 2 ^ s := by
  exact Nat.shiftRight_eq_div_pow i s

/-- For the three defined group sizes the shift is exact. -/
theorem group_sizes : (2 ^ 4 = 16) ∧ (2 ^ 5 = 32) ∧ (2 ^ 6 = 64) := by
  decide

-- § octets stay inside a block

/-- A 16-aligned base leaves room for 16 elements inside a 256-element Q8_K block. -/
theorem octet_in_q8k_block (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by
  omega

/-- Likewise for a 32-element Q8_0 block. -/
theorem octet_in_q8_0_block (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by
  omega

/-- All sixteen elements of the lane's step share the block index of the base. -/
theorem octet_same_block_q8_0 (i k : Nat) (h : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 := by
  omega

theorem octet_same_block_q8k (i k : Nat) (h : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 := by
  omega

/-- The lane stride advances by a multiple of 16, so alignment is preserved every iteration. -/
theorem stride_preserves_alignment (i : Nat) (h : i % 16 = 0) : (i + 32 * 16) % 16 = 0 := by
  omega

-- § code word widths

def codeWordBytes (codeBits : Nat) : Nat := 4 * codeBits / 8
def perByte (codeBits : Nat) : Nat := 8 / codeBits

/-- QW is 1, 2, 4 for S2, S4, S8. -/
theorem code_word_widths : codeWordBytes 2 = 1 ∧ codeWordBytes 4 = 2 ∧ codeWordBytes 8 = 4 := by
  decide

/-- Four codes per word, and `PER_BYTE` divides the byte. -/
theorem per_byte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by
  decide

/-- Q8_K needs n_in multiple of 256; `ffn_down_shexp` n_in = 640 is not, but is a multiple of 32. -/
theorem q8k_impossible_for_640 : (640 % 256 ≠ 0) ∧ (640 % 32 = 0) := by
  decide

/-- A multiple of 256 is a multiple of 32, so Q8_0 is the weaker requirement. -/
theorem q8k_implies_q8_0 (n : Nat) (h : n % 256 = 0) : n % 32 = 0 := by
  omega

end Strata.SGemvQuadIndexing
