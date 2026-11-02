/-!
# MODULE: s_gemv_quant_layout
Index and layout arithmetic of the quantised GEMV kernels: block strides, code widths, group shifts,
and the guarantee that a 16-element lane iteration never straddles an activation block.

## Source semantics
* Q8_K blocks are 292 bytes per 256 elements; Q8_0 blocks are 34 bytes per 32 elements.
* A code of width b bits is read as `(byte >> (k*b)) mod 2^b`.
* The group index is `i >> shift` for power-of-two groups.
* The IQ4_NL codebook is strictly increasing.

## Form template identifiers
* `s_gemv_q8_split_kernel` -> `quant_gemv_warp`; `kIq4Nl` -> `nl_codebook`
* `CODE_BITS` -> `code_bits`; `group_shift` -> `group_log2`; `QE` -> `lane_elems`
-/

namespace Strata.SGemv

-- § block strides

/-- Q8_K block size equals header, payload, and block sums. -/
theorem q8k_bytes : 292 = 4 + 256 + 16 * 2 := by decide
/-- Q8_0 block size equals header and payload. -/
theorem q8_0_bytes : 34 = 2 + 32 := by decide

/-- The Q8_K payload begins at a 4-byte aligned offset in every block whenever the base is aligned. -/
theorem q8k_block_aligned (b : Nat) : (b * 292) % 4 = 0 := by omega
/-- Q8_0 blocks start at even offsets, so the fp16 header is 2-aligned. -/
theorem q8_0_block_aligned (b : Nat) : (b * 34) % 2 = 0 := by omega

-- § lane iteration stays in one block

/-- A lane position that is a multiple of 16 leaves room for 16 elements in its Q8_K block. -/
theorem q8k_no_straddle (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega
/-- The same for Q8_0 blocks. -/
theorem q8_0_no_straddle (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Lane starting offsets `lane*16 + 512*t` are multiples of 16. -/
theorem lane_pos_mod (lane t : Nat) : (lane * 16 + 32 * 16 * t) % 16 = 0 := by omega

/-- The sixteen elements share one group for the three group sizes the format defines. -/
theorem quad_group16 (i : Nat) (h : i % 16 = 0) : i / 16 = (i + 15) / 16 := by omega
/-- Group size 32. -/
theorem quad_group32 (i : Nat) (h : i % 16 = 0) : i / 32 = (i + 15) / 32 := by omega
/-- Group size 64. -/
theorem quad_group64 (i : Nat) (h : i % 16 = 0) : i / 64 = (i + 15) / 64 := by omega

-- § code widths

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits
/-- Bytes in a four-code word. -/
def wordBytes (bits : Nat) : Nat := 4 * bits / 8

/-- Codes per byte for the three widths. -/
theorem perByte_vals : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide
/-- Four-code word bytes for the three widths. -/
theorem wordBytes_vals : wordBytes 2 = 1 ∧ wordBytes 4 = 2 ∧ wordBytes 8 = 4 := by decide

/-- A masked code is always below `2^bits`. -/
theorem code_lt (byte k bits : Nat) : (byte >>> (k * bits)) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- For 4-bit codes the extracted value is a valid codebook index. -/
theorem code4_lt16 (byte k : Nat) : (byte >>> (k * 4)) % 16 < 16 := Nat.mod_lt _ (by decide)

-- § group shift

/-- A shift by `s` equals division by `2^s`. -/
theorem shift_is_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

/-- The shifts for the three group sizes are 4, 5, 6. -/
theorem group_shifts : 2 ^ 4 = 16 ∧ 2 ^ 5 = 32 ∧ 2 ^ 6 = 64 := by decide

-- § codebook

/-- The IQ4_NL codebook values. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, so a 4-bit code always indexes it. -/
theorem iq4nl_len : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_sorted : iq4nl.Pairwise (· < ·) := by decide

/-- Every entry fits in int8. -/
theorem iq4nl_int8 : ∀ x ∈ iq4nl, -128 ≤ x ∧ x ≤ 127 := by decide

-- § launch shapes

/-- 256 threads form 8 warps, so blocks cover `n_out` rows with the ceiling quotient. -/
theorem warp_blocks_cover (n_out : Nat) : n_out ≤ (n_out + 8 - 1) / 8 * 8 := by omega

/-- The split kernel's tree reduction halves a power of two down to 1. -/
theorem halving_terminates (t : Nat) (h : 0 < t) : t / 2 < t := by omega

/-- The combined 16-way accumulator tree has exactly 16 leaves. -/
theorem acc_leaves : 2 * 2 * 2 * 2 = 16 := by decide

/-- `ffn_down_shexp`'s n_in = 640 is a multiple of 32 but not of 256. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

end Strata.SGemv
