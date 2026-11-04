/-!
# MODULE: s_gemv_quantized
Packing geometry, group-shift indexing, lane-octet block containment and activation block layouts of the
S-family quantized GEMV kernels (S2/S4/S8 weights, fp16 / Q8_0 / Q8_K activations).

## Source semantics
Weights are CODE_BITS-wide codes packed little-endian; PER_BYTE = 8/bits, QW = 4*bits/8 bytes per four-code
word. Group sizes are powers of two, so `i / group = i >> shift`. In the split kernel a lane takes QE=16
consecutive elements starting at `lane*16 + 512*k`; because 16 divides both 32 and 256 the 16 elements never
straddle a Q8_0 (32) or Q8_K (256) block, which licenses hoisting the block address.

## Form template identifiers
- `perByte`/`qw` ↔ `PER_BYTE`/`QW`; `groupShift` ↔ `group_shift`; `laneStart` ↔ `i`
-/

namespace Strata.SGemvQuantized

-- § packing

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Bytes per four-code word. -/
def qw (bits : Nat) : Nat := 4 * bits / 8

/-- The three supported widths: 4, 2, 1 codes per byte. -/
theorem perByte_vals : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- The three supported widths: 1, 2, 4 bytes per four-code word. -/
theorem qw_vals : qw 2 = 1 ∧ qw 4 = 2 ∧ qw 8 = 4 := by decide

/-- A four-code word is exactly four codes of `bits` each. -/
theorem qw_bits (bits : Nat) (h : bits = 2 ∨ bits = 4 ∨ bits = 8) : qw bits * 8 = 4 * bits := by
  rcases h with rfl | rfl | rfl <;> decide

/-- Bytes in one row of codes for `nIn` elements at width 2/4/8. -/
theorem codes_per_row_s4 (nIn : Nat) (h : nIn % 2 = 0) : nIn / perByte 4 * perByte 4 = nIn := by
  unfold perByte; omega

-- § group shift

/-- Smallest `s` with `2^s ≥ g`. -/
def groupShift (g : Nat) : Nat := Nat.log2 (2 * g - 1)

/-- For the format's group sizes the shift reproduces the division. -/
theorem shift_matches : (groupShift 16, groupShift 32, groupShift 64) = (4, 5, 6) := by decide

/-- A right shift by the log is the division by the group size (g = 2^s). -/
theorem shift_is_div (i s : Nat) : i >>> s = i / 2 ^ s := by
  simp [Nat.shiftRight_eq_div_pow]

-- § lane octets never straddle a block

/-- Start of a lane's `k`-th octet: `lane*16 + 512*k`. -/
def laneStart (lane k : Nat) : Nat := lane * 16 + 512 * k

/-- Every start is a multiple of 16. -/
theorem laneStart_mod16 (lane k : Nat) : laneStart lane k % 16 = 0 := by
  unfold laneStart; omega

/-- 16 consecutive elements from a multiple of 16 stay inside one 256-element Q8_K block. -/
theorem octet_in_q8k_block (i j : Nat) (h : i % 16 = 0) (hj : j < 16) :
    (i + j) / 256 = i / 256 := by
  omega

/-- ... and inside one 32-element Q8_0 block. -/
theorem octet_in_q8_0_block (i j : Nat) (h : i % 16 = 0) (hj : j < 16) :
    (i + j) / 32 = i / 32 := by
  omega

/-- The in-block offset leaves room for the whole octet (Q8_K). -/
theorem octet_fits_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by
  omega

/-- The in-block offset leaves room for the whole octet (Q8_0). -/
theorem octet_fits_q8_0 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by
  omega

/-- All 16 elements of an octet also share one weight group when the group is a multiple of 16. -/
theorem octet_one_group (i j g : Nat) (hg : g = 16 ∨ g = 32 ∨ g = 64) (h : i % 16 = 0) (hj : j < 16) :
    (i + j) / g = i / g := by
  rcases hg with rfl | rfl | rfl <;> omega

-- § activation block layouts

/-- Q8_K block bytes (f32 d + 256 qs + 16 int16 bsums). -/
theorem q8k_bytes : 4 + 256 + 16 * 2 = 292 := by decide

/-- Q8_0 block bytes (fp16 d + 32 qs). -/
theorem q8_0_bytes : 2 + 32 = 34 := by decide

/-- Byte offset of element `i` in a Q8_0 buffer. -/
def q8_0At (i : Nat) : Nat := i / 32 * 34 + 2 + i % 32

/-- The element byte lies within its block. -/
theorem q8_0At_in_block (i : Nat) : i / 32 * 34 + 2 ≤ q8_0At i ∧ q8_0At i < (i / 32 + 1) * 34 := by
  unfold q8_0At; omega

/-- Each block's scale (offset `34*b`) is 2-byte aligned because the stride is even. -/
theorem q8_0_scale_aligned (b : Nat) : (b * 34) % 2 = 0 := by omega

/-- ... and each Q8_K scale is 4-byte aligned because 292 is divisible by 4. -/
theorem q8k_scale_aligned (b : Nat) : (b * 292) % 4 = 0 := by omega

/-- Q8_K is impossible for the shared-expert down input: 640 is not a multiple of 256 but is of 32. -/
theorem shexp_down_needs_q8_0 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

-- § accumulator tree

/-- Sum of the sixteen accumulators in the kernel's fixed pairing is order-independent in exact arithmetic. -/
theorem acc_tree_exact (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15)))
    = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

end Strata.SGemvQuantized
