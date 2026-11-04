/-!
# MODULE: iquant_block_decode
SOURCES: src/kernels/cpu/iq_avx512.cpp
Multi-token AVX-512 dot products for the i-quant formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21), IQ2_S(22): block byte sizes, in-bounds reads of each 64-value chunk decode, scale-lane
mapping, sign application and the token-chunking loop.

## Source semantics
* Each format has a block of 256 values with a fixed byte size; chunk `j` (0..3) decodes 64 values.
* Every byte read performed by `Fmt<TY>::decode` for `j < 4` stays inside the block.
* A scale lane `l` of 32 int16 lanes takes scale `l / 8`; scales are odd numbers `2*s+1`.
* Signs are applied with a masked negate, an involution.
* `dot_rows_nt` processes tokens in groups of at most 4 and covers all `nt` tokens.

## Form template identifiers
* Fmt<TY>::bytes -> block_bytes, Fmt<TY>::K -> format_constant (numerator over 8)
* iq512_supported -> supported_type_predicate
-/

namespace Strata.Iquant

-- § supported types and block sizes
/-- The five supported ggml type ids. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Block bytes per format. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Supported types are exactly those with a non-zero block size (checked over all type ids below 64). -/
theorem supported_iff_sized : ∀ ty : Nat, ty < 64 → (supported ty = true ↔ 0 < blockBytes ty) := by
  decide

/-- Block sizes decompose into the fp16 scale plus payload fields. -/
theorem block_byte_decomposition :
    blockBytes 16 = 2 + 64 ∧ blockBytes 17 = 2 + 64 + 8 ∧ blockBytes 22 = 2 + 64 + 8 + 8 ∧
    blockBytes 18 = 2 + 96 ∧ blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by
  decide

/-- Block size of IQ2_XXS. -/
theorem bb16 : blockBytes 16 = 66 := by decide
/-- Block size of IQ2_XS. -/
theorem bb17 : blockBytes 17 = 74 := by decide
/-- Block size of IQ3_XXS. -/
theorem bb18 : blockBytes 18 = 98 := by decide
/-- Block size of IQ3_S. -/
theorem bb21 : blockBytes 21 = 110 := by decide
/-- Block size of IQ2_S. -/
theorem bb22 : blockBytes 22 = 82 := by decide

-- § in-bounds reads of the chunk decode (j = 0..3)
/-- IQ2_XXS: the 16-byte read at `2 + 16 j` ends at or before byte 66. -/
theorem iq2xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  rw [bb16]; omega

/-- IQ2_XS: code read and the two scale bytes are in bounds. -/
theorem iq2xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  rw [bb17]; omega

/-- IQ2_S: grid bytes, sign bytes, high bits and scale bytes are all in bounds. -/
theorem iq2s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 22 ∧
    74 + 2 * j + 1 < blockBytes 22 := by
  rw [bb22]; omega

/-- IQ3_XXS: grid bytes and the 8-byte sign/scale words are in bounds. -/
theorem iq3xxs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ blockBytes 18 := by
  rw [bb18]; omega

/-- IQ3_S: grid, high-bit, sign and scale reads are in bounds. -/
theorem iq3s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  rw [bb21]; omega

-- § scale lanes
/-- Scale index of int16 lane `l` (lanes 0-7 -> s0 ... 24-31 -> s3). -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every one of the 32 lanes maps to one of the four scales. -/
theorem scale_lane_lt : ∀ l : Nat, l < 32 → scaleLane l < 4 := by decide

/-- Odd scale from a 4-bit field: `2*s + 1` lies in 1..31 and is odd. -/
theorem odd_scale (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

-- § signs
/-- Masked negate: negate exactly when the sign bit is set. -/
def applySign (bit : Bool) (y : Int) : Int := if bit then -y else y

/-- Applying the same sign twice restores the value. -/
theorem applySign_involutive (bit : Bool) (y : Int) : applySign bit (applySign bit y) = y := by
  cases bit <;> simp [applySign]

/-- An unset sign bit leaves the activation untouched. -/
theorem applySign_false (y : Int) : applySign false y = y := by
  simp [applySign]

-- § token chunking in dot_rows_nt
/-- Size of the group starting at token `t0` (at most 4). -/
def groupSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Each group is non-empty while tokens remain. -/
theorem group_nonempty (nt t0 : Nat) (h : t0 < nt) : 0 < groupSize nt t0 := by
  unfold groupSize; omega

/-- Each group stays inside the token range. -/
theorem group_in_range (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize; omega

/-- Each group has at most 4 tokens, matching the instantiated widths 1..4. -/
theorem group_le_four (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize; omega

-- § row addressing
/-- Row `r` of a matrix with `rowBytes` per row starts at `r * rowBytes`; consecutive rows do not overlap. -/
theorem rows_disjoint (r rowBytes : Nat) : r * rowBytes + rowBytes = (r + 1) * rowBytes := by
  rw [Nat.add_mul, Nat.one_mul]

/-- Block `i` of a row starts at `i * bytes` and stays within the row when `i < nblocks`. -/
theorem block_in_row (i nblocks bytes : Nat) (h : i < nblocks) : i * bytes + bytes ≤ nblocks * bytes := by
  have : (i + 1) * bytes ≤ nblocks * bytes := Nat.mul_le_mul_right bytes h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

end Strata.Iquant
