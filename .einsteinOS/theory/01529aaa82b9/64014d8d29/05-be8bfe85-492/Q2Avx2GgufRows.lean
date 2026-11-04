/-!
# MODULE: q2_avx2_gguf_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
Q2_0 rows in the GGUF block layout (18 bytes per 64 weights) and the activation quantizer for CPUs
without AVX-512: 2-bit unpack addressing, chunk/scale association, block strides, token grouping.

## Source semantics
* Value `i` of a 64-value block lives in code byte `i / 4` at bit `2 * (i % 4)`; values 0..31 come from
  bytes 0..7 (the `lo` vector) and 32..63 from bytes 8..15 (`hi`).
* Block `b` of a row starts at `18 * b`: scale at `+0..1`, codes at `+2..17`.
* Value `i` of block `b` uses activation chunk `2*b + i/32`.
* Token groups have at most 4 tokens.

## Form template identifiers
* unpack64 -> unpack_codes_64, row_multi -> multi_token_row_dot, act_quant_q8_1_avx2 -> activation_quantizer
-/

namespace Strata.Q2Avx2

/-- Byte index holding value `i`. -/
def codeByte (i : Nat) : Nat := i / 4
/-- Bit shift of value `i` inside its byte. -/
def codeShift (i : Nat) : Nat := 2 * (i % 4)

/-- All 64 values read from within the 16 code bytes. -/
theorem code_byte_lt : ∀ i : Nat, i < 64 → codeByte i < 16 := by decide

/-- Shifts are 0, 2, 4 or 6, so a 2-bit field never straddles bytes. -/
theorem code_shift_le : ∀ i : Nat, i < 64 → codeShift i + 2 ≤ 8 := by decide

/-- The `lo` vector (values 0..31) uses bytes 0..7; the `hi` vector (32..63) uses bytes 8..15. -/
theorem lo_hi_split : ∀ i : Nat, i < 64 → (i < 32 → codeByte i < 8) ∧ (32 ≤ i → 8 ≤ codeByte i) := by
  decide

/-- Value index of a slot: inverse of (byte, shift). -/
theorem code_slot_roundtrip : ∀ i : Nat, i < 64 → 4 * codeByte i + codeShift i / 2 = i := by decide

/-- Distinct values never share a (byte, shift) slot. -/
theorem code_slot_injective (i j : Nat) (hi : i < 64) (hj : j < 64)
    (hb : codeByte i = codeByte j) (hs : codeShift i = codeShift j) : i = j := by
  have h1 := code_slot_roundtrip i hi
  have h2 := code_slot_roundtrip j hj
  rw [hb, hs] at h1
  omega

-- § GGUF block layout
/-- Start of block `b` in a row. -/
def blockStart (b : Nat) : Nat := 18 * b

/-- Scale bytes then code bytes fill the 18-byte block exactly. -/
theorem block_fields (b : Nat) : blockStart b + 2 + 16 = blockStart (b + 1) := by
  unfold blockStart; omega

/-- Block `b < nblocks` is inside a row of `18 * nblocks` bytes. -/
theorem block_in_row (b nblocks : Nat) (h : b < nblocks) : blockStart b + 18 ≤ 18 * nblocks := by
  unfold blockStart; omega

/-- Row `r` starts at `r * rowBytes`; rows are adjacent and disjoint. -/
theorem row_adjacent (r rowBytes : Nat) : r * rowBytes + rowBytes = (r + 1) * rowBytes := by
  rw [Nat.add_mul, Nat.one_mul]

-- § chunk association
/-- Activation chunk used by value `i` of block `b`. -/
def chunkOf (b i : Nat) : Nat := 2 * b + i / 32

/-- A block uses exactly two consecutive chunks. -/
theorem chunk_pair (b : Nat) : ∀ i : Nat, i < 64 → chunkOf b i = 2 * b ∨ chunkOf b i = 2 * b + 1 := by
  intro i hi
  unfold chunkOf
  omega

/-- Activation `q` offset of value `i` of block `b` (`b*64 + i`) lies in chunk `chunkOf b i`. -/
theorem activation_index_chunk (b i : Nat) (hi : i < 64) : (b * 64 + i) / 32 = chunkOf b i := by
  unfold chunkOf
  omega

-- § token grouping (avx2 handles 1..4 tokens per call)
/-- Group size at token offset `t0`. -/
def groupSize (nt t0 : Nat) : Nat := min 4 (nt - t0)

/-- Groups are non-empty and in range while tokens remain. -/
theorem group_ok (nt t0 : Nat) (h : t0 < nt) : 0 < groupSize nt t0 ∧ t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize; omega

/-- A group never exceeds the widest instantiated kernel. -/
theorem group_le_four (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize; omega

-- § quantizer
/-- Clamp to ±127 applied after the half-away rounding. -/
def clampQ (v : Int) : Int := max (-127) (min v 127)

/-- Clamped codes fit an `int8`. -/
theorem clampQ_int8 (v : Int) : -128 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ; omega

/-- Half-away-from-zero rounding step on an integer-scaled value `t2 = 2t`: `(t2 + sign) / 2`. -/
def roundHalfAway (t2 : Int) : Int := if 0 ≤ t2 then (t2 + 1) / 2 else (t2 - 1) / 2 + (if (t2 - 1) % 2 = 0 then 0 else 1)

/-- Ties round away from zero: `2.5 -> 3` and `-2.5 -> -3` (`t2 = ±5`). -/
theorem round_ties_away : roundHalfAway 5 = 3 ∧ roundHalfAway (-5) = -3 := by decide

/-- Chunk count of the quantizer for a width that is a multiple of 32. -/
theorem nchunks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by
  omega

end Strata.Q2Avx2
