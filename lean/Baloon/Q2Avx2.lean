-- Baloon.Q2Avx2: Q2_0 AVX2 kernel code-byte addressing, GGUF block layout,
-- chunk association and token grouping invariants.
-- Independent implementation proving the same mathematical facts as Ref.Q2Avx2.

namespace Baloon.Q2Avx2

def codeByte (i : Nat) : Nat := (i / 4)
def codeShift (i : Nat) : Nat := (2 * (i % 4))

theorem code_byte_lt : ∀ i : Nat, i < 64 → codeByte i < 16 := by { decide }
theorem code_shift_le : ∀ i : Nat, i < 64 → codeShift i + 2 ≤ 8 := by { decide }

theorem lo_hi_split : ∀ i : Nat, i < 64 →
    (i < 32 → codeByte i < 8) ∧ (32 ≤ i → 8 ≤ codeByte i) := by { decide }

theorem code_slot_roundtrip : ∀ i : Nat, i < 64 → 4 * codeByte i + codeShift i / 2 = i := by { decide }

theorem code_slot_injective (i j : Nat) (hi : i < 64) (hj : j < 64)
    (hb : codeByte i = codeByte j) (hs : codeShift i = codeShift j) : i = j := by
  have h1 := code_slot_roundtrip i hi
  have h2 := code_slot_roundtrip j hj
  rw [hb, hs] at h1; omega

def blockStart (b : Nat) : Nat := (18 * b)

theorem block_fields (b : Nat) : blockStart b + 2 + 16 = blockStart (b + 1) := by { unfold blockStart; omega }

theorem block_in_row (b nblocks : Nat) (h : b < nblocks) : blockStart b + 18 ≤ 18 * nblocks := by { unfold blockStart; omega }

theorem row_adjacent (r rowBytes : Nat) : r * rowBytes + rowBytes = (r + 1) * rowBytes := by { rw [Nat.add_mul, Nat.one_mul] }

def chunkOf (b i : Nat) : Nat := (2 * b + i / 32)

theorem chunk_pair (b : Nat) : ∀ i : Nat, i < 64 → chunkOf b i = 2 * b ∨ chunkOf b i = 2 * b + 1 := by
  intro i hi; unfold chunkOf; omega

theorem activation_index_chunk (b i : Nat) (hi : i < 64) : (b * 64 + i) / 32 = chunkOf b i := by
  unfold chunkOf; omega

-- ── token grouping and quantizer ─────────────────────────────────────────────

/-- Group size at token offset t0 (avx2 handles 1..4 tokens per call). -/
def groupSize (nt t0 : Nat) : Nat := (min 4 (nt - t0))

/-- Groups are non-empty and in range while tokens remain. -/
theorem group_ok (nt t0 : Nat) (h : t0 < nt) : 0 < groupSize nt t0 ∧ t0 + groupSize nt t0 ≤ nt := by { unfold groupSize; omega }

/-- A group never exceeds the widest instantiated kernel. -/
theorem group_le_four (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by { unfold groupSize; omega }
/-- Clamp to +-127 applied after the half-away rounding. -/
def clampQ (v : Int) : Int := (max (-127) (min v 127))

/-- Clamped codes fit an int8. -/
theorem clampQ_int8 (v : Int) : -128 ≤ clampQ v ∧ clampQ v ≤ 127 := by { unfold clampQ; omega }

/-- Half-away-from-zero rounding: (t2 + sign(t2)) / 2, negation-folded for the negative branch. -/
def roundHalfAway (t2 : Int) : Int :=
  if 0 ≤ t2 then (t2 + 1) / 2 else -((1 - t2) / 2)

/-- Ties round away from zero: 2.5 -> 3 and -2.5 -> -3 (t2 = ±5). -/
theorem round_ties_away : roundHalfAway 5 = 3 ∧ roundHalfAway (-5) = -3 := by { decide }

/-- Chunk count of the quantizer for a width that is a multiple of 32. -/
theorem nchunks_exact (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by { omega }

end Baloon.Q2Avx2
