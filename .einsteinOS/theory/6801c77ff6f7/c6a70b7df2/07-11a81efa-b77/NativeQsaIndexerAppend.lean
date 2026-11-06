/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Single-block indexer key append and block pooling bit-matched to llama.cpp: stores the raw tail,
pools on block completion, rotates the pooled row at the block's first cell, and rewrites the spare slot.

## Source semantics
* slot = pos % 4; tail row stored iff slot < 3; the block completes iff slot == 3 (or pos == 0 for the spare).
* rope position = 0 at pos 0, else pos_base + 4 * (pos / 4); pos_base must be a multiple of 4.
* pooled row b = pos / 4, spare row b + 1; at pos 0 the four gathered indices all name cell zero.
* NEOX rotation pairs d with d + 32 over n_rot = 64.
* Buffers must be pairwise non-overlapping.

## Form template identifiers
ropePos <- rope_pos, partner <- pair/pair + ROT/2, overlaps <- `overlaps(Span, Span)`.
-/

namespace Strata.NativeIndexer

-- § slots and rope positions

/-- Rotation position of the row completed at `pos`. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With an aligned `pos_base` the rotation position is block aligned. -/
theorem ropePos_aligned {pos posBase : Nat} (hb : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos; split <;> omega

/-- On completion the rotation position is the block's FIRST cell. -/
theorem ropePos_first_cell {pos posBase : Nat} (hs : pos % 4 = 3) :
    ropePos pos posBase = posBase + (pos - 3) := by
  unfold ropePos; split <;> omega

/-- It is never the block's last cell. -/
theorem ropePos_ne_last {pos posBase : Nat} (hs : pos % 4 = 3) :
    ropePos pos posBase ≠ posBase + pos := by
  unfold ropePos; split <;> omega

/-- Rotation positions stay in int32 under the validated capacity. -/
theorem ropePos_int32 {pos posBase maxCells : Nat} (hp : pos < maxCells)
    (hb : posBase + maxCells ≤ 2147483647) : ropePos pos posBase < 2147483648 := by
  unfold ropePos; split <;> omega

/-- A block completes exactly when the next cell count is a multiple of 4. -/
theorem completes_iff {pos : Nat} : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- Completing a block raises the block count by one. -/
theorem completion_blocks {pos : Nat} (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Only slots 0..2 are stored in the three-row tail. -/
theorem tail_slot_lt {pos : Nat} (h : pos % 4 < 3) : pos % 4 < 4 - 1 := by omega

-- § pooled row bounds

/-- The completed row is inside the pooled buffer of `max_cells / 4 + 1` rows. -/
theorem pooled_row_lt {pos maxCells : Nat} (h : pos < maxCells) : pos / 4 < maxCells / 4 + 1 := by
  omega

/-- The spare row written on completion is also inside it. -/
theorem spare_row_lt {pos maxCells : Nat} (h : pos < maxCells) (hs : pos % 4 = 3) :
    pos / 4 + 1 < maxCells / 4 + 1 := by
  omega

-- § pooling arithmetic

/-- At pos 0 the four gathered indices all name cell zero, so the sum is four copies. -/
theorem spare_sum (x : Int) : x + x + x + x = 4 * x := by omega

/-- The 0.25 scale of that sum returns the key itself (integer model). -/
theorem spare_mean (x : Int) : (x + x + x + x) / 4 = x := by omega

-- § NEOX pairing

/-- Partner dimension of the rotation over 64 rotated dims. -/
def partner (d : Nat) : Nat := if d < 32 then d + 32 else d - 32

/-- The pairing is an involution. -/
theorem partner_involutive {d : Nat} (h : d < 64) : partner (partner d) = d := by
  unfold partner; split <;> split <;> omega

/-- Partners stay inside the 64 rotated dimensions. -/
theorem partner_lt {d : Nat} (h : d < 64) : partner d < 64 := by
  unfold partner; split <;> omega

/-- Every rotated dimension `d` maps to a pair index `d % 32` below 32. -/
theorem pair_lt {d : Nat} (h : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by omega

-- § span disjointness

/-- Half-open byte-range overlap of (address, length) spans. -/
def overlaps (x xn y yn : Nat) : Prop := x < y + yn ∧ y < x + xn

/-- Overlap is symmetric, so checking pairs `i < j` suffices. -/
theorem overlaps_symm (x xn y yn : Nat) : overlaps x xn y yn ↔ overlaps y yn x xn := by
  unfold overlaps; omega

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (x xn : Nat) : ¬ overlaps x xn (x + xn) 4 := by
  unfold overlaps; omega

end Strata.NativeIndexer
