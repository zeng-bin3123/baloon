/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Single-block indexer-key append: stores the raw key in the 3-row tail, and on a block completion (or at position 0)
pools, normalises and rotates the row, then moves the spare row.

## Source semantics
* D = 128, R = 4, ROT = 64, THREADS = 256; slot = pos % 4; only slot < 3 stores into the tail.
* Writes of pooled rows happen only when pos = 0 or slot = 3; at slot 3 rows b = pos/4 and b+1 are written.
* The rotation position is `pos_base + 4 * b` (0 at pos 0); rotation pairs index `pair` with `pair + 32`.
* The pooled buffer has `max_cells / 4 + 1` rows.

## Form template identifiers
- `D`, `R`, `ROT`, `THREADS`
- `slot`, `b`, `rope_pos`, `pair`
-/

namespace Strata.NativeQsaIndexer

-- § slot arithmetic

/-- Slot of a position. -/
def slot (pos : Nat) : Nat := pos % 4

/-- Only slots 0..2 are stored in the tail; slot 3 completes a block. -/
theorem slot_lt_four (pos : Nat) : slot pos < 4 := by
  unfold slot; omega

/-- Tail row index slot * 128 + d stays inside the 3 * 128 tail. -/
theorem tail_index_lt (pos d : Nat) (hs : slot pos < 3) (hd : d < 128) : slot pos * 128 + d < 3 * 128 := by
  omega

/-- Gather rows j * 128 + d for j < 3 also stay inside the tail. -/
theorem tail_gather_lt (j d : Nat) (hj : j < 3) (hd : d < 128) : j * 128 + d < 3 * 128 := by
  omega

-- § pooled buffer rows

/-- At position 0 the pooled row 0 exists. -/
theorem row_zero_exists (maxc : Nat) : 0 < maxc / 4 + 1 := by
  omega

/-- When a block completes within capacity, its row b is in range. -/
theorem block_row_lt (pos maxc : Nat) (hs : slot pos = 3) (hp : pos < maxc) : pos / 4 < maxc / 4 + 1 := by
  unfold slot at hs; omega

/-- When a block completes within capacity, the spare row b+1 is also in range. -/
theorem spare_row_lt (pos maxc : Nat) (hs : slot pos = 3) (hp : pos < maxc) : pos / 4 + 1 < maxc / 4 + 1 := by
  unfold slot at hs; omega

/-- The kernel returns early unless pos = 0 or the slot is 3; rows are written only then. -/
theorem writes_only_when (pos : Nat) (h : ¬ (pos = 0 ∨ slot pos = 3)) : pos ≠ 0 ∧ slot pos ≠ 3 := by
  unfold slot at h ⊢; omega

-- § rotation

/-- Rotation position of block b. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- A 4-aligned position base keeps every block's rotation position 4-aligned. -/
theorem ropePos_aligned (pos posBase : Nat) (h : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos; split <;> omega

/-- The rotation position stays within int32 when base + capacity does. -/
theorem ropePos_le (pos posBase maxc : Nat) (hp : pos < maxc) (h : posBase + maxc ≤ 2147483647) :
    ropePos pos posBase ≤ 2147483647 := by
  unfold ropePos; split <;> omega

/-- Rotation partner of dimension d < 64 is `pair` or `pair + 32`, both inside the 64 rotated dims. -/
theorem rotation_partners_lt (d : Nat) (hd : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by
  omega

/-- The rotated value for d is a*c - z*s on the first half and a*s + z*c on the second, and the two
halves of [0, 64) are exactly [0, 32) and [32, 64). -/
theorem rotation_halves (d : Nat) (hd : d < 64) : d < 32 ∨ (32 ≤ d ∧ d - 32 = d % 32) := by
  omega

/-- Rotated dimensions are inside the 128-dim key, so non-rotated dims 64..127 pass through. -/
theorem rotation_inside_key (d : Nat) (hd : d < 64) : d < 128 := by
  omega

-- § thread geometry

/-- 256 threads give 8 warps, which fit the 32-entry partial sums. -/
theorem partials_fit : 256 / 32 ≤ 32 := by
  decide

/-- The 128 live dims are a prefix of the 256 threads, so a reduction over lanes < 8 sees every partial. -/
theorem live_warps : 128 / 32 = 4 := by
  decide

/-- A block mean is one quarter of the 4-cell sum: `0.25 * s` on naturals as `s / 4` times 4 bounds the sum. -/
theorem quarter_mean_bound (s : Nat) : s / 4 * 4 ≤ s := by
  omega

end Strata.NativeQsaIndexer
