/-!
# MODULE: native_qsa_indexer_append
Single-block append of one raw indexer key and, on block completion, pooling plus RoPE of the pooled row
(`src/kernels/cuda/native_qsa_indexer.cu`); geometry fixed at 128 dims, blocks of 4 cells, 64 rotary dims.

## Source semantics
* `slot = pos % 4`; slots 0..2 are stored in `tail` (3 rows of 128), slot 3 completes a block.
* The kernel works only for `pos = 0` (spare/first row) or `slot = 3`.
* A completed block `b = pos / 4` writes `pooled[b]` and moves the spare to `pooled[b+1]`.
* Rotation position is `pos_base + 4*b` (0 for `pos = 0`); pairs are `(pair, pair + 32)` with `pair = d % 32`.
* `pooled` holds `max_cells / 4 + 1` rows.

## Form template identifiers
* `slotOf`      <- `pos % R`
* `ropePos`     <- `pos == 0 ? 0 : pos_base + R * b`
* `pairOf`      <- `d % (ROT / 2)`
-/

namespace NativeQsaIndexerAppend

-- § slots and tail

/-- Slot of a position within its block. -/
def slotOf (pos : Nat) : Nat := pos % 4

/-- Tail rows are written only for slots 0..2 and stay inside the 3x128 tail buffer. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 384 := by
  omega

/-- The slot-3 cell is the one whose successor starts a new block. -/
theorem slot3_iff_next_block (pos : Nat) : slotOf pos = 3 ↔ (pos + 1) % 4 = 0 := by
  unfold slotOf
  omega

/-- After the first cell, the kernel does work only on slot 3: otherwise it returns early. -/
theorem active_iff (pos : Nat) : (pos = 0 ∨ slotOf pos = 3) ↔ (pos = 0 ∨ (pos + 1) % 4 = 0) := by
  unfold slotOf
  omega

-- § pooled rows

/-- A non-first active cell completes a block, so its block index is the previous completed count. -/
theorem block_complete_index (pos : Nat) (h0 : pos ≠ 0) (h3 : slotOf pos = 3) :
    4 * (pos / 4) + 3 = pos := by
  unfold slotOf at h3
  omega

/-- The spare row `b + 1` is inside the `max_cells / 4 + 1` rows whenever a block completes. -/
theorem spare_row_in_range (pos maxCells : Nat) (hp : pos < maxCells) (h3 : slotOf pos = 3) :
    pos / 4 + 1 ≤ maxCells / 4 := by
  unfold slotOf at h3
  omega

/-- ... so the element writes `(b+1)*128 + d` stay inside the pooled buffer. -/
theorem spare_index_lt (pos maxCells d : Nat) (hp : pos < maxCells) (h3 : slotOf pos = 3)
    (hd : d < 128) : (pos / 4 + 1) * 128 + d < (maxCells / 4 + 1) * 128 := by
  unfold slotOf at h3
  omega

/-- The row written for the completed (or first) block is inside the buffer as well. -/
theorem block_index_lt (pos maxCells d : Nat) (hp : pos < maxCells) (hd : d < 128) :
    (pos / 4) * 128 + d < (maxCells / 4 + 1) * 128 := by
  omega

-- § rotation

/-- Rotation position: 0 for the very first cell, else the block's first position. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With an aligned base the rotation position is block-aligned. -/
theorem ropePos_aligned (pos posBase : Nat) (hb : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos
  split <;> omega

/-- The rotation position never exceeds `pos_base + pos`. -/
theorem ropePos_le (pos posBase : Nat) : ropePos pos posBase ≤ posBase + pos := by
  unfold ropePos
  split <;> omega

/-- It fits in an int32 under the host's capacity check. -/
theorem ropePos_lt (pos posBase maxCells : Nat) (hp : pos < maxCells)
    (hc : posBase + maxCells ≤ 2147483647) : ropePos pos posBase < 2147483647 := by
  have := ropePos_le pos posBase
  omega

/-- Rotary pair index of dimension `d < 64`. -/
def pairOf (d : Nat) : Nat := d % 32

/-- Both elements of a rotary pair lie in the 64 rotated dimensions. -/
theorem pair_in_range (d : Nat) (hd : d < 64) : pairOf d < 32 ∧ pairOf d + 32 < 64 := by
  unfold pairOf
  omega

/-- The first half of the rotated dims reads `(d, d+32)`, the second `(d-32, d)`. -/
theorem pair_partner (d : Nat) (hd : d < 64) :
    (d < 32 → pairOf d = d) ∧ (32 ≤ d → pairOf d + 32 = d) := by
  unfold pairOf
  constructor <;> intro h <;> omega

end NativeQsaIndexerAppend
