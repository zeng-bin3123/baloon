/-!
# MODULE: native_qsa_indexer_append
Appends one raw indexer key (128-dim, F16 round-trip), keeps a 3-row raw tail, and on every completed block of
4 cells (or at position 0) pools, RMS-normalizes, rotates (first 64 dims, NEOX pairing) and writes a pooled row.

## Source semantics
* `slot = pos % 4`; slots 0..2 are stored in the tail, slot 3 completes a block and is consumed directly.
* Pooling fires iff `pos == 0` or `slot == 3`.
* Block `b = pos / 4`; rotation position is `pos_base + 4*b` (block's first cell), and `pos_base % 4 == 0`.
* Rotation pairs dim `d` with `d + 32` inside the first 64 dims.
* Pooled buffer holds `max_cells/4 + 1` rows: row `b` is the block, row `b+1` the spare (dead) row.

## Form template identifiers
* `D`, `R`, `ROT`       -> index dim 128, block cells 4, rotary dims 64
* `slotOf`, `blockOf`   -> `pos % R`, `pos / R`
* `pools`               -> pooling trigger predicate
-/

namespace Strata.NativeQsaIndexer

/-- Indexer key dimension. -/
def D : Nat := 128
/-- Cells per pooled block. -/
def R : Nat := 4
/-- Rotary dimension count. -/
def ROT : Nat := 64

/-- Slot of a position inside its block. -/
def slotOf (pos : Nat) : Nat := pos % R
/-- Block of a position. -/
def blockOf (pos : Nat) : Nat := pos / R

/-- Pooling trigger. -/
def pools (pos : Nat) : Prop := pos = 0 ∨ slotOf pos = R - 1

/-- A slot is always in range. -/
theorem slot_lt (pos : Nat) : slotOf pos < R := by
  unfold slotOf R
  omega

/-- Slot 3 is exactly the position whose successor starts a new block. -/
theorem last_slot_iff (pos : Nat) : slotOf pos = R - 1 ↔ (pos + 1) % 4 = 0 := by
  unfold slotOf R
  omega

/-- Tail writes (slots 0..2) land inside the `(R-1)*D` tail buffer. -/
theorem tail_index_in_range (slot d : Nat) (hs : slot < R - 1) (hd : d < D) :
    slot * D + d < (R - 1) * D := by
  unfold R D at *
  omega

/-- Block and slot recombine to the position. -/
theorem block_slot_decomposition (pos : Nat) : R * blockOf pos + slotOf pos = pos := by
  unfold blockOf slotOf R
  omega

/-- The spare row `b+1` is written only when a block completes (`slot = R-1`, `pos != 0`); in that case it fits
the `max_cells/4 + 1` row buffer when `pos < max_cells`. (Without completion, row `b+1` may be out of range,
which is why the kernel does not write it then.) -/
theorem pooled_rows_fit (pos max_cells d : Nat) (hp : pos < max_cells) (hd : d < D)
    (hs : slotOf pos = R - 1) :
    (blockOf pos + 1) * D + d < (max_cells / R + 1) * D := by
  unfold blockOf slotOf R D at *
  omega

/-- The block row itself fits as well. -/
theorem pooled_row_fits (pos max_cells d : Nat) (hp : pos < max_cells) (hd : d < D) :
    blockOf pos * D + d < (max_cells / R + 1) * D := by
  unfold blockOf R D at *
  omega

/-- Rotation position of the block's first cell. -/
def ropePos (pos_base pos : Nat) : Nat := pos_base + R * blockOf pos

/-- With an aligned base, the rotation position is block-aligned. -/
theorem rope_pos_aligned (pos_base pos : Nat) (h : pos_base % R = 0) : ropePos pos_base pos % R = 0 := by
  unfold ropePos blockOf R at *
  omega

/-- The rotation position never exceeds the current absolute position. -/
theorem rope_pos_le (pos_base pos : Nat) : ropePos pos_base pos ≤ pos_base + pos := by
  unfold ropePos blockOf R
  omega

/-- Capacity precondition keeps every rope position below INT32_MAX. -/
theorem rope_pos_int32 (pos_base pos max_cells : Nat) (hp : pos < max_cells)
    (hcap : pos_base + max_cells ≤ 2147483647) : ropePos pos_base pos ≤ 2147483647 := by
  unfold ropePos blockOf R
  omega

/-- NEOX pairing inside the first 64 dims: each `d < 64` maps to pair `d % 32`, partner `pair + 32`. -/
theorem neox_pairing (d : Nat) (hd : d < ROT) :
    d % (ROT / 2) < ROT / 2 ∧ (d < ROT / 2 → d % (ROT / 2) = d) ∧
    (ROT / 2 ≤ d → d % (ROT / 2) + ROT / 2 = d) := by
  unfold ROT at *
  omega

/-- The rotation pairs are a partition: the two halves map onto the same 32 pair indices. -/
theorem neox_two_halves (p : Nat) (hp : p < 32) :
    (p < 32 ∧ p % 32 = p) ∧ ((p + 32) % 32 = p ∧ 32 ≤ p + 32 ∧ p + 32 < 64) := by
  omega

/-- Position 0 is the only position pooled at slot 0; all other pooled positions have slot 3. -/
theorem pools_cases (pos : Nat) (h : pools pos) : pos = 0 ∨ pos % 4 = 3 := by
  unfold pools slotOf R at h
  omega

/-- Positions that do not pool are never at slot 3 (and are not 0). -/
theorem not_pool_slot (pos : Nat) (h : ¬ pools pos) : pos ≠ 0 ∧ pos % 4 ≠ 3 := by
  unfold pools slotOf R at h
  omega

end Strata.NativeQsaIndexer
