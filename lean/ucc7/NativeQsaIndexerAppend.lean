/-!
# MODULE: native_qsa_indexer_append
SOURCES: src/kernels/cuda/native_qsa_indexer.cu
Native indexer key append adapted from llama.cpp: stores the f16-rounded raw key into a 3-row tail, and on a
block completion (or at position 0 for the spare/dead key) mean-pools, RMS-normalises with gamma, rotates the
first 64 dimensions NEOX-style at the block's first position, writes the pooled row and re-seeds the spare row.

## Source semantics
* D = 128, R = 4, ROT = 64, 256 threads (only `d < D` act).
* `slot = pos % R`; slots 0..2 are stored in the tail (`slot*D + d`); slot 3 completes block `b = pos / R`.
* Position 0 builds the dead key (rope position 0); later completions write `pooled[b]` and `pooled[b+1] = dead`.
* rope position is `pos_base + R*b`; `pos_base % R = 0` is required and `pos_base + max_cells <= INT32_MAX`.
* The pooled buffer has `max_cells / R + 1` rows.

## Form template identifiers
* `slot` -> tail_slot, `ropePos` -> block_first_position, `pairOf` -> rope_pair_index.
-/

namespace Ref.NativeQsaIndexer

/-- Tail slot of a position. -/
def slot (pos : Nat) : Nat := pos % 4

/-- Block of a position. -/
def block (pos : Nat) : Nat := pos / 4

/-- Rotation position of block `b` (for `pos ≠ 0`). -/
def ropePos (pos_base b : Nat) : Nat := pos_base + 4 * b

/-- Rope pair index of dimension `d`. -/
def pairOf (d : Nat) : Nat := d % 32

/-- Tail addresses `slot*128 + d` are inside the `3*128`-float tail. -/
theorem tail_addr_lt (s d : Nat) (hs : s < 3) (hd : d < 128) : s * 128 + d < 3 * 128 := by omega

/-- Tail addressing is injective. -/
theorem tail_addr_inj (s d s' d' : Nat) (hd : d < 128) (hd' : d' < 128)
    (h : s * 128 + d = s' * 128 + d') : s = s' ∧ d = d' := by omega

/-- A position is a block completion exactly when `slot = 3`. -/
theorem complete_iff (pos : Nat) : slot pos = 3 ↔ (pos + 1) % 4 = 0 := by unfold slot; omega

/-- Completion at `pos` finishes block `pos / 4`, whose cells are `4b .. 4b+3 = pos`. -/
theorem block_last_cell (pos : Nat) (h : slot pos = 3) : block pos * 4 + 3 = pos := by
  unfold slot block at *; omega

/-- On completion the spare row `b+1` exists in a pool of `max_cells/4 + 1` rows. -/
theorem spare_row_exists (pos max_cells : Nat) (hp : pos < max_cells) (h : slot pos = 3) :
    block pos + 1 < max_cells / 4 + 1 := by
  unfold slot block at *; omega

/-- The completed row `b` itself exists too. -/
theorem row_exists (pos max_cells : Nat) (hp : pos < max_cells) : block pos < max_cells / 4 + 1 := by
  unfold block; omega

/-- A completed block's rotation position stays 4-aligned when the base is. -/
theorem ropePos_aligned (pos_base b : Nat) (h : pos_base % 4 = 0) : ropePos pos_base b % 4 = 0 := by
  unfold ropePos; omega

/-- The rotation position fits the 32-bit position space under the validated bound. -/
theorem ropePos_in_range (pos_base pos max_cells : Nat) (hp : pos < max_cells)
    (hb : pos_base + max_cells ≤ 2147483647) : ropePos pos_base (block pos) ≤ 2147483647 := by
  unfold ropePos block; omega

/-- The rotation position is the block's FIRST cell offset, not its last. -/
theorem ropePos_first_cell (pos_base pos : Nat) : ropePos pos_base (block pos) = pos_base + block pos * 4 := by
  unfold ropePos; omega

/-- Rope pairs index within the first half of the rotated span. -/
theorem pair_lt (d : Nat) : pairOf d < 32 := by unfold pairOf; omega

/-- For `d < 32` the pair is `d` itself; its partner element is `d + 32 < 64`. -/
theorem pair_low (d : Nat) (h : d < 32) : pairOf d = d ∧ pairOf d + 32 < 64 := by unfold pairOf; omega

/-- For `32 ≤ d < 64` the pair is `d - 32` and its partner element is `d`. -/
theorem pair_high (d : Nat) (h1 : 32 ≤ d) (h2 : d < 64) : pairOf d = d - 32 ∧ pairOf d + 32 = d := by
  unfold pairOf; omega

/-- The rotated span is inside the 128-wide row, so dimensions `64..127` pass through unrotated. -/
theorem rot_span_in_row : 64 ≤ 128 := by decide

/-- Pooled addresses `b*128 + d` for `b < rows` stay inside `rows*128` floats. -/
theorem pooled_addr_lt (b d rows : Nat) (hb : b < rows) (hd : d < 128) : b * 128 + d < rows * 128 := by
  omega

/-- 256 threads cover the 128 dimensions with the rest guarded off. -/
theorem threads_cover_dims : 128 ≤ 256 := by decide

/-- Block-sum: 256 threads give 8 warp partials, read by lanes `< 8`. -/
theorem sum_partials : 256 / 32 = 8 := by decide

/-- Position 0 is its own slot-0 position and block 0. -/
theorem pos_zero : slot 0 = 0 ∧ block 0 = 0 := by decide

/-- Capacity guard: positions at or beyond `max_cells` are ignored (early return). -/
theorem guard_excludes (pos max_cells : Nat) (h : max_cells ≤ pos) : ¬ pos < max_cells := by omega

/-- The mean of four equal incoming values is that value (position-0 case): `4*x/4 = x` over naturals. -/
theorem mean_of_four_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

end Ref.NativeQsaIndexer
