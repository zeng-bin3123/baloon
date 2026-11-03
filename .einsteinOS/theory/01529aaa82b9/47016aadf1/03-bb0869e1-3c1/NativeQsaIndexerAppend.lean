/-!
# MODULE: native_qsa_indexer_append
Slot/pooling bookkeeping of the native QSA indexer key append (one block of 256 threads, 128 active dims).

## Source semantics
Position pos has slot pos % 4. Slots 0..2 are stored in the tail; slot 3 completes a block and is consumed
directly. At pos = 0 the spare ("dead") key is computed from cell 0 and pooled row 0 is written.
On completion block b = pos/4 is written and the spare moves to row b+1. Rope position of the block is
pos_base + 4*b (0 at pos 0); the first 64 dims rotate in pairs (d, d+32).
Validation: aligned pos_base, pos_base + max_cells <= INT32_MAX, pooled has max_cells/4 + 1 rows.

## Form template identifiers
D -> 128; R -> 4; ROT -> 64; THREADS -> 256; tail -> (R-1)*D floats; dead -> spare key; pooled -> rows.
-/
namespace Strata.NativeQsaIndexer

/-- A position recomposes from block and slot. -/
theorem pos_decomp (pos : Nat) : (pos / 4) * 4 + pos % 4 = pos := by omega

/-- The tail slot index is below R-1 whenever the block does not complete. -/
theorem tail_slot_lt (pos : Nat) (h : pos % 4 ≠ 3) : pos % 4 < 3 := by omega

/-- Tail index slot*128 + d stays within the 3*128 tail buffer. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 := by omega

/-- Tail indices are unique per (slot, dim). -/
theorem tail_inj (s d s' d' : Nat) (hd : d < 128) (hd' : d' < 128) (e : s * 128 + d = s' * 128 + d') :
    s = s' ∧ d = d' := by omega

/-- A block completes exactly when slot = 3. -/
theorem completes (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the block uses slots 0..2 from the tail and slot 3 from the incoming key: four cells. -/
theorem block_cells : 3 + 1 = 4 := by decide

/-- Pooled row b+1 (the spare) exists for every pos < max_cells. -/
theorem spare_row_exists (pos mc : Nat) (h : pos < mc) : pos / 4 + 1 < mc / 4 + 1 + 1 := by omega

/-- Pooled row b is within capacity max_cells/4 + 1. -/
theorem block_row_exists (pos mc : Nat) (h : pos < mc) : pos / 4 < mc / 4 + 1 := by omega

/-- The spare row on completion also fits: (pos/4 + 1) <= mc/4 + 1 when pos < mc. -/
theorem spare_row_fits (pos mc : Nat) (h : pos < mc) : pos / 4 + 1 ≤ mc / 4 + 1 := by omega

/-- Pooled element index row*128 + d stays within (mc/4+1)*128. -/
theorem pooled_index_lt (row d mc : Nat) (hr : row < mc / 4 + 1) (hd : d < 128) :
    row * 128 + d < (mc / 4 + 1) * 128 := by omega

/-- With pos_base a multiple of 4, the rope position pos_base + 4b is a multiple of 4. -/
theorem rope_pos_aligned (pb b : Nat) (h : pb % 4 = 0) : (pb + 4 * b) % 4 = 0 := by omega

/-- The rope position never exceeds pos_base + pos. -/
theorem rope_pos_le (pb pos : Nat) : pb + 4 * (pos / 4) ≤ pb + pos := by omega

/-- Overflow guard: pos_base + max_cells <= INT32_MAX bounds every rope position. -/
theorem rope_pos_bound (pb pos mc : Nat) (h : pos < mc) (hb : pb + mc ≤ 2147483647) :
    pb + 4 * (pos / 4) ≤ 2147483647 := by omega

/-- Rotation pairs: d < 32 pairs with d + 32, covering the 64 rotated dims exactly. -/
theorem rot_pairs (d : Nat) (h : d < 64) : (d < 32 ∧ d + 32 < 64) ∨ (32 ≤ d ∧ d - 32 < 32) := by omega

/-- The pair index of a rotated dim is below ROT/2. -/
theorem pair_lt (d : Nat) (h : d < 64) : d % 32 < 32 := by omega

/-- Warp partials: 128 active threads fill four warps; 8 partial slots are enough for 256 threads. -/
theorem partial_slots : 256 / 32 = 8 ∧ 128 / 32 = 4 := by decide

/-- Mean of four equal values (the sum is four copies, scale 0.25). -/
theorem mean_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

/-- Span size of tail: (R-1)*D*4 bytes. -/
theorem tail_bytes : 3 * 128 * 4 = 1536 := by decide

/-- Overlap predicate used by the host checks. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

/-- Non-overlap gives separation. -/
theorem not_overlap_sep (a an b bn : Nat) (h : ¬ overlap a an b bn) : b + bn ≤ a ∨ a + an ≤ b := by
  unfold overlap at h; omega

/-- Valid spans do not wrap. -/
theorem span_no_wrap (addr bytes umax : Nat) (h : bytes ≤ umax - addr) (h2 : addr ≤ umax) :
    addr + bytes ≤ umax := by omega

end Strata.NativeQsaIndexer
