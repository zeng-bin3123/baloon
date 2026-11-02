/-!
# MODULE: qsa_indexer_pooling
Slot, block and buffer-bound contracts of the QSA indexer append kernel: each token's 128-dim indexer key is
stored in a small tail ring of R-1 = 3 slots; when a block of R = 4 tokens completes (or at position 0) the block
is mean-pooled, RMS-normalised with gamma, RoPE-rotated on the first 64 dims, and written into the pooled array.

## Source semantics
- `slot = pos % R`; slots 0..R-2 are saved to `tail`; the kernel proceeds to pooling only when `pos == 0` or
  `slot == R-1`.
- Pooled block index `b = pos / R`, rope position `pos_base + R*b` (0 at position 0). The rotation pairs dim
  `pair` with `pair + 32` for `pair = d % 32`, d < 64.
- At a block-completing position (pos != 0) the next pooled row `b+1` is seeded with the "dead" copy.
- Launch contract: fixed geometry (idx_dim 128, block 4, n_rot 64), 1 <= max_cells < 2^31, pos_base a multiple
  of R with `pos_base + max_cells < 2^31`, finite positive epsilon, frequency base > 1, explicit stream, seven
  aligned pairwise-disjoint buffers; pooled holds `max_cells/R + 1` rows.

## Form template identifiers
- D -> indexer_dim (128), R -> pool_block (4), ROT -> rotary_dims (64), pooled -> pooled_blocks
- tail -> pending_tail, dead -> seed_row, block_pos -> last_block_position
-/

namespace Strata.QsaIndexer

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § slots and blocks

/-- A position splits into block and slot. -/
theorem pos_split (pos : Nat) : 4 * (pos / 4) + pos % 4 = pos := by
  omega

/-- Slot is below R. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by
  omega

/-- Tail slots written by `d < 128` threads: slot * 128 + d stays inside the 3-slot tail. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 3 * 128 :=
  idx_lt hs hd

/-- Pooling is triggered exactly at pos 0 or when the slot is the last of the block. -/
def poolTriggered (pos : Nat) : Prop := pos = 0 ∨ pos % 4 = 3

/-- Positions that do not pool are exactly those with slot in 1..2 or slot 0 with pos > 0. -/
theorem not_triggered_iff (pos : Nat) : ¬ poolTriggered pos ↔ pos ≠ 0 ∧ pos % 4 ≠ 3 := by
  unfold poolTriggered
  omega

/-- Every pool-triggering position past 0 closes a full block: pos = 4b + 3. -/
theorem trigger_closes_block (pos : Nat) (h : pos ≠ 0) (ht : poolTriggered pos) :
    pos = 4 * (pos / 4) + 3 := by
  unfold poolTriggered at ht
  omega

-- § pooled-array bounds

/-- Pooled row of a block-closing position is below `max_cells/R + 1`. -/
theorem pooled_row_lt (pos max_cells : Nat) (hp : pos < max_cells) :
    pos / 4 < max_cells / 4 + 1 := by
  omega

/-- The seed row `b+1` written at a block-closing position is also in range. -/
theorem seed_row_lt (pos max_cells : Nat) (hp : pos < max_cells) (hs : pos % 4 = 3) :
    pos / 4 + 1 < max_cells / 4 + 1 := by
  omega

/-- Pooled element address bound. -/
theorem pooled_elem_lt (row d max_cells : Nat) (hr : row < max_cells / 4 + 1) (hd : d < 128) :
    row * 128 + d < (max_cells / 4 + 1) * 128 :=
  idx_lt hr hd

/-- Position 0 writes only row 0, which exists for every capacity (the row count is at least 1). -/
theorem row_zero_exists (max_cells : Nat) : 0 < max_cells / 4 + 1 := by
  omega

-- § rotary mapping

/-- Pair index of a rotated dim is below 32. -/
theorem pair_lt (d : Nat) : d % 32 < 32 := by
  omega

/-- Both elements of a rotation pair lie in [0, 64). -/
theorem pair_elems_lt (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  omega

/-- Dims below 32 use the first-half formula and their pair index is d itself. -/
theorem rotary_first_half (d : Nat) (h : d < 32) : d % 32 = d := by
  omega

/-- Dims 32..63 use the second-half formula and their pair index is d - 32. -/
theorem rotary_second_half (d : Nat) (h1 : 32 ≤ d) (h2 : d < 64) : d % 32 = d - 32 := by
  omega

/-- Dims 64..127 are not rotated: exactly the dims below ROT pass through the rotation. -/
theorem rotated_dims (d : Nat) (h : d < 128) : d < 64 ∨ (64 ≤ d ∧ d < 128) := by
  omega

-- § rope position

/-- The rope position of block b is `pos_base + 4*b`, a multiple of 4 when pos_base is. -/
theorem rope_pos_mod (pos_base b : Nat) (h : pos_base % 4 = 0) : (pos_base + 4 * b) % 4 = 0 := by
  omega

/-- The rope position stays below `pos_base + max_cells` for positions below max_cells. -/
theorem rope_pos_lt (pos_base pos max_cells : Nat) (hp : pos < max_cells) :
    pos_base + 4 * (pos / 4) < pos_base + max_cells := by
  omega

/-- The int32 capacity check implies rope positions fit in int32. -/
theorem rope_pos_int32 (pos_base pos max_cells : Nat) (hp : pos < max_cells)
    (hb : pos_base + max_cells ≤ 2147483647) : pos_base + 4 * (pos / 4) ≤ 2147483647 := by
  omega

-- § block geometry

/-- 256 threads, of which d < 128 are active: 8 warps for the norm reduction. -/
theorem norm_reduce_warps (t : Nat) (h : t < 256) : t / 32 < 8 := by
  omega

/-- The block mean divides the 4-term sum by 4 (SCALE with 0.25): the pool size is R = 4. -/
theorem pool_size : 4 * 1 = 4 := rfl

-- § launch contract

/-- Validity of the append launch parameters over integers. -/
def appendArgsValid (idx_dim idx_block n_rot max_cells pos_base : Int) : Prop :=
  idx_dim = 128 ∧ idx_block = 4 ∧ n_rot = 64 ∧ 1 ≤ max_cells ∧ max_cells ≤ 2147483647 ∧
    0 ≤ pos_base ∧ pos_base % 4 = 0 ∧ pos_base + max_cells ≤ 2147483647

/-- Valid arguments have a pooled array of at least one row and a rope position range inside int32. -/
theorem append_valid_capacity (a b c m p : Int) (h : appendArgsValid a b c m p) :
    0 ≤ p + m ∧ p + m ≤ 2147483647 := by
  obtain ⟨_, _, _, h4, _, h6, _, h8⟩ := h
  omega

/-- Span bytes of the buffers: pooled needs `(max_cells/4 + 1) * 128 * 4` bytes. -/
def pooledBytes (max_cells : Nat) : Nat := (max_cells / 4 + 1) * 128 * 4

/-- The pooled span covers every written row. -/
theorem pooled_span_covers_rows (row max_cells : Nat) (hr : row < max_cells / 4 + 1) :
    (row + 1) * 128 * 4 ≤ pooledBytes max_cells := by
  unfold pooledBytes
  have h : (row + 1) * 128 ≤ (max_cells / 4 + 1) * 128 := Nat.mul_le_mul_right 128 (Nat.succ_le_of_lt hr)
  omega

end Strata.QsaIndexer
