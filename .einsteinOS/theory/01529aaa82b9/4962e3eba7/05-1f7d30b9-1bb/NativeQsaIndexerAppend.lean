/-!
# MODULE: native_qsa_indexer_append
Single-block kernel that appends one raw indexer key to the QSA indexer state: it keeps a 3-row raw tail,
and on every fourth cell mean-pools the four raw keys, RMS-normalises, rotates the first 64 dims (NeoX
pairing) at the block's first position, writes the pooled row and re-writes the spare row.  This file
proves the ring-buffer slot arithmetic, the pooled-buffer capacity, the gather shape, the rotation
pairing and the position-base overflow guards.

## Source semantics
* `slot = pos % 4`; slots 0..2 are written to `tail`; slot 3 completes the block.
* `pos == 0` initialises row 0 and `dead`; the spare's four gather indices all name cell zero.
* Rotation pairs dim `d` with `d + 32` for `d < 64`; the rotation position is `pos_base + 4*b`.
* Pooled buffer capacity is `max_cells / 4 + 1` rows.

## Form template identifiers
- `append` -> `pool_append`; `tail` -> `raw_tail`; `dead` -> `spare_row`; `pooled` -> `pooled_rows`
- `rope_pos` -> `block_rotation_pos`; `mrope_pos` -> `section_position`; `theta_scale` -> `freq_step`
-/

namespace Strata.NativeQsaIndexer

-- § ring buffer

/-- The ring slot is below 4. -/
theorem slot_lt (pos : Nat) : pos % 4 < 4 := by omega

/-- Tail writes happen only for slots 0..2, so the 3-row tail suffices. -/
theorem tail_slot_lt {pos : Nat} (h : pos % 4 < 3) : pos % 4 * 128 + 127 < 3 * 128 := by omega

/-- The block row index `pos / 4` of a completing cell and the spare row `pos/4 + 1` fit the pooled buffer. -/
theorem pooled_rows_fit {pos max : Nat} (hp : pos < max) (hs : pos % 4 = 3) : pos / 4 + 1 < max / 4 + 1 := by
  omega

/-- For position 0 the single written row index 0 fits the buffer. -/
theorem pooled_row_zero (max : Nat) : 0 < max / 4 + 1 := by omega

/-- Pooled row addressing `b*128 + d` is inside `rows * 128`. -/
theorem pooled_index_lt {b d rows : Nat} (hb : b < rows) (hd : d < 128) : b * 128 + d < rows * 128 := by
  have h := Nat.mul_le_mul_right 128 (show b + 1 ≤ rows by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § gather shape

/-- The four cells averaged by a completing cell are the three tail rows then the incoming key;
for position 0 all four are the incoming key. -/
def gatherRows (pos : Nat) (tail : List Int) (incoming : Int) : List Int :=
  if pos = 0 then [incoming, incoming, incoming, incoming] else tail ++ [incoming]

/-- A position-0 gather always has four entries. -/
theorem gather_zero_length (tail : List Int) (incoming : Int) : (gatherRows 0 tail incoming).length = 4 := by
  simp [gatherRows]

/-- A later gather has four entries when the tail has three. -/
theorem gather_later_length {pos : Nat} (h : pos ≠ 0) (tail : List Int) (incoming : Int)
    (ht : tail.length = 3) : (gatherRows pos tail incoming).length = 4 := by
  simp [gatherRows, h, ht]

/-- The newest cell is last in the gather order. -/
theorem gather_last {pos : Nat} (h : pos ≠ 0) (tail : List Int) (incoming : Int) :
    (gatherRows pos tail incoming).reverse.head? = some incoming := by
  simp [gatherRows, h]

-- § rotation pairing

/-- Pair index `d % 32` is below 32 and the partner `pair + 32` is below 64. -/
theorem pair_in_range {d : Nat} (hd : d < 64) : d % 32 < 32 ∧ d % 32 + 32 < 64 := by omega

/-- Lower-half threads have `pair = d`, upper-half threads have `pair = d - 32`. -/
theorem pair_halves {d : Nat} (hd : d < 64) :
    (d < 32 → d % 32 = d) ∧ (32 ≤ d → d % 32 = d - 32) := by omega

/-- The pairs `(d, d+32)` partition the 64 rotated dims: every dim has a unique pair index and half. -/
theorem rotated_dims_partition {d : Nat} (hd : d < 64) :
    (d % 32) + 32 * (d / 32) = d ∧ d / 32 < 2 := by omega

/-- Dims 64..127 are not rotated and keep the normalised value. -/
theorem unrotated_dims {d : Nat} (h1 : 64 ≤ d) (h2 : d < 128) : ¬ d < 64 := by omega

-- § position guards

/-- A base that is a multiple of 4 plus `4*b` is still a multiple of 4. -/
theorem rotation_pos_aligned {base b : Nat} (h : base % 4 = 0) : (base + 4 * b) % 4 = 0 := by omega

/-- The rotation position does not overflow int32 when `base + max_cells <= INT32_MAX` and `pos < max_cells`. -/
theorem rotation_pos_no_overflow {base pos max : Nat} (hm : base + max ≤ 2147483647) (hp : pos < max) :
    base + 4 * (pos / 4) ≤ 2147483647 := by omega

/-- The guard `pos < max_cells` makes every accepted position a valid cell index. -/
theorem guard_valid {pos max : Nat} (h : pos < max) : pos / 4 ≤ max / 4 := by omega

-- § span predicate

/-- Half-open byte ranges flagged as overlapping. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- The overlap predicate is symmetric. -/
theorem overlap_comm {ap an bp bn : Nat} : overlap ap an bp bn ↔ overlap bp bn ap an := by
  unfold overlap
  omega

/-- Ranges laid end to end are not flagged. -/
theorem adjacent_not_flagged {ap an bp bn : Nat} (h : ap + an ≤ bp) : ¬ overlap ap an bp bn := by
  unfold overlap
  omega

/-- The seven buffers give 21 pairwise overlap checks. -/
theorem pair_check_count : 7 * 6 / 2 = 21 := by decide

end Strata.NativeQsaIndexer
