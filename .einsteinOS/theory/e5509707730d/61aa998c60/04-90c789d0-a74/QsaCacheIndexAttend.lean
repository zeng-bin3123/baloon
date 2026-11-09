/-!
# MODULE: qsa_cache_index_attend
SOURCES: src/kernels/cuda/qsa.cu
The QSA KV cache append, indexer-key pooling, block scoring, exact top-k threshold selection, KV gather,
and per-head attention. This module models the index arithmetic, counting and bound contracts of those kernels.

## Source semantics
* `qsa_step_fill`: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1); all derive from one position.
* A cell `pos` has slot `pos % r` and block `pos / r`; the cell with slot `r-1` completes a block, and then the
  pooled row b and the spare row b+1 are written (so the pooled buffer needs b+1 in range).
* `topk_512` emits every cell strictly above the width-th key plus the first `width - c_gt` cells equal to it.
* `kv_gather` decomposes a flat index into (selected id, kv head, uint2 lane); `qsa_attend` maps head h to kv head
  `h / (n_head / n_head_kv)`.

## Form template identifiers
- `selWidth`      : selection width = min(n_kv, top_k + r - 1)
- `r`             : idx_block (cells per pooled block)
- `n_bid`         : completed-block count n_kv / r
- `gt`, `eq`, `budget` : strictly-above count, equal-to-threshold count, `width - c_gt`
-/

namespace Strata.QsaCache

-- § step state

/-- Selection width as computed by `qsa_selection_width`: min(n_kv, top_k + r - 1). -/
def selWidth (nkv topk r : Nat) : Nat := min nkv (topk + r - 1)

/-- The selection never exceeds the number of cached cells. -/
theorem selWidth_le (nkv topk r : Nat) : selWidth nkv topk r ≤ nkv := by
  unfold selWidth; omega

/-- Below the bound the selection is the identity (width equals n_kv). -/
theorem selWidth_identity (nkv topk r : Nat) (h : nkv ≤ topk + r - 1) :
    selWidth nkv topk r = nkv := by
  unfold selWidth; omega

/-- For the 2048/4 geometry the width never exceeds 2051. -/
theorem selWidth_cap (nkv : Nat) : selWidth nkv 2048 4 ≤ 2051 := by
  unfold selWidth; omega

-- § slot / block arithmetic

/-- A position decomposes as block * r + slot. -/
theorem slot_block_decomp (pos r : Nat) : pos / r * r + pos % r = pos := by
  have h := Nat.div_add_mod pos r
  rw [Nat.mul_comm] at h
  exact h

/-- The cell that completes a block closes exactly (b+1) * r cells. -/
theorem complete_block_size (pos r : Nat) (hs : pos % r + 1 = r) :
    (pos / r + 1) * r = pos + 1 := by
  have h1 := slot_block_decomp pos r
  have h2 : (pos / r + 1) * r = pos / r * r + r := by
    rw [Nat.add_mul, Nat.one_mul]
  omega

/-- When a block completes inside capacity, both the pooled row b and the spare row b+1 exist
in a buffer of `max_cells / r + 1` rows. -/
theorem spare_row_in_bounds (pos r maxc : Nat) (hr : 0 < r) (hs : pos % r + 1 = r)
    (hp : pos < maxc) : pos / r + 1 ≤ maxc / r := by
  have h := complete_block_size pos r hs
  exact (Nat.le_div_iff_mul_le hr).2 (by omega)

/-- A block before the incomplete tail lies entirely inside the cache. -/
theorem full_block_inside (b nkv r : Nat) (hb : b < nkv / r) : (b + 1) * r ≤ nkv := by
  have h1 : (b + 1) * r ≤ nkv / r * r := Nat.mul_le_mul_right r hb
  have h2 : nkv / r * r ≤ nkv := Nat.div_mul_le_self nkv r
  omega

/-- Every cached cell belongs to a block index at most n_bid. -/
theorem cell_block_le (j nkv r : Nat) (hj : j < nkv) : j / r ≤ nkv / r :=
  Nat.div_le_div_right (Nat.le_of_lt hj)

/-- The rotation position of a completed block keeps the alignment of `pos_base`. -/
theorem block_pos_formula (pos_base b r : Nat) : pos_base + b * r = pos_base + r * b := by
  rw [Nat.mul_comm b r]

-- § pool indexing

/-- Mixed-radix index bound: a < n and b < m give a * m + b < n * m. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The `[page][kv_head][page_size][head_dim]` pool row of `kv_append`/`kv_gather` stays in the pool. -/
theorem pool_row_lt (page h slot np kvh ps : Nat) (hp : page < np) (hh : h < kvh) (hs : slot < ps) :
    (page * kvh + h) * ps + slot < np * kvh * ps :=
  idx_lt (page * kvh + h) slot (np * kvh) ps (idx_lt page h np kvh hp hh) hs

/-- Gather index decomposition: the selected-id component is recovered by division. -/
theorem gather_id (id rem m : Nat) (hm : 0 < m) (hr : rem < m) : (m * id + rem) / m = id := by
  rw [Nat.mul_add_div hm, Nat.div_eq_of_lt hr, Nat.add_zero]

/-- Gather index decomposition: the (kv head, lane) remainder is recovered by modulus. -/
theorem gather_rem (id rem m : Nat) (hr : rem < m) : (m * id + rem) % m = rem := by
  rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hr]

/-- The kv head chosen for query head `h` is in range (`iv2 = iq2 / rv2`). -/
theorem gqa_kv_in_range (h g nkv : Nat) (hh : h < nkv * g) : h / g < nkv := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact hh

-- § exact threshold selection

/-- Counting contract: gt strictly-above cells plus min(eq, width - gt) equal cells select exactly `width`
cells whenever the threshold bracket gt <= width <= gt + eq holds. -/
theorem selected_count (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + min eq (width - gt) = width := by
  omega

/-- With distinct scores (c_gt = width, budget 0), nothing at the threshold is taken. -/
theorem selected_count_distinct (eq width : Nat) : width + min eq (width - width) = width := by
  omega

/-- The prefix of equal cells before a chunk never lets a chunk overshoot the budget. -/
theorem eq_prefix_within_budget (eqBefore eqHere budget : Nat) :
    eqBefore + min eqHere (budget - eqBefore) ≤ max budget eqBefore := by
  omega

/-- A chunk entirely under the budget selects all its equal cells. -/
theorem eq_chunk_fully_taken (eqBefore eqHere budget : Nat) (h : eqBefore + eqHere ≤ budget) :
    min eqHere (budget - eqBefore) = eqHere := by
  omega

-- § attention launch contract

/-- Shared memory sized from the capacity always covers this token's real selection. -/
theorem smem_covers (n_ids max_ids : Nat) (h : n_ids ≤ max_ids) :
    (n_ids + 32) * 4 ≤ (max_ids + 32) * 4 := by
  omega

end Strata.QsaCache
