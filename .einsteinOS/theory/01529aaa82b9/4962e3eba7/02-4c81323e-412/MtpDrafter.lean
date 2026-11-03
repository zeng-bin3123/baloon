/-!
# MODULE: mtp_drafter
Arithmetic contracts of `src/core/mtp.cpp`: the MTP draft layer's bump arena, attention window capacity,
staging-row layout of the captured round/step graphs, prefill chunking and the draft chain length.

## Source semantics
* `Bump::take(n)` advances by `(n*sizeof + 255) & ~255`; a counting pass and the real pass see identical sizes so
  offsets agree and every region is 256-aligned and in bounds of the counted total.
* `cap_ = ceil(window_eff / 64) * 64`, `window_eff = window` if `0 < window < max_cells` else `max_cells`.
* Step/pos rows: rows `[0,T)` catch-up, row `2*max_t-1` the full-layer cell, rows `max_t + j - 1` (j = 1..max_t-2)
  the chain steps; all rows lie below `2*max_t`.
* Draft chain yields `1 <= n <= max_t - 1` drafts.

## Form template identifiers
* `align256` <- Bump's `(x + 255) & ~255`
* `bumpTotal` <- `Bump::used` after `carve`
* `effWindow`, `capOf` <- `window_`, `cap_`
* `chainRow` <- `max_t_ + j - 1`
-/

namespace Strata.MtpDrafter

-- § bump arena

/-- 256-byte round-up. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega

/-- Rounded value is 256-aligned. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Rounding adds at most 255 bytes. -/
theorem align256_le (n : Nat) : align256 n ≤ n + 255 := by unfold align256; omega

/-- Bytes consumed by a sequence of takes (byte sizes already multiplied by `sizeof`). -/
def bumpTotal (l : List Nat) : Nat := (l.map align256).sum

/-- Counting two phases separately and concatenating is the same as one pass. -/
theorem bump_append (l1 l2 : List Nat) : bumpTotal (l1 ++ l2) = bumpTotal l1 + bumpTotal l2 := by
  simp [bumpTotal, List.map_append, List.sum_append]

/-- The region of the `d`-th take, starting at the prefix total, lies inside the whole arena. -/
theorem bump_region_in_bounds (l1 l2 : List Nat) (d : Nat) :
    bumpTotal l1 + d ≤ bumpTotal (l1 ++ d :: l2) := by
  have h := align256_ge d
  simp only [bumpTotal, List.map_append, List.map_cons, List.sum_append, List.sum_cons]
  omega

/-- The whole arena is a multiple of 256. -/
theorem bump_total_mod (l : List Nat) : bumpTotal l % 256 = 0 := by
  induction l with
  | nil => simp [bumpTotal]
  | cons a t ih =>
    have h := align256_mod a
    simp only [bumpTotal, List.map_cons, List.sum_cons] at ih ⊢
    omega

/-- Every region start (a prefix total) is 256-aligned. -/
theorem bump_region_aligned (l1 l2 : List Nat) : bumpTotal l1 % 256 = 0 := bump_total_mod l1

-- § attention window capacity

/-- Effective attended cells. -/
def effWindow (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else maxCells

/-- Capacity `cap_`: effective window rounded up to 64. -/
def capOf (window maxCells : Nat) : Nat := (effWindow window maxCells + 63) / 64 * 64

/-- Effective window never exceeds the cache. -/
theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow; split <;> omega

/-- Capacity covers the effective window. -/
theorem capOf_ge (w m : Nat) : effWindow w m ≤ capOf w m := by
  unfold capOf; omega

/-- Capacity is a multiple of 64. -/
theorem capOf_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- Capacity wastes less than 64 cells. -/
theorem capOf_lt (w m : Nat) : capOf w m < effWindow w m + 64 := by
  unfold capOf; omega

/-- Identity table: entry `t*cap + i` (t < T, i < cap) is inside the `T*cap` table. -/
theorem ident_index_lt (t i T cap : Nat) (ht : t < T) (hi : i < cap) : t * cap + i < T * cap := by
  have h1 := Nat.mul_le_mul_right cap (show t + 1 ≤ T by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § staging rows

/-- Row of chain step `j` (1-based). -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows are after every catch-up row (`T <= max_t`). -/
theorem chain_row_after_catchup (maxT T j : Nat) (hT : T ≤ maxT) (hj : 1 ≤ j) :
    T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows are before the full-layer row `2*max_t-1`. -/
theorem chain_row_before_ra (maxT j : Nat) (hj : j + 2 ≤ maxT) :
    chainRow maxT j < 2 * maxT - 1 := by
  unfold chainRow; omega

/-- The full-layer row lies inside the `2*max_t` rows allocated for step/pos. -/
theorem ra_row_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT := by
  omega

/-- Distinct chain steps use distinct rows. -/
theorem chain_row_injective (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k)
    (h : chainRow maxT j = chainRow maxT k) : j = k := by
  unfold chainRow at h; omega

-- § chunk loops

/-- Eight-column quantize/GEMV chunks over `T*HC` columns: each chunk is non-empty and in range. -/
theorem mmvq_chunk_in_range (total c0 : Nat) (h : c0 < total) :
    1 ≤ min 8 (total - c0) ∧ min 8 (total - c0) ≤ 8 ∧ c0 + min 8 (total - c0) ≤ total := by
  omega

/-- Prefill chunking: a chunk starting before `n` is non-empty, at most `max_t`, and stays inside `n`. -/
theorem prefill_chunk_bounds (n c maxT : Nat) (h : c < n) (hm : 1 ≤ maxT) :
    1 ≤ min maxT (n - c) ∧ min maxT (n - c) ≤ maxT ∧ c + min maxT (n - c) ≤ n := by
  omega

/-- Draft chain: `n = 1 + k` with at most `max_t - 2` extra steps yields `1 <= n <= max_t - 1`. -/
theorem draft_count_bounds (n k maxT : Nat) (hn : n = 1 + k) (hk : k + 2 ≤ maxT) :
    1 ≤ n ∧ n ≤ maxT - 1 := by
  omega

/-- Draft arguments accepted by `draft`: the catch-up row `a` is an existing window row. -/
theorem draft_arg_row (T maxT a : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) (h3 : a < T) : a < maxT := by
  omega

end Strata.MtpDrafter
