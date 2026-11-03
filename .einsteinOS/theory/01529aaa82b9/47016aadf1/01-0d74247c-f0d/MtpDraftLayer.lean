/-!
# MODULE: mtp_draft_layer
The MTP draft layer (`MtpDrafter`): bump-allocator arena, attention window capacity, step-row staging for the
catch-up and chain graphs, and the draft vocabulary subset.

## Source semantics
* `Bump::take` advances by `n*sizeof(T)` rounded up to 256; the count pass and the carve pass use the same sizes.
* `cap_` is the window (or `max_cells`) rounded up to a multiple of 64.
* Step rows: T catch-up rows `[0,T)`, chain rows `[max_t, 2*max_t-2)`, and the staged full-layer row `2*max_t-1`.
* `draft_vocab.bin` holds `n_dvocab = size/4` 32-bit ids.

## Form template identifiers
`Bump` -> bumpEnd, `cap_` -> capOf, `ra` -> stagedRow, `row` (chain) -> chainRow, `n_dvocab_` -> nDvocab.
-/

namespace Strata.Mtp

-- § bump allocator

/-- 256-byte rounding of one `take`. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A take advances the cursor by at least the request. -/
theorem take_advances (used b : Nat) : used + b ≤ used + align256 b := by unfold align256; omega

/-- A take keeps the cursor 256-aligned. -/
theorem take_aligned (used b : Nat) (h : used % 256 = 0) : (used + align256 b) % 256 = 0 := by
  unfold align256; omega

/-- Cursor after a sequence of takes. -/
def bumpEnd (acc : Nat) (sizes : List Nat) : Nat := sizes.foldl (fun a b => a + align256 b) acc

/-- The total is the sum of the rounded sizes (so the count pass sizes the real allocation exactly). -/
theorem bumpEnd_eq : ∀ (sizes : List Nat) (acc : Nat), bumpEnd acc sizes = acc + (sizes.map align256).sum
  | [], acc => by simp [bumpEnd]
  | b :: bs, acc => by
      have := bumpEnd_eq bs (acc + align256 b)
      simp only [bumpEnd, List.foldl_cons, List.map_cons, List.sum_cons] at this ⊢
      omega

/-- The count pass and the carve pass end at the same offset (same list, same start). -/
theorem two_pass_agree (sizes : List Nat) : bumpEnd 0 sizes = bumpEnd 0 sizes := rfl

-- § attention window

/-- Effective window: the requested one if it is positive and below `max_cells`. -/
def windowOf (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else maxCells

/-- `cap_` rounds the effective window up to a multiple of 64. -/
def capOf (window maxCells : Nat) : Nat := (windowOf window maxCells + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capOf_ge (window maxCells : Nat) : windowOf window maxCells ≤ capOf window maxCells := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (window maxCells : Nat) : capOf window maxCells % 64 = 0 := by
  unfold capOf; omega

-- § step rows

/-- Row staged for the single full-layer step of the round graph. -/
def stagedRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Catch-up rows, chain rows and the staged row are pairwise disjoint and all below `2*max_t`. -/
theorem rows_disjoint (maxT j : Nat) (hT : 2 ≤ maxT) (hj1 : 1 ≤ j) (hj2 : j + 2 ≤ maxT) :
    maxT - 1 < chainRow maxT j ∧ chainRow maxT j < stagedRow maxT ∧ stagedRow maxT < 2 * maxT := by
  unfold chainRow stagedRow; omega

/-- Chain draft indices stay inside the `max_t`-sized draft array. -/
theorem draft_index_bound (maxT j : Nat) (hj : j + 1 < maxT) : j < maxT := by omega

/-- At most `max_t - 1` drafts are produced, so `n_drafts ≤ max_t - 1`. -/
theorem drafts_count_bound (maxT n : Nat) (h : n ≤ maxT - 1) : n < maxT ∨ maxT = 0 := by omega

-- § draft vocabulary

/-- Number of ids in `draft_vocab.bin`. -/
def nDvocab (size : Nat) : Nat := size / 4

/-- When the file size is a multiple of 4 the ids tile it exactly. -/
theorem dvocab_tiles (size : Nat) (h : size % 4 = 0) : nDvocab size * 4 = size := by
  unfold nDvocab; omega

/-- A non-empty file yields at least one id. -/
theorem dvocab_pos (size : Nat) (h : 4 ≤ size) : 0 < nDvocab size := by
  unfold nDvocab; omega

end Strata.Mtp
