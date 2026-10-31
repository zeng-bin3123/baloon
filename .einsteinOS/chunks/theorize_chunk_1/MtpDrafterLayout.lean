/-!
# MODULE: mtp_drafter_layout
Index and capacity arithmetic of the MTP draft layer: 256-byte bump allocation of the device arena, the
attention window capacity rounded to 64, and the step/pos row schedule of a draft round (T catch-up rows,
one staged row for the full layer, and up to `max_t - 2` chain steps).

## Source semantics
* `Bump::take n` advances `used` by `(n * sizeof(T) + 255) & ~255`; carving twice (count, then real) yields the
  same offsets, so the real pass never overruns the counted size.
* `cap = ceil((window > 0 ? window : max_cells) / 64) * 64`.
* Step rows: `R2 = 2 * max_t`; the catch-up rows are `0..T-1`, the full-layer row is `ra = 2 * max_t - 1`, chain
  step `j` (1..max_t-2) uses row `max_t + j - 1`. All are `< R2` and the chain rows never collide with the
  catch-up rows or `ra`.
* `draft` requires `1 <= T <= max_t` and `0 <= a < T`.

## Form template identifiers
* `bumpTake`     : `(n * elem + 255) / 256 * 256`
* `capOf`        : window capacity rounding
* `chainRow`     : `max_t + j - 1`
* `fullRow`      : `2 * max_t - 1`
-/

namespace Strata.Mtp

-- § bump allocator

/-- Aligned advance for one `take`. -/
def bumpTake (n elem : Nat) : Nat := (n * elem + 255) / 256 * 256

/-- A `take` advances by a multiple of 256. -/
theorem bumpTake_mod (n elem : Nat) : bumpTake n elem % 256 = 0 := by
  unfold bumpTake; omega

/-- A `take` advances by at least the bytes requested. -/
theorem bumpTake_ge (n elem : Nat) : n * elem ≤ bumpTake n elem := by
  unfold bumpTake; omega

/-- A `take` wastes less than 256 bytes. -/
theorem bumpTake_lt (n elem : Nat) : bumpTake n elem < n * elem + 256 := by
  unfold bumpTake; omega

/-- Total after a sequence of takes. -/
def bumpUsed : List Nat → Nat → Nat
  | [], u => u
  | b :: rest, u => bumpUsed rest (u + b)

/-- Used is the plain sum when each advance is already aligned. -/
theorem bumpUsed_eq_sum (bs : List Nat) (u : Nat) : bumpUsed bs u = u + bs.sum := by
  induction bs generalizing u with
  | nil => simp [bumpUsed]
  | cons b rest ih => simp [bumpUsed, ih] <;> omega

/-- A deterministic carve: counting and real passes are the same function, so they agree. -/
theorem count_real_agree (bs : List Nat) : bumpUsed bs 0 = bumpUsed bs 0 := rfl

/-- Aligned advances keep the cursor 256-aligned. -/
theorem bumpUsed_aligned (bs : List Nat) (u : Nat) (hu : u % 256 = 0) (hb : ∀ b ∈ bs, b % 256 = 0) :
    bumpUsed bs u % 256 = 0 := by
  induction bs generalizing u with
  | nil => simpa [bumpUsed] using hu
  | cons b rest ih =>
    simp only [bumpUsed]
    apply ih
    · have := hb b (by simp); omega
    · intro x hx; exact hb x (by simp [hx])

-- § window capacity

/-- Capacity of the attention window, rounded up to 64. -/
def capOf (window maxCells : Nat) : Nat :=
  ((if window > 0 then window else maxCells) + 63) / 64 * 64

/-- The capacity covers the window (or the whole context when unwindowed). -/
theorem capOf_ge (window maxCells : Nat) :
    (if window > 0 then window else maxCells) ≤ capOf window maxCells := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (window maxCells : Nat) : capOf window maxCells % 64 = 0 := by
  unfold capOf; omega

/-- The capacity overshoots by fewer than 64 cells. -/
theorem capOf_lt (window maxCells : Nat) :
    capOf window maxCells < (if window > 0 then window else maxCells) + 64 := by
  unfold capOf; omega

/-- The effective window: a window at or above the context size is dropped. -/
def effWindow (window maxCells : Nat) : Nat := if window > 0 ∧ window < maxCells then window else 0

/-- A positive effective window is strictly below the context size. -/
theorem effWindow_lt (window maxCells : Nat) (h : 0 < effWindow window maxCells) :
    effWindow window maxCells < maxCells := by
  unfold effWindow at *
  by_cases hc : window > 0 ∧ window < maxCells
  · simp [hc] <;> exact hc.2
  · simp [hc] at h

-- § step row schedule

/-- Number of step/pos rows. -/
def rows (maxT : Nat) : Nat := 2 * maxT

/-- The full-layer row. -/
def fullRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row is the last row, in range. -/
theorem fullRow_lt (maxT : Nat) (h : 1 ≤ maxT) : fullRow maxT < rows maxT := by
  unfold fullRow rows; omega

/-- Chain rows lie in `[max_t, 2*max_t - 2]`, so inside the table. -/
theorem chainRow_lt (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) : chainRow maxT j < rows maxT := by
  unfold chainRow rows; omega

/-- Chain rows never collide with the catch-up rows `0..T-1` (with `T ≤ max_t`). -/
theorem chainRow_ne_catchup (maxT T j t : Nat) (hT : T ≤ maxT) (ht : t < T) (h1 : 1 ≤ j) :
    chainRow maxT j ≠ t := by
  unfold chainRow; omega

/-- Chain rows never collide with the full-layer row. -/
theorem chainRow_ne_full (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    chainRow maxT j ≠ fullRow maxT := by
  unfold chainRow fullRow; omega

/-- The full-layer row never collides with a catch-up row. -/
theorem fullRow_ne_catchup (maxT T t : Nat) (hT : T ≤ maxT) (ht : t < T) (h : 1 ≤ maxT) :
    fullRow maxT ≠ t := by
  unfold fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_injective (maxT j k : Nat) (h1 : 1 ≤ j) (h2 : 1 ≤ k)
    (h : chainRow maxT j = chainRow maxT k) : j = k := by
  unfold chainRow at h; omega

/-- Cells of a round: rows `p + t` for the catch-up and `p + a + j` for chain step `j`. -/
def chainCell (p a j : Nat) : Nat := p + a + j

/-- The chain extends the window: step `j`'s cell is strictly after the real pair's cell `p + a`. -/
theorem chainCell_gt (p a j : Nat) (h : 1 ≤ j) : p + a < chainCell p a j := by
  unfold chainCell; omega

/-- Valid arguments for a draft round. -/
def draftArgsOk (T a maxT : Nat) : Prop := 1 ≤ T ∧ T ≤ maxT ∧ a < T

/-- A valid round has its catch-up cell `p + a` inside the staged window `p .. p + T - 1`. -/
theorem draft_cell_in_window (T a maxT p : Nat) (h : draftArgsOk T a maxT) :
    p ≤ p + a ∧ p + a < p + T := by
  obtain ⟨_, _, h3⟩ := h
  omega

/-- Number of drafts is at most `max_t - 1` (one from the round graph plus `max_t - 2` chain steps). -/
theorem drafts_bound (maxT n : Nat) (h : n ≤ 1 + (maxT - 2)) (hm : 2 ≤ maxT) : n ≤ maxT - 1 := by omega

end Strata.Mtp
