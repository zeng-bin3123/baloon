/-!
# MODULE: mtp_draft_layer
Multi-token-prediction draft layer: a single extra decoder layer with its own K/V cache that drafts up to
max_t-2 tokens by chaining one-row steps after a catch-up round over the verified window.

## Source semantics
- The attention capacity is the window (or context) rounded up to a multiple of 64.
- Step/pos staging has 2*max_t rows: rows [0, T) catch-up, rows [max_t, 2*max_t-3] chain steps,
  row 2*max_t-1 the catch-up row for the accepted position.
- Chain step j (1 <= j <= max_t-2) uses row max_t + j - 1.
- Each cell's step record is (cell, cell+1, (cell+1)/4, cell+1).

## Form template identifiers
cap_ -> capRound ; ra -> accRow ; row = max_t_ + j - 1 -> chainRow ; put(row, cell) -> stepRec
-/

namespace Strata.Mtp

-- § capacity

/-- Capacity rounded up to a multiple of 64. -/
def capRound (n : Nat) : Nat := ((n + 63) / 64) * 64

/-- Capacity covers the request. -/
theorem capRound_ge (n : Nat) : n ≤ capRound n := by unfold capRound; omega

/-- Capacity is a multiple of 64. -/
theorem capRound_mod (n : Nat) : capRound n % 64 = 0 := by unfold capRound; omega

/-- Capacity wastes less than one unit of rounding. -/
theorem capRound_lt (n : Nat) : capRound n < n + 64 := by unfold capRound; omega

-- § row layout

/-- Row of the accepted-position catch-up step. -/
def accRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows lie above every catch-up row. -/
theorem chainRow_above_catchup (maxT T j : Nat) (hT : T ≤ maxT) (hj : 1 ≤ j) :
    T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows are below the accepted-position row. -/
theorem chainRow_below_acc (maxT j : Nat) (hj : j + 2 ≤ maxT) :
    chainRow maxT j < accRow maxT := by
  unfold chainRow accRow; omega

/-- All rows fit in the 2*max_t staging allocation. -/
theorem accRow_lt_staging (maxT : Nat) (h : 1 ≤ maxT) : accRow maxT < 2 * maxT := by
  unfold accRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow maxT j = chainRow maxT k) :
    j = k := by
  unfold chainRow at h; omega

-- § step record

/-- The four-int step record for a cell. -/
def stepRec (cell : Nat) : Nat × Nat × Nat × Nat := (cell, cell + 1, (cell + 1) / 4, cell + 1)

/-- The block count never exceeds the cell count. -/
theorem stepRec_blocks_le (cell : Nat) : (stepRec cell).2.2.1 ≤ (stepRec cell).2.1 := by
  simp [stepRec]; omega

/-- The number of keys is one more than the cell index. -/
theorem stepRec_nkv (cell : Nat) : (stepRec cell).2.1 = cell + 1 := rfl

-- § draft chain bookkeeping

/-- Drafts produced: the first is always produced, later ones only while confident. -/
def draftCount (maxT stop : Nat) : Nat := min (maxT - 1) (max 1 stop)

/-- At least one and at most max_t - 1 drafts. -/
theorem draftCount_bounds (maxT stop : Nat) (h : 2 ≤ maxT) :
    1 ≤ draftCount maxT stop ∧ draftCount maxT stop ≤ maxT - 1 := by
  unfold draftCount; omega

end Strata.Mtp
