/-!
# MODULE: mtp_windowing
Staging-row layout, attention window capacity and the bounded draft chain of the multi-token-prediction drafter.

## Source semantics
* Step/pos staging has `2*max_t` rows: rows `[0, T)` are catch-up, row `2*max_t-1` is the full-layer row,
  and chain step `j` (1 ≤ j ≤ max_t-2) uses row `max_t + j - 1`. These never collide.
* The attention index capacity is the window (or the context if no window applies) rounded up to 64.
* The draft chain continues while the last draft is likely; it drafts between 1 and `max_t - 1` tokens.
* A draft-vocab file is a whole number of 4-byte ids.

## Form template identifiers
* `cap_ = ((w>0&&w<mc ? w : mc)+63)/64*64` → `capOf`
* `ra = 2*max_t_-1`, `row = max_t_+j-1` → `fullRow`, `chainRow`
* the `for (j...; pj >= min_p)` loop → `chainLen`
* `n_dvocab_ = raw.size()/4` → `nDvocab`
-/

namespace Strata.Mtp

-- § window capacity

/-- Effective window: the requested one if it is positive and below the context, else the whole context. -/
def baseWindow (w maxCells : Nat) : Nat := if 0 < w ∧ w < maxCells then w else maxCells

/-- Index capacity, rounded up to 64. -/
def capOf (w maxCells : Nat) : Nat := (baseWindow w maxCells + 63) / 64 * 64

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- The capacity covers the window. -/
theorem baseWindow_le_cap (w m : Nat) : baseWindow w m ≤ capOf w m := by
  unfold capOf; omega

/-- The window never exceeds the context. -/
theorem baseWindow_le (w m : Nat) : baseWindow w m ≤ m := by
  unfold baseWindow
  split
  · omega
  · exact Nat.le_refl _

-- § staging rows

/-- The full-layer row. -/
def fullRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows sit above the catch-up rows `[0, T)` since `T ≤ max_t`. -/
theorem chainRow_ge (maxT j T : Nat) (hT : T ≤ maxT) (hj : 1 ≤ j) : T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows sit strictly below the full-layer row for steps `j < max_t`. -/
theorem chainRow_lt_full (maxT j : Nat) (hj : j < maxT) (h1 : 1 ≤ j) :
    chainRow maxT j < fullRow maxT := by
  unfold chainRow fullRow; omega

/-- The full row is the last of the `2*max_t` rows. -/
theorem fullRow_lt (maxT : Nat) (h : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by
  unfold fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow maxT j = chainRow maxT k) :
    j = k := by
  unfold chainRow at h; omega

-- § the bounded draft chain

/-- Extra chain steps taken from step `j`, with `fuel` as the structural bound. -/
def go (mt : Nat) (likely : Nat → Bool) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, j => if j < mt - 1 ∧ likely j = true then 1 + go mt likely fuel (j + 1) else 0

/-- The chain never runs past step `max_t - 2`. -/
theorem go_le (mt : Nat) (likely : Nat → Bool) :
    ∀ fuel j, go mt likely fuel j ≤ mt - 1 - j
  | 0, j => by simp [go]
  | fuel + 1, j => by
    simp only [go]
    split
    · rename_i h
      have h1 : j < mt - 1 := h.1
      have := go_le mt likely fuel (j + 1)
      omega
    · exact Nat.zero_le _

/-- Number of drafts: the first one is unconditional, the rest follow the likelihood test. -/
def chainLen (mt : Nat) (likely : Nat → Bool) : Nat := 1 + go mt likely mt 1

/-- A round drafts at least one token. -/
theorem chainLen_pos (mt : Nat) (likely : Nat → Bool) : 1 ≤ chainLen mt likely := by
  unfold chainLen; omega

/-- A round drafts at most `max_t - 1` tokens, which is what the drafts array holds. -/
theorem chainLen_le (mt : Nat) (likely : Nat → Bool) (h : 2 ≤ mt) : chainLen mt likely ≤ mt - 1 := by
  unfold chainLen
  have := go_le mt likely mt 1
  omega

/-- With nothing likely the chain stops after the first draft. -/
theorem chainLen_unlikely (mt : Nat) : chainLen mt (fun _ => false) = 1 := by
  unfold chainLen
  cases mt with
  | zero => rfl
  | succ n => simp [go]

-- § draft vocabulary file

/-- Number of draft-vocab ids in a file of `size` bytes. -/
def nDvocab (size : Nat) : Nat := size / 4

/-- A size accepted by the loader (multiple of 4) is exactly `4 * n_dvocab` bytes. -/
theorem nDvocab_bytes (size : Nat) (h : size % 4 = 0) : nDvocab size * 4 = size := by
  unfold nDvocab; omega

end Strata.Mtp
