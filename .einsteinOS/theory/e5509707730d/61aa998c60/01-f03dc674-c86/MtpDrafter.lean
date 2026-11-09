/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
Multi-token-prediction draft layer: bump-allocated device arena, the attention window capacity, step-row staging
indices, and the early-stopping draft chain.

## Source semantics
* `Bump::take` advances by the 256-rounded byte count, so every buffer starts aligned.
* `cap = ceil(window / 64) * 64`.
* Chain step `j` uses staged row `max_t + j - 1`, which lies between the catch-up rows and the final row `2*max_t - 1`.
* The chain stops when `j` reaches `max_t - 1` or the last draft probability is below `min_p`.

## Form template identifiers
Bump::take → bumpTake ; cap_ → capOf ; step row → chainRow ; draft loop → chainAux
-/

namespace Strata.Mtp

-- § bump allocator

/-- New `used` after taking `bytes` bytes. -/
def bumpTake (used bytes : Nat) : Nat := used + (bytes + 255) / 256 * 256

/-- The arena only grows. -/
theorem bumpTake_ge (used bytes : Nat) : used + bytes ≤ bumpTake used bytes := by
  unfold bumpTake; omega

/-- Alignment is preserved, so every carve starts on a 256-byte boundary. -/
theorem bumpTake_mod (used bytes : Nat) (h : used % 256 = 0) : bumpTake used bytes % 256 = 0 := by
  unfold bumpTake; omega

/-- Over-allocation per buffer is under 256 bytes. -/
theorem bumpTake_lt (used bytes : Nat) : bumpTake used bytes < used + bytes + 256 := by
  unfold bumpTake; omega

-- § window capacity

/-- Attention capacity: the window rounded up to a multiple of 64. -/
def capOf (w : Nat) : Nat := (w + 63) / 64 * 64

/-- The capacity covers the window. -/
theorem capOf_ge (w : Nat) : w ≤ capOf w := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (w : Nat) : capOf w % 64 = 0 := by
  unfold capOf; omega

-- § staged rows

/-- Staged step row for chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows sit after the `max_t` catch-up rows. -/
theorem chainRow_ge (maxT j : Nat) (hj : 1 ≤ j) : maxT ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows sit strictly before the final-row slot `2*max_t - 1`, so the two never collide. -/
theorem chainRow_lt (maxT j : Nat) (hj : j < maxT - 1) (hm : 2 ≤ maxT) : chainRow maxT j < 2 * maxT - 1 := by
  unfold chainRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j j' : Nat) (hj : 1 ≤ j) (hj' : 1 ≤ j') (h : chainRow maxT j = chainRow maxT j') :
    j = j' := by
  unfold chainRow at h; omega

-- § draft chain

/-- The draft loop's count of drafts: continue while `j < max_t - 1` and the last draft was likely enough. -/
def chainAux (maxT : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, j => j
  | f + 1, j => if j < maxT - 1 ∧ ok j = true then chainAux maxT ok f (j + 1) else j

/-- The chain never produces fewer drafts than it started with. -/
theorem chainAux_ge (maxT : Nat) (ok : Nat → Bool) (f j : Nat) : j ≤ chainAux maxT ok f j := by
  induction f generalizing j with
  | zero => simp [chainAux]
  | succ f ih =>
    simp only [chainAux]
    split
    · have := ih (j + 1)
      omega
    · exact Nat.le_refl _

/-- The chain never produces more than `max_t - 1` drafts, so the `drafts` array is not overrun. -/
theorem chainAux_le (maxT : Nat) (ok : Nat → Bool) (f j : Nat) (hj : j ≤ maxT - 1) :
    chainAux maxT ok f j ≤ maxT - 1 := by
  induction f generalizing j with
  | zero => simpa [chainAux] using hj
  | succ f ih =>
    simp only [chainAux]
    split
    · rename_i h
      exact ih (j + 1) (by omega)
    · exact hj

/-- A first draft always exists: starting at 1 with `max_t ≥ 2` the count is at least 1. -/
theorem chainAux_pos (maxT : Nat) (ok : Nat → Bool) (f : Nat) : 1 ≤ chainAux maxT ok f 1 :=
  chainAux_ge maxT ok f 1

/-- An unlikely draft stops the chain immediately. -/
theorem chainAux_stop (maxT : Nat) (ok : Nat → Bool) (f j : Nat) (h : ok j = false) :
    chainAux maxT ok (f + 1) j = j := by
  simp [chainAux, h]

end Strata.Mtp
