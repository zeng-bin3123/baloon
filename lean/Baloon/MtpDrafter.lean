-- Baloon.MtpDrafter: multi-token prediction drafter window/capacity, step-row
-- staging and draft-chain termination invariants.
-- Independent implementation proving the same mathematical facts as Ref.MtpDrafter.

namespace Baloon.MtpDrafter

def effWindow (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else 0

def attendLen (window maxCells : Nat) : Nat :=
  if effWindow window maxCells > 0 then effWindow window maxCells else maxCells

def capOf (window maxCells : Nat) : Nat := (attendLen window maxCells + 63) / 64 * 64

theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow; split <;> omega

theorem attendLen_le (w m : Nat) : attendLen w m ≤ m := by
  unfold attendLen; have := effWindow_le w m; split <;> omega

theorem cap_covers (w m : Nat) : attendLen w m ≤ capOf w m := by unfold capOf; omega
theorem cap_mod (w m : Nat) : capOf w m % 64 = 0 := by unfold capOf; omega
theorem cap_lt (w m : Nat) : capOf w m < attendLen w m + 64 := by unfold capOf; omega

def chainRow (T j : Nat) : Nat := T + j - 1
def fullRow (T : Nat) : Nat := 2 * T - 1

theorem chainRow_above_catchup (T j : Nat) (hj : 1 ≤ j) : T ≤ chainRow T j := by
  unfold chainRow; omega

theorem chainRow_below_full (T j : Nat) (hj : j ≤ T - 2) (hT : 3 ≤ T) :
    chainRow T j < fullRow T ∧ fullRow T < 2 * T := by
  unfold chainRow fullRow; omega

theorem chainRow_inj (T j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow T j = chainRow T k) :
    j = k := by unfold chainRow at h; omega

theorem step_block_mono (a b : Nat) (h : a ≤ b) : (a + 1) / 4 ≤ (b + 1) / 4 := by omega

def chainFuel (maxT minP : Nat) (p : Nat → Nat) : Nat → Nat → Nat
  | 0, j => j
  | fuel + 1, j =>
    if j < maxT - 1 ∧ p (j - 1) ≥ minP then chainFuel maxT minP p fuel (j + 1) else j

theorem chainFuel_bounded (maxT minP : Nat) (p : Nat → Nat) (fuel j : Nat) (h : j < maxT) :
    chainFuel maxT minP p fuel j < maxT := by
  induction fuel generalizing j with
  | zero => simp [chainFuel]; exact h
  | succ n ih =>
    simp only [chainFuel]
    split_ifs with hc
    · exact ih (j + 1) (by omega)
    · exact h

end Baloon.MtpDrafter
