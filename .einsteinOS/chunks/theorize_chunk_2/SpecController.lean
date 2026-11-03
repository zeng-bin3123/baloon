/-!
# MODULE: spec_controller
Adaptive speculative-decoding controller: picks a draft source and length by expected tokens per millisecond,
and learns per-position acceptance rates from outcomes.

## Source semantics
- The verify pass always commits one token, so a window of k drafts has n = clamp(k+1, 1, K_MAX+1) tokens.
- `observe` clamps accepted into [0, k]; positions 0..accepted-1 were hits and position `accepted` a miss.
- The number of positions updated is min(k, accepted+1).
- A speculative choice is kept only if it beats baseline by the minimum gain.

## Form template identifiers
std::clamp -> clampNat ; seen -> seenCount ; K_MAX -> kMax
-/

namespace Strata.Spec

-- § clamping

/-- Clamp into [lo, hi]. -/
def clampNat (lo hi x : Nat) : Nat := max lo (min hi x)

/-- Clamp result is at least lo. -/
theorem clampNat_ge (lo hi x : Nat) : lo ≤ clampNat lo hi x := by unfold clampNat; omega

/-- Clamp result is at most hi when lo ≤ hi. -/
theorem clampNat_le (lo hi x : Nat) (h : lo ≤ hi) : clampNat lo hi x ≤ hi := by
  unfold clampNat; omega

/-- Window size index: n lies in 1..kMax+1, so `dense_ratio[n-1]` is in range. -/
theorem window_index_bounds (kMax k : Nat) :
    1 ≤ clampNat 1 (kMax + 1) (k + 1) ∧ clampNat 1 (kMax + 1) (k + 1) ≤ kMax + 1 := by
  unfold clampNat; omega

-- § observation bookkeeping

/-- Accepted count after clamping into [0, k]. -/
def acceptedClamped (k accepted : Nat) : Nat := min accepted k

/-- Positions whose rate is updated. -/
def seenCount (k accepted : Nat) : Nat := min k (accepted + 1)

/-- Clamped accepted never exceeds the draft length. -/
theorem acceptedClamped_le (k a : Nat) : acceptedClamped k a ≤ k := by
  unfold acceptedClamped; omega

/-- Updated positions never exceed the draft length. -/
theorem seenCount_le (k a : Nat) : seenCount k a ≤ k := by unfold seenCount; omega

/-- Every accepted position is among the updated ones. -/
theorem seenCount_ge_accepted (k a : Nat) : acceptedClamped k a ≤ seenCount k (acceptedClamped k a) := by
  unfold seenCount acceptedClamped; omega

/-- A rejection at position `a < k` is observed as the last updated position. -/
theorem seenCount_reject (k a : Nat) (h : a < k) : seenCount k a = a + 1 := by
  unfold seenCount; omega

/-- A full acceptance updates exactly k positions (none beyond the window). -/
theorem seenCount_full (k : Nat) : seenCount k k = k := by unfold seenCount; omega

-- § expected tokens

/-- Expected committed tokens in fixed point: one verified token plus the running acceptance products. -/
def expectedCount : List Nat → Nat
  | [] => 1
  | r :: rs => expectedCount rs + r

/-- Expected count is at least one: the verify pass always commits. -/
theorem expectedCount_ge_one (rs : List Nat) : 1 ≤ expectedCount rs := by
  induction rs with
  | nil => simp [expectedCount]
  | cons r rs ih => simp [expectedCount]; omega

/-- Adding a draft position never lowers the expectation. -/
theorem expectedCount_mono (r : Nat) (rs : List Nat) : expectedCount rs ≤ expectedCount (r :: rs) := by
  simp [expectedCount]

-- § keep-or-baseline rule

/-- Keep the speculative rate only when it exceeds baseline*(100+gainPct)/100 (percent fixed point). -/
def keepSpec (rate baseline gainPct : Nat) : Bool :=
  decide (rate * 100 ≥ baseline * (100 + gainPct))

/-- With zero gain threshold, a rate no lower than baseline is kept. -/
theorem keepSpec_zero_gain (rate baseline : Nat) (h : baseline ≤ rate) : keepSpec rate baseline 0 = true := by
  unfold keepSpec
  simp
  omega

end Strata.Spec
