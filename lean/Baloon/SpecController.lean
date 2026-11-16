-- Baloon.SpecController: invariants of the speculative-decoding controller.
-- Independent implementation in the project namespace; proves the same
-- mathematical facts as the reference theory with different names and terms.

namespace Baloon.SpecController

/-- K_MAX = 8: maximum number of draft tokens per step. -/
def K_MAX : Nat := 8

-- ── table index = min(k, K_MAX) ──────────────────────────────────────────────

/-- Table index is at most K_MAX for every draft length k. -/
theorem stepIdx_le (k : Nat) : min k 8 ≤ 8 := Nat.min_le_right k 8

/-- Table index is strictly less than the table size (K_MAX + 1). -/
theorem stepIdx_lt (k : Nat) : min k 8 < 9 := by omega

/-- A longer draft never maps to a lower table slot. -/
theorem stepIdx_mono (a b : Nat) (hab : a ≤ b) : min a 8 ≤ min b 8 := by omega

-- ── match-length bucket (four tiers) ─────────────────────────────────────────

/-- Bucket value is a valid 4-entry index (0–3). -/
theorem bucketBound (m : Nat) :
    (if 16 ≤ m then (3 : Nat) else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0) < 4 := by
  split_ifs <;> omega

/-- Longer match length → same or higher bucket (monotone). -/
theorem bucketMono (m n : Nat) (hmn : m ≤ n) :
    (if 16 ≤ m then (3 : Nat) else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0) ≤
    (if 16 ≤ n then (3 : Nat) else if 8 ≤ n then 2 else if 5 ≤ n then 1 else 0) := by
  split_ifs <;> omega

-- ── expected committed tokens (foldl running product) ────────────────────────

/-- Running acceptance product: start=S, fold p_i → acc * p_i / S. -/
def runFold (S : Nat) (ps : List Nat) : Nat :=
  ps.foldl (fun r p => r * p / S) S

/-- Expected committed tokens: S plus sum of prefix running products via scanl. -/
def commitSum (S : Nat) (ps : List Nat) : Nat :=
  S + (ps.scanl (fun r p => r * p / S) S).tail.sum

/-- Expected committed ≥ S (the bonus token is always there). -/
theorem commitSum_ge_S (S : Nat) (ps : List Nat) : S ≤ commitSum S ps := by
  unfold commitSum; exact Nat.le_add_right S _

-- ── min-gain gate ────────────────────────────────────────────────────────────

/-- Gate: if base * (1000 + gain) ≤ best * 1000, then base ≤ best. -/
theorem gatePass (best base gain : Nat) (h : base * (1000 + gain) ≤ best * 1000) :
    base ≤ best := by nlinarith

-- ── EMA update (Controller::observe) ─────────────────────────────────────────

/-- EMA hit update p + (S − p) * a / A stays in [0, S]. -/
theorem hitStep_le (S p a A : Nat) (hp : p ≤ S) (ha : a ≤ A) :
    p + (S - p) * a / A ≤ S := by
  have h1 : (S - p) * a ≤ A * (S - p) := by nlinarith
  have h2 : (S - p) * a / A ≤ S - p := Nat.div_le_of_le_mul h1
  omega

/-- EMA hit update never lowers the probability. -/
theorem hitStep_ge (S p a A : Nat) : p ≤ p + (S - p) * a / A :=
  Nat.le_add_right p _

/-- EMA miss update p − p * a / A stays in [0, S]. -/
theorem missStep_le (p a A S : Nat) (hp : p ≤ S) : p - p * a / A ≤ S :=
  Nat.le_trans (Nat.sub_le p _) hp

-- ── observe bookkeeping ───────────────────────────────────────────────────────

/-- observe touches at most k positions: min(k, min(acc, k) + 1) ≤ k. -/
theorem touchCount_le (k acc : Nat) : min k (min acc k + 1) ≤ k := by omega

/-- The last updated position index k − 1 is within K_MAX when 1 ≤ k < K_MAX. -/
theorem deepest_lt (k : Nat) (h1 : 1 ≤ k) (h2 : k < K_MAX) : k - 1 < K_MAX := by
  unfold K_MAX at *; omega

end Baloon.SpecController
