/-!
# MODULE: spec_controller
Speculative-decoding controller: chooses drafter (none / suffix lookup / MTP) and draft length k per step by
maximizing expected tokens per millisecond, keeping k = 0 unless the best choice beats plain decoding by `min_gain`.

## Source semantics
* `bucket(match)` maps a match length to one of 4 acceptance buckets (<5, 5..7, 8..15, >=16).
* `step_ms` clamps `n = k+1` into `1..K_MAX+1` before indexing the 9-entry ratio tables.
* `observe` clamps `accepted` into `0..k`, updates positions `0..min(k, accepted+1)-1`.
* The cost tables are monotone, and `U(n) <= n * U(1)` so extra-use cost is non-negative.

## Form template identifiers
K_MAX -> `KMAX`; bucket -> `bucket`; clamp(k+1,1,K_MAX+1) -> `stepN`; dense_ratio/distinct_ratio (x100) ->
`denseRatio`/`distinctRatio`; observe -> `observeSeen`/`observeAcc`.
-/

namespace Strata.Spec

-- § acceptance buckets

/-- Maximum draft length. -/
def KMAX : Nat := 8

/-- Match-length bucket. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- Buckets index a 4-entry array. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split_ifs <;> omega

/-- Longer matches never land in a lower bucket. -/
theorem bucket_mono (m n : Nat) (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split_ifs <;> omega

-- § cost-model indexing

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def stepN (k : Nat) : Nat := min (max (k + 1) 1) (KMAX + 1)

/-- The clamped count indexes the 9-entry tables at `n - 1`. -/
theorem stepN_idx (k : Nat) : 1 ≤ stepN k ∧ stepN k - 1 < 9 := by
  unfold stepN KMAX
  omega

/-- Inside the legal range the clamp is the identity. -/
theorem stepN_exact (k : Nat) (h : k ≤ KMAX) : stepN k = k + 1 := by
  unfold stepN
  unfold KMAX at *
  omega

/-- `dense_ratio` scaled by 100. -/
def denseRatio : List Nat := [100, 105, 130, 145, 185, 220, 260, 300, 330]

/-- `distinct_ratio` scaled by 100. -/
def distinctRatio : List Nat := [100, 170, 231, 288, 340, 389, 435, 480, 520]

/-- Both tables have `K_MAX + 1` entries. -/
theorem tables_size : denseRatio.length = KMAX + 1 ∧ distinctRatio.length = KMAX + 1 := by decide

/-- Dense ratio is monotone in the token count. -/
theorem denseRatio_mono :
    (List.range 8).all (fun i => decide (denseRatio[i]! ≤ denseRatio[i + 1]!)) = true := by decide

/-- Distinct-expert ratio is monotone in the token count. -/
theorem distinctRatio_mono :
    (List.range 8).all (fun i => decide (distinctRatio[i]! ≤ distinctRatio[i + 1]!)) = true := by decide

/-- `U(n) <= n * U(1)`: the uses-per-expert factor is at least 1, so the extra-use cost is non-negative. -/
theorem distinct_le_n :
    (List.range 9).all (fun i => decide (distinctRatio[i]! ≤ (i + 1) * 100)) = true := by decide

-- § observe

/-- Accepted count clamped to the drafted count. -/
def observeAcc (accepted k : Nat) : Nat := min accepted k

/-- Number of positions whose acceptance estimate is updated. -/
def observeSeen (accepted k : Nat) : Nat := min k (observeAcc accepted k + 1)

/-- Updated positions are within the draft, and cover every accepted position. -/
theorem observeSeen_bounds (accepted k : Nat) :
    observeAcc accepted k ≤ observeSeen accepted k ∧ observeSeen accepted k ≤ k := by
  unfold observeSeen observeAcc
  omega

/-- When some draft token was rejected, exactly the rejected position is also updated. -/
theorem observeSeen_rejected (accepted k : Nat) (h : accepted < k) :
    observeSeen accepted k = accepted + 1 := by
  unfold observeSeen observeAcc
  omega

/-- When everything was accepted, all k positions are updated and the window-growth index `k - 1` is valid. -/
theorem observe_all_accepted (accepted k : Nat) (h : k ≤ accepted) (hk : 1 ≤ k) (hmax : k ≤ KMAX) :
    observeSeen accepted k = k ∧ k - 1 < KMAX := by
  unfold observeSeen observeAcc KMAX at *
  omega

/-- Tokens committed by a verify pass: the bonus token plus the accepted prefix. -/
def committed (accepted k : Nat) : Nat := 1 + observeAcc accepted k

/-- A verify pass commits between 1 and k+1 tokens. -/
theorem committed_bounds (accepted k : Nat) : 1 ≤ committed accepted k ∧ committed accepted k ≤ k + 1 := by
  unfold committed observeAcc
  omega

-- § the gain threshold

/-- Keep a draft only when `best * 100 >= baseline * 105` (min_gain = 5%). -/
def keepDraft (best baseline : Nat) : Bool := decide (baseline * 105 ≤ best * 100)

/-- A kept draft is strictly faster than the baseline whenever the baseline is positive. -/
theorem keepDraft_faster (best baseline : Nat) (h : keepDraft best baseline = true) (hb : 0 < baseline) :
    baseline < best := by
  unfold keepDraft at h
  have h' : baseline * 105 ≤ best * 100 := of_decide_eq_true h
  omega

end Strata.Spec
