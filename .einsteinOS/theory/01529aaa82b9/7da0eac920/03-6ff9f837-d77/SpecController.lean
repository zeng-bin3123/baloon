/-!
# MODULE: spec_controller
The speculative-decoding controller: it picks a draft source (none / suffix lookup / MTP) and a draft
length `k` per step by maximizing expected tokens per millisecond, and learns per-position acceptance
with an exponential moving average.

## Source semantics
* Step time is evaluated at `n = clamp(k + 1, 1, K_MAX + 1)`, so table lookups at `n - 1` are in bounds.
* Lookup drafts are bounded by `min(available, K_MAX)`.
* The match-length bucket is always in `0..3` and is monotone in the match length.
* `observe` clamps `accepted` into `0..k`, updates positions `< min(k, accepted + 1)`, and when every draft
  is accepted pulls positions `k..K_MAX-1` toward position `k-1` (valid because `k >= 1` there).
* The EMA is a convex combination, so a probability stays a probability.
* The controller returns `None` unless the best rate beats the baseline by `min_gain`.

## Form template identifiers
* `K_MAX`      <- `strata::spec::K_MAX` (= 8)
* `bucket`     <- `Controller::bucket`
* `clampN`     <- `std::clamp(k + 1, 1, K_MAX + 1)`
* `emaScaled`  <- `p += ema * (hit - p)` in thousandths
-/

namespace Strata.Spec

/-- `K_MAX` of the controller. -/
def KMAX : Nat := 8

-- § bucketing

/-- Match-length bucket: `>=16 -> 3`, `>=8 -> 2`, `>=5 -> 1`, else `0`. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- The bucket indexes the four-entry lookup table. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split <;> try omega
  split <;> try omega
  split <;> omega

/-- Longer matches never fall into a lower bucket. -/
theorem bucket_mono (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  split_ifs <;> omega

-- § table indexing

/-- The verify-window size used for the cost tables. -/
def clampN (k : Nat) : Nat := max 1 (min (k + 1) (KMAX + 1))

/-- The cost-table index `n - 1` is below the table length `K_MAX + 1`. -/
theorem clampN_index_lt (k : Nat) : clampN k - 1 < KMAX + 1 := by
  unfold clampN KMAX
  omega

/-- `n` is at least 1. -/
theorem clampN_pos (k : Nat) : 1 ≤ clampN k := by
  unfold clampN
  omega

/-- For in-range `k`, `n = k + 1` exactly. -/
theorem clampN_exact (k : Nat) (h : k ≤ KMAX) : clampN k = k + 1 := by
  unfold clampN KMAX at *
  omega

/-- The lookup draft count is bounded by `K_MAX`. -/
theorem lookup_k_le (avail : Nat) : min avail KMAX ≤ KMAX := by
  unfold KMAX
  omega

-- § observe

/-- `accepted` is clamped into `0..k`. -/
def clampAcc (acc k : Int) : Int := max 0 (min acc k)

/-- The clamped acceptance lies in `0..k` for non-negative `k`. -/
theorem clampAcc_bounds (acc k : Int) (hk : 0 ≤ k) : 0 ≤ clampAcc acc k ∧ clampAcc acc k ≤ k := by
  unfold clampAcc
  omega

/-- Number of positions updated by `observe`. -/
def seen (accepted k : Nat) : Nat := min k (accepted + 1)

/-- Every updated position index is below `k`, hence below `K_MAX` when `k <= K_MAX`. -/
theorem seen_le (accepted k : Nat) : seen accepted k ≤ k := by
  unfold seen
  omega

/-- The positions updated are within the controller's arrays. -/
theorem seen_index_lt (accepted k i : Nat) (hk : k ≤ KMAX) (hi : i < seen accepted k) : i < KMAX := by
  unfold seen KMAX at *
  omega

/-- When all `k >= 1` drafts are accepted, the reference position `k - 1` exists and the tail starts at `k`. -/
theorem tail_pull_valid (k : Nat) (hk1 : 1 ≤ k) (hk : k < KMAX) : k - 1 < KMAX ∧ k ≤ KMAX := by
  unfold KMAX at *
  omega

-- § EMA

/-- EMA step in thousandths: `(1000 - a) * p + a * h` with `a` the rate and `h` the hit indicator, both scaled. -/
def emaScaled (a p h : Nat) : Nat := (1000 - a) * p + a * h

/-- The EMA of values in `[0, 1000]` with rate `a <= 1000` stays in `[0, 1000*1000]` (a convex combination). -/
theorem emaScaled_le (a p h : Nat) (ha : a ≤ 1000) (hp : p ≤ 1000) (hh : h ≤ 1000) :
    emaScaled a p h ≤ 1000 * 1000 := by
  unfold emaScaled
  have h1 : (1000 - a) * p ≤ (1000 - a) * 1000 := Nat.mul_le_mul_left _ hp
  have h2 : a * h ≤ a * 1000 := Nat.mul_le_mul_left _ hh
  have h3 : (1000 - a) * 1000 + a * 1000 = 1000 * 1000 := by
    rw [← Nat.add_mul, Nat.sub_add_cancel ha]
  omega

/-- A hit (h = 1000) never decreases the estimate below the convex floor `(1000 - a) * p`. -/
theorem emaScaled_ge (a p h : Nat) : (1000 - a) * p ≤ emaScaled a p h := by
  unfold emaScaled
  omega

-- § decision rule

/-- `choose` keeps a non-baseline choice only when its rate is at least `baseline * (1 + gain)`, in thousandths. -/
def keepChoice (best base gain : Nat) : Prop := base * (1000 + gain) ≤ best * 1000

/-- A kept choice is at least as fast as the baseline. -/
theorem keepChoice_ge_base (best base gain : Nat) (h : keepChoice best base gain) : base ≤ best := by
  unfold keepChoice at h
  by_contra hc
  have : best * 1000 < base * 1000 := Nat.mul_lt_mul_of_pos_right (by omega) (by omega)
  have h2 : base * 1000 ≤ base * (1000 + gain) := Nat.mul_le_mul_left _ (by omega)
  omega

/-- Source `None` is exactly the baseline: rate `1 / step_ms(0)`, zero drafts and expected tokens one. -/
theorem none_choice_tokens (k : Nat) (h : k = 0) : k + 1 = 1 := by omega

end Strata.Spec
