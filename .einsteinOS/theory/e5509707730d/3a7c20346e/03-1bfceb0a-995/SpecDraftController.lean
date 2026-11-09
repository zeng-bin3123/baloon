/-!
# MODULE: spec_draft_controller
SOURCES: src/spec/controller.cpp, include/strata/spec/controller.hpp
Chooses, per verify step, which drafter (none / suffix lookup / MTP) and how many draft tokens `k`
maximise expected committed tokens per millisecond, and learns per-position acceptance with an EMA.

## Source semantics
* `K_MAX = 8`; a verify pass of `k` drafts has `n = clamp(k+1, 1, K_MAX+1)` tokens, indexing the ratio
  tables at `n-1`, which must stay inside the `K_MAX+1`-entry arrays.
* `bucket(match)` maps a match length to one of four lookup-acceptance buckets (<5, 5..7, 8..15, >=16).
* `expected(p, k)` = 1 + sum of running products of per-position acceptance (the verify pass always
  commits one token).
* `choose` starts from plain decoding (k = 0) and only replaces it by a candidate that is strictly better,
  and finally falls back to plain decoding unless the gain exceeds `min_gain`.
* `observe` clamps `accepted` to `[0, k]` and updates positions `0 .. min(k, accepted+1) - 1` by an EMA
  step toward 1 (accepted) or 0 (the rejected position).

## Form template identifiers
bucket -> `Controller::bucket` (match-length bucketing)
expAux/expected -> `Controller::expected`
nTok -> `std::clamp(k + 1, 1, K_MAX + 1)` in `CostModel::step_ms`
seen -> `std::min(c.k, accepted + 1)` in `Controller::observe`
-/

namespace Strata.Spec

-- § constants and bucketing

/-- `K_MAX` of the controller header. -/
def kMax : Nat := 8

/-- Lookup acceptance bucket of a match length, as `Controller::bucket`. -/
def bucket (m : Nat) : Nat :=
  if m ≥ 16 then 3 else if m ≥ 8 then 2 else if m ≥ 5 then 1 else 0

/-- Every bucket index is a valid index into the 4-entry `lookup_q_` array. -/
theorem bucket_lt_four (m : Nat) : bucket m < 4 := by
  unfold bucket
  split <;> (try split) <;> (try split) <;> omega

/-- A longer match never falls into a lower bucket. -/
theorem bucket_mono {m n : Nat} (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split <;> split <;> (try split) <;> (try split) <;> (try split) <;> (try split) <;> omega

-- § verify-window size and table indexing

/-- Number of tokens of a verify pass with `k` drafts: `clamp(k+1, 1, K_MAX+1)`. -/
def nTok (k : Nat) : Nat := min (max (k + 1) 1) (kMax + 1)

/-- The table index `n - 1` is always inside the `K_MAX + 1`-entry ratio arrays. -/
theorem nTok_idx_lt (k : Nat) : nTok k - 1 < kMax + 1 := by
  unfold nTok kMax
  omega

/-- A verify pass always holds at least one token. -/
theorem nTok_pos (k : Nat) : 1 ≤ nTok k := by
  unfold nTok kMax
  omega

/-- The default ratio tables (scaled by 100) have `K_MAX + 1` entries. -/
theorem ratio_tables_length :
    [100, 105, 130, 145, 185, 220, 260, 300, 330].length = kMax + 1 ∧
    [100, 170, 231, 288, 340, 389, 435, 480, 520].length = kMax + 1 := by
  decide

/-- Both default ratio tables are non-decreasing: bigger windows never cost less. -/
theorem ratio_tables_sorted :
    List.Pairwise (· ≤ ·) [100, 105, 130, 145, 185, 220, 260, 300, 330] ∧
    List.Pairwise (· ≤ ·) [100, 170, 231, 288, 340, 389, 435, 480, 520] := by
  decide

-- § expected committed tokens

/-- Sum of running products of acceptance (percent), starting from run length `run`. -/
def expAux : Nat → List Nat → Nat
  | _, [] => 0
  | run, p :: ps => run * p / 100 + expAux (run * p / 100) ps

/-- `expected` scaled by 100: the committed token plus the accepted drafts. -/
def expected (ps : List Nat) : Nat := 100 + expAux 100 ps

/-- One acceptance step never increases the running product. -/
theorem runstep_le (run p : Nat) (hp : p ≤ 100) : run * p / 100 ≤ run := by
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left run hp
  omega

/-- The verify pass always commits at least one token. -/
theorem expected_ge_one (ps : List Nat) : 100 ≤ expected ps := by
  unfold expected
  omega

/-- With acceptances at most 100%, the accumulated drafts are bounded by `run * #drafts`. -/
theorem expAux_le (ps : List Nat) :
    ∀ run, (∀ p ∈ ps, p ≤ 100) → expAux run ps ≤ run * ps.length := by
  induction ps with
  | nil =>
    intro run _
    simp [expAux]
  | cons p ps ih =>
    intro run h
    have hp : p ≤ 100 := h p (by simp)
    have hr : run * p / 100 ≤ run := runstep_le run p hp
    have ih' := ih (run * p / 100) (fun q hq => h q (by simp [hq]))
    have hm : run * p / 100 * ps.length ≤ run * ps.length := Nat.mul_le_mul_right _ hr
    have hl : run * (ps.length + 1) = run * ps.length + run := by
      rw [Nat.mul_add, Nat.mul_one]
    simp only [expAux, List.length_cons]
    rw [hl]
    omega

/-- Expected committed tokens never exceed one plus the number of drafts (all accepted). -/
theorem expected_le (ps : List Nat) (h : ∀ p ∈ ps, p ≤ 100) :
    expected ps ≤ 100 * (ps.length + 1) := by
  have := expAux_le ps 100 h
  unfold expected
  omega

-- § choice: never worse than plain decoding

/-- Best rate of the candidates, starting from the plain-decode baseline (strict improvement only). -/
def bestOf (base : Nat) (cands : List Nat) : Nat := cands.foldl max base

/-- The fold of `max` never drops below its starting value. -/
theorem foldl_max_ge (l : List Nat) : ∀ a, a ≤ l.foldl max a := by
  induction l with
  | nil => intro a; simp
  | cons x xs ih =>
    intro a
    have := ih (max a x)
    simp only [List.foldl_cons]
    omega

/-- The chosen candidate rate is at least the plain-decoding baseline. -/
theorem bestOf_ge_base (base : Nat) (cands : List Nat) : base ≤ bestOf base cands :=
  foldl_max_ge cands base

/-- The `min_gain` gate (percent): keep the candidate only if it clears `base * (100 + gain)`. -/
def gate (base best gain : Nat) : Nat := if best * 100 < base * (100 + gain) then base else best

/-- After gating, the rate is still at least the baseline. -/
theorem gate_ge_base (base best gain : Nat) (h : base ≤ best) : base ≤ gate base best gain := by
  unfold gate
  split <;> omega

/-- A gated-in candidate beats the baseline by at least the configured margin. -/
theorem gate_margin (base best gain : Nat) (h : gate base best gain ≠ base) :
    base * (100 + gain) ≤ best * 100 := by
  unfold gate at h
  split at h
  · exact absurd rfl h
  · omega

-- § observe: clamping and the EMA

/-- `accepted` clamped to `[0, k]`. -/
def acc' (k accepted : Nat) : Nat := min accepted k

/-- Number of positions updated: `min(k, accepted + 1)`. -/
def seen (k accepted : Nat) : Nat := min k (acc' k accepted + 1)

/-- The clamped acceptance never exceeds the number of drafts. -/
theorem acc'_le (k a : Nat) : acc' k a ≤ k := by
  unfold acc'
  omega

/-- At most `k` positions are updated, so the `mtp_p_[i]` writes stay in range. -/
theorem seen_le (k a : Nat) : seen k a ≤ k := by
  unfold seen
  omega

/-- Every accepted position is updated. -/
theorem acc_le_seen (k a : Nat) : acc' k a ≤ seen k a := by
  unfold seen acc'
  omega

/-- If some draft was rejected, exactly one extra (the rejected) position is updated. -/
theorem seen_of_reject (k a : Nat) (h : a < k) : seen k a = a + 1 := by
  unfold seen acc'
  omega

/-- EMA step toward 1 (scaled: 100) keeps the estimate in `[0, 100]` (result scaled by 1000). -/
theorem ema_hit_bounds (x e : Int) (hx0 : 0 ≤ x) (hx1 : x ≤ 100) (he0 : 0 ≤ e) (he1 : e ≤ 1000) :
    0 ≤ 1000 * x + e * (100 - x) ∧ 1000 * x + e * (100 - x) ≤ 100000 := by
  have h1 : 0 ≤ e * (100 - x) := Int.mul_nonneg he0 (by omega)
  have h2 : e * (100 - x) ≤ 1000 * (100 - x) := Int.mul_le_mul_of_nonneg_right he1 (by omega)
  constructor <;> omega

/-- EMA step toward 0 keeps the estimate in `[0, 100]` (result scaled by 1000). -/
theorem ema_miss_bounds (x e : Int) (hx0 : 0 ≤ x) (hx1 : x ≤ 100) (he0 : 0 ≤ e) (he1 : e ≤ 1000) :
    0 ≤ 1000 * x + e * (-x) ∧ 1000 * x + e * (-x) ≤ 100000 := by
  have h1 : 0 ≤ e * x := Int.mul_nonneg he0 hx0
  have h2 : e * x ≤ 1000 * x := Int.mul_le_mul_of_nonneg_right he1 hx0
  have h3 : e * (-x) = -(e * x) := Int.mul_neg e x
  rw [h3]
  constructor <;> omega

end Strata.Spec
