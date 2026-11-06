/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp
The speculation controller: a cost model prices a verify step of `k` drafts, the controller picks the draft
source (none / lookup / MTP) and length `k` with the best expected tokens per millisecond, and learns per-position
acceptance rates from outcomes by exponential moving averages.

## Source semantics
* `step_ms(k)` indexes its ratio tables at `clamp(k+1, 1, K_MAX+1) - 1`.
* `expected(p, k)` = 1 + sum over `i < k` of the running product of acceptance rates (the verify pass always
  commits one token); `same` uses `p[0]` at every position.
* `choose` starts from the plain-decode rate and only replaces it by a strictly better candidate, and falls back
  to no speculation unless the best rate beats the baseline by `min_gain`.
* `observe` clamps `accepted` to `[0, k]`, updates positions `0 .. accepted-1` with a hit and position `accepted`
  (when drafted) with a miss, i.e. `min(k, accepted+1)` positions.
* Priors: 0.86 per MTP position and `{0.35, 0.6, 0.8, 0.92}` for lookup by match-length bucket.
The model below works on natural numbers (rates in permille) so that all arithmetic is exact.

## Form template identifiers
role mapping (source names -> abstract names):
  CostModel::step_ms    -> step_cost
  Controller::expected  -> expected_tokens
  Controller::choose    -> pick_source
  Controller::observe   -> learn_acceptance
  K_MAX                 -> max_draft_len
  min_gain              -> required_gain
  ema_                  -> smoothing
-/

namespace Strata.Spec

-- § table index clamp

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def clampN (kmax k : Nat) : Nat := min (max (k + 1) 1) (kmax + 1)

/-- The clamped window size is at least 1. -/
theorem clampN_ge_one (kmax k : Nat) : 1 ≤ clampN kmax k := by
  unfold clampN; omega

/-- The clamped window size is at most `K_MAX + 1`. -/
theorem clampN_le (kmax k : Nat) : clampN kmax k ≤ kmax + 1 := by
  unfold clampN; omega

/-- `n - 1` indexes a table of `K_MAX + 1` entries. -/
theorem clampN_index_lt (kmax k : Nat) : clampN kmax k - 1 < kmax + 1 := by
  unfold clampN; omega

/-- Within range the clamp is the identity on `k + 1`. -/
theorem clampN_exact (kmax k : Nat) (h : k ≤ kmax) : clampN kmax k = k + 1 := by
  unfold clampN; omega

-- § expected tokens

/-- Per-position factor: `p[0]` when `same`, else `p[i]`. -/
def factor (p : Nat → Nat) (same : Bool) (i : Nat) : Nat := if same = true then p 0 else p i

/-- Sum of running products over the next `k` draft positions, starting at position `i`. -/
def expSum (p : Nat → Nat) (same : Bool) : Nat → Nat → Nat → Nat
  | _, 0, _ => 0
  | i, k + 1, run => run * factor p same i + expSum p same (i + 1) k (run * factor p same i)

/-- Expected tokens of a step with `k` drafts, in the model's units. -/
def expected (p : Nat → Nat) (same : Bool) (k : Nat) : Nat := 1 + expSum p same 0 k 1

/-- The verify pass always commits one token. -/
theorem expected_ge_one (p : Nat → Nat) (same : Bool) (k : Nat) : 1 ≤ expected p same k := by
  unfold expected; omega

/-- With no drafts exactly one token is expected. -/
theorem expected_zero (p : Nat → Nat) (same : Bool) : expected p same 0 = 1 := by
  simp [expected, expSum]

/-- Drafting more positions never lowers the expectation. -/
theorem expSum_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    ∀ i run, expSum p same i k run ≤ expSum p same i (k + 1) run := by
  induction k with
  | zero => intro i run; simp [expSum]
  | succ k ih =>
    intro i run
    have e1 : expSum p same i (k + 1) run =
        run * factor p same i + expSum p same (i + 1) k (run * factor p same i) := rfl
    have e2 : expSum p same i (k + 1 + 1) run =
        run * factor p same i + expSum p same (i + 1) (k + 1) (run * factor p same i) := rfl
    have h := ih (i + 1) (run * factor p same i)
    rw [e1, e2]
    omega

/-- The expected tokens are monotone in the draft length. -/
theorem expected_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    expected p same k ≤ expected p same (k + 1) := by
  unfold expected
  have := expSum_mono p same k 0 1
  omega

-- § choosing with a gain threshold

/-- The best candidate must beat the plain-decode rate by `gainPm` permille to be kept. -/
def passesGain (best baseline gainPm : Nat) : Bool := decide (baseline * (1000 + gainPm) ≤ best * 1000)

/-- A candidate that passes the gain test is at least as fast as the baseline. -/
theorem passesGain_ge (best baseline gainPm : Nat) (h : passesGain best baseline gainPm = true) :
    baseline ≤ best := by
  have h' : baseline * (1000 + gainPm) ≤ best * 1000 := of_decide_eq_true h
  have h1 : baseline * 1000 ≤ baseline * (1000 + gainPm) := Nat.mul_le_mul_left _ (by omega)
  have h2 : baseline * 1000 ≤ best * 1000 := Nat.le_trans h1 h'
  omega

/-- With zero required gain the test is exactly `baseline <= best`. -/
theorem passesGain_zero (best baseline : Nat) :
    passesGain best baseline 0 = true ↔ baseline ≤ best := by
  unfold passesGain
  simp only [Nat.add_zero, decide_eq_true_eq]
  constructor
  · intro h; omega
  · intro h; omega

/-- Running maximum over candidate rates (the strict `>` update of `consider` keeps the maximum). -/
def bestRate (init : Nat) (rs : List Nat) : Nat := rs.foldl (fun b r => max b r) init

/-- The result is at least the starting (baseline) rate. -/
theorem bestRate_ge_init (rs : List Nat) : ∀ init, init ≤ bestRate init rs := by
  induction rs with
  | nil => intro init; simp [bestRate]
  | cons r rs ih =>
    intro init
    have h := ih (max init r)
    simp only [bestRate, List.foldl_cons] at h ⊢
    omega

/-- The result is at least every candidate rate. -/
theorem bestRate_ge_mem (rs : List Nat) : ∀ init, ∀ r ∈ rs, r ≤ bestRate init rs := by
  induction rs with
  | nil => intro init r hr; simp at hr
  | cons r' rs ih =>
    intro init r hr
    rcases List.mem_cons.1 hr with h | h
    · subst h
      have := bestRate_ge_init rs (max init r)
      simp only [bestRate, List.foldl_cons] at this ⊢
      omega
    · have := ih (max init r') r h
      simp only [bestRate, List.foldl_cons] at this ⊢
      exact this

-- § learning acceptance rates

/-- Accepted count clamped to `[0, k]`. -/
def accClamp (k acc : Nat) : Nat := min acc k

/-- Number of positions whose rate is updated: the accepted ones plus the first rejected. -/
def seenCount (k acc : Nat) : Nat := min k (acc + 1)

/-- The clamped acceptance count never exceeds the window. -/
theorem accClamp_le (k acc : Nat) : accClamp k acc ≤ k := by
  unfold accClamp; omega

/-- All accepted positions are among the updated ones. -/
theorem seen_ge_acc (k acc : Nat) : accClamp k acc ≤ seenCount k (accClamp k acc) := by
  unfold accClamp seenCount; omega

/-- At most one rejected position is updated. -/
theorem seen_le_acc_succ (k acc : Nat) : seenCount k (accClamp k acc) ≤ accClamp k acc + 1 := by
  unfold accClamp seenCount; omega

/-- No position beyond the draft window is updated. -/
theorem seen_le_k (k acc : Nat) : seenCount k acc ≤ k := by
  unfold seenCount; omega

/-- A rejected position is updated exactly when not everything was accepted. -/
theorem seen_eq_acc_iff (k acc : Nat) (h : acc ≤ k) : seenCount k acc = acc ↔ acc = k := by
  unfold seenCount; omega

/-- One exponential-moving-average step on a rate in permille with smoothing `e` permille. -/
def emaStep (v e h : Nat) : Nat := (v * (1000 - e) + h * e) / 1000

/-- A rate in `[0, 1000]` stays in `[0, 1000]` after any update toward a hit value in `[0, 1000]`. -/
theorem emaStep_le (v e h : Nat) (hv : v ≤ 1000) (he : e ≤ 1000) (hh : h ≤ 1000) :
    emaStep v e h ≤ 1000 := by
  unfold emaStep
  have h1 : v * (1000 - e) ≤ 1000 * (1000 - e) := Nat.mul_le_mul_right _ hv
  have h2 : h * e ≤ 1000 * e := Nat.mul_le_mul_right _ hh
  have h3 : 1000 * (1000 - e) + 1000 * e = 1000 * 1000 := by
    rw [← Nat.mul_add, Nat.sub_add_cancel he]
  exact Nat.div_le_of_le_mul (by omega)

/-- Moving a rate toward its own value changes nothing. -/
theorem emaStep_fixed (v e : Nat) (he : e ≤ 1000) : emaStep v e v = v := by
  unfold emaStep
  have h : v * (1000 - e) + v * e = v * 1000 := by
    rw [← Nat.mul_add, Nat.sub_add_cancel he]
  rw [h]
  exact Nat.mul_div_cancel v (by decide)

/-- The lookup prior rises with the match-length bucket and stays below 1. -/
theorem lookup_prior_sorted : [350, 600, 800, 920].Pairwise (· ≤ ·) ∧ ∀ x ∈ [350, 600, 800, 920], x ≤ 1000 := by
  decide

end Strata.Spec
