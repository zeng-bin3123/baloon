import Mathlib

/-!
# MODULE: spec_decode_controller
Adaptive controller for speculative decoding. It picks a draft source (none, suffix lookup, or MTP)
and a draft length `k` to maximize expected committed tokens per millisecond, using a step cost
model, and learns per-position acceptance rates with an exponential moving average.

## Source semantics
* `step_ms(k)` uses `n = clamp(k+1, 1, K_MAX+1)` for the table lookups; draft cost is linear in `k`.
* `expected(p, k, same)` = 1 + Σ_{j<k} Π_{i≤j} p_i (the verify pass always commits one token);
  with `same` every position uses `p[0]`.
* `choose` starts from baseline (no drafting), takes strictly better candidates, and falls back to
  baseline unless the best rate beats it by factor `1 + min_gain`.
* `observe` updates positions `0..min(k, accepted+1)-1`, hit = 1 for accepted positions and 0 for the
  first rejected one, via `x += ema * (hit - x)`.

## Form template identifiers
* Controller → DraftPolicy, CostModel → StepCost, Choice → DraftChoice, K_MAX → maxDraft,
  mtp_p_ → mtpAccept, lookup_q_ → lookupAccept, min_gain_ → minGain, ema_ → emaRate,
  choose → selectDraft, observe → learnOutcome, expected → expectedCommitted.
-/

namespace Strata.Spec.Controller

-- § cost-model index clamp

/-- Table index `n = clamp(k+1, 1, K_MAX+1)` (k is non-negative here). -/
def nIdx (kMax k : Nat) : Nat := min (k + 1) (kMax + 1)

/-- The index is at least 1, so `dense_ratio[n-1]` is in bounds. -/
theorem nIdx_pos (kMax k : Nat) : 1 ≤ nIdx kMax k := by
  unfold nIdx; omega

/-- The index is at most `K_MAX + 1`. -/
theorem nIdx_le (kMax k : Nat) : nIdx kMax k ≤ kMax + 1 := by
  unfold nIdx; omega

/-- With no drafting the index is 1. -/
theorem nIdx_zero (kMax : Nat) : nIdx kMax 0 = 1 := by
  unfold nIdx; omega

-- § expected committed tokens

/-- Running product loop: `aux p same k i run` sums `run * p_i`, then continues. -/
def expAux (p : Nat → ℚ) (same : Bool) : Nat → Nat → ℚ → ℚ
  | 0, _, _ => 0
  | k + 1, i, run =>
    let r := run * (if same then p 0 else p i)
    r + expAux p same k (i + 1) r

/-- Expected committed tokens: the verify pass commits one, plus the accepted-prefix sums. -/
def expected (p : Nat → ℚ) (k : Nat) (same : Bool) : ℚ := 1 + expAux p same k 0 1

/-- With zero drafted tokens, exactly one token is committed. -/
theorem expected_zero (p : Nat → ℚ) (same : Bool) : expected p 0 same = 1 := by
  simp [expected, expAux]

/-- Auxiliary: the sum is nonnegative when probabilities and the run are nonnegative. -/
theorem expAux_nonneg (p : Nat → ℚ) (same : Bool) (hp : ∀ i, 0 ≤ p i) :
    ∀ k i (run : ℚ), 0 ≤ run → 0 ≤ expAux p same k i run := by
  intro k
  induction k with
  | zero => intro i run _; simp [expAux]
  | succ n ih =>
    intro i run hr
    simp only [expAux]
    have hq : 0 ≤ (if same then p 0 else p i) := by split <;> exact hp _
    have hr' : 0 ≤ run * (if same then p 0 else p i) := mul_nonneg hr hq
    exact add_nonneg hr' (ih _ _ hr')

/-- Expected committed tokens is at least 1 for nonnegative acceptance probabilities. -/
theorem expected_ge_one (p : Nat → ℚ) (same : Bool) (hp : ∀ i, 0 ≤ p i) (k : Nat) :
    1 ≤ expected p k same := by
  have := expAux_nonneg p same hp k 0 1 (by norm_num)
  unfold expected; linarith

/-- Auxiliary: with probabilities in [0,1] and run in [0,1], the sum is at most `k`. -/
theorem expAux_le (p : Nat → ℚ) (same : Bool) (hp0 : ∀ i, 0 ≤ p i) (hp1 : ∀ i, p i ≤ 1) :
    ∀ k i (run : ℚ), 0 ≤ run → run ≤ 1 → expAux p same k i run ≤ k := by
  intro k
  induction k with
  | zero => intro i run _ _; simp [expAux]
  | succ n ih =>
    intro i run h0 h1
    simp only [expAux]
    have hq0 : 0 ≤ (if same then p 0 else p i) := by split <;> exact hp0 _
    have hq1 : (if same then p 0 else p i) ≤ 1 := by split <;> exact hp1 _
    have hr0 : 0 ≤ run * (if same then p 0 else p i) := mul_nonneg h0 hq0
    have hr1 : run * (if same then p 0 else p i) ≤ 1 := by nlinarith
    have := ih (i + 1) _ hr0 hr1
    push_cast
    linarith

/-- Drafting `k` tokens can commit at most `1 + k` tokens. -/
theorem expected_le (p : Nat → ℚ) (same : Bool) (hp0 : ∀ i, 0 ≤ p i) (hp1 : ∀ i, p i ≤ 1)
    (k : Nat) : expected p k same ≤ 1 + k := by
  have := expAux_le p same hp0 hp1 k 0 1 (by norm_num) (by norm_num)
  unfold expected; linarith

-- § choice gating

/-- Candidate comparison: a candidate replaces the incumbent only if strictly faster. -/
def consider (best rate : ℚ) : ℚ := if rate > best then rate else best

/-- Considering a candidate never lowers the best rate. -/
theorem consider_ge (best rate : ℚ) : best ≤ consider best rate := by
  unfold consider; split <;> linarith

/-- The result is one of the two inputs' maximum. -/
theorem consider_eq_max (best rate : ℚ) : consider best rate = max best rate := by
  unfold consider
  split
  · rename_i h; exact (max_eq_right h.le).symm
  · rename_i h; exact (max_eq_left (not_lt.mp h)).symm

/-- Final min-gain gate: fall back to baseline unless the best beats `baseline * (1 + gain)`. -/
def gate (best baseline gain : ℚ) : ℚ :=
  if best < baseline * (1 + gain) then baseline else best

/-- The gated rate is never below the baseline (nonnegative baseline and gain, best ≥ baseline). -/
theorem gate_ge_baseline (best baseline gain : ℚ) (hb : 0 ≤ baseline) (hg : 0 ≤ gain) :
    baseline ≤ gate best baseline gain := by
  unfold gate
  split
  · exact le_refl _
  · rename_i h
    have h' : baseline * (1 + gain) ≤ best := not_lt.mp h
    nlinarith

/-- A non-baseline answer survives the gate only if it clears the threshold. -/
theorem gate_survivor (best baseline gain : ℚ) (h : gate best baseline gain ≠ baseline) :
    baseline * (1 + gain) ≤ best := by
  unfold gate at h
  by_cases hc : best < baseline * (1 + gain)
  · simp [hc] at h
  · exact not_lt.mp hc

/-- With zero gain and nonnegative baseline, a best at least the baseline passes unchanged. -/
theorem gate_zero_gain (best baseline : ℚ) (h : baseline ≤ best) :
    gate best baseline 0 = best := by
  have hn : ¬ best < baseline * (1 + 0) := by
    rw [add_zero, mul_one]; exact not_lt.mpr h
  unfold gate
  rw [if_neg hn]

-- § observation update

/-- Number of positions that receive an update: `min k (accepted + 1)`. -/
def seen (k accepted : Nat) : Nat := min k (accepted + 1)

/-- Accepted count after clamping into `[0, k]` (as a Nat it is `min accepted k`). -/
def clampAcc (k accepted : Nat) : Nat := min accepted k

/-- At most `k` positions are updated. -/
theorem seen_le (k a : Nat) : seen k a ≤ k := by
  unfold seen; omega

/-- If not everything was accepted, the first rejected position is also updated. -/
theorem seen_of_lt (k a : Nat) (h : a < k) : seen k a = a + 1 := by
  unfold seen; omega

/-- If everything was accepted, exactly `k` positions are updated. -/
theorem seen_of_full (k : Nat) : seen k k = k := by
  unfold seen; omega

/-- After clamping, accepted never exceeds `k`. -/
theorem clampAcc_le (k a : Nat) : clampAcc k a ≤ k := by
  unfold clampAcc; omega

/-- Exponential moving average step. -/
def emaStep (ema x hit : ℚ) : ℚ := x + ema * (hit - x)

/-- The EMA step equals the convex combination `(1 - ema) x + ema hit`. -/
theorem emaStep_convex (ema x hit : ℚ) : emaStep ema x hit = (1 - ema) * x + ema * hit := by
  unfold emaStep; ring

/-- The EMA of values in [0,1] with rate in [0,1] stays in [0,1], so probabilities remain valid. -/
theorem emaStep_mem_unit (ema x hit : ℚ) (he0 : 0 ≤ ema) (he1 : ema ≤ 1)
    (hx0 : 0 ≤ x) (hx1 : x ≤ 1) (ht0 : 0 ≤ hit) (ht1 : hit ≤ 1) :
    0 ≤ emaStep ema x hit ∧ emaStep ema x hit ≤ 1 := by
  rw [emaStep_convex]
  constructor
  · have h1 : 0 ≤ (1 - ema) * x := mul_nonneg (by linarith) hx0
    have h2 : 0 ≤ ema * hit := mul_nonneg he0 ht0
    linarith
  · have h1 : (1 - ema) * x ≤ (1 - ema) * 1 := mul_le_mul_of_nonneg_left hx1 (by linarith)
    have h2 : ema * hit ≤ ema * 1 := mul_le_mul_of_nonneg_left ht1 he0
    linarith

/-- An accepted observation (hit = 1) never lowers the estimate. -/
theorem emaStep_hit_mono (ema x : ℚ) (he0 : 0 ≤ ema) (hx1 : x ≤ 1) : x ≤ emaStep ema x 1 := by
  unfold emaStep; nlinarith

/-- A rejected observation (hit = 0) never raises the estimate. -/
theorem emaStep_miss_mono (ema x : ℚ) (he0 : 0 ≤ ema) (hx0 : 0 ≤ x) : emaStep ema x 0 ≤ x := by
  unfold emaStep; nlinarith

/-- Pulling an untested position toward the deepest observed rate moves it monotonically toward it. -/
theorem emaStep_toward (ema x target : ℚ) (he0 : 0 ≤ ema) (he1 : ema ≤ 1) (h : x ≤ target) :
    x ≤ emaStep ema x target ∧ emaStep ema x target ≤ target := by
  unfold emaStep
  constructor <;> nlinarith

end Strata.Spec.Controller
