/-!
# MODULE: speculative_draft_controller
SOURCES: src/spec/controller.cpp
Chooses, per decode step, whether to speculate and with which drafter (suffix lookup or MTP) and how many
tokens `k`, by maximising expected committed tokens per millisecond under a cost model; then learns
per-position acceptance rates online with an EMA.

## Source semantics
* `expected(p, k, same) = 1 + Σ_{i<k} Π_{j≤i} p_j` (or `p_0^(i+1)` when `same`): the verify pass always
  commits one token, and draft i only counts if drafts 0..i were all accepted.
* `step_ms(k)` indexes the per-n ratio tables at `n = clamp(k+1, 1, K_MAX+1)`, so index `n-1 ≤ K_MAX`.
* `choose` keeps the best rate, but falls back to no speculation unless it beats the baseline by
  `(1 + min_gain)`.
* `observe` clamps `accepted` to `[0, k]`, updates positions `0..min(k, accepted+1)-1` with hit=1 for
  `i < accepted` and hit=0 for the single rejected position; if the whole window was accepted, positions
  beyond it are pulled toward the deepest observed rate so the window can grow.
* EMA update `p ← p + ema·(hit − p)` = `(1−ema)·p + ema·hit`.
Rates and probabilities are modelled in fixed point (`S` = 1.0, EMA = `an/ad`), integer floor division.

## Form template identifiers
Controller::expected → expectedFx / expectedTail, CostModel::step_ms table index → stepIndex,
Controller::observe → acceptedClamped / seenCount / emaStep, Controller::choose gain test → keepsDraft,
K_MAX → kMax
-/

namespace Strata.Spec

-- § cost-model table index

/-- `clamp(k+1, 1, K_MAX+1)`. -/
def stepN (kMax k : Nat) : Nat := max 1 (min (k + 1) (kMax + 1))

/-- The table index `n - 1` used by `step_ms`. -/
def stepIndex (kMax k : Nat) : Nat := stepN kMax k - 1

/-- The ratio tables (size K_MAX+1) are always indexed in bounds, for any requested k. -/
theorem stepIndex_in_bounds (kMax k : Nat) : stepIndex kMax k < kMax + 1 := by
  unfold stepIndex stepN
  simp only [Nat.max_def, Nat.min_def]
  split <;> split <;> omega

/-- Within the window, `n = k + 1` exactly (no clamping happens). -/
theorem stepN_exact (kMax k : Nat) (hk : k ≤ kMax) : stepN kMax k = k + 1 := by
  unfold stepN
  simp only [Nat.max_def, Nat.min_def]
  split <;> split <;> omega

-- § expected tokens per verify pass (fixed point, S = 1.0)

/-- Draft contributions: `run ← run·p/S`, summed. -/
def expectedTail (S : Nat) : Nat → List Nat → Nat
  | _, [] => 0
  | run, p :: ps => run * p / S + expectedTail S (run * p / S) ps

/-- `expected`: one committed token (S) plus the draft contributions. `same` is `List.replicate k q`. -/
def expectedFx (S : Nat) (ps : List Nat) : Nat := S + expectedTail S S ps

/-- A probability ≤ 1 never increases the running acceptance product. -/
theorem run_step_le (S run p : Nat) (hp : p ≤ S) : run * p / S ≤ run := by
  cases Nat.eq_zero_or_pos S with
  | inl h0 => subst h0; simp
  | inr hpos =>
    apply Nat.div_le_of_le_mul
    have : run * p ≤ run * S := Nat.mul_le_mul_left run hp
    rw [Nat.mul_comm S run]
    exact this

/-- Each draft adds at most the current run, so the tail is bounded by `run · k`. -/
theorem expectedTail_le (S : Nat) (ps : List Nat) (hps : ∀ p ∈ ps, p ≤ S) :
    ∀ run, expectedTail S run ps ≤ run * ps.length := by
  induction ps with
  | nil => intro run; simp [expectedTail]
  | cons p ps ih =>
    intro run
    have hp : p ≤ S := hps p (List.mem_cons_self ..)
    have hrest : ∀ q ∈ ps, q ≤ S := fun q hq => hps q (List.mem_cons_of_mem _ hq)
    have hr : run * p / S ≤ run := run_step_le S run p hp
    have ih' := ih hrest (run * p / S)
    have hmul : run * p / S * ps.length ≤ run * ps.length := Nat.mul_le_mul_right _ hr
    simp only [expectedTail, List.length_cons, Nat.mul_succ]
    omega

/-- At least one token is always committed. -/
theorem expected_ge_one (S : Nat) (ps : List Nat) : S ≤ expectedFx S ps := by
  unfold expectedFx; omega

/-- With probabilities ≤ 1, k drafts give at most k + 1 tokens. -/
theorem expected_le_k_plus_one (S : Nat) (ps : List Nat) (hps : ∀ p ∈ ps, p ≤ S) :
    expectedFx S ps ≤ S * (ps.length + 1) := by
  unfold expectedFx
  have := expectedTail_le S ps hps S
  rw [Nat.mul_succ]
  omega

/-- Once the run is zero it stays zero. -/
theorem expectedTail_zero (S : Nat) (ps : List Nat) : expectedTail S 0 ps = 0 := by
  induction ps with
  | nil => rfl
  | cons p ps ih => simp [expectedTail, ih]

/-- If the first draft is certainly rejected, speculation yields exactly the one verified token. -/
theorem expected_first_reject (S : Nat) (ps : List Nat) : expectedFx S (0 :: ps) = S := by
  simp [expectedFx, expectedTail, expectedTail_zero]

/-- Certain acceptance of every draft yields exactly k + 1 tokens. -/
theorem expected_all_accept (S k : Nat) (hS : 0 < S) :
    expectedFx S (List.replicate k S) = S * (k + 1) := by
  have hstep : S * S / S = S := Nat.mul_div_cancel S hS
  have htail : ∀ k, expectedTail S S (List.replicate k S) = S * k := by
    intro k
    induction k with
    | zero => rfl
    | succ n ih =>
      simp only [List.replicate_succ, expectedTail, hstep, ih, Nat.mul_succ]
      omega
  unfold expectedFx
  rw [htail k, Nat.mul_succ]
  omega

/-- Drafting one more token never lowers the expected token count (only its cost can stop it). -/
theorem expectedTail_append_ge (S : Nat) (ps : List Nat) (q : Nat) :
    ∀ run, expectedTail S run ps ≤ expectedTail S run (ps ++ [q]) := by
  induction ps with
  | nil => intro run; simp [expectedTail]
  | cons p ps ih =>
    intro run
    simp only [List.cons_append, expectedTail]
    have := ih (run * p / S)
    omega

/-- `expected` is monotone in the window length. -/
theorem expected_mono_append (S : Nat) (ps : List Nat) (q : Nat) :
    expectedFx S ps ≤ expectedFx S (ps ++ [q]) := by
  unfold expectedFx
  have := expectedTail_append_ge S ps q S
  omega

-- § choose: the minimum-gain gate

/-- Keep a drafting choice only if `rate ≥ baseline·(1 + num/den)`, cross-multiplied. -/
def keepsDraft (baseline rate num den : Nat) : Bool :=
  decide (baseline * (den + num) ≤ rate * den)

/-- A kept draft really beats the baseline by the required margin. -/
theorem keepsDraft_sound (baseline rate num den : Nat) (h : keepsDraft baseline rate num den = true) :
    baseline * (den + num) ≤ rate * den := by
  unfold keepsDraft at h
  exact of_decide_eq_true h

/-- With zero required gain, any rate at least the baseline is kept. -/
theorem keepsDraft_zero_gain (baseline rate den : Nat) (h : baseline ≤ rate) :
    keepsDraft baseline rate 0 den = true := by
  unfold keepsDraft
  apply decide_eq_true
  simp only [Nat.add_zero]
  exact Nat.mul_le_mul_right den h

-- § observe: clamping and the update window

/-- `accepted = clamp(accepted, 0, k)` (negative input clamps to 0 via `Int.toNat`). -/
def acceptedClamped (k : Nat) (accepted : Int) : Nat := min k accepted.toNat

/-- `seen = min(k, accepted + 1)`: positions that were actually tested. -/
def seenCount (k : Nat) (accepted : Int) : Nat := min k (acceptedClamped k accepted + 1)

/-- The clamped acceptance never exceeds the window. -/
theorem accepted_le_k (k : Nat) (a : Int) : acceptedClamped k a ≤ k := by
  unfold acceptedClamped; exact Nat.min_le_left _ _

/-- Updates stay inside the window (so inside `mtp_p_` whenever `k ≤ K_MAX`). -/
theorem seen_le_k (k : Nat) (a : Int) : seenCount k a ≤ k := by
  unfold seenCount; exact Nat.min_le_left _ _

/-- All accepted positions are updated as hits. -/
theorem accepted_le_seen (k : Nat) (a : Int) : acceptedClamped k a ≤ seenCount k a := by
  unfold seenCount
  have := accepted_le_k k a
  simp only [Nat.min_def]
  split <;> omega

/-- At most one position is recorded as a miss — the first rejected one. -/
theorem at_most_one_miss (k : Nat) (a : Int) : seenCount k a - acceptedClamped k a ≤ 1 := by
  unfold seenCount
  simp only [Nat.min_def]
  split <;> omega

/-- Exactly one miss is recorded iff not everything drafted was accepted. -/
theorem one_miss_iff_partial (k : Nat) (a : Int) :
    seenCount k a - acceptedClamped k a = 1 ↔ acceptedClamped k a < k := by
  have := accepted_le_k k a
  unfold seenCount
  simp only [Nat.min_def]
  split <;> omega

/-- The extrapolation branch (full acceptance, `k < K_MAX`) reads `mtp_p_[k-1]`, valid as `k ≥ 1`, and
writes `k..K_MAX-1`, all inside an array of size K_MAX. -/
theorem extrapolation_in_bounds (k kMax : Nat) (hk : 0 < k) (hlt : k < kMax) :
    k - 1 < kMax ∧ ∀ i, k ≤ i → i < kMax → i < kMax := by
  exact ⟨by omega, fun _ _ h => h⟩

-- § EMA update in fixed point

/-- `p + ema·(hit − p)` = `((1−ema)·p + ema·hit)` with `ema = an/ad`. -/
def emaStep (an ad p hit : Nat) : Nat := (p * (ad - an) + hit * an) / ad

/-- The EMA is a convex combination: it keeps a probability within [0, 1]. -/
theorem ema_stays_le_one (S an ad p hit : Nat) (hp : p ≤ S) (hh : hit ≤ S) (ha : an ≤ ad) :
    emaStep an ad p hit ≤ S := by
  unfold emaStep
  have h1 : p * (ad - an) ≤ S * (ad - an) := Nat.mul_le_mul_right _ hp
  have h2 : hit * an ≤ S * an := Nat.mul_le_mul_right _ hh
  have h3 : S * (ad - an) + S * an = ad * S := by
    rw [← Nat.mul_add, Nat.mul_comm]
    congr 1
    omega
  apply Nat.div_le_of_le_mul
  omega

/-- An observation equal to the current estimate leaves it unchanged (EMA fixed point). -/
theorem ema_fixed_point (an ad p : Nat) (ha : an ≤ ad) (hd : 0 < ad) : emaStep an ad p p = p := by
  unfold emaStep
  have h : p * (ad - an) + p * an = p * ad := by
    rw [← Nat.mul_add]
    congr 1
    omega
  rw [h]
  exact Nat.mul_div_cancel p hd

/-- A hit never lowers the estimate, a miss never raises it. -/
theorem ema_direction (an ad p S : Nat) (hp : p ≤ S) (ha : an ≤ ad) (hd : 0 < ad) :
    emaStep an ad p 0 ≤ p ∧ p ≤ emaStep an ad p S := by
  constructor
  · unfold emaStep
    apply Nat.div_le_of_le_mul
    have : p * (ad - an) ≤ p * ad := Nat.mul_le_mul_left p (by omega)
    rw [Nat.mul_comm ad p]
    omega
  · unfold emaStep
    apply (Nat.le_div_iff_mul_le hd).mpr
    have h1 : p * an ≤ S * an := Nat.mul_le_mul_right _ hp
    have h2 : p * (ad - an) + p * an = p * ad := by
      rw [← Nat.mul_add]
      congr 1
      omega
    omega

end Strata.Spec
