/-!
# MODULE: spec_controller
The speculative-decoding controller: choose drafter and draft length maximizing expected tokens per ms,
keep k = 0 unless the gain beats a threshold, and learn per-position acceptance with an EMA.

## Source semantics
Expected committed tokens is `1 + sum of running acceptance products` (the verify pass always commits one
token). Arrays are indexed by `n = k+1` clamped to `[1, K_MAX+1]`. Lookup acceptance is bucketed by match
length. The EMA is a convex combination so probabilities stay in range.

## Form template identifiers
- `K_MAX` <- spec::K_MAX
- `stepIdx` <- `clamp(k+1,1,K_MAX+1) - 1` in CostModel::step_ms
- `bucket` <- Controller::bucket
- `tail`/`expect` <- Controller::expected
- `pickBest` <- the strict-improvement scan in Controller::choose
- `emaStep` <- the `p += ema * (hit - p)` update (fixed-point form)
-/

namespace Strata.SpecController

def K_MAX : Nat := 8

-- § cost-model indexing

/-- Table index used for an `n = k+1` token verify pass. -/
def stepIdx (k : Nat) : Nat := (max 1 (min (k + 1) (K_MAX + 1))) - 1

/-- The table index is always inside the 9-entry cost tables. -/
theorem stepIdx_lt (k : Nat) : stepIdx k < K_MAX + 1 := by
  unfold stepIdx K_MAX
  omega

/-- For `k ≤ K_MAX` the index is exactly `k` (no clamping artefacts). -/
theorem stepIdx_id (k : Nat) (hk : k ≤ K_MAX) : stepIdx k = k := by
  unfold stepIdx
  unfold K_MAX at *
  omega

-- § lookup bucket

/-- Match-length bucket of the lookup acceptance table. -/
def bucket (m : Nat) : Nat := if m ≥ 16 then 3 else if m ≥ 8 then 2 else if m ≥ 5 then 1 else 0

/-- Buckets are 0..3 (the `lookup_q_` array has 4 entries). -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  repeat' split
  all_goals omega

/-- Longer matches never fall into a lower bucket. -/
theorem bucket_mono (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  repeat' split
  all_goals omega

-- § expected tokens

/-- Sum of running acceptance products for `k` drafted positions. -/
def tail (p : Nat → Nat) (i : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, run => run * p i + tail p (i + 1) k (run * p i)

/-- Expected committed tokens: the verified token plus the accepted-prefix sum. -/
def expect (p : Nat → Nat) (k : Nat) : Nat := 1 + tail p 0 k 1

/-- The verify pass always commits at least one token. -/
theorem expect_ge_one (p : Nat → Nat) (k : Nat) : 1 ≤ expect p k := by
  unfold expect
  omega

/-- Drafting more positions never lowers the expected token count. -/
theorem tail_mono (p : Nat → Nat) : ∀ k i run, tail p i k run ≤ tail p i (k + 1) run := by
  intro k
  induction k with
  | zero =>
    intro i run
    simp [tail]
  | succ k ih =>
    intro i run
    simp only [tail]
    exact Nat.add_le_add_left (ih _ _) _

/-- With certain acceptance every drafted token is committed: `k * run`. -/
theorem tail_all_one : ∀ k i run, tail (fun _ => 1) i k run = k * run := by
  intro k
  induction k with
  | zero =>
    intro i run
    simp [tail]
  | succ k ih =>
    intro i run
    simp only [tail, Nat.mul_one]
    rw [ih, Nat.add_mul, Nat.one_mul]
    omega

/-- Certain acceptance commits `k + 1` tokens. -/
theorem expect_all_one (k : Nat) : expect (fun _ => 1) k = k + 1 := by
  unfold expect
  rw [tail_all_one]
  omega

-- § choice scan

/-- Best rate over candidates with strict improvement, starting from the plain-decoding baseline. -/
def pickBest (init : Nat) (cands : List Nat) : Nat := cands.foldl max init

/-- The chosen rate is never below the baseline. -/
theorem pickBest_ge_init (cands : List Nat) : ∀ init, init ≤ pickBest init cands := by
  induction cands with
  | nil =>
    intro init
    exact Nat.le_refl _
  | cons c cs ih =>
    intro init
    exact Nat.le_trans (Nat.le_max_left init c) (ih (max init c))

/-- The chosen rate dominates every candidate considered. -/
theorem pickBest_ge_mem (cands : List Nat) : ∀ init x, x ∈ cands → x ≤ pickBest init cands := by
  induction cands with
  | nil =>
    intro init x hx
    simp at hx
  | cons c cs ih =>
    intro init x hx
    rcases List.mem_cons.mp hx with h | h
    · subst h
      exact Nat.le_trans (Nat.le_max_right init x) (pickBest_ge_init cs (max init x))
    · exact ih (max init c) x h

/-- Lookup draft length is capped by both the proposal and `K_MAX`. -/
theorem lookup_k_le (avail : Nat) : min avail K_MAX ≤ K_MAX := by
  unfold K_MAX
  omega

-- § learning

/-- Number of positions updated by `observe`: those accepted plus the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- `observe` never touches a position beyond the drafted window. -/
theorem seen_le (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- Every touched MTP position is a valid index of the 8-entry array. -/
theorem seen_index_lt (k accepted i : Nat) (hk : k ≤ K_MAX) (hi : i < seen k accepted) : i < K_MAX := by
  unfold seen at hi
  unfold K_MAX at *
  omega

/-- Fixed-point EMA step: `((S-e)*x + e*h) / S`. -/
def emaStep (S e x h : Nat) : Nat := ((S - e) * x + e * h) / S

/-- The EMA is a convex combination: an in-range estimate stays in range. -/
theorem emaStep_le (S e x h : Nat) (hS : 0 < S) (he : e ≤ S) (hx : x ≤ S) (hh : h ≤ S) :
    emaStep S e x h ≤ S := by
  unfold emaStep
  have h1 : (S - e) * x ≤ (S - e) * S := Nat.mul_le_mul_left _ hx
  have h2 : e * h ≤ e * S := Nat.mul_le_mul_left _ hh
  have h3 : (S - e) * S + e * S = S * S := by
    rw [← Nat.add_mul, Nat.sub_add_cancel he]
  apply Nat.div_le_of_le_mul
  omega

end Strata.SpecController
