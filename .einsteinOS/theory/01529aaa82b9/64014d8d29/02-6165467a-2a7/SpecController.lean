/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp, include/strata/spec/controller.hpp
Speculative-decoding controller: each step it maximizes `E[tokens] / T(step)` over
`none | lookup(k) | MTP(k)`, keeps `k = 0` unless the best beats plain decoding by `min_gain`,
and learns per-position acceptance with an EMA.

## Source semantics
* The verify size is `n = clamp(k+1, 1, K_MAX+1)` and indexes arrays of length `K_MAX+1` at `n-1`.
* Lookup acceptance is bucketed by match length: `>=16 -> 3`, `>=8 -> 2`, `>=5 -> 1`, else `0`.
* `expected = 1 + sum_i prod_{j<=i} p_j` (conditional-same uses `p[0]` throughout).
* The choice is the strict argmax of the rate, starting from the plain-decode baseline.
* The EMA update keeps acceptance in `[0,1]`.

## Form template identifiers
* `CostModel::step_ms` index `n-1`  -> `clampN`
* `Controller::bucket`              -> `bucket`
* `Controller::expected`            -> `runs`, `expectedFx` (fixed point, scale `S`)
* `Controller::choose` best-so-far  -> `pickBest`
* `Controller::observe` EMA         -> `emaUp`, `emaDown`
-/

namespace Strata.Spec

-- § window clamp

/-- `K_MAX`. -/
def K_MAX : Nat := 8

/-- `n = clamp(k + 1, 1, K_MAX + 1)`. -/
def clampN (k : Nat) : Nat := min (max (k + 1) 1) (K_MAX + 1)

/-- The verify size is at least one token (the pass always commits one). -/
theorem clampN_ge_one (k : Nat) : 1 ≤ clampN k := by
  unfold clampN K_MAX
  omega

/-- The array index `n - 1` is inside the `K_MAX + 1` ratio tables. -/
theorem clampN_idx_lt (k : Nat) : clampN k - 1 < K_MAX + 1 := by
  unfold clampN K_MAX
  omega

-- § match-length bucket

/-- Lookup acceptance bucket by match length. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- Buckets index the 4-entry lookup table. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  split_ifs <;> omega

/-- Longer matches never land in a lower bucket. -/
theorem bucket_mono (m n : Nat) (h : m ≤ n) : bucket m ≤ bucket n := by
  unfold bucket
  split_ifs <;> omega

-- § expected tokens (fixed point)

/-- Running products `run_{i+1} = run_i * p_i / S` for probabilities scaled by `S`. -/
def runs (S : Nat) : Nat → List Nat → List Nat
  | _, [] => []
  | run, p :: ps => (run * p / S) :: runs S (run * p / S) ps

/-- Expected committed tokens, scaled by `S`: `S + sum runs`. -/
def expectedFx (S : Nat) (ps : List Nat) : Nat := S + (runs S S ps).sum

/-- One run has one product per draft position. -/
theorem runs_length (S : Nat) : ∀ (ps : List Nat) (run : Nat), (runs S run ps).length = ps.length := by
  intro ps
  induction ps with
  | nil => intro run; simp [runs]
  | cons p ps ih => intro run; simp [runs, ih]

/-- If every probability is at most `S` and the start is, every running product is at most `S`. -/
theorem runs_le (S : Nat) : ∀ (ps : List Nat) (run : Nat), run ≤ S → (∀ p ∈ ps, p ≤ S) →
    ∀ x ∈ runs S run ps, x ≤ S := by
  intro ps
  induction ps with
  | nil => intro run _ _ x hx; simp [runs] at hx
  | cons p ps ih =>
    intro run hr hp x hx
    have hp0 : p ≤ S := hp p (List.mem_cons_self)
    have hmul : run * p ≤ S * S := Nat.mul_le_mul hr hp0
    have hdiv : run * p / S ≤ S := Nat.div_le_of_le_mul hmul
    rw [runs] at hx
    rcases List.mem_cons.mp hx with h | hx'
    · rw [h]; exact hdiv
    · exact ih (run * p / S) hdiv (fun q hq => hp q (List.mem_cons_of_mem _ hq)) x hx'

/-- A list of numbers each at most `S` sums to at most `length * S`. -/
theorem sum_le_len_mul (S : Nat) : ∀ (l : List Nat), (∀ x ∈ l, x ≤ S) → l.sum ≤ l.length * S := by
  intro l
  induction l with
  | nil => intro _; simp
  | cons a l ih =>
    intro h
    have h1 : a ≤ S := h a (List.mem_cons_self)
    have h2 := ih (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp only [List.sum_cons, List.length_cons]
    have h3 : (l.length + 1) * S = l.length * S + S := by rw [Nat.succ_mul]
    omega

/-- The expected token count never exceeds `k + 1` (all drafts accepted), in scaled units. -/
theorem expectedFx_le (S : Nat) (ps : List Nat) (hp : ∀ p ∈ ps, p ≤ S) :
    expectedFx S ps ≤ (ps.length + 1) * S := by
  unfold expectedFx
  have h1 := sum_le_len_mul S (runs S S ps) (runs_le S ps S (Nat.le_refl S) hp)
  rw [runs_length] at h1
  have h3 : (ps.length + 1) * S = ps.length * S + S := by rw [Nat.succ_mul]
  omega

/-- The expected token count is at least one token (scaled `S`). -/
theorem expectedFx_ge (S : Nat) (ps : List Nat) : S ≤ expectedFx S ps := by
  unfold expectedFx
  omega

-- § argmax with strict improvement

/-- Strict argmax fold: a candidate replaces the best only when strictly greater. -/
def pickBest : Nat → List Nat → Nat
  | b, [] => b
  | b, c :: cs => pickBest (if b < c then c else b) cs

/-- The chosen rate is never below the plain-decode baseline (the start value). -/
theorem pickBest_ge_init : ∀ (cs : List Nat) (b : Nat), b ≤ pickBest b cs := by
  intro cs
  induction cs with
  | nil => intro b; simp [pickBest]
  | cons c cs ih =>
    intro b
    rw [pickBest]
    have h1 := ih (if b < c then c else b)
    have h2 : b ≤ (if b < c then c else b) := by split_ifs <;> omega
    omega

/-- The chosen rate dominates every candidate. -/
theorem pickBest_ge_mem : ∀ (cs : List Nat) (b : Nat), ∀ x ∈ cs, x ≤ pickBest b cs := by
  intro cs
  induction cs with
  | nil => intro b x hx; simp at hx
  | cons c cs ih =>
    intro b x hx
    rw [pickBest]
    rcases List.mem_cons.mp hx with h | hx'
    · have h2 : c ≤ (if b < c then c else b) := by split_ifs <;> omega
      have h3 := pickBest_ge_init cs (if b < c then c else b)
      rw [h]
      omega
    · exact ih _ x hx'

/-- The min-gain gate: keep a drafted choice only if `baseline * (den + num) <= best * den`. -/
def gate (best baseline num den : Nat) : Bool := decide (baseline * (den + num) ≤ best * den)

/-- A choice that passes the gate beats the baseline by the factor `(den+num)/den`. -/
theorem gate_sound (best baseline num den : Nat) (h : gate best baseline num den = true) :
    baseline * (den + num) ≤ best * den := by
  unfold gate at h
  exact of_decide_eq_true h

-- § EMA acceptance update

/-- EMA move toward 1: `p + (S - p) * a / A`. -/
def emaUp (S p a A : Nat) : Nat := p + (S - p) * a / A

/-- EMA move toward 0: `p - p * a / A`. -/
def emaDown (p a A : Nat) : Nat := p - p * a / A

/-- Moving toward a hit keeps the probability within `[0, S]` when the rate `a <= A`. -/
theorem emaUp_le (S p a A : Nat) (hp : p ≤ S) (ha : a ≤ A) : emaUp S p a A ≤ S := by
  unfold emaUp
  have hmul : (S - p) * a ≤ A * (S - p) := by
    rw [Nat.mul_comm A]
    exact Nat.mul_le_mul_left _ ha
  have hdiv : (S - p) * a / A ≤ S - p := Nat.div_le_of_le_mul hmul
  omega

/-- A hit never lowers the probability. -/
theorem emaUp_ge (S p a A : Nat) : p ≤ emaUp S p a A := by
  unfold emaUp
  exact Nat.le_add_right _ _

/-- Moving toward a miss never raises the probability and keeps the bound. -/
theorem emaDown_le (p a A S : Nat) (hp : p ≤ S) : emaDown p a A ≤ S := by
  unfold emaDown
  exact Nat.le_trans (Nat.sub_le _ _) hp

-- § observe bookkeeping

/-- Positions updated by `observe`: those up to and including the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- `observe` never touches more than the `k` drafted positions. -/
theorem seen_le_k (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- The deepest-tested position index `k - 1` exists when the window is non-empty and below `K_MAX`. -/
theorem deepest_idx_lt (k : Nat) (h1 : 1 ≤ k) (h2 : k < K_MAX) : k - 1 < K_MAX := by
  unfold K_MAX at *
  omega

end Strata.Spec
