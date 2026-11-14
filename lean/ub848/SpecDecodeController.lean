/-!
# MODULE: spec_decode_controller
SOURCES: src/spec/controller.cpp
The speculative-decoding controller: a cost model for a verify step of `k` drafts, the expected
number of tokens a draft of `k` commits, a chooser that picks the draft source and length with the
best tokens per millisecond (or none), and the online update of acceptance rates.

## Source semantics
* `step_ms` indexes its per-window tables by `n = clamp(k + 1, 1, K_MAX + 1)`, so `n - 1` is always
  a valid index into tables of `K_MAX + 1` entries.
* `expected(p, k)` is `1 + Σ_{i<k} Π_{j≤i} p_j` (or `p_0^(i+1)` for lookup): the verify pass always
  commits one token, and each further token needs every earlier draft accepted.  It is ≥ 1 and
  grows with `k`.
* `choose` starts from the no-draft baseline, keeps any strictly better candidate, and falls back to
  no draft unless the best beats the baseline by the factor `1 + min_gain`.
* `observe` clamps `accepted` to `[0, k]` and updates the first `min(k, accepted + 1)` positions:
  `accepted` hits and at most one miss.
-/

namespace Ref.Spec

-- § cost-table index

/-- `std::clamp(k + 1, 1, K_MAX + 1)`. -/
def clampIdx (kmax : Int) (k : Int) : Int := max 1 (min (k + 1) (kmax + 1))

/-- Whatever `k` the caller passes, the table index `n - 1` is in `[0, K_MAX]`. -/
theorem clampIdx_valid (kmax k : Int) (h : 0 ≤ kmax) :
    0 ≤ clampIdx kmax k - 1 ∧ clampIdx kmax k - 1 ≤ kmax := by
  unfold clampIdx; omega

/-- In range, the clamp is the identity `k + 1`. -/
theorem clampIdx_id (kmax k : Int) (h0 : 0 ≤ k) (h1 : k ≤ kmax) : clampIdx kmax k = k + 1 := by
  unfold clampIdx; omega

-- § expected tokens per verify step

/-- The running product after `i` drafts; `same` uses `p 0` throughout (the lookup source). -/
def runProd (p : Nat → Nat) (same : Bool) : Nat → Nat
  | 0 => 1
  | i + 1 => runProd p same i * (if same then p 0 else p i)

/-- Expected committed tokens for `k` drafts. -/
def expected (p : Nat → Nat) (same : Bool) : Nat → Nat
  | 0 => 1
  | k + 1 => expected p same k + runProd p same (k + 1)

/-- The verify pass always commits at least one token. -/
theorem expected_ge_one (p : Nat → Nat) (same : Bool) (k : Nat) : 1 ≤ expected p same k := by
  induction k with
  | zero => simp [expected]
  | succ k ih => simp only [expected]; omega

/-- Drafting more never lowers the expectation (nonnegative acceptance rates). -/
theorem expected_mono (p : Nat → Nat) (same : Bool) (k : Nat) :
    expected p same k ≤ expected p same (k + 1) := by
  simp only [expected]; omega

/-- With certain acceptance, `k` drafts commit `k + 1` tokens. -/
theorem expected_certain (same : Bool) (k : Nat) : expected (fun _ => 1) same k = k + 1 := by
  have hr : ∀ i, runProd (fun _ => 1) same i = 1 := by
    intro i
    induction i with
    | zero => rfl
    | succ i ih => simp [runProd, ih]
  induction k with
  | zero => rfl
  | succ k ih => simp only [expected, ih, hr]

/-- With zero acceptance only the verify pass's own token is committed. -/
theorem expected_reject (same : Bool) (k : Nat) : expected (fun _ => 0) same k = 1 := by
  induction k with
  | zero => rfl
  | succ k ih => simp [expected, ih, runProd]

-- § choosing

/-- A candidate: source tag (0 = none), draft length, rate (tokens per ms, any monotone scale). -/
structure Cand where
  src : Nat
  k : Nat
  rate : Nat
  deriving DecidableEq, Repr

/-- `consider`: keep a candidate only if it is strictly better. -/
def pick (best : Cand) : List Cand → Cand
  | [] => best
  | c :: cs => pick (if c.rate > best.rate then c else best) cs

/-- The kept candidate is never worse than the starting baseline. -/
theorem pick_ge (best : Cand) (cs : List Cand) : best.rate ≤ (pick best cs).rate := by
  induction cs generalizing best with
  | nil => exact Nat.le_refl _
  | cons c cs ih =>
    simp only [pick]
    split
    · next h => have := ih c; omega
    · exact ih best

/-- The min-gain gate, in percent: fall back to no draft unless `best ≥ base * (1 + g/100)`. -/
def gate (base g : Nat) (best : Cand) : Cand :=
  if best.src ≠ 0 ∧ best.rate * 100 < base * (100 + g) then ⟨0, 0, base⟩ else best

/-- `choose`: start from the baseline (`src = 0`), consider the candidates, apply the gate. -/
def choose (base g : Nat) (cs : List Cand) : Cand := gate base g (pick ⟨0, 0, base⟩ cs)

/-- **Contract of `choose`**: either no draft at the baseline rate, or a choice that beats the
baseline by the required margin. -/
theorem choose_spec (base g : Nat) (cs : List Cand) :
    let r := choose base g cs
    (r.src = 0 ∧ base ≤ r.rate) ∨ base * (100 + g) ≤ r.rate * 100 := by
  simp only [choose, gate]
  have hge := pick_ge ⟨0, 0, base⟩ cs
  simp only at hge
  split
  · left; simp
  · next h =>
    by_cases hs : (pick ⟨0, 0, base⟩ cs).src = 0
    · left; exact ⟨hs, hge⟩
    · right; omega

-- § observing

/-- Clamp of the accepted count. -/
def clampAcc (k acc : Int) : Int := max 0 (min acc k)

/-- Positions observed in a round: the accepted ones plus the first rejected one, if drafted. -/
def seen (k acc : Int) : Int := min k (clampAcc k acc + 1)

/-- The update touches only drafted positions, records `accepted` hits and at most one miss. -/
theorem observe_counts (k acc : Int) (hk : 1 ≤ k) :
    0 ≤ clampAcc k acc ∧ clampAcc k acc ≤ seen k acc ∧ seen k acc ≤ k ∧
    seen k acc - clampAcc k acc ≤ 1 := by
  unfold seen clampAcc; omega

/-- A fully accepted window (`accepted = k`) observes no miss, which is when the deeper positions
are pulled toward the deepest observed rate. -/
theorem full_accept_no_miss (k : Int) (hk : 1 ≤ k) : seen k k - clampAcc k k = 0 := by
  unfold seen clampAcc; omega

end Ref.Spec

