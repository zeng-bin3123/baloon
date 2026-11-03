import Mathlib

/-!
# MODULE: speculative_verify_window
The speculative verify window: T tokens (one accepted + T-1 drafts) at consecutive positions processed in one
graph, the greedy acceptance rule, `commit(n_keep)`, and the PLE predecessor window advance.

## Source semantics
* `1 <= T <= 8` (`exec_[9]`, `last_tokens_[8]`).
* Token t's argmax is what greedy decode would emit after token t; draft t+1 is accepted iff it equals that
  argmax, so `n_keep = 1 + (length of the leading run of matches)`.
* `commit(n_keep)` requires `1 <= n_keep <= T`; positions `pos0 .. pos0 + n_keep - 1` become permanent.
* `ple_prev` holds the last two tokens, oldest first; advancing by the kept tokens keeps exactly two.

## Form template identifiers
* `leadingMatches`  -> `ACCEPTED_DRAFT_RUN`
* `nKeep`           -> `COMMIT_COUNT`
* `advancePrev`     -> `PLE_PREV_ADVANCE`
-/

namespace StrataVerify

-- § window size

/-- Window sizes accepted by the verifier. -/
def windowOk (T : Nat) : Prop := 1 ≤ T ∧ T ≤ 8

/-- Every legal window indexes the 9-entry graph array and the 8-entry token array in range. -/
theorem window_index_ok (T : Nat) (h : windowOk T) : T < 9 ∧ T ≤ 8 := by
  unfold windowOk at h; omega

-- § greedy acceptance

/-- Length of the leading run where draft t equals the argmax produced after token t-1. -/
def leadingMatches : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then 1 + leadingMatches ds os else 0
  | _, _ => 0

/-- The run cannot exceed the number of drafts. -/
theorem leadingMatches_le : ∀ (d o : List Nat), leadingMatches d o ≤ d.length := by
  intro d
  induction d with
  | nil => intro o; simp [leadingMatches]
  | cons x xs ih =>
    intro o
    cases o with
    | nil => simp [leadingMatches]
    | cons y ys =>
      have := ih ys
      simp only [leadingMatches, List.length_cons]
      split_ifs <;> omega

/-- If every draft equals the model's own argmax, all of them are accepted. -/
theorem leadingMatches_self : ∀ l : List Nat, leadingMatches l l = l.length := by
  intro l
  induction l with
  | nil => simp [leadingMatches]
  | cons x xs ih => simp [leadingMatches, ih]; omega

/-- A first draft that differs from the argmax is rejected at once. -/
theorem leadingMatches_first_miss (d o : Nat) (ds os : List Nat) (h : d ≠ o) :
    leadingMatches (d :: ds) (o :: os) = 0 := by
  simp [leadingMatches, h]

/-- Tokens kept: the accepted token plus the accepted draft run. -/
def nKeep (drafts outs : List Nat) : Nat := 1 + leadingMatches drafts outs

/-- `commit` is always handed a count in `1 .. T` where `T = drafts + 1`. -/
theorem nKeep_bounds (drafts outs : List Nat) :
    1 ≤ nKeep drafts outs ∧ nKeep drafts outs ≤ drafts.length + 1 := by
  have := leadingMatches_le drafts outs
  unfold nKeep
  omega

/-- With no drafts exactly one token is committed (plain greedy decode). -/
theorem nKeep_no_drafts (outs : List Nat) : nKeep [] outs = 1 := by
  simp [nKeep, leadingMatches]

-- § commit bookkeeping

/-- The position after the committed prefix. -/
def nextPos (pos0 nKeep : Nat) : Nat := pos0 + nKeep

/-- Committing one more token moves the next position by exactly one. -/
theorem nextPos_succ (pos0 n : Nat) : nextPos pos0 (n + 1) = nextPos pos0 n + 1 := by
  unfold nextPos; omega

/-- The last two tokens after appending the kept tokens to the old window. -/
def advancePrev (prev kept : List Int) : List Int :=
  (prev ++ kept).drop ((prev ++ kept).length - 2)

/-- A two-token predecessor window stays two tokens long. -/
theorem advancePrev_length (prev kept : List Int) (hp : prev.length = 2) :
    (advancePrev prev kept).length = 2 := by
  unfold advancePrev
  rw [List.length_drop, List.length_append, hp]
  omega

end StrataVerify
