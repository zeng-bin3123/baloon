import Mathlib

/-!
# MODULE: verify_window_commit
The speculative verify window (`verify.hpp`): a window of `T` tokens (the last accepted token plus `T - 1` drafts)
is run, and `commit(n_keep)` makes the first `n_keep` tokens permanent. Greedy-equivalence is stated here as:
a draft is accepted exactly while it matches what greedy decode produces, and committing a prefix then
continuing equals plain decoding of the whole sequence.

## Source semantics
* The window's argmax after token `t` is plain greedy's next token, so the accepted length is the longest prefix
  of the drafts that agrees with the greedy outputs; the verified token itself is always kept, so
  `1 <= n_keep <= T`.
* `commit(n_keep)` replays the first `n_keep` tokens through the state update (GDN conv history and recurrence,
  indexer tail, PLE history); decoding the remaining tokens afterwards reproduces decoding all of them.
* Committing nothing leaves the state unchanged.
* Window size is bounded by `max_t <= 8`.

## Form template identifiers
* `commitState`  <- `Verifier::commit` (replaying `n_keep` tokens through the state update `step`)
* `acceptLen`    <- the longest accepted draft prefix against `out[t]`
* `n_keep`       <- `acceptLen + 1`
-/

namespace Strata.Verify

-- § commit replays a prefix

/-- The state after committing the first `n` tokens of a window: replay them through the state update. -/
def commitState {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) (n : Nat) : σ :=
  (toks.take n).foldl step s

/-- Committing a prefix and then decoding the remaining tokens equals decoding the whole window plainly. -/
theorem commit_then_rest {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) (n : Nat) :
    (toks.drop n).foldl step (commitState step s toks n) = toks.foldl step s := by
  unfold commitState
  rw [← List.foldl_append, List.take_append_drop]

/-- Committing zero tokens leaves the state unchanged. -/
theorem commit_zero {σ : Type} (step : σ → Nat → σ) (s : σ) (toks : List Nat) :
    commitState step s toks 0 = s := by
  simp [commitState]

/-- Committing one more token applies exactly one more state update. -/
theorem commit_succ {σ : Type} (step : σ → Nat → σ) (s : σ) (t : Nat) (rest : List Nat) (n : Nat) :
    commitState step s (t :: rest) (n + 1) = commitState step (step s t) rest n := by
  simp [commitState]

-- § acceptance

/-- Length of the longest prefix of the drafts that matches the greedy outputs. -/
def acceptLen : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | d :: ds, g :: gs => if d = g then acceptLen ds gs + 1 else 0

/-- Never more drafts are accepted than were proposed. -/
theorem acceptLen_le (d g : List Nat) : acceptLen d g ≤ d.length := by
  induction d generalizing g with
  | nil => cases g <;> simp [acceptLen]
  | cons a ds ih =>
    cases g with
    | nil => simp [acceptLen]
    | cons b gs =>
      have := ih gs
      by_cases h : a = b
      · simp [acceptLen, h]
        omega
      · simp [acceptLen, h]

/-- If greedy decode reproduces every draft, all drafts are accepted. -/
theorem acceptLen_self (d r : List Nat) : acceptLen d (d ++ r) = d.length := by
  induction d with
  | nil => simp [acceptLen]
  | cons a ds ih => simp [acceptLen, ih]

/-- A first mismatch rejects everything after it. -/
theorem acceptLen_mismatch (a b : Nat) (ds gs : List Nat) (h : a ≠ b) :
    acceptLen (a :: ds) (b :: gs) = 0 := by
  simp [acceptLen, h]

/-- `n_keep = acceptLen + 1` is between 1 and the window size `T`: the verified token is always kept. -/
theorem keep_bounds (T : Nat) (drafts greedy : List Nat) (hT : drafts.length + 1 = T) :
    1 ≤ acceptLen drafts greedy + 1 ∧ acceptLen drafts greedy + 1 ≤ T := by
  have := acceptLen_le drafts greedy
  omega

/-- A window of at most `max_t <= 8` tokens has at most 7 drafts. -/
theorem drafts_bound (maxT T : Nat) (hmax : maxT ≤ 8) (hT : T ≤ maxT) (hpos : 1 ≤ T) : T - 1 ≤ 7 := by
  omega

end Strata.Verify
