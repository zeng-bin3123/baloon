/-!
# MODULE: speculative_verify_window
SOURCES: src/program/generate.cpp
Plan v0.3 P6 speculative decoding as the driver performs it: a window `[x, drafts...]` of size `T` is
verified in one pass, the longest agreeing draft prefix is accepted, `a + 1` tokens are emitted and the
position advances by `a + 1`.  Also the `--spec-min-p` window sizing loop.

## Source semantics
* `a` is the largest count with `window[i+1] == outv[i]` for all `i < a`, capped at `T - 1`.
* The round emits `outv[0..a]` (`a + 1` tokens): every round makes progress, at most `T` tokens.
* If the verifier's output `outv[j]` is the greedy prediction after `history ++ window[0..j]`, then every
  emitted token is the greedy prediction after `history ++ x ++ (emitted so far)`: speculation is exact.
* With `spec_min_p > 0`, `T` starts at 1 and grows while `T < S` and the draft probability clears the
  threshold; the first window is always `T = 1`.
-/

namespace Ref.Generate.Speculative

-- § acceptance

/-- Accepted drafts: the length of the longest common prefix of the drafts and the verifier output. -/
def acceptLen : List Nat → List Nat → Nat
  | d :: rest, o :: os => if d = o then acceptLen ds os + 1 else 0
  | _, _ => 0

/-- No more drafts are accepted than were proposed (`a ≤ T - 1`). -/
theorem acceptLen_le_drafts : ∀ (ds os : List Nat), acceptLen ds os ≤ ds.length
  | [], _ => by simp only [acceptLen]
  | _ :: _, [] => by simp only [acceptLen]
  | d :: rest, o :: os => by
    have ih := acceptLen_le_drafts ds os
    by_cases h : d = o
    · simp only [acceptLen, h]; exact (by omega)
    · simp only [acceptLen, h]

/-- No more drafts are accepted than the verifier produced outputs for. -/
theorem acceptLen_le_outputs : ∀ (ds os : List Nat), acceptLen ds os ≤ os.length
  | [], _ => by simp only [acceptLen]
  | _ :: _, [] => by simp only [acceptLen]
  | d :: rest, o :: os => by
    have ih := acceptLen_le_outputs ds os
    by_cases h : d = o
    · simp only [acceptLen, h]; exact (by omega)
    · simp only [acceptLen, h]

/-- The accepted drafts are literally the verifier's first `a` outputs. -/
theorem take_acceptLen : ∀ (ds os : List Nat), ds.take (acceptLen ds os) = os.take (acceptLen ds os)
  | [], _ => by simp only [acceptLen]
  | _ :: _, [] => by simp only [acceptLen]
  | d :: rest, o :: os => by
    by_cases h : d = o
    · subst h; simp only [acceptLen, take_acceptLen ds os]
    · simp only [acceptLen, h]

/-- Any shorter prefix of the accepted run agrees as well. -/
theorem take_eq_of_le {ws gs : List Nat} {i : Nat} (hi : i ≤ acceptLen ws gs) :
    ws.take i = gs.take i := by
  have h := congrArg (List.take j) (take_acceptLen ds os)
  simp only [List.take_take] at h
  rw [Nat.min_eq_left hj] at h
  exact h

-- § emission and progress

/-- Tokens emitted by a round: the `a + 1` first verifier outputs. -/
def emitted (ws gs : List Nat) : List Nat := gs.take (acceptLen ws gs + 1)

/-- A round emits `a + 1` tokens when the verifier returned a full window. -/
theorem emitted_length {ds os : List Nat} (hlen : ds.length + 1 ≤ os.length) :
    (emitted ws gs).length = acceptLen ds os + 1 := by
  have ha := acceptLen_le_drafts ds os
  simp only [emitted, List.length_take]; exact (by omega)

/-- Every round makes progress and never more than the window: `1 ≤ emitted ≤ T`. -/
theorem emitted_bounds {ds os : List Nat} (hlen : os.length = ds.length + 1) :
    1 ≤ (emitted ws gs).length ∧ (emitted ws gs).length ≤ ds.length + 1 := by
  have h := emitted_length (ds := ds) (os := os) (by omega)
  have ha := acceptLen_le_drafts ds os; exact (by omega)

/-- The position advance `p += a + 1` after `r` rounds is at least `r`: decoding terminates. -/
theorem rounds_progress (advances : List Nat) (hpos : ∀ a ∈ advances, 1 ≤ a) :
    advances.length ≤ advances.foldl (· + ·) 0 := by
  have key : ∀ (xs : List Nat) (acc : Nat), (∀ a ∈ xs, 1 ≤ a) → acc + xs.length ≤ xs.foldl (· + ·) acc := by
    intro xs
    induction xs with
    | nil => intro acc _; simp only
    | cons x xs ih =>
      intro acc h
      have hx := h x (List.mem_cons_self ..)
      have := ih (acc + x) (fun a ha => h a (List.mem_cons_of_mem _ ha))
      simp only [List.foldl_cons, List.length_cons]; exact (by omega)
  have := key advances 0 hpos; exact (by omega)

-- § exactness

/-- The verifier contract: output `j` is the greedy token after the history and window cells `0..j`. -/
def VerifyOk (g : List Nat → Nat) (hist : List Nat) (x : Nat) (ws os : List Nat) : Prop :=
  ∀ j, j < os.length → os[j]? = some (g (hist ++ x :: ds.take j))

/-- Every emitted token equals what plain greedy decoding would emit at that point. -/
theorem emitted_is_greedy {g : List Nat → Nat} {hist : List Nat} {x : Nat} {ws gs : List Nat}
    (hv : VerifyOk g hist x ws gs) {j : Nat} (hj : j ≤ acceptLen ds os) (hjl : j < os.length) :
    os[j]? = some (g (hist ++ x :: os.take j)) := by
  have h := hv j hjl
  rw [take_eq_of_le hj] at h
  exact h

/-- A wrong draft (the `--spec-corrupt` fault) stops acceptance right there. -/
theorem wrong_draft_stops {d o : Nat} (ds os : List Nat) (h : d ≠ o) :
    acceptLen (d :: ds) (o :: os) = 0 := by simp only [acceptLen, h]

-- § window sizing

/-- The `--spec-min-p` loop: grow `T` while `T < S` and draft `T - 1` clears the threshold. -/
def growFrom (Cap : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, T => T
  | fuel + 1, T => if T < Cap ∧ ok (T - 1) = true then growFrom S ok fuel (T + 1) else T

/-- Growth never shrinks the window and never passes `S`. -/
theorem growFrom_bounds (S : Nat) (ok : Nat → Bool) :
    ∀ fuel T, T ≤ S → T ≤ growFrom S ok fuel T ∧ growFrom S ok fuel T ≤ S := by
  intro fuel
  induction fuel with
  | zero => intro T h; simp only [growFrom]; exact h
  | succ fuel ih =>
    intro T h
    simp only [growFrom]
    split
    · rename_i hc; have := ih (T + 1) (by omega); exact (by omega)
    · exact (by omega)

/-- The verify window: start at 1, grow up to `S`. -/
def windowSize (Cap : Nat) (ok : Nat → Bool) : Nat := growFrom Cap ok Cap 1

/-- The window always holds the last token and at most `S - 1` drafts. -/
theorem windowSize_bounds {S : Nat} (hCap : 1 ≤ S) (ok : Nat → Bool) :
    1 ≤ windowSize S ok ∧ windowSize S ok ≤ S :=
  growFrom_bounds S ok S 1 hCap

end Ref.Generate.Speculative

