/-!
# MODULE: speculative_verify_window
SOURCES: include/strata/core/verify.hpp
The speculative verify window: T tokens (the last accepted one plus T-1 drafts) go through all 48 layers in
one graph; token t's argmax equals what greedy decode would emit, so a draft is accepted exactly when it
matches. `commit(n_keep)` keeps the first `n_keep` tokens.

## Source semantics
* `out[t]` is the argmax after token `t`; draft `t+1` is accepted iff `out[t] = draft[t+1]`, and acceptance stops
  at the first mismatch.
* `n_keep = 1 + accepted` is in `1..T`.
* `max_t <= kVerifyMaxT` and the per-`T` graph array has 9 entries.
* Flag slots are indexed by (layer, group): `2 * 64 * 2` entries.
* The commit record is `[n_keep, n_keep-1, pos_0 .. pos_{T-1}]`.

## Form template identifiers
Verifier -> window_verifier; run -> verify_window; commit -> keep_prefix; VerifyHits -> resident_view;
PoolMultiFn -> multi_token_pool; GpuPlanSink -> gpu_share_plan.
-/

namespace SpeculativeVerifyWindow

-- § acceptance

/-- Number of drafts accepted: the longest prefix of `drafts` equal to the window's argmaxes. -/
def accepted : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then 1 + accepted ds os else 0
  | _, _ => 0

/-- Acceptance never exceeds the number of drafts. -/
theorem accepted_le_drafts : ∀ (ds os : List Nat), accepted ds os ≤ ds.length := by
  intro ds
  induction ds with
  | nil => intro os; simp [accepted]
  | cons d ds ih =>
    intro os
    cases os with
    | nil => simp [accepted]
    | cons o os =>
      by_cases h : d = o
      · have := ih os
        simp [accepted, h]
        omega
      · simp [accepted, h]

/-- Acceptance never exceeds the number of argmaxes available. -/
theorem accepted_le_outs : ∀ (ds os : List Nat), accepted ds os ≤ os.length := by
  intro ds
  induction ds with
  | nil => intro os; simp [accepted]
  | cons d ds ih =>
    intro os
    cases os with
    | nil => simp [accepted]
    | cons o os =>
      by_cases h : d = o
      · have := ih os
        simp [accepted, h]
        omega
      · simp [accepted, h]

/-- If every draft matches its argmax, all drafts are accepted. -/
theorem accepted_all : ∀ ds : List Nat, accepted ds ds = ds.length := by
  intro ds
  induction ds with
  | nil => rfl
  | cons d ds ih => simp [accepted, ih]; omega

/-- A mismatch at the first draft rejects everything after it. -/
theorem accepted_first_mismatch {d o : Nat} (h : d ≠ o) (ds os : List Nat) :
    accepted (d :: ds) (o :: os) = 0 := by
  simp [accepted, h]

/-- `n_keep = 1 + accepted` is between 1 and `T = drafts + 1`. -/
theorem n_keep_range (ds os : List Nat) :
    1 ≤ 1 + accepted ds os ∧ 1 + accepted ds os ≤ ds.length + 1 := by
  have := accepted_le_drafts ds os
  omega

-- § array bounds

/-- `exec_[9]` is indexed by `T <= 8`. -/
theorem exec_index_lt {T : Nat} (h : T ≤ 8) : T < 9 := by omega

/-- Flag slot of (layer, group) lies in `flag_sets_[2 * 64 * 2]` for layers < 64 and two groups. -/
theorem flag_slot_lt {l g : Nat} (hl : l < 64) (hg : g < 2) : l * 2 + g < 2 * 64 * 2 := by omega

/-- The commit record holds `2 + T` words. -/
def commitWords (T : Nat) : Nat := 2 + T

/-- A window of at most 8 tokens has a commit record of at most 10 words. -/
theorem commitWords_le {T : Nat} (h : T ≤ 8) : commitWords T ≤ 10 := by
  unfold commitWords
  omega

/-- Entries of a window (tokens x 10 experts) stay within the plan's capacity of 128. -/
theorem window_entries_le {T : Nat} (h : T ≤ 8) : T * 10 ≤ 128 := by omega

end SpeculativeVerifyWindow
