/-!
# MODULE: speculative_verifier
The speculative verify window: `T` tokens through all layers in one graph, commit of the accepted prefix,
split into two pipelined token groups, flag-table indexing and the PLE predecessor history.

## Source semantics
* `T <= kVerifyMaxT = 8`; `commit(n_keep)` requires `1 <= n_keep <= T`.
* The window appends K/V for all `T` positions; only the first `n_keep` become permanent, so the next
  position is `pos0 + n_keep`.
* Splitting a window into two groups preserves the token count.
* Flag-set entries are indexed by `(layer, group, flag)` and stay inside `2 * 64 * 2` entries.
* The PLE predecessor pair after committing is the last two of `prev ++ kept tokens`.

## Form template identifiers
* `Verifier::commit` -> `commitOk`, `nextPos`
* `Verifier::set_split` -> `groupSizes`
* `flag_sets_` -> `flagIndex`
* `ss.ple_prev` -> `lastTwo`
-/

namespace Strata.SpeculativeVerifier

-- § window and commit

/-- Window sizes accepted by `init`/`run`. -/
def windowOk (T maxT : Nat) : Bool := decide (1 ≤ T ∧ T ≤ maxT ∧ maxT ≤ 8)

/-- Commit is accepted only for `1 <= n_keep <= T`. -/
def commitOk (nKeep T : Nat) : Bool := decide (1 ≤ nKeep ∧ nKeep ≤ T)

/-- Position of the next token after committing `n_keep` tokens of a window starting at `pos0`. -/
def nextPos (pos0 nKeep : Nat) : Nat := pos0 + nKeep

/-- Committing zero tokens is refused: the window must make progress. -/
theorem commit_zero_refused (T : Nat) : commitOk 0 T = false := by
  unfold commitOk
  exact decide_eq_false (by omega)

/-- Committing more tokens than the window holds is refused. -/
theorem commit_overflow_refused (nKeep T : Nat) (h : T < nKeep) : commitOk nKeep T = false := by
  unfold commitOk
  exact decide_eq_false (by omega)

/-- A valid commit advances the position by at least one and at most `T`. -/
theorem commit_progress (pos0 nKeep T : Nat) (h : commitOk nKeep T = true) :
    pos0 < nextPos pos0 nKeep ∧ nextPos pos0 nKeep ≤ pos0 + T := by
  unfold commitOk at h
  have h' := of_decide_eq_true h
  unfold nextPos
  omega

/-- The window arrays hold up to eight tokens: the last-token snapshot and all tables have room. -/
theorem window_fits_tables (T : Nat) (h : T ≤ 8) : T < 9 := by
  omega

-- § group split

/-- The two pipelined token groups of a window of `T` tokens. -/
def groupSizes (T : Nat) : Nat × Nat := (T / 2, T - T / 2)

/-- The two groups together hold exactly `T` tokens. -/
theorem groups_sum (T : Nat) : (groupSizes T).1 + (groupSizes T).2 = T := by
  unfold groupSizes
  omega

/-- For `T ≥ 2` both groups are non-empty, so the pipeline always has work in each. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) : 0 < (groupSizes T).1 ∧ 0 < (groupSizes T).2 := by
  unfold groupSizes
  omega

/-- Neither group is larger than the other by more than one token. -/
theorem groups_balanced (T : Nat) : (groupSizes T).1 ≤ (groupSizes T).2 ∧ (groupSizes T).2 ≤ (groupSizes T).1 + 1 := by
  unfold groupSizes
  omega

-- § flag table

/-- Index of the host-function argument for `(layer, group, flag)` in the `2 * 64 * 2` table. -/
def flagIndex (layer group flag : Nat) : Nat := (layer * 2 + group) * 2 + flag

/-- Every in-range `(layer, group, flag)` lands inside the 256-entry table. -/
theorem flagIndex_lt (layer group flag : Nat) (hl : layer < 64) (hg : group < 2) (hf : flag < 2) :
    flagIndex layer group flag < 256 := by
  unfold flagIndex
  omega

/-- Distinct triples never share a table cell. -/
theorem flagIndex_injective (l g f l' g' f' : Nat) (hg : g < 2) (hf : f < 2) (hg' : g' < 2) (hf' : f' < 2)
    (h : flagIndex l g f = flagIndex l' g' f') : l = l' ∧ g = g' ∧ f = f' := by
  unfold flagIndex at h
  omega

/-- A DMA fetch of `n` blobs is accepted only if it fits the 16 staging blobs. -/
def fetchOk (n : Nat) : Bool := decide (n ≤ 16)

/-- Seventeen blobs do not fit the staging area. -/
theorem fetch_overflow : fetchOk 17 = false ∧ fetchOk 16 = true := by
  decide

-- § PLE predecessor history

/-- The last two elements of a list (all of it if shorter than two). -/
def lastTwo (l : List Nat) : List Nat := l.drop (l.length - 2)

/-- The predecessor pair never holds more than two tokens. -/
theorem lastTwo_length_le (l : List Nat) : (lastTwo l).length ≤ 2 := by
  unfold lastTwo
  simp [List.length_drop]
  omega

/-- The predecessor pair after committing keeps the most recent kept token last. -/
theorem lastTwo_append_single (l : List Nat) (x : Nat) : (lastTwo (l ++ [x])).getLast? = some x := by
  unfold lastTwo
  simp [List.getLast?_drop]

/-- Advance the history by the kept tokens. -/
def advancePrev (prev toks : List Nat) (nKeep : Nat) : List Nat := lastTwo (prev ++ toks.take nKeep)

/-- Keeping no token leaves at most the old pair's last two. -/
theorem advance_zero (prev toks : List Nat) : advancePrev prev toks 0 = lastTwo prev := by
  simp [advancePrev]

/-- The history length after any commit is at most two. -/
theorem advance_length_le (prev toks : List Nat) (n : Nat) : (advancePrev prev toks n).length ≤ 2 := by
  unfold advancePrev
  exact lastTwo_length_le _

end Strata.SpeculativeVerifier
