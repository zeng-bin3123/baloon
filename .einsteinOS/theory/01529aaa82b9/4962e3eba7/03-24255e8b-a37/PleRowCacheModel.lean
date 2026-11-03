/-!
# MODULE: ple_row_cache_model
Behavioural model of the PLE reader's 8-way set-associative row cache and the PleTable single-token in-flight
protocol (`issue` then `collect`).

## Source semantics
* A set holds 8 keys; `insert` writes the new key at the set's round-robin pointer `w` and advances it to `(w+1) % 8`.
* After an insert the key is findable; the other 7 ways are untouched; a key survives the next 7 inserts into the
  same set, and the pointer returns to `w` after 8 inserts.
* `used` grows only when an EMPTY way is overwritten, so `used <= number of inserts`.
* `PleTable::issue` fails when a token is already in flight; `collect` without `issue` fails; an issue followed by a
  collect leaves nothing in flight.
* A batch gather of `n = T*16` rows writes row `i` at `out + i*160`, covering exactly `T*2560` floats.

## Form template identifiers
RowCache set -> `keys : Nat → Nat`; insert -> `insertAt`; replacement pointer -> `ptr`; pending flag -> `pending`;
issue/collect -> `issueStep`/`collectStep`.
-/

namespace Strata.Ngram.CacheModel

-- § row cache

/-- Write `row` into way `w` of a set. -/
def insertAt (keys : Nat → Nat) (w row : Nat) : Nat → Nat := fun i => if i = w then row else keys i

/-- After an insert the row is findable in one of the 8 ways. -/
theorem find_after_insert (keys : Nat → Nat) (w row : Nat) (hw : w < 8) :
    ∃ i, i < 8 ∧ insertAt keys w row i = row := by
  refine ⟨w, hw, ?_⟩
  simp [insertAt]

/-- The other ways are untouched by an insert. -/
theorem other_ways_untouched (keys : Nat → Nat) (w row i : Nat) (h : i ≠ w) :
    insertAt keys w row i = keys i := by
  simp [insertAt, h]

/-- The replacement pointer stays inside the set. -/
theorem ptr_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

/-- A freshly written way is not overwritten by the next 7 inserts into the same set. -/
theorem survives_seven (w k : Nat) (hk1 : 1 ≤ k) (hk2 : k ≤ 7) : (w + k) % 8 ≠ w % 8 := by
  omega

/-- After 8 inserts the pointer is back where it started (round-robin). -/
theorem ptr_cycle (w : Nat) : (w + 8) % 8 = w % 8 := by
  omega

/-- The set count `sets = rows / 8` times 8 ways never exceeds the requested rows. -/
theorem capacity_le_rows (rows : Nat) : rows / 8 * 8 ≤ rows := by
  omega

/-- The `used` counter: incremented only when overwriting an EMPTY way. -/
def usedAfter (used : Nat) (wasEmpty : Bool) : Nat := if wasEmpty then used + 1 else used

/-- `used` grows by at most one per insert. -/
theorem used_step_le (used : Nat) (wasEmpty : Bool) : usedAfter used wasEmpty ≤ used + 1 := by
  unfold usedAfter
  split <;> omega

/-- `used` never decreases. -/
theorem used_step_ge (used : Nat) (wasEmpty : Bool) : used ≤ usedAfter used wasEmpty := by
  unfold usedAfter
  split <;> omega

-- § single-token in-flight protocol

/-- `issue`: returns whether it was accepted and the new pending flag. -/
def issueStep (pending : Bool) : Bool × Bool := if pending then (false, pending) else (true, true)

/-- `collect`: returns whether it succeeded and the new pending flag. -/
def collectStep (pending : Bool) : Bool × Bool := if pending then (true, false) else (false, false)

/-- Issuing while a token is in flight is refused and changes nothing. -/
theorem issue_refused (pending : Bool) (h : pending = true) : issueStep pending = (false, pending) := by
  subst h
  rfl

/-- Collecting with nothing in flight is refused. -/
theorem collect_without_issue : collectStep false = (false, false) := rfl

/-- An accepted issue followed by a collect succeeds and leaves nothing in flight. -/
theorem issue_collect_roundtrip : collectStep (issueStep false).2 = (true, false) := rfl

-- § batch gather layout

/-- Row `i < T*16` written at `i*160` ends within the `T*2560`-float output. -/
theorem gather_batch_in_range (T i : Nat) (h : i < T * 16) : i * 160 + 160 ≤ T * 2560 := by
  omega

/-- Head-slowest layout: head `h` of a token occupies `[h*160, (h+1)*160)` of its 2560 floats. -/
theorem head_slowest_in_token (h k : Nat) (hh : h < 16) (hk : k < 160) : h * 160 + k < 2560 := by
  omega

end Strata.Ngram.CacheModel
