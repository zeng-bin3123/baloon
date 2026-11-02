/-!
# MODULE: expert_pool_scheduling
CPU expert thread pool: physical-core selection, worker count, equal-range row partitioning across tasks,
segment walking over expert boundaries, batching of experts, and task-count formulas.

## Source semantics
* Task `i` of `tasks` over `rows` rows owns `[rows*i/tasks, rows*(i+1)/tasks)`: ranges are contiguous,
  monotone and cover `[0, rows)` exactly.
* A range may span several experts; it is walked as segments `(e = r / per, r0 = r % per, r1 = min per (r0 + remaining))`
  and each segment makes progress and stays inside one expert.
* `skip_first` drops the first core (kept for the host); worker count is at least one.
* More experts than split buffers are processed in batches of at most `kMax`.
* Source note: on non-Windows, `physical_cores` has a dangling `else` that binds to the inner
  `if (CPU_ISSET(i, &set))`, so the `hardware_concurrency` fallback is never reached via `sched_getaffinity`
  failure and instead runs once per CPU *not* in the mask. Not modelled here.

## Form template identifiers
* `rangeLo` ~ `mrows_ * i / mtasks_`
* `segR0` / `segR1` ~ `r0` / `r1` in the drain loop
* `batchSize` ~ `nb = min(kMaxSplitMulti, n - b0)`
* `nWorkers` ~ `n_`, `dropFirst` ~ `physical_cores(skip_first)`
* `partsOf` ~ `parts_a_`
-/

namespace Strata.ExpertPool

-- § equal-range partition

/-- Start of task `i`'s range. -/
def rangeLo (rows i tasks : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem rangeLo_zero (rows tasks : Nat) : rangeLo rows 0 tasks = 0 := by
  simp [rangeLo]

/-- The sentinel after the last task is exactly `rows`. -/
theorem rangeLo_last (rows tasks : Nat) (h : 0 < tasks) : rangeLo rows tasks tasks = rows := by
  unfold rangeLo
  exact Nat.mul_div_cancel rows h

/-- Range starts are monotone in the task index. -/
theorem rangeLo_mono (rows tasks i j : Nat) (h : i ≤ j) : rangeLo rows i tasks ≤ rangeLo rows j tasks := by
  unfold rangeLo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows h)

/-- No range start exceeds `rows`. -/
theorem rangeLo_le_rows (rows tasks i : Nat) (h : i ≤ tasks) : rangeLo rows i tasks ≤ rows := by
  unfold rangeLo
  apply Nat.div_le_of_le_mul
  rw [Nat.mul_comm tasks rows]
  exact Nat.mul_le_mul_left rows h

-- § segment walking across expert boundaries

/-- Offset within the current expert. -/
def segR0 (per r : Nat) : Nat := r % per

/-- End offset of the segment: stay in the expert, stop at the range end. -/
def segR1 (per r g1 : Nat) : Nat := min per (segR0 per r + (g1 - r))

/-- Expert index and offset recompose the global row. -/
theorem seg_recompose (per r : Nat) : per * (r / per) + segR0 per r = r := by
  unfold segR0
  exact Nat.div_add_mod r per

/-- The offset lies inside the expert. -/
theorem segR0_lt (per r : Nat) (h : 0 < per) : segR0 per r < per := by
  unfold segR0
  exact Nat.mod_lt r h

/-- Each segment is non-empty, stays within one expert, and does not pass the range end: the walk terminates. -/
theorem seg_progress (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    segR0 per r < segR1 per r g1 ∧ segR1 per r g1 ≤ per ∧ r + (segR1 per r g1 - segR0 per r) ≤ g1 := by
  have h0 := segR0_lt per r hper
  unfold segR1
  generalize segR0 per r = r0 at h0 ⊢
  omega

-- § batching and task counts

/-- Experts handled in the batch starting at `b0`. -/
def batchSize (kMax n b0 : Nat) : Nat := min kMax (n - b0)

/-- A batch is non-empty while experts remain (given a positive buffer count). -/
theorem batchSize_pos (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) : 1 ≤ batchSize kMax n b0 := by
  unfold batchSize
  omega

/-- A batch fits in the buffers. -/
theorem batchSize_le (kMax n b0 : Nat) : batchSize kMax n b0 ≤ kMax := by
  unfold batchSize
  omega

/-- A batch never passes the last expert. -/
theorem batchSize_fits (kMax n b0 : Nat) (h : b0 < n) : b0 + batchSize kMax n b0 ≤ n := by
  unfold batchSize
  omega

/-- Parts per expert: about three tasks per thread, at least one. -/
def partsOf (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsOf_pos (threads n : Nat) : 1 ≤ partsOf threads n := by
  unfold partsOf
  exact Nat.le_max_left _ _

/-- The split phase has at least one task per expert. -/
theorem split_tasks_ge (threads n : Nat) : n ≤ n * partsOf threads n := by
  have h := Nat.mul_le_mul_left n (partsOf_pos threads n)
  simpa using h

-- § worker and core selection

/-- Optionally drop the first core (reserved for the host thread). -/
def dropFirst (skip : Bool) (cs : List Nat) : List Nat := if skip then cs.tail else cs

/-- Dropping the host core removes exactly one entry from a non-empty list. -/
theorem dropFirst_length (cs : List Nat) (h : cs ≠ []) : (dropFirst true cs).length = cs.length - 1 := by
  simp [dropFirst]

/-- Without `skip_first` the list is unchanged. -/
theorem dropFirst_false (cs : List Nat) : dropFirst false cs = cs := by
  simp [dropFirst]

/-- Worker count: the request if positive, otherwise one per core; never below one. -/
def nWorkers (req cores : Nat) : Nat := max 1 (if req > 0 then req else cores)

/-- There is always at least one worker, so a single-core machine still produces a token. -/
theorem nWorkers_pos (req cores : Nat) : 1 ≤ nWorkers req cores := by
  unfold nWorkers
  exact Nat.le_max_left _ _

/-- Worker `i` is pinned to core `cores[i]` when it exists, otherwise left unpinned. -/
def pinCore (cores : List Nat) (i : Nat) : Option Nat := cores[i]?

/-- Workers within the core list always get a core. -/
theorem pinCore_some (cores : List Nat) (i : Nat) (h : i < cores.length) :
    pinCore cores i = some cores[i] := by
  unfold pinCore
  exact List.getElem?_eq_getElem h

/-- Workers beyond the core list are unpinned. -/
theorem pinCore_none (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : pinCore cores i = none := by
  unfold pinCore
  exact List.getElem?_eq_none h

-- § affinity return protocol

/-- Pin return value: -1 when the core is negative (no-op), else the previous mask. -/
def pinReturn (core : Int) (prev : Int) : Int := if core < 0 then -1 else prev

/-- A negative core id is always reported as "nothing to restore". -/
theorem pinReturn_neg (core prev : Int) (h : core < 0) : pinReturn core prev = -1 := by
  unfold pinReturn
  rw [if_pos h]

/-- `restore_thread_affinity` ignores non-positive values, so -1 never restores. -/
theorem restore_ignores_failure : ¬ ((-1 : Int) > 0) := by
  decide

end Strata.ExpertPool
