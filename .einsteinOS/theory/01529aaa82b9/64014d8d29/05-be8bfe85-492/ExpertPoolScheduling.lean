/-!
# MODULE: expert_pool_scheduling
SOURCES: include/strata/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
The CPU expert pool: row-range splitting of experts across tasks, the flat-batch counter protocol
(`head`/`done`/`parked`), worker-count selection and core reservation.

## Source semantics
* A task `i` of a split phase covers rows `[FF*part/parts, FF*(part+1)/parts)`; ranges tile `[0, FF)`.
* In multi-token phases a task covers a global row range `[g0, g1)` that may span experts; each step
  takes `r1 = min per (r0 + (g1 - r))` and always makes progress.
* `run()` publishes only when `parked == n`, waits for `done == n`, then waits for `parked == n` again.
* At least one worker exists; `skip_first` drops exactly one core.

## Form template identifiers
* head_, done_, parked_, epoch_ -> claim_counter, completion_counter, park_counter, publish_epoch
* ExpertJob / ExpertJobMulti -> work_item, multi_token_work_item
* kMaxSplit, kMaxSplitMulti -> split_capacity
-/

namespace Strata.Pool

-- § row-range splitting
/-- Row start of task `part` out of `parts` over `total` rows (`FF * part / parts`). -/
def rowStart (total parts part : Nat) : Nat := total * part / parts

/-- The first range starts at row 0. -/
theorem rowStart_zero (total parts : Nat) : rowStart total parts 0 = 0 := by
  simp [rowStart]

/-- The last range ends at the total (needs at least one part). -/
theorem rowStart_last (total parts : Nat) (hp : 0 < parts) : rowStart total parts parts = total := by
  unfold rowStart
  exact Nat.mul_div_cancel total hp

/-- Range starts are monotone, so each range `[start part, start (part+1))` is well formed. -/
theorem rowStart_mono (total parts part : Nat) : rowStart total parts part ≤ rowStart total parts (part + 1) := by
  unfold rowStart
  exact Nat.div_le_div_right (Nat.mul_le_mul_left total (Nat.le_succ part))

/-- Every range is inside `[0, total]`. -/
theorem rowStart_le_total (total parts part : Nat) (hp : 0 < parts) (hpart : part ≤ parts) :
    rowStart total parts part ≤ total := by
  have h1 := rowStart_mono total parts part
  have h2 : rowStart total parts part ≤ rowStart total parts parts := by
    unfold rowStart
    exact Nat.div_le_div_right (Nat.mul_le_mul_left total hpart)
  rw [rowStart_last total parts hp] at h2
  exact h2

-- § task decomposition (mode 1 / mode 2: task i = expert e, part p)
/-- A task index below `parts * n` decodes to an expert below `n`. -/
theorem task_expert_lt (i parts n : Nat) (h : i < parts * n) : i / parts < n :=
  Nat.div_lt_of_lt_mul h

/-- A task index decodes to a part below `parts`. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : i % parts < parts :=
  Nat.mod_lt _ hp

/-- The decomposition is invertible: `i = parts * e + part`. -/
theorem task_decode (i parts : Nat) : parts * (i / parts) + i % parts = i :=
  Nat.div_add_mod i parts

-- § multi-token phases: a global row range cut at expert boundaries
/-- End of the current piece: stay inside the expert (`per` rows) and inside the task's range. -/
def pieceEnd (per r0 g1 r : Nat) : Nat := min per (r0 + (g1 - r))

/-- Global row `r` decodes to expert `r / per` and local row `r % per`. -/
theorem global_row_decode (r per : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

/-- The local start is inside the expert. -/
theorem local_start_lt (r per : Nat) (hper : 0 < per) : r % per < per :=
  Nat.mod_lt _ hper

/-- The piece is non-empty while rows remain in the task: guarantees the loop makes progress. -/
theorem piece_progress (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < pieceEnd per r0 g1 r := by
  unfold pieceEnd
  omega

/-- The piece never leaves the expert. -/
theorem piece_in_expert (per r0 g1 r : Nat) : pieceEnd per r0 g1 r ≤ per := by
  unfold pieceEnd
  omega

/-- The piece never consumes more rows than remain in the task, so `r += r1 - r0` cannot overshoot `g1`. -/
theorem piece_no_overshoot (per r0 g1 r : Nat) (hr : r ≤ g1) :
    pieceEnd per r0 g1 r - r0 ≤ g1 - r := by
  unfold pieceEnd
  omega

-- § counter protocol
/-- Observable pool state: claim counter, completion counter, parked workers. -/
structure PoolState where
  head : Nat
  done : Nat
  parked : Nat

/-- A claim returns the old head and bumps it. -/
def claim (s : PoolState) : Nat × PoolState := (s.head, { s with head := s.head + 1 })

/-- A claimed index is executed only when it is below the batch size. -/
def executes (n i : Nat) : Bool := decide (i < n)

/-- Claims return distinct, consecutive indices. -/
theorem claim_distinct (s : PoolState) : (claim s).1 < (claim (claim s).2).1 := by
  simp [claim]

/-- Publishing is only legal when every worker is parked (the host waits for this first). -/
def canPublish (n : Nat) (s : PoolState) : Prop := s.parked = n

/-- Publication resets head and done: afterwards no stale claim is possible. -/
def publish (s : PoolState) : PoolState := { s with head := 0, done := 0 }

/-- After publication the batch starts from a clean counter state. -/
theorem publish_clean (s : PoolState) : (publish s).head = 0 ∧ (publish s).done = 0 := by
  simp [publish]

/-- The batch is complete only when both `done == n` and `parked == nw`. -/
def batchComplete (n nw : Nat) (s : PoolState) : Prop := s.done = n ∧ s.parked = nw

/-- Completion implies every worker is parked, i.e. nobody can still claim from the old batch. -/
theorem complete_implies_parked (n nw : Nat) (s : PoolState) (h : batchComplete n nw s) : s.parked = nw :=
  h.2

-- § worker count and core reservation
/-- Worker count: requested count if positive, else every core; never below one. -/
def poolWorkers (requested cores : Nat) : Nat := max 1 (if 0 < requested then requested else cores)

/-- A pool always has at least one worker, even on a single-core machine. -/
theorem poolWorkers_pos (requested cores : Nat) : 1 ≤ poolWorkers requested cores := by
  unfold poolWorkers
  exact Nat.le_max_left _ _

/-- A positive request is honoured exactly. -/
theorem poolWorkers_request (requested cores : Nat) (h : 0 < requested) :
    poolWorkers requested cores = requested := by
  unfold poolWorkers
  rw [if_pos h]
  omega

/-- Core list after the optional reservation of core 0 for the host. -/
def reserveHost (skipFirst : Bool) (cores : List Nat) : List Nat := if skipFirst then cores.tail else cores

/-- Reserving the host's core removes exactly one core when there is one. -/
theorem reserveHost_length (cores : List Nat) (h : cores ≠ []) :
    (reserveHost true cores).length + 1 = cores.length := by
  cases cores with
  | nil => exact absurd rfl h
  | cons a t => simp [reserveHost]

/-- Without the reservation the list is untouched. -/
theorem reserveHost_false (cores : List Nat) : reserveHost false cores = cores := by
  simp [reserveHost]

/-- With `host_works` the drain uses workers plus the host thread. -/
def drainThreads (nw : Nat) (hostWorks : Bool) : Nat := nw + if hostWorks then 1 else 0

/-- Task count of the multi phases is three tasks per thread, hence strictly positive. -/
theorem multi_tasks_pos (nw : Nat) (hostWorks : Bool) (h : 1 ≤ nw) : 0 < 3 * drainThreads nw hostWorks := by
  unfold drainThreads
  split <;> omega

/-- Split phases are used only up to capacity; larger batches fall back to `run`/singles. -/
def useSplit (n kMax : Nat) (oracle : Bool) : Bool := decide (n ≤ kMax) && !oracle

/-- The oracle contract never uses the split path. -/
theorem oracle_never_split (n kMax : Nat) : useSplit n kMax true = false := by
  simp [useSplit]

end Strata.Pool
