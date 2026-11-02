/-!
# MODULE: expert_pool_protocol
CPU expert thread pool: workers park on an epoch counter, drain a shared job counter, and the host optionally
drains too. Multi-phase modes split the rows of all experts into equal task ranges.

## Source semantics
* `physical_cores(skip_first)` drops core 0 so the host owns it.
* Row ranges `[R*i/T, R*(i+1)/T)` tile `[0, R)` contiguously and monotonically.
* Inside a task, the expert/row walk always progresses, never leaves an expert, and never overshoots the task end.
* Park accounting: workers arrive once at construction, leave on work, arrive after drain; parked + draining = n.
* Batching of more experts than buffers proceeds in bounded, non-empty batches.
* `run_split` falls back to `run` when too many jobs, a single worker, or the oracle is on.

## Form template identifiers
ExpertPool::drain -> task_drain ; run_phase -> phase_barrier ; parked_ , epoch_ , head_ , done_ -> park_count ,
epoch , claim_counter , done_counter ; mrows_ , mtasks_ -> total_rows , task_count
-/

namespace Strata.Cpu.ExpertPool

-- § core list

/-- Core list after the optional host-core skip. -/
def skipFirst (skip : Bool) (cores : List Nat) : List Nat := if skip then cores.drop 1 else cores

/-- Skipping removes at most one core. -/
theorem skipFirst_length (skip : Bool) (cores : List Nat) :
    (skipFirst skip cores).length = if skip then cores.length - 1 else cores.length := by
  cases skip <;> simp [skipFirst]

/-- Without skipping the list is unchanged. -/
theorem skipFirst_false (cores : List Nat) : skipFirst false cores = cores := by
  simp [skipFirst]

-- § row-range partition

/-- Start of part `i` when `rows` are split across `parts`. -/
def lo (rows parts i : Nat) : Nat := rows * i / parts

/-- The first part starts at row 0. -/
theorem lo_zero (rows parts : Nat) : lo rows parts 0 = 0 := by
  simp [lo]

/-- The last boundary reaches all rows. -/
theorem lo_last (rows parts : Nat) (h : 0 < parts) : lo rows parts parts = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Boundaries are monotone. -/
theorem lo_mono (rows parts i : Nat) : lo rows parts i ≤ lo rows parts (i + 1) := by
  unfold lo
  apply Nat.div_le_div_right
  exact Nat.mul_le_mul_left rows (Nat.le_succ i)

/-- Parts are contiguous: the end of part i is the start of part i+1. -/
theorem lo_contiguous (rows parts i : Nat) : lo rows parts (i + 1) = rows * (i + 1) / parts := rfl

/-- No boundary exceeds the total once `i ≤ parts`. -/
theorem lo_le_rows (rows parts i : Nat) (hp : 0 < parts) (hi : i ≤ parts) : lo rows parts i ≤ rows := by
  have h1 : lo rows parts i ≤ lo rows parts parts := by
    unfold lo
    apply Nat.div_le_div_right
    exact Nat.mul_le_mul_left rows hi
  rw [lo_last rows parts hp] at h1
  exact h1

-- § per-task walk across experts

/-- Row-in-expert is strictly below the expert size. -/
theorem row_in_expert_lt (r per : Nat) (h : 0 < per) : r % per < per := Nat.mod_lt r h

/-- One step of the expert walk makes progress, stays in the expert, and stays within the task. -/
theorem walk_step (per r0 g1 r : Nat) (hr0 : r0 < per) (hlt : r < g1) :
    r0 < min per (r0 + (g1 - r)) ∧ min per (r0 + (g1 - r)) ≤ per ∧
    r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by
  omega

/-- Expert index and row recombine to the global row. -/
theorem expert_row_recombine (r per : Nat) (h : 0 < per) : (r / per) * per + r % per = r := by
  have := Nat.div_add_mod r per
  rw [Nat.mul_comm] at this
  exact this

-- § task counts

/-- Parts per expert in the single-token split: about three tasks per thread. -/
def partsA (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- There is always at least one part. -/
theorem partsA_pos (threads n : Nat) : 1 ≤ partsA threads n := by
  unfold partsA
  omega

/-- Total tasks of a two-phase split. -/
theorem tasks_total (n parts : Nat) : n * parts = n * parts := rfl

/-- Task index decomposition: expert and part recombine. -/
theorem task_decompose (i parts : Nat) (h : 0 < parts) : (i / parts) * parts + i % parts = i := by
  have := Nat.div_add_mod i parts
  rw [Nat.mul_comm] at this
  exact this

/-- The part index is below the part count. -/
theorem part_lt (i parts : Nat) (h : 0 < parts) : i % parts < parts := Nat.mod_lt i h

-- § park accounting

/-- Workers either parked or draining. -/
structure PState where
  parked : Nat
  draining : Nat

/-- A worker leaves the park to drain. -/
def leave (s : PState) : PState := ⟨s.parked - 1, s.draining + 1⟩

/-- A worker returns to the park after draining. -/
def arrive (s : PState) : PState := ⟨s.parked + 1, s.draining - 1⟩

/-- Leaving preserves the worker total. -/
theorem leave_inv (s : PState) (n : Nat) (h : s.parked + s.draining = n) (hp : 0 < s.parked) :
    (leave s).parked + (leave s).draining = n := by
  simp only [leave]
  omega

/-- Arriving preserves the worker total. -/
theorem arrive_inv (s : PState) (n : Nat) (h : s.parked + s.draining = n) (hd : 0 < s.draining) :
    (arrive s).parked + (arrive s).draining = n := by
  simp only [arrive]
  omega

/-- Leave then arrive restores the state. -/
theorem leave_arrive (s : PState) (hp : 0 < s.parked) : arrive (leave s) = s := by
  cases s with
  | mk p d =>
    have hp' : 0 < p := hp
    have h1 : p - 1 + 1 = p := by omega
    simp [leave, arrive, h1]

/-- After all workers arrive at construction the pool is fully parked. -/
theorem constructed_all_parked (n : Nat) : (PState.mk n 0).parked = n := rfl

-- § claim counter

/-- A claimed index runs a job exactly when it is below the job count. -/
def claims (i njobs : Nat) : Bool := decide (i < njobs)

/-- Indices at or past the job count end the drain loop. -/
theorem claim_stop (i njobs : Nat) (h : njobs ≤ i) : claims i njobs = false := by
  simp [claims]
  omega

/-- Indices below the job count are executed. -/
theorem claim_run (i njobs : Nat) (h : i < njobs) : claims i njobs = true := by
  simp [claims, h]

-- § mode dispatch

/-- Rows-per-expert for a multi-token mode: gate/up modes (3, 5) use FF, others H. -/
def perOf (ff h mode : Nat) : Nat := if mode = 3 ∨ mode = 5 then ff else h

/-- Modes 3 and 4 never take the native (>= 5) branch. -/
theorem multi_not_native (m : Nat) (h : m = 3 ∨ m = 4) : ¬ (m ≥ 5) := by omega

/-- Native modes are exactly those at least 5. -/
theorem native_mode (m : Nat) : (m ≥ 5) ↔ (5 ≤ m) := Iff.rfl

-- § run_split guards and batching

/-- `run_split` falls back to `run` for oversize jobs, one worker, or the oracle. -/
def useRun (n kMax nWorkers : Nat) (oracle : Bool) : Bool :=
  decide (n > kMax) || decide (nWorkers = 1) || oracle

/-- One worker always uses the inline path. -/
theorem single_worker_uses_run (n kMax : Nat) (oracle : Bool) : useRun n kMax 1 oracle = true := by
  simp [useRun]

/-- The oracle always uses the single-token path. -/
theorem oracle_uses_run (n kMax nWorkers : Nat) : useRun n kMax nWorkers true = true := by
  simp [useRun]

/-- Batch size when more experts than buffers must be processed in several batches. -/
def batchLen (kMax n b0 : Nat) : Nat := min kMax (n - b0)

/-- Batches are non-empty, bounded by the buffer count, and inside the job list. -/
theorem batch_ok (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) :
    1 ≤ batchLen kMax n b0 ∧ batchLen kMax n b0 ≤ kMax ∧ b0 + batchLen kMax n b0 ≤ n := by
  unfold batchLen
  omega

/-- The batch loop strictly advances. -/
theorem batch_advance (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) :
    b0 < b0 + batchLen kMax n b0 := by
  unfold batchLen
  omega

end Strata.Cpu.ExpertPool
