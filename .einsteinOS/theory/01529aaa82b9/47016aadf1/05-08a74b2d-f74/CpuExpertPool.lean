/-!
# MODULE: cpu_expert_pool
The CPU expert thread pool: task splitting across workers, row-range segmentation across experts, the native
Q2_0 / generic routing by phase mode, batching of experts, park accounting and core selection.

## Source semantics
* A phase of `rows` rows is cut into `tasks` equal-ish ranges `[rows*i/tasks, rows*(i+1)/tasks)`; the ranges
  are monotone, start at 0 and end at `rows`, so together they cover every row exactly once.
* A range may span several experts; each step of the segment loop makes progress and never overshoots.
* Phase modes 5/6 route to Q2_0 kernels when the pack type is 42, otherwise to the native dispatch.
* Experts beyond the split-buffer capacity are processed in batches.
* The first physical core is reserved for the host.

## Form template identifiers
run_phase -> run_phase, drain -> drain, mrows_/mtasks_ -> rows/tasks, split_ -> split_buffers,
physical_cores -> physical_cores, parked_ -> parked_count.
-/

namespace Strata.CpuExpertPool

-- § task splitting

/-- Start row of task `i` when `rows` rows are divided among `tasks` tasks. -/
def splitPt (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem splitPt_zero (rows tasks : Nat) : splitPt rows tasks 0 = 0 := by
  simp [splitPt]

/-- The last boundary is exactly `rows`, so no row is left uncovered. -/
theorem splitPt_last (rows tasks : Nat) (h : 0 < tasks) : splitPt rows tasks tasks = rows := by
  unfold splitPt
  exact Nat.mul_div_cancel rows h

/-- Task boundaries are monotone, so the ranges are well formed and disjoint. -/
theorem splitPt_mono (rows tasks i : Nat) : splitPt rows tasks i ≤ splitPt rows tasks (i + 1) := by
  unfold splitPt
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows (Nat.le_succ i))

/-- Every boundary lies within `[0, rows]` for `i ≤ tasks`. -/
theorem splitPt_le_rows (rows tasks i : Nat) (hi : i ≤ tasks) : splitPt rows tasks i ≤ rows := by
  unfold splitPt
  have h := Nat.mul_le_mul_left rows hi
  rw [Nat.mul_comm rows tasks] at h
  exact Nat.div_le_of_le_mul h

/-- Total number of threads working on a phase: the workers plus the host when it helps. -/
def threads (nWorkers : Nat) (hostWorks : Bool) : Nat := nWorkers + (if hostWorks then 1 else 0)

/-- Tasks per phase: about three per thread. -/
def tasksPerPhase (nWorkers : Nat) (hostWorks : Bool) : Nat := 3 * threads nWorkers hostWorks

/-- A pool with at least one worker has at least three tasks per phase. -/
theorem tasks_pos (n : Nat) (hostWorks : Bool) (hn : 1 ≤ n) : 3 ≤ tasksPerPhase n hostWorks := by
  unfold tasksPerPhase threads
  split <;> omega

/-- Parts per expert in the split-phase modes is at least one. -/
theorem parts_pos (threads n : Nat) : 1 ≤ max 1 ((3 * threads + n - 1) / n) :=
  Nat.le_max_left 1 _

-- § row-range segmentation across experts

/-- One step of the segment loop: with `r0` the offset inside the current expert (< per) and `g1 - r`
rows still wanted, the step end `r1` is `min per (r0 + (g1 - r))`. The step advances and never overshoots. -/
theorem seg_progress (per r0 r g1 : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < min per (r0 + (g1 - r)) ∧ r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by
  omega

/-- The offset inside an expert is always a valid row of that expert. -/
theorem seg_offset_lt (per r : Nat) (h : 0 < per) : r % per < per := Nat.mod_lt r h

-- § routing

/-- The kernel family a native-layer phase is routed to. -/
inductive Path
  | q2Gu
  | nativeGu
  | q2Down
  | nativeDown
  | other
  deriving DecidableEq

/-- Route a phase mode (5 = gate/up, 6 = down) given the pack's gate/up and down type ids. -/
def route (mode gu d : Nat) : Path :=
  if mode = 5 then (if gu = 42 then Path.q2Gu else Path.nativeGu)
  else if mode = 6 then (if d = 42 then Path.q2Down else Path.nativeDown)
  else Path.other

/-- A Q2_0 gate/up pack uses the Q2_0 kernels. -/
theorem route_q2_gu (d : Nat) : route 5 42 d = Path.q2Gu := by
  simp [route]

/-- Any other gate/up type uses the native dispatch. -/
theorem route_native_gu (gu d : Nat) (h : gu ≠ 42) : route 5 gu d = Path.nativeGu := by
  simp [route, h]

/-- A Q2_0 down projection uses the AVX-512 Q2_0 kernel. -/
theorem route_q2_down (gu : Nat) : route 6 gu 42 = Path.q2Down := by
  simp [route]

/-- Any other down type uses the native dispatch. -/
theorem route_native_down (gu d : Nat) (h : d ≠ 42) : route 6 gu d = Path.nativeDown := by
  simp [route, h]

-- § expert batching

/-- Experts handled in the batch starting at `b0` when `cap` split buffers exist. -/
def batchLen (cap n b0 : Nat) : Nat := min cap (n - b0)

/-- Each batch is non-empty and within the expert list (capacity must be positive). -/
theorem batch_progress (cap n b0 : Nat) (hc : 0 < cap) (hb : b0 < n) :
    0 < batchLen cap n b0 ∧ b0 + batchLen cap n b0 ≤ n := by
  unfold batchLen
  omega

-- § park accounting

/-- A worker leaving the park and arriving back restores the parked count. -/
theorem leave_then_arrive (p : Nat) (h : 1 ≤ p) : (p - 1) + 1 = p := by omega

/-- Construction starts with every worker counted as parked. -/
theorem parked_initially (n : Nat) : (List.replicate n 1).sum = n := by
  simp

-- § core selection

/-- Drop the first core, reserved for the host thread. -/
def skipFirst : List Nat → List Nat
  | [] => []
  | _ :: t => t

/-- Skipping the first core removes exactly one entry from a non-empty list. -/
theorem skipFirst_length (l : List Nat) : (skipFirst l).length = l.length - 1 := by
  cases l <;> simp [skipFirst]

/-- Core assigned to worker `i`: its own core when available, otherwise unpinned (-1). -/
def coreFor (cores : List Nat) (i : Nat) : Int :=
  match cores[i]? with
  | some c => (c : Int)
  | none => -1

/-- Workers beyond the core list are left unpinned. -/
theorem coreFor_overflow (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : coreFor cores i = -1 := by
  have : cores[i]? = none := List.getElem?_eq_none h
  simp [coreFor, this]

/-- A negative core means "do not pin". -/
def shouldPin (core : Int) : Bool := decide (0 ≤ core)

/-- Unpinned workers are never pinned. -/
theorem unpinned_not_pinned : shouldPin (-1) = false := by decide

end Strata.CpuExpertPool
