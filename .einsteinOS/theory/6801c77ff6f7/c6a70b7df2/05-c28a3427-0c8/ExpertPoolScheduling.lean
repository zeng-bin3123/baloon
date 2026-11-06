/-!
# MODULE: expert_pool_scheduling
SOURCES: src/kernels/cpu/pool.cpp
Arithmetic of the pool's drain loop: the `head_` claim counter hands each job out exactly once, row-range
partitions `N*i/parts .. N*(i+1)/parts` tile `[0,N)` exactly, task indices decompose uniquely into
(expert, part), and the multi-expert row walk always makes progress and never leaves an expert.

## Source semantics
* `head_.fetch_add(1)` returns 0,1,2,...; a claim succeeds iff it is below the job count, so exactly `n`
  claims succeed no matter how many threads race (extra claims just break out).
* Mode 1/2 tasks: task `i` is expert `i / parts`, part `i % parts`; part `p` covers rows `[N*p/parts, N*(p+1)/parts)`.
* Modes 3..6 cut `rows` global rows into `tasks` equal ranges; a range is walked expert by expert with
  `e = r / per`, `r0 = r % per`, `r1 = min per (r0 + (g1 - r))`.
* The workers' park counter returns to its initial value after every worker leaves and re-arrives.

## Form template identifiers
lo → range_start, widthSum → covered_rows, successes → claims_granted, decomposeTask → expert_part_split,
walkNext → row_walk_step.
-/

namespace Strata.Kernels.Cpu.ExpertPoolScheduling

-- § claim counter

/-- Number of successful claims among the first `k` values returned by `fetch_add` for `n` jobs. -/
def successes : Nat → Nat → Nat
  | 0, _ => 0
  | k + 1, n => successes k n + (if k < n then 1 else 0)

/-- Exactly `min k n` of the first `k` claims succeed. -/
theorem successes_eq (k n : Nat) : successes k n = min k n := by
  induction k with
  | zero => simp [successes]
  | succ k ih =>
    rw [successes, ih]
    split <;> omega

/-- However many extra (failing) claims the threads make, exactly `n` jobs are granted. -/
theorem every_job_claimed_once (n extra : Nat) : successes (n + extra) n = n := by
  rw [successes_eq]
  omega

-- § equal row-range partition

/-- Start of part `i` when `N` rows are cut into `parts` ranges. -/
def lo (N parts i : Nat) : Nat := N * i / parts

/-- The first part starts at row 0. -/
theorem lo_zero (N parts : Nat) : lo N parts 0 = 0 := by
  simp [lo]

/-- The last boundary is `N` (when there is at least one part). -/
theorem lo_last (N parts : Nat) (hp : 0 < parts) : lo N parts parts = N := by
  unfold lo
  exact Nat.mul_div_cancel N hp

/-- Boundaries are monotone, so every range has non-negative width. -/
theorem lo_mono (N parts i j : Nat) (h : i ≤ j) : lo N parts i ≤ lo N parts j := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left N h)

/-- Consecutive ranges abut: the end of part `i` is the start of part `i+1` (they are the same expression). -/
theorem range_abut (N parts i : Nat) : N * (i + 1) / parts = lo N parts (i + 1) := by
  rfl

/-- Total width of the first `k` parts. -/
def widthSum (N parts : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => widthSum N parts k + (lo N parts (k + 1) - lo N parts k)

/-- The widths telescope: the first `k` parts cover exactly `lo k` rows. -/
theorem widthSum_eq (N parts k : Nat) : widthSum N parts k = lo N parts k := by
  induction k with
  | zero => simp [widthSum, lo]
  | succ k ih =>
    rw [widthSum, ih]
    have := lo_mono N parts k (k + 1) (Nat.le_succ k)
    omega

/-- All `parts` ranges together cover exactly `N` rows, with no gap and no overlap. -/
theorem partition_covers (N parts : Nat) (hp : 0 < parts) : widthSum N parts parts = N := by
  rw [widthSum_eq, lo_last N parts hp]

-- § task decomposition (modes 1 and 2)

/-- Task `i` is (expert `i / parts`, part `i % parts`). -/
def decomposeTask (i parts : Nat) : Nat × Nat := (i / parts, i % parts)

/-- The expert index is below the expert count. -/
theorem task_expert_lt (i n parts : Nat) (h : i < n * parts) : (decomposeTask i parts).1 < n := by
  show i / parts < n
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- The part index is below the part count. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : (decomposeTask i parts).2 < parts :=
  Nat.mod_lt i hp

/-- The decomposition recombines to the task index. -/
theorem task_recombine (i parts : Nat) : parts * (decomposeTask i parts).1 + (decomposeTask i parts).2 = i :=
  Nat.div_add_mod i parts

/-- Distinct tasks map to distinct (expert, part) pairs: no (expert, part) is run twice. -/
theorem task_decompose_injective (i j parts : Nat)
    (h : decomposeTask i parts = decomposeTask j parts) : i = j := by
  have h1 := Nat.div_add_mod i parts
  have h2 := Nat.div_add_mod j parts
  have hd : i / parts = j / parts := congrArg Prod.fst h
  have hm : i % parts = j % parts := congrArg Prod.snd h
  rw [hd, hm] at h1
  exact h1.symm.trans h2

-- § the multi-expert row walk (modes 3..6)

/-- The expert a global row belongs to. -/
theorem walk_expert_lt (r n per : Nat) (h : r < n * per) : r / per < n :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- The end of the current sub-range inside one expert. -/
def walkEnd (per m remaining : Nat) : Nat := min per (m + remaining)

/-- The sub-range is non-empty: the walk makes progress (`m` is `r % per`, `remaining` is `g1 - r`). -/
theorem walk_progress (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    m < walkEnd per m (g1 - r) := by
  unfold walkEnd
  omega

/-- The sub-range never leaves its expert. -/
theorem walk_within_expert (per m remaining : Nat) : walkEnd per m remaining ≤ per := by
  unfold walkEnd
  omega

/-- The sub-range never overshoots the task's end `g1`. -/
theorem walk_within_task (m per remaining : Nat) (hm : m < per) :
    walkEnd per m remaining - m ≤ remaining := by
  unfold walkEnd
  omega

/-- The walk advances by a positive amount, so it terminates after at most `g1 - g0` steps. -/
theorem walk_advances (m per g1 r : Nat) (hm : m < per) (hr : r < g1) :
    0 < walkEnd per m (g1 - r) - m := by
  unfold walkEnd
  omega

-- § the phase-row arithmetic

/-- Rows per expert in phases 3..6: gate/up rows `FF` for modes 3 and 5, down rows `H` otherwise. -/
def rowsPerExpert (mode ff h : Nat) : Nat := if mode = 3 ∨ mode = 5 then ff else h

/-- Rows per expert is positive whenever both widths are. -/
theorem rowsPerExpert_pos (mode ff h : Nat) (hf : 0 < ff) (hh : 0 < h) : 0 < rowsPerExpert mode ff h := by
  unfold rowsPerExpert
  split <;> omega

-- § worker park counter

/-- Park counter after `k` workers leave. -/
def afterLeave (parked k : Nat) : Nat := parked - k

/-- Park counter after `k` workers re-arrive. -/
def afterArrive (parked k : Nat) : Nat := parked + k

/-- After every worker leaves the park and every worker re-arrives, the counter is back at `n`:
the `done`-then-`parked` handover leaves the pool in the state `run` started from. -/
theorem park_roundtrip (n : Nat) : afterArrive (afterLeave n n) n = n := by
  unfold afterArrive afterLeave
  omega

/-- A host that is told `parked == n` can safely publish: the counter equals `n` only if no worker is inside the drain. -/
theorem parked_full_means_none_draining (n inDrain : Nat) (h : n - inDrain = n) (hn : 0 < n) :
    inDrain = 0 := by
  omega

end Strata.Kernels.Cpu.ExpertPoolScheduling
