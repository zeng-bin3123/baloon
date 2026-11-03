/-!
# MODULE: expert_pool
Work partitioning and counter protocol of the CPU expert pool (`pool.hpp`, `pool.cpp`).

## Source semantics
A batch is split into row ranges `[total*i/parts, total*(i+1)/parts)`; ranges are adjacent and cover
the total exactly. A thread walks its global row range expert by expert; each step makes progress
and stays within its expert and its range. Jobs are claimed with a single `fetch_add` counter.

## Form template identifiers
* `FF * part / parts_a_` → `splitPoint 640 parts part`; `mrows_ * i / mtasks_` → `splitPoint rows tasks i`
* `head_` → `head`; `physical_cores(skip_first)` → `dropFirst`
-/

namespace StrataExpertPool

/-- Boundary `i` of a split of `total` rows into `parts` ranges. -/
def splitPoint (total parts i : Nat) : Nat := total * i / parts

-- § row range split

/-- The first range starts at zero. -/
theorem splitPoint_zero (total parts : Nat) : splitPoint total parts 0 = 0 := by
  simp [splitPoint]

/-- The last range ends at `total`. -/
theorem splitPoint_last (total parts : Nat) (h : 0 < parts) : splitPoint total parts parts = total := by
  unfold splitPoint; exact Nat.mul_div_cancel total h

/-- Boundaries are monotone. -/
theorem splitPoint_mono (total parts i j : Nat) (h : i ≤ j) :
    splitPoint total parts i ≤ splitPoint total parts j := by
  unfold splitPoint
  exact Nat.div_le_div_right (Nat.mul_le_mul_left total h)

/-- Every boundary within `0..parts` is at most `total`. -/
theorem splitPoint_le_total (total parts i : Nat) (h0 : 0 < parts) (hi : i ≤ parts) :
    splitPoint total parts i ≤ total := by
  have h := splitPoint_mono total parts i parts hi
  rw [splitPoint_last total parts h0] at h
  exact h

/-- Adjacent ranges: range `i` ends where range `i+1` begins, and has non-negative size. -/
theorem splitPoint_adjacent (total parts i : Nat) :
    splitPoint total parts i + (splitPoint total parts (i + 1) - splitPoint total parts i)
      = splitPoint total parts (i + 1) := by
  have h := splitPoint_mono total parts i (i + 1) (Nat.le_succ i)
  omega

-- § row walk across experts

/-- One walk step (`r0 = r % per`, `r1 = min per (r0 + (g1 - r))`) makes progress,
stays inside the expert and does not pass the range end. -/
theorem walk_step (per r g1 m : Nat) (hm : m = r % per) (hp : 0 < per) (hr : r < g1) :
    m < min per (m + (g1 - r)) ∧ min per (m + (g1 - r)) ≤ per ∧
      r + (min per (m + (g1 - r)) - m) ≤ g1 := by
  have h := Nat.mod_lt r hp
  rw [← hm] at h
  omega

/-- A row decomposes as `expert * per + offset`. -/
theorem walk_decomp (per r : Nat) (hp : 0 < per) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

-- § sizing

/-- Worker count: the request if positive, else the physical cores (at least one). -/
def workerCount (requested : Int) (cores : Nat) : Nat :=
  if 0 < requested then requested.toNat else max cores 1

/-- The pool always has at least one worker. -/
theorem workerCount_pos (requested : Int) (cores : Nat) : 1 ≤ workerCount requested cores := by
  unfold workerCount
  split <;> omega

/-- Core list after the optional reservation of the first core for the host loop. -/
def dropFirst (skip : Bool) (cores : List Nat) : List Nat :=
  if skip then cores.tail else cores

/-- Reserving a core removes exactly one entry. -/
theorem dropFirst_length (cores : List Nat) : (dropFirst true cores).length = cores.length - 1 := by
  simp [dropFirst]

/-- Total threads in a drain: workers plus optionally the host. -/
def threadCount (n : Nat) (hostWorks : Bool) : Nat := n + if hostWorks then 1 else 0

/-- Task parts per expert in `run_split`. -/
def partsA (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsA_pos (threads n : Nat) : 1 ≤ partsA threads n := by
  unfold partsA; exact Nat.le_max_left _ _

/-- `run_split_multi` cuts each phase into `3 * threads` tasks, at least three. -/
theorem multi_tasks_ge (n : Nat) (host : Bool) (h : 1 ≤ n) : 3 ≤ 3 * threadCount n host := by
  unfold threadCount; split <;> omega

-- § claim counter

/-- `fetch_add(1)` on the head counter: returns the old value and the new head. -/
def claim (head : Nat) : Nat × Nat := (head, head + 1)

/-- Each claim hands out a fresh index below the new head. -/
theorem claim_fresh (head : Nat) : (claim head).1 < (claim head).2 := by
  simp [claim]

/-- Consecutive claims never hand out the same index. -/
theorem claim_distinct (head : Nat) : (claim head).1 ≠ (claim (claim head).2).1 := by
  simp [claim]

end StrataExpertPool
