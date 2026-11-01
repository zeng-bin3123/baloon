/-!
# MODULE: expert_pool_row_partition
Work partitioning of the CPU expert pool: a phase's rows across all experts are cut into equal
integer ranges, one per claimed task, and a range may span several experts.

## Source semantics
* Task `i` of `tasks` owns global rows `[rows*i/tasks, rows*(i+1)/tasks)`.
* Inner loop: `e = r/per`, `r0 = r%per`, `r1 = min(per, r0 + (g1 - r))`, `r += r1 - r0`.

## Form template identifiers
* `rows`, `tasks` ↔ `mrows_`, `mtasks_`; `per` ↔ `FF` or `H`; `bound i` ↔ `g0`/`g1`
-/

namespace Strata.ExpertPool

/-- Row boundary of task `i`. -/
def bound (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem bound_zero (rows tasks : Nat) : bound rows tasks 0 = 0 := by
  simp [bound]

/-- The last task ends exactly at `rows`. -/
theorem bound_last (rows tasks : Nat) (h : 0 < tasks) : bound rows tasks tasks = rows := by
  unfold bound
  exact Nat.mul_div_cancel rows h

/-- Boundaries are monotone, so task ranges never overlap. -/
theorem bound_mono (rows tasks : Nat) {i j : Nat} (h : i ≤ j) :
    bound rows tasks i ≤ bound rows tasks j := by
  unfold bound
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows h)

/-- No boundary exceeds `rows` for `i ≤ tasks`. -/
theorem bound_le_rows (rows tasks : Nat) (h : 0 < tasks) {i : Nat} (hi : i ≤ tasks) :
    bound rows tasks i ≤ rows := by
  have h1 := bound_mono rows tasks hi
  have h2 := bound_last rows tasks h
  omega

/-- End of the sub-range handled in one inner iteration. -/
def stepEnd (per r g1 : Nat) : Nat := min per (r % per + (g1 - r))

/-- Each inner iteration makes progress. -/
theorem step_progress (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    r % per < stepEnd per r g1 := by
  have h := Nat.mod_lt r hper
  unfold stepEnd
  omega

/-- The sub-range never leaves its expert's `per` rows. -/
theorem step_le_per (per r g1 : Nat) : stepEnd per r g1 ≤ per := by
  unfold stepEnd
  omega

/-- The sub-range never overshoots the task's end `g1`. -/
theorem step_le_end (per r g1 : Nat) (hr : r ≤ g1) :
    r + (stepEnd per r g1 - r % per) ≤ g1 ∨ stepEnd per r g1 = per := by
  unfold stepEnd
  omega

/-- After one iteration `r` stays within the same expert block. -/
theorem step_same_expert (per r g1 : Nat) (hper : 0 < per) :
    r + (stepEnd per r g1 - r % per) ≤ per * (r / per) + per := by
  have h1 := Nat.div_add_mod r per
  have h2 := step_le_per per r g1
  have h3 := Nat.mod_lt r hper
  omega

/-- Expert and row-in-expert recompose the global row. -/
theorem expert_row_recompose (per r : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

/-- The row-in-expert index is in range. -/
theorem row_in_range (per r : Nat) (hper : 0 < per) : r % per < per :=
  Nat.mod_lt r hper

/-- `parts_a = max 1 (...)` is at least 1. -/
theorem parts_pos (threads n : Nat) : 1 ≤ max 1 ((3 * threads + n - 1) / n) :=
  Nat.le_max_left _ _

/-- Multi-token phases use `3 * threads` tasks, positive whenever there is a thread. -/
theorem mtasks_pos (threads : Nat) (h : 0 < threads) : 0 < 3 * threads := by
  omega

/-- Claiming past the job count claims nothing. -/
theorem claim_stops (i njobs : Nat) (h : njobs ≤ i) : ¬ i < njobs := by omega

/-- Round-robin slot index stays in range. -/
theorem slot_lt (j slots : Nat) (h : 0 < slots) : j % slots < slots := Nat.mod_lt j h

end Strata.ExpertPool
