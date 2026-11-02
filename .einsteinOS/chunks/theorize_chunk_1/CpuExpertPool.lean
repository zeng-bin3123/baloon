/-!
# MODULE: cpu_expert_pool
Work partitioning of the CPU expert pool: equal row ranges per task, and one thread per physical core.

## Source semantics
`run_split` cuts a phase of `rows` rows into `n` equal ranges `[i*rows/n, (i+1)*rows/n)`; they are contiguous, ordered,
and cover `[0, rows)` exactly. The pool is `cores - 1` workers plus the host when `host_works`, i.e. one thread per
physical core.

## Form template identifiers
- ExpertPool::run_split_multi (mtasks_, mrows_) -> rowLo / rowHi
- physical_cores(skip_first), host_works        -> poolThreads
-/
namespace Strata.CpuExpertPool

def rowLo (rows n i : Nat) : Nat := i * rows / n
def rowHi (rows n i : Nat) : Nat := (i + 1) * rows / n

/-- Task 0 starts at row 0. -/
theorem rowLo_zero (rows n : Nat) : rowLo rows n 0 = 0 := by simp [rowLo]

/-- Consecutive tasks abut: no row is skipped or read twice. -/
theorem rowHi_eq_rowLo_succ (rows n i : Nat) : rowHi rows n i = rowLo rows n (i + 1) := rfl

/-- The last task ends at the last row. -/
theorem rowHi_last {rows n : Nat} (h : 0 < n) : rowHi rows n (n - 1) = rows := by
  unfold rowHi
  have : n - 1 + 1 = n := by omega
  rw [this]
  exact Nat.mul_div_cancel_left rows h

/-- Each task's range is non-empty-or-empty but never inverted. -/
theorem rowLo_le_rowHi (rows n i : Nat) : rowLo rows n i ≤ rowHi rows n i := by
  unfold rowLo rowHi
  exact Nat.div_le_div_right (Nat.mul_le_mul_right rows (by omega))

/-- Row ranges stay inside the phase. -/
theorem rowHi_le {rows n i : Nat} (h : i < n) : rowHi rows n i ≤ rows := by
  unfold rowHi
  rcases Nat.eq_zero_or_pos n with hn | hn
  · omega
  · have h1 : (i + 1) * rows ≤ n * rows := Nat.mul_le_mul_right rows (by omega)
    have h2 : (i + 1) * rows / n ≤ n * rows / n := Nat.div_le_div_right h1
    rw [Nat.mul_div_cancel_left rows hn] at h2
    exact h2

/-- Workers plus the working host are exactly the physical cores (when at least one core exists). -/
def poolThreads (cores : Nat) (hostWorks : Bool) : Nat := (cores - 1) + (if hostWorks then 1 else 0)

/-- With `host_works` every physical core carries one thread. -/
theorem poolThreads_host {cores : Nat} (h : 1 ≤ cores) : poolThreads cores true = cores := by
  unfold poolThreads
  simp
  omega

/-- Without `host_works` core 0 stays free for the host's doorbell spin. -/
theorem poolThreads_nohost {cores : Nat} (h : 1 ≤ cores) : poolThreads cores false + 1 = cores := by
  unfold poolThreads
  simp
  omega

end Strata.CpuExpertPool
