/-!
# MODULE: cpu_expert_pool_scheduling
Work partitioning, job claiming and affinity bookkeeping of the CPU expert thread pool.

## Source semantics
* A phase splits `rows` rows across `tasks` tasks with task `i` owning `[rows*i/tasks, rows*(i+1)/tasks)`; the
  ranges are contiguous, start at 0 and end at `rows`.
* Inside a task, rows are walked expert by expert (`per` rows per expert); each step makes positive progress,
  stays inside one expert and never passes the task's end; the steps cover the whole range.
* Jobs are claimed with one `fetch_add` on a shared head; with `T` total claims over `n` jobs exactly
  `min T n` succeed, so `done_` reaches `n` once every thread has made its failing claim.
* The pool has at least one worker; `skip_first` removes core 0 from the core list.
* The Linux branch of `physical_cores` has a dangling `else`: it binds to `if (CPU_ISSET(i, &set))`, so
  every CPU NOT in the affinity mask appends `hardware_concurrency()` entries (modelled below).
* `pin_current_thread` returns the previous mask truncated to 64 CPUs as a signed value; a mask with bit 63
  set is negative and `restore_thread_affinity` skips it (`previous <= 0`).

## Form template identifiers
run_phase → run_phase, drain → claim_and_process, head_ → job_head, done_ → done_count, parked_ → parked_count,
mrows_/mtasks_ → phase_rows/phase_tasks, physical_cores → core_enumeration, pin_current_thread → pin_thread.
-/

namespace Strata.Cpu.Pool

-- § row partition across tasks

/-- Start of task `i`'s row range. -/
def lo (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem lo_zero (rows tasks : Nat) : lo rows tasks 0 = 0 := by simp [lo]

/-- The last boundary is `rows` when there is at least one task. -/
theorem lo_last (rows tasks : Nat) (h : 0 < tasks) : lo rows tasks tasks = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Boundaries are monotone, so every task's range is well formed (`g0 ≤ g1`). -/
theorem lo_mono (rows tasks i j : Nat) (h : i ≤ j) : lo rows tasks i ≤ lo rows tasks j := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows h)

/-- Total rows of the first `k` tasks. -/
def total (rows tasks : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => total rows tasks k + (lo rows tasks (k + 1) - lo rows tasks k)

/-- The first `k` tasks together own exactly `lo k` rows: ranges are contiguous with no gap or overlap. -/
theorem total_eq_lo (rows tasks k : Nat) : total rows tasks k = lo rows tasks k := by
  induction k with
  | zero => simp [total, lo]
  | succ k ih =>
    have hm := lo_mono rows tasks k (k + 1) (by omega)
    simp only [total, ih]
    omega

/-- All boundaries are at most `rows` for `i ≤ tasks`. -/
theorem lo_le_rows (rows tasks i : Nat) (h : 0 < tasks) (hi : i ≤ tasks) : lo rows tasks i ≤ rows := by
  have := lo_mono rows tasks i tasks hi
  rw [lo_last rows tasks h] at this
  exact this

-- § walking a task's range expert by expert

/-- The steps `(expert, r0, r1)` the inner loop performs, bounded by fuel. -/
def walk (per g : Nat) : Nat → Nat → List (Nat × Nat × Nat)
  | 0, _ => []
  | fuel + 1, r =>
    if r < g then
      (r / per, r % per, min per (r % per + (g - r))) ::
        walk per g fuel (r + (min per (r % per + (g - r)) - r % per))
    else []

/-- A single step makes positive progress, stays within an expert and does not pass the task end. -/
theorem step_progress (per r0 g r : Nat) (hr : r < g) (hp : r0 < per) :
    r0 < min per (r0 + (g - r)) ∧ min per (r0 + (g - r)) ≤ per ∧
      r + (min per (r0 + (g - r)) - r0) ≤ g := by
  omega

/-- The expert index of a global row stays below the number of experts. -/
theorem expert_index_lt (r per n : Nat) (h : r < n * per) : r / per < n := by
  by_cases hp : per = 0
  · subst hp; simp at h
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)

/-- The sum of the step widths equals the range length, given enough fuel. -/
theorem walk_cover (per g : Nat) (hp : 0 < per) :
    ∀ fuel r, g - r ≤ fuel → ((walk per g fuel r).map (fun t => t.2.2 - t.2.1)).sum = g - r := by
  intro fuel
  induction fuel with
  | zero =>
    intro r h
    simp [walk] <;> omega
  | succ f ih =>
    intro r h
    by_cases hr : r < g
    · have h0 : r % per < per := Nat.mod_lt _ hp
      have key := ih (r + (min per (r % per + (g - r)) - r % per)) (by omega)
      simp only [walk, hr, if_true, List.map_cons, List.sum_cons]
      rw [key]
      omega
    · simp [walk, hr] <;> omega

-- § phase decode: task index -> (expert, part)

/-- For `i < n * parts`, the decoded expert is `< n` and the part is `< parts`. -/
theorem decode_task (i n parts : Nat) (hp : 0 < parts) (h : i < n * parts) :
    i / parts < n ∧ i % parts < parts := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  · exact Nat.mod_lt _ hp

/-- `parts_a_ = max 1 ((3 * threads + n - 1) / n)` is positive. -/
theorem parts_positive (threads n : Nat) : 0 < max 1 ((3 * threads + n - 1) / n) := by omega

-- § job claiming

/-- Number of successful claims among the first `T` `fetch_add` results `0, 1, …, T-1`. -/
def claimOk (n : Nat) : Nat → Nat
  | 0 => 0
  | t + 1 => claimOk n t + (if t < n then 1 else 0)

/-- Over `n` jobs and `T` claims exactly `min T n` succeed. -/
theorem claimOk_eq (n T : Nat) : claimOk n T = min T n := by
  induction T with
  | zero => simp [claimOk]
  | succ t ih =>
    simp only [claimOk, ih]
    split <;> omega

/-- Once the claims number at least `n` plus one failing claim per thread, all `n` jobs were claimed once. -/
theorem all_jobs_claimed_once (n T : Nat) (h : n ≤ T) : claimOk n T = n := by
  rw [claimOk_eq]; omega

-- § pool sizing and core list

/-- Effective worker count: the requested count if positive, else the core count, never below 1. -/
def poolSize (requested cores : Nat) : Nat := max 1 (if 0 < requested then requested else cores)

/-- The pool always has at least one worker, so a single-core machine still produces a token. -/
theorem poolSize_pos (requested cores : Nat) : 1 ≤ poolSize requested cores := by
  unfold poolSize
  omega

/-- Dropping core 0 from a non-empty core list shortens it by exactly one. -/
theorem skip_first_len (cores : List Nat) (h : cores ≠ []) : (cores.drop 1).length = cores.length - 1 := by
  simp

-- § the dangling else in the Linux branch of physical_cores

/-- What the source actually builds: unset CPUs expand to `range hc`. Parameters: hc, index, mask. -/
def actualAux (hc : Nat) : Nat → List Bool → List Nat
  | _, [] => []
  | i, b :: bs => (if b then [i] else List.range hc) ++ actualAux hc (i + 1) bs

/-- What the indentation suggests: just the set CPUs. -/
def intendedAux : Nat → List Bool → List Nat
  | _, [] => []
  | i, b :: bs => (if b then [i] else []) ++ intendedAux (i + 1) bs

/-- When every CPU is in the mask the two coincide. -/
theorem actual_eq_intended_when_all_set (hc : Nat) (bs : List Bool) (h : ∀ b ∈ bs, b = true) :
    ∀ i, actualAux hc i bs = intendedAux i bs := by
  induction bs with
  | nil => intro i; rfl
  | cons b bs ih =>
    intro i
    have hb : b = true := h b (by simp)
    subst hb
    have h' : ∀ b ∈ bs, b = true := fun x hx => h x (by simp [hx])
    simp only [actualAux, intendedAux, if_true]
    rw [ih h' (i + 1)]

/-- Witness that they differ: with one unset CPU and `hardware_concurrency = 2`, the list is `[0,0,1]`
instead of `[0]`, so a pinned worker would be placed on cores 0 and 1 twice. -/
theorem dangling_else_witness :
    actualAux 2 0 [true, false] = [0, 0, 1] ∧ intendedAux 0 [true, false] = [0] := by decide

-- § affinity mask round trip

/-- Interpret a 64-bit mask as the signed `long long` the source returns. -/
def asSigned (m : Int) : Int := if m < 9223372036854775808 then m else m - 18446744073709551616

/-- A mask with bit 63 set becomes negative. -/
theorem high_bit_mask_negative (m : Int) (h1 : 9223372036854775808 ≤ m) (h2 : m < 18446744073709551616) :
    asSigned m < 0 := by
  unfold asSigned
  split <;> omega

/-- A negative stored mask makes `restore_thread_affinity` a no-op (`previous <= 0` returns early),
so an affinity containing CPU 63 is never restored. -/
theorem high_bit_mask_not_restored (m : Int) (h1 : 9223372036854775808 ≤ m) (h2 : m < 18446744073709551616) :
    asSigned m ≤ 0 := by
  have := high_bit_mask_negative m h1 h2
  omega

/-- Masks below bit 63 round-trip as non-negative values. -/
theorem low_mask_roundtrips (m : Int) (h : 0 ≤ m) (h2 : m < 9223372036854775808) : asSigned m = m := by
  unfold asSigned
  simp [h2]

end Strata.Cpu.Pool
