/-!
# MODULE: cpu_expert_pool
SOURCES: src/kernels/cpu/pool.cpp
The CPU expert pool: physical-core selection and pinning, worker-count rules, and the work partitioning of
the split phases (equal row ranges across tasks, the inner per-expert segment loop, and batching of native
jobs). This module models the integer partition contracts that make every row computed exactly once.

## Source semantics
* `physical_cores(skip_first)` drops the first core (the host's) so workers never share it.
* The pool has at least one worker; a pin core exists only when pinning is on and the index is in range.
* Phase task `i` of `tasks` over `rows` rows covers `[rows*i/tasks, rows*(i+1)/tasks)`: these ranges tile
  `[0, rows)`.
* The inner loop walks a range in segments that never cross an expert boundary and always make progress,
  covering exactly the range.
* Batches of at most `kMaxSplitMulti` experts cover all `n` experts.

## Form template identifiers
* `coresFor`       : physical_core_list
* `getCore`        : worker_pin_core
* `nWorkers`       : worker_count
* `lo`             : task_row_lo
* `stepLen`, `cover` : segment_walk
* `batches`        : native_batching
-/

namespace Ref.CpuExpertPool

-- § core selection

/-- Cores after optionally dropping the host's core. -/
def coresFor (skipFirst : Bool) (cores : List Nat) : List Nat := if skipFirst then cores.tail else cores

/-- Dropping the first core removes exactly one entry from a non-empty list. -/
theorem coresFor_length (cores : List Nat) (h : cores ≠ []) :
    (coresFor true cores).length = cores.length - 1 := by
  unfold coresFor
  simp

/-- Without skipping the list is unchanged. -/
theorem coresFor_noskip (cores : List Nat) : coresFor false cores = cores := by
  simp [coresFor]

/-- With distinct core ids the host core is never handed to a worker. -/
theorem host_core_unused (c : Nat) (cs : List Nat) (h : (c :: cs).Nodup) : c ∉ coresFor true (c :: cs) := by
  unfold coresFor
  simpa using (List.nodup_cons.mp h).1

/-- Pin core of worker `i`: only when pinning is on and `i` is in range. -/
def getCore (pin : Bool) (cores : List Nat) (i : Nat) : Option Nat := if pin then cores[i]? else none

/-- No pinning, no core. -/
theorem getCore_nopin (cores : List Nat) (i : Nat) : getCore false cores i = none := by
  simp [getCore]

/-- Out-of-range workers are left unpinned. -/
theorem getCore_oob (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : getCore true cores i = none := by
  simp [getCore, h]

-- § worker count

/-- Requested workers, else the core count, but never fewer than one. -/
def nWorkers (req nCores : Nat) : Nat := if (if 0 < req then req else nCores) < 1 then 1 else (if 0 < req then req else nCores)

/-- The pool always has a worker, so a single-core machine still produces a token. -/
theorem nWorkers_pos (req nCores : Nat) : 1 ≤ nWorkers req nCores := by
  unfold nWorkers
  by_cases h : 0 < req
  · rw [if_pos h]
    by_cases h2 : req < 1
    · rw [if_pos h2] <;> omega
    · rw [if_neg h2] <;> omega
  · rw [if_neg h]
    by_cases h2 : nCores < 1
    · rw [if_pos h2] <;> omega
    · rw [if_neg h2] <;> omega

/-- An explicit positive request is honoured. -/
theorem nWorkers_req (req nCores : Nat) (h : 0 < req) : nWorkers req nCores = req := by
  unfold nWorkers
  rw [if_pos h]
  have h2 : ¬ req < 1 := by omega
  rw [if_neg h2]

-- § task row ranges

/-- Start row of task `i` when `rows` rows are split over `tasks` tasks. -/
def lo (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem lo_zero (rows tasks : Nat) : lo rows tasks 0 = 0 := by
  simp [lo]

/-- The last boundary is exactly `rows`. -/
theorem lo_last (rows tasks : Nat) (h : 0 < tasks) : lo rows tasks tasks = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Ranges are ordered: a task's end is not before its start. -/
theorem lo_mono (rows tasks i : Nat) : lo rows tasks i ≤ lo rows tasks (i + 1) := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul (Nat.le_refl rows) (Nat.le_succ i))

/-- Adjacent tasks share a boundary, so the ranges tile with no gap and no overlap (by definition of `lo`). -/
theorem lo_adjacent (rows tasks i : Nat) : lo rows tasks (i + 1) = rows * (i + 1) / tasks := by
  rfl

/-- A flat task id `i < n*parts` splits into an expert `i/parts < n` and a part `i%parts < parts`. -/
theorem task_split (i n parts : Nat) (hp : 0 < parts) (hi : i < n * parts) :
    i / parts < n ∧ i % parts < parts ∧ parts * (i / parts) + i % parts = i := by
  refine ⟨?_, Nat.mod_lt _ hp, Nat.div_add_mod i parts⟩
  exact (Nat.div_lt_iff_lt_mul hp).2 hi

/-- The number of parts per expert is at least one. -/
theorem parts_pos (threads n : Nat) : 1 ≤ max 1 ((3 * threads + n - 1) / n) :=
  Nat.le_max_left 1 _

-- § inner segment walk

/-- Length of the segment taken from row `r` of a range ending at `g1`, in experts of `per` rows. -/
def stepLen (per g1 r : Nat) : Nat := min per (r % per + (g1 - r)) - r % per

/-- Each segment is non-empty while rows remain. -/
theorem stepLen_pos (per g1 r : Nat) (hper : 0 < per) (hr : r < g1) : 0 < stepLen per g1 r := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- A segment never overruns the range end. -/
theorem stepLen_le_range (per g1 r : Nat) (hper : 0 < per) (hr : r < g1) : r + stepLen per g1 r ≤ g1 := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- A segment never crosses its expert's boundary: the offset in the expert plus the length is at most `per`. -/
theorem stepLen_in_expert (per g1 r : Nat) (hper : 0 < per) : r % per + stepLen per g1 r ≤ per := by
  have hm : r % per < per := Nat.mod_lt _ hper
  unfold stepLen
  generalize r % per = r0 at *
  omega

/-- The expert and in-expert offset reassemble the row. -/
theorem row_decomp (r per : Nat) : per * (r / per) + r % per = r :=
  Nat.div_add_mod r per

/-- Rows processed by walking from `r` to `g1` with fuel `f`. -/
def cover (per g1 : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, r => if r < g1 then stepLen per g1 r + cover per g1 f (r + stepLen per g1 r) else 0

/-- The walk covers exactly the `g1 - r` remaining rows when the fuel suffices. -/
theorem cover_eq (per g1 : Nat) (hper : 0 < per) (f : Nat) :
    ∀ r, r ≤ g1 → g1 - r ≤ f → cover per g1 f r = g1 - r := by
  induction f with
  | zero =>
    intro r h1 h2
    have : g1 - r = 0 := by omega
    rw [this]
    rfl
  | succ f ih =>
    intro r h1 h2
    have e : cover per g1 (f + 1) r =
        if r < g1 then stepLen per g1 r + cover per g1 f (r + stepLen per g1 r) else 0 := rfl
    rw [e]
    by_cases hr : r < g1
    · rw [if_pos hr]
      have hp := stepLen_pos per g1 r hper hr
      have hl := stepLen_le_range per g1 r hper hr
      rw [ih (r + stepLen per g1 r) hl (by omega)]
      omega
    · rw [if_neg hr]
      omega

-- § native batching

/-- Sizes of the batches of at most `K` experts a loop over `n` experts produces, with fuel `f`. -/
def batches (K : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min K rem :: batches K f (rem - min K rem)

/-- The batches cover all `n` experts. -/
theorem batches_sum (K : Nat) (f : Nat) : ∀ rem, rem ≤ K * f → (batches K f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    rw [Nat.mul_succ] at h
    by_cases hr : rem = 0
    · subst hr
      simp [batches]
    · have e : batches K (f + 1) rem = min K rem :: batches K f (rem - min K rem) := by
        simp [batches, hr]
      rw [e, List.sum_cons, ih (rem - min K rem) (by omega)]
      omega

/-- A batch starting at `b0 < n` is non-empty and stays inside `n`. -/
theorem batch_bounds (K n b0 : Nat) (hK : 0 < K) (h : b0 < n) :
    1 ≤ min K (n - b0) ∧ b0 + min K (n - b0) ≤ n := by
  refine ⟨?_, ?_⟩ <;> omega

-- § affinity bookkeeping

/-- Windows `SetThreadAffinityMask` returns 0 on failure, so 0 maps to the error value -1. -/
def winPin (prev : Int) : Int := if prev = 0 then -1 else prev

/-- The error value is returned exactly when the OS call failed. -/
theorem winPin_err (prev : Int) : winPin prev = -1 ↔ prev = 0 ∨ prev = -1 := by
  unfold winPin
  split <;> omega

/-- A successful pin never yields the ambiguous value 0, so a returned mask is always restorable. -/
theorem winPin_ne_zero (prev : Int) : winPin prev ≠ 0 := by
  unfold winPin
  split <;> omega

/-- Restoring a non-positive previous mask is a no-op. -/
def restoreEffect (prev : Int) : Option Int := if prev ≤ 0 then none else some prev

/-- Error and unset masks are never restored. -/
theorem restore_noop (prev : Int) (h : prev ≤ 0) : restoreEffect prev = none := by
  simp [restoreEffect, h]

/-- Group/bit processor numbering: `group*64 + bit` decomposes back for `bit < 64`. -/
theorem cpu_index_roundtrip (g b : Nat) (hb : b < 64) : (g * 64 + b) / 64 = g ∧ (g * 64 + b) % 64 = b := by
  refine ⟨?_, ?_⟩ <;> omega

/-- `core & 63` is a valid bit position. -/
theorem core_bit_lt (core : Nat) : core % 64 < 64 := by
  omega

end Ref.CpuExpertPool
