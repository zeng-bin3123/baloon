/-!
# MODULE: expert_pool_protocol
SOURCES: include/strata/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
Abstract model of the CPU expert pool: row-range partitioning for split phases, the multi-expert flat row
index mapping, worker/thread counts, and the parked/active handover invariant of the completion protocol.

## Source semantics
* A split phase cuts `total` rows into `parts` ranges [total*p/parts, total*(p+1)/parts): they abut, start at 0 and
  end at `total`.
* A flat row r over several experts of `per` rows maps to expert r/per, local row r%per; each step makes progress
  and never crosses an expert or the task end.
* parked + active = n_workers always; `run()` waits for parked = n so no worker is in the drain loop when the
  host resets `head`.
* Workers >= 1; `physical_cores(true)` drops exactly the first core.

## Form template identifiers
lo/hi -> range_begin/range_end, PS -> park_state, Step -> worker_transition, Reach -> reachable
-/

namespace Strata.ExpertPool

-- § row-range partition
/-- Start row of part `p` of `parts`. -/
def lo (total parts p : Nat) : Nat := total * p / parts
/-- End row (exclusive) of part `p` of `parts`. -/
def hi (total parts p : Nat) : Nat := total * (p + 1) / parts

/-- The first range starts at row 0. -/
theorem lo_zero (total parts : Nat) : lo total parts 0 = 0 := by simp [lo]

/-- Consecutive ranges abut: the end of part p is the start of part p+1. -/
theorem hi_eq_lo_succ (total parts p : Nat) : hi total parts p = lo total parts (p + 1) := rfl

/-- The last range ends exactly at `total`. -/
theorem hi_last (total parts : Nat) (h : 0 < parts) : hi total parts (parts - 1) = total := by
  unfold hi
  have e : parts - 1 + 1 = parts := by omega
  rw [e]
  exact Nat.mul_div_cancel total h

/-- Each range is well-formed: begin <= end. -/
theorem lo_le_hi (total parts p : Nat) : lo total parts p ≤ hi total parts p := by
  unfold lo hi
  exact Nat.div_le_div_right (Nat.mul_le_mul_left total (Nat.le_succ p))

/-- Every range ends within `total`. -/
theorem hi_le_total (total parts p : Nat) (h0 : 0 < parts) (hp : p < parts) :
    hi total parts p ≤ total := by
  unfold hi
  calc total * (p + 1) / parts ≤ total * parts / parts :=
        Nat.div_le_div_right (Nat.mul_le_mul_left total hp)
    _ = total := Nat.mul_div_cancel total h0

-- § flat multi-expert row mapping
/-- Expert index and local row recombine to the flat row, and the local row is in range. -/
theorem flat_row_split (r per : Nat) (hper : 0 < per) :
    per * (r / per) + r % per = r ∧ r % per < per :=
  ⟨Nat.div_add_mod r per, Nat.mod_lt r hper⟩

/-- A flat row below per*n belongs to an expert index below n. -/
theorem expert_index_lt (r per n : Nat) (h : r < per * n) : r / per < n :=
  Nat.div_lt_of_lt_mul h

/-- Each inner step takes at least one row. -/
theorem step_progress (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r < g1) :
    r0 < min per (r0 + (g1 - r)) := by omega

/-- The step's end never passes the expert's last row. -/
theorem step_within_expert (per r0 g1 r : Nat) : min per (r0 + (g1 - r)) ≤ per := by omega

/-- The step never overshoots the task's end row `g1`. -/
theorem step_within_task (per r0 g1 r : Nat) (h0 : r0 < per) (hr : r ≤ g1) :
    r + (min per (r0 + (g1 - r)) - r0) ≤ g1 := by omega

-- § worker and thread counts
/-- Worker count: the request if positive, else the core count, never below one. -/
def workersOf (nw cores : Nat) : Nat := max 1 (if 0 < nw then nw else cores)

/-- At least one worker always exists. -/
theorem workersOf_pos (nw cores : Nat) : 1 ≤ workersOf nw cores := Nat.le_max_left _ _

/-- Thread count including the host when it also drains. -/
def threadsOf (n : Nat) (hostWorks : Bool) : Nat := n + if hostWorks then 1 else 0

/-- With `host_works` the pool is n+1 threads. -/
theorem threads_with_host (n : Nat) : threadsOf n true = n + 1 := rfl

/-- Without `host_works` the pool is exactly the workers. -/
theorem threads_without_host (n : Nat) : threadsOf n false = n := rfl

/-- Skipping the first core removes exactly one entry (when any exists). -/
theorem skip_first_len (cores : List Nat) : (cores.drop 1).length = cores.length - 1 := by simp

/-- Parts per expert in `run_split` is at least one. -/
def partsOf (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- `partsOf` is positive. -/
theorem partsOf_pos (threads n : Nat) : 1 ≤ partsOf threads n := Nat.le_max_left _ _

-- § park / drain handover
/-- Workers parked versus inside the drain loop. -/
structure PS where
  parked : Nat
  active : Nat

/-- A worker leaves the park, or returns to it. -/
inductive Step : PS → PS → Prop
  | leave (p a : Nat) : 0 < p → Step ⟨p, a⟩ ⟨p - 1, a + 1⟩
  | arrive (p a : Nat) : 0 < a → Step ⟨p, a⟩ ⟨p + 1, a - 1⟩

/-- Reachability by worker transitions. -/
inductive Reach (s : PS) : PS → Prop
  | refl : Reach s s
  | tail {t u : PS} : Reach s t → Step t u → Reach s u

/-- One transition preserves parked + active. -/
theorem step_preserves {s t : PS} (h : Step s t) : s.parked + s.active = t.parked + t.active := by
  cases h with
  | leave p a hp =>
      show p + a = (p - 1) + (a + 1)
      omega
  | arrive p a ha =>
      show p + a = (p + 1) + (a - 1)
      omega

/-- Reachable states preserve parked + active. -/
theorem reach_preserves {s t : PS} (h : Reach s t) : s.parked + s.active = t.parked + t.active := by
  induction h with
  | refl => rfl
  | tail _ hs ih => exact ih.trans (step_preserves hs)

/-- Handover safety: starting with all n workers parked, `parked = n` implies no worker is in the drain loop. -/
theorem handover_safe (n : Nat) (t : PS) (h : Reach ⟨n, 0⟩ t) (hp : t.parked = n) : t.active = 0 := by
  have e : n + 0 = t.parked + t.active := reach_preserves h
  omega

/-- Parked never exceeds the worker count. -/
theorem parked_le (n : Nat) (t : PS) (h : Reach ⟨n, 0⟩ t) : t.parked ≤ n := by
  have e : n + 0 = t.parked + t.active := reach_preserves h
  omega

end Strata.ExpertPool
