/-!
# MODULE: expert_pool_scheduling
SOURCES: src/kernels/cpu/pool.cpp
The work distribution of `ExpertPool`: the shared `head_` claim counter, the per-expert row split of
`run_split` (modes 1/2), the global row-range split of `run_split_multi` / `run_split_multi_native`
(modes 3–6) whose ranges may span two experts, the batching of experts beyond the buffer count, and the
park counter that makes every `run` start and end in the same state.

## Source semantics
* Claims: each thread does `i = head_.fetch_add(1)` and stops at `i >= njobs`; claims are 0,1,2,…, so
  exactly the jobs 0..njobs−1 run, each once, however many threads claim.
* Task i of `ntasks` covers global rows [rows·i/ntasks, rows·(i+1)/ntasks): consecutive, disjoint,
  covering [0, rows).
* Inside a task, global row r decomposes as expert e = r / per, local row r0 = r % per; the piece ends
  at r1 = min(per, r0 + (g1 − r)); each step makes progress and never passes g1.
* `run_split`: parts = max(1, ⌈3·threads / n⌉) per expert; job i is expert i/parts, part i%parts,
  rows [FF·part/parts, FF·(part+1)/parts).
* Parking: workers arrive at the park before the first wait, leave on a new epoch, return after draining;
  `parked == n` before every publish.

## Form template identifiers
claims → headClaims, bound → taskBoundary, partsFor → partsPerExpert, decompose → globalRowSplit,
batchSizes → expertBatches, parkedAfter → parkCounter
-/

namespace Strata.Kernels.Cpu.Pool

-- § the head_ claim counter

/-- Of the first m claims (values 0..m−1), those below `njobs` do work. -/
def worked (m njobs : Nat) : List Nat := (List.range m).filter (· < njobs)

/-- Exactly min(m, njobs) claims do work: every job once, late claimers get nothing. -/
theorem worked_length (m njobs : Nat) : (worked m njobs).length = min m njobs := by
  induction m with
  | zero => simp [worked]
  | succ k ih =>
    simp only [worked, List.range_succ, List.filter_append] at ih ⊢
    rw [List.length_append, ih]
    by_cases h : k < njobs
    · simp [h]; omega
    · simp [h]; omega

/-- A worked claim is a valid job index. -/
theorem worked_valid (m njobs i : Nat) (h : i ∈ worked m njobs) : i < njobs := by
  simp only [worked, List.mem_filter, decide_eq_true_eq] at h
  exact h.2

/-- Claims are distinct (each job runs at most once). -/
theorem worked_nodup (m njobs : Nat) : (worked m njobs).Nodup :=
  List.filter_sublist.nodup List.nodup_range

-- § the even global row split

/-- Boundary of task i: `rows·i / ntasks`. -/
def bound (rows ntasks i : Nat) : Nat := rows * i / ntasks

/-- Boundaries are monotone, so ranges are well formed. -/
theorem bound_mono (rows ntasks i : Nat) : bound rows ntasks i ≤ bound rows ntasks (i + 1) := by
  unfold bound
  apply Nat.div_le_div_right
  exact Nat.mul_le_mul_left _ (Nat.le_succ i)

/-- The first task starts at row 0. -/
theorem bound_zero (rows ntasks : Nat) : bound rows ntasks 0 = 0 := by simp [bound]

/-- The last task ends at the last row. -/
theorem bound_last (rows ntasks : Nat) (h : 0 < ntasks) : bound rows ntasks ntasks = rows := by
  unfold bound
  exact Nat.mul_div_cancel rows h

/-- The task lengths telescope to `bound k`: no row is skipped or done twice. -/
theorem ranges_cover (rows ntasks k : Nat) :
    ((List.range k).map (fun i => bound rows ntasks (i + 1) - bound rows ntasks i)).foldr (· + ·) 0 =
      bound rows ntasks k := by
  induction k with
  | zero => simp [bound_zero]
  | succ j ih =>
    rw [List.range_succ, List.map_append, List.foldr_append]
    have hm := bound_mono rows ntasks j
    have key : ∀ (l : List Nat) (a : Nat), l.foldr (· + ·) a = l.foldr (· + ·) 0 + a := by
      intro l a
      induction l with
      | nil => simp
      | cons x t iht => simp only [List.foldr_cons]; omega
    simp only [List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, Nat.add_zero]
    rw [key, ih]
    omega

/-- All tasks together cover exactly the phase's rows. -/
theorem all_tasks_cover (rows ntasks : Nat) (h : 0 < ntasks) :
    ((List.range ntasks).map (fun i => bound rows ntasks (i + 1) - bound rows ntasks i)).foldr
      (· + ·) 0 = rows := by
  rw [ranges_cover, bound_last rows ntasks h]

-- § a range spanning experts

/-- Global row r is local row r % per of expert r / per. -/
theorem decompose (r per : Nat) : r / per * per + r % per = r := by
  have := Nat.div_add_mod r per
  rw [Nat.mul_comm] at this
  exact this

/-- A global row below n·per belongs to one of the n experts of the phase. -/
theorem expert_in_range (r per n : Nat) (h : r < n * per) : r / per < n := by
  apply (Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero (by rintro rfl; simp at h))).2
  exact h

/-- One inner step: r1 = min(per, r0 + (g1 − r)) makes progress and stays within [r, g1]. -/
theorem inner_step (r g1 per : Nat) (hper : 0 < per) (hr : r < g1) :
    r % per < min per (r % per + (g1 - r)) ∧ r + (min per (r % per + (g1 - r)) - r % per) ≤ g1 := by
  have h1 : r % per < per := Nat.mod_lt r hper
  omega

-- § run_split: parts per expert

/-- `parts_a_ = max(1, ⌈3·threads / n⌉)`. -/
def partsFor (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- There is at least one part per expert. -/
theorem partsFor_pos (threads n : Nat) : 0 < partsFor threads n := by
  unfold partsFor; omega

/-- About three tasks per thread: n·parts ≥ 3·threads, so no thread idles for want of work. -/
theorem enough_tasks (threads n : Nat) (hn : 0 < n) : 3 * threads ≤ n * partsFor threads n := by
  unfold partsFor
  have hdm := Nat.div_add_mod (3 * threads + n - 1) n
  have hlt := Nat.mod_lt (3 * threads + n - 1) hn
  have hle : (3 * threads + n - 1) / n ≤ max 1 ((3 * threads + n - 1) / n) := Nat.le_max_right _ _
  have := Nat.mul_le_mul_left n hle
  generalize (3 * threads + n - 1) / n = q at hdm hle this
  generalize (3 * threads + n - 1) % n = m at hdm hlt
  omega

/-- Job i (< n·parts) is part i%parts of expert i/parts, a valid expert. -/
theorem job_decode (i n parts : Nat) (hi : i < n * parts) : i / parts < n ∧ i % parts < parts := by
  have hp : 0 < parts := Nat.pos_of_ne_zero (by rintro rfl; simp at hi)
  exact ⟨(Nat.div_lt_iff_lt_mul hp).2 hi, Nat.mod_lt i hp⟩

/-- A part's rows [FF·p/parts, FF·(p+1)/parts) are ordered and end inside the expert. -/
theorem part_rows (ff parts p : Nat) (hp : p < parts) :
    ff * p / parts ≤ ff * (p + 1) / parts ∧ ff * (p + 1) / parts ≤ ff := by
  constructor
  · exact Nat.div_le_div_right (Nat.mul_le_mul_left _ (Nat.le_succ p))
  · have hpos : 0 < parts := by omega
    have : ff * (p + 1) ≤ ff * parts := Nat.mul_le_mul_left _ hp
    calc ff * (p + 1) / parts ≤ ff * parts / parts := Nat.div_le_div_right this
      _ = ff := Nat.mul_div_cancel ff hpos

-- § batching beyond the split buffers

/-- Batch sizes of `for (b0 = 0; b0 < n; b0 += k) nb = min(k, n − b0)`. -/
def batchSizes (k : Nat) (hk : 0 < k) : Nat → List Nat
  | n => if _h : n = 0 then [] else min k n :: batchSizes k hk (n - min k n)
termination_by n => n
decreasing_by omega

/-- The batches cover every expert exactly once. -/
theorem batchSizes_sum (k : Nat) (hk : 0 < k) (n : Nat) :
    (batchSizes k hk n).foldr (· + ·) 0 = n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [batchSizes]
    split
    · simp_all
    · simp only [List.foldr_cons]
      rw [ih (n - min k n) (by omega)]
      omega

/-- No batch exceeds the buffer count. -/
theorem batchSizes_le (k : Nat) (hk : 0 < k) (n b : Nat) (hb : b ∈ batchSizes k hk n) : b ≤ k := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [batchSizes] at hb
    split at hb
    · simp at hb
    · rcases List.mem_cons.mp hb with h' | h'
      · omega
      · exact ih (n - min k n) (by omega) h'

-- § the park counter

/-- Park count after `left` workers left the park and `back` of them returned. -/
def parkedAfter (n left back : Nat) : Nat := n - left + back

/-- Once every worker that left has returned, `parked == n` again: the next publish is safe. -/
theorem parked_restored (n k : Nat) (hk : k ≤ n) : parkedAfter n k k = n := by
  unfold parkedAfter; omega

/-- Counting arrivals only on return would start at 0 ≠ n: the first `run` would deadlock. -/
theorem return_only_counting_deadlocks (n : Nat) (hn : 0 < n) : (0 : Nat) ≠ n := by omega

end Strata.Kernels.Cpu.Pool
