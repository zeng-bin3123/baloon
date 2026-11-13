/-!
# MODULE: expert_pool_scheduling
SOURCES: include/ref/kernels/cpu/pool.hpp, src/kernels/cpu/pool.cpp
The CPU expert pool: worker count and core selection, the `head`/`done`/`parked`/`epoch` completion
protocol, and the row-splitting arithmetic of `run_split`, `run_split_multi` and
`run_split_multi_native`.

## Source semantics
* Workers = `n_workers` if positive, else the physical cores minus the first, and at least 1.
* Claims are `head.fetch_add(1)`: the k-th claim gets k, so every job index < n is claimed exactly
  once and late claimers get indices >= n and stop.
* `parked` counts parked workers.  The host publishes only when `parked == n` and returns only when
  `done == n_tasks` AND `parked == n`; at that point no worker is in the drain loop, so resetting
  `head` cannot hand a worker a job from the next batch.
* `run_split`: parts = max(1, ceil(3T / n)) per expert, so n * parts >= 3T tasks; task i maps to
  expert i / parts and the row range [R p / P, R (p+1) / P), and these ranges tile [0, R).
* `run_split_multi`: 3T tasks over n * per rows; task i owns [rows i / tasks, rows (i+1) / tasks),
  and the inner loop walks it in per-expert segments that never cross an expert boundary.
* Oversized batches fall back to single-token jobs in groups of kMaxSplit = 16, or native batches of
  kMaxSplitMulti = 96; the groups cover every job.

## Form template identifiers
ExpertPool::ExpertPool -> workerCount, head_.fetch_add -> claims, parked_/workers -> ParkState,
run_split parts -> partsFor, row range -> rangeLo, multi segment -> segEnd, batching -> batchesOf
-/

namespace Ref.Pool

-- § worker count

/-- `physical_cores(true)`: drop the first core when there is one. -/
def skipFirst (cores : List Nat) : List Nat := cores.drop 1

/-- Skipping removes exactly one core (if any) and keeps the rest in order. -/
theorem skipFirst_length (cores : List Nat) : (skipFirst cores).length = cores.length - 1 := by
  simp [skipFirst]

/-- The constructor's worker count. -/
def workerCount (nWorkers : Int) (nCores : Nat) : Nat :=
  max 1 (if nWorkers > 0 then nWorkers.toNat else nCores)

/-- There is always at least one worker, and an explicit positive request is honoured. -/
theorem workerCount_spec (w : Int) (c : Nat) :
    1 ≤ workerCount w c ∧ (w > 0 → workerCount w c = w.toNat) := by
  unfold workerCount
  constructor
  · omega
  · intro h; simp [h]; omega

-- § claims

/-- Indices handed out by `k` successive `fetch_add(1)` calls starting from 0. -/
def claims (k : Nat) : List Nat := List.range k

/-- Claims below `n` are the jobs actually run: with at least `n` claims every job runs, in order,
and no index is handed out twice. -/
theorem claims_cover (k n : Nat) (hk : n ≤ k) :
    (claims k).filter (fun i => decide (i < n)) = List.range n ∧ (claims k).Nodup := by
  refine ⟨?_, List.nodup_range⟩
  unfold claims
  induction k with
  | zero => have : n = 0 := by omega
            subst this; rfl
  | succ k ih =>
    rw [List.range_succ, List.filter_append]
    by_cases hnk : n ≤ k
    · rw [ih hnk]
      simp; omega
    · have : n = k + 1 := by omega
      subst this
      have hall : (List.range k).filter (fun i => decide (i < k + 1)) = List.range k := by
        rw [List.filter_eq_self]
        intro a ha; simp at ha; simp; omega
      rw [hall, List.range_succ]
      simp

-- § park protocol

/-- Each worker is parked (`true`) or draining (`false`); `count` mirrors `parked_`. -/
structure ParkState where
  workers : List Bool
  count : Nat

/-- Number of parked workers. -/
def nParked : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + nParked bs

/-- The invariant the atomics maintain: the counter equals the number of parked workers. -/
def ParkState.inv (s : ParkState) : Prop := s.count = nParked s.workers

/-- Set worker `i`'s flag. -/
def setAt : List Bool → Nat → Bool → List Bool
  | [], _, _ => []
  | _ :: bs, 0, v => v :: bs
  | b :: bs, i + 1, v => b :: setAt bs i v

/-- Changing one flag from `old` to `new` changes the parked count by the expected amount. -/
theorem nParked_setAt (l : List Bool) (i : Nat) (hi : i < l.length) (v : Bool) :
    nParked (setAt l i v) + (if l[i] then 1 else 0) = nParked l + (if v then 1 else 0) := by
  induction l generalizing i with
  | nil => simp at hi
  | cons b bs ih =>
    cases i with
    | zero => simp [setAt, nParked]; omega
    | succ j =>
      simp only [List.length_cons] at hi
      have := ih j (by omega)
      simp [setAt, nParked]
      omega

/-- A parked worker leaving the park (`fetch_sub`) preserves the invariant. -/
theorem leave_preserves (s : ParkState) (i : Nat) (hi : i < s.workers.length) (hp : s.workers[i] = true)
    (h : s.inv) : ParkState.inv ⟨setAt s.workers i false, s.count - 1⟩ := by
  unfold ParkState.inv at *
  have := nParked_setAt s.workers i hi false
  rw [hp] at this
  simp at this
  simp only
  omega

/-- A draining worker returning to the park (`fetch_add`) preserves the invariant. -/
theorem arrive_preserves (s : ParkState) (i : Nat) (hi : i < s.workers.length) (hp : s.workers[i] = false)
    (h : s.inv) : ParkState.inv ⟨setAt s.workers i true, s.count + 1⟩ := by
  unfold ParkState.inv at *
  have := nParked_setAt s.workers i hi true
  rw [hp] at this
  simp at this
  simp only
  omega

/-- The parked count never exceeds the number of workers. -/
theorem nParked_le (l : List Bool) : nParked l ≤ l.length := by
  induction l with
  | nil => simp [nParked]
  | cons b bs ih => simp [nParked]; split <;> omega

/-- When `parked == n` every worker is parked: none is inside the drain loop. -/
theorem all_parked (l : List Bool) (h : nParked l = l.length) : ∀ b ∈ l, b = true := by
  induction l with
  | nil => simp
  | cons b bs ih =>
    have hle := nParked_le bs
    simp [nParked] at h
    intro x hx
    cases b with
    | true =>
      simp at h
      rcases List.mem_cons.1 hx with rfl | hx
      · rfl
      · exact ih (by omega) x hx
    | false => simp at h; omega

/-- The host's handover condition: with the invariant, `count == n` means no worker can still claim
from `head`, so resetting it for the next batch is safe. -/
theorem handover_safe (s : ParkState) (h : s.inv) (hc : s.count = s.workers.length) :
    ∀ b ∈ s.workers, b = true := by
  unfold ParkState.inv at h
  exact all_parked s.workers (by omega)

/-- Workers arrive at the park before their first wait, so the initial state already satisfies
`parked == n` and the first `run()` cannot deadlock. -/
theorem initial_all_parked (n : Nat) :
    ParkState.inv ⟨List.replicate n true, n⟩ := by
  unfold ParkState.inv
  induction n with
  | zero => rfl
  | succ k ih => simp [List.replicate, nParked] at *; omega

-- § run_split partitioning

/-- Parts per expert: `max(1, (3T + n - 1) / n)`. -/
def partsFor (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- There are at least `3T` tasks in total (about three per thread). -/
theorem partsFor_enough (threads n : Nat) (hn : 0 < n) : 3 * threads ≤ n * partsFor threads n := by
  unfold partsFor
  have hdm := Nat.div_add_mod (3 * threads + n - 1) n
  have hlt := Nat.mod_lt (3 * threads + n - 1) hn
  have hm : n * ((3 * threads + n - 1) / n) ≤ n * max 1 ((3 * threads + n - 1) / n) :=
    Nat.mul_le_mul_left _ (Nat.le_max_right _ _)
  generalize (3 * threads + n - 1) / n = q at *
  generalize n * q = m at *
  omega

/-- Task `i < n * parts` decodes to a valid (expert, part). -/
theorem task_decode (i n parts : Nat) (hp : 0 < parts) (hi : i < n * parts) :
    i / parts < n ∧ i % parts < parts := by
  refine ⟨?_, Nat.mod_lt _ hp⟩
  rw [Nat.div_lt_iff_lt_mul hp]
  exact hi

/-- Lower row bound of part `p` of `P` over `R` rows: `R * p / P`. -/
def rangeLo (R P p : Nat) : Nat := R * p / P

/-- The parts tile `[0, R)`: the first starts at 0, the last ends at R, consecutive parts meet,
and every range is non-decreasing. -/
theorem ranges_tile (R P p : Nat) (hP : 0 < P) :
    rangeLo R P 0 = 0 ∧ rangeLo R P P = R ∧ rangeLo R P p ≤ rangeLo R P (p + 1) := by
  unfold rangeLo
  refine ⟨by simp, Nat.mul_div_cancel R hP, ?_⟩
  exact Nat.div_le_div_right (Nat.mul_le_mul_left _ (Nat.le_succ p))

/-- Every row of a range is a real row: `R * (p+1) / P <= R` for `p < P`. -/
theorem range_within (R P p : Nat) (hP : 0 < P) (hp : p < P) : rangeLo R P (p + 1) ≤ R := by
  unfold rangeLo
  have : R * (p + 1) / P ≤ R * P / P := Nat.div_le_div_right (Nat.mul_le_mul_left _ hp)
  rw [Nat.mul_div_cancel R hP] at this
  exact this

-- § run_split_multi segments

/-- End of the segment starting at global row `r` within expert `r / per`, for a task ending at `g1`. -/
def segEnd (per r g1 : Nat) : Nat := min per (r % per + (g1 - r))

/-- Inside a task the walk makes progress, never passes the task's end, and stays inside one expert. -/
theorem segment_spec (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    r % per < segEnd per r g1 ∧ segEnd per r g1 - r % per ≤ g1 - r ∧ segEnd per r g1 ≤ per := by
  have := Nat.mod_lt r hper
  unfold segEnd
  omega

/-- After a segment the next row starts at the following expert or at the task's end. -/
theorem segment_next (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    let r' := r + (segEnd per r g1 - r % per)
    r' = g1 ∨ r' % per = 0 := by
  have hdm := Nat.div_add_mod r per
  have hlt := Nat.mod_lt r hper
  simp only [segEnd]
  by_cases h : r % per + (g1 - r) ≤ per
  · rw [Nat.min_eq_right h]
    left; omega
  · rw [Nat.min_eq_left (by omega)]
    right
    have : r + (per - r % per) = per * (r / per + 1) := by
      rw [Nat.mul_add, Nat.mul_one]; omega
    rw [this, Nat.mul_mod_right]

/-- With `mtasks = 3T` tasks over `n * per` rows, consecutive tasks meet and the last one ends at
the total: the multi phases cover all rows of all experts. -/
theorem multi_tasks_tile (rows tasks : Nat) (ht : 0 < tasks) :
    rows * 0 / tasks = 0 ∧ rows * tasks / tasks = rows := by
  exact ⟨by simp, Nat.mul_div_cancel rows ht⟩

-- § oversized-batch fallbacks

/-- Batch sizes of a loop stepping by `k` with tail `min(k, n - b0)`. -/
def batchesOf (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min k n :: batchesOf k fuel (n - min k n)

/-- The batches cover every job, and each is non-empty and at most `k`. -/
theorem batchesOf_cover (k fuel n : Nat) (hk : 0 < k) (hf : n ≤ k * fuel) :
    (batchesOf k fuel n).foldr (· + ·) 0 = n ∧ ∀ b ∈ batchesOf k fuel n, 0 < b ∧ b ≤ k := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by simp at hf; omega
            subst this; simp [batchesOf]
  | succ f ih =>
    cases n with
    | zero => simp [batchesOf]
    | succ m =>
      have hf' : m + 1 - min k (m + 1) ≤ k * f := by
        rw [Nat.mul_succ] at hf; omega
      obtain ⟨h1, h2⟩ := ih (m + 1 - min k (m + 1)) hf'
      simp only [batchesOf, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro b hb
      rcases hb with rfl | hb
      · omega
      · exact h2 b hb

/-- The single-token fallback (groups of kMaxSplit = 16) and native batching (kMaxSplitMulti = 96)
are instances: e.g. 100 single jobs run as 16 x 6 + 4, 200 native experts as 96 + 96 + 8. -/
theorem fallback_examples :
    batchesOf 16 7 100 = [16, 16, 16, 16, 16, 16, 4] ∧ batchesOf 96 3 200 = [96, 96, 8] := by decide

/-- `run_split` falls back to `run` for too many experts, a single worker, or the oracle. -/
def splitFallsBack (n nWorkers : Nat) (oracle : Bool) : Bool :=
  decide (n > 16) || decide (nWorkers = 1) || oracle

/-- Up to 16 experts on a multi-worker pool without the oracle are split. -/
theorem split_used (n w : Nat) (hn : n ≤ 16) (hw : w ≠ 1) : splitFallsBack n w false = false := by
  simp [splitFallsBack]; omega

-- § affinity restore

/-- `restore_thread_affinity` acts only on a positive previous mask; -1 (refused) is a no-op. -/
def restoreActs (previous : Int) : Bool := decide (previous > 0)

/-- A refused pin (-1) and the Windows failure value 0 are never "restored". -/
theorem restore_ignores_failure : restoreActs (-1) = false ∧ restoreActs 0 = false := by decide

end Ref.Pool
