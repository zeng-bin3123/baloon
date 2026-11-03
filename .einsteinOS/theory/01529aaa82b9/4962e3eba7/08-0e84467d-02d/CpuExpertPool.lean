/-!
# MODULE: cpu_expert_pool
The CPU expert pool: single-counter job claiming (`head_` fetch_add), the `done`-then-`parked` completion
handover, physical-core selection, worker count, and the arithmetic that cuts a phase into equal row ranges
across all experts of a multi-token batch.

## Source semantics
* The k-th `fetch_add` on `head_` returns k; the claim is a job iff k < n. So exactly n claims succeed and each
  index is handed out once.
* `run()` returns only when `done == n` AND `parked == n_`. `done` alone is not enough: a worker can still be
  inside the drain after its last `done` increment.
* `physical_cores(skip_first)` drops the first core; the worker count is at least 1.
* `run_split` cuts each expert into `parts` row ranges; task i maps to (expert i / parts, part i % parts).
* The multi phases cut `mrows` rows (n experts times per-expert rows) into `mtasks` ranges
  `[mrows*i/mtasks, mrows*(i+1)/mtasks)`, each walked expert by expert with the end clamped at the expert edge.
* Native batches of at most `kMaxSplitMulti = 96` experts cover all n experts.

## Form template identifiers
* `claim`, `validClaim` : `head_.fetch_add(1)` and `i < njobs_`
* `inDrain` : `n_ - parked_`
* `lo` : `mrows * i / mtasks`
* `chunks k n` : the `for (b0 = 0; b0 < n; b0 += k)` batching loop with `nb = min(k, n - b0)`
-/

namespace Strata.CpuPool

-- § claiming

/-- The k-th `fetch_add` returns k (counters start at 0 for each batch). -/
def claim (k : Nat) : Nat := k

/-- A claim is a job iff its index is below the job count. -/
def validClaim (i n : Nat) : Bool := decide (i < n)

/-- The successful claim indices are exactly `0 .. n-1`. -/
theorem valid_iff_mem_range (k n : Nat) : validClaim (claim k) n = true ↔ k ∈ List.range n := by
  unfold validClaim claim
  rw [List.mem_range]
  exact decide_eq_true_iff

/-- Distinct claims get distinct indices, so no job is handed to two workers. -/
theorem claim_injective (a b : Nat) (h : claim a = claim b) : a = b := h

/-- Among the first m claims exactly `min m n` succeed. -/
theorem valid_claims (m n : Nat) :
    ((List.range m).filter (fun k => decide (k < n))).length = min m n := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.filter_append, List.length_append, ih]
    by_cases h : m < n
    · simp [h]
      omega
    · simp [h]
      omega

-- § completion handover

/-- Workers currently outside the park. -/
def inDrain (n parked : Nat) : Nat := n - parked

/-- With every worker parked nobody is in the drain, so resetting `head_` cannot race a claim. -/
theorem parked_all_no_drain (n parked : Nat) (h : parked = n) : inDrain n parked = 0 := by
  simp [inDrain, h]

/-- Waiting for `done == njobs` alone is insufficient: a state exists where all jobs are done yet a worker is
still in the drain. -/
theorem done_alone_insufficient :
    ∃ done parked n njobs : Nat, done = njobs ∧ parked < n ∧ 0 < inDrain n parked :=
  ⟨3, 1, 2, 3, rfl, by decide, by decide⟩

/-- Workers arrive at the park once before the first wait, so `parked == n_` holds from construction. -/
theorem initial_parked (n : Nat) : (List.replicate n 1).sum = n := by
  simp

-- § cores and worker count

/-- `physical_cores(skip_first)`: drop the first core when asked. -/
def skipFirst (skip : Bool) (cores : List Nat) : List Nat := if skip then cores.drop 1 else cores

/-- Skipping removes exactly one core from a non-empty list. -/
theorem skipFirst_length (cores : List Nat) (h : cores ≠ []) :
    (skipFirst true cores).length + 1 = cores.length := by
  cases cores with
  | nil => exact absurd rfl h
  | cons a t => simp [skipFirst]

/-- Skipping never adds cores. -/
theorem skipFirst_le (skip : Bool) (cores : List Nat) : (skipFirst skip cores).length ≤ cores.length := by
  cases skip <;> simp [skipFirst]

/-- The pool's worker count: the request if positive, else every (remaining) core; at least one. -/
def nWorkers (req ncores : Nat) : Nat :=
  if (if 0 < req then req else ncores) < 1 then 1 else (if 0 < req then req else ncores)

/-- There is always at least one worker, so a token is always produced. -/
theorem nWorkers_pos (req ncores : Nat) : 1 ≤ nWorkers req ncores := by
  unfold nWorkers
  split_ifs <;> omega

/-- A positive request is honoured. -/
theorem nWorkers_req (req ncores : Nat) (h : 0 < req) : nWorkers req ncores = req := by
  unfold nWorkers
  split_ifs <;> omega

-- § row-range arithmetic

/-- Boundary of range `i` when `mrows` rows are cut into `mtasks` ranges. -/
def lo (mrows mtasks i : Nat) : Nat := mrows * i / mtasks

/-- The first range starts at 0. -/
theorem lo_zero (mrows mtasks : Nat) : lo mrows mtasks 0 = 0 := by simp [lo]

/-- The last range ends at `mrows`: every row of the phase is covered. -/
theorem lo_full (mrows mtasks : Nat) (h : 0 < mtasks) : lo mrows mtasks mtasks = mrows := by
  simp [lo, Nat.mul_div_cancel _ h]

/-- Boundaries are monotone, so ranges never overlap. -/
theorem lo_mono (mrows mtasks i j : Nat) (h : i ≤ j) : lo mrows mtasks i ≤ lo mrows mtasks j :=
  Nat.div_le_div_right (Nat.mul_le_mul_left mrows h)

/-- A flat row index decomposes as (expert, row within expert). -/
theorem flat_decomp (r per : Nat) (h : 0 < per) :
    r = per * (r / per) + r % per ∧ r % per < per := by
  have h1 := Nat.div_add_mod r per
  have h2 := Nat.mod_lt r h
  omega

/-- The walk clamps the range end at the expert edge and always makes progress. -/
theorem clamp_progress (r0 per rem : Nat) (h0 : r0 < per) (hr : 0 < rem) :
    r0 < min per (r0 + rem) ∧ min per (r0 + rem) ≤ per := by
  omega

/-- The expert index of a flat row is in range when the row is below `n * per`. -/
theorem expert_index_lt (r per n : Nat) (h : r < n * per) (hp : 0 < per) : r / per < n :=
  (Nat.div_lt_iff_lt_mul hp).2 (by have := Nat.mul_comm n per; omega)

-- § run_split task decomposition

/-- Parts per expert: about three tasks per thread, at least one part. -/
def partsPer (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsPer_pos (threads n : Nat) : 1 ≤ partsPer threads n := by
  unfold partsPer
  omega

/-- The unclamped quotient provides at least three tasks per thread. -/
theorem parts_cover_threads (threads n : Nat) (hn : 0 < n) :
    3 * threads ≤ n * ((3 * threads + n - 1) / n) := by
  have h1 := Nat.div_add_mod (3 * threads + n - 1) n
  have h2 := Nat.mod_lt (3 * threads + n - 1) hn
  omega

/-- Task i of `n * parts` names an expert below n. -/
theorem task_expert_lt (i n parts : Nat) (h : i < n * parts) (hp : 0 < parts) : i / parts < n :=
  (Nat.div_lt_iff_lt_mul hp).2 (by have := Nat.mul_comm n parts; omega)

/-- Task i names a part below `parts`. -/
theorem task_part_lt (i parts : Nat) (hp : 0 < parts) : i % parts < parts := Nat.mod_lt i hp

/-- Part boundaries `FF * part / parts` tile [0, FF): the last part ends at FF. -/
theorem part_end (ff parts : Nat) (hp : 0 < parts) : ff * parts / parts = ff := by
  simp [Nat.mul_div_cancel _ hp]

-- § batching of experts

/-- `for (b0 = 0; b0 < n; b0 += k) nb = min(k, n - b0)`: the list of batch sizes. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- The batches cover all n experts exactly once. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Every batch is non-empty and at most `k` (so at most kMaxSplitMulti = 96 buffers are used). -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.CpuPool
