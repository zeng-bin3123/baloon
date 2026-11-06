/-!
# MODULE: expert_pool_contract
SOURCES: include/strata/kernels/cpu/pool.hpp
The static contract of the CPU expert pool: capacity limits, worker/thread counting, cache-line separation of the
contended atomics, the split-path guard, task-count sizing, and batching of more experts than buffers.

## Source semantics
* `run_split` handles at most `kMaxSplit` = 16 experts; `run_split_multi` at most `kMaxSplitMulti` = 96;
  larger native layers are processed in batches of at most 96.
* The pool has at least one worker; the number of draining threads is workers plus one when the host also works.
* Each contended atomic (`head_`, `done_`, `parked_`, `epoch_`, `stop_`) sits alone on a 64-byte line.
* A phase is cut into `3 * threads` tasks (multi) or about three tasks per thread (split): enough to cover the work.
* `restore_thread_affinity` ignores non-positive masks, so a failed pin (-1) is never restored.

## Form template identifiers
kMaxSplit → split_capacity, kMaxSplitMulti → multi_capacity, MAXT → tokens_per_expert,
poolWorkers → worker_count, drainThreads → drain_thread_count, partsPerExpert → split_parts,
batchSizes → expert_batches, cacheLineOf → atomic_cache_line.
-/

namespace Strata.Kernels.Cpu.ExpertPoolContract

-- § capacities

/-- Maximum experts in `run_split`. -/
def kMaxSplit : Nat := 16
/-- Maximum experts in `run_split_multi`. -/
def kMaxSplitMulti : Nat := 96
/-- Maximum tokens routed to one expert. -/
def MAXT : Nat := 8

/-- The multi capacity is at least the single-token split capacity. -/
theorem multi_capacity_ge_split : kMaxSplit ≤ kMaxSplitMulti := by
  decide

/-- Guard of the split path: it is taken only if the job count is within capacity (and several workers, no oracle). -/
theorem split_path_bound (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) : n ≤ 16 :=
  Nat.le_of_not_lt (fun hn => h (Or.inl hn))

/-- The split path is never taken with a single worker. -/
theorem split_path_multi_worker (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) :
    nw ≠ 1 :=
  fun h1 => h (Or.inr (Or.inl h1))

/-- The split path is never taken under the oracle contract. -/
theorem split_path_no_oracle (n nw : Nat) (o : Bool) (h : ¬ (n > 16 ∨ nw = 1 ∨ o = true)) :
    o = false := by
  cases o
  · rfl
  · exact (h (Or.inr (Or.inr rfl))).elim

/-- Every expert index below the job count indexes a valid split buffer. -/
theorem split_buffer_index (e n : Nat) (hn : n ≤ 16) (he : e < n) : e < kMaxSplit := by
  have : kMaxSplit = 16 := by decide
  omega

-- § worker and thread counts

/-- Worker count: the request if positive, else one per usable core, never fewer than one. -/
def poolWorkers (requested : Int) (usableCores : Nat) : Nat :=
  max 1 (if 0 < requested then requested.toNat else usableCores)

/-- The pool always has at least one worker. -/
theorem poolWorkers_pos (requested : Int) (usableCores : Nat) : 1 ≤ poolWorkers requested usableCores :=
  Nat.le_max_left _ _

/-- Threads that drain a phase: the workers, plus the host when `host_works`. -/
def drainThreads (workers : Nat) (hostWorks : Bool) : Nat :=
  workers + (if hostWorks then 1 else 0)

/-- At least one thread drains. -/
theorem drainThreads_pos (workers : Nat) (hostWorks : Bool) (h : 1 ≤ workers) :
    1 ≤ drainThreads workers hostWorks := by
  unfold drainThreads
  split <;> omega

/-- The host joining the drain adds exactly one thread. -/
theorem drainThreads_host (workers : Nat) : drainThreads workers true = workers + 1 := by
  unfold drainThreads
  simp

/-- Excluding the first physical core removes exactly one entry (when there is one). -/
def selectCores (cores : List Nat) (skipFirst : Bool) : List Nat :=
  if skipFirst then cores.tail else cores

/-- Skipping the first core leaves one fewer core. -/
theorem selectCores_length (cores : List Nat) : (selectCores cores true).length = cores.length - 1 := by
  simp [selectCores]

/-- Not skipping leaves all cores. -/
theorem selectCores_all (cores : List Nat) : selectCores cores false = cores := by
  simp [selectCores]

-- § cache-line separation of the atomics

/-- Byte offset of the i-th atomic when each is `alignas(64)`. -/
def atomicOffset (i : Nat) : Nat := 64 * i

/-- The cache line (64-byte granule) an offset falls in. -/
def cacheLineOf (off : Nat) : Nat := off / 64

/-- Two distinct `alignas(64)` atomics never share a cache line (no false sharing between them). -/
theorem atomics_distinct_lines (i j : Nat) (h : i ≠ j) :
    cacheLineOf (atomicOffset i) ≠ cacheLineOf (atomicOffset j) := by
  unfold cacheLineOf atomicOffset
  omega

-- § task counts

/-- Parts per expert in `run_split`: about three tasks per thread, at least one part. -/
def partsPerExpert (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsPerExpert_pos (threads n : Nat) : 1 ≤ partsPerExpert threads n :=
  Nat.le_max_left _ _

/-- The ceiling division covers `3 * threads` tasks in total. -/
theorem split_tasks_cover (threads n : Nat) (hn : 0 < n) :
    3 * threads ≤ n * ((3 * threads + n - 1) / n) := by
  have h := Nat.div_add_mod (3 * threads + n - 1) n
  have hr := Nat.mod_lt (3 * threads + n - 1) hn
  omega

/-- Tasks of `run_split_multi` per phase: three per thread, so at least three. -/
theorem multi_tasks_ge_three (threads : Nat) (h : 1 ≤ threads) : 3 ≤ 3 * threads := by
  omega

-- § batching of experts beyond the buffer capacity

/-- Batch sizes when walking `n` experts in batches of at most `k` (fuel-bounded). -/
def batchSizes (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: batchSizes k f (n - min k n)

/-- The batches cover exactly `n` experts. -/
theorem batchSizes_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (batchSizes k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [batchSizes, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [batchSizes, if_pos h0]
      simp [h0]
    · rw [batchSizes, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every batch holds at most `k` experts, so at most `kMaxSplitMulti` buffers are ever used. -/
theorem batchSizes_le (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ batchSizes k f n, x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [batchSizes] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [batchSizes, if_pos h0] at hx
      simp at hx
    · rw [batchSizes, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- With fuel `n`, batches of 96 cover any number of native experts. -/
theorem native_batches_cover (n : Nat) : (batchSizes 96 n n).sum = n := by
  apply batchSizes_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 96 by decide)
  omega

-- § affinity restore

/-- `restore_thread_affinity` acts only on a positive previous mask. -/
def restoreApplies (previous : Int) : Bool := decide (0 < previous)

/-- A refused pin (returns -1) is never restored. -/
theorem restore_skips_failure : restoreApplies (-1) = false := by
  decide

/-- A real previous mask is restored. -/
theorem restore_applies_mask (m : Int) (h : 0 < m) : restoreApplies m = true := by
  unfold restoreApplies
  simp [h]

end Strata.Kernels.Cpu.ExpertPoolContract
