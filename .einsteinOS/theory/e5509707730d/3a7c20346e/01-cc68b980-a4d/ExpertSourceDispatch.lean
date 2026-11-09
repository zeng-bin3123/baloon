/-!
# MODULE: expert_source_dispatch
SOURCES: include/strata/core/expert_source.hpp, src/core/expert_source.cpp
The adapter between the host loop and the CPU expert pool: blob lookup in the mapped/arena source, the
GPU/CPU hit-miss split by router index, the PCIe share of misses, and the latching failure flag.

## Source semantics
* A blob index is `layer * n_expert + expert`; `expert >= n_expert` is rejected separately because the flat
  index alone aliases the next layer's expert 0.
* CPU rows for hits are zeroed and GPU rows for misses are zeroed, so `parts + hit_out` is the full answer.
* `njobs` (CPU jobs) counts the misses; hits + jobs = k.
* The PCIe share is `nmiss * pcie_num / 256 <= nmiss` of the distinct misses.
* `failed` latches: once set, every later dispatch returns immediately.
* The hit path is all-or-nothing (`hits_ready`).

## Form template identifiers
ExpertSource -> blob_provider; FileExpertSource -> mapped_provider; ArenaExpertSource -> resident_provider;
ExpertDispatch -> split_dispatcher; expert_pool_dispatch -> cpu_half; expert_hit_run -> gpu_half;
pcie_num -> pcie_share_numerator.
-/

namespace ExpertSourceDispatch

-- § blob indexing

/-- Flat blob index `layer * n_expert + expert`. -/
def blobIndex (ne l e : Nat) : Nat := l * ne + e

/-- With `expert < n_expert` and `layer < n_layers` the flat index is in range. -/
theorem blobIndex_lt {ne nl l e : Nat} (he : e < ne) (hl : l < nl) : blobIndex ne l e < nl * ne := by
  unfold blobIndex
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Why the separate `expert >= n_expert` check exists: without it, `blob(0, n_expert)` is `blob(1, 0)`. -/
theorem blobIndex_aliases_without_check (ne : Nat) : blobIndex ne 0 ne = blobIndex ne 1 0 := by
  unfold blobIndex
  simp

/-- With both coordinates checked, distinct (layer, expert) pairs give distinct indices. -/
theorem blobIndex_injective {ne l1 l2 e1 e2 : Nat} (hne : 0 < ne) (h1 : e1 < ne) (h2 : e2 < ne)
    (h : blobIndex ne l1 e1 = blobIndex ne l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold blobIndex at h
  have hd : (e1 + l1 * ne) / ne = (e2 + l2 * ne) / ne := by
    rw [Nat.add_comm e1, Nat.add_comm e2, h]
  rw [Nat.add_mul_div_right _ _ hne, Nat.add_mul_div_right _ _ hne,
    Nat.div_eq_of_lt h1, Nat.div_eq_of_lt h2] at hd
  have hl : l1 = l2 := by omega
  subst hl
  exact ⟨rfl, by omega⟩

-- § the hit/miss split

/-- The CPU's row for one router index: zero when the GPU owns it. -/
def cpuRow (hit : Bool) (v : Nat) : Nat := if hit then 0 else v

/-- The GPU's row for one router index: zero when the CPU owns it. -/
def gpuRow (hit : Bool) (v : Nat) : Nat := if hit then v else 0

/-- Summing the two zero-padded halves yields exactly the expert's output, whoever computed it. -/
theorem split_partition (hit : Bool) (v : Nat) : cpuRow hit v + gpuRow hit v = v := by
  cases hit <;> simp [cpuRow, gpuRow]

/-- Number of hits in a decision vector. -/
def countHit : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + countHit bs

/-- Number of misses (= CPU jobs, `njobs`) in a decision vector. -/
def countMiss : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + countMiss bs

/-- Every routed expert is either a hit or a CPU job; none is dropped or doubled. -/
theorem hits_plus_jobs (bs : List Bool) : countHit bs + countMiss bs = bs.length := by
  induction bs with
  | nil => rfl
  | cons b bs ih => cases b <;> simp [countHit, countMiss] <;> omega

-- § PCIe share of misses

/-- The GPU takes `nmiss * pcie_num / 256` of the distinct misses over PCIe. -/
def pcieShare (nmiss pn : Nat) : Nat := nmiss * pn / 256

/-- With `pcie_num <= 256`, the PCIe share never exceeds the number of misses. -/
theorem pcie_share_le {nmiss pn : Nat} (hp : pn ≤ 256) : pcieShare nmiss pn ≤ nmiss := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left nmiss hp
  omega

/-- The last `m` misses in routing order (rank `>= nmiss - m`) number exactly `m` when `m <= nmiss`. -/
theorem pcie_tail_count {nmiss m : Nat} (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  omega

/-- The plan's scratch arrays: at most 8 tokens x 10 experts fit the `kind[128]` array. -/
theorem plan_fits_kind_array {ntok k : Nat} (ht : ntok ≤ 8) (hk : k ≤ 10) : ntok * k ≤ 128 := by
  have : ntok * k ≤ 8 * 10 := Nat.mul_le_mul ht hk
  omega

-- § failure latch and configuration

/-- `failed` after one more dispatch step with outcome `ok`. -/
def stepFailed (failed ok : Bool) : Bool := failed || !ok

/-- Once failed, the adapter stays failed: clearing the message cannot re-arm it. -/
theorem failure_latches (ok : Bool) : stepFailed true ok = true := by
  simp [stepFailed]

/-- A success never clears an earlier failure, and a failure is always recorded. -/
theorem failure_recorded (failed : Bool) : stepFailed failed false = true := by
  simp [stepFailed]

/-- `hits_ready` is the conjunction of all its prerequisites. -/
def hitsReady (flags : List Bool) : Bool := flags.all id

/-- If the hit path is ready, every prerequisite holds (no half-configured hit path). -/
theorem hitsReady_each {flags : List Bool} (h : hitsReady flags = true) : ∀ b ∈ flags, b = true := by
  intro b hb
  unfold hitsReady at h
  rw [List.all_eq_true] at h
  simpa using h b hb

-- § artifact geometry (measured constants)

/-- Expert blob size: three 640x2560 matrices at 9/32 bytes per element (2.25 bits). -/
theorem blob_bytes_derivation : 3 * 640 * 2560 * 9 / 32 = 1382400 := by decide

/-- `experts.bin` holds exactly 48 x 512 blobs and is 33,973,862,400 bytes. -/
theorem experts_bin_size : 48 * 512 * 1382400 = 33973862400 := by decide

/-- One token reads 48 layers x 10 experts = 663,552,000 bytes (the quoted 663.6 MB). -/
theorem bytes_per_token : 48 * 10 * 1382400 = 663552000 := by decide

/-- A VRAM-resident tier of all 512 experts of one layer is 707,788,800 bytes (the MTP layer's ~708 MB). -/
theorem one_layer_experts_bytes : 512 * 1382400 = 707788800 := by decide

end ExpertSourceDispatch
