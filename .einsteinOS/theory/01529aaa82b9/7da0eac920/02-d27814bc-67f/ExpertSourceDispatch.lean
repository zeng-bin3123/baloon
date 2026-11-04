/-!
# MODULE: expert_source_dispatch
Expert blob sources (memory-mapped file, pinned arena) and the CPU-pool dispatch that splits routed experts
between GPU hits and CPU jobs.

## Source semantics
* A blob is addressed by `layer * n_expert + expert`; layer and expert are separate axes, so `expert < n_expert`
  must be checked on its own (the flat index alone aliases the next layer).
* In the dispatch loop `njobs` indexes the job array and `i` indexes the output; they agree only when nothing is a
  hit. Hits and misses partition the `k` routed entries.
* The PCIe share of the misses is `(nmiss * pcie_num) >> 8` and never exceeds `nmiss` when `pcie_num ≤ 256`.
* The GGUF loader reads 16 experts per chunk; every expert index it writes is below `n_expert`.

## Form template identifiers
* `FileExpertSource::blob`, `ArenaExpertSource::blob` -> `flatIdx`, `blobValid`
* `expert_pool_dispatch` job/output indexing -> `nMiss`, `nHit`
* `expert_pool_dispatch_multi` PCIe share -> `pcieShare`
* `load_experts_gguf` chunk walk -> `chunkExpert`
-/

namespace Strata.ExpertSource

-- § flat blob index

/-- Flat blob index. -/
def flatIdx (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- The bounds check of `blob()`: both axes, separately. -/
def blobValid (nLayers nExpert layer expert : Nat) : Prop := layer < nLayers ∧ expert < nExpert

/-- A valid (layer, expert) has a flat index inside the pack. -/
theorem flatIdx_lt (nl ne l e : Nat) (h : blobValid nl ne l e) : flatIdx ne l e < nl * ne := by
  obtain ⟨hl, he⟩ := h
  unfold flatIdx
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ hl
  have h2 : (l + 1) * ne = l * ne + ne := Nat.succ_mul l ne
  omega

/-- The aliasing hazard: `blob(0, n_expert)` has the same flat index as `blob(1, 0)`. -/
theorem flatIdx_alias (ne : Nat) : flatIdx ne 0 ne = flatIdx ne 1 0 := by
  simp [flatIdx]

/-- Hence a flat in-range check alone cannot validate the expert axis. -/
theorem flat_check_insufficient (nl ne : Nat) (hl : 2 ≤ nl) (hne : 0 < ne) :
    flatIdx ne 0 ne < nl * ne ∧ ¬ blobValid nl ne 0 ne := by
  refine ⟨?_, ?_⟩
  · have h1 : 2 * ne ≤ nl * ne := Nat.mul_le_mul_right _ hl
    unfold flatIdx
    omega
  · intro h
    exact Nat.lt_irrefl _ h.2

/-- On valid pairs the flat index is injective: distinct (layer, expert) never share a blob. -/
theorem flatIdx_inj (ne l1 e1 l2 e2 : Nat) (he1 : e1 < ne) (he2 : e2 < ne)
    (h : flatIdx ne l1 e1 = flatIdx ne l2 e2) : l1 = l2 ∧ e1 = e2 := by
  have hne : 0 < ne := by omega
  unfold flatIdx at h
  have h1 : (e1 + l1 * ne) / ne = l1 := by
    rw [Nat.add_mul_div_right _ _ hne, Nat.div_eq_of_lt he1]; simp
  have h2 : (e2 + l2 * ne) / ne = l2 := by
    rw [Nat.add_mul_div_right _ _ hne, Nat.div_eq_of_lt he2]; simp
  have h3 : e1 + l1 * ne = e2 + l2 * ne := by
    rw [Nat.add_comm e1, Nat.add_comm e2]; exact h
  have hl : l1 = l2 := by
    rw [← h1, ← h2, h3]
  subst hl
  constructor
  · rfl
  · omega

-- § hit / miss partition

/-- Number of CPU jobs (misses) among the hit flags. -/
def nMiss : List Bool → Nat
  | [] => 0
  | h :: t => (if h then 0 else 1) + nMiss t

/-- Number of GPU-owned rows (hits) among the hit flags. -/
def nHit : List Bool → Nat
  | [] => 0
  | h :: t => (if h then 1 else 0) + nHit t

/-- Hits and misses partition the routed entries. -/
theorem nMiss_add_nHit (hs : List Bool) : nMiss hs + nHit hs = hs.length := by
  induction hs with
  | nil => simp [nMiss, nHit]
  | cons h t ih =>
    cases h <;> simp [nMiss, nHit] <;> omega

/-- The job array never needs more than `k` entries. -/
theorem nMiss_le (hs : List Bool) : nMiss hs ≤ hs.length := by
  have := nMiss_add_nHit hs
  omega

/-- With no hits, job index and output index coincide. -/
theorem nMiss_no_hits (n : Nat) : nMiss (List.replicate n false) = n := by
  induction n with
  | zero => simp [nMiss]
  | succ n ih => simp [List.replicate_succ, nMiss, ih]; omega

/-- With all hits, no CPU job exists. -/
theorem nMiss_all_hits (n : Nat) : nMiss (List.replicate n true) = 0 := by
  induction n with
  | zero => simp [nMiss]
  | succ n ih => simp [List.replicate_succ, nMiss, ih]

-- § PCIe share

/-- The PCIe share of the misses, `(nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss num : Nat) : Nat := nmiss * num / 256

/-- The PCIe share never exceeds the number of misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  unfold pcieShare
  have h1 : nmiss * num ≤ 256 * nmiss := by
    rw [Nat.mul_comm 256]; exact Nat.mul_le_mul_left _ h
  exact Nat.div_le_of_le_mul h1

/-- Zero numerator gives no PCIe traffic. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

-- § GGUF chunk walk

/-- Expert index written for chunk `j`, entry `k` (16 experts per chunk). -/
def chunkExpert (j k : Nat) : Nat := j * 16 + k

/-- Every expert index the GGUF loader writes is below `n_expert`. -/
theorem chunkExpert_lt (ne j k : Nat) (hj : j * 16 < ne) (hk : k < min 16 (ne - j * 16)) :
    chunkExpert j k < ne := by
  unfold chunkExpert
  omega

/-- Chunks tile the experts: the next chunk starts where this one ends. -/
theorem chunkExpert_next (j : Nat) : chunkExpert j 16 = chunkExpert (j + 1) 0 := by
  unfold chunkExpert; omega

end Strata.ExpertSource
