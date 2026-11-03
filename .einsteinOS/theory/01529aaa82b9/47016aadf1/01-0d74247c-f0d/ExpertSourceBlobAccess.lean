/-!
# MODULE: expert_source_blob_access
Expert blob addressing (`FileExpertSource`, `ArenaExpertSource`) and the dispatch adapter's job/output indexing
(`expert_pool_dispatch`, `expert_pool_dispatch_multi`).

## Source semantics
* The flat blob index is `layer*n_expert + expert`; `expert >= n_expert` must be rejected on its own, because the flat
  index alone would alias into the next layer.
* The dispatch loop indexes the JOB ARRAY by `njobs` and the OUTPUT by `i`; hit rows are zeroed and skipped.
* The PCIe share `(nmiss * pcie_num) >> 8` never exceeds the number of misses.

## Form template identifiers
`n_expert_` -> ne, `blobs_` -> blobs, `BLOB` -> blobBytes, `d.is_hit` -> hitFlags, `njobs` -> misses.
-/

namespace Strata.ExpertSource

-- § blob addressing

/-- Blob index with the separate expert-axis check (negative inputs are modelled elsewhere: coordinates are naturals). -/
def blobIndex? (ne blobs layer expert : Nat) : Option Nat :=
  if ne ≤ expert then none
  else if blobs ≤ layer * ne + expert then none
  else some (layer * ne + expert)

/-- An expert id one past the end of a layer is rejected. -/
theorem blobIndex_expert_axis (ne blobs layer : Nat) : blobIndex? ne blobs layer ne = none := by
  unfold blobIndex?; simp

/-- Why the separate check is needed: `(layer, ne)` and `(layer+1, 0)` have the same flat index. -/
theorem flat_index_aliases (ne layer : Nat) : layer * ne + ne = (layer + 1) * ne + 0 := by
  rw [Nat.add_mul, Nat.one_mul, Nat.add_zero]

/-- A returned index is in range and equals the flat index. -/
theorem blobIndex_sound (ne blobs layer expert i : Nat) (h : blobIndex? ne blobs layer expert = some i) :
    expert < ne ∧ i < blobs ∧ i = layer * ne + expert := by
  unfold blobIndex? at h
  by_cases c1 : ne ≤ expert
  · simp [c1] at h
  · by_cases c2 : blobs ≤ layer * ne + expert
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      omega

/-- For a pack of exactly `nl * ne` blobs, every valid coordinate pair is served. -/
theorem blobIndex_complete (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    blobIndex? ne (nl * ne) layer expert = some (layer * ne + expert) := by
  have h := Nat.mul_le_mul_right ne (show layer + 1 ≤ nl by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  have c1 : ¬ ne ≤ expert := by omega
  have c2 : ¬ nl * ne ≤ layer * ne + expert := by omega
  unfold blobIndex?
  rw [if_neg c1, if_neg c2]

/-- The pinned test: a blob is pinned only if it ends inside the registered bytes. -/
def pinnedOk (off len pinnedBytes : Nat) : Bool := decide (off + len ≤ pinnedBytes)

/-- Pinned status is monotone in the registered size. -/
theorem pinnedOk_mono (off len a b : Nat) (hab : a ≤ b) (h : pinnedOk off len a = true) :
    pinnedOk off len b = true := by
  unfold pinnedOk at *
  simp at *
  omega

-- § dispatch: jobs versus output rows

/-- Number of misses (jobs) in a hit-flag list. -/
def misses : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + misses bs

/-- Number of hits in a hit-flag list. -/
def hitCount : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + hitCount bs

/-- Every routed entry is either a hit (GPU) or a job (CPU): no entry is computed twice or dropped. -/
theorem hits_add_misses : ∀ l : List Bool, hitCount l + misses l = l.length
  | [] => by simp [hitCount, misses]
  | b :: bs => by
      have ih := hits_add_misses bs
      cases b <;> simp [hitCount, misses] <;> omega

/-- The job array never needs more than `k` slots. -/
theorem jobs_le_k (l : List Bool) : misses l ≤ l.length := by
  have := hits_add_misses l
  omega

/-- Output row `i < k` lies inside the `k x n_embd` output buffer. -/
theorem out_row_in_bounds (k n i : Nat) (h : i < k) : i * n + n ≤ k * n := by
  have h1 := Nat.mul_le_mul_right n (show i + 1 ≤ k by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

/-- The verify-window job table of `n_tok * k` entries fits the `MAXT * k` allocation. -/
theorem jobs_multi_fit (nTok maxT k : Nat) (h : nTok ≤ maxT) : nTok * k ≤ maxT * k :=
  Nat.mul_le_mul_right k h

-- § the PCIe share

/-- Number of missed experts the GPU reads over PCIe: `(nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss num : Nat) : Nat := nmiss * num / 256

/-- With `pcie_num ≤ 256` the share never exceeds the misses. -/
theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  unfold pcieShare
  have h1 : nmiss * num ≤ nmiss * 256 := Nat.mul_le_mul_left nmiss h
  have h2 : nmiss * num ≤ 256 * nmiss := by
    calc nmiss * num ≤ nmiss * 256 := h1
      _ = 256 * nmiss := Nat.mul_comm _ _
  exact Nat.div_le_of_le_mul h2

end Strata.ExpertSource
