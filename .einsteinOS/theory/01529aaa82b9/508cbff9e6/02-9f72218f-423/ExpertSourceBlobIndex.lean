/-!
# MODULE: expert_source_blob_index
Blob addressing of the expert sources (file-mapped and arena) and the arithmetic of the verify window's
GPU/PCIe split of missed experts.

## Source semantics
* The blob index is `layer * n_expert + expert` and `expert < n_expert` is checked on its own axis: otherwise
  `blob(l, n_expert)` aliases `blob(l+1, 0)`.
* On valid pairs the index is injective and below `blobs = n_layers * n_expert`.
* The PCIe share of the misses is `(nmiss * pcie_num) >> 8`, which never exceeds `nmiss` for `pcie_num <= 256`.
* Hits are skipped by the pool: `njobs + hits = k`.

## Form template identifiers
* `FileExpertSource::blob`, `ArenaExpertSource::blob` -> `blobIndex`
* `expert_pool_dispatch_multi` PCIe count `m` -> `pcieShare`
* `expert_pool_dispatch` job counter -> `jobCount`
-/

namespace Strata.ExpertSource

-- § blob index

/-- Flat blob index of `(layer, expert)`. -/
def blobIndex (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- Helper: an in-layer expert index stays below the next layer's start. -/
theorem idx_lt_next (ne l e : Nat) (he : e < ne) : blobIndex ne l e < (l + 1) * ne := by
  unfold blobIndex
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Helper: layer starts are monotone. -/
theorem start_mono (ne a b : Nat) (h : a ≤ b) : a * ne ≤ b * ne := Nat.mul_le_mul_right ne h

/-- Why the separate `expert >= n_expert` check matters: one past the end aliases the next layer's first expert. -/
theorem overflow_aliases (ne l : Nat) : blobIndex ne l ne = blobIndex ne (l + 1) 0 := by
  unfold blobIndex
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- In-range pairs index inside the blob table. -/
theorem blobIndex_lt (n ne l e : Nat) (hl : l < n) (he : e < ne) : blobIndex ne l e < n * ne := by
  have h1 := idx_lt_next ne l e he
  have hl' : l + 1 ≤ n := hl
  have h2 := start_mono ne (l + 1) n hl'
  omega

/-- On valid pairs distinct `(layer, expert)` get distinct blobs (no aliasing). -/
theorem blobIndex_injective (ne l1 e1 l2 e2 : Nat) (h1 : e1 < ne) (h2 : e2 < ne)
    (h : blobIndex ne l1 e1 = blobIndex ne l2 e2) : l1 = l2 ∧ e1 = e2 := by
  rcases Nat.lt_trichotomy l1 l2 with hlt | heq | hgt
  · exfalso
    have a := idx_lt_next ne l1 e1 h1
    have b := start_mono ne (l1 + 1) l2 hlt
    have c : l2 * ne ≤ blobIndex ne l2 e2 := by unfold blobIndex; omega
    omega
  · subst heq
    unfold blobIndex at h
    omega
  · exfalso
    have a := idx_lt_next ne l2 e2 h2
    have b := start_mono ne (l2 + 1) l1 hgt
    have c : l1 * ne ≤ blobIndex ne l1 e1 := by unfold blobIndex; omega
    omega

-- § GPU/PCIe split

/-- Number of missed experts the GPU reads over PCIe: `(nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := nmiss * pcieNum / 256

/-- The PCIe share never exceeds the number of misses when `pcie_num <= 256`. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left nmiss h
  omega

/-- With no PCIe numerator the CPU keeps every miss. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

-- § hit / miss partition

/-- Jobs the CPU pool runs: every routed expert that is not a hit. -/
def jobCount (k hits : Nat) : Nat := k - hits

/-- Jobs plus hits account for every routed expert exactly once. -/
theorem jobs_add_hits (k hits : Nat) (h : hits ≤ k) : jobCount k hits + hits = k := by
  unfold jobCount; omega

/-- A layer with no hits gives the pool all `k` experts. -/
theorem jobs_no_hits (k : Nat) : jobCount k 0 = k := by
  simp [jobCount]

end Strata.ExpertSource
