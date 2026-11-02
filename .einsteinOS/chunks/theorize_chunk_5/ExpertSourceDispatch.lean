/-!
# MODULE: expert_source_dispatch
Adapter between the expert blob stores (mmap file, pinned arena) and the CPU expert pool: bounds
checks on blob lookup, hit/miss job compaction and the PCIe share of misses.

## Source semantics
* A blob index is `layer * n_expert + expert`; layer and expert are separate axes, so `expert ≥ n_expert`
  must be refused even though its flat index may be in range (it aliases the next layer).
* Job index and output index differ whenever some row is a GPU hit; the job count is the number of misses.
* The PCIe share of the missed experts is `(nmiss * pcie_num) >> 8`, never more than `nmiss` when `pcie_num ≤ 256`.

## Form template identifiers
* `FileExpertSource::blob` / `ArenaExpertSource::blob` → `blobOk`, `flat`
* `njobs` vs `i` in `expert_pool_dispatch` → `countMiss`
* `m = (nmiss * pcie_num) >> 8` → `pcieShare`
-/

namespace Strata.ExpertSource

-- § blob lookup

/-- Flat blob index. -/
def flat (n_expert l e : Nat) : Nat := l * n_expert + e

/-- The lookup guard: expert checked against its own axis, then the flat index. -/
def blobOk (n_expert blobs l e : Nat) : Bool :=
  decide (e < n_expert) && decide (flat n_expert l e < blobs)

/-- An out-of-range expert id aliases the next layer's expert under the flat index alone. -/
theorem flat_alias (n l e : Nat) (h : n ≤ e) : flat n l e = flat n (l + 1) (e - n) := by
  unfold flat
  have h2 : (l + 1) * n = l * n + n := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The separate expert check refuses exactly the aliasing ids. -/
theorem blobOk_rejects_alias (n blobs l e : Nat) (h : n ≤ e) : blobOk n blobs l e = false := by
  unfold blobOk
  have : ¬ e < n := by omega
  simp [this]

/-- Accepted lookups stay inside the geometry `blobs = layers * n_expert`. -/
theorem blobOk_in_geometry (n nl l e : Nat) (hl : l < nl) (he : e < n) :
    blobOk n (nl * n) l e = true := by
  unfold blobOk flat
  have h := Nat.mul_le_mul_right n (show l + 1 ≤ nl from hl)
  have h2 : (l + 1) * n = l * n + n := by rw [Nat.add_mul, Nat.one_mul]
  have : l * n + e < nl * n := by omega
  simp [he, this]

-- § job compaction

/-- Number of misses in a hit-flag row list (the number of CPU jobs). -/
def countMiss : List Bool → Nat
  | [] => 0
  | h :: hs => (if h then 0 else 1) + countMiss hs

/-- Jobs never outnumber output rows. -/
theorem countMiss_le (hs : List Bool) : countMiss hs ≤ hs.length := by
  induction hs with
  | nil => simp [countMiss]
  | cons h hs ih =>
    simp only [countMiss, List.length_cons]
    split <;> omega

/-- With no hits every row is a job: job index equals output index. -/
theorem countMiss_no_hits (n : Nat) : countMiss (List.replicate n false) = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [List.replicate_succ, countMiss, ih] <;> omega

/-- With only hits there are no jobs. -/
theorem countMiss_all_hits (n : Nat) : countMiss (List.replicate n true) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [List.replicate_succ, countMiss, ih] <;> omega

-- § PCIe share

/-- The number of missed experts routed over PCIe. -/
def pcieShare (nmiss pcie_num : Nat) : Nat := (nmiss * pcie_num) / 256

/-- A numerator of at most 256 never routes more experts than missed. -/
theorem pcieShare_le (nmiss p : Nat) (hp : p ≤ 256) : pcieShare nmiss p ≤ nmiss := by
  unfold pcieShare
  have h : nmiss * p ≤ nmiss * 256 := Nat.mul_le_mul_left _ hp
  apply Nat.div_le_of_le_mul
  omega

/-- A zero numerator disables the PCIe path. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

end Strata.ExpertSource
