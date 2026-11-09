/-!
# MODULE: expert_source_dispatch
SOURCES: src/core/expert_source.cpp, include/strata/core/expert_source.hpp
Adapter from the host loop to the expert pool: blob addressing (`layer * n_expert + expert`), the job/hit split
by router index, and the PCIe share of missed experts in a verify window.

## Source semantics
* The flat blob index alone cannot detect `expert >= n_expert`; layer and expert are validated separately.
* Jobs are indexed by `njobs`, outputs by `i`; hits consume no job.
* The PCIe share is `(nmiss * pcie_num) >> 8` and never exceeds `nmiss` when `pcie_num ≤ 256`.

## Form template identifiers
blob(layer, expert) → blobIndex ; expert_pool_dispatch job count → jobCount ; GPU PCIe share → pcieShare
-/

namespace Strata.ExpertSource

-- § blob addressing

/-- Flat blob index. -/
def blobIndex (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- Why the separate `expert >= n_expert` check exists: `blob(0, n)` aliases `blob(1, 0)`. -/
theorem blobIndex_alias (nExpert : Nat) : blobIndex nExpert 0 nExpert = blobIndex nExpert 1 0 := by
  simp [blobIndex]

/-- A valid expert id stays inside its own layer's row. -/
theorem blobIndex_lt_succ (nExpert l e : Nat) (he : e < nExpert) :
    blobIndex nExpert l e < (l + 1) * nExpert := by
  unfold blobIndex
  have h : (l + 1) * nExpert = l * nExpert + nExpert := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- With both axes validated, the flat index is below `layers * n_expert`. -/
theorem blobIndex_lt_total (nExpert layers l e : Nat) (hl : l < layers) (he : e < nExpert) :
    blobIndex nExpert l e < layers * nExpert := by
  have h1 := blobIndex_lt_succ nExpert l e he
  have h2 : (l + 1) * nExpert ≤ layers * nExpert := Nat.mul_le_mul_right _ hl
  omega

/-- On validated axes the flat index is injective: distinct `(layer, expert)` never share a blob. -/
theorem blobIndex_inj (nExpert l e l' e' : Nat) (he : e < nExpert) (he' : e' < nExpert)
    (h : blobIndex nExpert l e = blobIndex nExpert l' e') : l = l' ∧ e = e' := by
  have key : ∀ a b a' b' : Nat, b < nExpert → b' < nExpert → a < a' →
      a * nExpert + b ≠ a' * nExpert + b' := by
    intro a b a' b' hb hb' hlt heq
    have h1 : (a + 1) * nExpert ≤ a' * nExpert := Nat.mul_le_mul_right _ hlt
    have h2 : (a + 1) * nExpert = a * nExpert + nExpert := by rw [Nat.add_mul, Nat.one_mul]
    omega
  unfold blobIndex at h
  rcases Nat.lt_trichotomy l l' with hl | hl | hl
  · exact absurd h (key _ _ _ _ he he' hl)
  · subst hl
    exact ⟨rfl, by omega⟩
  · exact absurd h.symm (key _ _ _ _ he' he hl)

-- § jobs vs hits

/-- Number of CPU jobs: the entries that are not GPU hits. -/
def jobCount (hits : List Bool) : Nat := (hits.filter (fun b => !b)).length

/-- `njobs` never exceeds `k`, the number of routed entries (so the job array is large enough). -/
theorem jobCount_le (hits : List Bool) : jobCount hits ≤ hits.length := by
  unfold jobCount
  exact List.length_filter_le _ _

/-- With no hits, every entry is a job and `njobs = i` indexing coincide. -/
theorem jobCount_no_hits (k : Nat) : jobCount (List.replicate k false) = k := by
  unfold jobCount
  simp

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) >>> 8

/-- The PCIe share never exceeds the number of misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  rw [Nat.shiftRight_eq_div_pow]
  have h1 : nmiss * pcieNum ≤ nmiss * 256 := Nat.mul_le_mul_left _ h
  omega

/-- With `pcie_num = 0` the CPU keeps every miss. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

end Strata.ExpertSource
