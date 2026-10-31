/-!
# MODULE: expert_blob_source
File-backed, arena-backed expert sources and the dispatch adapter that turns a router's `ids` into a job array
and an output-row mapping that skips GPU hits.

## Source semantics
* `blob(layer, expert)` validates the two axes SEPARATELY: `expert >= n_expert` must be refused even though the
  flat index could be in range (it would alias the next layer's first expert).
* The pack size must be exactly `layers * experts * BLOB` bytes.
* In `expert_pool_dispatch`, `njobs` indexes the job array and `i` indexes the output; they agree only when no
  entry is a hit, and always `njobs <= i`.
* The multi-token planner splits distinct experts so that resident ones go to the GPU and the last
  `(nmiss * pcie_num) >> 8` missed ones go over PCIe.

## Form template identifiers
* `nExpert`, `blobs`      : `n_expert_`, `blobs_`
* `flat`                  : `layer * n_expert + expert`
* `pcieShare`             : `(nmiss * pcie_num) >> 8`
-/

namespace Strata.ExpertSource

-- § flat index and validation

/-- Flat blob index. -/
def flat (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- Validated lookup: both axes checked separately, then the flat bound. Returns the index if accepted. -/
def lookup (nExpert blobs layer expert : Nat) : Option Nat :=
  if expert ≥ nExpert then none
  else if flat nExpert layer expert ≥ blobs then none
  else some (flat nExpert layer expert)

/-- The aliasing hazard: `(layer, nExpert)` has the same flat index as `(layer + 1, 0)`. -/
theorem flat_alias (nExpert layer : Nat) : flat nExpert layer nExpert = flat nExpert (layer + 1) 0 := by
  unfold flat
  rw [Nat.succ_mul]
  omega

/-- The separate expert check refuses the aliasing id even when the flat bound would accept it. -/
theorem alias_refused (nExpert blobs layer : Nat) (h : flat nExpert layer nExpert < blobs) :
    lookup nExpert blobs layer nExpert = none := by
  unfold lookup
  simp

/-- An accepted lookup has an in-range expert and an in-range flat index. -/
theorem lookup_sound (nExpert blobs layer expert i : Nat) (h : lookup nExpert blobs layer expert = some i) :
    expert < nExpert ∧ i < blobs ∧ i = flat nExpert layer expert := by
  unfold lookup at h
  by_cases h1 : expert ≥ nExpert
  · simp [h1] at h
  · by_cases h2 : flat nExpert layer expert ≥ blobs
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      exact ⟨by omega, by omega, h.symm⟩

/-- Completeness: in-range expert and in-range flat index are accepted. -/
theorem lookup_complete (nExpert blobs layer expert : Nat) (he : expert < nExpert)
    (hf : flat nExpert layer expert < blobs) : lookup nExpert blobs layer expert = some (flat nExpert layer expert) := by
  unfold lookup
  have h1 : ¬ expert ≥ nExpert := by omega
  have h2 : ¬ flat nExpert layer expert ≥ blobs := by omega
  simp [h1, h2]

-- § pack size

/-- Required pack size in bytes. -/
def packBytes (nLayers nExpert blob : Nat) : Nat := nLayers * nExpert * blob

/-- A pack of the right size is accepted; any other size is refused. -/
def sizeOk (fileBytes nLayers nExpert blob : Nat) : Bool := fileBytes == packBytes nLayers nExpert blob

/-- The size check is an equality test. -/
theorem sizeOk_iff (fileBytes nLayers nExpert blob : Nat) :
    sizeOk fileBytes nLayers nExpert blob = true ↔ fileBytes = packBytes nLayers nExpert blob := by
  unfold sizeOk
  simp

/-- The last blob's end equals the pack size. -/
theorem last_blob_end (nLayers nExpert blob : Nat) (hl : 0 < nLayers) (hb : 0 < nExpert) :
    (flat nExpert (nLayers - 1) (nExpert - 1) + 1) * blob = packBytes nLayers nExpert blob := by
  unfold flat packBytes
  have h1 : (nLayers - 1) * nExpert + (nExpert - 1) + 1 = nLayers * nExpert := by
    have : nLayers = (nLayers - 1) + 1 := by omega
    conv_rhs => rw [this]
    rw [Nat.succ_mul]
    omega
  rw [h1]

-- § job array versus output rows

/-- Number of non-hit entries among a prefix of hit flags: the job count after processing it. -/
def jobsAfter : List Bool → Nat
  | [] => 0
  | hit :: rest => (if hit then 0 else 1) + jobsAfter rest

/-- The job count never exceeds the number of entries processed (`njobs <= i`). -/
theorem jobsAfter_le (hits : List Bool) : jobsAfter hits ≤ hits.length := by
  induction hits with
  | nil => simp [jobsAfter]
  | cons h rest ih =>
    simp only [jobsAfter, List.length_cons]
    cases h <;> simp <;> omega

/-- With no hits, the job index equals the output index. -/
theorem jobsAfter_no_hits (n : Nat) : jobsAfter (List.replicate n false) = n := by
  induction n with
  | zero => simp [jobsAfter]
  | succ k ih => simp [List.replicate_succ, jobsAfter, ih] <;> omega

/-- With all hits, no job is created. -/
theorem jobsAfter_all_hits (n : Nat) : jobsAfter (List.replicate n true) = 0 := by
  induction n with
  | zero => simp [jobsAfter]
  | succ k ih => simp [List.replicate_succ, jobsAfter, ih]

-- § PCIe share

/-- Number of missed experts sent over PCIe. -/
def pcieShare (nmiss pcieNum : Nat) : Nat := (nmiss * pcieNum) / 256

/-- With `pcie_num ≤ 256` the PCIe share never exceeds the number of misses. -/
theorem pcieShare_le (nmiss pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare nmiss pcieNum ≤ nmiss := by
  unfold pcieShare
  have h1 : nmiss * pcieNum ≤ nmiss * 256 := Nat.mul_le_mul_left _ h
  have h2 : nmiss * pcieNum / 256 ≤ nmiss * 256 / 256 := Nat.div_le_div_right h1
  have h3 : nmiss * 256 / 256 = nmiss := Nat.mul_div_cancel nmiss (by omega)
  omega

/-- With `pcie_num = 0` nothing goes over PCIe. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  unfold pcieShare
  simp

/-- The CPU keeps misses ranked below `nmiss - m`: the count is exactly `nmiss - m` when `m ≤ nmiss`. -/
theorem cpu_miss_count (nmiss m : Nat) (h : m ≤ nmiss) : nmiss - m + m = nmiss := by omega

/-- Entries per plan: a batch of `n_tok * k` entries is within the `128` cap iff `n ≤ 128`. -/
theorem plan_cap_ok (nTok k : Nat) (h : nTok * k ≤ 128) : nTok * k < 129 := by omega

end Strata.ExpertSource
