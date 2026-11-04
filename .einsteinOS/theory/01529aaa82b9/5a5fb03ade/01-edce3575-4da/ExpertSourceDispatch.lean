/-!
# MODULE: expert_source_dispatch
The expert source adapter (`expert_source.cpp`, `expert_source.hpp`): blob addressing in the file-backed and
arena sources, the pack-size check, the hit/miss job compaction used by `expert_pool_dispatch`, the PCIe share
arithmetic of the verify window, and the GGUF-to-blob chunked copy.

## Source semantics
* `blob(layer, expert)` is valid iff `layer < n_layers` and `expert < n_expert`; the flat index alone does not
  catch `expert >= n_expert` (it aliases the next layer's first expert), hence the separate axis check.
* A pack file is accepted iff its size is exactly `n_layers * n_expert * BLOB`.
* Job compaction: `njobs` indexes the job array and `i` the output; jobs are exactly the non-hit routed rows,
  in order, each pointing at a distinct output row.
* PCIe share `m = nmiss * pcie_num / 256` never exceeds `nmiss` when `pcie_num <= 256`.
* The GGUF load splits a blob as gate | up | down and copies 16 experts per read.

## Form template identifiers
* `FileExpertSource::blob` / `ArenaExpertSource::blob` -> `blobValid`, `blobIndex`
* `is_hit` / `njobs` / `jobs[njobs++]` -> `nMiss`, `nHit`, `jobRows`
* `(nmiss * pcie_num) >> 8` -> `nmiss * num / 256`
* `per[3]`, `at[3]` in `load_experts_gguf` -> `gateBytes`, `upBytes`, `downBytes`
-/

namespace ExpertSourceDispatch

-- § blob addressing

/-- Flat blob index `layer * n_expert + expert`. -/
def blobIndex (nExp layer expert : Nat) : Nat := layer * nExp + expert

/-- The source's validity rule: expert axis checked separately, flat index below the blob count. -/
def blobValid (nLayers nExp layer expert : Nat) : Prop :=
  expert < nExp ∧ blobIndex nExp layer expert < nLayers * nExp

/-- A blob is valid exactly when both the layer and the expert are in range. -/
theorem blobValid_iff (nLayers nExp layer expert : Nat) :
    blobValid nLayers nExp layer expert ↔ layer < nLayers ∧ expert < nExp := by
  constructor
  · rintro ⟨he, hi⟩
    refine ⟨?_, he⟩
    by_contra h
    have h1 : nLayers * nExp ≤ layer * nExp := Nat.mul_le_mul_right _ (by omega)
    unfold blobIndex at hi
    omega
  · rintro ⟨hl, he⟩
    refine ⟨he, ?_⟩
    unfold blobIndex
    have h2 : (layer + 1) * nExp ≤ nLayers * nExp := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.add_mul, Nat.one_mul] at h2
    omega

/-- One past the end of a layer's experts aliases the next layer's first expert in flat index space. -/
theorem flat_alias (nExp : Nat) : blobIndex nExp 0 nExp = blobIndex nExp 1 0 := by
  simp [blobIndex]

/-- The flat index alone is NOT enough: `(0, n_expert)` is below the blob count yet invalid. -/
theorem flat_check_insufficient (nLayers nExp : Nat) (hl : 2 ≤ nLayers) (hn : 0 < nExp) :
    blobIndex nExp 0 nExp < nLayers * nExp ∧ ¬ blobValid nLayers nExp 0 nExp := by
  have h2 : 2 * nExp ≤ nLayers * nExp := Nat.mul_le_mul_right _ hl
  refine ⟨?_, ?_⟩
  · simp [blobIndex]
    omega
  · intro h
    exact absurd h.1 (Nat.lt_irrefl _)

/-- A valid blob's byte range lies inside the mapped pack of `n_layers * n_expert * BLOB` bytes. -/
theorem blob_in_pack (nLayers nExp layer expert blob : Nat)
    (h : blobValid nLayers nExp layer expert) :
    (blobIndex nExp layer expert + 1) * blob ≤ nLayers * nExp * blob := by
  exact Nat.mul_le_mul_right _ h.2

/-- A pack file is accepted iff its size is exactly `n_layers * n_expert * BLOB`. -/
def packSizeOk (size nLayers nExp blob : Nat) : Prop := size = nLayers * nExp * blob

/-- The artifact's pack: 48 layers x 512 experts x 1,382,400 B is 33,973,862,400 B. -/
theorem pack_size_concrete : 48 * 512 * 1382400 = 33973862400 := by decide

-- § job compaction in expert_pool_dispatch

/-- Number of non-hit rows (jobs the CPU pool must run). -/
def nMiss : List Bool → Nat
  | [] => 0
  | b :: l => (if b then 0 else 1) + nMiss l

/-- Number of hit rows (rows the GPU owns). -/
def nHit : List Bool → Nat
  | [] => 0
  | b :: l => (if b then 1 else 0) + nHit l

/-- Every routed row is either a hit or a miss, never both and never neither. -/
theorem nMiss_add_nHit (l : List Bool) : nMiss l + nHit l = l.length := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    cases b <;> simp [nMiss, nHit] <;> omega

/-- The output rows of the jobs, in order, starting from output index `i`. -/
def jobRows : List Bool → Nat → List Nat
  | [], _ => []
  | b :: l, i => if b then jobRows l (i + 1) else i :: jobRows l (i + 1)

/-- The job array has exactly one entry per miss. -/
theorem jobRows_length : ∀ (l : List Bool) (i : Nat), (jobRows l i).length = nMiss l
  | [], _ => rfl
  | b :: l, i => by
    cases b <;> simp [jobRows, nMiss, jobRows_length l (i + 1)] <;> omega

/-- Every job points at an output row inside the layer's `k` rows. -/
theorem mem_jobRows : ∀ (l : List Bool) (i r : Nat), r ∈ jobRows l i → i ≤ r ∧ r < i + l.length
  | [], i, r, h => by simp [jobRows] at h
  | b :: l, i, r, h => by
    cases b
    · simp [jobRows] at h
      rcases h with h | h
      · subst h
        rw [List.length_cons]
        omega
      · have := mem_jobRows l (i + 1) r h
        rw [List.length_cons]
        omega
    · simp [jobRows] at h
      have := mem_jobRows l (i + 1) r h
      rw [List.length_cons]
      omega

/-- No two jobs share an output row: a hit's row can never get two experts summed into it. -/
theorem jobRows_nodup : ∀ (l : List Bool) (i : Nat), (jobRows l i).Nodup
  | [], _ => by simp [jobRows]
  | b :: l, i => by
    cases b
    · simp only [jobRows, Bool.false_eq_true, if_false]
      rw [List.nodup_cons]
      refine ⟨?_, jobRows_nodup l (i + 1)⟩
      intro hmem
      have := (mem_jobRows l (i + 1) i hmem).1
      omega
    · simp only [jobRows, if_true]
      exact jobRows_nodup l (i + 1)

/-- A hit row is never a job row: the pool cannot overwrite the GPU's rows. -/
theorem hit_row_not_job (l : List Bool) (i r : Nat) (hr : l.getD (r - i) false = true)
    (hri : i ≤ r) : r ∉ jobRows l i := by
  induction l generalizing i with
  | nil => simp [jobRows]
  | cons b l ih =>
    intro hmem
    by_cases hr0 : r = i
    · subst hr0
      simp at hr
      subst hr
      simp [jobRows] at hmem
      have := (mem_jobRows l (r + 1) r hmem).1
      omega
    · have hge : i + 1 ≤ r := by omega
      have hr' : l.getD (r - (i + 1)) false = true := by
        have : r - i = (r - (i + 1)) + 1 := by omega
        rw [this] at hr
        simpa using hr
      cases b
      · simp [jobRows] at hmem
        rcases hmem with h | h
        · exact hr0 h
        · exact ih (i + 1) hr' hge h
      · simp [jobRows] at hmem
        exact ih (i + 1) hr' hge hmem

-- § PCIe share of the verify window

/-- The GPU's PCIe share `m = nmiss * num / 256` never exceeds the number of misses when `num <= 256`. -/
theorem pcie_share_le (nmiss num : Nat) (h : num ≤ 256) : nmiss * num / 256 ≤ nmiss := by
  have : nmiss * num ≤ nmiss * 256 := Nat.mul_le_mul_left _ h
  omega

/-- With `num = 0` the PCIe share is empty. -/
theorem pcie_share_zero (nmiss : Nat) : nmiss * 0 / 256 = 0 := by
  simp

-- § the GGUF blob split in load_experts_gguf

/-- Gate bytes per expert: the up tensor starts at `up_off`, so the gate occupies `[0, up_off)`. -/
def gateBytes (upOff : Nat) : Nat := upOff

/-- Up bytes per expert: `[up_off, down_off)` with `down_off = 2 * up_off`. -/
def upBytes (upOff : Nat) : Nat := upOff

/-- Down bytes per expert: `[down_off, blob)`. -/
def downBytes (blob downOff : Nat) : Nat := blob - downOff

/-- The three copied pieces tile the blob exactly when the layout puts down at twice the gate size. -/
theorem blob_split_tiles (blob upOff downOff : Nat) (h1 : downOff = 2 * upOff) (h2 : downOff ≤ blob) :
    gateBytes upOff + upBytes upOff + downBytes blob downOff = blob := by
  unfold gateBytes upBytes downBytes
  omega

/-- A read of 16 experts divides the 512 experts of a layer into exactly 32 reads. -/
theorem reads_tile_layer (per : Nat) : per * 512 = per * 16 * 32 := by
  omega

end ExpertSourceDispatch
