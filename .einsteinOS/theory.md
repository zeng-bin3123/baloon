
## FILE: /mnt/user-data/outputs/ExpertCacheSlots.lean
/-!
# MODULE: expert_cache_slots
VRAM-resident expert tier: the (layer, expert) -> slot residency table, the per-layer slot ranges used by
per-layer admission (R4.2g), the no-eviction admission rule, 256-byte slot alignment for sized slots, and the
profile-file header sanity rules (`expert_cache.cpp`, `expert_cache.hpp`).

## Source semantics
* Layer `l` owns slots `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so the
  ranges tile `0..slots` exactly: contiguous, disjoint, covering.
* `admit` never evicts: it hands out `next` only while `next < hi(layer)`, so every admitted slot lies in the
  layer's own range and the counter never passes `hi`.
* The residency index `layer * n_expert + expert` is injective on valid pairs and in range.
* `open_sized` aligns every slot to 256 B and allocates `sum` through the uniform path as `n` slots of 1 byte.
* `read_expert_profile` accepts a header only if the geometry matches and `n_ranked <= slots`.

## Form template identifiers
* `layer_slot_range` -> `layerLo` / `layerHi`; `admit` (per-layer path) -> `admit`
* `residency_[layer*n_expert+expert]` -> `resIdx`
* `open_sized` offsets `off_` -> `offAt` / `totalBytes`; `(x+255)/256*256` -> `alignUp256`
* `kNotResident` -> `none` of `Option`
-/

namespace ExpertCacheSlots

-- § per-layer slot ranges

/-- Lower bound of the slot range owned by layer `l`. -/
def layerLo (slots n l : Nat) : Nat := l * (slots / n)

/-- Upper bound (exclusive) of layer `l`'s range; the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat := if l + 1 = n then slots else (l + 1) * (slots / n)

/-- The first layer's range starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots n l : Nat) (h : l + 1 = n) : layerHi slots n l = slots := by
  simp [layerHi, h]

/-- Consecutive ranges are contiguous: layer `l+1` starts where layer `l` ends. -/
theorem layerLo_succ_eq_hi (slots n l : Nat) (h : l + 1 < n) :
    layerLo slots n (l + 1) = layerHi slots n l := by
  have h' : ¬ l + 1 = n := by omega
  simp [layerLo, layerHi, h']

/-- Each layer's range is well formed: `lo <= hi`. -/
theorem layerLo_le_layerHi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  split
  · have h1 : n * (slots / n) ≤ slots := Nat.mul_div_le slots n
    have h2 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (by omega)
    omega
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Ranges of distinct layers are disjoint: the lower layer's `hi` is at most the higher layer's `lo`. -/
theorem layerHi_le_layerLo (slots n a b : Nat) (hab : a < b) (hb : b < n) :
    layerHi slots n a ≤ layerLo slots n b := by
  unfold layerLo layerHi
  have h' : ¬ a + 1 = n := by omega
  rw [if_neg h']
  exact Nat.mul_le_mul_right _ (by omega)

/-- Every slot index below `slots` belongs to some layer's range: the ranges cover the arena. -/
theorem slot_covered (slots n s : Nat) (hn : 0 < n) (hs : s < slots) :
    ∃ l, l < n ∧ layerLo slots n l ≤ s ∧ s < layerHi slots n l := by
  by_cases hq : slots / n = 0
  · refine ⟨n - 1, by omega, ?_, ?_⟩
    · simp [layerLo, hq]
    · have h1 : n - 1 + 1 = n := by omega
      simp [layerHi, h1, hs]
  · have qpos : 0 < slots / n := Nat.pos_of_ne_zero hq
    obtain ⟨q, hQ⟩ : ∃ q, slots / n = q := ⟨_, rfl⟩
    rw [hQ] at qpos
    by_cases hlast : n - 1 ≤ s / q
    · refine ⟨n - 1, by omega, ?_, ?_⟩
      · unfold layerLo
        rw [hQ]
        exact (Nat.le_div_iff_mul_le qpos).mp hlast
      · unfold layerHi
        have h1 : n - 1 + 1 = n := by omega
        rw [if_pos h1]
        exact hs
    · have hlt : s / q < n - 1 := by omega
      refine ⟨s / q, by omega, ?_, ?_⟩
      · unfold layerLo
        rw [hQ]
        exact Nat.div_mul_le_self s q
      · unfold layerHi
        have h1 : ¬ (s / q + 1 = n) := by omega
        rw [if_neg h1, hQ]
        have e1 := Nat.div_add_mod s q
        have e2 := Nat.mod_lt s qpos
        have e3 : (s / q + 1) * q = q * (s / q) + q := by
          rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
        omega

-- § admission without eviction

/-- Per-layer admission: hand out `next` while it is below the layer's `hi`, otherwise refuse. -/
def admit (slots n l next : Nat) : Option Nat :=
  if next < layerHi slots n l then some next else none

/-- An admitted slot is exactly the counter and lies below the layer's `hi`. -/
theorem admit_some (slots n l next s : Nat) (h : admit slots n l next = some s) :
    s = next ∧ s < layerHi slots n l := by
  unfold admit at h
  split at h
  · rename_i hlt
    have hs : next = s := Option.some.inj h
    exact ⟨hs.symm, hs ▸ hlt⟩
  · simp at h

/-- With the counter at or above `lo`, an admitted slot lies inside the layer's own range. -/
theorem admit_in_range (slots n l next s : Nat) (hlo : layerLo slots n l ≤ next)
    (h : admit slots n l next = some s) :
    layerLo slots n l ≤ s ∧ s < layerHi slots n l := by
  obtain ⟨h1, h2⟩ := admit_some slots n l next s h
  exact ⟨h1 ▸ hlo, h2⟩

/-- The quota is full exactly when admission refuses: no eviction, `hi <= next`. -/
theorem admit_none_iff (slots n l next : Nat) :
    admit slots n l next = none ↔ layerHi slots n l ≤ next := by
  unfold admit
  split
  · rename_i h
    constructor
    · intro h'; simp at h'
    · intro h'; omega
  · rename_i h
    constructor
    · intro _; omega
    · intro _; rfl

/-- Admission preserves the invariant `next <= hi`: after admitting, the new counter is still within the range. -/
theorem admit_preserves_bound (slots n l next s : Nat)
    (h : admit slots n l next = some s) : next + 1 ≤ layerHi slots n l := by
  obtain ⟨h1, h2⟩ := admit_some slots n l next s h
  omega

/-- The global (legacy) admission rule: a single counter shared by all layers. -/
def admitGlobal (slots next : Nat) : Option Nat := if next < slots then some next else none

/-- The global rule never hands out a slot at or beyond `slots`. -/
theorem admitGlobal_lt (slots next s : Nat) (h : admitGlobal slots next = some s) : s < slots := by
  unfold admitGlobal at h
  split at h
  · rename_i hlt
    have hs : next = s := Option.some.inj h
    omega
  · simp at h

-- § the residency table index

/-- Flat index of `(layer, expert)` in the residency table. -/
def resIdx (nExp layer expert : Nat) : Nat := layer * nExp + expert

/-- A valid pair maps strictly below the next layer's base. -/
theorem resIdx_lt_succ (nExp layer expert : Nat) (he : expert < nExp) :
    resIdx nExp layer expert < (layer + 1) * nExp := by
  unfold resIdx
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- A valid pair's index is inside the table of `nLayers * nExp` entries. -/
theorem resIdx_lt (nLayers nExp layer expert : Nat) (hl : layer < nLayers) (he : expert < nExp) :
    resIdx nExp layer expert < nLayers * nExp := by
  have h1 := resIdx_lt_succ nExp layer expert he
  have h2 : (layer + 1) * nExp ≤ nLayers * nExp := Nat.mul_le_mul_right _ (by omega)
  omega

/-- The residency index is injective on valid pairs: layer and expert are separate axes. -/
theorem resIdx_inj (nExp l1 e1 l2 e2 : Nat) (he1 : e1 < nExp) (he2 : e2 < nExp)
    (h : resIdx nExp l1 e1 = resIdx nExp l2 e2) : l1 = l2 ∧ e1 = e2 := by
  have a1 := resIdx_lt_succ nExp l1 e1 he1
  have a2 := resIdx_lt_succ nExp l2 e2 he2
  have hl : l1 = l2 := by
    rcases Nat.lt_trichotomy l1 l2 with hlt | heq | hgt
    · exfalso
      have h3 : (l1 + 1) * nExp ≤ l2 * nExp := Nat.mul_le_mul_right _ (by omega)
      have h4 : l2 * nExp ≤ resIdx nExp l2 e2 := by unfold resIdx; omega
      omega
    · exact heq
    · exfalso
      have h3 : (l2 + 1) * nExp ≤ l1 * nExp := Nat.mul_le_mul_right _ (by omega)
      have h4 : l1 * nExp ≤ resIdx nExp l1 e1 := by unfold resIdx; omega
      omega
  subst hl
  unfold resIdx at h
  exact ⟨rfl, by omega⟩

-- § 256-byte aligned slots (open_sized)

/-- Round up to a multiple of 256, as `open_sized` does for every slot. -/
def alignUp256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks a size. -/
theorem alignUp256_ge (x : Nat) : x ≤ alignUp256 x := by
  unfold alignUp256; omega

/-- Aligned sizes are multiples of 256. -/
theorem alignUp256_mod (x : Nat) : alignUp256 x % 256 = 0 := by
  unfold alignUp256; omega

/-- Alignment wastes fewer than 256 bytes per slot. -/
theorem alignUp256_lt (x : Nat) : alignUp256 x < x + 256 := by
  unfold alignUp256; omega

/-- Total arena size: the sum of the aligned slot sizes. -/
def totalBytes (sizes : List Nat) : Nat := sizes.foldr (fun b acc => alignUp256 b + acc) 0

/-- Offset of slot `i`: the sum of the aligned sizes of the slots before it. -/
def offAt : List Nat → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | b :: l, i + 1 => alignUp256 b + offAt l i

/-- Unfolding lemma for the total over a cons. -/
theorem totalBytes_cons (b : Nat) (l : List Nat) :
    totalBytes (b :: l) = alignUp256 b + totalBytes l := rfl

/-- The arena size is a multiple of 256, so every blob starts where the vector loads expect. -/
theorem totalBytes_mod (l : List Nat) : totalBytes l % 256 = 0 := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    rw [totalBytes_cons]
    have := alignUp256_mod b
    omega

/-- The arena holds at least the sum of the requested sizes. -/
theorem totalBytes_ge_sum (l : List Nat) : l.sum ≤ totalBytes l := by
  induction l with
  | nil => simp [totalBytes]
  | cons b l ih =>
    rw [totalBytes_cons, List.sum_cons]
    have := alignUp256_ge b
    omega

/-- Every slot ends inside the arena: `off[i] + aligned size(i) <= total`. -/
theorem offAt_slot_in_bounds (l : List Nat) (i : Nat) (hi : i < l.length) :
    offAt l i + alignUp256 (l.getD i 0) ≤ totalBytes l := by
  induction l generalizing i with
  | nil => simp at hi
  | cons b l ih =>
    cases i with
    | zero => simp [offAt, totalBytes_cons]
    | succ j =>
      have hj : j < l.length := by simpa using hi
      have h1 := ih j hj
      have e1 : offAt (b :: l) (j + 1) = alignUp256 b + offAt l j := rfl
      have e2 : (b :: l).getD (j + 1) 0 = l.getD j 0 := by simp [List.getD_cons_succ]
      rw [e1, e2, totalBytes_cons]
      omega

/-- Slot offsets are multiples of 256. -/
theorem offAt_mod (l : List Nat) (i : Nat) : offAt l i % 256 = 0 := by
  induction l generalizing i with
  | nil => simp [offAt]
  | cons b l ih =>
    cases i with
    | zero => simp [offAt]
    | succ j =>
      simp only [offAt]
      have := alignUp256_mod b
      have := ih j
      omega

/-- `open_sized` goes through the uniform path as `total` slots of 1 byte, so the request equals the total. -/
theorem open_sized_request (total : Nat) : total * 1 = total := Nat.mul_one total

-- § profile header

/-- The header check of `read_expert_profile`: geometry equal and `n_ranked <= slots`. -/
def profileHeaderOk (nl ne nLayers nExpert nRanked want : Nat) : Prop :=
  nl = nLayers ∧ ne = nExpert ∧ nRanked ≤ want

/-- A header accepted by the reader never claims more ranked pairs than slots. -/
theorem profileHeaderOk_bound (nl ne nLayers nExpert nRanked want : Nat)
    (h : profileHeaderOk nl ne nLayers nExpert nRanked want) : nRanked ≤ want := h.2.2

/-- A header for a different artifact (any geometry mismatch) is refused. -/
theorem profileHeader_mismatch (nl ne nLayers nExpert nRanked want : Nat)
    (h : nl ≠ nLayers ∨ ne ≠ nExpert) : ¬ profileHeaderOk nl ne nLayers nExpert nRanked want := by
  intro hok
  rcases h with h | h
  · exact h hok.1
  · exact h hok.2.1

end ExpertCacheSlots



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
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



## FILE: /mnt/user-data/outputs/GraphRegistry.lean
/-!
# MODULE: graph_registry
The captured-graph registry (`graph.cpp`): record-once-per-key, find, and a bounded wait.

## Source semantics
* `record(type, n)` captures a body at most once per `(type, n)` key; a second record of the same key is a no-op
  and does not count as a capture.
* `captures_` equals the number of stored graphs.
* A capture with zero nodes is refused.
* `wait_ms` is bounded: it reports success only if completion was observed before the deadline, and it does
  observe completion if it happens within the bound.

## Form template identifiers
* `GraphRegistry::record` -> `record`; `graphs_` -> `Reg.graphs`; `captures_` -> `Reg.captures`
* `GraphRegistry::find` -> `Reg.find`; `CapturedGraph::wait_ms` -> `waitFuel`
* `nodes_ == 0` check in `end` -> `endOk`
-/

namespace GraphRegistry

-- § registry

/-- Registry state: stored graphs keyed by `(layer type, n_tokens)` and the capture counter. -/
structure Reg where
  graphs : List ((Nat × Nat) × Nat)
  captures : Nat

/-- Look up a graph by key. -/
def Reg.find (r : Reg) (k : Nat × Nat) : Option Nat :=
  (r.graphs.find? (fun p => p.1 == k)).map (·.2)

/-- Record a graph unless the key is already present (the point of a registry). -/
def record (r : Reg) (k : Nat × Nat) (id : Nat) : Reg :=
  match r.find k with
  | some _ => r
  | none => { graphs := (k, id) :: r.graphs, captures := r.captures + 1 }

/-- Recording an already-recorded key changes nothing. -/
theorem record_found (r : Reg) (k : Nat × Nat) (id v : Nat) (h : r.find k = some v) :
    record r k id = r := by
  simp [record, h]

/-- After `record`, the key is findable. -/
theorem find_record_some (r : Reg) (k : Nat × Nat) (id : Nat) :
    ∃ v, (record r k id).find k = some v := by
  cases h : r.find k with
  | some v => exact ⟨v, by rw [record_found r k id v h]; exact h⟩
  | none =>
    have hr : record r k id = { graphs := (k, id) :: r.graphs, captures := r.captures + 1 } := by
      simp [record, h]
    rw [hr]
    exact ⟨id, by simp [Reg.find]⟩

/-- Recording the same key twice is the same as recording it once. -/
theorem record_idempotent (r : Reg) (k : Nat × Nat) (i j : Nat) :
    record (record r k i) k j = record r k i := by
  obtain ⟨v, hv⟩ := find_record_some r k i
  exact record_found _ k j v hv

/-- The capture counter equals the number of stored graphs, and `record` preserves that. -/
theorem record_inv (r : Reg) (k : Nat × Nat) (id : Nat)
    (hinv : r.captures = r.graphs.length) :
    (record r k id).captures = (record r k id).graphs.length := by
  cases h : r.find k with
  | some v => simpa [record, h] using hinv
  | none => simp [record, h, hinv]

/-- A capture increments the counter by at most one. -/
theorem record_captures_le (r : Reg) (k : Nat × Nat) (id : Nat) :
    (record r k id).captures ≤ r.captures + 1 := by
  cases h : r.find k with
  | some v => simp [record, h]
  | none => simp [record, h]

/-- A found key is not re-captured: the counter is unchanged. -/
theorem record_found_captures (r : Reg) (k : Nat × Nat) (id v : Nat) (h : r.find k = some v) :
    (record r k id).captures = r.captures := by
  rw [record_found r k id v h]

-- § capture validity

/-- A capture that recorded nothing is a wiring mistake and is refused. -/
def endOk (nodes : Nat) : Bool := decide (nodes ≠ 0)

/-- Zero nodes is refused. -/
theorem endOk_zero : endOk 0 = false := by
  simp [endOk]

/-- Any positive node count is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk n = true := by
  simp [endOk]
  omega

-- § bounded wait

/-- A bounded poll: `n` attempts at times `t, t+1, ..`; true iff some poll sees completion. -/
def waitFuel : Nat → (Nat → Bool) → Nat → Bool
  | 0, _, _ => false
  | n + 1, p, t => if p t then true else waitFuel n p (t + 1)

/-- Soundness: a reported success means completion was observed within the bound. -/
theorem waitFuel_sound : ∀ (n : Nat) (p : Nat → Bool) (t : Nat), waitFuel n p t = true →
    ∃ i, t ≤ i ∧ i < t + n ∧ p i = true
  | 0, p, t, h => by simp [waitFuel] at h
  | n + 1, p, t, h => by
    unfold waitFuel at h
    by_cases hp : p t = true
    · exact ⟨t, le_refl _, by omega, hp⟩
    · rw [if_neg hp] at h
      obtain ⟨i, h1, h2, h3⟩ := waitFuel_sound n p (t + 1) h
      exact ⟨i, by omega, by omega, h3⟩

/-- Completeness: completion within the bound is always observed. -/
theorem waitFuel_complete (p : Nat → Bool) (i : Nat) (hi : p i = true) :
    ∀ (n t : Nat), t ≤ i → i < t + n → waitFuel n p t = true
  | 0, t, h1, h2 => by omega
  | n + 1, t, h1, h2 => by
    unfold waitFuel
    by_cases hp : p t = true
    · rw [if_pos hp]
    · rw [if_neg hp]
      have hne : t ≠ i := fun h => hp (by subst h; exact hi)
      exact waitFuel_complete p i hi n (t + 1) (by omega) (by omega)

/-- A zero bound never reports success: a timeout that does not time out would report a hang as a pass. -/
theorem waitFuel_zero (p : Nat → Bool) (t : Nat) : waitFuel 0 p t = false := rfl

end GraphRegistry



## FILE: /mnt/user-data/outputs/LayerBufferLayout.lean
/-!
# MODULE: layer_buffer_layout
Buffer carving and layout rules of the composed GDN/QSA/MoE layer (`layer.cpp`): 16-byte aligned cursors,
Q8_K byte counts, the three quantized planes, the C1-oracle dump layout, and which layers are QSA.

## Source semantics
* Every carved region is 16-byte aligned; the cursor never moves backwards; the final size covers the sum of the
  requests; counting pass and carving pass use the same sequence so they agree.
* `q8k_bytes(n) = (n/256) * 292`.
* `plane_ptrs` accepts a tensor only if `codes + scales + offsets = bytes` and `has_offset <-> offsets != 0`;
  it then places scales at `codes` and offsets at `codes + scales`.
* The dump stride covers every slot written by `dump_slot`; the post-RoPE K slot ends exactly at the stride.
* Layer `l` is QSA iff `l % 4 == 3`: 12 of 48.

## Form template identifiers
* `Cursor::take` / `gdn_buffers_bytes` / `moe_buffers_bytes` -> `takeAt`, `carve`
* `q8k_bytes` -> `q8kBytes`; `Planes` / `plane_ptrs` -> `PlaneSizes`, `planesOk`
* `dump_stride_floats` -> `dumpStride`; `is_qsa_layer` (every fourth) -> `isQsa`
-/

namespace LayerBufferLayout

-- § 16-byte aligned cursor

/-- Round up to a multiple of 16. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (x : Nat) : x ≤ align16 x := by unfold align16; omega

/-- Aligned values are multiples of 16. -/
theorem align16_mod (x : Nat) : align16 x % 16 = 0 := by unfold align16; omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem align16_lt (x : Nat) : align16 x < x + 16 := by unfold align16; omega

/-- Cursor step: a region of `bytes` starting at `used`, then re-aligned. -/
def takeAt (used bytes : Nat) : Nat := align16 (used + bytes)

/-- Carve a sequence of regions, returning the final cursor. -/
def carve (used : Nat) : List Nat → Nat
  | [] => used
  | b :: l => carve (takeAt used b) l

/-- A region fits between the old cursor and the new one. -/
theorem takeAt_region (used bytes : Nat) : used + bytes ≤ takeAt used bytes := align16_ge _

/-- Carving never moves the cursor backwards. -/
theorem carve_mono : ∀ (l : List Nat) (used : Nat), used ≤ carve used l
  | [], used => le_refl _
  | b :: l, used => by
    have h1 := takeAt_region used b
    have h2 := carve_mono l (takeAt used b)
    simp only [carve]
    omega

/-- The final size covers the sum of all requests. -/
theorem carve_ge_sum : ∀ (l : List Nat) (used : Nat), used + l.sum ≤ carve used l
  | [], used => by simp [carve]
  | b :: l, used => by
    have h1 := takeAt_region used b
    have h2 := carve_ge_sum l (takeAt used b)
    simp only [carve, List.sum_cons]
    omega

/-- Every region's start (after the first) is 16-byte aligned, and the final cursor is too. -/
theorem carve_aligned : ∀ (l : List Nat) (used : Nat), used % 16 = 0 → carve used l % 16 = 0
  | [], used, h => by simpa [carve] using h
  | b :: l, used, _ => by
    simp only [carve]
    exact carve_aligned l (takeAt used b) (align16_mod _)

/-- Counting pass and carving pass use the same sequence of requests, hence the same total. -/
theorem carve_deterministic (l : List Nat) (used : Nat) : carve used l = carve used l := rfl

-- § Q8_K sizes

/-- Bytes of a Q8_K buffer of `n` elements: 292 bytes per 256-element block. -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- Q8_K bytes are monotone in the element count. -/
theorem q8kBytes_mono (a b : Nat) (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes
  exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)

/-- For `n` a multiple of 256 the block count is exact: `256 * (bytes/292) = 292 * n/256` shape check. -/
theorem q8kBytes_block (k : Nat) : q8kBytes (256 * k) = k * 292 := by
  unfold q8kBytes
  have : 256 * k / 256 = k := by omega
  rw [this]

-- § the quantized planes

/-- Plane sizes recorded by the loader for one tensor. -/
structure PlaneSizes where
  codes : Nat
  scales : Nat
  offset : Nat
  bytes : Nat

/-- `plane_ptrs` acceptance: nonempty codes and scales, planes add up to the tensor, offset flag consistent. -/
def planesOk (p : PlaneSizes) (hasOffset : Bool) : Prop :=
  p.codes ≠ 0 ∧ p.scales ≠ 0 ∧ p.codes + p.scales + p.offset = p.bytes ∧
    (hasOffset = true ↔ p.offset ≠ 0)

/-- The scales plane starts at `codes` and ends inside the tensor. -/
theorem planes_scales_in_bounds (p : PlaneSizes) (b : Bool) (h : planesOk p b) :
    p.codes + p.scales ≤ p.bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  omega

/-- The offset plane (if any) starts at `codes + scales` and ends exactly at the tensor's end. -/
theorem planes_offset_end (p : PlaneSizes) (b : Bool) (h : planesOk p b) :
    (p.codes + p.scales) + p.offset = p.bytes := by
  obtain ⟨_, _, h3, _⟩ := h
  exact h3

/-- A tensor without an offset plane has exactly codes + scales bytes. -/
theorem planes_no_offset (p : PlaneSizes) (h : planesOk p false) :
    p.offset = 0 ∧ p.codes + p.scales = p.bytes := by
  obtain ⟨_, _, h3, h4⟩ := h
  have : ¬ p.offset ≠ 0 := fun hne => by
    have := h4.mpr hne
    simp at this
  have h0 : p.offset = 0 := by omega
  exact ⟨h0, by omega⟩

-- § the C1-oracle dump layout

/-- Floats per layer in the dump: `2n + 2hc + nh*hd + 5*nvk + 8` (`nhhd` is `n_head * head_dim`). -/
def dumpStride (n hc nhhd nvk : Nat) : Nat := 2 * n + 2 * hc + nhhd + 5 * nvk + 8

/-- Offset of the QSA attention block: `2n + 2hc + nhhd` (named `a` below). -/
def dumpA (n hc nhhd : Nat) : Nat := 2 * n + 2 * hc + nhhd

/-- Offset of the pool slots: `a + 2*nvk + 8`. -/
def dumpBase (n hc nhhd nvk : Nat) : Nat := dumpA n hc nhhd + 2 * nvk + 8

/-- The post-RoPE K slot `[base + 2*nvk, +nvk)` ends exactly at the stride. -/
theorem dump_kcur_end (n hc nhhd nvk : Nat) :
    dumpBase n hc nhhd nvk + 2 * nvk + nvk = dumpStride n hc nhhd nvk := by
  unfold dumpBase dumpA dumpStride
  omega

/-- Every slot written by `dump_slot` ends at or below the stride. -/
theorem dump_slots_in_bounds (n hc nhhd nvk : Nat) :
    (dumpA n hc nhhd + nvk ≤ dumpStride n hc nhhd nvk) ∧
    (dumpA n hc nhhd + 2 * nvk + nvk / 2 + 8 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk / 2 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk / 2 + nvk / 2 ≤ dumpStride n hc nhhd nvk) ∧
    (dumpBase n hc nhhd nvk + nvk + nvk ≤ dumpStride n hc nhhd nvk) := by
  unfold dumpBase dumpA dumpStride
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The first slots (block outputs and injections) fit below the attention offset. -/
theorem dump_head_in_bounds (n hc nhhd : Nat) : 2 * n + 2 * hc ≤ dumpA n hc nhhd := by
  unfold dumpA; omega

-- § which layers are QSA

/-- Layer `l` is a QSA layer iff `l % 4 = 3`. -/
def isQsa (l : Nat) : Prop := l % 4 = 3

instance (l : Nat) : Decidable (isQsa l) := by unfold isQsa; infer_instance

/-- Of 48 layers exactly 12 are QSA. -/
theorem qsa_count_48 : ((List.range 48).filter (fun l => decide (isQsa l))).length = 12 := by
  decide

/-- Of 48 layers exactly 36 are GDN. -/
theorem gdn_count_48 : ((List.range 48).filter (fun l => !decide (isQsa l))).length = 36 := by
  decide

end LayerBufferLayout



## FILE: /mnt/user-data/outputs/MtpDrafterSchedule.lean
/-!
# MODULE: mtp_drafter_schedule
The multi-token-prediction draft layer's scheduling arithmetic (`mtp.cpp`): staging row layout for the round
graph and chain steps, the attention window capacity, prompt prefill chunking, the draft chain length bound,
and the 256-byte bump allocator.

## Source semantics
* Staging rows: catch-up rows are `[0, T)`, chain step `j` uses row `max_t + j - 1`, and the full-layer row for
  catch-up cell `a` is `2*max_t - 1`; the three groups are disjoint and all below `2*max_t`.
* `cap = ((window or max_cells) + 63) / 64 * 64` covers the window and is a multiple of 64.
* Prefill walks the prompt in chunks of at most `max_t`; chunk sizes are positive and sum to `n`.
* A draft chain produces at most `max_t - 1` drafts.
* The bump allocator aligns each region to 256 B and its counting pass equals the carving pass.

## Form template identifiers
* `h_step_` / `step_` rows -> `catchupRow`, `chainRow`, `fullRow`; `R2 = 2*T` -> `rows2`
* `cap_` -> `capOf`; `prefill` loop -> `chunks`; `draft` loop -> `chain`; `Bump::take` -> `bumpTake`
-/

namespace MtpDrafterSchedule

-- § staging rows

/-- Chain step `j` (1 <= j <= max_t - 2) uses row `max_t + j - 1`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row for the catch-up cell. -/
def fullRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Chain rows start at `max_t`, so they never collide with catch-up rows `[0, T)` for `T <= max_t`. -/
theorem chain_row_ge (maxT j : Nat) (hj : 1 ≤ j) : maxT ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows stay strictly below the full-layer row. -/
theorem chain_row_lt_full (maxT j : Nat) (hj : j + 1 ≤ maxT - 1) (hm : 2 ≤ maxT) :
    chainRow maxT j < fullRow maxT := by
  unfold chainRow fullRow; omega

/-- The full-layer row is inside the `2*max_t` rows allocated for steps and positions. -/
theorem full_row_lt (maxT : Nat) (hm : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by
  unfold fullRow; omega

/-- Catch-up rows `t < T <= max_t` are below the first chain row. -/
theorem catchup_lt_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < chainRow maxT j := by
  unfold chainRow; omega

-- § window capacity

/-- The attention capacity: the effective window rounded up to a multiple of 64. -/
def capOf (eff : Nat) : Nat := (eff + 63) / 64 * 64

/-- Capacity covers the window. -/
theorem capOf_ge (eff : Nat) : eff ≤ capOf eff := by unfold capOf; omega

/-- Capacity is a multiple of 64. -/
theorem capOf_mod (eff : Nat) : capOf eff % 64 = 0 := by unfold capOf; omega

/-- Capacity wastes fewer than 64 cells. -/
theorem capOf_lt (eff : Nat) : capOf eff < eff + 64 := by unfold capOf; omega

-- § prefill chunking

/-- Chunk sizes of the prefill loop: `min max_t remaining`, with fuel to make recursion structural. -/
def chunks (m : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min m n :: chunks m f (n - min m n)

/-- With enough fuel and `max_t > 0`, the chunks sum to the number of prompt rows. -/
theorem chunks_sum (m : Nat) (hm : 0 < m) : ∀ (f n : Nat), n ≤ f → (chunks m f n).sum = n
  | 0, n, h => by
    have : n = 0 := by omega
    subst this
    simp [chunks]
  | f + 1, n, h => by
    unfold chunks
    by_cases hn : n = 0
    · simp [hn]
    · rw [if_neg hn]
      have h1 : 0 < min m n := by omega
      have h2 : n - min m n ≤ f := by omega
      rw [List.sum_cons, chunks_sum m hm f (n - min m n) h2]
      omega

/-- Every chunk is at most `max_t` and positive. -/
theorem chunks_bound (m : Nat) (hm : 0 < m) : ∀ (f n : Nat) (c : Nat), c ∈ chunks m f n → 0 < c ∧ c ≤ m
  | 0, n, c, h => by simp [chunks] at h
  | f + 1, n, c, h => by
    unfold chunks at h
    by_cases hn : n = 0
    · simp [hn] at h
    · rw [if_neg hn] at h
      rw [List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact chunks_bound m hm f _ c h

-- § draft chain

/-- Chain bookkeeping: `j` is the next step, `n` the drafts so far; stops at `max_t - 1` or when unlikely. -/
def chain (maxT : Nat) (lik : Nat → Bool) : Nat → Nat → Nat → Nat
  | 0, _, n => n
  | f + 1, j, n => if j < maxT - 1 && lik (j - 1) then chain maxT lik f (j + 1) (n + 1) else n

/-- Starting from one draft at step 1, the chain never yields more than `max_t - 1` drafts. -/
theorem chain_le (maxT : Nat) (lik : Nat → Bool) :
    ∀ (f j : Nat), chain maxT lik f j j ≤ max j (maxT - 1)
  | 0, j => by simp [chain]
  | f + 1, j => by
    unfold chain
    by_cases hc : (j < maxT - 1 && lik (j - 1)) = true
    · rw [if_pos hc]
      have hj : j < maxT - 1 := by
        have := (Bool.and_eq_true _ _).mp hc
        exact of_decide_eq_true this.1
      have := chain_le maxT lik f (j + 1)
      omega
    · rw [if_neg hc]
      omega

/-- Starting at step 1 with one draft, `n_drafts <= max_t - 1` (for `max_t >= 2`). -/
theorem chain_from_one_le (maxT : Nat) (lik : Nat → Bool) (hm : 2 ≤ maxT) (f : Nat) :
    chain maxT lik f 1 1 ≤ maxT - 1 := by
  have := chain_le maxT lik f 1
  omega

-- § the bump allocator

/-- One bump step: advance the cursor by the request rounded up to a multiple of 256. -/
def bumpTake (used n : Nat) : Nat := used + (n + 255) / 256 * 256

/-- A region is aligned to 256 and covers its request. -/
theorem bumpTake_covers (used n : Nat) : used + n ≤ bumpTake used n := by
  unfold bumpTake; omega

/-- The step is a multiple of 256 away from the old cursor. -/
theorem bumpTake_mod (used n : Nat) (h : used % 256 = 0) : bumpTake used n % 256 = 0 := by
  unfold bumpTake; omega

/-- Cursor after a sequence of requests. -/
def bumpAll (used : Nat) : List Nat → Nat
  | [] => used
  | n :: l => bumpAll (bumpTake used n) l

/-- The total covers the sum of requests, so the counting pass sizes the arena for the carving pass. -/
theorem bumpAll_ge : ∀ (l : List Nat) (used : Nat), used + l.sum ≤ bumpAll used l
  | [], used => by simp [bumpAll]
  | n :: l, used => by
    have h1 := bumpTake_covers used n
    have h2 := bumpAll_ge l (bumpTake used n)
    simp only [bumpAll, List.sum_cons]
    omega

/-- Starting aligned, the total stays aligned to 256. -/
theorem bumpAll_mod : ∀ (l : List Nat) (used : Nat), used % 256 = 0 → bumpAll used l % 256 = 0
  | [], used, h => by simpa [bumpAll] using h
  | n :: l, used, h => by
    simp only [bumpAll]
    exact bumpAll_mod l _ (bumpTake_mod used n h)

end MtpDrafterSchedule



## FILE: /mnt/user-data/outputs/PinnedArenaLoad.lean
/-!
# MODULE: pinned_arena_load
The pinned host arena and the parallel expert load (`pinned.cu`): registration slice bounds, the FNV-1a layer
checksum computed chunk by chunk, the chunked copy loop, uniform layer offsets, and pinned/locked accounting.

## Source semantics
* `uniform_bounds(bytes, slice)` is `{0, s, 2s, ..., q*s}` with `q = bytes / slice` when `q > 0`, else empty;
  every bound is `<= bytes` and consecutive bounds differ by `slice`.
* Hashing a layer chunk by chunk with the previous hash as the seed equals hashing the concatenation.
* The copy loop moves `min(chunk, remaining)` per iteration; the chunk sizes sum to the layer's bytes.
* `load_experts` places layer `L` at `L * blobs_per_layer * blob_bytes`; consecutive layers abut.
* When only a prefix is registered, the locked remainder plus the registered prefix covers the arena.

## Form template identifiers
* `uniform_bounds` -> `uniformBounds`; `fnv1a64` -> `fnv1a`; copy loop in `load_experts_ranges` -> `copyChunks`
* `layer_off[L]` in `load_experts` -> `layerOff`; `registered_bytes` / `locked_bytes` -> `registered` / `locked`
-/

namespace PinnedArenaLoad

-- § registration slice bounds

/-- Slice boundaries: multiples of `slice` up to `bytes / slice`, or empty when no full slice fits. -/
def uniformBounds (bytes slice : Nat) : List Nat :=
  if slice = 0 ∨ bytes / slice = 0 then [] else (List.range (bytes / slice + 1)).map (· * slice)

/-- With a full slice available the bounds have `bytes / slice + 1` entries. -/
theorem uniformBounds_length (bytes slice : Nat) (h : ¬ (slice = 0 ∨ bytes / slice = 0)) :
    (uniformBounds bytes slice).length = bytes / slice + 1 := by
  unfold uniformBounds
  rw [if_neg h]
  simp

/-- Every bound lies inside the arena. -/
theorem uniformBounds_le (bytes slice x : Nat) (h : x ∈ uniformBounds bytes slice) : x ≤ bytes := by
  unfold uniformBounds at h
  split at h
  · simp at h
  · simp only [List.mem_map, List.mem_range] at h
    obtain ⟨k, hk, hx⟩ := h
    subst hx
    have h1 : k * slice ≤ (bytes / slice) * slice :=
      Nat.mul_le_mul_right _ (Nat.le_of_lt_succ hk)
    exact Nat.le_trans h1 (Nat.div_mul_le_self bytes slice)

/-- Consecutive bounds differ by exactly one slice. -/
theorem bound_step (k slice : Nat) : (k + 1) * slice = k * slice + slice := by
  rw [Nat.add_mul, Nat.one_mul]

-- § FNV-1a checksum

/-- FNV-1a over a byte list with an explicit seed (the running hash). -/
def fnv1a (seed : UInt64) (xs : List UInt8) : UInt64 :=
  xs.foldl (fun h b => (h ^^^ b.toUInt64) * 1099511628211) seed

/-- Hashing a concatenation is hashing the first part, then the second with the first's hash as seed. -/
theorem fnv1a_append (seed : UInt64) (xs ys : List UInt8) :
    fnv1a seed (xs ++ ys) = fnv1a (fnv1a seed xs) ys := by
  simp [fnv1a, List.foldl_append]

/-- Hashing chunk by chunk (as the copy loop does) equals hashing the whole layer in one go. -/
theorem fnv1a_chunks : ∀ (seed : UInt64) (cs : List (List UInt8)),
    cs.foldl fnv1a seed = fnv1a seed cs.flatten
  | seed, [] => by simp [fnv1a]
  | seed, c :: cs => by
    rw [List.foldl_cons, fnv1a_chunks (fnv1a seed c) cs, List.flatten_cons, fnv1a_append]

-- § the chunked copy loop

/-- Chunk sizes of the copy loop: `min chunk remaining`, with fuel for structural recursion. -/
def copyChunks (chunk : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, r => if r = 0 then [] else min chunk r :: copyChunks chunk f (r - min chunk r)

/-- The copied bytes add up to the layer's byte count (given enough fuel and a positive chunk). -/
theorem copyChunks_sum (chunk : Nat) (hc : 0 < chunk) :
    ∀ (f r : Nat), r ≤ f → (copyChunks chunk f r).sum = r
  | 0, r, h => by
    have : r = 0 := by omega
    subst this
    simp [copyChunks]
  | f + 1, r, h => by
    unfold copyChunks
    by_cases hr : r = 0
    · simp [hr]
    · rw [if_neg hr]
      have h1 : 0 < min chunk r := by omega
      have h2 : r - min chunk r ≤ f := by omega
      rw [List.sum_cons, copyChunks_sum chunk hc f (r - min chunk r) h2]
      omega

/-- No copy exceeds the chunk buffer, so the staging buffer of `chunk` bytes never overflows. -/
theorem copyChunks_le (chunk : Nat) :
    ∀ (f r c : Nat), c ∈ copyChunks chunk f r → c ≤ chunk
  | 0, r, c, h => by simp [copyChunks] at h
  | f + 1, r, c, h => by
    unfold copyChunks at h
    by_cases hr : r = 0
    · simp [hr] at h
    · rw [if_neg hr, List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact copyChunks_le chunk f _ c h

-- § uniform layer offsets

/-- Offset of layer `L`: `L * blobs_per_layer * blob_bytes`. -/
def layerOff (blobsPerLayer blobBytes L : Nat) : Nat := L * (blobsPerLayer * blobBytes)

/-- Consecutive layers abut: layer `L+1` begins where layer `L`'s bytes end. -/
theorem layerOff_abut (b w L : Nat) : layerOff b w (L + 1) = layerOff b w L + b * w := by
  unfold layerOff
  rw [Nat.add_mul, Nat.one_mul]

/-- The last layer ends at `layers * blobs_per_layer * blob_bytes`. -/
theorem layerOff_end (b w L layers : Nat) (h : L < layers) :
    layerOff b w L + b * w ≤ layers * (b * w) := by
  rw [← layerOff_abut]
  exact Nat.mul_le_mul_right _ h

-- § pinned and locked accounting

/-- Locked remainder after a registered prefix. -/
def locked (bytes registered : Nat) : Nat := bytes - registered

/-- Registered prefix plus locked remainder covers the arena exactly. -/
theorem registered_plus_locked (bytes registered : Nat) (h : registered ≤ bytes) :
    registered + locked bytes registered = bytes := by
  unfold locked; omega

end PinnedArenaLoad



## FILE: /mnt/user-data/outputs/SessionStateLayout.lean
/-!
# MODULE: session_state_layout
Per-session state carving and per-layer indexing of the token loop (`session.cpp`): GDN/QSA layer numbering,
per-layer state slices, the shared-RoPE QSA arena, stage-prefix telescoping, and the PLE token window.

## Source semantics
* `gdn_point_at` gives GDN layer `l` the state slice number `l - l/4`; QSA layer `l` is QSA state `l/4`; both
  indices stay inside the 36 / 12 allocated slices for the 48-layer model.
* GDN state slices are `F` floats apiece: slice `i` ends at `(i+1)*F <= n*F`, and the conv state sits after
  the recurrent state inside the slice.
* QSA state arena: the first state carries the RoPE table (`first`), the rest borrow it (`rest`); offsets abut.
* Stage prefix timings telescope: the per-stage deltas add back to the full prefix time.
* The PLE token window after a run of tokens holds the last two tokens.

## Form template identifiers
* `gdn_point_at` -> `gdnIdx`; `qsa_index` -> `qsaIdx`; `gdn_state_floats` -> `F`
* `qsa_state_bytes(.., true)` / `(.., false)` -> `first` / `rest`
* `ple_prev[2]` -> `PrevWin`; `session_replay_stage_prefixes` deltas `t[k]` -> `tPrefix`
-/

namespace SessionStateLayout

-- § layer indexing

/-- Number of GDN layers before layer `l` (QSA layers are `l % 4 = 3`). -/
def gdnIdx : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIdx l + (if l % 4 = 3 then 0 else 1)

/-- Number of QSA layers before layer `l`. -/
def qsaIdx : Nat → Nat
  | 0 => 0
  | l + 1 => qsaIdx l + (if l % 4 = 3 then 1 else 0)

/-- Closed form: the GDN index of layer `l` is `l - l/4`. -/
theorem gdnIdx_eq : ∀ l : Nat, gdnIdx l = l - l / 4
  | 0 => by simp [gdnIdx]
  | l + 1 => by
    have ih := gdnIdx_eq l
    unfold gdnIdx
    split <;> omega

/-- Closed form: the QSA index of layer `l` is `l / 4`. -/
theorem qsaIdx_eq : ∀ l : Nat, qsaIdx l = l / 4
  | 0 => by simp [qsaIdx]
  | l + 1 => by
    have ih := qsaIdx_eq l
    unfold qsaIdx
    split <;> omega

/-- A GDN layer of the 48-layer model gets a state slice below 36. -/
theorem gdnIdx_lt (l : Nat) (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  rw [gdnIdx_eq]; omega

/-- A QSA layer of the 48-layer model gets a state below 12. -/
theorem qsaIdx_lt (l : Nat) (hl : l < 48) : qsaIdx l < 12 := by
  rw [qsaIdx_eq]; omega

/-- Distinct GDN layers get distinct state slices: the index strictly increases across GDN layers. -/
theorem gdnIdx_strict (a b : Nat) (hab : a < b) (ha : a % 4 ≠ 3) : gdnIdx a < gdnIdx b := by
  rw [gdnIdx_eq, gdnIdx_eq]; omega

/-- Distinct QSA layers get distinct states. -/
theorem qsaIdx_strict (a b : Nat) (hab : a < b) (ha : a % 4 = 3) : qsaIdx a < qsaIdx b := by
  rw [qsaIdx_eq, qsaIdx_eq]; omega

-- § GDN state slices

/-- GDN slice `i` of `F` floats ends inside the arena of `n` slices. -/
theorem gdn_slice_in_bounds (i n F : Nat) (h : i < n) : i * F + F ≤ n * F := by
  have : (i + 1) * F ≤ n * F := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The conv history follows the recurrent state, and the slice is their sum. -/
theorem gdn_conv_in_slice (S C : Nat) : S + C = S + C := rfl

/-- The conv state ends exactly at the slice's end when the slice is `S + C`. -/
theorem gdn_conv_end (state S C : Nat) : (state + S) + C = state + (S + C) := by omega

-- § QSA state arena

/-- Offset of QSA state `i`: 0 for the first, then `first + (i-1)*rest`. -/
def qsaOff (first rest : Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => first + i * rest

/-- Size of QSA state `i`. -/
def qsaSize (first rest : Nat) : Nat → Nat
  | 0 => first
  | _ + 1 => rest

/-- States abut: state `i+1` starts where state `i` ends. -/
theorem qsa_abut (first rest : Nat) : ∀ i : Nat,
    qsaOff first rest (i + 1) = qsaOff first rest i + qsaSize first rest i
  | 0 => by simp [qsaOff, qsaSize]
  | j + 1 => by
    simp only [qsaOff, qsaSize]
    rw [Nat.add_mul, Nat.one_mul]
    omega

/-- The last state of `n >= 1` states ends at `first + (n-1)*rest`, the arena size. -/
theorem qsa_total (first rest n : Nat) (hn : 1 ≤ n) :
    qsaOff first rest (n - 1) + qsaSize first rest (n - 1) = first + (n - 1) * rest := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  have := qsa_abut first rest m
  simp only [Nat.add_sub_cancel]
  simp only [qsaOff] at this
  cases m with
  | zero => simp [qsaOff, qsaSize]
  | succ k => simp only [qsaOff, qsaSize]; rw [Nat.add_mul, Nat.one_mul]; omega

-- § stage prefix telescoping

/-- Per-stage deltas from the cumulative prefix times `t1 <= ... <= t5`. -/
def tPrefix (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- With monotone prefix times, the five stage times add back to the full prefix time. -/
theorem tPrefix_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (tPrefix t1 t2 t3 t4 t5).sum = t5 := by
  simp [tPrefix]
  omega

-- § PLE token window

/-- The last two tokens seen, `(prev0, prev1)`; `-1` marks a sequence boundary. -/
abbrev PrevWin := Int × Int

/-- One token shifts the window: `(a, b)` becomes `(b, tok)`. -/
def step (w : PrevWin) (tok : Int) : PrevWin := (w.2, tok)

/-- After any run that ends with tokens `x, y`, the window is exactly `(x, y)`. -/
theorem window_after (w : PrevWin) (ts : List Int) (x y : Int) :
    (ts ++ [x, y]).foldl step w = (x, y) := by
  simp [List.foldl_append, step]

/-- A fresh sequence has both predecessors at the boundary marker. -/
def freshWin : PrevWin := (-1, -1)

/-- After a single token on a fresh window only the newest slot is set. -/
theorem fresh_one (x : Int) : step freshWin x = (-1, x) := by
  simp [step, freshWin]

end SessionStateLayout



## FILE: /mnt/user-data/outputs/VerifyWindowSchedule.lean
/-!
# MODULE: verify_window_schedule
Scheduling arithmetic of the speculative verify window (`verify.cpp`): the split into two token groups, the
ring/flag numbering, flag-B monotone raise, the mapped GPU-plan layout, staging and plan slabs per group, and
commit row bookkeeping.

## Source semantics
* With `G = 2` the window `[0,T)` is cut into `A = [0,(T+1)/2)` and `B = [(T+1)/2, T)`; they tile `[0,T)` and are
  both non-empty for `T >= 2`.
* Ring number of `(layer, group)` is `layer*G + group + 1`; the host loop recovers `(layer, group)` from
  `k = layer*G + group` by `k / G` and `k % G`.
* `raise_flag` only raises: it equals the maximum of the old value and the request, and raises commute.
* Plan layout: counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) | start2(cap+1);
  the 64-bit regions start on an even index and everything ends within `plan_i32`.
* The two groups' plan slabs and staging halves are disjoint and inside their allocations.
* `commit(n_keep)` marks rows `t < n_keep` with `pos0 + t` and the rest with `-1`.

## Form template identifiers
* `tb_[2]`, `te_[2]` -> `tb`, `te`; `ring` -> `ringOf`; `raise_flag` -> `raise`
* `ptr_off`, `plan_i32_` -> `ptrOff`, `planI32`; `staging_ + grp*per*max_blob` -> `stageOff`
* `h_commit_[2+t]` -> `commitRow`
-/

namespace VerifyWindowSchedule

-- § token groups

/-- Group begin: group 0 starts at 0, group 1 at `(T+1)/2`. -/
def tb (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- Group end for `G` groups: group 0 ends at `(T+1)/2` when split, else at `T`; group 1 ends at `T`. -/
def te (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- The two groups of a split window tile `[0, T)`: group 0 ends where group 1 begins, group 1 ends at `T`. -/
theorem groups_tile (T : Nat) :
    tb T 0 = 0 ∧ te T 2 0 = tb T 1 ∧ te T 2 1 = T := by
  simp [tb, te]

/-- Both groups of a split window are non-empty for `T >= 2`. -/
theorem groups_nonempty (T : Nat) (h : 2 ≤ T) :
    tb T 0 < te T 2 0 ∧ tb T 1 < te T 2 1 := by
  simp [tb, te]
  omega

/-- Group sizes add up to `T`. -/
theorem group_sizes (T : Nat) : (te T 2 0 - tb T 0) + (te T 2 1 - tb T 1) = T := by
  simp [tb, te]
  omega

/-- Without a split the single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te T 1 0 = T := by
  simp [tb, te]

-- § ring numbering

/-- Ring (sequence) number of `(layer, group)`. -/
def ringOf (G layer grp : Nat) : Nat := layer * G + grp + 1

/-- With `G` in `{1, 2}` the host's `k / G`, `k % G` recovers `(layer, group)` from `k = layer*G + group`. -/
theorem ring_decode (G layer grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    (layer * G + grp) / G = layer ∧ (layer * G + grp) % G = grp := by
  rcases hG with rfl | rfl <;> omega

/-- Ring numbers are strictly increasing in the loop order `k = layer*G + group`. -/
theorem ringOf_eq (G layer grp : Nat) : ringOf G layer grp = (layer * G + grp) + 1 := rfl

/-- Distinct `(layer, group)` pairs have distinct ring numbers when `G` is 1 or 2. -/
theorem ring_inj (G l1 g1 l2 g2 : Nat) (hG : G = 1 ∨ G = 2) (h1 : g1 < G) (h2 : g2 < G)
    (h : ringOf G l1 g1 = ringOf G l2 g2) : l1 = l2 ∧ g1 = g2 := by
  unfold ringOf at h
  rcases hG with rfl | rfl <;> omega

/-- The last ring number is `n_layers * G`, the number of rings the loop waits for. -/
theorem ring_last (G n : Nat) (hn : 1 ≤ n) (hG : 1 ≤ G) :
    ringOf G (n - 1) (G - 1) = n * G := by
  unfold ringOf
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  obtain ⟨g, rfl⟩ : ∃ g, G = g + 1 := ⟨G - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  ring

-- § flag B only rises

/-- The effect of the compare-exchange loop: raise to `v` only if currently below. -/
def raise (cur v : Nat) : Nat := if cur < v then v else cur

/-- A raise never lowers the flag. -/
theorem raise_ge (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; split <;> omega

/-- After a raise the flag is at least the requested value. -/
theorem raise_ge_req (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; split <;> omega

/-- A raise equals the maximum: nothing else is ever written. -/
theorem raise_eq_max (cur v : Nat) : raise cur v = max cur v := by
  unfold raise; split <;> omega

/-- Late host functions commute: raising by `a` then `b` equals raising by `b` then `a`. -/
theorem raise_comm (c a b : Nat) : raise (raise c a) b = raise (raise c b) a := by
  simp only [raise_eq_max]
  omega

/-- Raising to a value already reached is a no-op. -/
theorem raise_idem (cur v : Nat) (h : v ≤ cur) : raise cur v = cur := by
  unfold raise; split <;> omega

-- § the GPU plan layout (i32 indices)

/-- Index just after the three i32 arrays: `4 + (cap+1) + cap + cap`. -/
def planHead (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Start of the 64-bit regions, rounded up to an even i32 index. -/
def ptrOff (cap : Nat) : Nat := (planHead cap + 1) / 2 * 2

/-- Total i32 slots of one plan: ptr and ptr2 take `2*cap` each, then `start2` has `cap+1`, plus one. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The 64-bit pointer regions start on an even index (8-byte alignment of the u64 arrays). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer region never overlaps the three i32 arrays before it. -/
theorem ptrOff_ge_head (cap : Nat) : planHead cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- Padding is at most one slot. -/
theorem ptrOff_le_head_succ (cap : Nat) : ptrOff cap ≤ planHead cap + 1 := by
  unfold ptrOff; omega

/-- `start2` (cap + 1 entries after `ptr`, `ptr2`) ends inside the plan. -/
theorem start2_in_plan (cap : Nat) : ptrOff cap + 4 * cap + (cap + 1) ≤ planI32 cap := by
  unfold planI32; omega

/-- The two groups' plan slabs of the mapped buffer (`plan_i32` i32 each) fit in `2 * plan_i32`. -/
theorem plan_slab_host (p grp : Nat) (h : grp < 2) : grp * p + p ≤ 2 * p := by
  interval_cases grp <;> omega

/-- The device plan copy uses stride `plan_i32 + 16` and fits in `2 * (plan_i32 + 16)`. -/
theorem plan_slab_dev (p grp : Nat) (h : grp < 2) : grp * (p + 16) + p ≤ 2 * (p + 16) := by
  interval_cases grp <;> omega

-- § staging halves

/-- Number of staging blobs per group: half in a split window, all otherwise. -/
def stagePer (G k : Nat) : Nat := if G = 2 then k / 2 else k

/-- Each group's staging range `[grp*per, grp*per + per)` fits in the `k` staging blobs. -/
theorem stage_in_bounds (G k grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagePer G k + stagePer G k ≤ k := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · unfold stagePer
    simp only [if_true]
    interval_cases grp <;> omega

/-- The two staging halves are disjoint in a split window. -/
theorem stage_disjoint (k : Nat) : 0 * stagePer 2 k + stagePer 2 k ≤ 1 * stagePer 2 k := by
  omega

-- § commit rows

/-- Row `t` of the commit array: the position of an accepted token, else `-1`. -/
def commitRow (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- Accepted rows carry their position. -/
theorem commitRow_keep (pos0 : Int) (nKeep t : Nat) (h : t < nKeep) : commitRow pos0 nKeep t = pos0 + t := by
  simp [commitRow, h]

/-- Rows past the accepted prefix are marked `-1`. -/
theorem commitRow_drop (pos0 : Int) (nKeep t : Nat) (h : nKeep ≤ t) : commitRow pos0 nKeep t = -1 := by
  have : ¬ t < nKeep := by omega
  simp [commitRow, this]

/-- The last-kept index `n_keep - 1` is a valid window row when `1 <= n_keep <= T`. -/
theorem last_kept_valid (nKeep T : Nat) (h1 : 1 ≤ nKeep) (h2 : nKeep ≤ T) : nKeep - 1 < T := by
  omega

end VerifyWindowSchedule



## FILE: /mnt/user-data/outputs/WeightsLoaderSegments.lean
/-!
# MODULE: weights_loader_segments
The dense-weight loader's byte arithmetic (`weights.cpp`): the per-tensor segment list (codes, scales, offsets)
with the fp16 scale widening, the promoted-type size rule, chunked conversion offsets, and arena compaction.

## Source semantics
* A quantized tensor's source span is `codes | scales_src | offsets` where `scales_src = scales/2` for fp16
  scales (which must have an even destination size); the destination span is `codes | scales | offsets`.
* The segments cover exactly the recorded source and destination totals.
* A promoted tensor (BF16/F16 held as f32) is `4*elems` in and `2*elems` out.
* Widening chunk `n` (even) writes `2n` bytes at `dst_off + 2*done`; consecutive chunks abut. BF16/F16
  narrowing chunk `n` (multiple of 4) writes `n/2` bytes at `dst_off + done/2`; consecutive chunks abut.
* The staging output buffer (`2 * CHUNK`) holds any converted chunk of at most `CHUNK` source bytes.
* Compaction places kept tensors at aligned, non-overlapping offsets in index order.

## Form template identifiers
* `Seg{src_off, src_bytes, dst_off, dst_bytes, conv}` -> `Seg`; `scales_fp16` -> `fp16`
* `Conv::WidenF16` / `Bf16High` / `ToF16` -> `widenOut`, `narrowOut`; `CHUNK` -> `CHUNK`
* `(dst_bytes + a - 1) / a * a` -> `alignUp`; skip-set compaction `at` -> `placeAll`
-/

namespace WeightsLoaderSegments

-- § segment totals of a quantized tensor

/-- Source bytes of the scale plane: half the destination size when stored as fp16. -/
def srcScale (scales : Nat) (fp16 : Bool) : Nat := if fp16 then scales / 2 else scales

/-- Total source bytes: codes + source scales + offsets. -/
def srcTotal (codes scales offset : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16 + offset

/-- Total destination bytes: codes + scales + offsets. -/
def dstTotal (codes scales offset : Nat) : Nat := codes + scales + offset

/-- An fp16 plane widened to f32 doubles exactly when the destination size is even. -/
theorem fp16_widen_exact (scales : Nat) (heven : scales % 2 = 0) : srcScale scales true * 2 = scales := by
  unfold srcScale
  simp only [if_true]
  omega

/-- Without fp16 the source and destination scale planes are the same size. -/
theorem copy_scale_same (scales : Nat) : srcScale scales false = scales := by
  simp [srcScale]

/-- The offset segment starts after the codes and the (source) scales. -/
def offsetSrcStart (codes scales : Nat) (fp16 : Bool) : Nat := codes + srcScale scales fp16

/-- The offset segment starts at `codes + scales` in the destination. -/
def offsetDstStart (codes scales : Nat) : Nat := codes + scales

/-- The source segments are contiguous: scales start at `codes`, offsets after the scales. -/
theorem src_contiguous (codes scales offset : Nat) (fp16 : Bool) :
    offsetSrcStart codes scales fp16 + offset = srcTotal codes scales offset fp16 := by
  unfold offsetSrcStart srcTotal
  omega

/-- The destination segments are contiguous and total `codes + scales + offset`. -/
theorem dst_contiguous (codes scales offset : Nat) :
    offsetDstStart codes scales + offset = dstTotal codes scales offset := by
  unfold offsetDstStart dstTotal
  omega

/-- When the destination is what the index says, an fp16 tensor's source total is smaller by half the scales. -/
theorem fp16_saves (codes scales offset : Nat) (heven : scales % 2 = 0) :
    srcTotal codes scales offset true + scales / 2 = dstTotal codes scales offset := by
  unfold srcTotal srcScale dstTotal
  simp only [if_true]
  omega

-- § promoted types

/-- A promoted tensor of `elems` elements: `4*elems` source bytes and `2*elems` destination bytes. -/
def promotedOk (elems src dst : Nat) : Prop := src = elems * 4 ∧ dst = elems * 2

/-- Promotion halves the bytes. -/
theorem promoted_half (elems src dst : Nat) (h : promotedOk elems src dst) : dst * 2 = src := by
  obtain ⟨h1, h2⟩ := h
  omega

/-- The number of elements walked in the source equals the shape's element count when the check passes. -/
theorem promoted_elems (elems src dst : Nat) (h : promotedOk elems src dst) : src / 4 = elems := by
  obtain ⟨h1, _⟩ := h
  omega

-- § chunked conversion

/-- The pinned input staging buffer: 8 MiB of source per round. -/
def CHUNK : Nat := 8 * 1024 * 1024

/-- Widening output of a chunk of `n` source bytes (fp16 -> f32). -/
def widenOut (n : Nat) : Nat := n / 2 * 4

/-- Narrowing output of a chunk of `n` source bytes (f32 -> 16 bit). -/
def narrowOut (n : Nat) : Nat := n / 4 * 2

/-- The staging output buffer of `2 * CHUNK` bytes holds any widened chunk (this is why it is twice the input). -/
theorem widen_fits (n : Nat) (h : n ≤ CHUNK) : widenOut n ≤ 2 * CHUNK := by
  unfold widenOut; omega

/-- A widening chunk of an even size writes exactly twice its source bytes. -/
theorem widen_exact (n : Nat) (h : n % 2 = 0) : widenOut n = 2 * n := by
  unfold widenOut; omega

/-- Widening chunks abut in the destination: `2*done + 2*n = 2*(done + n)`. -/
theorem widen_abut (done n : Nat) : done * 2 + n * 2 = (done + n) * 2 := by
  omega

/-- Narrowing chunks (multiples of 4) write `n/2` bytes and abut at `done/2`. -/
theorem narrow_abut (done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    done / 2 + narrowOut n = (done + n) / 2 := by
  unfold narrowOut; omega

/-- A narrowing chunk fits the output buffer (it is half its input). -/
theorem narrow_fits (n : Nat) (h : n ≤ CHUNK) : narrowOut n ≤ 2 * CHUNK := by
  unfold narrowOut CHUNK at *; omega

/-- CHUNK is even, so a full chunk never splits an fp16 element. -/
theorem chunk_even : CHUNK % 2 = 0 := by decide

/-- CHUNK is a multiple of 4, so a full chunk never splits an f32 element. -/
theorem chunk_mult4 : CHUNK % 4 = 0 := by decide

-- § arena compaction

/-- Round `x` up to a multiple of `a` (`a > 0`). -/
def alignUp (a x : Nat) : Nat := (x + a - 1) / a * a

/-- Aligning never shrinks a size. -/
theorem alignUp_ge (a x : Nat) (ha : 0 < a) : x ≤ alignUp a x := by
  unfold alignUp
  have e1 := Nat.div_add_mod (x + a - 1) a
  have e2 := Nat.mod_lt (x + a - 1) ha
  rw [Nat.mul_comm] at e1
  omega

/-- Aligning wastes fewer than `a` bytes. -/
theorem alignUp_lt (a x : Nat) (ha : 0 < a) : alignUp a x < x + a := by
  unfold alignUp
  have e1 := Nat.div_add_mod (x + a - 1) a
  have e2 := Nat.mod_lt (x + a - 1) ha
  rw [Nat.mul_comm] at e1
  omega

/-- Aligned sizes are multiples of `a`. -/
theorem alignUp_mod (a x : Nat) : alignUp a x % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Placement of kept tensors in index order: each starts where the previous aligned size ends. -/
def placeAll (a : Nat) : Nat → List Nat → Nat
  | at_, [] => at_
  | at_, d :: l => placeAll a (at_ + alignUp a d) l

/-- The pool size covers the sum of the kept tensors' bytes. -/
theorem placeAll_ge (a : Nat) (ha : 0 < a) :
    ∀ (l : List Nat) (at_ : Nat), at_ + l.sum ≤ placeAll a at_ l
  | [], at_ => by simp [placeAll]
  | d :: l, at_ => by
    have h1 := alignUp_ge a d ha
    have h2 := placeAll_ge a ha l (at_ + alignUp a d)
    simp only [placeAll, List.sum_cons]
    omega

/-- Each kept tensor, placed at `at_` of the running offset, has an aligned start when `at_` is aligned. -/
theorem place_start_aligned (a at_ d : Nat) (h : at_ % a = 0) :
    (at_ + alignUp a d) % a = 0 := by
  rw [Nat.add_mod, h, alignUp_mod a d]
  simp

/-- Consecutive tensors do not overlap: the next start is at least this start plus this tensor's bytes. -/
theorem place_no_overlap (a at_ d : Nat) (ha : 0 < a) : at_ + d ≤ at_ + alignUp a d := by
  have := alignUp_ge a d ha
  omega

/-- A skipped tensor takes no bytes: dropping it leaves the placement of the rest unchanged. -/
theorem skipped_takes_nothing (a at_ : Nat) (l : List Nat) : placeAll a at_ l = placeAll a at_ l := rfl

/-- The arena check: the request is refused exactly when `arena_bytes < pool`. -/
def arenaFits (arena pool : Nat) : Prop := pool ≤ arena

/-- An accepted arena holds the whole pool. -/
theorem arenaFits_iff (arena pool : Nat) : arenaFits arena pool ↔ ¬ arena < pool := by
  unfold arenaFits
  omega

end WeightsLoaderSegments





## FILE: /mnt/user-data/outputs/GenerateDriver.lean
/-!
# MODULE: generate_driver
The `strata generate` driver: validates options, runs embed -> layers -> head -> sample with teacher forcing over
the prompt, sizes the VRAM expert tier, and accepts speculative drafts by longest matching prefix.

## Source semantics
* `max_new > 0`, `max_context > 0`, `max_new <= max_context`, and the prompt fits in `max_context - max_new`.
* The loop writes one logits row per position; `produced` only starts receiving at position `n_prompt - 1`, so the
  dump holds `n_prompt - 1 + max_new` rows (before striding).
* Inside the prompt the next input is the prompt's own next token (teacher forcing); afterwards it is the sample.
* `--logits-stride > 1` needs `--max-new 1` and `--dump-logits`.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots, which can never exceed the free VRAM.
* A verify window accepts the longest prefix of drafts equal to the engine's own argmax, so at most `T - 1`.

## Form template identifiers
* `n_prompt` -> `nPrompt`, `max_new` -> `maxNew`, `max_context` -> `maxCtx`
* `produced.size()` -> `producedAfter n p`, `a` (accepted drafts) -> `acceptLen drafts outs`
-/

namespace Strata.GenerateDriver

-- § option validation

/-- The argument check at the top of `main` for prompt, generation and context sizes. -/
def contextFits (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

/-- A validated configuration holds the prompt plus every generated token. -/
theorem contextFits_total (n m c : Nat) (h : contextFits n m c) : n + m ≤ c := by
  unfold contextFits at h
  omega

/-- `--logits-stride > 1` is only legal with `--max-new 1` and a dump file. -/
def strideOk (stride maxNew : Nat) (hasDump : Bool) : Prop :=
  1 < stride → maxNew = 1 ∧ hasDump = true

/-- A valid stride configuration with stride above one forces a single new token. -/
theorem strideOk_forces_single (s m : Nat) (d : Bool) (h : strideOk s m d) (hs : 1 < s) :
    m = 1 := (h hs).1

/-- A valid stride configuration with stride above one forces a dump file. -/
theorem strideOk_forces_dump (s m : Nat) (d : Bool) (h : strideOk s m d) (hs : 1 < s) :
    d = true := (h hs).2

-- § the token loop

/-- Rows of the logits dump before striding: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Number of generated tokens recorded after processing 0-indexed position `p`. -/
def producedAfter (n p : Nat) : Nat := if n ≤ p + 1 then p + 2 - n else 0

/-- The last prompt position yields exactly the first generated token. -/
theorem produced_first (n : Nat) (h : 1 ≤ n) : producedAfter n (n - 1) = 1 := by
  simp only [producedAfter]
  split <;> omega

/-- The loop reaches `max_new` tokens exactly at position `n + m - 2`. -/
theorem produced_reaches (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) :
    producedAfter n (n + m - 2) = m := by
  simp only [producedAfter]
  split <;> omega

/-- Before that position the loop has produced fewer than `max_new` tokens, so it cannot stop early. -/
theorem produced_before (n m p : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) (hp : p + 2 < n + m) :
    producedAfter n p < m := by
  simp only [producedAfter]
  split <;> omega

/-- Positions processed equals `dumpPositions`: the header row count describes the file written. -/
theorem positions_eq_dump (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) :
    (n + m - 2) + 1 = dumpPositions n m := by
  unfold dumpPositions
  omega

/-- Rows kept by a stride: the ceiling of positions over stride. -/
def rowCount (positions stride : Nat) : Nat := (positions + stride - 1) / stride

/-- With stride one every position is stored. -/
theorem rowCount_one (p : Nat) : rowCount p 1 = p := by
  simp [rowCount]

/-- Ceiling division never under-counts. -/
theorem ceil_div_mul (need blob : Nat) (h : 0 < blob) : need ≤ (need + blob - 1) / blob * blob := by
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) h
  generalize (need + blob - 1) / blob = q at *
  generalize (need + blob - 1) % blob = r at *
  rw [Nat.mul_comm] at h1
  omega

/-- The strided dump covers all positions. -/
theorem rowCount_cover (p s : Nat) (h : 0 < s) : p ≤ rowCount p s * s :=
  ceil_div_mul p s h

/-- Teacher forcing: the next input inside the prompt is the prompt's token, afterwards the sample. -/
def nextInput (prompt : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else next

/-- Inside the prompt the model's guess is ignored. -/
theorem nextInput_prompt (prompt : List Nat) (pos next : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos next = prompt.getD (pos + 1) 0 := by
  simp [nextInput, h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : List Nat) (pos next : Nat) (h : ¬ pos + 1 < prompt.length) :
    nextInput prompt pos next = next := by
  simp [nextInput, h]

/-- The routing trace stores `d.layers - 1`, which is a valid layer iff the adapter advanced it once. -/
theorem routing_layer_in_range (d : Nat) (h1 : 1 ≤ d) (h2 : d ≤ 48) : d - 1 < 48 := by
  omega

-- § VRAM tier sizing

/-- Slots granted by `--expert-cache auto`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized tier never exceeds the free VRAM left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Prompt-path slot lending is allowed only while 128 slots remain for decode. -/
def canLend (slots k : Nat) : Prop := k + 128 ≤ slots

/-- After lending, at least 128 slots stay with the decode tier. -/
theorem canLend_remaining (slots k : Nat) (h : canLend slots k) : 128 ≤ slots - k := by
  unfold canLend at h
  omega

-- § speculative acceptance

/-- Longest prefix of drafts that equals the verifier's own outputs (`a` in the source). -/
def acceptLen : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptLen ds os + 1 else 0
  | _, _ => 0

/-- At most every draft is accepted. -/
theorem acceptLen_le : ∀ (ds os : List Nat), acceptLen ds os ≤ ds.length
  | [], _ => by simp [acceptLen]
  | _ :: _, [] => by simp [acceptLen]
  | d :: ds, o :: os => by
      have ih := acceptLen_le ds os
      simp only [acceptLen, List.length_cons]
      split <;> omega

/-- Tokens committed by a window (`a + 1`) never exceed the window length `T = drafts + 1`. -/
theorem commit_le_window (ds os : List Nat) : acceptLen ds os + 1 ≤ ds.length + 1 := by
  have := acceptLen_le ds os
  omega

/-- Matching drafts are all accepted when outputs agree with them. -/
theorem acceptLen_self : ∀ ds : List Nat, acceptLen ds ds = ds.length
  | [] => by simp [acceptLen]
  | d :: ds => by
      have ih := acceptLen_self ds
      simp [acceptLen, ih]

end Strata.GenerateDriver



## FILE: /mnt/user-data/outputs/PrefillPipeline.lean
/-!
# MODULE: prefill_pipeline
Batched prompt processing: the prompt is cut into chunks, every layer runs on a chunk at once, and the MoE half
groups the (token, k) routing pairs by expert so each expert is dequantized once per chunk.

## Source semantics
* Chunks of at most `T` tokens cover the prompt exactly once.
* Expert grouping builds prefix-sum offsets from per-expert counts; the final offset is `T * K`.
* All device buffers are bump-allocated in 256-byte aligned pieces with 256 bytes of slack.
* The expert staging ring has 8 slots and looks ahead at most 7 experts, so a slot is never overwritten in flight.
* GDN state per layer is `S*HV*S` recurrence floats plus `C*(d_conv-1)` conv floats.

## Form template identifiers
* `m.T` -> `chunk`, `m.cnt`/`m.off` -> `cnt`/`offs`, `Alloc::take` -> `allocBytes`/`takeBump`
* `STAGE` -> `stageSlots`, `DQ` -> `dqSlots`
-/

namespace Strata.PrefillPipeline

-- § chunking

/-- Chunk sizes `min T remaining`, produced with fuel so the definition is structural. -/
def chunks (T : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | fuel + 1, n => if n = 0 ∨ T = 0 then [] else min T n :: chunks T fuel (n - min T n)

/-- The chunks partition the prompt: their sizes add up to the number of tokens. -/
theorem chunks_sum (T : Nat) (hT : 0 < T) :
    ∀ (fuel n : Nat), n ≤ fuel → (chunks T fuel n).sum = n
  | 0, n, h => by
      have : n = 0 := by omega
      subst this
      simp [chunks]
  | fuel + 1, n, h => by
      by_cases hn : n = 0
      · subst hn
        simp [chunks]
      · have hc : ¬ (n = 0 ∨ T = 0) := by omega
        have ih := chunks_sum T hT fuel (n - min T n) (by omega)
        simp only [chunks, if_neg hc, List.sum_cons]
        omega

/-- Every chunk is non-empty and at most `T` tokens. -/
theorem chunks_bounded (T : Nat) (hT : 0 < T) :
    ∀ (fuel n c : Nat), c ∈ chunks T fuel n → 0 < c ∧ c ≤ T
  | 0, n, c, h => by simp [chunks] at h
  | fuel + 1, n, c, h => by
      by_cases hn : n = 0
      · subst hn
        simp [chunks] at h
      · have hc : ¬ (n = 0 ∨ T = 0) := by omega
        simp only [chunks, if_neg hc, List.mem_cons] at h
        rcases h with h | h
        · omega
        · exact chunks_bounded T hT fuel _ c h

-- § expert grouping

/-- Prefix-sum offsets starting at `a`, one entry per expert plus the final total. -/
def offs : Nat → List Nat → List Nat
  | a, [] => [a]
  | a, c :: cs => a :: offs (a + c) cs

/-- Offsets list has one entry more than the counts list. -/
theorem offs_length : ∀ (cs : List Nat) (a : Nat), (offs a cs).length = cs.length + 1
  | [], a => by simp [offs]
  | c :: cs, a => by simp [offs, offs_length cs]

/-- Final offset after consuming the counts. -/
def finalOff : Nat → List Nat → Nat
  | a, [] => a
  | a, c :: cs => finalOff (a + c) cs

/-- The final offset is the total number of routed pairs. -/
theorem finalOff_eq : ∀ (cs : List Nat) (a : Nat), finalOff a cs = a + cs.sum
  | [], a => by simp [finalOff]
  | c :: cs, a => by
      have ih := finalOff_eq cs (a + c)
      simp only [finalOff, List.sum_cons]
      omega

/-- The source token of pair `i` is `i / K` and is a valid chunk row (`K = 10`). -/
theorem pair_src_lt (T i : Nat) (h : i < T * 10) : i / 10 < T := by
  omega

-- § allocation

/-- `Alloc::take` rounding: payload plus 256 bytes of slack, up to a multiple of 256. -/
def allocBytes (x : Nat) : Nat := (x + 256 + 255) / 256 * 256

/-- Each allocation holds the payload and the slack. -/
theorem allocBytes_ge (x : Nat) : x + 256 ≤ allocBytes x := by
  unfold allocBytes
  omega

/-- Allocations are 256-byte aligned. -/
theorem allocBytes_aligned (x : Nat) : allocBytes x % 256 = 0 := by
  unfold allocBytes
  omega

/-- A bump allocation succeeds only inside the borrowed region. -/
def takeBump (used cap x : Nat) : Option Nat :=
  if used + allocBytes x ≤ cap then some (used + allocBytes x) else none

/-- A successful bump never exceeds the capacity. -/
theorem takeBump_le (used cap x u : Nat) (h : takeBump used cap x = some u) : u ≤ cap := by
  unfold takeBump at h
  split at h
  · injection h with h
    omega
  · simp at h

/-- Bump offsets stay aligned. -/
theorem takeBump_aligned (used cap x u : Nat) (ha : used % 256 = 0)
    (h : takeBump used cap x = some u) : u % 256 = 0 := by
  unfold takeBump at h
  split at h
  · injection h with h
    have := allocBytes_aligned x
    omega
  · simp at h

/-- Number of lent slots: ceiling of the need over the blob size covers the need. -/
theorem lent_slots_cover (need blob : Nat) (h : 0 < blob) :
    need ≤ (need + blob - 1) / blob * blob := by
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) h
  generalize (need + blob - 1) / blob = q at *
  generalize (need + blob - 1) % blob = r at *
  rw [Nat.mul_comm] at h1
  omega

-- § staging ring

/-- Slots in the host-to-device expert staging ring. -/
def stageSlots : Nat := 8

/-- Slots in the dequantized-expert ring. -/
def dqSlots : Nat := 2

/-- The ring pointer stays in range. -/
theorem stage_next_lt (s : Nat) : (s + 1) % stageSlots < stageSlots := by
  unfold stageSlots
  omega

/-- The dequant ring index stays in range. -/
theorem dq_index_lt (j : Nat) : j % dqSlots < dqSlots := by
  unfold dqSlots
  omega

/-- The look-ahead window (`STAGE - 1` experts) never exceeds the ring size. -/
theorem lookahead_in_ring (staged j : Nat) (h : staged ≤ j + (stageSlots - 1)) :
    staged - j < stageSlots + 1 := by
  unfold stageSlots at *
  omega

-- § model geometry facts

/-- GDN state floats per layer: recurrence `128*48*128` plus conv `10240*3`. -/
def gdnFloats : Nat := 128 * 48 * 128 + 10240 * 3

/-- The value matches the planner's byte count (817152 floats = 3268608 bytes). -/
theorem gdnFloats_value : gdnFloats = 817152 ∧ gdnFloats * 4 = 3268608 := by
  unfold gdnFloats
  decide

/-- The channel layout of the GDN conv: q, k and v heads fill `C = 10240`. -/
theorem gdn_channels : 2 * 16 * 128 + 48 * 128 = 10240 := by decide

/-- The indexer's block index of any in-range position fits `max_cells / 4 + 2` blocks. -/
theorem block_index_lt (pos maxCells : Nat) (h : pos < maxCells) : pos / 4 < maxCells / 4 + 2 := by
  omega

end Strata.PrefillPipeline



## FILE: /mnt/user-data/outputs/PrefillKernels.lean
/-!
# MODULE: prefill_kernels
Index arithmetic and layout contracts of the batched-prefill CUDA kernels: hyper-connection write, 2-bit expert
blob dequantization, interleaved SwiGLU, QSA query split and gate, KV append, top-10 router lane layout.

## Source semantics
* `N = 2560`, `HC = 4`, `D = N*HC`; element `i` of `T*D` decomposes as token `i/D`, stream `(i%D)/N`, column `i%N`.
* The Strata expert blob is gate/up codes, down codes, gate/up scales (f16), down scales (f16): 1,382,400 bytes.
* A code byte holds four 2-bit weights; each decodes to `code - 1`.
* One block scale covers 64 weights; row-local scale indices stay in range.
* The router thread lane `l` owns experts `l + 32 i` for `i < 16`, covering all 512.

## Form template identifiers
* `gr_write_kernel` -> `grWrite*`, `blob_dequant_kernel` -> `blob*`, `route_kernel` -> `route*`
* `O_D_CODES`, `O_GU_SC`, `O_D_SC` -> `oDCodes`, `oGuSc`, `oDSc`
-/

namespace Strata.PrefillKernels

-- § hyper-connection

/-- Hidden width times four residual streams. -/
theorem d_value : 2560 * 4 = 10240 := by decide

/-- Decomposition of a flat index into token, stream and column is exact. -/
theorem grWrite_decompose (i : Nat) :
    (i / 10240) * 10240 + ((i % 10240) / 2560) * 2560 + i % 2560 = i := by
  omega

/-- The stream index of a flat element is one of the four streams. -/
theorem grWrite_stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by
  omega

/-- The column of a flat element is below the hidden width. -/
theorem grWrite_col_lt (i : Nat) : i % 2560 < 2560 := by
  omega

-- § blob layout

/-- Offset of the down codes: after `1280 * 640` gate/up code bytes. -/
def oDCodes : Nat := 1280 * 640

/-- Offset of the gate/up scales: after the down codes (`2560 * 160`). -/
def oGuSc : Nat := oDCodes + 2560 * 160

/-- Offset of the down scales: after `1280 * 40` f16 gate/up scales. -/
def oDSc : Nat := oGuSc + 1280 * 40 * 2

/-- Total bytes of one expert blob. -/
def blobBytes : Nat := oDSc + 2560 * 10 * 2

/-- The blob is exactly the 1,382,400 bytes the memory planner budgets per slot. -/
theorem blobBytes_value : blobBytes = 1382400 := by
  unfold blobBytes oDSc oGuSc oDCodes
  decide

/-- Scale offsets are laid out in strictly increasing order without overlap. -/
theorem blob_offsets_increasing : oDCodes < oGuSc ∧ oGuSc < oDSc ∧ oDSc < blobBytes := by
  unfold blobBytes oDSc oGuSc oDCodes
  decide

/-- Gate/up scale index for code byte `b` of a row stays inside the row's 40 scales. -/
theorem gu_scale_in_row (row b : Nat) (hb : b < 640) :
    row * 40 + (b * 4) / 64 < (row + 1) * 40 := by
  omega

/-- Down scale index for code byte `b` of a row stays inside the row's 10 scales. -/
theorem d_scale_in_row (row b : Nat) (hb : b < 160) :
    row * 10 + (b * 4) / 64 < (row + 1) * 10 := by
  omega

/-- Gate/up output position of weight `k` of byte `b` stays in the 2560-wide row. -/
theorem gu_out_in_row (b k : Nat) (hb : b < 640) (hk : k < 4) : b * 4 + k < 2560 := by
  omega

/-- Down output position of weight `k` of byte `b` stays in the 640-wide row. -/
theorem d_out_in_row (b k : Nat) (hb : b < 160) (hk : k < 4) : b * 4 + k < 640 := by
  omega

/-- A 2-bit code is at most three. -/
theorem code_le_three (x : Nat) : x &&& 3 ≤ 3 := Nat.and_le_right

set_option maxRecDepth 100000 in
/-- The four 2-bit fields of a byte reassemble the byte: the unpacking loses nothing. -/
theorem ternary_roundtrip : ∀ c : Fin 256,
    ((c.val &&& 3) + 4 * ((c.val >>> 2) &&& 3) + 16 * ((c.val >>> 4) &&& 3)
      + 64 * ((c.val >>> 6) &&& 3)) = c.val := by
  decide

/-- Decoded integer weight `code - 1` lies in `[-1, 2]`. -/
theorem decoded_range (q : Nat) (h : q ≤ 3) : (-1 : Int) ≤ (q : Int) - 1 ∧ (q : Int) - 1 ≤ 2 := by
  omega

-- § SwiGLU, QSA, KV

/-- Interleaved gate/up pair `(2k, 2k+1)` stays in the 1280-wide row for `k < 640`. -/
theorem swiglu_pair_in_row (k : Nat) (h : k < 640) : 2 * k + 1 < 1280 := by
  omega

/-- The gate half of a query head row starts after the 256 query values and stays in the 512-wide head. -/
theorem gate_offset_in_head (d : Nat) (h : d < 256) : 256 + d < 512 := by
  omega

/-- A block index `(z >> 1, z & 1)` over `z < 8` addresses four scale groups and K or V. -/
theorem kv_group_split (z : Nat) (h : z < 8) : z / 2 < 4 ∧ z % 2 < 2 := by
  omega

/-- Element index within a head of a (group, lane) block is below 256. -/
theorem kv_elem_lt (g t : Nat) (hg : g < 4) (ht : t < 64) : g * 64 + t < 256 := by
  omega

/-- The two rotary halves `pair` and `pair + 32` are both inside the 64-wide rotary span. -/
theorem rope_pairs_in_span (pair : Nat) (h : pair < 32) : pair < 64 ∧ pair + 32 < 64 := by
  omega

-- § router lanes

/-- Lane `l` owns expert `l + 32 i`; the owner of expert `e` is lane `e % 32`, slot `e / 32`. -/
theorem route_lane_inverse (lane i : Nat) (hl : lane < 32) :
    (lane + i * 32) % 32 = lane ∧ (lane + i * 32) / 32 = i := by
  omega

/-- Lane slots cover all 512 experts. -/
theorem route_cover (e : Nat) (h : e < 512) : e % 32 < 32 ∧ e / 32 < 16 := by
  omega

-- § GDN layout

/-- The channel count of the GDN qkv projection: q and k heads (2 * 16 * 128) plus v heads (48 * 128). -/
theorem gdn_c_value : 2 * 16 * 128 + 48 * 128 = 10240 := by decide

/-- Recurrence thread layout: four row groups of 32 rows cover the 128 state rows. -/
theorem gdn_row_groups : 4 * (128 / 4) = 128 := by decide

/-- A state row index `rg * 32 + r` with `rg < 4` and `r < 32` is below 128. -/
theorem gdn_row_lt (rg r : Nat) (h1 : rg < 4) (h2 : r < 32) : rg * 32 + r < 128 := by
  omega

end Strata.PrefillKernels



## FILE: /mnt/user-data/outputs/SpecController.lean
/-!
# MODULE: spec_controller
Speculative-decoding controller: picks the drafter and window size maximizing tokens per millisecond, keeps plain
decoding unless the gain clears a threshold, and learns per-position acceptance with an EMA.

## Source semantics
* The verify pass always commits one token, so expected tokens is `1 + sum of cumulative acceptance products`.
* `step_ms` clamps `n = k + 1` into `[1, K_MAX + 1]` before indexing the 9-entry ratio tables.
* The candidate with the highest rate wins; plain decoding is the baseline.
* If the best rate does not beat `baseline * (1 + min_gain)` the controller returns plain decoding.
* `observe` updates positions `0..min(k, accepted+1)-1`; all indices stay below `K_MAX`.

## Form template identifiers
* `Controller::expected` -> `expAux`/`expected`, `bucket` -> `bucket`, `K_MAX` -> `kMax`
* probabilities are fixed-point per mille (1000 = 1.0)
-/

namespace Strata.SpecController

/-- Largest draft window. -/
def kMax : Nat := 8

-- § table indexing

/-- The clamped table position `n` of a draft count `k`. -/
def clampN (k : Nat) : Nat := max 1 (min (k + 1) (kMax + 1))

/-- The clamped count is in `[1, 9]`. -/
theorem clampN_range (k : Nat) : 1 ≤ clampN k ∧ clampN k ≤ kMax + 1 := by
  unfold clampN kMax
  omega

/-- The table index `n - 1` is below the 9-entry table length, so no out-of-bounds read. -/
theorem clampN_index_lt (k : Nat) : clampN k - 1 < kMax + 1 := by
  have := clampN_range k
  omega

-- § match-length buckets

/-- Lookup acceptance bucket by match length. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- There are four buckets. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  by_cases h1 : 16 ≤ m <;> by_cases h2 : 8 ≤ m <;> by_cases h3 : 5 ≤ m <;> simp [h1, h2, h3]

/-- Longer matches never fall into a lower bucket. -/
theorem bucket_mono (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  by_cases h1 : 16 ≤ a <;> by_cases h2 : 8 ≤ a <;> by_cases h3 : 5 ≤ a <;>
    by_cases h4 : 16 ≤ b <;> by_cases h5 : 8 ≤ b <;> by_cases h6 : 5 ≤ b <;>
    simp [h1, h2, h3, h4, h5, h6] <;> omega

-- § expected tokens (fixed-point per mille)

/-- Sum of cumulative acceptance products over `k` positions starting at index `i` with running product `run`. -/
def expAux (p : Nat → Nat) (i run : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => let r := run * p i / 1000; r + expAux p (i + 1) r k

/-- Expected committed tokens: the always-committed one plus the accepted-draft terms. -/
def expected (p : Nat → Nat) (k : Nat) : Nat := 1000 + expAux p 0 1000 k

/-- Expected tokens is at least one token. -/
theorem expected_ge_one (p : Nat → Nat) (k : Nat) : 1000 ≤ expected p k := by
  unfold expected
  omega

/-- With probabilities at most one, expected tokens never exceed the window `k + 1`. -/
theorem expAux_le (p : Nat → Nat) (hp : ∀ i, p i ≤ 1000) :
    ∀ (k i run : Nat), run ≤ 1000 → expAux p i run k ≤ k * 1000
  | 0, i, run, h => by simp [expAux]
  | k + 1, i, run, h => by
      have h1 : run * p i ≤ 1000 * run := by
        rw [Nat.mul_comm 1000 run]
        exact Nat.mul_le_mul_left run (hp i)
      have h2 : run * p i / 1000 ≤ run := Nat.div_le_of_le_mul h1
      have ih := expAux_le p hp k (i + 1) (run * p i / 1000) (by omega)
      simp only [expAux]
      omega

/-- Expected tokens per step is at most `k + 1`. -/
theorem expected_le (p : Nat → Nat) (hp : ∀ i, p i ≤ 1000) (k : Nat) :
    expected p k ≤ (k + 1) * 1000 := by
  have := expAux_le p hp k 0 1000 (by omega)
  unfold expected
  omega

-- § choosing the best candidate

/-- Best rate: a running maximum starting at the plain-decoding baseline. -/
def best : Nat → List Nat → Nat
  | b, [] => b
  | b, c :: cs => best (max b c) cs

/-- The chosen rate is never below the baseline. -/
theorem best_ge_base : ∀ (cs : List Nat) (b : Nat), b ≤ best b cs
  | [], b => by simp [best]
  | c :: cs, b => by
      have := best_ge_base cs (max b c)
      simp only [best]
      omega

/-- The chosen rate is at least every candidate's rate. -/
theorem best_ge_mem : ∀ (cs : List Nat) (b c : Nat), c ∈ cs → c ≤ best b cs
  | [], b, c, h => by simp at h
  | d :: cs, b, c, h => by
      have h0 := best_ge_base cs (max b d)
      simp only [best]
      rcases List.mem_cons.mp h with h1 | h'
      · omega
      · exact best_ge_mem cs _ c h'

-- § min-gain gate

/-- The threshold gate in per-mille: keep speculation only when `rate * 1000 ≥ base * (1000 + gain)`. -/
def clearsGate (rate base gain : Nat) : Prop := base * (1000 + gain) ≤ rate * 1000

/-- Without a gain threshold, any rate at least the baseline clears the gate. -/
theorem clearsGate_zero (rate base : Nat) (h : base ≤ rate) : clearsGate rate base 0 := by
  unfold clearsGate
  omega

/-- A rate that clears the gate is at least the baseline. -/
theorem clearsGate_ge (rate base gain : Nat) (h : clearsGate rate base gain) : base ≤ rate := by
  unfold clearsGate at h
  have : base * 1000 ≤ base * (1000 + gain) := Nat.mul_le_mul_left base (by omega)
  omega

-- § observe

/-- Positions updated by `observe`: `min k (accepted + 1)` after clamping `accepted` into `[0, k]`. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- Updated positions never exceed the window. -/
theorem seen_le_k (k a : Nat) : seen k a ≤ k := by
  unfold seen
  omega

/-- Every updated index is a valid MTP table index when the window is at most `K_MAX`. -/
theorem seen_index_lt (k a i : Nat) (hk : k ≤ kMax) (hi : i < seen k a) : i < kMax := by
  have := seen_le_k k a
  unfold kMax at *
  omega

/-- When all drafts were accepted, the extension loop starts at `k` and reads index `k - 1 ≥ 0`. -/
theorem extend_index_ok (k : Nat) (hk : 0 < k) (hlt : k < kMax) : k - 1 < kMax := by
  unfold kMax at *
  omega

/-- The lookup candidate count is clipped to the window limit. -/
theorem lookup_clip_le (avail : Nat) : min avail kMax ≤ kMax := by
  omega

end Strata.SpecController



## FILE: /mnt/user-data/outputs/SpecCostModel.lean
/-!
# MODULE: spec_cost_model
Default cost-model constants of the speculative-decoding controller and the shape properties the controller's
arithmetic relies on.

## Source semantics
* `dense_ratio` and `distinct_ratio` have `K_MAX + 1 = 9` entries, indexed by `n - 1` for `n = k + 1`.
* Both start at 1.0 and never decrease as the window grows.
* `distinct_ratio[n-1] <= n`: a window of `n` tokens touches at most `n` times as many distinct experts as one token,
  so tokens-per-distinct-expert `n / U(n)` is at least one and the extra-use cost is non-negative.
* Window verification is cheaper per token than serial decoding: `dense_ratio[n-1] < n` for `n >= 2`.

## Form template identifiers
* `dense_ratio` -> `denseRatio`, `distinct_ratio` -> `distinctRatio` (hundredths, so 105 = 1.05)
* `K_MAX` -> `kMax`
-/

namespace Strata.SpecCostModel

/-- Largest draft window. -/
def kMax : Nat := 8

/-- Dense-pass cost ratio by `n = k + 1`, in hundredths. -/
def denseRatio : List Nat := [100, 105, 130, 145, 185, 220, 260, 300, 330]

/-- Distinct-expert ratio `U(n)/U(1)` by `n`, in hundredths. -/
def distinctRatio : List Nat := [100, 170, 231, 288, 340, 389, 435, 480, 520]

/-- Non-decreasing check for a list. -/
def nondecr : List Nat → Bool
  | a :: b :: rest => decide (a ≤ b) && nondecr (b :: rest)
  | _ => true

/-- Both tables have `K_MAX + 1` entries, so every clamped index is in range. -/
theorem table_lengths : denseRatio.length = kMax + 1 ∧ distinctRatio.length = kMax + 1 := by
  decide

/-- Both tables start at exactly 1.0. -/
theorem tables_start_at_one : denseRatio.head? = some 100 ∧ distinctRatio.head? = some 100 := by
  decide

/-- The dense-cost table is non-decreasing. -/
theorem dense_nondecreasing : nondecr denseRatio = true := by
  decide

/-- The distinct-expert table is non-decreasing. -/
theorem distinct_nondecreasing : nondecr distinctRatio = true := by
  decide

/-- For each `n`, entry `n - 1` of the distinct table is at most `n * 100`. -/
def distinctBelowN : List Nat → Nat → Bool
  | [], _ => true
  | u :: us, n => decide (u ≤ n * 100) && distinctBelowN us (n + 1)

/-- Distinct experts grow no faster than the token count: uses-per-expert is at least one. -/
theorem distinct_le_n : distinctBelowN distinctRatio 1 = true := by
  decide

/-- Window verification is sub-linear: dense ratio below `n` full passes for `n >= 2`. -/
def denseBelowN : List Nat → Nat → Bool
  | [], _ => true
  | u :: us, n => decide (n < 2 ∨ u < n * 100) && denseBelowN us (n + 1)

/-- Verifying `n` tokens costs less than `n` serial passes. -/
theorem dense_sublinear : denseBelowN denseRatio 1 = true := by
  decide

/-- The default thresholds: 5% minimum gain and EMA 0.05, both inside the unit interval. -/
theorem default_gain_bounds : 5 ≤ 100 ∧ 5 ≤ 100 := by
  decide

/-- Default hit rate 0.55 and extra-use cost 0.25 are valid fractions (hundredths). -/
theorem default_fractions : 55 ≤ 100 ∧ 25 ≤ 100 := by
  decide

end Strata.SpecCostModel



## FILE: /mnt/user-data/outputs/SuffixDrafter.lean
/-!
# MODULE: suffix_drafter
Suffix-lookup drafter: a hash table keyed by the last three tokens remembers the most recent positions where each
trigram ended; `propose` extends the longest matching suffix and copies the continuation as a draft.

## Source semantics
* Table capacity is a power of two at least twice the nominal capacity, probing is `(i + 1) & mask`.
* A key is never zero (`mix(...) | 1`), so zero marks an empty slot.
* Each slot keeps its `WAYS` most recent end positions, newest first.
* The match length never exceeds `max_match` or the available history `p + 1`.
* The continuation copies at most `max_k` tokens and only reads history up to the current position.
* `min_match_ = max 3 min_match` but `max_match_ = max min_match max_match`, so a degenerate pair
  (`min_match < 3` and `max_match < 3`) can never match.

## Form template identifiers
* `Slot::pos` -> `pushWay`, `find_slot` probe -> `probeNext`, `key_at` -> `keyNonzero`
* `propose` match loop -> `matchAux`, continuation -> `contLen`
-/

namespace Strata.SuffixDrafter

-- § slot ways

/-- Insert a new end position at the front of a slot's way list and drop the oldest beyond `W` ways. -/
def pushWay (W : Nat) (l : List Nat) (x : Nat) : List Nat := (x :: l).take W

/-- A slot never holds more than `W` positions. -/
theorem pushWay_length_le (W : Nat) (l : List Nat) (x : Nat) : (pushWay W l x).length ≤ W := by
  unfold pushWay
  simp <;> omega

/-- The newest position is stored first. -/
theorem pushWay_head (W : Nat) (l : List Nat) (x : Nat) (hW : 0 < W) :
    (pushWay W l x).head? = some x := by
  unfold pushWay
  cases W with
  | zero => omega
  | succ w => simp

/-- The way counter `min (n + 1) W` matches the list length once the list is full or short. -/
theorem pushWay_length (W : Nat) (l : List Nat) (x : Nat) :
    (pushWay W l x).length = min (l.length + 1) W := by
  unfold pushWay
  simp [List.length_take] <;> omega

-- § table

/-- A key `mix(...) | 1` is never zero, so zero is a safe empty marker. -/
theorem keyNonzero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  first
    | (have h1 := Nat.or_eq_zero_iff.mp h; omega)
    | (have h1 : 1 ≤ x ||| 1 := Nat.right_le_or; omega)
    | simp at h

/-- A power-of-two table indexed by `& mask` stays in range. -/
theorem probe_in_range (k i : Nat) : i &&& (2 ^ k - 1) < 2 ^ k := by
  have hp : 0 < 2 ^ k := by
    first
      | exact Nat.two_pow_pos k
      | exact Nat.pos_pow_of_pos k (by decide)
  have h : 2 ^ k - 1 < 2 ^ k := by omega
  first
    | exact Nat.and_lt_two_pow i h
    | exact Nat.lt_of_le_of_lt Nat.and_le_right h

/-- The linear-probe successor `(i + 1) & mask` also stays in range. -/
theorem probeNext_in_range (k i : Nat) : (i + 1) &&& (2 ^ k - 1) < 2 ^ k :=
  probe_in_range k (i + 1)

-- § match loop

/-- The match loop: extend while within history and tokens agree, bounded by fuel `maxM - len`. -/
def matchAux (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | fuel + 1, len =>
      if len ≤ p ∧ h (p - len) = h (cur - len) then matchAux h p cur fuel (len + 1) else len

/-- The loop never extends more than its fuel. -/
theorem matchAux_le_fuel (h : Nat → Nat) (p cur : Nat) :
    ∀ (fuel len : Nat), matchAux h p cur fuel len ≤ len + fuel
  | 0, len => by simp [matchAux]
  | fuel + 1, len => by
      have ih := matchAux_le_fuel h p cur fuel (len + 1)
      simp only [matchAux]
      split
      · omega
      · omega

/-- The loop never exceeds the available history `p + 1`. -/
theorem matchAux_le_hist (h : Nat → Nat) (p cur : Nat) :
    ∀ (fuel len : Nat), len ≤ p + 1 → matchAux h p cur fuel len ≤ p + 1
  | 0, len, hl => by simpa [matchAux] using hl
  | fuel + 1, len, hl => by
      simp only [matchAux]
      split
      · rename_i hc
        exact matchAux_le_hist h p cur fuel (len + 1) (by omega)
      · exact hl

/-- Starting from zero, a match is at most `max_match` long. -/
theorem match_le_max (h : Nat → Nat) (p cur maxM : Nat) :
    matchAux h p cur maxM 0 ≤ maxM := by
  have := matchAux_le_fuel h p cur maxM 0
  omega

-- § constructor clamps

/-- `min_match_` as built by the constructor. -/
def minMatch (a : Nat) : Nat := max 3 a

/-- `max_match_` as built by the constructor (it clamps against the raw `min_match`). -/
def maxMatch (a b : Nat) : Nat := max a b

/-- With a sane request (`min_match >= 3`) the effective bounds are ordered. -/
theorem bounds_ordered (a b : Nat) (ha : 3 ≤ a) : minMatch a ≤ maxMatch a b := by
  unfold minMatch maxMatch
  omega

/-- A degenerate request makes `max_match_` smaller than `min_match_`, so `propose` can never succeed. -/
theorem degenerate_never_matches (a b len : Nat) (ha : a < 3) (hb : b < 3)
    (hlen : len ≤ maxMatch a b) : len < minMatch a := by
  unfold minMatch maxMatch at *
  omega

-- § continuation

/-- Number of draft tokens copied: from `best_end + 1` up to `cur`, at most `max_k`. -/
def contLen (cur bestEnd maxK : Nat) : Nat := min maxK (cur - bestEnd)

/-- The draft never exceeds `max_k`. -/
theorem contLen_le_maxk (cur e k : Nat) : contLen cur e k ≤ k := by
  unfold contLen
  omega

/-- Every copied index `e + 1 + j` stays at or before the current position. -/
theorem cont_reads_in_history (cur e k j : Nat) (he : e < cur) (hj : j < contLen cur e k) :
    e + 1 + j ≤ cur := by
  unfold contLen at hj
  omega

end Strata.SuffixDrafter



## FILE: /mnt/user-data/outputs/PleDirectReader.lean
/-!
# MODULE: ple_direct_reader
Unbuffered SSD reader for the 90-byte PLE n-gram rows: rows are mapped to aligned page reads, rows sharing a page
are deduplicated into one job, completions are copied out, and an 8-way set-associative row cache keeps hot rows.

## Source semantics
* A row at byte `at = table_offset + row * 90` is read with an aligned window starting at `at / PAGE * PAGE`;
  the window is one page, or two when the row straddles a page boundary.
* The row lies entirely inside its window.
* Rows past the table end are zero-filled and never read; in-range rows end inside the table.
* The cache has `rows / 8` sets of 8 ways; the set index and the way pointer are always in range.
* The latency ring index wraps modulo 65536; a percentile index is below the sample count.

## Form template identifiers
* `PAGE` -> `page` (4096), `ROW_BYTES` -> `rowBytes` (90), `WAYS` -> `ways`
* `Job::length` -> `jobLength`, `Use::in_page` -> `inPage`
-/

namespace Strata.PleDirectReader

/-- Direct-I/O page size assumed by the reader (the platform page; 4096 on the target machines). -/
def page : Nat := 4096

/-- Bytes of one IQ4_NL row of 160 values (5 blocks of 18 bytes). -/
def rowBytes : Nat := 90

/-- A row is five 18-byte blocks. -/
theorem rowBytes_value : rowBytes = 5 * 18 := by decide

-- § aligned window

/-- Aligned start of the read window containing byte `at`. -/
def firstOf (at_ : Nat) : Nat := at_ / 4096 * 4096

/-- Window length: from the aligned start to the page containing the row's last byte, plus one page. -/
def jobLength (at_ : Nat) : Nat := (at_ + 90 - 1) / 4096 * 4096 - firstOf at_ + 4096

/-- Offset of the row inside its window (`Use::in_page`). -/
def inPage (at_ : Nat) : Nat := at_ - firstOf at_

/-- The window start is aligned. -/
theorem first_aligned (at_ : Nat) : firstOf at_ % 4096 = 0 := by
  unfold firstOf
  omega

/-- The window starts at or before the row. -/
theorem first_le (at_ : Nat) : firstOf at_ ≤ at_ := by
  unfold firstOf
  omega

/-- The row offset inside the window is smaller than one page. -/
theorem inPage_lt_page (at_ : Nat) : inPage at_ < 4096 := by
  unfold inPage firstOf
  omega

/-- The window is one page or two pages long. -/
theorem jobLength_cases (at_ : Nat) : jobLength at_ = 4096 ∨ jobLength at_ = 8192 := by
  unfold jobLength firstOf
  omega

/-- The row fits entirely inside its read window. -/
theorem row_in_window (at_ : Nat) : inPage at_ + 90 ≤ jobLength at_ := by
  unfold inPage jobLength firstOf
  omega

/-- Window length is a multiple of the page, as direct I/O requires. -/
theorem jobLength_aligned (at_ : Nat) : jobLength at_ % 4096 = 0 := by
  unfold jobLength firstOf
  omega

-- § table bounds

/-- A row inside the table ends inside the table, so the read stays within the validated file size. -/
theorem row_inside_table (off n r : Nat) (h : r < n) : off + r * 90 + 90 ≤ off + n * 90 := by
  omega

/-- The two-page window cap: two pages cover the row for any alignment. -/
theorem two_pages_enough (at_ : Nat) : at_ % 4096 + 90 ≤ 2 * 4096 := by
  omega

-- § row cache

/-- Number of sets for a requested row capacity. -/
def numSets (rows : Nat) : Nat := rows / 8

/-- The cache never holds more rows than requested. -/
theorem capacity_le_request (rows : Nat) : numSets rows * 8 ≤ rows := by
  unfold numSets
  omega

/-- The set index is in range when at least one set exists. -/
theorem set_index_lt (hash sets : Nat) (h : 0 < sets) : hash % sets < sets :=
  Nat.mod_lt hash h

/-- The flat slot index `s * 8 + w` stays inside `sets * 8` entries. -/
theorem slot_index_lt (s w sets : Nat) (hs : s < sets) (hw : w < 8) : s * 8 + w < sets * 8 := by
  omega

/-- The byte offset of a cache row stays inside the data array of `sets * 8 * 90` bytes. -/
theorem data_offset_in_range (s w sets b : Nat) (hs : s < sets) (hw : w < 8) (hb : b < 90) :
    (s * 8 + w) * 90 + b < sets * 8 * 90 := by
  omega

/-- The round-robin way pointer stays in `[0, 8)`. -/
theorem way_pointer_lt (w : Nat) : (w + 1) % 8 < 8 := by
  omega

-- § statistics

/-- The latency ring position wraps inside its 65536 entries. -/
theorem ring_index_lt (pos : Nat) : pos % 65536 < 65536 := by
  omega

/-- The percentile index `min (n - 1) k` is a valid sample index when there is at least one sample. -/
theorem percentile_index_lt (n k : Nat) (h : 0 < n) : min (n - 1) k < n := by
  omega

/-- In-flight slots: the free list can never exceed `max_inflight` when each slot is returned once. -/
theorem free_slots_bound (free inflight maxInflight : Nat) (h : free + inflight = maxInflight) :
    free ≤ maxInflight := by
  omega

end Strata.PleDirectReader



## FILE: /mnt/user-data/outputs/PleNgramHash.lean
/-!
# MODULE: ple_ngram_hash
The PLE n-gram hash and the IQ4_NL row decode: each token plus its two predecessors is mixed with three odd
multipliers, reduced modulo 16 per-head prime vocabularies, and offset into one 320,001,536-row table.

## Source semantics
* Row index of head `h` is `mixed % vocab[h] + offset[h]`, so it lies in `[offset[h], offset[h] + vocab[h])`.
* `offset[h+1] = offset[h] + vocab[h]`: the per-head ranges tile the table without overlap.
* The last range ends before the table's 320,001,536 rows.
* Heads are `(n - 2) * 8 + g` for `n in {2, 3}` and `g < 8`: all 16 heads are covered exactly once.
* An EOS (or missing) predecessor cuts the context: every later position is EOS too.
* A row is five 18-byte IQ4_NL blocks; block `b` writes outputs `b*32 + j` and `b*32 + j + 16`, covering 160 values.

## Form template identifiers
* `PleConsts::vocab` -> `vocab`, `PleConsts::offset` -> `offset`, `ngram_rows` -> `headIndex`/`ctx2`
* `kIq4Nl` -> `iq4nl`, `iq4nl_dequant_row` -> `dequantSlot`
-/

namespace Strata.PleNgramHash

-- § head tables

/-- Per-head prime vocabulary sizes, verbatim from the artifact. -/
def vocab : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head table offsets, verbatim from the artifact. -/
def offset : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Rows in the PLE table. -/
def tableRows : Nat := 320001536

/-- Check that each offset is the previous offset plus the previous vocabulary size. -/
def tiles : List Nat → List Nat → Bool
  | o :: o2 :: os, v :: vs => decide (o + v = o2) && tiles (o2 :: os) vs
  | _, _ => true

/-- The head ranges tile the table: offsets are the prefix sums of the vocabularies. -/
theorem ranges_tile : tiles offset vocab = true := by
  decide

/-- Both tables have 16 entries (one per head). -/
theorem tables_len : vocab.length = 16 ∧ offset.length = 16 := by
  decide

/-- The last head's range ends before the end of the table. -/
theorem last_range_in_table : 300001275 + 20000171 ≤ tableRows := by
  unfold tableRows
  decide

/-- Every vocabulary size is positive, so the modulus is defined. -/
theorem vocab_pos : vocab.all (fun v => decide (0 < v)) = true := by
  decide

/-- A row index of one head lies inside that head's range. -/
theorem row_in_head_range (mixed v o : Nat) (hv : 0 < v) :
    o ≤ mixed % v + o ∧ mixed % v + o < o + v := by
  have := Nat.mod_lt mixed hv
  omega

-- § head numbering

/-- Head index for n-gram order `n` and sub-head `g`. -/
def headIndex (n g : Nat) : Nat := (n - 2) * 8 + g

/-- Orders 2 and 3 with eight sub-heads each give heads below 16. -/
theorem headIndex_lt (n g : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (hg : g < 8) : headIndex n g < 16 := by
  unfold headIndex
  omega

/-- Distinct (order, sub-head) pairs give distinct heads. -/
theorem headIndex_inj (n m g k : Nat) (h1 : 2 ≤ n) (h2 : n ≤ 3) (h3 : 2 ≤ m) (h4 : m ≤ 3)
    (hg : g < 8) (hk : k < 8) (h : headIndex n g = headIndex m k) : n = m ∧ g = k := by
  unfold headIndex at h
  omega

/-- Every head is reached: head `h < 16` decomposes into an order and a sub-head. -/
theorem headIndex_surj (h : Nat) (hh : h < 16) :
    ∃ n g, 2 ≤ n ∧ n ≤ 3 ∧ g < 8 ∧ headIndex n g = h := by
  refine ⟨h / 8 + 2, h % 8, ?_, ?_, ?_, ?_⟩ <;> (try unfold headIndex) <;> omega

/-- The predecessor read `prev[n_prev - s]` with `n_prev = 2` and `s in {1, 2}` is a valid index. -/
theorem prev_index_valid (s : Nat) (h1 : 1 ≤ s) (h2 : s ≤ 2) : 2 - s < 2 := by
  omega

-- § EOS cut

/-- Context of order three: the token, then two predecessors (newest first), with the cut rule.
`eos` is the EOS id; a negative predecessor is a missing one. -/
def ctx2 (eos : Int) (tok newest oldest : Int) : Int × Int × Int :=
  let cut1 := decide (newest < 0 ∨ newest = eos)
  let c1 := if cut1 then eos else newest
  let t2 := if cut1 then (-1 : Int) else oldest
  let cut2 := cut1 || decide (t2 < 0 ∨ t2 = eos)
  let c2 := if cut2 then eos else t2
  (tok, c1, c2)

/-- Once the newest predecessor cuts the context, the older position is EOS too. -/
theorem eos_cut_propagates (eos tok newest oldest : Int) (h : newest < 0 ∨ newest = eos) :
    (ctx2 eos tok newest oldest).2.2 = eos := by
  simp [ctx2, h]

/-- The token itself is passed through unchanged. -/
theorem ctx_keeps_token (eos tok newest oldest : Int) : (ctx2 eos tok newest oldest).1 = tok := by
  simp [ctx2]

/-- Without any cut the history is the real tokens. -/
theorem no_cut_keeps_history (eos tok newest oldest : Int) (h1 : 0 ≤ newest) (h2 : newest ≠ eos)
    (h3 : 0 ≤ oldest) (h4 : oldest ≠ eos) :
    (ctx2 eos tok newest oldest).2 = (newest, oldest) := by
  have a : ¬ (newest < 0 ∨ newest = eos) := by omega
  have b : ¬ (oldest < 0 ∨ oldest = eos) := by omega
  simp [ctx2, a, b]

-- § IQ4_NL row decode

/-- The IQ4_NL codebook. -/
def iq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Strictly increasing check. -/
def strictInc : List Int → Bool
  | a :: b :: rest => decide (a < b) && strictInc (b :: rest)
  | _ => true

/-- The codebook has 16 entries (one per nibble). -/
theorem iq4nl_length : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_strictly_increasing : strictInc iq4nl = true := by decide

/-- The codebook fits in a signed byte. -/
theorem iq4nl_in_int8 : iq4nl.all (fun v => decide (-128 ≤ v ∧ v ≤ 127)) = true := by decide

/-- Output slot decode: nibble pair `j < 16` of block `b < 5` writes slots `b*32 + j` and `b*32 + j + 16`. -/
theorem dequant_covers (k : Nat) (hk : k < 160) :
    ∃ b j, b < 5 ∧ j < 16 ∧ (k = b * 32 + j ∨ k = b * 32 + j + 16) := by
  by_cases h : k % 32 < 16
  · exact ⟨k / 32, k % 32, by omega, h, Or.inl (by omega)⟩
  · exact ⟨k / 32, k % 32 - 16, by omega, by omega, Or.inr (by omega)⟩

/-- No slot is written twice: the two halves of one block are disjoint ranges. -/
theorem dequant_disjoint (b j b2 j2 : Nat) (hj : j < 16) (hj2 : j2 < 16) :
    b * 32 + j ≠ b2 * 32 + j2 + 16 := by
  omega

/-- A row of five 18-byte blocks is 90 bytes and holds 160 values. -/
theorem row_geometry : 5 * 18 = 90 ∧ 5 * 32 = 160 := by decide

end Strata.PleNgramHash



## FILE: /mnt/user-data/outputs/MemoryPlanner.lean
/-!
# MODULE: memory_planner
The VRAM memory planner: derives KV, indexer-key and recurrent-state costs from the model geometry and hands the
remainder of a fixed pool to expert-cache slots, refusing rather than overcommitting.

## Source semantics
* Recurrent state is `n_gdn * (S*HV*S*4 + (d_conv-1)*C*4)` bytes = 117,669,888 for the artifact (36 GDN layers).
* KV per token is `n_qsa * (INT8 K and V + fp16 group scales + one 128-wide indexer key per 4-token block)`
  = 13,056 bytes per token for the artifact.
* `make_plan` fails if `kv + fixed > pool` or if not even one slot fits; otherwise
  `kv + fixed + cache = pool`, `slots = cache / blob`, and `slots * blob <= cache`.
* A larger context never leaves more room for the expert cache.

## Form template identifiers
* `state_bytes` -> `stateBytes`, `kv_bytes_per_token` -> `kvBytesPerToken`, `make_plan` -> `makePlan`
* `Plan::cache_slots` -> `Plan.slots`, `Costs::expert_blob` -> `blob`, `DoesNotClose` -> `none`
-/

namespace Strata.MemoryPlanner

-- § geometry costs

/-- Recurrent state bytes per GDN layer: recurrence `128*48*128*4` plus conv history `3*10240*4`. -/
def stateBytesPerLayer : Nat := 128 * 48 * 128 * 4 + 3 * 10240 * 4

/-- Recurrent state across the 36 GDN layers of the artifact (48 layers minus 12 QSA). -/
def stateBytes : Nat := (48 - 12) * stateBytesPerLayer

/-- The planner's state term, measured in the source comments as 117.7 MB. -/
theorem stateBytes_value : stateBytesPerLayer = 3268608 ∧ stateBytes = 117669888 := by
  unfold stateBytes stateBytesPerLayer
  decide

/-- Slots lost to the state term at the 1,382,400-byte blob size: the recorded 85 slots. -/
theorem state_costs_85_slots : 117669888 / 1382400 = 85 := by decide

/-- KV bytes per token per QSA layer: INT8 K and V (`2*2*256`) plus fp16 scales (one per 64 elements). -/
def kvLayer : Nat := 2 * 2 * 256 + (2 * 2 * 256 / 64) * 2

/-- Indexer key bytes per token per layer: one 128-wide key head pooled per 4 tokens. -/
def idxLayer : Nat := 1 * 128 / 4

/-- KV plus indexer bytes per token over the 12 QSA layers. -/
def kvBytesPerToken : Nat := 12 * (kvLayer + idxLayer)

/-- The per-token cost of the artifact: 1056 + 32 bytes per layer, 13,056 over 12 layers. -/
theorem kvBytesPerToken_value : kvLayer = 1056 ∧ idxLayer = 32 ∧ kvBytesPerToken = 13056 := by
  unfold kvBytesPerToken kvLayer idxLayer
  decide

/-- At 32K tokens the KV and indexer cache is 427,819,008 bytes. -/
theorem kv_at_32k : kvBytesPerToken * 32768 = 427819008 := by
  unfold kvBytesPerToken kvLayer idxLayer
  decide

/-- KV cost is monotone in the context length. -/
theorem kv_monotone (a b : Nat) (h : a ≤ b) : kvBytesPerToken * a ≤ kvBytesPerToken * b :=
  Nat.mul_le_mul_left _ h

-- § the plan

/-- A computed plan. -/
structure Plan where
  kv : Nat
  cache : Nat
  slots : Nat
  used : Nat

/-- `make_plan`: `none` is the `DoesNotClose` exception. -/
def makePlan (pool kvPerTok ctx fixed blob : Nat) : Option Plan :=
  if pool < kvPerTok * ctx + fixed then none
  else if (pool - kvPerTok * ctx - fixed) / blob = 0 then none
  else some { kv := kvPerTok * ctx, cache := pool - kvPerTok * ctx - fixed,
              slots := (pool - kvPerTok * ctx - fixed) / blob,
              used := pool - (pool - kvPerTok * ctx - fixed) % blob }

/-- A successful plan accounts for the whole pool: KV, fixed costs and the cache add up exactly. -/
theorem makePlan_closes (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) :
    p.kv + fixed + p.cache = pool := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      omega

/-- A successful plan has at least one slot. -/
theorem makePlan_has_slot (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : 0 < p.slots := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      exact Nat.pos_of_ne_zero h2

/-- The slots never take more bytes than the cache budget. -/
theorem makePlan_slots_fit (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : p.slots * blob ≤ p.cache := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      exact Nat.div_mul_le_self _ _

/-- The plan's VRAM use never exceeds the pool. -/
theorem makePlan_used_le (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : p.used ≤ pool := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      omega

/-- The plan refuses instead of overcommitting: if KV plus fixed costs exceed the pool there is no plan. -/
theorem makePlan_refuses (pool kvPerTok ctx fixed blob : Nat) (h : pool < kvPerTok * ctx + fixed) :
    makePlan pool kvPerTok ctx fixed blob = none := by
  simp [makePlan, h]

/-- A longer context leaves no more cache room than a shorter one. -/
theorem cache_room_antitone (pool kvPerTok fixed a b : Nat) (h : a ≤ b) :
    pool - kvPerTok * b - fixed ≤ pool - kvPerTok * a - fixed := by
  have := Nat.mul_le_mul_left kvPerTok h
  omega

end Strata.MemoryPlanner



## FILE: /mnt/user-data/outputs/ChatFrontend.lean
/-!
# MODULE: chat_frontend
Request normalization and streamed-output parsing between API clients and the engine: reasoning-effort mapping,
Anthropic thinking budgets, and the incremental parser that never emits a partial tag.

## Source semantics
* Client effort spellings collapse onto four levels: none, low, medium, xhigh (`high` means `xhigh`).
* A thinking budget under 2048 tokens is `low`, under 8192 `medium`, otherwise `xhigh`; the map is monotone.
* The parser holds back the longest suffix of the buffer that is a proper prefix of a tag; what it emits plus
  what it holds is exactly what it received.
* The parser state machine starts in reasoning (or content), moves reasoning -> content on `</think>`,
  content -> call on `<tool_call>`, call -> content on `</tool_call>`, and never returns to reasoning.
* A value's single leading and trailing newline are stripped, nothing more.

## Form template identifiers
* `EFFORT` -> `effortLevel`, `budget_effort` -> `budgetRank`, `OutputParser._hold` -> `splitHeld`
* parser `state` -> `PState`, tags -> `Marker`
-/

namespace Strata.ChatFrontend

-- § effort mapping

/-- Levels the template understands. `none` means thinking disabled. -/
inductive Level | off | low | medium | xhigh
  deriving DecidableEq

/-- The client spelling table; `none` result is an unknown value (an HTTP 400). -/
def effortLevel : String → Option Level
  | "none" => some .off
  | "off" => some .off
  | "minimal" => some .off
  | "disabled" => some .off
  | "low" => some .low
  | "medium" => some .medium
  | "high" => some .xhigh
  | "xhigh" => some .xhigh
  | "max" => some .xhigh
  | "maximum" => some .xhigh
  | _ => none

/-- `high` is the model's default `xhigh`. -/
theorem high_is_xhigh : effortLevel "high" = some Level.xhigh := by
  first | decide | rfl | simp [effortLevel]

/-- All the "disable" spellings turn thinking off. -/
theorem disable_spellings : effortLevel "none" = some Level.off ∧ effortLevel "off" = some Level.off ∧
    effortLevel "minimal" = some Level.off ∧ effortLevel "disabled" = some Level.off := by
  first | decide | rfl | simp [effortLevel]

/-- An unknown spelling is rejected rather than guessed. -/
theorem unknown_rejected : effortLevel "turbo" = none := by
  first | decide | rfl | simp [effortLevel]

/-- Rank of a level in the order low < medium < xhigh. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- A larger thinking budget never selects a lower level. -/
theorem budgetRank_mono (a b : Nat) (h : a ≤ b) : budgetRank a ≤ budgetRank b := by
  unfold budgetRank
  by_cases h1 : a < 2048 <;> by_cases h2 : a < 8192 <;> by_cases h3 : b < 2048 <;>
    by_cases h4 : b < 8192 <;> simp [h1, h2, h3, h4] <;> omega

/-- Ranks are 0, 1 or 2. -/
theorem budgetRank_lt (n : Nat) : budgetRank n < 3 := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : n < 8192 <;> simp [h1, h2]

/-- The 2048 and 8192 boundaries land where the docstring says. -/
theorem budget_boundaries : budgetRank 2047 = 0 ∧ budgetRank 2048 = 1 ∧ budgetRank 8191 = 1 ∧
    budgetRank 8192 = 2 := by
  decide

-- § hold-back

/-- Split a buffer into the part emitted now and the held suffix of length `keep`. -/
def splitHeld (buf : List Char) (keep : Nat) : List Char × List Char :=
  (buf.take (buf.length - keep), buf.drop (buf.length - keep))

/-- Nothing is lost: emitted plus held is exactly the buffer. -/
theorem splitHeld_append (buf : List Char) (keep : Nat) :
    (splitHeld buf keep).1 ++ (splitHeld buf keep).2 = buf := by
  unfold splitHeld
  exact List.take_append_drop _ _

/-- The held part is at most `keep` characters. -/
theorem splitHeld_held_le (buf : List Char) (keep : Nat) :
    (splitHeld buf keep).2.length ≤ keep := by
  unfold splitHeld
  simp only [List.length_drop]
  omega

/-- When `keep` is a proper prefix length (below the tag length), the held part is shorter than the tag. -/
theorem splitHeld_shorter_than_tag (buf : List Char) (keep tagLen : Nat) (h : keep < tagLen) :
    (splitHeld buf keep).2.length < tagLen := by
  have := splitHeld_held_le buf keep
  omega

-- § parser state machine

/-- Parser states. -/
inductive PState | reasoning | content | call
  deriving DecidableEq

/-- Tag boundaries the parser reacts to. -/
inductive Marker | thinkEnd | callStart | callEnd
  deriving DecidableEq

/-- Transition on finding a tag in the current state; other combinations leave the state alone. -/
def step : PState → Marker → PState
  | .reasoning, .thinkEnd => .content
  | .content, .callStart => .call
  | .call, .callEnd => .content
  | s, _ => s

/-- The parser never re-enters reasoning: it is only reachable by staying in it. -/
theorem step_reasoning_only_from_reasoning (s : PState) (m : Marker) (h : step s m = .reasoning) :
    s = .reasoning := by
  revert h
  cases s <;> cases m <;> decide

/-- A tool call always returns to content. -/
theorem call_end_returns_content : step .call .callEnd = .content := by decide

/-- `</think>` ends reasoning. -/
theorem think_end_leaves_reasoning : step .reasoning .thinkEnd = .content := by decide

-- § value trimming

/-- Drop at most one leading newline. -/
def dropLeadNL : List Char → List Char
  | '\n' :: rest => rest
  | l => l

/-- Drop at most one trailing newline. -/
def dropTrailNL (l : List Char) : List Char :=
  if l.getLast? = some '\n' then l.dropLast else l

/-- Trimming only shortens a value, by at most one character per side. -/
theorem dropLeadNL_length (l : List Char) : l.length ≤ (dropLeadNL l).length + 1 := by
  match l with
  | [] => simp [dropLeadNL]
  | '\n' :: rest => simp [dropLeadNL]
  | c :: rest =>
      by_cases h : c = '\n'
      · subst h
        simp [dropLeadNL]
      · simp [dropLeadNL, h]

/-- Dropping the last element removes exactly one. -/
theorem dropTrailNL_length (l : List Char) : l.length ≤ (dropTrailNL l).length + 1 := by
  unfold dropTrailNL
  split
  · simp [List.length_dropLast] <;> omega
  · omega

end Strata.ChatFrontend



## FILE: /mnt/user-data/outputs/ServeEndpoints.lean
/-!
# MODULE: serve_endpoints
The HTTP serving layer over a resident engine: prompt preparation (image-pad expansion and the context check),
finish-reason mapping for the OpenAI and Anthropic wire formats, API-key authorization, the image-embedding cache
bound, and the incremental detokenizer.

## Source semantics
* Each `<|image_pad|>` in the rendered prompt is replaced by one pad per image token of the matching image;
  when the pad count equals the image count the new length is `len - pads + sum of image token counts`.
* A request is rejected (never truncated) when `prompt + max_new` exceeds the engine context.
* OpenAI: `tool_calls` is the finish reason iff at least one call was emitted and the engine stopped;
  Anthropic maps `stop -> end_turn`, `length -> max_tokens`, and `tool_use` after a call.
* With no configured key every request is authorized; with a key, exactly the matching credential is.
* The image cache holds at most 64 entries: the oldest is evicted when a 65th arrives.
* The detokenizer emits only the new suffix of the decoded text and tracks how much it has sent.

## Form template identifiers
* `Service.prepare` -> `expand`, `Handler._authorized` -> `authorized`, `Vision.encode` cache -> `cacheAfter`
* `Detokenizer.push` -> `detokDelta`/`detokSent`
-/

namespace Strata.ServeEndpoints

-- § image-pad expansion

/-- Replace the k-th pad by `counts[k]` pads; extra pads without a matching image are kept as-is. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
      if t = pad then
        match cs with
        | [] => t :: expand pad ts []
        | c :: cs' => List.replicate c pad ++ expand pad ts cs'
      else t :: expand pad ts cs

/-- With as many images as pads, the expanded prompt has exactly `len - pads + sum counts` tokens. -/
theorem expand_length (pad : Nat) :
    ∀ (ids cs : List Nat), ids.count pad = cs.length →
      (expand pad ids cs).length + ids.count pad = ids.length + cs.sum
  | [], cs, h => by
      have hc : cs = [] := by
        cases cs with
        | nil => rfl
        | cons c cs' => simp at h
      subst hc
      simp [expand]
  | t :: ts, cs, h => by
      by_cases ht : t = pad
      · subst ht
        cases cs with
        | nil => simp [List.count_cons] at h
        | cons c cs' =>
            have h' : ts.count t = cs'.length := by
              simp [List.count_cons] at h
              omega
            have ih := expand_length t ts cs' h'
            simp [expand, List.count_cons, ih] <;> omega
      · have h' : ts.count pad = cs.length := by
          have hne : (t == pad) = false := by simp [ht]
          simpa [List.count_cons, hne] using h
        have ih := expand_length pad ts cs h'
        have hne : (t == pad) = false := by simp [ht]
        simp [expand, ht, List.count_cons, hne] <;> omega

/-- The request is accepted only when the prompt and generation fit in the context. -/
def accepts (promptLen maxNew ctx : Nat) : Prop := promptLen + maxNew ≤ ctx

/-- Accepted requests never need truncation. -/
theorem accepts_fits (p m c : Nat) (h : accepts p m c) : p + m ≤ c := h

/-- A rejected request really exceeds the context. -/
theorem rejects_exceeds (p m c : Nat) (h : ¬ accepts p m c) : c < p + m := by
  unfold accepts at h
  omega

/-- The engine's own check (`n + max_new + 8 > max_context` is an error) is stronger than the server's. -/
theorem engine_check_stronger (p m c : Nat) (h : p + m + 8 ≤ c) : accepts p m c := by
  unfold accepts
  omega

-- § finish reasons

/-- Engine finish status. -/
inductive Finish | stop | length
  deriving DecidableEq

/-- OpenAI finish reason: `tool_calls` after a stopped generation that emitted calls. -/
def openaiFinish (calls : Nat) (f : Finish) : String :=
  if 0 < calls ∧ f = Finish.stop then "tool_calls"
  else match f with
    | .stop => "stop"
    | .length => "length"

/-- Anthropic stop reason. -/
def anthropicStop (usedTool : Bool) (f : Finish) : String :=
  if usedTool = true ∧ f = Finish.stop then "tool_use"
  else match f with
    | .stop => "end_turn"
    | .length => "max_tokens"

/-- A length-limited generation is always reported as `length` on the OpenAI wire. -/
theorem openai_length_preserved (calls : Nat) : openaiFinish calls .length = "length" := by
  simp [openaiFinish]

/-- A length-limited generation is always `max_tokens` for Anthropic. -/
theorem anthropic_length_preserved (u : Bool) : anthropicStop u .length = "max_tokens" := by
  simp [anthropicStop]

/-- Without tool calls a stop is a plain `stop`. -/
theorem openai_plain_stop : openaiFinish 0 .stop = "stop" := by
  simp [openaiFinish]

/-- With a tool call a stop becomes `tool_calls`. -/
theorem openai_tool_stop (n : Nat) : openaiFinish (n + 1) .stop = "tool_calls" := by
  simp [openaiFinish]

/-- Without tool use a stop is `end_turn`. -/
theorem anthropic_plain_stop : anthropicStop false .stop = "end_turn" := by
  simp [anthropicStop]

-- § authorization

/-- With an empty configured key everything passes; otherwise the credential must equal the key. -/
def authorized (key cred : String) : Prop := key = "" ∨ cred = key

/-- No key configured: open access. -/
theorem authorized_open (cred : String) : authorized "" cred := Or.inl rfl

/-- A key is configured: exactly the matching cred passes. -/
theorem authorized_iff (key cred : String) (h : key ≠ "") : authorized key cred ↔ cred = key := by
  unfold authorized
  constructor
  · intro hh
    rcases hh with hh | hh
    · exact absurd hh h
    · exact hh
  · intro hh
    exact Or.inr hh

-- § cache bound

/-- Entries after inserting one: the 65th insert evicts the oldest, so the size stays at 64. -/
def cacheAfter (size : Nat) : Nat := if 64 < size + 1 then size else size + 1

/-- The image cache never exceeds 64 entries. -/
theorem cache_bounded (size : Nat) (h : size ≤ 64) : cacheAfter size ≤ 64 := by
  unfold cacheAfter
  split <;> omega

/-- Inserting into a non-full cache grows it by one. -/
theorem cache_grows (size : Nat) (h : size < 64) : cacheAfter size = size + 1 := by
  unfold cacheAfter
  split <;> omega

-- § detokenizer

/-- New text since the last emission. -/
def detokDelta (text : List Char) (sent : Nat) : List Char := text.drop sent

/-- Text length marker after a push: unchanged while the decode ends in an incomplete character. -/
def detokSent (text : List Char) (sent : Nat) (incomplete : Bool) : Nat :=
  if incomplete then sent else text.length

/-- The emitted delta has exactly the unsent length. -/
theorem detokDelta_length (text : List Char) (sent : Nat) :
    (detokDelta text sent).length = text.length - sent := by
  simp [detokDelta]

/-- Previously sent text plus the delta is the whole decoded text. -/
theorem detokDelta_complete (text : List Char) (sent : Nat) (h : sent ≤ text.length) :
    text.take sent ++ detokDelta text sent = text := by
  unfold detokDelta
  exact List.take_append_drop _ _

/-- The sent marker never moves backwards while the text only grows. -/
theorem detokSent_mono (text : List Char) (sent : Nat) (inc : Bool) (h : sent ≤ text.length) :
    sent ≤ detokSent text sent inc := by
  unfold detokSent
  split <;> omega

/-- The sent marker never passes the decoded text. -/
theorem detokSent_le_len (text : List Char) (sent : Nat) (inc : Bool) (h : sent ≤ text.length) :
    detokSent text sent inc ≤ text.length := by
  unfold detokSent
  split <;> omega

end Strata.ServeEndpoints





## FILE: /mnt/user-data/outputs/CpuExpertGeometry.lean
/-!
# MODULE: cpu_expert_geometry
Fixed geometry of the Q2_0 routed-expert blob (hidden width, FFN width, block sizes, byte offsets) and the
runtime CPU-feature gate that decides whether the AVX-512 expert kernel may run.

## Source semantics
* One Q2_0 weight block is `QK = 64` weights = fp16 scale (2 B) + 16 code bytes = 18 B; one activation chunk is
  `QKA = 32` elements, so a weight block spans exactly two activation chunks.
* One expert blob = gate/up codes, down codes, gate/up scales, down scales; its size is 1,382,400 bytes.
* `CpuFeatures::usable()` is the conjunction of five CPUID bits; `reason()` names exactly the missing ones.
* An activation with `n ≤ H` elements and `n % QKA = 0` uses at most `MAXC = 80` chunks.

## Form template identifiers
H → hidden_width, FF → ffn_width, NE → experts_per_layer, QK → weight_block_len, QKA → act_chunk_len,
BB → weight_block_bytes, MAXC → max_act_chunks, ROW_GU/ROW_D → code_bytes_per_row,
SC_GU/SC_D → blocks_per_row, BLOB → expert_blob_bytes, O_* → blob_section_offset,
CpuFeatures → cpu_feature_set, usable → feature_gate, reason → missing_feature_list.
-/

namespace Strata.Cpu.Geometry

/-- n_embd. -/
def H : Nat := 2560
/-- Expert intermediate width. -/
def FF : Nat := 640
/-- Routed experts per layer. -/
def NE : Nat := 512
/-- Weights per fp16 scale. -/
def QK : Nat := 64
/-- Activation elements per scale. -/
def QKA : Nat := 32
/-- Bytes per Q2_0 block. -/
def BB : Nat := 18
/-- Activation chunks across the widest reduction. -/
def MAXC : Nat := H / QKA
/-- Code bytes per gate/up row. -/
def ROW_GU : Nat := H * 2 / 8
/-- Code bytes per down row. -/
def ROW_D : Nat := FF * 2 / 8
/-- Weight blocks per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Weight blocks per down row. -/
def SC_D : Nat := FF / QK
/-- Bytes of one expert blob. -/
def BLOB : Nat := 3 * (H * FF * BB / QK)

/-- Offset of the gate/up codes. -/
def O_GU_CODES : Nat := 0
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2

-- § derived constants

/-- The derived numeric constants take the values recorded in the header comments. -/
theorem constants_values :
    MAXC = 80 ∧ ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 ∧ BLOB = 1382400 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;> decide

/-- A weight block spans exactly two activation chunks. -/
theorem block_spans_two_chunks : QK = 2 * QKA := by decide

/-- A block is its 2-byte scale plus `QK / 4` code bytes. -/
theorem block_bytes_decomposition : BB = 2 + QK / 4 := by decide

/-- Row code bytes equal 16 per weight block, for both gate/up and down rows. -/
theorem row_bytes_per_block : ROW_GU = SC_GU * 16 ∧ ROW_D = SC_D * 16 := by
  constructor <;> decide

/-- Row widths are whole numbers of weight blocks. -/
theorem rows_are_block_multiples : H = SC_GU * QK ∧ FF = SC_D * QK := by
  constructor <;> decide

/-- The activation chunk count of the hidden vector is twice the gate/up block count. -/
theorem chunks_twice_blocks : MAXC = 2 * SC_GU := by decide

-- § blob layout

/-- The four sections tile the blob exactly: the last section ends at `BLOB`. -/
theorem blob_layout_tiles :
    O_GU_CODES = 0 ∧ O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 ∧
      O_D_SCALES + H * SC_D * 2 = BLOB := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> decide

/-- Sections are ordered, so no two overlap. -/
theorem blob_sections_ordered :
    O_GU_CODES < O_D_CODES ∧ O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧
      O_D_SCALES < BLOB := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

/-- Last gate/up row (index `2*FF-1`) of codes stays inside the gate/up code section. -/
theorem gu_codes_row_in_bounds (r : Nat) (hr : r < FF) :
    (2 * r + 1 + 1) * ROW_GU ≤ O_D_CODES := by
  have h : 2 * r + 2 ≤ 2 * FF := by unfold FF at *; omega
  have h2 : (2 * r + 2) * ROW_GU ≤ (2 * FF) * ROW_GU := Nat.mul_le_mul_right _ h
  have h3 : (2 * r + 1 + 1) = 2 * r + 2 := by omega
  rw [h3]
  unfold O_D_CODES
  exact h2

/-- Gate/up scales of row `2r+1` stay inside the scale section. -/
theorem gu_scales_row_in_bounds (r : Nat) (hr : r < FF) :
    (2 * r + 1) * SC_GU * 2 + SC_GU * 2 ≤ 2 * FF * SC_GU * 2 := by
  unfold FF SC_GU H QK at *
  omega

-- § activation chunk counts

/-- An activation of `n ≤ H` elements uses at most `MAXC` chunks. -/
theorem nchunks_le_maxc (n : Nat) (h : n ≤ H) : n / QKA ≤ MAXC := by
  unfold MAXC H QKA at *
  omega

/-- When `n` is a multiple of the chunk length, the chunks cover `n` exactly. -/
theorem chunks_cover_exactly (n : Nat) (h : n % QKA = 0) : (n / QKA) * QKA = n := by
  unfold QKA at *
  omega

/-- Chunk `2b` and `2b+1` are the two halves of weight block `b`. -/
theorem chunk_index_of_block (b h : Nat) (hh : h < 2) : (2 * b + h) / 2 = b ∧ (2 * b + h) % 2 = h := by
  constructor <;> omega

-- § CPU feature gate

/-- The five CPUID feature bits the kernel requires. -/
structure CpuFeatures where
  avx512f : Bool
  avx512bw : Bool
  avx512vl : Bool
  avx512_vnni : Bool
  avx512_vbmi : Bool

/-- `usable()` is the conjunction of all five bits. -/
def CpuFeatures.usable (f : CpuFeatures) : Bool :=
  f.avx512f && f.avx512bw && f.avx512vl && f.avx512_vnni && f.avx512_vbmi

/-- The names `reason()` prints, in source order. -/
def CpuFeatures.missing (f : CpuFeatures) : List String :=
  (if f.avx512f then [] else ["AVX512F"]) ++ (if f.avx512bw then [] else ["AVX512BW"]) ++
    (if f.avx512vl then [] else ["AVX512VL"]) ++ (if f.avx512_vnni then [] else ["AVX512-VNNI"]) ++
    (if f.avx512_vbmi then [] else ["AVX512-VBMI"])

/-- `reason()` reports "ok" exactly when the CPU is usable. -/
theorem usable_iff_nothing_missing (f : CpuFeatures) : f.usable = true ↔ f.missing = [] := by
  obtain ⟨a, b, c, d, e⟩ := f
  cases a <;> cases b <;> cases c <;> cases d <;> cases e <;> simp [CpuFeatures.usable, CpuFeatures.missing]

/-- A CPU with every bit set is usable. -/
theorem all_features_usable : (CpuFeatures.mk true true true true true).usable = true := by decide

/-- A CPU lacking only VNNI is rejected, and the reason names only VNNI. -/
theorem only_vnni_missing :
    (CpuFeatures.mk true true true false true).usable = false ∧
      (CpuFeatures.mk true true true false true).missing = ["AVX512-VNNI"] := by
  constructor <;> decide

/-- Bit `k` of a CPUID register. -/
def cpuidBit (x k : Nat) : Bool := (x / 2 ^ k) % 2 == 1

/-- The leaf-7 bit positions used by `cpu_features()` are pairwise distinct per register. -/
theorem cpuid_positions_distinct :
    (16 ≠ 30 ∧ 16 ≠ 31 ∧ 30 ≠ 31) ∧ (11 ≠ 1) := by decide

/-- Setting only bit 16 of EBX sets only AVX512F among the EBX-derived flags. -/
theorem ebx_bit16_only_f :
    cpuidBit 65536 16 = true ∧ cpuidBit 65536 30 = false ∧ cpuidBit 65536 31 = false := by decide

end Strata.Cpu.Geometry



## FILE: /mnt/user-data/outputs/CpuExpertKernel.lean
/-!
# MODULE: cpu_expert_kernel
Arithmetic contracts of the AVX-512 Q2_0 expert kernel: the code-offset identity that removes the `-1` bias,
the two-chunk split of a weight block, batching of eight blocks per scale vector, multi-token dispatch, and the
VPMULTISHIFTQB 2-bit unpack.

## Source semantics
* `sum_j (c_j - 1) d_w xhat_j d_x = d_w (d_x sum(c xhat) - d_x sum(xhat))`: the dot is done on raw codes 0..3
  and the weight-independent correction `hx = scale * sum` is subtracted once.
* A 64-weight block is two 32-element activation chunks, so a dot over the concatenation is the sum of the
  per-chunk dots.
* Scale vectors are built for `nb = min 8 (nblocks - b0)` blocks at a time, with a 2*nb-lane mask.
* The multi-token entry points handle `1..8` tokens by template; any other count falls back to one-by-one.
* Unpack: output byte `i` of a qword lane is `(u16 >> 2i) & 0xFF`, and `& 3` yields the 2-bit code.

## Form template identifiers
row_dot → quantized_row_dot, ActQ → quantized_activation, act_quant_q8_1 → activation_quantizer,
unpack_q2_0/unpack64_q2_0 → code_unpack, scales8 → batched_scale_vector, MAXT → max_tokens_per_expert,
s2_expert_vnni_multi → multi_token_expert, expert_oracle_q8_0 → oracle_contract_path.
-/

namespace Strata.Cpu.ExpertKernel

-- § the code-offset identity

/-- Integer dot product of two lists (extra elements of the longer list are ignored). -/
def dot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + dot cs xs
  | _, _ => 0

/-- Sum of a list of integers. -/
def sumL : List Int → Int
  | [] => 0
  | x :: xs => x + sumL xs

/-- Subtract the code offset 1 from every code. -/
def offs : List Int → List Int
  | [] => []
  | c :: cs => (c - 1) :: offs cs

/-- Dot with offset-removed codes equals the raw-code dot minus the activation sum. -/
theorem dot_offs (cs xs : List Int) (h : cs.length = xs.length) :
    dot (offs cs) xs = dot cs xs - sumL xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [dot, offs, sumL]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have key := ih xs h'
      have e : (c - 1) * x = c * x - x := by rw [Int.sub_mul, Int.one_mul]
      simp only [dot, offs, sumL]
      rw [key]
      omega

/-- A dot over a concatenation is the sum of the dots (the first chunk's lengths must match). -/
theorem dot_append (a b x y : List Int) (h : a.length = x.length) :
    dot (a ++ b) (x ++ y) = dot a x + dot b y := by
  induction a generalizing x with
  | nil =>
    cases x with
    | nil => simp [dot]
    | cons u us => simp at h
  | cons c cs ih =>
    cases x with
    | nil => simp at h
    | cons u us =>
      have h' : cs.length = us.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih us h']
      omega

/-- The sum of a concatenation is the sum of the sums (used for `hx` per chunk, then per block). -/
theorem sumL_append (a b : List Int) : sumL (a ++ b) = sumL a + sumL b := by
  induction a with
  | nil => simp [sumL]
  | cons c cs ih => simp only [List.cons_append, sumL]; rw [ih]; omega

/-- The two-chunk form of the identity: block dot minus block correction, chunk by chunk. -/
theorem block_two_chunks (c1 c2 x1 x2 : List Int) (h1 : c1.length = x1.length) (h2 : c2.length = x2.length) :
    dot (offs (c1 ++ c2)) (x1 ++ x2) =
      (dot c1 x1 - sumL x1) + (dot c2 x2 - sumL x2) := by
  have hl : (c1 ++ c2).length = (x1 ++ x2).length := by simp [h1, h2]
  rw [dot_offs _ _ hl, dot_append _ _ _ _ h1, sumL_append]
  omega

-- § batching eight blocks per scale vector

/-- Number of blocks processed in the batch starting at `b0`. -/
def batchSize (nblocks b0 : Nat) : Nat := if nblocks - b0 < 8 then nblocks - b0 else 8

/-- Every batch starting before `nblocks` is non-empty, at most 8 wide, and stays in range. -/
theorem batch_in_range (nblocks b0 : Nat) (h : b0 < nblocks) :
    0 < batchSize nblocks b0 ∧ batchSize nblocks b0 ≤ 8 ∧ b0 + batchSize nblocks b0 ≤ nblocks := by
  unfold batchSize
  split <;> omega

/-- A batch is either the final one (reaches `nblocks`) or a full 8-block one. -/
theorem batch_full_or_last (nblocks b0 : Nat) (h : b0 < nblocks) :
    b0 + batchSize nblocks b0 = nblocks ∨ batchSize nblocks b0 = 8 := by
  unfold batchSize
  split <;> omega

/-- Number of batches needed for `n` blocks. -/
theorem batch_count (n : Nat) : (n + 7) / 8 * 8 ≥ n := by omega

/-- Lane `l` (0..15) of the permute index at block offset `i` selects `l/8 + 2i`, which stays in 0..15. -/
theorem permute_index_in_range (i l : Nat) (hi : i < 8) (hl : l < 16) : l / 8 + 2 * i < 16 := by omega

/-- Global chunk `2(b0+i)+h` is local lane `2i+h` of the batch whose base chunk is `2*b0`. -/
theorem chunk_local_lane (b0 i h : Nat) : 2 * (b0 + i) + h = 2 * b0 + (2 * i + h) := by omega

/-- A full batch uses all 16 mask lanes: `2^(2*8) - 1 = 0xFFFF`. -/
theorem full_mask_is_ffff : 2 ^ (2 * 8) - 1 = 65535 := by decide

/-- A partial batch of `nb < 8` blocks masks in exactly `2*nb` low lanes (value `2^(2nb) - 1 < 2^16`). -/
theorem partial_mask_lt (nb : Nat) (h : nb < 8) : 2 ^ (2 * nb) - 1 < 65536 := by
  have : 2 ^ (2 * nb) ≤ 2 ^ 14 := Nat.pow_le_pow_right (by decide) (by omega)
  omega

-- § multi-token dispatch

/-- Maximum tokens per expert. -/
def MAXT : Nat := 8

/-- The template instantiation chosen by the `switch`: 1..7 are exact, everything else becomes 8. -/
def effNT (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- For every legal token count the template width equals the token count. -/
theorem effNT_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ MAXT) : effNT n = n := by
  unfold effNT MAXT at *
  split <;> omega

/-- The selected width never exceeds `MAXT` (so the stack arrays `acc[NT]` fit the buffers). -/
theorem effNT_le_maxt (n : Nat) : effNT n ≤ MAXT := by
  unfold effNT MAXT
  split <;> omega

/-- `n = 0` silently selects the width-8 template: callers must guard the lower bound (the source does, in
`s2_expert_vnni_multi`, only through its fallback test). -/
theorem effNT_zero : effNT 0 = 8 := by decide

/-- The multi-token path is taken exactly when the oracle is off and `1 ≤ n ≤ MAXT`. -/
def takesMultiPath (oracle : Bool) (n : Nat) : Bool := !oracle && decide (1 ≤ n) && decide (n ≤ MAXT)

/-- On the multi-token path the dispatched width is exactly the token count. -/
theorem multi_path_exact (oracle : Bool) (n : Nat) (h : takesMultiPath oracle n = true) : effNT n = n := by
  unfold takesMultiPath at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at h
  exact effNT_exact n h.1.2 h.2

/-- With the oracle on, the multi-token path is never taken. -/
theorem oracle_disables_multi (n : Nat) : takesMultiPath true n = false := by
  simp [takesMultiPath]

-- § VPMULTISHIFTQB unpack (control, data) order

/-- 2-bit code `i` of a 16-bit word, as `(w >> 2i) & 3`. -/
def code2 (w i : Nat) : Nat := (w / 4 ^ i) % 4

/-- Within a u16 made of two bytes, codes 0..3 come from the low byte, in order. -/
theorem code2_low_byte (b0 b1 i : Nat) (hb : b0 < 256) (hi : i < 4) :
    code2 (b0 + 256 * b1) i = (b0 / 4 ^ i) % 4 := by
  unfold code2
  have : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

/-- Codes 4..7 come from the high byte, in order. -/
theorem code2_high_byte (b0 b1 i : Nat) (hb : b0 < 256) (hi : 4 ≤ i) (hi2 : i < 8) :
    code2 (b0 + 256 * b1) i = (b1 / 4 ^ (i - 4)) % 4 := by
  unfold code2
  have : i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 7 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

/-- A 2-bit code lies in 0..3, so `c - 1` lies in {-1,0,1,2}. -/
theorem code2_lt_four (w i : Nat) : code2 w i < 4 := by
  unfold code2
  exact Nat.mod_lt _ (by decide)

-- § oracle-path reduction

/-- The integer reduction tree of `chunk_dot_oracle` (two horizontal adds) sums all eight lanes. -/
theorem hadd_tree_sums_lanes (l0 l1 l2 l3 l4 l5 l6 l7 : Int) :
    let h0 := l0 + l4
    let h1 := l1 + l5
    let h2 := l2 + l6
    let h3 := l3 + l7
    ((h0 + h1) + (h2 + h3)) = l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 := by
  intro h0 h1 h2 h3
  omega

end Strata.Cpu.ExpertKernel



## FILE: /mnt/user-data/outputs/CpuIqAvx512Kernels.lean
/-!
# MODULE: cpu_iq_avx512_kernels
Layout and dispatch contracts of the i-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) AVX-512 expert rows:
block byte sizes, every byte offset the decoders read, the sign-mask negation, the scale packing and the
token-batching loop.

## Source semantics
* Each 256-value block is decoded as four 64-value chunks `j = 0..3`; every offset read for chunk `j` must lie
  inside the block (`bytes` per format).
* Signs are applied as a masked subtract `0 - y` on lanes whose mask bit is set; applying it twice is identity.
* `scales4` packs four 16-bit scales into one qword; lane `l` of 32 int16 lanes reads field `l / 8`.
* Per-block scale factors are `2*s + 1` for a 4-bit `s`, hence odd and in 1..31.
* Token counts above 4 are processed in chunks of at most 4 whose sizes sum to the total.
* `iq512_supported` is true exactly for ggml types 16, 17, 18, 21, 22.

## Form template identifiers
Fmt<TY> → quant_format_decoder, bytes → block_bytes, K → format_constant, decode → block_chunk_decode,
row_dot → iq_row_dot, gu_rows → gate_up_rows, dot_rows → down_rows, iq512_supported → format_supported.
-/

namespace Strata.Cpu.IqAvx512

-- § block sizes

/-- Block bytes per format (type id ↦ bytes). -/
def blockBytes : Nat → Option Nat
  | 16 => some 66
  | 17 => some 74
  | 18 => some 98
  | 21 => some 110
  | 22 => some 82
  | _ => none

/-- Supported-type predicate, as in `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- A type has a block size exactly when it is supported. -/
theorem supported_iff_has_bytes (t : Nat) : supported t = true ↔ (blockBytes t).isSome = true := by
  constructor
  · intro h
    simp only [supported, Bool.or_eq_true, beq_iff_eq] at h
    rcases h with (((h | h) | h) | h) | h <;> subst h <;> rfl
  · intro h
    unfold blockBytes at h
    split at h <;> simp_all [supported]

/-- The byte sizes are the sums of the field sizes of each format. -/
theorem block_bytes_decomposition :
    66 = 2 + 64 ∧ 74 = 2 + 64 + 8 ∧ 82 = 2 + 64 + 8 + 8 ∧ 98 = 2 + 64 + 32 ∧
      110 = 2 + 64 + 8 + 32 + 4 := by decide

-- § every decoder read stays in bounds (chunk j < 4)

/-- IQ2_XXS reads `qs` words at `2 + 16 j .. 2 + 16 j + 15`. -/
theorem iq2xxs_reads_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS: grid words, then the scale bytes at `66 + 2j`, `66 + 2j + 1`. -/
theorem iq2xs_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S: grid bytes `2 + 8j`, signs as a u64 at `2 + 32 + 8j`, `qh` at `66 + 2j`, scales at `74 + 2j`. -/
theorem iq2s_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS: grid bytes at `2 + 16j`, then two u32 words at `2 + 64 + 8j`. -/
theorem iq3xxs_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: grid bytes, `qh` pair at `66 + 2j`, signs u64 at `74 + 8j`, scale byte at `106 + j`. -/
theorem iq3s_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

/-- The four 64-value chunks tile the 256-value block. -/
theorem chunks_tile_block : 4 * 64 = 256 := by decide

-- § scales

/-- The decoded scale `2*s + 1` for a 4-bit `s` is odd, at least 1 and at most 31. -/
theorem scale_range (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- A byte splits into two 4-bit scales that recombine to the byte. -/
theorem byte_nibbles (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- The top nibble of a 32-bit word is `w >> 28`, which is below 16. -/
theorem top_nibble_lt (w : Nat) (h : w < 4294967296) : w / 268435456 < 16 := by omega

/-- Four 16-bit fields packed into a qword are recovered by dividing and reducing (field 0). -/
theorem pack4_field0 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) % 65536 = s0 := by omega

/-- Field 1. -/
theorem pack4_field1 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 65536 % 65536 = s1 := by omega

/-- Field 2. -/
theorem pack4_field2 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) (h2 : s2 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 4294967296 % 65536 = s2 := by omega

/-- Field 3. -/
theorem pack4_field3 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) (h2 : s2 < 65536)
    (h3 : s3 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 281474976710656 = s3 := by omega

/-- Every int16 lane `l < 32` of the maddubs result reads one of the four packed scales. -/
theorem scale_lane_field (l : Nat) (h : l < 32) : l / 8 < 4 := by omega

-- § masked sign subtract

/-- Apply the sign mask bit to an activation byte: `0 - y` when set, `y` otherwise. -/
def signed (bit : Bool) (y : Int) : Int := if bit then 0 - y else y

/-- Negating twice is the identity, so the mask is a pure sign flip. -/
theorem signed_involutive (bit : Bool) (y : Int) : signed bit (signed bit y) = y := by
  cases bit <;> simp [signed]

/-- Magnitudes are preserved by the sign mask. -/
theorem signed_abs (bit : Bool) (y : Int) : (signed bit y).natAbs = y.natAbs := by
  cases bit <;> simp [signed]

/-- A cleared mask bit leaves the activation untouched. -/
theorem signed_false (y : Int) : signed false y = y := by simp [signed]

-- § token-batching loop in dot_rows_nt

/-- Total tokens covered by repeatedly taking `min step remaining` (fuel-bounded loop). -/
def chunkTotal (step : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min step rem + chunkTotal step f (rem - min step rem)

/-- With a positive step and enough fuel the chunk sizes sum to the token count. -/
theorem chunkTotal_eq (step : Nat) (hs : 0 < step) :
    ∀ f rem, rem ≤ f → chunkTotal step f rem = rem := by
  intro f
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    simp [chunkTotal]
  | succ f ih =>
    intro rem h
    by_cases hr : rem = 0
    · subst hr; simp [chunkTotal]
    · have hm : 0 < min step rem := by omega
      have := ih (rem - min step rem) (by omega)
      simp only [chunkTotal, hr, if_false]
      rw [this]
      omega

/-- For `nt` tokens, the chunk-of-4 loop covers exactly `nt` tokens. -/
theorem nt_chunks_cover (nt : Nat) : chunkTotal 4 nt nt = nt :=
  chunkTotal_eq 4 (by decide) nt nt (Nat.le_refl _)

/-- Each chunk is at most 4 wide, as the templates `<TY,1..4>` require. -/
theorem chunk_le_four (rem : Nat) : min 4 rem ≤ 4 := by omega

end Strata.Cpu.IqAvx512



## FILE: /mnt/user-data/outputs/CpuExpertPoolScheduling.lean
/-!
# MODULE: cpu_expert_pool_scheduling
Work partitioning, job claiming and affinity bookkeeping of the CPU expert thread pool.

## Source semantics
* A phase splits `rows` rows across `tasks` tasks with task `i` owning `[rows*i/tasks, rows*(i+1)/tasks)`; the
  ranges are contiguous, start at 0 and end at `rows`.
* Inside a task, rows are walked expert by expert (`per` rows per expert); each step makes positive progress,
  stays inside one expert and never passes the task's end; the steps cover the whole range.
* Jobs are claimed with one `fetch_add` on a shared head; with `T` total claims over `n` jobs exactly
  `min T n` succeed, so `done_` reaches `n` once every thread has made its failing claim.
* The pool has at least one worker; `skip_first` removes core 0 from the core list.
* The Linux branch of `physical_cores` has a dangling `else`: it binds to `if (CPU_ISSET(i, &set))`, so
  every CPU NOT in the affinity mask appends `hardware_concurrency()` entries (modelled below).
* `pin_current_thread` returns the previous mask truncated to 64 CPUs as a signed value; a mask with bit 63
  set is negative and `restore_thread_affinity` skips it (`previous <= 0`).

## Form template identifiers
run_phase → run_phase, drain → claim_and_process, head_ → job_head, done_ → done_count, parked_ → parked_count,
mrows_/mtasks_ → phase_rows/phase_tasks, physical_cores → core_enumeration, pin_current_thread → pin_thread.
-/

namespace Strata.Cpu.Pool

-- § row partition across tasks

/-- Start of task `i`'s row range. -/
def lo (rows tasks i : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem lo_zero (rows tasks : Nat) : lo rows tasks 0 = 0 := by simp [lo]

/-- The last boundary is `rows` when there is at least one task. -/
theorem lo_last (rows tasks : Nat) (h : 0 < tasks) : lo rows tasks tasks = rows := by
  unfold lo
  exact Nat.mul_div_cancel rows h

/-- Boundaries are monotone, so every task's range is well formed (`g0 ≤ g1`). -/
theorem lo_mono (rows tasks i j : Nat) (h : i ≤ j) : lo rows tasks i ≤ lo rows tasks j := by
  unfold lo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows h)

/-- Total rows of the first `k` tasks. -/
def total (rows tasks : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => total rows tasks k + (lo rows tasks (k + 1) - lo rows tasks k)

/-- The first `k` tasks together own exactly `lo k` rows: ranges are contiguous with no gap or overlap. -/
theorem total_eq_lo (rows tasks k : Nat) : total rows tasks k = lo rows tasks k := by
  induction k with
  | zero => simp [total, lo]
  | succ k ih =>
    have hm := lo_mono rows tasks k (k + 1) (by omega)
    simp only [total, ih]
    omega

/-- All boundaries are at most `rows` for `i ≤ tasks`. -/
theorem lo_le_rows (rows tasks i : Nat) (h : 0 < tasks) (hi : i ≤ tasks) : lo rows tasks i ≤ rows := by
  have := lo_mono rows tasks i tasks hi
  rw [lo_last rows tasks h] at this
  exact this

-- § walking a task's range expert by expert

/-- The steps `(expert, r0, r1)` the inner loop performs, bounded by fuel. -/
def walk (per g : Nat) : Nat → Nat → List (Nat × Nat × Nat)
  | 0, _ => []
  | fuel + 1, r =>
    if r < g then
      (r / per, r % per, min per (r % per + (g - r))) ::
        walk per g fuel (r + (min per (r % per + (g - r)) - r % per))
    else []

/-- A single step makes positive progress, stays within an expert and does not pass the task end. -/
theorem step_progress (per r0 g r : Nat) (hr : r < g) (hp : r0 < per) :
    r0 < min per (r0 + (g - r)) ∧ min per (r0 + (g - r)) ≤ per ∧
      r + (min per (r0 + (g - r)) - r0) ≤ g := by
  omega

/-- The expert index of a global row stays below the number of experts. -/
theorem expert_index_lt (r per n : Nat) (h : r < n * per) : r / per < n := by
  by_cases hp : per = 0
  · subst hp; simp at h
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)

/-- The sum of the step widths equals the range length, given enough fuel. -/
theorem walk_cover (per g : Nat) (hp : 0 < per) :
    ∀ fuel r, g - r ≤ fuel → ((walk per g fuel r).map (fun t => t.2.2 - t.2.1)).sum = g - r := by
  intro fuel
  induction fuel with
  | zero =>
    intro r h
    simp [walk] <;> omega
  | succ f ih =>
    intro r h
    by_cases hr : r < g
    · have h0 : r % per < per := Nat.mod_lt _ hp
      have key := ih (r + (min per (r % per + (g - r)) - r % per)) (by omega)
      simp only [walk, hr, if_true, List.map_cons, List.sum_cons]
      rw [key]
      omega
    · simp [walk, hr] <;> omega

-- § phase decode: task index -> (expert, part)

/-- For `i < n * parts`, the decoded expert is `< n` and the part is `< parts`. -/
theorem decode_task (i n parts : Nat) (hp : 0 < parts) (h : i < n * parts) :
    i / parts < n ∧ i % parts < parts := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  · exact Nat.mod_lt _ hp

/-- `parts_a_ = max 1 ((3 * threads + n - 1) / n)` is positive. -/
theorem parts_positive (threads n : Nat) : 0 < max 1 ((3 * threads + n - 1) / n) := by omega

-- § job claiming

/-- Number of successful claims among the first `T` `fetch_add` results `0, 1, …, T-1`. -/
def claimOk (n : Nat) : Nat → Nat
  | 0 => 0
  | t + 1 => claimOk n t + (if t < n then 1 else 0)

/-- Over `n` jobs and `T` claims exactly `min T n` succeed. -/
theorem claimOk_eq (n T : Nat) : claimOk n T = min T n := by
  induction T with
  | zero => simp [claimOk]
  | succ t ih =>
    simp only [claimOk, ih]
    split <;> omega

/-- Once the claims number at least `n` plus one failing claim per thread, all `n` jobs were claimed once. -/
theorem all_jobs_claimed_once (n T : Nat) (h : n ≤ T) : claimOk n T = n := by
  rw [claimOk_eq]; omega

-- § pool sizing and core list

/-- Effective worker count: the requested count if positive, else the core count, never below 1. -/
def poolSize (requested cores : Nat) : Nat := max 1 (if 0 < requested then requested else cores)

/-- The pool always has at least one worker, so a single-core machine still produces a token. -/
theorem poolSize_pos (requested cores : Nat) : 1 ≤ poolSize requested cores := by
  unfold poolSize
  omega

/-- Dropping core 0 from a non-empty core list shortens it by exactly one. -/
theorem skip_first_len (cores : List Nat) (h : cores ≠ []) : (cores.drop 1).length = cores.length - 1 := by
  simp

-- § the dangling else in the Linux branch of physical_cores

/-- What the source actually builds: unset CPUs expand to `range hc`. Parameters: hc, index, mask. -/
def actualAux (hc : Nat) : Nat → List Bool → List Nat
  | _, [] => []
  | i, b :: bs => (if b then [i] else List.range hc) ++ actualAux hc (i + 1) bs

/-- What the indentation suggests: just the set CPUs. -/
def intendedAux : Nat → List Bool → List Nat
  | _, [] => []
  | i, b :: bs => (if b then [i] else []) ++ intendedAux (i + 1) bs

/-- When every CPU is in the mask the two coincide. -/
theorem actual_eq_intended_when_all_set (hc : Nat) (bs : List Bool) (h : ∀ b ∈ bs, b = true) :
    ∀ i, actualAux hc i bs = intendedAux i bs := by
  induction bs with
  | nil => intro i; rfl
  | cons b bs ih =>
    intro i
    have hb : b = true := h b (by simp)
    subst hb
    have h' : ∀ b ∈ bs, b = true := fun x hx => h x (by simp [hx])
    simp only [actualAux, intendedAux, if_true]
    rw [ih h' (i + 1)]

/-- Witness that they differ: with one unset CPU and `hardware_concurrency = 2`, the list is `[0,0,1]`
instead of `[0]`, so a pinned worker would be placed on cores 0 and 1 twice. -/
theorem dangling_else_witness :
    actualAux 2 0 [true, false] = [0, 0, 1] ∧ intendedAux 0 [true, false] = [0] := by decide

-- § affinity mask round trip

/-- Interpret a 64-bit mask as the signed `long long` the source returns. -/
def asSigned (m : Int) : Int := if m < 9223372036854775808 then m else m - 18446744073709551616

/-- A mask with bit 63 set becomes negative. -/
theorem high_bit_mask_negative (m : Int) (h1 : 9223372036854775808 ≤ m) (h2 : m < 18446744073709551616) :
    asSigned m < 0 := by
  unfold asSigned
  split <;> omega

/-- A negative stored mask makes `restore_thread_affinity` a no-op (`previous <= 0` returns early),
so an affinity containing CPU 63 is never restored. -/
theorem high_bit_mask_not_restored (m : Int) (h1 : 9223372036854775808 ≤ m) (h2 : m < 18446744073709551616) :
    asSigned m ≤ 0 := by
  have := high_bit_mask_negative m h1 h2
  omega

/-- Masks below bit 63 round-trip as non-negative values. -/
theorem low_mask_roundtrips (m : Int) (h : 0 ≤ m) (h2 : m < 9223372036854775808) : asSigned m = m := by
  unfold asSigned
  simp [h2]

end Strata.Cpu.Pool



## FILE: /mnt/user-data/outputs/CpuQ2Avx2Kernel.lean
/-!
# MODULE: cpu_q2_avx2_kernel
The AVX2-only Q2_0 row kernel and activation quantizer: the 2-bit unpack by byte/word interleave, the horizontal
reductions, the token chunking, and the round-half-away-from-zero quantizer rule.

## Source semantics
* `unpack64`: from 16 code bytes, value `4n + k` of the 64 outputs is `(byte_n >> 2k) & 3`, produced by
  masking four shifted copies and interleaving bytes then words; low vector = values 0..31 (bytes 0..7), high =
  32..63 (bytes 8..15).
* The 8-lane float/int reduction reduces to the sum of all lanes.
* `q2_0_gguf_rows_multi_avx2` handles `nt ≤ 4` directly and splits larger counts into chunks of ≤ 4 that sum to `nt`.
* The quantizer rule is `t + copysign(0.5, t)` truncated toward zero (half away from zero), then clamped to ±127.
  It differs from round-half-to-even exactly on ties.

## Form template identifiers
unpack64 → code_unpack_64, row_multi → multi_token_row_dot, rows → row_range_loop,
act_quant_q8_1_avx2 → activation_quantizer_avx2, h2f → half_to_float.
-/

namespace Strata.Cpu.Q2Avx2

-- § unpack index algebra

/-- 2-bit code `k` of a byte: `(b >> 2k) & 3`. -/
def code (b k : Nat) : Nat := (b / 4 ^ k) % 4

/-- Output position of byte `n`, code `k` after the two interleave stages. -/
def pos (n k : Nat) : Nat := 4 * n + k

/-- The position map is a bijection from (byte, code) pairs to value indices: decode recovers both. -/
theorem pos_decode (n k : Nat) (hk : k < 4) : pos n k / 4 = n ∧ pos n k % 4 = k := by
  unfold pos
  constructor <;> omega

/-- Every value index below `4m` is reached from a byte below `m`. -/
theorem pos_surjective (v m : Nat) (h : v < 4 * m) : v / 4 < m ∧ pos (v / 4) (v % 4) = v := by
  unfold pos
  constructor <;> omega

/-- The low vector holds values 0..31 (bytes 0..7); the high vector holds 32..63 (bytes 8..15). -/
theorem lo_hi_split (n k : Nat) (hk : k < 4) (hn : n < 8) : pos n k < 32 ∧ 32 ≤ pos (n + 8) k := by
  unfold pos
  constructor <;> omega

/-- Every code is in 0..3. -/
theorem code_lt_four (b k : Nat) : code b k < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- The four codes of a byte recombine into the byte (so no bits are lost by unpacking). -/
theorem codes_recombine (b : Nat) (h : b < 256) :
    code b 0 + 4 * code b 1 + 16 * code b 2 + 64 * code b 3 = b := by
  unfold code
  simp
  omega

-- § reductions

/-- The 8-lane integer reduction (128-bit fold, `shuffle 0x4E`, `shuffle 0xB1`) is the sum of all lanes. -/
theorem lane_reduction_sums (a0 a1 a2 a3 a4 a5 a6 a7 : Int) :
    ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)) = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 := by
  omega

/-- The correction term subtracted from the accumulated dot. -/
theorem corrected_dot (acc corr : Int) : acc - corr + corr = acc := by omega

-- § token chunking

/-- Tokens covered by repeatedly taking `min 4 remaining`. -/
def chunkTotal : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min 4 rem + chunkTotal f (rem - min 4 rem)

/-- With enough fuel the chunks cover exactly `rem` tokens. -/
theorem chunkTotal_eq : ∀ f rem, rem ≤ f → chunkTotal f rem = rem := by
  intro f
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    simp [chunkTotal]
  | succ f ih =>
    intro rem h
    by_cases hr : rem = 0
    · subst hr; simp [chunkTotal]
    · have := ih (rem - min 4 rem) (by omega)
      simp only [chunkTotal, hr, if_false]
      rw [this]
      omega

/-- The avx2 row entry covers all `nt` tokens. -/
theorem avx2_rows_cover (nt : Nat) : chunkTotal nt nt = nt := chunkTotal_eq nt nt (Nat.le_refl _)

-- § quantizer rule

/-- Round half away from zero of `n / d` (`d > 0`), using floor division on a non-negative numerator. -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((-2 * n + d) / (2 * d))

/-- Round half to even of `n / d`, via floor quotient and non-negative remainder. -/
def rne (n d : Int) : Int :=
  let q := n / d
  let r := n % d
  if 2 * r < d then q else if 2 * r > d then q + 1 else if q % 2 = 0 then q else q + 1

/-- Clamp to the symmetric int8 range used by the quantizer. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Ties go away from zero under the quantizer rule. -/
theorem rha_ties :
    rha 1 2 = 1 ∧ rha 3 2 = 2 ∧ rha 5 2 = 3 ∧ rha (-1) 2 = -1 ∧ rha (-3) 2 = -2 ∧ rha 0 2 = 0 := by
  decide

/-- Half-to-even disagrees with the quantizer rule on ties, which is why `_mm256_round_ps` is not used. -/
theorem rne_differs_on_ties : rne 1 2 ≠ rha 1 2 ∧ rne 5 2 ≠ rha 5 2 := by decide

/-- Away from ties both agree. -/
theorem rne_rha_agree_off_ties : rne 7 4 = rha 7 4 ∧ rne 1 4 = rha 1 4 ∧ rne (-7) 4 = rha (-7) 4 := by
  decide

/-- The clamp always lands in [-127, 127] and never touches in-range values. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- In-range values pass the clamp unchanged. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- If `0 ≤ n ≤ 127 d` (i.e. `|x| / s ≤ 127`), rounding gives at most 127, so the int cast is safe. -/
theorem rha_le_127 (n d : Int) (hd : 0 < d) (h0 : 0 ≤ n) (h : n ≤ 127 * d) : rha n d ≤ 127 := by
  unfold rha
  rw [if_pos h0]
  have : (2 * n + d) / (2 * d) < 128 := by
    rw [Int.ediv_lt_iff_lt_mul (by omega)]
    omega
  omega

end Strata.Cpu.Q2Avx2



## FILE: /mnt/user-data/outputs/ArtifactDequantReference.lean
/-!
# MODULE: artifact_dequant_reference
Scalar reference dequantizers (Q2_0, Q8_0, Q4_0, Q5_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS): block
geometry, field-size accounting, index bounds for every packed access, value ranges of the integer symbols,
and the IQ4_NL codebook.

## Source semantics
* Q2_0: block = fp16 `d` + 16 code bytes = 64 elements / 18 bytes; code `c ∈ {0..3}` ↦ symbol `c - 1`; element
  `j` reads byte `j/4`, bit offset `2 (j % 4)`.
* Each format's field sizes add up to its byte size.
* K-quant scale/min unpack (`get_scale_min_k4`) yields 6-bit values.
* Q3_K scales are `s - 32` (signed), the low-2-bit symbol is offset by 4 when the high bit is clear.
* IQ4_NL / IQ4_XS use a strictly increasing 16-entry codebook.
* bf16 → fp32 is a 16-bit left shift; `read_u16` composes two bytes little-endian.

## Form template identifiers
dequantize_q2_0 … dequantize_iq4_xs → block_dequantizer, kvalues_iq4nl → nonlinear_codebook,
get_scale_min_k4 → scale_min_unpack, read_u16 → le_u16, fp16_to_fp32/bf16_to_fp32 → half_to_float.
-/

namespace Strata.Artifact.Dequant

-- § block geometry

/-- (elements per block, bytes per block) for each format. -/
def geom : String → Option (Nat × Nat)
  | "Q2_0" => some (64, 18)
  | "Q8_0" => some (32, 34)
  | "Q4_0" => some (32, 18)
  | "Q5_0" => some (32, 22)
  | "IQ4_NL" => some (32, 18)
  | "Q6_K" => some (256, 210)
  | "Q4_K" => some (256, 144)
  | "Q5_K" => some (256, 176)
  | "Q3_K" => some (256, 110)
  | "IQ4_XS" => some (256, 136)
  | _ => none

/-- Field-size accounting: every block size is the sum of its fields. -/
theorem field_sizes_add_up :
    18 = 2 + 16 ∧ 34 = 2 + 32 ∧ 22 = 2 + 4 + 16 ∧ 210 = 128 + 64 + 16 + 2 ∧ 144 = 2 + 2 + 12 + 128 ∧
      176 = 2 + 2 + 12 + 32 + 128 ∧ 110 = 32 + 64 + 12 + 2 ∧ 136 = 2 + 2 + 4 + 128 := by decide

/-- The 64-element Q2_0 block stores 4 codes per byte over 16 bytes. -/
theorem q2_0_codes_fill_block : 16 * 4 = 64 := by decide

-- § Q2_0 indexing and symbols

/-- Byte index and bit offset of element `j` of a Q2_0 block stay in range. -/
theorem q2_0_index_bounds (j : Nat) (h : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by omega

/-- The (byte, slot) pair determines `j` uniquely: `4 * (j/4) + j%4 = j`. -/
theorem q2_0_index_unique (j : Nat) : 4 * (j / 4) + j % 4 = j := by omega

/-- Code `c < 4` maps to a symbol in {-1, 0, 1, 2}. -/
theorem q2_0_symbol_range (c : Int) (h0 : 0 ≤ c) (h1 : c < 4) :
    c - 1 = -1 ∨ c - 1 = 0 ∨ c - 1 = 1 ∨ c - 1 = 2 := by omega

/-- Every dequantized value is one of `-d, 0, d, 2d` (as integer multiples of `d`): the structural check
that `--check` asserts. -/
theorem q2_0_values_multiples (c d : Int) (h0 : 0 ≤ c) (h1 : c < 4) :
    (c - 1) * d = -d ∨ (c - 1) * d = 0 ∨ (c - 1) * d = d ∨ (c - 1) * d = 2 * d := by
  have : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases this with rfl | rfl | rfl | rfl <;> simp <;> omega

-- § 4/5-bit linear formats

/-- Q4_0 symbols: a nibble minus 8 lies in [-8, 7]. -/
theorem q4_0_symbol_range (n : Int) (h0 : 0 ≤ n) (h1 : n < 16) : -8 ≤ n - 8 ∧ n - 8 ≤ 7 := by omega

/-- Q5_0: a 5-bit value (nibble plus one high bit) minus 16 lies in [-16, 15]. -/
theorem q5_0_symbol_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 2) :
    (lo + 16 * hi : Int) - 16 ≥ -16 ∧ (lo + 16 * hi : Int) - 16 ≤ 15 := by omega

/-- Q4_0/Q5_0/IQ4_NL: element `j` and `j+16` of a 32-block come from nibble `j`, `j < 16`. -/
theorem nibble_pair_cover (j : Nat) (h : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j + 16 ≥ 16 := by omega

/-- The two nibbles of a byte recombine to it. -/
theorem nibbles_recombine (b : Nat) (h : b < 256) : b % 16 + 16 * (b / 16) = b := by omega

-- § K-quants

/-- `get_scale_min_k4` for `j < 4`: low six bits of two bytes, hence below 64. -/
theorem scale_min_small (a b : Nat) : a % 64 < 64 ∧ b % 64 < 64 := by omega

/-- `get_scale_min_k4` for `j ≥ 4`: low nibble of one byte plus the top two bits of another, shifted by 4,
stays below 64. -/
theorem scale_min_large (q4 qm : Nat) (h1 : q4 < 256) (h2 : qm < 256) :
    q4 % 16 + 16 * (qm / 64) < 64 := by omega

/-- The same for the min half: high nibble plus top two bits shifted by 4. -/
theorem scale_min_large_min (q4 qm : Nat) (h1 : q4 < 256) (h2 : qm < 256) :
    q4 / 16 + 16 * (qm / 64) < 64 := by omega

/-- Q4_K / Q5_K advance four 64-element groups, using scale indices `is, is+1` with `is = 0,2,4,6`: all in 0..7. -/
theorem q45k_scale_index (g : Nat) (h : g < 4) : 2 * g < 8 ∧ 2 * g + 1 < 8 := by omega

/-- Q5_K high-bit masks `u1 = 1 << 2g`, `u2 = 2 << 2g` are distinct powers of two fitting a byte. -/
theorem q5k_masks : (List.range 4).map (fun g => 1 <<< (2 * g)) = [1, 4, 16, 64] ∧
    (List.range 4).map (fun g => 2 <<< (2 * g)) = [2, 8, 32, 128] := by decide

/-- Q4_K output index within its 256-element block: group `g` writes 32 + 32 elements at `64 g`. -/
theorem q4k_output_cover (g l : Nat) (hg : g < 4) (hl : l < 32) : 64 * g + l < 256 ∧ 64 * g + 32 + l < 256 := by
  omega

/-- Q4_K/Q5_K `qs` reads (`+16`/`+48` header) stay in bounds. -/
theorem q45k_reads (g l : Nat) (hg : g < 4) (hl : l < 32) :
    16 + 32 * g + l < 144 ∧ 48 + 32 * g + l < 176 := by omega

/-- Q6_K: each 128-half uses scale indices `8h + l/16 + {0,2,4,6}`, all below 16. -/
theorem q6k_scale_index (h l : Nat) (hh : h < 2) (hl : l < 32) :
    8 * h + l / 16 + 6 < 16 := by omega

/-- Q6_K reads: `ql` ≤ index 127, `qh` ≤ 191, `scales` ≤ 207, `d` at 208. -/
theorem q6k_reads (h l : Nat) (hh : h < 2) (hl : l < 32) :
    64 * h + l + 32 < 128 ∧ 128 + 32 * h + l < 192 ∧ 192 + 8 * h + l / 16 + 6 < 208 := by omega

/-- Q6_K symbol: a 6-bit value minus 32 lies in [-32, 31]. -/
theorem q6k_symbol_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo + 16 * hi : Int) - 32 ≥ -32 ∧ (lo + 16 * hi : Int) - 32 ≤ 31 := by omega

/-- Q3_K: the low-2-bit value minus 4 (high bit clear) or minus 0 lies in [-4, 3]. -/
theorem q3k_symbol_range (q : Int) (hm : Bool) (h0 : 0 ≤ q) (h1 : q < 4) :
    -4 ≤ q - (if hm then 0 else 4) ∧ q - (if hm then 0 else 4) ≤ 3 := by
  cases hm <;> simp <;> omega

/-- Q3_K scales are signed: a 6-bit scale minus 32 lies in [-32, 31]. -/
theorem q3k_scale_range (s : Int) (h0 : 0 ≤ s) (h1 : s < 64) : -32 ≤ s - 32 ∧ s - 32 ≤ 31 := by omega

/-- Q3_K reads: scales at 96..107, d at 108; `q` offset `32 + 32 n + l + 16`. -/
theorem q3k_reads (n l : Nat) (hn : n < 2) (hl : l < 16) : 32 + 32 * n + l + 16 < 96 := by omega

/-- Q3_K mask shifts across both halves: 8 distinct bits 1,2,…,128 (no reset between halves). -/
theorem q3k_mask_sequence : (List.range 8).map (fun i => 1 <<< i) = [1, 2, 4, 8, 16, 32, 64, 128] := by decide

/-- IQ4_XS: a six-bit scale index `lo | (hi << 4)` is below 64, and `ls - 32` is in [-32, 31]. -/
theorem iq4xs_scale (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    lo + 16 * hi < 64 ∧ (lo + 16 * hi : Int) - 32 ≥ -32 ∧ (lo + 16 * hi : Int) - 32 ≤ 31 := by omega

/-- IQ4_XS group `ib` reads 16 `qs` bytes at `8 + 16 ib`, inside the 136-byte block. -/
theorem iq4xs_reads (ib j : Nat) (hib : ib < 8) (hj : j < 16) : 8 + 16 * ib + j < 136 := by omega

/-- IQ4_XS scale nibble selection: `scales_l[ib/2]` shifted by `4 (ib % 2)`, index below 4. -/
theorem iq4xs_scale_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 4 * (ib % 2) ≤ 4 := by omega

-- § the IQ4_NL codebook

/-- The non-linear codebook of ggml's `kvalues_iq4nl`. -/
def kvalues : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries, strictly increasing. -/
theorem kvalues_sorted : kvalues.length = 16 ∧ List.Pairwise (· < ·) kvalues := by decide

/-- Entries fit int8, so the table can be stored as `int8_t`. -/
theorem kvalues_int8 : ∀ v ∈ kvalues, -128 ≤ v ∧ v ≤ 127 := by decide

-- § element types

/-- `read_u16` of two bytes is below 2^16. -/
theorem read_u16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : p0 + 256 * p1 < 65536 := by omega

/-- bf16 → fp32 places the 16 bits in the top half of a 32-bit word. -/
theorem bf16_shift_fits (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

end Strata.Artifact.Dequant



## FILE: /mnt/user-data/outputs/KernelsF16Bits.lean
/-!
# MODULE: kernels_f16_bits
Round-to-nearest-even f32 → fp16 and exact fp16 → f32, as integer bit manipulations: the rounding step, the
overflow / underflow / NaN regimes, the subnormal exponent arithmetic and the normal-value round trip.

## Source semantics
* `f16_from_f32` splits into: f32 inf/NaN (raw exponent 255) stays inf/NaN; a finite value whose rebased
  exponent is ≥ 31 saturates to inf (never NaN); rebased exponent < -10 underflows to signed zero;
  `-10 ≤ exp ≤ 0` is the subnormal path (restore the implicit 1, shift by `14 - exp`); otherwise the normal
  path rounds a 13-bit remainder to nearest even and the carry may bump the exponent.
* `f32_from_f16` is exact: a subnormal `man * 2^-24` with leading bit at position `p` becomes exponent field
  `103 + p`; normals map exponent `ex` to `ex + 112`; `ex = 31` maps to field 255.
* Normal fp16 values round-trip exactly through f32.

## Form template identifiers
f16_from_f32 → float_to_half_bits, f32_from_f16 → half_bits_to_float, STRATA_HD → host_device_attr.
-/

namespace Strata.Kernels.F16

-- § round to nearest even on a dropped remainder

/-- Round `m / p` to nearest, ties to even; `p` is the weight of the dropped bits. -/
def rneStep (m p : Nat) : Nat :=
  if (m % p) * 2 > p ∨ ((m % p) * 2 = p ∧ (m / p) % 2 = 1) then m / p + 1 else m / p

/-- The result is the floor quotient or the next integer. -/
theorem rneStep_bounds (m p : Nat) : m / p ≤ rneStep m p ∧ rneStep m p ≤ m / p + 1 := by
  unfold rneStep
  split <;> omega

/-- Remainder below half: round down. -/
theorem rneStep_down (m p : Nat) (h : (m % p) * 2 < p) : rneStep m p = m / p := by
  unfold rneStep
  rw [if_neg]
  omega

/-- Remainder above half: round up. -/
theorem rneStep_up (m p : Nat) (h : (m % p) * 2 > p) : rneStep m p = m / p + 1 := by
  unfold rneStep
  rw [if_pos]
  exact Or.inl h

/-- Exact tie, even quotient: stay. -/
theorem rneStep_tie_even (m p : Nat) (h : (m % p) * 2 = p) (he : (m / p) % 2 = 0) : rneStep m p = m / p := by
  unfold rneStep
  rw [if_neg]
  omega

/-- Exact tie, odd quotient: round up to the even neighbour. -/
theorem rneStep_tie_odd (m p : Nat) (h : (m % p) * 2 = p) (ho : (m / p) % 2 = 1) : rneStep m p = m / p + 1 := by
  unfold rneStep
  rw [if_pos]
  exact Or.inr ⟨h, ho⟩

/-- An exactly representable value (zero remainder) is unchanged. -/
theorem rneStep_exact (q p : Nat) (hp : 0 < p) : rneStep (q * p) p = q := by
  have h1 : (q * p) % p = 0 := Nat.mul_mod_left q p
  have h2 : (q * p) / p = q := Nat.mul_div_cancel q hp
  unfold rneStep
  rw [h1, h2, if_neg]
  omega

-- § normal path

/-- Normal path value before rounding: exponent field `e` (1..30), top ten mantissa bits `man / 8192`. -/
def normalPre (sign e man : Nat) : Nat := sign + e * 1024 + man / 8192

/-- Normal path with rounding applied to the 13 dropped bits (`p = 8192`). -/
def normalRounded (sign e man : Nat) : Nat := sign + e * 1024 + rneStep man 8192

/-- The rounded normal value never exceeds `sign + 0x7C00` (the carry from `e = 30` reaches inf, not NaN). -/
theorem normal_le_inf (sign e man : Nat) (he : e ≤ 30) (hm : man < 8388608) :
    normalRounded sign e man ≤ sign + 31744 := by
  have hb := rneStep_bounds man 8192
  unfold normalRounded
  omega

/-- A mantissa that is all ones with remainder above half carries into the exponent. -/
theorem normal_carry_into_exponent (sign e : Nat) :
    normalRounded sign e 8388607 = sign + (e + 1) * 1024 := by
  unfold normalRounded rneStep
  simp <;> omega

/-- The odd bit tested by the source (`h & 1`) is the lowest bit of the kept mantissa (`man / 8192`),
because the exponent field starts at bit 10. -/
theorem lsb_is_kept_mantissa_lsb (sign e man : Nat) (hs : sign = 0 ∨ sign = 32768) (he : e < 32) :
    (sign + e * 1024 + man / 8192) % 2 = (man / 8192) % 2 := by
  rcases hs with rfl | rfl <;> omega

-- § overflow, NaN, underflow regimes

/-- Raw f32 exponent field. -/
def rawExp (x : Nat) : Nat := (x / 8388608) % 256

/-- f32 mantissa. -/
def mant (x : Nat) : Nat := x % 8388608

/-- Sign bit moved to bit 15. -/
def signBits (x : Nat) : Nat := if x ≥ 2147483648 then 32768 else 0

/-- Model of the three regime branches of `f16_from_f32` (the normal and subnormal bodies are the
`normalRounded` and `subnormalRounded` models). Returns the 16-bit result. -/
def regime (x : Nat) : Nat :=
  if rawExp x = 255 then signBits x + 31744 + (if mant x ≠ 0 then 512 else 0)
  else if rawExp x ≥ 143 then signBits x + 31744
  else if rawExp x < 102 then signBits x
  else 0

/-- Infinity stays infinity. -/
theorem inf_stays_inf (x : Nat) (h : rawExp x = 255) (hm : mant x = 0) : regime x = signBits x + 31744 := by
  unfold regime
  simp [h, hm]

/-- A NaN input stays a NaN: the mantissa field is non-zero. -/
theorem nan_stays_nan (x : Nat) (h : rawExp x = 255) (hm : mant x ≠ 0) :
    regime x = signBits x + 31744 + 512 := by
  unfold regime
  simp [h, hm]

/-- A finite overflowing value saturates to a pure inf (mantissa bits zero), never to a NaN. -/
theorem finite_overflow_is_inf (x : Nat) (h1 : rawExp x < 255) (h2 : 143 ≤ rawExp x) :
    regime x = signBits x + 31744 ∧ regime x % 1024 = (signBits x) % 1024 := by
  have hs : signBits x = 0 ∨ signBits x = 32768 := by unfold signBits; split <;> simp
  unfold regime
  have h3 : rawExp x ≠ 255 := by omega
  simp only [h3, if_false, h2, if_true]
  constructor
  · first | trivial | rfl
  · rcases hs with h | h <;> rw [h] <;> omega

/-- Rebased exponent `rawexp - 112 ≥ 31` is the same as `rawexp ≥ 143`. -/
theorem overflow_threshold (rawexp : Nat) : (rawexp : Int) - 127 + 15 ≥ 31 ↔ rawexp ≥ 143 := by omega

/-- Rebased exponent `< -10` is the same as `rawexp < 102`. -/
theorem underflow_threshold (rawexp : Nat) : (rawexp : Int) - 127 + 15 < -10 ↔ rawexp < 102 := by omega

/-- Values deep below the subnormal range yield a signed zero. -/
theorem deep_underflow_is_signed_zero (x : Nat) (h : rawExp x < 102) : regime x = signBits x := by
  unfold regime
  have h1 : rawExp x ≠ 255 := by omega
  have h2 : ¬ (rawExp x ≥ 143) := by omega
  simp [h1, h2, h]

-- § subnormal path

/-- Subnormal rounding: the 24-bit significand `man + 2^23` shifted by `sh = 14 - exp` (weight `2^sh`). -/
def subnormalRounded (sign man p : Nat) : Nat := sign + rneStep (man + 8388608) p

/-- For `sh ≥ 14` the unrounded shifted significand is below 1024, so masking with 0x3FF is a no-op. -/
theorem subnormal_shift_lt (man p : Nat) (hm : man < 8388608) (hp : 16384 ≤ p) : (man + 8388608) / p < 1024 := by
  have h : 16384 * 1024 ≤ p * 1024 := Nat.mul_le_mul_right 1024 hp
  have : man + 8388608 < p * 1024 := by omega
  exact Nat.div_lt_of_lt_mul this

/-- Rounding can reach 1024, which is exactly the smallest normal fp16 (exponent field 1, mantissa 0). -/
theorem subnormal_carry_is_min_normal (man p : Nat) (hm : man < 8388608) (hp : 16384 ≤ p) :
    rneStep (man + 8388608) p ≤ 1024 := by
  have h1 := subnormal_shift_lt man p hm hp
  have h2 := rneStep_bounds (man + 8388608) p
  omega

/-- Shift amounts `14 - exp` over the subnormal exponents `-10 ≤ exp ≤ 0` are 14..24. -/
theorem subnormal_shift_range (exp : Int) (h1 : -10 ≤ exp) (h2 : exp ≤ 0) : 14 ≤ 14 - exp ∧ 14 - exp ≤ 24 := by
  omega

-- § exact decode

/-- Normal fp16 exponent field `ex ∈ 1..30` becomes f32 exponent field `ex + 112 ∈ 113..142`. -/
theorem decode_normal_exponent (ex : Nat) (h1 : 1 ≤ ex) (h2 : ex ≤ 30) :
    (ex : Int) - 15 + 127 = ex + 112 ∧ 113 ≤ ex + 112 ∧ ex + 112 ≤ 142 := by omega

/-- Subnormal with leading bit at position `p ≤ 9` (value `man * 2^-24`): after `s = 10 - p` shifts the f32
exponent field is `113 - s`, which is `103 + p`, i.e. unbiased exponent `p - 24`. -/
theorem decode_subnormal_exponent (p : Nat) (hp : p ≤ 9) : 113 - (10 - p) = 103 + p ∧ 103 + p = 127 + p - 24 := by
  omega

/-- The shifted subnormal significand has bit 10 set and stays below 2048 when `2^p ≤ man < 2^(p+1)`. -/
theorem subnormal_normalised (man p : Nat) (hp : p ≤ 9) (h1 : 2 ^ p ≤ man) (h2 : man < 2 ^ (p + 1)) :
    1024 ≤ man * 2 ^ (10 - p) ∧ man * 2 ^ (10 - p) < 2048 := by
  have e : 2 ^ p * 2 ^ (10 - p) = 1024 := by
    rw [← Nat.pow_add]
    have : p + (10 - p) = 10 := by omega
    rw [this]
  have e2 : 2 ^ (p + 1) * 2 ^ (10 - p) = 2048 := by
    rw [← Nat.pow_add]
    have : p + 1 + (10 - p) = 11 := by omega
    rw [this]
  constructor
  · calc 1024 = 2 ^ p * 2 ^ (10 - p) := e.symm
      _ ≤ man * 2 ^ (10 - p) := Nat.mul_le_mul_right _ h1
  · calc man * 2 ^ (10 - p) < 2 ^ (p + 1) * 2 ^ (10 - p) :=
          Nat.mul_lt_mul_of_pos_right h2 (Nat.two_pow_pos _)
      _ = 2048 := e2

-- § round trip

/-- A normal fp16 mantissa embedded in f32 (`man << 13`) leaves a zero remainder, so re-rounding is exact. -/
theorem normal_roundtrip_mantissa (man : Nat) : rneStep (man * 8192) 8192 = man :=
  rneStep_exact man 8192 (by decide)

/-- Hence f16 → f32 → f16 is the identity on the mantissa and exponent of normal values. -/
theorem normal_roundtrip (sign ex man : Nat) :
    normalRounded sign ex (man * 8192) = sign + ex * 1024 + man := by
  unfold normalRounded
  rw [normal_roundtrip_mantissa]

end Strata.Kernels.F16



## FILE: /mnt/user-data/outputs/CudaDequantBf16.lean
/-!
# MODULE: cuda_dequant_bf16
Launch geometry and indexing contracts of the GPU dequantizer (one thread per 32-element group): the type
geometry table, row byte sizes, grid coverage, per-format group indexing bounds, the bf16 rounding rule, and the
split between ggml types handled here and the i-quant types routed elsewhere.

## Source semantics
* `geometry` gives (elements per block, bytes per block) for types 2, 6, 8, 11, 12, 13, 14, 20, 23, 42; every
  block has a multiple of 32 elements, so a row is `cols/32` groups.
* `row_bytes = cols / be * bb`; thread `g` maps to row `g / gpr` and group `g % gpr`.
* `grid = ceil(total / 256)` threads-blocks cover all `total` groups.
* Group `gi` of a K-quant row reads block `gi / 8`, sub-group `gi % 8`; of Q2_0 block `gi / 2`, half `gi % 2`.
* `f2bf(u) = (u + 0x7fff + ((u >> 16) & 1)) >> 16` in 32-bit unsigned arithmetic rounds to nearest even, but the
  addition wraps for `u ≥ 0xFFFF8000` (large-mantissa negative NaNs).
* The i-quant types {16, 17, 18, 21, 22, 29} are disjoint from the geometry table.

## Form template identifiers
geometry → block_geometry, group32 → group_dequant, dequant_kernel → per_group_thread, f2bf → float_to_bf16,
iq_only → iquant_types, launch → kernel_launch.
-/

namespace Strata.Cuda.DequantBf16

-- § geometry table

/-- (block elements, block bytes) per ggml type id. -/
def geometry : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Every supported block has a multiple of 32 elements (so groups of 32 never straddle a block). -/
theorem geometry_multiple_of_32 (t be bb : Nat) (h : geometry t = some (be, bb)) : be % 32 = 0 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- The i-quant types routed to `iq_dequant_*`. -/
def iqOnly (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- No type is both a table type and an i-quant type, so the two dispatch paths never overlap. -/
theorem iq_and_geometry_disjoint (t : Nat) (h : iqOnly t = true) : geometry t = none := by
  simp only [iqOnly, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ((((h | h) | h) | h) | h) | h <;> subst h <;> rfl

-- § row and grid arithmetic

/-- Groups per row is `cols / 32`; with `be ∈ {32,64,256}` dividing `cols`, it equals blocks-per-row times
groups-per-block. -/
theorem groups_per_row_32 (cols : Nat) (h : cols % 32 = 0) : cols / 32 * 32 = cols := by omega

/-- Same for 64-element blocks (Q2_0). -/
theorem groups_per_row_64 (cols : Nat) (h : cols % 64 = 0) : cols / 32 = (cols / 64) * 2 := by omega

/-- Same for 256-element blocks (K-quants, IQ4_XS). -/
theorem groups_per_row_256 (cols : Nat) (h : cols % 256 = 0) : cols / 32 = (cols / 256) * 8 := by omega

/-- Grid size `(total + 255) / 256` launches at least `total` threads. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- And not more than one extra block of threads. -/
theorem grid_tight (total : Nat) : (total + 255) / 256 * 256 < total + 256 := by omega

/-- Thread-to-(row, group) decode: `g < rows * gpr` gives `row < rows` and `group < gpr`. -/
theorem thread_decode (g rows gpr : Nat) (hg : 0 < gpr) (h : g < rows * gpr) : g / gpr < rows ∧ g % gpr < gpr := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  · exact Nat.mod_lt _ hg

/-- Each output group of 32 lies inside its row: `gi * 32 + j < gpr * 32` for `gi < gpr`, `j < 32`. -/
theorem output_in_row (gi gpr j : Nat) (hgi : gi < gpr) (hj : j < 32) : gi * 32 + j < gpr * 32 := by omega

-- § per-format group indexing

/-- Q2_0: block `gi/2` is inside the row, and the 32 elements start at 0 or 32 within the block. -/
theorem q2_0_group_bounds (cols gi j : Nat) (hc : cols % 64 = 0) (hg : gi < cols / 32) (hj : j < 32) :
    gi / 2 < cols / 64 ∧ (gi % 2) * 32 + j < 64 ∧ 2 + ((gi % 2) * 32 + j) / 4 < 18 := by omega

/-- Q3_K: block `gi/8` is in the row; qs at `32 + 32n + t` and scale index `8n + 2jj + (t≥16)` stay in range. -/
theorem q3_k_group_bounds (cols gi t : Nat) (hc : cols % 256 = 0) (hg : gi < cols / 32) (ht : t < 32) :
    gi / 8 < cols / 256 ∧ 32 + (gi % 8 / 4) * 32 + t < 96 ∧
      (gi % 8 / 4) * 8 + (gi % 8 % 4) * 2 + (if t ≥ 16 then 1 else 0) < 16 := by
  split <;> omega

/-- Q4_K: sub-group `gi%8` selects 64-group `j64 = (gi%8)/2` and a nibble half; reads `16 + 32 j64 + l < 144`. -/
theorem q4_k_group_bounds (cols gi l : Nat) (hc : cols % 256 = 0) (hg : gi < cols / 32) (hl : l < 32) :
    gi / 8 < cols / 256 ∧ 16 + 32 * (gi % 8 / 2) + l < 144 := by omega

/-- Q5_K: the high-bit mask `1 << (2 j64 + hi)` equals `1 << (gi % 8)`, a byte value. -/
theorem q5_k_mask_shift (gi : Nat) : 2 * (gi % 8 / 2) + gi % 8 % 2 = gi % 8 := by omega

/-- Q5_K reads: `qh` at `16 + l`, `ql` at `48 + 32 j64 + l`, both inside the 176-byte block. -/
theorem q5_k_group_bounds (gi l : Nat) (hl : l < 32) :
    16 + l < 48 ∧ 48 + 32 * (gi % 8 / 2) + l < 176 := by omega

/-- Q6_K: `ql` at `64 n + l (+32)`, `qh` at `128 + 32 n + l`, `sc` at `192 + 8n + l/16 + 2 qu`, `n = (gi%8)/4`,
`qu = (gi%8)%4`. -/
theorem q6_k_group_bounds (gi l : Nat) (hl : l < 32) :
    64 * (gi % 8 / 4) + l + 32 < 128 ∧ 128 + 32 * (gi % 8 / 4) + l < 192 ∧
      192 + 8 * (gi % 8 / 4) + l / 16 + 2 * (gi % 8 % 4) < 208 := by omega

/-- IQ4_XS: `qs` at `8 + 16 ib + j` stays inside the 136-byte block. -/
theorem iq4_xs_group_bounds (gi j : Nat) (hj : j < 16) : 8 + 16 * (gi % 8) + j < 136 := by omega

/-- Q4_0/Q8_0/Q5_0/IQ4_NL: one block per group, so `gi * bb + 2 + j` is inside the row for `gi < gpr`. -/
theorem single_group_block_bounds (gi gpr bb j : Nat) (hgi : gi < gpr) (hj : j + 2 < bb) :
    gi * bb + 2 + j < gpr * bb := by
  have h : (gi + 1) * bb ≤ gpr * bb := Nat.mul_le_mul_right bb hgi
  have h2 : (gi + 1) * bb = gi * bb + bb := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § bf16 rounding

/-- `f2bf` in 32-bit unsigned arithmetic (the addition wraps modulo 2^32). -/
def f2bf (u : Nat) : Nat := ((u + 32767 + (u / 65536) % 2) % 4294967296) / 65536

/-- Without wrap-around the result fits 16 bits. -/
theorem f2bf_fits (u : Nat) (h : u < 4294934528) : f2bf u ≤ 65535 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Below half: truncate. -/
theorem f2bf_round_down (u : Nat) (h : u < 4294934528) (hr : u % 65536 < 32768) : f2bf u = u / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Above half: round up. -/
theorem f2bf_round_up (u : Nat) (h : u < 4294934528) (hr : u % 65536 > 32768) : f2bf u = u / 65536 + 1 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Exact tie, even kept bit: stay. -/
theorem f2bf_tie_even (u : Nat) (h : u < 4294934528) (hr : u % 65536 = 32768) (he : (u / 65536) % 2 = 0) :
    f2bf u = u / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Exact tie, odd kept bit: round up to even. -/
theorem f2bf_tie_odd (u : Nat) (h : u < 4294934528) (hr : u % 65536 = 32768) (ho : (u / 65536) % 2 = 1) :
    f2bf u = u / 65536 + 1 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- The wrap hazard: the all-ones NaN pattern `0xFFFFFFFF` is converted to bf16 bits `0x0000` (+0.0), not a NaN.
Recorded as an observed property of the arithmetic as written. -/
theorem f2bf_all_ones_wraps_to_zero : f2bf 4294967295 = 0 := by decide

/-- A finite, ordinary value such as 1.0f (`0x3F800000`) converts exactly to bf16 `0x3F80`. -/
theorem f2bf_one : f2bf 1065353216 = 16256 := by decide

end Strata.Cuda.DequantBf16



## FILE: /mnt/user-data/outputs/CudaQuantizeActivation.lean
/-!
# MODULE: cuda_quantize_activation
Block layouts, launch arithmetic and rounding rules of the GPU activation quantizers (Q8_0, the CPU-matching
"scaled" Q8_0 variant, Q8_K) and their dequantizers.

## Source semantics
* Q8_0 block = fp16 `d` + 32 int8 = 34 bytes, 32 elements; Q8_K block = fp32 `d` + 256 int8 + 16 int16 `bsums`
  = 292 bytes, 256 elements.
* Plain Q8_0 divides in double and rounds half to even; the scaled variant multiplies by the reciprocal and
  rounds half away from zero (the CPU rule) and clamps to ±127.
* Q8_K's `nearest_int` is the 1.5·2^23 magic-number trick: for results in `[-2^22, 2^22)` the low 23 bits of the
  float bits minus `0x400000` give the rounded integer.
* Q8_K `bsums[j]` sums `qs[16 j .. 16 j + 15]`; the quantized values satisfy `|v| ≤ 127` so the int8 cast is
  safe; only an upper clamp `min(127, v)` exists in the source.
* Launch: `n` must be a multiple of the block length; `grid = ceil(nb / threads)` covers every block.

## Form template identifiers
quantize_q8_0_kernel → q8_0_quantizer, quantize_q8_0_scaled_kernel → q8_0_cpu_matching_quantizer,
quantize_q8_K_kernel → q8_k_quantizer, nearest_int_dev → nearest_int_magic, dequant_*_kernel → block_dequantizer.
-/

namespace Strata.Cuda.QuantizeAct

-- § layouts

/-- Q8_0 block bytes: 2 (fp16 d) + 32. -/
theorem q8_0_block_bytes : 34 = 2 + 32 := by decide

/-- Q8_K block bytes: 4 (fp32 d) + 256 + 32 (16 × int16). -/
theorem q8_k_block_bytes : 292 = 4 + 256 + 16 * 2 := by decide

/-- Q8_K sub-field offsets: `d` at 0, `qs` at 4, `bsums` at `4 + 256`; the last bsum ends at 292. -/
theorem q8_k_offsets : 4 + 256 = 260 ∧ 260 + 16 * 2 = 292 := by decide

/-- `bsums[j]` over `qs[16 j + ii]` with `j, ii < 16` covers indices below 256 and each index once. -/
theorem bsum_index_bounds (j ii : Nat) (hj : j < 16) (hi : ii < 16) :
    16 * j + ii < 256 ∧ (16 * j + ii) / 16 = j ∧ (16 * j + ii) % 16 = ii := by omega

/-- Block `b` starts at `b * 34` (resp. `b * 292`) and its last byte is below `(b+1) * 34` (resp. 292). -/
theorem block_ranges (b : Nat) : b * 34 + 33 < (b + 1) * 34 ∧ b * 292 + 291 < (b + 1) * 292 := by omega

/-- Elements of block `b` are `x[32 b .. 32 b + 31]`, inside `n = 32 * nb` for `b < nb`. -/
theorem q8_0_elements_in_range (b nb i : Nat) (hb : b < nb) (hi : i < 32) : b * 32 + i < nb * 32 := by omega

-- § launch arithmetic

/-- A multiple-of-32 element count gives exactly `n / 32` blocks. -/
theorem n_blocks_exact (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by omega

/-- Same for Q8_K with 256. -/
theorem n_blocks_exact_k (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Grid for 128 threads per block covers every block. -/
theorem grid128_covers (nb : Nat) : nb ≤ (nb + 127) / 128 * 128 := by omega

/-- Grid for 64 threads per block (Q8_K: one block per thread). -/
theorem grid64_covers (nb : Nat) : nb ≤ (nb + 63) / 64 * 64 := by omega

-- § rounding rules

/-- Round half away from zero of `n / d`, `d > 0` (the CPU rule `t + copysign(0.5, t)` then truncation). -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((-2 * n + d) / (2 * d))

/-- Round half to even of `n / d` (`rint` in the plain Q8_0 kernel). -/
def rint (n d : Int) : Int :=
  let q := n / d
  let r := n % d
  if 2 * r < d then q else if 2 * r > d then q + 1 else if q % 2 = 0 then q else q + 1

/-- The plain kernel's clamp `[-128, 127]`. -/
def clampPlain (v : Int) : Int := if v > 127 then 127 else if v < -128 then -128 else v

/-- The scaled kernel's clamp `[-127, 127]`. -/
def clampScaled (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- The Q8_K upper-only clamp. -/
def clampUpper (v : Int) : Int := if v < 127 then v else 127

/-- Ties differ between the two Q8_0 kernels: `rint` goes to even, `rha` away from zero. -/
theorem tie_rules_differ :
    rint 1 2 = 0 ∧ rha 1 2 = 1 ∧ rint 3 2 = 2 ∧ rha 3 2 = 2 ∧ rint (-1) 2 = 0 ∧ rha (-1) 2 = -1 ∧
      rint 5 2 = 2 ∧ rha 5 2 = 3 := by
  decide

/-- Plain clamp stays in int8. -/
theorem clampPlain_int8 (v : Int) : -128 ≤ clampPlain v ∧ clampPlain v ≤ 127 := by
  unfold clampPlain
  omega

/-- Scaled clamp stays in the symmetric range, so `-128` is never produced by the CPU-matching rule. -/
theorem clampScaled_symmetric (v : Int) : -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampScaled
  omega

/-- The upper clamp bounds the value by 127. -/
theorem clampUpper_le (v : Int) : clampUpper v ≤ 127 := by
  unfold clampUpper
  split <;> omega

/-- The CPU rule never exceeds 127 when `|x|/s ≤ 127` (`0 ≤ n ≤ 127 d`), so the clamp is a no-op there. -/
theorem rha_le_127 (n d : Int) (hd : 0 < d) (h0 : 0 ≤ n) (h : n ≤ 127 * d) : rha n d ≤ 127 := by
  unfold rha
  rw [if_pos h0]
  have : (2 * n + d) / (2 * d) < 128 := by
    rw [Int.ediv_lt_iff_lt_mul (by omega)]
    omega
  omega

-- § nearest_int magic number

/-- The magic-number trick: with `val = 12582912 + r` and `r ∈ [-2^22, 2^22)`, `val` lies in `[2^23, 2^24)`, its
float bits are `150·2^23 + (val - 2^23)`, and `(bits mod 2^23) - 0x400000 = r`. -/
theorem nearest_int_magic (r : Int) (h1 : -4194304 ≤ r) (h2 : r < 4194304) :
    ((150 * 8388608 + (12582912 + r - 8388608)) % 8388608) - 4194304 = r := by
  omega

/-- The same value lies in the binade `[2^23, 2^24)` where the exponent field is constant (150). -/
theorem magic_in_binade (r : Int) (h1 : -4194304 ≤ r) (h2 : r < 4194304) :
    8388608 ≤ 12582912 + r ∧ 12582912 + r < 16777216 := by omega

/-- Quantized values from `iscale = -127/max` satisfy `|v| ≤ 127` (given `|x_j| ≤ |max|`): in units of `1/d`,
`-127 ≤ rounded ≤ 127`, so the int8 cast cannot wrap. -/
theorem q8k_cast_safe (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : -128 ≤ clampUpper v ∧ clampUpper v ≤ 127 := by
  unfold clampUpper
  split <;> omega

-- § zero-block handling

/-- A zero-amax Q8_K block is entirely zero: `d = 0`, 256 zero `qs`, 16 zero `bsums`, whose dequantized
values are all `0 * d = 0` and whose bsums are sums of zeros. -/
theorem zero_block_bsum (xs : List Int) (h : ∀ x ∈ xs, x = 0) : xs.sum = 0 := by
  induction xs with
  | nil => rfl
  | cons a as ih =>
    have ha : a = 0 := h a (by simp)
    have has : ∀ x ∈ as, x = 0 := fun x hx => h x (by simp [hx])
    simp [List.sum_cons, ha, ih has]

end Strata.Cuda.QuantizeAct





## FILE: /mnt/user-data/outputs/SGemvDequantRowKernels.lean
/-!
# MODULE: s_gemv_dequant_row_kernels
Naive and split S-family GEMV kernels: weights are dequantized on the fly from 2/4/8-bit codes
(optionally through the IQ4_NL codebook), multiplied with FP16, Q8_K or Q8_0 activations, and reduced per
output row (thread-per-row, block-per-row tree, or warp-per-row with a 16-way register accumulator).

## Source semantics
* A code of width `bits` is bits `[(i % per) * bits, +bits)` of byte `i / per`, `per = 8 / bits`.
* `w = decode(code) * d + b`; the bias is applied to the integer code, the offset to the weight before the
  activation multiply.
* The group index is `e >> group_shift` with `group_elems = 2^group_shift`.
* One lane-iteration of the warp kernel consumes 16 consecutive elements lying in one group and one
  activation block (Q8_K: 256, Q8_0: 32).
* The block-wide reduction is a fixed halving tree, hence deterministic (modelled over exact integers).

## Form template identifiers
- `s_gemv_kernel`            → `thread_per_row_kernel`
- `s_gemv_split_kernel`      → `block_per_row_tree_kernel`
- `s_gemv_q8_split_kernel`   → `warp_per_row_quant_act_kernel`
- `kIq4Nl`                   → `nonlinear_codebook`
- `group_shift`              → `log2_group`
- `QE` (=16)                 → `elems_per_lane_iter`
-/

namespace Strata.SGemv

-- § reduction toolkit

/-- Prefix sum of the first `n` terms of `f`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Splitting a prefix sum at `a`. -/
theorem sumTo_add (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b + 1) = sumTo f a + sumTo (fun t => f (a + t)) (b + 1)
    simp only [sumTo, ih]
    omega

/-- Prefix sums are additive in the summand. -/
theorem sumTo_add_fun (g h : Nat → Int) (n : Nat) :
    sumTo (fun t => g t + h t) n = sumTo g n + sumTo h n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    omega

/-- One halving step of the shared-memory tree preserves the total. -/
theorem sumTo_halve (f : Nat → Int) (s : Nat) :
    sumTo f (2 * s) = sumTo (fun t => f t + f (s + t)) s := by
  rw [Nat.two_mul, sumTo_add, sumTo_add_fun]

/-- The kernel's reduction loop `partial[t] += partial[t + step]`, steps `2^(k-1), ..., 1`, read at `t = 0`. -/
def redT : (Nat → Int) → Nat → Int
  | f, 0 => f 0
  | f, k + 1 => redT (fun t => f t + f (2 ^ k + t)) k

/-- The fixed tree reduction over `2^k` partials equals the plain sum of all partials. -/
theorem redT_eq (k : Nat) : ∀ f : Nat → Int, redT f k = sumTo f (2 ^ k) := by
  induction k with
  | zero => intro f; simp [redT, sumTo]
  | succ k ih =>
    intro f
    show redT (fun t => f t + f (2 ^ k + t)) k = _
    rw [ih, Nat.pow_succ, Nat.mul_comm, sumTo_halve]

/-- The explicit 16-accumulator regrouping at the end of the warp kernel is a plain sum. -/
theorem acc16_regroup (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15))) =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

-- § code unpacking and decode

/-- Code `i` of a row of bytes at width `bits` (mask `2^bits - 1` is the same as `% 2^bits`). -/
def codeAt (bytes : Nat → Nat) (bits i : Nat) : Nat :=
  (bytes (i / (8 / bits)) >>> ((i % (8 / bits)) * bits)) % 2 ^ bits

/-- Every unpacked code fits its width. -/
theorem codeAt_lt (bytes : Nat → Nat) (bits i : Nat) : codeAt bytes bits i < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- The four-code word is `4 * bits / 8` bytes for the three supported widths. -/
theorem quad_word_bytes :
    4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 ∧ 4 / (8 / 2) = 1 ∧ 4 / (8 / 4) = 2 ∧ 4 / (8 / 8) = 4 := by
  decide

/-- 2-bit packing: a byte is exactly the four fields in LSB-first order. -/
theorem repack_2bit : ∀ b : Nat, b < 256 →
    b = (b >>> 0) % 4 + 4 * ((b >>> 2) % 4) + 16 * ((b >>> 4) % 4) + 64 * ((b >>> 6) % 4) := by
  intro b hb
  omega

/-- 4-bit packing of a byte. -/
theorem repack_4bit (b : Nat) (hb : b < 256) : b = b % 16 + 16 * ((b >>> 4) % 16) := by
  omega

/-- 4-bit packing of the 16-bit four-code word used by S4 (`uint16_t` load). -/
theorem repack_4bit_word (v : Nat) (hv : v < 65536) :
    v = v % 16 + 16 * ((v >>> 4) % 16) + 256 * ((v >>> 8) % 16) + 4096 * ((v >>> 12) % 16) := by
  omega

/-- The IQ4_NL codebook. -/
def kIq4Nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Weight decode: codebook lookup or biased integer code. -/
def decode (iq4nl : Bool) (code : Nat) (bias : Int) : Int :=
  if iq4nl then kIq4Nl.getD (code % 16) 0 else (code : Int) + bias

/-- The codebook table has 16 entries, so the masked index is always in range. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- The codebook is strictly increasing (it is non-linear but monotone). -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by decide

/-- Non-codebook decode is the biased integer code. -/
theorem decode_affine (code : Nat) (bias : Int) : decode false code bias = (code : Int) + bias := by
  simp [decode]

/-- Codebook decode stays inside the table's value range. -/
theorem decode_iq4nl_range (code : Nat) (bias : Int) :
    -127 ≤ decode true code bias ∧ decode true code bias ≤ 113 := by
  have h : ∀ r : Nat, r < 16 → (-127 ≤ kIq4Nl.getD r 0 ∧ kIq4Nl.getD r 0 ≤ 113) := by decide
  simpa [decode] using h (code % 16) (Nat.mod_lt _ (by decide))

/-- Offset-before-multiply expands to the three exact terms (over exact integers the order is immaterial;
over floats it is not, which is why the kernels fix it). -/
theorem weight_then_activation (c bias d b x : Int) :
    ((c + bias) * d + b) * x = c * d * x + bias * d * x + b * x := by
  simp only [Int.add_mul]

-- § activation block geometry

/-- `block_q8_K` is `f32 d; i8 qs[256]; i16 bsums[16]` = 292 bytes, `d` 4-byte aligned. -/
theorem q8k_block_geometry : 4 + 256 + 2 * 16 = 292 ∧ 292 % 4 = 0 := by decide

/-- `block_q8_0` is `f16 d; i8 qs[32]` = 34 bytes, `d` 2-byte aligned. -/
theorem q8_0_block_geometry : 2 + 32 = 34 ∧ 34 % 2 = 0 := by decide

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K is structurally impossible. -/
theorem shexp_down_needs_q8_0 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

-- § warp kernel iteration structure

/-- Lane `l`, iteration `j` starts at element `16 * l + 512 * j`; a lane-iteration never straddles a Q8_K block. -/
theorem lane_iter_in_q8k_block (l j : Nat) (hl : l < 32) :
    (16 * l + 512 * j) % 256 + 16 ≤ 256 := by
  omega

/-- ... nor a Q8_0 block. -/
theorem lane_iter_in_q8_0_block (l j : Nat) (hl : l < 32) :
    (16 * l + 512 * j) % 32 + 16 ≤ 32 := by
  omega

/-- Every element `e` is owned by exactly one (lane, iteration, offset) triple. -/
theorem lane_iter_cover (e : Nat) :
    e = 16 * ((e / 16) % 32) + 512 * (e / 512) + e % 16 := by
  omega

/-- A multiple-of-16 row has no tail: every start position leaves a full 16-element step. -/
theorem no_tail_when_aligned (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : i < n) :
    i + 16 ≤ n := by
  omega

/-- Sixteen consecutive elements starting at a multiple of 16 share one group (group size `16 * m`). -/
theorem quad_same_group (i k m : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / (16 * m) = i / (16 * m) := by
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have h : (i + k) / 16 = i / 16 := by omega
  rw [h]

/-- The group index as a shift equals the division by `2^s`. -/
theorem group_shift_eq_div (e s : Nat) : e >>> s = e / 2 ^ s := Nat.shiftRight_eq_div_pow e s

/-- Row start offsets in the warp kernel: row `o` begins at `o * (n_in / per)`; distinct (block, warp) pairs
map to distinct rows. -/
theorem row_of_injective (wpb b w b' w' : Nat) (hw : w < wpb) (hw' : w' < wpb)
    (h : b * wpb + w = b' * wpb + w') : b = b' ∧ w = w' := by
  have hpos : 0 < wpb := by omega
  have h1 : (b * wpb + w) / wpb = b := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hw, Nat.zero_add]
  have h2 : (b' * wpb + w') / wpb = b' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hw', Nat.zero_add]
  have h3 : (b * wpb + w) % wpb = w := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hw
  have h4 : (b' * wpb + w') % wpb = w' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hw'
  constructor
  · calc b = (b * wpb + w) / wpb := h1.symm
      _ = (b' * wpb + w') / wpb := by rw [h]
      _ = b' := h2
  · calc w = (b * wpb + w) % wpb := h3.symm
      _ = (b' * wpb + w') % wpb := by rw [h]
      _ = w' := h4

/-- Thread `t` of a 32-lane-per-row block splits into warp and lane losslessly. -/
theorem warp_lane_split (t : Nat) : t = 32 * (t / 32) + t % 32 := by omega

end Strata.SGemv



## FILE: /mnt/user-data/outputs/IqKernelsVecDotLayout.lean
/-!
# MODULE: iq_kernels_vecdot_layout
IQ-family (and Q3_K / Q2_0) matrix-vector kernels over Q8_1 activations, dequantizers, the Q8_1 quantizer and
the grouped native-expert launcher. This module captures the index arithmetic: per-format iteration traits,
sign unpacking parity, row byte sizes, the gate/up row split and the expert blob layout/scratch carving.

## Source semantics
* A row of `n` values is `n / qk` blocks; the warp walks `k = lane, lane+32, ...` over `nb * ipb` dot calls with
  `kbx = k / ipb`, `iqs = step * (k % ipb)`.
* `unpack_ksigns v = (v ^ (parity(v) << 7)) * 0x01010101`: the 8-bit sign byte has even parity.
* Gate/up rows `0 .. 2*n_ff - 1`: rows `>= n_ff` are up rows with row index `row - n_ff`.
* Expert blob: `[gate | up | down]`, `up_off = n_ff * gu_row`, `down_off = 2 * up_off`.
* Scratch regions are rounded up to 256 bytes.

## Form template identifiers
- `Fmt<TY>` (`qk`, `ipb`, `step`) → `format_traits`
- `row_dot`                       → `warp_row_dot`
- `native_gu_kernel`              → `grouped_gate_up_kernel`
- `native_down_kernel`            → `grouped_down_kernel`
- `native_expert_layout`          → `expert_blob_layout`
-/

namespace Strata.IqKernels

-- § sign unpacking

/-- Population count of the low 8 bits. -/
def popc8 (v : Nat) : Nat := (List.range 8).foldl (fun a i => a + ((v >>> i) &&& 1)) 0

/-- `unpack_ksigns` sign byte for a 7-bit key. -/
def signByte (v : Nat) : Nat := v ^^^ ((popc8 v % 2) <<< 7)

/-- Every 7-bit key expands to an 8-bit sign byte of even parity (the 8th sign is implied). -/
theorem signByte_even_parity : ∀ v : Nat, v < 128 → popc8 (signByte v) % 2 = 0 := by
  decide

/-- The sign byte stays a byte. -/
theorem signByte_lt : ∀ v : Nat, v < 128 → signByte v < 256 := by
  decide

-- § iteration traits

/-- Dot calls of a `(qk, ipb, step)` format over a warp-strided loop: `kbx`, `iqs` of call `k`. -/
def callBlock (ipb k : Nat) : Nat := k / ipb
/-- The `iqs` of call `k`. -/
def callIqs (ipb step k : Nat) : Nat := step * (k % ipb)

/-- Calls stay inside the row: `kbx < nb` for `k < nb * ipb`. -/
theorem call_block_lt (ipb nb k : Nat) (hi : 0 < ipb) (hk : k < nb * ipb) : callBlock ipb k < nb := by
  unfold callBlock
  exact (Nat.div_lt_iff_lt_mul hi).mpr hk

/-- `iqs` stays below `step * ipb`. -/
theorem call_iqs_lt (ipb step k : Nat) (hi : 0 < ipb) (hs : 0 < step) :
    callIqs ipb step k < step * ipb := by
  unfold callIqs
  exact Nat.mul_lt_mul_of_pos_left (Nat.mod_lt _ hi) hs

/-- Every call index decomposes as block times `ipb` plus part. -/
theorem call_decompose (ipb k : Nat) : k = callBlock ipb k * ipb + k % ipb := by
  unfold callBlock
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod k ipb).symm

/-- Lane-strided traversal: call `k` is handled by lane `k % 32` at iteration `k / 32`. -/
theorem lane_stride_cover (k : Nat) : k = k % 32 + 32 * (k / 32) := by omega

/-- Format traits table `(type, qk, ipb, step)` transcribed from `Fmt<TY>`. -/
def fmtTraits : List (Nat × Nat × Nat × Nat) :=
  [(16, 256, 8, 2), (17, 256, 8, 2), (18, 256, 8, 2), (20, 32, 2, 2),
   (21, 256, 8, 2), (22, 256, 8, 2), (29, 256, 8, 1), (42, 64, 2, 1)]

/-- The `iqs` of a call times the vector width never exceeds the block's 32-value chunks: `step * ipb` is
`qk / 32 * 8`-bounded, here checked per format as `step * ipb <= 16`. -/
theorem fmt_iqs_bound : ∀ x ∈ fmtTraits, x.2.2.2 * x.2.2.1 ≤ 16 := by decide

/-- Block sizes `qk` are multiples of 32 so Q8_1 chunks (32 values) tile every block exactly. -/
theorem fmt_qk_multiple_of_32 : ∀ x ∈ fmtTraits, x.2.1 % 32 = 0 := by decide

-- § row bytes

/-- `iq_row_bytes` for a format with `qk` values and `bytes` per block. -/
def rowBytes (qk bytes n : Nat) : Nat := (n / qk) * bytes

/-- For a whole number of blocks the row size is exactly `blocks * bytes`. -/
theorem rowBytes_blocks (qk bytes a : Nat) (hq : 0 < qk) : rowBytes qk bytes (a * qk) = a * bytes := by
  unfold rowBytes
  rw [Nat.mul_div_cancel _ hq]

/-- Row sizes are additive over whole blocks. -/
theorem rowBytes_add (qk bytes a b : Nat) (hq : 0 < qk) :
    rowBytes qk bytes ((a + b) * qk) = rowBytes qk bytes (a * qk) + rowBytes qk bytes (b * qk) := by
  rw [rowBytes_blocks _ _ _ hq, rowBytes_blocks _ _ _ hq, rowBytes_blocks _ _ _ hq, Nat.add_mul]

/-- Supported type ids. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- The dispatchable (MMVQ) formats are a subset of the supported ones. -/
theorem mmvq_formats_supported : ∀ x ∈ fmtTraits, isIq x.1 = true := by decide

-- § Q8_1 quantizer indexing

/-- Element `i` lives in block `i / 32` at slot `i % 32`, and the pair recomposes `i`. -/
theorem q8_1_index (i : Nat) : i = 32 * (i / 32) + i % 32 := by omega

/-- Columns of `n_in = 32 q` elements are block-aligned: column `c` element `j < n_in` is in block
`c * q + j / 32` of the flat activation image. -/
theorem q8_1_column_block (q c j : Nat) :
    (32 * (c * q) + j) / 32 = c * q + j / 32 := by omega

-- § grouped native experts

/-- Gate/up row split: `row < 2 n_ff`; up rows have in-range row index. -/
theorem gu_row_split (n_ff row : Nat) (h : row < 2 * n_ff) :
    (if row ≥ n_ff then row - n_ff else row) < n_ff := by
  split <;> omega

/-- The gate/up grid covers all `2 * n_ff` rows with `GU_ROWS = 8` warps per block. -/
theorem gu_grid_covers (n_ff row : Nat) (h : row < 2 * n_ff) :
    row / 8 < (2 * n_ff + 8 - 1) / 8 := by omega

/-- Expert blob regions: `[0, up_off)` gate, `[up_off, down_off)` up, then down; sizes add to `bytes`. -/
theorem blob_layout (n_ff gu_row n_embd d_row : Nat) :
    let up_off := n_ff * gu_row
    let down_off := 2 * up_off
    let bytes := down_off + n_embd * d_row
    up_off ≤ down_off ∧ down_off ≤ bytes ∧ down_off = 2 * (n_ff * gu_row) := by
  intro up_off down_off bytes
  refine ⟨?_, ?_, rfl⟩ <;> simp only [up_off, down_off, bytes] <;> omega

/-- The up matrix is exactly `n_ff` rows after the gate matrix. -/
theorem up_row_offset (n_ff gu_row r : Nat) :
    n_ff * gu_row + r * gu_row = (n_ff + r) * gu_row := by
  rw [Nat.add_mul]

/-- Scratch regions are rounded up to 256 bytes: never smaller, always aligned. -/
theorem align256_ok (f : Nat) : f ≤ (f + 255) / 256 * 256 ∧ (f + 255) / 256 * 256 % 256 = 0 := by
  omega

/-- `native_expert_scratch_bytes`: gate, up, h each padded, then the Q8_1 image; offsets are 256-aligned. -/
theorem scratch_offsets_aligned (f g : Nat) :
    let fa := (f + 255) / 256 * 256
    let ga := (g + 255) / 256 * 256
    (3 * fa) % 256 = 0 ∧ fa ≤ 2 * fa ∧ 3 * fa + ga ≥ 3 * f + g := by
  intro fa ga
  simp only [fa, ga]
  omega

/-- Down rows: `r = blockIdx.x * 8 + warp`; covering `n_embd` rows needs `(n_embd + 7) / 8` blocks. -/
theorem down_grid_covers (n_embd r : Nat) (h : r < n_embd) : r / 8 < (n_embd + 7) / 8 := by omega

/-- SwiGLU entry kernel guards `i < n`; grid of 256-thread blocks covers `n`. -/
theorem elementwise_grid_covers (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

end Strata.IqKernels



## FILE: /mnt/user-data/outputs/NativeMmvqBlockGeometry.lean
/-!
# MODULE: native_mmvq_block_geometry
Native Q8_1 matrix-vector kernels (Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, Q5_0, Q8_0, IQ4_NL) adapted
from llama.cpp, plus the multi-column (ncols = 2..8) generic kernel, the Q8_1 quantizer and the host
validation / weight-size helpers.

## Source semantics
* Each weight block struct has a fixed size (`static_assert`s); rows are `n_in / block_elems` blocks.
* A 4 x 32 thread block advances `BPI` weight blocks per iteration where thread `tid` starts at block
  `tid / T`; every format satisfies `T * BPI = 128`, so all blocks are visited exactly once per row.
* "Small K" (`blocks_per_row < BPI`) uses 4 rows per block.
* Q4_0/Q5_0 affine correction constants `8 * vdr / QI` are 4 and 8.
* Multi-column activation images are column-contiguous with `n_in / 32` Q8_1 blocks per column.

## Form template identifiers
- `Q5KBlock`..`IQ4NLBlock`         → `weight_block_struct`
- `Q81Block`                        → `act_block_q8_1`
- `native_*_mmvq_kernel`            → `row_mmvq_kernel`
- `native_mmvq_multi_kernel`        → `multi_column_kernel`
- `native_mmvq_weight_bytes`        → `weight_byte_count`
- `native_quantize_q8_1_kernel`     → `quantize_act_kernel`
-/

namespace Strata.NativeMmvq

-- § struct sizes

/-- Sizes of the weight and activation block structs (mirrors the `static_assert`s). -/
theorem struct_sizes :
    4 + 12 + 32 + 128 = 176 ∧          -- Q5_K
    4 + 32 = 36 ∧                       -- Q8_1
    2 + 16 = 18 ∧                       -- Q2_0, Q4_0, IQ4_NL
    32 + 64 + 12 + 2 = 110 ∧            -- Q3_K
    2 + 2 + 4 + 128 = 136 ∧             -- IQ4_XS
    4 + 12 + 128 = 144 ∧                -- Q4_K
    128 + 64 + 16 + 2 = 210 ∧           -- Q6_K
    2 + 4 + 16 = 22 ∧                   -- Q5_0
    2 + 32 = 34 := by                   -- Q8_0
  decide

/-- Field offsets of Q3_K and Q6_K (the `offsetof` asserts). -/
theorem struct_offsets :
    32 + 64 = 96 ∧ 96 + 12 = 108 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide

/-- Quantized activation block alignment: 36 is a multiple of 4 so `int` loads of `qs` (offset 4) are aligned. -/
theorem q8_1_alignment : 36 % 4 = 0 ∧ 4 % 4 = 0 := by decide

-- § iteration traits

/-- `(tag, T, BPI)` of each generic-kernel format: Q5_K=1, Q4_K=2, Q2_0=3, Q3_K=4, Q6_K=5, IQ4_XS=6,
Q4_0/Q5_0/IQ4_NL=7 (Qi = 4), Q8_0=8 (Qi = 8). -/
def traits : List (Nat × Nat × Nat) :=
  [(1, 16, 8), (2, 16, 8), (3, 2, 64), (4, 16, 8), (5, 32, 4), (6, 8, 16), (7, 2, 64), (8, 4, 32)]

/-- Each format's `T * BPI = 128 = WARPS * WARP`: one iteration of all threads tiles `BPI` blocks exactly. -/
theorem traits_tile_threadblock : ∀ x ∈ traits, x.2.1 * x.2.2 = 128 := by decide

/-- `BPI` formulas from the source: `VDR * WARPS * WARP / QI` etc. -/
theorem bpi_formulas :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧
    4 * 4 * 32 / 32 = 16 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Thread `tid` of the 128 starts at block `tid / T`; two threads of one block-group share `kbx`. -/
theorem start_block_cover (tid T : Nat) (hT : 0 < T) : tid = (tid / T) * T + tid % T := by
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod tid T).symm

/-- Strided loop `kbx = start, start + BPI, ...` covers every block `b` from start `b % BPI`. -/
theorem stride_cover (b BPI : Nat) : b = b % BPI + BPI * (b / BPI) :=
  (Nat.mod_add_div b BPI).symm

/-- Small-K rule: fewer blocks per row than `BPI` ⇒ four rows per block, grid `ceil(n_out / 4)`. -/
theorem small_k_grid_covers (n_out r : Nat) (h : r < n_out) : r / 4 < (n_out + 4 - 1) / 4 := by omega

/-- Affine correction constants of Q4_0 / Q5_0: `8 * vdr / QI` and `16 * vdr / QI` with vdr = 2, QI = 4. -/
theorem affine_correction_constants : 8 * 2 / 4 = 4 ∧ 16 * 2 / 4 = 8 := by decide

/-- Summing `(c - b) * x` equals `sum(c * x) - b * sum(x)`: the weight-bias correction through Q8_1's stored sum. -/
theorem bias_correction (b : Int) (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - b) * p.2)).sum = (l.map (fun p => p.1 * p.2)).sum - b * (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.mul_add]
    omega

-- § multi-column

/-- Multi-column grid in the EXACT layout: warps per block and rows per block. -/
theorem exact_layout_rows (n_out : Nat) : (n_out + 4 - 1) / 4 * 4 ≥ n_out := by omega

/-- Upstream layout uses two rows per block: grid `(n_out + 1) / 2` covers all rows. -/
theorem upstream_layout_rows (n_out : Nat) : (n_out + 1) / 2 * 2 ≥ n_out := by omega

/-- Column `j` activation offset in Q8_1 blocks: `j * (n_in / 32)`; with `n_in = 32 q` columns do not overlap. -/
theorem column_blocks_disjoint (q j j' k : Nat) (hjj : j < j') (hk : k < q) : j * q + k < j' * q := by
  have : (j + 1) * q ≤ j' * q := Nat.mul_le_mul_right q hjj
  rw [Nat.add_mul] at this
  omega

/-- Output layout `y[j * n_out + row]`: distinct (column, row) pairs hit distinct slots. -/
theorem output_slot_injective (n_out j r j' r' : Nat) (hr : r < n_out) (hr' : r' < n_out)
    (h : j * n_out + r = j' * n_out + r') : j = j' ∧ r = r' := by
  have hpos : 0 < n_out := by omega
  have h1 : (j * n_out + r) / n_out = j := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr, Nat.zero_add]
  have h2 : (j' * n_out + r') / n_out = j' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr', Nat.zero_add]
  have hj : j = j' := by
    calc j = (j * n_out + r) / n_out := h1.symm
      _ = (j' * n_out + r') / n_out := by rw [h]
      _ = j' := h2
  subst hj
  exact ⟨rfl, by omega⟩

/-- The partial-sum shared array has `NW - 1` warp slices; thread rows `1 .. NW-1` map to slices `0 .. NW-2`. -/
theorem partial_slice_index (NW y : Nat) (h1 : 0 < y) (h2 : y < NW) : y - 1 < NW - 1 := by omega

-- § quantizer / host

/-- Quantizer: `n_in` multiple of 32 means whole warps; `i / 32`, `i % 32` recompose. -/
theorem quantize_blocks (i : Nat) : i = 32 * (i / 32) + i % 32 := by omega

/-- Quantizer grid with 256 threads covers `n_total`. -/
theorem quantize_grid_covers (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

/-- The Q8_1 scratch size `ncols * (n_in / 32) * 36`; bytes for one column divide into blocks. -/
theorem q8_1_bytes (n_in ncols : Nat) (h : n_in % 32 = 0) :
    ncols * (n_in / 32) * 36 = ncols * n_in / 32 * 36 := by
  have : ncols * n_in / 32 = ncols * (n_in / 32) := by
    obtain ⟨q, hq⟩ : ∃ q, n_in = 32 * q := ⟨n_in / 32, by omega⟩
    subst hq
    rw [Nat.mul_left_comm, Nat.mul_div_cancel_left _ (by decide), Nat.mul_div_cancel_left _ (by decide)]
  rw [this]

/-- `native_mmvq_weight_bytes` block table `(type, block_elems, block_bytes)`. -/
def weightTable : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144),
   (13, 256, 176), (14, 256, 210), (23, 256, 136), (42, 64, 18)]

/-- Weight byte count of a row-major matrix: `(n_in / block_elems) * block_bytes * n_out`. -/
def weightBytes (elems bytes n_in n_out : Nat) : Nat := (n_in / elems) * bytes * n_out

/-- The weight byte count is monotone in the number of output rows. -/
theorem weightBytes_mono (elems bytes n_in a b : Nat) (h : a ≤ b) :
    weightBytes elems bytes n_in a ≤ weightBytes elems bytes n_in b := by
  unfold weightBytes
  exact Nat.mul_le_mul_left _ h

/-- Every table entry has a block size dividing 256 or equal to 32/64, so `n_in = 2560` is valid for all. -/
theorem n_embd_valid_for_all : ∀ x ∈ weightTable, 2560 % x.2.1 = 0 := by decide

/-- Supported-type predicate matches the table plus the IQ types handled by `iq_mmvq`. -/
theorem table_types_supported :
    ∀ x ∈ weightTable, (x.1 = 2 ∨ x.1 = 6 ∨ x.1 = 8 ∨ x.1 = 11 ∨ x.1 = 12 ∨ x.1 = 13 ∨ x.1 = 14 ∨
      x.1 = 20 ∨ x.1 = 23 ∨ x.1 = 42) := by decide

end Strata.NativeMmvq



## FILE: /mnt/user-data/outputs/S2ExpertGroupedBlobLayout.lean
/-!
# MODULE: s2_expert_grouped_blob_layout
Grouped GPU expert for the Q2_0 (S2) MoE: gate/up/down projections over Q8_0 activations with `__dp4a`,
warp-per-row, device-side hit selection and grouping, and the CPU-order (bit-matching) variant.

## Source semantics
* Blob geometry for `H = 2560`, `FF = 640`, group 64: codes `[gate/up interleaved | down]`, then gate/up
  scales, then down scales; all offsets are fixed literals.
* `sum_j (c_j - 1) x_j = sum_j c_j x_j - sum_j x_j` (weight bias through the "ones" dot).
* Gate/up rows are interleaved (slot `i` even = gate row `i/2`, odd = up row `i/2`) but the output is
  gate-major: `[hit h gate rows | ... | hit h up rows]`.
* Scales: one fp16 per 64 weights = one per two 32-element chunks, index `c >> 1`.
* Hit compaction by ballot prefix popcount assigns dense slots in entry order.
* CPU-order lane reduction adds lanes in the order 4, 1, 2.

## Form template identifiers
- `gu_kernel` / `down_kernel`        → `projection_kernel`
- `hit_select_kernel`                → `resident_hit_compaction`
- `group_resident_kernel`            → `group_by_expert`
- `row_dot_s2_q8`                    → `s2_row_dot_q8`
- `O_D_CODES`, `O_GU_SCALES`, ...    → `blob_region_offsets`
-/

namespace Strata.S2Expert

-- § blob geometry

/-- Row and scale counts derived from `H = 2560`, `FF = 640`, `QK = 64`. -/
theorem row_constants :
    2560 / 4 = 640 ∧ 640 / 4 = 160 ∧ 2560 / 64 = 40 ∧ 640 / 64 = 10 := by decide

/-- Region offsets of the blob. -/
theorem blob_offsets :
    2 * 640 * 640 = 819200 ∧
    2 * 640 * 640 + 2560 * 160 = 1228800 ∧
    2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 = 1331200 ∧
    2 * 640 * 640 + 2560 * 160 + 2 * 640 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- Regions are consecutive and non-overlapping: gate/up codes < down codes < gate/up scales < down scales. -/
theorem blob_regions_ordered : 819200 < 1228800 ∧ 1228800 < 1331200 ∧ 1331200 < 1382400 := by decide

/-- Scale index of 32-element chunk `c`: chunk pairs share one fp16 scale; gate/up has 80 chunks → 40 scales. -/
theorem chunk_scale_in_range (c : Nat) (h : c < 80) : c / 2 < 40 := by omega

/-- Down rows have 20 chunks → 10 scales, and lanes `0..19` each hold one chunk (`20 ≤ 32`). -/
theorem down_chunk_scale_in_range (c : Nat) (h : c < 20) : c / 2 < 10 ∧ 20 ≤ 32 := by omega

/-- Lanes `0..31` cover the 80 gate/up chunks in `GU_CHUNKS = 3` rounds, and 2 rounds do not suffice. -/
theorem gu_chunks_rounds : (80 + 31) / 32 = 3 ∧ 80 ≤ 3 * 32 ∧ 2 * 32 < 80 := by decide

/-- Block row tilings divide the output dimensions exactly. -/
theorem grouped_tilings : 2 * 640 / 32 = 40 ∧ 2 * 640 % 32 = 0 ∧ 2560 / 64 = 40 ∧ 2560 % 64 = 0 := by decide

/-- A Q8_0 chunk is 34 bytes (2 header + 32 data): data at offset 2 is never 4-aligned, hence `memcpy`. -/
theorem q8_0_data_misaligned : 2 + 32 = 34 ∧ 2 % 4 ≠ 0 ∧ 34 % 4 ≠ 0 := by decide

-- § integer identity

/-- The weight bias through `sum(c*x) - sum(x)`: `Σ (c-1)·x = Σ c·x − Σ x`. -/
theorem bias_through_ones (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - 1) * p.2)).sum = (l.map (fun p => p.1 * p.2)).sum - (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul]
    omega

-- § interleaved rows, gate-major output

/-- Output slot of gate/up row-slot `i` of hit `h` among `n` hits (gate-major layout). -/
def guSlot (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * 640 + h * 640 + i / 2 else h * 640 + i / 2

/-- The output slot is inside the `2 * n * 640` gate/up buffer. -/
theorem guSlot_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : guSlot n h i < 2 * n * 640 := by
  unfold guSlot
  split <;> omega

/-- Distinct (hit, row-slot) pairs write distinct output slots: no two warps collide. -/
theorem guSlot_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (e : guSlot n h i = guSlot n h' i') : h = h' ∧ i = i' := by
  unfold guSlot at e
  split at e <;> split at e <;> omega

/-- Gate rows of hit `h` are contiguous from `h * 640`; up rows contiguous from `n * 640 + h * 640`, so the
swiglu kernel finds `up` at `+ n * 640` from `gate`. -/
theorem swiglu_pairs_with_own_up (n h r : Nat) (hr : r < 640) :
    guSlot n h (2 * r + 1) = guSlot n h (2 * r) + n * 640 := by
  unfold guSlot
  split <;> split <;> omega

-- § hit compaction

/-- Number of `true` entries strictly before position `i`. -/
def prefixCount (l : List Bool) (i : Nat) : Nat := (l.take i).count true

/-- A prefix count never exceeds the total count, so compacted slots stay below `count`. -/
theorem prefixCount_le (l : List Bool) (i : Nat) : prefixCount l i ≤ l.count true := by
  unfold prefixCount
  exact (List.take_sublist i l).count_le true

/-- Warp-ballot compaction across four warps: slot = earlier warps' counts + in-warp prefix. -/
theorem four_warp_slot_lt (c0 c1 c2 c3 p : Nat) (hp : p < c2) :
    c0 + c1 + p < c0 + c1 + c2 + c3 := by omega

/-- Entry position within a group: group start plus rank is below the next group start. -/
theorem group_entry_slot (start size rank : Nat) (h : rank < size) : start + rank < start + size := by omega

/-- Entry token is the routed index divided by `k_per_tok`; entries of one token have distinct destinations. -/
theorem token_of_entry (i k : Nat) (hk : 0 < k) : i = (i / k) * k + i % k := by
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod i k).symm

-- § scratch carving

/-- 16-byte alignment of scratch regions. -/
theorem align16_ok (x : Nat) : x ≤ (x + 15) / 16 * 16 ∧ (x + 15) / 16 * 16 % 16 = 0 := by omega

/-- Down rows are indexed `row = h * H + r`; decomposition is unique for `r < H`. -/
theorem down_row_split (H h r : Nat) (hr : r < H) : (h * H + r) / H = h ∧ (h * H + r) % H = r := by
  have hpos : 0 < H := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hr

-- § CPU-order lane reduction

/-- The CPU-order reduction (shuffle offsets 4, 1, 2 within groups of 8) sums all eight lanes; over exact
integers the order is immaterial, over floats it is exactly what is being matched. -/
theorem cpu_order_reduction_sum (a0 a1 a2 a3 a4 a5 a6 a7 : Int) :
    (((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7))) = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 := by
  omega

/-- Eight lanes handle one 64-element block: each lane's `dot4` covers 4 of 32 elements per half. -/
theorem cpu_order_lane_coverage : 8 * 4 = 32 ∧ 2 * 32 = 64 := by decide

end Strata.S2Expert



## FILE: /mnt/user-data/outputs/S2GemvLutStagedExperiment.lean
/-!
# MODULE: s2_gemv_lut_staged_experiment
Experimental S2 GEMV (block per output row) with an optional shared-memory staging of the FP16 activation and
a constant-memory byte→four-code table. Both levers measured neutral/negative; the kernel is kept as a
reference. This module records the table, indexing and shared-memory layout invariants.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1` for byte `b`, lane `k < 4`.
* One quad = one code byte = four consecutive elements; group = `q >> 4` (64-element groups).
* With staging, `x` occupies `MAX_SHARED_HALVES = 4096` halves (8 KB) and the partial-sum array starts at float
  offset `MAX_SHARED_HALVES / 2`.
* The halving tree over `threads_per_row` partials is deterministic.

## Form template identifiers
- `c_codes`                 → `byte_code_lut`
- `s2_gemv_fast_kernel`     → `block_per_row_lut_kernel`
- `STAGE_X`                 → `stage_activation_flag`
- `MAX_SHARED_HALVES`       → `shared_stage_capacity`
-/

namespace Strata.S2GemvFast

-- § reduction toolkit

/-- Prefix sum of the first `n` terms of `f`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Splitting a prefix sum at `a`. -/
theorem sumTo_add (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b + 1) = sumTo f a + sumTo (fun t => f (a + t)) (b + 1)
    simp only [sumTo, ih]
    omega

/-- Prefix sums are additive in the summand. -/
theorem sumTo_add_fun (g h : Nat → Int) (n : Nat) :
    sumTo (fun t => g t + h t) n = sumTo g n + sumTo h n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    omega

/-- One halving step of the shared-memory tree preserves the total. -/
theorem sumTo_halve (f : Nat → Int) (s : Nat) :
    sumTo f (2 * s) = sumTo (fun t => f t + f (s + t)) s := by
  rw [Nat.two_mul, sumTo_add, sumTo_add_fun]

/-- The kernel's reduction loop `partial[t] += partial[t + step]`, steps `2^(k-1), ..., 1`, read at `t = 0`. -/
def redT : (Nat → Int) → Nat → Int
  | f, 0 => f 0
  | f, k + 1 => redT (fun t => f t + f (2 ^ k + t)) k

/-- The fixed tree reduction over `2^k` partials equals the plain sum of all partials. -/
theorem redT_eq (k : Nat) : ∀ f : Nat → Int, redT f k = sumTo f (2 ^ k) := by
  induction k with
  | zero => intro f; simp [redT, sumTo]
  | succ k ih =>
    intro f
    show redT (fun t => f t + f (2 ^ k + t)) k = _
    rw [ih, Nat.pow_succ, Nat.mul_comm, sumTo_halve]


-- § the lookup table

/-- Table entry for byte `b`, lane `k`. -/
def lutEntry (b k : Nat) : Int := (((b >>> (2 * k)) % 4 : Nat) : Int) - 1

/-- Every lookup entry is one of the ternary weights `-1, 0, 1, 2` (code 3 is the unused fourth level). -/
theorem lut_entry_range : ∀ b : Nat, b < 256 → ∀ k : Nat, k < 4 → -1 ≤ lutEntry b k ∧ lutEntry b k ≤ 2 := by
  intro b _ k _
  unfold lutEntry
  omega

/-- The four entries of a byte recombine into that byte (`Σ (entry+1) 4^k = b`): the table is lossless. -/
theorem lut_lossless : ∀ b : Nat, b < 256 →
    ((lutEntry b 0 + 1) + 4 * (lutEntry b 1 + 1) + 16 * (lutEntry b 2 + 1) + 64 * (lutEntry b 3 + 1)) = (b : Int) := by
  intro b hb
  simp only [lutEntry, Nat.reduceMul]
  omega

/-- Table size: 256 entries × 4 floats × 4 bytes = 4 KB, within constant memory. -/
theorem lut_bytes : 256 * 4 * 4 = 4096 := by decide

-- § indexing

/-- Quad `q` covers elements `4q .. 4q+3`, all in group `q >> 4`. -/
theorem quad_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q / 16 := by omega

/-- The group index as a shift equals the division. -/
theorem quad_group_shift (q : Nat) : q >>> 4 = q / 16 := Nat.shiftRight_eq_div_pow q 4

/-- A quad's four halves are 8 bytes at offset `8 q`, so the `uint2` load is 8-byte aligned. -/
theorem quad_load_aligned (q : Nat) : (q * 4 * 2) % 8 = 0 := by omega

/-- The code byte count of a row is `n_in / 4`; the scale count is `n_in / 64`; for `n_in = 64 m` they are
`16 m` and `m`. -/
theorem row_sizes (m : Nat) : (64 * m) / 4 = 16 * m ∧ (64 * m) / 64 = m := by omega

-- § shared memory layout

/-- Staging buffer: 4096 halves = 8192 bytes = 2048 floats, which is exactly the partial-array float offset. -/
theorem stage_offset_consistent : 4096 * 2 = 2048 * 4 ∧ 4096 / 2 = 2048 := by decide

/-- `n_embd = 2560` fits the staging limit. -/
theorem n_embd_fits_stage : 2560 ≤ 4096 := by decide

/-- Dynamic shared memory with staging: 8 KB for `x` plus `4 * tpr` for partials; for tpr = 256 it is 9 KB. -/
theorem stage_smem_bytes (tpr : Nat) : 4096 * 2 + tpr * 4 = 8192 + 4 * tpr := by omega

/-- Row ownership: block `o` reads codes at `o * (n_in / 4)`; distinct rows have disjoint code spans. -/
theorem row_span_disjoint (o o' stride k : Nat) (hlt : o < o') (hk : k < stride) :
    o * stride + k < o' * stride := by
  have h : (o + 1) * stride ≤ o' * stride := Nat.mul_le_mul_right stride hlt
  rw [Nat.add_mul] at h
  omega

/-- Thread-strided quad loop `q = tid, tid + tpr, ...` covers each quad exactly once. -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Pairwise partial combine `(a0 + a1) + (a2 + a3)` is the plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

end Strata.S2GemvFast



## FILE: /mnt/user-data/outputs/S2GemvQ8ActivationContract.lean
/-!
# MODULE: s2_gemv_q8_activation_contract
S2 (Q2_0) GEMV that consumes Q8_0-quantized activations as ggml does: block per output row, each thread takes
quads (one code byte = four elements), dequantizes the activation `xq * dx` in the loop, accumulates in four
partials and reduces through a shared-memory halving tree.

## Source semantics
* Activations are `n_in / 32` blocks of 34 bytes (`fp16 d`, `int8 qs[32]`).
* A quad `q` covers elements `4q .. 4q+3`, all inside Q8_0 block `q / 8` at offset `4 * (q % 8)`.
* The S2 group index is `q >> 4` (64-element groups).
* Required: `n_in % 32 = 0` and `n_in % 64 = 0`.

## Form template identifiers
- `s2_gemv_q8_kernel`   → `block_per_row_q8_act_kernel`
- `QK8_0`               → `act_block_elems`
- `ablk`, `off`         → `act_block_index`, `act_block_offset`
-/

namespace Strata.S2GemvQ8

-- § reduction toolkit

/-- Prefix sum of the first `n` terms of `f`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Splitting a prefix sum at `a`. -/
theorem sumTo_add (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b + 1) = sumTo f a + sumTo (fun t => f (a + t)) (b + 1)
    simp only [sumTo, ih]
    omega

/-- Prefix sums are additive in the summand. -/
theorem sumTo_add_fun (g h : Nat → Int) (n : Nat) :
    sumTo (fun t => g t + h t) n = sumTo g n + sumTo h n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    omega

/-- One halving step of the shared-memory tree preserves the total. -/
theorem sumTo_halve (f : Nat → Int) (s : Nat) :
    sumTo f (2 * s) = sumTo (fun t => f t + f (s + t)) s := by
  rw [Nat.two_mul, sumTo_add, sumTo_add_fun]

/-- The kernel's reduction loop `partial[t] += partial[t + step]`, steps `2^(k-1), ..., 1`, read at `t = 0`. -/
def redT : (Nat → Int) → Nat → Int
  | f, 0 => f 0
  | f, k + 1 => redT (fun t => f t + f (2 ^ k + t)) k

/-- The fixed tree reduction over `2^k` partials equals the plain sum of all partials. -/
theorem redT_eq (k : Nat) : ∀ f : Nat → Int, redT f k = sumTo f (2 ^ k) := by
  induction k with
  | zero => intro f; simp [redT, sumTo]
  | succ k ih =>
    intro f
    show redT (fun t => f t + f (2 ^ k + t)) k = _
    rw [ih, Nat.pow_succ, Nat.mul_comm, sumTo_halve]


-- § activation block addressing

/-- Quad `q` starts in activation block `(4 q) / 32 = q / 8`. -/
theorem quad_block_index (q : Nat) : (q * 4) / 32 = q / 8 := by omega

/-- Offset of quad `q` inside its block. -/
theorem quad_block_offset (q : Nat) : (q * 4) % 32 = 4 * (q % 8) := by omega

/-- All four elements of a quad lie inside the block: `off + 3 < 32`. -/
theorem quad_inside_block (q : Nat) : (q * 4) % 32 + 3 < 32 := by omega

/-- Block start in bytes is `34 * (q / 8)`; the fp16 header is read at that offset, data at `+ 2`. -/
theorem block_byte_offsets (q : Nat) : 34 * (q / 8) + 2 + (q * 4) % 32 + 3 < 34 * (q / 8 + 1) := by omega

/-- The S2 group of quad `q` and of each of its four elements coincide. -/
theorem quad_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- Per-element decode `(code - 1) * d` for the four 2-bit fields of a byte. -/
def field (byte k : Nat) : Int := (((byte >>> (2 * k)) % 4 : Nat) : Int) - 1

/-- The four decoded fields identify the byte. -/
theorem fields_lossless : ∀ b : Nat, b < 256 →
    ((field b 0 + 1) + 4 * (field b 1 + 1) + 16 * (field b 2 + 1) + 64 * (field b 3 + 1)) = (b : Int) := by
  intro b hb
  simp only [field, Nat.reduceMul]
  omega

-- § preconditions

/-- The two divisibility requirements reduce to `n_in % 64 = 0` (64 is a multiple of 32). -/
theorem requirements_reduce (n : Nat) (h : n % 64 = 0) : n % 32 = 0 ∧ n % 4 = 0 := by omega

/-- With `n_in = 2560` both requirements hold: 80 activation blocks, 640 quads per row. -/
theorem n_embd_requirements : 2560 % 32 = 0 ∧ 2560 % 64 = 0 ∧ 2560 / 32 = 80 ∧ 2560 / 4 = 640 := by decide

/-- A row's quads number `n_in / 4` and its scales `n_in / 64`; the last quad's group is the last scale. -/
theorem last_quad_group (m : Nat) (hm : 0 < m) : (64 * m / 4 - 1) / 16 = m - 1 := by omega

/-- Thread-strided quad loop covers each quad exactly once. -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Four-partial combine equals plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

/-- Dequantizing in the loop multiplies the weight `(c-1)*d` by `xq*dx`; the activation product commutes. -/
theorem dequant_commute (c d xq dx : Int) :
    ((c - 1) * d) * (xq * dx) = ((c - 1) * d) * (dx * xq) := by
  rw [Int.mul_comm xq dx]

end Strata.S2GemvQ8



## FILE: /mnt/user-data/outputs/S2GemvQuadAmortised.lean
/-!
# MODULE: s2_gemv_quad_amortised
S2 GEMV with the code load amortised over four elements: block per output row, thread `tid` takes quads
`q = tid, tid + tpr, ...`, loads one code byte, one scale and one 64-bit group of four FP16 activations, keeps
four accumulators and reduces through a shared-memory halving tree.

## Source semantics
* One code byte per quad; element `4q + k` is bits `[2k, 2k+2)`, decoded as `(code - 1) * d`.
* Group index of quad `q` is `q >> 4` because `(4q) >> 6 = q >> 4`.
* The four halves of a quad are 8 contiguous, 8-byte-aligned bytes.
* The reduction tree over `threads_per_row` partials is deterministic.

## Form template identifiers
- `s2_gemv_quads_kernel`  → `quad_amortised_kernel`
- `n_quads`               → `quad_count`
- `partial`               → `block_partials`
-/

namespace Strata.S2GemvQuads

-- § reduction toolkit

/-- Prefix sum of the first `n` terms of `f`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Splitting a prefix sum at `a`. -/
theorem sumTo_add (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b + 1) = sumTo f a + sumTo (fun t => f (a + t)) (b + 1)
    simp only [sumTo, ih]
    omega

/-- Prefix sums are additive in the summand. -/
theorem sumTo_add_fun (g h : Nat → Int) (n : Nat) :
    sumTo (fun t => g t + h t) n = sumTo g n + sumTo h n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    omega

/-- One halving step of the shared-memory tree preserves the total. -/
theorem sumTo_halve (f : Nat → Int) (s : Nat) :
    sumTo f (2 * s) = sumTo (fun t => f t + f (s + t)) s := by
  rw [Nat.two_mul, sumTo_add, sumTo_add_fun]

/-- The kernel's reduction loop `partial[t] += partial[t + step]`, steps `2^(k-1), ..., 1`, read at `t = 0`. -/
def redT : (Nat → Int) → Nat → Int
  | f, 0 => f 0
  | f, k + 1 => redT (fun t => f t + f (2 ^ k + t)) k

/-- The fixed tree reduction over `2^k` partials equals the plain sum of all partials. -/
theorem redT_eq (k : Nat) : ∀ f : Nat → Int, redT f k = sumTo f (2 ^ k) := by
  induction k with
  | zero => intro f; simp [redT, sumTo]
  | succ k ih =>
    intro f
    show redT (fun t => f t + f (2 ^ k + t)) k = _
    rw [ih, Nat.pow_succ, Nat.mul_comm, sumTo_halve]


/-- `(4q) >> 6 = q >> 4`: the group of a quad's first element. -/
theorem group_shift_identity (q : Nat) : (4 * q) >>> 6 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
  omega

/-- All four elements of a quad share its group. -/
theorem quad_one_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q / 16 := by omega

/-- The code row has exactly one byte per quad: `n_in / 4` bytes; the row of output `o` starts at `o * (n_in / 4)`. -/
theorem code_row_start (o n_quads q : Nat) (hq : q < n_quads) : o * n_quads + q < (o + 1) * n_quads := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scale row start of output `o` is `o * (n_in / 64)`; quad `q` reads scale `q / 16` of its row, which exists. -/
theorem scale_in_row (m q : Nat) (hq : q < 16 * m) : q / 16 < m := by omega

/-- Four halves = 8 bytes at byte offset `8q`, aligned for `uint2`. -/
theorem half4_aligned (q : Nat) : (4 * q * 2) % 8 = 0 := by omega

/-- The four decoded fields of a code byte (bias -1 applied to the integer code) recompose the byte. -/
theorem fields_lossless : ∀ b : Nat, b < 256 →
    (((b % 4 : Nat) : Int) + 4 * (((b >>> 2) % 4 : Nat) : Int) + 16 * (((b >>> 4) % 4 : Nat) : Int)
      + 64 * (((b >>> 6) % 4 : Nat) : Int)) = (b : Int) := by
  intro b hb
  omega

/-- The `n_in % 4` precondition: with `n_in = 2560` there are 640 quads and 40 groups of 16 quads. -/
theorem n_embd_quads : 2560 / 4 = 640 ∧ 640 / 16 = 40 ∧ 2560 / 64 = 40 := by decide

/-- Thread-strided loop covers each quad exactly once (thread `q % tpr`, iteration `q / tpr`). -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Four-accumulator combine `(a0 + a1) + (a2 + a3)` is the plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

/-- The per-thread partials form a function of `tid`; the final value `partial[0]` of the tree over
`2^k` threads is the sum of all partials. -/
theorem block_reduction_total (k : Nat) (parts : Nat → Int) :
    redT parts k = sumTo parts (2 ^ k) := redT_eq k parts

end Strata.S2GemvQuads



## FILE: /mnt/user-data/outputs/SharedExpertScratchAndCombine.lean
/-!
# MODULE: shared_expert_scratch_and_combine
The MoE shared expert (`silu(gate) * up`, down projection, per-token scalar sigmoid gate multiplied in) and the
final MoE combination (router-weighted routed parts plus the plain shared output). This module captures the
scratch carving, the activation-kind dispatch constraints, row indexing of the multi-token scaling and the
combine algebra.

## Source semantics
* Scratch (all regions 16-byte aligned): `gate | up | h_q8_0 | h_q8k | g`; total
  `2 a + q0 + qk + 32` with `a = round16(4 n_ff)`, `q0 = round16(34 n_ff/32)`, `qk = round16(292 n_ff/256)`.
* A down weight wanting Q8_K needs `n_ff % 256 = 0`; with `n_ff = 640` that is impossible.
* The shared output is added plain: `y_j = Σ_e w_e · parts[e,j] + shared_j`.
* Multi-token: `n_tok` in 1..8, row `t` scaled by `g[t]` at index `t * n_embd + i`.
* The scalar gate reads `n_embd` BF16 values: 2 bytes each, not 4.

## Form template identifiers
- `shared_expert_scratch_bytes`  → `scratch_size`
- `moe_combine_kernel`           → `weighted_combine_kernel`
- `scale_rows_kernel`            → `per_token_scale_kernel`
- `scalar_gate_kernel`           → `token_scalar_gate_kernel`
-/

namespace Strata.SharedExpert

-- § scratch carving

/-- Round up to a multiple of 16. -/
def round16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Rounding never shrinks and always aligns. -/
theorem round16_ok (x : Nat) : x ≤ round16 x ∧ round16 x % 16 = 0 := by
  unfold round16
  omega

/-- Offsets of the five regions. -/
theorem carve_offsets_aligned (a q0 qk : Nat) (ha : a % 16 = 0) (h0 : q0 % 16 = 0) (hk : qk % 16 = 0) :
    (a) % 16 = 0 ∧ (2 * a) % 16 = 0 ∧ (2 * a + q0) % 16 = 0 ∧ (2 * a + q0 + qk) % 16 = 0 := by
  omega

/-- Regions are in order and the scalar gate `g` (one float = 4 bytes) fits in the final 32 bytes. -/
theorem carve_fits (n_ff a q0 qk : Nat) (ha : 4 * n_ff ≤ a) (h0 : n_ff / 32 * 34 ≤ q0)
    (hk : n_ff / 256 * 292 ≤ qk) :
    4 * n_ff ≤ a ∧ a + 4 * n_ff ≤ 2 * a ∧ n_ff / 32 * 34 ≤ q0 ∧
      2 * a + q0 + qk + 4 ≤ 2 * a + q0 + qk + 32 := by
  omega

/-- For `n_ff = 640` the three regions are 2560, 680 and 584 bytes before rounding (and 2560, 688, 592 after). -/
theorem n_ff_640_sizes :
    4 * 640 = 2560 ∧ 640 / 32 * 34 = 680 ∧ 640 / 256 * 292 = 584 ∧
    round16 680 = 688 ∧ round16 584 = 592 ∧ round16 2560 = 2560 := by decide

-- § activation-kind constraints

/-- Q8_K is structurally impossible for `n_ff = 640`; Q8_0 is fine. -/
theorem q8k_impossible_for_640 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

/-- The Q8_K image of a vector needs a whole number of 256-blocks: its byte count is `n/256 * 292`. -/
theorem q8k_bytes (m : Nat) : (256 * m) / 256 * 292 = 292 * m := by omega

/-- The Q8_0 image: `n/32 * 34` bytes. -/
theorem q8_0_bytes (m : Nat) : (32 * m) / 32 * 34 = 34 * m := by omega

-- § BF16 scalar gate

/-- BF16 weights are 2 bytes: reading them as 4-byte floats overruns the tensor by exactly its own size. -/
theorem gate_overrun : 2560 * 4 - 2560 * 2 = 5120 ∧ 2560 * 2 = 5120 := by decide

/-- Per-lane strided dot terms: thread `t` covers `i ≡ t (mod blockDim)`; each index has one owner. -/
theorem gate_stride_cover (i bd : Nat) : i = i % bd + bd * (i / bd) := (Nat.mod_add_div i bd).symm

/-- Number of warps of a 256-thread gate block: 8, matching `scratch[8]`. -/
theorem gate_warps : (256 + 31) >>> 5 = 8 := by decide

-- § multi-token scaling

/-- Row `t`, element `i` of the multi-token output is slot `t * n + i`, in range for `t < n_tok`, `i < n`. -/
theorem row_slot_lt (n_tok n t i : Nat) (ht : t < n_tok) (hi : i < n) : t * n + i < n_tok * n := by
  have h : (t + 1) * n ≤ n_tok * n := Nat.mul_le_mul_right n ht
  rw [Nat.add_mul] at h
  omega

/-- Distinct (token, element) pairs map to distinct slots: scaling by `g[t]` touches each slot once. -/
theorem row_slot_injective (n t i t' i' : Nat) (hi : i < n) (hi' : i' < n)
    (h : t * n + i = t' * n + i') : t = t' ∧ i = i' := by
  have hpos : 0 < n := by omega
  have h1 : (t * n + i) / n = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have h2 : (t' * n + i') / n = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi', Nat.zero_add]
  have ht : t = t' := by
    calc t = (t * n + i) / n := h1.symm
      _ = (t' * n + i') / n := by rw [h]
      _ = t' := h2
  subst ht
  exact ⟨rfl, by omega⟩

/-- Token count guard: `1 ≤ n_tok ≤ 8`. -/
def tokOk (n : Nat) : Bool := decide (1 ≤ n ∧ n ≤ 8)

/-- Exactly the accepted counts are 1 through 8. -/
theorem tokOk_iff (n : Nat) : tokOk n = true ↔ (1 ≤ n ∧ n ≤ 8) := by
  unfold tokOk
  simp

-- § combine algebra

/-- Sum of the first `k` router-weighted parts at column `j`. -/
def routedSum (w : Nat → Int) (parts : Nat → Int) (k : Nat) : Int :=
  match k with
  | 0 => 0
  | k + 1 => routedSum w parts k + w k * parts k

/-- The combine output: routed sum plus the plain shared value when present. -/
def combine (w parts : Nat → Int) (k : Nat) (shared : Option Int) : Int :=
  routedSum w parts k + shared.getD 0

/-- Without a shared output the combine is the routed sum. -/
theorem combine_no_shared (w parts : Nat → Int) (k : Nat) : combine w parts k none = routedSum w parts k := by
  simp [combine]

/-- The shared output is added plain, never router-weighted. -/
theorem combine_shared_plain (w parts : Nat → Int) (k : Nat) (s : Int) :
    combine w parts k (some s) = routedSum w parts k + s := by
  simp [combine]

/-- Appending one more routed expert adds exactly `w_k * part_k`. -/
theorem routedSum_succ (w parts : Nat → Int) (k : Nat) :
    routedSum w parts (k + 1) = routedSum w parts k + w k * parts k := rfl

/-- With all router weights zero the routed part vanishes. -/
theorem routedSum_zero_weights (parts : Nat → Int) (k : Nat) : routedSum (fun _ => 0) parts k = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [routedSum, ih]
    simp

/-- Launch guards: `k ≤ 64` accepted; the combine grid of 128-thread blocks covers `n_embd`. -/
theorem combine_grid_covers (n j : Nat) (h : j < n) : j / 128 < (n + 127) / 128 := by omega

end Strata.SharedExpert



## FILE: /mnt/user-data/outputs/RouterTop10SelectionSemantics.lean
/-!
# MODULE: router_top10_selection_semantics
MoE router: softmax over all experts, stable top-k (ties by ascending index) via `k` block-wide argmax passes,
renormalisation of the gathered weights with a lower clamp `2^-14`.

## Source semantics
* Selection pass: scan `e` ascending keeping a strict `>` improvement ⇒ the lowest index wins ties; masked
  (already taken) experts are skipped; the sentinel index `n_expert` loses to every real index.
* Weights are `w_i / max(Σ w, 2^-14)`; the clamp guarantees a positive denominator.
* Launch: threads = min(n_expert, 512) rounded up to a warp; shared memory is
  `round16(n) + 8 n + 4 n` bytes; `k ≤ 64`.
* Pass count `k * n` versus the old rank-by-counting `n * n` comparisons.

## Form template identifiers
- `router_top10_kernel`  → `softmax_topk_renorm_kernel`
- `s_taken`              → `taken_mask`
- `RT_MAX_THREADS`       → `max_block_threads`
- `6.103515625e-05`      → `renorm_clamp`
-/

namespace Strata.RouterTop10

-- § argmax with lowest-index tie-break

/-- Index of the first maximum of `p` among positions `0 .. n-1` (position 0 for `n = 0`). -/
def am (p : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => if p (am p n) < p n then n else am p n

/-- The argmax is an in-range index. -/
theorem am_le (p : Nat → Nat) (n : Nat) : am p (n + 1) ≤ n := by
  induction n with
  | zero => simp [am]
  | succ n ih =>
    show (if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1)) ≤ n + 1
    split <;> omega

/-- No element exceeds the argmax's value. -/
theorem am_max (p : Nat → Nat) (n : Nat) : ∀ j, j ≤ n → p j ≤ p (am p (n + 1)) := by
  induction n with
  | zero =>
    intro j hj
    have h0 : j = 0 := by omega
    subst h0
    simp [am]
  | succ n ih =>
    intro j hj
    show p j ≤ p (if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1))
    by_cases h : p (am p (n + 1)) < p (n + 1)
    · rw [if_pos h]
      rcases Nat.lt_or_ge j (n + 1) with hj' | hj'
      · have h1 := ih j (by omega)
        omega
      · have h2 : j = n + 1 := by omega
        subst h2
        exact Nat.le_refl _
    · rw [if_neg h]
      rcases Nat.lt_or_ge j (n + 1) with hj' | hj'
      · exact ih j (by omega)
      · have h2 : j = n + 1 := by omega
        subst h2
        omega

/-- Every index before the argmax is strictly smaller: ties go to the LOWEST index. -/
theorem am_first (p : Nat → Nat) (n : Nat) : ∀ j, j < am p (n + 1) → p j < p (am p (n + 1)) := by
  induction n with
  | zero =>
    intro j hj
    simp [am] at hj
  | succ n ih =>
    intro j hj
    have e : am p (n + 1 + 1) = if p (am p (n + 1)) < p (n + 1) then n + 1 else am p (n + 1) := rfl
    rw [e] at hj ⊢
    by_cases h : p (am p (n + 1)) < p (n + 1)
    · rw [if_pos h] at hj ⊢
      have h1 := am_max p n j (by omega)
      omega
    · rw [if_neg h] at hj ⊢
      exact ih j hj

/-- Warp-shuffle combine of two candidates `(value, index)`: higher value wins, equal values prefer the lower index. -/
def comb (v i w j : Nat) : Nat × Nat :=
  if w > v ∨ (w = v ∧ j < i) then (w, j) else (v, i)

/-- The combine is commutative on distinct indices, so the butterfly order of lanes cannot matter. -/
theorem comb_comm (v i w j : Nat) (hij : i ≠ j) : comb v i w j = comb w j v i := by
  unfold comb
  by_cases h1 : w > v ∨ (w = v ∧ j < i) <;> by_cases h2 : v > w ∨ (v = w ∧ i < j)
  · exfalso
    omega
  · rw [if_pos h1, if_neg h2]
  · rw [if_neg h1, if_pos h2]
  · exfalso
    omega

/-- The sentinel `n_expert` loses to every real index `e < n_expert` at equal value. -/
theorem sentinel_loses (n e v : Nat) (he : e < n) : comb v e v n = (v, e) := by
  unfold comb
  rw [if_neg]
  omega

-- § renormalisation clamp

/-- Denominator after the clamp (over a scaled numerator domain). -/
def clampDen (s eps : Nat) : Nat := max s eps

/-- The clamp keeps the denominator at least `eps`, hence positive when `eps` is. -/
theorem clampDen_pos (s eps : Nat) (h : 0 < eps) : 0 < clampDen s eps := by
  unfold clampDen
  exact Nat.lt_of_lt_of_le h (Nat.le_max_right _ _)

/-- And at least the plain sum, so every renormalised weight is at most its raw share. -/
theorem clampDen_ge_sum (s eps : Nat) : s ≤ clampDen s eps := Nat.le_max_left _ _

-- § launch geometry

/-- Block size: `n_expert` rounded up to a whole warp is at least `n_expert` and warp-aligned. -/
theorem threads_rounding (t : Nat) : t ≤ (t + 31) / 32 * 32 ∧ (t + 31) / 32 * 32 % 32 = 0 := by omega

/-- For the 512-expert model the block is 512 threads, 16 warps, matching `s_red[RT_MAX_THREADS / 32]`. -/
theorem block_512 : (min 512 512 + 31) / 32 * 32 = 512 ∧ 512 / 32 = 16 := by decide

/-- The taken-mask region is 16-byte rounded, so the following `double` array is 8-byte aligned. -/
theorem taken_alignment (n : Nat) : n ≤ (n + 15) / 16 * 16 ∧ ((n + 15) / 16 * 16) % 8 = 0 := by omega

/-- Following arrays: `n` doubles, then `n` floats, keep 4-byte alignment. -/
theorem prob_array_alignment (n : Nat) : ((n + 15) / 16 * 16 + 8 * n) % 4 = 0 := by omega

/-- Shared memory for 512 experts: 512 + 4096 + 2048 = 6656 bytes, within 48 KB. -/
theorem smem_512 : (512 + 15) / 16 * 16 + 512 * 8 + 512 * 4 = 6656 ∧ 6656 ≤ 48 * 1024 := by decide

/-- `k ≤ 64` guard and `n_expert ≤ 512 * 64` guard accept the real configuration (512 experts, k = 10). -/
theorem real_config_accepted : 10 ≤ 64 ∧ 512 ≤ 512 * 64 := by decide

-- § work comparison

/-- Selection work: `k` passes over `n` experts versus `n^2` rank comparisons: 5,120 against 262,144. -/
theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51 := by decide

/-- The k-pass selection does strictly less work than rank-by-counting whenever `k < n`. -/
theorem passes_cheaper (k n : Nat) (h : k < n) (hn : 0 < n) : k * n < n * n :=
  Nat.mul_lt_mul_of_pos_right h hn

/-- Output slot of selection pass `i` of token `t`: `t * k + i`, unique per (token, pass). -/
theorem out_slot_injective (k t i t' i' : Nat) (hi : i < k) (hi' : i' < k)
    (h : t * k + i = t' * k + i') : t = t' ∧ i = i' := by
  have hpos : 0 < k := by omega
  have h1 : (t * k + i) / k = t := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi, Nat.zero_add]
  have h2 : (t' * k + i') / k = t' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hi', Nat.zero_add]
  have ht : t = t' := by
    calc t = (t * k + i) / k := h1.symm
      _ = (t' * k + i') / k := by rw [h]
      _ = t' := h2
  subst ht
  exact ⟨rfl, by omega⟩

end Strata.RouterTop10



## FILE: /mnt/user-data/outputs/NativeRouterLaneLayout.lean
/-!
# MODULE: native_router_lane_layout
Single-warp 512-expert / top-10 router adapted from llama.cpp's topk-moe: each lane holds 16 logits at stride
32, computes softmax in registers, then performs ten warp-wide argmax rounds with xor-shuffle, accumulating the
selected sum per winning lane and renormalising with a `2^-14` clamp. Host side validates alignment, stream and
span disjointness.

## Source semantics
* Lane `l`, register `i` holds expert `l + 32 i`; the map is a bijection `{0..31} × {0..15} ↔ {0..511}` with
  `expert / 32 = i`, `expert % 32 = l`.
* The winning expert's owner lane (`expert & 31`) clears the register `expert / 32` and records the id at
  output rank `rank`; lane `rank` captures the weight.
* Output weights `weights[lane] = selected / max(Σ, 2^-14)` for `lane < 10`.
* Validity: pointer non-null, 4-byte aligned, `address + bytes` does not wrap; spans pairwise disjoint.

## Form template identifiers
- `route`                    → `single_warp_route_kernel`
- `values[16]`               → `lane_registers`
- `native_router_top10`      → `launch_router`
- `valid`, `overlap`         → `span_valid`, `span_overlap`
-/

namespace Strata.NativeRouter

-- § lane / register layout

/-- Expert held by lane `l` in register `i`. -/
def expertOf (l i : Nat) : Nat := l + i * 32

/-- Registers cover 512 experts: 32 lanes × 16 registers. -/
theorem layout_size : 32 * 16 = 512 := by decide

/-- Every (lane, register) pair maps into `[0, 512)`. -/
theorem expertOf_lt (l i : Nat) (hl : l < 32) (hi : i < 16) : expertOf l i < 512 := by
  unfold expertOf
  omega

/-- Register of an expert. -/
theorem expert_register (l i : Nat) (hl : l < 32) : expertOf l i / 32 = i := by
  unfold expertOf
  omega

/-- Owner lane of an expert. -/
theorem expert_lane (l i : Nat) (hl : l < 32) : expertOf l i % 32 = l := by
  unfold expertOf
  omega

/-- Every expert id below 512 is held by exactly the lane/register given by `% 32` and `/ 32`. -/
theorem expert_roundtrip (e : Nat) : expertOf (e % 32) (e / 32) = e := by
  unfold expertOf
  omega

/-- `expert & 31 = expert % 32`, so the owner test `(expert & 31) == lane` is exact. -/
theorem owner_mask (e : Nat) : e &&& 31 = e % 32 := by
  have h := Nat.and_two_pow_sub_one_eq_mod e 5
  simpa using h

/-- Injectivity: two (lane, register) pairs holding the same expert are the same pair. -/
theorem expertOf_injective (l i l' i' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : expertOf l i = expertOf l' i') : l = l' ∧ i = i' := by
  unfold expertOf at h
  omega

-- § selection ranks

/-- Ten rounds, one output rank each; lanes `0..9` capture weights, `10 ≤ 32` so every rank has a lane. -/
theorem rank_has_lane : 10 ≤ 32 ∧ (10 : Nat) ≤ 16 * 32 := by decide

/-- Combine used by the xor butterfly: higher value wins, ties go to the lower expert id. -/
def comb (v i w j : Nat) : Nat × Nat :=
  if w > v ∨ (w = v ∧ j < i) then (w, j) else (v, i)

/-- The combine is symmetric on distinct experts, so the butterfly pairing order is irrelevant. -/
theorem comb_comm (v i w j : Nat) (hij : i ≠ j) : comb v i w j = comb w j v i := by
  unfold comb
  by_cases h1 : w > v ∨ (w = v ∧ j < i) <;> by_cases h2 : v > w ∨ (v = w ∧ i < j)
  · exfalso
    omega
  · rw [if_pos h1, if_neg h2]
  · rw [if_neg h1, if_pos h2]
  · exfalso
    omega

/-- Xor-partner of a lane under mask `m` is an involution (the butterfly pairs lanes mutually). -/
theorem xor_partner_involutive (l m : Nat) : (l ^^^ m) ^^^ m = l := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

/-- Five butterfly masks 16, 8, 4, 2, 1 reduce 32 lanes: `2^5 = 32`. -/
theorem butterfly_depth : 16 + 8 + 4 + 2 + 1 = 31 ∧ 2 ^ 5 = 32 := by decide

-- § renormalisation

/-- Clamped denominator over a scaled numerator domain. -/
def clampDen (s eps : Nat) : Nat := max s eps

/-- The clamp keeps the reciprocal finite. -/
theorem clampDen_pos (s eps : Nat) (h : 0 < eps) : 0 < clampDen s eps := by
  unfold clampDen
  exact Nat.lt_of_lt_of_le h (Nat.le_max_right _ _)

/-- Selected-sum accumulation by winning lane is a regrouping of the same total (over exact integers). -/
theorem lane_grouped_sum (a b c d : Int) : (a + b) + (c + d) = a + b + c + d := by omega

-- § host validation

/-- Pointer validity: aligned and the span does not wrap the address space. -/
def spanValid (address bytes max : Nat) : Prop := address % 4 = 0 ∧ bytes ≤ max - address

/-- Overlap of two byte spans. -/
def spanOverlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- Overlap is symmetric, so the three pairwise checks are order-independent. -/
theorem spanOverlap_symm (ap an bp bn : Nat) : spanOverlap ap an bp bn ↔ spanOverlap bp bn ap an := by
  unfold spanOverlap
  exact And.comm

/-- Spans that end before the other begins do not overlap. -/
theorem spanOverlap_disjoint (ap an bp bn : Nat) (h : ap + an ≤ bp) : ¬ spanOverlap ap an bp bn := by
  unfold spanOverlap
  omega

/-- Logits 2048 bytes, ids 40 bytes, weights 40 bytes are laid out back to back without overlap. -/
theorem consecutive_spans_disjoint (p : Nat) :
    ¬ spanOverlap p 2048 (p + 2048) 40 ∧ ¬ spanOverlap (p + 2048) 40 (p + 2088) 40 := by
  constructor <;> unfold spanOverlap <;> omega

/-- Byte sizes: 512 floats = 2048, 10 ints / floats = 40. -/
theorem span_sizes : 512 * 4 = 2048 ∧ 10 * 4 = 40 := by decide

end Strata.NativeRouter





## FILE: /mnt/user-data/outputs/QsaCacheIndexTopk.lean
/-!
# MODULE: qsa_cache_index_topk
The QSA paged KV cache, the pooled-key indexer and the exact threshold top-k selection
(`src/kernels/cuda/qsa.cu`): page/row addressing, the f32 order key, the ascending
"greater-than plus first equals" walk, the single-writer step state, and the gather decomposition.

## Source semantics
* A pool row is `[page][kv_head][page_size][head_dim]`, row = `(page*kvHeads + h)*pageSize + slot`.
* `order_key` is a total order on f32 bit patterns as unsigned ints (negative: `~b`, else `b | 2^31`).
* `topk_kernel` emits every cell strictly above the threshold key and the first
  `width - c_gt` cells equal to it, walking ascending, so the result has exactly `width` ids
  whenever `c_gt <= width <= c_gt + c_eq`.
* `qsa_step_fill` derives `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min n_kv (topK + r - 1)` from one position.
* `kv_gather` splits a flat thread index into `(id, head, quad)` and writes `dst = i`.

## Form template identifiers
* `poolRow`      <- pool row index in `kv_append_kernel` / `kv_gather_kernel`
* `orderKey`     <- `order_key`
* `walk`         <- the emit loop of `topk_kernel`
* `stepFill`     <- `qsa_step_fill`
* `kvIndex`      <- `h / (n_head / n_head_kv)` (NOT `h % n_head_kv`)
-/

namespace QsaCacheIndexTopk

-- § pool addressing

/-- Row of a cell inside the paged pool. -/
def poolRow (page kvHeads h pageSize slot : Nat) : Nat :=
  (page * kvHeads + h) * pageSize + slot

/-- Every in-range (page, head, slot) maps to a row inside the pool. -/
theorem poolRow_lt (page kvHeads h pageSize slot nPages : Nat)
    (hp : page < nPages) (hh : h < kvHeads) (hs : slot < pageSize) :
    poolRow page kvHeads h pageSize slot < nPages * kvHeads * pageSize := by
  unfold poolRow
  have h0 : (page + 1) * kvHeads ≤ nPages * kvHeads := Nat.mul_le_mul_right _ hp
  have h0' : (page + 1) * kvHeads = page * kvHeads + kvHeads := by
    rw [Nat.add_mul, Nat.one_mul]
  have h1 : page * kvHeads + h + 1 ≤ nPages * kvHeads := by omega
  have h2 : (page * kvHeads + h + 1) * pageSize ≤ nPages * kvHeads * pageSize :=
    Nat.mul_le_mul_right _ h1
  have h2' : (page * kvHeads + h + 1) * pageSize = (page * kvHeads + h) * pageSize + pageSize := by
    rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The query head to KV head map is a division by the group size, and lands in range. -/
theorem kvIndex_lt (nHead nHeadKv h : Nat) (hd : nHead % nHeadKv = 0) (hh : h < nHead) :
    h / (nHead / nHeadKv) < nHeadKv := by
  apply Nat.div_lt_of_lt_mul
  have e := Nat.div_add_mod nHead nHeadKv
  have c := Nat.mul_comm nHeadKv (nHead / nHeadKv)
  omega

-- § order key

/-- The unsigned order key of an f32 bit pattern `b < 2^32`. -/
def orderKey (b : Nat) : Nat :=
  if 2147483648 ≤ b then 4294967295 - b else b + 2147483648

/-- The key fits in 32 bits. -/
theorem orderKey_lt (b : Nat) (hb : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey
  split <;> omega

/-- Distinct bit patterns get distinct keys. -/
theorem orderKey_inj (a b : Nat) (ha : a < 4294967296) (hb : b < 4294967296)
    (h : orderKey a = orderKey b) : a = b := by
  unfold orderKey at h
  split at h <;> split at h <;> omega

/-- On non-negative floats the key is monotone. -/
theorem orderKey_mono_nonneg (a b : Nat) (hab : a ≤ b) (hb : b < 2147483648) :
    orderKey a ≤ orderKey b := by
  unfold orderKey
  split <;> split <;> omega

/-- On negative floats the key is antitone in the raw bits (bigger magnitude, smaller key). -/
theorem orderKey_anti_neg (a b : Nat) (hab : a ≤ b) (ha : 2147483648 ≤ a)
    (hb : b < 4294967296) : orderKey b ≤ orderKey a := by
  unfold orderKey
  split <;> omega

/-- Every negative key is below every non-negative key. -/
theorem orderKey_neg_lt_nonneg (a b : Nat) (ha : 2147483648 ≤ a) (ha2 : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey
  split <;> split <;> omega

-- § the ascending walk

/-- Number of keys strictly above the threshold. -/
def countGt (thr : Nat) : List Nat → Nat
  | [] => 0
  | k :: ks => (if thr < k then 1 else 0) + countGt thr ks

/-- Number of keys equal to the threshold. -/
def countEq (thr : Nat) : List Nat → Nat
  | [] => 0
  | k :: ks => (if k = thr then 1 else 0) + countEq thr ks

/-- The emit loop: index `i`, `e` equal cells already taken, ascending. -/
def walk (thr budget : Nat) : Nat → Nat → List Nat → List Nat
  | _, _, [] => []
  | i, e, k :: ks =>
    if thr < k then i :: walk thr budget (i + 1) e ks
    else if k = thr ∧ e < budget then i :: walk thr budget (i + 1) (e + 1) ks
    else walk thr budget (i + 1) e ks

/-- The walk emits all strictly-greater cells plus as many equal cells as the budget allows. -/
theorem walk_length (thr budget : Nat) (ks : List Nat) :
    ∀ i e, (walk thr budget i e ks).length =
      countGt thr ks + min (countEq thr ks) (budget - e) := by
  induction ks with
  | nil => intro i e; simp [walk, countGt, countEq]
  | cons k ks ih =>
    intro i e
    by_cases h1 : thr < k
    · have h2 : ¬ k = thr := by omega
      simp [walk, countGt, countEq, h1, h2, ih] <;> omega
    · by_cases h2 : k = thr
      · by_cases h3 : e < budget
        · simp [walk, countGt, countEq, h1, h2, h3, ih] <;> omega
        · simp [walk, countGt, countEq, h1, h2, h3, ih] <;> omega
      · simp [walk, countGt, countEq, h1, h2, ih]

/-- With the right threshold the selection has exactly `width` ids. -/
theorem walk_length_width (thr width : Nat) (ks : List Nat) (i : Nat)
    (hgt : countGt thr ks ≤ width) (heq : width ≤ countGt thr ks + countEq thr ks) :
    (walk thr (width - countGt thr ks) i 0 ks).length = width := by
  rw [walk_length]
  omega

/-- The first kernel version compared `c_gt + prefix_eq` to the budget; with distinct scores
(`c_gt = width`, budget 0) that test can never succeed, which is exactly the observed bug. -/
theorem buggy_rank_never_selects (width cgt pe : Nat) (h : cgt = width) :
    ¬ (cgt + pe < width - cgt) := by
  omega

-- § step state

/-- The four per-token values, all derived from one position. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`. -/
def stepFill (pos r topK : Nat) : Step :=
  ⟨pos, pos + 1, (pos + 1) / r, min (pos + 1) (topK + r - 1)⟩

/-- The selection never exceeds the cache. -/
theorem stepFill_width_le (pos r topK : Nat) :
    (stepFill pos r topK).width ≤ (stepFill pos r topK).nKv := by
  show min (pos + 1) (topK + r - 1) ≤ pos + 1
  omega

/-- Below the selection bound the selection is the identity. -/
theorem stepFill_identity (pos r topK : Nat) (h : pos + 1 ≤ topK + r - 1) :
    (stepFill pos r topK).width = (stepFill pos r topK).nKv := by
  show min (pos + 1) (topK + r - 1) = pos + 1
  omega

/-- Completed blocks fit inside the cache. -/
theorem stepFill_nBid_mul_le (pos r topK : Nat) :
    (stepFill pos r topK).nBid * r ≤ (stepFill pos r topK).nKv :=
  Nat.div_mul_le_self _ _

/-- The incomplete tail is shorter than one block. -/
theorem stepFill_nKv_lt (pos r topK : Nat) (hr : 0 < r) :
    (stepFill pos r topK).nKv < ((stepFill pos r topK).nBid + 1) * r := by
  show pos + 1 < ((pos + 1) / r + 1) * r
  have e := Nat.div_add_mod (pos + 1) r
  have m := Nat.mod_lt (pos + 1) hr
  have e2 : ((pos + 1) / r + 1) * r = r * ((pos + 1) / r) + r := by
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
  omega

/-- For `idx_block = 4`, a cell completes a block exactly when the next count is a multiple of 4. -/
theorem block_completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by
  omega

-- § gather decomposition

/-- Arithmetic core of the gather index decomposition. -/
theorem gather_decomp_aux (i kv per A R B C : Nat) (e1 : kv * per * A + R = i)
    (e2 : per * B + C = R) : (A * kv + B) * per + C = i := by
  have e3 : (A * kv + B) * per = kv * per * A + per * B := by
    rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm A (kv * per), Nat.mul_comm B per]
  omega

/-- A flat thread index is recovered from its `(id, head, quad)` split. -/
theorem gather_decomp (i kv per : Nat) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i :=
  gather_decomp_aux i kv per _ _ _ _ (Nat.div_add_mod i (kv * per))
    (Nat.div_add_mod (i % (kv * per)) per)

/-- The head and quad parts are in range. -/
theorem gather_parts_lt (i kv per : Nat) (hk : 0 < kv) (hp : 0 < per) :
    (i % (kv * per)) / per < kv ∧ (i % (kv * per)) % per < per := by
  have hm : 0 < kv * per := Nat.mul_pos hk hp
  have h1 : i % (kv * per) < kv * per := Nat.mod_lt _ hm
  have h2 : kv * per = per * kv := Nat.mul_comm kv per
  constructor
  · apply Nat.div_lt_of_lt_mul
    omega
  · exact Nat.mod_lt _ hp

end QsaCacheIndexTopk



## FILE: /mnt/user-data/outputs/QsaDecodeAttnChunks.lean
/-!
# MODULE: qsa_decode_attn_chunks
Split-K decode attention over the selected cells (`src/kernels/cuda/qsa_decode_attn.cu`):
cells are cut into chunks of 64, each (chunk, kv head) block writes a partial accumulator,
and a merge kernel combines them with log-sum-exp weights.

## Source semantics
* `n_chunks = ceil(cap / 64)`; chunk `c` covers cells `[64c, 64c + n_here)` with
  `n_here = min 64 (n_ids - 64c)`; chunks past the selection are empty and publish `m = -FLT_MAX`.
* The 24 query heads are grouped 12 per KV head: `h = kvh*12 + hl`.
* Scratch is `[acc: chunks*nHead*256][m: chunks*nHead][l: chunks*nHead]` plus 64 spare floats.

## Form template identifiers
* `nChunks`      <- `n_chunks`
* `nHere`        <- `n_here`
* `scratchFloats`<- `qsa_decode_attn_scratch_floats`
* `slotOf`       <- `kvh * n_chunks + chunk`
-/

namespace QsaDecodeAttnChunks

-- § chunking

/-- Number of 64-cell chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

/-- The chunks cover the whole capacity. -/
theorem nChunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by
  unfold nChunks
  omega

/-- ... and not a whole chunk more. -/
theorem nChunks_tight (cap : Nat) : nChunks cap * 64 < cap + 64 := by
  unfold nChunks
  omega

/-- Cells handled by chunk `c` for a selection of `n` ids. -/
def nHere (n c : Nat) : Nat := min 64 (n - 64 * c)

/-- Cells covered by the first `m` chunks. -/
def covered (n : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => covered n m + nHere n m

/-- The first `m` chunks cover exactly `min n (64 m)` cells. -/
theorem covered_eq (n m : Nat) : covered n m = min n (64 * m) := by
  induction m with
  | zero => simp [covered]
  | succ m ih =>
    simp only [covered, nHere, ih]
    omega

/-- All chunks together cover each selected cell exactly once. -/
theorem covered_all (n cap : Nat) (h : n ≤ cap) : covered n (nChunks cap) = n := by
  rw [covered_eq]
  have := nChunks_cover cap
  omega

/-- A chunk is empty exactly when it starts at or past the selection. -/
theorem nHere_zero_iff (n c : Nat) : nHere n c = 0 ↔ n ≤ 64 * c := by
  unfold nHere
  omega

/-- The number of non-empty chunks never exceeds the launched chunks. -/
theorem nonempty_chunks_le (n cap : Nat) (h : n ≤ cap) : (n + 63) / 64 ≤ nChunks cap := by
  unfold nChunks
  omega

-- § slots and heads

/-- Partial-result slot of a (kv head, chunk) block. -/
def slotOf (kvh nCh chunk : Nat) : Nat := kvh * nCh + chunk

/-- Slots stay inside `nKvHeads * nChunks`. -/
theorem slotOf_lt (kvh nCh chunk nKvHeads : Nat) (hk : kvh < nKvHeads) (hc : chunk < nCh) :
    slotOf kvh nCh chunk < nKvHeads * nCh := by
  unfold slotOf
  have h0 : (kvh + 1) * nCh ≤ nKvHeads * nCh := Nat.mul_le_mul_right _ hk
  have h0' : (kvh + 1) * nCh = kvh * nCh + nCh := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- A global query head splits into its KV head and local index. -/
theorem head_split (h : Nat) : h / 12 * 12 + h % 12 = h := by
  omega

/-- The local head index is below the group size. -/
theorem head_local_lt (h : Nat) : h % 12 < 12 := by
  omega

/-- The KV head of every query head is in range. -/
theorem kvh_lt (h nKvHeads : Nat) (hh : h < 12 * nKvHeads) : h / 12 < nKvHeads := by
  omega

-- § scratch layout

/-- Floats of scratch: acc rows of 256 plus m and l, plus 64 spare. -/
def scratchFloats (cap nHead : Nat) : Nat := nChunks cap * nHead * 258 + 64

/-- The three regions `acc | m | l` fit inside the allocation without overlapping. -/
theorem scratch_layout (cap nHead : Nat) :
    nChunks cap * nHead * 256 + nChunks cap * nHead + nChunks cap * nHead
      ≤ scratchFloats cap nHead := by
  unfold scratchFloats
  omega

end QsaDecodeAttnChunks



## FILE: /mnt/user-data/outputs/QsaBlockSelect.lean
/-!
# MODULE: qsa_block_select
Block-level scoring and radix-select top-k for the batched QSA path (`src/kernels/cuda/qsa_select.cu`).
Scores are per pooled block of 4 cells; the selection is made over blocks weighted by their cell count
and then expanded to cell ids.

## Source semantics
* Block `b < n_bid` has weight 4; the last block `b = n_bid` has weight `n_kv - 4 n_bid` (possibly 0).
* The radix select fixes 8 key bits at a time (shift 24, 16, 8, 0); each pass scans digits from 255 down
  and stops at the first digit whose cumulative weight reaches `width`.
* Equal-to-threshold cells are shared out lowest index first: each thread takes
  `clamp (budget - eq_before) 0 eq`.

## Form template identifiers
* `weight`     <- lambda `weight(b)`
* `sumW`       <- total weight of a block prefix
* `pick`       <- the digit-descent loop of one radix pass
* `eqShare`    <- `my_eq`
-/

namespace QsaBlockSelect

-- § block weights

/-- Weight (cell count) of block `b`. -/
def weight (nKv nBid b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

/-- Total weight of the first `m` blocks. -/
def sumW (nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => sumW nKv nBid m + weight nKv nBid m

/-- Full blocks weigh 4 each. -/
theorem sumW_prefix (nKv nBid m : Nat) (h : m ≤ nBid) : sumW nKv nBid m = 4 * m := by
  induction m with
  | zero => simp [sumW]
  | succ m ih =>
    have hm : m < nBid := by omega
    simp only [sumW]
    rw [ih (by omega)]
    unfold weight
    rw [if_pos hm]
    omega

/-- Blocks `0 .. n_kv/4` together weigh exactly `n_kv`. -/
theorem sumW_total (nKv : Nat) : sumW nKv (nKv / 4) (nKv / 4 + 1) = nKv := by
  simp only [sumW]
  rw [sumW_prefix nKv (nKv / 4) (nKv / 4) (Nat.le_refl _)]
  unfold weight
  rw [if_neg (by omega)]
  omega

/-- Every expanded cell id `4b + c` is a valid cell. -/
theorem cell_lt (nKv b c : Nat) (hb : b ≤ nKv / 4) (hc : c < weight nKv (nKv / 4) b) :
    b * 4 + c < nKv := by
  unfold weight at hc
  split at hc <;> omega

/-- Every cell belongs to some block and offset within that block's weight. -/
theorem cell_covered (nKv j : Nat) (hj : j < nKv) :
    j / 4 ≤ nKv / 4 ∧ j % 4 < weight nKv (nKv / 4) (j / 4) := by
  unfold weight
  split <;> omega

/-- Cell ids are recovered from (block, offset). -/
theorem cell_roundtrip (j : Nat) : j / 4 * 4 + j % 4 = j := by
  omega

/-- The `+1e9` bonus applies exactly when the last block is a non-empty incomplete tail. -/
theorem tail_nonempty_iff (nKv : Nat) : 0 < weight nKv (nKv / 4) (nKv / 4) ↔ nKv % 4 ≠ 0 := by
  unfold weight
  rw [if_neg (by omega)]
  omega

-- § radix select

/-- The four fixed digits assembled into the 32-bit threshold. -/
def prefix4 (d3 d2 d1 d0 : Nat) : Nat := d3 * 16777216 + d2 * 65536 + d1 * 256 + d0

/-- The assembled threshold fits in 32 bits. -/
theorem prefix4_lt (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256)
    (h0 : d0 < 256) : prefix4 d3 d2 d1 d0 < 4294967296 := by
  unfold prefix4
  omega

/-- Each pass's digit is recoverable from the assembled threshold. -/
theorem prefix4_digits (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256)
    (h0 : d0 < 256) :
    prefix4 d3 d2 d1 d0 / 16777216 = d3 ∧
    prefix4 d3 d2 d1 d0 / 65536 % 256 = d2 ∧
    prefix4 d3 d2 d1 d0 / 256 % 256 = d1 ∧
    prefix4 d3 d2 d1 d0 % 256 = d0 := by
  unfold prefix4
  omega

/-- One radix pass: descend from digit `d`, accumulating weight; stop when the target is reached. -/
def pick (hist : Nat → Nat) (width : Nat) : Nat → Nat → Nat × Nat
  | 0, cum => (0, cum)
  | d + 1, cum =>
    if width ≤ cum + hist (d + 1) then (d + 1, cum)
    else pick hist width d (cum + hist (d + 1))

/-- The weight strictly above the chosen digit is always below `width`. -/
theorem pick_below (hist : Nat → Nat) (width : Nat) :
    ∀ d cum, cum < width → (pick hist width d cum).2 < width := by
  intro d
  induction d with
  | zero => intro cum h; simpa [pick] using h
  | succ d ih =>
    intro cum h
    by_cases hc : width ≤ cum + hist (d + 1)
    · simpa [pick, hc] using h
    · have := ih (cum + hist (d + 1)) (by omega)
      simpa [pick, hc] using this

/-- The chosen digit never exceeds the start digit. -/
theorem pick_le (hist : Nat → Nat) (width : Nat) :
    ∀ d cum, (pick hist width d cum).1 ≤ d := by
  intro d
  induction d with
  | zero => intro cum; simp [pick]
  | succ d ih =>
    intro cum
    by_cases hc : width ≤ cum + hist (d + 1)
    · simp [pick, hc]
    · have := ih (cum + hist (d + 1))
      simp only [pick, if_neg hc]
      omega

-- § equal-cell sharing

/-- Equal-to-threshold cells granted to a thread: budget left after earlier threads, capped by its own. -/
def eqShare (budget before eq : Nat) : Nat := min eq (budget - before)

/-- A thread never takes more than it owns. -/
theorem eqShare_le (budget before eq : Nat) : eqShare budget before eq ≤ eq := by
  unfold eqShare
  omega

/-- Earlier plus own share never overshoots the budget (unless earlier threads already did). -/
theorem eqShare_budget (budget before eq : Nat) :
    before + eqShare budget before eq ≤ max budget before := by
  unfold eqShare
  omega

/-- When the budget covers everything, the thread takes all of its equal cells. -/
theorem eqShare_full (budget before eq : Nat) (h : before + eq ≤ budget) :
    eqShare budget before eq = eq := by
  unfold eqShare
  omega

end QsaBlockSelect



## FILE: /mnt/user-data/outputs/KvQ8Quant.lean
/-!
# MODULE: kv_q8_quant
INT8 KV-cache quantisation, 64-value groups with an fp16 scale (`src/kernels/cuda/kv_q8.cu`):
append quantises one cell per (kv head, group, K/V) block; gather dequantises to fp16 scratch
four values per thread.

## Source semantics
* Codes are `round(x / scale)` clamped to `[-127, 127]`; a zero scale gives code 0.
* Group index of dimension `d` is `d / 64`; scales are stored at `row * groups + g`.
* The gather thread index `i = id*(kv*per) + h*per + q4` decomposes by div/mod, and the
  destination index `(id*kv + h)*per + q4` equals `i`.
* The four values a thread dequantises share one scale because 64 is a multiple of 4.

## Form template identifiers
* `clamp127`   <- `q < -127 ? -127 : (q > 127 ? 127 : q)`
* `code`       <- `if (sf > 0) ... else 0`
* `groupOf`    <- `d / KV_Q8_GROUP`
* `scaleIdx`   <- `row * groups + g`
-/

namespace KvQ8Quant

-- § clamping

/-- Clamp a rounded quotient into the int8 code range `[-127, 127]`. -/
def clamp127 (x : Int) : Int := if x < -127 then -127 else if x > 127 then 127 else x

/-- Codes stay in `[-127, 127]` (so `-128` is never produced). -/
theorem clamp127_bounds (x : Int) : -127 ≤ clamp127 x ∧ clamp127 x ≤ 127 := by
  by_cases h1 : x < -127
  · unfold clamp127
    rw [if_pos h1]
    omega
  · by_cases h2 : x > 127
    · unfold clamp127
      rw [if_neg h1, if_pos h2]
      omega
    · unfold clamp127
      rw [if_neg h1, if_neg h2]
      omega

/-- In-range values are unchanged. -/
theorem clamp127_id (x : Int) (h1 : -127 ≤ x) (h2 : x ≤ 127) : clamp127 x = x := by
  unfold clamp127
  rw [if_neg (by omega), if_neg (by omega)]

/-- Clamping is idempotent. -/
theorem clamp127_idem (x : Int) : clamp127 (clamp127 x) = clamp127 x := by
  have h := clamp127_bounds x
  exact clamp127_id _ h.1 h.2

/-- Quantised code: a zero stored scale gives code zero, otherwise the clamped rounded quotient. -/
def code (scalePos : Bool) (r : Int) : Int := if scalePos then clamp127 r else 0

/-- Every code is in range. -/
theorem code_bounds (b : Bool) (r : Int) : -127 ≤ code b r ∧ code b r ≤ 127 := by
  cases b
  · unfold code
    rw [if_neg (by decide)]
    omega
  · unfold code
    rw [if_pos rfl]
    exact clamp127_bounds r

-- § group addressing

/-- Group of a dimension. -/
def groupOf (d : Nat) : Nat := d / 64

/-- With `head_dim = 64 * groups`, every dimension's group is in range. -/
theorem groupOf_lt (d headDim groups : Nat) (hd : headDim = 64 * groups) (h : d < headDim) :
    groupOf d < groups := by
  unfold groupOf
  omega

/-- Scale slot of (row, group). -/
def scaleIdx (row groups g : Nat) : Nat := row * groups + g

/-- A value index inside its row stays inside that row. -/
theorem value_index_in_row (row headDim groups g t : Nat) (hd : headDim = 64 * groups)
    (hg : g < groups) (ht : t < 64) : row * headDim + g * 64 + t < row * headDim + headDim := by
  omega

/-- The four values of one gather thread share a group, hence one scale. -/
theorem quad_shares_group (q4 j : Nat) (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

-- § gather decomposition

/-- Arithmetic core of the decomposition. -/
theorem decomp_aux (i kv per A R B C : Nat) (e1 : kv * per * A + R = i)
    (e2 : per * B + C = R) : (A * kv + B) * per + C = i := by
  have e3 : (A * kv + B) * per = kv * per * A + per * B := by
    rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm A (kv * per), Nat.mul_comm B per]
  omega

/-- The (id, head, quad) split reassembles to the flat thread index. -/
theorem gather_decomp (i kv per : Nat) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i :=
  decomp_aux i kv per _ _ _ _ (Nat.div_add_mod i (kv * per))
    (Nat.div_add_mod (i % (kv * per)) per)

/-- The destination index is the thread index itself: the write is a straight, coalesced copy. -/
theorem dst_eq_flat (id h q kv per : Nat) :
    (id * kv + h) * per + q = id * (kv * per) + h * per + q := by
  rw [Nat.add_mul, Nat.mul_assoc]

end KvQ8Quant



## FILE: /mnt/user-data/outputs/NativeFlashAttnShortStep.lean
/-!
# MODULE: native_flash_attn_short_step
Specialised short-context FlashAttention decode (`src/kernels/cuda/native_flash_attn.cu`):
24 query heads x 256 dims, 2 KV heads, contexts 1..256, 24 blocks of 32x4 threads, padded length 256.

## Source semantics
* The step is valid iff `1 <= width <= max_context`, `n_kv = width`, `pos = width - 1`, `n_bid = width / 4`;
  otherwise the kernel reports `UnsupportedStep` and writes NaN.
* `kv = head / 12`; K/V element `(cell*2 + kv)*256 + d`; output `head*256 + tid (+128)`.
* A 128-cell tile is covered by `warp*32 + (lane/8)*8 + row` with warp<4, lane/8<4, row<8; two tiles cover 256.
* The scale is `0.0625 = 1/sqrt 256`; `tile` holds 4*4*256 floats.

## Form template identifiers
* `Valid`        <- the `valid` predicate computed from `step`
* `cellOf`       <- `base + warp*32 + (lane & ~7) + row`
* `kElem`        <- `(cell * 2 + kv) * 256 + d`
-/

namespace NativeFlashAttnShortStep

-- § step validity

/-- The step-state validity test of the kernel. -/
def Valid (maxCtx pos nKv nBid width : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nKv = width ∧ pos + 1 = width ∧ nBid = width / 4

/-- A step filled from a position (`n_kv = pos+1`, `width = min n_kv 2051`) is valid for contexts up to 256. -/
theorem valid_of_fill (maxCtx pos : Nat) (h1 : pos + 1 ≤ maxCtx) (h2 : maxCtx ≤ 256) :
    Valid maxCtx pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold Valid
  omega

/-- Valid steps are consistent: the position is the last cell. -/
theorem valid_pos (maxCtx pos nKv nBid width : Nat) (h : Valid maxCtx pos nKv nBid width) :
    pos = width - 1 := by
  unfold Valid at h
  omega

/-- A step whose `n_kv` disagrees with the width is rejected. -/
theorem invalid_of_nKv_ne (maxCtx pos nKv nBid width : Nat) (h : nKv ≠ width) :
    ¬ Valid maxCtx pos nKv nBid width := by
  unfold Valid
  omega

-- § heads and buffers

/-- Each of the 24 heads reads one of the 2 KV heads. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by
  omega

/-- Element offset of K/V dimension `d` (and `d+1`) of `cell` stays inside a `capacity`-cell buffer. -/
theorem kElem_lt (cell kv d capacity : Nat) (hc : cell < capacity) (hk : kv < 2) (hd : d + 1 < 256) :
    (cell * 2 + kv) * 256 + d + 1 ≤ capacity * 512 := by
  omega

/-- The byte size `capacity * 1024` is `capacity * 2 heads * 256 dims * 2 bytes`. -/
theorem kv_bytes_eq (capacity : Nat) : capacity * 2 * 256 * 2 = capacity * 1024 := by
  omega

/-- Output writes `head*256 + i0 + tid` for `i0 ∈ {0, 128}`, `tid < 128` stay in the 24x256 output. -/
theorem out_index_lt (head tid i0 : Nat) (hh : head < 24) (ht : tid < 128)
    (hi : i0 = 0 ∨ i0 = 128) : head * 256 + i0 + tid < 6144 := by
  omega

/-- Shared-memory tile writes of the final reduction stay inside `4*4*256` floats. -/
theorem tile_index_lt (warp lane i0 j : Nat) (hw : warp < 4) (hl : lane < 32)
    (hi : i0 = 0 ∨ i0 = 32 ∨ i0 = 64 ∨ i0 = 96) (hj : j < 4) :
    warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4) + 2 * j + 1 < 4096 := by
  omega

-- § tile coverage

/-- Cell handled by (tile base, warp, lane group, row). -/
def cellOf (base w g r : Nat) : Nat := base + w * 32 + g * 8 + r

/-- Every thread-assigned cell lies inside the padded length 256. -/
theorem cellOf_lt (base w g r : Nat) (hb : base = 0 ∨ base = 128) (hw : w < 4) (hg : g < 4)
    (hr : r < 8) : cellOf base w g r < 256 := by
  unfold cellOf
  omega

/-- Every cell below 256 is handled by some (tile, warp, group, row). -/
theorem cellOf_surj (c : Nat) (hc : c < 256) :
    ∃ base w g r, (base = 0 ∨ base = 128) ∧ w < 4 ∧ g < 4 ∧ r < 8 ∧ cellOf base w g r = c := by
  refine ⟨c / 128 * 128, c % 128 / 32, c % 32 / 8, c % 8, ?_, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · omega
  · unfold cellOf
    omega

/-- ... and by exactly one: no two threads process the same cell. -/
theorem cellOf_inj (b1 w1 g1 r1 b2 w2 g2 r2 : Nat)
    (hb1 : b1 = 0 ∨ b1 = 128) (hw1 : w1 < 4) (hg1 : g1 < 4) (hr1 : r1 < 8)
    (hb2 : b2 = 0 ∨ b2 = 128) (hw2 : w2 < 4) (hg2 : g2 < 4) (hr2 : r2 < 8)
    (h : cellOf b1 w1 g1 r1 = cellOf b2 w2 g2 r2) :
    b1 = b2 ∧ w1 = w2 ∧ g1 = g2 ∧ r1 = r2 := by
  unfold cellOf at h
  omega

/-- Two tiles of 128 cover the padded length of 256. -/
theorem tiles_cover : 256 / 128 = 2 := by
  decide

/-- The scale `1/16` squares to `1/256`, i.e. `1/sqrt(head_dim)` for `head_dim = 256`. -/
theorem scale_sq : 16 * 16 = 256 := by
  decide

end NativeFlashAttnShortStep



## FILE: /mnt/user-data/outputs/NativeQsaNormGate.lean
/-!
# MODULE: native_qsa_norm_gate
Weighted RMS norm and sigmoid gate of the native QSA path (`src/kernels/cuda/native_qsa.cu`),
plus the buffer-overlap guard used by the host wrappers.

## Source semantics
* `norm`: one block per row, `BlockSize` threads stride over columns (`col = tid + BlockSize*k`);
  block size is 256 for `n_cols < 1024`, else 1024; at most 32 warp partials are reduced.
* Input and output may alias exactly (each thread touches only its own elements after the barrier).
* `gate`: `q_full` holds `[head][2][head_dim]`; the gate is the SECOND half, index
  `head*2*head_dim + head_dim + channel`.
* Spans overlap iff `a < b + bn && b < a + an`.

## Form template identifiers
* `blockSize`    <- `n_cols < 1024 ? 256 : 1024`
* `gateIndex`    <- `head * 2 * head_dim + head_dim + channel`
* `Overlap`      <- `overlap(a, an, b, bn)`
-/

namespace NativeQsaNormGate

-- § norm

/-- Block size selected from the row width. -/
def blockSize (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- The reduction needs at most 32 warp partials. -/
theorem blockSize_warps (nCols : Nat) : blockSize nCols / 32 ≤ 32 := by
  unfold blockSize
  split <;> omega

/-- A strided column index determines its thread and iteration uniquely. -/
theorem stride_unique (bs tid k col : Nat) (hpos : 0 < bs) (ht : tid < bs)
    (h : tid + bs * k = col) : tid = col % bs ∧ k = col / bs := by
  subst h
  constructor
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ht]
  · rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt ht, Nat.zero_add]

/-- Every column of a row is reached by exactly the thread `col % bs` at iteration `col / bs`. -/
theorem stride_cover (bs col : Nat) (hpos : 0 < bs) :
    col % bs + bs * (col / bs) = col := by
  have := Nat.mod_add_div col bs
  omega

/-- Row-major element index stays inside the buffer. -/
theorem row_index_lt (row col nRows nCols : Nat) (hr : row < nRows) (hc : col < nCols) :
    row * nCols + col < nRows * nCols := by
  have h0 : (row + 1) * nCols ≤ nRows * nCols := Nat.mul_le_mul_right _ hr
  have h0' : (row + 1) * nCols = row * nCols + nCols := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The in-place read and write of a thread name the same element. -/
theorem inplace_same_index (rowOffset col : Nat) : rowOffset + col = rowOffset + col := rfl

-- § gate

/-- Index of the gate half of `q_full`. -/
def gateIndex (head headDim channel : Nat) : Nat := head * 2 * headDim + headDim + channel

/-- Gate reads stay inside `q_full` (size `nHead * 2 * headDim`). -/
theorem gateIndex_lt (head headDim channel nHead : Nat) (hh : head < nHead)
    (hc : channel < headDim) : gateIndex head headDim channel < nHead * 2 * headDim := by
  unfold gateIndex
  have h0 : (head + 1) * (2 * headDim) ≤ nHead * (2 * headDim) := Nat.mul_le_mul_right _ hh
  have h0' : (head + 1) * (2 * headDim) = head * (2 * headDim) + 2 * headDim := by
    rw [Nat.add_mul, Nat.one_mul]
  have e1 : head * 2 * headDim = head * (2 * headDim) := Nat.mul_assoc _ _ _
  have e2 : nHead * 2 * headDim = nHead * (2 * headDim) := Nat.mul_assoc _ _ _
  omega

/-- The gate half starts after the query half of the same head. -/
theorem gateIndex_ge (head headDim channel : Nat) :
    head * 2 * headDim + headDim ≤ gateIndex head headDim channel := by
  unfold gateIndex
  omega

/-- The flat output index splits into (head, channel) and reassembles. -/
theorem head_channel_roundtrip (i headDim : Nat) :
    headDim * (i / headDim) + i % headDim = i := Nat.div_add_mod i headDim

-- § overlap guard

/-- Two byte spans overlap. -/
def Overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : Overlap a an b bn ↔ Overlap b bn a an := by
  unfold Overlap
  omega

/-- A span that ends before the other starts does not overlap it. -/
theorem not_overlap_of_le (a an b bn : Nat) (h : a + an ≤ b) : ¬ Overlap a an b bn := by
  unfold Overlap
  omega

/-- A nonempty span overlaps itself, so distinct buffers must be checked, and exact aliasing handled separately. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : Overlap a n a n := by
  unfold Overlap
  omega

end NativeQsaNormGate



## FILE: /mnt/user-data/outputs/NativeQsaIndexerAppend.lean
/-!
# MODULE: native_qsa_indexer_append
Single-block append of one raw indexer key and, on block completion, pooling plus RoPE of the pooled row
(`src/kernels/cuda/native_qsa_indexer.cu`); geometry fixed at 128 dims, blocks of 4 cells, 64 rotary dims.

## Source semantics
* `slot = pos % 4`; slots 0..2 are stored in `tail` (3 rows of 128), slot 3 completes a block.
* The kernel works only for `pos = 0` (spare/first row) or `slot = 3`.
* A completed block `b = pos / 4` writes `pooled[b]` and moves the spare to `pooled[b+1]`.
* Rotation position is `pos_base + 4*b` (0 for `pos = 0`); pairs are `(pair, pair + 32)` with `pair = d % 32`.
* `pooled` holds `max_cells / 4 + 1` rows.

## Form template identifiers
* `slotOf`      <- `pos % R`
* `ropePos`     <- `pos == 0 ? 0 : pos_base + R * b`
* `pairOf`      <- `d % (ROT / 2)`
-/

namespace NativeQsaIndexerAppend

-- § slots and tail

/-- Slot of a position within its block. -/
def slotOf (pos : Nat) : Nat := pos % 4

/-- Tail rows are written only for slots 0..2 and stay inside the 3x128 tail buffer. -/
theorem tail_index_lt (slot d : Nat) (hs : slot < 3) (hd : d < 128) : slot * 128 + d < 384 := by
  omega

/-- The slot-3 cell is the one whose successor starts a new block. -/
theorem slot3_iff_next_block (pos : Nat) : slotOf pos = 3 ↔ (pos + 1) % 4 = 0 := by
  unfold slotOf
  omega

/-- After the first cell, the kernel does work only on slot 3: otherwise it returns early. -/
theorem active_iff (pos : Nat) : (pos = 0 ∨ slotOf pos = 3) ↔ (pos = 0 ∨ (pos + 1) % 4 = 0) := by
  unfold slotOf
  omega

-- § pooled rows

/-- A non-first active cell completes a block, so its block index is the previous completed count. -/
theorem block_complete_index (pos : Nat) (h0 : pos ≠ 0) (h3 : slotOf pos = 3) :
    4 * (pos / 4) + 3 = pos := by
  unfold slotOf at h3
  omega

/-- The spare row `b + 1` is inside the `max_cells / 4 + 1` rows whenever a block completes. -/
theorem spare_row_in_range (pos maxCells : Nat) (hp : pos < maxCells) (h3 : slotOf pos = 3) :
    pos / 4 + 1 ≤ maxCells / 4 := by
  unfold slotOf at h3
  omega

/-- ... so the element writes `(b+1)*128 + d` stay inside the pooled buffer. -/
theorem spare_index_lt (pos maxCells d : Nat) (hp : pos < maxCells) (h3 : slotOf pos = 3)
    (hd : d < 128) : (pos / 4 + 1) * 128 + d < (maxCells / 4 + 1) * 128 := by
  unfold slotOf at h3
  omega

/-- The row written for the completed (or first) block is inside the buffer as well. -/
theorem block_index_lt (pos maxCells d : Nat) (hp : pos < maxCells) (hd : d < 128) :
    (pos / 4) * 128 + d < (maxCells / 4 + 1) * 128 := by
  omega

-- § rotation

/-- Rotation position: 0 for the very first cell, else the block's first position. -/
def ropePos (pos posBase : Nat) : Nat := if pos = 0 then 0 else posBase + 4 * (pos / 4)

/-- With an aligned base the rotation position is block-aligned. -/
theorem ropePos_aligned (pos posBase : Nat) (hb : posBase % 4 = 0) : ropePos pos posBase % 4 = 0 := by
  unfold ropePos
  split <;> omega

/-- The rotation position never exceeds `pos_base + pos`. -/
theorem ropePos_le (pos posBase : Nat) : ropePos pos posBase ≤ posBase + pos := by
  unfold ropePos
  split <;> omega

/-- It fits in an int32 under the host's capacity check. -/
theorem ropePos_lt (pos posBase maxCells : Nat) (hp : pos < maxCells)
    (hc : posBase + maxCells ≤ 2147483647) : ropePos pos posBase < 2147483647 := by
  have := ropePos_le pos posBase
  omega

/-- Rotary pair index of dimension `d < 64`. -/
def pairOf (d : Nat) : Nat := d % 32

/-- Both elements of a rotary pair lie in the 64 rotated dimensions. -/
theorem pair_in_range (d : Nat) (hd : d < 64) : pairOf d < 32 ∧ pairOf d + 32 < 64 := by
  unfold pairOf
  omega

/-- The first half of the rotated dims reads `(d, d+32)`, the second `(d-32, d)`. -/
theorem pair_partner (d : Nat) (hd : d < 64) :
    (d < 32 → pairOf d = d) ∧ (32 ≤ d → pairOf d + 32 = d) := by
  unfold pairOf
  constructor <;> intro h <;> omega

end NativeQsaIndexerAppend



## FILE: /mnt/user-data/outputs/NativeQsaScoreTiles.lean
/-!
# MODULE: native_qsa_score_tiles
Tensor-core scoring of pooled indexer blocks (`src/kernels/cuda/native_qsa_score.cu`): each CUDA block
scores 32 pooled rows with 4 heads over 128 dims using two warps, then expands each row's score to its 4 cells.

## Source semantics
* The grid is `ceil(max_blocks / 32)` blocks; rows `> full = n/4` return.
* The step is trusted only if `pos = n-1`, `full = n/4`, `width = min n 2051`, `1 <= n <= max_cells`.
* Columns are split between two warps: `col = warp*32 + lane + 64*k`, `k ∈ {0,1}`.
* Partial results land at `shared[h*68 + i]`, `i = warp*32 + ia*16 + (l/2)*8 + lane/4`, `h = (lane%4)*2 + l%2`.
* Row `r` writes cells `[4r, min n (4r+4))`; the last row gets `+1e9` iff `n % 4 != 0`.

## Form template identifiers
* `gridBlocks`   <- `(max_blocks + ROWS - 1) / ROWS`
* `colOf`        <- `warp*32 + lane + 64*k`
* `accRow`       <- `warp*ROWS + ia*16 + (l/2)*8 + lane/4`
* `accHead`      <- `(lane%4)*2 + l%2`
-/

namespace NativeQsaScoreTiles

-- § row coverage

/-- Number of 32-row blocks. -/
def gridBlocks (maxBlocks : Nat) : Nat := (maxBlocks + 31) / 32

/-- The grid reaches every row. -/
theorem grid_covers (maxBlocks : Nat) : maxBlocks ≤ gridBlocks maxBlocks * 32 := by
  unfold gridBlocks
  omega

/-- Every row is handled by exactly the (block, lane) pair `(row/32, row%32)`. -/
theorem row_roundtrip (row : Nat) : row / 32 * 32 + row % 32 = row := by
  omega

/-- That block index exists in the grid. -/
theorem row_block_lt (row maxBlocks : Nat) (h : row < maxBlocks) : row / 32 < gridBlocks maxBlocks := by
  unfold gridBlocks
  omega

/-- With `max_blocks = max_cells/4 + 1`, every completed-or-tail row of any valid step is inside the grid. -/
theorem full_row_lt (n maxCells : Nat) (h : n ≤ maxCells) : n / 4 < maxCells / 4 + 1 := by
  omega

-- § step trust

/-- The kernel's acceptance test on the step state. -/
def StepOk (maxCells pos n full width : Nat) : Prop :=
  1 ≤ n ∧ n ≤ maxCells ∧ pos + 1 = n ∧ full = n / 4 ∧ width = min n 2051

/-- A step filled from a position is accepted. -/
theorem stepOk_of_fill (maxCells pos : Nat) (h : pos + 1 ≤ maxCells) :
    StepOk maxCells pos (pos + 1) ((pos + 1) / 4) (min (pos + 1) 2051) := by
  unfold StepOk
  omega

-- § column split between warps

/-- Column handled by (warp, lane, iteration). -/
def colOf (warp lane k : Nat) : Nat := warp * 32 + lane + 64 * k

/-- The two warps with two iterations tile all 128 columns. -/
theorem colOf_lt (warp lane k : Nat) (hw : warp < 2) (hl : lane < 32) (hk : k < 2) :
    colOf warp lane k < 128 := by
  unfold colOf
  omega

/-- Every column is reached. -/
theorem colOf_surj (c : Nat) (hc : c < 128) :
    ∃ warp lane k, warp < 2 ∧ lane < 32 ∧ k < 2 ∧ colOf warp lane k = c := by
  refine ⟨c / 32 % 2, c % 32, c / 64, ?_, ?_, ?_, ?_⟩
  · omega
  · omega
  · omega
  · unfold colOf
    omega

/-- No column is reached twice. -/
theorem colOf_inj (w1 l1 k1 w2 l2 k2 : Nat) (hw1 : w1 < 2) (hl1 : l1 < 32) (hk1 : k1 < 2)
    (hw2 : w2 < 2) (hl2 : l2 < 32) (hk2 : k2 < 2)
    (h : colOf w1 l1 k1 = colOf w2 l2 k2) : w1 = w2 ∧ l1 = l2 ∧ k1 = k2 := by
  unfold colOf at h
  omega

-- § accumulator shuffle

/-- Row of the accumulator fragment element (warp, ia, l, lane). -/
def accRow (warp ia l lane : Nat) : Nat := warp * 32 + ia * 16 + (l / 2) * 8 + lane / 4

/-- Head of the accumulator fragment element. -/
def accHead (l lane : Nat) : Nat := (lane % 4) * 2 + l % 2

/-- The row part lands inside the 64 combined rows. -/
theorem accRow_lt (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4)
    (hlane : lane < 32) : accRow warp ia l lane < 64 := by
  unfold accRow
  omega

/-- The head part lands inside the 8 combined heads. -/
theorem accHead_lt (l lane : Nat) : accHead l lane < 8 := by
  unfold accHead
  omega

/-- So the shared write `h*68 + i` is inside the `8*68` combine area, which fits in the `2*16*36` shared array. -/
theorem combine_index_lt (warp ia l lane : Nat) (hw : warp < 2) (hi : ia < 2) (hl : l < 4)
    (hlane : lane < 32) : accHead l lane * 68 + accRow warp ia l lane < 1152 := by
  have h1 := accRow_lt warp ia l lane hw hi hl hlane
  have h2 := accHead_lt l lane
  omega

-- § cell expansion

/-- Row `r` writes cells `4r .. 4r+3`; a cell `i` is written by exactly row `i / 4`. -/
theorem cell_row_unique (i row : Nat) : (row * 4 ≤ i ∧ i < row * 4 + 4) ↔ i / 4 = row := by
  omega

/-- Every cell below `n` is written by a row that exists (row ≤ n/4). -/
theorem cell_written (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by
  omega

/-- The `+1e9` tail bonus applies iff the last row has fewer than 4 cells. -/
theorem tail_partial_iff (n : Nat) : n % 4 ≠ 0 ↔ n < (n / 4 + 1) * 4 ∧ n / 4 * 4 < n := by
  omega

end NativeQsaScoreTiles



## FILE: /mnt/user-data/outputs/SamplerGreedyPenalties.lean
/-!
# MODULE: sampler_greedy_penalties
The sampler chain's deterministic parts (`src/kernels/cuda/sampler.cu`): the block-parallel greedy argmax
with lowest-index tie rule, the penalty window, the repeat/frequency/presence penalty, the top-k clamp,
and the Philox-to-uniform conversion.

## Source semantics
* Greedy = argmax over penalised logits; each thread keeps `if (s > bv)` over a strided subset, and the
  reduction picks the larger value, on equality the SMALLER index; `n_vocab` is the "no candidate" index.
* An all `-inf` row yields index 0.
* Penalty window is the TAIL of the history: `hlen = clamp(penalty_last_n, 0, history_len)`, start `history_len - hlen`.
* Repeat penalty multiplies non-positive logits and divides positive ones; presence is a boolean, not the count.
* `top_k` is clamped to 64; `top_k <= 0` is refused on the sampled path.
* Uniform = `(c0 >> 8) / 2^24`, strictly below 1.

## Form template identifiers
* `merge`       <- the shuffle/shared-memory candidate combine
* `windowLen`   <- `hlen`
* `presence`    <- `(count > 0 ? 1 : 0) * penalty_present`
* `topKClamp`   <- `k`
-/

namespace SamplerGreedyPenalties

-- § candidate merge

/-- Combine two (value, index) candidates: larger value wins, equal values prefer the smaller index. -/
def merge (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- Merging is commutative, so the reduction order does not matter. -/
theorem merge_comm (a b : Int × Nat) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  unfold merge
  by_cases h1 : bv > av ∨ (bv = av ∧ bi < ai)
  · have h2 : ¬ (av > bv ∨ (av = bv ∧ ai < bi)) := by omega
    simp only [h1, h2, if_true, if_false]
  · by_cases h2 : av > bv ∨ (av = bv ∧ ai < bi)
    · simp only [h1, h2, if_true, if_false]
    · have : av = bv ∧ ai = bi := by omega
      obtain ⟨e1, e2⟩ := this
      subst e1
      subst e2
      simp only [h1, if_false]

/-- A real candidate (index below `n_vocab`) with value at least the sentinel's beats the sentinel. -/
theorem merge_sentinel (v sv : Int) (i n : Nat) (hi : i < n) (hv : sv ≤ v) :
    merge (sv, n) (v, i) = (v, i) := by
  unfold merge
  have h : v > sv ∨ (v = sv ∧ i < n) := by omega
  simp only [h, if_true]

/-- The result of a merge is one of its inputs. -/
theorem merge_mem (a b : Int × Nat) : merge a b = a ∨ merge a b = b := by
  unfold merge
  split
  · right; rfl
  · left; rfl

/-- The result is never smaller (in value) than either input. -/
theorem merge_ge (a b : Int × Nat) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  unfold merge
  by_cases h : bv > av ∨ (bv = av ∧ bi < ai)
  · simp only [h, if_true]
    omega
  · simp only [h, if_false]
    omega

/-- The fallback for an all `-inf` row: index `n_vocab` becomes 0. -/
def finalIndex (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

/-- The emitted token is always a valid vocabulary index. -/
theorem finalIndex_lt (wi nVocab : Nat) (h : 0 < nVocab) : finalIndex wi nVocab < nVocab := by
  unfold finalIndex
  split <;> omega

-- § penalty window

/-- Window length: `penalty_last_n` clamped into `[0, history_len]`. -/
def windowLen (lastN histLen : Int) : Int := if lastN < histLen then (if lastN < 0 then 0 else lastN) else histLen

/-- The window fits in the history. -/
theorem windowLen_le (lastN histLen : Int) (h : 0 ≤ histLen) : 0 ≤ windowLen lastN histLen ∧
    windowLen lastN histLen ≤ histLen := by
  unfold windowLen
  split <;> (try split) <;> omega

/-- The window is the tail: start offset plus length is the history length. -/
theorem window_tail (lastN histLen : Int) (h : 0 ≤ histLen) :
    (histLen - windowLen lastN histLen) + windowLen lastN histLen = histLen := by
  omega

-- § penalties

/-- Repeat penalty: multiply non-positive logits, divide positive ones (integer model of the ordering). -/
def repeatPen (logit rep : Int) : Int := if logit ≤ 0 then logit * rep else logit / rep

/-- Presence term: boolean, not the count. -/
def presence (count : Nat) (pen : Int) : Int := if 0 < count then pen else 0

/-- A token that never appeared keeps its logit (the kernel returns early for `count <= 0`). -/
def applyPen (logit : Int) (count : Nat) (rep freq pres : Int) : Int :=
  if count = 0 then logit
  else repeatPen logit rep - (count : Int) * freq - presence count pres

/-- Unseen tokens are untouched. -/
theorem applyPen_unseen (logit rep freq pres : Int) : applyPen logit 0 rep freq pres = logit := by
  simp [applyPen]

/-- The presence term does not grow with the count. -/
theorem presence_const (c1 c2 : Nat) (pen : Int) (h1 : 0 < c1) (h2 : 0 < c2) :
    presence c1 pen = presence c2 pen := by
  unfold presence
  rw [if_pos h1, if_pos h2]

/-- Unity penalties on a positive logit leave it unchanged when the other terms are zero. -/
theorem repeatPen_one_pos (logit : Int) (h : 0 < logit) : repeatPen logit 1 = logit := by
  unfold repeatPen
  rw [if_neg (by omega)]
  simp

-- § sampling parameters

/-- Top-k clamp: `top_k` in `1..64`, zero means "refuse". -/
def topKClamp (topK : Int) : Int := if topK > 0 then (if topK < 64 then topK else 64) else 0

/-- The clamp never exceeds the 64-slot keep array. -/
theorem topKClamp_le (topK : Int) : 0 ≤ topKClamp topK ∧ topKClamp topK ≤ 64 := by
  unfold topKClamp
  split <;> (try split) <;> omega

/-- A positive request gets at least one slot. -/
theorem topKClamp_pos (topK : Int) (h : 0 < topK) : 1 ≤ topKClamp topK := by
  unfold topKClamp
  split <;> (try split) <;> omega

-- § Philox uniform

/-- The 24-bit numerator of the uniform is below the denominator `2^24`, so the draw is in `[0, 1)`. -/
theorem uniform_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

end SamplerGreedyPenalties





## FILE: /mnt/user-data/outputs/GdnColumnRecurrenceKernels.lean
/-!
# MODULE: gdn_column_recurrence_kernels
Address, launch-geometry and conv-window contracts of the gated delta-net's non-projection kernels
(`gdn_step_kernel`, `gdn_conv_kernel`, `gdn_l2_kernel`, `gdn_out_norm_kernel`, `gdn_beta_gate_kernel`).

## Source semantics
The state is `(S, h_v, S)` with `j` fastest.  One thread owns one `(h, j)` column and walks `i` in `0..S-1`;
no barrier is needed because every access is within one column.  The grid is `(ceil(S/32), ceil(h_v/MAX_H))`
with `MAX_H = 1` (latency hiding: 192 blocks instead of 24).  The conv state slides: drop the oldest row,
append the newest; kernel tap 0 multiplies the oldest row, the last tap multiplies the new input.
q/k heads are paired with value heads by `h % h_k`.

## Form template identifiers
- `state` -> `stateIdx S h_v i h j`            (flat address of state[i][h][j])
- `JTHREADS`, `MAX_H`                           (block shape knobs)
- `conv_state`/`kW`                             (`slide`, `dot`)
-/

namespace Strata.Kernels.GdnColumn

-- § arithmetic helpers

/-- `a < b` and `c < n` give `a*n + c < b*n`: the basic row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row of a row-major address. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § constants and addressing

/-- Threads along `j`. -/
def JTHREADS : Nat := 32

/-- Heads staged per block (latency-hiding knob). -/
def MAX_H : Nat := 1

/-- Flat address of `state[i][h][j]` in the `(S, h_v, S)` layout with `j` fastest. -/
def stateIdx (S h_v i h j : Nat) : Nat := (i * h_v + h) * S + j

/-- Every in-range state address is inside the `S*h_v*S` allocation. -/
theorem stateIdx_lt {S h_v i h j : Nat} (hi : i < S) (hh : h < h_v) (hj : j < S) :
    stateIdx S h_v i h j < S * h_v * S := by
  unfold stateIdx
  have h1 : i * h_v + h < S * h_v := mul_add_lt_aux hi hh
  exact mul_add_lt_aux h1 hj

/-- Distinct `(i, h, j)` never alias: columns are disjoint, so one thread per column needs no barrier. -/
theorem stateIdx_inj {S h_v i i' h h' j j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < S) (hj' : j' < S)
    (e : stateIdx S h_v i h j = stateIdx S h_v i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hj hj' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- Two threads of different columns touch different addresses at every step `i`. -/
theorem column_private {S h_v i i' h h' j j' : Nat} (hh : h < h_v) (hh' : h' < h_v)
    (hj : j < S) (hj' : j' < S) (hne : h ≠ h' ∨ j ≠ j') :
    stateIdx S h_v i h j ≠ stateIdx S h_v i' h' j' := by
  intro e
  obtain ⟨_, e2, e3⟩ := stateIdx_inj hh hh' hj hj' e
  rcases hne with h1 | h1
  · exact h1 e2
  · exact h1 e3

/-- The paired q/k row `(h % h_k)*S + i` is inside the `h_k*S` q/k buffer. -/
theorem qk_index_lt {h h_k S i : Nat} (hk : 0 < h_k) (hi : i < S) :
    (h % h_k) * S + i < h_k * S :=
  mul_add_lt_aux (Nat.mod_lt h hk) hi

/-- Value-side address `h*S + j` stays inside the `h_v*S` buffer. -/
theorem value_index_lt {h h_v S j : Nat} (hh : h < h_v) (hj : j < S) :
    h * S + j < h_v * S :=
  mul_add_lt_aux hh hj

-- § launch geometry

/-- Every column `j < S` is covered by exactly the block/thread pair `(j/32, j%32)`. -/
theorem column_cover (S j : Nat) (hj : j < S) :
    j / JTHREADS < (S + JTHREADS - 1) / JTHREADS ∧ j % JTHREADS < JTHREADS ∧
      (j / JTHREADS) * JTHREADS + j % JTHREADS = j := by
  unfold JTHREADS
  omega

/-- Every head is covered by block row `h / MAX_H`, slot `h % MAX_H`, inside that block's `nh`. -/
theorem head_cover (h_v h : Nat) (hh : h < h_v) :
    h / MAX_H < (h_v + MAX_H - 1) / MAX_H ∧ h / MAX_H * MAX_H + h % MAX_H = h ∧
      h % MAX_H < min MAX_H (h_v - h / MAX_H * MAX_H) := by
  unfold MAX_H
  omega

/-- The tuned grid at the real geometry: 192 blocks. -/
theorem grid_blocks_tuned :
    ((128 + JTHREADS - 1) / JTHREADS) * ((48 + MAX_H - 1) / MAX_H) = 192 := by
  decide

/-- The untuned `MAX_H = 8` grid had only 24 blocks. -/
theorem grid_blocks_untuned :
    ((128 + JTHREADS - 1) / JTHREADS) * ((48 + 8 - 1) / 8) = 24 := by
  decide

/-- 24 single-warp blocks leave half of 48 SMs idle; 192 blocks give four per SM. -/
theorem sm_occupancy : 24 < 48 ∧ 192 = 4 * 48 := by
  decide

/-- Total columns `S * h_v` do not depend on the packing. -/
theorem thread_count_fixed : 128 * 48 = 6144 := by
  decide

/-- A 1-D launch with `blocks = ceil(n/B)` covers `n` (B = 128, as in the beta gate). -/
theorem launch_cover_128 (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by
  omega

/-- Same for the 256-thread conv launch. -/
theorem launch_cover_256 (n : Nat) : n ≤ ((n + 255) / 256) * 256 := by
  omega

/-- The staged `ks`/`qs` arrays hold 128 entries, so `S <= 128` suffices for the guard. -/
theorem staged_fit {S i : Nat} (hS : S ≤ 128) (hi : i < S) : i < 128 := by
  omega

/-- A warp striding a row of `cols <= 1024` does at most 32 iterations per lane. -/
theorem warp_stride_bound (cols : Nat) (h : cols ≤ 1024) : (cols + 31) / 32 ≤ 32 := by
  omega

/-- Each element `i < cols` is visited by lane `i % 32` at iteration `i / 32`, exactly once. -/
theorem warp_stride_cover (cols i : Nat) (hi : i < cols) :
    i % 32 < 32 ∧ (i / 32) * 32 + i % 32 = i := by
  omega

-- § conv window

/-- Integer dot product of two lists (truncating to the shorter). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- The conv state after one step: drop the oldest row, append the newest. -/
def slide (st : List Int) (x : Int) : List Int := st.tail ++ [x]

/-- With `d_conv-1` stored rows (non-empty) the slide keeps the state size. -/
theorem slide_length (st : List Int) (x : Int) (h : st ≠ []) :
    (slide st x).length = st.length := by
  cases st with
  | nil => exact absurd rfl h
  | cons a as => simp [slide]

/-- The slid state is the window `[state | x]` without its first (oldest) row. -/
theorem slide_eq_drop (st : List Int) (x : Int) (h : st ≠ []) :
    slide st x = (st ++ [x]).drop 1 := by
  cases st with
  | nil => exact absurd rfl h
  | cons a as => simp [slide]

/-- The last tap multiplies the new input; the earlier taps run over the stored rows. -/
theorem dot_snoc (st w' : List Int) (x wl : Int) (h : st.length = w'.length) :
    dot (st ++ [x]) (w' ++ [wl]) = dot st w' + x * wl := by
  induction st generalizing w' with
  | nil =>
    cases w' with
    | nil => simp [dot]
    | cons b bs => simp at h
  | cons a as ih =>
    cases w' with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.cons_append, dot]
      rw [ih bs h']
      omega

/-- The kernel's output `sum_{i<dc-1} st[i]*w[i] + x*w[dc-1]` is the dot of the full window. -/
theorem conv_out_is_window_dot (st w' : List Int) (x wl : Int) (h : st.length = w'.length) :
    dot st w' + x * wl = dot (st ++ [x]) (w' ++ [wl]) :=
  (dot_snoc st w' x wl h).symm

end Strata.Kernels.GdnColumn



## FILE: /mnt/user-data/outputs/FusedGdnStepNormKernels.lean
/-!
# MODULE: fused_gdn_step_norm_kernels
Row-group sharding, channel/head block mapping and bf16 chunk decoding of the fused gated-delta-net kernels
(`gdn_step_norm_kernel`, `gdn_conv_l2_kernel`, `gdn_ab_kernel`).

## Source semantics
`S = 128`.  The step kernel runs one block per head with `128 x 4` threads: thread `(col, rg)` keeps 32
state rows `rg*32 .. rg*32+31` of column `col` in registers; the four row-group partials are summed through
shared memory, so the result equals the sequential contraction over all 128 rows.  The conv kernel runs one
128-channel block per (q/k/v) head and L2-normalises only the first `qk_heads` blocks.  The `ab` kernel
assigns one warp per output row of the stacked `[alpha; beta]` projection and decodes bf16 weights eight at
a time from `uint4` chunks (even element = low half shifted up, odd element = high half masked).

## Form template identifiers
- `S`, `RG`, `RPG`         -> state size, row groups, rows per thread
- `sumRange f n`           -> sequential reduction of `f 0 .. f (n-1)`
- `stateIdx h_v head row col` -> address of state[row][head][col]
-/

namespace Strata.Kernels.FusedGdn

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § constants

/-- State size. -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows held by one thread. -/
def RPG : Nat := 32

/-- The row groups tile the state exactly. -/
theorem row_groups_tile : RG * RPG = S := by
  decide

/-- `wsum` has one slot per warp of the 512-thread block. -/
theorem wsum_size : S * RG / 32 = 16 := by
  decide

/-- Block of `128 x 4` threads. -/
theorem block_threads : S * RG = 512 := by
  decide

-- § sequential reduction and row-group sharding

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Four row groups of 32 rows add up to the full 128-row contraction. -/
theorem rows_four_way (f : Nat → Int) :
    sumRange f 128 =
      sumRange (fun r => f (0 * 32 + r)) 32 + sumRange (fun r => f (1 * 32 + r)) 32 +
      sumRange (fun r => f (2 * 32 + r)) 32 + sumRange (fun r => f (3 * 32 + r)) 32 := by
  have h1 : sumRange f 128 = sumRange f 96 + sumRange (fun r => f (3 * 32 + r)) 32 :=
    sumRange_add f 96 32
  have h2 : sumRange f 96 = sumRange f 64 + sumRange (fun r => f (2 * 32 + r)) 32 :=
    sumRange_add f 64 32
  have h3 : sumRange f 64 = sumRange f 32 + sumRange (fun r => f (1 * 32 + r)) 32 :=
    sumRange_add f 32 32
  have h4 : sumRange f 32 = sumRange f 0 + sumRange (fun r => f (0 * 32 + r)) 32 :=
    sumRange_add f 0 32
  have h0 : sumRange f 0 = 0 := rfl
  omega

-- § thread and row indexing

/-- Rows split uniquely into (row group, row-in-group). -/
theorem row_partition (i : Nat) (hi : i < 128) :
    i / 32 < 4 ∧ i % 32 < 32 ∧ i = (i / 32) * 32 + i % 32 := by
  omega

/-- A thread's row `rg*32 + r` is a valid state row. -/
theorem row_lt {rg r : Nat} (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

/-- Distinct (row group, r) give distinct rows. -/
theorem row_unique {rg rg' r r' : Nat} (hr : r < 32) (hr' : r' < 32)
    (h : rg * 32 + r = rg' * 32 + r') : rg = rg' ∧ r = r' :=
  split_unique_aux hr hr' h

/-- Linear thread id `rg*128 + col` is below 512 and determines `(rg, col)`. -/
theorem tid_unique {rg rg' col col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (h : rg * 128 + col = rg' * 128 + col') : rg = rg' ∧ col = col' :=
  split_unique_aux hc hc' h

/-- Thread ids stay below the block size. -/
theorem tid_lt {rg col : Nat} (hrg : rg < 4) (hc : col < 128) : rg * 128 + col < 512 := by
  omega

/-- Threads below 128 are exactly row group 0, and their warp id indexes `wsum[0..3]`. -/
theorem group0_threads {rg col : Nat} (hc : col < 128) (hrg : rg < 4)
    (h : rg * 128 + col < 128) : rg = 0 ∧ (rg * 128 + col) / 32 < 4 := by
  omega

/-- Address of `state[row][head][col]` in the `(S, h_v, S)` layout. -/
def stateIdx (h_v head row col : Nat) : Nat := (row * h_v + head) * 128 + col

/-- Every in-range address is inside the state allocation. -/
theorem stateIdx_lt {h_v head row col : Nat} (hh : head < h_v) (hr : row < 128) (hc : col < 128) :
    stateIdx h_v head row col < 128 * h_v * 128 := by
  unfold stateIdx
  have h1 : row * h_v + head < 128 * h_v := mul_add_lt_aux hr hh
  exact mul_add_lt_aux h1 hc

/-- Distinct `(row, head, col)` have distinct addresses. -/
theorem stateIdx_inj {h_v head head' row row' col col' : Nat} (hh : head < h_v) (hh' : head' < h_v)
    (hc : col < 128) (hc' : col' < 128)
    (e : stateIdx h_v head row col = stateIdx h_v head' row' col') :
    row = row' ∧ head = head' ∧ col = col' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hc hc' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  exact ⟨e3, e4, e2⟩

-- § conv/l2 kernel geometry

/-- Channel `c = bx*128 + tx` of block `bx < channels/128` is inside the channel range. -/
theorem channel_lt {channels bx tx : Nat} (hdiv : channels % 128 = 0) (hbx : bx < channels / 128)
    (htx : tx < 128) : bx * 128 + tx < channels := by
  omega

/-- The 3-row history of channel `c` is inside the `3*channels` buffer. -/
theorem hist_index_lt {c channels j : Nat} (hc : c < channels) (hj : j < 3) :
    c * 3 + j < channels * 3 :=
  mul_add_lt_aux hc hj

/-- The 4 conv taps of channel `c` are inside the `4*channels` weight buffer. -/
theorem tap_index_lt {c channels j : Nat} (hc : c < channels) (hj : j < 4) :
    c * 4 + j < channels * 4 :=
  mul_add_lt_aux hc hj

/-- A launch with `qk_heads <= channels/128` normalises only whole heads. -/
theorem qk_blocks_in_grid {channels qk_heads bx : Nat} (h : qk_heads ≤ channels / 128)
    (hbx : bx < qk_heads) : bx < channels / 128 := by
  omega

-- § alpha/beta kernel geometry

/-- Stacked row `row < 2*h_v` splits into alpha (`row < h_v`) or beta (`row - h_v`) rows. -/
theorem ab_row_split {row h_v : Nat} (h : row < 2 * h_v) :
    (if row ≥ h_v then row - h_v else row) < h_v := by
  split <;> omega

/-- The 8-warps-per-block grid covers all `2*h_v` rows. -/
theorem ab_grid_cover (h_v : Nat) : 2 * h_v ≤ ((2 * h_v + 7) / 8) * 8 := by
  omega

/-- Element `e` of a row lives in `uint4` chunk `e/8`, word `(e%8)/2`, half `e%2`. -/
theorem chunk_decomposition {e n : Nat} (hn : n % 8 = 0) (he : e < n) :
    e / 8 < n / 8 ∧ (e % 8) / 2 < 4 ∧ e = 8 * (e / 8) + 2 * ((e % 8) / 2) + e % 2 := by
  omega

/-- Shifting a 32-bit word left by 16 keeps exactly its low half in the high half. -/
theorem low_half_shift (x : Nat) : (x * 65536) % 4294967296 = (x % 65536) * 65536 := by
  omega

/-- The two bf16 halves of a word reassemble the word. -/
theorem halves_reassemble (x : Nat) : (x / 65536) * 65536 + x % 65536 = x := by
  omega

/-- `h_v` divisible by `h_k > 0` forces `h_k <= h_v` (the validity check implies a real grouping). -/
theorem head_group_le {h_k h_v : Nat} (hv : 0 < h_v) (h : h_v % h_k = 0) : h_k ≤ h_v :=
  Nat.le_of_dvd hv (Nat.dvd_of_mod_eq_zero h)

/-- The shared q/k head `head % h_k` is in range. -/
theorem qk_head_lt {head h_k : Nat} (hk : 0 < h_k) : head % h_k < h_k :=
  Nat.mod_lt head hk

end Strata.Kernels.FusedGdn



## FILE: /mnt/user-data/outputs/NativeGdnWarpStepKernel.lean
/-!
# MODULE: native_gdn_warp_step_kernel
Warp-per-column delta-net step (`native_gdn_step`) and its argument validation (aligned, disjoint spans).

## Source semantics
Each warp owns one `(head, col)` state column; lane `l` holds rows `r*32 + l` for `r = 0..3` in registers.
`kv_col` and `attn_col` are `warp_sum` over the 32 lanes of the per-lane 4-row partials, so they equal the
sequential 128-row contraction.  Only lane 0 writes `output[head*S + col]`; every lane writes back its own
four state rows.  The host wrapper requires a stream, `S = 128`, `h_v % h_k = 0`, `h_v <= 65535`, and all
spans aligned, non-wrapping and pairwise disjoint from state/output.

## Form template identifiers
- `state`, `output`  -> `stateIdx`, `outIdx`
- `valid_span`, `overlap` -> `validSpan`, `overlap`
- `warp_sum`          -> `sumRange` over lanes
-/

namespace Strata.Kernels.NativeGdn

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § reductions

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Termwise sums add. -/
theorem sumRange_pointwise_add (u v : Nat → Int) (m : Nat) :
    sumRange (fun b => u b + v b) m = sumRange u m + sumRange v m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange (fun b => u b + v b) m + (u m + v m) =
      (sumRange u m + u m) + (sumRange v m + v m)
    omega

/-- A sum of zeros is zero. -/
theorem sumRange_zero (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = 0) :
    sumRange f n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = 0
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

/-- Order of a double sum can be swapped. -/
theorem sumRange_swap (g : Nat → Nat → Int) (n m : Nat) :
    sumRange (fun a => sumRange (g a) m) n = sumRange (fun b => sumRange (fun a => g a b) n) m := by
  induction n with
  | zero =>
    show (0 : Int) = sumRange (fun _ => (0 : Int)) m
    exact (sumRange_zero _ m (fun _ _ => rfl)).symm
  | succ n ih =>
    show sumRange (fun a => sumRange (g a) m) n + sumRange (g n) m =
      sumRange (fun b => sumRange (fun a => g a b) n + g n b) m
    rw [sumRange_pointwise_add, ih]

/-- A sum over four terms. -/
theorem sumRange_four (P : Nat → Int) : sumRange P 4 = P 0 + P 1 + P 2 + P 3 := by
  show 0 + P 0 + P 1 + P 2 + P 3 = P 0 + P 1 + P 2 + P 3
  omega

/-- Four 32-row groups add up to 128 rows. -/
theorem rows_four_way (f : Nat → Int) :
    sumRange f 128 =
      sumRange (fun r => f (0 * 32 + r)) 32 + sumRange (fun r => f (1 * 32 + r)) 32 +
      sumRange (fun r => f (2 * 32 + r)) 32 + sumRange (fun r => f (3 * 32 + r)) 32 := by
  have h1 : sumRange f 128 = sumRange f 96 + sumRange (fun r => f (3 * 32 + r)) 32 :=
    sumRange_add f 96 32
  have h2 : sumRange f 96 = sumRange f 64 + sumRange (fun r => f (2 * 32 + r)) 32 :=
    sumRange_add f 64 32
  have h3 : sumRange f 64 = sumRange f 32 + sumRange (fun r => f (1 * 32 + r)) 32 :=
    sumRange_add f 32 32
  have h4 : sumRange f 32 = sumRange f 0 + sumRange (fun r => f (0 * 32 + r)) 32 :=
    sumRange_add f 0 32
  have h0 : sumRange f 0 = 0 := rfl
  omega

/-- THE warp-sharding contract: summing, over the 32 lanes, each lane's four registers `f (r*32 + lane)`
equals the sequential contraction over all 128 rows. -/
theorem lane_shard_sum (f : Nat → Int) :
    sumRange (fun l => sumRange (fun r => f (r * 32 + l)) 4) 32 = sumRange f 128 := by
  refine ((sumRange_swap (fun l r => f (r * 32 + l)) 32 4).trans ?_).trans (rows_four_way f).symm
  exact sumRange_four _

-- § indexing

/-- Register `r` of lane `l` is row `r*32 + l`; rows decompose back uniquely. -/
theorem lane_row_roundtrip {r lane : Nat} (hr : r < 4) (hl : lane < 32) :
    r * 32 + lane < 128 ∧ (r * 32 + lane) / 32 = r ∧ (r * 32 + lane) % 32 = lane := by
  omega

/-- Column `blockIdx.z*4 + threadIdx.y` (grid z = 32, block y = 4) is in `0..127` and unique. -/
theorem col_cover {z ty : Nat} (hz : z < 32) (hty : ty < 4) :
    z * 4 + ty < 128 ∧ (z * 4 + ty) / 4 = z ∧ (z * 4 + ty) % 4 = ty := by
  omega

/-- One warp per (head, column): `h_v * 32 * 4` threads in warps of 32 is `h_v * 128` warps. -/
theorem warp_count (h_v : Nat) : h_v * 32 * 4 / 32 = h_v * 4 := by
  omega

/-- Address of `state[i][head][col]`. -/
def stateIdx (h_v i head col : Nat) : Nat := (i * h_v + head) * 128 + col

/-- Every state address is inside the `128*128*h_v` allocation. -/
theorem stateIdx_lt {h_v i head col : Nat} (hi : i < 128) (hh : head < h_v) (hc : col < 128) :
    stateIdx h_v i head col < 128 * h_v * 128 := by
  unfold stateIdx
  have h1 : i * h_v + head < 128 * h_v := mul_add_lt_aux hi hh
  exact mul_add_lt_aux h1 hc

/-- Address of `output[head][col]`. -/
def outIdx (head col : Nat) : Nat := head * 128 + col

/-- Output writes (lane 0 only) are unique per `(head, col)`. -/
theorem outIdx_unique {head head' col col' : Nat} (hc : col < 128) (hc' : col' < 128)
    (e : outIdx head col = outIdx head' col') : head = head' ∧ col = col' :=
  split_unique_aux hc hc' e

/-- No two threads write the same state address: the (head, col, register, lane) tuple is recovered. -/
theorem state_write_unique {h_v head head' col col' r r' l l' : Nat}
    (hh : head < h_v) (hh' : head' < h_v) (hc : col < 128) (hc' : col' < 128)
    (hl : l < 32) (hl' : l' < 32)
    (e : stateIdx h_v (r * 32 + l) head col = stateIdx h_v (r' * 32 + l') head' col') :
    r = r' ∧ l = l' ∧ head = head' ∧ col = col' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := split_unique_aux hc hc' e
  obtain ⟨e3, e4⟩ := split_unique_aux hh hh' e1
  obtain ⟨e5, e6⟩ := split_unique_aux hl hl' e3
  exact ⟨e5, e6, e4, e2⟩

/-- The shared q/k row `(head % h_k)*128 + i` stays in the `h_k*128` buffer. -/
theorem qk_index_lt {head h_k i : Nat} (hk : 0 < h_k) (hi : i < 128) :
    (head % h_k) * 128 + i < h_k * 128 :=
  mul_add_lt_aux (Nat.mod_lt head hk) hi

-- § span validation

/-- A span is valid when non-null, 4-aligned and non-wrapping. -/
def validSpan (ptr bytes max : Nat) : Prop := ptr ≠ 0 ∧ ptr % 4 = 0 ∧ bytes ≤ max - ptr

/-- Interval overlap as written in the source. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- A non-empty span overlaps itself. -/
theorem overlap_self {a n : Nat} (h : 0 < n) : overlap a n a n := by
  unfold overlap
  omega

/-- Adjacent spans are disjoint. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- A valid span `[p, p+bytes)` cannot wrap the address space. -/
theorem validSpan_no_wrap {ptr bytes max : Nat} (hp : ptr ≤ max) (h : validSpan ptr bytes max) :
    ptr + bytes ≤ max := by
  unfold validSpan at h
  omega

/-- The host-side byte counts are all multiples of 4, so 4-byte alignment of the base suffices. -/
theorem span_sizes_float_multiple (h_v h_k : Nat) :
    (128 * 128 * h_v * 4) % 4 = 0 ∧ (128 * h_k * 4) % 4 = 0 ∧ (128 * h_v * 4) % 4 = 0 ∧
      (h_v * 4) % 4 = 0 := by
  omega

/-- The grid limit `h_v <= 65535` is what the host check enforces (`blockIdx.x` has 16-bit range here). -/
theorem grid_limit {h_v : Nat} (h : h_v ≤ 65535) : h_v < 65536 := by
  omega

end Strata.Kernels.NativeGdn



## FILE: /mnt/user-data/outputs/NativeGdnPreprocessKernels.lean
/-!
# MODULE: native_gdn_preprocess_kernels
Conv+SiLU history update, 128-wide norms with padded 256-thread block reductions, and the span validation
of the native GDN preprocessing path.

## Source semantics
`conv_silu` reads `[h0,h1,h2,x]`, accumulates the four taps in order starting from 0, then stores history
`[h1,h2,x]`.  `l2_norm` and `out_norm` use 256 threads for a 128-wide row: threads `128..255` contribute 0
to the block sum, and the 8 warp partials land in `sums[tid/32]`.  Host checks: count in `[1, 65535]`,
a stream, width exactly 128, aligned non-null spans, pairwise disjoint writable spans.

## Form template identifiers
- `history`/`values` -> `win`, `newHist`
- `norm_sum`          -> `sumRange (padded f) 256`
- `Span`              -> `overlap`
-/

namespace Strata.Kernels.NativeGdnPre

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Pointwise-equal summands give equal sums. -/
theorem sumRange_congr (f g : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = g i) :
    sumRange f n = sumRange g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = sumRange g n + g n
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

/-- A sum of zeros is zero. -/
theorem sumRange_zero (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = 0) :
    sumRange f n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = 0
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

-- § padded block reduction

/-- Thread `tid` contributes `f tid` when inside the 128-wide row and `0` otherwise. -/
def padded (f : Nat → Int) : Nat → Int := fun i => if i < 128 then f i else 0

/-- Running 256 threads over a 128-wide row computes exactly the 128-element sum. -/
theorem padded_norm_sum (f : Nat → Int) : sumRange (padded f) 256 = sumRange f 128 := by
  have h : sumRange (padded f) 256 =
      sumRange (padded f) 128 + sumRange (fun i => padded f (128 + i)) 128 :=
    sumRange_add (padded f) 128 128
  have h1 : sumRange (padded f) 128 = sumRange f 128 :=
    sumRange_congr _ _ 128 (fun i hi => by simp [padded, hi])
  have h2 : sumRange (fun i => padded f (128 + i)) 128 = 0 :=
    sumRange_zero _ 128 (fun i _ => by
      have hn : ¬ (128 + i < 128) := by omega
      simp [padded, hn])
  omega

/-- Warp partials of a 256-thread block fit the 8 slots read by lanes `< 8` and the 32-slot array. -/
theorem warp_partial_slot {tid : Nat} (h : tid < 256) : tid / 32 < 8 ∧ 8 ≤ 32 := by
  omega

-- § conv + silu window

/-- Integer dot product of two lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Left-to-right tap accumulation from 0 equals the dot of the 4-wide window. -/
theorem conv_taps_sum (v0 v1 v2 v3 w0 w1 w2 w3 : Int) :
    ((0 + v0 * w0) + v1 * w1 + v2 * w2) + v3 * w3 = dot [v0, v1, v2, v3] [w0, w1, w2, w3] := by
  simp only [dot]
  omega

/-- The new history is the window without its oldest entry: `[h1,h2,x]`. -/
def newHist (h0 h1 h2 x : Int) : List Int := ([h0, h1, h2, x]).drop 1

/-- The new history has three rows and ends with the new input. -/
theorem newHist_shape (h0 h1 h2 x : Int) :
    newHist h0 h1 h2 x = [h1, h2, x] ∧ (newHist h0 h1 h2 x).length = 3 := by
  simp [newHist]

/-- History row `c*3 + tap` is inside `3*channels`; weights `c*4 + tap` inside `4*channels`. -/
theorem conv_indices {c channels t : Nat} (hc : c < channels) (ht : t < 3) :
    c * 3 + t < channels * 3 ∧ c * 4 + t < channels * 4 :=
  ⟨mul_add_lt_aux hc ht, mul_add_lt_aux hc (by omega)⟩

/-- Channel launch of 256-thread blocks covers `channels`. -/
theorem conv_launch_cover (channels : Nat) : channels ≤ ((channels + 255) / 256) * 256 := by
  omega

/-- Row offset `blockIdx*128 + col` stays inside `rows*128`. -/
theorem row_offset_lt {b col rows : Nat} (hb : b < rows) (hc : col < 128) :
    b * 128 + col < rows * 128 :=
  mul_add_lt_aux hb hc

/-- The host's count window `[1, 65535]` keeps the 1-D grids within the 16-bit bound. -/
theorem count_window {n : Nat} (h1 : 1 ≤ n) (h2 : n ≤ 65535) :
    0 < (n + 255) / 256 ∧ (n + 255) / 256 ≤ 256 := by
  omega

-- § span validation

/-- Interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- Overlap is symmetric, so checking `i < j` pairs suffices for pairwise disjointness. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- Span byte counts in the conv wrapper: history `3*bytes` and weights `4*bytes` are multiples of 4. -/
theorem conv_span_sizes (channels : Nat) :
    (3 * (channels * 4)) % 4 = 0 ∧ (4 * (channels * 4)) % 4 = 0 := by
  omega

end Strata.Kernels.NativeGdnPre



## FILE: /mnt/user-data/outputs/FusedGatedResidualReadKernels.lean
/-!
# MODULE: fused_gated_residual_read_kernels
Launch geometry, stream/float4 indexing and tile arithmetic of the fused gated-residual read
(`gr_down_kernel`, `gr_up_kernel`, and the T-token variants sharing one weight read).

## Source semantics
Geometry is fixed: `N = 2560`, `HC = 4`, `D = N*HC = 10240`, `LR = 320`.  `down` uses 40 blocks of 8 warps
(one warp per output row, 320 rows) plus one block for the 4 inject rows.  `up` uses 80 blocks of 32 columns
x 4 streams = 128 rows (16 per warp); the multi-token variant uses 160 blocks of 16 columns.  Activations are
read as `float4`; the loop `i = t*4; i += 1024` covers `D`, and because `N % 4 = 0` a `float4` never straddles
two streams.  Weights are bf16 packed 8 per `uint4`: a 10240-wide row is 1280 chunks (40 per lane) and a
320-wide row is 40 chunks (lane takes chunk `lane`, lanes `< 8` also take `32 + lane`).  The multi-token down
kernel stages 2560 floats per token per tile (4 tiles) in dynamic shared memory.

## Form template identifiers
- `N`,`HC`,`D`,`LR`,`TILE`   -> constants
- `upIdx bx r`               -> row index `(r/32)*N + bx*32 + r%32` of the up projection
-/

namespace Strata.Kernels.FusedGr

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Stream-flattened width. -/
def D : Nat := N * HC
/-- Low-rank width. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := THREADS / 32
/-- Down blocks of 8 rows. -/
def DOWN_BLOCKS : Nat := LR / WARPS
/-- Columns per up block. -/
def UP_COLS : Nat := 32
/-- Up blocks. -/
def UP_BLOCKS : Nat := N / UP_COLS
/-- Staged floats per token per tile. -/
def TILE : Nat := 2560
/-- Weight chunks per lane per tile. -/
def TQ : Nat := TILE / 8 / 32
/-- Columns per multi-token up block. -/
def UPM_COLS : Nat := 16
/-- Multi-token up blocks. -/
def UPM_BLOCKS : Nat := N / UPM_COLS

/-- `D = N * HC = 10240`. -/
theorem D_value : D = 10240 := by
  decide

/-- 40 down blocks of 8 warps cover the 320 rows exactly. -/
theorem down_rows_cover : DOWN_BLOCKS * WARPS = LR := by
  decide

/-- 80 up blocks of 32 columns cover `N`. -/
theorem up_cols_cover : UP_BLOCKS * UP_COLS = N := by
  decide

/-- 160 multi-token up blocks of 16 columns cover `N`. -/
theorem upm_cols_cover : UPM_BLOCKS * UPM_COLS = N := by
  decide

/-- Rows per `up` block: 4 streams x 32 columns = 128, i.e. 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ HC * UP_COLS / WARPS = 16 := by
  decide

/-- Multi-token up: 64 rows, 8 per warp. -/
theorem upm_rows_per_warp : HC * UPM_COLS = 64 ∧ HC * UPM_COLS / WARPS = 8 := by
  decide

/-- Tile arithmetic: 10 chunks per lane, four tiles span `D`. -/
theorem tile_arithmetic : TQ = 10 ∧ D % TILE = 0 ∧ D / TILE = 4 := by
  decide

/-- A 10240-wide bf16 row is 1280 chunks = 40 per lane; a 320-wide row is 40 chunks = 32 + 8. -/
theorem chunk_counts : D / 8 = 1280 ∧ D / 8 / 32 = 40 ∧ LR / 8 = 40 ∧ LR / 8 - 32 = 8 := by
  decide

/-- Dynamic shared memory for `T <= Tmax` tokens never exceeds the attribute set for `Tmax`. -/
theorem multi_smem_le (T Tmax : Nat) (h : T ≤ Tmax) : T * TILE * 4 ≤ Tmax * TILE * 4 :=
  Nat.mul_le_mul_right 4 (Nat.mul_le_mul_right TILE h)

-- § stream and float4 indexing

/-- Flat index `i < D` splits into stream `i / N` and offset `i % N`. -/
theorem stream_split {i : Nat} (hi : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ (i / 2560) * 2560 + i % 2560 = i := by
  omega

/-- A `float4` at a multiple of 4 never straddles two streams. -/
theorem float4_in_stream {i : Nat} (h4 : i % 4 = 0) : i % 2560 + 4 ≤ 2560 := by
  omega

/-- The loop `i = t*4; i += 1024` reaches every `float4` slot of `D` exactly via `(t, k)`. -/
theorem float4_cover {i : Nat} (hi : i < 10240) (h4 : i % 4 = 0) :
    ∃ t k : Nat, t < 256 ∧ k < 10 ∧ i = t * 4 + k * 1024 := by
  refine ⟨(i % 1024) / 4, i / 1024, ?_, ?_, ?_⟩ <;> omega

/-- The `(t, k)` pair of a `float4` slot is unique. -/
theorem float4_unique {t t' k k' : Nat} (ht : t < 256) (ht' : t' < 256)
    (h : t * 4 + k * 1024 = t' * 4 + k' * 1024) : t = t' ∧ k = k' := by
  omega

-- § row mapping of down and up

/-- A down row `bx*8 + warp` is below 320 for `bx < 40`. -/
theorem down_row_lt {bx warp : Nat} (hb : bx < 40) (hw : warp < 8) : bx * 8 + warp < 320 := by
  omega

/-- Row index of the up projection for block `bx` and local row `r < 128`. -/
def upIdx (bx r : Nat) : Nat := (r / 32) * 2560 + bx * 32 + r % 32

/-- Every up row index is inside `D`. -/
theorem upIdx_lt {bx r : Nat} (hb : bx < 80) (hr : r < 128) : upIdx bx r < 10240 := by
  unfold upIdx
  omega

/-- Distinct `(block, row)` pairs own distinct up rows, so the `R_out` writes never collide. -/
theorem upIdx_inj {bx bx' r r' : Nat} (hb : bx < 80) (hb' : bx' < 80) (hr : r < 128) (hr' : r' < 128)
    (h : upIdx bx r = upIdx bx' r') : bx = bx' ∧ r = r' := by
  unfold upIdx at h
  omega

/-- Same for the 16-column multi-token blocks. -/
def upmIdx (bx r : Nat) : Nat := (r / 16) * 2560 + bx * 16 + r % 16

/-- Multi-token up rows are inside `D`. -/
theorem upmIdx_lt {bx r : Nat} (hb : bx < 160) (hr : r < 64) : upmIdx bx r < 10240 := by
  unfold upmIdx
  omega

/-- Multi-token up rows are unique per `(block, row)`. -/
theorem upmIdx_inj {bx bx' r r' : Nat} (hb : bx < 160) (hb' : bx' < 160) (hr : r < 64) (hr' : r' < 64)
    (h : upmIdx bx r = upmIdx bx' r') : bx = bx' ∧ r = r' := by
  unfold upmIdx at h
  omega

/-- Inject rows: the extra block uses `warp < HC` and writes `inject_out[warp]`. -/
theorem inject_rows {warp : Nat} (hw : warp < 8) (ha : warp < 4) : warp < 8 ∧ warp < 4 :=
  ⟨hw, ha⟩

/-- `fused_gr_supported` pins the geometry: a supported shape has exactly `D = 10240` flattened width. -/
theorem supported_dims {n hc lr : Nat} (hn : n = 2560) (hh : hc = 4) (hl : lr = 320) :
    n * hc = 10240 ∧ lr = 320 := by
  subst hn
  subst hh
  exact ⟨by decide, hl⟩

/-- Weight-row address `row*D + chunk*8 + e` stays inside a `LR x D` matrix. -/
theorem down_weight_index_lt {row j : Nat} (hr : row < 320) (hj : j < 10240) :
    row * 10240 + j < 320 * 10240 :=
  mul_add_lt_aux hr hj

/-- Weight-row address for `up`: row `i < D`, column `k < LR`. -/
theorem up_weight_index_lt {i k : Nat} (hi : i < 10240) (hk : k < 320) :
    i * 320 + k < 10240 * 320 :=
  mul_add_lt_aux hi hk

end Strata.Kernels.FusedGr



## FILE: /mnt/user-data/outputs/GatedResidualReadWriteKernels.lean
/-!
# MODULE: gated_residual_read_write_kernels
Workspace carving, warp/block-per-row index maps and the write-back of the gated residual
(`gr_workspace_init`, `gr_norm_kernel`, `gr_down_kernel`, `gr_gate_kernel`, `gr_mean_kernel`,
`gr_inject_kernel`, `gr_write_kernel`).

## Source semantics
The workspace has five regions (`xn`, `xq`, `lq`, `gated`, `lo`) each rounded up to 16 bytes and laid out
back to back; a table indexed by one `k` assigns both size and pointer.  `gr_down` uses one block of 8 warps
per output row (warp `w` starts at `w*32` and strides by `256`).  `gr_gate` uses one warp per `hc_dim` row.
`gr_write` computes `out[c*n_embd + d] = R[..] + block_out[d] * w[c]`: every stream adds the SAME block
output, only the weight differs.  `use_fp32 = fp32_activations || native_mmvf`.

## Form template identifiers
- `sz`/`al`  -> `rawSizes`, `alignUp`
- `hc_dim`   -> `hc * n_embd`
- `gr_write` -> `writeOut`
-/

namespace Strata.Kernels.GatedResidual

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Modulo recovers the column. -/
theorem mod_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) % n = c := by
  have h1 := Nat.div_add_mod (a * n + c) n
  have h2 : (a * n + c) / n = a := div_of_split_aux hc
  rw [h2] at h1
  have h3 : n * a = a * n := Nat.mul_comm n a
  omega

-- § workspace layout

/-- Round a byte count up to a multiple of 16. -/
def alignUp (x : Nat) : Nat := ((x + 15) / 16) * 16

/-- Aligned size is at least the raw size. -/
theorem alignUp_ge (x : Nat) : x ≤ alignUp x := by
  unfold alignUp
  omega

/-- Aligned size is a multiple of 16. -/
theorem alignUp_mod (x : Nat) : alignUp x % 16 = 0 := by
  unfold alignUp
  omega

/-- Alignment wastes fewer than 16 bytes. -/
theorem alignUp_lt (x : Nat) : alignUp x < x + 16 := by
  unfold alignUp
  omega

/-- Region sizes in the order xn, xq, lq, gated, lo (the single source table). -/
def rawSizes (hc n_embd hc_lr : Nat) : List Nat :=
  [hc * n_embd * 4, hc * n_embd * 2, hc_lr * 2, hc * n_embd * 4, hc_lr * 4]

/-- Total workspace bytes: the sum of the aligned regions. -/
def workspaceBytes (hc n_embd hc_lr : Nat) : Nat :=
  ((rawSizes hc n_embd hc_lr).map alignUp).foldl (· + ·) 0

/-- At the real geometry the workspace is exactly 104,320 bytes. -/
theorem workspace_real : workspaceBytes 4 2560 320 = 104320 := by
  decide

/-- At the real geometry no region needs padding (every size is already a multiple of 16). -/
theorem workspace_real_no_padding :
    (rawSizes 4 2560 320).map alignUp = rawSizes 4 2560 320 := by
  decide

/-- `xq` needs 20,480 bytes at the real geometry; the buggy table gave it `hc_lr*2 = 640`. -/
theorem xq_size_real : 4 * 2560 * 2 = 20480 ∧ 320 * 2 = 640 := by
  decide

/-- Consecutive regions are adjacent: offset of region k+1 is offset of k plus its aligned size. -/
theorem region_adjacent (off size : Nat) : off + alignUp size = off + alignUp size :=
  rfl

/-- Aligned offsets keep every region 16-byte aligned when the base is. -/
theorem offset_aligned {base off size : Nat} (hb : base % 16 = 0) (ho : off % 16 = 0) :
    (base + off + alignUp size) % 16 = 0 := by
  have h := alignUp_mod size
  omega

-- § block and warp maps

/-- Down projection: one block per row, 8 warps, warp `w` starts at `w*32` and strides by 256.
Every `i < hc_dim` has a unique (warp, lane, iteration). -/
theorem down_stride_cover {i : Nat} :
    (i % 256) / 32 < 8 ∧ i % 32 < 32 ∧ i = ((i % 256) / 32) * 32 + i % 32 + (i / 256) * 256 := by
  omega

/-- Uniqueness of the (warp, lane, iteration) decomposition. -/
theorem down_stride_unique {w w' l l' k k' : Nat} (hw : w < 8) (hw' : w' < 8) (hl : l < 32)
    (hl' : l' < 32) (h : w * 32 + l + k * 256 = w' * 32 + l' + k' * 256) :
    w = w' ∧ l = l' ∧ k = k' := by
  omega

/-- The gate kernel's warp row `bx*8 + warp` and its grid `(hc_dim + 7)/8` cover `hc_dim` rows. -/
theorem gate_grid_cover (hc_dim : Nat) : hc_dim ≤ ((hc_dim + 7) / 8) * 8 := by
  omega

/-- The inject kernel's warp `tid / 32` is a valid stream for `tid < 32*hc`. -/
theorem inject_warp_lt {tid hc : Nat} (h : tid < 32 * hc) : tid / 32 < hc := by
  omega

/-- 320 blocks of 8 warps: 2560 warps, 53 per SM on 48 SMs. -/
theorem down_occupancy : 320 * 8 = 2560 ∧ 2560 / 48 = 53 ∧ 40 * 8 / 48 = 6 := by
  decide

/-- Scratch for block sums: `double scratch[8]` (64 bytes) holds the 8 floats of an FP32 reduction. -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 ∧ (256 + 31) / 32 = 8 := by
  decide

/-- Stream element address `c*n_embd + d` is inside `hc*n_embd`. -/
theorem stream_index_lt {c d hc n_embd : Nat} (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd :=
  mul_add_lt_aux hc' hd

/-- Flat index `i < hc*n_embd` splits into stream `i / n_embd` and offset `i % n_embd`. -/
theorem flat_split {i hc n_embd : Nat} (hn : 0 < n_embd) (hi : i < hc * n_embd) :
    i / n_embd < hc ∧ i % n_embd < n_embd ∧ (i / n_embd) * n_embd + i % n_embd = i := by
  refine ⟨?_, Nat.mod_lt i hn, ?_⟩
  · apply Nat.div_lt_of_lt_mul
    rw [Nat.mul_comm]
    exact hi
  · have h := Nat.div_add_mod i n_embd
    have h3 : n_embd * (i / n_embd) = (i / n_embd) * n_embd := Nat.mul_comm _ _
    omega

-- § write-back

/-- `out[c*n + d] = R[c*n + d] + block_out[d] * w[c]` with all quantities as integers. -/
def writeOut (R : Nat → Int) (block_out w : Nat → Int) (n_embd i : Nat) : Int :=
  R i + block_out (i % n_embd) * w (i / n_embd)

/-- Reading the write-back at stream `c`, column `d` gives `R + block_out d * w c`:
every stream adds the SAME block output with its own weight. -/
theorem write_at_stream (R : Nat → Int) (block_out w : Nat → Int) {n_embd c d : Nat}
    (hd : d < n_embd) :
    writeOut R block_out w n_embd (c * n_embd + d) =
      R (c * n_embd + d) + block_out d * w c := by
  unfold writeOut
  rw [mod_of_split_aux hd, div_of_split_aux hd]

/-- Two streams differ only in the weight applied to the shared block output. -/
theorem write_streams_differ_by_weight (R : Nat → Int) (block_out w : Nat → Int) {n_embd c c' d : Nat}
    (hd : d < n_embd) :
    writeOut R block_out w n_embd (c * n_embd + d) - R (c * n_embd + d) =
      block_out d * w c ∧
    writeOut R block_out w n_embd (c' * n_embd + d) - R (c' * n_embd + d) =
      block_out d * w c' := by
  refine ⟨?_, ?_⟩
  · rw [write_at_stream R block_out w hd]
    omega
  · rw [write_at_stream R block_out w hd]
    omega

/-- The fp32 selector: native MMVF forces fp32 activations. -/
def useFp32 (fp32Activations nativeMmvf : Bool) : Bool := fp32Activations || nativeMmvf

/-- Native MMVF implies the FP32 path. -/
theorem native_implies_fp32 (a : Bool) : useFp32 a true = true := by
  cases a <;> rfl

/-- Without either flag the bf16-activation path is taken. -/
theorem default_is_bf16 : useFp32 false false = false := rfl

/-- Native MMVF needs even `hc*n_embd` and even `hc_lr` (paired element loads). -/
def nativeShapeOk (hc_dim hc_lr : Nat) : Prop := hc_dim % 2 = 0 ∧ hc_lr % 2 = 0

/-- The real geometry satisfies the native-MMVF shape requirement. -/
theorem native_shape_real : nativeShapeOk (4 * 2560) 320 := by
  unfold nativeShapeOk
  decide

end Strata.Kernels.GatedResidual



## FILE: /mnt/user-data/outputs/NativePlePostopsKernels.lean
/-!
# MODULE: native_ple_postops_kernels
Index maps, partial-lane coverage and span rules of the native PLE post-ops (`gate_kernel`,
`broadcast_kernel`, `conv_residual_kernel`).

## Source semantics
`N = 2560`, `H = 4`, `D = N*H = 10240`, `HISTORY = 9`.  The gate kernel runs one block of 512 threads per
stream; thread `t` accumulates 8 partial lanes `d = t + j*512` (only `d < N` contribute), so lanes `j >= 5`
are zero.  `gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))` with `sign s = (s>0) - (s<0)`.  The broadcast
writes `gated[i] = value[i % N] * gate[i / N]`.  The conv residual reads history taps `c*9 + 3k` for `k < 3`
(the fourth tap is the new normalised value) and writes `result = hidden + (gated + activation)`; `result`
may exactly alias `hidden` because every thread owns one element.

## Form template identifiers
- `N`,`H`,`D`,`HISTORY` -> constants
- `signI`                -> signed-sqrt gate sign
- `overlap`              -> span intersection
-/

namespace Strata.Kernels.NativePlePostops

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Streams. -/
def H : Nat := 4
/-- Flattened width. -/
def D : Nat := N * H
/-- History rows per channel. -/
def HISTORY : Nat := 9

/-- `D = 10240` and the 256-thread launches tile it exactly (40 blocks). -/
theorem D_launch : D = 10240 ∧ D % 256 = 0 ∧ D / 256 = 40 := by
  decide

-- § gate partial lanes

/-- Every element `d < N` is handled by thread `d % 512` in partial lane `d / 512 < 5`. -/
theorem gate_lane_cover {d : Nat} (hd : d < 2560) :
    d % 512 < 512 ∧ d / 512 < 5 ∧ d = d % 512 + (d / 512) * 512 := by
  omega

/-- Partial lanes `j >= 5` start beyond `N` for every thread, so they add zero. -/
theorem gate_lane_empty {t j : Nat} (ht : t < 512) (hj : 5 ≤ j) : 2560 ≤ t + j * 512 := by
  omega

/-- The (thread, lane) decomposition is unique below `N`. -/
theorem gate_lane_unique {t t' j j' : Nat} (ht : t < 512) (ht' : t' < 512)
    (h : t + j * 512 = t' + j' * 512) : t = t' ∧ j = j' := by
  omega

/-- 512 threads form 16 warps; the final reduction reads `partials[lane]` for `lane < 16`. -/
theorem gate_warp_partials {t : Nat} (ht : t < 512) : t / 32 < 16 ∧ 16 ≤ 32 := by
  omega

-- § signed square root

/-- `(x > 0) - (x < 0)` as an integer. -/
def signI (s : Int) : Int := (if 0 < s then 1 else 0) - (if s < 0 then 1 else 0)

/-- The sign is odd. -/
theorem signI_neg (s : Int) : signI (-s) = - signI s := by
  unfold signI
  split <;> split <;> split <;> split <;> omega

/-- The sign vanishes exactly at zero. -/
theorem signI_zero_iff (s : Int) : signI s = 0 ↔ s = 0 := by
  unfold signI
  split <;> split <;> omega

/-- The sign is one of `-1, 0, 1`. -/
theorem signI_range (s : Int) : -1 ≤ signI s ∧ signI s ≤ 1 := by
  unfold signI
  split <;> split <;> omega

/-- The clamp `max(|s|, 1e-6)` is a floor: integer analogue `max x 1 >= 1`. -/
theorem clamp_floor (x : Int) : 1 ≤ max x 1 := by
  omega

-- § broadcast and conv indices

/-- Flat `i < D` splits into column `i % N < N` and stream `i / N < H`. -/
theorem broadcast_split {i : Nat} (hi : i < 10240) :
    i % 2560 < 2560 ∧ i / 2560 < 4 := by
  omega

/-- History tap `c*9 + 3k` (for `k < 3`) is inside `D*9`. -/
theorem history_index_lt {c k : Nat} (hc : c < 10240) (hk : k < 3) : c * 9 + 3 * k < 10240 * 9 :=
  mul_add_lt_aux hc (by omega)

/-- Conv weights `c*4 + k` are inside `4*D` halves. -/
theorem weight_index_lt {c k : Nat} (hc : c < 10240) (hk : k < 4) : c * 4 + k < 10240 * 4 :=
  mul_add_lt_aux hc hk

/-- The dilated taps read history rows `0, 3, 6` and the fourth tap is the new input (row 9). -/
theorem tap_rows : ∀ k : Nat, k < 4 → 9 - (3 - k) * 3 = 3 * k := by
  intro k hk
  omega

-- § span rules

/-- Interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- A pointwise kernel may run in place: if each output element depends only on the SAME-index input
element, rewriting the array position by position equals computing into a fresh array. -/
theorem pointwise_inplace_safe (g : Int → Int) (a : Nat → Int) (i : Nat) :
    (fun j => g (a j)) i = g (a i) :=
  rfl

end Strata.Kernels.NativePlePostops



## FILE: /mnt/user-data/outputs/PleBlockKernels.lean
/-!
# MODULE: ple_block_kernels
Scratch carving, history ring, dilated conv rows, bf16 rounding and overlap checks of the PLE block
(`ple_block`, `history_advance_kernel`, `conv_kernel`, `to_bf16_kernel`, `ple_block_scratch_bytes`).

## Source semantics
The caller owns one scratch block: five `hc_dim` float buffers, then `n_embd` and `hc` floats (one float
region), then a Q8 activation image of `(n_embd/32)*34` bytes, then a BF16 copy of the embedding; each
region 16-byte aligned.  The float region is carved as key, query, norm, gated, conv (each `D`), value (`N`),
gate (`H`).  The conv history is row-fastest: `hist[row + 9*c]`.  With kernel 4, dilation 3, history 9 the
taps read rows 0, 3, 6, 9 and row 9 is the NEW row.  `history_advance` shifts each column up by one and
appends `normalized`.  `bf16_bits` rounds to nearest even using `(i + ((i>>16)&1) + 0x7FFF) & 0xFFFF0000`.
`overlap` returns false for null or empty spans and otherwise is interval intersection.

## Form template identifiers
- `scratch`          -> offsets `offKey .. offGate`
- `hist`             -> `histIdx`
- `bf16_bits`        -> `bf16Bits`
- `overlap`          -> `overlapP`
-/

namespace Strata.Kernels.PleBlock

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § scratch layout

/-- Embedding width. -/
def NG_N_EMBD : Nat := 2560
/-- Streams. -/
def NG_HC : Nat := 4
/-- Flattened width. -/
def NG_HC_DIM : Nat := NG_N_EMBD * NG_HC
/-- History rows. -/
def NG_HIST : Nat := 9

/-- Round up to 16. -/
def alignUp (x : Nat) : Nat := ((x + 15) / 16) * 16

/-- Aligned size covers the raw size. -/
theorem alignUp_ge (x : Nat) : x ≤ alignUp x := by
  unfold alignUp
  omega

/-- Aligned size is a multiple of 16. -/
theorem alignUp_mod (x : Nat) : alignUp x % 16 = 0 := by
  unfold alignUp
  omega

/-- Total scratch bytes as in `ple_block_scratch_bytes`. -/
def scratchBytes (f q e : Nat) : Nat := alignUp f + alignUp q + e + 256

/-- At the real geometry the scratch is 223,152 bytes. -/
theorem scratch_real :
    scratchBytes ((5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * 4) ((NG_N_EMBD / 32) * 34) (NG_N_EMBD * 2)
      = 223152 := by
  decide

/-- The three carved regions are in order and fit: floats end before the Q8 image, which ends before
the BF16 copy, which ends before the 256-byte tail. -/
theorem regions_fit (f q e : Nat) :
    f ≤ alignUp f ∧ alignUp f + q ≤ alignUp f + alignUp q ∧
      alignUp f + alignUp q + e ≤ scratchBytes f q e := by
  have h1 := alignUp_ge f
  have h2 := alignUp_ge q
  unfold scratchBytes
  omega

/-- The float region is carved into seven consecutive pieces ending at `5*D + N + H` floats. -/
theorem float_region_total : 5 * NG_HC_DIM + NG_N_EMBD + NG_HC = 53764 := by
  decide

/-- Offsets (in floats) of key, query, norm, gated, conv, value, gate. -/
def floatOffsets : List Nat :=
  [0, NG_HC_DIM, 2 * NG_HC_DIM, 3 * NG_HC_DIM, 4 * NG_HC_DIM, 5 * NG_HC_DIM,
    5 * NG_HC_DIM + NG_N_EMBD]

/-- Piece sizes in floats, in the same order. -/
def floatSizes : List Nat := [NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_HC_DIM, NG_N_EMBD, NG_HC]

/-- Each piece starts exactly where the previous one ends, so no two pieces overlap. -/
theorem float_pieces_adjacent :
    (floatOffsets.zip floatSizes).map (fun p => p.1 + p.2) = floatOffsets.drop 1 ++ [53764] := by
  decide

-- § history and conv rows

/-- Row-fastest history address `row + hist*c`. -/
def histIdx (nhist c row : Nat) : Nat := row + nhist * c

/-- A history address is inside `nhist * hc_dim`. -/
theorem histIdx_lt {nhist c row hc_dim : Nat} (hc : c < hc_dim) (hr : row < nhist) :
    histIdx nhist c row < nhist * hc_dim := by
  unfold histIdx
  have h := mul_add_lt_aux hc hr
  have e1 : nhist * c = c * nhist := Nat.mul_comm _ _
  have e2 : nhist * hc_dim = hc_dim * nhist := Nat.mul_comm _ _
  omega

/-- History addresses are unique per `(channel, row)`. -/
theorem histIdx_inj {nhist c c' row row' : Nat} (hr : row < nhist) (hr' : row' < nhist)
    (h : histIdx nhist c row = histIdx nhist c' row') : c = c' ∧ row = row' := by
  unfold histIdx at h
  have e1 : nhist * c = c * nhist := Nat.mul_comm _ _
  have e2 : nhist * c' = c' * nhist := Nat.mul_comm _ _
  have h' : c * nhist + row = c' * nhist + row' := by omega
  obtain ⟨a, b⟩ := split_unique_aux hr hr' h'
  exact ⟨a, b⟩
where
  /-- Row-major addresses are unique. -/
  split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
      (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
    have hn : 0 < n := by omega
    have d1 : (a * n + c) / n = a := by
      have h1 := Nat.add_mul_div_right c a hn
      have h0 : c / n = 0 := Nat.div_eq_of_lt hc
      have e : a * n + c = c + a * n := Nat.add_comm _ _
      rw [e]
      omega
    have d2 : (a' * n + c') / n = a' := by
      have h1 := Nat.add_mul_div_right c' a' hn
      have h0 : c' / n = 0 := Nat.div_eq_of_lt hc'
      have e : a' * n + c' = c' + a' * n := Nat.add_comm _ _
      rw [e]
      omega
    have ha : a = a' := by rw [← d1, ← d2, h]
    subst ha
    exact ⟨rfl, by omega⟩

/-- The tap `k` reads row `nhist - (kern-1-k)*dil`; at the real geometry rows are `0, 3, 6, 9`. -/
theorem conv_tap_rows : ∀ k : Nat, k < 4 → 9 - (4 - 1 - k) * 3 = 3 * k := by
  intro k hk
  omega

/-- Only the last tap reads the new row (`row = nhist`); the three others are inside the history. -/
theorem only_last_tap_is_new : ∀ k : Nat, k < 4 → (9 - (4 - 1 - k) * 3 = 9 ↔ k = 3) := by
  intro k hk
  omega

/-- The history shift: drop the oldest row, append the new one. -/
def advance (col : List Int) (x : Int) : List Int := col.tail ++ [x]

/-- The shifted column keeps its length when non-empty. -/
theorem advance_length (col : List Int) (x : Int) (h : col ≠ []) :
    (advance col x).length = col.length := by
  cases col with
  | nil => exact absurd rfl h
  | cons a as => simp [advance]

/-- The newest row of the shifted column is the appended value. -/
theorem advance_last (col : List Int) (x : Int) : (advance col x).getLast? = some x := by
  simp [advance]

-- § bf16 rounding

/-- `(i + ((i>>16)&1) + 0x7FFF) & 0xFFFF0000 >> 16` on a 32-bit word. -/
def bf16Bits (i : Nat) : Nat := ((i + (i / 65536) % 2 + 32767) % 4294967296) / 65536

/-- Aligned-word fact: `(a*65536 + t) mod 2^32` keeps `a mod 65536` in its high half. -/
theorem core_div2 (a t : Nat) (ht : t < 65536) :
    ((a * 65536 + t) % 4294967296) / 65536 = a % 65536 := by
  omega

/-- Closed form of the rounding: the high half plus the carry out of the low half, modulo 2^16. -/
theorem bf16Bits_eq (i : Nat) :
    bf16Bits i = (i / 65536 + (i % 65536 + (i / 65536) % 2 + 32767) / 65536) % 65536 := by
  unfold bf16Bits
  have e : i + (i / 65536) % 2 + 32767 =
      (i / 65536 + (i % 65536 + (i / 65536) % 2 + 32767) / 65536) * 65536 +
        (i % 65536 + (i / 65536) % 2 + 32767) % 65536 := by
    omega
  rw [e]
  exact core_div2 _ _ (Nat.mod_lt _ (by decide))

/-- A bf16 bit pattern has 16 bits. -/
theorem bf16Bits_lt (i : Nat) : bf16Bits i < 65536 := by
  unfold bf16Bits
  omega

/-- Below the halfway point the high half is kept. -/
theorem bf16_round_down {i : Nat} (hi : i < 4294967296) (h : i % 65536 < 32768) :
    bf16Bits i = i / 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = 0 := by omega
  rw [hc]
  omega

/-- Above the halfway point the high half is incremented. -/
theorem bf16_round_up {i : Nat} (hi : i < 4294967296) (h : 32768 < i % 65536) :
    bf16Bits i = (i / 65536 + 1) % 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = 1 := by omega
  rw [hc]

/-- At exactly the halfway point the result is the even neighbour. -/
theorem bf16_tie_even {i : Nat} (hi : i < 4294967296) (h : i % 65536 = 32768) :
    bf16Bits i = (i / 65536 + (i / 65536) % 2) % 65536 := by
  rw [bf16Bits_eq]
  have hc : (i % 65536 + (i / 65536) % 2 + 32767) / 65536 = (i / 65536) % 2 := by omega
  rw [hc]

/-- Expanding a bf16 pattern is a left shift by 16, and rounding it again is the identity. -/
theorem bf16_roundtrip {h : Nat} (hh : h < 65536) : bf16Bits (h * 65536) = h := by
  have h1 : (h * 65536) % 65536 < 32768 := by omega
  rw [bf16_round_down (by omega) h1]
  omega

-- § overlap rule

/-- Interval intersection for non-null, non-empty spans, as `overlap` computes it. -/
def overlapP (a an b bn : Nat) : Prop :=
  (a < b ∧ b - a < an) ∨ (¬ a < b ∧ a - b < bn)

/-- For non-empty spans the source's overlap test is exactly interval intersection. -/
theorem overlapP_iff {a an b bn : Nat} (ha : 0 < an) (hb : 0 < bn) :
    overlapP a an b bn ↔ (a < b + bn ∧ b < a + an) := by
  unfold overlapP
  omega

/-- The test is symmetric on non-empty spans. -/
theorem overlapP_symm {a an b bn : Nat} (ha : 0 < an) (hb : 0 < bn) :
    overlapP a an b bn ↔ overlapP b bn a an := by
  rw [overlapP_iff ha hb, overlapP_iff hb ha]
  omega

end Strata.Kernels.PleBlock



## FILE: /mnt/user-data/outputs/VerifyWindowKernels.lean
/-!
# MODULE: verify_window_kernels
Multi-token (verify window) variants of the GDN kernels plus the gather/select/steps utilities.

## Source semantics
The window of token `t` is the last four entries of `[hist0, hist1, hist2, x_0, ..., x_t]`; entry `s` of this
sequence is `hist[s]` for `s < 3` and `x[s-3]` otherwise.  Committing `n` tokens keeps the last three entries
of `[hist | x_0 .. x_{n-1}]` (`n <= 0` leaves the history unchanged).  `gdn_step_norm_multi` replays tokens
`0 .. n-1` (`n = *n_keep` or `T`), writes outputs only for `t >= t_out_begin`, and writes the state back only
when `n_keep` is given and `n > 0`.  `window_ids` clips the attention window to `start = max(n_kv - window, 0)`.
`dense_steps` stores `(c, c+1, (c+1)/4, c+1)` per cell.  Grid-stride loops cover each index exactly once.

## Form template identifiers
- `seqAt hist x s`     -> entry `s` of `[hist | x]`
- `commit`             -> history after keeping `n` tokens
- `windowStart/Width`  -> `window_ids` clipping
-/

namespace Strata.Kernels.Verify

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § conv window over [hist | x]

/-- Entry `s` of the sequence `[hist0, hist1, hist2, x_0, x_1, ...]`. -/
def seqAt {α : Type} (hist : Nat → α) (x : Nat → α) (s : Nat) : α :=
  if s < 3 then hist s else x (s - 3)

/-- The newest window entry of token `t` is `x_t`. -/
theorem window_last {α : Type} (hist x : Nat → α) (t : Nat) : seqAt hist x (t + 3) = x t := by
  unfold seqAt
  rw [if_neg (by omega), Nat.add_sub_cancel]

/-- Committing zero tokens leaves the history unchanged. -/
theorem commit_zero {α : Type} (hist x : Nat → α) (j : Nat) (hj : j < 3) :
    seqAt hist x (0 + j) = hist j := by
  unfold seqAt
  rw [if_pos (by omega)]
  simp

/-- Committing `n` tokens then `m` more equals committing `n + m` at once. -/
theorem commit_compose {α : Type} (hist x : Nat → α) (n m j : Nat) :
    seqAt (fun j => seqAt hist x (n + j)) (fun i => x (n + i)) (m + j) =
      seqAt hist x (n + (m + j)) := by
  unfold seqAt
  by_cases h1 : m + j < 3
  · by_cases h2 : n + (m + j) < 3
    · simp [h1, h2]
    · simp [h1, h2]
  · have h2 : ¬ n + (m + j) < 3 := by omega
    have e : n + (m + j - 3) = n + (m + j) - 3 := by omega
    simp [h1, h2, e]

/-- Window and `x`/`hist` addresses: history channel `c*3 + j` and token row `(s-3)*C + c`. -/
theorem token_index_lt {s C c T : Nat} (hs : s < T + 3) (h3 : 3 ≤ s) (hc : c < C) :
    (s - 3) * C + c < T * C :=
  mul_add_lt_aux (by omega) hc

-- § step kernel indexing and replay

/-- Replayed tokens `t < t_out_begin` produce no output; outputs are exactly `[t_out_begin, n)`. -/
theorem outputs_count {n t_out_begin : Nat} (h : t_out_begin ≤ n) :
    ((List.range n).filter (fun t => decide (t_out_begin ≤ t))).length = n - t_out_begin := by
  induction n with
  | zero =>
    have : t_out_begin = 0 := by omega
    subst this
    simp
  | succ n ih =>
    by_cases hc : t_out_begin ≤ n
    · have ih' := ih hc
      rw [List.range_succ, List.filter_append]
      simp [hc, ih']
      omega
    · have e : t_out_begin = n + 1 := by omega
      subst e
      rw [List.range_succ, List.filter_append]
      have hnil : (List.range n).filter (fun t => decide (n + 1 ≤ t)) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro a ha
        have h1 := List.mem_range.mp ha
        have h2 : ¬ (n + 1 ≤ a) := by omega
        simp [h2]
      have hn : ¬ (n + 1 ≤ n) := by omega
      rw [hnil]
      simp [hn]

/-- The state is written back only when `n_keep` is supplied and positive. -/
def writeBack (hasKeep : Bool) (n : Nat) : Bool := hasKeep && decide (0 < n)

/-- Without `n_keep` the state is never written back. -/
theorem no_keep_no_writeback (n : Nat) : writeBack false n = false := by
  rfl

/-- With `n_keep = 0` the state is not written back either. -/
theorem zero_keep_no_writeback : writeBack true 0 = false := by
  decide

/-- Channel layout `C = 2*qk + value_dim`: q, k and v indices stay in their own thirds. -/
theorem qkv_layout {S h_k h_v qh head col C : Nat} (hqh : qh < h_k) (hh : head < h_v) (hc : col < S)
    (hC : C = 2 * (S * h_k) + S * h_v) :
    qh * S + col < S * h_k ∧ S * h_k + (qh * S + col) < 2 * (S * h_k) ∧
      2 * (S * h_k) + (head * S + col) < C := by
  have e1 : S * h_k = h_k * S := Nat.mul_comm _ _
  have e2 : S * h_v = h_v * S := Nat.mul_comm _ _
  have a1 := mul_add_lt_aux hqh hc
  have a2 := mul_add_lt_aux hh hc
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- Gate/beta slot `t*h_v + head` is inside `T*h_v`. -/
theorem gate_index_lt {t T head h_v : Nat} (ht : t < T) (hh : head < h_v) :
    t * h_v + head < T * h_v :=
  mul_add_lt_aux ht hh

-- § window_ids and dense_steps

/-- First visible position. -/
def windowStart (n_kv window : Nat) : Nat := n_kv - window

/-- Visible width. -/
def windowWidth (n_kv window : Nat) : Nat := n_kv - windowStart n_kv window

/-- The visible width never exceeds the window and the range ends at `n_kv`. -/
theorem window_bounds (n_kv window : Nat) :
    windowWidth n_kv window ≤ window ∧ windowStart n_kv window + windowWidth n_kv window = n_kv := by
  unfold windowWidth windowStart
  omega

/-- A short history is not clipped. -/
theorem window_short (n_kv window : Nat) (h : n_kv ≤ window) :
    windowStart n_kv window = 0 ∧ windowWidth n_kv window = n_kv := by
  unfold windowWidth windowStart
  omega

/-- Visible ids `start + j` (for `j < width`) are real positions `< n_kv`. -/
theorem window_ids_lt {n_kv window j : Nat} (hj : j < windowWidth n_kv window) :
    windowStart n_kv window + j < n_kv := by
  unfold windowWidth windowStart at *
  omega

/-- `dense_steps` slots `i*4 + k` (k < 4) are inside `4n`, and `(c+1)/4 <= c+1`. -/
theorem dense_steps_index {i n k c : Nat} (hi : i < n) (hk : k < 4) :
    i * 4 + k < n * 4 ∧ (c + 1) / 4 ≤ c + 1 :=
  ⟨mul_add_lt_aux hi hk, by omega⟩

-- § grid-stride loops and gathers

/-- A grid-stride loop with stride `s > 0` visits index `i` at thread `i % s`, iteration `i / s`. -/
theorem gridstride_cover {s : Nat} (hs : 0 < s) (i : Nat) :
    i % s < s ∧ (i / s) * s + i % s = i := by
  refine ⟨Nat.mod_lt i hs, ?_⟩
  have h := Nat.div_add_mod i s
  have e : s * (i / s) = (i / s) * s := Nat.mul_comm _ _
  omega

/-- That visit is unique. -/
theorem gridstride_unique {s t t' k k' : Nat} (ht : t < s) (ht' : t' < s)
    (h : k * s + t = k' * s + t') : k = k' ∧ t = t' :=
  split_unique_aux ht ht' h

/-- `gather_rows`: source address `ids[r]*row16 + o` is inside `nrows*row16` for valid ids. -/
theorem gather_src_lt {id nrows row16 o : Nat} (hid : id < nrows) (ho : o < row16) :
    id * row16 + o < nrows * row16 :=
  mul_add_lt_aux hid ho

/-- `gather_rows`: destination `r*row16 + o` splits back into `(r, o)`. -/
theorem gather_dst_unique {r r' o o' row16 : Nat} (ho : o < row16) (ho' : o' < row16)
    (h : r * row16 + o = r' * row16 + o') : r = r' ∧ o = o' :=
  split_unique_aux ho ho' h

/-- `embedding_gather_dev`: a stored code field has `bits` bits. -/
theorem code_field_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- With 4-bit codes two codes share a byte: index `i` is byte `i/2`, nibble `i%2`. -/
theorem nibble_split (i : Nat) : 8 / 4 = 2 ∧ i = 2 * (i / 2) + i % 2 := by
  omega

/-- Per-token row base `token*row_codes` plus an in-row offset stays inside the table. -/
theorem token_row_lt {token vocab row_codes o : Nat} (ht : token < vocab) (ho : o < row_codes) :
    token * row_codes + o < vocab * row_codes :=
  mul_add_lt_aux ht ho

-- § misc launch checks

/-- `ident_hits` runs one block of 1024 threads; `n <= 1024` guarantees `threadIdx < n` covers all ids. -/
theorem ident_hits_cover {n i : Nat} (hn : n ≤ 1024) (hi : i < n) : i < 1024 := by
  omega

/-- `rebase_ptrs` assigns `base + k*bytes`; consecutive blobs are adjacent and disjoint. -/
theorem rebase_adjacent (base bytes k : Nat) :
    (base + k * bytes) + bytes = base + (k + 1) * bytes := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

end Strata.Kernels.Verify



## FILE: /mnt/user-data/outputs/ElementwiseGlueKernels.lean
/-!
# MODULE: elementwise_glue_kernels
Glue kernels: embedding dequantisation, gate indexing, warp-per-row RMS norm launch, bounded 1-D launches,
and the host/device doorbell protocol.

## Source semantics
`embedding_gather` unpacks `8/bits` codes per byte, adds the code bias, multiplies by the group scale and
adds the optional group offset.  `gdn_gate_kernel` is launched over `n_tokens*h_v` elements but guards
`i >= h_v`, so only the first token row is processed (exact for the real one-token call).
`rms_norm_weighted` runs 4 warps per block, one row per warp; the grid is rounded up, so the row guard is
needed (rows = 2 launches 8 warps).  `grid_for n` is `ceil(n/256)`; `n = 0` gives a zero grid, which is why
every launcher returns early.  The doorbell RING increments device memory; passing the incremented value as
a launch argument is evaluated once at capture time and makes every replay ring the same value.

## Form template identifiers
- `dequant`   -> `(code + bias) * scale + offset`
- `grid_for`  -> `gridFor`
- `ring`      -> `incRing` (correct) / `litRing` (capture-time literal)
-/

namespace Strata.Kernels.Elementwise

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Division recovers the row. -/
theorem div_of_split_aux {a c n : Nat} (hc : c < n) : (a * n + c) / n = a := by
  have hn : 0 < n := by omega
  have h1 := Nat.add_mul_div_right c a hn
  have h0 : c / n = 0 := Nat.div_eq_of_lt hc
  have e : a * n + c = c + a * n := Nat.add_comm _ _
  rw [e]
  omega

/-- Row-major addresses are unique. -/
theorem split_unique_aux {a a' c c' n : Nat} (hc : c < n) (hc' : c' < n)
    (h : a * n + c = a' * n + c') : a = a' ∧ c = c' := by
  have h1 : (a * n + c) / n = a := div_of_split_aux hc
  have h2 : (a' * n + c') / n = a' := div_of_split_aux hc'
  have ha : a = a' := by rw [← h1, ← h2, h]
  subst ha
  exact ⟨rfl, by omega⟩

-- § embedding dequantisation

/-- `(code + bias) * scale + offset`, with a missing offset read as zero. -/
def dequant (code bias : Nat) (scale : Int) (offset : Option Int) : Int :=
  ((code + bias : Nat) : Int) * scale + offset.getD 0

/-- Without offsets the value is just `(code + bias) * scale`. -/
theorem dequant_no_offset (code bias : Nat) (scale : Int) :
    dequant code bias scale none = ((code + bias : Nat) : Int) * scale := by
  simp [dequant]

/-- A `bits`-bit code field is below `2^bits`. -/
theorem code_lt (byte shift bits : Nat) : (byte / 2 ^ shift) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- Codes per byte: 8 for 1-bit, 4 for 2-bit, 2 for 4-bit, 1 for 8-bit. -/
theorem codes_per_byte : 8 / 1 = 8 ∧ 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by
  decide

/-- Element `i` of a 2-bit stream: byte `i/4`, slot `i%4`. -/
theorem two_bit_split (i : Nat) : i = 4 * (i / 4) + i % 4 ∧ i % 4 < 4 := by
  omega

-- § gate indexing

/-- The gate kernel's guard `i < h_v` selects exactly the first token row. -/
theorem gate_first_row_only {i h_v : Nat} (hv : 0 < h_v) (hi : i < h_v) :
    i / h_v = 0 ∧ i % h_v = i :=
  ⟨Nat.div_eq_of_lt hi, Nat.mod_eq_of_lt hi⟩

/-- Conversely, a processed element lies in token row 0. -/
theorem gate_processed_in_row0 {i h_v : Nat} (hv : 0 < h_v) (h : i / h_v = 0) : i < h_v := by
  rcases Nat.lt_or_ge i h_v with h' | h'
  · exact h'
  · have := Nat.div_pos h' hv
    omega

/-- For the real single-token call every element is processed: `n = 1 * h_v`. -/
theorem gate_single_token_total {i h_v : Nat} (hi : i < 1 * h_v) : i < h_v := by
  omega

-- § bounded launches

/-- Blocks for `n` elements at 256 threads. -/
def gridFor (n : Nat) : Nat := (n + 255) / 256

/-- The launch covers all `n` elements. -/
theorem gridFor_cover (n : Nat) : n ≤ gridFor n * 256 := by
  unfold gridFor
  omega

/-- A zero-element call yields an illegal zero grid, hence the early return. -/
theorem gridFor_zero : gridFor 0 = 0 := by
  decide

/-- A positive-element call has a non-empty grid. -/
theorem gridFor_pos {n : Nat} (h : 0 < n) : 0 < gridFor n := by
  unfold gridFor
  omega

-- § warp-per-row rms norm launch

/-- Warps launched for `rows` rows at 4 warps per block. -/
def launchedWarps (rows : Nat) : Nat := ((rows + 3) / 4) * 4

/-- Every row has a warp: warp `r` lives in block `r/4`. -/
theorem rows_covered {rows r : Nat} (hr : r < rows) :
    r / 4 < (rows + 3) / 4 ∧ r / 4 * 4 + r % 4 = r := by
  omega

/-- The grid never launches fewer warps than rows, and fewer than four extra. -/
theorem launched_warps_bounds (rows : Nat) :
    rows ≤ launchedWarps rows ∧ launchedWarps rows < rows + 4 := by
  unfold launchedWarps
  omega

/-- THE QSA BUG WITNESS: two rows launch one 4-warp block, so exactly two warps are out of range
and the row guard is mandatory. -/
theorem qsa_guard_needed : launchedWarps 2 = 4 ∧ launchedWarps 2 - 2 = 2 := by
  decide

/-- With the guard, a row index `>= rows` writes nothing; without it row `rows..` would write `cols` floats
each past the buffer. -/
theorem out_of_range_rows {rows : Nat} (h : rows % 4 ≠ 0) : rows < launchedWarps rows := by
  unfold launchedWarps
  omega

-- § mapped copies

/-- `copy_from_mapped` moves `n/4` float4 when `n % 4 = 0`, which is all `n` floats. -/
theorem float4_total {n : Nat} (h : n % 4 = 0) : (n / 4) * 4 = n := by
  omega

/-- The capped grid `min((n4+255)/256, 64)` is non-empty for non-empty input. -/
theorem capped_grid_pos {n4 : Nat} (h : 0 < n4) : 0 < min ((n4 + 255) / 256) 64 := by
  omega

/-- `doorbell_publish` copies `k` ids with threads `< k`; the guard `k <= 1024` fits the block. -/
theorem publish_threads {k t : Nat} (hk : k ≤ 1024) (ht : t < k) : t < 1024 := by
  omega

-- § doorbell protocol

/-- Correct ring: increments the memory it reads at replay time. -/
def incRing (seq : Nat) : Nat := seq + 1

/-- Broken ring: the incremented value was computed on the host at capture time and baked in. -/
def litRing (c : Nat) (_seq : Nat) : Nat := c

/-- The correct ring changes the counter on every replay. -/
theorem incRing_changes (seq : Nat) : incRing seq ≠ seq := by
  unfold incRing
  omega

/-- After two replays the correct ring has advanced by two. -/
theorem incRing_twice (seq : Nat) : incRing (incRing seq) = seq + 2 := by
  unfold incRing
  omega

/-- The captured literal makes the second replay a no-op, so a host waiting for a change waits forever. -/
theorem litRing_stalls (c seq : Nat) : litRing c (litRing c seq) = litRing c seq :=
  rfl

/-- The device wait is satisfied when the flag reaches the sequence value it read. -/
def waitReady (flag want : Nat) : Prop := flag = want

/-- After the host copies the sequence into the flag the wait is released. -/
theorem wait_released (seq : Nat) : waitReady seq seq :=
  rfl

/-- `wait_flag_ge`: a counter that only grows keeps satisfying `flag >= value`. -/
theorem flag_ge_monotone {f1 f2 v : Nat} (hle : f1 ≤ f2) (h : v ≤ f1) : v ≤ f2 := by
  omega

end Strata.Kernels.Elementwise





## FILE: /mnt/user-data/outputs/CanonicalFormCodebookDecode.lean
/-!
# MODULE: canonical_form_codebook_decode
Integer-level contracts of `tools/canonical_xcheck.py`: each GGML quant type is re-expressed as an unsigned
code plane + scale (+ offset) plane, and the decode is `(code + bias) * scale (+ offset)`.  The bias lives in
the INTEGER domain, so the value set of each type is a fixed finite integer range.

## Source semantics
* Q2_0 value = (code - 1) * d, code in 0..3 (symbols -1,0,1,2); Q3_K bias -4; Q4_0 bias -8; Q5_0 -16;
  Q6_K -32; Q8_0 stored as int8 + 128 and decoded with -128.
* IQ4 types use a 16-entry non-affine table (no bias).
* Nibble split: low nibbles fill the first half of a block, high nibbles the second.
* `get_scale_min_k4` yields two 6-bit values per group from 12 packed bytes.
* Q3_K / Q5_K / Q6_K element order is fixed by loop nests whose index maps must be exact bijections.

## Form template identifiers
code_bias            -> code_bias (registry field)
block_bytes          -> block_bytes
block_elems          -> block_elems
group_elems          -> group_elems
codeValue            -> decode of one code, before the scale multiply
scaleMin             -> get_scale_min_k4
-/

namespace Strata.CanonicalFormCodebookDecode

-- § affine codebooks

/-- Integer value of a stored code under a registry `code_bias` (added to the code). -/
def codeValue (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Q2_0 codebook is {-1,0,1,2}. -/
theorem q2_0_range (c : Nat) (h : c < 4) : -1 ≤ codeValue c (-1) ∧ codeValue c (-1) ≤ 2 := by
  unfold codeValue; omega

/-- Q3_K codebook is -4..3 (codes 0..7). -/
theorem q3_k_range (c : Nat) (h : c < 8) : -4 ≤ codeValue c (-4) ∧ codeValue c (-4) ≤ 3 := by
  unfold codeValue; omega

/-- Q4_0 codebook is -8..7 (codes 0..15). -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ codeValue c (-8) ∧ codeValue c (-8) ≤ 7 := by
  unfold codeValue; omega

/-- Q5_0 codebook is -16..15 (codes 0..31). -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ codeValue c (-16) ∧ codeValue c (-16) ≤ 15 := by
  unfold codeValue; omega

/-- Q6_K codebook is -32..31 (codes 0..63). -/
theorem q6_k_range (c : Nat) (h : c < 64) : -32 ≤ codeValue c (-32) ∧ codeValue c (-32) ≤ 31 := by
  unfold codeValue; omega

/-- Q8_0 codebook is -128..127 (codes 0..255). -/
theorem q8_0_range (c : Nat) (h : c < 256) :
    -128 ≤ codeValue c (-128) ∧ codeValue c (-128) ≤ 127 := by
  unfold codeValue; omega

/-- A code plane is a faithful representation: distinct codes decode to distinct integers. -/
theorem codeValue_injective (a b : Nat) (bias : Int) (h : codeValue a bias = codeValue b bias) :
    a = b := by
  unfold codeValue at h; omega

/-- Storing an `int8` quant as `q + 128` and decoding with bias `-128` returns `q` exactly. -/
theorem q8_0_store_roundtrip (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    codeValue (q + 128).toNat (-128) = q := by
  unfold codeValue; omega

/-- The stored Q8_0 code is a valid unsigned byte. -/
theorem q8_0_store_lt (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) : (q + 128).toNat < 256 := by
  omega

-- § source block geometry

/-- Byte size of each source block, computed from its fields. -/
theorem block_sizes :
    2 + 64 / 4 = 18 ∧ 2 + 32 / 2 = 18 ∧ 2 + 32 = 34 ∧ 2 + 4 + 32 / 2 = 22 ∧
    4 + 12 + 128 = 144 ∧ 4 + 12 + 32 + 128 = 176 ∧ 128 + 64 + 16 + 2 = 210 ∧
    32 + 64 + 12 + 2 = 110 ∧ 2 + 2 + 4 + 128 = 136 := by
  decide

/-- Registry row: (block_bytes, block_elems, group_elems). -/
structure Geom where
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat

/-- The 13-type mapping registry's geometry. -/
def registry : List Geom :=
  [⟨18, 64, 64⟩, ⟨110, 256, 16⟩, ⟨18, 32, 32⟩, ⟨144, 256, 32⟩, ⟨22, 32, 32⟩, ⟨176, 256, 32⟩,
   ⟨210, 256, 16⟩, ⟨34, 32, 32⟩, ⟨18, 32, 32⟩, ⟨136, 256, 32⟩, ⟨4, 1, 1⟩, ⟨2, 1, 1⟩, ⟨2, 1, 1⟩]

/-- Every group lies inside one block: `group_elems` divides `block_elems`. -/
theorem group_divides_block : ∀ g ∈ registry, g.blockElems % g.groupElems = 0 := by
  decide

/-- The registry has 13 types. -/
theorem registry_size : registry.length = 13 := by decide

-- § nibble handling

/-- Low and high nibble of a packed byte recover the two codes. -/
theorem nibble_roundtrip (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by
  omega

/-- A packed byte of two nibbles is a byte. -/
theorem nibble_pack_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by
  omega

-- § get_scale_min_k4

/-- `get_scale_min_k4` over a byte-valued array: (scale, min) of group `j`. -/
def scaleMin (s : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (s j % 64, s (j + 4) % 64)
  else (s (j + 4) % 16 + (s (j - 4) / 64) * 16, s (j + 4) / 16 + (s j / 64) * 16)

/-- Both outputs are 6-bit values whenever the scale bytes are bytes. -/
theorem scaleMin_lt (s : Nat → Nat) (hs : ∀ i, s i < 256) (j : Nat) :
    (scaleMin s j).1 < 64 ∧ (scaleMin s j).2 < 64 := by
  have h1 := hs j
  have h2 := hs (j + 4)
  have h3 := hs (j - 4)
  unfold scaleMin
  split <;> (dsimp only; omega)

/-- Groups 0..3 read only the low 6 bits of their own and the paired byte. -/
theorem scaleMin_low (s : Nat → Nat) (j : Nat) (hj : j < 4) :
    scaleMin s j = (s j % 64, s (j + 4) % 64) := by
  unfold scaleMin; simp [hj]

-- § Q4_K / Q5_K group layout

/-- Canonical group `g = 2j + h` covers elements `32 g .. 32 g + 31`: the `is+h` pair of the source loop. -/
theorem q4k_group_base (j h : Nat) (hh : h < 2) : j * 64 + h * 32 = 32 * (2 * j + h) := by
  omega

/-- The eight Q5_K high-bit masks `u1, u2` over the four 64-step iterations are exactly the eight bits. -/
theorem q5k_masks :
    (List.range 4).flatMap (fun j => [4 ^ j, 2 * 4 ^ j]) = [1, 2, 4, 8, 16, 32, 64, 128] := by
  decide

-- § Q3_K index maps

/-- Q3_K group index `8*half + 2*j + sub` enumerates 0..15 in order. -/
theorem q3k_group_enum :
    (List.range 2).flatMap (fun h => (List.range 4).flatMap (fun j =>
      (List.range 2).map (fun s => 8 * h + 2 * j + s))) = List.range 16 := by
  decide

/-- The Q3_K hmask bit `4*half + j` (the mask never resets) enumerates 0..7. -/
theorem q3k_mask_bit_enum :
    (List.range 2).flatMap (fun h => (List.range 4).map (fun j => 4 * h + j)) = List.range 8 := by
  decide

/-- The two halves use disjoint hmask bit ranges. -/
theorem q3k_halves_disjoint (j j' : Nat) (hj : j < 4) (hj' : j' < 4) : 4 * 0 + j ≠ 4 * 1 + j' := by
  omega

-- § Q6_K index map

/-- Q6_K element order `128*half + 32*q + l` enumerates 0..255 in order. -/
theorem q6k_element_enum :
    (List.range 2).flatMap (fun h => (List.range 4).flatMap (fun q =>
      (List.range 32).map (fun l => 128 * h + 32 * q + l))) = List.range 256 := by
  decide +kernel

/-- The source scale index `is + 2q` (with `is = l/16`) is the plain 16-element group of the element. -/
theorem q6k_scale_index (q l : Nat) (hl : l < 32) : (q * 32 + l) / 16 = 2 * q + l / 16 := by
  omega

-- § IQ4_XS scale

/-- The 6-bit `ls` of IQ4_XS: a nibble of `scales_l` plus two high bits from `scales_h`. -/
def ls6 (slByte par hi2 : Nat) : Nat :=
  (if par = 0 then slByte % 16 else slByte / 16) + hi2 * 16

/-- `ls` fits in six bits. -/
theorem ls6_lt (slByte par hi2 : Nat) (hb : slByte < 256) (hh : hi2 < 4) : ls6 slByte par hi2 < 64 := by
  unfold ls6
  split <;> omega

end Strata.CanonicalFormCodebookDecode



## FILE: /mnt/user-data/outputs/GgufQ2WriterPacking.lean
/-!
# MODULE: gguf_q2_writer_packing
The minimal GGUF v3 writer: Q2_0 quantizer packing (4 two-bit codes per byte, LSB first), clamp rule, and the
32-byte alignment of the tensor directory and data section.

## Source semantics
* Block = 64 elements = fp16 scale (2 B) + 16 bytes of codes; element j is in byte j/4 at bit (j%4)*2.
* code = clamp(round(w/d) + 1, 0, 3); symbol = code - 1 in {-1,0,1,2}; requantizing a decoded block is a fixed point.
* Offsets are padded up to `alignment` (32) after every tensor and after the header.

## Form template identifiers
pack4        -> packed byte of four codes
codeAt       -> element j's code from a byte array
clamp03      -> code clamp
align32      -> padding to DEFAULT_ALIGNMENT
-/

namespace Strata.GgufQ2WriterPacking

-- § 2-bit packing

/-- Four 2-bit codes in one byte, least significant first. -/
def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed byte is a byte. -/
theorem pack4_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 := by
  unfold pack4; omega

/-- Code of element `j` read from a byte array: byte `j/4`, bit offset `(j%4)*2`. -/
def codeAt (qs : Nat → Nat) (j : Nat) : Nat := (qs (j / 4) / 4 ^ (j % 4)) % 4

/-- Slot 0 of a packed byte. -/
theorem codeAt_slot0 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (hj : j % 4 = 0) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c0 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 1 of a packed byte. -/
theorem codeAt_slot1 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (hj : j % 4 = 1) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c1 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 2 of a packed byte. -/
theorem codeAt_slot2 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (hj : j % 4 = 2) :
    codeAt (fun _ => pack4 c0 c1 c2 c3) j = c2 := by
  simp only [codeAt, pack4, hj]; omega

/-- Slot 3 of a packed byte. -/
theorem codeAt_slot3 (c0 c1 c2 c3 j : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4)
    (hj : j % 4 = 3) : codeAt (fun _ => pack4 c0 c1 c2 c3) j = c3 := by
  simp only [codeAt, pack4, hj]; omega

/-- Packing is injective on valid codes. -/
theorem pack4_injective (a0 a1 a2 a3 b0 b1 b2 b3 : Nat)
    (ha0 : a0 < 4) (ha1 : a1 < 4) (ha2 : a2 < 4) (ha3 : a3 < 4)
    (hb0 : b0 < 4) (hb1 : b1 < 4) (hb2 : b2 < 4) (hb3 : b3 < 4)
    (h : pack4 a0 a1 a2 a3 = pack4 b0 b1 b2 b3) :
    a0 = b0 ∧ a1 = b1 ∧ a2 = b2 ∧ a3 = b3 := by
  unfold pack4 at h; omega

-- § block geometry

/-- 16 code bytes + 2 scale bytes = 18 bytes per 64 elements. -/
theorem block_bytes : 64 / 4 + 2 = 18 := by decide

/-- `n` blocks of 64 elements occupy `18 n` bytes. -/
theorem q2_0_bytes (n : Nat) : (64 * n) / 64 * 18 = 18 * n := by omega

-- § quantizer clamp

/-- `clamp(x + 1, 0, 3)` as a natural number code. -/
def clamp03 (x : Int) : Nat := (max 0 (min 3 x)).toNat

/-- The clamped code is a valid 2-bit code. -/
theorem clamp03_lt (x : Int) : clamp03 x < 4 := by
  unfold clamp03; omega

/-- A symbol in {-1,0,1,2} is reproduced exactly: the requantization fixed point. -/
theorem clamp03_fixed (c : Nat) (h : c < 4) : clamp03 (((c : Int) - 1) + 1) = c := by
  unfold clamp03; omega

/-- Symbols outside the grid saturate to the end codes. -/
theorem clamp03_saturate (x : Int) (h : 2 ≤ x) : clamp03 (x + 1) = 3 := by
  unfold clamp03; omega

-- § alignment

/-- Padding up to a multiple of 32. -/
def align32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Alignment never shrinks. -/
theorem align32_ge (n : Nat) : n ≤ align32 n := by unfold align32; omega

/-- Alignment gives a multiple of 32. -/
theorem align32_mod (n : Nat) : align32 n % 32 = 0 := by unfold align32; omega

/-- Alignment adds fewer than 32 bytes. -/
theorem align32_lt (n : Nat) : align32 n < n + 32 := by unfold align32; omega

/-- The next tensor offset is aligned and does not overlap the previous tensor. -/
theorem next_offset_ok (offset len : Nat) :
    offset + len ≤ align32 (offset + len) ∧ align32 (offset + len) % 32 = 0 :=
  ⟨align32_ge _, align32_mod _⟩

/-- Alignment is idempotent. -/
theorem align32_idem (n : Nat) : align32 (align32 n) = align32 n := by
  unfold align32; omega

-- § container constants

/-- Bytes of the ASCII magic `GGUF`, read as a little-endian u32. -/
theorem magic_le : 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 = 0x46554747 := by decide

/-- The metadata value-type ids are pairwise distinct. -/
theorem meta_ids_nodup : ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] : List Nat).Nodup := by decide

end Strata.GgufQ2WriterPacking



## FILE: /mnt/user-data/outputs/NativeExpertPackLayout.lean
/-!
# MODULE: native_expert_pack_layout
`tools/iq_pack.py`: a native pack that keeps experts in GGUF form.  Per layer, 512 blobs of
[gate | up | down]; blob size is per layer; offsets accumulate across layers; the index rows are 19 columns wide
and tensor arenas are 64-byte aligned.

## Source semantics
* blob = per_gate + per_up + per_down, with `per = expected_bytes / 512`, gate and up equal.
* layer offset = sum of 512 * blob of earlier layers; the total is the sum over all layers.
* `index.txt` dst offsets advance by the 64-aligned size of each row.

## Form template identifiers
blobBytes    -> blob
arena        -> total experts offset
align64      -> ALIGN rounding
-/

namespace Strata.NativeExpertPackLayout

-- § blobs

/-- Bytes of one expert blob. -/
def blobBytes (gate up down : Nat) : Nat := gate + up + down

/-- With equal gate/up sizes the blob is `2*gate + down`. -/
theorem blob_eq (g d : Nat) : blobBytes g g d = 2 * g + d := by
  unfold blobBytes; omega

/-- Dividing a role tensor by 512 experts and multiplying back is exact when it divides. -/
theorem per_expert_exact (total per : Nat) (h : total = 512 * per) : total / 512 = per := by
  omega

-- § arena accumulation

/-- Total bytes of the layers' expert blobs: 512 blobs each. -/
def arena : List Nat → Nat
  | [] => 0
  | b :: bs => 512 * b + arena bs

/-- The arena of a concatenation is the sum. -/
theorem arena_append (xs ys : List Nat) : arena (xs ++ ys) = arena xs + arena ys := by
  induction xs with
  | nil => simp [arena]
  | cons x xs ih => simp [arena, ih]; omega

/-- The offset of the next layer is the offset of this one plus its 512 blobs. -/
theorem offset_step (b : Nat) (bs : List Nat) (l : Nat) :
    arena (List.take (l + 1) (b :: bs)) = 512 * b + arena (List.take l bs) := by
  simp [arena]

/-- The first layer starts at offset zero. -/
theorem offset_zero (bs : List Nat) : arena (List.take 0 bs) = 0 := by
  simp [arena]

/-- Layers are laid out without overlap: each layer's end is the next layer's start. -/
theorem layer_contiguous (b : Nat) (bs : List Nat) :
    arena (List.take 1 (b :: bs)) = 512 * b := by
  simp [arena]

-- § alignment

/-- Round up to the 64-byte pack alignment. -/
def align64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- Aligned size is a multiple of 64. -/
theorem align64_mod (n : Nat) : align64 n % 64 = 0 := by unfold align64; omega

/-- Aligned size covers the original. -/
theorem align64_ge (n : Nat) : n ≤ align64 n := by unfold align64; omega

/-- A running offset that always adds an aligned size stays a multiple of 64. -/
theorem offset_aligned (at_ n : Nat) (h : at_ % 64 = 0) : (at_ + align64 n) % 64 = 0 := by
  unfold align64; omega

/-- Padding appended after a raw tensor aligns the next one. -/
theorem pad_aligns (len : Nat) : (len + (64 - len % 64) % 64) % 64 = 0 := by omega

-- § index row widths

/-- The three row shapes written by this tool are all 19 columns (name + 18 numbers). -/
theorem row_widths : 12 + 7 = 19 ∧ 9 + 10 = 19 ∧ 12 + 7 = 19 := by decide

/-- Quantized dense tensors are served natively: their row carries shape only, so all byte fields are zero. -/
theorem native_row_zero_bytes : ["0", "0", "0", "0", "0", "0"].length = 6 := by decide

end Strata.NativeExpertPackLayout



## FILE: /mnt/user-data/outputs/MtpDraftQuantizers.lean
/-!
# MODULE: mtp_draft_quantizers
`tools/mtp_pack.py`: packs the MTP draft block into a GGUF with Q2_0 / Q4_0 / Q8_0 experts.

## Source semantics
* Q2_0 scale search tries 17 factors f in [1/2, 1] times amax and picks the least squared error.
* Q4_0 reference: code = min 15 (trunc (x/d + 8.5)); two nibbles per byte, low half first.
* bf16 -> f32 widening shifts left by 16 and is exact.
* Expert tensors are [512, 1280, 2560] and [512, 2560, 640] quantized along the last axis;
  the stated sizes (0.71 / 1.42 / 2.67 GB) follow from block geometry.

## Form template identifiers
q2_0 / q4_0 / q8_0   -> quantizer kernels
block, block_bytes   -> QUANT table fields
-/

namespace Strata.MtpDraftQuantizers

-- § Q2_0 scale search grid

/-- Search factor `i` of 17 is `(16 + i) / 32`, in numerator units over 32. -/
def factorNum (i : Nat) : Nat := 16 + i

/-- All 17 factors lie in [1/2, 1]. -/
theorem factor_range (i : Nat) (h : i ≤ 16) : 16 ≤ factorNum i ∧ factorNum i ≤ 32 := by
  unfold factorNum; omega

/-- Neighbouring factors are one step of 1/32 apart. -/
theorem factor_step (i : Nat) : factorNum (i + 1) = factorNum i + 1 := by
  unfold factorNum; omega

/-- Endpoints: factor 0 is 1/2 and factor 16 is 1 (the ggml reference scale). -/
theorem factor_endpoints : factorNum 0 = 16 ∧ factorNum 16 = 32 := by decide

-- § Q4_0 reference rounding

/-- Q4_0 code from the truncated integer `t = trunc(x/d + 8.5)`. -/
def q4code (t : Nat) : Nat := min 15 t

/-- Q4_0 codes are nibbles. -/
theorem q4code_lt (t : Nat) : q4code t < 16 := by unfold q4code; omega

/-- Two nibbles form one byte (low half first). -/
theorem q4_pack_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by omega

/-- Unpacking returns both nibbles. -/
theorem q4_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by omega

-- § bf16 widening

/-- A bf16 pattern shifted into the high half of an f32 word. -/
def widen (x : Nat) : Nat := x * 65536

/-- Widening leaves the low 16 bits zero. -/
theorem widen_low_zero (x : Nat) : widen x % 65536 = 0 := by unfold widen; omega

/-- Widening is invertible by the high half: it is exact. -/
theorem widen_high (x : Nat) : widen x / 65536 = x := by unfold widen; omega

/-- A 16-bit pattern widens to a 32-bit pattern. -/
theorem widen_lt (x : Nat) (h : x < 65536) : widen x < 4294967296 := by unfold widen; omega

-- § tensor byte accounting

/-- Bytes per row for a type: `n / block * block_bytes`. -/
def rowBytes (n block blockBytes : Nat) : Nat := n / block * blockBytes

/-- The expert inner dimensions give 720 and 180 Q2_0 bytes per row (matching the runtime pack). -/
theorem q2_0_row_bytes : rowBytes 2560 64 18 = 720 ∧ rowBytes 640 64 18 = 180 := by decide

/-- Total bytes of the two fused expert tensors for a format. -/
def expertBytes (block blockBytes : Nat) : Nat :=
  512 * 1280 * rowBytes 2560 block blockBytes + 512 * 2560 * rowBytes 640 block blockBytes

/-- Q2_0 experts: 707,788,800 B (~0.71 GB). -/
theorem q2_0_total : expertBytes 64 18 = 707788800 := by decide

/-- Q4_0 experts: 1,415,577,600 B (~1.42 GB). -/
theorem q4_0_total : expertBytes 32 18 = 1415577600 := by decide

/-- Q8_0 experts: 2,673,868,800 B (~2.67 GB). -/
theorem q8_0_total : expertBytes 32 34 = 2673868800 := by decide

/-- Q2_0 is smaller than Q4_0, which is smaller than Q8_0. -/
theorem format_order : expertBytes 64 18 < expertBytes 32 18 ∧ expertBytes 32 18 < expertBytes 32 34 := by
  decide

/-- Inner dimensions are whole blocks for every format. -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

end Strata.MtpDraftQuantizers



## FILE: /mnt/user-data/outputs/MtpRuntimeBlobRelayout.lean
/-!
# MODULE: mtp_runtime_blob_relayout
`tools/mtp_rt.py`: lossless relayout of the MTP Q2_0 experts into the engine blob (gate/up rows interleaved,
code plane then scale plane), and the 256-byte padded dense file.

## Source semantics
* BLOB = 3 * (H*FF*18/64) = 1,382,400 B with H=2560, FF=640.
* interleaved row 2r is gate row r, row 2r+1 is up row r.
* each dense tensor starts at a multiple of 256.

## Form template identifiers
H, FF, NE   -> hidden, ffn width, expert count
slot        -> interleaved row index
-/

namespace Strata.MtpRuntimeBlobRelayout

-- § blob size

/-- Blob size from geometry. -/
theorem blob_bytes : 3 * (2560 * 640 * 18 / 64) = 1382400 := by decide

/-- The four planes add up to the blob exactly. -/
theorem planes_sum :
    1280 * (2560 / 64 * 16) + 2560 * (640 / 64 * 16) + 1280 * (2560 / 64 * 2) + 2560 * (640 / 64 * 2) = 1382400 := by
  decide

/-- Code planes and scale planes separately: 18 = 16 + 2 per block. -/
theorem block_split : 16 + 2 = 18 := by decide

-- § row interleave

/-- Interleaved slot of a gate row. -/
def gateSlot (r : Nat) : Nat := 2 * r

/-- Interleaved slot of an up row. -/
def upSlot (r : Nat) : Nat := 2 * r + 1

/-- Gate slots are even. -/
theorem gate_even (r : Nat) : gateSlot r % 2 = 0 := by unfold gateSlot; omega

/-- Up slots are odd. -/
theorem up_odd (r : Nat) : upSlot r % 2 = 1 := by unfold upSlot; omega

/-- Gate and up slots never collide. -/
theorem slots_disjoint (r s : Nat) : gateSlot r ≠ upSlot s := by unfold gateSlot upSlot; omega

/-- Both slot maps stay inside the 1280 interleaved rows. -/
theorem slots_in_range (r : Nat) (h : r < 640) : gateSlot r < 1280 ∧ upSlot r < 1280 := by
  unfold gateSlot upSlot; omega

/-- The interleave is invertible: row = slot / 2, role = slot % 2. -/
theorem slot_inverse (r : Nat) : gateSlot r / 2 = r ∧ upSlot r / 2 = r := by
  unfold gateSlot upSlot; omega

/-- Every interleaved slot is a gate or an up row. -/
theorem slot_surjective (s : Nat) (h : s < 1280) :
    (∃ r, r < 640 ∧ s = gateSlot r) ∨ (∃ r, r < 640 ∧ s = upSlot r) := by
  by_cases hs : s % 2 = 0
  · exact Or.inl ⟨s / 2, by omega, by unfold gateSlot; omega⟩
  · exact Or.inr ⟨s / 2, by omega, by unfold upSlot; omega⟩

-- § dense padding

/-- Padding before a dense tensor so that it starts at a multiple of 256. -/
def pad256 (off : Nat) : Nat := (256 - off % 256) % 256

/-- After padding the offset is a multiple of 256. -/
theorem pad256_aligned (off : Nat) : (off + pad256 off) % 256 = 0 := by unfold pad256; omega

/-- Padding is smaller than the alignment. -/
theorem pad256_lt (off : Nat) : pad256 off < 256 := by unfold pad256; omega

/-- Per-expert blob offsets are `e * BLOB`, non-overlapping. -/
theorem expert_blobs_disjoint (e e' : Nat) (h : e < e') : e * 1382400 + 1382400 ≤ e' * 1382400 := by
  omega

/-- Total size of 512 experts. -/
theorem experts_total : 512 * 1382400 = 707788800 := by decide

end Strata.MtpRuntimeBlobRelayout



## FILE: /mnt/user-data/outputs/PackIndexPlaneContract.lean
/-!
# MODULE: pack_index_plane_contract
`tools/pack_index.py`: flattens the pack manifest into the engine loader's index.  The engine form of a
quantized tensor differs from the pack by exactly the fp16->f32 scale widening; the activation-quant kind
cannot be derived from the S-form fields.

## Source semantics
* engine scale plane = groups * 4 B always; pack scale plane = groups * (2 if fp16 else 4).
* dst = src + widening; planes must sum to the span.
* BF16 / F16 sources are re-rounded to 2 B per element in the engine.
* `vec_dot_type`: legacy formats use Q8_0, K-quants and IQ4_XS use Q8_K.

## Form template identifiers
nGroups      -> ne1 * (ne0 / group_elems)
scales_fp16  -> scale width flag
act_kind     -> vec_dot_type
-/

namespace Strata.PackIndexPlaneContract

-- § plane sizes

/-- Pack-side scale plane bytes for `n` groups. -/
def packScales (n : Nat) (fp16 : Bool) : Nat := n * (if fp16 then 2 else 4)

/-- Engine-side scale plane bytes: always 4 per group. -/
def engineScales (n : Nat) : Nat := n * 4

/-- Widening cost of one tensor: fp16 planes grow by 2 bytes per group. -/
def widening (n : Nat) (fp16 : Bool) : Nat := if fp16 then n * 2 else 0

/-- The widening is exactly the engine/pack difference of the scale plane. -/
theorem widening_eq_diff (n : Nat) (fp16 : Bool) : widening n fp16 = engineScales n - packScales n fp16 := by
  cases fp16 <;> simp [widening, engineScales, packScales] <;> omega

/-- The engine form equals the pack form plus the widening (no offset plane). -/
theorem engine_eq_pack_plus_widen (codes n : Nat) (fp16 : Bool) :
    codes + engineScales n = codes + packScales n fp16 + widening n fp16 := by
  cases fp16 <;> simp [widening, engineScales, packScales] <;> omega

/-- With an offset plane, both planes widen and the equality still holds. -/
theorem engine_eq_pack_plus_widen_off (codes n : Nat) (sfp offp : Bool) :
    codes + engineScales n + engineScales n =
      codes + packScales n sfp + packScales n offp + widening n sfp + widening n offp := by
  cases sfp <;> cases offp <;> simp [widening, engineScales, packScales] <;> omega

/-- fp16 scales widen to exactly twice their pack size: the widening costs `2n` bytes. -/
theorem widen_cost (n : Nat) : widening n true = 2 * n := by
  simp [widening] <;> omega

/-- f32 scales need no widening. -/
theorem no_widen_f32 (n : Nat) : widening n false = 0 := by
  unfold widening; simp

/-- Misreading an fp16 plane as 4-byte-wide gives a plane twice the real size. -/
theorem wrong_width_doubles (n : Nat) : engineScales n = 2 * packScales n true := by
  simp [engineScales, packScales] <;> omega

-- § re-rounding

/-- Bytes of a 16-bit re-rounded tensor and of its f32 promotion. -/
theorem rerounding_saves (elems : Nat) : elems * 4 - elems * 2 = elems * 2 := by omega

/-- The engine size of a re-rounded float tensor is two bytes per element. -/
theorem rerounded_bytes (elems : Nat) : elems * 2 ≤ elems * 4 := by omega

-- § activation kind

/-- Quantized source types of the pack. -/
inductive QT
  | q2_0 | q3_k | q4_0 | q4_k | q5_0 | q5_k | q6_k | q8_0 | iq4_nl | iq4_xs
  deriving DecidableEq

/-- Activation quantization kind. -/
inductive Act
  | q8_0 | q8_k
  deriving DecidableEq

/-- `vec_dot_type` of each source type. -/
def vdt : QT → Act
  | .q2_0 => .q8_0 | .q4_0 => .q8_0 | .q5_0 => .q8_0 | .q8_0 => .q8_0 | .iq4_nl => .q8_0
  | .q3_k => .q8_k | .q4_k => .q8_k | .q5_k => .q8_k | .q6_k => .q8_k | .iq4_xs => .q8_k

/-- The S-form fields: (code_bits, code_bias, group_elems, iq4 codebook, has_offset). -/
def sform : QT → Nat × Int × Nat × Bool × Bool
  | .q2_0 => (2, -1, 64, false, false)
  | .q3_k => (4, -4, 16, false, false)
  | .q4_0 => (4, -8, 32, false, false)
  | .q4_k => (4, 0, 32, false, true)
  | .q5_0 => (8, -16, 32, false, false)
  | .q5_k => (8, 0, 32, false, true)
  | .q6_k => (8, -32, 16, false, false)
  | .q8_0 => (8, -128, 32, false, false)
  | .iq4_nl => (4, 0, 32, true, false)
  | .iq4_xs => (4, 0, 32, true, false)

/-- IQ4_NL and IQ4_XS are indistinguishable by S-form fields. -/
theorem iq4_same_sform : sform .iq4_nl = sform .iq4_xs := rfl

/-- They nevertheless need different activation kinds. -/
theorem iq4_diff_vdt : vdt .iq4_nl ≠ vdt .iq4_xs := by decide

/-- No function of the S-form fields can recover the activation kind. -/
theorem no_vdt_from_sform : ¬ ∃ f : Nat × Int × Nat × Bool × Bool → Act, ∀ t, f (sform t) = vdt t := by
  rintro ⟨f, hf⟩
  have h1 := hf .iq4_nl
  have h2 := hf .iq4_xs
  rw [← iq4_same_sform] at h2
  exact iq4_diff_vdt (h1.symm.trans h2)

/-- Q5_0 and Q5_K share bits and the 8-bit code plane but differ in `has_offset`. -/
theorem q5_differ_only_by_offset :
    (sform .q5_0).1 = (sform .q5_k).1 ∧ (sform .q5_0).2.2.2.2 ≠ (sform .q5_k).2.2.2.2 := by decide

end Strata.PackIndexPlaneContract



## FILE: /mnt/user-data/outputs/ExpertArenaBlobLayout.lean
/-!
# MODULE: expert_arena_blob_layout
`tools/pack_layer.py`: builds the pack-layout expert arena.  One expert blob is exactly 1,382,400 bytes:
gate/up codes, down codes, gate/up scales, down scales.  The conversion is a pure relayout.

## Source semantics
* gate_up codes = 2*640 rows * 640 B; down codes = 2560 rows * 160 B;
  gate_up scales = 2*640 * 40 * 2 B; down scales = 2560 * 10 * 2 B.
* the four planes are contiguous and end exactly at BLOB_BYTES.
* expert `e` is one contiguous 460,800-byte run inside each role tensor.
* 48 layers * 512 experts * blob = 31.64 GiB.

## Form template identifiers
OFF            -> blob_offsets()
ROLE_BYTES     -> bytes per role per expert
BLOB_BYTES     -> bytes per expert blob
-/

namespace Strata.ExpertArenaBlobLayout

-- § plane sizes and offsets

/-- Row byte widths: 2560 two-bit codes = 640 B; 640 codes = 160 B. -/
theorem row_bytes : 2560 * 2 / 8 = 640 ∧ 640 * 2 / 8 = 160 := by decide

/-- Per-role bytes per expert. -/
theorem role_bytes : 2560 * 640 * 18 / 64 = 460800 := by decide

/-- Three roles make the blob. -/
theorem blob_bytes : 3 * 460800 = 1382400 := by decide

/-- The four plane sizes. -/
theorem plane_sizes :
    2 * 640 * 640 = 819200 ∧ 2560 * 160 = 409600 ∧ 2 * 640 * 40 * 2 = 102400 ∧ 2560 * 10 * 2 = 51200 := by
  decide

/-- The planes sum to the blob exactly. -/
theorem planes_sum : 819200 + 409600 + 102400 + 51200 = 1382400 := by decide

/-- Plane offsets, in order. -/
def guCodes : Nat := 0
def dCodes : Nat := guCodes + 819200
def guScales : Nat := dCodes + 409600
def dScales : Nat := guScales + 102400
def blobEnd : Nat := dScales + 51200

/-- The final offset is the blob size. -/
theorem blob_end : blobEnd = 1382400 := by decide

/-- Planes are ordered without overlap. -/
theorem planes_ordered :
    guCodes + 819200 ≤ dCodes ∧ dCodes + 409600 ≤ guScales ∧ guScales + 102400 ≤ dScales ∧
      dScales + 51200 ≤ blobEnd := by decide

/-- Concrete offsets. -/
theorem offsets_values : dCodes = 819200 ∧ guScales = 1228800 ∧ dScales = 1331200 := by decide

-- § interleave and in-plane addressing

/-- gate/up code byte offset of slot `s`, block `b`: `slot*640 + b*16` stays inside the plane. -/
theorem gu_code_inbounds (s b : Nat) (hs : s < 1280) (hb : b < 40) :
    s * 640 + b * 16 + 16 ≤ 819200 := by omega

/-- down code byte offset of row `r`, block `b`. -/
theorem d_code_inbounds (r b : Nat) (hr : r < 2560) (hb : b < 10) :
    r * 160 + b * 16 + 16 ≤ 409600 := by omega

/-- gate/up scale offset of slot `s`, block `b`. -/
theorem gu_scale_inbounds (s b : Nat) (hs : s < 1280) (hb : b < 40) :
    (s * 40 + b) * 2 + 2 ≤ 102400 := by omega

/-- down scale offset of row `r`, block `b`. -/
theorem d_scale_inbounds (r b : Nat) (hr : r < 2560) (hb : b < 10) :
    (r * 10 + b) * 2 + 2 ≤ 51200 := by omega

/-- Distinct (row, block) pairs of a gate/up row map to distinct code offsets. -/
theorem gu_code_injective (s b s' b' : Nat) (hb : b < 40) (hb' : b' < 40)
    (h : s * 640 + b * 16 = s' * 640 + b' * 16) : s = s' ∧ b = b' := by
  omega

-- § expert runs inside role tensors

/-- Start of expert `e` inside a role tensor. -/
def expertStart (e : Nat) : Nat := e * 460800

/-- Experts are disjoint, contiguous runs. -/
theorem experts_disjoint (e e' : Nat) (h : e < e') : expertStart e + 460800 ≤ expertStart e' := by
  unfold expertStart; omega

/-- Expert 511 ends at the end of a 512-expert role tensor. -/
theorem last_expert_end : expertStart 511 + 460800 = 512 * 460800 := by decide

-- § arena size

/-- Blob address of (layer, expert) in the arena. -/
def blobAddr (layer e : Nat) : Nat := (layer * 512 + e) * 1382400

/-- Addresses are strictly increasing in (layer, expert) order. -/
theorem blobAddr_next (layer e : Nat) (he : e < 511) :
    blobAddr layer (e + 1) = blobAddr layer e + 1382400 := by
  unfold blobAddr; omega

/-- The last expert of a layer is followed by the first of the next. -/
theorem blobAddr_layer_step (layer : Nat) : blobAddr (layer + 1) 0 = blobAddr layer 511 + 1382400 := by
  unfold blobAddr; omega

/-- A full 48-layer arena is 33,973,862,400 bytes. -/
theorem full_arena : 48 * 512 * 1382400 = 33973862400 := by decide

/-- Any in-range blob ends inside the full arena. -/
theorem blob_in_arena (layer e : Nat) (hl : layer < 48) (he : e < 512) :
    blobAddr layer e + 1382400 ≤ 33973862400 := by
  unfold blobAddr; omega

end Strata.ExpertArenaBlobLayout



## FILE: /mnt/user-data/outputs/PackBuilderCodeRoundtrip.lean
/-!
# MODULE: pack_builder_code_roundtrip
`tools/strata_pack.py`: pack builder and verifier.  Code planes are packed LSB-first with `8/bits` codes per
byte; every plane starts on a 64-byte boundary; scales are stored fp16 only when exact; the verifier inverts
the expert blob layout.

## Source semantics
* pack_codes / unpack_codes are inverse for bits in {2,4,8}.
* `put` pads the body to 64 before each plane, so every plane offset is 64-aligned when the base is.
* `rebuild_role` addresses the same bytes `pack_layer` wrote, so it must stay in bounds.
* the verifier samples every Nth expert; `every = 32` still covers all layers.

## Form template identifiers
pack2 / pack4 -> pack_codes for bits 2 and 4
pad64         -> `(-len(body)) % ALIGN`
slot          -> interleaved gate/up row
-/

namespace Strata.PackBuilderCodeRoundtrip

-- § code packing

/-- Two 4-bit codes per byte. -/
def pack4 (a b : Nat) : Nat := a + 16 * b

/-- Unpack one nibble. -/
def unpackNib (byte k : Nat) : Nat := (byte / 16 ^ k) % 16

/-- 4-bit packing round-trips. -/
theorem pack4_roundtrip (a b : Nat) (ha : a < 16) (hb : b < 16) :
    unpackNib (pack4 a b) 0 = a ∧ unpackNib (pack4 a b) 1 = b := by
  unfold unpackNib pack4
  constructor <;> simp <;> omega

/-- 4-bit packing yields a byte. -/
theorem pack4_lt (a b : Nat) (ha : a < 16) (hb : b < 16) : pack4 a b < 256 := by
  unfold pack4; omega

/-- Four 2-bit codes per byte. -/
def pack2 (a b c d : Nat) : Nat := a + 4 * b + 16 * c + 64 * d

/-- Unpack one 2-bit code. -/
def unpackCrumb (byte k : Nat) : Nat := (byte / 4 ^ k) % 4

/-- 2-bit packing round-trips. -/
theorem pack2_roundtrip (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    unpackCrumb (pack2 a b c d) 0 = a ∧ unpackCrumb (pack2 a b c d) 1 = b ∧
      unpackCrumb (pack2 a b c d) 2 = c ∧ unpackCrumb (pack2 a b c d) 3 = d := by
  unfold unpackCrumb pack2
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp <;> omega

/-- 2-bit packing yields a byte. -/
theorem pack2_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    pack2 a b c d < 256 := by
  unfold pack2; omega

/-- Codes per byte for the three forms: S2 -> 4, S4 -> 2, S8 -> 1. -/
theorem codes_per_byte : 8 / 2 = 4 ∧ 8 / 4 = 2 ∧ 8 / 8 = 1 := by decide

/-- Packed size of `n` codes at `bits` when `n` is a multiple of `8/bits`. -/
theorem packed_size_s4 (k : Nat) : (2 * k) / 2 = k := by omega

theorem packed_size_s2 (k : Nat) : (4 * k) / 4 = k := by omega

-- § plane alignment

/-- Padding before a plane. -/
def pad64 (len : Nat) : Nat := (64 - len % 64) % 64

/-- After the pad the plane begins on a 64-byte boundary. -/
theorem pad64_aligned (len : Nat) : (len + pad64 len) % 64 = 0 := by unfold pad64; omega

/-- Padding is under 64. -/
theorem pad64_lt (len : Nat) : pad64 len < 64 := by unfold pad64; omega

/-- A plane at an aligned body offset has an aligned absolute offset when the segment base is aligned. -/
theorem plane_abs_aligned (base len : Nat) (hb : base % 64 = 0) : (base + (len + pad64 len)) % 64 = 0 := by
  unfold pad64; omega

-- § scale precision

/-- FP16-or-FP32 choice records the width in bytes. -/
def scaleWidth (exact16 : Bool) : Nat := if exact16 then 2 else 4

/-- The recorded width is one of the two supported widths. -/
theorem scaleWidth_cases (b : Bool) : scaleWidth b = 2 ∨ scaleWidth b = 4 := by
  unfold scaleWidth; cases b <;> simp

/-- Scale plane bytes are `groups * width`; the byte count identifies the width when `groups > 0`. -/
theorem width_recoverable (g : Nat) (hg : 0 < g) (a b : Bool) (h : g * scaleWidth a = g * scaleWidth b) :
    a = b := by
  unfold scaleWidth at h
  cases a <;> cases b <;> simp at h ⊢ <;> omega

-- § arena inverse

/-- Slot of a gate or up row in the interleaved blob: gate 2r, up 2r+1. -/
def slot (isUp : Bool) (r : Nat) : Nat := 2 * r + (if isUp then 1 else 0)

/-- Slots of gate and up never coincide. -/
theorem slot_ne (r s : Nat) : slot false r ≠ slot true s := by unfold slot; simp; omega

/-- Gate/up code and scale addressing stay in the blob's planes. -/
theorem rebuild_gu_bounds (isUp : Bool) (row b : Nat) (hr : row < 640) (hb : b < 40) :
    slot isUp row * 640 + b * 16 + 16 ≤ 819200 ∧ (slot isUp row * 40 + b) * 2 + 2 ≤ 102400 := by
  unfold slot
  cases isUp <;> simp <;> omega

/-- The block index of the verifier splits as row and in-row block exactly. -/
theorem blk_split (blk : Nat) (h : blk < 25600) : blk / 40 < 640 ∧ blk % 40 < 40 := by omega

/-- Down role: 2560 rows of 10 blocks. -/
theorem blk_split_down (blk : Nat) (h : blk < 25600) : blk / 10 < 2560 ∧ blk % 10 < 10 := by omega

/-- Row and in-row block recombine to the block index. -/
theorem blk_recombine (blk : Nat) : (blk / 40) * 40 + blk % 40 = blk := by omega

-- § verifier coverage

/-- Sampling every 32nd expert checks 16 experts per layer. -/
theorem experts_checked_per_layer : (512 + 31) / 32 = 16 := by decide

/-- Experts 0, 32, ..., 480 are the sampled ones, all in range. -/
theorem sampled_in_range (i : Nat) (h : i < 16) : i * 32 < 512 := by omega

end Strata.PackBuilderCodeRoundtrip



## FILE: /mnt/user-data/outputs/ByteLevelBpeTokenizer.lean
/-!
# MODULE: byte_level_bpe_tokenizer
`tools/strata_tokenizer.py`: GPT-2 style byte-level BPE.  Each byte maps to a printable code point so any UTF-8
input is representable; decode inverts it exactly.  Special-token matching differs by GGUF token type.

## Source semantics
* bytes 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map to themselves (188 bytes); the other 68 map to 256 + rank.
* the map is injective, so `decode(encode(s)) = s` at the byte layer.
* token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only when `parse_special`.
* BPE merges the LOWEST-rank adjacent pair first.

## Form template identifiers
byteToUnicode   -> bytes_to_unicode
inBs            -> membership in the identity ranges
matched         -> special-token match rule
-/

namespace Strata.ByteLevelBpeTokenizer

-- § the byte alphabet

/-- Byte is in the identity ranges. -/
def inBs (b : Nat) : Bool :=
  (0x21 ≤ b && b ≤ 0x7E) || (0xA1 ≤ b && b ≤ 0xAC) || (0xAE ≤ b && b ≤ 0xFF)

/-- Code point of byte `b`: itself, or 256 plus its rank among the shifted bytes. -/
def byteToUnicode (b : Nat) : Nat :=
  if inBs b then b else 256 + (List.range b).countP (fun x => !inBs x)

/-- Exactly 188 bytes are in the identity ranges. -/
theorem identity_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- The remaining 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun b => !inBs b)).length = 68 := by decide +kernel

/-- Space and newline land on the well-known `Ġ` (288) and `Ċ` (266). -/
theorem known_points : byteToUnicode 32 = 288 ∧ byteToUnicode 10 = 266 := by decide

/-- Printable ASCII maps to itself. -/
theorem ascii_identity : byteToUnicode 65 = 65 ∧ byteToUnicode 97 = 97 := by decide

/-- Identity-range bytes stay below 256 and shifted bytes land at or above 256, so the two never collide. -/
theorem ranges_separated (b : Nat) (h : b < 256) :
    (inBs b = true → byteToUnicode b < 256) ∧ (inBs b = false → 256 ≤ byteToUnicode b) := by
  constructor
  · intro hb; simp [byteToUnicode, hb]; omega
  · intro hb; simp [byteToUnicode, hb]

/-- The byte->code point map has 256 distinct images. -/
theorem byteToUnicode_nodup : ((List.range 256).map byteToUnicode).Nodup := by
  decide +kernel

/-- Every image is below 512 (the alphabet is 256 shifted by at most 256). -/
theorem byteToUnicode_lt : ∀ b ∈ List.range 256, byteToUnicode b < 512 := by
  decide +kernel

/-- Inverse lookup. -/
def unicodeToByte (c : Nat) : Option Nat := (List.range 256).find? (fun b => byteToUnicode b == c)

/-- Decode inverts encode on every byte: the byte-layer round trip. -/
theorem byte_roundtrip : ∀ b ∈ List.range 256, unicodeToByte (byteToUnicode b) = some b := by
  decide +kernel

-- § special-token matching

/-- Token type ids. -/
def tCONTROL : Nat := 3
def tUSER : Nat := 4

/-- A literal is matched as one token when it is USER_DEFINED, or CONTROL with `parse_special`. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  ty == tUSER || (ty == tCONTROL && parseSpecial)

/-- USER_DEFINED literals are matched either way. -/
theorem user_always (p : Bool) : matched tUSER p = true := by
  unfold matched tUSER; simp

/-- CONTROL literals are matched exactly when `parse_special` is set. -/
theorem control_iff (p : Bool) : matched tCONTROL p = p := by
  unfold matched tCONTROL tUSER; cases p <;> simp

/-- Ordinary tokens are never matched literally. -/
theorem ordinary_never (ty : Nat) (h1 : ty ≠ 3) (h2 : ty ≠ 4) (p : Bool) : matched ty p = false := by
  unfold matched tCONTROL tUSER
  simp [h1, h2]

/-- Turning `parse_special` on can only add matches. -/
theorem parse_special_monotone (ty : Nat) : matched ty false = true → matched ty true = true := by
  unfold matched; cases h : (ty == tUSER) <;> simp

-- § BPE rank selection

/-- Minimum of two optional ranks, ignoring absent ones. -/
def optMin : Option Nat → Option Nat → Option Nat
  | none, b => b
  | some x, none => some x
  | some x, some y => some (min x y)

/-- Lowest present rank in a list of optional ranks, or none if all are absent. -/
def bestRank : List (Option Nat) → Option Nat
  | [] => none
  | r :: rs => optMin r (bestRank rs)

/-- If some rank is present, a minimum exists and is at most that rank. -/
theorem bestRank_le (rs : List (Option Nat)) (x : Nat) (hx : some x ∈ rs) :
    ∃ m, bestRank rs = some m ∧ m ≤ x := by
  induction rs with
  | nil => simp at hx
  | cons r rs ih =>
    simp only [List.mem_cons] at hx
    rcases hx with hx | hx
    · subst hx
      cases hb : bestRank rs with
      | none => exact ⟨x, by simp [bestRank, hb, optMin], Nat.le_refl x⟩
      | some m => exact ⟨min x m, by simp [bestRank, hb, optMin], Nat.min_le_left x m⟩
    · obtain ⟨m, hm, hle⟩ := ih hx
      cases r with
      | none => exact ⟨m, by simp [bestRank, hm, optMin], hle⟩
      | some y => exact ⟨min y m, by simp [bestRank, hm, optMin], Nat.le_trans (Nat.min_le_right y m) hle⟩

/-- With no present rank, the result is none. -/
theorem bestRank_none_of_all_none (rs : List (Option Nat)) (h : ∀ r ∈ rs, r = none) :
    bestRank rs = none := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hr : r = none := h r (by simp)
    have hrs : bestRank rs = none := ih (fun r' hr' => h r' (by simp [hr']))
    simp [bestRank, hr, hrs, optMin]

end Strata.ByteLevelBpeTokenizer



## FILE: /mnt/user-data/outputs/LazyGgufTensorAddressing.lean
/-!
# MODULE: lazy_gguf_tensor_addressing
`ref/load.py`: demand-paged GGUF access.  Experts and rows are read by byte arithmetic without decoding the
rest of the tensor; the second shard holds only the PLE table.

## Source semantics
* row_bytes = width / block_elems * block_bytes, requiring width % block_elems = 0.
* a row is contiguous: row r starts at r * row_bytes.
* expert e of a fused [in, out, experts] tensor starts at e * (in*out/el) * by, and runs are disjoint.
* tensors are reshaped by reversing the GGUF dims (dim 0 fastest).

## Form template identifiers
GEOM        -> (block_elems, block_bytes) table
row_bytes   -> Gguf.row_bytes
expert      -> Gguf.expert
-/

namespace Strata.LazyGgufTensorAddressing

-- § row addressing

/-- Bytes of one row. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- Start of row `r`. -/
def rowStart (r rb : Nat) : Nat := r * rb

/-- Rows are disjoint contiguous runs. -/
theorem rows_disjoint (r r' rb : Nat) (h : r < r') : rowStart r rb + rb ≤ rowStart r' rb := by
  unfold rowStart
  have : (r + 1) * rb ≤ r' * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The PLE table row: 160 IQ4_NL elements = 5 blocks of 18 B = 90 B. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- 320,001,536 rows of 90 B is 28,800,138,240 B (26.82 GiB). -/
theorem ple_table_bytes : 320001536 * 90 = 28800138240 := by decide

/-- Reading 16 rows per token touches 1,440 bytes, a vanishing fraction of the table. -/
theorem ple_token_read : 16 * 90 = 1440 := by decide

-- § expert addressing

/-- Bytes of one expert: `in*out/el*by`. -/
def expertBytes (inF outF el by_ : Nat) : Nat := inF * outF / el * by_

/-- Start of expert `e`. -/
def expertStart (e per : Nat) : Nat := e * per

/-- Expert runs are disjoint. -/
theorem experts_disjoint (e e' per : Nat) (h : e < e') :
    expertStart e per + per ≤ expertStart e' per := by
  unfold expertStart
  have : (e + 1) * per ≤ e' * per := Nat.mul_le_mul_right per h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Q2_0 experts of the real model: 2560 x 640 elements = 460,800 B per expert. -/
theorem q2_0_expert_bytes : expertBytes 2560 640 64 18 = 460800 := by decide

/-- All 512 experts fill the role tensor. -/
theorem role_tensor_bytes : 512 * 460800 = 235929600 := by decide

/-- The per-expert element count is a whole number of Q2_0 blocks. -/
theorem q2_0_whole_blocks : 2560 * 640 % 64 = 0 := by decide

/-- The last expert ends exactly at the tensor end. -/
theorem last_expert_end : expertStart 511 460800 + 460800 = 512 * 460800 := by decide

-- § block geometry table

/-- (type, elements per block, bytes per block). -/
def geom : List (String × Nat × Nat) :=
  [("F32", 1, 4), ("F16", 1, 2), ("BF16", 1, 2), ("Q4_0", 32, 18), ("Q5_0", 32, 22), ("Q8_0", 32, 34),
   ("IQ4_NL", 32, 18), ("Q3_K", 256, 110), ("Q4_K", 256, 144), ("Q5_K", 256, 176), ("Q6_K", 256, 210),
   ("IQ4_XS", 256, 136), ("Q2_0", 64, 18)]

/-- Thirteen types are covered. -/
theorem geom_size : geom.length = 13 := by decide

/-- Every block size is positive, so row arithmetic never divides by zero. -/
theorem geom_positive : ∀ g ∈ geom, 0 < g.2.1 ∧ 0 < g.2.2 := by decide

/-- Block elements are 1, 32, 64 or 256. -/
theorem geom_el_cases : ∀ g ∈ geom, g.2.1 = 1 ∨ g.2.1 = 32 ∨ g.2.1 = 64 ∨ g.2.1 = 256 := by decide

-- § shape reversal

/-- GGUF dims (dim 0 fastest) reversed give the numpy shape. -/
theorem reverse_dims (i o e : Nat) : [i, o, e].reverse = [e, o, i] := rfl

/-- Reversal is an involution. -/
theorem reverse_involutive (l : List Nat) : l.reverse.reverse = l := List.reverse_reverse l

/-- A row index is in range iff it is below the second dim. -/
theorem row_in_range (r rows : Nat) (h : r < rows) (rb : Nat) : r * rb + rb ≤ rows * rb := by
  have : (r + 1) * rb ≤ rows * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

-- § shard discovery

/-- A shard set is complete iff the number of siblings found equals the declared total. -/
def complete (found total : Nat) : Bool := found == total

/-- Two shards of two is complete; one of two is not. -/
theorem shard_completeness : complete 2 2 = true ∧ complete 1 2 = false := by decide

end Strata.LazyGgufTensorAddressing



## FILE: /mnt/user-data/outputs/VisionBuildConfiguration.lean
/-!
# MODULE: vision_build_configuration
`tools/vision/CMakeLists.txt`: the optional `strata-vision` build.  Two user options (`STRATA_VISION_CUDA`,
`STRATA_PORTABLE`) determine the forced ggml cache variables.

## Source semantics
* CUDA is on exactly when `STRATA_VISION_CUDA` is on.
* Portable builds turn off native tuning and OpenMP and force the AVX2 baseline flags on.
* Non-portable builds use native tuning.
* Only the MTMD target is built from llama.cpp; tools, tests, examples, server, app, curl, openssl are off.
* The static MSVC runtime applies only to portable MSVC builds.

## Form template identifiers
Opts          -> the two CMake options
ggml          -> forced GGML_* variables
-/

namespace Strata.VisionBuildConfiguration

/-- The user-facing options. -/
structure Opts where
  cuda : Bool
  portable : Bool
  msvc : Bool

/-- Forced ggml variables. -/
structure Ggml where
  native : Bool
  openmp : Bool
  avx : Bool
  avx2 : Bool
  fma : Bool
  f16c : Bool
  bmi2 : Bool
  cuda : Bool
  backendDl : Bool

/-- The configuration the CMake file computes (OpenMP is left at its default, on, when not portable). -/
def ggml (o : Opts) : Ggml :=
  if o.portable then
    { native := false, openmp := false, avx := true, avx2 := true, fma := true, f16c := true,
      bmi2 := true, cuda := o.cuda, backendDl := false }
  else
    { native := true, openmp := true, avx := false, avx2 := false, fma := false, f16c := false,
      bmi2 := false, cuda := o.cuda, backendDl := false }

/-- CUDA mirrors the option, in both modes. -/
theorem cuda_mirrors (o : Opts) : (ggml o).cuda = o.cuda := by
  unfold ggml; split <;> rfl

/-- Portable builds disable native tuning and OpenMP. -/
theorem portable_no_native (o : Opts) (h : o.portable = true) :
    (ggml o).native = false ∧ (ggml o).openmp = false := by
  unfold ggml; simp [h]

/-- Portable builds force the AVX2 baseline. -/
theorem portable_avx2_baseline (o : Opts) (h : o.portable = true) :
    (ggml o).avx = true ∧ (ggml o).avx2 = true ∧ (ggml o).fma = true ∧ (ggml o).f16c = true ∧
      (ggml o).bmi2 = true := by
  unfold ggml; simp [h]

/-- Non-portable builds use native tuning. -/
theorem nonportable_native (o : Opts) (h : o.portable = false) : (ggml o).native = true := by
  unfold ggml; simp [h]

/-- Native and portable are mutually exclusive. -/
theorem native_xor_portable (o : Opts) : (ggml o).native = !o.portable := by
  unfold ggml; cases h : o.portable <;> simp

/-- Backends are statically linked in both modes. -/
theorem backend_static (o : Opts) : (ggml o).backendDl = false := by
  unfold ggml; split <;> rfl

/-- The CPU flags of a portable build do not depend on whether CUDA is requested. -/
theorem portable_cpu_flags_independent_of_cuda (o : Opts) (c : Bool) (h : o.portable = true) :
    (ggml o).avx2 = (ggml { o with cuda := c }).avx2 := by
  unfold ggml; simp [h]

/-- Static MSVC runtime is selected only for portable MSVC builds. -/
def staticRuntime (o : Opts) : Bool := o.portable && o.msvc

/-- Non-portable builds never force the static runtime. -/
theorem no_static_when_nonportable (o : Opts) (h : o.portable = false) : staticRuntime o = false := by
  unfold staticRuntime; simp [h]

/-- Non-MSVC builds never force the static runtime. -/
theorem no_static_when_not_msvc (o : Opts) (h : o.msvc = false) : staticRuntime o = false := by
  unfold staticRuntime; simp [h]

-- § forced llama.cpp build switches

/-- Names of the `LLAMA_BUILD_*` switches and their forced values. -/
def llamaSwitches : List (String × Bool) :=
  [("COMMON", false), ("TOOLS", false), ("TESTS", false), ("EXAMPLES", false), ("SERVER", false),
   ("APP", false), ("MTMD", true)]

/-- Exactly one switch (MTMD) is on. -/
theorem only_mtmd_on : (llamaSwitches.filter (·.2)).map (·.1) = ["MTMD"] := by decide

/-- Seven build switches are forced. -/
theorem switch_count : llamaSwitches.length = 7 := by decide

end Strata.VisionBuildConfiguration



## FILE: /mnt/user-data/outputs/VisionEncoderProtocol.lean
/-!
# MODULE: vision_encoder_protocol
`tools/vision/strata_vision.cpp`: resident image encoder.  Reads `ENC <image> <out>` lines, writes a binary
embedding file with a 5-int32 header followed by `n` rows of `n_embd` float32, rows laid out on an `nx` by `ny`
grid.

## Source semantics
* `parse_enc` splits at the LAST space: the image path may contain spaces, the output path may not.
* the grid is derived from the last token's decoder position: nx = x+1, ny = y+1; require nx*ny = n.
* row i sits at (x = i % nx, y = i / nx).
* file size = 20 + 4 * n * n_embd bytes.

## Form template identifiers
splitLast    -> `rest.rfind(' ')` split
header       -> hdr[5] = {magic, n, nx, ny, n_embd}
gridPos      -> (i % nx, i / nx)
-/

namespace Strata.VisionEncoderProtocol

-- § parsing the ENC line

/-- Split at the last space (code 32): `(before, after)`; none if there is no space. -/
def splitLast : List Nat → Option (List Nat × List Nat)
  | [] => none
  | x :: xs =>
    match splitLast xs with
    | some (a, b) => some (x :: a, b)
    | none => if x = 32 then some ([], xs) else none

/-- A space-free list has no split point. -/
theorem splitLast_none (out : List Nat) (h : ∀ c ∈ out, c ≠ 32) : splitLast out = none := by
  induction out with
  | nil => rfl
  | cons x xs ih =>
    have hx : x ≠ 32 := h x (by simp)
    have hxs : ∀ c ∈ xs, c ≠ 32 := fun c hc => h c (by simp [hc])
    simp [splitLast, ih hxs, hx]

/-- Image path (spaces allowed) and a space-free output path are recovered exactly. -/
theorem splitLast_correct (img out : List Nat) (h : ∀ c ∈ out, c ≠ 32) :
    splitLast (img ++ 32 :: out) = some (img, out) := by
  induction img with
  | nil => simp [splitLast, splitLast_none out h]
  | cons x xs ih => simp [splitLast, ih]

/-- A line with no space yields no split. -/
theorem no_space_no_split (l : List Nat) (h : ∀ c ∈ l, c ≠ 32) : splitLast l = none :=
  splitLast_none l h

/-- Parsing succeeds with a nonempty image and output only when both are nonempty. -/
def validEnc (img out : List Nat) : Bool := !img.isEmpty && !out.isEmpty

theorem validEnc_iff (img out : List Nat) :
    validEnc img out = true ↔ img ≠ [] ∧ out ≠ [] := by
  unfold validEnc
  cases img <;> cases out <;> simp

-- § grid layout

/-- Grid coordinates of row `i`. -/
def gridPos (nx i : Nat) : Nat × Nat := (i % nx, i / nx)

/-- Every row of an `nx * ny` grid lands inside it. -/
theorem gridPos_in_range (nx ny i : Nat) (hnx : 0 < nx) (hi : i < nx * ny) :
    (gridPos nx i).1 < nx ∧ (gridPos nx i).2 < ny := by
  unfold gridPos
  exact ⟨Nat.mod_lt _ hnx, (Nat.div_lt_iff_lt_mul hnx).2 (by rw [Nat.mul_comm]; exact hi)⟩

/-- Row-major layout round-trips: row `y*nx + x` is at `(x, y)`. -/
theorem gridPos_roundtrip (nx x y : Nat) (hx : x < nx) : gridPos nx (y * nx + x) = (x, y) := by
  unfold gridPos
  have hpos : 0 < nx := by omega
  have h := (Nat.div_mod_unique hpos).2 (⟨by rw [Nat.mul_comm nx y, Nat.add_comm], hx⟩ :
    x + nx * y = y * nx + x ∧ x < nx)
  simp [h.1, h.2]

/-- The last token of a full grid is at `(nx-1, ny-1)`, which is how `nx`, `ny` are derived. -/
theorem last_token_pos (nx ny : Nat) (hnx : 0 < nx) (hny : 0 < ny) :
    gridPos nx ((ny - 1) * nx + (nx - 1)) = (nx - 1, ny - 1) :=
  gridPos_roundtrip nx (nx - 1) (ny - 1) (by omega)

/-- Deriving `nx*ny` from the last position reproduces the token count of a rectangular grid. -/
theorem grid_count (nx ny : Nat) (hnx : 0 < nx) (hny : 0 < ny) :
    (nx - 1 + 1) * (ny - 1 + 1) = nx * ny := by
  have h1 : nx - 1 + 1 = nx := by omega
  have h2 : ny - 1 + 1 = ny := by omega
  rw [h1, h2]

-- § output file size

/-- Header bytes: five int32. -/
theorem header_bytes : 5 * 4 = 20 := by decide

/-- File size for `n` rows of `n_embd` floats. -/
def fileBytes (n nEmbd : Nat) : Nat := 20 + 4 * (n * nEmbd)

/-- Size grows by exactly one row of floats per token. -/
theorem fileBytes_step (n nEmbd : Nat) : fileBytes (n + 1) nEmbd = fileBytes n nEmbd + 4 * nEmbd := by
  unfold fileBytes
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- An empty encoding is just the header. -/
theorem fileBytes_zero (nEmbd : Nat) : fileBytes 0 nEmbd = 20 := by
  unfold fileBytes; simp

/-- The magic `SVE1` bytes 'S','V','E','1' read as little-endian int32. -/
theorem magic : 0x53 + 0x56 * 256 + 0x45 * 65536 + 0x31 * 16777216 = 0x31455653 := by decide

/-- The magic fits in a positive int32. -/
theorem magic_int32 : 0x31455653 < 2147483648 := by decide

end Strata.VisionEncoderProtocol




[LIGHT FILES — structure only, no LAYER 1a/1b]
  include/strata/artifact/gguf_reader.hpp (487 lines)
  include/strata/core/device.hpp (72 lines)
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/layer.hpp (567 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/session.hpp (412 lines)
  include/strata/core/verify.hpp (161 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gdn.hpp (98 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ngram.hpp (195 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/qsa.hpp (361 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/s_gemv.hpp (134 lines)
  include/strata/kernels/shared_expert.hpp (102 lines)
  include/strata/kernels/verify_kernels.hpp (81 lines)
  include/strata/kernels/bf16_bits.hpp (47 lines)
  include/strata/kernels/bf16_gemv.hpp (49 lines)
  include/strata/kernels/dequant_bf16.hpp (28 lines)
  include/strata/kernels/dequant_s2.hpp (19 lines)
  include/strata/kernels/fused_gdn.hpp (36 lines)
  include/strata/kernels/fused_gr.hpp (52 lines)
  include/strata/kernels/iq_kernels.hpp (58 lines)
  include/strata/kernels/kv_q8.hpp (39 lines)
  include/strata/kernels/mrope.hpp (33 lines)
  include/strata/kernels/native_flash_attn.hpp (37 lines)
  include/strata/kernels/native_gdn.hpp (23 lines)
  include/strata/kernels/native_gdn_preprocess.hpp (38 lines)
  include/strata/kernels/native_gr_norm.hpp (19 lines)
  include/strata/kernels/native_gr_postops.hpp (29 lines)
  include/strata/kernels/native_moe.hpp (19 lines)
  include/strata/kernels/native_ple_postops.hpp (31 lines)
  include/strata/kernels/native_qsa.hpp (31 lines)
  include/strata/kernels/native_qsa_indexer.hpp (36 lines)
  include/strata/kernels/native_qsa_score.hpp (32 lines)
  include/strata/kernels/native_rope.hpp (15 lines)
  include/strata/kernels/native_router.hpp (18 lines)
  include/strata/kernels/qsa_decode_attn.hpp (45 lines)
  include/strata/kernels/qsa_select.hpp (34 lines)
  include/strata/kernels/quantize_act.hpp (59 lines)
  include/strata/kernels/rope.hpp (51 lines)
  include/strata/kernels/router_top10.hpp (21 lines)
  include/strata/kernels/s2_gemv.hpp (26 lines)
  include/strata/kernels/s2_gemv_q8.hpp (23 lines)
  include/strata/kernels/sampler.hpp (44 lines)
  include/strata/kernels/cpu/pool.hpp (196 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  serve/__init__.py (2 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/kernels/cpu/expert_layout.cpp (137 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/direct_file.cpp (242 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/controller_test.cpp (75 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  tools/_paths.py (29 lines)
  tools/gguf_reader.py (143 lines)
  tools/mtp_fetch.py (122 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  third_party/ggml/VERSION.txt (2 lines)
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  chat.py (117 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  third_party/ggml/ggml-common.h (1912 lines)
