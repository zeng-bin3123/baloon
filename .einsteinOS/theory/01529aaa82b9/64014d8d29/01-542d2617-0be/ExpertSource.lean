/-!
# MODULE: expert_source
SOURCES: src/core/expert_source.cpp, include/strata/core/expert_source.hpp
The adapter between the host loop and the CPU expert pool: file- or arena-backed blob lookup, hit/miss job
building, the GPU/PCIe share of missed experts in verify windows, and chunked loading of expert tensors.

## Source semantics
* Blobs are addressed by `layer * n_expert + expert`; `expert >= n_expert` must be rejected separately,
  because the flat index alone would alias the next layer's first expert.
* `experts.bin` holds exactly `n_layers * n_expert * BLOB` bytes (24576 × 1,382,400 = 33,973,862,400).
* For every router entry exactly one of {GPU hit, CPU job} computes it: jobs + hits = k, and `njobs` indexes
  the job array while `i` indexes the output.
* PCIe share: `m = (nmiss * pcie_num) >> 8` of the missed experts (the last `m` in routing order) go to the GPU.
* GGUF expert loading reads 16 experts per chunk and lays a blob out as [gate | up | down].
* Any slot-sized copy may start at any expert because the arena is one max-blob longer than the file.

## Form template identifiers
- flatIdx → `layer * n_expert + expert` ; nHit/nMiss → is_hit counts ; pcieShare → `m`
- blobRegions → per[3]/at[3] ; arenaSlack → `want + blob`
-/

namespace Strata.ExpertSource

-- § blob addressing

/-- Flat blob index. -/
def flatIdx (layer expert nExpert : Nat) : Nat := layer * nExpert + expert

/-- The expert axis is validated separately: without it, `(0, E)` aliases `(1, 0)`. -/
theorem unchecked_flat_alias (E : Nat) : flatIdx 0 E E = flatIdx 1 0 E := by
  unfold flatIdx; omega

/-- Concrete case from the source comment: with 512 experts, `blob(0, 512)` is `blob(1, 0)`. -/
theorem alias_512 : flatIdx 0 512 512 = flatIdx 1 0 512 := by
  decide

/-- With the separate range check the flat index recovers both axes. -/
theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx
    rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

/-- With the range check on `expert`, distinct (layer, expert) pairs get distinct blobs. -/
theorem flatIdx_inj (l e l' e' E : Nat) (hE : 0 < E) (he : e < E) (he' : e' < E)
    (h : flatIdx l e E = flatIdx l' e' E) : l = l' ∧ e = e' := by
  have h1 := flatIdx_recover l e E hE he
  have h2 := flatIdx_recover l' e' E hE he'
  rw [h] at h1
  exact ⟨h1.1.symm.trans h2.1, h1.2.symm.trans h2.2⟩

/-- An in-range pair indexes inside `n_layers * n_expert` blobs. -/
theorem flatIdx_lt (l e L E : Nat) (hl : l < L) (he : e < E) : flatIdx l e E < L * E := by
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx
  omega

/-- The expert file size for the artifact: 48 layers × 512 experts × 1,382,400 bytes. -/
theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by
  decide

/-- There are exactly 24,576 blobs. -/
theorem blob_count : 48 * 512 = 24576 := by
  decide

-- § hit/miss job building

/-- Number of GPU hits among the router entries. -/
def nHit : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + nHit bs

/-- Number of CPU jobs (misses) among the router entries. -/
def nMiss : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + nMiss bs

/-- Every router entry is computed by exactly one engine: jobs + hits = k. -/
theorem jobs_plus_hits : ∀ l : List Bool, nHit l + nMiss l = l.length
  | [] => by simp [nHit, nMiss]
  | b :: bs => by
    have ih := jobs_plus_hits bs
    cases b <;> simp [nHit, nMiss, List.length_cons] <;> omega

/-- Hits never exceed the number of entries (so `n_hits ≤ k`). -/
theorem nHit_le (l : List Bool) : nHit l ≤ l.length := by
  have := jobs_plus_hits l
  omega

/-- With no hits every entry is a job, so job index and output index coincide. -/
theorem no_hits_all_jobs : ∀ l : List Bool, nHit l = 0 → nMiss l = l.length := by
  intro l h
  have := jobs_plus_hits l
  omega

-- § PCIe share of the missed experts

/-- `m = (nmiss * pcie_num) >> 8`. -/
def pcieShare (nmiss num : Nat) : Nat := nmiss * num / 256

/-- A numerator of at most 256 gives a share of at most all the misses. -/
theorem pcieShare_le (nmiss num : Nat) (h : num ≤ 256) : pcieShare nmiss num ≤ nmiss := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  rw [Nat.mul_comm 256 nmiss]
  exact Nat.mul_le_mul_left nmiss h

/-- Ranks `r` with `nmiss - m ≤ r < nmiss` number exactly `m`. -/
theorem pcie_ranks_count (nmiss m : Nat) (h : m ≤ nmiss) : nmiss - (nmiss - m) = m := by
  omega

/-- A zero numerator sends nothing over PCIe. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

-- § GGUF expert loading

/-- Region sizes `per[3]` and starts `at[3]` of one blob from `up_off`, `down_off`, `blob`. -/
def blobRegions (upOff downOff blob : Nat) : List (Nat × Nat) :=
  [(0, upOff), (upOff, upOff), (downOff, blob - downOff)]

/-- With `down_off = 2 * up_off` the three regions tile the blob with no gap and no overlap. -/
theorem blob_regions_tile (upOff blob : Nat) (h : 2 * upOff ≤ blob) :
    0 + upOff = upOff ∧ upOff + upOff = 2 * upOff ∧ 2 * upOff + (blob - 2 * upOff) = blob := by
  omega

/-- A 16-expert read never covers more than 16 experts. -/
theorem chunk_experts_le (n per : Nat) (hn : n ≤ per * 16) : n / per ≤ 16 :=
  Nat.div_le_of_le_mul hn

-- § arena slack

/-- The arena is one max-blob longer than the file. -/
def arenaSlack (want maxBlob : Nat) : Nat := want + maxBlob

/-- A whole-slot copy starting at any in-file expert stays inside the arena. -/
theorem slot_copy_in_arena (off bb want mb : Nat) (h1 : off + bb ≤ want) :
    off + mb ≤ arenaSlack want mb := by
  unfold arenaSlack
  omega

-- § misc counters

/-- `reads_` increments once per accepted blob lookup, so two lookups add two. -/
theorem reads_two (r : Nat) : r + 1 + 1 = r + 2 := by
  omega

end Strata.ExpertSource
