/-!
# MODULE: expert_blob_addressing
SOURCES: include/strata/core/expert_source.hpp, src/core/expert_source.cpp
How a routed `(layer, expert)` pair becomes a byte offset into `experts.bin` (memory-mapped,
`FileExpertSource`) or into the resident arena (`ArenaExpertSource`), and the size check
`open` performs before trusting the file.

## Source semantics
* `experts.bin` holds exactly `n_layers * n_expert` blobs of `BLOB = 1,382,400` bytes, no padding;
  `48 * 512 * BLOB = 33,973,862,400`.
* `blob(l, e)` returns null for negative axes, for `e ≥ n_expert` (checked SEPARATELY, because the
  flat index `l * n_expert + e` alone would alias `(l, n_expert)` onto `(l+1, 0)`), and for a flat
  index past `blobs`.  Otherwise it is `base + (l * n_expert + e) * BLOB`.
* `open` refuses an empty geometry and any file whose size differs from `blobs * BLOB`.
* The arena uses a per-layer layout: `blob_offset(l, e) = layer_offset(l) + e * blob_bytes(l)`;
  `pinned(l, e)` holds iff the whole blob lies in the registered prefix `[0, pinned_bytes)`.

## Form template identifiers
`FileExpertSource::blob` → `fileBlob`; `FileExpertSource::open` → `openCheck`;
`ArenaExpertSource::pinned` → `arenaPinned`; `ExpertLayout::blob_offset` → `blobOffset`;
`ExpertLayout::layer_offset` → `layerOffset`; `BLOB` → `BLOB`; `H` → `H`.
-/

namespace Strata.ExpertBlobAddressing

-- § constants

/-- Bytes per expert blob in the canonical pack (`strata::kernels::cpu::BLOB`). -/
def BLOB : Nat := 1382400

/-- Hidden width the expert kernel is compiled for (`strata::kernels::cpu::H`). -/
def H : Nat := 2560

/-- The canonical pack's size is exactly 48 layers × 512 experts × BLOB bytes, as the header states. -/
theorem experts_bin_size : 48 * 512 * BLOB = 33973862400 := by decide

/-- The canonical pack holds exactly 24,576 blobs. -/
theorem experts_bin_blobs : 48 * 512 = 24576 := by decide

-- § the file-backed source

/-- Geometry of an opened source: experts per layer and total blobs. -/
structure Src where
  nExpert : Nat
  blobs : Nat

/-- `FileExpertSource::blob`: the byte offset of `(layer, expert)`, or `none` for null. -/
def fileBlob (s : Src) (layer expert : Int) : Option Nat :=
  if layer < 0 ∨ expert < 0 then none
  else if expert ≥ s.nExpert then none
  else
    let i := layer.toNat * s.nExpert + expert.toNat
    if i ≥ s.blobs then none else some (i * BLOB)

/-- Negative layer or expert ids never produce a blob. -/
theorem fileBlob_neg (s : Src) (l e : Int) (h : l < 0 ∨ e < 0) : fileBlob s l e = none := by
  simp [fileBlob, h]

/-- An expert id at or past `n_expert` never produces a blob, whatever the layer. -/
theorem fileBlob_expert_oob (s : Src) (l e : Int) (h : e ≥ s.nExpert) : fileBlob s l e = none := by
  unfold fileBlob
  split
  · rfl
  · simp [h]

/-- Why the separate expert check exists: without it, `(l, n)` has the same flat index as `(l+1, 0)`. -/
theorem flat_alias (l n : Nat) : l * n + n = (l + 1) * n + 0 := by
  rw [Nat.add_mul, Nat.one_mul, Nat.add_zero]

/-- A produced offset is the flat index times BLOB, with both axes in range. -/
theorem fileBlob_some (s : Src) (l e : Int) (off : Nat) (h : fileBlob s l e = some off) :
    0 ≤ l ∧ 0 ≤ e ∧ e < s.nExpert ∧ l.toNat * s.nExpert + e.toNat < s.blobs ∧
      off = (l.toNat * s.nExpert + e.toNat) * BLOB := by
  unfold fileBlob at h
  by_cases h1 : l < 0 ∨ e < 0
  · simp [h1] at h
  · simp only [h1, if_false] at h
    by_cases h2 : e ≥ s.nExpert
    · simp [h2] at h
    · simp only [h2, if_false] at h
      by_cases h3 : l.toNat * s.nExpert + e.toNat ≥ s.blobs
      · simp [h3] at h
      · simp only [h3, if_false, Option.some.injEq] at h
        refine ⟨by omega, by omega, by omega, by omega, h.symm⟩

/-- The flat index determines the layer and expert uniquely when `e < n` (mod/div decomposition). -/
theorem flat_inj (n l e l' e' : Nat) (he : e < n) (he' : e' < n)
    (h : l * n + e = l' * n + e') : l = l' ∧ e = e' := by
  have hm : ∀ a b : Nat, b < n → (a * n + b) % n = b := by
    intro a b hb
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]
  have hd : ∀ a b : Nat, b < n → (a * n + b) / n = a := by
    intro a b hb
    have hn : 0 < n := by omega
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hb, Nat.zero_add]
  have h1 := hm l e he
  have h2 := hd l e he
  rw [h] at h1 h2
  rw [hm l' e' he'] at h1
  rw [hd l' e' he'] at h2
  exact ⟨h2.symm, h1.symm⟩

/-- Two different routed pairs never share a blob: the file source is injective on valid ids. -/
theorem fileBlob_injective (s : Src) (l e l' e' : Int) (off : Nat)
    (h1 : fileBlob s l e = some off) (h2 : fileBlob s l' e' = some off) : l = l' ∧ e = e' := by
  obtain ⟨hl, he, hen, _, ho⟩ := fileBlob_some s l e off h1
  obtain ⟨hl', he', hen', _, ho'⟩ := fileBlob_some s l' e' off h2
  have hB : 0 < BLOB := by decide
  have hidx : l.toNat * s.nExpert + e.toNat = l'.toNat * s.nExpert + e'.toNat := by
    rw [ho] at ho'
    exact Nat.eq_of_mul_eq_mul_right hB ho'
  obtain ⟨a, b⟩ := flat_inj s.nExpert l.toNat e.toNat l'.toNat e'.toNat (by omega) (by omega) hidx
  exact ⟨by omega, by omega⟩

/-- A produced blob ends inside the file: `off + BLOB ≤ blobs * BLOB`. -/
theorem fileBlob_in_file (s : Src) (l e : Int) (off : Nat) (h : fileBlob s l e = some off) :
    off + BLOB ≤ s.blobs * BLOB := by
  obtain ⟨_, _, _, hi, ho⟩ := fileBlob_some s l e off h
  rw [ho, ← Nat.succ_mul]
  exact Nat.mul_le_mul_right _ hi

-- § the size check in `open`

/-- `FileExpertSource::open`'s refusal logic: `some blobs` on success. -/
def openCheck (nLayers nExpert fileSize : Nat) : Option Nat :=
  if nLayers = 0 ∨ nExpert = 0 then none
  else if fileSize ≠ nLayers * nExpert * BLOB then none
  else some (nLayers * nExpert)

/-- A successful open means the file is exactly `blobs * BLOB` bytes and the geometry is non-empty. -/
theorem openCheck_some (nl ne sz b : Nat) (h : openCheck nl ne sz = some b) :
    0 < nl ∧ 0 < ne ∧ b = nl * ne ∧ sz = b * BLOB := by
  unfold openCheck at h
  by_cases h1 : nl = 0 ∨ ne = 0
  · simp [h1] at h
  · simp only [h1, if_false] at h
    by_cases h2 : sz ≠ nl * ne * BLOB
    · simp [h2] at h
    · simp only [h2, if_false, Option.some.injEq] at h
      exact ⟨by omega, by omega, h.symm, by rw [← h]; omega⟩

/-- A short or over-long file is refused. -/
theorem openCheck_wrong_size (nl ne sz : Nat) (h : sz ≠ nl * ne * BLOB) : openCheck nl ne sz = none := by
  unfold openCheck
  split
  · rfl
  · simp [h]

-- § the arena layout

/-- `ExpertLayout::layer_offset`: sum of the previous layers' `blob_bytes * n_expert`. -/
def layerOffset (bytes : Nat → Nat) (nE : Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => layerOffset bytes nE l + bytes l * nE

/-- `ExpertLayout::blob_offset`. -/
def blobOffset (bytes : Nat → Nat) (nE l e : Nat) : Nat :=
  layerOffset bytes nE l + e * bytes l

/-- `ArenaExpertSource::pinned`: the whole blob lies in the registered prefix. -/
def arenaPinned (bytes : Nat → Nat) (nE pinnedBytes l e : Nat) : Bool :=
  e < nE && blobOffset bytes nE l e + bytes l ≤ pinnedBytes

/-- Within a layer, every blob ends at or before the next layer starts. -/
theorem blob_within_layer (bytes : Nat → Nat) (nE l e : Nat) (he : e < nE) :
    blobOffset bytes nE l e + bytes l ≤ layerOffset bytes nE (l + 1) := by
  unfold blobOffset
  simp only [layerOffset]
  have : e * bytes l + bytes l ≤ bytes l * nE := by
    rw [← Nat.succ_mul, Nat.mul_comm (bytes l) nE]
    exact Nat.mul_le_mul_right _ he
  omega

/-- Layer offsets are monotone, so layers never overlap. -/
theorem layerOffset_mono (bytes : Nat → Nat) (nE l m : Nat) (h : l ≤ m) :
    layerOffset bytes nE l ≤ layerOffset bytes nE m := by
  induction m with
  | zero => have : l = 0 := by omega
            subst this; exact Nat.le_refl _
  | succ m ih =>
    by_cases hl : l = m + 1
    · subst hl; exact Nat.le_refl _
    · have := ih (by omega)
      simp only [layerOffset]
      omega

/-- Pinned residency is downward closed in the expert index (the registered region is a prefix). -/
theorem arenaPinned_prefix (bytes : Nat → Nat) (nE pb l e e' : Nat) (he : e' ≤ e)
    (h : arenaPinned bytes nE pb l e = true) : arenaPinned bytes nE pb l e' = true := by
  simp only [arenaPinned, blobOffset, Bool.and_eq_true, decide_eq_true_eq] at *
  obtain ⟨h1, h2⟩ := h
  have : e' * bytes l ≤ e * bytes l := Nat.mul_le_mul_right _ he
  exact ⟨by omega, by omega⟩

/-- With a uniform blob size, layer `l` starts at `l * nE * b`. -/
theorem layerOffset_uniform (b nE l : Nat) : layerOffset (fun _ => b) nE l = l * nE * b := by
  induction l with
  | zero => simp [layerOffset]
  | succ l ih =>
    show layerOffset (fun _ => b) nE l + b * nE = (l + 1) * nE * b
    rw [ih, Nat.succ_mul, Nat.add_mul, Nat.mul_comm b nE]

/-- With a uniform blob size the arena layout coincides with the file source's flat layout. -/
theorem uniform_layout (b nE l e : Nat) :
    blobOffset (fun _ => b) nE l e = (l * nE + e) * b := by
  show layerOffset (fun _ => b) nE l + e * b = (l * nE + e) * b
  rw [layerOffset_uniform, Nat.add_mul]

end Strata.ExpertBlobAddressing
