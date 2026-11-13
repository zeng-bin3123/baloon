/-!
# MODULE: expert_blob_addressing
SOURCES: include/ref/core/expert_source.hpp, src/core/expert_source.cpp
Models how an expert source turns a routed `(layer, expert)` pair into a byte range of the expert
arena: the flat index `layer * n_expert + expert` for the uniform `experts.bin`, and the per-layer
layout (`layer_offset + expert * blob_bytes(layer)`) used by `ArenaExpertSource`.

## Source semantics
* `experts.bin` holds exactly `48 * 512 = 24576` blobs of `BLOB = 1,382,400` bytes, no padding.
* `blob(layer, expert)` refuses negative ids, refuses `expert >= n_expert` SEPARATELY from the flat-index
  bound (otherwise `blob(0, 512)` would alias `blob(1, 0)`), and refuses a flat index past the file.
* Every accepted blob lies wholly inside the file, and distinct valid pairs give disjoint byte ranges.
* Per-layer layout: layer `l+1` starts where layer `l`'s `n_expert` blobs end; the arena is one
  max-blob longer than the file so a whole-slot copy may start at any expert.

## Form template identifiers
`FileExpertSource::blob` → `blob`; `layer * n_expert_ + expert` → `flatIndex`;
`base_ + i * BLOB` → `byteOffset`; `ExpertLayout::layer_offset` → `layerOffset`;
`ExpertLayout::blob_offset` → `blobOffsetL`; `ExpertLayout::blob_bytes` → `bytes`.
-/

namespace Ref.ExpertBlobAddressing

-- § Geometry constants

/-- Bytes in one expert blob (`ref::kernels::cpu::BLOB`). -/
def BLOB : Nat := 1382400

/-- Layers of the artifact. -/
def nLayers : Nat := 48

/-- Experts per layer. -/
def nExpert : Nat := 512

/-- `experts.bin` holds exactly 24,576 blobs. -/
theorem experts_bin_blobs : nLayers * nExpert = 24576 := by decide

/-- `experts.bin` is exactly 33,973,862,400 bytes, the size `open` checks against. -/
theorem experts_bin_size : nLayers * nExpert * BLOB = 33973862400 := by decide

-- § Flat index

/-- The flat blob index `layer * n_expert + expert`. -/
def flatIndex (nE l e : Nat) : Nat := l * nE + e

/-- Helper: a layer below `nL` leaves room for one more full layer of `nE` blobs. -/
theorem layer_room {l nL : Nat} (nE : Nat) (h : l < nL) : l * nE + nE ≤ nL * nE := by
  have h1 : (l + 1) * nE ≤ nL * nE := Nat.mul_le_mul_right nE h
  rw [Nat.succ_mul] at h1
  exact h1

/-- A valid pair has a flat index inside the file. -/
theorem flatIndex_lt {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    flatIndex nE l e < nL * nE := by
  have h := layer_room nE hl
  unfold flatIndex
  omega

/-- The layer is recovered from the flat index by division. -/
theorem flatIndex_div {nE l e : Nat} (he : e < nE) : flatIndex nE l e / nE = l := by
  have hpos : 0 < nE := by omega
  unfold flatIndex
  rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he, Nat.add_zero]

/-- The expert is recovered from the flat index by remainder. -/
theorem flatIndex_mod {nE l e : Nat} (he : e < nE) : flatIndex nE l e % nE = e := by
  unfold flatIndex
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt he]

/-- Distinct in-range pairs never share a flat index (layer and expert are separate axes). -/
theorem flatIndex_injective {nE l1 e1 l2 e2 : Nat} (h1 : e1 < nE) (h2 : e2 < nE)
    (h : flatIndex nE l1 e1 = flatIndex nE l2 e2) : l1 = l2 ∧ e1 = e2 := by
  have hl : l1 = l2 :=
    calc l1 = flatIndex nE l1 e1 / nE := (flatIndex_div h1).symm
      _ = flatIndex nE l2 e2 / nE := by rw [h]
      _ = l2 := flatIndex_div h2
  have he : e1 = e2 :=
    calc e1 = flatIndex nE l1 e1 % nE := (flatIndex_mod h1).symm
      _ = flatIndex nE l2 e2 % nE := by rw [h]
      _ = e2 := flatIndex_mod h2
  exact ⟨hl, he⟩

/-- THE HAZARD the separate expert check exists for: without it, expert `nE` of layer 0 is the
flat index of expert 0 of layer 1 - finite, correctly sized, and wrong. -/
theorem unchecked_alias (nE : Nat) : flatIndex nE 0 nE = flatIndex nE 1 0 := by
  simp [flatIndex]

-- § Byte ranges

/-- Byte offset of a blob in the uniform file. -/
def byteOffset (nE l e : Nat) : Nat := flatIndex nE l e * BLOB

/-- Every valid blob lies wholly inside the file. -/
theorem blob_in_file {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    byteOffset nE l e + BLOB ≤ nL * nE * BLOB := by
  have h := flatIndex_lt hl he
  have h2 : (flatIndex nE l e + 1) * BLOB ≤ nL * nE * BLOB := Nat.mul_le_mul_right BLOB h
  rw [Nat.succ_mul] at h2
  exact h2

/-- Helper: blob `i < j` ends at or before blob `j` starts. -/
theorem blob_before {i j : Nat} (h : i < j) : i * BLOB + BLOB ≤ j * BLOB := by
  have h2 : (i + 1) * BLOB ≤ j * BLOB := Nat.mul_le_mul_right BLOB h
  rw [Nat.succ_mul] at h2
  exact h2

/-- Distinct blob indices give disjoint byte ranges. -/
theorem blobs_disjoint {i j : Nat} (hne : i ≠ j) :
    i * BLOB + BLOB ≤ j * BLOB ∨ j * BLOB + BLOB ≤ i * BLOB := by
  rcases Nat.lt_or_gt_of_ne hne with h | h
  · exact Or.inl (blob_before h)
  · exact Or.inr (blob_before h)

-- § The `blob()` model

/-- `FileExpertSource::blob` on signed ids: `none` is the null pointer. -/
def blob (nL nE : Nat) (l e : Int) : Option Nat :=
  if l < 0 ∨ e < 0 then none
  else if (nE : Int) ≤ e then none
  else if nL * nE ≤ flatIndex nE l.toNat e.toNat then none
  else some (byteOffset nE l.toNat e.toNat)

/-- The expert one past the end of a layer is refused, never aliased. -/
theorem blob_refuses_expert_overflow (nL nE : Nat) (l : Int) : blob nL nE l nE = none := by
  unfold blob
  split
  · rfl
  · simp

/-- A valid pair is accepted at its byte offset. -/
theorem blob_valid {nL nE : Nat} {l e : Int} (hl0 : 0 ≤ l) (he0 : 0 ≤ e)
    (he : e.toNat < nE) (hl : l.toNat < nL) :
    blob nL nE l e = some (byteOffset nE l.toNat e.toNat) := by
  have hlt := flatIndex_lt hl he
  have hneg : ¬ (l < 0 ∨ e < 0) := by omega
  have hexp : ¬ ((nE : Int) ≤ e) := by omega
  have hflat : ¬ (nL * nE ≤ flatIndex nE l.toNat e.toNat) := by omega
  unfold blob
  simp [hneg, hexp, hflat]

/-- Anything accepted is a valid pair whose blob lies inside the file. -/
theorem blob_some_sound {nL nE : Nat} {l e : Int} {o : Nat} (h : blob nL nE l e = some o) :
    0 ≤ l ∧ 0 ≤ e ∧ e.toNat < nE ∧ o + BLOB ≤ nL * nE * BLOB := by
  unfold blob at h
  split at h
  · cases h
  · rename_i hneg
    split at h
    · cases h
    · rename_i hexp
      split at h
      · cases h
      · rename_i hflat
        have he : e.toNat < nE := by omega
        have hl : l.toNat < nL := by
          refine Nat.lt_of_not_le fun hge => hflat ?_
          have := Nat.mul_le_mul_right nE hge
          unfold flatIndex
          omega
        obtain rfl := Option.some.inj h
        exact ⟨by omega, by omega, he, blob_in_file hl he⟩

-- § Per-layer layout (ArenaExpertSource)

/-- `layer_offset`: layers laid out back to back, `nE` blobs of `bytes l` each. -/
def layerOffset (bytes : Nat → Nat) (nE : Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => layerOffset bytes nE l + nE * bytes l

/-- `blob_offset(l, e)`. -/
def blobOffsetL (bytes : Nat → Nat) (nE l e : Nat) : Nat :=
  layerOffset bytes nE l + e * bytes l

/-- A blob of layer `l` ends at or before layer `l+1` begins. -/
theorem blobOffsetL_in_layer (bytes : Nat → Nat) {nE l e : Nat} (he : e < nE) :
    blobOffsetL bytes nE l e + bytes l ≤ layerOffset bytes nE (l + 1) := by
  have h := layer_room (bytes l) he
  simp only [blobOffsetL, layerOffset]
  omega

/-- Layer offsets are monotone. -/
theorem layerOffset_mono (bytes : Nat → Nat) (nE : Nat) :
    ∀ {a b : Nat}, a ≤ b → layerOffset bytes nE a ≤ layerOffset bytes nE b := by
  intro a b hab
  induction b with
  | zero =>
    have : a = 0 := by omega
    subst this; exact Nat.le_refl _
  | succ b ih =>
    rcases Nat.lt_or_ge a (b + 1) with h | h
    · have := ih (by omega)
      simp only [layerOffset]
      omega
    · have : a = b + 1 := by omega
      subst this; exact Nat.le_refl _

/-- Every valid arena blob lies inside the arena's file-sized prefix `layerOffset nL`. -/
theorem blobOffsetL_in_total (bytes : Nat → Nat) {nL nE l e : Nat} (hl : l < nL) (he : e < nE) :
    blobOffsetL bytes nE l e + bytes l ≤ layerOffset bytes nE nL := by
  have h1 := blobOffsetL_in_layer bytes (l := l) he
  have h2 := layerOffset_mono bytes nE (a := l + 1) (b := nL) hl
  omega

/-- The arena is one max-blob longer than the file, so a whole-slot copy of `maxBlob` bytes
starting at any valid expert stays inside the allocation. -/
theorem slot_copy_in_arena (bytes : Nat → Nat) {nL nE l e maxBlob : Nat}
    (hl : l < nL) (he : e < nE) :
    blobOffsetL bytes nE l e + maxBlob ≤ layerOffset bytes nE nL + maxBlob := by
  have := blobOffsetL_in_total bytes hl he
  omega

end Ref.ExpertBlobAddressing

