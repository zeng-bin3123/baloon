/-!
# MODULE: expert_blob_indexing
Flat blob indexing for the expert arena (`layer * n_expert + expert`), the separate per-axis bounds check
that prevents cross-layer aliasing, and the weight loader's per-plane segment accounting.

## Source semantics
* `blob(layer, expert)` is null unless `0 ≤ layer`, `0 ≤ expert < n_expert` and the flat index `< blobs`.
* `blob(0, n_expert)` would equal `blob(1, 0)` as a flat index, which is why `expert < n_expert` is checked on its own.
* The pack file holds exactly `n_layers * n_expert * BLOB` bytes with no padding.
* A quantized tensor's planes (codes, scales, offsets) are contiguous; widening fp16 scales to f32 doubles that plane.

## Form template identifiers
* `flatIndex`  ← `layer * n_expert_ + expert`
* `nExpert`    ← `n_expert_`
* `blobs`      ← `blobs_`
* `BLOB`       ← `strata::kernels::cpu::BLOB` (1,382,400)
-/

namespace Strata.ExpertBlobIndexing

-- § flat index

def flatIndex (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- The source's guard (Nat model: negatives are already excluded). -/
def blobOk (nExpert blobs layer expert : Nat) : Bool :=
  decide (expert < nExpert) && decide (flatIndex nExpert layer expert < blobs)

/-- Without the per-axis check, `(layer, n_expert)` aliases `(layer+1, 0)`. -/
theorem flat_alias (nExpert layer : Nat) :
    flatIndex nExpert layer nExpert = flatIndex nExpert (layer + 1) 0 := by
  unfold flatIndex
  rw [Nat.add_mul, one_mul, Nat.add_zero, Nat.add_comm nExpert]
  omega

/-- The per-axis guard rejects the aliasing index. -/
theorem guard_rejects_overrun (nExpert blobs layer : Nat) :
    blobOk nExpert blobs layer nExpert = false := by
  unfold blobOk
  simp

/-- Within the guard, flat indexing is injective in `(layer, expert)`. -/
theorem flat_injective (nExpert l1 e1 l2 e2 : Nat) (h1 : e1 < nExpert) (h2 : e2 < nExpert)
    (h : flatIndex nExpert l1 e1 = flatIndex nExpert l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold flatIndex at h
  have hl : l1 = l2 := by
    rcases Nat.lt_trichotomy l1 l2 with hlt | heq | hgt
    · exfalso
      have : (l1 + 1) * nExpert ≤ l2 * nExpert := Nat.mul_le_mul_right _ hlt
      rw [Nat.add_mul, one_mul] at this
      omega
    · exact heq
    · exfalso
      have : (l2 + 1) * nExpert ≤ l1 * nExpert := Nat.mul_le_mul_right _ hgt
      rw [Nat.add_mul, one_mul] at this
      omega
  subst hl
  exact ⟨rfl, by omega⟩

/-- A guarded index lies inside the file: the byte offset plus one blob fits. -/
theorem blob_in_file (nExpert blobs blobBytes layer expert : Nat)
    (h : blobOk nExpert blobs layer expert = true) :
    flatIndex nExpert layer expert * blobBytes + blobBytes ≤ blobs * blobBytes := by
  unfold blobOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨_, hi⟩ := h
  have h1 : (flatIndex nExpert layer expert + 1) * blobBytes ≤ blobs * blobBytes :=
    Nat.mul_le_mul_right _ hi
  rw [Nat.add_mul, one_mul] at h1
  exact h1

/-- The real artifact: 48 × 512 blobs of 1,382,400 B is 33,973,862,400 B. -/
theorem artifact_size : 48 * 512 * 1382400 = 33973862400 := by decide

-- § loader segments

/-- Source and destination bytes of a quantized tensor's planes. -/
structure Planes where
  codes : Nat
  scales : Nat
  offsets : Nat

/-- Destination size (always f32 scales). -/
def Planes.dst (p : Planes) : Nat := p.codes + p.scales + p.offsets

/-- Source size: an fp16 scale plane is half the width in the pack. -/
def Planes.src (p : Planes) (fp16 : Bool) : Nat :=
  p.codes + (if fp16 then p.scales / 2 else p.scales) + p.offsets

/-- With even scale bytes, widening exactly doubles the source scale plane. -/
theorem widen_doubles (p : Planes) (h : p.scales % 2 = 0) :
    p.src true + p.scales / 2 = p.dst := by
  unfold Planes.src Planes.dst
  simp only [if_true]
  omega

/-- Without fp16 scales the two layouts agree. -/
theorem copy_same (p : Planes) : p.src false = p.dst := by
  unfold Planes.src Planes.dst
  simp

/-- A promoted (bf16/f16-in-f32) tensor halves in size: 4 B/elem in, 2 B/elem out. -/
theorem promoted_halves (elems : Nat) : (elems * 4) / 2 = elems * 2 := by
  omega

end Strata.ExpertBlobIndexing
