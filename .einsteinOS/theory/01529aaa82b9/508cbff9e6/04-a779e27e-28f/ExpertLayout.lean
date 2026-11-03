/-!
# MODULE: expert_layout
Per-layer expert table: contiguity check of `native_experts.txt`, maximum blob size and slot
rounding (`expert_layout.cpp`).

## Source semantics
Layer offsets must be contiguous: layer `l` starts where layer `l-1` ended, each layer occupying
`bytes * n_expert`. `max_blob` is the largest per-layer blob. Cache slots round each blob up to a
multiple of 256 bytes.

## Form template identifiers
* `L.offset[l]`, `L.bytes[l]` → `(o, b)` pairs; `at` → `start`; `L.max_blob` → `maxBlob`
* `(blob_bytes + 255) / 256 * 256` → `roundSlot`
-/

namespace StrataExpertLayout

/-- End of a sequence of layers starting at `start`, each of `bytes * ne`. -/
def layoutEnd (bytes : List Nat) (ne start : Nat) : Nat :=
  bytes.foldl (fun at' b => at' + b * ne) start

/-- Contiguity check over `(offset, bytes)` pairs. -/
def checkContig (ne : Nat) : Nat → List (Nat × Nat) → Bool
  | _, [] => true
  | at', (o, b) :: r => (o == at') && checkContig ne (at' + b * ne) r

-- § contiguity

/-- The end of an empty layout is its start. -/
theorem layoutEnd_nil (ne start : Nat) : layoutEnd [] ne start = start := rfl

/-- Stepping over the first layer advances the start by its size. -/
theorem layoutEnd_cons (b : Nat) (bs : List Nat) (ne start : Nat) :
    layoutEnd (b :: bs) ne start = layoutEnd bs ne (start + b * ne) := rfl

/-- The layout end is never before the start. -/
theorem layoutEnd_ge (bytes : List Nat) (ne start : Nat) : start ≤ layoutEnd bytes ne start := by
  induction bytes generalizing start with
  | nil => exact Nat.le_refl _
  | cons b bs ih =>
    rw [layoutEnd_cons]
    exact Nat.le_trans (Nat.le_add_right start (b * ne)) (ih _)

/-- A passing check forces the first offset to equal the running start. -/
theorem checkContig_first (ne at' o b : Nat) (r : List (Nat × Nat))
    (h : checkContig ne at' ((o, b) :: r) = true) : o = at' := by
  simp [checkContig] at h
  exact h.1

/-- A passing check passes for the remaining layers from the advanced start. -/
theorem checkContig_rest (ne at' o b : Nat) (r : List (Nat × Nat))
    (h : checkContig ne at' ((o, b) :: r) = true) : checkContig ne (at' + b * ne) r = true := by
  simp [checkContig] at h
  exact h.2

/-- The empty table passes. -/
theorem checkContig_nil (ne at' : Nat) : checkContig ne at' [] = true := rfl

-- § maximum blob

/-- Running maximum never drops below the accumulator. -/
theorem foldl_max_ge (l : List Nat) (acc : Nat) : acc ≤ l.foldl max acc := by
  induction l generalizing acc with
  | nil => exact Nat.le_refl _
  | cons a as ih => exact Nat.le_trans (Nat.le_max_left acc a) (ih (max acc a))

/-- Every blob size is at most `max_blob`. -/
theorem le_maxBlob (l : List Nat) (acc x : Nat) (hx : x ∈ l) : x ≤ l.foldl max acc := by
  induction l generalizing acc with
  | nil => cases hx
  | cons a as ih =>
    rcases List.mem_cons.mp hx with h | h
    · subst h
      exact Nat.le_trans (Nat.le_max_right acc x) (foldl_max_ge as (max acc x))
    · exact ih (max acc a) h

-- § slot rounding

/-- Slot size: blob bytes rounded up to a multiple of 256. -/
def roundSlot (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A rounded slot holds the blob. -/
theorem roundSlot_ge (b : Nat) : b ≤ roundSlot b := by
  unfold roundSlot; omega

/-- A rounded slot is 256-aligned. -/
theorem roundSlot_aligned (b : Nat) : roundSlot b % 256 = 0 := by
  unfold roundSlot; omega

/-- Rounding wastes fewer than 256 bytes. -/
theorem roundSlot_lt (b : Nat) : roundSlot b < b + 256 := by
  unfold roundSlot; omega

-- § CPU capability probe

/-- XCR0 mask `0xE6` requires the SSE/AVX/opmask/ZMM state bits. -/
theorem xcr0_mask : 0xE6 = 2 ^ 1 + 2 ^ 2 + 2 ^ 5 + 2 ^ 6 + 2 ^ 7 := by decide

end StrataExpertLayout
