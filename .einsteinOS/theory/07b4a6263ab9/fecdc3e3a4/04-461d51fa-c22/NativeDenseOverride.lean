/-!
# MODULE: native_dense_override
SOURCES: src/core/native_dense.cpp
Attaches GGUF-form ("native") copies of the GDN/QSA/shared-expert projection matrices to the canonical
weight table.  Every directory span of every shard is validated (block geometry, overflow-free extent,
payload bound, non-overlap with the next tensor), every eligible matrix must match its canonical shape,
and only when ALL checks and allocations succeed are the references published, sharing one Q8_1 scratch
sized for the widest input.

## Source semantics
* A tensor's byte count is `(elements / block_elements) * block_bytes`; `elements` is the product of the
  dimensions, each multiplication guarded by `elements ≤ MAX / dim`.
* A span is valid iff `offset ≤ payload` and `bytes ≤ payload - offset`; it does not overlap the next
  tensor (by sorted offset) iff `bytes ≤ next - offset`.
* Publication is all-or-nothing: either every pending reference is published or none is.
* The shared scratch is sized for the maximum `ne0` over the published matrices, so every projection's
  input fits.

## Form template identifiers
extentOk ↔ `elements > MAX / dimension` guard · spanOk ↔ truncated-payload check ·
noOverlap ↔ `bytes > *next - offset` check · publishAll ↔ the final publish loop ·
maxIn ↔ max_in · shapeMatch ↔ incompatible-matrix check
-/

namespace NativeDenseOverride

-- § overflow-free extents

/-- The multiplication guard: if `e ≤ M / d` then `e * d` does not exceed `M`. -/
theorem guarded_mul_le (e d M : Nat) (hd : 0 < d) (h : e ≤ M / d) : e * d ≤ M :=
  (Nat.le_div_iff_mul_le hd).mp h

/-- Extent of a shape, guarded: `none` on a zero dimension or a would-be overflow past `M`. -/
def extent (M : Nat) : List Nat → Option Nat
  | [] => some 1
  | d :: ds =>
    match extent M ds with
    | none => none
    | some e => if d = 0 then none else if e ≤ M / d then some (e * d) else none

/-- A guarded extent never exceeds the bound, for any bound at least 1. -/
theorem extent_le (M : Nat) (hM : 1 ≤ M) : ∀ (ds : List Nat) (e : Nat), extent M ds = some e → e ≤ M
  | [], e, h => by
    simp only [extent, Option.some.injEq] at h
    omega
  | d :: ds, e, h => by
    simp only [extent] at h
    split at h
    · exact absurd h (by simp)
    · rename_i e' _
      by_cases hd : d = 0
      · simp [hd] at h
      · by_cases hg : e' ≤ M / d
        · simp only [hd, hg, ite_false, ite_true, Option.some.injEq] at h
          subst h
          exact guarded_mul_le e' d M (Nat.pos_of_ne_zero hd) hg
        · simp [hd, hg] at h

/-- A shape with a zero dimension is rejected. -/
theorem extent_zero_dim (M : Nat) (ds : List Nat) : extent M (0 :: ds) = none := by
  simp only [extent]
  split <;> simp

-- § block geometry and payload spans

/-- Bytes of a tensor of `elements` elements stored in blocks. -/
def tensorBytes (elements blockElems blockBytes : Nat) : Nat := elements / blockElems * blockBytes

/-- When the row width is a multiple of the block, a whole number of blocks covers the tensor exactly. -/
theorem blocks_exact (rows blocksPerRow blockElems blockBytes : Nat) (hb : 0 < blockElems) :
    tensorBytes (rows * (blocksPerRow * blockElems)) blockElems blockBytes =
      rows * blocksPerRow * blockBytes := by
  unfold tensorBytes
  rw [← Nat.mul_assoc, Nat.mul_div_cancel _ hb]

/-- The span check `load` applies to every directory entry. -/
def spanOk (offset bytes payload : Nat) : Prop := offset ≤ payload ∧ bytes ≤ payload - offset

/-- A valid span lies inside the payload. -/
theorem span_inside (o b p : Nat) (h : spanOk o b p) : o + b ≤ p := by
  unfold spanOk at h; omega

/-- The non-overlap check against the next tensor by sorted offset. -/
def noOverlap (offset bytes next : Nat) : Prop := offset < next → bytes ≤ next - offset

/-- A tensor passing the check ends no later than the next one starts. -/
theorem no_overlap_end (o b n : Nat) (hlt : o < n) (h : noOverlap o b n) : o + b ≤ n := by
  have := h hlt
  omega

/-- Two checked tensors with offsets `o1 < o2` occupy disjoint byte ranges. -/
theorem checked_disjoint (o1 b1 o2 b2 p : Nat) (hlt : o1 < o2) (h1 : noOverlap o1 b1 o2)
    (h2 : spanOk o2 b2 p) : o1 + b1 ≤ o2 ∧ o2 + b2 ≤ p :=
  ⟨no_overlap_end o1 b1 o2 hlt h1, span_inside o2 b2 p h2⟩

-- § shape agreement with the canonical table

/-- A native matrix replaces a canonical one only if both dimensions match and fit an `int`. -/
def shapeMatch (s0 s1 ne0 ne1 : Nat) : Prop :=
  s0 = ne0 ∧ s1 = ne1 ∧ 0 < ne0 ∧ ne0 ≤ 2147483647 ∧ 0 < ne1 ∧ ne1 ≤ 2147483647

/-- A matched matrix has positive dimensions that survive the `(int)` cast. -/
theorem shapeMatch_int_safe (s0 s1 ne0 ne1 : Nat) (h : shapeMatch s0 s1 ne0 ne1) :
    0 < s0 ∧ s0 < 2147483648 ∧ 0 < s1 ∧ s1 < 2147483648 := by
  unfold shapeMatch at h; omega

-- § all-or-nothing publication

/-- Publish the pending set only if every check passed. -/
def publishAll {α : Type} (checks : List Bool) (pending : List α) : List α :=
  if checks.all id then pending else []

/-- If anything was published, every check passed. -/
theorem published_implies_checked {α : Type} (checks : List Bool) (pending : List α)
    (h : publishAll checks pending ≠ []) : ∀ c ∈ checks, c = true := by
  unfold publishAll at h
  split at h
  · rename_i hall
    intro c hc
    have := List.all_eq_true.mp hall c hc
    simpa using this
  · exact absurd rfl h

/-- A single failing check publishes nothing. -/
theorem one_failure_publishes_nothing {α : Type} (checks : List Bool) (pending : List α)
    (h : false ∈ checks) : publishAll checks pending = [] := by
  unfold publishAll
  have : checks.all id = false := by
    cases hc : checks.all id with
    | false => rfl
    | true =>
      have := List.all_eq_true.mp hc false h
      simp at this
  simp [this]

-- § the shared scratch

/-- The widest input over the published matrices (`max_in`). -/
def maxIn : List Nat → Nat
  | [] => 0
  | x :: xs => max x (maxIn xs)

/-- Every published matrix's input width fits the shared Q8_1 scratch. -/
theorem input_fits_scratch : ∀ (xs : List Nat), ∀ x ∈ xs, x ≤ maxIn xs
  | [], x, hx => by simp at hx
  | y :: ys, x, hx => by
    simp only [List.mem_cons] at hx
    simp only [maxIn]
    rcases hx with rfl | hx
    · omega
    · have := input_fits_scratch ys x hx
      omega

end NativeDenseOverride
