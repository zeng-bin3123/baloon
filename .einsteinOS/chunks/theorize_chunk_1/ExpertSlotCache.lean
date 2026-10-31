/-!
# MODULE: expert_slot_cache
Slot storage and residency table for a VRAM expert cache: per-layer slot ranges that exactly tile `0..slots`,
a no-eviction admit counter per layer, 256-byte aligned variable-size slot offsets, and the flat
`(layer, expert)` residency index.

## Source semantics
* `layer_slot_range`: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the LAST layer takes the
  remainder, so ranges cover `0..slots` with no gap and no overlap.
* `admit` never evicts: a layer's counter only advances while it is below that layer's `hi`.
* `open_sized`: slot offsets are prefix sums of 256-aligned sizes, so every slot starts 256-aligned.
* The residency index `layer * n_expert + expert` is in range iff both axes are in range.

## Form template identifiers
* `slots`, `nLayers`, `nExpert`   : source `slots_`, `n_layers_`, `n_expert_`
* `layerLo`, `layerHi`            : `layer_slot_range` outputs `lo`, `hi`
* `align256`                      : `(x + 255) / 256 * 256`
-/

namespace Strata.ExpertCache

-- § per-layer ranges

/-- Quota per layer. -/
def quota (slots nLayers : Nat) : Nat := slots / nLayers

/-- Lower bound of layer `l`'s range. -/
def layerLo (slots nLayers l : Nat) : Nat := l * quota slots nLayers

/-- Upper bound (exclusive) of layer `l`'s range; the last layer absorbs the remainder. -/
def layerHi (slots nLayers l : Nat) : Nat :=
  if l = nLayers - 1 then slots else (l + 1) * quota slots nLayers

/-- Every range is well formed: `lo ≤ hi`. -/
theorem lo_le_hi (slots nLayers l : Nat) (hn : 0 < nLayers) (hl : l < nLayers) :
    layerLo slots nLayers l ≤ layerHi slots nLayers l := by
  unfold layerLo layerHi quota
  have hq : nLayers * (slots / nLayers) ≤ slots := Nat.mul_div_le slots nLayers
  by_cases h : l = nLayers - 1
  · simp only [h, if_true]
    have h1 : (nLayers - 1) * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (Nat.sub_le _ _)
    omega
  · simp only [h, if_false]
    exact Nat.mul_le_mul_right _ (Nat.le_succ _)

/-- Adjacent ranges abut: layer `l`'s `hi` is layer `l+1`'s `lo` (no slot orphaned, none shared). -/
theorem hi_eq_next_lo (slots nLayers l : Nat) (hl : l + 1 < nLayers) :
    layerHi slots nLayers l = layerLo slots nLayers (l + 1) := by
  unfold layerHi layerLo
  have h : ¬ (l = nLayers - 1) := by omega
  simp [h]

/-- The first range starts at zero. -/
theorem first_lo (slots nLayers : Nat) : layerLo slots nLayers 0 = 0 := by
  unfold layerLo
  simp

/-- The last range ends exactly at `slots`, so the ranges cover `0..slots`. -/
theorem last_hi (slots nLayers : Nat) (hn : 0 < nLayers) :
    layerHi slots nLayers (nLayers - 1) = slots := by
  unfold layerHi
  simp

/-- A non-final layer's range has exactly the quota size. -/
theorem nonlast_size (slots nLayers l : Nat) (hl : l + 1 < nLayers) :
    layerHi slots nLayers l - layerLo slots nLayers l = quota slots nLayers := by
  unfold layerHi layerLo
  have h : ¬ (l = nLayers - 1) := by omega
  simp only [h, if_false]
  rw [Nat.succ_mul]
  omega

/-- No slot below `slots` is outside the last layer's range or an earlier one: a slot in a layer's range is `< slots`. -/
theorem hi_le_slots (slots nLayers l : Nat) (hn : 0 < nLayers) (hl : l < nLayers) :
    layerHi slots nLayers l ≤ slots := by
  unfold layerHi quota
  have hq : nLayers * (slots / nLayers) ≤ slots := Nat.mul_div_le slots nLayers
  by_cases h : l = nLayers - 1
  · simp [h]
  · simp only [h, if_false]
    have h1 : (l + 1) * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (by omega)
    omega

-- § admit without eviction

/-- One step of the per-layer admit: returns the new counter and whether a slot was granted. -/
def admitStep (next hi : Nat) : Nat × Bool :=
  if next ≥ hi then (next, false) else (next + 1, true)

/-- A refused admit leaves the counter unchanged (no eviction, no drift). -/
theorem admit_refused_unchanged (next hi : Nat) (h : next ≥ hi) :
    (admitStep next hi).1 = next ∧ (admitStep next hi).2 = false := by
  unfold admitStep
  simp [h]

/-- A granted admit hands out slot `next`, strictly below `hi`, and advances by one. -/
theorem admit_granted (next hi : Nat) (h : next < hi) :
    (admitStep next hi).1 = next + 1 ∧ (admitStep next hi).2 = true ∧ next < hi := by
  unfold admitStep
  have h' : ¬ (next ≥ hi) := by omega
  simp [h', h]

/-- The counter never exceeds `hi` if it started at or below `hi`. -/
theorem admit_counter_bounded (next hi : Nat) (h : next ≤ hi) : (admitStep next hi).1 ≤ hi := by
  unfold admitStep
  by_cases h1 : next ≥ hi
  · simp [h1] <;> omega
  · simp [h1] <;> omega

-- § aligned slot offsets

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- The aligned size is at least the request. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256
  omega

/-- The aligned size is a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256
  omega

/-- The aligned size wastes less than 256 bytes. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256
  omega

/-- Prefix-sum offsets of aligned slot sizes. -/
def offsets : List Nat → Nat → List Nat
  | [], acc => [acc]
  | s :: rest, acc => acc :: offsets rest (acc + align256 s)

/-- There is one more offset than slots (the final entry is the total). -/
theorem offsets_length (sizes : List Nat) (acc : Nat) : (offsets sizes acc).length = sizes.length + 1 := by
  induction sizes generalizing acc with
  | nil => simp [offsets]
  | cons s rest ih => simp [offsets, ih]

/-- Every offset is a multiple of 256 when the starting accumulator is. -/
theorem offsets_aligned (sizes : List Nat) (acc : Nat) (h : acc % 256 = 0) :
    ∀ o ∈ offsets sizes acc, o % 256 = 0 := by
  induction sizes generalizing acc with
  | nil => intro o ho; simp [offsets] at ho; omega
  | cons s rest ih =>
    intro o ho
    simp only [offsets, List.mem_cons] at ho
    rcases ho with ho | ho
    · omega
    · apply ih (acc + align256 s) _ o ho
      have := align256_mod s
      omega

-- § residency index

/-- Flat residency index. -/
def flatIndex (nExpert layer expert : Nat) : Nat := layer * nExpert + expert

/-- In-range axes give an in-range flat index. -/
theorem flatIndex_lt (nLayers nExpert layer expert : Nat) (hl : layer < nLayers) (he : expert < nExpert) :
    flatIndex nExpert layer expert < nLayers * nExpert := by
  unfold flatIndex
  have : (layer + 1) * nExpert ≤ nLayers * nExpert := Nat.mul_le_mul_right _ hl
  rw [Nat.succ_mul] at this
  omega

/-- The flat index is injective on in-range pairs, so two experts never share a residency cell. -/
theorem flatIndex_injective (nExpert l1 e1 l2 e2 : Nat) (h1 : e1 < nExpert) (h2 : e2 < nExpert)
    (h : flatIndex nExpert l1 e1 = flatIndex nExpert l2 e2) : l1 = l2 ∧ e1 = e2 := by
  unfold flatIndex at h
  have hd1 : (l1 * nExpert + e1) / nExpert = l1 := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega : 0 < nExpert)]
    simp [Nat.div_eq_of_lt h1]
  have hd2 : (l2 * nExpert + e2) / nExpert = l2 := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega : 0 < nExpert)]
    simp [Nat.div_eq_of_lt h2]
  have hl : l1 = l2 := by rw [← hd1, ← hd2, h]
  subst hl
  exact ⟨rfl, by omega⟩

end Strata.ExpertCache
