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
