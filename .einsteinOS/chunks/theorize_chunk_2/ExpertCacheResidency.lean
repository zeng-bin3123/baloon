/-!
# MODULE: expert_cache_residency
VRAM-resident expert tier: a slot arena plus a `(layer, expert) -> slot | none` residency table, with
never-evicting admission (global arrival order, or per-layer slot ranges) and bounds-checked lookup.

## Source semantics
* `slot_of` is bounds-checked: out-of-range layer/expert yields `kNotResident`.
* `admit` claims the next free slot from one global counter and returns none when full; it never evicts,
  so every other residency entry is unchanged.
* Per-layer admission: layer `l` may use only `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last
  layer takes the remainder. Ranges are pairwise disjoint and contained in `[0, slots)`.
* Uniform arena size is `slots * blob`; a sized arena's size is its last offset.
* A profile is valid when every ranked `(layer, expert)` pair is in range.

## Form template identifiers
* `ExpertCache` -> `Cache`, `slot_of` -> `slotOf`, `admit` -> `admit`, `kNotResident` -> `none`
* `layer_slot_range` -> `layerLo` / `layerHi`, `next_free_` -> `nextFree`, `bytes()` -> `arenaBytes`
-/

namespace StrataExpertCache

-- § state and lookup

/-- Cache state: capacity, geometry, global admission counter, residency table. -/
structure Cache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  nextFree : Nat
  res : Nat → Nat → Option Nat

/-- Bounds-checked residency lookup. -/
def slotOf (c : Cache) (l e : Nat) : Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert then c.res l e else none

/-- Global admission: take the next free slot, never evict. -/
def admit (c : Cache) (l e : Nat) : Cache × Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots then
    ({ c with nextFree := c.nextFree + 1,
              res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e' },
     some c.nextFree)
  else (c, none)

/-- Out-of-range layer gives no slot. -/
theorem slotOf_layer_oob (c : Cache) (l e : Nat) (h : ¬ l < c.nLayers) :
    slotOf c l e = none := by
  unfold slotOf
  simp [h]

/-- Out-of-range expert gives no slot. -/
theorem slotOf_expert_oob (c : Cache) (l e : Nat) (h : ¬ e < c.nExpert) :
    slotOf c l e = none := by
  unfold slotOf
  simp [h]

-- § admission

/-- A full cache refuses and is left unchanged. -/
theorem admit_full (c : Cache) (l e : Nat) (h : c.slots ≤ c.nextFree) :
    admit c l e = (c, none) := by
  unfold admit
  have : ¬ (l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots) := by omega
  simp [this]

/-- A successful admission returns the counter value as slot. -/
theorem admit_slot (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : (admit c l e).2 = some c.nextFree := by
  unfold admit
  simp [hl, he, hs]

/-- A successful admission advances the counter by one. -/
theorem admit_counter (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : (admit c l e).1.nextFree = c.nextFree + 1 := by
  unfold admit
  simp [hl, he, hs]

/-- The admitted pair becomes resident at the claimed slot. -/
theorem admit_resident (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : slotOf (admit c l e).1 l e = some c.nextFree := by
  unfold admit slotOf
  simp [hl, he, hs]

/-- No eviction: every other pair keeps its slot. -/
theorem admit_no_evict (c : Cache) (l e l' e' : Nat) (h : ¬ (l' = l ∧ e' = e)) :
    slotOf (admit c l e).1 l' e' = slotOf c l' e' := by
  unfold admit slotOf
  by_cases hc : l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots
  · simp [hc, h]
  · simp [hc]

/-- Admission never changes capacity. -/
theorem admit_slots (c : Cache) (l e : Nat) : (admit c l e).1.slots = c.slots := by
  unfold admit
  split <;> rfl

/-- The counter never exceeds capacity once it is within capacity. -/
theorem admit_counter_le (c : Cache) (l e : Nat) (h : c.nextFree ≤ c.slots) :
    (admit c l e).1.nextFree ≤ c.slots := by
  unfold admit
  split
  · rename_i hc
    simp only
    omega
  · exact h

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots n l : Nat) : Nat := l * (slots / n)

/-- Upper bound (exclusive); the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat := if l + 1 = n then slots else (l + 1) * (slots / n)

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- `n * (slots / n) ≤ slots`. -/
theorem n_mul_div_le (slots n : Nat) : n * (slots / n) ≤ slots :=
  Nat.mul_div_le slots n

/-- The upper bound lies within the arena. -/
theorem layerHi_le_slots (slots n l : Nat) (hl : l < n) : layerHi slots n l ≤ slots := by
  unfold layerHi
  split
  · exact Nat.le_refl _
  · rename_i h
    have h1 : l + 1 ≤ n := hl
    have h2 : (l + 1) * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ h1
    have h3 := n_mul_div_le slots n
    omega

/-- Each layer's range is well formed. -/
theorem layerLo_le_layerHi (slots n l : Nat) (hl : l < n) : layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  split
  · rename_i h
    have h2 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
    have h3 := n_mul_div_le slots n
    omega
  · have : l * (slots / n) ≤ (l + 1) * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_succ l)
    exact this

/-- Ranges of distinct layers are disjoint: the lower layer ends before the higher begins. -/
theorem layer_ranges_disjoint (slots n l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < n) :
    layerHi slots n l1 ≤ layerLo slots n l2 := by
  unfold layerHi layerLo
  have hne : ¬ (l1 + 1 = n) := by omega
  simp only [hne, ite_false]
  exact Nat.mul_le_mul_right _ h12

/-- The last layer's range ends exactly at `slots`. -/
theorem last_layer_takes_remainder (slots n : Nat) (hn : 0 < n) : layerHi slots n (n - 1) = slots := by
  unfold layerHi
  have : n - 1 + 1 = n := by omega
  simp [this]

/-- Non-last layers get exactly `q = slots / n` slots. -/
theorem layer_width (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l - layerLo slots n l = slots / n := by
  unfold layerHi layerLo
  have hne : ¬ (l + 1 = n) := by omega
  simp only [hne, ite_false]
  rw [Nat.add_mul, Nat.one_mul]
  exact Nat.add_sub_cancel_left _ _

/-- A per-layer admission decision: the layer's own next-free slot if below its upper bound. -/
def admitPerLayerSlot (slots n : Nat) (next : Nat → Nat) (l : Nat) : Option Nat :=
  if next l < layerHi slots n l then some (next l) else none

/-- A granted per-layer slot lies in the layer's own range. -/
theorem per_layer_slot_in_range (slots n : Nat) (next : Nat → Nat) (l s : Nat)
    (hlo : layerLo slots n l ≤ next l) (h : admitPerLayerSlot slots n next l = some s) :
    layerLo slots n l ≤ s ∧ s < layerHi slots n l := by
  unfold admitPerLayerSlot at h
  split at h
  · rename_i hc
    have hs : next l = s := Option.some.inj h
    subst hs
    exact ⟨hlo, hc⟩
  · cases h

/-- Slots granted to two different layers are different. -/
theorem per_layer_slots_distinct (slots n : Nat) (next : Nat → Nat) (l1 l2 s1 s2 : Nat)
    (h12 : l1 < l2) (h2 : l2 < n)
    (hlo1 : layerLo slots n l1 ≤ next l1) (hlo2 : layerLo slots n l2 ≤ next l2)
    (g1 : admitPerLayerSlot slots n next l1 = some s1)
    (g2 : admitPerLayerSlot slots n next l2 = some s2) : s1 ≠ s2 := by
  have a := per_layer_slot_in_range slots n next l1 s1 hlo1 g1
  have b := per_layer_slot_in_range slots n next l2 s2 hlo2 g2
  have d := layer_ranges_disjoint slots n l1 l2 h12 h2
  omega

-- § sizes and profile

/-- Slot offsets for slots of the given sizes, back to back. -/
def slotOffsets (sizes : List Nat) : List Nat := List.scanl (· + ·) 0 sizes

/-- A sized arena has one more offset than slots. -/
theorem slotOffsets_length (sizes : List Nat) : (slotOffsets sizes).length = sizes.length + 1 := by
  unfold slotOffsets
  simp

/-- Arena bytes: last offset if sized, else `slots * blob`. -/
def arenaBytes (offs : List Nat) (slots blob : Nat) : Nat :=
  match offs.getLast? with
  | none => slots * blob
  | some x => x

/-- Uniform arena size. -/
theorem arenaBytes_uniform (slots blob : Nat) : arenaBytes [] slots blob = slots * blob := by
  rfl

/-- Resident count: the per-layer total when per-layer admission is on, else the global counter. -/
def residentCount (perLayer : Bool) (admitted nextFree : Nat) : Nat :=
  if perLayer then admitted else nextFree

/-- With per-layer admission off, the global counter is the resident count. -/
theorem residentCount_off (a n : Nat) : residentCount false a n = n := by
  rfl

/-- A profile's ranked pairs are all in range. -/
def validProfile (ranked : List (Nat × Nat)) (nLayers nExpert : Nat) : Prop :=
  ∀ p ∈ ranked, p.1 < nLayers ∧ p.2 < nExpert

/-- The empty profile is valid. -/
theorem validProfile_nil (nl ne : Nat) : validProfile [] nl ne := by
  intro p hp
  cases hp

/-- Validity of a cons list splits into head and tail. -/
theorem validProfile_cons (p : Nat × Nat) (ps : List (Nat × Nat)) (nl ne : Nat)
    (h1 : p.1 < nl ∧ p.2 < ne) (h2 : validProfile ps nl ne) : validProfile (p :: ps) nl ne := by
  intro q hq
  cases hq with
  | head => exact h1
  | tail _ hq' => exact h2 q hq'

end StrataExpertCache
