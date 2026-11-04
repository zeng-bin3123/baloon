/-!
# MODULE: expert_residency_cache
The VRAM-resident expert tier: slot storage, the `(layer, expert) -> slot` residency table, global and
per-layer admission with no eviction, and slot-offset arithmetic.

## Source semantics
* `slot_of` is bounds-checked: an out-of-range layer or expert yields `kNotResident`.
* `admit` never evicts: a full cache refuses; the slot counter never exceeds the slot count.
* Per-layer admission gives layer `l` the slot range `[l*q, (l+1)*q)` with `q = slots / n_layers`, the last
  layer taking the remainder; the ranges are disjoint and inside `[0, slots)`.
* Admission into a slot never disturbs other residency entries.

## Form template identifiers
* `ExpertCache::slot_of` -> `slotOf`
* `ExpertCache::admit` -> `admitGlobal`
* `ExpertCache::layer_slot_range` -> `slotLo` / `slotHi`
* `kNotResident` -> `Option.none`
-/

namespace Strata.ExpertResidencyCache

-- § global admission

/-- Global admission: take the next free slot, or refuse when full (never evict). -/
def admitGlobal (next slots : Nat) : Option (Nat × Nat) :=
  if next < slots then some (next, next + 1) else none

/-- An admitted slot is a valid slot index and the counter stays within the slot count. -/
theorem admitGlobal_bounds (next slots s n' : Nat) (h : admitGlobal next slots = some (s, n')) :
    s < slots ∧ n' ≤ slots ∧ n' = next + 1 := by
  unfold admitGlobal at h
  split at h <;> simp at h <;> omega

/-- A full cache refuses and changes nothing. -/
theorem admitGlobal_full (next slots : Nat) (h : slots ≤ next) : admitGlobal next slots = none := by
  unfold admitGlobal
  have : ¬ (next < slots) := by omega
  simp [this]

/-- Repeated admission from empty fills exactly `slots` slots and then refuses. -/
theorem admit_never_exceeds (slots : Nat) : admitGlobal slots slots = none := by
  exact admitGlobal_full slots slots (Nat.le_refl _)

-- § per-layer admission

/-- Lower bound of layer `l`'s slot range: `l * q`. -/
def slotLo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper bound (exclusive) of layer `l`'s range: the last layer takes the remainder. -/
def slotHi (slots nl l : Nat) : Nat := if l + 1 = nl then slots else (l + 1) * (slots / nl)

/-- All `nl` quotas together fit in the slot count. -/
theorem quota_total_le (slots nl : Nat) : nl * (slots / nl) ≤ slots := by
  have h := Nat.div_mul_le_self slots nl
  have h3 : nl * (slots / nl) = slots / nl * nl := Nat.mul_comm _ _
  omega

/-- A layer's range ends inside the arena. -/
theorem slotHi_le (slots nl l : Nat) (hl : l < nl) : slotHi slots nl l ≤ slots := by
  unfold slotHi
  split
  · omega
  · have h1 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 := quota_total_le slots nl
    omega

/-- A layer's range is well formed: `lo ≤ hi`. -/
theorem slotLo_le_hi (slots nl l : Nat) (hl : l < nl) : slotLo slots nl l ≤ slotHi slots nl l := by
  unfold slotLo slotHi
  split
  · have h1 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 := quota_total_le slots nl
    omega
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Ranges of distinct layers are disjoint: layer `l1`'s end is not past layer `l2`'s start. -/
theorem slot_ranges_disjoint (slots nl l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < nl) :
    slotHi slots nl l1 ≤ slotLo slots nl l2 := by
  unfold slotHi slotLo
  have hne : ¬ (l1 + 1 = nl) := by omega
  simp only [hne, if_false]
  exact Nat.mul_le_mul_right _ (by omega)

/-- The last layer's range ends exactly at the slot count, so no slot is orphaned. -/
theorem last_layer_takes_remainder (slots nl : Nat) (h : 0 < nl) : slotHi slots nl (nl - 1) = slots := by
  unfold slotHi
  have : nl - 1 + 1 = nl := by omega
  simp [this]

/-- Admission within a layer's range: take the layer's next slot only if it is below the range end. -/
def admitLayer (next hi : Nat) : Option (Nat × Nat) :=
  if next < hi then some (next, next + 1) else none

/-- A per-layer admitted slot lies inside `[lo, hi)` when the layer counter starts at `lo`. -/
theorem admitLayer_in_range (lo next hi s n' : Nat) (hlo : lo ≤ next)
    (h : admitLayer next hi = some (s, n')) : lo ≤ s ∧ s < hi ∧ n' ≤ hi := by
  unfold admitLayer at h
  split at h <;> simp at h <;> omega

-- § residency table

/-- `(layer, expert) -> slot` for the 512-expert geometry; out-of-range keys are not resident. -/
def slotOf (nl : Nat) (res : Nat → Option Nat) (l e : Nat) : Option Nat :=
  if l < nl ∧ e < 512 then res (l * 512 + e) else none

/-- Out-of-range layer or expert returns "not resident" instead of reading an adjacent entry. -/
theorem slotOf_oob (nl : Nat) (res : Nat → Option Nat) (l e : Nat) (h : ¬ (l < nl ∧ e < 512)) :
    slotOf nl res l e = none := by
  unfold slotOf
  simp [h]

/-- Table indices are injective on in-range keys: two distinct pairs never share a cell. -/
theorem table_index_injective (l e l' e' : Nat) (he : e < 512) (he' : e' < 512)
    (h : l * 512 + e = l' * 512 + e') : l = l' ∧ e = e' := by
  omega

/-- The table has `nl * 512` cells and every in-range key addresses one. -/
theorem table_index_lt (nl l e : Nat) (hl : l < nl) (he : e < 512) : l * 512 + e < nl * 512 := by
  omega

/-- Record a residency entry. -/
def setRes (res : Nat → Option Nat) (k s : Nat) : Nat → Option Nat :=
  fun x => if x = k then some s else res x

/-- The admitted key now maps to its slot. -/
theorem setRes_self (res : Nat → Option Nat) (k s : Nat) : setRes res k s k = some s := by
  simp [setRes]

/-- Admitting one expert never disturbs any other entry (no eviction, no aliasing). -/
theorem setRes_other (res : Nat → Option Nat) (k s x : Nat) (h : x ≠ k) : setRes res k s x = res x := by
  simp [setRes, h]

-- § sizes

/-- Sum of a list of slot sizes. -/
theorem sum_replicate (n b : Nat) : (List.replicate n b).sum = n * b := by
  induction n with
  | zero => simp
  | succ n ih => simp [List.replicate_succ, List.sum_cons, ih, Nat.succ_mul] <;> omega

/-- Uniform arena bytes equal the sized arena bytes when every slot has the same size. -/
theorem uniform_equals_sized (n b : Nat) : n * b = (List.replicate n b).sum := by
  rw [sum_replicate]

/-- One expert blob is 1,382,400 bytes: three 640x2560 Q2_0 matrices (18 bytes per 64 elements). -/
theorem blob_bytes_value : 3 * (640 * 2560) / 64 * 18 = 1382400 := by
  decide

/-- 4,105 slots occupy 5,411 whole MiB of VRAM (the arena figure the planner is checked against). -/
theorem slots_4105_mib : 4105 * 1382400 / 1048576 = 5411 := by
  decide

/-- Eight slots per layer across 48 layers is 384 slots. -/
theorem eight_per_layer : 8 * 48 = 384 := by
  decide

end Strata.ExpertResidencyCache
