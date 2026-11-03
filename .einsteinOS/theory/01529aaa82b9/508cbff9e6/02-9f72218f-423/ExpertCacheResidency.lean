/-!
# MODULE: expert_cache_residency
VRAM expert tier: a (layer, expert) -> slot residency table, a no-eviction admission counter, and the
per-layer slot ranges of R4.2g. Also the 256-byte slot alignment of `open_sized`.

## Source semantics
* `admit` never evicts: a resident pair keeps its slot, a full cache answers "not resident".
* Per-layer admission: layer `l` may only use `[l*q, (l+1)*q)` with `q = slots / n_layers`, the last layer taking
  the remainder, so the ranges tile `0..slots` exactly.
* `slot_of` is bounds-checked: layer and expert are separate axes, the flat index is `layer*n_expert + expert`.
* `open_sized` rounds each slot up to 256 bytes, so the arena is at least the sum of the blob sizes.

## Form template identifiers
* `ExpertCache::admit` -> `admitBounded` / `admitResident`
* `ExpertCache::layer_slot_range` -> `layerLo` / `layerHi`
* `ExpertCache::slot_of` flat index -> `layer * nExpert + expert`
* `open_sized` offset rounding -> `align256`, `totalAligned`
-/

namespace Strata.ExpertCache

-- § per-layer slot ranges

/-- `q = slots / n_layers`, the quota of each layer under per-layer admission. -/
def perLayerQuota (slots nLayers : Nat) : Nat := slots / nLayers

/-- Lower end of layer `layer`'s slot range. -/
def layerLo (slots nLayers layer : Nat) : Nat := layer * perLayerQuota slots nLayers

/-- Upper end (exclusive) of layer `layer`'s range; the last layer takes the remainder. -/
def layerHi (slots nLayers layer : Nat) : Nat :=
  if layer + 1 = nLayers then slots else (layer + 1) * perLayerQuota slots nLayers

/-- `q * n_layers` never exceeds the slot count. -/
theorem quota_mul_le (slots n : Nat) : perLayerQuota slots n * n ≤ slots :=
  Nat.div_mul_le_self slots n

/-- Helper: `layer * q ≤ slots` whenever `layer ≤ n_layers`. -/
theorem lo_le_slots (slots n layer : Nat) (h : layer ≤ n) : layerLo slots n layer ≤ slots := by
  unfold layerLo
  have h1 := Nat.mul_le_mul_right (perLayerQuota slots n) h
  have h2 := quota_mul_le slots n
  have h3 : n * perLayerQuota slots n = perLayerQuota slots n * n := Nat.mul_comm _ _
  omega

/-- Each layer's range is well formed: `lo ≤ hi`. -/
theorem lo_le_hi (slots n layer : Nat) (h : layer < n) :
    layerLo slots n layer ≤ layerHi slots n layer := by
  unfold layerHi
  split
  · exact lo_le_slots slots n layer (Nat.le_of_lt h)
  · unfold layerLo
    exact Nat.mul_le_mul_right _ (Nat.le_succ layer)

/-- Each layer's range stays inside the arena. -/
theorem hi_le_slots (slots n layer : Nat) (h : layer < n) : layerHi slots n layer ≤ slots := by
  unfold layerHi
  split
  · exact Nat.le_refl _
  · have hl : layer + 1 ≤ n := h
    have h1 := Nat.mul_le_mul_right (perLayerQuota slots n) hl
    have h2 := quota_mul_le slots n
    have h3 : n * perLayerQuota slots n = perLayerQuota slots n * n := Nat.mul_comm _ _
    omega

/-- Consecutive ranges abut: no slot is orphaned between layers. -/
theorem hi_eq_next_lo (slots n layer : Nat) (h : layer + 1 < n) :
    layerHi slots n layer = layerLo slots n (layer + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

/-- The last layer's range ends exactly at `slots`. -/
theorem last_hi (slots n : Nat) (h : 0 < n) : layerHi slots n (n - 1) = slots := by
  unfold layerHi
  rw [if_pos (by omega)]

/-- The first layer's range starts at 0. -/
theorem first_lo (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- Distinct layers' ranges never overlap. -/
theorem ranges_disjoint (slots n a b : Nat) (hab : a < b) (hb : b < n) :
    layerHi slots n a ≤ layerLo slots n b := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]
  exact Nat.mul_le_mul_right _ hab

-- § admission

/-- Claim the slot `cur` if it is below the ceiling `hi`; the new counter is `cur + 1`. -/
def admitBounded (cur hi : Nat) : Option (Nat × Nat) :=
  if hi ≤ cur then none else some (cur, cur + 1)

/-- A granted slot lies in `[lo, hi)` when the counter started at or above `lo`; the counter advances by one. -/
theorem admitBounded_spec {cur lo hi s cur' : Nat} (hlo : lo ≤ cur)
    (h : admitBounded cur hi = some (s, cur')) : lo ≤ s ∧ s < hi ∧ cur' = s + 1 := by
  by_cases hc : hi ≤ cur
  · simp [admitBounded, hc] at h
  · simp [admitBounded, hc] at h
    omega

/-- Admission is refused exactly when the quota (or the whole cache) is full. -/
theorem admitBounded_none_iff (cur hi : Nat) : admitBounded cur hi = none ↔ hi ≤ cur := by
  unfold admitBounded
  split <;> simp_all

/-- No eviction: an already resident pair keeps its slot whatever the counter says. -/
def admitResident (r : Option Nat) (fresh : Option Nat) : Option Nat :=
  match r with
  | some s => some s
  | none => fresh

/-- A resident pair is returned unchanged. -/
theorem admitResident_keeps {r : Option Nat} {s : Nat} (fresh : Option Nat) (h : r = some s) :
    admitResident r fresh = some s := by
  simp [admitResident, h]

-- § residency table indexing

/-- The flat index of an in-range pair stays inside the table. -/
theorem flat_index_lt (n ne layer expert : Nat) (hl : layer < n) (he : expert < ne) :
    layer * ne + expert < n * ne := by
  have hl' : layer + 1 ≤ n := hl
  have h2 := Nat.mul_le_mul_right ne hl'
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

-- § slot alignment (open_sized)

/-- Round a byte count up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Rounding never shrinks a slot. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Rounded sizes are multiples of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Rounding wastes fewer than 256 bytes. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- Sum of raw blob sizes. -/
def totalRaw : List Nat → Nat
  | [] => 0
  | b :: bs => b + totalRaw bs

/-- Sum of 256-aligned slot sizes (the arena size). -/
def totalAligned : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + totalAligned bs

/-- The arena holds every blob: aligned total is at least the raw total. -/
theorem totalAligned_ge (bs : List Nat) : totalRaw bs ≤ totalAligned bs := by
  induction bs with
  | nil => simp [totalRaw, totalAligned]
  | cons b bs ih =>
    simp only [totalRaw, totalAligned]
    have := align256_ge b
    omega

end Strata.ExpertCache
