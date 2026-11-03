/-!
# MODULE: expert_cache_residency
Slot arena bookkeeping of the VRAM expert cache: the per-layer slot ranges, the no-eviction admission counters
and the flat `(layer, expert)` residency index.

## Source semantics
- Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so the ranges
  tile `0..slots` exactly.
- `admit` never evicts: a slot is handed out only while the counter is below the ceiling (global `slots_`, or the
  layer's own `hi`), and the counter then advances by one.
- The residency table is indexed `layer * n_expert + expert` and both axes are range-checked separately.

## Form template identifiers
- `layerLo`, `layerHi`  <- `ExpertCache::layer_slot_range` (lo, hi)
- `admitGlobal`         <- the shared `next_free_` path of `ExpertCache::admit`
- `admitLayer`          <- the `per_layer_` path (`layer_next_[layer]` against `hi`)
- `flatIdx`             <- `layer * n_expert_ + expert` into `residency_`
-/

namespace Strata.ExpertCache

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper bound (exclusive) of layer `l`'s slot range; the last layer takes the remainder. -/
def layerHi (slots nl l : Nat) : Nat := if l + 1 = nl then slots else (l + 1) * (slots / nl)

/-- Flat index into the residency table. -/
def flatIdx (ne layer expert : Nat) : Nat := layer * ne + expert

/-- Global compulsory-miss admission: returns the slot (if any) and the new counter. -/
def admitGlobal (next slots : Nat) : Option Nat × Nat :=
  if next < slots then (some next, next + 1) else (none, next)

/-- Per-layer admission against the layer's own ceiling `hi`. -/
def admitLayer (nxt hi : Nat) : Option Nat × Nat :=
  if nxt < hi then (some nxt, nxt + 1) else (none, nxt)

/-- The first layer's range starts at slot 0. -/
theorem layerLo_zero (s nl : Nat) : layerLo s nl 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`: no slot is orphaned by the division. -/
theorem layerHi_last (s nl l : Nat) (h : l + 1 = nl) : layerHi s nl l = s := by
  simp [layerHi, h]

/-- Adjacent ranges abut: layer `l`'s upper bound is layer `l+1`'s lower bound. -/
theorem layerHi_succ (s nl l : Nat) (h : l + 1 < nl) :
    layerHi s nl l = layerLo s nl (l + 1) := by
  have hne : ¬ (l + 1 = nl) := by omega
  simp [layerHi, layerLo, hne]

/-- Every in-range layer has a well-formed (possibly empty) range. -/
theorem layerLo_le_hi (s nl l : Nat) (h : l < nl) : layerLo s nl l ≤ layerHi s nl l := by
  unfold layerLo layerHi
  split
  · have h1 := Nat.mul_div_le s nl
    have h2 := Nat.mul_le_mul_right (s / nl) (show l ≤ nl by omega)
    omega
  · exact Nat.mul_le_mul_right _ (by omega)

/-- A slot handed out by the global path is below the slot count. -/
theorem admitGlobal_lt (next slots s : Nat) (h : (admitGlobal next slots).1 = some s) : s < slots := by
  unfold admitGlobal at h
  split at h <;> simp_all

/-- The global counter never passes `slots`. -/
theorem admitGlobal_next_le (next slots : Nat) (h : next ≤ slots) :
    (admitGlobal next slots).2 ≤ slots := by
  unfold admitGlobal
  split <;> simp <;> omega

/-- A slot handed out by the per-layer path stays inside that layer's range. -/
theorem admitLayer_lt (nxt hi s : Nat) (h : (admitLayer nxt hi).1 = some s) : s < hi := by
  unfold admitLayer at h
  split at h <;> simp_all

/-- The per-layer counter never passes the layer's ceiling. -/
theorem admitLayer_next_le (nxt hi : Nat) (h : nxt ≤ hi) : (admitLayer nxt hi).2 ≤ hi := by
  unfold admitLayer
  split <;> simp <;> omega

/-- A full layer admits nothing (no eviction). -/
theorem admitLayer_full (nxt hi : Nat) (h : hi ≤ nxt) : (admitLayer nxt hi).1 = none := by
  unfold admitLayer
  have : ¬ (nxt < hi) := by omega
  simp [this]

/-- In-range `(layer, expert)` pairs land inside the residency table. -/
theorem flatIdx_lt (nl ne layer expert : Nat) (hl : layer < nl) (he : expert < ne) :
    flatIdx ne layer expert < nl * ne := by
  unfold flatIdx
  have h := Nat.mul_le_mul_right ne (Nat.succ_le_of_lt hl)
  simp only [Nat.succ_mul] at h
  omega

/-- The aliasing trap: expert id `ne` of layer 0 has the same flat index as expert 0 of layer 1,
so the expert axis must be validated separately from the flat index. -/
theorem flatIdx_alias (ne : Nat) : flatIdx ne 0 ne = flatIdx ne 1 0 := by
  simp [flatIdx]

end Strata.ExpertCache
