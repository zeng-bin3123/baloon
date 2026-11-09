/-!
# MODULE: expert_cache_residency
SOURCES: src/core/expert_cache.cpp
Slot arena for the VRAM expert tier: per-layer slot ranges (R4.2g), the no-eviction admit counter,
256-byte aligned variable-size slots, and the clamped fill size.

## Source semantics
* Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so ranges tile `0..slots`.
* `admit` never evicts: it hands out `layer_next` while it is below the layer's `hi`, otherwise refuses.
* `open_sized` aligns each slot to 256 bytes; offsets are cumulative.
* `fill_slot` copies `min`-style `bytes` when `0 < bytes ≤ blob`, else the whole blob.

## Form template identifiers
layer_slot_range → layerLo/layerHi ; admit → admitStep ; open_sized offsets → align256 ;
fill_slot byte count → fillBytes
-/

namespace Strata.ExpertCache

-- § per-layer ranges

/-- Quota per layer. -/
def layerQ (slots nLayers : Nat) : Nat := slots / nLayers

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots nLayers l : Nat) : Nat := l * layerQ slots nLayers

/-- Upper bound (exclusive) of layer `l`'s slot range; the last layer takes the remainder. -/
def layerHi (slots nLayers l : Nat) : Nat :=
  if l = nLayers - 1 then slots else (l + 1) * layerQ slots nLayers

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots n : Nat) (hn : 0 < n) : layerHi slots n (n - 1) = slots := by
  simp [layerHi]

/-- Every range is well formed: `lo ≤ hi`. -/
theorem layerLo_le_layerHi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h1 : layerQ slots n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : l * layerQ slots n ≤ n * layerQ slots n := Nat.mul_le_mul_right _ (by omega)
    have h3 : n * layerQ slots n = layerQ slots n * n := Nat.mul_comm _ _
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Adjacent layers' ranges abut with no gap and no overlap. -/
theorem layerHi_eq_layerLo_succ (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

-- § admit counter (no eviction)

/-- One `admit` step on the per-layer path: `(slot or refusal, new counter)`. -/
def admitStep (next hi : Nat) : Option Nat × Nat :=
  if hi ≤ next then (none, next) else (some next, next + 1)

/-- A full quota refuses and leaves the counter untouched. -/
theorem admitStep_full (next hi : Nat) (h : hi ≤ next) : admitStep next hi = (none, next) := by
  simp [admitStep, h]

/-- A granted slot is below the ceiling and the counter advances by exactly one. -/
theorem admitStep_slot_lt (next hi s next' : Nat) (h : admitStep next hi = (some s, next')) :
    s < hi ∧ next' = s + 1 := by
  unfold admitStep at h
  split at h
  · simp at h
  · simp at h
    omega

/-- The counter never decreases. -/
theorem admitStep_mono (next hi : Nat) : next ≤ (admitStep next hi).2 := by
  unfold admitStep
  split <;> simp

-- § aligned slots

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem le_align256 (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Alignment wastes fewer than 256 bytes. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- A cumulative offset stays 256-aligned and covers the slot just added. -/
theorem offset_step (off s : Nat) (h : off % 256 = 0) :
    (off + align256 s) % 256 = 0 ∧ off + s ≤ off + align256 s := by
  have h1 := align256_mod s
  have h2 := le_align256 s
  omega

-- § fill size

/-- Byte count `fill_slot` copies. -/
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never writes past the slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- With a positive blob, a fill copies at least one byte. -/
theorem fillBytes_pos (bytes blob : Nat) (hb : 0 < blob) : 0 < fillBytes bytes blob := by
  unfold fillBytes
  split <;> omega

end Strata.ExpertCache
