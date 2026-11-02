/-!
# MODULE: expert_cache_residency
Slot storage and residency table of the VRAM expert cache: the per-layer slot ranges, the
"no eviction" admission rule, 256-byte slot alignment and the clamped fill length.

## Source semantics
* Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so
  the ranges tile `0..slots` exactly and no slot is orphaned.
* `admit` never evicts: a layer hands out `layer_next` while it is below the layer's `hi`, otherwise it refuses.
* A flat residency index `layer * n_expert + expert` is only meaningful for in-range layer AND expert.
* Variable-size slots are 256-byte aligned; offsets are prefix sums of the aligned sizes.
* A fill copies `bytes` when `0 < bytes ≤ blob`, else the whole `blob`.

## Form template identifiers
* `layer_slot_range` → `layerLo`, `layerHi` (partition of an index range among owners)
* `ExpertCache::admit` → `admitStep` (bump allocator with a ceiling)
* `open_sized` offsets → `align256`, `total`
* `fill_slot` length → `fillLen`
-/

namespace Strata.ExpertCache

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots n l : Nat) : Nat := l * (slots / n)

/-- Upper bound of layer `l`'s slot range; the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat :=
  if l = n - 1 then slots else (l + 1) * (slots / n)

/-- The first layer starts at slot zero. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- Each in-range layer owns a (possibly empty) well-formed interval. -/
theorem layerLo_le_hi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  split
  · have h1 : slots / n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (by omega)
    exact Nat.le_trans h2 (by rw [Nat.mul_comm]; exact h1)
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Adjacent layers' ranges abut: no gap and no overlap. -/
theorem layerHi_eq_lo_succ (slots n l : Nat) (h : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

/-- The last layer's range ends exactly at `slots`: no slot is orphaned. -/
theorem layerHi_last (slots n : Nat) : layerHi slots n (n - 1) = slots := by
  unfold layerHi
  rw [if_pos rfl]

/-- Every layer's upper bound stays within the arena. -/
theorem layerHi_le (slots n l : Nat) (hl : l < n) : layerHi slots n l ≤ slots := by
  unfold layerHi
  split
  · exact Nat.le_refl _
  · have h1 : slots / n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : (l + 1) * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (by omega)
    exact Nat.le_trans h2 (by rw [Nat.mul_comm]; exact h1)

-- § admission without eviction

/-- One admission attempt against a layer ceiling: returns the slot (if any) and the new cursor. -/
def admitStep (next hi : Nat) : Option Nat × Nat :=
  if next ≥ hi then (none, next) else (some next, next + 1)

/-- A granted slot is strictly below the ceiling and the cursor advances by exactly one. -/
theorem admitStep_some (next hi s n' : Nat) (h : admitStep next hi = (some s, n')) :
    s < hi ∧ n' = s + 1 := by
  unfold admitStep at h
  split at h
  · simp at h
  · simp at h
    omega

/-- A full layer refuses and leaves its cursor unchanged. -/
theorem admitStep_full (next hi : Nat) (h : next ≥ hi) : admitStep next hi = (none, next) := by
  unfold admitStep
  rw [if_pos h]

-- § flat residency index

/-- In-range layer and expert give a flat index inside the table. -/
theorem flat_lt (nl ne l e : Nat) (hl : l < nl) (he : e < ne) : l * ne + e < nl * ne := by
  have h := Nat.mul_le_mul_right ne (show l + 1 ≤ nl from hl)
  have h2 : (l + 1) * ne = l * ne + ne := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § slot alignment and fills

/-- Round a byte count up to 256. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (b : Nat) : align256 b % 256 = 0 := by
  unfold align256; omega

/-- Aligning never shrinks a slot. -/
theorem le_align256 (b : Nat) : b ≤ align256 b := by
  unfold align256; omega

/-- Sum of aligned slot sizes (the final offset). -/
def total : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + total bs

/-- Every slot offset is 256-aligned. -/
theorem total_mod (bs : List Nat) : total bs % 256 = 0 := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [total]
    have := align256_mod b
    omega

/-- Clamped fill length. -/
def fillLen (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never exceeds the slot's blob size. -/
theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by
  unfold fillLen
  split
  · omega
  · exact Nat.le_refl _

end Strata.ExpertCache
