/-!
# MODULE: expert_residency_cache
VRAM-resident expert tier: a fixed pool of slots, a (layer, expert) residency table, and a no-eviction
admission policy. Under per-layer admission each layer owns a contiguous slot range.

## Source semantics
* `layer_slot_range(l)` is `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder,
  so the ranges tile `0..slots` exactly.
* `admit` never evicts; it fails (returns `kNotResident`) once the relevant ceiling is reached.
* Out-of-range layer/expert yields `kNotResident`, never an adjacent table entry.

## Form template identifiers
* `slots`      ← `ExpertCache::slots_`
* `nLayers`    ← `ExpertCache::n_layers_`
* `loRange`    ← `lo` out-param of `layer_slot_range`
* `hiRange`    ← `hi` out-param of `layer_slot_range`
* `nextFree`   ← `next_free_`
-/

namespace Strata.ExpertResidencyCache

-- § slot ranges

/-- Quota per layer: `slots_ / n_layers_`. -/
def quota (slots n : Nat) : Nat := slots / n

/-- Lower bound of layer `l`'s range. -/
def loRange (slots n l : Nat) : Nat := l * quota slots n

/-- Upper bound (exclusive) of layer `l`'s range; the last layer absorbs the remainder. -/
def hiRange (slots n l : Nat) : Nat :=
  if l = n - 1 then slots else (l + 1) * quota slots n

/-- `n` layers' quotas never exceed the slot count. -/
theorem n_quota_le (slots n : Nat) : n * quota slots n ≤ slots := by
  unfold quota
  exact Nat.mul_div_le slots n

/-- A layer's range is non-empty-or-empty but well formed: `lo ≤ hi`. -/
theorem lo_le_hi (slots n l : Nat) (hl : l < n) : loRange slots n l ≤ hiRange slots n l := by
  unfold loRange hiRange
  split_ifs with h
  · have h1 : l * quota slots n ≤ n * quota slots n := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
    exact le_trans h1 (n_quota_le slots n)
  · exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Every layer's range stays inside the arena. -/
theorem hi_le_slots (slots n l : Nat) (hl : l < n) : hiRange slots n l ≤ slots := by
  unfold hiRange
  split_ifs with h
  · exact le_rfl
  · have h1 : (l + 1) * quota slots n ≤ n * quota slots n := Nat.mul_le_mul_right _ hl
    exact le_trans h1 (n_quota_le slots n)

/-- Consecutive layers' ranges abut: no slot is skipped. -/
theorem hi_eq_next_lo (slots n l : Nat) (hl : l + 1 < n) :
    hiRange slots n l = loRange slots n (l + 1) := by
  unfold hiRange loRange
  have hne : l ≠ n - 1 := by omega
  rw [if_neg hne]

/-- The last layer's range ends exactly at `slots`: no slot is orphaned by the division. -/
theorem last_hi (slots n : Nat) (hn : 0 < n) : hiRange slots n (n - 1) = slots := by
  unfold hiRange
  rw [if_pos rfl]

/-- The first layer's range starts at zero. -/
theorem first_lo (slots n : Nat) : loRange slots n 0 = 0 := by
  unfold loRange
  simp

/-- Ranges of distinct layers are disjoint: if `l1 < l2 < n`, then `hi l1 ≤ lo l2`. -/
theorem ranges_disjoint (slots n l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < n) :
    hiRange slots n l1 ≤ loRange slots n l2 := by
  have hne : l1 ≠ n - 1 := by omega
  unfold hiRange loRange
  rw [if_neg hne]
  exact Nat.mul_le_mul_right _ h12

/-- Every non-last layer owns exactly `quota` slots. -/
theorem range_size_of_not_last (slots n l : Nat) (hl : l + 1 < n) :
    hiRange slots n l - loRange slots n l = quota slots n := by
  have h := hi_eq_next_lo slots n l hl
  have hne : l ≠ n - 1 := by omega
  unfold hiRange loRange at *
  rw [if_neg hne]
  rw [Nat.add_mul, one_mul]
  omega

-- § admission

/-- Global admission: a slot is handed out iff the counter is below `slots`. -/
def admitGlobal (slots nextFree : Nat) : Option Nat :=
  if nextFree < slots then some nextFree else none

/-- A global admission returns an in-range slot. -/
theorem admitGlobal_lt (slots nextFree s : Nat) (h : admitGlobal slots nextFree = some s) : s < slots := by
  unfold admitGlobal at h
  split_ifs at h with hlt
  · cases h
    exact hlt

/-- A full cache refuses: it never evicts. -/
theorem admitGlobal_full (slots nextFree : Nat) (h : slots ≤ nextFree) :
    admitGlobal slots nextFree = none := by
  unfold admitGlobal
  rw [if_neg (by omega)]

/-- Per-layer admission: layer `l` hands out `next` iff `next < hi(l)`. -/
def admitLayer (slots n l next : Nat) : Option Nat :=
  if next < hiRange slots n l then some next else none

/-- A per-layer admission whose cursor sits in the layer's range returns a slot in that range. -/
theorem admitLayer_in_range (slots n l next s : Nat) (hlo : loRange slots n l ≤ next)
    (h : admitLayer slots n l next = some s) :
    loRange slots n l ≤ s ∧ s < hiRange slots n l := by
  unfold admitLayer at h
  split_ifs at h with hlt
  · obtain rfl : next = s := by simpa using h
    exact ⟨hlo, hlt⟩

/-- Per-layer admission never produces a slot owned by a different layer. -/
theorem admitLayer_not_other (slots n l1 l2 next s : Nat) (h12 : l1 < l2) (h2 : l2 < n)
    (hlo : loRange slots n l1 ≤ next) (h : admitLayer slots n l1 next = some s) :
    s < loRange slots n l2 := by
  obtain ⟨_, hs⟩ := admitLayer_in_range slots n l1 next s hlo h
  exact lt_of_lt_of_le hs (ranges_disjoint slots n l1 l2 h12 h2)

end Strata.ExpertResidencyCache
