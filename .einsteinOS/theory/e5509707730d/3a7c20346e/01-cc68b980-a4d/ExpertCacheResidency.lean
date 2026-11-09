/-!
# MODULE: expert_cache_residency
SOURCES: include/strata/core/expert_cache.hpp, src/core/expert_cache.cpp
VRAM-resident expert tier: slot storage, the (layer, expert) -> slot residency table, the
per-layer admission quotas (R4.2g), the STRP profile reader's validation, and 256-byte slot alignment.

## Source semantics
* `slot_of` is bounds-checked and returns `kNotResident` (-1) out of range.
* `admit` never evicts. Global mode uses one counter `next_free_ <= slots_`; per-layer mode gives layer `l`
  the slot range `[l*q, (l+1)*q)` with `q = slots/n_layers`, the last layer taking the remainder.
* The ranges tile `0..slots` exactly; each layer's cursor never exceeds its upper bound.
* `read_expert_profile` accepts only a header whose geometry matches, `n_ranked <= slots`, and pairs in range.
* `open_sized` aligns each slot to 256 bytes.

## Form template identifiers
ExpertCache -> tier_store; admit -> claim_slot; slot_of -> lookup_slot; layer_slot_range -> quota_range;
kNotResident -> absent_marker; read_expert_profile -> load_static_plan; open_sized -> open_variable_slots.
-/

namespace ExpertCacheResidency

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range: `l * (slots / nl)`. -/
def lo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper bound of layer `l`'s slot range; the last layer takes the remainder. -/
def hi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Each layer's range is well formed: `lo <= hi`. -/
theorem lo_le_hi {slots nl l : Nat} (h : 0 < nl) (hl : l < nl) : lo slots nl l ≤ hi slots nl l := by
  unfold lo hi
  split
  · have h1 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl
    exact Nat.le_trans h1 h2
  · exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- No layer's range extends past the arena: `hi <= slots`. -/
theorem hi_le_slots {slots nl l : Nat} (hl : l < nl) : hi slots nl l ≤ slots := by
  unfold hi
  split
  · exact Nat.le_refl _
  · have h1 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
    have h2 : nl * (slots / nl) ≤ slots := Nat.mul_div_le slots nl
    exact Nat.le_trans h1 h2

/-- Adjacent layers' ranges abut: layer `l`'s upper bound is layer `l+1`'s lower bound. -/
theorem hi_eq_next_lo {slots nl l : Nat} (hl : l + 1 < nl) : hi slots nl l = lo slots nl (l + 1) := by
  have hne : l ≠ nl - 1 := by omega
  simp [hi, lo, hne]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem last_hi {slots nl : Nat} : hi slots nl (nl - 1) = slots := by
  simp [hi]

/-- Sum of the per-layer quotas over the first `m` layers. -/
def quotaSum (slots nl : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => quotaSum slots nl m + (hi slots nl m - lo slots nl m)

/-- Before the last layer, the first `m` quotas sum to `m * q`. -/
theorem quotaSum_prefix {slots nl : Nat} (hnl : 0 < nl) :
    ∀ m, m ≤ nl - 1 → quotaSum slots nl m = m * (slots / nl) := by
  intro m
  induction m with
  | zero => intro _; simp [quotaSum]
  | succ m ih =>
    intro hm
    have hm' : m ≤ nl - 1 := by omega
    have hne : m ≠ nl - 1 := by omega
    have e1 : hi slots nl m = (m + 1) * (slots / nl) := by simp [hi, hne]
    have e2 : lo slots nl m = m * (slots / nl) := rfl
    simp only [quotaSum]
    rw [ih hm', e1, e2, Nat.add_mul, Nat.one_mul, Nat.add_sub_cancel_left]

/-- The quotas of all `nl` layers sum to exactly `slots`: per-layer admission can fill the arena and no more. -/
theorem quotaSum_total {slots nl : Nat} (hnl : 0 < nl) : quotaSum slots nl nl = slots := by
  obtain ⟨n, rfl⟩ : ∃ n, nl = n + 1 := ⟨nl - 1, by omega⟩
  have hp : quotaSum slots (n + 1) n = n * (slots / (n + 1)) :=
    quotaSum_prefix (nl := n + 1) hnl n (by omega)
  have hh : hi slots (n + 1) n = slots := by simp [hi]
  have hl : lo slots (n + 1) n = n * (slots / (n + 1)) := rfl
  have h1 : n * (slots / (n + 1)) ≤ (n + 1) * (slots / (n + 1)) := Nat.mul_le_mul_right _ (Nat.le_succ n)
  have h2 : (n + 1) * (slots / (n + 1)) ≤ slots := Nat.mul_div_le slots (n + 1)
  show quotaSum slots (n + 1) n + (hi slots (n + 1) n - lo slots (n + 1) n) = slots
  rw [hp, hh, hl]
  omega

-- § admission

/-- One admission attempt against a cursor and an upper bound: claim the slot if one is free, else refuse. -/
def admitStep (next hi : Nat) : Nat := if next < hi then next + 1 else next

/-- Admission never pushes the cursor past the bound. -/
theorem admitStep_le {next hi : Nat} (h : next ≤ hi) : admitStep next hi ≤ hi := by
  unfold admitStep
  split <;> omega

/-- A full range refuses: the cursor does not move (no eviction). -/
theorem admitStep_full {next hi : Nat} (h : hi ≤ next) : admitStep next hi = next := by
  unfold admitStep
  split <;> omega

/-- Repeating `k` admission attempts. -/
def admitN : Nat → Nat → Nat → Nat
  | 0, n, _ => n
  | k + 1, n, h => admitN k (admitStep n h) h

/-- However many attempts are made, a cursor that started within its bound stays within it. -/
theorem admitN_le {hi : Nat} : ∀ (k n : Nat), n ≤ hi → admitN k n hi ≤ hi := by
  intro k
  induction k with
  | zero => intro n h; simpa [admitN] using h
  | succ k ih =>
    intro n h
    exact ih _ (admitStep_le h)

-- § why the global counter failed (R4.2g measurement)

/-- 256 slots under global arrival-order admission are consumed inside 26 layers' worth of position 0
(10 experts each), i.e. more than 25 and at most 26 layers. -/
theorem global_policy_confinement : 25 * 10 < 256 ∧ 256 ≤ 26 * 10 := by decide

-- § STRP profile validation

/-- Every ranked pair lies inside the `nl x ne` table. -/
def pairsOk (nl ne : Nat) (ps : List (Nat × Nat)) : Bool :=
  ps.all fun p => decide (p.1 < nl) && decide (p.2 < ne)

/-- A pair in a validated list is in range. -/
theorem pairsOk_mem {nl ne : Nat} {ps : List (Nat × Nat)} (h : pairsOk nl ne ps = true)
    {p : Nat × Nat} (hp : p ∈ ps) : p.1 < nl ∧ p.2 < ne := by
  unfold pairsOk at h
  rw [List.all_eq_true] at h
  simpa using h p hp

/-- Header acceptance: matching geometry and `n_ranked <= slots`. -/
def headerOk (nl ne want nr L E : Nat) : Bool :=
  decide (nl = L) && decide (ne = E) && decide (nr ≤ want)

/-- An accepted header never claims more ranked pairs than slots. -/
theorem headerOk_ranked_le {nl ne want nr L E : Nat} (h : headerOk nl ne want nr L E = true) :
    nr ≤ want := by
  simp [headerOk] at h
  omega

/-- An accepted header has exactly the model's geometry. -/
theorem headerOk_geometry {nl ne want nr L E : Nat} (h : headerOk nl ne want nr L E = true) :
    nl = L ∧ ne = E := by
  simp [headerOk] at h
  omega

-- § byte accounting and alignment

/-- Rounds `n` up to a multiple of 256 (the slot alignment in `open_sized`). -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks a slot. -/
theorem align256_ge (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Aligned slot sizes are multiples of 256, so kernel vector loads start aligned. -/
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by
  unfold align256
  omega

/-- Alignment wastes fewer than 256 bytes per slot. -/
theorem align256_lt (n : Nat) : align256 n < n + 256 := by
  unfold align256
  omega

/-- Uniform-slot arena size: `slots * blob`. -/
def arenaBytes (slots blob : Nat) : Nat := slots * blob

/-- 4,105 slots of 1,382,400 B is no more than the 5,413 MiB the design note quotes. -/
theorem slots4105_fits_note : arenaBytes 4105 1382400 ≤ 5413 * 1048576 := by decide

/-- ... and fits in the 6,414 MiB measured free at steady state. -/
theorem slots4105_fits_free : arenaBytes 4105 1382400 ≤ 6414 * 1048576 := by decide

end ExpertCacheResidency
