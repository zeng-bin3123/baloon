/-!
# MODULE: expert_residency_cache
SOURCES: include/strata/core/expert_cache.hpp, src/core/expert_cache.cpp
The VRAM-resident expert tier: a slot arena plus a `(layer, expert) -> slot or -1` residency table. Admission never
evicts; an optional per-layer mode confines each layer to its own contiguous slot range.

## Source semantics
* `admit` is idempotent on resident pairs, refuses (returns -1) when no slot is free, and NEVER evicts or moves an
  already-resident pair. Distinct pairs get distinct slots.
* Per-layer admission: layer `l` may only use `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the
  remainder, so the ranges tile `0..slots` and no two layers share a slot.
* `fill_slot` copies `bytes` when `0 < bytes <= blob`, else the whole slot. Sized slots are 256-byte aligned.
* The profile reader accepts only a header whose geometry matches, with `n_ranked <= slots`, and in-range pairs.

## Form template identifiers
ExpertCache::admit -> admit ; ExpertCache::slot_of -> lookup ; next_free_ -> nextFree ; layer_next_ -> layerNext ;
layer_slot_range -> layerLo / layerHi ; residency_ -> res ; kNotResident -> none (toInt none = -1)
-/

namespace Strata.ExpertCache

-- § slot encoding

/-- The C++ encoding of an optional slot: `-1` for not resident. -/
def toInt : Option Nat → Int
  | none => -1
  | some n => (n : Int)

/-- `kNotResident` (-1) is exactly the `none` case. -/
theorem toInt_eq_neg_one_iff (o : Option Nat) : toInt o = -1 ↔ o = none := by
  cases o with
  | none => exact ⟨fun _ => rfl, fun _ => rfl⟩
  | some n => exact ⟨fun h => by have h' : (n : Int) = -1 := h; omega, fun h => by cases h⟩

-- § per-layer slot ranges

/-- `q = slots / n_layers`. -/
def layerQ (slots n : Nat) : Nat := slots / n
/-- Lower bound of layer `l`'s range. -/
def layerLo (slots n l : Nat) : Nat := l * layerQ slots n
/-- Upper bound (exclusive); the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat := if l = n - 1 then slots else (l + 1) * layerQ slots n

/-- Layer 0's range starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots n : Nat) : layerHi slots n (n - 1) = slots := by
  simp [layerHi]

/-- Each range is well formed. -/
theorem layerLo_le_hi (slots n l : Nat) (hl : l < n) : layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi layerQ
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h1 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
    have h2 : n * (slots / n) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots n
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Consecutive ranges abut: the ranges tile with no gap. -/
theorem layerHi_eq_next_lo (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  have h : ¬ l = n - 1 := by omega
  rw [if_neg h]

/-- Ranges of different layers do not overlap: this is why per-layer admission cannot share a slot. -/
theorem ranges_disjoint (slots n l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < n) :
    layerHi slots n l1 ≤ layerLo slots n l2 := by
  unfold layerHi layerLo
  have h : ¬ l1 = n - 1 := by omega
  rw [if_neg h]
  exact Nat.mul_le_mul_right _ h12

/-- Every layer's quota is at least `q` wide. -/
theorem layer_width_ge_quota (slots n l : Nat) (hl : l < n) :
    layerLo slots n l + layerQ slots n ≤ layerHi slots n l := by
  unfold layerLo layerHi layerQ
  have e : (l + 1) * (slots / n) = l * (slots / n) + slots / n := by
    rw [Nat.add_mul, Nat.one_mul]
  by_cases h : l = n - 1
  · rw [if_pos h]
    have h2 : n * (slots / n) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots n
    have h3 : (l + 1) * (slots / n) ≤ slots := by
      have : l + 1 = n := by omega
      rw [this]; exact h2
    omega
  · rw [if_neg h]
    omega

-- § global admission (no eviction)

/-- The cache state relevant to admission. -/
structure Cache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  res : Nat → Nat → Option Nat
  nextFree : Nat

/-- `slot_of`: bounds-checked lookup. -/
def Cache.lookup (c : Cache) (l e : Nat) : Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert then c.res l e else none

/-- A new slot is claimed exactly when in bounds, not resident, and a slot is free. -/
def Cache.canAdmit (c : Cache) (l e : Nat) : Bool :=
  decide (l < c.nLayers) && decide (e < c.nExpert) && (c.res l e).isNone && decide (c.nextFree < c.slots)

/-- `admit`: returns the slot (or none) and the new state. -/
def Cache.admit (c : Cache) (l e : Nat) : Option Nat × Cache :=
  if c.canAdmit l e = true then
    (some c.nextFree,
      { c with res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e',
               nextFree := c.nextFree + 1 })
  else (c.lookup l e, c)

/-- Characterisation of `canAdmit`. -/
theorem canAdmit_iff (c : Cache) (l e : Nat) :
    c.canAdmit l e = true ↔ ((l < c.nLayers ∧ e < c.nExpert) ∧ c.res l e = none ∧ c.nextFree < c.slots) := by
  unfold Cache.canAdmit
  cases hr : c.res l e <;> simp [hr, and_assoc]

/-- The fields of the post-state when a slot is claimed. -/
theorem admit_new (c : Cache) (l e : Nat) (hc : c.canAdmit l e = true) :
    (c.admit l e).2.nextFree = c.nextFree + 1 ∧
    (∀ l' e', (c.admit l e).2.res l' e' = if l' = l ∧ e' = e then some c.nextFree else c.res l' e') ∧
    (c.admit l e).2.slots = c.slots ∧ (c.admit l e).2.nLayers = c.nLayers ∧
    (c.admit l e).2.nExpert = c.nExpert := by
  unfold Cache.admit
  rw [if_pos hc]
  exact ⟨rfl, fun _ _ => rfl, rfl, rfl, rfl⟩

/-- When nothing is claimed the state is untouched. -/
theorem admit_not (c : Cache) (l e : Nat) (hc : c.canAdmit l e = false) :
    c.admit l e = (c.lookup l e, c) := by
  unfold Cache.admit
  have hn : ¬ (c.canAdmit l e = true) := by simp [hc]
  rw [if_neg hn]

/-- Invariants: the counter is within the arena, resident slots are below the counter, slots are injective. -/
def Cache.Inv (c : Cache) : Prop :=
  c.nextFree ≤ c.slots ∧
  (∀ l e s, c.res l e = some s → s < c.nextFree) ∧
  (∀ l e l' e' s, c.res l e = some s → c.res l' e' = some s → l = l' ∧ e = e')

/-- Admission preserves the invariants (no slot is handed out twice, none beyond the arena). -/
theorem admit_inv (c : Cache) (l e : Nat) (h : c.Inv) : (c.admit l e).2.Inv := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨_, hnone, hlt⟩ := (canAdmit_iff c l e).1 hc
    obtain ⟨hnf, hres, hs, _, _⟩ := admit_new c l e hc
    obtain ⟨h1, h2, h3⟩ := h
    refine ⟨?_, ?_, ?_⟩
    · rw [hnf, hs]; omega
    · intro l' e' s hs'
      rw [hres] at hs'
      rw [hnf]
      by_cases hh : l' = l ∧ e' = e
      · rw [if_pos hh] at hs'
        have := Option.some.inj hs'
        omega
      · rw [if_neg hh] at hs'
        have := h2 l' e' s hs'
        omega
    · intro a b a' b' s ha hb
      rw [hres] at ha hb
      by_cases k1 : a = l ∧ b = e
      · by_cases k2 : a' = l ∧ b' = e
        · exact ⟨k1.1.trans k2.1.symm, k1.2.trans k2.2.symm⟩
        · rw [if_pos k1] at ha
          rw [if_neg k2] at hb
          have e1 := Option.some.inj ha
          have e2 := h2 a' b' s hb
          omega
      · by_cases k2 : a' = l ∧ b' = e
        · rw [if_neg k1] at ha
          rw [if_pos k2] at hb
          have e1 := Option.some.inj hb
          have e2 := h2 a b s ha
          omega
        · rw [if_neg k1] at ha
          rw [if_neg k2] at hb
          exact h3 a b a' b' s ha hb
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc']
    exact h

/-- A resident, in-bounds pair is returned as is and the state does not change (admit is idempotent). -/
theorem admit_resident (c : Cache) (l e s : Nat) (hb : l < c.nLayers ∧ e < c.nExpert)
    (hr : c.res l e = some s) : c.admit l e = (some s, c) := by
  have hc' : c.canAdmit l e = false := by
    cases hh : c.canAdmit l e
    · rfl
    · have := ((canAdmit_iff c l e).1 hh).2.1
      rw [hr] at this
      cases this
  rw [admit_not c l e hc']
  simp [Cache.lookup, hb, hr]

/-- A full cache never admits, and never changes: there is no eviction. -/
theorem admit_full (c : Cache) (l e : Nat) (hfull : c.slots ≤ c.nextFree) :
    (c.admit l e).2 = c := by
  have hc' : c.canAdmit l e = false := by
    cases hh : c.canAdmit l e
    · rfl
    · have := ((canAdmit_iff c l e).1 hh).2.2
      omega
  rw [admit_not c l e hc']

/-- Any slot returned lies inside the arena. -/
theorem admit_slot_lt (c : Cache) (l e s : Nat) (h : c.Inv) (hs : (c.admit l e).1 = some s) :
    s < c.slots := by
  by_cases hc : c.canAdmit l e = true
  · have h1 := ((canAdmit_iff c l e).1 hc).2.2
    unfold Cache.admit at hs
    rw [if_pos hc] at hs
    have := Option.some.inj hs
    omega
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc'] at hs
    have hs' : c.lookup l e = some s := hs
    unfold Cache.lookup at hs'
    by_cases hb : l < c.nLayers ∧ e < c.nExpert
    · rw [if_pos hb] at hs'
      obtain ⟨h1, h2, _⟩ := h
      have := h2 l e s hs'
      omega
    · rw [if_neg hb] at hs'
      cases hs'

/-- NO EVICTION: every pair resident before an admission is resident in the same slot after it. -/
theorem admit_no_evict (c : Cache) (l e l' e' s : Nat) (h : c.res l' e' = some s) :
    (c.admit l e).2.res l' e' = some s := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨hnf, hres, _⟩ := admit_new c l e hc
    have hnone := ((canAdmit_iff c l e).1 hc).2.1
    rw [hres]
    by_cases hh : l' = l ∧ e' = e
    · obtain ⟨h1, h2⟩ := hh
      subst h1; subst h2
      rw [hnone] at h
      cases h
    · rw [if_neg hh]; exact h
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc']
    exact h

/-- After an admission that returned a slot, the pair is resident in exactly that slot. -/
theorem admit_then_resident (c : Cache) (l e s : Nat) (hs : (c.admit l e).1 = some s) :
    (c.admit l e).2.lookup l e = some s := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨_, hres, _, hL, hE⟩ := admit_new c l e hc
    have hb := ((canAdmit_iff c l e).1 hc).1
    unfold Cache.admit at hs
    rw [if_pos hc] at hs
    have hs2 := Option.some.inj hs
    unfold Cache.lookup
    rw [hL, hE, if_pos hb, hres, if_pos ⟨rfl, rfl⟩, hs2]
  · have hc' : c.canAdmit l e = false := by simpa using hc
    rw [admit_not c l e hc'] at hs ⊢
    exact hs

-- § per-layer admission (R4.2g)

/-- Cache state under per-layer admission. -/
structure LCache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  res : Nat → Nat → Option Nat
  layerNext : Nat → Nat

/-- Claim a slot only when in bounds, not resident and the layer's own quota is not full. -/
def LCache.canAdmit (c : LCache) (l e : Nat) : Bool :=
  decide (l < c.nLayers) && decide (e < c.nExpert) && (c.res l e).isNone &&
    decide (c.layerNext l < layerHi c.slots c.nLayers l)

/-- Per-layer `admit`. -/
def LCache.admit (c : LCache) (l e : Nat) : Option Nat × LCache :=
  if c.canAdmit l e = true then
    (some (c.layerNext l),
      { c with res := fun l' e' => if l' = l ∧ e' = e then some (c.layerNext l) else c.res l' e',
               layerNext := fun l' => if l' = l then c.layerNext l + 1 else c.layerNext l' })
  else (if l < c.nLayers ∧ e < c.nExpert then c.res l e else none, c)

/-- Characterisation of the per-layer `canAdmit`. -/
theorem LCache.canAdmit_iff (c : LCache) (l e : Nat) :
    c.canAdmit l e = true ↔
      ((l < c.nLayers ∧ e < c.nExpert) ∧ c.res l e = none ∧
        c.layerNext l < layerHi c.slots c.nLayers l) := by
  unfold LCache.canAdmit
  cases hr : c.res l e <;> simp [hr, and_assoc]

/-- Post-state fields when a slot is claimed per-layer. -/
theorem LCache.admit_new (c : LCache) (l e : Nat) (hc : c.canAdmit l e = true) :
    (∀ l', (c.admit l e).2.layerNext l' = if l' = l then c.layerNext l + 1 else c.layerNext l') ∧
    (∀ l' e', (c.admit l e).2.res l' e' =
        if l' = l ∧ e' = e then some (c.layerNext l) else c.res l' e') ∧
    (c.admit l e).2.slots = c.slots ∧ (c.admit l e).2.nLayers = c.nLayers ∧
    (c.admit l e).1 = some (c.layerNext l) := by
  unfold LCache.admit
  rw [if_pos hc]
  exact ⟨fun _ => rfl, fun _ _ => rfl, rfl, rfl, rfl⟩

/-- Per-layer invariants: counters stay inside the layer's range; resident slots lie in their own layer's range. -/
def LCache.Inv (c : LCache) : Prop :=
  (∀ l, l < c.nLayers → layerLo c.slots c.nLayers l ≤ c.layerNext l ∧
      c.layerNext l ≤ layerHi c.slots c.nLayers l) ∧
  (∀ l e s, c.res l e = some s →
      l < c.nLayers ∧ layerLo c.slots c.nLayers l ≤ s ∧ s < c.layerNext l)

/-- A claimed slot lies inside the claiming layer's range. -/
theorem LCache.admit_slot_in_range (c : LCache) (l e s : Nat) (h : c.Inv)
    (hc : c.canAdmit l e = true) (hs : (c.admit l e).1 = some s) :
    layerLo c.slots c.nLayers l ≤ s ∧ s < layerHi c.slots c.nLayers l := by
  obtain ⟨_, _, _, _, h5⟩ := LCache.admit_new c l e hc
  rw [h5] at hs
  have hs2 := Option.some.inj hs
  obtain ⟨hb, _, hlt⟩ := (LCache.canAdmit_iff c l e).1 hc
  have := h.1 l hb.1
  omega

/-- Per-layer admission preserves the invariants. -/
theorem LCache.admit_inv (c : LCache) (l e : Nat) (h : c.Inv) : (c.admit l e).2.Inv := by
  by_cases hc : c.canAdmit l e = true
  · obtain ⟨hnx, hres, hs, hn, _⟩ := LCache.admit_new c l e hc
    obtain ⟨hb, hnone, hlt⟩ := (LCache.canAdmit_iff c l e).1 hc
    obtain ⟨h1, h2⟩ := h
    refine ⟨?_, ?_⟩
    · intro l' hl'
      rw [hn] at hl'
      rw [hs, hn, hnx l']
      by_cases hh : l' = l
      · subst hh
        rw [if_pos rfl]
        have := h1 l' hb.1
        omega
      · rw [if_neg hh]
        exact h1 l' hl'
    · intro a b s hab
      rw [hres] at hab
      rw [hs, hn, hnx a]
      by_cases k : a = l ∧ b = e
      · rw [if_pos k] at hab
        have e1 := Option.some.inj hab
        obtain ⟨ka, _⟩ := k
        subst ka
        rw [if_pos rfl]
        have := h1 a hb.1
        omega
      · rw [if_neg k] at hab
        obtain ⟨p1, p2, p3⟩ := h2 a b s hab
        by_cases ka : a = l
        · rw [if_pos ka]
          subst ka
          exact ⟨p1, p2, by omega⟩
        · rw [if_neg ka]
          exact ⟨p1, p2, p3⟩
  · have hc' : c.canAdmit l e = false := by simpa using hc
    have : (c.admit l e).2 = c := by
      unfold LCache.admit
      have hn : ¬ (c.canAdmit l e = true) := by simp [hc']
      rw [if_neg hn]
    rw [this]
    exact h

/-- TWO LAYERS NEVER SHARE A SLOT: the same slot in two resident pairs forces the same layer. -/
theorem LCache.no_cross_layer_sharing (c : LCache) (h : c.Inv) (l e l' e' s : Nat)
    (h1 : c.res l e = some s) (h2 : c.res l' e' = some s) : l = l' := by
  obtain ⟨hl, lo1, hi1⟩ := h.2 l e s h1
  obtain ⟨hl', lo2, hi2⟩ := h.2 l' e' s h2
  rcases Nat.lt_trichotomy l l' with hlt | heq | hgt
  · have d := ranges_disjoint c.slots c.nLayers l l' hlt hl'
    have a := (h.1 l hl).2
    omega
  · exact heq
  · have d := ranges_disjoint c.slots c.nLayers l' l hgt hl
    have a := (h.1 l' hl').2
    omega

-- § byte arithmetic of slots

/-- Slot sizes are rounded up to 256 bytes. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounded sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Rounding adds less than 256 bytes. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

/-- Total arena size of a list of sized slots. -/
def totalBytes : List Nat → Nat
  | [] => 0
  | b :: bs => alignUp256 b + totalBytes bs

/-- Sum of the raw sizes. -/
def sumBytes : List Nat → Nat
  | [] => 0
  | b :: bs => b + sumBytes bs

/-- The arena holds at least the raw bytes of every slot. -/
theorem totalBytes_ge (bs : List Nat) : sumBytes bs ≤ totalBytes bs := by
  induction bs with
  | nil => simp [sumBytes, totalBytes]
  | cons b bs ih =>
    simp only [sumBytes, totalBytes]
    have := alignUp256_ge b
    omega

/-- The byte count `fill_slot` copies: the blob's own size if it fits the slot, otherwise the whole slot. -/
def fillBytes (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never writes past the slot. -/
theorem fillBytes_le (bytes blob : Nat) : fillBytes bytes blob ≤ blob := by
  unfold fillBytes
  split <;> omega

/-- `bytes = 0` means the whole slot. -/
theorem fillBytes_zero (blob : Nat) : fillBytes 0 blob = blob := by
  simp [fillBytes]

/-- The VRAM check: refuse when free memory is below what was requested. -/
def vramFits (free want : Nat) : Bool := decide (want ≤ free)

/-- Refusal happens exactly when the request exceeds free memory. -/
theorem vramFits_false_iff (free want : Nat) : vramFits free want = false ↔ free < want := by
  unfold vramFits
  simp

-- § profile reader

/-- A ranked pair is in range for the model. -/
def pairValid (nl ne : Nat) (p : Nat × Nat) : Bool := decide (p.1 < nl) && decide (p.2 < ne)

/-- Header + payload acceptance of `read_expert_profile`. -/
def profileAccepts (hdrNl hdrNe want nRanked nl ne : Nat) (pairs : List (Nat × Nat)) : Bool :=
  decide (hdrNl = nl) && decide (hdrNe = ne) && decide (nRanked ≤ want) &&
    decide (pairs.length = nRanked) && pairs.all (pairValid nl ne)

/-- An accepted profile matches the model, has no more pairs than slots, and every pair is in range. -/
theorem profileAccepts_sound (hdrNl hdrNe want nRanked nl ne : Nat) (pairs : List (Nat × Nat))
    (h : profileAccepts hdrNl hdrNe want nRanked nl ne pairs = true) :
    hdrNl = nl ∧ hdrNe = ne ∧ nRanked ≤ want ∧ ∀ p ∈ pairs, p.1 < nl ∧ p.2 < ne := by
  simp only [profileAccepts, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, _⟩, h5⟩ := h
  refine ⟨h1, h2, h3, fun p hp => ?_⟩
  have := h5 p hp
  simp only [pairValid, Bool.and_eq_true, decide_eq_true_eq] at this
  exact this

end Strata.ExpertCache
