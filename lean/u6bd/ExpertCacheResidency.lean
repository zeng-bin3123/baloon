/-!
# MODULE: expert_cache_residency
SOURCES: src/core/expert_cache.cpp
The VRAM expert cache's bookkeeping: the residency table, compulsory-miss admission with no
eviction (global and per-layer quota variants), the per-layer slot ranges, slot byte addressing
(uniform and 256-byte-aligned sized slots), fill sizing and the `STRP` profile validation.

## Source semantics
* `layer_slot_range`: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer
  takes the remainder, so the ranges tile `0..slots` exactly and are pairwise disjoint.
* `admit` is idempotent on a resident expert, never hands out a slot `>= slots`, and never hands the
  same slot to two `(layer, expert)` pairs (no eviction: the free pointer only moves up).
* Per-layer admission keeps each layer inside its own range, so two layers never share a slot.
* `fill_slot` copies `bytes` if `0 < bytes <= blob`, else `blob` - never more than a slot.
* Sized slots are rounded up to 256 bytes, so every slot start is 256-aligned and holds its blob.
* A profile is accepted only if its geometry matches and every ranked pair is in range.

## Form template identifiers
`ExpertCache::layer_slot_range` → `layerSlotRange`; `ExpertCache::admit` → `admit`;
`layer_next_` → `Quota.next`; `residency_` → `Cache.res`; `next_free_` → `Cache.nextFree`;
`fill_slot` byte count → `fillBytes`; `device_slot` → `deviceSlotOff`;
`open_sized` offsets → `sizedOffsets`; `read_expert_profile` → `profileAccepts`.
-/

namespace Ref.ExpertCacheResidency

-- § Per-layer slot ranges

/-- `ExpertCache::layer_slot_range`: `(lo, hi)`, or `(0, 0)` when out of range. -/
def layerSlotRange (slots nL l : Nat) : Nat × Nat :=
  if nL = 0 ∨ slots = 0 ∨ nL ≤ l then (0, 0)
  else (l * (slots / nL), if l = nL - 1 then slots else (l + 1) * (slots / nL))

/-- Helper: `j` full quotas fit in the arena when `j ≤ nL`. -/
theorem quota_fits {slots nL j : Nat} (hj : j ≤ nL) : j * (slots / nL) ≤ slots := by
  have h1 : j * (slots / nL) ≤ nL * (slots / nL) := Nat.mul_le_mul_right _ hj
  have h2 : nL * (slots / nL) ≤ slots := by
    rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nL
  omega

/-- Lower end of a valid layer's range. -/
theorem range_lo {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l < nL) :
    (layerSlotRange slots nL l).1 = l * (slots / nL) := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ l) := by omega
  simp [layerSlotRange, h]

/-- Upper end of a layer that is not the last one. -/
theorem range_hi_inner {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l + 1 < nL) :
    (layerSlotRange slots nL l).2 = (l + 1) * (slots / nL) := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ l) := by omega
  have h2 : l ≠ nL - 1 := by omega
  simp [layerSlotRange, h, h2]

/-- Upper end of the last layer: it takes every slot left over by the division. -/
theorem range_hi_last {slots nL : Nat} (hn : 0 < nL) (hs : 0 < slots) :
    (layerSlotRange slots nL (nL - 1)).2 = slots := by
  have h : ¬ (nL = 0 ∨ slots = 0 ∨ nL ≤ nL - 1) := by omega
  simp [layerSlotRange, h]

/-- The first layer starts at slot 0. -/
theorem range_first_lo {slots nL : Nat} (hn : 0 < nL) (hs : 0 < slots) :
    (layerSlotRange slots nL 0).1 = 0 := by
  rw [range_lo hn hs hn]; simp

/-- Adjacent ranges meet exactly: no slot is orphaned between two layers. -/
theorem range_adjacent {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l + 1 < nL) :
    (layerSlotRange slots nL l).2 = (layerSlotRange slots nL (l + 1)).1 := by
  rw [range_hi_inner hn hs hl, range_lo hn hs hl]

/-- Every range is well formed and inside the arena. -/
theorem range_bounds {slots nL l : Nat} (hn : 0 < nL) (hs : 0 < slots) (hl : l < nL) :
    (layerSlotRange slots nL l).1 ≤ (layerSlotRange slots nL l).2 ∧
      (layerSlotRange slots nL l).2 ≤ slots := by
  rw [range_lo hn hs hl]
  by_cases hlast : l + 1 < nL
  · rw [range_hi_inner hn hs hlast]
    have h1 : l * (slots / nL) ≤ (l + 1) * (slots / nL) := Nat.mul_le_mul_right _ (by omega)
    exact ⟨h1, quota_fits (by omega)⟩
  · have he : l = nL - 1 := by omega
    subst he
    rw [range_hi_last hn hs]
    exact ⟨quota_fits (by omega), Nat.le_refl _⟩

/-- Ranges of different layers are disjoint: an earlier layer ends before a later one begins. -/
theorem range_disjoint {slots nL l1 l2 : Nat} (hn : 0 < nL) (hs : 0 < slots)
    (h12 : l1 < l2) (h2 : l2 < nL) :
    (layerSlotRange slots nL l1).2 ≤ (layerSlotRange slots nL l2).1 := by
  rw [range_hi_inner hn hs (by omega), range_lo hn hs h2]
  exact Nat.mul_le_mul_right _ h12

-- § Global admission (no eviction)

/-- The residency table and free pointer. -/
structure Cache where
  slots : Nat
  nextFree : Nat
  res : Nat → Nat → Option Nat

/-- Cache invariant: free pointer within the arena, every resident slot below it, and no slot
shared by two pairs. -/
def Cache.Inv (c : Cache) : Prop :=
  c.nextFree ≤ c.slots ∧
  (∀ l e s, c.res l e = some s → s < c.nextFree) ∧
  (∀ l1 e1 l2 e2 s, c.res l1 e1 = some s → c.res l2 e2 = some s → l1 = l2 ∧ e1 = e2)

/-- `ExpertCache::admit` (global path): result slot (`none` = `kNotResident`) and new state. -/
def admit (c : Cache) (l e : Nat) : Option Nat × Cache :=
  match c.res l e with
  | some s => (some s, c)
  | none =>
    if c.nextFree < c.slots then
      (some c.nextFree,
        { c with nextFree := c.nextFree + 1,
                 res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e' })
    else (none, c)

/-- The empty cache satisfies the invariant. -/
theorem empty_inv (slots : Nat) : (Cache.mk slots 0 (fun _ _ => none)).Inv := by
  refine ⟨Nat.zero_le _, ?_, ?_⟩
  · intro l e s h; cases h
  · intro l1 e1 l2 e2 s h; cases h

/-- Admitting a resident expert returns its slot and changes nothing. -/
theorem admit_resident {c : Cache} {l e s : Nat} (h : c.res l e = some s) :
    admit c l e = (some s, c) := by
  simp [admit, h]

/-- A full cache refuses a non-resident expert and changes nothing. -/
theorem admit_full {c : Cache} {l e : Nat} (h : c.res l e = none) (hfull : c.slots ≤ c.nextFree) :
    admit c l e = (none, c) := by
  have : ¬ c.nextFree < c.slots := by omega
  simp [admit, h, this]

/-- After a successful admission the pair is resident at the returned slot. -/
theorem admit_then_resident {c : Cache} {l e s : Nat} (h : (admit c l e).1 = some s) :
    (admit c l e).2.res l e = some s := by
  unfold admit at h ⊢
  split at h
  · rename_i s' hs
    simp only [hs]
    exact (Option.some.inj h) ▸ rfl
  · rename_i hn
    split at h
    · obtain rfl := Option.some.inj h
      simp [hn, *]
    · cases h

/-- Admission preserves the cache invariant: in particular no slot is ever handed out twice. -/
theorem admit_inv {c : Cache} (hc : c.Inv) (l e : Nat) : (admit c l e).2.Inv := by
  obtain ⟨hle, hlt, huniq⟩ := hc
  unfold admit
  split
  · exact ⟨hle, hlt, huniq⟩
  · split
    · rename_i hfree
      refine ⟨by simp only; omega, ?_, ?_⟩
      · intro l' e' s hs
        simp only at hs ⊢
        by_cases h : l' = l ∧ e' = e
        · simp only [h, and_self, if_true] at hs
          obtain rfl := Option.some.inj hs
          omega
        · simp only [h, if_false] at hs
          have := hlt _ _ _ hs
          omega
      · intro l1 e1 l2 e2 s h1 h2
        simp only at h1 h2
        by_cases ha : l1 = l ∧ e1 = e <;> by_cases hb : l2 = l ∧ e2 = e
        · exact ⟨ha.1.trans hb.1.symm, ha.2.trans hb.2.symm⟩
        · simp only [ha, hb, and_self, if_true, if_false] at h1 h2
          obtain rfl := Option.some.inj h1
          have := hlt _ _ _ h2
          omega
        · simp only [ha, hb, and_self, if_true, if_false] at h1 h2
          obtain rfl := Option.some.inj h2
          have := hlt _ _ _ h1
          omega
        · simp only [ha, hb, if_false] at h1 h2
          exact huniq _ _ _ _ _ h1 h2
    · exact ⟨hle, hlt, huniq⟩

/-- Any slot `admit` returns from a well-formed cache is inside the arena. -/
theorem admit_slot_lt {c : Cache} (hc : c.Inv) {l e s : Nat} (h : (admit c l e).1 = some s) :
    s < c.slots := by
  obtain ⟨hle, hlt, _⟩ := hc
  unfold admit at h
  split at h
  · rename_i s' hs
    obtain rfl := Option.some.inj h
    have := hlt _ _ _ hs
    omega
  · split at h
    · rename_i hfree
      obtain rfl := Option.some.inj h
      exact hfree
    · cases h

-- § Per-layer admission (R4.2g)

/-- Per-layer quota pointers. -/
structure Quota where
  slots : Nat
  nL : Nat
  next : Nat → Nat

/-- Quota invariant: each layer's pointer stays in its own range. -/
def Quota.Inv (q : Quota) : Prop :=
  ∀ l, l < q.nL → (layerSlotRange q.slots q.nL l).1 ≤ q.next l ∧
    q.next l ≤ (layerSlotRange q.slots q.nL l).2

/-- Per-layer admission of a non-resident expert: take this layer's next slot if its quota allows. -/
def admitLayer (q : Quota) (l : Nat) : Option Nat × Quota :=
  if q.next l < (layerSlotRange q.slots q.nL l).2 then
    (some (q.next l), { q with next := fun l' => if l' = l then q.next l + 1 else q.next l' })
  else (none, q)

/-- A per-layer admission returns a slot inside that layer's own range. -/
theorem admitLayer_in_range {q : Quota} (hq : q.Inv) {l s : Nat} (hl : l < q.nL)
    (h : (admitLayer q l).1 = some s) :
    (layerSlotRange q.slots q.nL l).1 ≤ s ∧ s < (layerSlotRange q.slots q.nL l).2 := by
  have hb := hq l hl
  unfold admitLayer at h
  split at h
  · rename_i hlt
    obtain rfl := Option.some.inj h
    exact ⟨hb.1, hlt⟩
  · cases h

/-- Per-layer admission preserves the quota invariant. -/
theorem admitLayer_inv {q : Quota} (hq : q.Inv) (l : Nat) : (admitLayer q l).2.Inv := by
  unfold admitLayer
  split
  · rename_i hlt
    intro l' hl'
    have hb := hq l' hl'
    simp only
    by_cases h : l' = l
    · subst h; simp only [if_true]; omega
    · simp only [h, if_false]; exact hb
  · exact hq

/-- Two layers never receive the same slot under per-layer admission. -/
theorem layers_never_share {slots nL l1 l2 s1 s2 : Nat} (hn : 0 < nL) (hs : 0 < slots)
    (h12 : l1 < l2) (h2 : l2 < nL)
    (hs1 : s1 < (layerSlotRange slots nL l1).2) (hs2 : (layerSlotRange slots nL l2).1 ≤ s2) :
    s1 ≠ s2 := by
  have := range_disjoint hn hs h12 h2
  omega

-- § Slot addressing and fills

/-- `fill_slot` byte count: the requested size if it is positive and fits, else a whole slot. -/
def fillBytes (blob bytes : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot. -/
theorem fillBytes_le (blob bytes : Nat) : fillBytes blob bytes ≤ blob := by
  unfold fillBytes; split <;> omega

/-- Uniform `device_slot`: `none` is the null pointer for an out-of-range slot. -/
def deviceSlotOff (slots blob s : Nat) : Option Nat := if s < slots then some (s * blob) else none

/-- A slot returned by `device_slot` lies wholly in the arena. -/
theorem deviceSlot_in_arena {slots blob s o : Nat} (h : deviceSlotOff slots blob s = some o) :
    o + blob ≤ slots * blob := by
  unfold deviceSlotOff at h
  split at h
  · rename_i hs
    obtain rfl := Option.some.inj h
    have h2 := Nat.mul_le_mul_right blob hs
    rw [Nat.succ_mul] at h2
    exact h2
  · cases h

/-- Round a slot size up to 256 bytes. -/
def round256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A rounded slot holds its blob. -/
theorem round256_ge (b : Nat) : b ≤ round256 b := by unfold round256; omega

/-- A rounded slot is a multiple of 256. -/
theorem round256_mod (b : Nat) : round256 b % 256 = 0 := by unfold round256; omega

/-- `open_sized` offsets: a prefix sum of rounded sizes, starting at `start`. -/
def sizedOffsets : List Nat → Nat → List Nat
  | [], o => [o]
  | b :: t, o => o :: sizedOffsets t (o + round256 b)

/-- Every slot start is 256-aligned, as the kernels' vector loads require. -/
theorem sizedOffsets_aligned : ∀ (sizes : List Nat) (o : Nat), o % 256 = 0 →
    ∀ x ∈ sizedOffsets sizes o, x % 256 = 0
  | [], o, ho, x, hx => by
    simp [sizedOffsets] at hx; omega
  | b :: t, o, ho, x, hx => by
    simp only [sizedOffsets, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact ho
    · have hr := round256_mod b
      exact sizedOffsets_aligned t (o + round256 b) (by omega) x hx

-- § Profile validation (`read_expert_profile`)

/-- A ranked pair is inside the model's geometry. -/
def pairOk (nL nE : Nat) (p : Nat × Nat) : Bool := decide (p.1 < nL) && decide (p.2 < nE)

/-- The reader's acceptance test: matching geometry, at most `want` pairs, all in range. -/
def profileAccepts (fileL fileE want : Nat) (ranked : List (Nat × Nat)) (nL nE : Nat) : Bool :=
  (fileL == nL) && (fileE == nE) && decide (ranked.length ≤ want) && ranked.all (pairOk nL nE)

/-- An accepted profile names only in-range pairs. -/
theorem accepts_pairs_in_range {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : profileAccepts fileL fileE want ranked nL nE = true) :
    ∀ p ∈ ranked, p.1 < nL ∧ p.2 < nE := by
  intro p hp
  simp only [profileAccepts, Bool.and_eq_true, List.all_eq_true] at h
  have := h.2 p hp
  simp only [pairOk, Bool.and_eq_true, decide_eq_true_eq] at this
  exact this

/-- An accepted profile never ranks more pairs than it reserves slots. -/
theorem accepts_len {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : profileAccepts fileL fileE want ranked nL nE = true) : ranked.length ≤ want := by
  simp only [profileAccepts, Bool.and_eq_true, decide_eq_true_eq] at h
  exact h.1.2

/-- A profile for a different artifact is refused. -/
theorem refuses_other_geometry {fileL fileE want nL nE : Nat} {ranked : List (Nat × Nat)}
    (h : fileL ≠ nL) : profileAccepts fileL fileE want ranked nL nE = false := by
  simp [profileAccepts, h]

end Ref.ExpertCacheResidency

