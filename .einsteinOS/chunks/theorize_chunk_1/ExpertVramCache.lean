import Mathlib

/-!
# MODULE: expert_vram_cache
The VRAM-resident expert tier: a slot arena plus a `(layer, expert) -> slot | -1` residency table, with a
no-eviction admission policy (global counter, or per-layer quota ranges), 256-byte-aligned sized slots, and a
validated profile header.

## Source semantics
* `slot_of` is bounds-checked; out-of-range returns `kNotResident` (-1).
* `admit` never evicts: when the counter reaches its ceiling it returns `kNotResident` and changes nothing.
* Per-layer admission: layer `l` owns `[l*q, (l+1)*q)`, `q = slots / n_layers`; the last layer takes the
  remainder so the ranges tile `0..slots` exactly.
* Sized slots are rounded up to 256 bytes; `open_sized` builds `slots + 1` prefix offsets.
* `read_expert_profile` refuses a header whose geometry differs from the model, or that claims more ranked
  pairs than slots, and refuses any ranked pair out of range.

## Form template identifiers
* `kNotResident`        -> `NOT_RESIDENT`
* `layerLo/layerHi`     -> `LAYER_SLOT_RANGE_LO/HI`
* `admitBelow`          -> `ADMIT_NO_EVICT`
* `flatIdx`             -> `RESIDENCY_FLAT_INDEX`
* `alignUp256/offList`  -> `SLOT_OFFSETS`
-/

namespace StrataExpertCache

/-- The sentinel for "this expert has no VRAM slot". -/
def kNotResident : Int := -1

-- § per-layer slot ranges

/-- Lower end of layer `l`'s slot range under per-layer admission. -/
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper end (exclusive); the last layer takes the remainder. -/
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by
  simp [layerLo]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerHi_last (slots nl : Nat) : layerHi slots nl (nl - 1) = slots := by
  simp [layerHi]

/-- Each layer's range is well formed. -/
theorem layerLo_le_layerHi (slots nl l : Nat) (hl : l < nl) :
    layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  have hq := Nat.div_mul_le_self slots nl
  generalize slots / nl = q at hq ⊢
  split
  · have h1 : l * q ≤ nl * q := Nat.mul_le_mul_right _ hl.le
    have h2 : nl * q = q * nl := Nat.mul_comm _ _
    omega
  · exact Nat.mul_le_mul_right _ (Nat.le_succ l)

/-- Ranges of distinct layers are disjoint (earlier layer ends before the later one begins). -/
theorem layerHi_le_layerLo (slots nl l1 l2 : Nat) (h : l1 < l2) (h2 : l2 < nl) :
    layerHi slots nl l1 ≤ layerLo slots nl l2 := by
  unfold layerLo layerHi
  generalize slots / nl = q
  have hne : l1 ≠ nl - 1 := by omega
  rw [if_neg hne]
  exact Nat.mul_le_mul_right _ (by omega)

/-- Adjacent layers' ranges are contiguous: no gap between layer `l` and `l+1`. -/
theorem layerHi_eq_layerLo_succ (slots nl l : Nat) (h : l + 1 < nl) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  unfold layerLo layerHi
  have hne : l ≠ nl - 1 := by omega
  rw [if_neg hne]

/-- Every layer's range stays inside the arena. -/
theorem layerHi_le_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by
  unfold layerHi
  have hq := Nat.div_mul_le_self slots nl
  generalize slots / nl = q at hq ⊢
  split
  · exact Nat.le_refl _
  · rename_i hne
    have h1 : (l + 1) * q ≤ nl * q := Nat.mul_le_mul_right _ (by omega)
    have h2 : nl * q = q * nl := Nat.mul_comm _ _
    omega

/-- The planned configuration: 4105 slots over 48 layers is 85 per layer, the last layer getting 110. -/
theorem planned_quota :
    4105 / 48 = 85 ∧ layerHi 4105 48 47 - layerLo 4105 48 47 = 110 := by
  decide

-- § admission without eviction

/-- One admission step against a ceiling: claim the next slot or refuse; never evicts. -/
def admitBelow (hi nx : Nat) : Option Nat × Nat :=
  if hi ≤ nx then (none, nx) else (some nx, nx + 1)

/-- The counter never passes its ceiling. -/
theorem admitBelow_le (hi nx : Nat) (h : nx ≤ hi) : (admitBelow hi nx).2 ≤ hi := by
  unfold admitBelow
  split <;> simp <;> omega

/-- A granted slot lies below the ceiling and equals the old counter. -/
theorem admitBelow_some (hi nx s : Nat) (h : (admitBelow hi nx).1 = some s) : s = nx ∧ s < hi := by
  unfold admitBelow at h
  split at h
  · simp at h
  · rename_i hn
    simp at h
    omega

/-- When the quota is full the call is a no-op returning "not resident": no eviction. -/
theorem admitBelow_full (hi nx : Nat) (h : hi ≤ nx) : admitBelow hi nx = (none, nx) := by
  simp [admitBelow, h]

/-- The counter never decreases. -/
theorem admitBelow_mono (hi nx : Nat) : nx ≤ (admitBelow hi nx).2 := by
  unfold admitBelow
  split <;> simp

/-- Run `n` admission requests against one ceiling and return the final counter. -/
def runAdmit (hi : Nat) : Nat → Nat → Nat
  | nx, 0 => nx
  | nx, n + 1 => runAdmit hi (admitBelow hi nx).2 n

/-- However many requests arrive, a counter that starts within its ceiling stays within it. -/
theorem runAdmit_le (hi : Nat) : ∀ (n nx : Nat), nx ≤ hi → runAdmit hi nx n ≤ hi := by
  intro n
  induction n with
  | zero => intro nx h; simpa [runAdmit] using h
  | succ n ih =>
    intro nx h
    exact ih _ (admitBelow_le hi nx h)

/-- An expert already resident keeps its existing slot; only a non-resident one takes the fresh grant. -/
def admitSlot (cur fresh : Option Nat) : Option Nat :=
  match cur with
  | some s => some s
  | none => fresh

/-- Re-admitting a resident expert is idempotent. -/
theorem admitSlot_resident (s : Nat) (fresh : Option Nat) : admitSlot (some s) fresh = some s := rfl

-- § residency table indexing

/-- Bounds predicate guarding `slot_of` and `admit`. -/
def inBounds (nl ne l e : Nat) : Prop := l < nl ∧ e < ne

/-- Flat index of `(layer, expert)` in the `n_layers * n_expert` table. -/
def flatIdx (ne l e : Nat) : Nat := l * ne + e

/-- In-bounds lookups stay inside the table. -/
theorem flatIdx_lt (nl ne l e : Nat) (h : inBounds nl ne l e) : flatIdx ne l e < nl * ne := by
  obtain ⟨hl, he⟩ := h
  unfold flatIdx
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ hl
  have h2 : (l + 1) * ne = l * ne + ne := Nat.succ_mul l ne
  omega

/-- With the expert in range the flat index determines both coordinates. -/
theorem flatIdx_inj (ne l e l' e' : Nat) (he : e < ne) (he' : e' < ne)
    (h : flatIdx ne l e = flatIdx ne l' e') : l = l' ∧ e = e' := by
  have hne : 0 < ne := by omega
  unfold flatIdx at h
  have h1 : (e + l * ne) / ne = l := by
    rw [Nat.add_mul_div_right _ _ hne, Nat.div_eq_of_lt he, Nat.zero_add]
  have h2 : (e' + l' * ne) / ne = l' := by
    rw [Nat.add_mul_div_right _ _ hne, Nat.div_eq_of_lt he', Nat.zero_add]
  have h3 : e + l * ne = e' + l' * ne := by omega
  have hl : l = l' := by rw [← h1, ← h2, h3]
  subst hl
  omega

/-- Why the expert axis needs its own bounds check: `(0, ne)` aliases `(1, 0)` in the flat table. -/
theorem flatIdx_alias (ne : Nat) : flatIdx ne 0 ne = flatIdx ne 1 0 := by
  simp [flatIdx]

-- § sized slots (256-byte aligned)

/-- Round a slot size up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- A rounded slot is never smaller than the blob. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Rounding adds less than one alignment unit. -/
theorem alignUp256_lt (n : Nat) : alignUp256 n < n + 256 := by
  unfold alignUp256; omega

/-- A rounded slot is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Prefix offsets of back-to-back aligned slots, starting at `acc`; `slots + 1` entries. -/
def offList : Nat → List Nat → List Nat
  | acc, [] => [acc]
  | acc, s :: ss => acc :: offList (acc + alignUp256 s) ss

/-- `open_sized` produces exactly one more offset than slots. -/
theorem offList_length : ∀ (ss : List Nat) (acc : Nat), (offList acc ss).length = ss.length + 1 := by
  intro ss
  induction ss with
  | nil => intro acc; simp [offList]
  | cons s ss ih => intro acc; simp [offList, ih]

/-- Every slot starts at a 256-aligned offset. -/
theorem offList_aligned : ∀ (ss : List Nat) (acc : Nat), acc % 256 = 0 →
    ∀ o ∈ offList acc ss, o % 256 = 0 := by
  intro ss
  induction ss with
  | nil =>
    intro acc h o ho
    simp [offList] at ho
    subst ho
    exact h
  | cons s ss ih =>
    intro acc h o ho
    simp only [offList, List.mem_cons] at ho
    rcases ho with ho | ho
    · subst ho; exact h
    · exact ih (acc + alignUp256 s) (by unfold alignUp256; omega) o ho

-- § profile header validation

/-- Header agreement required by `read_expert_profile`. -/
def profileHeaderOk (nl ne hnl hne want nr : Nat) : Prop := hnl = nl ∧ hne = ne ∧ nr ≤ want

/-- A ranked pair must lie inside the model's geometry. -/
def pairOk (nl ne : Nat) (p : Nat × Nat) : Prop := p.1 < nl ∧ p.2 < ne

/-- A header that passes cannot claim more ranked pairs than slots. -/
theorem profile_ranked_le_slots (nl ne hnl hne want nr : Nat) (h : profileHeaderOk nl ne hnl hne want nr) :
    nr ≤ want := h.2.2

/-- Once every ranked pair is validated, each indexes the residency table in bounds. -/
theorem ranked_flat_lt (nl ne : Nat) (ranked : List (Nat × Nat)) (h : ∀ p ∈ ranked, pairOk nl ne p) :
    ∀ p ∈ ranked, flatIdx ne p.1 p.2 < nl * ne := by
  intro p hp
  exact flatIdx_lt nl ne p.1 p.2 (h p hp)

-- § the measured VRAM budget

/-- 4105 slots of 1,382,400 B fit within 5413 MiB. -/
theorem planned_bytes_fit : 4105 * 1382400 ≤ 5413 * 1048576 := by decide

/-- The live-measured 6414 MiB free leaves about 1 GB spare over 5413 MiB. -/
theorem planned_spare : 6414 - 5413 = 1001 := by decide

end StrataExpertCache
