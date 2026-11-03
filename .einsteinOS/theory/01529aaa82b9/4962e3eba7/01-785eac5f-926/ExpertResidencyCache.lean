import Mathlib

/-!
# MODULE: expert_residency_cache
The VRAM-resident expert tier's slot storage and residency table (`ExpertCache`, `read_expert_profile`).
It answers `(layer, expert) -> slot or none`, admits experts into free slots without ever evicting, optionally
partitions the slots into per-layer ranges (R4.2g), and lays out variable-size slots on 256-byte boundaries.

## Source semantics
* `slot_of` is bounds-checked: an out-of-range layer or expert is "not resident", never an adjacent cell.
* `admit` never evicts: the admission counter is monotone and never passes its ceiling (`slots_` globally,
  the layer's range end under per-layer admission).
* Per-layer ranges `[l*q, (l+1)*q)`, `q = slots / n_layers`, last layer takes the remainder; they are
  pairwise disjoint and lie inside `0..slots`.
* `open_sized` aligns each slot to 256 bytes; offsets are monotone and 256-divisible.
* `read_expert_profile` refuses a profile for a different geometry or with out-of-range pairs.
* `fill_slot` copies at most one slot's bytes.

## Form template identifiers
* `layerRange`      <- `ExpertCache::layer_slot_range`
* `admitSlot`/`admitNext` <- `ExpertCache::admit` (counter `next_free_` / `layer_next_[l]`, ceiling `slots_` / `hi`)
* `slotOf`          <- `ExpertCache::slot_of`
* `align256`/`offAt` <- `ExpertCache::open_sized` offsets `off_`
* `fillBytes`       <- the `n` computed in `fill_slot`/`fill_slot_blocking`/`verify_slot`
* `readProfile`     <- `read_expert_profile`
-/

namespace Strata.ExpertCache

variable {slots nLayers layer l1 l2 : Nat}

-- § per-layer slot ranges

/-- The slot range `[lo, hi)` layer `layer` may admit into under per-layer admission. -/
def layerRange (slots nLayers layer : Nat) : Nat × Nat :=
  if nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer then (0, 0)
  else (layer * (slots / nLayers),
        if layer = nLayers - 1 then slots else (layer + 1) * (slots / nLayers))

/-- The lower end of a valid layer's range is `layer * q`. -/
theorem layerRange_lo (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).1 = layer * (slots / nLayers) := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc]

/-- The last layer's range ends exactly at `slots`, so no slot is orphaned by the division. -/
theorem layerRange_hi_last (hl : layer < nLayers) (hs : 0 < slots) (hlast : layer = nLayers - 1) :
    (layerRange slots nLayers layer).2 = slots := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc, if_pos hlast]

/-- A non-last layer's range ends at `(layer + 1) * q`. -/
theorem layerRange_hi_not_last (hl : layer < nLayers) (hs : 0 < slots) (hlast : layer ≠ nLayers - 1) :
    (layerRange slots nLayers layer).2 = (layer + 1) * (slots / nLayers) := by
  have hc : ¬ (nLayers = 0 ∨ slots = 0 ∨ nLayers ≤ layer) := by omega
  simp only [layerRange, if_neg hc, if_neg hlast]

/-- A valid layer's range is well formed: `lo ≤ hi`. -/
theorem layerRange_lo_le_hi (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).1 ≤ (layerRange slots nLayers layer).2 := by
  rw [layerRange_lo hl hs]
  by_cases hlast : layer = nLayers - 1
  · rw [layerRange_hi_last hl hs hlast]
    have h1 : layer * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : nLayers * (slots / nLayers) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nLayers
    omega
  · rw [layerRange_hi_not_last hl hs hlast]
    exact Nat.mul_le_mul_right _ (Nat.le_succ _)

/-- Every layer's range stays inside the arena: `hi ≤ slots`. -/
theorem layerRange_hi_le_slots (hl : layer < nLayers) (hs : 0 < slots) :
    (layerRange slots nLayers layer).2 ≤ slots := by
  by_cases hlast : layer = nLayers - 1
  · exact (layerRange_hi_last hl hs hlast).le
  · rw [layerRange_hi_not_last hl hs hlast]
    have h1 : (layer + 1) * (slots / nLayers) ≤ nLayers * (slots / nLayers) :=
      Nat.mul_le_mul_right _ (by omega)
    have h2 : nLayers * (slots / nLayers) ≤ slots := by
      rw [Nat.mul_comm]; exact Nat.div_mul_le_self slots nLayers
    omega

/-- Ranges of distinct layers never overlap: a lower layer's `hi` is at most a higher layer's `lo`. -/
theorem layerRange_disjoint (hs : 0 < slots) (h12 : l1 < l2) (hl2 : l2 < nLayers) :
    (layerRange slots nLayers l1).2 ≤ (layerRange slots nLayers l2).1 := by
  have hl1 : l1 < nLayers := by omega
  have hne : l1 ≠ nLayers - 1 := by omega
  rw [layerRange_hi_not_last hl1 hs hne, layerRange_lo hl2 hs]
  exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt h12)

-- § admission without eviction

/-- The slot `admit` hands out: the next free one, or none when the ceiling `hi` is reached. -/
def admitSlot (hi next : Nat) : Option Nat := if hi ≤ next then none else some next

/-- The admission counter after an `admit` call. -/
def admitNext (hi next : Nat) : Nat := if hi ≤ next then next else next + 1

/-- An admitted slot is always below the ceiling, so it is inside the arena (or the layer's range). -/
theorem admitSlot_lt (hi next s : Nat) (h : admitSlot hi next = some s) : s < hi := by
  unfold admitSlot at h
  split_ifs at h with hc
  simp at h
  omega

/-- A full cache refuses: no eviction, no slot. -/
theorem admitSlot_full (hi next : Nat) (h : hi ≤ next) : admitSlot hi next = none := by
  unfold admitSlot
  rw [if_pos h]

/-- The counter never decreases: there is no eviction. -/
theorem admitNext_ge (hi next : Nat) : next ≤ admitNext hi next := by
  unfold admitNext
  split_ifs <;> omega

/-- The counter never passes the ceiling if it started at or below it. -/
theorem admitNext_le (hi next : Nat) (h : next ≤ hi) : admitNext hi next ≤ hi := by
  unfold admitNext
  split_ifs with hc
  · exact h
  · omega

/-- The handed-out slot is exactly the counter's old value, so slots are claimed in order. -/
theorem admitSlot_eq_counter (hi next s : Nat) (h : admitSlot hi next = some s) : s = next := by
  unfold admitSlot at h
  split_ifs at h with hc
  simp at h
  omega

-- § the residency table

/-- `(layer, expert) -> slot`, bounds-checked against both axes. -/
def slotOf (res : Nat → Option Nat) (nl ne l e : Nat) : Option Nat :=
  if l < nl ∧ e < ne then res (l * ne + e) else none

/-- An out-of-range layer or expert is "not resident". -/
theorem slotOf_oob (res : Nat → Option Nat) (nl ne l e : Nat) (h : ¬ (l < nl ∧ e < ne)) :
    slotOf res nl ne l e = none := by
  unfold slotOf
  rw [if_neg h]

/-- Why both axes are checked: a flat index one expert past a layer's end is the next layer's first cells. -/
theorem flat_index_alias (l e' ne : Nat) : l * ne + (ne + e') = (l + 1) * ne + e' := by
  ring

/-- A valid `(layer, expert)` pair lands inside the flat table. -/
theorem flat_index_lt (nl ne l e : Nat) (hl : l < nl) (he : e < ne) : l * ne + e < nl * ne := by
  have h1 : (l + 1) * ne ≤ nl * ne := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hl)
  have h2 : (l + 1) * ne = l * ne + ne := by ring
  omega

/-- The flat index is injective on valid pairs: `layer * n_expert + expert` has no padding or collision. -/
theorem flat_index_inj (ne l l' e e' : Nat) (hne : 0 < ne) (he : e < ne) (he' : e' < ne)
    (h : l * ne + e = l' * ne + e') : l = l' ∧ e = e' := by
  have hm : (l * ne + e) % ne = e := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt he
  have hm' : (l' * ne + e') % ne = e' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt he'
  have hee : e = e' := by
    calc e = (l * ne + e) % ne := hm.symm
      _ = (l' * ne + e') % ne := by rw [h]
      _ = e' := hm'
  have hll : l * ne = l' * ne := by omega
  exact ⟨Nat.eq_of_mul_eq_mul_right hne hll, hee⟩

-- § sized slots (plan v0.3 P6)

/-- Round a byte count up to a multiple of 256, as `open_sized` does. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- An aligned size is a multiple of 256. -/
theorem align256_dvd (n : Nat) : 256 ∣ align256 n := by
  unfold align256
  exact Nat.dvd_mul_left 256 _

/-- Alignment never shrinks a blob: the slot holds its blob. -/
theorem le_align256 (n : Nat) : n ≤ align256 n := by
  unfold align256
  omega

/-- Byte offset of slot `i` in the arena: the running sum of the aligned sizes before it. -/
def offAt (sizes : List Nat) : Nat → Nat
  | 0 => 0
  | i + 1 => offAt sizes i + align256 (sizes.getD i 0)

/-- Offsets are monotone, so slots are laid out back to back without overlap. -/
theorem offAt_mono (sizes : List Nat) (i : Nat) : offAt sizes i ≤ offAt sizes (i + 1) := by
  show offAt sizes i ≤ offAt sizes i + align256 (sizes.getD i 0)
  exact Nat.le_add_right _ _

/-- Every slot offset is 256-byte aligned, which is what the kernels' vector loads expect. -/
theorem offAt_dvd (sizes : List Nat) (i : Nat) : 256 ∣ offAt sizes i := by
  induction i with
  | zero => exact ⟨0, rfl⟩
  | succ n ih =>
    show 256 ∣ offAt sizes n + align256 (sizes.getD n 0)
    exact Nat.dvd_add ih (align256_dvd _)

/-- Slot `i` spans exactly its aligned size, so it is at least as large as the blob it holds. -/
theorem slot_span (sizes : List Nat) (i : Nat) :
    offAt sizes (i + 1) - offAt sizes i = align256 (sizes.getD i 0) := by
  show offAt sizes i + align256 (sizes.getD i 0) - offAt sizes i = align256 (sizes.getD i 0)
  omega

-- § fills

/-- Bytes a fill copies: the blob's own size when it fits in the slot, else the whole slot. -/
def fillBytes (blob bytes : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never copies more than one slot, so it cannot run past the arena. -/
theorem fillBytes_le (blob bytes : Nat) : fillBytes blob bytes ≤ blob := by
  unfold fillBytes
  split_ifs with h
  · exact h.2
  · exact Nat.le_refl _

/-- `bytes = 0` means "the whole slot". -/
theorem fillBytes_zero (blob : Nat) : fillBytes blob 0 = blob := by
  unfold fillBytes
  rw [if_neg (by omega)]

-- § the profile reader

/-- `read_expert_profile`'s acceptance rule: same geometry, `n_ranked <= slots`, a complete list, in-range pairs. -/
def readProfile (nl ne hdrNl hdrNe want nRanked : Nat) (pairs : List (Nat × Nat)) :
    Option (List (Nat × Nat)) :=
  if hdrNl = nl ∧ hdrNe = ne ∧ nRanked ≤ want ∧ pairs.length = nRanked ∧
      pairs.all (fun p => decide (p.1 < nl ∧ p.2 < ne)) = true
  then some pairs else none

/-- An accepted profile has at most `slots` pairs and every pair is a real `(layer, expert)`. -/
theorem readProfile_sound (nl ne a b want nr : Nat) (pairs r : List (Nat × Nat))
    (h : readProfile nl ne a b want nr pairs = some r) :
    r.length ≤ want ∧ ∀ p ∈ r, p.1 < nl ∧ p.2 < ne := by
  unfold readProfile at h
  split_ifs at h with hc
  obtain ⟨_, _, h3, h4, h5⟩ := hc
  have hr : pairs = r := by simpa using h
  subst hr
  refine ⟨by omega, ?_⟩
  intro p hp
  have hp' := List.all_eq_true.mp h5 p hp
  simpa using hp'

/-- A profile for a different artifact (wrong layer count) is refused. -/
theorem readProfile_wrong_geometry (nl ne a b want nr : Nat) (pairs : List (Nat × Nat))
    (h : a ≠ nl) : readProfile nl ne a b want nr pairs = none := by
  unfold readProfile
  rw [if_neg (by intro hh; exact h hh.1)]

end Strata.ExpertCache
