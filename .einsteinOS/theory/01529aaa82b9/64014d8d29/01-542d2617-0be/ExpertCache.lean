/-!
# MODULE: expert_cache
SOURCES: src/core/expert_cache.cpp, include/strata/core/expert_cache.hpp
The VRAM-resident expert tier: profile-file validation, slot arena sizing, the (layer, expert) → slot residency
table, per-layer slot ranges for admission, variable-size slot offsets and bounded slot fills.

## Source semantics
* A profile header must match the model's `n_layers × n_expert`, and may rank at most `slots` pairs; every
  ranked pair must be in range.
* The residency table is flat row-major `layer * n_expert + expert`; out-of-range queries are "not resident".
* Per-layer admission: layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the
  remainder, so the ranges tile `[0, slots)`; admission never evicts.
* Sized slots are 256-byte aligned, back to back; `fill_slot` copies at most the slot's blob size.

## Form template identifiers
- flatIdx → `layer * n_expert + expert` ; layerLo/layerHi → layer_slot_range ; offsets → `off_`
- fillLen → `n = (bytes > 0 && bytes <= blob_) ? bytes : blob_`
-/

namespace Strata.ExpertCache

-- § profile validation

/-- Header acceptance of `read_expert_profile`. -/
def headerOk (nl ne want nRanked nLayers nExpert : Nat) : Prop :=
  nl = nLayers ∧ ne = nExpert ∧ nRanked ≤ want

/-- Accepted headers rank no more pairs than there are slots. -/
theorem header_ranked_le (nl ne want nr L E : Nat) (h : headerOk nl ne want nr L E) : nr ≤ want := by
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- A pair is in range. -/
def pairOk (l e nLayers nExpert : Nat) : Prop := l < nLayers ∧ e < nExpert

/-- Flat residency index. -/
def flatIdx (l e nExpert : Nat) : Nat := l * nExpert + e

/-- An in-range pair indexes inside the residency table. -/
theorem flatIdx_lt (l e L E : Nat) (h : pairOk l e L E) : flatIdx l e E < L * E := by
  obtain ⟨hl, he⟩ := h
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx
  omega

/-- The flat index determines the pair (layer = idx / E, expert = idx % E). -/
theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx
    rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

-- § per-layer slot ranges

/-- Lower end of layer `l`'s slot range. -/
def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)

/-- Upper end: the last layer takes whatever is left. -/
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by
  simp [layerLo]

/-- A non-last layer's range ends where the next one starts (no gap, no overlap). -/
theorem layer_ranges_adjacent (slots nl l : Nat) (h : l ≠ nl - 1) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  simp [layerHi, layerLo, h]

/-- The last layer ends at `slots`, so the ranges cover `0..slots` exactly. -/
theorem last_layer_ends_at_slots (slots m : Nat) : layerHi slots (m + 1) m = slots := by
  simp [layerHi]

/-- Every range has `lo ≤ hi` (for layers below `nl`). -/
theorem layer_range_ordered (slots nl l : Nat) (hl : l < nl) : layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  by_cases h : l = nl - 1
  · rw [if_pos h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2
    omega
  · rw [if_neg h]
    exact Nat.mul_le_mul_right _ (by omega)

/-- Every range stays inside `0..slots`. -/
theorem layer_range_in_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by
  unfold layerHi
  by_cases h : l = nl - 1
  · simp [h]
  · rw [if_neg h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2
    omega

/-- The last layer's range is at least the uniform quota `q`: it gets the remainder on top. -/
theorem last_layer_has_quota (slots m q : Nat) (hq : q = slots / (m + 1)) :
    q ≤ layerHi slots (m + 1) m - layerLo slots (m + 1) m := by
  have h := Nat.div_mul_le_self slots (m + 1)
  rw [← hq] at h
  rw [Nat.mul_add, Nat.mul_one] at h
  have hhi : layerHi slots (m + 1) m = slots := last_layer_ends_at_slots slots m
  have hlo : layerLo slots (m + 1) m = m * q := by
    unfold layerLo
    rw [← hq]
  rw [hhi, hlo]
  rw [Nat.mul_comm q m] at h
  omega

-- § admission

/-- Per-layer admission: take the layer's next slot if it is below the layer's ceiling. -/
def admitPerLayer (next hi : Nat) : Option Nat := if next < hi then some next else none

/-- An admitted slot lies in `[lo, hi)` when the cursor was at or above `lo`. -/
theorem admitPerLayer_in_range (next lo hi s : Nat) (hlo : lo ≤ next)
    (h : admitPerLayer next hi = some s) : lo ≤ s ∧ s < hi := by
  unfold admitPerLayer at h
  split at h
  · rename_i hc
    cases h
    exact ⟨hlo, hc⟩
  · cases h

/-- A full layer admits nothing (no eviction). -/
theorem admitPerLayer_full (next hi : Nat) (h : hi ≤ next) : admitPerLayer next hi = none := by
  unfold admitPerLayer
  rw [if_neg (by omega)]

/-- Global admission: the shared counter hands out slots below `slots` only. -/
def admitGlobal (nextFree slots : Nat) : Option Nat := if nextFree < slots then some nextFree else none

/-- Global admission never produces an out-of-range slot. -/
theorem admitGlobal_lt (nf slots s : Nat) (h : admitGlobal nf slots = some s) : s < slots := by
  unfold admitGlobal at h
  split at h
  · rename_i hc
    cases h
    exact hc
  · cases h

-- § sized slots

/-- Round a slot up to 256 bytes. -/
def al256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Slot offsets: start at `acc`, each slot advances by its aligned size; the list ends with the total. -/
def offsets : Nat → List Nat → List Nat
  | acc, [] => [acc]
  | acc, b :: bs => acc :: offsets (acc + al256 b) bs

/-- The offset list has one more entry than there are slots. -/
theorem offsets_length : ∀ (bs : List Nat) (acc : Nat), (offsets acc bs).length = bs.length + 1
  | [], acc => by simp [offsets]
  | b :: bs, acc => by
    simp only [offsets, List.length_cons]
    rw [offsets_length bs]

/-- Every slot offset is 256-byte aligned when the start is. -/
theorem offsets_aligned : ∀ (bs : List Nat) (acc : Nat), acc % 256 = 0 →
    ∀ x ∈ offsets acc bs, x % 256 = 0
  | [], acc, h, x, hx => by
    simp [offsets] at hx
    omega
  | b :: bs, acc, h, x, hx => by
    simp only [offsets, List.mem_cons] at hx
    rcases hx with hx | hx
    · omega
    · have h1 : (acc + al256 b) % 256 = 0 := by unfold al256; omega
      exact offsets_aligned bs (acc + al256 b) h1 x hx

/-- `fill_slot`'s copy length: the given size when it fits the slot, else the whole slot. -/
def fillLen (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never exceeds the slot's blob size. -/
theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by
  unfold fillLen
  split <;> omega

/-- A fill is never empty when the blob size is positive. -/
theorem fillLen_pos (bytes blob : Nat) (h : 0 < blob) : 0 < fillLen bytes blob := by
  unfold fillLen
  split <;> omega

/-- Slot arena size is `slots * blob`, so every slot start is inside it. -/
theorem slot_start_in_arena (slot slots blob : Nat) (h : slot < slots) : slot * blob + blob ≤ slots * blob := by
  have h1 : (slot + 1) * blob ≤ slots * blob := Nat.mul_le_mul_right blob (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  exact h1

end Strata.ExpertCache
