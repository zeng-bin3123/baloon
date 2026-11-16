-- Baloon.ExpertCache: VRAM expert-cache profile validation, flat residency index,
-- per-layer slot ranges and slot fill bounds.
-- Independent implementation proving the same mathematical facts as Ref.ExpertCache.

namespace Baloon.ExpertCache

def headerOk (nl ne want nRanked nLayers nExpert : Nat) : Prop :=
  nl = nLayers ∧ ne = nExpert ∧ nRanked ≤ want

theorem header_ranked_le (nl ne want nr L E : Nat) (h : headerOk nl ne want nr L E) : nr ≤ want := by
  obtain ⟨_, _, h3⟩ := h; exact h3

def pairOk (l e nLayers nExpert : Nat) : Prop := l < nLayers ∧ e < nExpert
def flatIdx (l e nExpert : Nat) : Nat := l * nExpert + e

theorem flatIdx_lt (l e L E : Nat) (h : pairOk l e L E) : flatIdx l e E < L * E := by
  obtain ⟨hl, he⟩ := h
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx; omega

theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx; rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

def layerLo (slots nl l : Nat) : Nat := l * (slots / nl)
def layerHi (slots nl l : Nat) : Nat := if l = nl - 1 then slots else (l + 1) * (slots / nl)

theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by simp [layerLo]

theorem layer_ranges_adjacent (slots nl l : Nat) (h : l ≠ nl - 1) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by simp [layerHi, layerLo, h]

theorem last_layer_ends_at_slots (slots m : Nat) : layerHi slots (m + 1) m = slots := by
  simp [layerHi]

theorem layer_range_ordered (slots nl l : Nat) (hl : l < nl) :
    layerLo slots nl l ≤ layerHi slots nl l := by
  unfold layerLo layerHi
  by_cases h : l = nl - 1
  · rw [if_pos h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2; omega
  · rw [if_neg h]; exact Nat.mul_le_mul_right _ (by omega)

theorem layer_range_in_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by
  unfold layerHi
  by_cases h : l = nl - 1
  · simp [h]
  · rw [if_neg h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    linarith [Nat.mul_comm nl (slots / nl)]

end Baloon.ExpertCache
