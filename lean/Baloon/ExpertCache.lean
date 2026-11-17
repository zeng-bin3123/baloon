-- Baloon.ExpertCache: independent implementation; proves the same invariants
-- as Ref.ExpertCache with different definitions and proof strategies.

namespace Baloon.ExpertCache

-- § profile validation

def headerOk (nl ne want nRanked nLayers nExpert : Nat) : Prop :=
  (nl = nLayers) ∧ (ne = nExpert) ∧ (nRanked ≤ want)

theorem header_ranked_le (nl ne want nr L E : Nat) (h : headerOk nl ne want nr L E) : nr ≤ want :=
  h.2.2

def pairOk (l e nLayers nExpert : Nat) : Prop := (l < nLayers) ∧ (e < nExpert)

/-- expert-major flat index: row l, column e, stride nExpert. -/
def flatIdx (l e nExpert : Nat) : Nat := e + l * nExpert

theorem flatIdx_lt (l e L E : Nat) (h : pairOk l e L E) : flatIdx l e E < L * E := by
  obtain ⟨hl, he⟩ := h
  simp only [flatIdx]
  have hLE : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E hl
  have hlin : (l + 1) * E = l * E + E := by ring
  omega

theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  simp only [flatIdx]
  have heq : e + l * E = e + E * l := by ring
  exact ⟨by rw [heq, Nat.add_mul_div_left _ _ hE, Nat.div_eq_of_lt he, Nat.zero_add],
         by rw [heq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]⟩

-- § per-layer slot ranges (quota = slots / nl; last layer takes the remainder)

/-- Lower end of layer l's slot range: quota × l. -/
def layerLo (slots nl l : Nat) : Nat := (slots / nl) * l

/-- Upper end: when l+1 = nl (last layer) return slots; else next quota boundary. -/
def layerHi (slots nl l : Nat) : Nat :=
  if l + 1 = nl then slots else (l + 1) * (slots / nl)

theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by
  unfold layerLo; ring

theorem layer_ranges_adjacent (slots nl l : Nat) (h : l ≠ nl - 1) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by
  simp only [layerHi, layerLo]
  rw [if_neg (by omega)]; ring

theorem last_layer_ends_at_slots (slots m : Nat) : layerHi slots (m + 1) m = slots := by
  unfold layerHi; simp

theorem layer_range_ordered (slots nl l : Nat) (hl : l < nl) :
    layerLo slots nl l ≤ layerHi slots nl l := by
  simp only [layerLo, layerHi]
  by_cases h : l + 1 = nl
  · rw [if_pos h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : slots / nl * l ≤ slots / nl * nl := Nat.mul_le_mul_left _ (by omega)
    omega
  · rw [if_neg h]
    rw [Nat.mul_comm (slots / nl) l]
    exact Nat.mul_le_mul_right _ (Nat.le_succ l)

theorem layer_range_in_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by
  simp only [layerHi]
  by_cases h : l + 1 = nl
  · simp [h]
  · rw [if_neg h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2; omega

theorem last_layer_has_quota (slots m q : Nat) (hq : q = slots / (m + 1)) :
    q ≤ layerHi slots (m + 1) m - layerLo slots (m + 1) m := by
  have h := Nat.div_mul_le_self slots (m + 1)
  rw [← hq] at h; rw [Nat.mul_add, Nat.mul_one] at h
  have hhi : layerHi slots (m + 1) m = slots := last_layer_ends_at_slots slots m
  have hlo : layerLo slots (m + 1) m = q * m := by
    simp only [layerLo]; rw [← hq]
  rw [hhi, hlo]; omega

-- § admission

/-- Admit next slot if cursor is strictly below ceiling; refuse when full. -/
def admitPerLayer (next hi : Nat) : Option Nat := if next ≥ hi then none else some next

theorem admitPerLayer_in_range (next lo hi s : Nat) (hlo : lo ≤ next)
    (h : admitPerLayer next hi = some s) : lo ≤ s ∧ s < hi := by
  unfold admitPerLayer at h
  by_cases hge : next ≥ hi
  · simp [hge] at h
  · simp [hge] at h
    subst h
    exact ⟨hlo, by omega⟩

theorem admitPerLayer_full (next hi : Nat) (h : hi ≤ next) : admitPerLayer next hi = none := by
  simp only [admitPerLayer, if_pos h]

/-- Global cursor: hand out next free slot if below total capacity. -/
def admitGlobal (nextFree slots : Nat) : Option Nat := if nextFree ≥ slots then none else some nextFree

theorem admitGlobal_lt (nf slots s : Nat) (h : admitGlobal nf slots = some s) : s < slots := by
  unfold admitGlobal at h
  by_cases hge : nf ≥ slots
  · simp [hge] at h
  · simp [hge] at h
    subst h
    omega

-- § sized slots

/-- Round n up to the next 256-byte boundary (gap formula). -/
def al256 (n : Nat) : Nat := n + (256 - n % 256) % 256

/-- Slot offsets: start at acc; each slot advances by its aligned size. -/
def offsets : Nat → List Nat → List Nat
  | acc, [] => [acc]
  | acc, b :: bs => acc :: offsets (al256 b + acc) bs

theorem offsets_length (bs : List Nat) (acc : Nat) : (offsets acc bs).length = bs.length + 1 := by
  induction bs generalizing acc with
  | nil => simp [offsets]
  | cons b bs ih =>
    intro acc
    simp only [offsets, List.length_cons]
    have := ih (al256 b + acc)
    omega

theorem offsets_aligned (bs : List Nat) (acc : Nat) (h : acc % 256 = 0) :
    ∀ x ∈ offsets acc bs, x % 256 = 0 := by
  induction bs generalizing acc with
  | nil =>
    intro acc h x hx
    simp [offsets] at hx
    omega
  | cons b bs ih =>
    intro acc h x hx
    simp only [offsets, List.mem_cons] at hx
    cases hx with
    | inl heq => omega
    | inr hmem =>
      have h1 : (al256 b + acc) % 256 = 0 := by unfold al256; omega
      exact ih _ h1 x hmem

def fillLen (bytes blob : Nat) : Nat := if bytes > 0 ∧ bytes ≤ blob then bytes else blob

theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by
  simp only [fillLen]; split_ifs <;> omega

theorem fillLen_pos (bytes blob : Nat) (h : 0 < blob) : 0 < fillLen bytes blob := by
  simp only [fillLen]; split_ifs <;> omega

theorem slot_start_in_arena (slot slots blob : Nat) (h : slot < slots) :
    slot * blob + blob ≤ slots * blob :=
  calc slot * blob + blob = (slot + 1) * blob := by ring
    _ ≤ slots * blob := Nat.mul_le_mul_right blob h

end Baloon.ExpertCache
