-- Baloon.ExpertCache: VRAM expert-cache profile validation, flat residency index,
-- per-layer slot ranges and slot fill bounds.
-- Independent implementation proving the same mathematical facts as Ref.ExpertCache.

namespace Baloon.ExpertCache

def headerOk (nl ne want nRanked nLayers nExpert : Nat) : Prop :=
  (nl = nLayers) ∧ (ne = nExpert) ∧ (nRanked ≤ want)

theorem header_ranked_le (nl ne want nr L E : Nat) (h : headerOk nl ne want nr L E) : nr ≤ want := by {
  obtain ⟨_, _, h3⟩ := h; exact h3 }

def pairOk (l e nLayers nExpert : Nat) : Prop := (l < nLayers) ∧ (e < nExpert)
def flatIdx (l e nExpert : Nat) : Nat := (l * nExpert + e)

theorem flatIdx_lt (l e L E : Nat) (h : pairOk l e L E) : flatIdx l e E < L * E := by {
  obtain ⟨hl, he⟩ := h
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx; omega }

theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by {
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx; rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he] }

def layerLo (slots nl l : Nat) : Nat := (l * (slots / nl))
def layerHi (slots nl l : Nat) : Nat :=
  if (l = nl - 1) then slots else (l + 1) * (slots / nl)

theorem layerLo_zero (slots nl : Nat) : layerLo slots nl 0 = 0 := by {
  simp [layerLo] }

theorem layer_ranges_adjacent (slots nl l : Nat) (h : l ≠ nl - 1) :
    layerHi slots nl l = layerLo slots nl (l + 1) := by {
  simp [layerHi, layerLo, h] }

theorem last_layer_ends_at_slots (slots m : Nat) : layerHi slots (m + 1) m = slots := by {
  simp [layerHi] }

theorem layer_range_ordered (slots nl l : Nat) (hl : l < nl) :
    layerLo slots nl l ≤ layerHi slots nl l := by {
  unfold layerLo layerHi
  by_cases h : l = nl - 1
  · rw [if_pos h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : l * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    rw [Nat.mul_comm nl] at h2; omega
  · rw [if_neg h]; exact Nat.mul_le_mul_right _ (by omega) }

theorem layer_range_in_slots (slots nl l : Nat) (hl : l < nl) : layerHi slots nl l ≤ slots := by {
  unfold layerHi
  by_cases h : l = nl - 1
  · simp [h]
  · rw [if_neg h]
    have h1 : slots / nl * nl ≤ slots := Nat.div_mul_le_self slots nl
    have h2 : (l + 1) * (slots / nl) ≤ nl * (slots / nl) := Nat.mul_le_mul_right _ (by omega)
    linarith [Nat.mul_comm nl (slots / nl)] }

-- § last-layer quota

theorem last_layer_has_quota (slots m q : Nat) (hq : q = slots / (m + 1)) :
    q ≤ layerHi slots (m + 1) m - layerLo slots (m + 1) m := by {
  have hhi : layerHi slots (m + 1) m = slots := last_layer_ends_at_slots slots m
  have hlo : layerLo slots (m + 1) m = m * q := by { unfold layerLo; rw [← hq] }
  rw [hhi, hlo]
  -- goal: q ≤ slots - m * q; derive m * q + q ≤ slots via the div bound
  have hstep : m * q + q ≤ slots := by {
    have h := Nat.div_mul_le_self slots (m + 1)
    rw [← hq] at h                          -- h : q * (m + 1) ≤ slots
    have heq : q * (m + 1) = m * q + q := by ring
    linarith }
  omega }

-- § per-layer admission

def admitPerLayer (next hi : Nat) : Option Nat :=
  if (next < hi) then some next else none

theorem admitPerLayer_in_range (next lo hi s : Nat) (hlo : lo ≤ next)
    (h : admitPerLayer next hi = some s) : lo ≤ s ∧ s < hi := by {
  unfold admitPerLayer at h
  split at h
  · rename_i hc; cases h; exact ⟨hlo, hc⟩
  · cases h }

theorem admitPerLayer_full (next hi : Nat) (h : hi ≤ next) : admitPerLayer next hi = none := by {
  unfold admitPerLayer
  rw [if_neg (by omega)] }

-- § global admission

def admitGlobal (nextFree slots : Nat) : Option Nat :=
  if (nextFree < slots) then some nextFree else none

theorem admitGlobal_lt (nf slots s : Nat) (h : admitGlobal nf slots = some s) : s < slots := by {
  unfold admitGlobal at h
  split at h
  · rename_i hc; cases h; exact hc
  · cases h }

-- § sized slot offsets (gap-padding alignment; different variable names from Ref)

def al256 (n : Nat) : Nat := n + (256 - n % 256) % 256

def offsets : Nat → List Nat → List Nat
  | a, []     => [a]
  | a, x :: xs => a :: offsets (a + al256 x) xs

theorem offsets_length : ∀ (bs : List Nat) (acc : Nat), (offsets acc bs).length = bs.length + 1
  | [], acc => by { simp [offsets] }
  | b :: bs, acc => by {
    simp only [offsets, List.length_cons]
    rw [offsets_length bs] }

theorem offsets_aligned : ∀ (bs : List Nat) (acc : Nat), acc % 256 = 0 →
    ∀ x ∈ offsets acc bs, x % 256 = 0
  | [], acc, h, x, hx => by {
    simp [offsets] at hx; omega }
  | b :: bs, acc, h, x, hx => by {
    simp only [offsets, List.mem_cons] at hx
    rcases hx with hx | hx
    · omega
    · have h1 : (acc + al256 b) % 256 = 0 := by { unfold al256; omega }
      exact offsets_aligned bs (acc + al256 b) h1 x hx }

-- § fill length and arena bounds

def fillLen (bytes blob : Nat) : Nat :=
  if (0 < bytes ∧ bytes ≤ blob) then bytes else blob

theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by {
  unfold fillLen; split <;> omega }

theorem fillLen_pos (bytes blob : Nat) (h : 0 < blob) : 0 < fillLen bytes blob := by {
  unfold fillLen; split <;> omega }

theorem slot_start_in_arena (slot slots blob : Nat) (h : slot < slots) :
    slot * blob + blob ≤ slots * blob := by {
  have h1 : (slot + 1) * blob ≤ slots * blob := Nat.mul_le_mul_right blob (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  exact h1 }

end Baloon.ExpertCache
