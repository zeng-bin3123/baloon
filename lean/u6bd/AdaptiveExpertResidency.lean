/-!
# MODULE: adaptive_expert_residency
SOURCES: src/program/generate.cpp
How `ref generate` sizes, shrinks, lends and adapts the VRAM expert tier: `--expert-cache auto`
slot count, 256-byte-rounded sized slots kept as a prefix of the ranked profile, the prompt path
borrowing the last slots, and the adaptive swap that moves a hot missing expert into a cold resident's
slot.

## Source semantics
* auto: `slots = max(0, (free - reserve) / max_blob)`, capped by the profile length; the slots never
  use more than `free - reserve`.
* Sized slots: walk the ranked profile, each blob rounded up to 256 bytes, stop at the first that does
  not fit; the kept slots are a prefix and their rounded sum fits the budget.  Shrinking re-runs this.
* Borrowing takes the last `k` slots only when `k + 128 ≤ slots`, so at least 128 stay resident; the
  prompt chunk is halved (never below 256) until its buffers fit.
* Adaptive swap: a missing expert with decayed usage `≥ 2` replaces its layer's least-used resident one
  when it was used at least `1.5` more; usage then decays by `0.7`.  The victim is evicted at once and
  the new expert becomes resident on the victim's slot when the copy lands, so no slot is ever owned by
  two experts.  Usage is kept here in tenths so the thresholds are 20, 15 and `×7/10`.

## Form template identifiers
auto sizing → autoSlots · `(b + 255) / 256 * 256` → round256 · the sized-slot walk → keepPrefix ·
lend condition → lendFirst · swap test → swapWorth · host_res → Residency · evict/apply_pending →
evict / admit · `v *= 0.7f` → decay
-/

namespace Ref.Generate.Residency

-- § auto sizing

/-- `--expert-cache auto`: the slots that fit in what is free after the reserve. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized slots never eat into the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Capping by the profile length can only shrink the tier. -/
theorem capped_fit (free reserve blob plen : Nat) :
    min (autoSlots free reserve blob) plen * blob ≤ free - reserve := by
  have h := autoSlots_fit free reserve blob
  have hm : min (autoSlots free reserve blob) plen ≤ autoSlots free reserve blob := Nat.min_le_left _ _
  exact Nat.le_trans (Nat.mul_le_mul_right blob hm) h

-- § sized slots

/-- A blob's footprint rounded up to 256 bytes. -/
def round256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Rounding never loses bytes. -/
theorem le_round256 (b : Nat) : b ≤ round256 b := by
  unfold round256
  omega

/-- Rounded footprints are 256-aligned. -/
theorem round256_aligned (b : Nat) : round256 b % 256 = 0 := by
  unfold round256
  omega

/-- Rounding adds less than 256 bytes. -/
theorem round256_lt (b : Nat) : round256 b < b + 256 := by
  unfold round256
  omega

/-- Keep the longest prefix of the ranked sizes whose rounded footprints fit in `room`. -/
def keepPrefix : List Nat → Nat → List Nat
  | [], _ => []
  | s :: ss, room => if round256 s ≤ room then s :: keepPrefix ss (room - round256 s) else []

/-- Sum of rounded footprints. -/
def roundedSum : List Nat → Nat
  | [] => 0
  | s :: ss => round256 s + roundedSum ss

/-- The kept slots fit the budget. -/
theorem keepPrefix_fits : ∀ (ss : List Nat) (room : Nat), roundedSum (keepPrefix ss room) ≤ room
  | [], _ => by simp [keepPrefix, roundedSum]
  | s :: ss, room => by
    simp only [keepPrefix]
    split
    · rename_i h
      have ih := keepPrefix_fits ss (room - round256 s)
      simp only [roundedSum]
      omega
    · simp [roundedSum]

/-- The kept slots are a prefix of the ranking: the most-routed experts are the ones kept. -/
theorem keepPrefix_prefix : ∀ (ss : List Nat) (room : Nat), ∃ t, keepPrefix ss room ++ t = ss
  | [], _ => ⟨[], by simp [keepPrefix]⟩
  | s :: ss, room => by
    simp only [keepPrefix]
    split
    · obtain ⟨t, ht⟩ := keepPrefix_prefix ss (room - round256 s)
      exact ⟨t, by simp [ht]⟩
    · exact ⟨s :: ss, by simp⟩

/-- Shrinking the budget never keeps more slots. -/
theorem keepPrefix_mono : ∀ (ss : List Nat) {r r' : Nat}, r ≤ r' →
    (keepPrefix ss r).length ≤ (keepPrefix ss r').length
  | [], _, _, _ => by simp [keepPrefix]
  | s :: ss, r, r', h => by
    simp only [keepPrefix]
    by_cases h1 : round256 s ≤ r
    · have h2 : round256 s ≤ r' := Nat.le_trans h1 h
      simp only [h1, h2, if_true, List.length_cons]
      have := keepPrefix_mono ss (r := r - round256 s) (r' := r' - round256 s) (by omega)
      omega
    · simp only [h1, if_false, List.length_nil]
      exact Nat.zero_le _

-- § lending to the prompt path

/-- The first lent slot when the prompt path borrows `k` of `slots`. -/
def lendFirst (slots k : Nat) : Nat := slots - k

/-- Lending is only allowed with 128 slots to spare, and those stay resident for decode. -/
theorem lend_keeps_128 {slots k : Nat} (h : k + 128 ≤ slots) : 128 ≤ lendFirst slots k := by
  unfold lendFirst
  omega

/-- The lent slots are exactly the last `k`. -/
theorem lent_count {slots k : Nat} (h : k ≤ slots) : slots - lendFirst slots k = k := by
  unfold lendFirst
  omega

/-- Halving the prompt chunk strictly shrinks it while the loop runs (`chunk ≥ 256`). -/
theorem halve_shrinks {chunk : Nat} (h : 256 ≤ chunk) : chunk / 2 < chunk := by
  omega

-- § adaptive swap

/-- A candidate (missing, decayed usage in tenths) displaces a victim (resident) only when it was
routed at least twice and at least 1.5 more often. -/
def swapWorth (cand vict : Nat) : Bool := decide (20 ≤ cand ∧ vict + 15 ≤ cand)

/-- An accepted swap always has positive gain, so sorting by gain is meaningful. -/
theorem swap_gain_pos {c v : Nat} (h : swapWorth c v = true) : v < c := by
  simp only [swapWorth, decide_eq_true_eq] at h
  omega

/-- Decay by 0.7 (in integer tenths) never raises a usage count. -/
def decay (u : Nat) : Nat := u * 7 / 10

/-- Decay is non-increasing. -/
theorem decay_le (u : Nat) : decay u ≤ u := by
  unfold decay
  omega

/-- Residency: expert index ↦ slot, `none` when not resident. -/
abbrev Residency := Nat → Option Nat

/-- Pointwise update. -/
def upd (r : Residency) (k : Nat) (v : Option Nat) : Residency := fun j => if j = k then v else r j

/-- No slot owned by two experts. -/
def SlotsUnique (r : Residency) : Prop := ∀ a b s, r a = some s → r b = some s → a = b

/-- The victim is evicted at once (the CPU computes it meanwhile). -/
def evict (r : Residency) (out : Nat) : Residency := upd r out none

/-- The incoming expert takes the slot once the copy has landed. -/
def admit (r : Residency) (inn slot : Nat) : Residency := upd r inn (some slot)

/-- Evicting preserves slot uniqueness. -/
theorem evict_unique {r : Residency} (h : SlotsUnique r) (out : Nat) : SlotsUnique (evict r out) := by
  intro a b s ha hb
  simp only [evict, upd] at ha hb
  by_cases hao : a = out
  · simp [hao] at ha
  · by_cases hbo : b = out
    · simp [hbo] at hb
    · simp only [hao, hbo, if_false] at ha hb
      exact h a b s ha hb

/-- During flight the victim's slot belongs to no expert. -/
theorem slot_free_in_flight {r : Residency} (h : SlotsUnique r) {out s : Nat} (hs : r out = some s) :
    ∀ j, evict r out j ≠ some s := by
  intro j hj
  simp only [evict, upd] at hj
  by_cases hjo : j = out
  · simp [hjo] at hj
  · simp only [hjo, if_false] at hj
    exact hjo (h j out s hj hs)

/-- The full swap (evict, then admit on the freed slot) keeps every slot single-owner. -/
theorem swap_unique {r : Residency} (h : SlotsUnique r) {out inn s : Nat} (hs : r out = some s) :
    SlotsUnique (admit (evict r out) inn s) := by
  have hev := evict_unique h out
  have hfree := slot_free_in_flight h hs
  intro a b t ha hb
  simp only [admit, upd] at ha hb
  by_cases hai : a = inn
  · by_cases hbi : b = inn
    · rw [hai, hbi]
    · simp only [hai, hbi, if_true, if_false, Option.some.injEq] at ha hb
      subst ha
      exact absurd hb (hfree b)
  · by_cases hbi : b = inn
    · simp only [hai, hbi, if_true, if_false, Option.some.injEq] at ha hb
      subst hb
      exact absurd ha (hfree a)
    · simp only [hai, hbi, if_false] at ha hb
      exact hev a b t ha hb

/-- After the swap the incoming expert is resident on the victim's slot and the victim is not. -/
theorem swap_moves_slot {r : Residency} {out inn s : Nat} (hne : inn ≠ out) :
    admit (evict r out) inn s inn = some s ∧ admit (evict r out) inn s out = none := by
  have hne' : out ≠ inn := fun h => hne h.symm
  simp [admit, evict, upd, hne']

end Ref.Generate.Residency

