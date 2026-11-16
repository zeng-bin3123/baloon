-- Baloon.Tokenizer: BPE byte-alphabet encoding/decoding, injectivity and round-trip
-- proofs, plus special-token matching rules.
-- Independent implementation proving the same mathematical facts as Ref.Tokenizer.

namespace Baloon.Tokenizer

def byteToCp (b : Nat) : Nat :=
  if ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255)) then b
  else if (b ≤ 32) then 256 + b
  else if (b ≤ 160) then 256 + b - 94
  else 323

def cpToByte (c : Nat) : Nat :=
  if (c < 256) then c
  else if (c < 289) then c - 256
  else if (c < 323) then c - 256 + 94
  else 173

theorem printable_count : (94 : Nat) + 12 + 82 = 188 := by { decide }
theorem nonprintable_count : (33 : Nat) + 34 + 1 = 68 ∧ 188 + 68 = 256 := by { decide }

theorem byteToCp_injective (a b : Nat) (ha : a < 256) (hb : b < 256) (h : byteToCp a = byteToCp b) :
    a = b := by unfold byteToCp at h; split_ifs at h <;> omega

theorem byteToCp_lt (b : Nat) (hb : b < 256) : byteToCp b < 324 := by
  unfold byteToCp; split_ifs <;> omega

theorem cpToByte_byteToCp (b : Nat) (hb : b < 256) : cpToByte (byteToCp b) = b := by
  unfold cpToByte byteToCp; split_ifs <;> omega

theorem printable_fixed (b : Nat) (h : ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) :
    byteToCp b = b := by unfold byteToCp; rw [if_pos h]

theorem nonprintable_shifted (b : Nat) (hb : b < 256)
    (h : ¬ ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) :
    256 ≤ byteToCp b := by unfold byteToCp; split_ifs <;> omega

theorem roundtrip_list (bs : List Nat) (h : ∀ b ∈ bs, b < 256) :
    (bs.map byteToCp).map cpToByte = bs := by
  induction bs with
  | nil => rfl
  | cons a t ih =>
    have ha : a < 256 := h a (by simp)
    have ht : ∀ b ∈ t, b < 256 := fun b hb => h b (by simp [hb])
    simp only [List.map_cons]; rw [cpToByte_byteToCp a ha, ih ht]

def matchedAlways (ty : Nat) : Bool := (ty == 4)
def matchedWithSpecial (ty : Nat) : Bool := (ty == 3 || ty == 4)

theorem always_subset_special (ty : Nat) (h : matchedAlways ty = true) :
    matchedWithSpecial ty = true := by
  unfold matchedAlways at h; unfold matchedWithSpecial; simp_all

-- ── special-token classes and BPE merge rules ────────────────────────────────

theorem control_needs_parse_special : matchedAlways 3 = false ∧ matchedWithSpecial 3 = true := by { decide }
theorem user_defined_always : matchedAlways 4 = true ∧ matchedWithSpecial 4 = true := by { decide }

/-- A merge line must split into exactly two tokens. -/
def validMergeArity (parts : Nat) : Bool := (parts == 2)

theorem merge_arity_iff (n : Nat) : validMergeArity n = true ↔ n = 2 := by { simp [validMergeArity] }
/-- BPE picks the lowest rank: a candidate replaces the best only when strictly smaller. -/
def betterRank (best : Option Nat) (r : Nat) : Bool :=
  match best with
  | none => (true)
  | some b => (decide (r < b))

/-- The first candidate is always taken. -/
theorem betterRank_none (r : Nat) : betterRank none r = true := by { rfl }

/-- Ties keep the earlier (leftmost) candidate. -/
theorem betterRank_tie (r : Nat) : betterRank (some r) r = false := by { simp [betterRank] }

/-- A strictly lower rank displaces the current best. -/
theorem betterRank_lower (best r : Nat) (h : r < best) : betterRank (some best) r = true := by
  simp [betterRank, h]

end Baloon.Tokenizer
