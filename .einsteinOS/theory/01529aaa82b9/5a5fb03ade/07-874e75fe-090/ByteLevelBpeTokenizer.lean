/-!
# MODULE: byte_level_bpe_tokenizer
`tools/strata_tokenizer.py`: GPT-2 style byte-level BPE.  Each byte maps to a printable code point so any UTF-8
input is representable; decode inverts it exactly.  Special-token matching differs by GGUF token type.

## Source semantics
* bytes 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map to themselves (188 bytes); the other 68 map to 256 + rank.
* the map is injective, so `decode(encode(s)) = s` at the byte layer.
* token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only when `parse_special`.
* BPE merges the LOWEST-rank adjacent pair first.

## Form template identifiers
byteToUnicode   -> bytes_to_unicode
inBs            -> membership in the identity ranges
matched         -> special-token match rule
-/

namespace Strata.ByteLevelBpeTokenizer

-- § the byte alphabet

/-- Byte is in the identity ranges. -/
def inBs (b : Nat) : Bool :=
  (0x21 ≤ b && b ≤ 0x7E) || (0xA1 ≤ b && b ≤ 0xAC) || (0xAE ≤ b && b ≤ 0xFF)

/-- Code point of byte `b`: itself, or 256 plus its rank among the shifted bytes. -/
def byteToUnicode (b : Nat) : Nat :=
  if inBs b then b else 256 + (List.range b).countP (fun x => !inBs x)

/-- Exactly 188 bytes are in the identity ranges. -/
theorem identity_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- The remaining 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun b => !inBs b)).length = 68 := by decide +kernel

/-- Space and newline land on the well-known `Ġ` (288) and `Ċ` (266). -/
theorem known_points : byteToUnicode 32 = 288 ∧ byteToUnicode 10 = 266 := by decide

/-- Printable ASCII maps to itself. -/
theorem ascii_identity : byteToUnicode 65 = 65 ∧ byteToUnicode 97 = 97 := by decide

/-- Identity-range bytes stay below 256 and shifted bytes land at or above 256, so the two never collide. -/
theorem ranges_separated (b : Nat) (h : b < 256) :
    (inBs b = true → byteToUnicode b < 256) ∧ (inBs b = false → 256 ≤ byteToUnicode b) := by
  constructor
  · intro hb; simp [byteToUnicode, hb]; omega
  · intro hb; simp [byteToUnicode, hb]

/-- The byte->code point map has 256 distinct images. -/
theorem byteToUnicode_nodup : ((List.range 256).map byteToUnicode).Nodup := by
  decide +kernel

/-- Every image is below 512 (the alphabet is 256 shifted by at most 256). -/
theorem byteToUnicode_lt : ∀ b ∈ List.range 256, byteToUnicode b < 512 := by
  decide +kernel

/-- Inverse lookup. -/
def unicodeToByte (c : Nat) : Option Nat := (List.range 256).find? (fun b => byteToUnicode b == c)

/-- Decode inverts encode on every byte: the byte-layer round trip. -/
theorem byte_roundtrip : ∀ b ∈ List.range 256, unicodeToByte (byteToUnicode b) = some b := by
  decide +kernel

-- § special-token matching

/-- Token type ids. -/
def tCONTROL : Nat := 3
def tUSER : Nat := 4

/-- A literal is matched as one token when it is USER_DEFINED, or CONTROL with `parse_special`. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  ty == tUSER || (ty == tCONTROL && parseSpecial)

/-- USER_DEFINED literals are matched either way. -/
theorem user_always (p : Bool) : matched tUSER p = true := by
  unfold matched tUSER; simp

/-- CONTROL literals are matched exactly when `parse_special` is set. -/
theorem control_iff (p : Bool) : matched tCONTROL p = p := by
  unfold matched tCONTROL tUSER; cases p <;> simp

/-- Ordinary tokens are never matched literally. -/
theorem ordinary_never (ty : Nat) (h1 : ty ≠ 3) (h2 : ty ≠ 4) (p : Bool) : matched ty p = false := by
  unfold matched tCONTROL tUSER
  simp [h1, h2]

/-- Turning `parse_special` on can only add matches. -/
theorem parse_special_monotone (ty : Nat) : matched ty false = true → matched ty true = true := by
  unfold matched; cases h : (ty == tUSER) <;> simp

-- § BPE rank selection

/-- Minimum of two optional ranks, ignoring absent ones. -/
def optMin : Option Nat → Option Nat → Option Nat
  | none, b => b
  | some x, none => some x
  | some x, some y => some (min x y)

/-- Lowest present rank in a list of optional ranks, or none if all are absent. -/
def bestRank : List (Option Nat) → Option Nat
  | [] => none
  | r :: rs => optMin r (bestRank rs)

/-- If some rank is present, a minimum exists and is at most that rank. -/
theorem bestRank_le (rs : List (Option Nat)) (x : Nat) (hx : some x ∈ rs) :
    ∃ m, bestRank rs = some m ∧ m ≤ x := by
  induction rs with
  | nil => simp at hx
  | cons r rs ih =>
    simp only [List.mem_cons] at hx
    rcases hx with hx | hx
    · subst hx
      cases hb : bestRank rs with
      | none => exact ⟨x, by simp [bestRank, hb, optMin], Nat.le_refl x⟩
      | some m => exact ⟨min x m, by simp [bestRank, hb, optMin], Nat.min_le_left x m⟩
    · obtain ⟨m, hm, hle⟩ := ih hx
      cases r with
      | none => exact ⟨m, by simp [bestRank, hm, optMin], hle⟩
      | some y => exact ⟨min y m, by simp [bestRank, hm, optMin], Nat.le_trans (Nat.min_le_right y m) hle⟩

/-- With no present rank, the result is none. -/
theorem bestRank_none_of_all_none (rs : List (Option Nat)) (h : ∀ r ∈ rs, r = none) :
    bestRank rs = none := by
  induction rs with
  | nil => rfl
  | cons r rs ih =>
    have hr : r = none := h r (by simp)
    have hrs : bestRank rs = none := ih (fun r' hr' => h r' (by simp [hr']))
    simp [bestRank, hr, hrs, optMin]

end Strata.ByteLevelBpeTokenizer
