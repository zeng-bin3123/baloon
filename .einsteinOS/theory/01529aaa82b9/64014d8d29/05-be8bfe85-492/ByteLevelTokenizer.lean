/-!
# MODULE: byte_level_tokenizer
SOURCES: tools/strata_tokenizer.py
The GPT-2 byte-level BPE alphabet used by Strata's tokenizer, the special-token classes, and the
merge-validity conditions.

## Source semantics
* `bytes_to_unicode` maps the 188 printable bytes (33..126, 161..172, 174..255) to themselves and the
  other 68 to code points 256..323 in increasing byte order; it is injective with an explicit inverse,
  so `decode(encode(s)) = s` at the byte layer.
* Token type 4 (USER_DEFINED) literals are matched always; type 3 (CONTROL) only with `parse_special`.
* A merge is a pair of tokens; BPE applies the lowest rank first.

## Form template identifiers
* BYTE_TO_UNICODE / UNICODE_TO_BYTE -> byte_alphabet / byte_alphabet_inverse
* special_tokens (types 3, 4) -> literal_token_classes
-/

namespace Strata.Tokenizer

-- § the byte alphabet
/-- Code point assigned to byte `b`: itself when printable, otherwise `256 + (rank among non-printables)`. -/
def byteToCp (b : Nat) : Nat :=
  if ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255)) then b
  else if b ≤ 32 then 256 + b
  else if b ≤ 160 then 256 + b - 94
  else 323

/-- Inverse map from code point back to byte. -/
def cpToByte (c : Nat) : Nat :=
  if c < 256 then c
  else if c < 289 then c - 256
  else if c < 323 then c - 256 + 94
  else 173

/-- Counting: 94 + 12 + 82 = 188 printable bytes. -/
theorem printable_count : (94 : Nat) + 12 + 82 = 188 := by decide

/-- Counting: 33 + 34 + 1 = 68 non-printable bytes, and 188 + 68 = 256. -/
theorem nonprintable_count : (33 : Nat) + 34 + 1 = 68 ∧ 188 + 68 = 256 := by decide

/-- The encoder is injective on bytes. -/
theorem byteToCp_injective (a b : Nat) (ha : a < 256) (hb : b < 256) (h : byteToCp a = byteToCp b) :
    a = b := by
  unfold byteToCp at h
  split_ifs at h <;> omega

/-- Every encoded code point is below 324. -/
theorem byteToCp_lt (b : Nat) (hb : b < 256) : byteToCp b < 324 := by
  unfold byteToCp
  split_ifs <;> omega

/-- Decoding inverts encoding on every byte: the byte layer is lossless. -/
theorem cpToByte_byteToCp (b : Nat) (hb : b < 256) : cpToByte (byteToCp b) = b := by
  unfold cpToByte byteToCp
  split_ifs <;> omega

/-- Printable bytes are fixed points. -/
theorem printable_fixed (b : Nat) (h : ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) : byteToCp b = b := by
  unfold byteToCp
  rw [if_pos h]

/-- Non-printable bytes land at or above 256 (so they never collide with a printable byte). -/
theorem nonprintable_shifted (b : Nat) (hb : b < 256) (h : ¬ ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255))) : 256 ≤ byteToCp b := by
  unfold byteToCp
  split_ifs <;> omega

/-- A string round-trips when each byte does. -/
theorem roundtrip_list (bs : List Nat) (h : ∀ b ∈ bs, b < 256) :
    (bs.map byteToCp).map cpToByte = bs := by
  induction bs with
  | nil => rfl
  | cons a t ih =>
    have ha : a < 256 := h a (by simp)
    have ht : ∀ b ∈ t, b < 256 := fun b hb => h b (by simp [hb])
    simp only [List.map_cons]
    rw [cpToByte_byteToCp a ha, ih ht]

-- § special tokens
/-- Literal classes: GGUF token type 3 = CONTROL, 4 = USER_DEFINED. -/
def matchedAlways (ty : Nat) : Bool := ty == 4

/-- With `parse_special`, both classes are matched. -/
def matchedWithSpecial (ty : Nat) : Bool := ty == 3 || ty == 4

/-- Whatever is matched without `parse_special` is also matched with it. -/
theorem always_subset_special (ty : Nat) (h : matchedAlways ty = true) : matchedWithSpecial ty = true := by
  unfold matchedAlways at h
  unfold matchedWithSpecial
  simp_all

/-- CONTROL tokens such as `<|im_end|>` are NOT matched without `parse_special`. -/
theorem control_needs_parse_special : matchedAlways 3 = false ∧ matchedWithSpecial 3 = true := by decide

/-- USER_DEFINED tokens such as `<think>` are matched either way. -/
theorem user_defined_always : matchedAlways 4 = true ∧ matchedWithSpecial 4 = true := by decide

-- § merges
/-- A merge line must split into exactly two tokens. -/
def validMergeArity (parts : Nat) : Bool := parts == 2

/-- Only two-part merge lines are accepted. -/
theorem merge_arity_iff (n : Nat) : validMergeArity n = true ↔ n = 2 := by
  simp [validMergeArity]

/-- BPE picks the lowest rank: a candidate replaces the best only when strictly smaller. -/
def betterRank (best : Option Nat) (r : Nat) : Bool :=
  match best with
  | none => true
  | some b => decide (r < b)

/-- The first candidate is always taken. -/
theorem betterRank_none (r : Nat) : betterRank none r = true := rfl

/-- Ties keep the earlier (leftmost) candidate. -/
theorem betterRank_tie (r : Nat) : betterRank (some r) r = false := by
  simp [betterRank]

end Strata.Tokenizer
