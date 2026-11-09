/-!
# MODULE: tokenizer_byte_level_bpe
SOURCES: tools/strata_tokenizer.py
Model of the GPT-2 byte-to-unicode map, BPE merge step, special-token classes and the decode homomorphism that
underlies the `decode(encode(s)) == s` round trip.

## Source semantics
bytes_to_unicode sends 188 printable bytes to themselves and the other 68 to 256+n in byte order; it is injective.
A BPE step replaces an adjacent pair by one merged token, shrinking the list by exactly one, so merging terminates.
Token type 4 (USER_DEFINED) is matched always; type 3 (CONTROL) only with parse_special. decode concatenates the
byte strings of its tokens, so it distributes over concatenation of id lists.

## Form template identifiers
`bytes_to_unicode` -> `cp`; `_bpe` merge -> `mergeAt`; `special_tokens`/`_always_re` -> `matchedAlways`/`matchedSpecial`;
`decode` -> `decodeBytes`; `_encode_matching` splice -> `split_decode`.
-/

namespace StrataTokenizer

-- § byte <-> unicode

/-- Bytes that map to themselves: 33..126, 161..172, 174..255. -/
def inBs (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below b (the `n` counter of the source). -/
def rank (b : Nat) : Nat := ((List.range b).filter (fun x => !inBs x)).length

/-- Code point of byte b: itself if printable, else 256 plus its rank among the others. -/
def cp (b : Nat) : Nat := if inBs b then b else 256 + rank b

/-- 188 bytes map to themselves. -/
theorem inBs_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by decide +kernel

/-- Printable bytes are fixed points. -/
theorem cp_printable (b : Nat) (h : inBs b = true) : cp b = b := by
  unfold cp; simp [h]

/-- The map is injective on bytes: all 256 code points are distinct, so decode inverts encode. -/
theorem cp_nodup : ((List.range 256).map cp).Nodup := by decide +kernel

/-- Shifted code points lie in 256..323. -/
theorem cp_bound : ∀ b ∈ List.range 256, cp b < 324 := by decide +kernel

-- § BPE merge

/-- Replace the adjacent pair at i by a single merged token. -/
def mergeAt {α : Type} (l : List α) (i : Nat) (x : α) : List α := l.take i ++ x :: l.drop (i + 2)

/-- A merge shrinks the list by exactly one, so BPE terminates after fewer than len steps. -/
theorem mergeAt_length {α : Type} (l : List α) (i : Nat) (x : α) (h : i + 1 < l.length) :
    (mergeAt l i x).length + 1 = l.length := by
  unfold mergeAt
  simp [List.length_take, List.length_drop]
  omega

/-- A list of length at most one has no pair left to merge. -/
theorem no_pair_when_short {α : Type} (l : List α) (h : l.length ≤ 1) : ¬ (0 + 1 < l.length) := by omega

-- § special tokens

/-- Type 4 USER_DEFINED tokens are matched with or without parse_special. -/
def matchedAlways (ty : Nat) : Prop := ty = 4
/-- Types 3 and 4 are matched when parse_special is set. -/
def matchedSpecial (ty : Nat) : Prop := ty = 3 ∨ ty = 4

/-- Anything matched always is matched under parse_special. -/
theorem always_subset_special (ty : Nat) (h : matchedAlways ty) : matchedSpecial ty := by
  unfold matchedAlways matchedSpecial at *; omega

/-- CONTROL tokens are matched only under parse_special. -/
theorem control_needs_flag (ty : Nat) (h : ty = 3) : matchedSpecial ty ∧ ¬ matchedAlways ty := by
  unfold matchedAlways matchedSpecial; omega

-- § decode homomorphism

/-- Decoding concatenates the byte strings of the tokens. -/
def decodeBytes (toks : Nat → List Nat) (ids : List Nat) : List Nat := ids.flatMap toks

/-- Decode distributes over concatenation of id lists. -/
theorem decode_append (toks : Nat → List Nat) (a b : List Nat) :
    decodeBytes toks (a ++ b) = decodeBytes toks a ++ decodeBytes toks b := by
  simp [decodeBytes]

/-- Splicing a special token between two encoded pieces decodes to piece, token bytes, piece. -/
theorem split_decode (toks : Nat → List Nat) (a b : List Nat) (sp : Nat) :
    decodeBytes toks (a ++ [sp] ++ b) = decodeBytes toks a ++ toks sp ++ decodeBytes toks b := by
  simp [decodeBytes]

/-- Round trip is preserved by splicing: if both pieces round-trip, so does the spliced text. -/
theorem split_roundtrip (toks : Nat → List Nat) (ea eb ta tb : List Nat) (sp : Nat)
    (ha : decodeBytes toks ea = ta) (hb : decodeBytes toks eb = tb) :
    decodeBytes toks (ea ++ [sp] ++ eb) = ta ++ toks sp ++ tb := by
  rw [split_decode, ha, hb]

/-- Decoding the empty id list gives the empty string, so the empty corpus entry round-trips. -/
theorem decode_nil (toks : Nat → List Nat) : decodeBytes toks [] = [] := by
  simp [decodeBytes]

end StrataTokenizer
