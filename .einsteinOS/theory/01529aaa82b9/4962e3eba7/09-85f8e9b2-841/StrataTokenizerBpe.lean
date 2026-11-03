/-!
# MODULE: strata_tokenizer_bpe
Structure of `tools/strata_tokenizer.py`: GPT-2 byte-level BPE with the `qwen35` pre-tokenizer.

## Source semantics
* `bytes_to_unicode`: 188 bytes (33..126, 161..172, 174..255) map to themselves, the other 68 map to
  256 + rank; so space -> 288 and newline -> 266.
* BPE merges the adjacent pair of LOWEST rank first; each merge shortens the word by one and keeps the
  concatenation unchanged.
* Special tokens: type 4 (USER_DEFINED) matched always, type 3 (CONTROL) only with parse_special.
* decode(encode(s)) = s holds at the byte layer when the byte map has a left inverse.

## Form template identifiers
* inBs / byteToUnicode  <- `bytes_to_unicode`
* mergeAt               <- `parts[best:best+2] = [parts[best] + parts[best+1]]`
* matched               <- `_always_re` / `_special_re` selection in `encode`
-/

namespace Strata.Tokenizer

-- § Byte alphabet

/-- Bytes that map to themselves. -/
def inBs (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) ||
    (decide (174 ≤ b) && decide (b ≤ 255))

/-- Number of shifted bytes below `b`. -/
def rank (b : Nat) : Nat := ((List.range b).filter (fun x => !inBs x)).length

/-- GPT-2 byte -> code point. -/
def byteToUnicode (b : Nat) : Nat := if inBs b then b else 256 + rank b

set_option maxRecDepth 100000 in
/-- Exactly 188 bytes are printable as themselves. -/
theorem identity_byte_count : ((List.range 256).filter inBs).length = 188 := by
  decide

set_option maxRecDepth 100000 in
/-- Exactly 68 bytes are shifted. -/
theorem shifted_byte_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by
  decide

set_option maxRecDepth 100000 in
/-- The shifted bytes receive ranks 0..67 exactly once, so they land on 256..323 without collision. -/
theorem shifted_ranks_are_0_to_67 :
    ((List.range 256).filter (fun x => !inBs x)).map rank = List.range 68 := by
  decide

/-- Space and newline map to the familiar characters U+0120 and U+010A. -/
theorem space_newline_map : byteToUnicode 32 = 288 ∧ byteToUnicode 10 = 266 := by
  decide

/-- Letters map to themselves. -/
theorem letters_identity : byteToUnicode 65 = 65 ∧ byteToUnicode 97 = 97 := by
  decide

-- § BPE merge step

/-- Merge the pair at positions i and i+1 of a word held as a list of pieces. -/
def mergeAt {α : Type} : List (List α) → Nat → List (List α)
  | a :: b :: t, 0 => (a ++ b) :: t
  | x :: t, i + 1 => x :: mergeAt t i
  | l, _ => l

/-- A merge keeps the underlying symbol sequence unchanged. -/
theorem mergeAt_flatten {α : Type} :
    ∀ (i : Nat) (l : List (List α)), i + 1 < l.length → (mergeAt l i).flatten = l.flatten
  | 0, [], h => by simp at h
  | 0, [_], h => by simp at h
  | 0, a :: b :: t, _ => by simp [mergeAt, List.append_assoc]
  | i + 1, [], h => by simp at h
  | i + 1, x :: t, h => by
    have ih := mergeAt_flatten i t (by simp at h; omega)
    simp [mergeAt, ih]

/-- A merge shortens the word by exactly one piece. -/
theorem mergeAt_length {α : Type} :
    ∀ (i : Nat) (l : List (List α)), i + 1 < l.length → (mergeAt l i).length + 1 = l.length
  | 0, [], h => by simp at h
  | 0, [_], h => by simp at h
  | 0, a :: b :: t, _ => by simp [mergeAt]
  | i + 1, [], h => by simp at h
  | i + 1, x :: t, h => by
    have ih := mergeAt_length i t (by simp at h; omega)
    simp [mergeAt]
    omega

/-- Byte-level round trip: a left inverse of the byte map recovers any byte string. -/
theorem decode_encode_bytes (m u : Nat → Nat) (h : ∀ b, b < 256 → u (m b) = b) :
    ∀ bs : List Nat, (∀ b ∈ bs, b < 256) → (bs.map m).map u = bs
  | [], _ => rfl
  | b :: t, hb => by
    have hb0 : u (m b) = b := h b (hb b (by simp))
    have ih := decode_encode_bytes m u h t (fun x hx => hb x (List.mem_cons_of_mem b hx))
    simp [hb0, ih]

-- § Special tokens

/-- Whether a token of GGUF type `ty` is matched as a literal. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  ty == 4 || (ty == 3 && parseSpecial)

/-- USER_DEFINED tokens (type 4) are matched with or without parse_special. -/
theorem user_defined_always : matched 4 true = true ∧ matched 4 false = true := by decide

/-- CONTROL tokens (type 3) are matched only with parse_special. -/
theorem control_needs_flag : matched 3 true = true ∧ matched 3 false = false := by decide

/-- Ordinary types (e.g. normal=1, byte=6) are never literal matches. -/
theorem normal_never : matched 1 true = false ∧ matched 6 true = false := by decide

/-- Vocabulary sizes of the shipped tokenizer: more tokens than merges. -/
theorem vocab_sizes : 247587 < 248320 := by decide

end Strata.Tokenizer
