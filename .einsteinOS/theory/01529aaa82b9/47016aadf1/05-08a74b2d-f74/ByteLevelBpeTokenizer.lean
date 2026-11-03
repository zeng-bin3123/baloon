/-!
# MODULE: byte_level_bpe_tokenizer
The GPT-2 byte-level BPE tokenizer: the byte-to-unicode bijection, the single-step merge length law,
special-token matching classes and vocabulary/merge bookkeeping.

## Source semantics
* Bytes 33..126, 161..172 and 174..255 map to themselves; the other 68 bytes map to `256 + n` in byte order.
  The map is a bijection onto 256 code points, so every UTF-8 input is representable and decode inverts encode.
* A BPE step replaces an adjacent pair by one merged token, shortening the word by exactly one.
* Token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only with `parse_special`.
* Every merge names two vocabulary tokens; there are fewer merges than tokens.

## Form template identifiers
bytes_to_unicode -> byte_map, _bpe -> bpe_merge, special_tokens -> special_literals,
Tokenizer.encode/decode -> encode/decode.
-/

namespace Strata.ByteLevelBpe

set_option maxRecDepth 100000

-- § byte <-> unicode

/-- A byte maps to itself iff it is in the printable ranges 33..126, 161..172, 174..255. -/
def printable (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below `b`. -/
def nonPrintBefore (b : Nat) : Nat := ((List.range b).filter (fun x => !printable x)).length

/-- The code point assigned to byte `b`. -/
def enc (b : Nat) : Nat := if printable b then b else 256 + nonPrintBefore b

/-- Exactly 68 of the 256 bytes are shifted. -/
theorem shifted_count : nonPrintBefore 256 = 68 := by decide

/-- The 188 printable bytes map to themselves. -/
theorem printable_count : ((List.range 256).filter printable).length = 188 := by decide

/-- Shifted bytes land in `256 .. 323`; printable bytes stay below 256. -/
theorem enc_range : ∀ b : Fin 256, enc b.val < 324 := by decide

/-- The map is injective on bytes, hence a bijection onto its 256-element image. -/
theorem enc_nodup : ((List.range 256).map enc).Nodup := by decide

/-- Decode searches for the byte whose code point matches. -/
def dec (c : Nat) : Nat := ((List.range 256).find? (fun b => enc b == c)).getD 0

/-- Decode inverts encode on every byte: the byte layer is lossless. -/
theorem dec_enc : ∀ b : Fin 256, dec (enc b.val) = b.val := by decide

/-- Space (32) and newline (10) are shifted into the 256+ range, which is why tokens print as visible glyphs. -/
theorem space_newline_shifted : printable 32 = false ∧ printable 10 = false := by decide

-- § BPE step

/-- Merge the adjacent pair at `i` into the single token `m`. -/
def mergeAt (l : List Nat) (i m : Nat) : List Nat := l.take i ++ m :: l.drop (i + 2)

/-- A merge step shortens the word by exactly one, so BPE terminates. -/
theorem mergeAt_length (l : List Nat) (i m : Nat) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length = l.length - 1 := by
  simp only [mergeAt, List.length_append, List.length_take, List.length_drop, List.length_cons]
  omega

/-- A step strictly decreases the length of a word with at least two parts. -/
theorem mergeAt_decreases (l : List Nat) (i m : Nat) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length < l.length := by
  have := mergeAt_length l i m h
  omega

/-- A word of one part (or none) is never merged: a pair needs two parts. -/
theorem no_pair_in_short (n : Nat) (h : n < 2) : ¬ (0 + 2 ≤ n) := by omega

-- § special tokens

/-- Whether a token of type `ty` is matched as a literal, given `parse_special`. -/
def matched (ty : Nat) (parse : Bool) : Bool := ty == 4 || (ty == 3 && parse)

/-- USER_DEFINED tokens (type 4) are matched with or without `parse_special`. -/
theorem user_defined_always (parse : Bool) : matched 4 parse = true := by
  simp [matched]

/-- CONTROL tokens (type 3) are matched only with `parse_special`. -/
theorem control_needs_parse : matched 3 false = false ∧ matched 3 true = true := by decide

/-- Ordinary tokens (type 1) are never literal-matched. -/
theorem normal_never (parse : Bool) : matched 1 parse = false := by
  simp [matched]

-- § vocabulary

/-- The qwen35 vocabulary size. -/
def vocabSize : Nat := 248320
/-- The qwen35 merge count. -/
def nMerges : Nat := 247587

/-- There are fewer merges than vocabulary entries. -/
theorem merges_lt_vocab : nMerges < vocabSize := by decide

/-- The 256 byte tokens plus the merges fit inside the vocabulary. -/
theorem bytes_plus_merges_fit : 256 + nMerges ≤ vocabSize := by decide

/-- A merge line must split into exactly two parts. -/
def validMerge (nParts : Nat) : Bool := nParts == 2

/-- A merge with any other number of parts is rejected. -/
theorem bad_merge_rejected (n : Nat) (h : n ≠ 2) : validMerge n = false := by
  simp [validMerge, h]

/-- Token ids must be inside the vocabulary when decoding. -/
def idInRange (i : Int) (vocab : Nat) : Bool := decide (0 ≤ i ∧ i < (vocab : Int))

/-- A negative id is rejected. -/
theorem negative_id_rejected (i : Int) (vocab : Nat) (h : i < 0) : idInRange i vocab = false := by
  simp only [idInRange, decide_eq_false_iff_not]
  omega

end Strata.ByteLevelBpe
