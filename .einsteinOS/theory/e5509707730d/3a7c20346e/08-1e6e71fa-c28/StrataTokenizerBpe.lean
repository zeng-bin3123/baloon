/-!
# MODULE: strata_tokenizer_bpe
SOURCES: tools/strata_tokenizer.py
Reference byte-level BPE tokenizer (GPT-2 style, `qwen35` pre-tokenizer): the byte to printable-unicode map,
merge-by-lowest-rank, and the special-token matching rules.

## Source semantics
* `bytes_to_unicode`: the 188 bytes in 33..126, 161..172, 174..255 map to themselves; the other 68 map to
  `256 + n` in byte order, so the image is in 0..255 (printable) or 256..323 (shifted).
* BPE repeatedly merges ONE adjacent pair (the lowest-rank one); every merge shortens the sequence by one.
* GGUF token types: type 4 (USER_DEFINED) is matched always; type 3 (CONTROL) only when `parse_special`.
* The vocabulary (248,320) holds the 256 byte tokens, the 247,587 merges' results and the specials.

## Form template identifiers
- bytes_to_unicode               → `b2u` (with `isPrint`, `rank`)
- Tokenizer._bpe merge step      → `mergeAt`
- special_tokens / _always_re    → `matched`
- tokenizer.ggml.tokens / merges → `vocabSize` / `nMerges`
-/

namespace Strata.TokenizerBpe

-- § byte <-> unicode

/-- Bytes that map to themselves. -/
def isPrint (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) ||
    (decide (174 ≤ b) && decide (b ≤ 255))

/-- Number of shifted bytes before `b`. -/
def rank (b : Nat) : Nat := (List.range b).countP (fun x => !isPrint x)

/-- The GPT-2 byte encoder as code points. -/
def b2u (b : Nat) : Nat := if isPrint b then b else 256 + rank b

/-- Exactly 188 bytes are printable and 68 are shifted. -/
theorem printable_counts :
    (List.range 256).countP isPrint = 188 ∧ (List.range 256).countP (fun x => !isPrint x) = 68 := by
  decide +kernel

/-- Printable bytes map into 0..255 and shifted bytes into 256..323, so the two classes never collide. -/
theorem b2u_range : ∀ b, b < 256 → b2u b < 324 := by
  decide +kernel

/-- Printable bytes are fixed points; shifted bytes land strictly above 255. -/
theorem b2u_classes : ∀ b, b < 256 → (isPrint b = true → b2u b = b) ∧ (isPrint b = false → 256 ≤ b2u b) := by
  decide +kernel

/-- The encoder is injective on bytes, which is what makes `decode (encode s) = s` exact. -/
theorem b2u_nodup : ((List.range 256).map b2u).Nodup := by
  decide +kernel

-- § merging

/-- Merge the adjacent pair at index `i` with the combiner `f`. -/
def mergeAt {α : Type} (f : α → α → α) : List α → Nat → List α
  | a :: b :: rest, 0 => f a b :: rest
  | a :: rest, i + 1 => a :: mergeAt f rest i
  | l, _ => l

/-- Each merge of a valid pair shortens the sequence by exactly one, so BPE terminates. -/
theorem mergeAt_length {α : Type} (f : α → α → α) :
    ∀ (l : List α) (i : Nat), i + 1 < l.length → (mergeAt f l i).length + 1 = l.length := by
  intro l
  induction l with
  | nil => intro i h; simp at h
  | cons a rest ih =>
    intro i h
    cases i with
    | zero =>
      cases rest with
      | nil => simp at h
      | cons b rest' => simp [mergeAt]
    | succ i =>
      have h' : i + 1 < rest.length := by
        simp at h
        omega
      have := ih i h'
      simp [mergeAt] <;> omega

/-- An index past the end leaves the sequence unchanged. -/
theorem mergeAt_nil {α : Type} (f : α → α → α) (i : Nat) : mergeAt f [] i = [] := by
  cases i <;> rfl

-- § special tokens

/-- Whether a token of GGUF type `ty` is matched as a literal: type 4 always, type 3 only with
`parse_special`. -/
def matched (ps : Bool) (ty : Nat) : Prop := ty = 4 ∨ (ty = 3 ∧ ps = true)

/-- Everything matched without `parse_special` is also matched with it. -/
theorem matched_monotone (ty : Nat) (h : matched false ty) : matched true ty := by
  rcases h with h | ⟨_, h'⟩
  · exact Or.inl h
  · exact absurd h' (by decide)

/-- Without `parse_special`, CONTROL tokens such as `<|im_end|>` (type 3) are not matched. -/
theorem control_not_matched_by_default : ¬ matched false 3 := by
  intro h
  rcases h with h | ⟨_, h'⟩
  · exact absurd h (by decide)
  · exact absurd h' (by decide)

/-- USER_DEFINED tokens such as `<think>` (type 4) are matched either way. -/
theorem user_defined_always_matched (ps : Bool) : matched ps 4 := by
  exact Or.inl rfl

-- § vocabulary arithmetic

/-- Vocabulary and merge counts of the GGUF. -/
def vocabSize : Nat := 248320
/-- Number of merge rules. -/
def nMerges : Nat := 247587

/-- The 256 byte tokens plus one token per merge fit in the vocabulary, leaving 477 for specials etc. -/
theorem vocab_covers_merges : 256 + nMerges ≤ vocabSize ∧ vocabSize - (256 + nMerges) = 477 := by
  decide

end Strata.TokenizerBpe
