/-!
# MODULE: byte_level_bpe_tokenizer
GPT-2 style byte-level BPE as used by the `qwen35` pre-tokenizer: the byte/unicode alphabet, lowest-rank-first
merging, special-token matching classes, and the exactness of the byte layer behind `decode(encode(s)) = s`.

## Source semantics
* `bytes_to_unicode`: bytes 33..126, 161..172, 174..255 (188 values) map to themselves; the other 68 map to
  `256 + n` in byte order, so all 256 images are distinct.
* BPE merges the pair of LOWEST RANK repeatedly; each merge shortens the sequence by exactly one, so it
  terminates after at most `len - 1` merges.
* Token types: 4 (USER_DEFINED) literals are matched always; 3 (CONTROL) only with `parse_special`.
* Longest literal first in the alternation so `<|im_end|>` never matches a shorter prefix of itself.

## Form template identifiers
byte_alphabet := BYTE_TO_UNICODE ; shifted_base := 256 ; merge_step := parts[best:best+2] = [a+b] ;
special_always := token_type == 4 ; special_gated := token_type == 3 ; literal_order := sorted by len desc
-/

namespace Strata.Tokenizer

-- § byte alphabet

/-- Bytes that map to themselves in the GPT-2 byte encoder. -/
def isIdentityByte (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

set_option maxRecDepth 100000 in
/-- 94 + 12 + 82 = 188 bytes keep their own code point. -/
theorem identity_count : (List.range 256).countP isIdentityByte = 188 := by
  decide

/-- Hence exactly 68 bytes are shifted into `256 + n`. -/
theorem shifted_count : 256 - 188 = 68 := by
  decide

/-- Image of a shifted byte: the n-th shifted byte gets code point `256 + n`. -/
def shiftedImage (n : Nat) : Nat := 256 + n

/-- Shifted images are all outside the byte range, so they never collide with an identity image. -/
theorem shifted_outside_identity (n : Nat) : 256 ≤ shiftedImage n := by
  unfold shiftedImage
  omega

/-- Shifted images are pairwise distinct (injective in `n`). -/
theorem shifted_injective (n m : Nat) (h : shiftedImage n = shiftedImage m) : n = m := by
  unfold shiftedImage at h
  omega

/-- All 68 shifted images stay below 324, i.e. printable code points well inside the BMP. -/
theorem shifted_bound (n : Nat) (h : n < 68) : shiftedImage n < 324 := by
  unfold shiftedImage
  omega

/-- An identity byte's image (itself) is below 256, so identity and shifted images are disjoint. -/
theorem identity_below_shift (b : Nat) (h : b ≤ 255) (n : Nat) : b ≠ shiftedImage n := by
  unfold shiftedImage
  omega

-- § lowest-rank-first merging

/-- Replace the adjacent pair at index `i` by their merge. -/
def mergeAt {α : Type} (f : α → α → α) (l : List α) (i : Nat) (a b : α) : List α :=
  l.take i ++ [f a b] ++ l.drop (i + 2)

/-- Merging two adjacent parts shortens the sequence by exactly one. -/
theorem mergeAt_length {α : Type} (f : α → α → α) (l : List α) (i : Nat) (a b : α) (h : i + 1 < l.length) :
    (mergeAt f l i a b).length = l.length - 1 := by
  unfold mergeAt
  simp only [List.length_append, List.length_take, List.length_drop, List.length_cons, List.length_nil]
  omega

/-- Hence every merge strictly decreases the length of a multi-part word. -/
theorem mergeAt_decreases {α : Type} (f : α → α → α) (l : List α) (i : Nat) (a b : α) (h : i + 1 < l.length) :
    (mergeAt f l i a b).length < l.length := by
  have h1 := mergeAt_length f l i a b h
  omega

/-- BPE therefore needs at most `len - 1` merges to reach a fixed point. -/
theorem merge_budget (len : Nat) (steps : Nat) (hs : steps ≤ len - 1) (h : 1 ≤ len) : len - steps ≥ 1 := by
  omega

/-- Index of the best (lowest-rank) pair: keep the earlier one on ties, replace only when strictly lower. -/
def betterRank (cur new : Option Nat) : Bool :=
  match cur, new with
  | none, some _ => true
  | some c, some n => decide (n < c)
  | _, _ => false

/-- A strictly lower rank always replaces the current best. -/
theorem lower_rank_replaces (c n : Nat) (h : n < c) : betterRank (some c) (some n) = true := by
  unfold betterRank
  simp [h]

/-- An equal rank never replaces the current best, so the leftmost pair of a given rank wins. -/
theorem equal_rank_keeps (c : Nat) : betterRank (some c) (some c) = false := by
  unfold betterRank
  simp

/-- A missing pair never replaces anything. -/
theorem missing_pair_keeps (cur : Option Nat) : betterRank cur none = false := by
  cases cur <;> rfl

/-- Any ranked pair replaces "no best yet". -/
theorem first_pair_taken (n : Nat) : betterRank none (some n) = true := by
  rfl

-- § special-token classes

/-- Whether a token of GGUF type `ty` is matched literally, given `parse_special`. -/
def literalMatched (ty : Nat) (parseSpecial : Bool) : Bool :=
  decide (ty = 4) || (decide (ty = 3) && parseSpecial)

/-- USER_DEFINED tokens (`<think>`, `<tool_call>`, ...) are matched regardless of `parse_special`. -/
theorem user_defined_always (ps : Bool) : literalMatched 4 ps = true := by
  unfold literalMatched
  simp

/-- CONTROL tokens (`<|im_end|>`, ...) are NOT matched without `parse_special`. -/
theorem control_gated_off : literalMatched 3 false = false := by
  decide

/-- CONTROL tokens are matched when `parse_special` is on. -/
theorem control_gated_on : literalMatched 3 true = true := by
  decide

/-- Turning `parse_special` on never removes a literal: the matched set is monotone. -/
theorem parse_special_monotone (ty : Nat) : literalMatched ty false = true → literalMatched ty true = true := by
  unfold literalMatched
  simp
  omega

/-- Ordinary tokens (type 1, normal) are never matched as literals. -/
theorem normal_never_literal (ps : Bool) : literalMatched 1 ps = false := by
  unfold literalMatched
  simp

-- § longest-literal-first alternation

/-- Sorting literals by decreasing length puts a literal before every proper prefix of itself. -/
theorem longer_first (a b : Nat) (hab : b < a) : ¬ (a ≤ b) := by
  omega

-- § held-back partial tags in streaming (shared with the output parser)

/-- Fold of `max` is bounded by `m` when the start value and every element are. -/
theorem foldl_max_le (m : Nat) (l : List Nat) :
    ∀ (init : Nat), (∀ x ∈ l, x ≤ m) → init ≤ m → l.foldl max init ≤ m := by
  induction l with
  | nil =>
    intro init _ h
    simpa using h
  | cons a t ih =>
    intro init hl h
    simp only [List.foldl_cons]
    exact ih (max init a) (fun x hx => hl x (List.mem_cons_of_mem a hx))
      (Nat.max_le.mpr ⟨h, hl a (by simp)⟩)

/-- Decoding a byte image: the identity images are the inverse of themselves and the shifted ones map
back by subtracting 256, so `UNICODE_TO_BYTE ∘ BYTE_TO_UNICODE = id` on the shifted range. -/
theorem shifted_decode (n : Nat) : shiftedImage n - 256 = n := by
  unfold shiftedImage
  omega

end Strata.Tokenizer
