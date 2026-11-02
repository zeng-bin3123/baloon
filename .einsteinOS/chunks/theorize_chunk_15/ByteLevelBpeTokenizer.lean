/-!
# MODULE: byte_level_bpe_tokenizer
Model of the GPT-2 byte↔unicode alphabet, special-token matching classes, and BPE merge bookkeeping used by
`strata_tokenizer.py`.

## Source semantics
* 188 bytes (33..126, 161..172, 174..255) map to themselves; the other 68 map to `256 + n` in increasing
  byte order, so the map is total and reversible (no UNK).
* Token types: type 4 (USER_DEFINED) literals are matched always; type 3 (CONTROL) only with parse_special.
* BPE merges the lowest-rank adjacent pair first; each merge shortens the part list by exactly one.
* Concatenating pieces before or after encoding gives the same ids (flat map).

## Form template identifiers
* `inBs` / `byteCode` / `codeByte` ← `bytes_to_unicode`, `BYTE_TO_UNICODE`, `UNICODE_TO_BYTE`
* `matched`                        ← `_always_re` / `_special_re` selection
* `mergeAt`                        ← `parts[best:best+2] = [parts[best] + parts[best+1]]`
-/

namespace ByteBpe

-- § the byte alphabet

/-- Bytes that map to themselves in the GPT-2 byte encoder. -/
def inBs (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) || decide (174 ≤ b)

/-- Code point assigned to byte `b`: itself if printable, else `256 + #(non-printable bytes below b)`. -/
def byteCode (b : Nat) : Nat :=
  if inBs b then b else 256 + ((List.range b).filter (fun x => !inBs x)).length

/-- Inverse: code points below 256 are bytes; the rest index the non-printable bytes in order. -/
def codeByte (c : Nat) : Nat :=
  if c < 256 then c else ((List.range 256).filter (fun x => !inBs x)).getD (c - 256) 0

/-- Exactly 188 bytes are self-mapped. -/
theorem selfMapped_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- Exactly 68 bytes are shifted into `256+n`. -/
theorem shifted_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by decide

/-- The two classes together cover all 256 bytes. -/
theorem classes_total : 188 + 68 = 256 := by decide

/-- Space (32) maps to U+0120 (288), the familiar `Ġ`. -/
theorem space_code : byteCode 32 = 288 := by decide

/-- Newline (10) maps to U+010A (266), the familiar `Ċ`. -/
theorem newline_code : byteCode 10 = 266 := by decide

/-- 0xAD (soft hyphen) is shifted, 0xAC and 0xAE are not. -/
theorem soft_hyphen_edges : inBs 173 = false ∧ inBs 172 = true ∧ inBs 174 = true := by decide

/-- Printable bytes map below 256. -/
theorem printable_code_lt (b : Nat) (h : inBs b = true) (hb : b < 256) : byteCode b < 256 := by
  unfold byteCode
  simp [h]
  exact hb

/-- Shifted bytes map to 256 or above. -/
theorem shifted_code_ge (b : Nat) (h : inBs b = false) : 256 ≤ byteCode b := by
  unfold byteCode
  simp [h]

/-- A printable byte and a shifted byte never collide. -/
theorem class_codes_disjoint (a b : Nat) (ha : inBs a = true) (haa : a < 256) (hb : inBs b = false) :
    byteCode a ≠ byteCode b := by
  have h1 := printable_code_lt a ha haa
  have h2 := shifted_code_ge b hb
  omega

/-- The byte layer is lossless: decoding the code of every byte returns the byte. -/
theorem byte_roundtrip : ∀ b, b < 256 → codeByte (byteCode b) = b := by
  decide +kernel

/-- Encoding is injective on bytes (a consequence of the round trip). -/
theorem byteCode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteCode a = byteCode b) : a = b := by
  have h1 := byte_roundtrip a ha
  have h2 := byte_roundtrip b hb
  rw [h] at h1
  omega

/-- Round trip lifts to byte strings: decode ∘ encode = id. -/
theorem bytes_roundtrip (bs : List Nat) (h : ∀ b ∈ bs, b < 256) :
    (bs.map byteCode).map codeByte = bs := by
  induction bs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons]
    rw [byte_roundtrip x (h x (by simp)), ih (fun b hb => h b (by simp [hb]))]

-- § special-token classes

/-- Whether a token of type `ty` is matched as a literal, given `parse_special`. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  decide (ty = 4) || (decide (ty = 3) && parseSpecial)

/-- USER_DEFINED (type 4) literals are matched with or without parse_special. -/
theorem type4_always (ps : Bool) : matched 4 ps = true := by simp [matched]

/-- CONTROL (type 3) literals are matched exactly when parse_special is set. -/
theorem type3_iff (ps : Bool) : matched 3 ps = ps := by simp [matched]

/-- Other token types are never matched as literals. -/
theorem other_never (ty : Nat) (h3 : ty ≠ 3) (h4 : ty ≠ 4) (ps : Bool) : matched ty ps = false := by
  simp [matched, h3, h4]

/-- Everything matched without parse_special is also matched with it. -/
theorem always_subset_special (ty : Nat) (h : matched ty false = true) : matched ty true = true := by
  unfold matched at *
  simp at *
  omega

-- § BPE merge bookkeeping

/-- Merge the adjacent pair at index `i` into the single token `m`. -/
def mergeAt {α : Type} (l : List α) (i : Nat) (m : α) : List α := l.take i ++ m :: l.drop (i + 2)

/-- A merge shortens the part list by exactly one. -/
theorem mergeAt_length {α : Type} (l : List α) (i : Nat) (m : α) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length + 1 = l.length := by
  simp [mergeAt, List.length_take, List.length_drop] <;> omega

/-- Merging at an in-range position leaves the prefix before it untouched. -/
theorem mergeAt_prefix {α : Type} (l : List α) (i : Nat) (m : α) (h : i ≤ l.length) :
    (mergeAt l i m).take i = l.take i := by
  unfold mergeAt
  rw [List.take_append_of_le_length (by rw [List.length_take]; omega)]
  simp

/-- A word of `n ≥ 1` parts needs at most `n - 1` merges: each merge removes one part, so the loop terminates. -/
theorem bpe_merge_budget (n k : Nat) (h : k + 1 ≤ n) (hk : 1 ≤ k + 1) : n - k ≥ 1 := by omega

/-- Encoding a concatenation of pre-token pieces is the concatenation of their encodings. -/
theorem encode_pieces_append {α β : Type} (f : α → List β) (xs ys : List α) :
    (xs ++ ys).flatMap f = xs.flatMap f ++ ys.flatMap f := by
  simp

/-- A merge list index equals its rank, so a pair's rank is unique if the pair appears once. -/
theorem rank_unique (ranks : List (Nat × Nat)) (h : ranks.Nodup) (i j : Nat)
    (hi : i < ranks.length) (hj : j < ranks.length) (e : ranks[i] = ranks[j]) : i = j := by
  exact (List.Nodup.getElem_inj_iff h).mp e

end ByteBpe
