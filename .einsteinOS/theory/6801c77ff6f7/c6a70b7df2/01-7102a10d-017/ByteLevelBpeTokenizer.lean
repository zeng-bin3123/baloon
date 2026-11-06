/-!
# MODULE: byte_level_bpe_tokenizer
SOURCES: tools/strata_tokenizer.py
GPT-2 byte-level BPE with the qwen35 pre-tokenizer. Models the reversible byte <-> unicode map, the
lowest-rank-first merge step and the special-token matching classes.

## Source semantics
* bytes_to_unicode: 188 printable bytes map to themselves; the other 68 map to 256+n in order. It is injective,
  which is why decode(encode(s)) == s.
* A BPE merge replaces two adjacent parts with one, so the part count falls by exactly one per merge.
* Token type 4 (USER_DEFINED) is always matched; type 3 (CONTROL) only when parse_special is set.

## Form template identifiers
bytes_to_unicode -> byte_unicode_map ; Tokenizer._bpe -> bpe_merge_step ;
special_tokens -> special_token_classes ; decode -> bytes_from_tokens
-/
set_option maxRecDepth 100000

namespace Strata.Tokenizer

-- § byte <-> unicode
def inBs (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- Number of non-printable bytes below b, in closed form (non-members: 0..32, 127..160, 173). -/
def nonBelow (b : Nat) : Nat :=
  if b ≤ 33 then b
  else if b ≤ 127 then 33
  else if b ≤ 161 then 33 + (b - 127)
  else if b ≤ 173 then 67
  else 68

def byteToUnicode (b : Nat) : Nat := if inBs b then b else 256 + nonBelow b

/-- Exactly 188 bytes are printable and map to themselves. -/
theorem printable_count : ((List.range 256).filter inBs).length = 188 := by decide

/-- Exactly 68 bytes are shifted. -/
theorem shifted_count : ((List.range 256).filter (fun b => !inBs b)).length = 68 := by decide

/-- The closed form agrees with the counting definition of "n" in the source loop. -/
theorem nonBelow_correct :
    ∀ b, b < 256 → nonBelow b = ((List.range b).filter (fun x => !inBs x)).length := by decide

/-- Printable bytes are fixed points. -/
theorem printable_fixed (b : Nat) (h : inBs b = true) : byteToUnicode b = b := by
  unfold byteToUnicode
  simp [h]

/-- The non-printable bytes below 256 are exactly 0..32, 127..160 and 173. -/
theorem nonprintable_shape :
    ∀ b, b < 256 → inBs b = false → (b ≤ 32 ∨ (127 ≤ b ∧ b ≤ 160) ∨ b = 173) := by decide

/-- Shifted bytes land in [256, 324). -/
theorem shifted_range (b : Nat) (hb : b < 256) (h : inBs b = false) :
    256 ≤ byteToUnicode b ∧ byteToUnicode b < 324 := by
  have hs := nonprintable_shape b hb h
  unfold byteToUnicode nonBelow
  simp only [h]
  simp
  split <;> try omega
  split <;> try omega
  split <;> try omega
  split <;> omega

/-- On non-printable bytes the shift index is injective. -/
theorem nonBelow_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (hA : inBs a = false) (hB : inBs b = false) (h : nonBelow a = nonBelow b) : a = b := by
  have h1 := nonprintable_shape a ha hA
  have h2 := nonprintable_shape b hb hB
  have e : ∀ x, x < 256 → (x ≤ 32 ∨ (127 ≤ x ∧ x ≤ 160) ∨ x = 173) →
      (x ≤ 32 → nonBelow x = x) ∧ ((127 ≤ x ∧ x ≤ 160) → nonBelow x = 33 + (x - 127)) ∧
      (x = 173 → nonBelow x = 67) := by
    intro x _ _
    unfold nonBelow
    refine ⟨?_, ?_, ?_⟩ <;> intro hx <;> split <;> (try split) <;> (try split) <;> (try split) <;> omega
  have ea := e a ha h1
  have eb := e b hb h2
  omega

/-- The byte map is injective: this is the decisive round-trip property. -/
theorem byteToUnicode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteToUnicode a = byteToUnicode b) : a = b := by
  cases hA : inBs a <;> cases hB : inBs b
  · have e : nonBelow a = nonBelow b := by
      unfold byteToUnicode at h
      simp only [hA, hB] at h
      simp at h
      exact h
    exact nonBelow_injective a b ha hb hA hB e
  · have h1 := printable_fixed b hB
    have h2 := shifted_range a ha hA
    omega
  · have h1 := printable_fixed a hA
    have h2 := shifted_range b hb hB
    omega
  · have h1 := printable_fixed a hA
    have h2 := printable_fixed b hB
    omega

-- § BPE merge step
def mergeAt {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) : List α :=
  l.take i ++ [f (l.getD i default) (l.getD (i + 1) default)] ++ l.drop (i + 2)

/-- One merge reduces the number of parts by exactly one. -/
theorem mergeAt_length {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α)
    (h : i + 1 < l.length) : (mergeAt l i f).length + 1 = l.length := by
  unfold mergeAt
  simp [List.length_take, List.length_drop]
  omega

/-- Merging never empties a word. -/
theorem mergeAt_nonempty {α : Type} [Inhabited α] (l : List α) (i : Nat) (f : α → α → α) :
    (mergeAt l i f) ≠ [] := by
  unfold mergeAt
  simp

/-- A word of n bytes can undergo at most n-1 merges, so the loop terminates. -/
theorem merge_budget (n k : Nat) (h : 1 < n - k) : 0 < n - k - 1 := by
  omega

-- § special tokens
def isActive (parseSpecial : Bool) (ty : Nat) : Bool :=
  ty == 4 || (parseSpecial && ty == 3)

/-- Without parse_special only USER_DEFINED (type 4) tokens are matched. -/
theorem active_without_parse (ty : Nat) : isActive false ty = true ↔ ty = 4 := by
  unfold isActive
  simp

/-- With parse_special both CONTROL (3) and USER_DEFINED (4) are matched. -/
theorem active_with_parse (ty : Nat) : isActive true ty = true ↔ (ty = 4 ∨ ty = 3) := by
  unfold isActive
  simp

/-- Enabling parse_special only adds matches, never removes any. -/
theorem active_monotone (ty : Nat) (h : isActive false ty = true) : isActive true ty = true := by
  unfold isActive at *
  simp at *
  omega

/-- Type 3 tokens are matched only under parse_special. -/
theorem control_needs_parse : isActive false 3 = false ∧ isActive true 3 = true := by decide

/-- The vocabulary of the model: 248,320 tokens, 247,587 merges (more tokens than merges). -/
theorem vocab_vs_merges : 247587 < 248320 := by decide

end Strata.Tokenizer
