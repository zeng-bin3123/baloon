/-!
# MODULE: byte_level_bpe_tokenizer
SOURCES: tools/strata_tokenizer.py
The reference GPT-2 byte-level BPE tokenizer (qwen35 pre-tokenizer) and the export of the pack's
`tokenizer/` directory. Its decisive property is the lossless byte layer, so `decode ∘ encode = id`.

## Source semantics
* `bytes_to_unicode`: bytes 33..126, 161..172, 174..255 (188 of them) map to themselves; the other 68
  map to `256 + n` in increasing byte order. The map is a bijection onto its image.
* `_bpe`: repeatedly merge the adjacent pair of LOWEST rank (leftmost on equal rank, by strict `<`) until
  no pair has a rank. Each merge concatenates two pieces, so the concatenation of pieces is invariant.
* Special literals: GGUF types 3 (CONTROL) and 4 (USER_DEFINED). Type 4 is matched always; types 3 and 4
  when `parse_special`. The always-set is therefore a subset of the special-set.
Tokens are modelled as byte lists; ranks as a partial function on pairs.

## Form template identifiers
bytes_to_unicode → byteToCode (closed form) / pyByteTable (the Python loop), UNICODE_TO_BYTE → codeToByte,
Tokenizer._bpe → bpeLoop, one merge → mergeAt, best-pair search → pairRanks / pickMin,
special_tokens / _always_re → specialIdx / alwaysIdx
-/

namespace Strata.Tokenizer

-- § the byte ↔ code map

/-- Bytes that map to themselves: 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF. -/
def printableList : List Nat := List.range' 33 94 ++ List.range' 161 12 ++ List.range' 174 82

/-- The same set as a predicate. -/
def isPrintable (b : Nat) : Bool :=
  (33 ≤ b && b ≤ 126) || (161 ≤ b && b ≤ 172) || (174 ≤ b && b ≤ 255)

/-- The Python loop, verbatim in shape: printable pairs first, then the shifted ones in byte order. -/
def pyByteTable : List (Nat × Nat) :=
  let rest := (List.range 256).filter (fun b => !(printableList.contains b))
  printableList.map (fun b => (b, b)) ++ (rest.zip (List.range rest.length)).map (fun (b, n) => (b, 256 + n))

/-- Rank of a non-printable byte among the non-printables: 0..32, then 127..160, then 173. -/
def shiftRank (b : Nat) : Nat := if b ≤ 32 then b else if b ≤ 160 then b - 94 else 67

/-- Closed form of `bytes_to_unicode` (as code points). -/
def byteToCode (b : Nat) : Nat := if isPrintable b then b else 256 + shiftRank b

/-- Inverse of `shiftRank`. -/
def unshiftRank (r : Nat) : Nat := if r ≤ 32 then r else if r ≤ 66 then r + 94 else 173

/-- `UNICODE_TO_BYTE`, closed form. -/
def codeToByte (c : Nat) : Nat := if c < 256 then c else unshiftRank (c - 256)

set_option maxRecDepth 20000 in
/-- 188 printable bytes, so 68 are shifted. -/
theorem printable_count : printableList.length = 188 ∧
    ((List.range 256).filter (fun b => !(printableList.contains b))).length = 68 := by decide

set_option maxRecDepth 20000 in
/-- The predicate and the list agree on every byte. -/
theorem isPrintable_eq_list : (List.range 256).all (fun b => isPrintable b == printableList.contains b) = true := by
  decide

set_option maxRecDepth 20000 in
/-- The closed form agrees with the Python construction on all 256 bytes, and the table covers 256 bytes. -/
theorem closed_form_matches_python :
    pyByteTable.length = 256 ∧ pyByteTable.all (fun (b, c) => byteToCode b == c) = true := by
  decide

/-- Every byte gets a code below 256 + 68. -/
theorem byteToCode_lt (b : Nat) (hb : b < 256) : byteToCode b < 324 := by
  unfold byteToCode shiftRank
  repeat' split
  all_goals omega

/-- The decode map inverts the encode map on every byte: the byte layer is lossless. -/
theorem codeToByte_byteToCode (b : Nat) (hb : b < 256) : codeToByte (byteToCode b) = b := by
  unfold byteToCode codeToByte
  by_cases hp : isPrintable b = true
  · simp only [hp, if_true, hb]
  · simp only [hp, Bool.false_eq_true, if_false]
    have hnp : ¬ ((33 ≤ b ∧ b ≤ 126) ∨ (161 ≤ b ∧ b ≤ 172) ∨ (174 ≤ b ∧ b ≤ 255)) := by
      intro h; apply hp; unfold isPrintable; simp only [Bool.or_eq_true, Bool.and_eq_true,
        decide_eq_true_eq]; exact or_assoc.mpr h
    have h256 : ¬ (256 + shiftRank b < 256) := by omega
    simp only [h256, if_false, Nat.add_sub_cancel_left]
    unfold shiftRank unshiftRank
    repeat' split
    all_goals omega

/-- Hence `bytes_to_unicode` is injective: distinct bytes never share a character. -/
theorem byteToCode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteToCode a = byteToCode b) : a = b := by
  have h1 := codeToByte_byteToCode a ha
  have h2 := codeToByte_byteToCode b hb
  rw [h] at h1
  omega

/-- Decoding a byte-mapped string recovers the bytes. -/
theorem decode_encode_bytes (bs : List Nat) (hb : ∀ b ∈ bs, b < 256) :
    (bs.map byteToCode).map codeToByte = bs := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [List.map_cons]
    rw [codeToByte_byteToCode b (hb b (List.mem_cons_self ..))]
    rw [ih (fun x hx => hb x (List.mem_cons_of_mem _ hx))]

-- § BPE merging

/-- Merge pieces `i` and `i+1`. -/
def mergeAt : List (List Nat) → Nat → List (List Nat)
  | a :: b :: rest, 0 => (a ++ b) :: rest
  | a :: rest, i + 1 => a :: mergeAt rest i
  | l, _ => l

/-- A merge never changes the underlying byte string. -/
theorem mergeAt_flatten : ∀ (l : List (List Nat)) (i : Nat), (mergeAt l i).flatten = l.flatten
  | [], _ => rfl
  | [_], 0 => rfl
  | a :: b :: rest, 0 => by simp [mergeAt, List.append_assoc]
  | a :: rest, i + 1 => by simp [mergeAt, mergeAt_flatten rest i]

/-- A valid merge removes exactly one piece (so the BPE loop terminates). -/
theorem mergeAt_length : ∀ (l : List (List Nat)) (i : Nat), i + 1 < l.length →
    (mergeAt l i).length + 1 = l.length
  | [], _, h => by simp at h
  | [_], _, h => by simp at h
  | a :: b :: rest, 0, _ => by simp [mergeAt]
  | a :: rest, i + 1, h => by
    simp only [List.length_cons] at h
    have := mergeAt_length rest i (by omega)
    simp only [mergeAt, List.length_cons]
    omega

/-- Ranks of adjacent pairs (`None` = no merge rule). -/
def pairRanks (rk : List Nat → List Nat → Option Nat) : List (List Nat) → List (Option Nat)
  | a :: b :: rest => rk a b :: pairRanks rk (b :: rest)
  | _ => []

/-- The scan of `_bpe`: keep the strictly lowest rank, so the leftmost wins ties. -/
def pickMin : List (Option Nat) → Nat → Option (Nat × Nat) → Option (Nat × Nat)
  | [], _, best => best
  | none :: rs, i, best => pickMin rs (i + 1) best
  | some r :: rs, i, best =>
    pickMin rs (i + 1)
      (match best with
       | some (bi, br) => if r < br then some (i, r) else some (bi, br)
       | none => some (i, r))

/-- The chosen rank is no larger than the incoming best and than every ranked pair: lowest rank first,
never left-to-right list order. -/
theorem pickMin_le : ∀ (rs : List (Option Nat)) (i : Nat) (best : Option (Nat × Nat)) (j r : Nat),
    pickMin rs i best = some (j, r) →
      (∀ b, best = some b → r ≤ b.2) ∧ (∀ r', some r' ∈ rs → r ≤ r')
  | [], _, best, j, r, h => by
    simp only [pickMin] at h
    refine ⟨fun b hb => ?_, fun _ h' => by simp at h'⟩
    rw [h] at hb; simp only [Option.some.injEq] at hb; subst hb; exact Nat.le_refl _
  | none :: rs, i, best, j, r, h => by
    simp only [pickMin] at h
    obtain ⟨h1, h2⟩ := pickMin_le rs (i + 1) best j r h
    refine ⟨h1, fun r' hr' => ?_⟩
    simp only [List.mem_cons, reduceCtorEq, false_or] at hr'
    exact h2 r' hr'
  | some r0 :: rs, i, best, j, r, h => by
    simp only [pickMin] at h
    cases best with
    | none =>
      obtain ⟨h1, h2⟩ := pickMin_le rs (i + 1) (some (i, r0)) j r h
      refine ⟨fun b hb => by simp at hb, fun r' hr' => ?_⟩
      simp only [List.mem_cons, Option.some.injEq] at hr'
      cases hr' with
      | inl he => subst he; exact h1 (i, r') rfl
      | inr hm => exact h2 r' hm
    | some bb =>
      obtain ⟨bi, br⟩ := bb
      by_cases hlt : r0 < br
      · simp only [hlt, if_true] at h
        obtain ⟨h1, h2⟩ := pickMin_le rs (i + 1) (some (i, r0)) j r h
        have hr0 := h1 (i, r0) rfl
        refine ⟨fun b hb => ?_, fun r' hr' => ?_⟩
        · simp only [Option.some.injEq] at hb; subst hb; simp only at hr0 ⊢; omega
        · simp only [List.mem_cons, Option.some.injEq] at hr'
          cases hr' with
          | inl he => subst he; exact hr0
          | inr hm => exact h2 r' hm
      · simp only [hlt, if_false] at h
        obtain ⟨h1, h2⟩ := pickMin_le rs (i + 1) (some (bi, br)) j r h
        have hbr := h1 (bi, br) rfl
        refine ⟨fun b hb => ?_, fun r' hr' => ?_⟩
        · simp only [Option.some.injEq] at hb; subst hb; exact hbr
        · simp only [List.mem_cons, Option.some.injEq] at hr'
          cases hr' with
          | inl he => subst he; simp only at hbr; omega
          | inr hm => exact h2 r' hm

/-- The BPE loop (fuel = number of pieces suffices, each merge removes one). -/
def bpeLoop (rk : List Nat → List Nat → Option Nat) : Nat → List (List Nat) → List (List Nat)
  | 0, parts => parts
  | fuel + 1, parts =>
    match pickMin (pairRanks rk parts) 0 none with
    | none => parts
    | some (i, _) => bpeLoop rk fuel (mergeAt parts i)

/-- BPE is lossless whatever the merge table: the pieces always concatenate back to the word.
(So `decode(encode(s)) == s` checks the byte layer, not merge order — merge order is checked by
`pickMin_le`.) -/
theorem bpeLoop_flatten (rk : List Nat → List Nat → Option Nat) :
    ∀ (fuel : Nat) (parts : List (List Nat)), (bpeLoop rk fuel parts).flatten = parts.flatten := by
  intro fuel
  induction fuel with
  | zero => intro parts; rfl
  | succ f ih =>
    intro parts
    simp only [bpeLoop]
    split
    · rfl
    · rw [ih, mergeAt_flatten]

/-- Starting from single bytes, the BPE output spells the word. -/
theorem bpe_word_roundtrip (rk : List Nat → List Nat → Option Nat) (word : List Nat) :
    (bpeLoop rk word.length (word.map (fun b => [b]))).flatten = word := by
  rw [bpeLoop_flatten]
  induction word with
  | nil => rfl
  | cons b w ih => simp [ih]

-- § special literals

/-- Token ids matched only with parse_special or always: types 3 and 4. -/
def specialIdx (types : List Nat) : List Nat :=
  (List.range types.length).filter (fun i => types[i]! == 3 || types[i]! == 4)

/-- Token ids matched always: type 4 (USER_DEFINED) only. -/
def alwaysIdx (types : List Nat) : List Nat :=
  (List.range types.length).filter (fun i => types[i]! == 4)

/-- `parse_special = False` never matches anything `parse_special = True` would not. -/
theorem always_subset_special (types : List Nat) (i : Nat) (h : i ∈ alwaysIdx types) :
    i ∈ specialIdx types := by
  unfold alwaysIdx at h
  unfold specialIdx
  simp only [List.mem_filter, List.mem_range, beq_iff_eq, Bool.or_eq_true] at h ⊢
  exact ⟨h.1, Or.inr h.2⟩

/-- CONTROL tokens (type 3) are never matched without parse_special. -/
theorem control_not_always (types : List Nat) (i : Nat) (h : types[i]! = 3) : i ∉ alwaysIdx types := by
  unfold alwaysIdx
  simp only [List.mem_filter, List.mem_range, beq_iff_eq, h, not_and]
  intro _ h'
  exact absurd h' (by decide)

end Strata.Tokenizer
