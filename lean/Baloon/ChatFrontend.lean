-- Baloon.ChatFrontend: thinking-budget rank, partial-tag hold-back, newline stripping,
-- image-pad expansion, context admission and detokenizer delta lengths.
-- Independent implementation proving the same mathematical facts as Ref.Frontend.

namespace Baloon.ChatFrontend

-- § thinking budget → level

/-- Level rank of a thinking budget: 0 = low, 1 = medium, 2 = xhigh. -/
def budgetRank (n : Int) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- More budget never yields a lower level. -/
theorem budgetRank_mono (n m : Int) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank; split_ifs <;> omega

/-- Levels are one of the three template values. -/
theorem budgetRank_lt (n : Int) : budgetRank n < 3 := by
  unfold budgetRank; split_ifs <;> omega

-- § partial-tag hold-back

/-- Longest n in 1..tag.length-1 such that the last n symbols of text equal tag.take n. -/
def holdOne (text tag : List Nat) : Nat :=
  (List.range tag.length).foldl
    (fun b n => if 1 ≤ n ∧ n ≤ text.length ∧ text.drop (text.length - n) = tag.take n
                then max b n else b) 0

/-- Fold of max-style steps stays below a bound if every candidate is below it. -/
theorem fold_bound (tl tl' : Nat) (f : Nat → Nat → Nat) (hf : ∀ b n, f b n ≤ max b n) :
    ∀ (l : List Nat) (b : Nat), (∀ n ∈ l, n < tl) → b < tl → l.foldl f b < tl := by
  intro l
  induction l with
  | nil => intro b _ hb; simpa using hb
  | cons n l ih =>
    intro b hl hb
    simp only [List.foldl_cons]
    apply ih
    · intro x hx; exact hl x (List.mem_cons_of_mem _ hx)
    · have h1 := hf b n
      have h2 : n < tl := hl n List.mem_cons_self
      omega

/-- The held suffix is a proper prefix of the tag: strictly shorter than the tag. -/
theorem holdOne_lt (text tag : List Nat) (h : 0 < tag.length) : holdOne text tag < tag.length := by
  unfold holdOne
  apply fold_bound tag.length 0
  · intro b n; split_ifs <;> omega
  · intro n hn; exact List.mem_range.mp hn
  · exact h

/-- Hold-back splits the buffer into an emitted prefix and a held suffix that recombine exactly. -/
theorem hold_split (buf : List Nat) (keep : Nat) :
    buf.take (buf.length - keep) ++ buf.drop (buf.length - keep) = buf :=
  List.take_append_drop _ _

/-- The emitted prefix is no longer than the buffer. -/
theorem emitted_le (buf : List Nat) (keep : Nat) :
    (buf.take (buf.length - keep)).length ≤ buf.length := by
  simp [List.length_take]

-- § newline stripping of parameter values

/-- Remove one leading newline (code 10) if present. -/
def dropLead : List Nat → List Nat
  | 10 :: rest => rest
  | l => l

/-- Remove one trailing newline if present. -/
def dropTrail (l : List Nat) : List Nat := if l.getLast? = some 10 then l.dropLast else l

/-- Strip at most one newline at each end. -/
def stripNl (l : List Nat) : List Nat := dropTrail (dropLead l)

/-- Leading strip removes at most one symbol. -/
theorem dropLead_length (l : List Nat) : l.length ≤ (dropLead l).length + 1 := by
  cases l with
  | nil => simp [dropLead]
  | cons h t =>
    cases Nat.decEq h 10 with
    | isTrue he => simp [dropLead, he]
    | isFalse hne => simp [dropLead, show h ≠ 10 from hne]

/-- Trailing strip removes at most one symbol. -/
theorem dropTrail_length (l : List Nat) : l.length ≤ (dropTrail l).length + 1 := by
  unfold dropTrail
  split_ifs
  · simp only [List.length_dropLast]; omega
  · omega

/-- Stripping never lengthens a value. -/
theorem dropTrail_le (l : List Nat) : (dropTrail l).length ≤ l.length := by
  unfold dropTrail
  split_ifs
  · simp only [List.length_dropLast]; omega
  · exact Nat.le_refl _

-- § image pad expansion

/-- Expand each pad token into c pads using the per-image counts in order. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
    if t = pad then
      match cs with
      | [] => t :: expand pad ts []
      | c :: cs' => List.replicate c pad ++ expand pad ts cs'
    else t :: expand pad ts cs

/-- With no image counts the ids are unchanged. -/
theorem expand_nil_counts (pad : Nat) : ∀ ids : List Nat, expand pad ids [] = ids := by
  intro ids
  induction ids with
  | nil => rfl
  | cons t ts ih =>
    by_cases h : t = pad <;> simp [expand, h, ih]

/-- Ids without any pad token are unchanged for any counts. -/
theorem expand_no_pad (pad : Nat) :
    ∀ (ids cs : List Nat), (∀ t ∈ ids, t ≠ pad) → expand pad ids cs = ids := by
  intro ids
  induction ids with
  | nil => intro cs _; rfl
  | cons t ts ih =>
    intro cs h
    have ht : t ≠ pad := h t List.mem_cons_self
    have h2 := ih cs (fun x hx => h x (List.mem_cons_of_mem _ hx))
    simp [expand, ht, h2]

-- § context admission

/-- A request is admitted iff prompt plus max tokens fit the engine context. -/
def fits (promptLen maxNew maxCtx : Nat) : Prop := promptLen + maxNew ≤ maxCtx

/-- An admitted request's prompt alone fits the context. -/
theorem fits_prompt (p m c : Nat) (h : fits p m c) : p ≤ c := by
  unfold fits at h; omega

/-- Detokenizer: next delta starts at sent. -/
def deltaLen (textLen sent : Nat) : Nat := textLen - sent

/-- Emitted deltas never overlap: sent + delta = text length when sent is not ahead. -/
theorem delta_exact (textLen sent : Nat) (h : sent ≤ textLen) :
    sent + deltaLen textLen sent = textLen := by
  unfold deltaLen; omega

end Baloon.ChatFrontend
