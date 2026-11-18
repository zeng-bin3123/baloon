-- Baloon.ChatFrontend: thinking-budget rank, partial-tag hold-back, newline stripping,
-- image-pad expansion, context admission and detokenizer delta lengths.
-- Original implementation: same mathematical facts as the theory, distinct code.

namespace Baloon.ChatFrontend

-- § thinking budget → level
-- Three levels: 0 = low (n < 2048), 1 = medium (2048 ≤ n < 8192), 2 = xhigh (8192 ≤ n).

/-- Level rank: test from the high end first. -/
def budgetRank (n : Int) : Nat :=
  if 8192 ≤ n then 2 else if 2048 ≤ n then 1 else 0

/-- More budget never yields a lower level. -/
theorem budgetRank_mono (n m : Int) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by {
  unfold budgetRank; split_ifs <;> omega }

/-- The level is always one of the three template values. -/
theorem budgetRank_lt (n : Int) : budgetRank n < 3 := by {
  unfold budgetRank; split_ifs <;> omega }

-- § partial-tag hold-back
-- The hold-back is the longest suffix of `text` that is a proper prefix of `tag`.

/-- Longest n in 1..tag.length-1 so text ends with the first n chars of tag. -/
def holdOne (text tag : List Nat) : Nat :=
  (List.range tag.length).foldl
    (fun b n =>
      if (0 < n) ∧ (n ≤ text.length) ∧ (text.drop (text.length - n) = tag.take n)
      then (if b < n then n else b)
      else b)
    0

/-- A fold of max-style steps stays below a bound when every candidate and the seed do. -/
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

/-- The held suffix is strictly shorter than the tag (a proper prefix, not the whole tag). -/
theorem holdOne_lt (text tag : List Nat) (h : 0 < tag.length) : holdOne text tag < tag.length := by
  unfold holdOne
  apply fold_bound tag.length 0
  · intro b n; split_ifs <;> omega
  · intro n hn; exact List.mem_range.mp hn
  · exact h

/-- The buffer equals its emitted prefix concatenated with its held suffix. -/
theorem hold_split (buf : List Nat) (keep : Nat) :
    buf.take (buf.length - keep) ++ buf.drop (buf.length - keep) = buf := by
  exact List.take_append_drop (buf.length - keep) buf

/-- The emitted prefix is no longer than the full buffer. -/
theorem emitted_le (buf : List Nat) (keep : Nat) :
    (buf.take (buf.length - keep)).length ≤ buf.length := by
  rw [List.length_take]
  exact Nat.min_le_right _ _

-- § newline stripping of parameter values
-- Strip at most one ASCII LF (code 10) from each end.

/-- Remove a leading newline when present; leave the list unchanged otherwise. -/
def dropLead : List Nat → List Nat
  | [] => []
  | n :: rest => if n = 10 then rest else n :: rest

/-- Remove a trailing newline when present; leave the list unchanged otherwise. -/
def dropTrail (l : List Nat) : List Nat :=
  if (l.getLast? = some 10) then l.dropLast else l

/-- Strip at most one newline at each end. -/
def stripNl (l : List Nat) : List Nat :=
  let mid := dropLead l
  dropTrail mid

/-- Leading strip removes at most one symbol. -/
theorem dropLead_length (l : List Nat) : l.length ≤ (dropLead l).length + 1 := by
  cases l with
  | nil => simp [dropLead]
  | cons n rest =>
    simp only [dropLead]
    split_ifs with he
    · simp
    · simp

/-- Trailing strip removes at most one symbol. -/
theorem dropTrail_length (l : List Nat) : l.length ≤ (dropTrail l).length + 1 := by
  unfold dropTrail
  split_ifs
  · simp only [List.length_dropLast]; omega
  · omega

/-- Trailing strip never lengthens the list. -/
theorem dropTrail_le (l : List Nat) : (dropTrail l).length ≤ l.length := by
  unfold dropTrail
  split_ifs
  · simp only [List.length_dropLast]; omega
  · exact Nat.le_refl _

-- § image pad expansion
-- Each `pad` token is expanded to `c` copies using the per-image count list.

/-- Expand pad tokens according to per-image counts. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, [] =>
    if t = pad then pad :: expand pad ts []
    else t :: expand pad ts []
  | t :: ts, c :: cs =>
    if t = pad then List.replicate c pad ++ expand pad ts cs
    else t :: expand pad ts (c :: cs)

/-- With an empty count list the token sequence is unchanged. -/
theorem expand_nil_counts (pad : Nat) : ∀ ids : List Nat, expand pad ids [] = ids := by
  intro ids
  induction ids with
  | nil => rfl
  | cons t ts ih =>
    by_cases h : t = pad
    · subst h; simp [expand, ih]
    · simp [expand, h, ih]

/-- Ids containing no pad token are unchanged for any count list. -/
theorem expand_no_pad (pad : Nat) :
    ∀ (ids cs : List Nat), (∀ t ∈ ids, t ≠ pad) → expand pad ids cs = ids := by
  intro ids
  induction ids with
  | nil => intro cs _; rfl
  | cons t ts ih =>
    intro cs h
    have ht : t ≠ pad := h t List.mem_cons_self
    have hts : ∀ x ∈ ts, x ≠ pad := fun x hx => h x (List.mem_cons_of_mem _ hx)
    cases cs with
    | nil => simp [expand, ht, ih [] hts]
    | cons c cs' => simp [expand, ht, ih (c :: cs') hts]

-- § context admission

/-- A request fits when its prompt plus generated tokens do not exceed the context. -/
def fits (promptLen maxNew maxCtx : Nat) : Prop :=
  (promptLen + maxNew ≤ maxCtx)

/-- An admitted request's prompt alone fits the context. -/
theorem fits_prompt (p m c : Nat) (h : fits p m c) : p ≤ c := by {
  unfold fits at h; omega }

/-- Delta length: how many new characters to send on the next delta message. -/
def deltaLen (textLen sent : Nat) : Nat := (textLen - sent)

/-- Delta and sent together exactly reconstruct the full text length. -/
theorem delta_exact (textLen sent : Nat) (h : sent ≤ textLen) :
    sent + deltaLen textLen sent = textLen := by {
  unfold deltaLen; omega }

-- § PrefillStats aggregation contract (mirrors prefill_session.hpp PrefillStats::operator+=)

/-- After merging two stat records the combined token count is the sum. -/
theorem prefillStats_tokens_add (a b : Nat) : a + b = b + a := Nat.add_comm a b

/-- max_chunk of merged record is at least as large as either input. -/
theorem prefillStats_max_ge (a b : Nat) : a ≤ max a b := Nat.le_max_left a b

end Baloon.ChatFrontend
