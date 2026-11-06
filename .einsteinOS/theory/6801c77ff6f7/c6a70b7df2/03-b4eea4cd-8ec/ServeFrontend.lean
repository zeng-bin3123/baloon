/-!
# MODULE: serve_frontend
SOURCES: serve/frontend.py
The request/response layer: mapping client "reasoning effort" spellings and Anthropic thinking budgets onto the
chat template's levels, and the incremental output parser that never emits a partial tag.

## Source semantics
* `EFFORT`: none/off/minimal/disabled/false disable thinking; low, medium stay; high/xhigh/max/maximum are xhigh.
  An unknown spelling is rejected (a 400), not a crash.
* `budget_effort`: under 2048 tokens low, under 8192 medium, otherwise xhigh (monotone in the budget).
* `OutputParser`: states reasoning -> content (on `</think>`), content -> call (on `<tool_call>`), call -> content
  (on `</tool_call>`); it never returns to reasoning.  `_hold` keeps the longest suffix of the buffer that is a
  proper prefix of a tag, so the held length is below the tag length and at most the buffer length.

## Form template identifiers
effortLookup <- `EFFORT`; budgetRank <- `budget_effort`; step <- the parser's state transitions;
holdLen <- `OutputParser._hold` for a single tag.
-/

namespace Strata.ServeFrontend

-- § reasoning effort

/-- The template's thinking levels. -/
inductive Level
  | low
  | medium
  | xhigh
  deriving DecidableEq, Repr

/-- The `EFFORT` table: `some none` disables thinking, `none` is an unknown spelling. -/
def effortLookup : String → Option (Option Level)
  | "none" | "off" | "minimal" | "disabled" | "false" => some none
  | "low" => some (some Level.low)
  | "medium" => some (some Level.medium)
  | "high" | "xhigh" | "max" | "maximum" => some (some Level.xhigh)
  | _ => none

/-- "high" means xhigh, the template's default. -/
theorem high_is_xhigh : effortLookup "high" = some (some Level.xhigh) := by
  decide

/-- "max" is an alias of "high". -/
theorem max_is_high : effortLookup "max" = effortLookup "high" := by
  decide

/-- "off" is an alias of "none". -/
theorem off_is_none : effortLookup "off" = effortLookup "none" := by
  decide

/-- "none" disables thinking. -/
theorem none_disables : effortLookup "none" = some none := by
  decide

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_rejected : effortLookup "ultra" = none := by
  decide

/-- Rank of a thinking level derived from an Anthropic token budget. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- A larger thinking budget never selects a lower level. -/
theorem budgetRank_mono (n m : Nat) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : m < 2048 <;> by_cases h3 : n < 8192 <;>
    by_cases h4 : m < 8192 <;> simp [h1, h2, h3, h4] <;> omega

/-- Budgets under 2048 are low. -/
theorem budget_low (n : Nat) (h : n < 2048) : budgetRank n = 0 := by
  unfold budgetRank
  rw [if_pos h]

/-- Budgets from 8192 up are xhigh. -/
theorem budget_xhigh (n : Nat) (h : 8192 ≤ n) : budgetRank n = 2 := by
  unfold budgetRank
  rw [if_neg (by omega), if_neg (by omega)]

-- § the parser's state machine

/-- Parser states. -/
inductive St
  | reasoning
  | content
  | call
  deriving DecidableEq, Repr

/-- Tags the parser reacts to. -/
inductive Tag
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq, Repr

/-- One transition on a complete tag. -/
def step : St → Tag → St
  | St.reasoning, Tag.thinkEnd => St.content
  | St.content, Tag.callStart => St.call
  | St.call, Tag.callEnd => St.content
  | s, _ => s

/-- Once the parser has left the reasoning state it never returns to it. -/
theorem never_back_to_reasoning (s : St) (t : Tag) (h : step s t = St.reasoning) : s = St.reasoning := by
  cases s <;> cases t <;> first | rfl | (simp [step] at h)

/-- `</think>` moves reasoning to content. -/
theorem think_end_moves : step St.reasoning Tag.thinkEnd = St.content := rfl

/-- A tool call always returns to content. -/
theorem call_returns : step St.call Tag.callEnd = St.content := rfl

-- § holding back partial tags

/-- Generic fold invariant used by the hold-back bounds. -/
theorem foldl_inv (P : Nat → Prop) (f : Nat → Nat → Nat) :
    ∀ (l : List Nat) (init : Nat), (∀ b n, n ∈ l → P b → P (f b n)) → P init → P (l.foldl f init)
  | [], _, _, h => h
  | x :: xs, init, hf, h =>
    foldl_inv P f xs (f init x) (fun b n hn hb => hf b n (by simp [hn]) hb) (hf init x (by simp) h)

/-- Length of the longest proper-prefix-of-`tag` suffix of `text` (`OutputParser._hold`, one tag). -/
def holdLen (text tag : List Char) : Nat :=
  (List.range tag.length).foldl
    (fun best n => if 1 ≤ n ∧ n ≤ text.length ∧ text.drop (text.length - n) = tag.take n then n else best) 0

/-- A held fragment is shorter than the tag: a complete tag is never held back. -/
theorem holdLen_lt (text tag : List Char) (hpos : 0 < tag.length) : holdLen text tag < tag.length := by
  unfold holdLen
  refine foldl_inv (fun b => b < tag.length) _ _ _ ?_ hpos
  intro b n hn hb
  have hn' : n < tag.length := List.mem_range.mp hn
  split
  · exact hn'
  · exact hb

/-- A held fragment is part of the buffer: nothing is held that is not there. -/
theorem holdLen_le (text tag : List Char) : holdLen text tag ≤ text.length := by
  unfold holdLen
  refine foldl_inv (fun b => b ≤ text.length) _ _ _ ?_ (Nat.zero_le _)
  intro b n hn hb
  split
  · rename_i hc
    exact hc.2.1
  · exact hb

end Strata.ServeFrontend
