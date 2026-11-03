/-!
# MODULE: serve_frontend
Request normalization and the incremental output parser of the serving frontend: reasoning-effort mapping,
Anthropic thinking-budget buckets, partial-tag hold-back and the reasoning/content/tool-call state machine.

## Source semantics
Client effort names map onto {none, low, medium, xhigh}; "high" means xhigh. Budgets under 2K tokens are
low, under 8K medium, else xhigh (monotone in the budget). A streamed delta never exposes a partial tag:
the held suffix is a proper prefix of the tag. The parser only moves reasoning -> content -> call -> content
and never returns to reasoning.

## Form template identifiers
- `Level`, `effortOf` <- EFFORT / effort_kwargs
- `budgetRank` <- budget_effort
- `holdFrom`, `hold` <- OutputParser._hold
- `PState`, `next` <- OutputParser.state transitions
-/

namespace Strata.ServeFrontend

-- § effort mapping

inductive Level where
  | low
  | medium
  | xhigh
  deriving DecidableEq

/-- Client spelling -> template level (`none` inside means "thinking disabled"); `none` outside = unknown. -/
def effortOf (s : String) : Option (Option Level) :=
  if s = "none" ∨ s = "off" ∨ s = "minimal" ∨ s = "disabled" ∨ s = "false" then some none
  else if s = "low" then some (some Level.low)
  else if s = "medium" then some (some Level.medium)
  else if s = "high" ∨ s = "xhigh" ∨ s = "max" ∨ s = "maximum" then some (some Level.xhigh)
  else none

/-- "high" means the model's default top level, xhigh. -/
theorem high_is_xhigh : effortOf "high" = some (some Level.xhigh) := by decide

/-- "off" disables thinking. -/
theorem off_disables : effortOf "off" = some none := by decide

/-- "low" maps to low. -/
theorem low_is_low : effortOf "low" = some (some Level.low) := by decide

/-- An unrecognized effort is rejected (a 400), not guessed. -/
theorem unknown_rejected : effortOf "turbo" = none := by decide

-- § budget buckets

/-- Rank of the level chosen from an Anthropic thinking budget: 0 low, 1 medium, 2 xhigh. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- More thinking budget never selects a lower level. -/
theorem budgetRank_mono (a b : Nat) (h : a ≤ b) : budgetRank a ≤ budgetRank b := by
  unfold budgetRank
  repeat' split
  all_goals omega

/-- Ranks are at most 2 (three levels). -/
theorem budgetRank_le (n : Nat) : budgetRank n ≤ 2 := by
  unfold budgetRank
  repeat' split
  all_goals omega

-- § partial-tag hold-back

/-- Longest suffix of `text` (up to `n` long) that is a prefix of `tag`. -/
def holdFrom (text tag : List Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => if text.drop (text.length - (n + 1)) = tag.take (n + 1) then n + 1 else holdFrom text tag n

/-- Hold-back candidates start at `len(tag) - 1`: only proper prefixes are held. -/
def hold (text tag : List Nat) : Nat := holdFrom text tag (tag.length - 1)

/-- The scan never reports more than its starting bound. -/
theorem holdFrom_le (text tag : List Nat) : ∀ n, holdFrom text tag n ≤ n := by
  intro n
  induction n with
  | zero => simp [holdFrom]
  | succ n ih =>
    unfold holdFrom
    split
    · exact Nat.le_refl _
    · exact Nat.le_succ_of_le ih

/-- A non-zero hold is exactly a suffix of the text equal to a prefix of the tag. -/
theorem holdFrom_spec (text tag : List Nat) : ∀ n, 0 < holdFrom text tag n →
    text.drop (text.length - holdFrom text tag n) = tag.take (holdFrom text tag n) := by
  intro n
  induction n with
  | zero =>
    intro h
    simp [holdFrom] at h
  | succ n ih =>
    intro h
    by_cases c : text.drop (text.length - (n + 1)) = tag.take (n + 1)
    · simp only [holdFrom, if_pos c]
      exact c
    · simp only [holdFrom, if_neg c] at h ⊢
      exact ih h

/-- A complete tag is never held: the hold is strictly shorter than the tag. -/
theorem hold_lt (text tag : List Nat) (h : 0 < tag.length) : hold text tag < tag.length := by
  have := holdFrom_le text tag (tag.length - 1)
  unfold hold
  omega

-- § parser state machine

inductive PState where
  | reasoning
  | content
  | call
  deriving DecidableEq

inductive Ev where
  | thinkEnd
  | callStart
  | callEnd
  | other

/-- State after one recognized marker. -/
def next : PState → Ev → PState
  | PState.reasoning, Ev.thinkEnd => PState.content
  | PState.content, Ev.callStart => PState.call
  | PState.call, Ev.callEnd => PState.content
  | s, _ => s

/-- Once out of reasoning, the parser never returns to it. -/
theorem next_never_reasoning (s : PState) (e : Ev) (h : s ≠ PState.reasoning) :
    next s e ≠ PState.reasoning := by
  cases s <;> cases e <;> simp [next] at *

/-- A tool call always returns to content after its closing tag. -/
theorem call_end_returns_content : next PState.call Ev.callEnd = PState.content := rfl

/-- `</think>` ends reasoning. -/
theorem think_end_to_content : next PState.reasoning Ev.thinkEnd = PState.content := rfl

/-- The one newline strip on a parameter value removes at most one character. -/
def stripOne (l : List Nat) : List Nat := if l.head? = some 10 then l.tail else l

/-- Stripping never lengthens the value and removes at most one element. -/
theorem stripOne_length (l : List Nat) : l.length ≤ (stripOne l).length + 1 ∧ (stripOne l).length ≤ l.length := by
  unfold stripOne
  split
  · simp
    omega
  · omega

end Strata.ServeFrontend
