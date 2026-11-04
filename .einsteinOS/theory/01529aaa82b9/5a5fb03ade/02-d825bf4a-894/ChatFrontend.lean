/-!
# MODULE: chat_frontend
Request normalization and streamed-output parsing between API clients and the engine: reasoning-effort mapping,
Anthropic thinking budgets, and the incremental parser that never emits a partial tag.

## Source semantics
* Client effort spellings collapse onto four levels: none, low, medium, xhigh (`high` means `xhigh`).
* A thinking budget under 2048 tokens is `low`, under 8192 `medium`, otherwise `xhigh`; the map is monotone.
* The parser holds back the longest suffix of the buffer that is a proper prefix of a tag; what it emits plus
  what it holds is exactly what it received.
* The parser state machine starts in reasoning (or content), moves reasoning -> content on `</think>`,
  content -> call on `<tool_call>`, call -> content on `</tool_call>`, and never returns to reasoning.
* A value's single leading and trailing newline are stripped, nothing more.

## Form template identifiers
* `EFFORT` -> `effortLevel`, `budget_effort` -> `budgetRank`, `OutputParser._hold` -> `splitHeld`
* parser `state` -> `PState`, tags -> `Marker`
-/

namespace Strata.ChatFrontend

-- § effort mapping

/-- Levels the template understands. `none` means thinking disabled. -/
inductive Level | off | low | medium | xhigh
  deriving DecidableEq

/-- The client spelling table; `none` result is an unknown value (an HTTP 400). -/
def effortLevel : String → Option Level
  | "none" => some .off
  | "off" => some .off
  | "minimal" => some .off
  | "disabled" => some .off
  | "low" => some .low
  | "medium" => some .medium
  | "high" => some .xhigh
  | "xhigh" => some .xhigh
  | "max" => some .xhigh
  | "maximum" => some .xhigh
  | _ => none

/-- `high` is the model's default `xhigh`. -/
theorem high_is_xhigh : effortLevel "high" = some Level.xhigh := by
  first | decide | rfl | simp [effortLevel]

/-- All the "disable" spellings turn thinking off. -/
theorem disable_spellings : effortLevel "none" = some Level.off ∧ effortLevel "off" = some Level.off ∧
    effortLevel "minimal" = some Level.off ∧ effortLevel "disabled" = some Level.off := by
  first | decide | rfl | simp [effortLevel]

/-- An unknown spelling is rejected rather than guessed. -/
theorem unknown_rejected : effortLevel "turbo" = none := by
  first | decide | rfl | simp [effortLevel]

/-- Rank of a level in the order low < medium < xhigh. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- A larger thinking budget never selects a lower level. -/
theorem budgetRank_mono (a b : Nat) (h : a ≤ b) : budgetRank a ≤ budgetRank b := by
  unfold budgetRank
  by_cases h1 : a < 2048 <;> by_cases h2 : a < 8192 <;> by_cases h3 : b < 2048 <;>
    by_cases h4 : b < 8192 <;> simp [h1, h2, h3, h4] <;> omega

/-- Ranks are 0, 1 or 2. -/
theorem budgetRank_lt (n : Nat) : budgetRank n < 3 := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : n < 8192 <;> simp [h1, h2]

/-- The 2048 and 8192 boundaries land where the docstring says. -/
theorem budget_boundaries : budgetRank 2047 = 0 ∧ budgetRank 2048 = 1 ∧ budgetRank 8191 = 1 ∧
    budgetRank 8192 = 2 := by
  decide

-- § hold-back

/-- Split a buffer into the part emitted now and the held suffix of length `keep`. -/
def splitHeld (buf : List Char) (keep : Nat) : List Char × List Char :=
  (buf.take (buf.length - keep), buf.drop (buf.length - keep))

/-- Nothing is lost: emitted plus held is exactly the buffer. -/
theorem splitHeld_append (buf : List Char) (keep : Nat) :
    (splitHeld buf keep).1 ++ (splitHeld buf keep).2 = buf := by
  unfold splitHeld
  exact List.take_append_drop _ _

/-- The held part is at most `keep` characters. -/
theorem splitHeld_held_le (buf : List Char) (keep : Nat) :
    (splitHeld buf keep).2.length ≤ keep := by
  unfold splitHeld
  simp only [List.length_drop]
  omega

/-- When `keep` is a proper prefix length (below the tag length), the held part is shorter than the tag. -/
theorem splitHeld_shorter_than_tag (buf : List Char) (keep tagLen : Nat) (h : keep < tagLen) :
    (splitHeld buf keep).2.length < tagLen := by
  have := splitHeld_held_le buf keep
  omega

-- § parser state machine

/-- Parser states. -/
inductive PState | reasoning | content | call
  deriving DecidableEq

/-- Tag boundaries the parser reacts to. -/
inductive Marker | thinkEnd | callStart | callEnd
  deriving DecidableEq

/-- Transition on finding a tag in the current state; other combinations leave the state alone. -/
def step : PState → Marker → PState
  | .reasoning, .thinkEnd => .content
  | .content, .callStart => .call
  | .call, .callEnd => .content
  | s, _ => s

/-- The parser never re-enters reasoning: it is only reachable by staying in it. -/
theorem step_reasoning_only_from_reasoning (s : PState) (m : Marker) (h : step s m = .reasoning) :
    s = .reasoning := by
  revert h
  cases s <;> cases m <;> decide

/-- A tool call always returns to content. -/
theorem call_end_returns_content : step .call .callEnd = .content := by decide

/-- `</think>` ends reasoning. -/
theorem think_end_leaves_reasoning : step .reasoning .thinkEnd = .content := by decide

-- § value trimming

/-- Drop at most one leading newline. -/
def dropLeadNL : List Char → List Char
  | '\n' :: rest => rest
  | l => l

/-- Drop at most one trailing newline. -/
def dropTrailNL (l : List Char) : List Char :=
  if l.getLast? = some '\n' then l.dropLast else l

/-- Trimming only shortens a value, by at most one character per side. -/
theorem dropLeadNL_length (l : List Char) : l.length ≤ (dropLeadNL l).length + 1 := by
  match l with
  | [] => simp [dropLeadNL]
  | '\n' :: rest => simp [dropLeadNL]
  | c :: rest =>
      by_cases h : c = '\n'
      · subst h
        simp [dropLeadNL]
      · simp [dropLeadNL, h]

/-- Dropping the last element removes exactly one. -/
theorem dropTrailNL_length (l : List Char) : l.length ≤ (dropTrailNL l).length + 1 := by
  unfold dropTrailNL
  split
  · simp [List.length_dropLast] <;> omega
  · omega

end Strata.ChatFrontend
