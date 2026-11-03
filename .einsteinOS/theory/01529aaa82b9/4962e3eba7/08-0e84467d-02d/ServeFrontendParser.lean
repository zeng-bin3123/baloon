/-!
# MODULE: serve_frontend_parser
The request/response text layer: reasoning-effort normalization, Anthropic thinking-budget bucketing, the
incremental output parser's state machine, and the tag-holdback rule that stops a partial tag reaching clients.

## Source semantics
* `EFFORT` maps client spellings onto four template settings: none (thinking off), low, medium, xhigh;
  "high" means xhigh. Unknown spellings are rejected (400), never crashed on.
* `budget_effort`: budget under 2048 is low, under 8192 is medium, otherwise xhigh; monotone in the budget.
* `OutputParser` states are reasoning, content, call. Tags move reasoning to content (`</think>`), content to
  call (`<tool_call>`), call to content (`</tool_call>`); the parser never returns to reasoning.
* `_hold(text, tags)` is the length of the longest suffix of `text` that is a PROPER prefix of a tag, so it is
  always shorter than the tag and a complete tag is never held back.

## Form template identifiers
* `effortLevel` : `EFFORT` lookup after `strip().lower()`
* `budgetLevel` : `budget_effort`
* `St`, `Tag`, `step` : `OutputParser.state` and its transitions
* `holdAux text tag k` : the inner loop of `_hold`, scanning n = k downto 1
-/

namespace Strata.ServeFrontend

-- § reasoning effort

inductive Level
  | low
  | medium
  | xhigh
  deriving DecidableEq, Repr

/-- Rank of a level for ordering. -/
def Level.rank : Level → Nat
  | .low => 0
  | .medium => 1
  | .xhigh => 2

/-- `EFFORT`: `none` = thinking disabled, `some l` = a template level, outer `none` = unknown (a 400). -/
def effortLevel : String → Option (Option Level)
  | "none" => some none
  | "off" => some none
  | "minimal" => some none
  | "disabled" => some none
  | "low" => some (some .low)
  | "medium" => some (some .medium)
  | "high" => some (some .xhigh)
  | "xhigh" => some (some .xhigh)
  | "max" => some (some .xhigh)
  | "maximum" => some (some .xhigh)
  | _ => none

/-- "high" and "xhigh" are the same template level. -/
theorem high_is_xhigh : effortLevel "high" = effortLevel "xhigh" := by
  simp [effortLevel]

/-- "max" and "maximum" are xhigh too. -/
theorem max_is_xhigh : effortLevel "max" = some (some Level.xhigh) := by
  simp [effortLevel]

/-- "none" turns thinking off. -/
theorem none_disables : effortLevel "none" = some none := by
  simp [effortLevel]

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_rejected : effortLevel "bogus" = none := by
  simp [effortLevel]

/-- `budget_effort`: under 2K low, under 8K medium, else xhigh. -/
def budgetLevel (n : Nat) : Level := if n < 2048 then .low else if n < 8192 then .medium else .xhigh

/-- A larger thinking budget never selects a lower level. -/
theorem budget_monotone (n m : Nat) (h : n ≤ m) : (budgetLevel n).rank ≤ (budgetLevel m).rank := by
  unfold budgetLevel
  by_cases a : n < 2048 <;> by_cases b : m < 2048 <;> by_cases c : n < 8192 <;> by_cases d : m < 8192 <;>
    simp [a, b, c, d, Level.rank] <;> omega

/-- Boundary behaviour: 2047 is low, 2048 is medium, 8191 is medium, 8192 is xhigh. -/
theorem budget_boundaries :
    budgetLevel 2047 = .low ∧ budgetLevel 2048 = .medium ∧ budgetLevel 8191 = .medium ∧
      budgetLevel 8192 = .xhigh := by
  decide

-- § output parser state machine

inductive St
  | reasoning
  | content
  | call
  deriving DecidableEq, Repr

inductive Tag
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq, Repr

/-- The parser's transition when it finds a complete tag in its current state. -/
def step : St → Tag → St
  | .reasoning, .thinkEnd => .content
  | .content, .callStart => .call
  | .call, .callEnd => .content
  | s, _ => s

/-- A tag in the wrong state is ignored (it is just text). -/
theorem step_ignores_foreign (s : St) (t : Tag) (h : step s t ≠ s) :
    (s = .reasoning ∧ t = .thinkEnd) ∨ (s = .content ∧ t = .callStart) ∨ (s = .call ∧ t = .callEnd) := by
  cases s <;> cases t <;> simp [step] at h ⊢

/-- Once out of reasoning the parser never re-enters it. -/
theorem step_not_reasoning (s : St) (t : Tag) (h : s ≠ .reasoning) : step s t ≠ .reasoning := by
  revert h
  cases s <;> cases t <;> simp [step]

/-- ...for any sequence of tags. -/
theorem run_never_reasoning (l : List Tag) : ∀ s : St, s ≠ .reasoning → l.foldl step s ≠ .reasoning := by
  induction l with
  | nil => intro s h; simpa using h
  | cons t rest ih =>
    intro s h
    exact ih (step s t) (step_not_reasoning s t h)

/-- A non-thinking request starts in content, so it can never emit reasoning events. -/
theorem nonthinking_never_reasoning (l : List Tag) : l.foldl step .content ≠ .reasoning :=
  run_never_reasoning l .content (by decide)

/-- After a complete call the parser is back in content. -/
theorem call_end_returns : step .call .callEnd = .content := by
  rfl

/-- `finish()` flushes held text as reasoning or content; an unterminated call comes back as content. -/
def flushIsContent : St → Bool
  | .reasoning => false
  | .content => true
  | .call => true

/-- An unterminated tool call is surfaced as content, never silently dropped. -/
theorem unterminated_call_is_content : flushIsContent .call = true := by
  rfl

-- § tag holdback

/-- Inner loop of `_hold`: the largest n <= k whose n-prefix of `tag` is a suffix of `text`. -/
def holdAux (text tag : List Char) : Nat → Nat
  | 0 => 0
  | n + 1 => if (tag.take (n + 1)).isSuffixOf text then n + 1 else holdAux text tag n

/-- `_hold` scans only proper prefixes: n from `len tag - 1` down to 1. -/
def hold (text tag : List Char) : Nat := holdAux text tag (tag.length - 1)

/-- The holdback never exceeds the scan bound. -/
theorem holdAux_le (text tag : List Char) (k : Nat) : holdAux text tag k ≤ k := by
  induction k with
  | zero => simp [holdAux]
  | succ n ih =>
    unfold holdAux
    split
    · exact Nat.le_refl _
    · exact Nat.le_succ_of_le ih

/-- A held length is always strictly shorter than the tag: a complete tag is never held back. -/
theorem hold_lt (text tag : List Char) (h : 0 < tag.length) : hold text tag < tag.length := by
  have := holdAux_le text tag (tag.length - 1)
  unfold hold
  omega

/-- A positive holdback is a real match: that prefix of the tag is a suffix of the text. -/
theorem holdAux_spec (text tag : List Char) (k : Nat) (h : 0 < holdAux text tag k) :
    (tag.take (holdAux text tag k)).isSuffixOf text = true := by
  induction k with
  | zero => simp [holdAux] at h
  | succ n ih =>
    unfold holdAux at h ⊢
    by_cases hs : (tag.take (n + 1)).isSuffixOf text = true
    · rw [if_pos hs]
      exact hs
    · rw [if_neg hs] at h ⊢
      exact ih h

end Strata.ServeFrontend
