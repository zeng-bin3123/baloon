/-!
# MODULE: chat_frontend_parser
SOURCES: serve/frontend.py
Request normalisation (OpenAI / Anthropic -> chat template messages) and the incremental output parser that
splits model text into reasoning, content and tool calls without ever emitting a partial tag.

## Source semantics
* Reasoning effort spellings map onto {none, low, medium, xhigh}: "high", "max", "maximum" mean xhigh;
  "none", "off", "minimal", "disabled" disable thinking; any other spelling is a client error (400).
* Anthropic `budget_tokens`: under 2048 low, under 8192 medium, otherwise xhigh (monotone in the budget).
* `_hold` returns the length of the longest suffix of the buffered text that is a PROPER prefix of a tag,
  so it is always strictly shorter than the tag; that many characters are withheld from the client.
* Tool-call parameter values drop one leading and one trailing newline.

## Form template identifiers
- EFFORT table                   → `effortLevel`
- budget_effort                  → `budgetRank`
- OutputParser._hold             → `holdLen` / `isHold`
- value[1:] / value[:-1] newline → `stripNL`
-/

namespace Strata.ChatFrontend

-- § reasoning effort

/-- Thinking levels as seen by the template: 0 = none, 1 = low, 2 = medium, 3 = xhigh. -/
def effortLevel : String → Option Nat
  | "none" => some 0
  | "off" => some 0
  | "minimal" => some 0
  | "disabled" => some 0
  | "low" => some 1
  | "medium" => some 2
  | "high" => some 3
  | "xhigh" => some 3
  | "max" => some 3
  | "maximum" => some 3
  | _ => none

/-- "high" means xhigh (the template's default), not a separate level. -/
theorem high_is_xhigh : effortLevel "high" = effortLevel "xhigh" := by
  rfl

/-- Aliases that disable thinking. -/
theorem disabling_aliases :
    effortLevel "none" = some 0 ∧ effortLevel "off" = some 0 ∧ effortLevel "minimal" = some 0 ∧
      effortLevel "disabled" = some 0 := by
  decide

/-- Unknown spellings are rejected rather than guessed. -/
theorem unknown_effort_rejected : effortLevel "turbo" = none := by
  rfl

-- § Anthropic thinking budget

/-- Level chosen from `budget_tokens`. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 1 else if n < 8192 then 2 else 3

/-- A larger budget never selects a lower level. -/
theorem budget_mono (n m : Nat) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> by_cases c : m < 2048 <;> by_cases d : m < 8192 <;>
    simp [a, b, c, d] <;> omega

/-- The levels are only ever low, medium or xhigh (never "none"). -/
theorem budget_range (n : Nat) : 1 ≤ budgetRank n ∧ budgetRank n ≤ 3 := by
  unfold budgetRank
  by_cases a : n < 2048 <;> by_cases b : n < 8192 <;> simp [a, b]

-- § partial-tag hold-back

/-- Is a suffix of `text` of length `n` a proper non-empty prefix of `tag`? -/
def isHold (text tag : List Nat) (n : Nat) : Bool :=
  decide (0 < n) && decide (n < tag.length) && decide (n ≤ text.length) &&
    (text.drop (text.length - n) == tag.take n)

/-- Longest held-back suffix length. -/
def holdLen (text tag : List Nat) : Nat := ((List.range tag.length).filter (isHold text tag)).foldl max 0

/-- A fold of `max` over elements below `k` stays below `k`. -/
theorem foldl_max_lt (k : Nat) :
    ∀ (l : List Nat) (a : Nat), a < k → (∀ x ∈ l, x < k) → l.foldl max a < k := by
  intro l
  induction l with
  | nil => intro a ha _; simpa using ha
  | cons x xs ih =>
    intro a ha hl
    simp only [List.foldl_cons]
    have hx : x < k := hl x (by simp)
    exact ih (max a x) (by omega) (fun y hy => hl y (by simp [hy]))

/-- The withheld suffix is always strictly shorter than the tag, so a full tag is never withheld. -/
theorem holdLen_lt (text tag : List Nat) (hne : 0 < tag.length) : holdLen text tag < tag.length := by
  unfold holdLen
  apply foldl_max_lt _ _ _ hne
  intro x hx
  rw [List.mem_filter, List.mem_range] at hx
  exact hx.1

-- § newline stripping of parameter values

/-- Drop one leading newline (code 10). -/
def dropLead : List Nat → List Nat
  | 10 :: t => t
  | l => l

/-- Drop one trailing newline. -/
def dropTrail (l : List Nat) : List Nat :=
  match l.reverse with
  | 10 :: t => t.reverse
  | _ => l

/-- Strip one newline at each end. -/
def stripNL (l : List Nat) : List Nat := dropTrail (dropLead l)

/-- A value wrapped in exactly one newline on each side is recovered unchanged. -/
theorem stripNL_wrapped (l : List Nat) : stripNL (10 :: (l ++ [10])) = l := by
  simp [stripNL, dropLead, dropTrail]

end Strata.ChatFrontend
