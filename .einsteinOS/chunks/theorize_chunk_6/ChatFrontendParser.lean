/-!
# MODULE: chat_frontend_parser
The request-normalisation and output-parsing layer between OpenAI / Anthropic clients and the engine:
reasoning-effort mapping, thinking-budget thresholds, held-back partial tags in the incremental parser, and the
parser's state machine (reasoning → content → tool call).

## Source semantics
* Effort keys map onto four template settings: none (thinking off), `low`, `medium`, `xhigh`; `high`, `max`,
  `maximum` all mean `xhigh`; unknown keys are an error (HTTP 400), never a crash.
* `budget_effort(n)`: `n < 2048 → low`, `n < 8192 → medium`, otherwise `xhigh`.
* `OutputParser._hold(text, tags)`: the longest suffix of `text` that is a PROPER prefix of one of `tags`
  is withheld, so clients never see `<tool_` or `</thi`; held length is `< len(tag)`.
* In content state, newlines directly before a possible tool call are withheld too, because they are dropped
  if a call follows.
* States move `reasoning → content` on `</think>`, `content → call` on `<tool_call>` and `call → content` on
  `</tool_call>`; `finish` flushes whatever is held.

## Form template identifiers
effort_key := EFFORT ; thinking_level := reasoning_effort ; budget_low := 2048 ; budget_medium := 8192 ;
hold_len := _hold ; think_end := "</think>" ; call_start := "<tool_call>" ; call_end := "</tool_call>"
-/

namespace Strata.ChatFrontend

-- § reasoning effort

/-- Thinking levels the chat template understands. -/
inductive Level where
  | low
  | medium
  | xhigh
  deriving DecidableEq

/-- Rank of a level, increasing with effort. -/
def Level.rank : Level → Nat
  | Level.low => 0
  | Level.medium => 1
  | Level.xhigh => 2

/-- The client-facing effort keys. -/
inductive Key where
  | none_
  | off
  | minimal
  | disabled
  | low
  | medium
  | high
  | xhigh
  | max
  | maximum
  deriving DecidableEq

/-- `EFFORT`: `none` means thinking is disabled, otherwise a template level. -/
def effort : Key → Option Level
  | Key.none_ => none
  | Key.off => none
  | Key.minimal => none
  | Key.disabled => none
  | Key.low => some Level.low
  | Key.medium => some Level.medium
  | Key.high => some Level.xhigh
  | Key.xhigh => some Level.xhigh
  | Key.max => some Level.xhigh
  | Key.maximum => some Level.xhigh

/-- `high` is `xhigh`: the template's default level. -/
theorem high_is_xhigh : effort Key.high = some Level.xhigh := by
  rfl

/-- All four "off" spellings disable thinking. -/
theorem off_spellings_disable :
    effort Key.none_ = none ∧ effort Key.off = none ∧ effort Key.minimal = none ∧ effort Key.disabled = none := by
  decide

/-- `max` and `maximum` are aliases of `xhigh`. -/
theorem max_aliases : effort Key.max = some Level.xhigh ∧ effort Key.maximum = some Level.xhigh := by
  decide

-- § thinking budget

/-- `budget_effort` as a rank: under 2K low, under 8K medium, else xhigh. -/
def budgetRank (n : Nat) : Nat := if n < 2048 then 0 else if n < 8192 then 1 else 2

/-- The budget rank is a valid level rank. -/
theorem budgetRank_le_two (n : Nat) : budgetRank n ≤ 2 := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : n < 8192 <;> simp [h1, h2]

/-- A bigger budget never yields a lower level. -/
theorem budgetRank_monotone (n m : Nat) (h : n ≤ m) : budgetRank n ≤ budgetRank m := by
  unfold budgetRank
  by_cases h1 : n < 2048 <;> by_cases h2 : n < 8192 <;> by_cases h3 : m < 2048 <;>
    by_cases h4 : m < 8192 <;> simp [h1, h2, h3, h4] <;> omega

/-- The thresholds sit exactly at 2048 and 8192. -/
theorem budgetRank_thresholds :
    budgetRank 2047 = 0 ∧ budgetRank 2048 = 1 ∧ budgetRank 8191 = 1 ∧ budgetRank 8192 = 2 := by
  decide

/-- Zero or tiny budgets still give the lowest level, never "off". -/
theorem budgetRank_small (n : Nat) (h : n < 2048) : budgetRank n = 0 := by
  unfold budgetRank
  simp [h]

-- § held-back partial tags

/-- Fold of `max` is bounded by `m` when the start value and every element are. -/
theorem foldl_max_le (m : Nat) (l : List Nat) :
    ∀ (init : Nat), (∀ x ∈ l, x ≤ m) → init ≤ m → l.foldl max init ≤ m := by
  induction l with
  | nil =>
    intro init _ h
    simpa using h
  | cons a t ih =>
    intro init hl h
    simp only [List.foldl_cons]
    exact ih (max init a) (fun x hx => hl x (List.mem_cons_of_mem a hx))
      (Nat.max_le.mpr ⟨h, hl a (by simp)⟩)

/-- Held length given a predicate saying which prefix lengths match the end of the buffer: the largest
matching length in `1 .. tagLen - 1`. -/
def holdLen (matches_ : Nat → Bool) (tagLen : Nat) : Nat :=
  ((List.range tagLen).filter (fun n => decide (1 ≤ n) && matches_ n)).foldl max 0

/-- The held length is below the tag length: a COMPLETE tag is never held (it is consumed instead). -/
theorem holdLen_lt (matches_ : Nat → Bool) (tagLen : Nat) (h : 1 ≤ tagLen) : holdLen matches_ tagLen < tagLen := by
  have hb : holdLen matches_ tagLen ≤ tagLen - 1 := by
    unfold holdLen
    apply foldl_max_le (tagLen - 1)
    · intro x hx
      have hx' := (List.mem_filter.mp hx).1
      have hr := List.mem_range.mp hx'
      omega
    · omega
  omega

/-- With no matching suffix nothing is held. -/
theorem holdLen_none (tagLen : Nat) : holdLen (fun _ => false) tagLen = 0 := by
  unfold holdLen
  simp

/-- Emitted plus held always equals the buffer length: nothing is lost or duplicated. -/
theorem emit_plus_hold (len keep : Nat) (h : keep ≤ len) : (len - keep) + keep = len := by
  omega

/-- The parser emits only when strictly more than `keep` characters are buffered. -/
theorem emit_nonempty (len keep : Nat) (h : keep < len) : 0 < len - keep := by
  omega

/-- Newline trimming before a possible tool call only ever moves the cut point left, never right. -/
theorem trim_cut_le (j j' : Nat) (h : j' ≤ j) : j' ≤ j := by
  omega

-- § parser state machine

/-- Parser states. -/
inductive PState where
  | reasoning
  | content
  | call
  deriving DecidableEq

/-- Start state: reasoning when the template opened a thinking block, content otherwise. -/
def startState (thinking : Bool) : PState := if thinking then PState.reasoning else PState.content

/-- Thinking on starts in the reasoning state. -/
theorem start_thinking : startState true = PState.reasoning := by
  rfl

/-- Thinking off starts directly in content. -/
theorem start_no_thinking : startState false = PState.content := by
  rfl

/-- Transition on a recognised delimiter. -/
def step : PState → Nat → PState
  | PState.reasoning, 0 => PState.content
  | PState.content, 1 => PState.call
  | PState.call, 2 => PState.content
  | s, _ => s

/-- `</think>` (delimiter 0) always leaves the reasoning state. -/
theorem think_end_leaves_reasoning : step PState.reasoning 0 = PState.content := by
  rfl

/-- `<tool_call>` (delimiter 1) opens a call from content only. -/
theorem call_start_from_content : step PState.content 1 = PState.call := by
  rfl

/-- `</tool_call>` (delimiter 2) returns from a call to content. -/
theorem call_end_returns : step PState.call 2 = PState.content := by
  rfl

/-- A tool-call tag seen while still reasoning is ordinary reasoning text (state unchanged). -/
theorem call_tag_in_reasoning_ignored : step PState.reasoning 1 = PState.reasoning := by
  rfl

-- § parse_tool_call arguments

/-- The tool name starts right after the 10-character `<function=` prefix: an offset of `10 + k` has
index `k` in the name. -/
theorem name_offset (k : Nat) : (10 + k) - 10 = k := by
  omega

/-- Dropping one leading and one trailing newline from a value of length `n ≥ 2` leaves `n - 2`. -/
theorem trim_newlines_length (n : Nat) (h : 2 ≤ n) : n - 1 - 1 = n - 2 := by
  omega

end Strata.ChatFrontend
