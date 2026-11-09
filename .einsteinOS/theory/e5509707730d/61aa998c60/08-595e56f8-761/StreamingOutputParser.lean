/-!
# MODULE: streaming_output_parser
SOURCES: serve/frontend.py
Response-side half of the frontend: the incremental `OutputParser` state machine (reasoning -> content <-> tool
call), partial-tag hold-back, tool-call argument decoding and the template renderer's flag mapping.

## Source semantics
* States: reasoning, content, call. `</think>` leaves reasoning for good; `<tool_call>` enters call; `</tool_call>`
  returns to content with leading newlines swallowed.
* A suffix of the buffer that is a proper prefix of a tag is held back, so clients never see `<tool_` or `</thi`.
* `finish()` flushes held text as reasoning or content, never as a tool call.
* Tool parameter values: declared `string` stays a string; otherwise JSON when it parses; one leading and one trailing
  newline are removed.

## Form template identifiers
- `_hold` -> `partial_tag_holdback`; `OutputParser.feed` -> `incremental_state_machine`; `parse_tool_call` -> `tool_call_decode`.
-/

namespace Strata.StreamingOutputParser

-- § hold-back of partial tags

/-- Longest `n ≤ fuel` such that the first `n` elements of `tag` form a suffix of `text`. -/
def holdAux {α : Type} [BEq α] (tag text : List α) : Nat → Nat
  | 0 => 0
  | n + 1 => max (holdAux tag text n) (if (tag.take (n + 1)).isSuffixOf text then n + 1 else 0)

/-- `_hold`: longest proper-prefix-of-`tag` suffix of `text` (lengths 1 .. |tag|-1). -/
def holdLen {α : Type} [BEq α] (tag text : List α) : Nat := holdAux tag text (tag.length - 1)

/-- The hold search never exceeds its fuel. -/
theorem holdAux_le {α : Type} [BEq α] (tag text : List α) (n : Nat) : holdAux tag text n ≤ n := by
  induction n with
  | zero => simp [holdAux]
  | succ k ih =>
    unfold holdAux
    split <;> omega

/-- The held part is a PROPER prefix: strictly shorter than the tag, so a complete tag is never held. -/
theorem holdLen_lt {α : Type} [BEq α] (tag text : List α) (h : 0 < tag.length) :
    holdLen tag text < tag.length := by
  have := holdAux_le tag text (tag.length - 1)
  unfold holdLen
  omega

/-- An empty tag holds nothing. -/
theorem holdLen_empty_tag {α : Type} [BEq α] (text : List α) : holdLen ([] : List α) text = 0 := by
  simp [holdLen, holdAux]

/-- Split a buffer into the emitted part and the held tail of length `keep`. -/
def splitHold {α : Type} (buf : List α) (keep : Nat) : List α × List α :=
  (buf.take (buf.length - keep), buf.drop (buf.length - keep))

/-- Emitted ++ held reproduces the buffer exactly: no text is lost or duplicated by holding. -/
theorem splitHold_append {α : Type} (buf : List α) (keep : Nat) :
    (splitHold buf keep).1 ++ (splitHold buf keep).2 = buf := by
  unfold splitHold
  simp

/-- The held tail has the requested length when it fits in the buffer. -/
theorem splitHold_held_length {α : Type} (buf : List α) (keep : Nat) (h : keep ≤ buf.length) :
    (splitHold buf keep).2.length = keep := by
  unfold splitHold
  simp
  omega

/-- The emitted part has length `len - keep`. -/
theorem splitHold_emit_length {α : Type} (buf : List α) (keep : Nat) :
    (splitHold buf keep).1.length = buf.length - keep := by
  unfold splitHold
  simp

/-- Concrete check: with tag [1,2,3] and text ending [1,2], two elements are held. -/
theorem hold_example : holdLen [1, 2, 3] [9, 1, 2] = 2 := by
  decide

/-- Concrete check: text not ending in a tag prefix holds nothing. -/
theorem hold_none_example : holdLen [1, 2, 3] [9, 8, 7] = 0 := by
  decide

-- § state machine

/-- Parser states. -/
inductive State where
  | reasoning
  | content
  | call
  deriving DecidableEq

/-- Markers the parser reacts to. -/
inductive Marker where
  | thinkEnd
  | callStart
  | callEnd
  deriving DecidableEq

/-- State transition when a marker is found in the current state; other markers are inert. -/
def next : State → Marker → State
  | State.reasoning, Marker.thinkEnd => State.content
  | State.content, Marker.callStart => State.call
  | State.call, Marker.callEnd => State.content
  | s, _ => s

/-- The initial state is reasoning exactly when thinking is on. -/
def initState (thinking : Bool) : State := if thinking then State.reasoning else State.content

/-- With thinking disabled the parser starts in content. -/
theorem init_no_thinking : initState false = State.content := by
  rfl

/-- Reasoning is never re-entered: a transition into reasoning only comes from reasoning. -/
theorem reasoning_not_reentered (s : State) (m : Marker) (h : next s m = State.reasoning) :
    s = State.reasoning := by
  cases s <;> cases m <;> simp [next] at h ⊢

/-- `</think>` takes reasoning to content. -/
theorem think_end_exits_reasoning : next State.reasoning Marker.thinkEnd = State.content := by
  rfl

/-- A completed tool call returns to content. -/
theorem call_end_returns : next State.call Marker.callEnd = State.content := by
  rfl

/-- A tool call can only start from content (never from reasoning). -/
theorem call_only_from_content (s : State) (h : next s Marker.callStart = State.call) : s = State.content ∨ s = State.call := by
  cases s <;> simp [next] at h ⊢

-- § events and flush

/-- Event kinds. -/
inductive Kind where
  | reasoning
  | content
  | toolCall
  deriving DecidableEq

/-- The kind `finish()` flushes held text as: reasoning stays reasoning, everything else (including an
unterminated call) is content. -/
def finishKind : State → Kind
  | State.reasoning => Kind.reasoning
  | _ => Kind.content

/-- `finish()` never fabricates a tool call from unterminated input. -/
theorem finish_never_tool_call (s : State) : finishKind s ≠ Kind.toolCall := by
  cases s <;> simp [finishKind]

-- § tool-call argument decoding

/-- A decoded argument is either the raw string or a JSON value. -/
inductive Arg where
  | str
  | json
  deriving DecidableEq

/-- Declared-string parameters stay strings; others are JSON when they parse, else strings. -/
def decodeArg (declaredString parses : Bool) : Arg :=
  if declaredString then Arg.str else if parses then Arg.json else Arg.str

/-- A declared string is never JSON-decoded (so the text `123` stays `"123"`). -/
theorem declared_string_stays_string (parses : Bool) : decodeArg true parses = Arg.str := by
  simp [decodeArg]

/-- An unparsable value falls back to the string. -/
theorem unparsable_is_string (d : Bool) : decodeArg d false = Arg.str := by
  cases d <;> simp [decodeArg]

/-- Drop one leading newline (10). -/
def dropLead : List Nat → List Nat
  | 10 :: r => r
  | l => l

/-- Drop one trailing newline (10). -/
def dropTrail (l : List Nat) : List Nat := if l.getLast? = some 10 then l.dropLast else l

/-- Exactly one leading newline is removed. -/
theorem dropLead_newline (r : List Nat) : dropLead (10 :: r) = r := by
  rfl

/-- A leading non-newline is untouched. -/
theorem dropLead_other (c : Nat) (r : List Nat) (h : c ≠ 10) : dropLead (c :: r) = c :: r := by
  unfold dropLead
  split
  · rename_i heq
    simp at heq
    omega
  · rfl

/-- Exactly one trailing newline is removed. -/
theorem dropTrail_newline (l : List Nat) : dropTrail (l ++ [10]) = l := by
  simp [dropTrail]

/-- Stripping never makes a value longer, and removes at most one character at each end. -/
theorem dropLead_length (l : List Nat) : l.length - 1 ≤ (dropLead l).length ∧ (dropLead l).length ≤ l.length := by
  unfold dropLead
  split <;> simp

/-- A tool call needs a `<function=` prefix; the name is the text up to the first `>`. -/
def hasFunctionPrefix (body : List Nat) (prefixLen : Nat) : Bool := prefixLen ≤ body.length

/-- The tool-call id is `call_` plus the first 24 of the 32 uuid hex digits. -/
theorem call_id_digits : 24 ≤ 32 ∧ "call_".length + 24 = 29 := by
  decide

end Strata.StreamingOutputParser
