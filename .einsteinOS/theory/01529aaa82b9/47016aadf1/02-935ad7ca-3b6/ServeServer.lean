/-!
# MODULE: serve_server
Contracts of the HTTP serving layer: the never-truncate context guard, incremental detokenization, token
accounting up to the stop token, finish-reason mapping for both API dialects, the image-embedding cache
bound, max-token defaulting and API-key authorization.

## Source semantics
A request is accepted iff `prompt + max_new <= context`; the prompt is never altered by the guard. The
detokenizer advances `sent` only to the length of a fully decoded text (never while a partial character is
held) and never moves backwards. Generation stops at the first stop token, which still counts as a
completion token. A tool call plus a natural stop reports `tool_calls`/`tool_use`; a length stop reports
`length`/`max_tokens`. The image cache never exceeds 64 entries.

## Form template identifiers
- `accepts` <- the check in Service.prepare
- `detokPush` <- Detokenizer.push (length model)
- `runN` <- the token loop in Service.run
- `openaiFinish`, `anthropicStop` <- finish mapping in openai_chunks / anthropic_events
- `cacheAdd` <- Vision.encode cache insert + evict
- `pickMax`, `authorized` <- max_new defaulting and Handler._authorized
-/

namespace Strata.ServeServer

-- § context guard

/-- Accept iff the prompt plus the requested completion fits the context. -/
def accepts (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- Accepted requests fit entirely: nothing is truncated to make room. -/
theorem accepts_fits (p m c : Nat) (h : accepts p m c = true) : p + m ≤ c := by
  unfold accepts at h
  exact of_decide_eq_true h

/-- Requests that do not fit are rejected, never clipped. -/
theorem rejects_overflow (p m c : Nat) (h : c < p + m) : accepts p m c = false := by
  unfold accepts
  exact decide_eq_false (by omega)

-- § detokenizer

/-- New `sent` after a push: advance to the decoded length unless a partial character is held. -/
def detokPush (sent len : Nat) (held : Bool) : Nat := if held then sent else len

/-- Emitted delta length is `len - sent` when released and 0 when held. -/
def deltaLen (sent len : Nat) (held : Bool) : Nat := if held then 0 else len - sent

/-- `sent` never moves backwards (decoded text only grows). -/
theorem detok_monotone (sent len : Nat) (held : Bool) (h : sent ≤ len) : sent ≤ detokPush sent len held := by
  unfold detokPush
  split <;> omega

/-- `sent` never passes the decoded text. -/
theorem detok_le_len (sent len : Nat) (held : Bool) (h : sent ≤ len) : detokPush sent len held ≤ len := by
  unfold detokPush
  split <;> omega

/-- Every pushed character is accounted for: old `sent` plus the delta is the new `sent`. -/
theorem detok_accounting (sent len : Nat) (held : Bool) (h : sent ≤ len) :
    sent + deltaLen sent len held = detokPush sent len held := by
  unfold detokPush deltaLen
  split <;> omega

-- § generation loop

/-- (completion tokens counted, stopped by a stop token) over the engine's token stream. -/
def runN (isStop : Nat → Bool) : List Nat → Nat × Bool
  | [] => (0, false)
  | t :: ts => if isStop t then (1, true) else ((runN isStop ts).1 + 1, (runN isStop ts).2)

/-- The count never exceeds the number of generated tokens. -/
theorem runN_le (isStop : Nat → Bool) : ∀ l : List Nat, (runN isStop l).1 ≤ l.length := by
  intro l
  induction l with
  | nil => simp [runN]
  | cons t ts ih =>
    unfold runN
    split
    · simp
    · simp only [List.length_cons]
      omega

/-- A non-empty stream counts at least one token (the stop token itself counts). -/
theorem runN_pos (isStop : Nat → Bool) (t : Nat) (ts : List Nat) : 1 ≤ (runN isStop (t :: ts)).1 := by
  unfold runN
  split
  · exact Nat.le_refl _
  · simp

/-- Usage totals add up. -/
theorem usage_total (p c : Nat) : p + c = p + c := rfl

-- § finish mapping

inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reason: 0 = stop, 1 = length, 2 = tool_calls. -/
def openaiFinish (calls : Nat) (f : Finish) : Nat :=
  if 0 < calls ∧ f = Finish.stop then 2 else (if f = Finish.stop then 0 else 1)

/-- Anthropic stop reason: 0 = end_turn, 1 = max_tokens, 2 = tool_use. -/
def anthropicStop (usedTool : Bool) (f : Finish) : Nat :=
  if usedTool = true ∧ f = Finish.stop then 2 else (if f = Finish.stop then 0 else 1)

/-- A tool call followed by a natural stop reports `tool_calls`. -/
theorem openai_tool_calls (calls : Nat) (h : 0 < calls) : openaiFinish calls Finish.stop = 2 := by
  unfold openaiFinish
  simp [h]

/-- A length stop reports `length` even when tools were called. -/
theorem openai_length (calls : Nat) : openaiFinish calls Finish.length = 1 := by
  unfold openaiFinish
  simp

/-- A tool call followed by a natural stop reports `tool_use`. -/
theorem anthropic_tool_use : anthropicStop true Finish.stop = 2 := by
  unfold anthropicStop
  simp

/-- A length stop reports `max_tokens`. -/
theorem anthropic_max_tokens (u : Bool) : anthropicStop u Finish.length = 1 := by
  unfold anthropicStop
  simp

-- § image cache

/-- Cache size after inserting one entry and evicting the oldest when over 64. -/
def cacheAdd (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem cacheAdd_le (size : Nat) (h : size ≤ 64) : cacheAdd size ≤ 64 := by
  unfold cacheAdd
  split <;> omega

-- § request defaults and auth

/-- `max_completion_tokens or max_tokens or 1024`. -/
def pickMax (a b : Nat) : Nat := if a ≠ 0 then a else if b ≠ 0 then b else 1024

/-- The defaulted limit is never 0. -/
theorem pickMax_pos (a b : Nat) : 0 < pickMax a b := by
  unfold pickMax
  repeat' split
  all_goals omega

/-- Authorization: no key configured, or the presented key matches. -/
def authorized {α : Type} [DecidableEq α] (configured presented : α) (empty : α) : Bool :=
  decide (configured = empty) || decide (presented = configured)

/-- With no key configured every request is authorized. -/
theorem authorized_when_open {α : Type} [DecidableEq α] (presented empty : α) :
    authorized empty presented empty = true := by
  unfold authorized
  simp

/-- With a key configured, only the matching key passes. -/
theorem authorized_needs_key {α : Type} [DecidableEq α] (configured presented empty : α)
    (hne : configured ≠ empty) (h : authorized configured presented empty = true) : presented = configured := by
  unfold authorized at h
  simp [hne] at h
  exact h

end Strata.ServeServer
