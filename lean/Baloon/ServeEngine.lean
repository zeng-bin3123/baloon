/-!
# MODULE: serve_engine
SOURCES: serve/server.py
OpenAI / Anthropic HTTP front for a single resident engine: a FIFO-serialized `Service`, a
`RefEngine` speaking the `GEN / T / DONE / ERR` line protocol, an image-embedding cache
bounded to 64 entries, API-key authorization and finish-reason mapping.

## Source semantics
* Requests are admitted only if prompt + max tokens fit the engine context (400 otherwise).
* The engine emits at most `max_new` `T` lines per request and finishes with exactly one `DONE`.
* The vision cache never holds more than 64 entries (oldest evicted first).
* With no API key configured every request is authorized; otherwise the supplied key must equal it.
* OpenAI finish is `tool_calls` iff at least one call was made and generation stopped naturally;
  Anthropic stop reason is `tool_use` under the same condition.

## Form template identifiers
* `Service.prepare` context check -> `admit`
* `Vision.cache` eviction        -> `insertCache`
* `_authorized`                  -> `authorized`
* `openai_chunks` finish         -> `openaiFinish`
* `anthropic_events` stop        -> `anthropicStop`
-/

namespace Baloon.ServeEngine

-- § admission

/-- Prompt + max tokens must not exceed the engine context. -/
def admit (promptLen maxNew maxCtx : Nat) : Bool := decide (promptLen + maxNew ≤ maxCtx)

/-- An admitted request fits the context. -/
theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  simp only [admit, decide_eq_true_eq] at h; exact h

/-- A request that does not fit is rejected, never truncated. -/
theorem admit_reject (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  simp only [admit, decide_eq_false_iff_not]; omega

-- § engine line protocol

/-- Server-side view of a request in flight: tokens streamed so far. -/
structure Req where
  streamed : Nat
  maxNew : Nat
  done : Bool
  deriving Repr

/-- Receiving a `T` line streams one token only while below `max_new`. -/
def onToken (r : Req) : Req :=
  if r.done = false ∧ r.streamed < r.maxNew then { r with streamed := r.streamed + 1 } else r

/-- `DONE` finishes the request. -/
def onDone (r : Req) : Req := { streamed := r.streamed, maxNew := r.maxNew, done := true }

/-- Token delivery never exceeds `max_new`. -/
theorem onToken_le (r : Req) (h : r.streamed ≤ r.maxNew) : (onToken r).streamed ≤ r.maxNew := by
  unfold onToken
  split_ifs with hc
  · have := hc.2; omega
  · exact h

/-- After `DONE` no further token is delivered. -/
theorem onToken_after_done (r : Req) : (onToken (onDone r)).streamed = r.streamed := by
  unfold onDone onToken; simp

/-- `DONE` is idempotent: a request finishes exactly once. -/
theorem onDone_idem (r : Req) : onDone (onDone r) = onDone r := by
  show onDone (onDone r) = onDone r; rfl

-- § vision cache bound

/-- Insert and evict the oldest when more than 64 entries. -/
def insertCache (cache : List Nat) (x : Nat) : List Nat :=
  if cache.length ≥ 64 then (cache ++ [x]).tail else cache ++ [x]

/-- The cache never exceeds 64 entries (given it started within the bound). -/
theorem insertCache_len (cache : List Nat) (x : Nat) (h : cache.length ≤ 64) :
    (insertCache cache x).length ≤ 64 := by
  unfold insertCache
  split_ifs with hc
  · rw [List.length_tail, List.length_append, List.length_singleton]; omega
  · rw [List.length_append, List.length_singleton]; omega

-- § authorization

/-- No key configured -> open; else the supplied key must match. -/
def authorized (key gk : Nat) (configured : Bool) : Bool :=
  if !configured then true else decide (gk = key)

/-- With no API key every request is authorized. -/
theorem authorized_open (key gk : Nat) : authorized key gk false = true := by
  simp only [authorized, Bool.not_false, if_true]

/-- With a key configured, only the matching key is authorized. -/
theorem authorized_iff (key gk : Nat) : authorized key gk true = true ↔ gk = key := by
  simp only [authorized, Bool.not_true, if_false, decide_eq_true_eq]

-- § finish reasons

/-- Finish reasons exposed by the OpenAI side. -/
inductive Finish where
  | stop
  | length
  | toolCalls
  deriving DecidableEq, BEq

/-- `tool_calls` iff at least one call and a natural stop; else `stop` / `length`. -/
def openaiFinish (calls : Nat) (stopped : Bool) : Finish :=
  if 0 < calls ∧ stopped = true then Finish.toolCalls else if stopped = true then Finish.stop else Finish.length

/-- The OpenAI `tool_calls` finish reason is exactly "stopped naturally after calling a tool". -/
theorem openaiFinish_toolCalls_iff (c : Nat) (s : Bool) :
    openaiFinish c s = Finish.toolCalls ↔ (s = true ∧ 0 < c) := by
  rcases Nat.eq_zero_or_pos c with h | h <;> cases s <;> simp [openaiFinish, h]

/-- Anthropic stop reasons. -/
inductive Stop where
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq, BEq

/-- `tool_use` iff a tool block was emitted and generation stopped naturally. -/
def anthropicStop (usedTool stopped : Bool) : Stop :=
  if stopped = false then Stop.maxTokens else if usedTool = true then Stop.toolUse else Stop.endTurn

/-- A length-limited generation never reports `tool_use`. -/
theorem anthropicStop_length (u : Bool) : anthropicStop u false = Stop.maxTokens := by
  cases u <;> rfl

/-- A natural stop without tools is `end_turn`. -/
theorem anthropicStop_end (u : Bool) (h : u = false) : anthropicStop u true = Stop.endTurn := by
  subst h; rfl

-- § usage accounting

/-- OpenAI usage total = prompt + completion tokens. -/
def totalTokens (prompt completion : Nat) : Nat := completion + prompt

/-- The reported total is the sum of its parts. -/
theorem totalTokens_eq (p c : Nat) : totalTokens p c = p + c := Nat.add_comm c p

end Baloon.ServeEngine
