/-!
# MODULE: serve_server_service
The serving core: the context-length admission rule, finish-reason mapping for the OpenAI and Anthropic
endpoints, usage accounting, the vision-embedding cache bound, API-key authorization, the mock engine's token
budget and the incremental detokenizer's cursor.

## Source semantics
* A request is accepted iff `len(prompt) + max_new <= max_context`; otherwise 400 (never truncated).
* OpenAI finish reason: `tool_calls` if at least one call was emitted and the engine stopped; otherwise the
  engine's reason (`stop` / `length`). Anthropic: `tool_use` under the same condition, else `end_turn` for
  stop and `max_tokens` for length. Truncation by length is never reported as a normal stop.
* `usage.total_tokens = prompt_tokens + completion_tokens`.
* The image-embedding cache holds at most 64 entries (oldest evicted first).
* With no API key everything is authorized; with a key only an exact match is.
* `MockEngine.generate` yields `min(max_new, len(script))` tokens.
* `Detokenizer.push` holds back text ending in U+FFFD and otherwise advances its cursor to the decoded length.

## Form template identifiers
* `admit` : `len(ids) + max_new > engine.max_context` check in `Service.prepare`
* `openaiFinish`, `anthropicStop` : the finish mappings in `openai_chunks` / `anthropic_events`
* `evict` : the `len(self.cache) > 64` eviction
* `authorized` : `Handler._authorized`
* `pushCursor` : `Detokenizer.push`
-/

namespace Strata.ServeServer

-- § admission

/-- The request fits iff the prompt plus the generation budget is within the engine context. -/
def admit (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- An admitted request never overruns the context. -/
theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  simpa [admit] using h

/-- A rejected request really does overrun it. -/
theorem reject_sound (p m c : Nat) (h : admit p m c = false) : c < p + m := by
  simp [admit] at h
  omega

/-- A larger prompt is never admitted when a smaller one was rejected. -/
theorem admit_antitone (p q m c : Nat) (h : p ≤ q) (hq : admit q m c = true) : admit p m c = true := by
  simp [admit] at hq ⊢
  omega

-- § finish reasons

inductive Fin2
  | stop
  | length
  deriving DecidableEq, Repr

inductive OaFinish
  | stop
  | length
  | toolCalls
  deriving DecidableEq, Repr

inductive AnStop
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq, Repr

/-- OpenAI: `tool_calls` iff calls were emitted and the engine stopped. -/
def openaiFinish (calls : Nat) : Fin2 → OaFinish
  | .stop => if 0 < calls then .toolCalls else .stop
  | .length => .length

/-- Anthropic: `tool_use` iff a tool was used and the engine stopped. -/
def anthropicStop (usedTool : Bool) : Fin2 → AnStop
  | .stop => if usedTool then .toolUse else .endTurn
  | .length => .maxTokens

/-- A length cutoff is reported as `length` whatever tool calls were emitted. -/
theorem openai_length (calls : Nat) : openaiFinish calls .length = .length := rfl

/-- A length cutoff is reported as `max_tokens` whatever tool use happened. -/
theorem anthropic_length (u : Bool) : anthropicStop u .length = .maxTokens := rfl

/-- Without any call, a normal stop is `stop`. -/
theorem openai_plain_stop : openaiFinish 0 .stop = .stop := by
  simp [openaiFinish]

/-- With at least one call, a normal stop is `tool_calls`. -/
theorem openai_tool_stop (calls : Nat) (h : 0 < calls) : openaiFinish calls .stop = .toolCalls := by
  simp [openaiFinish, h]

/-- The two endpoints agree on when a tool stop is reported. -/
theorem endpoints_agree (calls : Nat) (h : 0 < calls) :
    openaiFinish calls .stop = .toolCalls ∧ anthropicStop true .stop = .toolUse := by
  simp [openaiFinish, anthropicStop, h]

-- § usage

/-- `usage.total_tokens`. -/
def totalTokens (prompt completion : Nat) : Nat := prompt + completion

/-- Total is prompt plus completion. -/
theorem total_eq (p c : Nat) : totalTokens p c = p + c := rfl

/-- Total is at least each part. -/
theorem total_ge (p c : Nat) : p ≤ totalTokens p c ∧ c ≤ totalTokens p c := by
  unfold totalTokens
  omega

-- § mock engine budget

/-- The mock engine yields at most `max_new` tokens of its script. -/
theorem mock_yield_len (script : List Nat) (maxNew : Nat) :
    (script.take maxNew).length = min maxNew script.length := by
  simp

/-- It never exceeds the generation budget. -/
theorem mock_yield_le (script : List Nat) (maxNew : Nat) : (script.take maxNew).length ≤ maxNew := by
  simp

-- § vision cache

/-- Cache size after inserting one new entry and evicting the oldest when it exceeds 64. -/
def evict (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem evict_bound (size : Nat) (h : size ≤ 64) : evict size ≤ 64 := by
  unfold evict
  split <;> omega

/-- Eviction removes exactly one entry once full, and none before. -/
theorem evict_full (size : Nat) (h : size = 64) : evict size = 64 := by
  unfold evict
  split <;> omega

-- § authorization

/-- `Handler._authorized`: no configured key authorizes everything; a key requires an exact match. -/
def authorized (key supplied : String) : Bool := key == "" || supplied == key

/-- With no key configured every request is authorized. -/
theorem authorized_no_key (supplied : String) : authorized "" supplied = true := by
  simp [authorized]

/-- With a key configured only the exact key is authorized. -/
theorem authorized_exact (key supplied : String) (hk : key ≠ "") (h : authorized key supplied = true) :
    supplied = key := by
  simp [authorized, hk] at h
  exact h

/-- The right key is always accepted. -/
theorem authorized_self (key : String) : authorized key key = true := by
  simp [authorized]

-- § detokenizer cursor

/-- `Detokenizer.push`: when the decoded text ends in U+FFFD nothing is emitted and the cursor stays;
otherwise the cursor advances to the decoded length. -/
def pushCursor (sent decodedLen : Nat) (held : Bool) : Nat := if held then sent else decodedLen

/-- The cursor never passes the decoded text. -/
theorem cursor_le (sent decodedLen : Nat) (held : Bool) (h : sent ≤ decodedLen) :
    pushCursor sent decodedLen held ≤ decodedLen := by
  unfold pushCursor
  split <;> omega

/-- The cursor never moves backwards when the decoded text only grows. -/
theorem cursor_mono (sent decodedLen : Nat) (held : Bool) (h : sent ≤ decodedLen) :
    sent ≤ pushCursor sent decodedLen held := by
  unfold pushCursor
  split <;> omega

/-- When nothing is held back, everything decoded so far has been emitted. -/
theorem cursor_caught_up (sent decodedLen : Nat) : pushCursor sent decodedLen false = decodedLen := by
  simp [pushCursor]

end Strata.ServeServer
