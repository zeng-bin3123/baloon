/-!
# MODULE: inference_server_core
Pure logic of the OpenAI / Anthropic HTTP server in front of a token-id engine: context admission,
finish-reason mapping, API-key check, image-pad expansion, bounded embedding cache, incremental
detokenisation and Anthropic content-block bookkeeping.

## Source semantics
* A request is admitted iff prompt tokens + max new tokens ≤ engine context; otherwise 400, never truncation.
* OpenAI finish reason becomes `tool_calls` iff at least one call was emitted and the engine stopped;
  Anthropic stop reason becomes `tool_use` under the same condition, else `end_turn`/`max_tokens`.
* When no API key is configured everything is authorised; otherwise the presented key must equal it.
* Each `<|image_pad|>` in the prompt is expanded to that image's token count; prompts with no pad are unchanged.
* The vision cache holds at most 64 entries (oldest evicted first).
* The detokenizer emits nothing while the decoded text ends in a replacement character, and otherwise emits
  exactly the not-yet-sent suffix, so concatenated deltas equal the decoded text.
* An Anthropic content block is opened when the kind changes, and always for a tool_use.

## Form template identifiers
* `fitsContext` ↔ `Service.prepare` length check; `openaiFinish` ↔ `openai_chunks` finish; `anthropicStop` ↔ `anthropic_events`
* `authorized` ↔ `Handler._authorized`; `expandPads` ↔ pad expansion loop in `Service.prepare`
* `cacheInsert` ↔ `Vision.encode` eviction; `detokPush` ↔ `Detokenizer.push`; `blockStarts` ↔ `open_kind != want or want == "tool_use"`
-/

namespace Strata.Server

-- § context admission

/-- Admission rule: prompt plus requested completion must fit the engine's context. -/
def fitsContext (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- Admitted requests never need truncation. -/
theorem fitsContext_sound {p m c : Nat} (h : fitsContext p m c = true) : p + m ≤ c := by
  simpa [fitsContext] using h

/-- Rejected requests really exceed the context. -/
theorem fitsContext_complete {p m c : Nat} (h : fitsContext p m c = false) : c < p + m := by
  simp [fitsContext] at h
  omega

/-- A shorter prompt is admitted whenever a longer one is. -/
theorem fitsContext_mono {p q m c : Nat} (hpq : p ≤ q) (h : fitsContext q m c = true) :
    fitsContext p m c = true := by
  have := fitsContext_sound h
  simp [fitsContext]
  omega

-- § finish reasons

/-- Engine finish reasons. -/
inductive Finish
  | stop | length
  deriving DecidableEq, Repr

/-- OpenAI finish reasons. -/
inductive OpenFinish
  | stop | length | toolCalls
  deriving DecidableEq, Repr

/-- Anthropic stop reasons. -/
inductive AnthStop
  | endTurn | maxTokens | toolUse
  deriving DecidableEq, Repr

/-- OpenAI: `tool_calls` iff a call was emitted and the engine stopped. -/
def openaiFinish (calls : Nat) (f : Finish) : OpenFinish :=
  match f with
  | .stop => if 0 < calls then .toolCalls else .stop
  | .length => .length

/-- Anthropic: `tool_use` iff a tool was used and the engine stopped. -/
def anthropicStop (usedTool : Bool) (f : Finish) : AnthStop :=
  match f with
  | .stop => if usedTool then .toolUse else .endTurn
  | .length => .maxTokens

/-- Hitting the token limit is always reported as `length`, even with tool calls. -/
theorem openaiFinish_length (calls : Nat) : openaiFinish calls .length = .length := rfl

/-- A completed turn with a call reports `tool_calls`. -/
theorem openaiFinish_stop_calls {calls : Nat} (h : 0 < calls) : openaiFinish calls .stop = .toolCalls := by
  simp [openaiFinish, h]

/-- A completed turn without a call reports `stop`. -/
theorem openaiFinish_stop_nocalls : openaiFinish 0 .stop = .stop := rfl

/-- Truncated Anthropic output is `max_tokens`. -/
theorem anthropicStop_length (u : Bool) : anthropicStop u .length = .maxTokens := rfl

/-- A completed Anthropic turn with a tool reports `tool_use`. -/
theorem anthropicStop_stop_tool : anthropicStop true .stop = .toolUse := rfl

/-- A completed Anthropic turn without a tool reports `end_turn`. -/
theorem anthropicStop_stop_plain : anthropicStop false .stop = .endTurn := rfl

/-- `tool_use` is only reported for a completed turn that used a tool. -/
theorem anthropicStop_toolUse_iff (u : Bool) (f : Finish) :
    anthropicStop u f = .toolUse ↔ u = true ∧ f = .stop := by
  cases u <;> cases f <;> simp [anthropicStop]

-- § authorisation

/-- With an empty configured key everything is authorised; otherwise the key must match. -/
def authorized (key presented : String) : Bool := key.isEmpty || presented == key

/-- No configured key → open access. -/
theorem authorized_open (presented : String) : authorized "" presented = true := by
  simp [authorized]

/-- The right key is accepted. -/
theorem authorized_self (key : String) : authorized key key = true := by
  simp [authorized]

/-- A non-empty configured key rejects any different key. -/
theorem authorized_wrong {key presented : String} (hk : key.isEmpty = false) (hg : presented ≠ key) :
    authorized key presented = false := by
  simp [authorized, hk, hg]

-- § image pad expansion

/-- Expand each `pad` token into as many pads as its image has tokens; returns output and unused counts.
A pad with no remaining count is kept as is (the source then raises "do not match"). -/
def expandPads (pad : Nat) : List Nat → List Nat → List Nat × List Nat
  | [], cs => ([], cs)
  | t :: ts, cs =>
    if t = pad then
      match cs with
      | [] => ((t :: (expandPads pad ts []).1), (expandPads pad ts []).2)
      | c :: cs' => (List.replicate c pad ++ (expandPads pad ts cs').1, (expandPads pad ts cs').2)
    else
      (t :: (expandPads pad ts cs).1, (expandPads pad ts cs).2)

/-- A prompt without pad tokens (a text-only request) is unchanged and consumes no counts. -/
theorem expandPads_no_pad (pad : Nat) (ids cs : List Nat) (h : ∀ t ∈ ids, t ≠ pad) :
    expandPads pad ids cs = (ids, cs) := by
  induction ids generalizing cs with
  | nil => simp [expandPads]
  | cons t ts ih =>
    have ht : t ≠ pad := h t (by simp)
    have hts : ∀ x ∈ ts, x ≠ pad := fun x hx => h x (List.mem_cons_of_mem _ hx)
    have := ih cs hts
    simp [expandPads, ht, this]

/-- A single pad with a single count expands to exactly that many pads. -/
theorem expandPads_single (pad c : Nat) : expandPads pad [pad] [c] = (List.replicate c pad, []) := by
  simp [expandPads]

/-- Worked example with two images of 3 and 2 tokens. -/
theorem expandPads_example :
    expandPads 7 [1, 7, 2, 7] [3, 2] = ([1, 7, 7, 7, 2, 7, 7], []) := by decide

-- § bounded cache

/-- Insert at the end; evict the oldest entry when the size exceeds 64. -/
def cacheInsert {α : Type} (c : List α) (x : α) : List α :=
  if (c ++ [x]).length > 64 then (c ++ [x]).drop 1 else c ++ [x]

/-- The cache never holds more than 64 entries. -/
theorem cacheInsert_length_le {α : Type} (c : List α) (x : α) (h : c.length ≤ 64) :
    (cacheInsert c x).length ≤ 64 := by
  unfold cacheInsert
  split
  · simp only [List.length_drop, List.length_append, List.length_singleton] at *
    omega
  · rename_i hn
    omega

-- § incremental detokenisation

/-- One push: with a pending partial character emit nothing; otherwise emit the unsent suffix. -/
def detokPush {α : Type} (text : List α) (sent : Nat) (partial_ : Bool) : List α × Nat :=
  if partial_ then ([], sent) else (text.drop sent, text.length)

/-- Held pushes emit nothing and keep the cursor. -/
theorem detokPush_partial {α : Type} (text : List α) (sent : Nat) :
    detokPush text sent true = ([], sent) := rfl

/-- Emitted text continues exactly where the previous output stopped. -/
theorem detokPush_continues {α : Type} (text : List α) (sent : Nat) :
    text.take sent ++ (detokPush text sent false).1 = text := by
  simp [detokPush]

/-- After an emitting push everything decoded so far has been sent. -/
theorem detokPush_cursor {α : Type} (text : List α) (sent : Nat) :
    (detokPush text sent false).2 = text.length := rfl

-- § Anthropic content blocks

/-- Content block kinds. -/
inductive BlockKind
  | thinking | text | toolUse
  deriving DecidableEq, Repr

/-- A new block starts when the kind changes or the event is a tool_use. -/
def blockStarts (openKind : Option BlockKind) (want : BlockKind) : Bool :=
  decide (openKind ≠ some want) || decide (want = .toolUse)

/-- The first event always opens a block. -/
theorem blockStarts_first (want : BlockKind) : blockStarts none want = true := by
  simp [blockStarts]

/-- Consecutive text (or thinking) deltas stay in one block. -/
theorem blockStarts_continue_text : blockStarts (some .text) .text = false := by decide

/-- Consecutive thinking deltas stay in one block. -/
theorem blockStarts_continue_thinking : blockStarts (some .thinking) .thinking = false := by decide

/-- Every tool_use gets its own block. -/
theorem blockStarts_toolUse (o : Option BlockKind) : blockStarts o .toolUse = true := by
  simp [blockStarts]

/-- Usage totals add up. -/
theorem usage_total (prompt completion : Nat) : prompt + completion = prompt + completion := rfl

end Strata.Server
