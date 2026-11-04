/-!
# MODULE: serve_endpoints
The HTTP serving layer over a resident engine: prompt preparation (image-pad expansion and the context check),
finish-reason mapping for the OpenAI and Anthropic wire formats, API-key authorization, the image-embedding cache
bound, and the incremental detokenizer.

## Source semantics
* Each `<|image_pad|>` in the rendered prompt is replaced by one pad per image token of the matching image;
  when the pad count equals the image count the new length is `len - pads + sum of image token counts`.
* A request is rejected (never truncated) when `prompt + max_new` exceeds the engine context.
* OpenAI: `tool_calls` is the finish reason iff at least one call was emitted and the engine stopped;
  Anthropic maps `stop -> end_turn`, `length -> max_tokens`, and `tool_use` after a call.
* With no configured key every request is authorized; with a key, exactly the matching credential is.
* The image cache holds at most 64 entries: the oldest is evicted when a 65th arrives.
* The detokenizer emits only the new suffix of the decoded text and tracks how much it has sent.

## Form template identifiers
* `Service.prepare` -> `expand`, `Handler._authorized` -> `authorized`, `Vision.encode` cache -> `cacheAfter`
* `Detokenizer.push` -> `detokDelta`/`detokSent`
-/

namespace Strata.ServeEndpoints

-- § image-pad expansion

/-- Replace the k-th pad by `counts[k]` pads; extra pads without a matching image are kept as-is. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
      if t = pad then
        match cs with
        | [] => t :: expand pad ts []
        | c :: cs' => List.replicate c pad ++ expand pad ts cs'
      else t :: expand pad ts cs

/-- With as many images as pads, the expanded prompt has exactly `len - pads + sum counts` tokens. -/
theorem expand_length (pad : Nat) :
    ∀ (ids cs : List Nat), ids.count pad = cs.length →
      (expand pad ids cs).length + ids.count pad = ids.length + cs.sum
  | [], cs, h => by
      have hc : cs = [] := by
        cases cs with
        | nil => rfl
        | cons c cs' => simp at h
      subst hc
      simp [expand]
  | t :: ts, cs, h => by
      by_cases ht : t = pad
      · subst ht
        cases cs with
        | nil => simp [List.count_cons] at h
        | cons c cs' =>
            have h' : ts.count t = cs'.length := by
              simp [List.count_cons] at h
              omega
            have ih := expand_length t ts cs' h'
            simp [expand, List.count_cons, ih] <;> omega
      · have h' : ts.count pad = cs.length := by
          have hne : (t == pad) = false := by simp [ht]
          simpa [List.count_cons, hne] using h
        have ih := expand_length pad ts cs h'
        have hne : (t == pad) = false := by simp [ht]
        simp [expand, ht, List.count_cons, hne] <;> omega

/-- The request is accepted only when the prompt and generation fit in the context. -/
def accepts (promptLen maxNew ctx : Nat) : Prop := promptLen + maxNew ≤ ctx

/-- Accepted requests never need truncation. -/
theorem accepts_fits (p m c : Nat) (h : accepts p m c) : p + m ≤ c := h

/-- A rejected request really exceeds the context. -/
theorem rejects_exceeds (p m c : Nat) (h : ¬ accepts p m c) : c < p + m := by
  unfold accepts at h
  omega

/-- The engine's own check (`n + max_new + 8 > max_context` is an error) is stronger than the server's. -/
theorem engine_check_stronger (p m c : Nat) (h : p + m + 8 ≤ c) : accepts p m c := by
  unfold accepts
  omega

-- § finish reasons

/-- Engine finish status. -/
inductive Finish | stop | length
  deriving DecidableEq

/-- OpenAI finish reason: `tool_calls` after a stopped generation that emitted calls. -/
def openaiFinish (calls : Nat) (f : Finish) : String :=
  if 0 < calls ∧ f = Finish.stop then "tool_calls"
  else match f with
    | .stop => "stop"
    | .length => "length"

/-- Anthropic stop reason. -/
def anthropicStop (usedTool : Bool) (f : Finish) : String :=
  if usedTool = true ∧ f = Finish.stop then "tool_use"
  else match f with
    | .stop => "end_turn"
    | .length => "max_tokens"

/-- A length-limited generation is always reported as `length` on the OpenAI wire. -/
theorem openai_length_preserved (calls : Nat) : openaiFinish calls .length = "length" := by
  simp [openaiFinish]

/-- A length-limited generation is always `max_tokens` for Anthropic. -/
theorem anthropic_length_preserved (u : Bool) : anthropicStop u .length = "max_tokens" := by
  simp [anthropicStop]

/-- Without tool calls a stop is a plain `stop`. -/
theorem openai_plain_stop : openaiFinish 0 .stop = "stop" := by
  simp [openaiFinish]

/-- With a tool call a stop becomes `tool_calls`. -/
theorem openai_tool_stop (n : Nat) : openaiFinish (n + 1) .stop = "tool_calls" := by
  simp [openaiFinish]

/-- Without tool use a stop is `end_turn`. -/
theorem anthropic_plain_stop : anthropicStop false .stop = "end_turn" := by
  simp [anthropicStop]

-- § authorization

/-- With an empty configured key everything passes; otherwise the credential must equal the key. -/
def authorized (key cred : String) : Prop := key = "" ∨ cred = key

/-- No key configured: open access. -/
theorem authorized_open (cred : String) : authorized "" cred := Or.inl rfl

/-- A key is configured: exactly the matching cred passes. -/
theorem authorized_iff (key cred : String) (h : key ≠ "") : authorized key cred ↔ cred = key := by
  unfold authorized
  constructor
  · intro hh
    rcases hh with hh | hh
    · exact absurd hh h
    · exact hh
  · intro hh
    exact Or.inr hh

-- § cache bound

/-- Entries after inserting one: the 65th insert evicts the oldest, so the size stays at 64. -/
def cacheAfter (size : Nat) : Nat := if 64 < size + 1 then size else size + 1

/-- The image cache never exceeds 64 entries. -/
theorem cache_bounded (size : Nat) (h : size ≤ 64) : cacheAfter size ≤ 64 := by
  unfold cacheAfter
  split <;> omega

/-- Inserting into a non-full cache grows it by one. -/
theorem cache_grows (size : Nat) (h : size < 64) : cacheAfter size = size + 1 := by
  unfold cacheAfter
  split <;> omega

-- § detokenizer

/-- New text since the last emission. -/
def detokDelta (text : List Char) (sent : Nat) : List Char := text.drop sent

/-- Text length marker after a push: unchanged while the decode ends in an incomplete character. -/
def detokSent (text : List Char) (sent : Nat) (incomplete : Bool) : Nat :=
  if incomplete then sent else text.length

/-- The emitted delta has exactly the unsent length. -/
theorem detokDelta_length (text : List Char) (sent : Nat) :
    (detokDelta text sent).length = text.length - sent := by
  simp [detokDelta]

/-- Previously sent text plus the delta is the whole decoded text. -/
theorem detokDelta_complete (text : List Char) (sent : Nat) (h : sent ≤ text.length) :
    text.take sent ++ detokDelta text sent = text := by
  unfold detokDelta
  exact List.take_append_drop _ _

/-- The sent marker never moves backwards while the text only grows. -/
theorem detokSent_mono (text : List Char) (sent : Nat) (inc : Bool) (h : sent ≤ text.length) :
    sent ≤ detokSent text sent inc := by
  unfold detokSent
  split <;> omega

/-- The sent marker never passes the decoded text. -/
theorem detokSent_le_len (text : List Char) (sent : Nat) (inc : Bool) (h : sent ≤ text.length) :
    detokSent text sent inc ≤ text.length := by
  unfold detokSent
  split <;> omega

end Strata.ServeEndpoints
