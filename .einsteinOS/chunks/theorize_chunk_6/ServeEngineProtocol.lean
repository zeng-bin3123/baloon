/-!
# MODULE: serve_engine_protocol
The serving core between HTTP clients and the resident engine: context admission, image-pad expansion,
usage accounting, finish-reason mapping for the OpenAI and Anthropic wire formats, the incremental
detokenizer, the image-embedding cache bound, and API-key authorisation.

## Source semantics
* A request is admitted iff `len(prompt_ids) + max_new <= engine.max_context`; otherwise HTTP 400, never truncated.
* Every `<|image_pad|>` of the rendered prompt is expanded to one pad per image token of that image; the prompt
  grows by `Σ (n_k - 1)`.
* Usage: `prompt_tokens = len(ids)`, `total_tokens = prompt + completion`; the stop token itself is counted
  in `completion_tokens` but never emitted.
* Finish reason: generation ended on a stop id → `stop`, else `length`; OpenAI reports `tool_calls` when at
  least one call was emitted and the finish was `stop`; Anthropic maps `stop → end_turn`, `length → max_tokens`
  and reports `tool_use` under the same condition.
* Detokenizer: re-decode all ids, emit only the new complete suffix; a text ending in U+FFFD is withheld.
* The vision cache keeps at most 64 entries, evicting the oldest first.
* Auth: with an empty key every request is allowed, otherwise the Bearer token or `x-api-key` must equal it.

## Form template identifiers
admit := len + max_new <= max_context ; pad_expand := [pad] * n_k ; completion_tokens := n ;
finish_stop := "stop" ; finish_length := "length" ; cache_cap := 64 ; sent_offset := self.sent
-/

namespace Strata.ServeProtocol

-- § context admission

/-- Admission test of `Service.prepare`. -/
def admitted (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- An admitted request fits the context, so nothing is ever truncated. -/
theorem admitted_fits (p m c : Nat) (h : admitted p m c = true) : p + m ≤ c := by
  unfold admitted at h
  simpa using h

/-- A request that does not fit is rejected. -/
theorem rejected_overflow (p m c : Nat) (h : c < p + m) : admitted p m c = false := by
  unfold admitted
  simp
  omega

/-- Admission is monotone in the context: a larger context admits everything a smaller one did. -/
theorem admitted_mono (p m c c' : Nat) (h : admitted p m c = true) (hc : c ≤ c') : admitted p m c' = true := by
  have h1 := admitted_fits p m c h
  unfold admitted
  simp
  omega

/-- A longer prompt can only be rejected if a shorter one with the same budget was. -/
theorem admitted_antitone (p p' m c : Nat) (h : admitted p' m c = true) (hp : p ≤ p') : admitted p m c = true := by
  have h1 := admitted_fits p' m c h
  unfold admitted
  simp
  omega

-- § image pad expansion

/-- Length of the prompt after expanding each of `k` pads to its image's token count. -/
def expandedLen (base : Nat) (counts : List Nat) : Nat := base - counts.length + counts.sum

/-- A list of ones sums to its length. -/
theorem sum_of_ones (l : List Nat) : (∀ c ∈ l, c = 1) → l.sum = l.length := by
  induction l with
  | nil =>
    intro _
    rfl
  | cons a t ih =>
    intro h
    have ha : a = 1 := h a (by simp)
    have ht : ∀ c ∈ t, c = 1 := fun c hc => h c (List.mem_cons_of_mem a hc)
    have h1 := ih ht
    simp only [List.sum_cons, List.length_cons]
    omega

/-- With one token per image the prompt length is unchanged. -/
theorem expand_unit (base : Nat) (counts : List Nat) (hall : ∀ c ∈ counts, c = 1) (hk : counts.length ≤ base) :
    expandedLen base counts = base := by
  have h1 := sum_of_ones counts hall
  unfold expandedLen
  omega

/-- With no images nothing changes. -/
theorem expand_nil (base : Nat) : expandedLen base [] = base := by
  unfold expandedLen
  simp

/-- A single image of `n` tokens grows the prompt by `n - 1`. -/
theorem expand_single (base n : Nat) (hb : 1 ≤ base) : expandedLen base [n] = base - 1 + n := by
  unfold expandedLen
  simp

/-- Replacing one pad by `n ≥ 1` pads never shrinks the prompt. -/
theorem expand_single_grows (base n : Nat) (hb : 1 ≤ base) (hn : 1 ≤ n) : base ≤ expandedLen base [n] := by
  unfold expandedLen
  simp
  omega

-- § usage accounting

/-- Total tokens reported in `usage`. -/
def totalTokens (prompt completion : Nat) : Nat := prompt + completion

/-- Total is the sum of its parts. -/
theorem total_is_sum (p c : Nat) : totalTokens p c = p + c := by
  rfl

/-- The total never exceeds the context when the request was admitted and generation respected `max_new`. -/
theorem total_within_context (p c m ctx : Nat) (hadm : p + m ≤ ctx) (hc : c ≤ m) : totalTokens p c ≤ ctx := by
  unfold totalTokens
  omega

-- § finish reasons

/-- Why generation ended. -/
inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- The engine side of finishing: a stop id ends the turn, running out of tokens does not. -/
def finishOf (hitStop : Bool) : Finish := if hitStop then Finish.stop else Finish.length

/-- A stop id gives `stop`. -/
theorem finish_stop_id : finishOf true = Finish.stop := by
  rfl

/-- No stop id gives `length`. -/
theorem finish_no_stop : finishOf false = Finish.length := by
  rfl

/-- OpenAI finish reason code: 0 stop, 1 length, 2 tool_calls. -/
def openaiFinish (calls : Nat) (f : Finish) : Nat :=
  match f with
  | Finish.stop => if 0 < calls then 2 else 0
  | Finish.length => 1

/-- Calls plus a clean stop is reported as `tool_calls`. -/
theorem openai_tool_calls (calls : Nat) (h : 0 < calls) : openaiFinish calls Finish.stop = 2 := by
  unfold openaiFinish
  simp [h]

/-- A truncated generation stays `length` even when calls were emitted. -/
theorem openai_length_wins (calls : Nat) : openaiFinish calls Finish.length = 1 := by
  rfl

/-- Anthropic stop reason code: 0 end_turn, 1 max_tokens, 2 tool_use. -/
def anthropicStop (usedTool : Bool) (f : Finish) : Nat :=
  match f with
  | Finish.stop => if usedTool then 2 else 0
  | Finish.length => 1

/-- Both wire formats agree on when a tool call ends the turn. -/
theorem formats_agree_on_tools (calls : Nat) (h : 0 < calls) :
    openaiFinish calls Finish.stop = anthropicStop true Finish.stop := by
  unfold openaiFinish anthropicStop
  simp [h]

/-- ... and on truncation. -/
theorem formats_agree_on_length (calls : Nat) (used : Bool) :
    openaiFinish calls Finish.length = anthropicStop used Finish.length := by
  rfl

-- § detokenizer

/-- New `sent` offset after pushing a token: unchanged when the text ends in a replacement character,
else the decoded length. -/
def sentAfter (sent textLen : Nat) (endsReplacement : Bool) : Nat := if endsReplacement then sent else textLen

/-- The emitted delta has length `textLen - sent` when it is emitted. -/
def deltaLen (sent textLen : Nat) (endsReplacement : Bool) : Nat := if endsReplacement then 0 else textLen - sent

/-- Delta plus old offset reaches the new offset: emitted text is contiguous. -/
theorem delta_contiguous (sent textLen : Nat) (e : Bool) (h : sent ≤ textLen) :
    sent + deltaLen sent textLen e = sentAfter sent textLen e := by
  unfold deltaLen sentAfter
  cases e <;> simp <;> omega

/-- The sent offset never decreases (given a stable decoded prefix). -/
theorem sent_monotone (sent textLen : Nat) (e : Bool) (h : sent ≤ textLen) : sent ≤ sentAfter sent textLen e := by
  unfold sentAfter
  cases e <;> simp <;> omega

/-- The sent offset never passes the decoded length, so no text is emitted twice. -/
theorem sent_bounded (sent textLen : Nat) (e : Bool) (h : sent ≤ textLen) : sentAfter sent textLen e ≤ textLen := by
  unfold sentAfter
  cases e <;> simp <;> omega

-- § vision cache

/-- Cache size after inserting one new entry with the oldest-first bound of 64. -/
def cacheAfterInsert (size : Nat) : Nat := if 64 < size + 1 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem cache_bounded (size : Nat) (h : size ≤ 64) : cacheAfterInsert size ≤ 64 := by
  unfold cacheAfterInsert
  split <;> omega

/-- The cache grows by one until full. -/
theorem cache_grows (size : Nat) (h : size < 64) : cacheAfterInsert size = size + 1 := by
  unfold cacheAfterInsert
  simp
  omega

/-- A full cache stays full: the oldest entry is evicted for the new one. -/
theorem cache_full_stays : cacheAfterInsert 64 = 64 := by
  decide

-- § authorisation

/-- Authorisation: an empty configured key allows everything. -/
def authorized (keyEmpty givenMatches : Bool) : Bool := keyEmpty || givenMatches

/-- No key configured → every request is authorised. -/
theorem auth_open (g : Bool) : authorized true g = true := by
  rfl

/-- With a key configured a request is authorised exactly when the given token matches. -/
theorem auth_requires_match (g : Bool) : authorized false g = g := by
  unfold authorized
  simp

end Strata.ServeProtocol
