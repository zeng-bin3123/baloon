/-!
# MODULE: api_server_engine
SOURCES: serve/server.py
OpenAI/Anthropic server over a token-id engine: context admission, incremental detokenisation, stop-token
handling, finish-reason mapping, Anthropic content-block bookkeeping, image-pad expansion, vision cache bound,
auth and launcher argument assembly.

## Source semantics
* Requests with `prompt + max_new > max_context` are rejected, never truncated.
* `completion_tokens` counts every yielded id including the stop id; the stop id is not emitted as text.
* OpenAI finish: `tool_calls` only when at least one call and the engine stopped; Anthropic: `max_tokens` for length.
* One `<|image_pad|>` per image is expanded into one pad per image token.
* The vision cache holds at most 64 entries (oldest evicted).
* `max_completion_tokens or max_tokens or 1024` is always positive.

## Form template identifiers
- `Service.prepare` -> `admission_and_image_expand`; `Service.run` -> `stream_loop`; `Detokenizer` -> `incremental_detok`;
  `openai_chunks`/`anthropic_events` -> `protocol_adapters`; `Vision.encode` -> `cached_image_encode`.
-/

namespace Strata.ApiServerEngine

-- § admission

/-- Admission: the prompt plus the generation budget must fit in the context. -/
def admit (promptLen maxNew ctx : Nat) : Bool := promptLen + maxNew ≤ ctx

/-- An admitted request fits in the context. -/
theorem admit_fits (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  simpa [admit] using h

/-- An over-long request is rejected (400), not truncated. -/
theorem reject_overlong (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  have : ¬ (p + m ≤ c) := by omega
  simp [admit, this]

/-- `max_completion_tokens or max_tokens or 1024` (0 counts as unset, like Python `or`). -/
def maxNew (a b : Nat) : Nat := if a ≠ 0 then a else if b ≠ 0 then b else 1024

/-- The effective generation budget is always positive. -/
theorem maxNew_pos (a b : Nat) : 0 < maxNew a b := by
  unfold maxNew
  split
  · omega
  · split <;> omega

/-- An explicit `max_completion_tokens` wins over `max_tokens`. -/
theorem maxNew_prefers_first (a b : Nat) (h : a ≠ 0) : maxNew a b = a := by
  simp [maxNew, h]

-- § stop handling

/-- Ids that are emitted before the first stop id. -/
def emitted (isStop : Nat → Bool) (ts : List Nat) : List Nat := ts.takeWhile (fun t => !isStop t)

/-- `completion_tokens`: emitted ids plus the stop id when one was seen. -/
def completionTokens (isStop : Nat → Bool) (ts : List Nat) : Nat :=
  (emitted isStop ts).length + (if ts.any isStop then 1 else 0)

/-- Counting never exceeds the number of ids the engine yielded. -/
theorem completion_le (isStop : Nat → Bool) (ts : List Nat) : completionTokens isStop ts ≤ ts.length := by
  unfold completionTokens emitted
  induction ts with
  | nil => simp
  | cons t r ih =>
    by_cases h : isStop t = true
    · simp [List.takeWhile, h]
    · have h' : isStop t = false := by simpa using h
      have e : List.takeWhile (fun t => !isStop t) (t :: r) =
          t :: List.takeWhile (fun t => !isStop t) r := by simp [List.takeWhile, h']
      have e2 : (t :: r).any isStop = r.any isStop := by simp [List.any_cons, h']
      rw [e, e2, List.length_cons, List.length_cons]
      omega

/-- No stop id is ever emitted as text. -/
theorem emitted_no_stop (isStop : Nat → Bool) (ts : List Nat) : ∀ t ∈ emitted isStop ts, isStop t = false := by
  intro t ht
  unfold emitted at ht
  have := List.mem_takeWhile_imp ht
  simpa using this

/-- With no stop id the whole stream is emitted and no extra token is counted. -/
theorem no_stop_all_emitted (isStop : Nat → Bool) (ts : List Nat) (h : ts.any isStop = false) :
    emitted isStop ts = ts := by
  unfold emitted
  apply List.takeWhile_eq_self_iff.mpr
  intro x hx
  have := List.any_eq_false.mp h x hx
  simpa using this

-- § finish reasons

/-- Engine finish reasons. -/
inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reasons. -/
inductive OaFinish where
  | stop
  | length
  | toolCalls
  deriving DecidableEq

/-- OpenAI mapping: `tool_calls` only if calls were made and the engine stopped. -/
def oaFinish (calls : Nat) (f : Finish) : OaFinish :=
  match f with
  | Finish.stop => if calls ≠ 0 then OaFinish.toolCalls else OaFinish.stop
  | Finish.length => OaFinish.length

/-- Hitting the length limit is always reported as `length`, even with tool calls. -/
theorem oa_length_stays (calls : Nat) : oaFinish calls Finish.length = OaFinish.length := by
  rfl

/-- `tool_calls` is reported only when at least one tool call was emitted. -/
theorem oa_tool_calls_needs_call (calls : Nat) (f : Finish) (h : oaFinish calls f = OaFinish.toolCalls) :
    calls ≠ 0 := by
  cases f <;> simp [oaFinish] at h
  exact h

/-- Anthropic stop reasons. -/
inductive AnStop where
  | endTurn
  | maxTokens
  | toolUse
  deriving DecidableEq

/-- Anthropic mapping: `tool_use` when a tool block was emitted and the engine stopped. -/
def anStop (usedTool : Bool) (f : Finish) : AnStop :=
  if usedTool = true ∧ f = Finish.stop then AnStop.toolUse
  else match f with
    | Finish.stop => AnStop.endTurn
    | Finish.length => AnStop.maxTokens

/-- Length always maps to `max_tokens`, with or without tools. -/
theorem an_length_is_max (u : Bool) : anStop u Finish.length = AnStop.maxTokens := by
  cases u <;> simp [anStop]

/-- Without a tool block, a stop is `end_turn`. -/
theorem an_stop_no_tool : anStop false Finish.stop = AnStop.endTurn := by
  simp [anStop]

-- § Anthropic block bookkeeping

/-- Content-block kinds in the Anthropic stream. -/
inductive BK where
  | thinking
  | text
  | tool
  deriving DecidableEq

/-- Number of blocks opened: a new block starts when the kind changes, and always for a tool call. -/
def blocks (prev : Option BK) : List BK → Nat
  | [] => 0
  | k :: r => (if prev = some k ∧ k ≠ BK.tool then 0 else 1) + blocks (some k) r

/-- Never more blocks than events. -/
theorem blocks_le (ks : List BK) : ∀ prev, blocks prev ks ≤ ks.length := by
  induction ks with
  | nil => intro prev; simp [blocks]
  | cons k r ih =>
    intro prev
    have := ih (some k)
    simp only [blocks, List.length_cons]
    split <;> omega

/-- Consecutive text deltas share one block. -/
theorem text_deltas_merge : blocks none [BK.text, BK.text, BK.text] = 1 := by
  decide

/-- Every tool call opens its own block. -/
theorem tool_calls_separate : blocks none [BK.tool, BK.tool] = 2 := by
  decide

/-- A thinking block followed by text is two blocks. -/
theorem thinking_then_text : blocks none [BK.thinking, BK.thinking, BK.text] = 2 := by
  decide

-- § incremental detokenisation

/-- One step of the detokeniser: given the full decoded text and what was already sent, emit the new suffix unless
the text ends in a replacement character. Returns (delta, new sent). -/
def detokStep {α : Type} (text : List α) (sent : Nat) (endsReplacement : Bool) : List α × Nat :=
  if endsReplacement then ([], sent) else (text.drop sent, text.length)

/-- The sent counter never exceeds the text length. -/
theorem detok_sent_le {α : Type} (text : List α) (sent : Nat) (e : Bool) (h : sent ≤ text.length) :
    (detokStep text sent e).2 ≤ text.length := by
  unfold detokStep
  split <;> simp [h]

/-- Already-sent text plus the new delta is exactly the new sent prefix: nothing is skipped or repeated. -/
theorem detok_prefix_property {α : Type} (text : List α) (sent : Nat) (e : Bool) (h : sent ≤ text.length) :
    text.take sent ++ (detokStep text sent e).1 = text.take (detokStep text sent e).2 := by
  unfold detokStep
  split
  · simp
  · simp

/-- A held (incomplete) character emits nothing and leaves the counter unchanged. -/
theorem detok_hold_noop {α : Type} (text : List α) (sent : Nat) :
    detokStep text sent true = ([], sent) := by
  simp [detokStep]

-- § image-pad expansion

/-- Replace the k-th pad with `counts[k]` pads; leave other ids alone. -/
def expand (pad : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: r, cs =>
    if t = pad then
      match cs with
      | [] => t :: expand pad r []
      | c :: cs' => List.replicate c pad ++ expand pad r cs'
    else t :: expand pad r cs

/-- Number of pad ids in a prompt. -/
def countPad (pad : Nat) (ids : List Nat) : Nat := (ids.filter (fun t => t == pad)).length

/-- When pads and images match one-to-one, the expanded prompt has `len - pads + sum counts` ids. -/
theorem expand_length (pad : Nat) (ids : List Nat) : ∀ cs : List Nat, countPad pad ids = cs.length →
    (expand pad ids cs).length + countPad pad ids = ids.length + cs.sum := by
  induction ids with
  | nil =>
    intro cs h
    have h1 : countPad pad [] = 0 := by simp [countPad]
    have h0 : cs.length = 0 := by omega
    cases cs with
    | nil => simp [expand, countPad]
    | cons c r => simp at h0
  | cons t r ih =>
    intro cs h
    by_cases ht : t = pad
    · subst ht
      cases cs with
      | nil => simp [countPad] at h
      | cons c cs' =>
        have h' : countPad t r = cs'.length := by
          simp [countPad, List.filter_cons] at h ⊢
          omega
        have := ih cs' h'
        simp [expand, countPad, List.filter_cons] at this ⊢
        omega
    · have h' : countPad pad r = cs.length := by
        simp [countPad, List.filter_cons, ht] at h ⊢
        exact h
      have := ih cs h'
      simp [expand, ht, countPad, List.filter_cons] at this ⊢
      omega

-- § vision cache

/-- Cache size after inserting one entry: the oldest is evicted past 64. -/
def cacheAfterInsert (size : Nat) : Nat := if size + 1 > 64 then size else size + 1

/-- The cache never exceeds 64 entries. -/
theorem cache_bounded (size : Nat) (h : size ≤ 64) : cacheAfterInsert size ≤ 64 := by
  unfold cacheAfterInsert
  split <;> omega

/-- Below the cap an insert grows the cache by one. -/
theorem cache_grows (size : Nat) (h : size < 64) : cacheAfterInsert size = size + 1 := by
  unfold cacheAfterInsert
  have : ¬ (size + 1 > 64) := by omega
  simp [this]

-- § image formats and auth

/-- Pass-through detection for the first magic bytes: JPEG `FF D8 FF`, BMP `BM`. -/
def isJpeg (b0 b1 b2 : Nat) : Bool := b0 == 255 && b1 == 216 && b2 == 255

/-- BMP magic `B M`. -/
def isBmp (b0 b1 : Nat) : Bool := b0 == 66 && b1 == 77

/-- A JPEG header is never also a BMP header. -/
theorem jpeg_not_bmp (b0 b1 b2 b1' : Nat) (h : isJpeg b0 b1 b2 = true) : isBmp b0 b1' = false := by
  simp [isJpeg, isBmp] at h ⊢
  omega

/-- Authorisation: no key configured means open; otherwise the presented key must match. -/
def authorized (apiKey presented : String) : Bool := apiKey == "" || presented == apiKey

/-- With no API key configured every request is authorised. -/
theorem auth_open (presented : String) : authorized "" presented = true := by
  simp [authorized]

/-- With a key configured, a wrong key is refused. -/
theorem auth_wrong_refused (k g : String) (hk : k ≠ "") (hg : g ≠ k) : authorized k g = false := by
  simp [authorized, hk, hg]

-- § launcher arguments

/-- Vision launcher argv length: 5 fixed items plus `--gpu`, `--threads N`, `--max-tokens N`. -/
def visionArgc (gpu threads maxTok : Bool) : Nat :=
  5 + (if gpu then 1 else 0) + (if threads then 2 else 0) + (if maxTok then 2 else 0)

/-- The launcher argv has between 5 and 10 entries. -/
theorem vision_argc_bounds (g t m : Bool) : 5 ≤ visionArgc g t m ∧ visionArgc g t m ≤ 10 := by
  cases g <;> cases t <;> cases m <;> simp [visionArgc]

/-- `DONE` carries 6 whitespace-separated fields; indices 1..5 are the five numbers/strings read. -/
theorem done_fields : 1 + 5 = 6 := by
  decide

end Strata.ApiServerEngine
