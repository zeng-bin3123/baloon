/-!
# MODULE: serve_server_protocol
SOURCES: serve/server.py
OpenAI / Anthropic endpoints over a token-id engine: context admission, image-pad expansion, incremental
detokenisation, stop reasons, content-block opening, the image-embedding cache bound and API-key checks.

## Source semantics
* A request is accepted only if `len(prompt) + max_new <= max_context`; requests are rejected, never truncated.
* Each `<|image_pad|>` is replaced by as many pads as its image has tokens; text without pads is unchanged.
* The detokenizer emits `text[sent:]` of the re-decoded ids, so the deltas concatenate to the full text.
* OpenAI finish: `tool_calls` iff a call was emitted and generation stopped; otherwise `stop`/`length`.
  Anthropic: `tool_use` iff a tool was used and generation stopped; `end_turn` for stop, `max_tokens` for length.
* A tool call always opens a new content block; reasoning/text blocks continue while the kind is unchanged.
* The image cache holds at most 64 entries (oldest evicted first).
* With an empty API key everything is authorised; otherwise the presented key must equal it.

## Form template identifiers
- len(ids)+max_new > max_context → `accepts`
- the `<|image_pad|>` loop        → `expand`
- Detokenizer.push               → `deltaOf`
- finish mapping                 → `openaiFinish` / `anthropicStop`
- block open decision            → `newBlock`
- cache eviction                 → `evict`
- Service._authorized            → `authorized`
-/

namespace Strata.ServeServer

-- § context admission

/-- The context check of `Service.prepare`. -/
def accepts (promptLen maxNew ctx : Nat) : Prop := promptLen + maxNew ≤ ctx

/-- An accepted request fits wholesale: nothing needs truncating. -/
theorem accepted_fits (p m c : Nat) (h : accepts p m c) : p ≤ c ∧ m ≤ c := by
  unfold accepts at h
  omega

/-- A request over the context is not accepted. -/
theorem over_context_rejected (p m c : Nat) (h : c < p + m) : ¬ accepts p m c := by
  unfold accepts
  omega

-- § image pad expansion

/-- Replace each pad token `p` by `c` pads, consuming one count per pad. -/
def expand (p : Nat) : List Nat → List Nat → List Nat
  | [], _ => []
  | t :: ts, cs =>
    if t = p then
      (match cs with
        | [] => t :: expand p ts []
        | c :: cs' => List.replicate c p ++ expand p ts cs')
    else t :: expand p ts cs

/-- Without pad tokens the prompt is unchanged. -/
theorem expand_no_pad (p : Nat) :
    ∀ (ts cs : List Nat), (∀ t ∈ ts, t ≠ p) → expand p ts cs = ts := by
  intro ts
  induction ts with
  | nil => intro cs _; rfl
  | cons t ts ih =>
    intro cs h
    have ht : t ≠ p := h t (by simp)
    have hr : ∀ x ∈ ts, x ≠ p := fun x hx => h x (by simp [hx])
    simp [expand, ht, ih cs hr]

/-- One pad and one image of `c` tokens expands to exactly `c` pads. -/
theorem expand_single (p c : Nat) : expand p [p] [c] = List.replicate c p := by
  simp [expand]

/-- A prompt of one pad per image grows to exactly the total image-token count. -/
theorem expand_length (p : Nat) (cs : List Nat) :
    (expand p (List.replicate cs.length p) cs).length = cs.sum := by
  induction cs with
  | nil => simp [expand]
  | cons c cs ih => simp [expand, List.replicate_succ, ih]

-- § incremental detokenisation

/-- The new suffix emitted when the decoded text grows from `old` to `new`. -/
def deltaOf (old new : List Nat) : List Nat := new.drop old.length

/-- If decoding is prefix-extending, the delta is exactly the appended text. -/
theorem delta_of_append (old x : List Nat) : deltaOf old (old ++ x) = x := by
  simp [deltaOf]

/-- Successive deltas concatenate back to the final text. -/
theorem deltas_concatenate (a x y : List Nat) :
    a ++ deltaOf a (a ++ x) ++ deltaOf (a ++ x) (a ++ x ++ y) = a ++ x ++ y := by
  simp [deltaOf]

-- § stop reasons and content blocks

/-- Generation finish reasons from the engine. -/
inductive Finish where
  | stop
  | length
  deriving DecidableEq

/-- OpenAI finish reason code: 0 = stop, 1 = length, 2 = tool_calls. -/
def openaiFinish (calls : Nat) (f : Finish) : Nat :=
  if calls > 0 ∧ f = Finish.stop then 2 else match f with
    | Finish.stop => 0
    | Finish.length => 1

/-- `tool_calls` is reported only when generation stopped normally. -/
theorem openai_tool_calls_only_on_stop (calls : Nat) (f : Finish) (h : openaiFinish calls f = 2) :
    f = Finish.stop := by
  unfold openaiFinish at h
  cases f <;> simp_all

/-- A truncated generation is still reported as `length`, even with calls emitted. -/
theorem openai_length_preserved (calls : Nat) : openaiFinish calls Finish.length = 1 := by
  simp [openaiFinish]

/-- Anthropic stop reason code: 0 = end_turn, 1 = max_tokens, 2 = tool_use. -/
def anthropicStop (usedTool : Bool) (f : Finish) : Nat :=
  if usedTool = true ∧ f = Finish.stop then 2 else match f with
    | Finish.stop => 0
    | Finish.length => 1

/-- `tool_use` only on a normal stop. -/
theorem anthropic_tool_use_only_on_stop (u : Bool) (f : Finish) (h : anthropicStop u f = 2) :
    f = Finish.stop := by
  unfold anthropicStop at h
  cases f <;> simp_all

/-- Kinds of streamed events. -/
inductive Kind where
  | reasoning
  | content
  | tool
  deriving DecidableEq

/-- Whether an event of kind `k` opens a new block given the currently open kind. -/
def newBlock (cur : Option Kind) (k : Kind) : Bool :=
  match k with
  | Kind.tool => true
  | _ => decide (cur ≠ some k)

/-- A tool call always opens a fresh block. -/
theorem tool_always_new (cur : Option Kind) : newBlock cur Kind.tool = true := by
  rfl

/-- Consecutive text deltas continue the same block. -/
theorem text_continues : newBlock (some Kind.content) Kind.content = false := by
  decide

/-- Switching from reasoning to content opens a new block. -/
theorem switch_opens : newBlock (some Kind.reasoning) Kind.content = true := by
  decide

-- § image cache bound

/-- Drop the oldest entry when the cache exceeds 64. -/
def evict {α : Type} (l : List α) : List α := if l.length > 64 then l.drop 1 else l

/-- Inserting into a cache of at most 64 entries and evicting keeps it at most 64. -/
theorem evict_bound {α : Type} (l : List α) (h : l.length ≤ 65) : (evict l).length ≤ 64 := by
  unfold evict
  split
  · rw [List.length_drop]
    omega
  · omega

-- § authorisation

/-- The presented key is accepted when no key is configured or it matches. -/
def authorized (configured presented : String) : Prop := configured = "" ∨ presented = configured

/-- With no configured key, anything is accepted. -/
theorem no_key_open (presented : String) : authorized "" presented := by
  exact Or.inl rfl

/-- The right key is always accepted. -/
theorem right_key_accepted (k : String) : authorized k k := by
  exact Or.inr rfl

end Strata.ServeServer
