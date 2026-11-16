-- Baloon.Serve: serve-loop admission, engine line protocol (Req state machine),
-- vision cache bound and API-key authorization.
-- Independent implementation proving the same mathematical facts as Ref.Serve.

namespace Baloon.Serve

def admit (promptLen maxNew maxCtx : Nat) : Bool := (decide (promptLen + maxNew ≤ maxCtx))

theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  unfold admit at h; exact of_decide_eq_true h

theorem admit_reject (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  unfold admit; exact decide_eq_false (by omega)

structure Req where
  streamed : Nat
  maxNew : Nat
  done : Bool
  deriving Repr

def onToken (r : Req) : Req :=
  if (r.done = false ∧ r.streamed < r.maxNew) then { r with streamed := r.streamed + 1 } else r

def onDone (r : Req) : Req := ({ r with done := true })

theorem onToken_le (r : Req) (h : r.streamed ≤ r.maxNew) : (onToken r).streamed ≤ r.maxNew := by
  unfold onToken
  split_ifs with hc
  · show r.streamed + 1 ≤ r.maxNew; have := hc.2; omega
  · exact h

theorem onToken_after_done (r : Req) : (onToken (onDone r)).streamed = r.streamed := by { simp [onToken, onDone] }

theorem onDone_idem (r : Req) : onDone (onDone r) = onDone r := by { rfl }

def insertCache (cache : List Nat) (x : Nat) : List Nat :=
  if (64 < cache.length + 1) then (cache ++ [x]).tail else cache ++ [x]

theorem insertCache_len (cache : List Nat) (x : Nat) (h : cache.length ≤ 64) :
    (insertCache cache x).length ≤ 64 := by
  unfold insertCache
  split_ifs with hc
  · rw [List.length_tail, List.length_append, List.length_singleton]; omega
  · rw [List.length_append, List.length_singleton]; omega

def authorized (key gk : Nat) (configured : Bool) : Bool :=
  if (configured = false) then true else decide (gk = key)

theorem authorized_open (key gk : Nat) : authorized key gk false = true := by { simp [authorized] }

theorem authorized_key_match (key : Nat) : authorized key key true = true := by
  simp [authorized]

theorem authorized_key_mismatch (key gk : Nat) (h : gk ≠ key) : authorized key gk true = false := by
  simp [authorized]; exact h

-- ── finish reasons and usage accounting ──────────────────────────────────────

/-- Anthropic stop reasons. -/
inductive Stop where
  | endTurn | maxTokens | toolUse
  deriving DecidableEq, Repr

/-- toolUse iff a tool block was emitted and generation stopped naturally. -/
def anthropicStop (usedTool stopped : Bool) : Stop :=
  if (usedTool = true ∧ stopped = true) then Stop.toolUse
  else if (stopped = true) then Stop.endTurn
  else Stop.maxTokens

/-- A length-limited generation never reports tool_use. -/
theorem anthropicStop_length (u : Bool) : anthropicStop u false = Stop.maxTokens := by { simp [anthropicStop] }

/-- A natural stop without tools is end_turn. -/
theorem anthropicStop_end (u : Bool) (h : u = false) : anthropicStop u true = Stop.endTurn := by { simp [anthropicStop, h] }

/-- OpenAI usage total = prompt + completion tokens. -/
def totalTokens (prompt completion : Nat) : Nat := (prompt + completion)

theorem totalTokens_eq (p c : Nat) : totalTokens p c = p + c := by { rfl }

end Baloon.Serve
