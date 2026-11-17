-- Baloon.ServeEngine: independent implementation; proves the same invariants
-- as Ref.Serve with different definitions and proof strategies.

namespace Baloon.ServeEngine

-- § admission

/-- Admit when the total request fits (max_new + prompt ≤ ctx). -/
def admit (promptLen maxNew maxCtx : Nat) : Bool := decide (maxNew + promptLen ≤ maxCtx)

theorem admit_sound (p m c : Nat) (h : admit p m c = true) : p + m ≤ c := by
  unfold admit at h; have := of_decide_eq_true h; omega

theorem admit_reject (p m c : Nat) (h : c < p + m) : admit p m c = false := by
  unfold admit; exact decide_eq_false (by omega)

-- § engine line protocol

structure Req where
  streamed : Nat
  maxNew   : Nat
  done     : Bool
  deriving Repr

/-- Stream one token: increment only while not done and below max_new. -/
def onToken (r : Req) : Req :=
  if r.done != true ∧ r.streamed < r.maxNew then { r with streamed := r.streamed + 1 } else r

/-- Mark a request complete. -/
def onDone (r : Req) : Req := Req.mk r.streamed r.maxNew true

theorem onToken_le (r : Req) (h : r.streamed ≤ r.maxNew) : (onToken r).streamed ≤ r.maxNew := by
  unfold onToken
  split_ifs with hc
  · obtain ⟨_, hlt⟩ := hc; omega
  · exact h

theorem onToken_after_done (r : Req) : (onToken (onDone r)).streamed = r.streamed := by
  unfold onToken onDone; simp

theorem onDone_idem (r : Req) : onDone (onDone r) = onDone r := by rfl

-- § vision cache bound (64-entry LRU by evicting head)

def insertCache (cache : List Nat) (x : Nat) : List Nat :=
  if cache.length ≥ 64 then (cache ++ [x]).tail else cache ++ [x]

theorem insertCache_len (cache : List Nat) (x : Nat) (h : cache.length ≤ 64) :
    (insertCache cache x).length ≤ 64 := by
  simp only [insertCache]
  by_cases hc : cache.length ≥ 64
  · simp only [if_pos hc, List.length_tail, List.length_append, List.length_singleton]; omega
  · simp only [if_neg hc, List.length_append, List.length_singleton]; omega

-- § authorization

/-- Open when no key configured; otherwise given key must equal stored key. -/
def authorized (key gk : Nat) (configured : Bool) : Bool :=
  !configured || decide (gk = key)

theorem authorized_open (key gk : Nat) : authorized key gk false = true := by
  unfold authorized; rfl

theorem authorized_iff (key gk : Nat) : authorized key gk true = true ↔ gk = key := by
  constructor
  · intro h; unfold authorized at h
    rw [Bool.not_true, Bool.false_or] at h; exact of_decide_eq_true h
  · intro h; unfold authorized
    rw [Bool.not_true, Bool.false_or]; exact decide_eq_true h

-- § finish reasons

inductive Finish where | stop | length | toolCalls
  deriving DecidableEq, Repr

def openaiFinish (calls : Nat) (stopped : Bool) : Finish :=
  if stopped && (calls != 0) then Finish.toolCalls
  else if stopped then Finish.stop
  else Finish.length

theorem openaiFinish_toolCalls_iff (c : Nat) (s : Bool) :
    openaiFinish c s = Finish.toolCalls ↔ (s = true ∧ 0 < c) := by
  cases s
  · simp [openaiFinish]
  · cases c with
    | zero => simp [openaiFinish]
    | succ n => simp [openaiFinish]

inductive Stop where | endTurn | maxTokens | toolUse
  deriving DecidableEq, Repr

def anthropicStop (usedTool stopped : Bool) : Stop :=
  if usedTool && stopped then Stop.toolUse
  else if stopped then Stop.endTurn
  else Stop.maxTokens

theorem anthropicStop_length (u : Bool) : anthropicStop u false = Stop.maxTokens := by
  cases u <;> simp [anthropicStop]

theorem anthropicStop_end (u : Bool) (h : u = false) : anthropicStop u true = Stop.endTurn := by
  subst h; simp [anthropicStop]

-- § usage accounting

/-- Total tokens consumed = completion + prompt. -/
def totalTokens (prompt completion : Nat) : Nat := completion + prompt

theorem totalTokens_eq (p c : Nat) : totalTokens p c = p + c := by
  unfold totalTokens; omega

end Baloon.ServeEngine
