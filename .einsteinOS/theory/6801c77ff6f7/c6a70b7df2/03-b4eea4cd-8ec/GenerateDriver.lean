/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `strata generate` driver: option validation, teacher forcing of the prompt, how many rows the loop produces,
the speculative-decoding acceptance rule, the routing-trace layer check, and the lent-slot sizing of the batched
prompt path.

## Source semantics
* Validation: `max_new > 0`, `max_context > 0`, `max_new <= max_context`, prompt fits in `max_context - max_new`,
  and `max_context` stays below 2^31 so positions fit int32.
* Loop: for every position before the last prompt position the next input is the prompt's own next token;
  afterwards it is the sampled token.  `produced` receives a token from position `n_prompt - 1` on, so the loop
  stops at position `n_prompt - 2 + max_new` and writes `n_prompt - 1 + max_new` rows.
* Speculation: a window is `[last token, drafts...]`; the number of accepted drafts is the length of the longest
  prefix of drafts equal to the window's argmax outputs; a round emits `accepted + 1` tokens.
* Routing trace: `d.layers` is already advanced when the trace is written, so the recorded layer is
  `d.layers - 1`, valid iff `1 <= d.layers <= 48`.
* Prompt borrowing: `k = ceil(need / blob)` cache slots are lent, so `k * blob >= need`.

## Form template identifiers
validOpts <- the option checks in `main`; nextInput <- `tok = (pos + 1 < n_prompt) ? tokens[pos+1] : next`;
producedAt <- `produced.size()`; accepted <- `a` in the verify loop; ceilDiv <- `(need + blob - 1) / blob`.
-/

namespace Strata.GenerateDriver

-- § option validation

/-- The numeric part of the option checks. -/
structure Opts where
  nTokens : Nat
  maxNew : Nat
  maxContext : Nat

/-- Accepted configurations. -/
def validOpts (o : Opts) : Prop :=
  0 < o.nTokens ∧ 0 < o.maxNew ∧ 0 < o.maxContext ∧ o.maxNew ≤ o.maxContext ∧
    o.nTokens ≤ o.maxContext - o.maxNew ∧ o.maxContext ≤ 2147483647 - 8

/-- A valid configuration always fits prompt plus generation (the later redundant check cannot fire). -/
theorem valid_fits (o : Opts) (h : validOpts o) : o.nTokens + o.maxNew ≤ o.maxContext := by
  unfold validOpts at h
  omega

/-- Every position the loop can reach is a valid int32 with the +8 margin the device path needs. -/
theorem valid_position_int32 (o : Opts) (h : validOpts o) (pos : Nat) (hp : pos < o.maxContext) :
    pos + 8 < 2147483648 := by
  unfold validOpts at h
  omega

-- § teacher forcing

/-- The next input token: the prompt's next token while the prompt lasts, otherwise the sampled one. -/
def nextInput (prompt : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else next

/-- While the prompt lasts the model's guess is ignored. -/
theorem nextInput_prompt (prompt : List Nat) (pos next : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos next = prompt.getD (pos + 1) 0 := by
  unfold nextInput
  rw [if_pos h]

/-- After the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : List Nat) (pos next : Nat) (h : prompt.length ≤ pos + 1) :
    nextInput prompt pos next = next := by
  unfold nextInput
  rw [if_neg (by omega)]

-- § how many rows the loop writes

/-- Tokens in `produced` after processing position `pos` of an `n`-token prompt. -/
def producedAt (n pos : Nat) : Nat := if n ≤ pos + 1 then pos + 2 - n else 0

/-- The last prompt position yields the first generated token. -/
theorem first_token_at_last_prompt (n : Nat) (hn : 1 ≤ n) : producedAt n (n - 1) = 1 := by
  unfold producedAt
  split <;> omega

/-- The loop stops at position `n - 2 + max_new`, where `produced` reaches `max_new`. -/
theorem stops_at_max_new (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) : producedAt n (n + m - 2) = m := by
  unfold producedAt
  split <;> omega

/-- Before that position `produced` is still short. -/
theorem short_before_stop (n m pos : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) (hp : pos < n + m - 2) :
    producedAt n pos < m := by
  unfold producedAt
  split <;> omega

/-- Rows written = positions processed = `n - 1 + max_new`, which is the dump header's row count. -/
theorem rows_written (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) : (n + m - 2) + 1 = n - 1 + m := by
  omega

-- § speculative acceptance

/-- Length of the longest common prefix of the drafts and the verified outputs. -/
def accepted : List Int → List Int → Nat
  | d :: ds, o :: os => if d = o then accepted ds os + 1 else 0
  | _, _ => 0

/-- Never more accepted drafts than drafts. -/
theorem accepted_le : ∀ (ds os : List Int), accepted ds os ≤ ds.length
  | [], _ => by simp [accepted]
  | _ :: _, [] => by simp [accepted]
  | d :: ds, o :: os => by
    simp only [accepted, List.length_cons]
    split
    · have := accepted_le ds os
      omega
    · omega

/-- When every draft is right, all of them are accepted. -/
theorem accepted_self : ∀ ds : List Int, accepted ds ds = ds.length
  | [] => by simp [accepted]
  | d :: ds => by simp [accepted, accepted_self ds]

/-- The first wrong draft ends the round with nothing accepted from it. -/
theorem accepted_stop (d o : Int) (ds os : List Int) (h : d ≠ o) : accepted (d :: ds) (o :: os) = 0 := by
  simp [accepted, h]

/-- A round of window size T (drafts = T - 1) emits at most T tokens. -/
theorem round_emits_le (ds os : List Int) : accepted ds os + 1 ≤ ds.length + 1 := by
  have := accepted_le ds os
  omega

-- § routing trace and prompt borrowing

/-- The layer recorded in the routing trace, from the already-advanced `d.layers`. -/
def traceLayer (layers : Nat) : Nat := layers - 1

/-- The recorded layer is in 0..47 exactly when the advanced counter is in 1..48. -/
theorem traceLayer_valid (layers : Nat) (h1 : 1 ≤ layers) (h2 : layers ≤ 48) : traceLayer layers < 48 := by
  unfold traceLayer
  omega

/-- `(need + blob - 1) / blob`: slots to lend to the prompt path. -/
def ceilDiv (need blob : Nat) : Nat := (need + blob - 1) / blob

/-- The lent slots really hold the prompt path's buffers. -/
theorem lent_slots_hold (need blob : Nat) (hb : 0 < blob) : need ≤ ceilDiv need blob * blob := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) hb
  have h3 := Nat.mul_comm blob ((need + blob - 1) / blob)
  omega

/-- ...and never more than one slot too many. -/
theorem lent_slots_tight (need blob : Nat) (hb : 0 < blob) : ceilDiv need blob * blob < need + blob := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h3 := Nat.mul_comm blob ((need + blob - 1) / blob)
  omega

end Strata.GenerateDriver
