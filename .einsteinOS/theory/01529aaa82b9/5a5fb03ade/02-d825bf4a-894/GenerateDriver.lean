/-!
# MODULE: generate_driver
The `strata generate` driver: validates options, runs embed -> layers -> head -> sample with teacher forcing over
the prompt, sizes the VRAM expert tier, and accepts speculative drafts by longest matching prefix.

## Source semantics
* `max_new > 0`, `max_context > 0`, `max_new <= max_context`, and the prompt fits in `max_context - max_new`.
* The loop writes one logits row per position; `produced` only starts receiving at position `n_prompt - 1`, so the
  dump holds `n_prompt - 1 + max_new` rows (before striding).
* Inside the prompt the next input is the prompt's own next token (teacher forcing); afterwards it is the sample.
* `--logits-stride > 1` needs `--max-new 1` and `--dump-logits`.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots, which can never exceed the free VRAM.
* A verify window accepts the longest prefix of drafts equal to the engine's own argmax, so at most `T - 1`.

## Form template identifiers
* `n_prompt` -> `nPrompt`, `max_new` -> `maxNew`, `max_context` -> `maxCtx`
* `produced.size()` -> `producedAfter n p`, `a` (accepted drafts) -> `acceptLen drafts outs`
-/

namespace Strata.GenerateDriver

-- § option validation

/-- The argument check at the top of `main` for prompt, generation and context sizes. -/
def contextFits (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

/-- A validated configuration holds the prompt plus every generated token. -/
theorem contextFits_total (n m c : Nat) (h : contextFits n m c) : n + m ≤ c := by
  unfold contextFits at h
  omega

/-- `--logits-stride > 1` is only legal with `--max-new 1` and a dump file. -/
def strideOk (stride maxNew : Nat) (hasDump : Bool) : Prop :=
  1 < stride → maxNew = 1 ∧ hasDump = true

/-- A valid stride configuration with stride above one forces a single new token. -/
theorem strideOk_forces_single (s m : Nat) (d : Bool) (h : strideOk s m d) (hs : 1 < s) :
    m = 1 := (h hs).1

/-- A valid stride configuration with stride above one forces a dump file. -/
theorem strideOk_forces_dump (s m : Nat) (d : Bool) (h : strideOk s m d) (hs : 1 < s) :
    d = true := (h hs).2

-- § the token loop

/-- Rows of the logits dump before striding: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Number of generated tokens recorded after processing 0-indexed position `p`. -/
def producedAfter (n p : Nat) : Nat := if n ≤ p + 1 then p + 2 - n else 0

/-- The last prompt position yields exactly the first generated token. -/
theorem produced_first (n : Nat) (h : 1 ≤ n) : producedAfter n (n - 1) = 1 := by
  simp only [producedAfter]
  split <;> omega

/-- The loop reaches `max_new` tokens exactly at position `n + m - 2`. -/
theorem produced_reaches (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) :
    producedAfter n (n + m - 2) = m := by
  simp only [producedAfter]
  split <;> omega

/-- Before that position the loop has produced fewer than `max_new` tokens, so it cannot stop early. -/
theorem produced_before (n m p : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) (hp : p + 2 < n + m) :
    producedAfter n p < m := by
  simp only [producedAfter]
  split <;> omega

/-- Positions processed equals `dumpPositions`: the header row count describes the file written. -/
theorem positions_eq_dump (n m : Nat) (hn : 1 ≤ n) (hm : 1 ≤ m) :
    (n + m - 2) + 1 = dumpPositions n m := by
  unfold dumpPositions
  omega

/-- Rows kept by a stride: the ceiling of positions over stride. -/
def rowCount (positions stride : Nat) : Nat := (positions + stride - 1) / stride

/-- With stride one every position is stored. -/
theorem rowCount_one (p : Nat) : rowCount p 1 = p := by
  simp [rowCount]

/-- Ceiling division never under-counts. -/
theorem ceil_div_mul (need blob : Nat) (h : 0 < blob) : need ≤ (need + blob - 1) / blob * blob := by
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) h
  generalize (need + blob - 1) / blob = q at *
  generalize (need + blob - 1) % blob = r at *
  rw [Nat.mul_comm] at h1
  omega

/-- The strided dump covers all positions. -/
theorem rowCount_cover (p s : Nat) (h : 0 < s) : p ≤ rowCount p s * s :=
  ceil_div_mul p s h

/-- Teacher forcing: the next input inside the prompt is the prompt's token, afterwards the sample. -/
def nextInput (prompt : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else next

/-- Inside the prompt the model's guess is ignored. -/
theorem nextInput_prompt (prompt : List Nat) (pos next : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos next = prompt.getD (pos + 1) 0 := by
  simp [nextInput, h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : List Nat) (pos next : Nat) (h : ¬ pos + 1 < prompt.length) :
    nextInput prompt pos next = next := by
  simp [nextInput, h]

/-- The routing trace stores `d.layers - 1`, which is a valid layer iff the adapter advanced it once. -/
theorem routing_layer_in_range (d : Nat) (h1 : 1 ≤ d) (h2 : d ≤ 48) : d - 1 < 48 := by
  omega

-- § VRAM tier sizing

/-- Slots granted by `--expert-cache auto`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized tier never exceeds the free VRAM left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Prompt-path slot lending is allowed only while 128 slots remain for decode. -/
def canLend (slots k : Nat) : Prop := k + 128 ≤ slots

/-- After lending, at least 128 slots stay with the decode tier. -/
theorem canLend_remaining (slots k : Nat) (h : canLend slots k) : 128 ≤ slots - k := by
  unfold canLend at h
  omega

-- § speculative acceptance

/-- Longest prefix of drafts that equals the verifier's own outputs (`a` in the source). -/
def acceptLen : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptLen ds os + 1 else 0
  | _, _ => 0

/-- At most every draft is accepted. -/
theorem acceptLen_le : ∀ (ds os : List Nat), acceptLen ds os ≤ ds.length
  | [], _ => by simp [acceptLen]
  | _ :: _, [] => by simp [acceptLen]
  | d :: ds, o :: os => by
      have ih := acceptLen_le ds os
      simp only [acceptLen, List.length_cons]
      split <;> omega

/-- Tokens committed by a window (`a + 1`) never exceed the window length `T = drafts + 1`. -/
theorem commit_le_window (ds os : List Nat) : acceptLen ds os + 1 ≤ ds.length + 1 := by
  have := acceptLen_le ds os
  omega

/-- Matching drafts are all accepted when outputs agree with them. -/
theorem acceptLen_self : ∀ ds : List Nat, acceptLen ds ds = ds.length
  | [] => by simp [acceptLen]
  | d :: ds => by
      have ih := acceptLen_self ds
      simp [acceptLen, ih]

end Strata.GenerateDriver
