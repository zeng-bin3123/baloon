/-!
# MODULE: teacher_forced_decode_loop
SOURCES: src/program/generate.cpp
The token loop `feed -> 48 layers -> head -> sample -> feed`: teacher forcing while the prompt lasts,
recording generated tokens from the last prompt position on, the stopping rule, the PLE predecessor
window, and the per-token reset of the expert blob's layer axis.

## Source semantics
* Every position is sampled; only positions `pos ≥ n_prompt - 1` push into `produced`.  The prediction
  at the last prompt position IS the first generated token.
* The next input is the prompt's own next token while `pos + 1 < n_prompt`, otherwise the sample.
* The loop stops as soon as `produced.size() ≥ max_new`, so it visits exactly
  `n_prompt - 1 + max_new` positions (the logits-dump header count).
* `ple_prev` is oldest-first: after a token the window becomes `(prev[1], tok)`.
* `d.layers` must restart at 0 every token: blob index `layer * 512 + expert` is only in range and
  only injective for `layer < 48`.

## Form template identifiers
produced.push_back gate → isDecodePos · produced.size() → producedCount · tok update → nextInput ·
ple_prev shift → pleShift · experts.bin index → blobIndex
-/

namespace Strata.Generate.DecodeLoop

-- § which positions produce

/-- A position records its sample iff it is at or after the last prompt position. -/
def isDecodePos (nPrompt pos : Nat) : Bool := decide (nPrompt - 1 ≤ pos)

/-- `produced.size()` after the loop has processed positions `0 .. p-1`. -/
def producedCount (nPrompt : Nat) : Nat → Nat
  | 0 => 0
  | p + 1 => producedCount nPrompt p + (if isDecodePos nPrompt p then 1 else 0)

/-- Closed form: the count is the number of processed positions past the prompt's conditioning ones. -/
theorem producedCount_eq (nPrompt p : Nat) : producedCount nPrompt p = p - (nPrompt - 1) := by
  induction p with
  | zero => simp [producedCount]
  | succ p ih =>
    simp only [producedCount, ih, isDecodePos]
    by_cases h : nPrompt - 1 ≤ p
    · simp [h]; omega
    · simp [h]; omega

/-- No token is produced by the conditioning positions `0 .. n_prompt - 2`. -/
theorem nothing_before_last_prompt (nPrompt : Nat) : producedCount nPrompt (nPrompt - 1) = 0 := by
  rw [producedCount_eq]
  omega

/-- The first generated token comes from the last prompt position. -/
theorem first_token_from_last_prompt (nPrompt : Nat) :
    producedCount nPrompt (nPrompt - 1 + 1) = 1 := by
  rw [producedCount_eq]
  omega

/-- After `n_prompt - 1 + max_new` positions exactly `max_new` tokens are produced. -/
theorem produced_at_stop (nPrompt maxNew : Nat) :
    producedCount nPrompt (nPrompt - 1 + maxNew) = maxNew := by
  rw [producedCount_eq]
  omega

/-- Before that many positions the stop condition has not fired yet, so the loop visits exactly
`n_prompt - 1 + max_new` positions (`max_new > 0` is enforced by the fit check). -/
theorem not_stopped_before {nPrompt maxNew p : Nat} (hm : 0 < maxNew) (hp : p < nPrompt - 1 + maxNew) :
    producedCount nPrompt p < maxNew := by
  rw [producedCount_eq]
  omega

/-- The wall-clock charge goes to decode exactly when the position records a token, so the reported
rate divides decode time by the produced count. -/
theorem decode_charge_matches (nPrompt pos : Nat) :
    isDecodePos nPrompt pos = decide (nPrompt - 1 ≤ pos) := rfl

-- § teacher forcing

/-- The next input: the prompt's next token while it lasts, otherwise the sample. -/
def nextInput (prompt : List Nat) (sample pos : Nat) : Nat :=
  if h : pos + 1 < prompt.length then prompt[pos + 1] else sample

/-- Inside the prompt the next input does not depend on the model's guess. -/
theorem nextInput_forced {prompt : List Nat} {pos : Nat} (h : pos + 1 < prompt.length) (s₁ s₂ : Nat) :
    nextInput prompt s₁ pos = nextInput prompt s₂ pos := by
  simp [nextInput, h]

/-- From the last prompt position on, the sample is fed back. -/
theorem nextInput_free {prompt : List Nat} {pos : Nat} (h : prompt.length ≤ pos + 1) (s : Nat) :
    nextInput prompt s pos = s := by
  have hn : ¬ pos + 1 < prompt.length := by omega
  simp [nextInput, hn]

-- § the PLE predecessor window

/-- `ple_prev[0] = ple_prev[1]; ple_prev[1] = tok` (oldest first). -/
def pleShift (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- After any token stream ending in `a, b` the window holds exactly `(a, b)`, oldest first. -/
theorem pleShift_last_two (w : Int × Int) (xs : List Int) (a b : Int) :
    (xs ++ [a, b]).foldl pleShift w = (a, b) := by
  simp [List.foldl_append, pleShift]

/-- The newest predecessor is always the last token fed. -/
theorem pleShift_newest (w : Int × Int) (t : Int) : (pleShift w t).2 = t := rfl

-- § the expert blob's layer axis

/-- `experts.bin` is indexed `layer * n_expert + expert` with 512 experts per layer. -/
def blobIndex (layer expert : Nat) : Nat := layer * 512 + expert

/-- With the axis reset each token, every index stays inside the 48-layer file. -/
theorem blobIndex_lt {l e : Nat} (hl : l < 48) (he : e < 512) : blobIndex l e < 48 * 512 := by
  unfold blobIndex
  omega

/-- Distinct (layer, expert) pairs read distinct blobs. -/
theorem blobIndex_injective {l l' e e' : Nat} (he : e < 512) (he' : e' < 512)
    (h : blobIndex l e = blobIndex l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h
  omega

/-- A counter left running into the second token asks for layer 48, which lies past the file. -/
theorem stale_layer_out_of_file (e : Nat) : ¬ blobIndex 48 e < 48 * 512 := by
  unfold blobIndex
  omega

end Strata.Generate.DecodeLoop
