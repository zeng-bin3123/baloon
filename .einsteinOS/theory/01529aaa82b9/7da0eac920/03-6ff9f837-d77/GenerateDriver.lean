/-!
# MODULE: generate_driver
The `strata generate` driver: validates the option budget, runs the teacher-forced
`embed -> layers -> head -> sample` loop, sizes the VRAM expert tier, and runs the speculative
verify rounds (accept-prefix, adaptive tier swaps).  This theory covers its arithmetic contracts.

## Source semantics
* The prompt is conditioned position by position (teacher forcing); the prediction at the last prompt
  position is the first generated token, so `n_prompt - 1 + max_new` positions are processed.
* `--max-new` and the prompt must fit `--max-context`; positions never reach `max_context`.
* The routing trace layer index is `d.layers - 1` and must lie in `0..47`.
* Auto expert-cache sizing never grants more slots than the free VRAM minus the reserve holds.
* Lent prompt slots: `ceil(need / blob)` slots always cover `need` bytes.
* A speculative round commits `accepted + 1` tokens where `accepted` is the matching draft prefix.
* An adaptive tier swap is only performed when the candidate is routed clearly more often.

## Form template identifiers
* `nPrompt`   <- `n_prompt` / `o.tokens.size()`
* `maxNew`    <- `o.max_new`
* `maxCtx`    <- `o.max_context`
* `autoSlots` <- the `o.expert_cache < 0` branch
* `lentSlots` <- `k` in the prefill borrow computation
* `acceptP`   <- the `while (a < T-1 && window[a+1] == outv[a]) ++a` loop
-/

namespace Strata.Generate

-- § option budget

/-- The budget check in `main`: positive `max_new`, `max_context`, and the prompt fits beside `max_new`. -/
def validBudget (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

/-- A valid budget means prompt plus generation fits in the context. -/
theorem validBudget_fits (n m c : Nat) (h : validBudget n m c) : n + m ≤ c := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

/-- With a nonempty prompt, every processed position index is below `max_context`. -/
theorem validBudget_positions_in_range (n m c : Nat) (h : validBudget n m c) (hn : 1 ≤ n) :
    n - 1 + m < c := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  omega

-- § positions, produced tokens and the logits dump

/-- Number of positions the token loop processes: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Tokens recorded after processing `k` positions: recording starts at position `n_prompt - 1`. -/
def producedAfter (nPrompt k : Nat) : Nat := k - (nPrompt - 1)

/-- Processing all `dumpPositions` positions produces exactly `max_new` tokens. -/
theorem producedAfter_dumpPositions (n m : Nat) : producedAfter n (dumpPositions n m) = m := by
  unfold producedAfter dumpPositions
  omega

/-- Before the last position the loop has not yet produced `max_new` tokens, so it does not stop early. -/
theorem producedAfter_lt_before_end (n m k : Nat) (hm : 0 < m) (h : k < dumpPositions n m) :
    producedAfter n k < m := by
  unfold producedAfter dumpPositions at *
  omega

/-- The header row count equals the positions that carry a generated token plus the conditioning ones. -/
theorem dump_decode_rows (n m : Nat) : dumpPositions n m - (n - 1) = m := by
  unfold dumpPositions
  omega

-- § teacher forcing and the PLE window

/-- The next input token: the prompt's own next token while the prompt lasts, else the sampled one. -/
def nextInput (prompt : List Nat) (sampled pos : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else sampled

/-- While the prompt lasts the input is forced to the prompt token. -/
theorem nextInput_forced (prompt : List Nat) (s pos : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt s pos = prompt.getD (pos + 1) 0 := by
  simp [nextInput, h]

/-- After the prompt the sampled token is used. -/
theorem nextInput_free (prompt : List Nat) (s pos : Nat) (h : prompt.length ≤ pos + 1) :
    nextInput prompt s pos = s := by
  have : ¬ (pos + 1 < prompt.length) := by omega
  simp [nextInput, this]

/-- The PLE predecessor window (oldest first) advances by dropping the oldest and appending the token. -/
def advance (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- Two advances leave exactly the last two tokens in the window. -/
theorem advance_twice (w : Int × Int) (a b : Int) : advance (advance w a) b = (a, b) := rfl

-- § routing trace layer index

/-- The layer index recorded in the trace is the adapter's layer counter minus one. -/
def traceLayer (layersAfter : Int) : Int := layersAfter - 1

/-- The trace index is in `0..47` exactly when the adapter's counter is in `1..48`. -/
theorem traceLayer_in_range (L : Int) :
    (0 ≤ traceLayer L ∧ traceLayer L < 48) ↔ (1 ≤ L ∧ L ≤ 48) := by
  unfold traceLayer
  omega

-- § VRAM tier sizing

/-- `--expert-cache auto`: slots that fit in the free memory minus the reserve. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- Auto sizing never over-commits the free memory. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Slots needed to cover `need` bytes with blobs of `blob` bytes. -/
def lentSlots (need blob : Nat) : Nat := (need + blob - 1) / blob

/-- The lent slots always cover the prompt path's buffer requirement. -/
theorem lentSlots_cover (need blob : Nat) (hb : 0 < blob) : need ≤ blob * lentSlots need blob := by
  unfold lentSlots
  have h := Nat.div_add_mod (need + blob - 1) blob
  have hm := Nat.mod_lt (need + blob - 1) hb
  generalize (need + blob - 1) / blob = q at h ⊢
  generalize (need + blob - 1) % blob = r at h hm
  omega

/-- The PCIe share (in 1/256 units) is clamped to `0..256`. -/
def pcieNum (x : Int) : Int := max 0 (min 256 x)

/-- The clamp stays inside the valid range. -/
theorem pcieNum_bounds (x : Int) : 0 ≤ pcieNum x ∧ pcieNum x ≤ 256 := by
  unfold pcieNum
  omega

-- § speculative rounds

/-- Length of the matching draft prefix: `drafts[i] == outputs[i]` for all `i < a`. -/
def acceptP : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptP ds os + 1 else 0
  | _, _ => 0

/-- The accepted prefix is never longer than the drafts. -/
theorem acceptP_le_drafts (ds os : List Nat) : acceptP ds os ≤ ds.length := by
  induction ds generalizing os with
  | nil => simp [acceptP]
  | cons d ds ih =>
    cases os with
    | nil => simp [acceptP]
    | cons o os =>
      have := ih os
      by_cases h : d = o <;> simp [acceptP, h, this] <;> omega

/-- The accepted prefix is never longer than the verifier outputs. -/
theorem acceptP_le_outputs (ds os : List Nat) : acceptP ds os ≤ os.length := by
  induction ds generalizing os with
  | nil => simp [acceptP]
  | cons d ds ih =>
    cases os with
    | nil => simp [acceptP]
    | cons o os =>
      have := ih os
      by_cases h : d = o <;> simp [acceptP, h, this] <;> omega

/-- Identical lists are fully accepted. -/
theorem acceptP_self (ds : List Nat) : acceptP ds ds = ds.length := by
  induction ds with
  | nil => simp [acceptP]
  | cons d ds ih => simp [acceptP, ih]

/-- A round commits `a + 1` tokens, which is at least one and at most the window size `T`. -/
theorem commit_bounds (T a : Nat) (hT : 1 ≤ T) (ha : a ≤ T - 1) : 1 ≤ a + 1 ∧ a + 1 ≤ T := by
  omega

-- § adaptive tier swaps

/-- A swap is allowed when `cand >= vict + 1.5`, written with both sides doubled. -/
def swapAllowed (cand2 vict2 : Nat) : Prop := vict2 + 3 ≤ cand2

/-- Doubled usages: an allowed swap has a strictly more used candidate than the victim. -/
theorem swapAllowed_gain_pos (c v : Nat) (h : swapAllowed (2 * c) (2 * v)) : v < c := by
  unfold swapAllowed at h
  omega

/-- Truncating the ranked swaps to `--adapt-swaps` never exceeds the cap. -/
theorem swaps_truncated {α : Type} (l : List α) (cap : Nat) : (l.take cap).length ≤ cap :=
  List.length_take_le _ _

-- § native pack constraints

/-- A native (IQ) pack needs `--native`, `--spec >= 2`, no `--keep-canonical`, and a prefill chunk when the prompt is long. -/
def nativePackOk (haveNative : Bool) (spec : Nat) (keep : Bool) (prefillChunk nTok : Nat) : Prop :=
  haveNative = true ∧ 2 ≤ spec ∧ keep = false ∧ (0 < prefillChunk ∨ nTok ≤ 1)

/-- Accepted native-pack configuration has a verify window of at least two tokens. -/
theorem nativePackOk_spec (h : Bool) (s : Nat) (k : Bool) (c n : Nat) (ok : nativePackOk h s k c n) : 2 ≤ s :=
  ok.2.1

end Strata.Generate
