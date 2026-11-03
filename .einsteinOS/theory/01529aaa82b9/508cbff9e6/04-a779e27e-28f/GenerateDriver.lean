/-!
# MODULE: generate_driver
Pure contracts of the `strata generate` driver: argument validation, teacher-forced decode loop,
stop rule, logits row count, routing-trace layer index, PCIe share clamp, auto cache sizing and the
speculative window-size rule.

## Source semantics
* The prompt must fit: `0 < max_new ≤ max_context` and `n_prompt ≤ max_context - max_new`.
* The logits dump holds `n_prompt - 1 + max_new` positions.
* While the prompt lasts the next input is the prompt token (teacher forcing); afterwards the sample.
* The routing trace layer index is `d.layers - 1` and must lie in `0..47`.
* `pcie_num` is clamped into `0..256`.
* The verify window `T` stays in `1..S`.

## Form template identifiers
* `o.max_new` → `maxNew`; `o.max_context` → `maxCtx`; `o.tokens` → `prompt`; `produced` → `produced`
* `drive.d.layers` → `layersAfter`; `o.pcie_frac*256` → `x` in `clampPcie`; `o.spec` → `S`
-/

namespace StrataGenerateDriver

/-- The prompt plus generation must fit the configured context. -/
def fits (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

-- § argument validation

/-- A valid configuration bounds the total number of positions by the context. -/
theorem fits_total (nPrompt maxNew maxCtx : Nat) (h : fits nPrompt maxNew maxCtx) :
    nPrompt + maxNew ≤ maxCtx := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

/-- Number of logit rows written: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- The dump count is one less than the total positions when the prompt is non-empty. -/
theorem dumpPositions_eq (nPrompt maxNew : Nat) (h : 1 ≤ nPrompt) :
    dumpPositions nPrompt maxNew + 1 = nPrompt + maxNew := by
  unfold dumpPositions; omega

/-- Parsing of a token-id list: non-empty and every id in `[0, 2^31)`. -/
def parseIds (xs : List Int) : Option (List Int) :=
  if xs = [] then none
  else if xs.all (fun v => decide (0 ≤ v ∧ v ≤ 2147483647)) = true then some xs else none

/-- The empty list is rejected ("token list was empty"). -/
theorem parseIds_nil : parseIds [] = none := by
  simp [parseIds]

/-- A successful parse is never empty. -/
theorem parseIds_ne_nil (xs ys : List Int) (h : parseIds xs = some ys) : ys ≠ [] := by
  unfold parseIds at h
  by_cases h1 : xs = []
  · rw [if_pos h1] at h; cases h
  · rw [if_neg h1] at h
    by_cases h2 : xs.all (fun v => decide (0 ≤ v ∧ v ≤ 2147483647)) = true
    · rw [if_pos h2] at h; cases h; exact h1
    · rw [if_neg h2] at h; cases h

-- § decode loop

/-- The next input token: teacher-forced while the prompt lasts, else the sampled token. -/
def nextInput (prompt : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else next

/-- Inside the prompt the next input is the prompt's own next token. -/
theorem nextInput_prompt (prompt : List Nat) (pos next : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos next = prompt.getD (pos + 1) 0 := by
  unfold nextInput; rw [if_pos h]

/-- Past the prompt the next input is the sampled token. -/
theorem nextInput_gen (prompt : List Nat) (pos next : Nat) (h : prompt.length ≤ pos + 1) :
    nextInput prompt pos next = next := by
  unfold nextInput; rw [if_neg (by omega)]

/-- A position produces an output token iff it is at or after the last prompt position. -/
def recordsFrom (pos nPrompt : Nat) : Prop := nPrompt - 1 ≤ pos

/-- The last prompt position already produces the first generated token. -/
theorem records_at_last_prompt (nPrompt : Nat) : recordsFrom (nPrompt - 1) nPrompt := by
  unfold recordsFrom; omega

/-- Stop rule: enough tokens, or EOS emitted at a generating position. -/
def shouldStop (producedLen maxNew : Nat) (stopEos : Bool) (pos nPrompt : Nat) (isEos : Bool) : Bool :=
  decide (maxNew ≤ producedLen) || (stopEos && decide (nPrompt - 1 ≤ pos) && isEos)

/-- While conditioning on the prompt with nothing produced, the loop never stops. -/
theorem no_stop_in_prompt (maxNew : Nat) (stopEos : Bool) (pos nPrompt : Nat) (isEos : Bool)
    (hm : 0 < maxNew) (hp : pos + 1 < nPrompt) :
    shouldStop 0 maxNew stopEos pos nPrompt isEos = false := by
  have h1 : ¬ maxNew ≤ 0 := by omega
  have h2 : ¬ nPrompt - 1 ≤ pos := by omega
  simp [shouldStop, h1, h2]

/-- PLE window advance: the older slot takes the newer, the newer takes the token. -/
def advancePle (w : Int × Int) (tok : Int) : Int × Int := (w.2, tok)

/-- After advancing, the newest slot holds the token. -/
theorem advancePle_newest (w : Int × Int) (tok : Int) : (advancePle w tok).2 = tok := rfl

/-- After advancing, the older slot holds the previous newest. -/
theorem advancePle_older (w : Int × Int) (tok : Int) : (advancePle w tok).1 = w.2 := rfl

/-- The PLE window at the start of the token path (`-1` = no predecessor). -/
def initPle (prompt : List Int) (ps : Nat) : Int × Int :=
  (if 2 ≤ ps then prompt.getD (ps - 2) (-1) else -1,
   if 1 ≤ ps then prompt.getD (ps - 1) (-1) else -1)

/-- At position zero there are no predecessors. -/
theorem initPle_zero (prompt : List Int) : initPle prompt 0 = (-1, -1) := by
  simp [initPle]

-- § traces and resources

/-- The routing trace layer index, taken after the adapter advanced `layers`. -/
def routingLayer (layersAfter : Int) : Int := layersAfter - 1

/-- A layer index is a valid trace layer. -/
def routingValid (l : Int) : Prop := 0 ≤ l ∧ l < 48

/-- The trace index is valid exactly when the adapter's counter is in `1..48`. -/
theorem routingValid_iff (a : Int) : routingValid (routingLayer a) ↔ 1 ≤ a ∧ a ≤ 48 := by
  unfold routingValid routingLayer
  constructor <;> intro h <;> omega

/-- Bytes of one routing record: layer, k, k ids, k weights. -/
def recordBytes (k : Nat) : Nat := 8 + 4 * k + 4 * k

/-- For the top-10 router a record is 88 bytes. -/
theorem recordBytes_ten : recordBytes 10 = 88 := by decide

/-- Clamp of the PCIe share numerator. -/
def clampPcie (x : Int) : Int := max 0 (min 256 x)

/-- The clamped share lies in `0..256`. -/
theorem clampPcie_bounds (x : Int) : 0 ≤ clampPcie x ∧ clampPcie x ≤ 256 := by
  unfold clampPcie; omega

/-- In range, clamping is the identity. -/
theorem clampPcie_id (x : Int) (h0 : 0 ≤ x) (h1 : x ≤ 256) : clampPcie x = x := by
  unfold clampPcie; omega

/-- `--expert-cache auto`: slots that fit after the reserve. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized slots never exceed the available memory. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- Swaps kept after the resize to `adapt_swaps`. -/
def swapsKept (n cap : Nat) : Nat := min n cap

/-- At most `adapt_swaps` swaps are submitted. -/
theorem swapsKept_le (n cap : Nat) : swapsKept n cap ≤ cap := by
  unfold swapsKept; omega

-- § speculative window

/-- Grow the verify window while drafts stay probable, bounded by `S`. -/
def growT (S : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, T => T
  | fuel + 1, T => if T < S ∧ ok T = true then growT S ok fuel (T + 1) else T

/-- The window size stays within `1..S`. -/
theorem growT_bounds (S : Nat) (ok : Nat → Bool) (fuel T : Nat) (h1 : 1 ≤ T) (h2 : T ≤ S) :
    1 ≤ growT S ok fuel T ∧ growT S ok fuel T ≤ S := by
  induction fuel generalizing T with
  | zero => exact ⟨h1, h2⟩
  | succ n ih =>
    simp only [growT]
    by_cases h : T < S ∧ ok T = true
    · have hT : T < S := h.1
      rw [if_pos h]
      exact ih (T + 1) (by omega) (by omega)
    · rw [if_neg h]
      exact ⟨h1, h2⟩

end StrataGenerateDriver
