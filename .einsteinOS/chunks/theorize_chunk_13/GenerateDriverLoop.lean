/-!
# MODULE: generate_driver_loop
Pure arithmetic and control-flow contracts of the `strata generate` driver: token-list validation, context
fitting, argmax, routing-trace layer index, teacher forcing, VRAM expert-cache sizing, PCIe share clamping
and speculative-window acceptance.

## Source semantics
* Token ids parse only when the list is non-empty and every id lies in [0, 2^31).
* `max_new > 0`, `max_new <= max_context` and `prompt <= max_context - max_new` jointly imply the
  prompt plus generation fits the context.
* `drive_pool` records `d.layers - 1`, valid only in 0..47.
* While the prompt lasts the next input is the prompt's own next token (teacher forcing).
* The logits dump has `n_prompt - 1 + max_new` positions.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots (never negative).
* The PCIe share numerator is clamped to [0, 256].
* A verify window accepts the longest prefix of drafts matching the model's argmax outputs.

## Form template identifiers
- `parse_i64_list` → `tokenListValid`
- `argmax` → `argmaxIdx`
- `drive_pool` layer index → `traceLayerIdx`
- `o.pcie_frac * 256` clamp → `clampPcie`
- `a` accepted count in verify loop → `acceptCount`
- `dump_positions` → `dumpPositions`
-/

-- § token list validation

/-- A parsed token list is valid when non-empty and every id is below 2^31. -/
def tokenListValid (l : List Nat) : Prop := l ≠ [] ∧ ∀ x ∈ l, x < 2147483648

/-- The empty token list is rejected. -/
theorem tokenListValid_nil : ¬ tokenListValid [] := by
  intro h
  exact h.1 rfl

/-- A singleton in range is accepted. -/
theorem tokenListValid_single (x : Nat) (hx : x < 2147483648) : tokenListValid [x] := by
  refine ⟨by simp, ?_⟩
  intro y hy
  have : y = x := by simpa using hy
  omega

-- § context fitting

/-- The CLI's checks imply the prompt plus generation fits in the context. -/
theorem prompt_fits_context (n m c : Nat) (hm : 0 < m) (hmc : m ≤ c) (hn : n ≤ c - m) :
    n + m ≤ c := by
  omega

/-- Number of logits rows positions: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- With a non-empty prompt, dump positions plus one equals prompt plus new tokens. -/
theorem dumpPositions_succ (n m : Nat) (hn : 1 ≤ n) : dumpPositions n m + 1 = n + m := by
  unfold dumpPositions
  omega

-- § argmax

/-- Loop of `argmax`: scanning from index `i` with the current best index and value. -/
def argmaxAux : List Int → Nat → Nat → Int → Nat
  | [], _, best, _ => best
  | x :: xs, i, best, bv =>
      if bv < x then argmaxAux xs (i + 1) i x else argmaxAux xs (i + 1) best bv

/-- Argmax of a vector, starting from index 0 as the C++ does. -/
def argmaxIdx : List Int → Nat
  | [] => 0
  | x :: xs => argmaxAux xs 1 0 x

/-- Helper: the scan result stays below `i + length`. -/
theorem argmaxAux_lt (l : List Int) : ∀ (i best : Nat) (bv : Int), best < i →
    argmaxAux l i best bv < i + l.length := by
  induction l with
  | nil => intro i best bv h; simpa [argmaxAux] using h
  | cons x xs ih =>
    intro i best bv h
    unfold argmaxAux
    by_cases hc : bv < x
    · rw [if_pos hc]
      have := ih (i + 1) i x (by omega)
      simp only [List.length_cons]
      omega
    · rw [if_neg hc]
      have := ih (i + 1) best bv (by omega)
      simp only [List.length_cons]
      omega

/-- The argmax of a non-empty vector is a valid index. -/
theorem argmaxIdx_lt (v : List Int) (hv : v ≠ []) : argmaxIdx v < v.length := by
  cases v with
  | nil => exact absurd rfl hv
  | cons x xs =>
    have := argmaxAux_lt xs 1 0 x (by omega)
    simp only [argmaxIdx, List.length_cons]
    omega

-- § routing trace layer index

/-- The layer index written to the routing trace (the adapter has already advanced the counter). -/
def traceLayerIdx (layersAfter : Int) : Int := layersAfter - 1

/-- The trace index is accepted exactly when the counter is in 1..48. -/
theorem traceLayerIdx_valid (n : Int) :
    (0 ≤ traceLayerIdx n ∧ traceLayerIdx n < 48) ↔ (1 ≤ n ∧ n ≤ 48) := by
  unfold traceLayerIdx
  omega

-- § teacher forcing

/-- The next input token: the prompt's own while it lasts, else the sampled one. -/
def nextInput (toks : List Nat) (pos nxt : Nat) : Nat :=
  if pos + 1 < toks.length then toks.getD (pos + 1) nxt else nxt

/-- Past the end of the prompt, the sampled token is fed back. -/
theorem nextInput_decode (toks : List Nat) (pos nxt : Nat) (h : toks.length ≤ pos + 1) :
    nextInput toks pos nxt = nxt := by
  unfold nextInput
  rw [if_neg (by omega)]

/-- Inside the prompt the prompt's own token is fed. -/
theorem nextInput_prompt (toks : List Nat) (pos nxt : Nat) (h : pos + 1 < toks.length) :
    nextInput toks pos nxt = toks.getD (pos + 1) nxt := by
  unfold nextInput
  rw [if_pos h]

-- § VRAM expert cache sizing

/-- Slots of `--expert-cache auto`: free memory minus the reserve, divided by the largest blob. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots never exceed the available bytes. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve := by
  unfold autoSlots
  exact Nat.div_mul_le_self _ _

/-- With no memory beyond the reserve, no slots are granted. -/
theorem autoSlots_zero (free reserve blob : Nat) (h : free ≤ reserve) :
    autoSlots free reserve blob = 0 := by
  unfold autoSlots
  have : free - reserve = 0 := by omega
  simp [this]

-- § PCIe share

/-- The PCIe numerator clamp to [0, 256]. -/
def clampPcie (x : Int) : Int := max 0 (min 256 x)

/-- The clamped share is within range. -/
theorem clampPcie_range (x : Int) : 0 ≤ clampPcie x ∧ clampPcie x ≤ 256 := by
  unfold clampPcie
  omega

/-- Clamping is idempotent. -/
theorem clampPcie_idem (x : Int) : clampPcie (clampPcie x) = clampPcie x := by
  unfold clampPcie
  omega

-- § speculative verify window

/-- Number of leading drafts that match the model's outputs. -/
def acceptCount : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptCount ds os + 1 else 0
  | _, _ => 0

/-- Accepted drafts never exceed the number of drafts. -/
theorem acceptCount_le_drafts (ds os : List Nat) : acceptCount ds os ≤ ds.length := by
  induction ds generalizing os with
  | nil => simp [acceptCount]
  | cons d ds ih =>
    cases os with
    | nil => simp [acceptCount]
    | cons o os =>
      unfold acceptCount
      by_cases h : d = o
      · rw [if_pos h]
        have := ih os
        simp only [List.length_cons]
        omega
      · rw [if_neg h]
        omega

/-- When every draft equals the output, all drafts are accepted. -/
theorem acceptCount_self (ds : List Nat) : acceptCount ds ds = ds.length := by
  induction ds with
  | nil => simp [acceptCount]
  | cons d ds ih => simp [acceptCount, ih]

/-- A round emits `accepted + 1` tokens, at most the window size `T`. -/
theorem round_emits_le_window (ds os : List Nat) :
    acceptCount ds os + 1 ≤ ds.length + 1 := by
  have := acceptCount_le_drafts ds os
  omega
