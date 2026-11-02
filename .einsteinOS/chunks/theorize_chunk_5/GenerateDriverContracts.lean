/-!
# MODULE: generate_driver_contracts
Pure contracts of the `strata generate` driver: argument validation, the teacher-forced decode loop and its
row accounting, prompt batching and lent-slot arithmetic, expert-cache sizing, speculative acceptance,
M-RoPE image positions and the serve request guards.

## Source semantics
* Validation: `0 < max_new ≤ max_context` and `n_prompt ≤ max_context - max_new`.
* Decode loop: positions `0 ..`; the next input is the prompt token while `pos+1 < n_prompt`, else the sample;
  a token is recorded from `pos ≥ n_prompt-1`; the loop stops once `max_new` tokens exist, so the number of
  processed positions (= logits rows) is `n_prompt - 1 + max_new`.
* Batched prefill covers `[0, n_batched)` with `n_batched = prefill_until` if `0 < prefill_until < n_prompt-1`
  else `n_prompt-1`; the token loop resumes at `pos_start = n_batched`.
* Borrowing: `k + 128 ≤ slots` slots from the end are lent, the first lent slot index is `slots - k`.
* Speculation: window `[x, d_1 .. d_{T-1}]`; `a` = longest prefix with `window[i+1] == out[i]`; emit `a+1`
  tokens, advance `p += a+1`.
* Image rows: row `j` of an image with `nx` columns sits at `(t, h, w) = (p, p + j/nx, p + j%nx)`; text after
  the image continues at `p + max(nx, ny)`.

## Form template identifiers
Options.max_new -> max_new ; Options.max_context -> max_context ; produced -> generated ;
pos_start -> resume_position ; lent -> lent_slots ; window/outv -> verify_window/verify_outputs ;
put(c,t,h,w) -> mrope_cell ; pcie_num -> pcie_share
-/

namespace Strata.Program.Generate

-- § argument validation

/-- The accepted argument shape. -/
def validContext (max_new max_ctx ntok : Nat) : Prop :=
  0 < max_new ∧ 0 < max_ctx ∧ max_new ≤ max_ctx ∧ ntok ≤ max_ctx - max_new

/-- Validation implies the later "context cannot hold prompt + new tokens" check never fires. -/
theorem validContext_fits (max_new max_ctx ntok : Nat) (h : validContext max_new max_ctx ntok) :
    ntok + max_new ≤ max_ctx := by
  unfold validContext at h
  omega

/-- With `max_context < 2^31 - 8` every position and cell index fits int32. -/
theorem positions_fit_int32 (max_ctx pos : Nat) (h1 : max_ctx ≤ 2147483647 - 8) (h2 : pos < max_ctx) :
    pos + 8 ≤ 2147483647 := by omega

/-- A token id accepted by the parser. -/
def validTokenId (id : Int) : Prop := 0 ≤ id ∧ id ≤ 2147483647

/-- Every accepted id fits a non-negative int32. -/
theorem validTokenId_int32 (id : Int) (h : validTokenId id) : -2147483648 ≤ id ∧ id ≤ 2147483647 := by
  unfold validTokenId at h
  omega

-- § the decode loop

/-- Number of logits rows written by the loop (`n_prompt - 1 + max_new`). -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Tokens recorded after processing position `pos` (0-based). -/
def producedAfter (nPrompt pos : Nat) : Nat := if pos + 1 ≥ nPrompt then pos + 2 - nPrompt else 0

/-- The last prompt position records the first generated token. -/
theorem first_token_at_last_prompt (nPrompt : Nat) (h : 1 ≤ nPrompt) :
    producedAfter nPrompt (nPrompt - 1) = 1 := by
  unfold producedAfter
  split <;> omega

/-- Prompt positions before the last record nothing. -/
theorem prompt_positions_record_nothing (nPrompt pos : Nat) (h : pos + 1 < nPrompt) :
    producedAfter nPrompt pos = 0 := by
  unfold producedAfter
  split <;> omega

/-- The loop stops at position `n_prompt + max_new - 2`, where `max_new` tokens exist; it is the last row written. -/
theorem stop_position_rows (nPrompt maxNew : Nat) (h1 : 1 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAfter nPrompt (nPrompt + maxNew - 2) = maxNew ∧
      (nPrompt + maxNew - 2) + 1 = dumpPositions nPrompt maxNew := by
  unfold producedAfter dumpPositions
  constructor
  · split <;> omega
  · omega

/-- Teacher forcing: the next input. -/
def nextInput (pos nPrompt : Nat) (promptTok sampled : Int) : Int :=
  if pos + 1 < nPrompt then promptTok else sampled

/-- While the prompt lasts the model's own guess is ignored. -/
theorem teacher_forced (pos nPrompt : Nat) (promptTok sampled : Int) (h : pos + 1 < nPrompt) :
    nextInput pos nPrompt promptTok sampled = promptTok := by
  unfold nextInput
  simp [h]

/-- Past the prompt the sampled token is fed back. -/
theorem free_running (pos nPrompt : Nat) (promptTok sampled : Int) (h : nPrompt ≤ pos + 1) :
    nextInput pos nPrompt promptTok sampled = sampled := by
  unfold nextInput
  have : ¬ pos + 1 < nPrompt := by omega
  simp [this]

-- § prompt batching

/-- Number of batched positions. -/
def nBatched (nPrompt prefillUntil : Nat) : Nat :=
  if 0 < prefillUntil ∧ prefillUntil < nPrompt - 1 then prefillUntil else nPrompt - 1

/-- The batched prefix never swallows the last prompt position. -/
theorem nBatched_le (nPrompt pu : Nat) : nBatched nPrompt pu ≤ nPrompt - 1 := by
  unfold nBatched
  split <;> omega

/-- The resume position is a valid prompt index. -/
theorem resume_in_prompt (nPrompt pu : Nat) (h : 1 ≤ nPrompt) : nBatched nPrompt pu < nPrompt := by
  have := nBatched_le nPrompt pu
  omega

/-- Lent-slot arithmetic: the lent block is `[slots - k, slots)` and at least 128 slots stay in service. -/
theorem lent_slots (k slots : Nat) (h : k + 128 ≤ slots) :
    128 ≤ slots - k ∧ (slots - k) + k = slots := by omega

/-- Expert-cache auto sizing: the granted slots fit the free memory minus the reserve. -/
theorem auto_slots_fit (free reserve blob : Nat) :
    (free - reserve) / blob * blob ≤ free - reserve := Nat.div_mul_le_self _ _

/-- A profile truncates the grant. -/
theorem profile_truncates (slots psize : Nat) : min slots psize ≤ slots := Nat.min_le_left _ _

-- § routing trace and clamps

/-- The adapter has advanced `layers` past the layer it just ran, so the traced index is `layers - 1`. -/
theorem trace_layer_index (l : Nat) (h : l < 48) : (l + 1) - 1 = l ∧ l + 1 ≤ 48 := by omega

/-- The PCIe share clamp lies in [0, 256]. -/
theorem pcie_clamp (x : Int) : 0 ≤ max 0 (min 256 x) ∧ max 0 (min 256 x) ≤ 256 := by omega

/-- Truncating a swap list to the budget never exceeds the budget. -/
theorem swaps_truncated {α : Type} (l : List α) (n : Nat) : (l.take n).length ≤ n :=
  List.length_take_le _ _

-- § speculative acceptance

/-- Longest prefix of drafts equal to the verifier outputs. -/
def accept : List Int → List Int → Nat
  | d :: ds, o :: os => if d = o then 1 + accept ds os else 0
  | _, _ => 0

/-- Acceptance never exceeds the number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Int), accept ds os ≤ ds.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      unfold accept
      have := accept_le_left ds os
      split <;> simp <;> omega

/-- Acceptance never exceeds the number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Int), accept ds os ≤ os.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      unfold accept
      have := accept_le_right ds os
      split <;> simp <;> omega

/-- The accepted drafts are exactly the verifier's own outputs, so the emitted tokens equal greedy decode. -/
theorem accept_prefix : ∀ (ds os : List Int), os.take (accept ds os) = ds.take (accept ds os)
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      by_cases h : d = o
      · have e : accept (d :: ds) (o :: os) = accept ds os + 1 := by
          simp only [accept]
          rw [if_pos h]
          omega
        rw [e, List.take_succ_cons, List.take_succ_cons, accept_prefix ds os, h]
      · have e : accept (d :: ds) (o :: os) = 0 := by
          simp only [accept]
          rw [if_neg h]
        rw [e]
        simp

/-- A round emits at most `T` tokens and advances by at most `T` cells. -/
theorem round_emits_le (T a : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by omega

/-- The emitted index range `0..a` is inside the `T` verifier outputs. -/
theorem emitted_in_outputs (T a i : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) (hi : i ≤ a) : i < T := by
  omega

/-- Window growth loop with fuel: extend while below `S` and the draft probability gate holds. -/
def growWindow (S : Nat) (ok : Nat → Bool) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then growWindow S ok f (T + 1) else T

/-- The window size stays in `[1, S]`. -/
theorem growWindow_bounds (S : Nat) (ok : Nat → Bool) :
    ∀ (f T : Nat), 1 ≤ T → T ≤ S → 1 ≤ growWindow S ok f T ∧ growWindow S ok f T ≤ S := by
  intro f
  induction f with
  | zero => intro T h1 h2; simp [growWindow]; omega
  | succ f ih =>
      intro T h1 h2
      unfold growWindow
      by_cases h : T < S ∧ ok (T - 1) = true
      · simp only [h, if_true]
        exact ih (T + 1) (by omega) (by omega)
      · simp only [h, if_false]
        omega

-- § serve request guards and M-RoPE

/-- An accepted request leaves room for the eight-cell margin. -/
theorem request_room (n maxNew ctx : Nat) (h : ¬ (n + maxNew + 8 > ctx)) : n + maxNew + 8 ≤ ctx := by
  omega

/-- Image row decomposition: row `j` of an `nx × ny` image has a valid (y, x). -/
theorem image_row_split (j nx ny : Nat) (hnx : 0 < nx) (h : j < nx * ny) :
    j / nx < ny ∧ j % nx < nx := by
  refine ⟨?_, Nat.mod_lt _ hnx⟩
  rw [Nat.div_lt_iff_lt_mul hnx]
  rw [Nat.mul_comm] at h
  exact h

/-- Image positions stay below the position where the following text resumes. -/
theorem image_positions_below_resume (p y x nx ny : Nat) (hy : y < ny) (hx : x < nx) :
    p + y < p + max nx ny ∧ p + x < p + max nx ny := by omega

/-- Text cells after an image resume strictly after every image position. -/
theorem text_resume_after_image (p nx ny : Nat) (hx : 0 < nx) (hy : 0 < ny) :
    p + (max nx ny - 1) < p + max nx ny := by omega

/-- The position table has three int32 per cell; the last component of cell `c` is inside it. -/
theorem mrope_table_index (c cells : Nat) (h : c < cells) : c * 3 + 2 < cells * 3 := by omega

/-- A request's cells (prompt + generation + margin) fit inside `max_context + 64` table cells. -/
theorem mrope_cells_cover (n maxNew ctx : Nat) (h : n + maxNew + 8 ≤ ctx) : n + maxNew < ctx + 64 := by
  omega

end Strata.Program.Generate
