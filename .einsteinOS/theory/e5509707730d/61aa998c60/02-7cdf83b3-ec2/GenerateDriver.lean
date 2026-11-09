/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
The `strata generate` driver: option validation, the token loop (prompt teacher forcing, first generated token at
the last prompt position), logits-row selection, speculative-window acceptance, VRAM expert-tier sizing and
lending, and the image M-RoPE position assignment of `--serve`.

## Source semantics
* The prompt plus `max_new` must fit in `max_context`.
* The dump header counts `n_prompt - 1 + max_new` positions; with a stride, every N-th row plus the last is stored.
* While the prompt lasts the next input is the prompt's own next token, afterwards the sampled one.
* A verify window accepts the longest prefix of drafts equal to the model's own outputs; it emits accepted + 1 tokens.
* Slots lent to the prompt path are a suffix of the cache and are given back afterwards.

## Form template identifiers
Options.max_new -> maxNew; Options.max_context -> ctx; `produced` -> producedAt; `a` (accepted) -> acc;
`lend_first` -> lendFirst; `pcie_num` -> pcieNum; `winLoop` <- the `T` selection `while` loop.
-/

namespace Strata.GenerateDriver

-- § option validation

/-- The size check of `main`: positive budgets and prompt + generation within the context. -/
def OptsOk (maxNew ctx promptLen : Nat) : Prop :=
  0 < maxNew ∧ 0 < ctx ∧ maxNew ≤ ctx ∧ promptLen ≤ ctx - maxNew

/-- Accepted options guarantee the prompt and the new tokens fit together. -/
theorem opts_fit (maxNew ctx promptLen : Nat) (h : OptsOk maxNew ctx promptLen) : promptLen + maxNew ≤ ctx := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

/-- The `--serve` guard `n + max_new + 8 <= ctx` keeps the whole request inside the context. -/
theorem serve_fit (n maxNew ctx : Nat) (h : ¬ (ctx < n + maxNew + 8)) : n + maxNew ≤ ctx := by omega

-- § the token loop

/-- Number of logged dump positions: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Tokens recorded after processing position `pos` (recording starts at the last prompt position). -/
def producedAt (nPrompt pos : Nat) : Nat := if nPrompt ≤ pos + 1 then pos + 2 - nPrompt else 0

/-- Nothing is produced before the last prompt position. -/
theorem produced_before_prompt_end (nPrompt pos : Nat) (h : pos + 1 < nPrompt) : producedAt nPrompt pos = 0 := by
  unfold producedAt
  rw [if_neg (by omega)]

/-- The last prompt position yields exactly the first generated token. -/
theorem produced_first (nPrompt : Nat) (h : 1 ≤ nPrompt) : producedAt nPrompt (nPrompt - 1) = 1 := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- For a prompt of two or more tokens the loop reaches `max_new` tokens at position `n_prompt - 2 + max_new`. -/
theorem produced_reaches_max (nPrompt maxNew : Nat) (h1 : 2 ≤ nPrompt) (h2 : 1 ≤ maxNew) :
    producedAt nPrompt (nPrompt - 2 + maxNew) = maxNew := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- For a one-token prompt the loop reaches `max_new` tokens at position `max_new - 1`. -/
theorem produced_reaches_max_single (maxNew : Nat) (h2 : 1 ≤ maxNew) :
    producedAt 1 (maxNew - 1) = maxNew := by
  unfold producedAt
  rw [if_pos (by omega)]
  omega

/-- Teacher forcing: the next input is the prompt token while the prompt lasts, the sample afterwards. -/
def nextInput (prompt : Nat → Nat) (n pos next : Nat) : Nat := if pos + 1 < n then prompt (pos + 1) else next

/-- Inside the prompt the model's guess is ignored. -/
theorem nextInput_prompt (prompt : Nat → Nat) (n pos next : Nat) (h : pos + 1 < n) :
    nextInput prompt n pos next = prompt (pos + 1) := by
  unfold nextInput
  rw [if_pos h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (prompt : Nat → Nat) (n pos next : Nat) (h : ¬ pos + 1 < n) :
    nextInput prompt n pos next = next := by
  unfold nextInput
  rw [if_neg h]

-- § logits-row selection (`--logits-stride`)

/-- A position's row is stored when it is a multiple of the stride or the final position. -/
def selected (pos total stride : Nat) : Prop := pos % stride = 0 ∨ pos + 1 = total

/-- Stride 1 stores every row. -/
theorem selected_stride_one (pos total : Nat) : selected pos total 1 := by
  left
  exact Nat.mod_one pos

/-- The final position is always stored. -/
theorem selected_last (total stride : Nat) (h : 0 < total) : selected (total - 1) total stride := by
  right
  omega

-- § speculative verification

/-- Length of the longest prefix of the drafts that equals the model's outputs (`a` in the source). -/
def acc : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | d :: ds, o :: os => if d = o then acc ds os + 1 else 0

/-- Never more accepted than drafted. -/
theorem acc_le_left (ds os : List Nat) : acc ds os ≤ ds.length := by
  induction ds generalizing os with
  | nil => simp [acc]
  | cons d ds ih =>
    cases os with
    | nil => simp [acc]
    | cons o os =>
      have h := ih os
      simp only [acc, List.length_cons]
      split <;> omega

/-- Never more accepted than the window produced. -/
theorem acc_le_right (ds os : List Nat) : acc ds os ≤ os.length := by
  induction ds generalizing os with
  | nil => simp [acc]
  | cons d ds ih =>
    cases os with
    | nil => simp [acc]
    | cons o os =>
      have h := ih os
      simp only [acc, List.length_cons]
      split <;> omega

/-- Perfect drafts are fully accepted. -/
theorem acc_self (ds : List Nat) : acc ds ds = ds.length := by
  induction ds with
  | nil => simp [acc]
  | cons d ds ih => simp [acc, ih]

/-- A round emits `accepted + 1` tokens, never more than the window length `1 + drafts`. -/
theorem emitted_le_window (ds os : List Nat) : acc ds os + 1 ≤ 1 + ds.length := by
  have := acc_le_left ds os
  omega

/-- Window-size loop: grow `T` while it is below `S` and the next draft probability test passes. -/
def winLoop (ok : Nat → Bool) (S : Nat) : Nat → Nat → Nat
  | 0, T => T
  | f + 1, T => if T < S ∧ ok (T - 1) = true then winLoop ok S f (T + 1) else T

/-- The loop never shrinks the window. -/
theorem winLoop_ge (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ winLoop ok S f T := by
  induction f with
  | zero => intro T; exact Nat.le_refl T
  | succ f ih =>
    intro T
    simp only [winLoop]
    split
    · have := ih (T + 1)
      omega
    · exact Nat.le_refl T

/-- The loop never exceeds the speculation depth `S`. -/
theorem winLoop_le (ok : Nat → Bool) (S : Nat) (f : Nat) : ∀ T, T ≤ S → winLoop ok S f T ≤ S := by
  induction f with
  | zero => intro T h; exact h
  | succ f ih =>
    intro T h
    simp only [winLoop]
    split
    · rename_i hc
      obtain ⟨hlt, _⟩ := hc
      exact ih (T + 1) (by omega)
    · exact h

-- § VRAM expert tier

/-- Slots that fit after the reserve: `(free - reserve) / blob`. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The automatically sized tier never exceeds the free memory left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- First lent slot when `k` slots are borrowed (only if `k + 128 ≤ slots`). -/
def lendFirst (slots k : Nat) : Nat := slots - k

/-- Lent slots are exactly the suffix `[lendFirst, slots)` and at least 128 slots stay resident. -/
theorem lend_suffix (slots k : Nat) (h : k + 128 ≤ slots) : lendFirst slots k + k = slots ∧ 128 ≤ lendFirst slots k := by
  unfold lendFirst
  constructor <;> omega

/-- The PCIe share numerator is clamped to `0..256`. -/
def pcieNum (x : Nat) : Nat := min 256 x

/-- The clamp is within range. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by unfold pcieNum; omega

-- § image M-RoPE positions (`GENI`)

/-- Within an image of `nx * ny` rows at base position `p`, the (h, w) positions stay below `p + max nx ny`,
which is where the text after the image resumes. -/
theorem mrope_image_pos_lt (p j nx ny : Nat) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny ∧ p + j % nx < p + max nx ny := by
  have h1 : j / nx < ny := Nat.div_lt_of_lt_mul hj
  have hnx : 0 < nx := by
    apply Nat.pos_of_ne_zero
    intro h
    subst h
    simp at hj
  have h2 : j % nx < nx := Nat.mod_lt j hnx
  constructor <;> omega

end Strata.GenerateDriver
