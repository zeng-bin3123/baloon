/-!
# MODULE: generate_driver
Arithmetic contracts of `src/program/generate.cpp`: the decode loop's position accounting (teacher forcing then
sampling), logits-dump row count, expert-cache auto sizing, cache-slot borrowing for the prompt path, the PCIe
share clamp and the speculative draft cell bounds.

## Source semantics
* The first generated token comes from the last prompt position `n_prompt-1`; the loop stops once `max_new`
  tokens are produced, i.e. at position `n_prompt - 2 + max_new`; rows written = `n_prompt - 1 + max_new`.
* `n_prompt + max_new <= max_context` is checked up front, so the loop never runs out of context.
* `slots = (free - reserve)/blob` (negative clamps to 0); borrowing takes `k = ceil(need/blob)` slots from the end.
* `pcie_num = clamp(round(frac*256), 0, 256)`.
* Speculative round: window `T <= S`, accepted `a <= T-1`, emitted `a+1 <= S`, draft cells `p + a + j`.

## Form template identifiers
* `producedAt` <- `produced.size()` after processing `pos`
* `slotsAuto` <- `o.expert_cache` under `--expert-cache auto`
* `ceilMul` <- `(need + blob - 1) / blob * blob`
* `clampPcie` <- `pcie_num` clamp
-/

namespace Strata.GenerateDriver

-- § decode loop positions

/-- Tokens produced after processing position `pos` (none before the last prompt position). -/
def producedAt (np pos : Nat) : Nat := if np ≤ pos + 1 then pos + 2 - np else 0

/-- The last prompt position yields exactly the first generated token. -/
theorem first_token_at_last_prompt (np : Nat) (h : 1 ≤ np) : producedAt np (np - 1) = 1 := by
  unfold producedAt; split <;> omega

/-- Prompt-only positions produce nothing. -/
theorem no_token_before_last_prompt (np pos : Nat) (h : pos + 1 < np) : producedAt np pos = 0 := by
  unfold producedAt; split <;> omega

/-- The loop stops at `pos` iff `pos + 1 = (np - 1) + max_new`: rows written equal `dump_positions`. -/
theorem stop_position_is_row_count (np mn pos : Nat) (h1 : 1 ≤ np) (h2 : 1 ≤ mn)
    (h : producedAt np pos = mn) : pos + 1 = (np - 1) + mn := by
  unfold producedAt at h; split at h <;> omega

/-- The checked context budget means the stopping position is inside the context. -/
theorem stop_position_in_context (np mn ctx pos : Nat) (h1 : 1 ≤ np) (h2 : 1 ≤ mn)
    (hctx : np + mn ≤ ctx) (h : producedAt np pos = mn) : pos < ctx := by
  have := stop_position_is_row_count np mn pos h1 h2 h
  omega

/-- The up-front check also bounds the prompt itself. -/
theorem prompt_fits (np mn ctx : Nat) (h : np ≤ ctx - mn) (hm : mn ≤ ctx) : np + mn ≤ ctx := by
  omega

-- § expert cache sizing

/-- Slots granted by `--expert-cache auto` (saturating subtraction = clamp at 0). -/
def slotsAuto (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots always fit in the free memory. -/
theorem slots_fit (free reserve blob : Nat) : slotsAuto free reserve blob * blob ≤ free := by
  unfold slotsAuto
  have h := Nat.div_mul_le_self (free - reserve) blob
  omega

/-- Smallest multiple of `blob` covering `need`. -/
def ceilMul (need blob : Nat) : Nat := (need + blob - 1) / blob * blob

/-- Borrowed slots cover the prompt path's buffers. -/
theorem ceil_covers (need blob : Nat) (hb : 0 < blob) : need ≤ ceilMul need blob := by
  unfold ceilMul
  have h1 := Nat.div_add_mod (need + blob - 1) blob
  have h2 := Nat.mod_lt (need + blob - 1) hb
  rw [Nat.mul_comm ((need + blob - 1) / blob) blob]
  omega

/-- Borrowing leaves at least 128 slots for decode when `k + 128 <= slots`. -/
theorem borrow_leaves_slots (slots k : Nat) (h : k + 128 ≤ slots) : 128 ≤ slots - k := by
  omega

/-- The first lent slot index is inside the cache. -/
theorem lend_first_lt (slots k : Nat) (hk : 0 < k) (h : k + 128 ≤ slots) : slots - k < slots := by
  omega

-- § PCIe share

/-- `pcie_num` after the clamp to `[0,256]`. -/
def clampPcie (x : Nat) : Nat := min 256 x

/-- The share numerator is at most 256. -/
theorem clampPcie_le (x : Nat) : clampPcie x ≤ 256 := by
  unfold clampPcie; omega

-- § speculative rounds

/-- Emitted tokens per round (`a + 1`) never exceed the window `T`, hence never exceed `S`. -/
theorem emitted_le_window (T S a : Nat) (hT : T ≤ S) (ha : a + 1 ≤ T) : a + 1 ≤ S := by
  omega

/-- The next window starts at `p + a + 1`, inside the cells just verified. -/
theorem next_start_in_window (p T a : Nat) (ha : a + 1 ≤ T) : p + a + 1 ≤ p + T := by
  omega

/-- FINDING: draft-chain cells `p + a + j` can exceed `max_context` by up to `S - 3` after the window check
`p + T <= ctx`, so the draft layer's K/V and RoPE need `S` cells of slack beyond the context. -/
theorem draft_cell_overshoot (p T a j S ctx : Nat) (hw : p + T ≤ ctx) (ha : a + 1 ≤ T)
    (hj : j + 2 ≤ S) : p + a + j + 3 ≤ ctx + S := by
  omega

/-- `--serve` admits a request only when `n + max_new + 8 <= max_context`, hence the prompt fits. -/
theorem serve_admission (n mn ctx : Nat) (h : n + mn + 8 ≤ ctx) : n + mn ≤ ctx := by
  omega

end Strata.GenerateDriver
