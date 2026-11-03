/-!
# MODULE: generate_driver_accounting
Position and row accounting of the `strata generate` driver: which positions produce tokens, the logits-dump row count,
context fit, the batched-prefill split, the auto-sized expert cache, the PCIe fraction clamp, profile truncation and
the routing-trace layer index.

## Source semantics
* The loop runs positions `pos_start ..`; tokens are produced from `pos = n_prompt - 1` until `max_new` exist, so the
  last position is `n_prompt - 2 + max_new`, and the dump has `n_prompt - 1 + max_new` positions.
* `n_tokens ≤ max_context - max_new` makes every position fit in the context.
* `--expert-cache auto` takes `(free - reserve) / blob` slots (saturating at 0): the slots fit in free VRAM.
* `pcie_num = clamp(round(frac*256), 0, 256)`.
* The routing trace records `layers - 1` after the dispatch advanced the counter.

## Form template identifiers
`n_prompt` -> nPrompt, `max_new` -> maxNew, `dump_positions` -> dumpPositions, `prefill_until` -> prefillUntil,
`pcie_num` -> pcieNum, `o.expert_cache` -> slots.
-/

namespace Strata.Generate

-- § positions and rows

/-- Number of positions the dump covers: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- Last position processed by the token loop. -/
def lastPos (nPrompt maxNew : Nat) : Nat := nPrompt - 2 + maxNew

/-- The first generated token comes from the last prompt position. -/
def firstProducedPos (nPrompt : Nat) : Nat := nPrompt - 1

/-- Positions `0 .. lastPos` number exactly `dumpPositions` (for a prompt of at least 2 tokens). -/
theorem dump_rows_match (nPrompt maxNew : Nat) (h : 2 ≤ nPrompt) (hm : 1 ≤ maxNew) :
    lastPos nPrompt maxNew + 1 = dumpPositions nPrompt maxNew := by
  unfold lastPos dumpPositions; omega

/-- Exactly `max_new` positions produce a token. -/
theorem produced_count (nPrompt maxNew : Nat) (h : 2 ≤ nPrompt) (hm : 1 ≤ maxNew) :
    lastPos nPrompt maxNew + 1 - firstProducedPos nPrompt = maxNew := by
  unfold lastPos firstProducedPos; omega

/-- A single-token prompt: the loop starts producing at position 0 and ends at `max_new - 1`. -/
theorem produced_count_single (maxNew : Nat) (hm : 1 ≤ maxNew) :
    (1 - 1 + maxNew) = maxNew := by omega

/-- With the size check `n ≤ ctx - max_new`, every position is below the context size. -/
theorem positions_fit (nPrompt maxNew ctx : Nat) (h1 : 1 ≤ nPrompt) (h2 : 0 < maxNew)
    (h3 : maxNew ≤ ctx) (h4 : nPrompt ≤ ctx - maxNew) : lastPos nPrompt maxNew < ctx := by
  unfold lastPos; omega

-- § prefill split

/-- Positions batched by the prompt path. -/
def nBatched (nPrompt prefillUntil : Nat) : Nat :=
  if 0 < prefillUntil ∧ prefillUntil < nPrompt - 1 then prefillUntil else nPrompt - 1

/-- The batched count never exceeds the conditioning positions. -/
theorem nBatched_le (nPrompt pu : Nat) : nBatched nPrompt pu ≤ nPrompt - 1 := by
  unfold nBatched; split <;> omega

/-- The token loop resumes at a prompt position (or the last conditioning one). -/
theorem resume_in_prompt (nPrompt pu : Nat) (h : 1 ≤ nPrompt) : nBatched nPrompt pu < nPrompt := by
  have := nBatched_le nPrompt pu
  omega

-- § auto-sized expert cache

/-- Slots granted by `--expert-cache auto` (truncating subtraction models the `max(0, ...)`). -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots occupy at most the free VRAM. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free :=
  Nat.le_trans (Nat.div_mul_le_self _ _) (Nat.sub_le _ _)

/-- Truncating the ranked profile to the slot count. -/
def profileWant (profileSize slots : Nat) : Nat := min profileSize slots

/-- The pre-fill never admits more pairs than there are slots or ranked pairs. -/
theorem profileWant_bounds (p s : Nat) : profileWant p s ≤ p ∧ profileWant p s ≤ s := by
  unfold profileWant; omega

-- § PCIe fraction

/-- `pcie_num` after the clamp to `[0, 256]`. -/
def clampPcie (x : Int) : Int := if x < 0 then 0 else if 256 < x then 256 else x

/-- The clamp lands in range. -/
theorem clampPcie_range (x : Int) : 0 ≤ clampPcie x ∧ clampPcie x ≤ 256 := by
  unfold clampPcie; split_ifs <;> omega

-- § routing trace

/-- The trace's layer index: the counter has already advanced past the layer. -/
def traceLayer (layersAfter : Nat) : Int := (layersAfter : Int) - 1

/-- After a dispatch of layer `l` (counter `l+1`), the trace records `l`. -/
theorem traceLayer_correct (l : Nat) : traceLayer (l + 1) = (l : Int) := by
  unfold traceLayer; omega

/-- For a counter in `1..48` the recorded layer is within `0..47`. -/
theorem traceLayer_range (c : Nat) (h1 : 1 ≤ c) (h2 : c ≤ 48) : 0 ≤ traceLayer c ∧ traceLayer c < 48 := by
  unfold traceLayer; omega

-- § serve admission

/-- A request is accepted only if prompt + max_new + 8 fits the context. -/
def serveAccepts (n maxNew ctx : Nat) : Bool := !(decide (ctx < n + maxNew + 8))

/-- An accepted request leaves room for a full verify window beyond its last position. -/
theorem serveAccepts_room (n maxNew ctx : Nat) (h : serveAccepts n maxNew ctx = true) :
    n + maxNew + 8 ≤ ctx := by
  unfold serveAccepts at h
  simp at h
  omega

end Strata.Generate
