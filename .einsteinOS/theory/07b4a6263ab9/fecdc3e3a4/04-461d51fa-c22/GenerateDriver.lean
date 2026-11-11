/-!
# MODULE: generate_driver
SOURCES: src/program/generate.cpp
`strata generate`, the driver: argument validation, VRAM expert-tier sizing (`--expert-cache auto`,
per-pair sized slots), the teacher-forced token loop (`feed → 48 layers → head → sample → feed`), the
speculative loop's window sizing and accounting, the PCIe share clamp and the adaptive tier's swap rule.
Covers the core decision logic of `main`; the I/O plumbing is out of scope.

## Source semantics
* A run is accepted only if `0 < max_new ≤ max_context` and `n_prompt ≤ max_context - max_new`
  (so prompt + generation fit), and `max_context < 2^31 - 8`.
* The logits dump has `n_prompt - 1 + max_new` positions: sampling starts at the last prompt position.
* Teacher forcing: the input after position `p` is the prompt's token `p + 1` while it lasts, the sampled
  token afterwards; a token is recorded only from position `n_prompt - 1`.
* Auto cache slots: `(free - reserve) / max_blob`, never negative, never more VRAM than granted.
* Sized slots are taken greedily from the profile while the 256-rounded running sum fits the cap.
* `pcie_num = clamp(round(frac * 256), 0, 256)`.
* With `spec_min_p > 0` a window has `T = 1 + (leading drafts above min_p)` tokens, capped at `spec`;
  the first window holds one token.
* The adaptive tier swaps a missing expert in only if routed at least 1.5 more (decayed) times than the
  resident victim, and keeps at most `adapt_swaps` swaps.

## Form template identifiers
runAccepted ↔ the max_new/max_context guard · dumpPositions ↔ dump_positions · nextInput ↔ teacher forcing ·
autoSlots ↔ `(free - reserve) / max_blob` · fillUsed ↔ sized_slots loop · pcieClamp ↔ pcie_num ·
windowT ↔ `while (T < spec && dprob[T-1] >= min_p)` · swapWorth ↔ `cand >= vict + 1.5`
-/

namespace GenerateDriver

-- § argument validation

/-- The run-shape guard in `main`. -/
def runAccepted (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew ∧ maxCtx < 2147483640

/-- An accepted run fits prompt and generation in the context. -/
theorem accepted_fits (p n c : Nat) (h : runAccepted p n c) : p + n ≤ c := by
  unfold runAccepted at h; omega

/-- Every position of an accepted run is a valid int32 cell. -/
theorem accepted_int32 (p n c pos : Nat) (h : runAccepted p n c) (hp : pos < p + n) :
    pos < 2147483647 := by
  unfold runAccepted at h; omega

-- § positions and the logits dump

/-- Positions that produce a logits row: every prompt position but the last only conditions. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- With a non-empty prompt, prompt positions plus generated tokens minus one is the row count. -/
theorem dump_rows (p n : Nat) (hp : 1 ≤ p) : dumpPositions p n + 1 = p + n := by
  unfold dumpPositions; omega

/-- The input fed after position `pos`: teacher forcing while the prompt lasts. -/
def nextInput (prompt : List Nat) (pos sampled : Nat) : Nat :=
  if pos + 1 < prompt.length then prompt.getD (pos + 1) 0 else sampled

/-- Inside the prompt the model's own guess is ignored. -/
theorem forced_in_prompt (prompt : List Nat) (pos s s' : Nat) (h : pos + 1 < prompt.length) :
    nextInput prompt pos s = nextInput prompt pos s' := by
  simp [nextInput, h]

/-- From the last prompt position on the sampled token is fed back. -/
theorem free_after_prompt (prompt : List Nat) (pos s : Nat) (h : prompt.length ≤ pos + 1) :
    nextInput prompt pos s = s := by
  have : ¬ pos + 1 < prompt.length := by omega
  simp [nextInput, this]

/-- A position records a generated token iff it is at or past the last prompt position. -/
def records (nPrompt pos : Nat) : Bool := decide (nPrompt - 1 ≤ pos)

/-- Positions `0 .. n_prompt + max_new - 2` record exactly `max_new` tokens. -/
theorem recorded_count (p n : Nat) (hp : 1 ≤ p) :
    ((List.range (p - 1 + n)).filter (records p)).length = n := by
  have key : ∀ m, ((List.range (p - 1 + m)).filter (records p)).length = m := by
    intro m
    induction m with
    | zero =>
      simp only [Nat.add_zero]
      rw [List.filter_eq_nil_iff.mpr]
      · rfl
      · intro x hx
        simp only [List.mem_range] at hx
        simp [records]
        omega
    | succ k ih =>
      rw [show p - 1 + (k + 1) = (p - 1 + k) + 1 by omega, List.range_succ, List.filter_append,
        List.length_append, ih]
      simp [records]
  exact key n

-- § VRAM tier sizing

/-- `--expert-cache auto`: slots from the VRAM left above the reserve. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto slots never use more than the free VRAM above the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) : autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- A slot size rounded to 256 bytes. -/
def round256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Final used bytes of the greedy sized-slot fill. -/
def fillUsed : List Nat → Nat → Nat → Nat
  | [], _, used => used
  | b :: bs, cap, used =>
    if used + round256 b > cap then used else fillUsed bs cap (used + round256 b)

/-- The greedy fill never exceeds the cap it started under. -/
theorem fillUsed_le : ∀ (bs : List Nat) (cap used : Nat), used ≤ cap → fillUsed bs cap used ≤ cap
  | [], _, _, h => h
  | b :: bs, cap, used, h => by
    simp only [fillUsed]
    split
    · exact h
    · exact fillUsed_le bs cap _ (by omega)

/-- The fill only grows the used total. -/
theorem fillUsed_ge : ∀ (bs : List Nat) (cap used : Nat), used ≤ fillUsed bs cap used
  | [], _, _ => Nat.le_refl _
  | b :: bs, cap, used => by
    simp only [fillUsed]
    split
    · exact Nat.le_refl _
    · have := fillUsed_ge bs cap (used + round256 b)
      omega

-- § the PCIe share

/-- `pcie_num`: the share of misses read over PCIe, out of 256. -/
def pcieClamp (x : Int) : Int := max 0 (min 256 x)

/-- The share is a valid fraction of 256. -/
theorem pcie_bounds (x : Int) : 0 ≤ pcieClamp x ∧ pcieClamp x ≤ 256 := by
  unfold pcieClamp; omega

/-- An in-range request is kept unchanged. -/
theorem pcie_identity (x : Int) (h0 : 0 ≤ x) (h1 : x ≤ 256) : pcieClamp x = x := by
  unfold pcieClamp; omega

-- § speculative window sizing

/-- Leading drafts whose probability passes `min_p`. -/
def leading : List Bool → Nat
  | [] => 0
  | b :: bs => if b then 1 + leading bs else 0

/-- Window size: one plus the leading confident drafts, capped at `spec`; one token for the first window. -/
def windowT (first : Bool) (spec : Nat) (gates : List Bool) : Nat :=
  if first then 1 else min spec (1 + leading gates)

/-- Leading count is bounded by the number of gates. -/
theorem leading_le : ∀ bs : List Bool, leading bs ≤ bs.length
  | [] => Nat.le_refl 0
  | b :: bs => by
    simp only [leading, List.length_cons]
    have := leading_le bs
    split <;> omega

/-- A window is never empty and never larger than the verifier was built for. -/
theorem windowT_bounds (first : Bool) (spec : Nat) (gates : List Bool) (hs : 1 ≤ spec) :
    1 ≤ windowT first spec gates ∧ windowT first spec gates ≤ spec := by
  unfold windowT
  split <;> omega

/-- Accepted drafts never exceed proposed ones, so tokens per round is at most the window. -/
theorem tokens_per_round (rounds ok total spec : Nat) (hs : 1 ≤ spec) (h1 : ok ≤ total)
    (h2 : total ≤ rounds * (spec - 1)) : ok + rounds ≤ rounds * spec := by
  have hsucc : spec - 1 + 1 = spec := by omega
  have h3 : rounds * (spec - 1) + rounds = rounds * spec := by
    calc rounds * (spec - 1) + rounds = rounds * (spec - 1 + 1) := by rw [Nat.mul_add, Nat.mul_one]
      _ = rounds * spec := by rw [hsucc]
  omega

-- § the adaptive tier

/-- Swap rule with usage scaled by 2 (so 1.5 becomes 3): the candidate must clearly out-route the victim. -/
def swapWorth (cand2 vict2 : Nat) : Bool := decide (vict2 + 3 ≤ cand2)

/-- A worthwhile swap has a positive gain. -/
theorem swap_gain_pos (c v : Nat) (h : swapWorth c v = true) : v < c := by
  unfold swapWorth at h
  have := of_decide_eq_true h
  omega

/-- Equally routed experts are never swapped: no churn at a tie. -/
theorem no_swap_tie (v : Nat) : swapWorth v v = false := by
  unfold swapWorth
  exact decide_eq_false (by omega)

/-- Decay by 0.7 (scaled as 7/10) never increases a usage count. -/
theorem decay_shrinks (u : Nat) : u * 7 / 10 ≤ u := by
  omega

end GenerateDriver
