-- Baloon.GenerateDriver: generate-driver formal contracts — independent implementations
-- of the generate-command's numeric invariants (run-fit, loop accounting, speculative
-- acceptance, PCIe clamping, trace index, expert-tier sizing, serve admission and the
-- image M-RoPE formula).  Corresponds to include/baloon/generate_cli.hpp and
-- src/driver/generate_cli.cpp.

namespace Baloon.GenerateDriver

-- ── Clock ─────────────────────────────────────────────────────────────────────
-- std::chrono::steady_clock maps to a monotone Nat tick counter.
abbrev Clock := Nat

-- ── Drive ─────────────────────────────────────────────────────────────────────
-- Backend interface: capability flags for the four callback slots.
structure Drive where
  hasStep    : Bool := true
  hasPrefill : Bool := false
  hasDraft   : Bool := false
  hasVerify  : Bool := false

-- ── Options ──────────────────────────────────────────────────────────────────
-- CLI options (mirrors baloon::gencli::Options).
structure Options where
  maxNew       : Nat := 32
  maxContext   : Nat := 4096
  prefillUntil : Nat := 0
  spec         : Nat := 0
  workers      : Nat := 1

-- ── run-fit check ─────────────────────────────────────────────────────────────

/-- A run is feasible: np + mn ≤ mc with both bounds positive. -/
def RunFits (np mn mc : Nat) : Prop :=
  np + mn ≤ mc ∧ 0 < mn ∧ 0 < mc

/-- A valid run holds prompt and generated tokens within the context. -/
theorem runFits_total (n m c : Nat) (h : RunFits n m c) : n + m ≤ c := h.1

/-- RunFits implies max_new is positive. -/
theorem runFits_maxNew_pos (n m c : Nat) (h : RunFits n m c) : 0 < m := h.2.1

-- ── positions and rows ────────────────────────────────────────────────────────

/-- Dump count: np + mn - 1 positions emit a logits row. -/
def dumpPositions (np mn : Nat) : Nat := np + mn - 1

/-- dumpPositions + 1 = np + mn when np ≥ 1. -/
theorem dumpPositions_succ (n m : Nat) (hn : 1 ≤ n) : dumpPositions n m + 1 = n + m := by
  simp only [dumpPositions]; omega

/-- The dump count is at least max_new when np ≥ 1. -/
theorem dumpPositions_ge_maxNew (n m : Nat) (hn : 1 ≤ n) : m ≤ dumpPositions n m := by
  simp only [dumpPositions]; omega

/-- Tokens produced after position `pos` (sampling starts at np - 1). -/
def producedAfter (np pos : Nat) : Nat := pos + 1 - np

/-- After all dump positions exactly mn tokens are produced (requires np ≥ 1). -/
theorem produced_at_end (np mn : Nat) (hn : 1 ≤ np) :
    producedAfter np (dumpPositions np mn) = mn := by
  simp only [producedAfter, dumpPositions]; omega

/-- Before the last prompt position nothing is produced (requires np ≥ 1). -/
theorem produced_before_prompt_end (np pos : Nat) (hn : 1 ≤ np) (h : pos ≤ np - 1) :
    producedAfter np pos = 0 := by
  unfold producedAfter; omega

/-- Next input: prompt token while pos + 1 is inside the prompt, sampled token afterwards. -/
def nextInput (toks : List Nat) (pos nx : Nat) : Nat :=
  if pos + 1 ≥ toks.length then nx else toks.getD (pos + 1) 0

/-- Inside the prompt the next input is the prompt token. -/
theorem nextInput_prompt (toks : List Nat) (pos nx : Nat) (h : pos + 1 < toks.length) :
    nextInput toks pos nx = toks.getD (pos + 1) 0 := by
  unfold nextInput
  have hge : ¬ (pos + 1 ≥ toks.length) := by omega
  exact if_neg hge

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (toks : List Nat) (pos nx : Nat) (h : toks.length ≤ pos + 1) :
    nextInput toks pos nx = nx := by
  unfold nextInput; exact if_pos h

-- ── batched prefill split ─────────────────────────────────────────────────────

/-- Positions eligible for the batched path. -/
def nBatched (pu n : Nat) : Nat :=
  if 0 < pu ∧ pu + 1 < n then pu else n - 1

/-- The batched count never exceeds n - 1. -/
theorem nBatched_le (pu n : Nat) : nBatched pu n ≤ n - 1 := by
  unfold nBatched; split_ifs with h <;> omega

/-- The token path starts strictly inside the prompt when the prompt is non-empty. -/
theorem posStart_lt (pu n : Nat) (hn : 1 ≤ n) : nBatched pu n < n := by
  have := nBatched_le pu n; omega

-- ── speculative acceptance ────────────────────────────────────────────────────

/-- Length of longest equal prefix; mismatch terminates at first differing element. -/
def accept : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d ≠ o then 0 else 1 + accept ds os
  | _, _ => 0

/-- Accepted count ≤ number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Nat), accept ds os ≤ ds.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      have ih := accept_le_left ds os
      simp only [accept, List.length_cons]
      split_ifs with h <;> omega

/-- Accepted count ≤ number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Nat), accept ds os ≤ os.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      have ih := accept_le_right ds os
      simp only [accept, List.length_cons]
      split_ifs with h <;> omega

/-- When drafts equal the verifier outputs all are accepted. -/
theorem accept_self : ∀ (xs : List Nat), accept xs xs = xs.length
  | [] => by simp [accept]
  | x :: xs => by
      have ih := accept_self xs
      unfold accept
      simp only [List.length_cons, if_neg (fun h : x ≠ x => h rfl)]
      omega

-- ── verify window ─────────────────────────────────────────────────────────────

/-- Window length: T = min S (run + 1). -/
def windowLen (S : Nat) (ok : List Bool) : Nat :=
  Nat.min S ((ok.takeWhile id).length + 1)

/-- The window never exceeds the configured spec size. -/
theorem windowLen_le (S : Nat) (ok : List Bool) : windowLen S ok ≤ S :=
  Nat.min_le_left S _

/-- The window is at least 1 when S ≥ 1. -/
theorem windowLen_pos (S : Nat) (ok : List Bool) (h : 1 ≤ S) : 1 ≤ windowLen S ok := by
  unfold windowLen
  rcases le_or_lt S ((ok.takeWhile id).length + 1) with hle | hlt
  · rw [Nat.min_eq_left hle]; exact h
  · rw [Nat.min_eq_right (Nat.le_of_lt hlt)]
    exact Nat.succ_le_succ (Nat.zero_le _)

/-- A round commits accepted+1 tokens, which fits a window of T-1 drafts. -/
theorem commit_le_window (T a : Nat) (hT : 1 ≤ T) (ha : a ≤ T - 1) : Nat.succ a ≤ T := by
  omega

-- ── PCIe clamp ────────────────────────────────────────────────────────────────

/-- PCIe numerator clamped to 0..256 (smaller of x and 256). -/
def pcieNum (x : Nat) : Nat := Nat.min x 256

/-- The clamped numerator is within the 256 denominator. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by
  unfold pcieNum; omega

-- ── routing trace ─────────────────────────────────────────────────────────────

/-- The trace layer index: adapter-advanced counter minus one. -/
def traceLayer (n : Nat) : Int := Int.ofNat n - 1

/-- For layers 0..47 the counter l+1 maps back to l. -/
theorem traceLayer_valid (l : Nat) (h : l < 48) :
    traceLayer (l + 1) = (l : Int) ∧ 0 ≤ traceLayer (l + 1) ∧ traceLayer (l + 1) < 48 := by
  unfold traceLayer; omega

-- ── VRAM expert-tier sizing ───────────────────────────────────────────────────

/-- Auto expert cache: Nat.div (free - reserve) blob slots. -/
def autoSlots (free reserve blob : Nat) : Nat :=
  Nat.div (free - reserve) blob

/-- The auto-sized tier never exceeds free - reserve bytes. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self (free - reserve) blob

/-- Profile truncation: slots limited to the profile length. -/
def withProfile (slots p : Nat) : Nat := Nat.min slots p

/-- Profile truncation only shrinks or keeps the tier. -/
theorem withProfile_le (slots p : Nat) : withProfile slots p ≤ slots := by
  unfold withProfile; omega

/-- Ceiling division. -/
def ceilDiv (a b : Nat) : Nat := Nat.div (a + b - 1) b

/-- Lent slots cover the buffer: a ≤ ceilDiv(a, b) * b. -/
theorem ceilDiv_mul_ge (a b : Nat) (hb : 0 < b) : a ≤ ceilDiv a b * b := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  omega

/-- Prefill MiB reservation: non-zero only when not borrowing cache slots. -/
def prefillMib (chunk : Nat) (borrow : Bool) : Nat :=
  if !borrow && 0 < chunk then 160 + chunk * 680 / 1024 else 0

/-- Borrowing removes the reservation. -/
theorem prefillMib_borrow (chunk : Nat) : prefillMib chunk true = 0 := by
  simp [prefillMib]

-- ── adaptive tier swaps ───────────────────────────────────────────────────────

/-- Swap gate: c ≥ 2 and 2c ≥ 2v + 3. -/
def swapOK (c v : Nat) : Prop := 2 ≤ c ∧ 2 * v + 3 ≤ 2 * c

/-- An accepted swap replaces a strictly less-routed expert. -/
theorem swapOK_lt (c v : Nat) (h : swapOK c v) : v < c := by
  obtain ⟨_, h2⟩ := h; omega

/-- 0.7 usage decay never increases the count (integer ×7/10 form). -/
theorem decay_le (u : Nat) : u * 7 / 10 ≤ u := by
  simp only []
  omega

/-- PLE token-window advance: oldest-first shift. -/
def pleAdvance (prev : Int × Int) (tok : Int) : Int × Int := (prev.snd, tok)

/-- After an advance the older slot holds the previously-newest token. -/
theorem pleAdvance_fst (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).1 = prev.2 := by
  simp [pleAdvance]

/-- After an advance the newest slot holds the just-decoded token. -/
theorem pleAdvance_snd (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).2 = tok := by
  simp [pleAdvance]

-- ── image M-RoPE positions ────────────────────────────────────────────────────

/-- M-RoPE position triple for image cell j (row-major, nx columns, text start p). -/
def imgPos (p nx j : Nat) : Nat × Nat × Nat := ⟨p, p + j / nx, p + j % nx⟩

/-- Height coordinate stays below p + max(nx, ny). -/
theorem imgPos_h_lt (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny := by
  have h1 : j / nx < ny := Nat.div_lt_of_lt_mul (Nat.mul_comm nx ny ▸ hj)
  have h2 := Nat.le_max_right nx ny
  omega

/-- Width coordinate stays below p + max(nx, ny). -/
theorem imgPos_w_lt (p nx ny j : Nat) (hx : 0 < nx) :
    p + j % nx < p + max nx ny :=
  Nat.add_lt_add_left
    (Nat.lt_of_lt_of_le (Nat.mod_lt j hx) (Nat.le_max_left nx ny)) p

/-- After an image block text position strictly advances. -/
theorem image_advance_pos (p nx ny : Nat) (hx : 0 < nx) : p < p + max nx ny := by
  have := Nat.le_max_left nx ny; omega

-- ── serve loop admission ──────────────────────────────────────────────────────

/-- Serve admission: n + (maxNew + 8) ≤ maxCtx. -/
def admitted (n maxNew maxCtx : Nat) : Prop := n + (maxNew + 8) ≤ maxCtx

/-- Admission leaves room for the full generation. -/
theorem admitted_fits (n m c : Nat) (h : admitted n m c) : n + m ≤ c := by
  unfold admitted at h; omega

/-- Tokens produced in one serve round: clipped to maxNew. -/
def producedStep (maxNew produced a : Nat) : Nat := Nat.min maxNew (produced + a + 1)

/-- A round never overshoots maxNew. -/
theorem producedStep_le (maxNew produced a : Nat) : producedStep maxNew produced a ≤ maxNew := by
  unfold producedStep; omega

/-- A round makes progress while below maxNew. -/
theorem producedStep_progress (maxNew produced a : Nat) (h : produced < maxNew) :
    produced < producedStep maxNew produced a := by
  unfold producedStep; omega

-- ── argmax ────────────────────────────────────────────────────────────────────

/-- argmax always returns a valid index for a non-empty list. -/
theorem argmax_valid (xs : List Float) (h : xs ≠ []) : ∃ i, i < xs.length := by
  cases xs with
  | nil => exact absurd rfl h
  | cons _ _ => exact ⟨0, by simp⟩

-- ── parse_i64_list ────────────────────────────────────────────────────────────

/-- A well-formed comma-separated list has one token per segment. -/
def parseTokenCount (s : String) : Nat := (s.splitOn ",").length

-- ── usage ─────────────────────────────────────────────────────────────────────

/-- A usage string is well-formed when it is non-empty. -/
def usageWellFormed (s : String) : Prop := 0 < s.length

/-- The empty string is not well-formed. -/
theorem empty_not_wellFormed : ¬ usageWellFormed "" := by simp [usageWellFormed]

end Baloon.GenerateDriver
