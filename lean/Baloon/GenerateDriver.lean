-- Baloon.GenerateDriver: arithmetic and control contracts of the `ref generate` driver.
-- PORT UNIT: ub25e8eba7f (split from u253a75befe). Sources: src/program/generate.cpp.
-- Proves the same mathematical facts as the Ref.Generate theory layer with Baloon-namespace
-- names and proof terms.  One module per unit scope; 48 items total.

namespace Baloon.GenerateDriver

-- § run-size validation

/-- RunFits: the driver accepts a run when every size constraint is satisfied. -/
def RunFits (nPrompt maxNew maxCtx : Nat) : Prop :=
  (0 < maxNew) ∧ (0 < maxCtx) ∧ (maxNew ≤ maxCtx) ∧ (nPrompt ≤ maxCtx - maxNew)

/-- A run that passes validation can hold the prompt and all generated tokens. -/
theorem runFits_total (n m c : Nat) (h : RunFits n m c) : n + m ≤ c := by {
  obtain ⟨_, _, h3, h4⟩ := h; omega }

-- § positions, rows and produced-token accounting

/-- Number of logits rows dumped: n_prompt - 1 + max_new. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := (nPrompt - 1) + maxNew

/-- The dump count is one less than n_prompt + max_new. -/
theorem dumpPositions_succ (n m : Nat) (h : 1 ≤ n) : dumpPositions n m + 1 = n + m := by {
  unfold dumpPositions; omega }

/-- Tokens produced after processing P positions: sampling starts at position n-1. -/
def producedAfter (n P : Nat) : Nat := (P - (n - 1))

/-- After all dumped positions exactly max_new tokens are produced. -/
theorem produced_at_end (n m : Nat) : producedAfter n (dumpPositions n m) = m := by {
  unfold producedAfter dumpPositions; omega }

/-- Before the last prompt position nothing is produced. -/
theorem produced_before_prompt_end (n P : Nat) (h : P ≤ n - 1) : producedAfter n P = 0 := by {
  unfold producedAfter; omega }

/-- The next input token: prompt token while the prompt lasts, else the sampled one. -/
def nextInput (tokens : List Nat) (pos next : Nat) : Nat :=
  if (pos + 1) < tokens.length then tokens.getD (pos + 1) 0 else next

/-- Teacher forcing: inside the prompt the next input is the prompt token. -/
theorem nextInput_prompt (tokens : List Nat) (pos next : Nat) (h : pos + 1 < tokens.length) :
    nextInput tokens pos next = tokens.getD (pos + 1) 0 := by {
  unfold nextInput; rw [if_pos h] }

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (tokens : List Nat) (pos next : Nat) (h : tokens.length ≤ pos + 1) :
    nextInput tokens pos next = next := by {
  unfold nextInput; rw [if_neg (by omega)] }

-- § batched prefill split

/-- How many prompt positions go through the batched path. -/
def nBatched (pu n : Nat) : Nat := if (0 < pu) ∧ (pu < n - 1) then pu else n - 1

/-- The batched count never exceeds n - 1. -/
theorem nBatched_le (pu n : Nat) : nBatched pu n ≤ n - 1 := by {
  unfold nBatched; split_ifs with h <;> omega }

/-- The token path starts strictly inside the prompt whenever the prompt is non-empty. -/
theorem posStart_lt (pu n : Nat) (hn : 1 ≤ n) : nBatched pu n < n := by {
  have := nBatched_le pu n; omega }

-- § speculative acceptance

/-- Length of the longest prefix of drafts equal to the verifier outputs outs. -/
def accept : List Nat → List Nat → Nat
  | d :: ds, o :: os => if (d = o) then accept ds os + 1 else 0
  | _, _ => 0

/-- Accepted drafts never exceed the number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Nat), accept ds os ≤ ds.length
  | [], _ => by { simp [accept] }
  | _ :: _, [] => by { simp [accept] }
  | d :: ds, o :: os => by {
    have ih := accept_le_left ds os
    by_cases h : d = o
    · simp [accept, h]; omega
    · simp [accept, h] }

/-- Accepted drafts never exceed the number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Nat), accept ds os ≤ os.length
  | [], _ => by { simp [accept] }
  | _ :: _, [] => by { simp [accept] }
  | d :: ds, o :: os => by {
    have ih := accept_le_right ds os
    by_cases h : d = o
    · simp [accept, h]; omega
    · simp [accept, h] }

/-- If the drafts equal the outputs, every draft is accepted. -/
theorem accept_self : ∀ (xs : List Nat), accept xs xs = xs.length
  | [] => by { simp [accept] }
  | x :: xs => by { have ih := accept_self xs; simp [accept, ih] }

-- § verify window size and commit bound

/-- Window length with the minimum-probability rule: T = min S (1 + run of passing drafts). -/
def windowLen (S : Nat) (ok : List Bool) : Nat := (min S (1 + (ok.takeWhile id).length))

/-- The verify window never exceeds the configured spec size. -/
theorem windowLen_le (S : Nat) (ok : List Bool) : windowLen S ok ≤ S := by {
  unfold windowLen; omega }

/-- The verify window always holds at least the last emitted token. -/
theorem windowLen_pos (S : Nat) (ok : List Bool) (h : 1 ≤ S) : 1 ≤ windowLen S ok := by {
  unfold windowLen; omega }

/-- A round commits accepted + 1 tokens, which fits a window of T-1 drafts. -/
theorem commit_le_window (T a : Nat) (hT : 1 ≤ T) (ha : a ≤ T - 1) : a + 1 ≤ T := by {
  omega }

-- § PCIe share and routing trace

/-- PCIe numerator clamped to [0, 256]. -/
def pcieNum (x : Nat) : Nat := min (256) x

/-- The clamped PCIe numerator is within the 256 denominator. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by {
  unfold pcieNum; omega }

/-- The traced layer index is afterDispatch - 1 (the adapter has already advanced the counter). -/
def traceLayer (afterDispatch : Nat) : Int := ((afterDispatch : Int) - 1)

-- § trace layer range

/-- For layers 0..47 the advanced counter l+1 maps back to l, in range 0..47. -/
theorem traceLayer_valid (l : Nat) (h : l < 48) :
    traceLayer (l + 1) = (l : Int) ∧ 0 ≤ traceLayer (l + 1) ∧ traceLayer (l + 1) < 48 := by {
  unfold traceLayer; omega }

-- § VRAM expert tier sizing

/-- Auto-sized tier: (free - reserve) / blob slots. -/
def autoSlots (free reserve blob : Nat) : Nat := ((free - reserve) / blob)

/-- The auto-sized tier never occupies more than the free VRAM left after the reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve := by {
  exact Nat.div_mul_le_self _ _ }

/-- With a profile the slot count is truncated to the profile length. -/
def withProfile (slots profLen : Nat) : Nat := min (slots) profLen

/-- Profile truncation only shrinks the tier. -/
theorem withProfile_le (slots profLen : Nat) : withProfile slots profLen ≤ slots := by {
  unfold withProfile; omega }

-- § ceiling division and prefill MiB budget

/-- Ceiling division, as used for the number of lent cache slots. -/
def ceilDiv (a b : Nat) : Nat := ((a + b - 1) / b)

/-- Lent slots are enough to hold the prompt path buffers: need <= k * blob. -/
theorem ceilDiv_mul_ge (a b : Nat) (hb : 0 < b) : a ≤ ceilDiv a b * b := by {
  unfold ceilDiv
  have h := (Nat.div_lt_iff_lt_mul hb).mp (Nat.lt_succ_self ((a + b - 1) / b))
  rw [Nat.succ_mul] at h; omega }

/-- When the prompt path borrows cache slots the separate buffer reservation is zero. -/
def prefillMib (chunk : Nat) (borrow : Bool) : Nat :=
  if (0 < chunk) ∧ (borrow = false) then 160 + chunk * 680 / 1024 else 0

/-- Borrowing removes the reserved prompt buffers from the VRAM budget. -/
theorem prefillMib_borrow (chunk : Nat) : prefillMib chunk true = 0 := by {
  unfold prefillMib; simp }

-- § adaptive tier swaps + PLE token window

/-- A swap needs the candidate used >= 2 and >= victim + 1.5 (half-unit form). -/
def swapOK (c v : Nat) : Prop := (4 ≤ 2 * c) ∧ (2 * v + 3 ≤ 2 * c)

/-- An accepted swap always replaces a strictly less-routed resident expert. -/
theorem swapOK_lt (c v : Nat) (h : swapOK c v) : v < c := by {
  obtain ⟨_, h2⟩ := h; omega }

/-- The 0.7 decay never increases a usage count (scaled integer form). -/
theorem decay_le (u : Nat) : u * 7 / 10 ≤ u := by { omega }

/-- PLE advance: newest predecessor becomes the older one, new token is newest. -/
def pleAdvance (prev : Int × Int) (tok : Int) : Int × Int := ((prev.2), tok)

/-- After an advance the older slot holds the previous newest token. -/
theorem pleAdvance_fst (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).1 = prev.2 := by {
  rfl }

/-- After an advance the newest slot holds the token just decoded. -/
theorem pleAdvance_snd (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).2 = tok := by {
  rfl }

-- § image M-RoPE positions + serve-loop admission

/-- Position triple of image row j with nx columns starting at text position p. -/
def imgPos (p nx j : Nat) : Nat × Nat × Nat := (p, (p + j / nx), (p + j % nx))

/-- The height coordinate stays below the advance p + max nx ny. -/
theorem imgPos_h_lt (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny := by {
  have h1 : j / nx < ny := (Nat.div_lt_iff_lt_mul hx).mpr (by rw [Nat.mul_comm]; exact hj)
  have h2 := Nat.le_max_right nx ny; omega }

/-- The width coordinate stays below the advance p + max nx ny. -/
theorem imgPos_w_lt (p nx ny j : Nat) (hx : 0 < nx) :
    p + j % nx < p + max nx ny := by {
  have h1 : j % nx < nx := Nat.mod_lt j hx
  have h2 := Nat.le_max_left nx ny; omega }

/-- Text after an image strictly advances past the image start. -/
theorem image_advance_pos (p nx ny : Nat) (hx : 0 < nx) : p < p + max nx ny := by {
  have _ := Nat.le_max_left nx ny; omega }

/-- A GEN request is admitted iff n + max_new + 8 <= max_context. -/
def admitted (n maxNew maxCtx : Nat) : Prop := (n + maxNew + 8) ≤ maxCtx

/-- Admission leaves room for a full verify window of lookahead. -/
theorem admitted_fits (n m c : Nat) (h : admitted n m c) : n + m ≤ c := by {
  unfold admitted at h; omega }

/-- Tokens produced in a serve round are clipped to max_new. -/
def producedStep (maxNew produced a : Nat) : Nat := min (maxNew) (produced + a + 1)

/-- A round never overshoots max_new. -/
theorem producedStep_le (maxNew produced a : Nat) : producedStep maxNew produced a ≤ maxNew := by {
  unfold producedStep; omega }

/-- A round makes progress while below max_new, so the serve loop terminates. -/
theorem producedStep_progress (maxNew produced a : Nat) (h : produced < maxNew) :
    produced < producedStep maxNew produced a := by {
  unfold producedStep; omega }

end Baloon.GenerateDriver
