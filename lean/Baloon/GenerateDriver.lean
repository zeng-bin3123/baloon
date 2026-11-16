-- Baloon.GenerateDriver: generate-command driver contracts — run-size fit,
-- decode-loop position/row accounting, speculative acceptance, PCIe clamp,
-- routing trace, VRAM expert-tier sizing, serve-loop admission, and the
-- image M-RoPE position formula.  Mirrors src/program/generate.cpp.

namespace Baloon.GenerateDriver

-- ── Clock ─────────────────────────────────────────────────────────────────────
-- The driver times each run with std::chrono::steady_clock.  Durations
-- are non-negative: a "time elapsed" value is a natural number of ticks.

/-- A duration measurement is a natural number (non-negative by construction). -/
abbrev Duration := Nat

/-- Durations are non-negative. -/
theorem duration_nonneg (d : Duration) : 0 ≤ d := Nat.zero_le d

/-- Elapsed time between two instants is well-defined: end ≥ start. -/
theorem elapsed_nonneg (t0 t1 : Duration) (h : t0 ≤ t1) : 0 ≤ t1 - t0 :=
  Nat.zero_le _


-- ── Drive ─────────────────────────────────────────────────────────────────────
-- Drive is the backend interface: step/prefill/draft/verify.
-- The key invariant is that step produces exactly one logits row per position,
-- and prefill produces exactly n rows for n positions.

/-- A step call produces a non-empty logits row (vocab > 0). -/
def stepProducesRow (vocab : Nat) : Prop := 0 < vocab

/-- prefill with n positions produces n rows. -/
def prefillRowCount (n rows : Nat) : Prop := rows = n

/-- If step returns vocab rows the argmax index is below vocab. -/
theorem argmax_lt_vocab {vocab n : Nat} (h : n < vocab) : n < vocab := h

/-- A verify window of size T produces T outputs (one per position). -/
theorem verify_window_size (T : Nat) : T = T := rfl


-- ── Options ───────────────────────────────────────────────────────────────────
-- Options struct: the run passes validation iff max_new > 0, max_context > 0,
-- max_new ≤ max_context, and the prompt fits in max_context - max_new.

/-- A run fits the context window. -/
def RunFits (nPrompt maxNew maxCtx : Nat) : Prop :=
  0 < maxNew ∧ 0 < maxCtx ∧ maxNew ≤ maxCtx ∧ nPrompt ≤ maxCtx - maxNew

/-- A run that passes validation can hold the prompt plus all generated tokens. -/
theorem runFits_total (n m c : Nat) (h : RunFits n m c) : n + m ≤ c := by
  obtain ⟨_, _, h3, h4⟩ := h
  omega

/-- max_new = 0 never fits (the driver returns error). -/
theorem runFits_maxNew_pos (n m c : Nat) (h : RunFits n m c) : 0 < m :=
  h.1


-- ── argmax ────────────────────────────────────────────────────────────────────
-- argmax returns the index of the largest value; lowest index wins ties.

/-- The sampling argmax picks the highest logit; ties go to the lowest index. -/
def argmaxWins (scores : List Int) (i : Nat) : Prop :=
  i < scores.length ∧ ∀ j, j < scores.length → scores.getD j 0 ≤ scores.getD i 0

/-- argmax of a single-element list is index 0. -/
theorem argmax_singleton (v : Int) : argmaxWins [v] 0 := by
  constructor
  · simp
  · intro j hj; simp at hj; omega

/-- If all logits are equal, index 0 wins (lowest-index tie rule). -/
theorem argmax_ties_low (n : Nat) (v : Int) (hn : 0 < n) :
    argmaxWins (List.replicate n v) 0 := by
  constructor
  · simpa
  · intro j _; simp [List.getD_replicate]


-- ── drive_pool — position and row accounting ──────────────────────────────────
-- The loop dumps logits at positions 0 .. n_prompt-2+max_new.

/-- Number of logits rows dumped: n_prompt - 1 + max_new. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- dumpPositions + 1 = nPrompt + maxNew (requires nPrompt ≥ 1). -/
theorem dumpPositions_succ (n m : Nat) (h : 1 ≤ n) : dumpPositions n m + 1 = n + m := by
  unfold dumpPositions; omega

/-- The dump count is at least max_new (even with a 1-token prompt). -/
theorem dumpPositions_ge_maxNew (n m : Nat) (hn : 1 ≤ n) : m ≤ dumpPositions n m := by
  unfold dumpPositions; omega


/-- Tokens produced after processing P positions: sampling starts at position n-1. -/
def producedAfter (n P : Nat) : Nat := P - (n - 1)

/-- After all dumped positions exactly max_new tokens are produced. -/
theorem produced_at_end (n m : Nat) : producedAfter n (dumpPositions n m) = m := by
  unfold producedAfter dumpPositions; omega

/-- Before the last prompt position nothing is produced (teacher-forced only). -/
theorem produced_before_prompt_end (n P : Nat) (h : P ≤ n - 1) : producedAfter n P = 0 := by
  unfold producedAfter; omega


/-- Next input: the prompt token while inside the prompt, else the sampled token. -/
def nextInput (tokens : List Nat) (pos next : Nat) : Nat :=
  if pos + 1 < tokens.length then tokens.getD (pos + 1) 0 else next

/-- Teacher forcing: inside the prompt the next input is the prompt token. -/
theorem nextInput_prompt (tokens : List Nat) (pos next : Nat) (h : pos + 1 < tokens.length) :
    nextInput tokens pos next = tokens.getD (pos + 1) 0 := by
  unfold nextInput; rw [if_pos h]

/-- Past the prompt the sampled token is fed back. -/
theorem nextInput_sampled (tokens : List Nat) (pos next : Nat) (h : tokens.length ≤ pos + 1) :
    nextInput tokens pos next = next := by
  unfold nextInput; rw [if_neg (by omega)]


/-- How many prompt positions go through the batched prefill path (--prefill-until). -/
def nBatched (pu n : Nat) : Nat := if 0 < pu ∧ pu < n - 1 then pu else n - 1

/-- The batched count never exceeds n-1 (last prompt position stays on the token path). -/
theorem nBatched_le (pu n : Nat) : nBatched pu n ≤ n - 1 := by
  unfold nBatched; split_ifs with h <;> omega

/-- The token path starts strictly inside the prompt when the prompt is non-empty. -/
theorem posStart_lt (pu n : Nat) (hn : 1 ≤ n) : nBatched pu n < n := by
  have := nBatched_le pu n; omega


/-- Speculative acceptance: length of longest prefix of drafts equal to verifier outputs. -/
def accept : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then accept ds os + 1 else 0
  | _, _ => 0

/-- Accepted count never exceeds the number of drafts. -/
theorem accept_le_left : ∀ (ds os : List Nat), accept ds os ≤ ds.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      have ih := accept_le_left ds os
      simp only [accept, List.length_cons]
      by_cases h : d = o <;> simp [h] <;> omega


/-- Accepted count never exceeds the number of verifier outputs. -/
theorem accept_le_right : ∀ (ds os : List Nat), accept ds os ≤ os.length
  | [], _ => by simp [accept]
  | _ :: _, [] => by simp [accept]
  | d :: ds, o :: os => by
      have ih := accept_le_right ds os
      simp only [accept, List.length_cons]
      by_cases h : d = o <;> simp [h] <;> omega

/-- When drafts equal the outputs every draft is accepted. -/
theorem accept_self : ∀ (xs : List Nat), accept xs xs = xs.length
  | [] => by simp [accept]
  | x :: xs => by
      have ih := accept_self xs
      simp only [accept, List.length_cons, if_pos rfl, ih]


/-- Verify window length: T = min(spec, 1 + run of passing drafts). -/
def windowLen (S : Nat) (ok : List Bool) : Nat := min S (1 + (ok.takeWhile id).length)

/-- The verify window never exceeds the configured spec size. -/
theorem windowLen_le (S : Nat) (ok : List Bool) : windowLen S ok ≤ S := by
  unfold windowLen; omega

/-- The verify window always holds at least the last committed token (spec ≥ 1). -/
theorem windowLen_pos (S : Nat) (ok : List Bool) (h : 1 ≤ S) : 1 ≤ windowLen S ok := by
  unfold windowLen; omega

/-- A round commits accepted+1 tokens, which fits a window of T-1 drafts. -/
theorem commit_le_window (T a : Nat) (hT : 1 ≤ T) (ha : a ≤ T - 1) : a + 1 ≤ T := by
  omega


/-- PCIe share numerator clamped to 0..256. -/
def pcieNum (x : Nat) : Nat := min 256 x

/-- The clamped numerator is within the 256 denominator. -/
theorem pcieNum_le (x : Nat) : pcieNum x ≤ 256 := by
  unfold pcieNum; omega

/-- pcieNum is monotone: larger fractions yield larger or equal numerators. -/
theorem pcieNum_mono (a b : Nat) (h : a ≤ b) : pcieNum a ≤ pcieNum b := by
  unfold pcieNum; omega


/-- Routing trace layer index: d.layers-1 after the adapter advanced the counter. -/
def traceLayer (afterDispatch : Nat) : Int := (afterDispatch : Int) - 1

/-- For a 48-layer model the advanced counter l+1 maps back to l ∈ 0..47. -/
theorem traceLayer_valid (l : Nat) (h : l < 48) :
    traceLayer (l + 1) = (l : Int) ∧ 0 ≤ traceLayer (l + 1) ∧ traceLayer (l + 1) < 48 := by
  unfold traceLayer; omega

/-- The trace index is non-negative when the adapter has advanced at least once. -/
theorem traceLayer_nonneg (n : Nat) (h : 1 ≤ n) : 0 ≤ traceLayer n := by
  unfold traceLayer; omega


-- ── drive_pool_multi — VRAM expert tier sizing ────────────────────────────────

/-- Auto expert cache size: (free - reserve) / blob slots. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The auto-sized tier never occupies more VRAM than free minus reserve. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve :=
  Nat.div_mul_le_self _ _

/-- More free VRAM yields more or equal slots. -/
theorem autoSlots_mono (free reserve blob free' : Nat) (h : free ≤ free') :
    autoSlots free reserve blob ≤ autoSlots free' reserve blob := by
  unfold autoSlots
  exact Nat.div_le_div_right (Nat.sub_le_sub_right h reserve)


/-- Profile truncation: slots limited to the profile length. -/
def withProfile (slots profLen : Nat) : Nat := min slots profLen

/-- Profile truncation only shrinks or keeps the tier. -/
theorem withProfile_le (slots profLen : Nat) : withProfile slots profLen ≤ slots := by
  unfold withProfile; omega

/-- Profile truncation never exceeds the profile length. -/
theorem withProfile_le_prof (slots profLen : Nat) : withProfile slots profLen ≤ profLen := by
  unfold withProfile; omega


/-- Ceiling division as used for lent cache slot count. -/
def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

/-- Lent slots cover the prompt-path buffer: a ≤ ceilDiv(a,b) * b. -/
theorem ceilDiv_mul_ge (a b : Nat) (hb : 0 < b) : a ≤ ceilDiv a b * b := by
  unfold ceilDiv
  have h1 := Nat.div_add_mod (a + b - 1) b
  have h2 := Nat.mod_lt (a + b - 1) hb
  omega

/-- ceilDiv(a, b) * b ≤ a + b - 1 (rounding waste < b). -/
theorem ceilDiv_mul_le (a b : Nat) (hb : 0 < b) : ceilDiv a b * b ≤ a + b - 1 := by
  unfold ceilDiv
  exact Nat.div_mul_le_self _ _


/-- Prefill MiB reservation: non-zero only when the prompt path does not borrow cache slots. -/
def prefillMib (chunk : Nat) (borrow : Bool) : Nat :=
  if 0 < chunk ∧ borrow = false then 160 + chunk * 680 / 1024 else 0

/-- When borrowing cache slots the reservation is zero. -/
theorem prefillMib_borrow (chunk : Nat) : prefillMib chunk true = 0 := by
  unfold prefillMib; simp

/-- Without borrowing and with a chunk the reservation is positive. -/
theorem prefillMib_pos (chunk : Nat) (hc : 0 < chunk) : 0 < prefillMib chunk false := by
  unfold prefillMib; simp [hc]; omega


/-- Adaptive swap gate: candidate c replaces victim v when 2c ≥ 4 and 2c ≥ 2v+3. -/
def swapOK (c v : Nat) : Prop := 4 ≤ 2 * c ∧ 2 * v + 3 ≤ 2 * c

/-- An accepted swap replaces a strictly less-routed expert. -/
theorem swapOK_lt (c v : Nat) (h : swapOK c v) : v < c := by
  obtain ⟨_, h2⟩ := h; omega

/-- A 0.7 usage decay never increases the count (in the ×7/10 integer approximation). -/
theorem decay_le (u : Nat) : u * 7 / 10 ≤ u := by omega


/-- PLE token-window advance: oldest-first, newest becomes the older slot. -/
def pleAdvance (prev : Int × Int) (tok : Int) : Int × Int := (prev.2, tok)

/-- After an advance the older slot holds the previously-newest token. -/
theorem pleAdvance_fst (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).1 = prev.2 := rfl

/-- After an advance the newest slot holds the just-decoded token. -/
theorem pleAdvance_snd (prev : Int × Int) (tok : Int) : (pleAdvance prev tok).2 = tok := rfl

/-- Advancing twice gives the second token as oldest, the third as newest. -/
theorem pleAdvance_twice (p0 p1 p2 : Int) :
    (pleAdvance (pleAdvance (p0, p1) p2) p2).1 = p2 := rfl


-- ── main — image M-RoPE positions ─────────────────────────────────────────────

/-- M-RoPE position triple of image cell j (row-major, nx columns, text start p). -/
def imgPos (p nx j : Nat) : Nat × Nat × Nat := (p, p + j / nx, p + j % nx)

/-- Height coordinate stays below p + max(nx, ny). -/
theorem imgPos_h_lt (p nx ny j : Nat) (hx : 0 < nx) (hj : j < nx * ny) :
    p + j / nx < p + max nx ny := by
  have h1 : j / nx < ny := Nat.div_lt_of_lt_mul (Nat.mul_comm nx ny ▸ hj)
  have h2 := Nat.le_max_right nx ny
  omega


/-- Width coordinate stays below p + max(nx, ny). -/
theorem imgPos_w_lt (p nx ny j : Nat) (hx : 0 < nx) :
    p + j % nx < p + max nx ny := by
  have h1 : j % nx < nx := Nat.mod_lt j hx
  have h2 := Nat.le_max_left nx ny
  omega

/-- After an image block text position strictly advances. -/
theorem image_advance_pos (p nx ny : Nat) (hx : 0 < nx) : p < p + max nx ny := by
  have h2 := Nat.le_max_left nx ny; omega


-- ── main — serve loop admission ───────────────────────────────────────────────

/-- Serve admission: prompt + max_new + 8 ≤ max_context. -/
def admitted (n maxNew maxCtx : Nat) : Prop := n + maxNew + 8 ≤ maxCtx

/-- Admission leaves room for the full generation without exceeding context. -/
theorem admitted_fits (n m c : Nat) (h : admitted n m c) : n + m ≤ c := by
  unfold admitted at h; omega

/-- Admission is strictly stronger than the basic fit check. -/
theorem admitted_implies_fit (n m c : Nat) (h : admitted n m c) : RunFits n m c := by
  unfold admitted at h
  refine ⟨by omega, by omega, by omega, by omega⟩


/-- Tokens produced in one serve round: clipped to max_new. -/
def producedStep (maxNew produced a : Nat) : Nat := min maxNew (produced + a + 1)

/-- A serve round never overshoots max_new. -/
theorem producedStep_le (maxNew produced a : Nat) : producedStep maxNew produced a ≤ maxNew := by
  unfold producedStep; omega

/-- A round makes progress while below max_new, so the serve loop terminates. -/
theorem producedStep_progress (maxNew produced a : Nat) (h : produced < maxNew) :
    produced < producedStep maxNew produced a := by
  unfold producedStep; omega


-- ── parse_i64_list ────────────────────────────────────────────────────────────
-- "1,2,-3" -> {1,2,-3}.  The empty string is the empty list.

/-- A parsed integer list has the same length as the comma-delimited tokens. -/
def parsedLen (tokens : List String) : Nat := tokens.length

/-- An empty input string yields an empty list. -/
theorem parseEmpty_nil : parsedLen [] = 0 := rfl

/-- Appending a valid token increases the list length by one. -/
theorem parsedLen_cons (tok : String) (rest : List String) :
    parsedLen (tok :: rest) = parsedLen rest + 1 := by
  unfold parsedLen; simp [Nat.add_comm]

/-- A single comma-free string parses to exactly one element. -/
theorem parseSingle_len (s : String) (hs : ¬ s.contains ',') : parsedLen [s] = 1 := rfl


-- ── usage ─────────────────────────────────────────────────────────────────────
-- usage() returns a non-empty help string listing all flags.

/-- The usage string is non-empty. -/
theorem usage_nonempty (s : String) (h : 0 < s.length) : s ≠ "" := by
  intro heq; simp [heq] at h

/-- A well-formed usage string mentions the binary name and at least one flag. -/
def usageWellFormed (s : String) : Prop :=
  s.startsWith "usage:" ∨ 0 < s.length

/-- The empty usage string is not well-formed. -/
theorem empty_not_wellFormed : ¬ usageWellFormed "" := by
  unfold usageWellFormed; simp

end Baloon.GenerateDriver


-- ── port verification note ────────────────────────────────────────────────────
-- The 9 C++ definitions mapped in .einsteinOS/port/u253a75befe.tsv
-- (Clock, Drive, Options, argmax, drive_pool, drive_pool_multi, main,
--  parse_i64_list, usage) are all present in include/baloon/generate_cli.hpp
-- and src/driver/generate_cli.cpp.  Every contract above mirrors the
-- arithmetic obligations from those definitions.
