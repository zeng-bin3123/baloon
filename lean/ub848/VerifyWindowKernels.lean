/-!
# MODULE: verify_window_kernels
SOURCES: src/kernels/cuda/verify_kernels.cu
Multi-token (speculative verify) versions of the GDN conv/L2, alpha/beta and step+norm kernels,
the conv-history commit after acceptance, and the small index/gather kernels of the verify loop
(window ids, dense steps, blob fetch, row gather, embedding gather, MTP select).

## Source semantics
* Token `t` of a window sees the conv window `[hist(3) | x_0 .. x_t]` at indices `t..t+3`:
  entry `src = t + j` is `hist[src]` if `src < 3`, else `x[src-3]`. This reproduces plain
  decode bit for bit (same operation order per token).
* The conv window kernel does NOT write history; `gdn_conv_commit(n_keep)` writes the last three
  of `[hist | x_0..x_{n-1}]`, and is a no-op for `n ≤ 0`.
* `gdn_step_norm_multi` runs the recurrence over `n = n_keep ? *n_keep : T` tokens, emits
  outputs only for `t ≥ t_out_begin` (replayed tokens update state only), and writes the state
  back only when `n_keep` is given and `n > 0`.
* `window_ids`: `start = max(n_kv - window, 0)`, `width = n_kv - start`, ids `start + j`.
* `dense_steps`: `steps[4i..4i+3] = [c, c+1, (c+1)/4, c+1]`.
* Embedding gather: `code = (byte >> ((i % per_byte)*bits)) & (2^bits - 1)`, `per_byte = 8/bits`.
* `n_tok ∈ [1, kVerifyMaxT]`, blob and row sizes multiples of 16 bytes.
-/

namespace Ref.VerifyWindow

-- § conv window reproduces sequential decode

/-- The virtual sequence `[hist(3) | x_0, x_1, ...]`. -/
def seqVal (hist x : Nat → Int) (n : Nat) : Int := if n < 3 then hist n else x (n - 3)

/-- The 3-entry conv history seen by token `t` in the window kernel (`win[j] = seq[t+j]`). -/
def stateAt (hist x : Nat → Int) (t : Nat) : Int × Int × Int :=
  (seqVal hist x t, seqVal hist x (t + 1), seqVal hist x (t + 2))

/-- One decode step of the conv history: drop the oldest, append the input. -/
def stepHist (h : Int × Int × Int) (v : Int) : Int × Int × Int := (h.2.1, h.2.2, v)

/-- History after feeding `x_0 .. x_{t-1}` one token at a time (plain decode). -/
def seqState (hist x : Nat → Int) : Nat → Int × Int × Int
  | 0 => (hist 0, hist 1, hist 2)
  | t + 1 => stepHist (seqState hist x t) (x t)

/-- The token after the window is the token's own input: `seq[t+3] = x_t`. -/
theorem seqVal_input (hist x : Nat → Int) (t : Nat) : seqVal hist x (t + 3) = x t := by
  have h : ¬ (t + 3 < 3) := by omega
  unfold seqVal; simp only [h, ↓reduceIte, Nat.add_sub_cancel]

/-- Main theorem: the window kernel's history for token `t` equals plain decode's history
    after `t` sequential steps. -/
theorem window_eq_sequential (hist x : Nat → Int) (t : Nat) :
    stateAt hist x t = seqState hist x t := by
  induction t with
  | zero => simp [stateAt, seqState, seqVal]
  | succ t ih =>
    have hs : seqState hist x (t + 1) = stepHist (seqState hist x t) (x t) := rfl
    rw [hs, ← ih]
    simp only [stateAt, stepHist]
    rw [← seqVal_input hist x t]

/-- The 4-tap conv sum over a history and the new input. -/
def convSum (h : Int × Int × Int) (v w0 w1 w2 w3 : Int) : Int :=
  h.1 * w0 + h.2.1 * w1 + h.2.2 * w2 + v * w3

/-- Hence every token's conv output in the window equals the plain-decode output. -/
theorem window_output_eq (hist x : Nat → Int) (t : Nat) (w0 w1 w2 w3 : Int) :
    convSum (stateAt hist x t) (x t) w0 w1 w2 w3 =
      convSum (seqState hist x t) (x t) w0 w1 w2 w3 := by
  rw [window_eq_sequential]

/-- `gdn_conv_commit`: write the last three of `[hist | x_0..x_{n-1}]`; no-op for `n = 0`. -/
def commit (hist x : Nat → Int) (n : Nat) : Int × Int × Int :=
  if n = 0 then (hist 0, hist 1, hist 2) else stateAt hist x n

/-- Committing `n` accepted tokens yields exactly the history plain decode would have. -/
theorem commit_eq_sequential (hist x : Nat → Int) (n : Nat) :
    commit hist x n = seqState hist x n := by
  unfold commit
  split
  · subst_vars; rfl
  · exact window_eq_sequential hist x n

-- § recurrence replay

/-- Fold of a per-token state transition over tokens `0 .. n-1`. -/
def foldSteps {σ : Type} (f : σ → Nat → σ) (s0 : σ) : Nat → σ
  | 0 => s0
  | n + 1 => f (foldSteps f s0 n) n

/-- State written back by `gdn_step_norm_multi`: only with `n_keep` given and positive. -/
def committedState {σ : Type} (f : σ → Nat → σ) (s0 : σ) (nKeep : Option Nat) : σ :=
  match nKeep with
  | some n => if n > 0 then foldSteps f s0 n else s0
  | none => s0

/-- A pure verify pass (no `n_keep`) never mutates the recurrent state. -/
theorem verify_without_keep_is_pure {σ : Type} (f : σ → Nat → σ) (s0 : σ) :
    committedState f s0 none = s0 := rfl

/-- Accepting `n` tokens commits the state of `n` sequential steps. -/
theorem keep_commits_prefix {σ : Type} (f : σ → Nat → σ) (s0 : σ) (n : Nat) :
    committedState f s0 (some n) = foldSteps f s0 n := by
  cases n with
  | zero => rfl
  | succ n => simp [committedState]

/-- Tokens whose output is emitted: `t_out_begin ≤ t < n`. -/
def emits (tOutBegin n t : Nat) : Bool := tOutBegin ≤ t && t < n

/-- Replayed tokens (`t < t_out_begin`) produce no output. -/
theorem replay_no_output {b n t : Nat} (h : t < b) : emits b n t = false := by
  simp [emits]; omega

-- § window ids and dense steps

/-- `start = n_kv > window ? n_kv - window : 0`. -/
def windowStart (nkv window : Nat) : Nat := if nkv > window then nkv - window else 0

/-- `width = n_kv - start`. -/
def windowWidth (nkv window : Nat) : Nat := nkv - windowStart nkv window

/-- The window width is `min(n_kv, window)`. -/
theorem windowWidth_eq_min (nkv window : Nat) : windowWidth nkv window = min nkv window := by
  unfold windowWidth windowStart; split <;> omega

/-- Every emitted id `start + j`, `j < width`, is a valid past position `< n_kv`. -/
theorem window_ids_lt {nkv window j : Nat} (hj : j < windowWidth nkv window) :
    windowStart nkv window + j < nkv := by
  unfold windowWidth at hj; unfold windowStart at hj ⊢
  by_cases hc : nkv > window
  · simp only [hc, ↓reduceIte] at hj ⊢; omega
  · simp only [hc, ↓reduceIte] at hj ⊢; omega

/-- `dense_steps` row for cell `c`. -/
def denseStep (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

/-- The dense step's `n_kv` field is one past the cell, and its block count is `⌊(c+1)/4⌋`. -/
theorem denseStep_fields (c : Nat) :
    (denseStep c)[1]! = c + 1 ∧ (denseStep c)[2]! * 4 ≤ c + 1 := by
  simp [denseStep]; omega

-- § gathers

/-- Extracted code of element with in-byte slot `k`. -/
def codeAt (byte bits k : Nat) : Nat := (byte / 2 ^ (k * bits)) % 2 ^ bits

/-- An extracted code always fits in `bits` bits. -/
theorem codeAt_lt (byte bits k : Nat) : codeAt byte bits k < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- For the supported widths the slots of a byte tile it exactly: `per_byte * bits = 8`. -/
theorem per_byte_tiles : ∀ b ∈ [1, 2, 4, 8], (8 / b) * b = 8 := by decide

/-- Flat blob index decomposition `i = k*per + off` with `off < per` is unique. -/
theorem blob_index_unique {per k off k' off' : Nat} (ho : off < per) (ho' : off' < per)
    (h : k * per + off = k' * per + off') : k = k' ∧ off = off' := by
  have hp : 0 < per := by omega
  have m1 : (off + k * per) % per = off := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho]
  have m2 : (off' + k' * per) % per = off' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ho']
  have hoo : off = off' := by rw [← m1, ← m2, Nat.add_comm off, Nat.add_comm off', h]
  subst hoo
  exact ⟨Nat.eq_of_mul_eq_mul_right hp (by omega), rfl⟩

/-- `copy_indexed` with a negative index copies nothing. -/
def copyIndexed (idx : Int) (src dst : Nat → Int) (stride : Nat) : Nat → Int :=
  if idx < 0 then dst else fun i => src (idx.toNat * stride + i)

/-- A negative index leaves the destination untouched. -/
theorem copyIndexed_neg {idx : Int} (h : idx < 0) (src dst : Nat → Int) (stride : Nat) :
    copyIndexed idx src dst stride = dst := by simp [copyIndexed, h]

/-- `ident_hits` followed by `map_ids`: slot `i` holds `table[ids[i]]`, destination `i`. -/
theorem ident_then_map (ids table : Nat → Nat) (i : Nat) :
    (fun j => table (ids j)) i = table (ids i) := rfl

/-- Host bound on the verify window length. -/
def nTokOk (maxT n : Nat) : Bool := 1 ≤ n && n ≤ maxT

/-- A passing window has at least one token and fits the per-thread accumulator array. -/
theorem nTokOk_bounds {m n : Nat} (h : nTokOk m n = true) : 1 ≤ n ∧ n ≤ m := by
  simp [nTokOk] at h; omega

end Ref.VerifyWindow

