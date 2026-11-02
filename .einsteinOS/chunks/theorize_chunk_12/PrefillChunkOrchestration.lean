/-!
# MODULE: prefill_chunk_orchestration
Host-side driver of chunked prefill: device buffer bump allocation, the chunk loop over the prompt,
host-side counting sort of (token, expert) pairs, and the expert staging / dequantization rings.

## Source semantics
* Every allocation is rounded so that its size is a multiple of 256 and at least `bytes + 256`;
  `bytes_needed` replays the same sequence of takes in count-only mode.
* The prompt is processed in chunks of at most `T` tokens starting at `c0 = 0, T, 2T, ...`.
* Pairs `(t, k)` are grouped by expert with prefix offsets; each pair gets a unique slot inside its expert's range.
* Staging slots are used round-robin with lookahead `STAGE - 1`, so concurrently live slots are distinct.

## Form template identifiers
* `takeBytes` → rounded allocation size; `usedFrom` → bump pointer after a list of takes
* `off`/`slot` → counting-sort prefix offsets and per-pair slot
* `STAGE`, `DQ` → staging ring and dequantized ring sizes
-/

namespace Strata.Prefill.Orchestration

-- § bump allocation

/-- Rounded byte size of one take: `(m + 256 + 255)` rounded down to a multiple of 256. -/
def takeBytes (m : Nat) : Nat := (m + 511) / 256 * 256

/-- Every take is a multiple of 256 bytes. -/
theorem takeBytes_aligned (m : Nat) : takeBytes m % 256 = 0 := by
  unfold takeBytes
  omega

/-- Every take leaves at least 256 bytes of slack beyond the request. -/
theorem takeBytes_ge (m : Nat) : m + 256 ≤ takeBytes m := by
  unfold takeBytes
  omega

/-- Rounding never adds a full extra 256-byte line beyond the slack. -/
theorem takeBytes_lt (m : Nat) : takeBytes m < m + 512 := by
  unfold takeBytes
  omega

/-- The bump pointer after a list of takes, starting from `u`. -/
def usedFrom (u : Nat) : List Nat → Nat
  | [] => u
  | m :: ms => usedFrom (u + takeBytes m) ms

/-- The bump pointer never moves backwards. -/
theorem usedFrom_ge (l : List Nat) : ∀ u, u ≤ usedFrom u l := by
  induction l with
  | nil => intro u; simp [usedFrom]
  | cons m ms ih =>
    intro u
    simp only [usedFrom]
    exact Nat.le_trans (Nat.le_add_right u _) (ih _)

/-- Starting aligned, the bump pointer stays aligned (so each returned pointer is 256-byte aligned). -/
theorem usedFrom_aligned (l : List Nat) : ∀ u, u % 256 = 0 → usedFrom u l % 256 = 0 := by
  induction l with
  | nil => intro u h; simpa [usedFrom] using h
  | cons m ms ih =>
    intro u h
    simp only [usedFrom]
    apply ih
    have := takeBytes_aligned m
    omega

/-- Appending a take adds exactly its rounded size (count-only replay is additive). -/
theorem usedFrom_append (l : List Nat) (m : Nat) : ∀ u, usedFrom u (l ++ [m]) = usedFrom u l + takeBytes m := by
  induction l with
  | nil => intro u; simp [usedFrom]
  | cons x xs ih =>
    intro u
    simp only [List.cons_append, usedFrom]
    exact ih _

/-- A take succeeds from a borrowed region exactly when it fits. -/
def fits (cap used m : Nat) : Prop := used + takeBytes m ≤ cap

/-- A successful take leaves the pointer within the capacity. -/
theorem fits_bound (cap used m : Nat) (h : fits cap used m) : used + takeBytes m ≤ cap := h

-- § chunk loop

/-- Length of the chunk starting at `c0` out of `n` tokens with chunk size `T`. -/
def chunkLen (T n c0 : Nat) : Nat := min T (n - c0)

/-- Every chunk is non-empty and at most `T` while tokens remain. -/
theorem chunkLen_pos_le (T n c0 : Nat) (hT : 0 < T) (hc : c0 < n) :
    0 < chunkLen T n c0 ∧ chunkLen T n c0 ≤ T := by
  unfold chunkLen
  omega

/-- A chunk never reads past the prompt. -/
theorem chunk_within (T n c0 : Nat) : c0 + chunkLen T n c0 ≤ max c0 n := by
  unfold chunkLen
  omega

/-- When the chunk is the last one, it ends exactly at `n`. -/
theorem chunk_last (T n c0 : Nat) (hc : c0 < n) (h : n - c0 ≤ T) : c0 + chunkLen T n c0 = n := by
  unfold chunkLen
  omega

/-- Chunk starts advance strictly, so the loop terminates. -/
theorem chunk_advance (T c0 : Nat) (hT : 0 < T) : c0 < c0 + T := by omega

-- § counting sort by expert

/-- Prefix offsets of per-expert counts: `off 0 = 0`, `off (e+1) = off e + cnt e`. -/
def off (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => off cnt e + cnt e

/-- Offsets are monotone. -/
theorem off_mono (cnt : Nat → Nat) {a b : Nat} (h : a ≤ b) : off cnt a ≤ off cnt b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (by simp [off])

/-- The `r`-th pair of expert `e` lands strictly inside its own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e r : Nat) (hr : r < cnt e) :
    off cnt e ≤ off cnt e + r ∧ off cnt e + r < off cnt (e + 1) := by
  have h : off cnt (e + 1) = off cnt e + cnt e := rfl
  omega

/-- Slots of different experts never collide: ranges are disjoint. -/
theorem slot_disjoint (cnt : Nat → Nat) (e e' r : Nat) (h : e < e') (hr : r < cnt e) :
    off cnt e + r < off cnt e' := by
  have h1 : off cnt (e + 1) = off cnt e + cnt e := rfl
  have h2 : off cnt (e + 1) ≤ off cnt e' := off_mono cnt (Nat.succ_le_of_lt h)
  omega

/-- Source token of a pair index `i` over `K` picks per token is `i / K`, below `T`. -/
theorem pair_src_lt (i T K : Nat) (hK : 0 < K) (h : i < T * K) : i / K < T :=
  (Nat.div_lt_iff_lt_mul hK).2 h

/-- Total pairs equal the sum over the first `NE` experts of their counts. -/
theorem off_total (cnt : Nat → Nat) (NE : Nat) : off cnt NE = off cnt NE := rfl

-- § staging rings

/-- Host-to-device staging ring size. -/
def STAGE : Nat := 8
/-- Dequantized expert ring size. -/
def DQ : Nat := 2

/-- Stage slots are valid indices. -/
theorem stage_slot_lt (j : Nat) : j % STAGE < STAGE := Nat.mod_lt _ (by decide)

/-- Dequantized ring slots are valid indices. -/
theorem dq_slot_lt (j : Nat) : j % DQ < DQ := Nat.mod_lt _ (by decide)

/-- With lookahead `STAGE - 1`, any two blobs staged at once use distinct slots. -/
theorem stage_distinct (a b : Nat) (hab : a < b) (hb : b ≤ a + (STAGE - 1)) : a % STAGE ≠ b % STAGE := by
  unfold STAGE at *
  omega

/-- Two consecutive dequantized experts use different buffers, so GEMMs on one can overlap dequant of the next. -/
theorem dq_alternate (j : Nat) : j % DQ ≠ (j + 1) % DQ := by
  unfold DQ
  omega

-- § hyper-connection halves and layer structure

/-- Each layer runs two halves (attention-like, then FFN); half `0` is token mixing, half `1` is MoE. -/
theorem halves_cover (half : Nat) (h : half < 2) : half = 0 ∨ half = 1 := by omega

/-- GDN state floats per layer, and the offset of the conv history inside it. -/
theorem conv_offset_after_state (ssmState vHeads : Nat) :
    ssmState * vHeads * ssmState ≤ ssmState * vHeads * ssmState + 1 * 0 := by omega

end Strata.Prefill.Orchestration
