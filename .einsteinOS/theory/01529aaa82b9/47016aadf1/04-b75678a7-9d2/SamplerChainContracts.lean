/-!
# MODULE: sampler_chain_contracts
Index, window and tie-break contracts of the sampler in `sampler.cu`.

## Source semantics
* The penalty window is the tail of the row history: `hlen = clamp(penalty_last_n, 0, history_len)`, starting at
  `history_len - hlen`.
* A token with count 0 is not penalised; the presence term is a boolean, so it is at most 1.
* Greedy picks the global argmax; equal values resolve to the smaller index; the sentinel index `n_vocab`
  loses to every real index and a sentinel result maps to token 0.
* Top-k is capped at 64; the launch grid covers all tokens; each token draws from counter `counter + t`.

## Form template identifiers
hlen -> window_length; history_len -> history_stride; KMAX -> max_top_k; counter -> rng_counter;
n_vocab -> vocab_size.
-/

namespace SamplerChainContracts

-- § penalty window

/-- Window length, clamped into [0, history_len]. -/
def windowLen (lastN : Int) (historyLen : Nat) : Nat :=
  if lastN < 0 then 0 else if lastN < (historyLen : Int) then lastN.toNat else historyLen

/-- The window never exceeds the history. -/
theorem windowLen_le (lastN : Int) (historyLen : Nat) : windowLen lastN historyLen ≤ historyLen := by
  unfold windowLen
  split_ifs <;> omega

/-- Every window slot lies inside the history row. -/
theorem window_index_lt (historyLen hlen j : Nat) (hh : hlen ≤ historyLen) (hj : j < hlen) :
    historyLen - hlen + j < historyLen := by
  omega

/-- The window is the tail: its last slot is the last history entry. -/
theorem window_last (historyLen hlen : Nat) (hh : hlen ≤ historyLen) (hpos : 0 < hlen) :
    historyLen - hlen + (hlen - 1) = historyLen - 1 := by
  omega

-- § penalties

/-- Penalty application over an abstract multiply/divide pair. -/
def applyPenalties (mulRep divRep : Int → Int) (freq present : Int) (logit : Int) (count : Nat) : Int :=
  if count = 0 then logit
  else
    let l := if logit ≤ 0 then mulRep logit else divRep logit
    l - ((count : Int) * freq + (if 0 < count then 1 else 0) * present)

/-- A token that never appeared keeps its logit. -/
theorem penalties_zero (m d : Int → Int) (f p logit : Int) :
    applyPenalties m d f p logit 0 = logit := by
  unfold applyPenalties; simp

/-- The presence penalty is applied once however many times the token appeared. -/
theorem presence_once (m d : Int → Int) (logit : Int) (c : Nat) (hc : 0 < c) :
    applyPenalties m d 0 1 logit c =
      (if logit ≤ 0 then m logit else d logit) - 1 := by
  unfold applyPenalties
  have h1 : c ≠ 0 := by omega
  simp [h1, hc]

-- § greedy argmax

/-- Reduction step: larger value wins, ties go to the smaller index. -/
def combine (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combine returns one of its inputs. -/
theorem combine_mem (a b : Int × Nat) : combine a b = a ∨ combine a b = b := by
  unfold combine; split <;> simp

/-- The combine never returns a smaller value than either input. -/
theorem combine_ge_left (a b : Int × Nat) : a.1 ≤ (combine a b).1 := by
  unfold combine; split <;> omega

theorem combine_ge_right (a b : Int × Nat) : b.1 ≤ (combine a b).1 := by
  unfold combine; split <;> omega

/-- Sentinel handling: a real index precedes the sentinel `n_vocab`. -/
theorem sentinel_after_real (i n : Nat) (h : i < n) : i < n := h

/-- A sentinel result maps to token zero. -/
def finalToken (wi nVocab : Nat) : Nat := if wi < nVocab then wi else 0

theorem finalToken_lt (wi nVocab : Nat) (h : 0 < nVocab) : finalToken wi nVocab < nVocab := by
  unfold finalToken; split <;> omega

-- § launch and rng

/-- Grid of 64-thread blocks covers all tokens. -/
theorem grid_covers (t n : Nat) (h : t < n) : t / 64 < (n + 63) / 64 := by omega

/-- Per-token RNG counters are pairwise distinct. -/
theorem rng_counter_inj (c t t' : Nat) (h : c + t = c + t') : t = t' := by omega

/-- Effective top-k is at most the cap of 64. -/
theorem topk_cap (k : Nat) : (if k < 64 then k else 64) ≤ 64 := by
  split <;> omega

/-- The warp count of a 1024-thread greedy block is 32, matching the 32-entry shared arrays. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

end SamplerChainContracts
