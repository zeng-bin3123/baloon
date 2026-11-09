/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu, include/strata/kernels/sampler.hpp
Device sampler for one token per row: repeat/frequency/presence penalties over a history tail, greedy arg-max
(one block per token, strided scan plus warp/block reduction), top-k by repeated maximum scans, top-p cut,
temperature, and a counter-based Philox uniform draw.

## Source semantics
* Greedy (`greedy` or `temperature <= 0`) is an arg-max over the PENALISED logits, ties to the lowest index.
  Temperature must not be applied there: `inv_t = 0` for `temperature <= 0` would zero every logit.
* Repeat penalty multiplies non-positive logits and divides positive ones; the presence penalty is a boolean
  (applied once), the frequency penalty scales with the count. Nothing happens for count 0.
* The history window is the TAIL of the row: the last `min(penalty_last_n, history_len)` entries.
* The Philox uniform keeps 24 mantissa bits, so it lies in [0, 1).
* `top_k` is clamped to at most 64 on the sampled path; `top_p` keeps at least `min_keep` survivors.

## Form template identifiers
`sample_tokens` -> `sample_rows`; `SamplerParams` -> `sampling_config`; `history` -> `token_history`;
`sampler_greedy_kernel` -> `argmax_kernel`; `philox_uniform` -> `counter_uniform`; `KMAX` -> `top_k_cap`.
-/

namespace Strata.Sampler

-- § penalties

/-- Penalty model over integers: repeat (multiply if non-positive, divide otherwise), then frequency and
presence. A zero count leaves the logit untouched. -/
def applyPenalty (logit : Int) (count : Nat) (rep freq present : Int) : Int :=
  if count = 0 then logit
  else (if logit ≤ 0 then logit * rep else logit / rep) - ((count : Int) * freq + 1 * present)

/-- A token that never occurred is not penalised. -/
theorem penalty_zero_count (logit rep freq present : Int) :
    applyPenalty logit 0 rep freq present = logit := by
  unfold applyPenalty
  simp

/-- The presence penalty is a boolean: with no frequency penalty the count does not matter. -/
theorem presence_is_boolean (logit rep present : Int) (c₁ c₂ : Nat) (h₁ : 0 < c₁) (h₂ : 0 < c₂) :
    applyPenalty logit c₁ rep 0 present = applyPenalty logit c₂ rep 0 present := by
  have n₁ : c₁ ≠ 0 := by omega
  have n₂ : c₂ ≠ 0 := by omega
  unfold applyPenalty
  rw [if_neg n₁, if_neg n₂]
  simp

/-- A negative logit is multiplied: with repeat penalty 2, -3 becomes -6 (less likely). -/
theorem penalty_negative_multiplies : applyPenalty (-3) 1 2 0 0 = -6 := by decide

/-- A positive logit is divided: with repeat penalty 2, 6 becomes 3. -/
theorem penalty_positive_divides : applyPenalty 6 1 2 0 0 = 3 := by decide

/-- Dividing a negative logit unconditionally would RAISE it (-3 / 2 = -2 > -6): the inversion the source avoids. -/
theorem naive_division_inverts : (-6 : Int) < (-3) / 2 := by decide

-- § history window

/-- Length of the penalty window: clamp of `penalty_last_n` into `[0, history_len]`. -/
def winLen (plen hl : Int) : Int := max 0 (min plen hl)

/-- The window length is non-negative. -/
theorem winLen_nonneg (plen hl : Int) : 0 ≤ winLen plen hl := by
  unfold winLen
  omega

/-- The window never exceeds the history. -/
theorem winLen_le (plen hl : Int) (h : 0 ≤ hl) : winLen plen hl ≤ hl := by
  unfold winLen
  omega

/-- The window is the tail: it starts at `hl - len` and ends exactly at the end of the row. -/
theorem winStart_tail (plen hl : Int) : (hl - winLen plen hl) + winLen plen hl = hl := by
  omega

-- § greedy arg-max with lowest-index tie rule

/-- Preference order: larger score, or equal score with smaller index. -/
abbrev Beats (v i w j : Nat) : Prop := v > w ∨ (v = w ∧ i < j)

/-- Asymmetry of the preference order. -/
theorem beats_asymm (v i w j : Nat) : Beats v i w j → ¬ Beats w j v i := by
  intro h1 h2
  unfold Beats at h1 h2
  omega

/-- Transitivity of the preference order. -/
theorem beats_trans (a i b j c k : Nat) : Beats a i b j → Beats b j c k → Beats a i c k := by
  intro h1 h2
  unfold Beats at h1 h2 ⊢
  omega

/-- Totality for distinct indices, which makes the strided scan plus reduction order-independent. -/
theorem beats_total (v w i j : Nat) (h : i ≠ j) : Beats v i w j ∨ Beats w j v i := by
  unfold Beats
  omega

/-- The "no candidate" sentinel (index `n_vocab`, score at most any real score) loses to every real index. -/
theorem sentinel_loses (v w i n : Nat) (hw : w ≤ v) (hi : i < n) : Beats v i w n := by
  unfold Beats
  omega

/-- The final answer is a valid index whenever the vocabulary is non-empty (`-inf` ties answer 0). -/
theorem final_index_lt (wi n : Nat) (hn : 0 < n) : (if wi < n then wi else 0) < n := by
  split <;> omega

/-- A 32-lane warp is folded by `shfl_down` offsets 16, 8, 4, 2, 1, which together span all 31 other lanes. -/
theorem warp_fold_span : 16 + 8 + 4 + 2 + 1 = 32 - 1 := by decide

/-- A 1024-thread block has 32 warps, so the second-stage warp reduction needs exactly one full warp. -/
theorem block_warps : (1024 + 31) / 32 = 32 := by decide

-- § temperature

/-- Scaling by zero collapses every logit to the same value: the bug the greedy branch had. -/
theorem zero_scale_collapses (a b : Int) : 0 * a = 0 * b := by
  simp

/-- Scaling by a positive constant preserves the order, so greedy never needs temperature. -/
theorem pos_scale_monotone (c a b : Int) (hc : 0 < c) : a < b ↔ c * a < c * b := by
  constructor
  · intro h
    exact Int.mul_lt_mul_of_pos_left h hc
  · intro h
    by_contra hn
    have h2 : b ≤ a := Int.not_lt.mp hn
    have h3 := Int.mul_le_mul_of_nonneg_left h2 (Int.le_of_lt hc)
    omega

-- § Philox uniform, top-k and top-p bookkeeping

/-- The top 24 bits of a 32-bit word give an integer below 2^24, so `u24 / 2^24 < 1`. -/
theorem philox_u24_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

/-- Effective `top_k` on the sampled path: zero disables it, otherwise at most 64. -/
def kClamp (topK : Nat) : Nat := if topK > 0 then min topK 64 else 0

/-- The clamped `top_k` never exceeds the 64-slot buffers. -/
theorem kClamp_le (topK : Nat) : kClamp topK ≤ 64 := by
  unfold kClamp
  split <;> omega

/-- The top-p cut after the `min_keep` correction. -/
def cutAdj (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The adjusted cut never exceeds the survivor count (given the raw cut does not). -/
theorem cutAdj_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : cutAdj cut minKeep nKeep ≤ nKeep := by
  unfold cutAdj
  split <;> omega

/-- When survivors exist, the adjusted cut keeps at least one of them for `min_keep >= 1`. -/
theorem cutAdj_pos (cut minKeep nKeep : Nat) (h1 : 1 ≤ minKeep) (h2 : 1 ≤ nKeep) (h3 : 1 ≤ cut) :
    1 ≤ cutAdj cut minKeep nKeep := by
  unfold cutAdj
  split <;> omega

-- § launch geometry

/-- The sampled-path grid of 64-thread blocks covers every token and wastes fewer than 64 threads. -/
theorem grid_covers (n : Nat) : n ≤ (n + 63) / 64 * 64 ∧ (n + 63) / 64 * 64 < n + 64 := by
  omega

end Strata.Sampler
