/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
GPU sampler: penalties, top-k, top-p, temperature, then one categorical draw from a counter-based Philox
stream; greedy path is a block-wide argmax with lowest-index tie-breaking.

## Source semantics
- Repeat penalty multiplies non-positive logits and divides positive ones; frequency penalty scales with the
  count, presence penalty is a boolean; none of them can increase a logit.
- Greedy is the argmax of the penalised logits; ties go to the lowest index; temperature must not be applied
  (at temperature <= 0 the inverse temperature is 0 and every scaled logit would be 0).
- Penalty window is the tail of the history row of length min(penalty_last_n, history_len).
- Philox uniform: top 24 bits of a 32-bit word divided by 2^24, so the value is in [0,1).
- Sampled path needs 1 <= top_k <= 64.

## Form template identifiers
- sampler_greedy_kernel -> argmax reduction ; apply_penalties -> penalise ; philox_uniform -> uniform24
-/

namespace Strata.Sampler

-- § argmax reduction
/-- Candidate (value, index) pair. -/
abbrev Cand := Int × Nat

/-- Reduction step: replace `a` by `b` iff `b` has a larger value, or an equal value and a smaller index. -/
def better (a b : Cand) : Cand :=
  if a.1 < b.1 ∨ (a.1 = b.1 ∧ b.2 < a.2) then b else a

/-- The reduction result is at least as large as both operands. -/
theorem better_value_ge (a b : Cand) : a.1 ≤ (better a b).1 ∧ b.1 ≤ (better a b).1 := by
  unfold better
  split <;> omega

/-- On equal values the smaller index wins. -/
theorem better_tie_min (a b : Cand) (h : a.1 = b.1) : (better a b).2 = min a.2 b.2 := by
  unfold better
  split <;> omega

/-- The result is always one of the operands. -/
theorem better_is_operand (a b : Cand) : better a b = a ∨ better a b = b := by
  unfold better
  split
  · exact Or.inr rfl
  · exact Or.inl rfl

/-- The order is symmetric on distinct indices: the reduction does not depend on operand order. -/
theorem better_comm (a b : Cand) (h : a.2 ≠ b.2) : (better a b).2 = (better b a).2 := by
  unfold better
  split <;> split <;> omega

/-- A candidate with no elements (index = n_vocab) loses ties against real indices. -/
theorem sentinel_loses (v : Int) (i n : Nat) (h : i < n) : (better (v, n) (v, i)).2 = i := by
  unfold better
  split
  · rfl
  · rename_i hc
    exact absurd (Or.inr ⟨rfl, h⟩) hc

-- § penalties
/-- Penalised logit: repeat (multiply if non-positive, divide if positive), then subtract freq/presence. -/
def penalise (logit : Int) (count : Int) (rep freq pres : Int) : Int :=
  if count ≤ 0 then logit
  else
    let l := if logit ≤ 0 then logit * rep else logit / rep
    l - (count * freq + pres)

/-- Multiplying a non-positive logit by `rep ≥ 1` does not increase it. -/
theorem mul_branch_le (logit rep : Int) (h0 : logit ≤ 0) (h1 : 1 ≤ rep) : logit * rep ≤ logit := by
  have h := Int.mul_nonneg (by omega : 0 ≤ -logit) (by omega : 0 ≤ rep - 1)
  simp only [Int.neg_mul, Int.mul_sub, Int.mul_one] at h
  omega

/-- Dividing a positive logit by `rep ≥ 1` does not increase it. -/
theorem div_branch_le (logit rep : Int) (h0 : 0 < logit) (h1 : 1 ≤ rep) : logit / rep ≤ logit := by
  exact Int.ediv_le_self rep (by omega)

/-- No penalty combination with `rep ≥ 1`, `freq ≥ 0`, `pres ≥ 0` can raise a logit. -/
theorem penalise_le (logit count rep freq pres : Int) (hr : 1 ≤ rep) (hf : 0 ≤ freq) (hp : 0 ≤ pres) :
    penalise logit count rep freq pres ≤ logit := by
  unfold penalise
  by_cases hc : count ≤ 0
  · simp [hc]
  · simp only [hc, if_false]
    have hcf : 0 ≤ count * freq := Int.mul_nonneg (by omega) hf
    by_cases hl : logit ≤ 0
    · have := mul_branch_le logit rep hl hr
      simp only [hl, if_true]
      omega
    · have := div_branch_le logit rep (by omega) hr
      simp only [hl, if_false]
      omega

/-- With no occurrences the logit is untouched. -/
theorem penalise_zero_count (logit rep freq pres : Int) : penalise logit 0 rep freq pres = logit := by
  simp [penalise]

/-- Scaling by an inverse temperature of 0 collapses every logit to 0 (the old greedy bug). -/
theorem zero_inv_temp_collapses (logit : Int) : logit * 0 = 0 := by
  omega

-- § penalty window
/-- Window length: `penalty_last_n` clamped to `[0, history_len]`. -/
def windowLen (lastN histLen : Int) : Int := max 0 (min lastN histLen)

/-- The window never exceeds the history length. -/
theorem window_le_history (lastN histLen : Int) (h : 0 ≤ histLen) : windowLen lastN histLen ≤ histLen := by
  unfold windowLen
  omega

/-- The window is non-negative. -/
theorem window_nonneg (lastN histLen : Int) : 0 ≤ windowLen lastN histLen := by
  unfold windowLen
  omega

/-- The tail starts at `histLen - window`, and start plus window is the history length. -/
theorem window_tail_start (lastN histLen : Int) (h : 0 ≤ histLen) :
    (histLen - windowLen lastN histLen) + windowLen lastN histLen = histLen := by
  omega

-- § Philox
/-- Top 24 bits of a 32-bit word. -/
def top24 (w : Nat) : Nat := w / 256

/-- The 24-bit mantissa is below 2^24 for 32-bit input, so `top24 / 2^24 < 1`. -/
theorem top24_lt (w : Nat) (h : w < 4294967296) : top24 w < 16777216 := by
  unfold top24
  omega

-- § top-k
/-- Effective k: zero for non-positive request, otherwise clamped to 64. -/
def kEff (topK : Int) : Int := if topK ≤ 0 then 0 else min topK 64

/-- Effective k never exceeds the 64-slot arrays. -/
theorem kEff_le (topK : Int) : kEff topK ≤ 64 := by
  unfold kEff
  split <;> omega

/-- Effective k is non-negative. -/
theorem kEff_nonneg (topK : Int) : 0 ≤ kEff topK := by
  unfold kEff
  split <;> omega

/-- A valid sampled-path request gets k between 1 and 64. -/
theorem kEff_valid (topK : Int) (h : 1 ≤ topK) : 1 ≤ kEff topK ∧ kEff topK ≤ 64 := by
  unfold kEff
  split <;> omega

-- § launch geometry
/-- Greedy kernel: 1024 threads give 32 warps, matching the 32-slot shared arrays. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

/-- Any block of at most 1024 threads has at most 32 warps. -/
theorem warps_le_32 (b : Nat) (h : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by
  omega

/-- Sampled kernel: 64 threads per block, grid covers all tokens. -/
theorem sampler_grid_covers (n : Nat) : n ≤ ((n + 63) / 64) * 64 := by
  omega

end Strata.Sampler
