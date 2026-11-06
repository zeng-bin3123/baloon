/-!
# MODULE: sampler_chain
SOURCES: src/kernels/cuda/sampler.cu
GPU sampler: penalties (repeat/frequency/presence over the history tail), top-k by repeated max scans,
top-p over survivors, temperature, Philox-driven draw; a parallel greedy argmax with lowest-index tie rule.

## Source semantics
* Penalty applies only to tokens with count > 0: repeat multiplies non-positive logits and divides positive
  ones; then subtract count*freq + (count > 0 ? 1 : 0)*presence (a boolean, not the count).
* The window is the TAIL of the history: hlen = clamp(penalty_last_n, 0, history_len).
* Greedy = argmax over penalised logits; ties go to the lowest index; the block reduction preserves that order.
* Sampled path needs 1 <= top_k <= 64; top_p cut >= min_keep (capped at the survivor count).
* Philox draw is uniform in [0,1): the top 24 bits of a 32-bit word divided by 2^24.

## Form template identifiers
apply_penalties -> penalty_fn ; sampler_greedy_kernel -> parallel_argmax ; philox_uniform -> counter_rng ;
KMAX -> topk_cap ; history_count -> token_count
-/

namespace Strata.Sampler

-- § penalties

/-- Presence indicator: 1 for any positive count. -/
def presence (count : Nat) : Nat := if count > 0 then 1 else 0

/-- The presence term is a boolean. -/
theorem presence_le_one (c : Nat) : presence c ≤ 1 := by
  unfold presence
  split <;> omega

/-- Presence does not depend on how many times the token occurred. -/
theorem presence_const (c : Nat) (h : 0 < c) : presence c = 1 := by
  unfold presence
  simp [h]

/-- Total subtraction for a token seen `count` times. -/
def penaltySub (count freq present : Nat) : Nat := count * freq + presence count * present

/-- Unseen tokens subtract nothing. -/
theorem penaltySub_zero (freq present : Nat) : penaltySub 0 freq present = 0 := by
  simp [penaltySub, presence]

/-- Seen tokens subtract count*freq + present (present counted once). -/
theorem penaltySub_seen (c freq present : Nat) (h : 0 < c) :
    penaltySub c freq present = c * freq + present := by
  unfold penaltySub
  rw [presence_const c h]
  omega

-- § history window

/-- Window length. -/
def hlen (lastN hist : Nat) : Nat := min lastN hist

/-- Window start offset in the history row (the tail). -/
def hstart (lastN hist : Nat) : Nat := hist - hlen lastN hist

/-- The window is a suffix: start + length = history_len. -/
theorem window_is_suffix (lastN hist : Nat) : hstart lastN hist + hlen lastN hist = hist := by
  unfold hstart hlen
  omega

/-- The window never exceeds the history. -/
theorem window_le (lastN hist : Nat) : hlen lastN hist ≤ hist := by
  unfold hlen
  omega

-- § argmax with the lowest-index tie rule

/-- Candidate (value, index) beats the current best: larger value, or equal value and smaller index. -/
def combine (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The reduction operator is commutative on candidates with distinct indices. -/
theorem combine_comm (a b : Int × Nat) (h : a.2 ≠ b.2) : combine a b = combine b a := by
  have key : ¬ ((b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) ∧ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2))) := by
    omega
  have key2 : (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) ∨ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)) := by
    omega
  by_cases h1 : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)
  · have h2 : ¬ (a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)) := fun h' => key ⟨h1, h'⟩
    unfold combine
    rw [if_pos h1, if_neg h2]
  · have h2 := key2.resolve_left h1
    unfold combine
    rw [if_neg h1, if_pos h2]

/-- Equal values: the lower index wins (second candidate has the lower index). -/
theorem combine_tie_low (v : Int) (i j : Nat) (h : j < i) : combine (v, i) (v, j) = (v, j) := by
  unfold combine
  rw [if_pos]
  simp
  omega

/-- Equal values: the lower index wins (first candidate has the lower index). -/
theorem combine_tie_keep (v : Int) (i j : Nat) (h : i ≤ j) : combine (v, i) (v, j) = (v, i) := by
  unfold combine
  rw [if_neg]
  intro hc
  simp at hc
  omega

/-- A strictly larger value always wins regardless of index. -/
theorem combine_greater (a b : Int × Nat) (h : b.1 > a.1) : combine a b = b := by
  unfold combine
  rw [if_pos (Or.inl h)]

-- § top-k / top-p bounds

/-- top-k clamp: the number of passes is at most KMAX = 64. -/
def kPasses (topK : Nat) : Nat := if topK > 0 then min topK 64 else 0

/-- The clamp is bounded by 64. -/
theorem kPasses_le (topK : Nat) : kPasses topK ≤ 64 := by
  unfold kPasses
  split <;> omega

/-- top-p cut after the min_keep floor. -/
def cutAfter (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The cut never exceeds the survivor count (given the cumulative scan bound). -/
theorem cutAfter_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : cutAfter cut minKeep nKeep ≤ nKeep := by
  unfold cutAfter
  split <;> omega

/-- The cut keeps at least min_keep survivors when available and at least one when cut >= 1. -/
theorem cutAfter_ge (cut minKeep nKeep : Nat) (h : minKeep ≤ nKeep) : min minKeep cut ≤ cutAfter cut minKeep nKeep := by
  unfold cutAfter
  split <;> omega

-- § RNG and launch

/-- 24-bit mantissa draw is strictly below 2^24, hence the uniform is below 1. -/
theorem philox_mantissa_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by
  omega

/-- Sampled-path grid (64 threads, one per token) covers all tokens. -/
theorem sample_grid_covers (n : Nat) : n ≤ 64 * ((n + 63) / 64) := by
  omega

/-- Warp count of a block of 1024 threads fits the 32-slot shared reduction arrays. -/
theorem greedy_warps_fit : (1024 + 31) / 32 ≤ 32 := by
  decide

end Strata.Sampler
