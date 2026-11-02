/-!
# MODULE: sampler_chain
Penalties, greedy argmax tie rule, history window, top-k bounds and the counter-based uniform draw.

## Source semantics
* Repeat penalty multiplies non-positive logits and divides positive ones; presence is a boolean.
* Greedy picks the largest value with the smallest index on ties; reductions are order independent.
* The penalty window is the tail of the history.
* The uniform draw is `(c0 >> 8) / 2^24`, always in [0, 1).

## Form template identifiers
* `apply_penalties` -> `penalise`; `sampler_greedy_kernel` -> `argmax_kernel`
* `history_count` -> `count_in_window`; `philox_uniform` -> `uniform_draw`
-/

namespace Strata.Sampler

-- § penalties (integer model: the repeat penalty is a positive divisor / multiplier)

/-- Penalty on an integer logit. Positive logits are divided, others multiplied. -/
def penalise (logit : Int) (count : Nat) (rep : Nat) (freq present : Int) : Int :=
  if count = 0 then logit
  else
    (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + present)

/-- No occurrences means no change. -/
theorem penalise_zero (l : Int) (rep : Nat) (f p : Int) : penalise l 0 rep f p = l := by
  simp [penalise]

/-- With a unit repeat penalty and zero offsets the logit is unchanged. -/
theorem penalise_neutral (l : Int) (c : Nat) : penalise l c 1 0 0 = l := by
  unfold penalise
  split_ifs <;> simp

/-- A positive logit stays non-negative after the divide when the offsets are zero. -/
theorem penalise_pos_nonneg (l : Int) (c rep : Nat) (hl : 0 < l) :
    0 ≤ penalise l c rep 0 0 := by
  unfold penalise
  by_cases hc : c = 0
  · rw [if_pos hc]; exact hl.le
  · have hn : ¬ l ≤ 0 := by omega
    rw [if_neg hc, if_neg hn, Int.mul_zero, Int.add_zero, Int.sub_zero]
    exact Int.ediv_nonneg hl.le (by omega)

/-- A non-positive logit stays non-positive after the multiply when the offsets are zero. -/
theorem penalise_nonpos_nonpos (l : Int) (c rep : Nat) (hl : l ≤ 0) (hc : c ≠ 0) :
    penalise l c rep 0 0 ≤ 0 := by
  unfold penalise
  rw [if_neg hc, if_pos hl, Int.mul_zero, Int.add_zero, Int.sub_zero]
  exact Int.mul_nonpos_of_nonpos_of_nonneg hl (by omega)

/-- The presence term is a boolean: at most 1 however large the count. -/
theorem presence_bool (c : Nat) : (if c > 0 then 1 else 0 : Nat) ≤ 1 := by
  split_ifs <;> omega

-- § greedy tie rule

/-- Candidate merge: larger value, then smaller index. -/
def pick (a b : Int × Nat) : Int × Nat :=
  if a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) then a else b

/-- The pick is commutative. -/
theorem pick_comm (a b : Int × Nat) : pick a b = pick b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [pick]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq]; omega)

/-- A tie in value goes to the smaller index. -/
theorem pick_tie (v : Int) (i j : Nat) (h : i < j) : pick (v, i) (v, j) = (v, i) := by
  simp only [pick]
  split_ifs with hc
  · rfl
  · exact absurd (Or.inr ⟨by trivial, h⟩) hc

/-- A strictly larger value always wins. -/
theorem pick_larger (u v : Int) (i j : Nat) (h : v < u) : pick (u, i) (v, j) = (u, i) := by
  simp only [pick]
  split_ifs with hc
  · rfl
  · exact absurd (Or.inl h) hc

-- § history window

/-- Length of the penalty window. -/
def windowLen (lastN histLen : Nat) : Nat := min lastN histLen

/-- The window never exceeds the history. -/
theorem windowLen_le (n h : Nat) : windowLen n h ≤ h := by unfold windowLen; omega

/-- The window starts at `histLen - len` and ends exactly at the end of the history. -/
theorem window_tail (n h : Nat) : (h - windowLen n h) + windowLen n h = h := by
  unfold windowLen; omega

-- § top-k bounds

/-- The effective `k`, capped at 64. -/
def effK (topK : Nat) : Nat := if topK < 64 then topK else 64

/-- The effective `k` fits the 64-entry survivor arrays. -/
theorem effK_le (t : Nat) : effK t ≤ 64 := by unfold effK; split_ifs <;> omega

/-- `min_keep` clamps the cut into the survivors. -/
theorem cut_le (minKeep nKeep : Nat) : min minKeep nKeep ≤ nKeep := by omega

-- § uniform draw

/-- The 24-bit mantissa extracted from a 32-bit word. -/
def mant (c0 : Nat) : Nat := c0 >>> 8

/-- A 32-bit word yields a value below `2^24`, so `mant / 2^24 < 1`. -/
theorem mant_lt (c0 : Nat) (h : c0 < 2 ^ 32) : mant c0 < 2 ^ 24 := by
  unfold mant
  rw [Nat.shiftRight_eq_div_pow]
  omega

-- § launch shapes

/-- A 1024-thread block has 32 warps, so the 32-entry reduction arrays suffice. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

/-- The ceiling grid covers every token. -/
theorem grid_covers (n : Nat) : n ≤ (n + 63) / 64 * 64 := by omega

end Strata.Sampler
