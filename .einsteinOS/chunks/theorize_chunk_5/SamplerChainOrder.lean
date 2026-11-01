/-!
# MODULE: sampler_chain_order
Argmax tie rule, penalty guard and uniform-range facts of the sampler (`sampler.cu`).

## Source semantics
Greedy picks the highest penalised logit, the LOWEST index winning ties; the warp/block reduction resolves
two candidates by larger value, then smaller index. Penalties only act when `count > 0`. The Philox draw
uses 24 mantissa bits so the uniform is strictly below 1.

## Form template identifiers
better -> reduction comparison; counts -> history_count; u24 -> `c0 >> 8`.
-/

namespace Strata.SamplerChainOrder

/-- Candidate `(v1,i1)` beats `(v2,i2)`: larger value, or equal value and smaller index. -/
def better (v1 : Int) (i1 : Nat) (v2 : Int) (i2 : Nat) : Prop :=
  v1 > v2 ∨ (v1 = v2 ∧ i1 < i2)

-- § total order

/-- Irreflexive. -/
theorem better_irrefl (v : Int) (i : Nat) : ¬ better v i v i := by
  unfold better
  omega

/-- Transitive, so the tree reduction is order-independent. -/
theorem better_trans (a : Int) (i : Nat) (b : Int) (j : Nat) (c : Int) (k : Nat)
    (h1 : better a i b j) (h2 : better b j c k) : better a i c k := by
  unfold better at *
  omega

/-- Distinct indices are always comparable, so a unique winner exists. -/
theorem better_total (a : Int) (i : Nat) (b : Int) (j : Nat) (h : i ≠ j) :
    better a i b j ∨ better b j a i := by
  unfold better
  omega

/-- A tie goes to the smaller index. -/
theorem tie_smaller_index (v : Int) (i j : Nat) (h : i < j) : better v i v j := by
  unfold better
  omega

-- § penalties

/-- With no occurrences the logit is unchanged (modelled by the guard). -/
def guardCount (count : Nat) (f : Int → Int) (logit : Int) : Int :=
  if count = 0 then logit else f logit

theorem no_history_no_penalty (f : Int → Int) (logit : Int) :
    guardCount 0 f logit = logit := by
  simp [guardCount]

/-- The presence term is boolean: it is the same for any positive count. -/
def presence (count : Nat) : Nat := if count > 0 then 1 else 0

theorem presence_boolean (a b : Nat) (ha : 0 < a) (hb : 0 < b) : presence a = presence b := by
  unfold presence
  rw [if_pos ha, if_pos hb]

-- § uniform draw

/-- `c0 >> 8` fits in 24 bits for any 32-bit word, so `u < 1`. -/
theorem u24_lt (c0 : Nat) (h : c0 < 2 ^ 32) : c0 / 256 < 2 ^ 24 := by
  omega

/-- Temperature is irrelevant to the greedy argmax (positive scaling preserves order). -/
theorem positive_scale_preserves (s : Int) (a b : Int) (hs : 0 < s) (h : a > b) : s * a > s * b := by
  exact Int.mul_lt_mul_of_pos_left h hs

end Strata.SamplerChainOrder
