/-!
# MODULE: token_sampler_chain
GPU sampler: repetition/frequency/presence penalties, greedy argmax, top-k, top-p, temperature and a
counter-based Philox draw.

## Source semantics
Chain order: penalties, top_k, top_p, temperature, pick. Greedy is the argmax of the PENALISED logits with
ties resolved to the LOWEST index; temperature must not be applied (inverse temperature is 0 for
temperature <= 0). The penalty window is the tail of the history. Penalties do nothing for count 0. The
Philox uniform uses 24 mantissa bits so it is in [0, 1). The sampled path needs `top_k` in 1..64.

## Form template identifiers
- `apply_penalties` -> applyPenalties, `sampler_greedy_kernel` tie rule -> better
- `philox_uniform` -> uniform bits, `KMAX` -> 64
-/

namespace Strata.Sampler

-- § Philox

/-- 24-bit mantissa extraction from a 32-bit word. -/
def mantissa24 (c0 : Nat) : Nat := c0 / 256

/-- The mantissa is below 2^24, so the uniform `m / 2^24` is strictly below 1. -/
theorem mantissa24_lt (c0 : Nat) (h : c0 < 4294967296) : mantissa24 c0 < 16777216 := by
  unfold mantissa24
  omega

/-- 24 bits fit exactly in a float mantissa: 2^24 = 16777216. -/
theorem two_pow_24 : (16777216 : Nat) = 2 ^ 24 := by decide

-- § greedy argmax tie rule

/-- Candidate 2 beats candidate 1: larger value, or equal value with smaller index. -/
def better (v1 : Int) (i1 : Nat) (v2 : Int) (i2 : Nat) : Prop := v1 < v2 ∨ (v1 = v2 ∧ i2 < i1)

/-- Irreflexive: a candidate never beats itself. -/
theorem better_irrefl (v : Int) (i : Nat) : ¬ better v i v i := by
  unfold better
  omega

/-- Asymmetric: two candidates never beat each other. -/
theorem better_asymm (v1 v2 : Int) (i1 i2 : Nat) (h : better v1 i1 v2 i2) : ¬ better v2 i2 v1 i1 := by
  unfold better at *
  omega

/-- Total on distinct indices: one of the two candidates wins. -/
theorem better_total (v1 v2 : Int) (i1 i2 : Nat) (hne : i1 ≠ i2) :
    better v1 i1 v2 i2 ∨ better v2 i2 v1 i1 := by
  unfold better
  omega

/-- Transitive, so the warp reduction order does not affect the winner. -/
theorem better_trans (v1 v2 v3 : Int) (i1 i2 i3 : Nat) (h12 : better v1 i1 v2 i2)
    (h23 : better v2 i2 v3 i3) : better v1 i1 v3 i3 := by
  unfold better at *
  omega

-- § penalties

/-- Integer model of the additive part of the penalty step. -/
def additivePenalty (count : Nat) (freq present : Int) : Int :=
  (count : Int) * freq + (if 0 < count then 1 else 0) * present

/-- Tokens never seen get no additive penalty. -/
theorem additive_zero (freq present : Int) : additivePenalty 0 freq present = 0 := by
  simp [additivePenalty]

/-- The presence penalty is boolean: it applies once however many times the token appeared. -/
theorem presence_is_boolean (n : Nat) (freq present : Int) (h : 0 < n) :
    additivePenalty n freq present = (n : Int) * freq + present := by
  simp [additivePenalty, h]

/-- Penalty window: the tail of at most `history_len` entries. -/
def windowLen (lastN histLen : Nat) : Nat := min lastN histLen

/-- The tail window starts inside the history and ends at its end. -/
theorem window_tail (lastN histLen : Nat) : (histLen - windowLen lastN histLen) + windowLen lastN histLen = histLen := by
  unfold windowLen
  omega

-- § temperature

/-- Inverse temperature model: zero for non-positive temperature. -/
def invT (t : Int) : Int := if 0 < t then 1000000 / t else 0

/-- At temperature <= 0 the inverse is 0, which is why scaling logits there erased the argmax. -/
theorem invT_nonpos (t : Int) (h : t ≤ 0) : invT t = 0 := by
  unfold invT
  rw [if_neg (by omega)]

/-- A zero inverse temperature flattens every logit. -/
theorem scaled_flat (l : Int) : l * 0 = 0 := by omega

-- § top-k and top-p

/-- Effective k: 0 means no top-k, else capped at 64. -/
def kEff (topK : Int) : Int := if topK ≤ 0 then 0 else min topK 64

/-- The effective k never exceeds 64. -/
theorem kEff_le (topK : Int) : kEff topK ≤ 64 := by
  unfold kEff
  split <;> omega

/-- The effective k is non-negative. -/
theorem kEff_nonneg (topK : Int) : 0 ≤ kEff topK := by
  unfold kEff
  split <;> omega

/-- Cut after top-p is at least min_keep (capped by the survivors) and at most the survivors. -/
def cutAdjust (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The cut never exceeds the survivor count when the unadjusted cut does not. -/
theorem cutAdjust_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : cutAdjust cut minKeep nKeep ≤ nKeep := by
  unfold cutAdjust
  split <;> omega

-- § launch geometry

/-- Greedy blocks have 1024 threads, i.e. 32 warps, matching the 32-entry shared arrays. -/
theorem greedy_warps : (1024 + 31) / 32 = 32 ∧ (1024 + 31) / 32 ≤ 32 := by decide

/-- The sampled-path grid covers every token with 64 threads per block. -/
theorem token_grid_cover (n : Nat) : n ≤ ((n + 63) / 64) * 64 := by omega

end Strata.Sampler
