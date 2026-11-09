/-!
# MODULE: activation_quant_contract
SOURCES: include/strata/kernels/cpu/expert.hpp, src/kernels/cpu/expert.cpp, src/kernels/cpu/q2_avx2.cpp
Integer-level contract of `act_quant_q8_1` / `act_quant_q8_1_avx2`: per-32-element chunk scale amax/127,
codes clamped to [-127,127], round half AWAY from zero (not half-to-even), planar `ActQ` storage.

## Source semantics
* Each code is clamped to [-127,127]; the chunk sum therefore lies in [-127*32, 127*32] and fits int32.
* Rounding is half away from zero (odd-symmetric); nearest-even would differ on ties.
* n must be a multiple of QKA=32, giving n/32 chunks, at most MAXC=80.
* hx = scale * sum is zero when the chunk scale is zero (all-zero chunk).

## Form template identifiers
clamp127 -> code_clamp, rha -> round_half_away, chunk_sum_bound -> chunk_sum_range
-/

namespace Strata.ActQuant

-- § clamp
/-- Clamp to the symmetric int8 range used by the quantizer. -/
def clamp127 (v : Int) : Int := max (-127) (min 127 v)

/-- Clamped values lie in [-127,127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- Clamping an in-range value is the identity. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v :=
  clamp127_id _ (clamp127_bounds v).1 (clamp127_bounds v).2

/-- Clamping is monotone. -/
theorem clamp127_mono (a b : Int) (h : a ≤ b) : clamp127 a ≤ clamp127 b := by
  unfold clamp127
  omega

/-- Clamping is odd-symmetric. -/
theorem clamp127_neg (v : Int) : clamp127 (-v) = - clamp127 v := by
  unfold clamp127
  omega

-- § chunk sums
/-- A list of in-range codes has a sum bounded by 127 times its length. -/
theorem chunk_sum_bound : ∀ l : List Int, (∀ v ∈ l, -127 ≤ v ∧ v ≤ 127) →
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: l, h => by
      have hx := h x (by simp)
      have ih := chunk_sum_bound l (fun v hv => h v (by simp [hv]))
      simp only [List.sum_cons, List.length_cons]
      omega

/-- A 32-code chunk sum is at most 4064 in magnitude and fits int32. -/
theorem chunk32_sum_fits : 32 * 127 = 4064 ∧ 4064 < 2 ^ 31 := by decide

-- § chunk geometry
/-- With n a multiple of 32, the chunks tile n exactly. -/
theorem nchunks_mul (n : Nat) (h : n % 32 = 0) : (n / 32) * 32 = n := by omega

/-- Widest reduction (2560) has 80 chunks; any n up to 2560 has at most 80. -/
theorem nchunks_le_maxc (n : Nat) (hn : n ≤ 2560) : n / 32 ≤ 80 := by omega

/-- Weight block b covers activation chunks 2b and 2b+1, at byte offsets 64b and 64b+32. -/
theorem weight_block_chunks (b : Nat) : b * 64 = (2 * b) * 32 ∧ b * 64 + 32 = (2 * b + 1) * 32 := by
  omega

/-- The second chunk of the last gate/up block still lies inside the 2560-wide `q` array. -/
theorem last_chunk_in_q (b : Nat) (h : b < 40) : (2 * b + 1) * 32 + 32 ≤ 2560 := by omega

-- § rounding
/-- Round n/d (d>0) half away from zero, via nonnegative integer division. -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((2 * (-n) + d) / (2 * d))

/-- Half-away rounding is odd-symmetric for positive numerators. -/
theorem rha_odd (n d : Int) (h : 0 < n) : rha (-n) d = - rha n d := by
  have h1 : ¬ (0 ≤ -n) := by omega
  have h2 : 0 ≤ n := by omega
  unfold rha
  rw [if_neg h1, if_pos h2, Int.neg_neg]

/-- A tie 1/2 rounds up to 1. -/
theorem rha_tie_pos : rha 1 2 = 1 := by decide

/-- A tie -1/2 rounds down to -1. -/
theorem rha_tie_neg : rha (-1) 2 = -1 := by decide

/-- Round-to-nearest-even would send the tie 1/2 to 0, so the two contracts differ. -/
theorem tie_differs_from_even : (0 : Int) ≠ rha 1 2 := by decide

-- § correction term
/-- hx = scale * sum vanishes for a zero scale. -/
theorem hx_zero_of_zero_scale (s sm : Int) (h : s = 0) : s * sm = 0 := by
  subst h
  exact Int.zero_mul sm

end Strata.ActQuant
