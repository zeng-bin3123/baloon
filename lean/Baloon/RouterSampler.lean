-- Baloon.RouterSampler: router argmax-combine, greedy output, Philox uniform draw
-- and repeat/frequency penalties.
-- Independent implementation proving the same mathematical facts as Ref.RouterSampler.

namespace Baloon.RouterSampler

/-- Combine two (value, index) candidates exactly as the shuffle reduction does. -/
def comb (a b : Int × Nat) : Int × Nat :=
  if (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) then b else a

theorem comb_pos (a b : Int × Nat) (h : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) : comb a b = b := by
  unfold comb; exact if_pos h

theorem comb_neg (a b : Int × Nat) (h : ¬ (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2))) : comb a b = a := by
  unfold comb; exact if_neg h

theorem comb_tie (v : Int) (i j : Nat) (h : i < j) :
    comb (v, j) (v, i) = (v, i) ∧ comb (v, i) (v, j) = (v, i) := by
  constructor
  · exact (comb_pos _ _ (Or.inr ⟨rfl, h⟩))
  · exact comb_neg _ _ (by dsimp only; omega)

theorem comb_assoc (a b c : Int × Nat) : comb (comb a b) c = comb a (comb b c) := by
  by_cases h1 : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)
  · rw [comb_pos a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · rw [comb_pos a c h3]
      · exfalso; omega
    · rw [comb_neg b c h2, comb_pos a b h1]
  · rw [comb_neg a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
    · rw [comb_neg b c h2, comb_neg a b h1]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · exfalso; omega
      · rw [comb_neg a c h3]

/-- The greedy output is always a valid token id. -/
def greedyOut (wi n : Nat) : Nat := (if wi < n then wi else 0)
theorem greedyOut_lt (wi n : Nat) (hn : 0 < n) : greedyOut wi n < n := by
  unfold greedyOut; split <;> omega

/-- The 24-bit mantissa draw is strictly below 2^24, so the uniform is below 1. -/
theorem philox_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by { omega }

/-- The repeat penalty: multiply non-positive logits, divide positive ones. -/
def repeatPen (l r : Int) : Int := (if l ≤ 0 then l * r else l / r)

-- ── thread sizing, shared memory and selection order ─────────────────────────

/-- Threads per token block: min(n_expert, 512) rounded up to a whole warp. -/
def thr (n : Nat) : Nat := ((min n 512 + 31) / 32 * 32)

/-- The thread count is a warp multiple, covers the experts and needs at most 16 warp slots. -/
theorem thr_props (n : Nat) (hn : 0 < n) :
    32 ≤ thr n ∧ thr n ≤ 512 ∧ thr n % 32 = 0 ∧ min n 512 ≤ thr n ∧ (thr n + 31) / 32 ≤ 16 := by
  unfold thr; omega

/-- Round up to 16 (the taken mask padding). -/
def pad16 (n : Nat) : Nat := ((n + 15) / 16 * 16)

theorem smem_alignment (n : Nat) : pad16 n % 16 = 0 ∧ (pad16 n + 8 * n) % 4 = 0 ∧ n ≤ pad16 n := by
  unfold pad16; omega

/-- Shared memory for the full 512-expert router. -/
theorem smem_512 : pad16 512 + 512 * 8 + 512 * 4 = 6656 := by { decide }

/-- The clamp keeps the renormalisation denominator positive. -/
theorem clamp_positive (s floor : Nat) (h : 0 < floor) : 0 < max s floor := by { omega }

/-- The router always selects the lowest index on a value tie. -/
theorem pass_takes_lowest_on_tie (v : Int) (i j : Nat) (h : i < j) : (comb (v, i) (v, j)).2 = i := by
  have := (comb_tie v i j h).2; rw [this]

end Baloon.RouterSampler
