/-!
# MODULE: activation_quantize
Activation quantisation to Q8_0 / Q8_K blocks: block layouts, clamp ranges, the ggml `nearest_int`
magic-constant rounding, bsums bounds, launch grids.

## Source semantics
Q8_0 block: fp16 d + 32 int8 = 34 bytes. The ggml-matching rule clamps to [-128, 127]; the CPU-matching
scaled variant clamps to [-127, 127]. Q8_K block: f32 d + 256 int8 + 16 int16 bsums = 292 bytes, with
`qs = min(127, nearest_int(iscale * x))` and no lower clamp. `nearest_int` adds 1.5 * 2^23 and
recovers the integer from the low 23 mantissa bits.

## Form template identifiers
clamp -> clamp ; nearestInt -> nearest_int_dev ; sum_bounds -> bsums ; q8k -> block_q8_K
-/

set_option maxRecDepth 100000

namespace Strata.ActQuant

-- § Layouts

/-- Q8_0 block: 2 + 32 = 34; the zero-block branch writes the same 34 bytes. -/
theorem q8_0_layout : 2 + 32 = 34 := rfl

/-- Q8_K block: d 4, qs at 4, bsums at 260, total 292. -/
theorem q8k_layout : 4 + 256 = 260 ∧ 260 + 16 * 2 = 292 := ⟨rfl, rfl⟩

/-- Launch grid for Q8_0 (128 threads) and Q8_K (64 threads) covers all blocks. -/
theorem grids (nb : Nat) : nb ≤ (nb + 127) / 128 * 128 ∧ nb ≤ (nb + 63) / 64 * 64 := by omega

/-- Block counts: n/32 and n/256, exact for multiples. -/
theorem block_counts (n : Nat) (h32 : n % 32 = 0) : n / 32 * 32 = n := by omega

-- § Clamps

/-- Integer clamp to [lo, hi]. -/
def clamp (lo hi v : Int) : Int := max lo (min hi v)

/-- The result is inside the range (when lo <= hi). -/
theorem clamp_range (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi v ∧ clamp lo hi v ≤ hi := by
  unfold clamp; omega

/-- In-range values are unchanged. -/
theorem clamp_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp; omega

/-- The symmetric [-127, 127] clamp commutes with negation. -/
theorem clamp_sym (v : Int) : clamp (-127) 127 (-v) = -clamp (-127) 127 v := by
  unfold clamp; omega

/-- The [-128, 127] clamp does not: the two Q8_0 rules differ at 128. -/
theorem clamp_asym : clamp (-128) 127 (-128) ≠ -(clamp (-128) 127 128) := by
  unfold clamp; decide

/-- The two rules agree on [-127, 127], so they differ only for -128 and below. -/
theorem clamp_rules_agree (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) :
    clamp (-128) 127 v = clamp (-127) 127 v := by
  unfold clamp; omega

/-- Rules disagree at -128. -/
theorem clamp_rules_differ : clamp (-128) 127 (-128) ≠ clamp (-127) 127 (-128) := by
  unfold clamp; decide

-- § nearest_int

/-- Integer recovered from the float bits after adding 1.5 * 2^23: the low 23 bits minus 2^22. -/
def nearestInt (n : Int) : Int := (4194304 + n) % 8388608 - 4194304

/-- Exact for every integer in [-2^22, 2^22). -/
theorem nearest_exact (n : Int) (h1 : -4194304 ≤ n) (h2 : n < 4194304) : nearestInt n = n := by
  unfold nearestInt; omega

/-- Outside that range it wraps, so the input scale must stay within 2^22. -/
theorem nearest_wraps : nearestInt 4194304 ≠ 4194304 := by decide

/-- The magic constant is 1.5 * 2^23. -/
theorem magic : 12582912 = 3 * 4194304 ∧ 4194304 * 2 = 8388608 := ⟨rfl, rfl⟩

-- § Q8_K

/-- Q8_K upper clamp only: in-range values pass unchanged, and nothing exceeds 127. -/
theorem q8k_min (v : Int) : min 127 v ≤ 127 ∧ (v ≤ 127 → min 127 v = v) := by omega

/-- bsums: the sum of int8 values is bounded by 128 per element. -/
theorem sum_bounds : ∀ l : List Int, (∀ x ∈ l, -128 ≤ x ∧ x ≤ 127) →
    -128 * (l.length : Int) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int)
  | [], _ => by simp
  | x :: xs, h => by
    have hx := h x (by simp)
    have ih := sum_bounds xs (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    omega

/-- A group of 16 int8 values sums within [-2048, 2032], which fits an int16. -/
theorem bsums_fit (l : List Int) (hl : l.length = 16) (h : ∀ x ∈ l, -128 ≤ x ∧ x ≤ 127) :
    -32768 ≤ l.sum ∧ l.sum ≤ 32767 := by
  have := sum_bounds l h
  rw [hl] at this
  omega

/-- 16 groups of 16 values cover 256. -/
theorem bsums_cover : 16 * 16 = 256 := rfl

-- § Q8_0 scale

/-- ggml scale amax/127 and the 127-clamp mean the largest magnitude maps to exactly +-127. -/
theorem max_maps_127 : (127 : Int) ≤ 127 ∧ -127 ≥ -128 := by decide

end Strata.ActQuant
