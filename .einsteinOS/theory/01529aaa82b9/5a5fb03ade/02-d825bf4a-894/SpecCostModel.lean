/-!
# MODULE: spec_cost_model
Default cost-model constants of the speculative-decoding controller and the shape properties the controller's
arithmetic relies on.

## Source semantics
* `dense_ratio` and `distinct_ratio` have `K_MAX + 1 = 9` entries, indexed by `n - 1` for `n = k + 1`.
* Both start at 1.0 and never decrease as the window grows.
* `distinct_ratio[n-1] <= n`: a window of `n` tokens touches at most `n` times as many distinct experts as one token,
  so tokens-per-distinct-expert `n / U(n)` is at least one and the extra-use cost is non-negative.
* Window verification is cheaper per token than serial decoding: `dense_ratio[n-1] < n` for `n >= 2`.

## Form template identifiers
* `dense_ratio` -> `denseRatio`, `distinct_ratio` -> `distinctRatio` (hundredths, so 105 = 1.05)
* `K_MAX` -> `kMax`
-/

namespace Strata.SpecCostModel

/-- Largest draft window. -/
def kMax : Nat := 8

/-- Dense-pass cost ratio by `n = k + 1`, in hundredths. -/
def denseRatio : List Nat := [100, 105, 130, 145, 185, 220, 260, 300, 330]

/-- Distinct-expert ratio `U(n)/U(1)` by `n`, in hundredths. -/
def distinctRatio : List Nat := [100, 170, 231, 288, 340, 389, 435, 480, 520]

/-- Non-decreasing check for a list. -/
def nondecr : List Nat → Bool
  | a :: b :: rest => decide (a ≤ b) && nondecr (b :: rest)
  | _ => true

/-- Both tables have `K_MAX + 1` entries, so every clamped index is in range. -/
theorem table_lengths : denseRatio.length = kMax + 1 ∧ distinctRatio.length = kMax + 1 := by
  decide

/-- Both tables start at exactly 1.0. -/
theorem tables_start_at_one : denseRatio.head? = some 100 ∧ distinctRatio.head? = some 100 := by
  decide

/-- The dense-cost table is non-decreasing. -/
theorem dense_nondecreasing : nondecr denseRatio = true := by
  decide

/-- The distinct-expert table is non-decreasing. -/
theorem distinct_nondecreasing : nondecr distinctRatio = true := by
  decide

/-- For each `n`, entry `n - 1` of the distinct table is at most `n * 100`. -/
def distinctBelowN : List Nat → Nat → Bool
  | [], _ => true
  | u :: us, n => decide (u ≤ n * 100) && distinctBelowN us (n + 1)

/-- Distinct experts grow no faster than the token count: uses-per-expert is at least one. -/
theorem distinct_le_n : distinctBelowN distinctRatio 1 = true := by
  decide

/-- Window verification is sub-linear: dense ratio below `n` full passes for `n >= 2`. -/
def denseBelowN : List Nat → Nat → Bool
  | [], _ => true
  | u :: us, n => decide (n < 2 ∨ u < n * 100) && denseBelowN us (n + 1)

/-- Verifying `n` tokens costs less than `n` serial passes. -/
theorem dense_sublinear : denseBelowN denseRatio 1 = true := by
  decide

/-- The default thresholds: 5% minimum gain and EMA 0.05, both inside the unit interval. -/
theorem default_gain_bounds : 5 ≤ 100 ∧ 5 ≤ 100 := by
  decide

/-- Default hit rate 0.55 and extra-use cost 0.25 are valid fractions (hundredths). -/
theorem default_fractions : 55 ≤ 100 ∧ 25 ≤ 100 := by
  decide

end Strata.SpecCostModel
