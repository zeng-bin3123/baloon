-- Baloon.MmvqGeometry: MMVQ (mixed-precision matrix-vector quantised) kernel geometry:
-- threads-per-block coverage, small-K thresholds, grid layouts and multi-column strides.
-- Independent implementation proving the same mathematical facts as Ref.MmvqGeometry.

namespace Baloon.MmvqGeometry

/-- (T threads per block, BPI blocks per iteration) for Q5_K, Q2_0, Q3_K, Q6_K, IQ4_XS, Q4_0/Q5_0/IQ4_NL, Q8_0. -/
def traits : List (Nat × Nat) := [(16, 8), (2, 64), (16, 8), (32, 4), (8, 16), (2, 64), (4, 32)]

/-- Every format assigns all 128 threads exactly once per iteration. -/
theorem threads_cover_block : ∀ p ∈ traits, p.1 * p.2 = 128 := by decide

/-- The BPI formulas of the source. -/
theorem bpi_values :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧
      4 * 4 * 32 / 32 = 16 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Thread tid < T*BPI handles a block index below BPI. -/
theorem thread_block_lt (T BPI tid : Nat) (h : tid < T * BPI) : tid / T < BPI :=
  Nat.div_lt_of_lt_mul h

/-- The (block, slot) pair of a thread determines the thread. -/
theorem thread_decomp (T tid : Nat) : T * (tid / T) + tid % T = tid := Nat.div_add_mod tid T

theorem smallk_4 (n : Nat) (h : n % 32 = 0) : n / 32 < 64 ↔ n < 2048 := by omega
theorem smallk_q8_0 (n : Nat) (h : n % 32 = 0) : n / 32 < 32 ↔ n < 1024 := by omega
theorem smallk_k (n : Nat) (h : n % 256 = 0) : n / 256 < 8 ↔ n < 2048 := by omega
theorem smallk_q2_0 (n : Nat) (h : n % 64 = 0) : n / 64 < 64 ↔ n < 4096 := by omega
theorem smallk_iq4xs (n : Nat) (h : n % 256 = 0) : n / 256 < 16 ↔ n < 4096 := by omega
theorem smallk_q6k (n : Nat) (h : n % 256 = 0) : n / 256 < 4 ↔ n < 1024 := by omega

theorem grid_rows4 (n : Nat) : n ≤ ((n + 3) / 4) * 4 := by omega
theorem grid_rows2 (n : Nat) : n ≤ ((n + 1) / 2) * 2 := by omega
theorem two_warp_traits : ∀ p ∈ [(16, 4), (2, 32), (32, 2), (8, 8), (4, 16)], p.1 * p.2 = 64 := by decide
theorem two_warp_bpi : 8 * 2 / 4 = 4 ∧ 64 * 2 / 4 = 32 ∧ 4 * 2 / 4 = 2 ∧ 16 * 2 / 4 = 8 ∧ 32 * 2 / 4 = 16 := by
  decide

end Baloon.MmvqGeometry
