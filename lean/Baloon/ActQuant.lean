-- Baloon.ActQuant: activation quantisation invariants (Q8_0, Q8_1, Q8_K block layouts,
-- clamp arithmetic, bsum bounds, nearest_int magic constant).
-- Independent implementation proving the same mathematical facts as Ref.ActQuant.

namespace Baloon.ActQuant

def clampI (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

theorem clampI_bounds (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clampI lo hi v ∧ clampI lo hi v ≤ hi := by
  simp only [clampI]; split_ifs <;> omega

theorem clampI_id (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clampI lo hi v = v := by
  simp only [clampI]; split_ifs <;> omega

theorem symmetric_range_closed (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : -127 ≤ -v ∧ -v ≤ 127 := by
  omega

theorem q8_0_layout (j : Nat) (h : j < 32) : 2 + 32 = 34 ∧ 2 + j < 34 := by omega
theorem q8_1_layout (j : Nat) (h : j < 32) : 4 + 32 = 36 ∧ 4 + j < 36 := by omega
theorem q8_k_layout (j : Nat) (h : j < 16) :
    4 + 256 + 32 = 292 ∧ 260 % 2 = 0 ∧ 4 + 256 + 2 * j + 2 ≤ 292 := by omega
theorem block_start_alignment (b : Nat) : (b * 292) % 4 = 0 ∧ (b * 34) % 2 = 0 := by omega
theorem q81_index (i : Nat) : i = (i / 32) * 32 + i % 32 ∧ i % 32 < 32 := by omega
theorem q81_in_range (i n : Nat) (hn : n % 32 = 0) (hi : i < n) : i / 32 < n / 32 := by omega
theorem bsums_fit : 16 * 127 = 2032 ∧ 2032 ≤ 32767 := by decide
theorem nearest_int_magic (n : Int) (h1 : -4194304 ≤ n) (h2 : n < 4194304) :
    ((4194304 + n) % 8388608) - 4194304 = n := by omega
theorem nearest_int_range : (127 : Int) < 4194304 := by decide
theorem q8k_min_only (v : Int) (h : v ≤ 127) : min 127 v = v := by omega

end Baloon.ActQuant
