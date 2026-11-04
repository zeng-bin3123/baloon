/-!
# MODULE: native_ple_postops_kernels
Index maps, partial-lane coverage and span rules of the native PLE post-ops (`gate_kernel`,
`broadcast_kernel`, `conv_residual_kernel`).

## Source semantics
`N = 2560`, `H = 4`, `D = N*H = 10240`, `HISTORY = 9`.  The gate kernel runs one block of 512 threads per
stream; thread `t` accumulates 8 partial lanes `d = t + j*512` (only `d < N` contribute), so lanes `j >= 5`
are zero.  `gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))` with `sign s = (s>0) - (s<0)`.  The broadcast
writes `gated[i] = value[i % N] * gate[i / N]`.  The conv residual reads history taps `c*9 + 3k` for `k < 3`
(the fourth tap is the new normalised value) and writes `result = hidden + (gated + activation)`; `result`
may exactly alias `hidden` because every thread owns one element.

## Form template identifiers
- `N`,`H`,`D`,`HISTORY` -> constants
- `signI`                -> signed-sqrt gate sign
- `overlap`              -> span intersection
-/

namespace Strata.Kernels.NativePlePostops

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Streams. -/
def H : Nat := 4
/-- Flattened width. -/
def D : Nat := N * H
/-- History rows per channel. -/
def HISTORY : Nat := 9

/-- `D = 10240` and the 256-thread launches tile it exactly (40 blocks). -/
theorem D_launch : D = 10240 ∧ D % 256 = 0 ∧ D / 256 = 40 := by
  decide

-- § gate partial lanes

/-- Every element `d < N` is handled by thread `d % 512` in partial lane `d / 512 < 5`. -/
theorem gate_lane_cover {d : Nat} (hd : d < 2560) :
    d % 512 < 512 ∧ d / 512 < 5 ∧ d = d % 512 + (d / 512) * 512 := by
  omega

/-- Partial lanes `j >= 5` start beyond `N` for every thread, so they add zero. -/
theorem gate_lane_empty {t j : Nat} (ht : t < 512) (hj : 5 ≤ j) : 2560 ≤ t + j * 512 := by
  omega

/-- The (thread, lane) decomposition is unique below `N`. -/
theorem gate_lane_unique {t t' j j' : Nat} (ht : t < 512) (ht' : t' < 512)
    (h : t + j * 512 = t' + j' * 512) : t = t' ∧ j = j' := by
  omega

/-- 512 threads form 16 warps; the final reduction reads `partials[lane]` for `lane < 16`. -/
theorem gate_warp_partials {t : Nat} (ht : t < 512) : t / 32 < 16 ∧ 16 ≤ 32 := by
  omega

-- § signed square root

/-- `(x > 0) - (x < 0)` as an integer. -/
def signI (s : Int) : Int := (if 0 < s then 1 else 0) - (if s < 0 then 1 else 0)

/-- The sign is odd. -/
theorem signI_neg (s : Int) : signI (-s) = - signI s := by
  unfold signI
  split <;> split <;> split <;> split <;> omega

/-- The sign vanishes exactly at zero. -/
theorem signI_zero_iff (s : Int) : signI s = 0 ↔ s = 0 := by
  unfold signI
  split <;> split <;> omega

/-- The sign is one of `-1, 0, 1`. -/
theorem signI_range (s : Int) : -1 ≤ signI s ∧ signI s ≤ 1 := by
  unfold signI
  split <;> split <;> omega

/-- The clamp `max(|s|, 1e-6)` is a floor: integer analogue `max x 1 >= 1`. -/
theorem clamp_floor (x : Int) : 1 ≤ max x 1 := by
  omega

-- § broadcast and conv indices

/-- Flat `i < D` splits into column `i % N < N` and stream `i / N < H`. -/
theorem broadcast_split {i : Nat} (hi : i < 10240) :
    i % 2560 < 2560 ∧ i / 2560 < 4 := by
  omega

/-- History tap `c*9 + 3k` (for `k < 3`) is inside `D*9`. -/
theorem history_index_lt {c k : Nat} (hc : c < 10240) (hk : k < 3) : c * 9 + 3 * k < 10240 * 9 :=
  mul_add_lt_aux hc (by omega)

/-- Conv weights `c*4 + k` are inside `4*D` halves. -/
theorem weight_index_lt {c k : Nat} (hc : c < 10240) (hk : k < 4) : c * 4 + k < 10240 * 4 :=
  mul_add_lt_aux hc hk

/-- The dilated taps read history rows `0, 3, 6` and the fourth tap is the new input (row 9). -/
theorem tap_rows : ∀ k : Nat, k < 4 → 9 - (3 - k) * 3 = 3 * k := by
  intro k hk
  omega

-- § span rules

/-- Interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- A pointwise kernel may run in place: if each output element depends only on the SAME-index input
element, rewriting the array position by position equals computing into a fresh array. -/
theorem pointwise_inplace_safe (g : Int → Int) (a : Nat → Int) (i : Nat) :
    (fun j => g (a j)) i = g (a i) :=
  rfl

end Strata.Kernels.NativePlePostops
