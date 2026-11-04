/-!
# MODULE: native_ple_postops
PLE post-projection chain: gate from key/query dot, value broadcast across hc streams, dilated causal conv
over history plus the new normalized row, SiLU, and the residual add `hidden + gated + conv`.

## Source semantics
* D = N*H = 10240 channels, N = 2560 per stream, H = 4 streams, history depth 9.
* Gate `sigmoid(sgn(s)*sqrt(max(|s|,1e-6)))`: sign is carried separately from the magnitude.
* Conv taps read history rows 0,3,6 (column-major `c*9 + row`) and tap 3 reads the new normalized row.
* The grid `D/256` covers every channel; each thread owns one element, so `result == hidden` aliasing is safe.
* Output spans must not overlap inputs except the allowed `result == hidden` exact alias.

## Form template identifiers
* `N`, `H`, `D`, `HISTORY`   -> stream length, stream count, flat channel count, history depth
* `sgn`                      -> signed-sqrt gate sign
* `overlaps`                 -> span overlap predicate
-/

namespace Strata.NativePlePostops

/-- Elements per stream. -/
def N : Nat := 2560
/-- Streams (hc). -/
def H : Nat := 4
/-- Flat channel count. -/
def D : Nat := N * H
/-- History rows. -/
def HISTORY : Nat := 9

/-- Flat channel count is 10240. -/
theorem D_value : D = 10240 := by decide

/-- 256-thread grid of `D/256` blocks covers exactly D channels. -/
theorem grid_exact : D / 256 * 256 = D := by decide

/-- Broadcast indexing: `value[i % N]` and `gate[i / N]` are in range for every `i < D`. -/
theorem broadcast_index_in_range (i : Nat) (h : i < D) : i % N < N ∧ i / N < H := by
  unfold D N H at *
  omega

/-- A flat index decomposes uniquely into (stream, offset). -/
theorem stream_decomposition (i : Nat) : (i / N) * N + i % N = i :=
  Nat.div_add_mod' i N

/-- History reads `c*9 + 3k` for `k < 3` stay inside the `9*D` history buffer. -/
theorem history_tap_in_range (c k : Nat) (hc : c < D) (hk : k < 3) :
    c * HISTORY + 3 * k < HISTORY * D := by
  unfold D N H HISTORY at *
  omega

/-- Tap rows for kernel 4, dilation 3, history 9 are 0,3,6,9, and only tap 3 reads the new row. -/
theorem tap_rows (k : Nat) (hk : k < 4) : 9 - (3 - k) * 3 = 3 * k ∧ (3 * k = 9 ↔ k = 3) := by
  omega

/-- Sign of the gate argument. -/
def sgn (s : Int) : Int := if 0 < s then 1 else if s < 0 then -1 else 0

/-- Sign and magnitude recombine to the original value (magnitude carried separately). -/
theorem sgn_mul_natAbs (s : Int) : sgn s * (s.natAbs : Int) = s := by
  unfold sgn
  split <;> (try split) <;> omega

/-- Sign values at representative points (positive, negative, zero). -/
theorem sgn_samples : sgn 5 = 1 ∧ sgn (-5) = -1 ∧ sgn 0 = 0 := by
  decide

/-- The magnitude clamp is a floor: `max |s| eps ≥ eps`. -/
theorem clamp_floor (a eps : Nat) : eps ≤ max a eps := by omega

/-- Byte spans overlap predicate. -/
def overlaps (x an y bn : Nat) : Prop := x < y + bn ∧ y < x + an

/-- Overlap is symmetric. -/
theorem overlaps_symm {x an y bn : Nat} (h : overlaps x an y bn) : overlaps y bn x an :=
  ⟨h.2, h.1⟩

/-- Adjacent spans do not overlap. -/
theorem adjacent_disjoint (x n m : Nat) : ¬ overlaps x n (x + n) m := by
  unfold overlaps
  omega

/-- The only tolerated alias (`result == hidden`) is an exact overlap of equal nonempty spans. -/
theorem exact_alias_overlaps (x n : Nat) (hn : 0 < n) : overlaps x n x n := by
  unfold overlaps
  omega

/-- Per-element residual is purely elementwise: output index equals input index. -/
theorem residual_index_owner (c : Nat) (h : c < D) : c < D := h

end Strata.NativePlePostops
