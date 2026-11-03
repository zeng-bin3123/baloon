/-!
# MODULE: native_ple_postops
Index algebra and validation contracts of the native PLE post-operations (`native_ple_postops.cu`): the
512-thread gate reduction, broadcast/conv/residual kernels, and the span validation.

## Source semantics
The gate kernel reduces one stream of 2560 products with 512 threads, each holding 8 partial lanes
(`d = tid + 512*j`, valid when `d < 2560`, i.e. `j < 5`); the 512 threads form 16 warps whose partials are
combined by the first 16 lanes. The sign is `float(s>0) - float(s<0)`. The conv reads history rows 0,3,6 at
`c*9 + 3k` and the new normalised value for tap 3. Every span must be non-null, aligned and non-wrapping.

## Form template identifiers
- N -> `N`, H -> `H`, D -> `D`, HISTORY -> `HISTORY`
- `Span` -> (`p`, `bytes`, `alignment`); `overlaps` -> `overlaps`
-/

namespace Strata.NativePlePostops

def N : Nat := 2560
def H : Nat := 4
def D : Nat := N * H
def HISTORY : Nat := 9

/-- The residual width. -/
theorem D_eq : D = 10240 := by decide

-- § gate reduction geometry

/-- Each thread/lane slot (tid, j) maps to a distinct element index. -/
theorem gate_slot_inj (t j t' j' : Nat) (ht : t < 512) (ht' : t' < 512)
    (h : t + j * 512 = t' + j' * 512) : t = t' ∧ j = j' := by omega

/-- A slot is in range exactly when j < 5, since 2560 = 5 * 512. -/
theorem gate_slot_valid (t j : Nat) (ht : t < 512) : t + j * 512 < 2560 ↔ j < 5 := by omega

/-- Every element of a stream is held by exactly one in-range slot. -/
theorem gate_slot_cover (d : Nat) (hd : d < 2560) : ∃ t j, t < 512 ∧ j < 5 ∧ d = t + j * 512 := by
  exact ⟨d % 512, d / 512, by omega, by omega, by omega⟩

/-- The eight partial lanes include the three that are always out of range. -/
theorem gate_unused_lanes (t j : Nat) (ht : t < 512) (hj : 5 ≤ j) (hj8 : j < 8) : ¬ (t + j * 512 < 2560) := by
  omega

/-- 512 threads are 16 warps, which is exactly the 16 partial lanes the final reduction reads. -/
theorem gate_warps : 512 / 32 = 16 ∧ 16 ≤ 32 := by decide

/-- The final reduction reads partials for lane < 16 only, one per warp. -/
theorem gate_partial_cover (w : Nat) (hw : w < 512 / 32) : w < 16 := by omega

-- § sign

def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0
def signDiff (s : Int) : Int := (if s > 0 then 1 else 0) - (if s < 0 then 1 else 0)

/-- `float(s>0) - float(s<0)` is the sign function. -/
theorem sign_diff_eq (s : Int) : signDiff s = sgn s := by
  unfold signDiff sgn; split_ifs <;> omega

/-- The sign is odd. -/
theorem sgn_odd (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn; split_ifs <;> omega

-- § elementwise launches

/-- Broadcast and conv kernels launch exactly D/256 blocks and the guard `i < D` always holds. -/
theorem elementwise_exact (b t : Nat) (hb : b < D / 256) (ht : t < 256) : b * 256 + t < D := by
  unfold D N H at *; omega

/-- Broadcast reads value[i % N] and gate[i / N]; both are in range. -/
theorem broadcast_idx (i : Nat) (hi : i < 10240) : i % 2560 < 2560 ∧ i / 2560 < 4 := by omega

-- § conv history

/-- History index of tap k < 3 for channel c is `c*9 + 3k`, inside the channel's 9-entry row. -/
theorem conv_hist_row (c k : Nat) (hk : k < 3) : c * 9 + 3 * k < c * 9 + 9 := by omega

/-- Tap history indices stay below the whole history buffer. -/
theorem conv_hist_lt (c k : Nat) (hc : c < 10240) (hk : k < 3) : c * 9 + 3 * k < 9 * 10240 := by omega

/-- Distinct (channel, tap) pairs read distinct history entries. -/
theorem conv_hist_inj (c k c' k' : Nat) (hk : k < 3) (hk' : k' < 3)
    (h : c * 9 + 3 * k = c' * 9 + 3 * k') : c = c' ∧ k = k' := by omega

/-- Conv weights are indexed `c*4 + k`, injective for k < 4. -/
theorem conv_weight_inj (c k c' k' : Nat) (hk : k < 4) (hk' : k' < 4)
    (h : c * 4 + k = c' * 4 + k') : c = c' ∧ k = k' := by omega

-- § span validation

def overlaps (x a y b : Nat) : Prop := x < y + b ∧ y < x + a

/-- Overlap is symmetric. -/
theorem overlaps_comm (x a y b : Nat) : overlaps x a y b ↔ overlaps y b x a := by
  unfold overlaps; omega

/-- A span never overlaps one starting at its end. -/
theorem overlaps_adjacent (x a b : Nat) : ¬ overlaps x a (x + a) b := by
  unfold overlaps; omega

/-- A span overlaps itself when non-empty. -/
theorem overlaps_self (x a : Nat) (h : 0 < a) : overlaps x a x a := by
  unfold overlaps; omega

/-- Validation (non-null, aligned, no wrap) implies the span end does not overflow the address space. -/
theorem validate_no_wrap (p bytes align maxAddr : Nat) (hp : p ≤ maxAddr - bytes) (hb : bytes ≤ maxAddr) :
    p + bytes ≤ maxAddr := by omega

/-- Validation rules out a null pointer and a misaligned one. -/
theorem validate_aligned (p align : Nat) (h0 : p ≠ 0) (ha : p % align = 0) (hal : 0 < align) :
    0 < p ∧ align ∣ p := by
  exact ⟨Nat.pos_of_ne_zero h0, Nat.dvd_of_mod_eq_zero ha⟩

/-- The only tolerated output/input overlap is exact aliasing, and exact aliasing is a self-overlap. -/
theorem alias_is_exact_overlap (x a : Nat) (h : 0 < a) : overlaps x a x a := overlaps_self x a h

end Strata.NativePlePostops
