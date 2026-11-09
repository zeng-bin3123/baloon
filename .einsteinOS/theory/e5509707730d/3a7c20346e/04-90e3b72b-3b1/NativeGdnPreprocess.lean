/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
Native GDN preprocessing kernels adapted from llama.cpp: conv + SiLU with history slide, per-row L2/RMS norm
with a scale, beta sigmoid, gate softplus, and output RMS-norm with sigmoid gating. Host wrappers validate
counts, geometry and span disjointness.

## Source semantics
* conv requires `d_conv = 4`; history holds 3 taps per channel (`c*3 + tap`), weights 4 (`c*4 + tap`).
* norms require width 128 and are launched with 256 threads (8 warps); the block sum reads 8 warp partials.
* counts must lie in `[1, 65535]`; grids are `ceil(count/256)`.
* writable spans must be pairwise disjoint and disjoint from inputs.

## Form template identifiers
* `S` -> head_width, `convHistoryNext` -> history_slide, `countOk` -> launch_count_guard.
-/

namespace Strata.NativeGdnPreprocess

/-- Head width. -/
def S : Nat := 128

/-- Count guard of the host wrappers. -/
def countOk (count : Nat) : Prop := 0 < count ∧ count ≤ 65535

/-- The grid for an accepted count has between 1 and 256 blocks. -/
theorem grid_bounds (count : Nat) (h : countOk count) :
    1 ≤ (count + 255) / 256 ∧ (count + 255) / 256 ≤ 256 := by
  unfold countOk at h; omega

/-- The grid covers every element. -/
theorem grid_covers (count : Nat) : count ≤ (count + 255) / 256 * 256 := by omega

/-- Block sums: 256 threads are 8 warps and the second pass reads lanes `< 8`. -/
theorem norm_warps : 256 / 32 = 8 := by decide

/-- 128 columns fit in 256 threads (the `col < S` guards cover the rest). -/
theorem norm_threads_cover : S ≤ 256 := by decide

/-- History addressing `c*3 + tap` is injective for taps `< 3`. -/
theorem hist_inj (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

/-- Weight addressing `c*4 + tap` is injective for taps `< 4`. -/
theorem weight_inj (c t c' t' : Nat) (ht : t < 4) (ht' : t' < 4)
    (h : c * 4 + t = c' * 4 + t') : c = c' ∧ t = t' := by omega

/-- The history region has `3*channels` floats and the last address stays inside it. -/
theorem hist_last_lt (channels c t : Nat) (hc : c < channels) (ht : t < 3) : c * 3 + t < 3 * channels := by
  omega

/-- The weight region has `4*channels` floats. -/
theorem weight_last_lt (channels c t : Nat) (hc : c < channels) (ht : t < 4) : c * 4 + t < 4 * channels := by
  omega

/-- Next history after consuming input `x`: taps shift left, input enters last. -/
def convHistoryNext (v1 v2 x : Int) : Int × Int × Int := (v1, v2, x)

/-- The shifted history equals the old taps 1..3 (`values[tap+1]` in the source). -/
theorem history_shift (v0 v1 v2 x : Int) : convHistoryNext v1 v2 x = (v1, v2, x) := rfl

/-- Tap sum over four values. -/
def convSum (v0 v1 v2 v3 w0 w1 w2 w3 : Int) : Int := v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3

/-- Adding the zero bias after the sum is the identity (the source adds `0.0f` explicitly). -/
theorem zero_bias_identity (s : Int) : s + 0 = s := by omega

/-- Gate softplus branch: for `v > 20` the value is returned unchanged (abstract threshold check). -/
def softplusBranchLarge (v : Int) : Bool := decide (v > 20)

/-- The large branch fires exactly above 20. -/
theorem softplus_large_iff (v : Int) : softplusBranchLarge v = true ↔ v > 20 := by
  unfold softplusBranchLarge; simp

/-- Output bytes of a norm over `heads` rows of width 128. -/
theorem norm_bytes (heads : Nat) : heads * 128 * 4 = 512 * heads := by omega

/-- Gamma is a single 128-float row: `S * 4 = 512` bytes. -/
theorem gamma_bytes : S * 4 = 512 := by decide

/-- Span interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric, so pairwise checks `j < i` suffice. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; constructor <;> intro h <;> omega

/-- Adjacent spans are disjoint: the byte just past the first is the start of the second. -/
theorem adjacent_disjoint (a an bn : Nat) : ¬ overlap a an (a + an) bn := by
  unfold overlap; omega

/-- Triple pairwise-disjointness of the three writable conv spans from their cumulative layout. -/
theorem layout_disjoint3 (a n1 n2 n3 : Nat) :
    ¬ overlap a n1 (a + n1) n2 ∧ ¬ overlap a n1 (a + n1 + n2) n3 ∧ ¬ overlap (a + n1) n2 (a + n1 + n2) n3 := by
  unfold overlap; omega

end Strata.NativeGdnPreprocess
