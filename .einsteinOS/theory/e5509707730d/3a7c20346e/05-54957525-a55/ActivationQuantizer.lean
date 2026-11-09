/-!
# MODULE: activation_quantizer
SOURCES: src/kernels/cuda/quantize_act.cu
Quantizing an fp32 activation to ggml's vec_dot_type: Q8_0 (34-byte blocks of 32), a CPU-order Q8_0 variant with
fp32 scales, and Q8_K (292-byte blocks of 256 with block sums), plus matching dequantizers.

## Source semantics
Q8_0: d = amax/127, ints are x/d clamped to [-128,127] (ggml bytes) or [-127,127] (CPU rule). Zero blocks are all
zero. Q8_K: iscale = -127/max (signed max at largest magnitude), q = min 127 (nearest_int (iscale x)) with
nearest_int the 1.5*2^23 magic trick; bsums[j] = sum of 16 consecutive qs; n must be a multiple of 256.
Sizes must be multiples of the block (32 resp. 256).

## Form template identifiers
clamp -> "int8 saturation"; nearestInt -> "magic-number rounding"; bsum -> "block sum".
-/
namespace Strata.ActivationQuantizer

-- § saturation
/-- Clamp an integer into [lo, hi]. -/
def clamp (lo hi v : Int) : Int := if v < lo then lo else if v > hi then hi else v

/-- Clamping into [-127,127] bounds the result. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp (-127) 127 v ∧ clamp (-127) 127 v ≤ 127 := by
  unfold clamp
  split_ifs <;> omega

/-- Clamping is idempotent. -/
theorem clamp_idem (lo hi v : Int) (h : lo ≤ hi) : clamp lo hi (clamp lo hi v) = clamp lo hi v := by
  unfold clamp
  split_ifs <;> omega

/-- In-range values are unchanged. -/
theorem clamp_of_mem (lo hi v : Int) (h1 : lo ≤ v) (h2 : v ≤ hi) : clamp lo hi v = v := by
  unfold clamp
  split_ifs <;> omega

-- § magic-number rounding
/-- `nearest_int`: low 23 mantissa bits minus 2^22. -/
def nearestInt (bits : Nat) : Int := ((bits &&& 8388607 : Nat) : Int) - 4194304

/-- The result lies in [-2^22, 2^22 - 1], so the trick is valid for |x| < 2^22. -/
theorem nearestInt_range (bits : Nat) : -4194304 ≤ nearestInt bits ∧ nearestInt bits ≤ 4194303 := by
  have h : (bits &&& 8388607) ≤ 8388607 := Nat.and_le_right
  unfold nearestInt
  omega

-- § sizes and counts
/-- Q8_0 block count. -/
def q80Blocks (n : Nat) : Nat := n / 32

/-- Q8_K block count. -/
def q8kBlocks (n : Nat) : Nat := n / 256

/-- A grid of 128 threads per block covers every Q8_0 block. -/
theorem q80_grid_covers (n : Nat) : q80Blocks n ≤ ((q80Blocks n + 127) / 128) * 128 := by
  unfold q80Blocks
  omega

/-- A grid of 64 threads per block covers every Q8_K block. -/
theorem q8k_grid_covers (n : Nat) : q8kBlocks n ≤ ((q8kBlocks n + 63) / 64) * 64 := by
  unfold q8kBlocks
  omega

/-- Whole blocks cover the array when n is a multiple of the block. -/
theorem q80_blocks_cover (n : Nat) (h : n % 32 = 0) : q80Blocks n * 32 = n := by
  unfold q80Blocks
  omega

/-- Same for Q8_K. -/
theorem q8k_blocks_cover (n : Nat) (h : n % 256 = 0) : q8kBlocks n * 256 = n := by
  unfold q8kBlocks
  omega

-- § block sums
/-- Sum of a block of quantized values. -/
def bsum (qs : List Int) : Int := qs.sum

/-- Block sums are additive over concatenation: total of 16 sub-sums equals the sum over the 256 values. -/
theorem bsum_append (a b : List Int) : bsum (a ++ b) = bsum a + bsum b := by
  unfold bsum
  exact List.sum_append

/-- A zero block has zero sum (the amax = 0 case writes zeroes everywhere). -/
theorem bsum_zero (n : Nat) : bsum (List.replicate n 0) = 0 := by
  unfold bsum
  simp

end Strata.ActivationQuantizer
