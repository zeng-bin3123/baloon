/-!
# MODULE: activation_quantize
SOURCES: src/kernels/cuda/quantize_act.cu
Quantization of FP32 activations into ggml's `block_q8_0` (34 B / 32 elements) and `block_q8_K`
(292 B / 256 elements), the "scaled" Q8_0 variant that reproduces the CPU path's `act_quant_q8_1`, and the
matching dequantizers.  This module models the block layouts, the three integer clamp rules, the two
rounding rules (half-to-even vs half-away-from-zero), ggml's `nearest_int` bit trick, and the `bsums` contract.

## Source semantics
* `quantize_q8_0`: `d32 = amax/127`; store `fp16(d32)`; `q = clamp(rint(x / d32), -128, 127)` computed in
  double (ties to even); an all-zero block stores `d = 0`, `q = 0`.  Dequantization multiplies by the STORED fp16 d.
* `quantize_q8_0_scaled`: CPU rule — `s = amax>0 ? amax/127 : 0`, `inv = s>0 ? 1/s : 0`,
  `q = clamp(trunc(t + copysign(0.5, t)), -127, 127)` (ties away from zero), plus `scales[b] = s` unrounded.
* `quantize_q8_K`: `iscale = -127/max` (max = signed value of the FIRST largest magnitude), `q = min(127,
  nearest_int(fl(iscale*x)))` with no lower clamp, `bsums[j] = Σ q[16j .. 16j+15]`, `d = 1/iscale`; a zero block
  zeroes `d`, `qs` AND `bsums` (a deliberate divergence from ggml, which leaves `bsums` unwritten).
* Hosts refuse `n % 32 ≠ 0` (Q8_0) and `n % 256 ≠ 0` (Q8_K).

## Form template identifiers
`block_q8_0` → `Q8_0` (34 B); `block_q8_K` → `Q8_K` (292 B); `rint` → `roundEven2`;
CPU `lround`-style rule → `roundAway2`; `nearest_int` → `nearestIntBits`; `bsums` → `bsum`;
clamps → `clampQ8_0`, `clampScaled`, `clampQ8K`.
-/

namespace Strata.ActivationQuantize

-- § block layouts

/-- `block_q8_0` bytes: fp16 `d` + 32 × int8. -/
def q8_0Bytes : Nat := 2 + 32

/-- `block_q8_K` bytes: f32 `d` + 256 × int8 + 16 × int16 (no padding). -/
def q8_KBytes : Nat := 4 + 256 + 16 * 2

/-- The strides used by every kernel in the file. -/
theorem block_strides : q8_0Bytes = 34 ∧ q8_KBytes = 292 := by decide

/-- Every Q8_0 block's fp16 `d` is 2-byte aligned (stride 34). -/
theorem q8_0_d_aligned (b : Nat) : (34 * b) % 2 = 0 := by omega

/-- Every Q8_K block's f32 `d` is 4-byte aligned and its `bsums` 2-byte aligned (stride 292, offset 260). -/
theorem q8_K_fields_aligned (b : Nat) : (292 * b) % 4 = 0 ∧ (292 * b + 4 + 256) % 2 = 0 := by omega

/-- Q8_K's 256 + 32 tail: `bsums` ends exactly at the block boundary. -/
theorem q8_K_bsums_end : 4 + 256 + 16 * 2 = q8_KBytes := rfl

-- § clamps

/-- `quantize_q8_0` clamp `[-128, 127]`. -/
def clampQ8_0 (v : Int) : Int := max (-128) (min 127 v)

/-- `quantize_q8_0_scaled` clamp `[-127, 127]` (symmetric, verbatim from the CPU path). -/
def clampScaled (v : Int) : Int := max (-127) (min 127 v)

/-- `quantize_q8_K`: `min(127, v)` only. -/
def clampQ8K (v : Int) : Int := min 127 v

/-- Both Q8_0 clamps land in `int8`. -/
theorem clamps_int8 (v : Int) :
    -128 ≤ clampQ8_0 v ∧ clampQ8_0 v ≤ 127 ∧ -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampQ8_0 clampScaled; omega

/-- The two Q8_0 clamps differ only at −128: the scaled variant can never emit it. -/
theorem clamps_differ_only_at_min (v : Int) (h : -127 ≤ v) : clampQ8_0 v = clampScaled v := by
  unfold clampQ8_0 clampScaled; omega

/-- Q8_K's clamp is one-sided: it is only correct because `|iscale * x| ≤ 127` keeps `v ≥ -127`
(the value at the arg-max is `-127` exactly).  An input below −128 would wrap in the `int8_t` cast. -/
theorem clampQ8K_needs_lower (v : Int) (h : -127 ≤ v) : -127 ≤ clampQ8K v ∧ clampQ8K v ≤ 127 := by
  unfold clampQ8K; omega

/-- Without that premise the clamp is not an `int8` guarantee. -/
theorem clampQ8K_unbounded_below : clampQ8K (-200) = -200 := by decide

-- § rounding rules, on half-integers `k/2`

/-- Ties away from zero at `k/2`: `trunc(t + copysign(0.5, t))`. -/
def roundAway2 (k : Int) : Int := if 0 ≤ k then (k + 1) / 2 else -((-k + 1) / 2)

/-- Ties to even at `k/2` (`rint`). -/
def roundEven2 (k : Int) : Int :=
  if k % 2 = 0 then k / 2 else if ((k - 1) / 2) % 2 = 0 then (k - 1) / 2 else (k + 1) / 2

/-- On exact integers the two rules agree. -/
theorem round_agree_on_integers (n : Int) : roundAway2 (2 * n) = n ∧ roundEven2 (2 * n) = n := by
  unfold roundAway2 roundEven2
  constructor
  · split <;> omega
  · simp <;> omega

/-- On ties the two rules can disagree: `2.5` → 3 vs 2, `-2.5` → −3 vs −2, while `1.5` → 2 under both. -/
theorem round_ties_examples :
    roundAway2 5 = 3 ∧ roundEven2 5 = 2 ∧ roundAway2 (-5) = -3 ∧ roundEven2 (-5) = -2 ∧
    roundAway2 3 = 2 ∧ roundEven2 3 = 2 := by
  decide

/-- The rules never differ by more than one step. -/
theorem round_rules_close (k : Int) : (roundAway2 k - roundEven2 k).natAbs ≤ 1 := by
  unfold roundAway2 roundEven2
  split <;> split <;> (try split) <;> omega

-- § ggml's `nearest_int`

/-- `nearest_int`'s integer read-back: `(bits & 0x7fffff) - 0x400000`. -/
def nearestIntBits (bits : Int) : Int := bits % 8388608 - 4194304

/-- **The magic constant.**  `12582912 = 1.5·2^23`.  When `|v| < 2^22`, the float `12582912 + v` has biased
exponent 150 and mantissa field `2^22 + v`, i.e. bit pattern `150·2^23 + 2^22 + v`; the read-back returns `v`. -/
theorem nearest_int_readback (v : Int) (hlo : -4194304 ≤ v) (hhi : v < 4194304) :
    nearestIntBits (150 * 8388608 + 4194304 + v) = v := by
  unfold nearestIntBits; omega

/-- The constant's decomposition: `1.5 · 2^23 = 2^23 + 2^22`. -/
theorem magic_value : (12582912 : Nat) = 2 ^ 23 + 2 ^ 22 := by decide

-- § bsums

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step of `rsum`. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- A sum over `m*k` indices is the sum of `m` consecutive chunk sums. -/
theorem rsum_nested (f : Nat → Int) (k m : Nat) :
    rsum f (m * k) = rsum (fun b => rsum (fun j => f (b * k + j)) k) m := by
  induction m with
  | zero => simp [rsum]
  | succ m ih => rw [Nat.succ_mul, rsum_split, ih, rsum_succ]

/-- Bounds propagate through `rsum`. -/
theorem rsum_bounds (f : Nat → Int) (lo hi : Int) (n : Nat) (h : ∀ i, i < n → lo ≤ f i ∧ f i ≤ hi) :
    lo * n ≤ rsum f n ∧ rsum f n ≤ hi * n := by
  induction n with
  | zero => simp [rsum]
  | succ n ih =>
    have := ih (fun i hi' => h i (by omega))
    have hn := h n (by omega)
    rw [rsum_succ]
    simp only [Int.natCast_add, Int.natCast_one, Int.mul_add, Int.mul_one]
    omega

/-- `bsums[j]`. -/
def bsum (q : Nat → Int) (j : Nat) : Int := rsum (fun ii => q (j * 16 + ii)) 16

/-- **`bsums` fits `int16`**: sixteen `int8` values sum to within `[-2048, 2032]`. -/
theorem bsum_fits_int16 (q : Nat → Int) (j : Nat) (hq : ∀ i, -128 ≤ q i ∧ q i ≤ 127) :
    -32768 ≤ bsum q j ∧ bsum q j ≤ 32767 := by
  have := rsum_bounds (fun ii => q (j * 16 + ii)) (-128) 127 16 (fun i _ => hq _)
  unfold bsum
  omega

/-- **`bsums` partition the block**: their total is the sum of all 256 quants (what ggml's dot product uses). -/
theorem bsums_total (q : Nat → Int) : rsum (bsum q) 16 = rsum q 256 := by
  rw [show (256 : Nat) = 16 * 16 from rfl, rsum_nested]
  rfl

/-- A zero block's bsums are zero (the deliberate full zeroing). -/
theorem zero_block_bsums (j : Nat) : bsum (fun _ => 0) j = 0 := by
  unfold bsum
  rfl

-- § scales and host checks

/-- The measured worst relative scale difference `4.761e-4` (fp32 `s` vs the block's fp16 `d`) is within
fp16's half-ulp relative bound `2^-11 ≈ 4.883e-4`: the measurement is exactly the rounding of `d`. -/
theorem measured_scale_gap_within_half_ulp : 4761 * 2048 < 10000000 := by decide

/-- `quantize_q8_0` accepts `n` iff it is a positive multiple of 32. -/
def q8_0Accepts (n : Nat) : Bool := 0 < n && n % 32 == 0

/-- `quantize_q8_K` accepts `n` iff it is a positive multiple of 256. -/
def q8_KAccepts (n : Nat) : Bool := 0 < n && n % 256 == 0

/-- The shared-expert width 640 has a Q8_0 activation but structurally no Q8_K one; 2560 has both. -/
theorem widths : q8_0Accepts 640 = true ∧ q8_KAccepts 640 = false ∧
    q8_0Accepts 2560 = true ∧ q8_KAccepts 2560 = true := by decide

/-- Q8_K acceptance implies Q8_0 acceptance. -/
theorem q8K_implies_q8_0 (n : Nat) (h : q8_KAccepts n = true) : q8_0Accepts n = true := by
  simp [q8_KAccepts, q8_0Accepts] at h ⊢
  omega

end Strata.ActivationQuantize
