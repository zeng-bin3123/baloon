/-!
# MODULE: activation_quantize
SOURCES: src/kernels/cuda/quantize_act.cu
Quantisation of an FP32 activation to ggml's `vec_dot_type` blocks: Q8_0 (two variants: ggml-exact and
CPU-path-exact with a parallel fp32 scale array), Q8_K, and the matching dequantisers.

## Source semantics
* `block_q8_0` = `{fp16 d; int8 qs[32]}` = 34 bytes; `d` stored little-endian in bytes 0,1.
* `quantize_q8_0`: `d32 = amax/127`; `q = rint(x/d32)` in double, clamped to `[-128,127]`; zero block
  writes `d = +0` and all `q = 0`.
* `quantize_q8_0_scaled` (CPU-path parity): `s = amax/127` (0 if `amax = 0`), `inv = 1/s`;
  `q = trunc(x*inv + copysign(0.5))` — round half AWAY from zero — clamped to `[-127,127]`; also writes
  `scales[b] = s` unrounded.
* `block_q8_K` = `{f32 d; int8 qs[256]; int16 bsums[16]}` = 292 bytes; `qs[j] = min(127, nearest_int(iscale*x))`
  with `iscale = -127/max`, `bsums[j] = Σ qs[16j..16j+16)`; the zero block also zeroes `bsums`.
* `nearest_int` adds `1.5·2^23` and reads the low 23 mantissa bits minus `2^22`.
* Host: `n ≤ 0` no-op, `n % 32` (Q8_0) / `n % 256` (Q8_K) must be 0; `scales` must be non-null.

## Form template identifiers
* `Q8_0_BYTES = 34`, `Q8K_BYTES = 292`, `QK8_0 = 32`, `QK_K = 256`
* `clampI` ↔ the two clamp sequences; `roundAway` ↔ `t + copysign(0.5)` then `(int)`
* `nearestIntBits`, `decodeNearest` ↔ `nearest_int_dev`
* `bsum` ↔ the `bsums` loop
-/

namespace Strata.QuantAct

-- § block layouts

/-- Q8_0 block size and field offsets. -/
theorem q8_0_layout : 2 + 32 = 34 := by decide

/-- Q8_K block: `d` at 0, `qs` at 4, `bsums` at 260, total 292 with no padding (ggml's `static_assert`). -/
theorem q8_K_layout : 4 + 256 + 2 * 16 = 292 ∧ 4 + 256 = 260 ∧ 260 % 2 = 0 := by decide

/-- Element `i` of a Q8_0 buffer lives at byte `34·(i/32) + 2 + i%32`, inside the `n/32` blocks. -/
theorem q8_0_element_addr (n i : Nat) (hn : n % 32 = 0) (hi : i < n) :
    34 * (i / 32) + 2 + i % 32 < 34 * (n / 32) := by omega

/-- Element `i` of a Q8_K buffer lives at byte `292·(i/256) + 4 + i%256`, before that block's `bsums`. -/
theorem q8_K_element_addr (n i : Nat) (hn : n % 256 = 0) (hi : i < n) :
    292 * (i / 256) + 4 + i % 256 < 292 * (i / 256) + 260 ∧
    292 * (i / 256) + 292 ≤ 292 * (n / 256) := by omega

/-- The fp16 scale's byte split `out[0] = d & 0xFF, out[1] = d >> 8` and the reader's
`blk[0] | (blk[1] << 8)` round-trip every 16-bit pattern. -/
theorem f16_bytes_roundtrip (d : Nat) (hd : d < 65536) :
    (d &&& 255) + 256 * (d >>> 8) = d ∧ d &&& 255 < 256 ∧ d >>> 8 < 256 := by
  have h1 : d &&& 255 = d % 256 := by simpa using Nat.and_two_pow_sub_one_eq_mod d 8
  rw [h1, Nat.shiftRight_eq_div_pow]
  simp
  omega

/-- …and the reader's OR is an addition, because the low byte is below `2^8`. -/
theorem f16_read_or (lo hi : Nat) (hlo : lo < 256) : (hi <<< 8) ||| lo = lo + 256 * hi := by
  rw [← Nat.shiftLeft_add_eq_or_of_lt (by simpa using hlo), Nat.shiftLeft_eq]
  simp; omega

-- § clamps

/-- Integer clamp to `[lo, hi]`. -/
def clampI (lo hi v : Int) : Int := if v < lo then lo else if hi < v then hi else v

/-- A clamp lands in its interval. -/
theorem clampI_range (lo hi v : Int) (h : lo ≤ hi) : lo ≤ clampI lo hi v ∧ clampI lo hi v ≤ hi := by
  unfold clampI
  split <;> (try split) <;> omega

/-- Both Q8_0 variants produce int8 values; the scaled variant never produces `-128`. -/
theorem q8_0_values (v : Int) :
    -128 ≤ clampI (-128) 127 v ∧ clampI (-128) 127 v ≤ 127 ∧
    -127 ≤ clampI (-127) 127 v ∧ clampI (-127) 127 v ≤ 127 :=
  ⟨(clampI_range _ _ v (by decide)).1, (clampI_range _ _ v (by decide)).2,
   (clampI_range _ _ v (by decide)).1, (clampI_range _ _ v (by decide)).2⟩

/-- The scaled variant's symmetric clamp commutes with negation (the `[-128,127]` one does not):
quantising `-x` gives exactly `-q`. -/
theorem symmetric_clamp_odd (v : Int) : clampI (-127) 127 (-v) = - clampI (-127) 127 v := by
  unfold clampI
  by_cases h1 : -v < -127 <;> by_cases h2 : 127 < -v <;> by_cases h3 : v < -127 <;>
    by_cases h4 : 127 < v <;> simp [h1, h2, h3, h4] <;> omega

/-- …whereas the ggml-exact clamp is asymmetric: `128` and `-128` do not map to negatives of each other. -/
theorem asymmetric_clamp_witness : clampI (-128) 127 (-128) ≠ - clampI (-128) 127 128 := by decide

-- § rounding rules

/-- Round half away from zero of `a / D` (`D > 0`), as the CPU path computes it:
`trunc(t + copysign(0.5, t))`. -/
def roundAway (a : Int) (D : Nat) : Int :=
  if 0 ≤ a then ((2 * a.toNat + D) / (2 * D) : Nat) else - (((2 * (-a).toNat + D) / (2 * D) : Nat) : Int)

/-- For non-negative inputs the result `v` satisfies `|2a − 2Dv| ≤ D` with ties upward:
`2Dv ≤ 2a + D < 2Dv + 2D`. -/
theorem roundAway_nonneg (a D : Nat) (hD : 0 < D) :
    let v := (2 * a + D) / (2 * D)
    2 * (D * v) ≤ 2 * a + D ∧ 2 * a + D < 2 * (D * v) + 2 * D := by
  intro v
  have h1 := Nat.div_add_mod (2 * a + D) (2 * D)
  have h2 := Nat.mod_lt (2 * a + D) (show 0 < 2 * D by omega)
  have h3 : 2 * D * v = 2 * (D * v) := Nat.mul_assoc 2 D v
  generalize (2 * a + D) % (2 * D) = r at *
  rw [h3] at h1
  omega

/-- Half away from zero is odd-symmetric: `round(-t) = -round(t)`. -/
theorem roundAway_neg (a : Int) (D : Nat) (ha : 0 < a) : roundAway (-a) D = - roundAway a D := by
  unfold roundAway
  have h1 : ¬ (0 ≤ -a) := by omega
  have h2 : 0 ≤ a := by omega
  simp [h1, h2]
  intro h; omega

/-- The tie `2.5` (`a = 5, D = 2`) goes to 3 under half-away and to 2 under `rint`: the two Q8_0
variants genuinely differ, which is why each is used against its own reference. -/
theorem tie_rules_differ : roundAway 5 2 = 3 ∧ roundAway (-5) 2 = -3 := by decide

-- § nearest_int

/-- Bit pattern of the FP32 sum `1.5·2^23 + n` for an integer `n` with `|n| < 2^22`: sign 0, biased
exponent `150`, mantissa field `m = 2^22 + n`. -/
def nearestIntBits (m : Nat) : Nat := 150 * 2 ^ 23 + m

/-- `nearest_int`'s decode of a bit pattern: `(i & 0x7fffff) - 0x400000`. -/
def decodeNearest (i : Nat) : Int := ((i &&& 0x7fffff : Nat) : Int) - 0x400000

/-- The sum lies in the binade `[2^23, 2^24)`, so its ulp is 1 and its mantissa field is `sum − 2^23`. -/
theorem nearest_binade (n : Int) (h1 : -2 ^ 22 ≤ n) (h2 : n < 2 ^ 22) :
    2 ^ 23 ≤ 12582912 + n ∧ 12582912 + n < 2 ^ 24 ∧ 12582912 + n - 2 ^ 23 = 2 ^ 22 + n := by
  refine ⟨?_, ?_, ?_⟩ <;> simp at * <;> omega

/-- **`nearest_int` recovers the integer**: masking the low 23 bits and subtracting `2^22` returns `n`. -/
theorem nearest_int_correct (n : Int) (h1 : -2 ^ 22 ≤ n) (h2 : n < 2 ^ 22) :
    decodeNearest (nearestIntBits (n + 2 ^ 22).toNat) = n := by
  have hm : (n + 2 ^ 22).toNat < 2 ^ 23 := by simp at *; omega
  unfold decodeNearest nearestIntBits
  have hmask : (0x7fffff : Nat) = 2 ^ 23 - 1 := by decide
  rw [hmask, Nat.and_two_pow_sub_one_eq_mod, Nat.add_comm, Nat.add_mul_mod_self_right,
    Nat.mod_eq_of_lt hm]
  simp at *
  omega

-- § Q8_K block sums

/-- A list whose entries lie in `[lo, hi]` sums into `[len·lo, len·hi]`. -/
theorem sum_bounds (lo hi : Int) : ∀ l : List Int, (∀ v ∈ l, lo ≤ v ∧ v ≤ hi) →
    l.length * lo ≤ l.sum ∧ l.sum ≤ l.length * hi
  | [], _ => by simp
  | x :: xs, h => by
    have hx := h x (by simp)
    have ih := sum_bounds lo hi xs (fun v hv => h v (by simp [hv]))
    simp only [List.length_cons, List.sum_cons]
    push_cast
    constructor <;> (rw [Int.add_mul, Int.one_mul]; omega)

/-- **`bsums` cannot overflow `int16`**: sixteen int8 values sum into `[-2048, 2032]`. -/
theorem bsum_fits_int16 (l : List Int) (hl : l.length = 16) (h : ∀ v ∈ l, -128 ≤ v ∧ v ≤ 127) :
    -32768 ≤ l.sum ∧ l.sum ≤ 32767 ∧ -2048 ≤ l.sum ∧ l.sum ≤ 2032 := by
  have := sum_bounds (-128) 127 l h
  rw [hl] at this
  omega

/-- `qs` has an upper clamp only (`min(127, v)`); the result is at most 127 and unchanged below it. -/
theorem q8_K_upper_clamp (v : Int) : min 127 v ≤ 127 ∧ (v ≤ 127 → min 127 v = v) := by omega

-- § host checks

/-- Q8_0 launches need `n % 32 = 0`; Q8_K launches need `n % 256 = 0`, which is strictly stronger.
The 640-wide shared-expert axis is Q8_0-quantisable and not Q8_K-quantisable. -/
theorem launch_divisibility : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 ∧ (∀ n, n % 256 = 0 → n % 32 = 0) := by
  refine ⟨by decide, by decide, ?_⟩
  intro n h; omega

/-- Grids cover every block: Q8_0 with 128 threads, Q8_K with 64. -/
theorem grids_cover (nb b : Nat) (h : b < nb) :
    b < (nb + 127) / 128 * 128 ∧ b < (nb + 63) / 64 * 64 := by omega

end Strata.QuantAct
