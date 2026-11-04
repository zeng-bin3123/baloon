/-!
# MODULE: cuda_quantize_activation
Block layouts, launch arithmetic and rounding rules of the GPU activation quantizers (Q8_0, the CPU-matching
"scaled" Q8_0 variant, Q8_K) and their dequantizers.

## Source semantics
* Q8_0 block = fp16 `d` + 32 int8 = 34 bytes, 32 elements; Q8_K block = fp32 `d` + 256 int8 + 16 int16 `bsums`
  = 292 bytes, 256 elements.
* Plain Q8_0 divides in double and rounds half to even; the scaled variant multiplies by the reciprocal and
  rounds half away from zero (the CPU rule) and clamps to ±127.
* Q8_K's `nearest_int` is the 1.5·2^23 magic-number trick: for results in `[-2^22, 2^22)` the low 23 bits of the
  float bits minus `0x400000` give the rounded integer.
* Q8_K `bsums[j]` sums `qs[16 j .. 16 j + 15]`; the quantized values satisfy `|v| ≤ 127` so the int8 cast is
  safe; only an upper clamp `min(127, v)` exists in the source.
* Launch: `n` must be a multiple of the block length; `grid = ceil(nb / threads)` covers every block.

## Form template identifiers
quantize_q8_0_kernel → q8_0_quantizer, quantize_q8_0_scaled_kernel → q8_0_cpu_matching_quantizer,
quantize_q8_K_kernel → q8_k_quantizer, nearest_int_dev → nearest_int_magic, dequant_*_kernel → block_dequantizer.
-/

namespace Strata.Cuda.QuantizeAct

-- § layouts

/-- Q8_0 block bytes: 2 (fp16 d) + 32. -/
theorem q8_0_block_bytes : 34 = 2 + 32 := by decide

/-- Q8_K block bytes: 4 (fp32 d) + 256 + 32 (16 × int16). -/
theorem q8_k_block_bytes : 292 = 4 + 256 + 16 * 2 := by decide

/-- Q8_K sub-field offsets: `d` at 0, `qs` at 4, `bsums` at `4 + 256`; the last bsum ends at 292. -/
theorem q8_k_offsets : 4 + 256 = 260 ∧ 260 + 16 * 2 = 292 := by decide

/-- `bsums[j]` over `qs[16 j + ii]` with `j, ii < 16` covers indices below 256 and each index once. -/
theorem bsum_index_bounds (j ii : Nat) (hj : j < 16) (hi : ii < 16) :
    16 * j + ii < 256 ∧ (16 * j + ii) / 16 = j ∧ (16 * j + ii) % 16 = ii := by omega

/-- Block `b` starts at `b * 34` (resp. `b * 292`) and its last byte is below `(b+1) * 34` (resp. 292). -/
theorem block_ranges (b : Nat) : b * 34 + 33 < (b + 1) * 34 ∧ b * 292 + 291 < (b + 1) * 292 := by omega

/-- Elements of block `b` are `x[32 b .. 32 b + 31]`, inside `n = 32 * nb` for `b < nb`. -/
theorem q8_0_elements_in_range (b nb i : Nat) (hb : b < nb) (hi : i < 32) : b * 32 + i < nb * 32 := by omega

-- § launch arithmetic

/-- A multiple-of-32 element count gives exactly `n / 32` blocks. -/
theorem n_blocks_exact (n : Nat) (h : n % 32 = 0) : n / 32 * 32 = n := by omega

/-- Same for Q8_K with 256. -/
theorem n_blocks_exact_k (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Grid for 128 threads per block covers every block. -/
theorem grid128_covers (nb : Nat) : nb ≤ (nb + 127) / 128 * 128 := by omega

/-- Grid for 64 threads per block (Q8_K: one block per thread). -/
theorem grid64_covers (nb : Nat) : nb ≤ (nb + 63) / 64 * 64 := by omega

-- § rounding rules

/-- Round half away from zero of `n / d`, `d > 0` (the CPU rule `t + copysign(0.5, t)` then truncation). -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((-2 * n + d) / (2 * d))

/-- Round half to even of `n / d` (`rint` in the plain Q8_0 kernel). -/
def rint (n d : Int) : Int :=
  let q := n / d
  let r := n % d
  if 2 * r < d then q else if 2 * r > d then q + 1 else if q % 2 = 0 then q else q + 1

/-- The plain kernel's clamp `[-128, 127]`. -/
def clampPlain (v : Int) : Int := if v > 127 then 127 else if v < -128 then -128 else v

/-- The scaled kernel's clamp `[-127, 127]`. -/
def clampScaled (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- The Q8_K upper-only clamp. -/
def clampUpper (v : Int) : Int := if v < 127 then v else 127

/-- Ties differ between the two Q8_0 kernels: `rint` goes to even, `rha` away from zero. -/
theorem tie_rules_differ :
    rint 1 2 = 0 ∧ rha 1 2 = 1 ∧ rint 3 2 = 2 ∧ rha 3 2 = 2 ∧ rint (-1) 2 = 0 ∧ rha (-1) 2 = -1 ∧
      rint 5 2 = 2 ∧ rha 5 2 = 3 := by
  decide

/-- Plain clamp stays in int8. -/
theorem clampPlain_int8 (v : Int) : -128 ≤ clampPlain v ∧ clampPlain v ≤ 127 := by
  unfold clampPlain
  omega

/-- Scaled clamp stays in the symmetric range, so `-128` is never produced by the CPU-matching rule. -/
theorem clampScaled_symmetric (v : Int) : -127 ≤ clampScaled v ∧ clampScaled v ≤ 127 := by
  unfold clampScaled
  omega

/-- The upper clamp bounds the value by 127. -/
theorem clampUpper_le (v : Int) : clampUpper v ≤ 127 := by
  unfold clampUpper
  split <;> omega

/-- The CPU rule never exceeds 127 when `|x|/s ≤ 127` (`0 ≤ n ≤ 127 d`), so the clamp is a no-op there. -/
theorem rha_le_127 (n d : Int) (hd : 0 < d) (h0 : 0 ≤ n) (h : n ≤ 127 * d) : rha n d ≤ 127 := by
  unfold rha
  rw [if_pos h0]
  have : (2 * n + d) / (2 * d) < 128 := by
    rw [Int.ediv_lt_iff_lt_mul (by omega)]
    omega
  omega

-- § nearest_int magic number

/-- The magic-number trick: with `val = 12582912 + r` and `r ∈ [-2^22, 2^22)`, `val` lies in `[2^23, 2^24)`, its
float bits are `150·2^23 + (val - 2^23)`, and `(bits mod 2^23) - 0x400000 = r`. -/
theorem nearest_int_magic (r : Int) (h1 : -4194304 ≤ r) (h2 : r < 4194304) :
    ((150 * 8388608 + (12582912 + r - 8388608)) % 8388608) - 4194304 = r := by
  omega

/-- The same value lies in the binade `[2^23, 2^24)` where the exponent field is constant (150). -/
theorem magic_in_binade (r : Int) (h1 : -4194304 ≤ r) (h2 : r < 4194304) :
    8388608 ≤ 12582912 + r ∧ 12582912 + r < 16777216 := by omega

/-- Quantized values from `iscale = -127/max` satisfy `|v| ≤ 127` (given `|x_j| ≤ |max|`): in units of `1/d`,
`-127 ≤ rounded ≤ 127`, so the int8 cast cannot wrap. -/
theorem q8k_cast_safe (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : -128 ≤ clampUpper v ∧ clampUpper v ≤ 127 := by
  unfold clampUpper
  split <;> omega

-- § zero-block handling

/-- A zero-amax Q8_K block is entirely zero: `d = 0`, 256 zero `qs`, 16 zero `bsums`, whose dequantized
values are all `0 * d = 0` and whose bsums are sums of zeros. -/
theorem zero_block_bsum (xs : List Int) (h : ∀ x ∈ xs, x = 0) : xs.sum = 0 := by
  induction xs with
  | nil => rfl
  | cons a as ih =>
    have ha : a = 0 := h a (by simp)
    have has : ∀ x ∈ as, x = 0 := fun x hx => h x (by simp [hx])
    simp [List.sum_cons, ha, ih has]

end Strata.Cuda.QuantizeAct
