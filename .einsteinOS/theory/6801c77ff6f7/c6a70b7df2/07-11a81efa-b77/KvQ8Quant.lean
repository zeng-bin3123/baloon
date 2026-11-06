/-!
# MODULE: kv_q8_quant
SOURCES: src/kernels/cuda/kv_q8.cu
INT8 KV cache with one fp16 scale per 64-value group: append quantizes against the stored scale,
gather dequantizes four values per thread.

## Source semantics
* scale = fp16(amax / 127); code = clamp(round(x / scale), -127, 127); scale 0 gives code 0.
* scale index = row * (head_dim / 64) + d / 64.
* Gather handles 4 consecutive values per thread; 64 is a multiple of 4, so the 4 share a group.

## Form template identifiers
clamp127 <- the `q < -127 ? -127 : q > 127 ? 127 : q` clamp, scaleIdx <- row * groups + g.
-/

namespace Strata.KvQ8

-- § clamp

/-- The symmetric int8 clamp used by `kv_append_q8_kernel`. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Codes stay in [-127, 127] (so -128 is never produced). -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; split <;> (try split) <;> omega

/-- In-range codes are unchanged. -/
theorem clamp127_id {q : Int} (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127; split <;> (try split) <;> omega

/-- Clamping is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q :=
  clamp127_id (clamp127_bounds q).1 (clamp127_bounds q).2

-- § scale indexing

/-- Scale index of group `g` of row `row`. -/
def scaleIdx (row groups g : Nat) : Nat := row * groups + g

/-- A valid group has a scale slot inside its row's range. -/
theorem scaleIdx_lt {row groups g : Nat} (hg : g < groups) :
    scaleIdx row groups g < (row + 1) * groups := by
  unfold scaleIdx; rw [Nat.add_mul, Nat.one_mul]; omega

/-- The real head_dim 256 has four groups of 64. -/
theorem real_groups : 256 / 64 = 4 := by decide

/-- Every dimension `d < 256` lands in one of the four groups. -/
theorem group_lt {d : Nat} (h : d < 256) : d / 64 < 4 := by omega

/-- The four values one gather thread handles share one scale group. -/
theorem gather_group_uniform {q4 j : Nat} (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

/-- 64 threads per block, two warps of 32. -/
theorem block_two_warps : 64 / 32 = 2 := by decide

end Strata.KvQ8
