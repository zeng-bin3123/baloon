/-!
# MODULE: s2_gemv_fast_experiment
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
A measured-neutral/negative S2 (2-bit) GEMV experiment: block per output row, optional shared-memory staging of
the activation, and a constant-memory byte -> four-code lookup table. Kept as a record, not a fast path.

## Source semantics
Codes are 2 bits LSB-first, four per byte; the LUT entry for byte b lists (code_k - 1) for k = 0..3. One scale per
64 elements, so quad q (elements 4q..4q+3) uses scale q >> 4. Shared staging needs n_in <= 4096 halves; the
partial-sum array begins right after the 4096 halves (2048 floats). n_in must be a multiple of 64.

## Form template identifiers
lut -> "byte decode table"; scaleIdx -> "per-64 group scale index"; smemBytes -> "dynamic shared size".
-/
namespace Strata.S2GemvFast

-- § lookup table
/-- LUT value: the k-th 2-bit field of byte b, minus the bias 1. -/
def lut (b k : Nat) : Int := ((b / 4 ^ k) % 4 : Nat) - 1

/-- Every LUT value lies in {-1, 0, 1, 2}. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  have h := Nat.mod_lt (b / 4 ^ k) (by decide : 4 > 0)
  unfold lut
  omega

/-- The four fields of a byte reassemble the byte. -/
theorem byte_fields_reassemble (b : Nat) (hb : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- Scale index of quad q (elements 4q..4q+3, 64 per group) is q >> 4. -/
theorem scale_index_quad (q : Nat) : (q * 4) / 64 = q / 16 := by
  omega

-- § shared memory budget
/-- Max halves staged in shared memory. -/
def maxSharedHalves : Nat := 4096

/-- Dynamic shared bytes: staged halves (if any) plus one float per thread. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat :=
  (if stage then maxSharedHalves * 2 else 0) + tpr * 4

/-- The partial array offset (`MAX/2` floats) is exactly the end of the staged halves. -/
theorem partial_offset_after_halves : (maxSharedHalves / 2) * 4 = maxSharedHalves * 2 := by
  decide

/-- Staging a 2560-element activation fits the staging area. -/
theorem stage_fits_2560 : 2560 ≤ maxSharedHalves := by
  decide

/-- With staging, the shared size is at least the staged halves. -/
theorem smem_with_stage (tpr : Nat) : 8192 ≤ smemBytes true tpr := by
  unfold smemBytes maxSharedHalves
  simp

/-- Multiple of 64 implies multiple of 4 (the quad requirement is redundant). -/
theorem mult64_implies_mult4 (n : Nat) (h : n % 64 = 0) : n % 4 = 0 := by
  omega

/-- Quads cover the row exactly. -/
theorem quads_cover (n : Nat) (h : n % 4 = 0) : n / 4 * 4 = n := by
  omega

end Strata.S2GemvFast
