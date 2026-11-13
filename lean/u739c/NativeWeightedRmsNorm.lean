/-!
# MODULE: native_weighted_rms_norm
SOURCES: src/kernels/cuda/native_gr_norm.cu
The llama.cpp-pinned weighted F32 RMS norm used by the native gated-residual path:
one block per row, block size 256 or 1024 chosen by row width, block reduction where every
warp repeats the final xor reduction, then `out = rsqrt(mean + eps) * x * gamma`.
-/

namespace Ref.NativeGrNorm

/-- Block size selected by `native_gr_rms_norm_weighted`. -/
def blockSize (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- The block size is one of the two template instances. -/
theorem blockSize_cases (n : Nat) : blockSize n = 256 ∨ blockSize n = 1024 := by
  unfold blockSize; split <;> simp

/-- The warp count `B/32` never exceeds the 32 shared partial slots. -/
theorem warp_slots_fit (n : Nat) : blockSize n / 32 ≤ 32 := by
  rcases blockSize_cases n with h | h <;> rw [h] <;> decide

/-- Block sizes are whole warps. -/
theorem blockSize_warps (n : Nat) : blockSize n % 32 = 0 := by
  rcases blockSize_cases n with h | h <;> rw [h] <;> decide

/-- Column visited by thread `tid` at step `m`. -/
def colVisit (B tid m : Nat) : Nat := tid + m * B

/-- Every column is visited by thread `col % B` at step `col / B`. -/
theorem colVisit_cover (B col : Nat) (hB : 0 < B) :
    colVisit B (col % B) (col / B) = col ∧ col % B < B := by
  refine ⟨?_, Nat.mod_lt _ hB⟩
  unfold colVisit
  have := Nat.mod_add_div col B
  rw [Nat.mul_comm] at this
  exact this

/-- No column is visited by two (thread, step) pairs. -/
theorem colVisit_unique {B t m t' m' : Nat} (ht : t < B) (ht' : t' < B)
    (h : colVisit B t m = colVisit B t' m') : t = t' ∧ m = m' := by
  unfold colVisit at h
  have hB : 0 < B := by omega
  have m1 : (t + m * B) % B = t := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
  have m2 : (t' + m' * B) % B = t' := by rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht']
  have htt : t = t' := by rw [← m1, ← m2, h]
  subst htt
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hB (by omega)⟩

/-- Row `r`, column `c` of a contiguous `n_rows × n_cols` buffer is in range. -/
theorem row_addr_lt {r c nRows nCols : Nat} (hr : r < nRows) (hc : c < nCols) :
    r * nCols + c < nRows * nCols := by
  have : (r + 1) * nCols ≤ nRows * nCols := Nat.mul_le_mul_right _ (by omega)
  rw [Nat.add_mul] at this; omega

/-- Output element in exact arithmetic: `scale * x * gamma`. -/
def outElem (scale x gamma : Int) : Int := scale * x * gamma

/-- A unit gamma leaves the plain RMS-normalised value. -/
theorem outElem_unit_gamma (scale x : Int) : outElem scale x 1 = scale * x := by
  simp [outElem]

/-- The sum of squares is non-negative, so `mean + eps > 0` whenever `eps > 0`. -/
theorem sumsq_nonneg (xs : List Int) : 0 ≤ (xs.map (fun x => x * x)).foldr (· + ·) 0 := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map, List.foldr]
    have : 0 ≤ x * x := by
      rcases Int.le_total 0 x with h | h
      · exact Int.mul_nonneg h h
      · have h' : 0 ≤ -x := by omega
        have := Int.mul_nonneg h' h'
        rwa [Int.neg_mul_neg] at this
    omega

/-- Host precondition (finiteness of epsilon abstracted as a flag). -/
def argsOk (nCols nRows : Int) (epsFinite : Bool) (eps : Int) (aligned : Bool) : Bool :=
  0 < nCols && 0 < nRows && epsFinite && 0 ≤ eps && aligned

/-- A passing call has positive dimensions, so the launch grid is non-empty. -/
theorem argsOk_dims {c r e : Int} {f a : Bool} (h : argsOk c r f e a = true) : 0 < c ∧ 0 < r := by
  unfold argsOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1.1, h.1.1.1.2⟩

end Ref.NativeGrNorm

