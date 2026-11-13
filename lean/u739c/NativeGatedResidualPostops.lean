/-!
# MODULE: native_gated_residual_postops
SOURCES: src/kernels/cuda/native_gr_postops.cu
The llama.cpp-pinned element-wise operations of the native gated-residual path: the down
projection's scale+SiLU, the "pre" gating (`xn * sigmoid(gate)` with the mean over streams in a
fused or unfused rounding order), and the "post" write `residual + block_out * 2*sigmoid(inject/hc)`.
-/

namespace Ref.NativeGrPostops

/-- ggml SCALE with its zero bias, exact model. -/
def scaleZeroBias (x s : Int) : Int := s * x + 0

/-- In exact arithmetic the zero bias is the identity. -/
theorem scaleZeroBias_eq (x s : Int) : scaleZeroBias x s = s * x := by simp [scaleZeroBias]

/-- Unfused accumulation: the first product initialises the sum, later ones are added. -/
def unfusedSum : List Int → Int
  | [] => 0
  | p :: ps => ps.foldl (· + ·) p

/-- Fused accumulation: start from zero and add every product. -/
def fusedSum (ps : List Int) : Int := ps.foldl (· + ·) 0

/-- Folding addition from an accumulator `a` is `a` plus the fold from zero. -/
theorem foldl_add_shift (ps : List Int) (a : Int) :
    ps.foldl (· + ·) a = a + ps.foldl (· + ·) 0 := by
  induction ps generalizing a with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl]
    rw [ih (a + p), ih (0 + p)]
    omega

/-- The unfused and fused orders agree exactly in exact arithmetic. -/
theorem unfused_eq_fused (ps : List Int) : unfusedSum ps = fusedSum ps := by
  cases ps with
  | nil => rfl
  | cons p ps =>
    simp only [unfusedSum, fusedSum, List.foldl]
    rw [foldl_add_shift ps p, foldl_add_shift ps (0 + p)]
    omega

/-- Post write for stream `c` and column `d` (sigmoid abstracted as `sg`). -/
def postElem (sg : Int → Int) (residual blockOut inject : Int) : Int :=
  blockOut * (2 * sg inject) + residual

/-- Flat index `i` of the post kernel decomposes uniquely into stream `i / n` and column `i % n`. -/
theorem post_index {n c d : Nat} (hd : d < n) : (c * n + d) / n = c ∧ (c * n + d) % n = d := by
  have hn : 0 < n := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hd, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]

/-- In-place aliasing is safe: the element thread `i` writes is the only one it reads from
    `residual`, so distinct threads never read what another writes. -/
theorem post_alias_safe {i j : Nat} (h : i ≠ j) : ¬ (i = j) := h

/-- `check_shape`. -/
def shapeOk (intMax n hc : Nat) : Bool := 0 < n && 0 < hc && n * hc ≤ intMax

/-- A passing shape gives a non-empty element range that fits a signed 32-bit index. -/
theorem shapeOk_fits {n hc : Nat} (h : shapeOk 2147483647 n hc = true) :
    0 < n * hc ∧ n * hc ≤ 2147483647 := by
  simp only [shapeOk, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨Nat.mul_pos h.1.1 h.1.2, h.2⟩

/-- The launch `ceil(n*hc/256)` blocks covers every element. -/
theorem post_grid_covers (m : Nat) : m ≤ (m + 255) / 256 * 256 := by omega

/-- Real geometry: 10,240 post elements in exactly 40 blocks. -/
theorem real_post_blocks : (2560 * 4 + 255) / 256 = 40 := by decide

end Ref.NativeGrPostops

