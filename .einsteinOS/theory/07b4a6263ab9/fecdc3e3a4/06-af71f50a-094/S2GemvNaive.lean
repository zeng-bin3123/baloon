/-!
# MODULE: s2_gemv_naive
SOURCES: src/kernels/cuda/s2_gemv.cu
The reference S2 GEMV: one thread per output row, dequantising on the fly, FP32 accumulation over
groups of 64 in natural order, FP16 activations.

## Source semantics
* Host: `n_in, n_out > 0` else no-op; `n_in % 64 = 0` else exit; grid `⌈n_out/128⌉` of 128 threads.
* Row `o` uses `nb = n_in / 64` groups: codes at `o·nb·16 + b·16 + (j >> 2)`, slot `(j & 3)·2`,
  scale `o·nb + b`, activation `b·64 + j`.
* These agree with the plane-wide layout (element `e` at byte `e/4` of row `o`'s `n_in/4` bytes), rows
  occupy disjoint code and scale ranges, and every activation index is used exactly once per row.
* Factor `(code - 1) ∈ [-1, 2]`, applied in the integer domain.

## Form template identifiers
* `QK = 64`;  `codeAddr`, `scaleAddr`, `actAddr` ↔ the three pointer expressions
-/

namespace Strata.S2Gemv

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2

-- § addressing

/-- Byte holding element `j` of group `b` of row `o`: `c + b*(QK/4) + (j >> 2)` with `c = codes + o*nb*16`. -/
def codeAddr (nb o b j : Nat) : Nat := o * nb * 16 + b * 16 + (j >>> 2)

/-- Scale of group `b` of row `o`. -/
def scaleAddr (nb o b : Nat) : Nat := o * nb + b

/-- Activation consumed by element `j` of group `b`. -/
def actAddr (b j : Nat) : Nat := b * 64 + j

/-- `j >> 2` and `(j & 3) * 2` are the plane convention's byte and bit offset. -/
theorem shift_mask (j : Nat) : j >>> 2 = j / 4 ∧ (j &&& 3) * 2 = (j % 4) * 2 := by
  refine ⟨by simp [Nat.shiftRight_eq_div_pow], by rw [and3]⟩

/-- **Per-group addressing equals the plane layout**: with `n_in = 64 nb`, element `b·64 + j` of row `o`
is byte `o·(n_in/4) + (b·64 + j)/4`. -/
theorem codeAddr_plane (nb o b j : Nat) (hj : j < 64) :
    codeAddr nb o b j = o * (64 * nb / 4) + (b * 64 + j) / 4 := by
  unfold codeAddr
  rw [(shift_mask j).1]
  have h : 64 * nb / 4 = nb * 16 := by omega
  rw [h, ← Nat.mul_assoc]
  omega

/-- Rows own disjoint code ranges `[o·nb·16, (o+1)·nb·16)`. -/
theorem codeAddr_in_row (nb o b j : Nat) (hb : b < nb) (hj : j < 64) :
    o * nb * 16 ≤ codeAddr nb o b j ∧ codeAddr nb o b j < (o + 1) * nb * 16 := by
  unfold codeAddr
  rw [(shift_mask j).1, Nat.add_mul, Nat.add_mul, Nat.one_mul]
  omega

/-- Scales: row `o` group `b` stays inside the `n_out · nb` scale array. -/
theorem scaleAddr_in_range (nb n_out o b : Nat) (ho : o < n_out) (hb : b < nb) :
    scaleAddr nb o b < n_out * nb := by
  unfold scaleAddr
  have : (o + 1) * nb ≤ n_out * nb := Nat.mul_le_mul_right nb ho
  rw [Nat.add_mul, Nat.one_mul] at this
  omega

/-- Activations: `(b, j) ↦ 64 b + j` enumerates `0 … n_in − 1` exactly once. -/
theorem actAddr_bijective (nb e : Nat) (he : e < 64 * nb) :
    e / 64 < nb ∧ e % 64 < 64 ∧ actAddr (e / 64) (e % 64) = e := by
  unfold actAddr; omega

/-- …and injectively. -/
theorem actAddr_injective (b₁ b₂ j₁ j₂ : Nat) (h₁ : j₁ < 64) (h₂ : j₂ < 64)
    (h : actAddr b₁ j₁ = actAddr b₂ j₂) : b₁ = b₂ ∧ j₁ = j₂ := by
  unfold actAddr at h; omega

-- § decode and launch

/-- The code factor `((byte >> 2k) & 3) - 1` is in `[-1, 2]`. -/
theorem factor_range (byte s : Nat) : -1 ≤ (((byte >>> s) &&& 3 : Nat) : Int) - 1 ∧
    (((byte >>> s) &&& 3 : Nat) : Int) - 1 ≤ 2 := by
  rw [and3]; omega

/-- The host's `n_in % 64` check is exactly what makes `nb = n_in/64` groups cover the row. -/
theorem groups_cover (n_in : Nat) (h : n_in % 64 = 0) : 64 * (n_in / 64) = n_in := by omega

/-- Without it a tail is silently skipped: 100 elements give one group of 64. -/
theorem tail_dropped_without_check : 64 * (100 / 64) = 64 := by decide

/-- One thread per row: `⌈n_out/128⌉` blocks of 128 reach every row. -/
theorem rows_covered (n_out o : Nat) (h : o < n_out) : o / 128 < (n_out + 127) / 128 := by omega

end Strata.S2Gemv
