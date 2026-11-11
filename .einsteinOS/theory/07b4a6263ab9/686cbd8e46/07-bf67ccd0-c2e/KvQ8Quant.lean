/-!
# MODULE: kv_q8_quant
SOURCES: src/kernels/cuda/kv_q8.cu
Symmetric int8 KV quantization with one fp16 scale per 64-value group (`kv_append_q8_kernel`) and the
dequantizing gather (`kv_gather_q8_kernel`).

## Source semantics
* Code is `clamp(round(x / sf), -127, 127)` against the STORED scale, and `0` when the scale is `0`.
* Codes are symmetric: `-128` is never produced, so every code fits `int8` and negation is closed.
* Scale of element `d` of row `row` is at `row * groups + d / 64`; a 4-value gather lane never
  straddles a group, and every scale index stays inside its row.

## Form template identifiers
`KV_Q8_GROUP` → `64`, clamp → `clampCode`, `q` → `code`, `k_scale/v_scale` index → `scaleIdx`,
`q4` → gather lane.
-/

namespace Strata.KvQ8

-- § codes

/-- The clamp to the symmetric int8 range. -/
def clampCode (q : Int) : Int := max (-127) (min 127 q)

/-- The stored code: clamped rounding when the stored scale is positive, else zero. -/
def code (scalePos : Bool) (rounded : Int) : Int := if scalePos then clampCode rounded else 0

/-- Clamped codes lie in `[-127, 127]`. -/
theorem clampCode_bounds (q : Int) : -127 ≤ clampCode q ∧ clampCode q ≤ 127 := by
  unfold clampCode
  omega

/-- Values already in range are untouched. -/
theorem clampCode_of_mem (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampCode q = q := by
  unfold clampCode
  omega

/-- Clamping is idempotent. -/
theorem clampCode_idem (q : Int) : clampCode (clampCode q) = clampCode q := by
  obtain ⟨h1, h2⟩ := clampCode_bounds q
  exact clampCode_of_mem _ h1 h2

/-- The clamp commutes with negation: the code range is symmetric. -/
theorem clampCode_neg (q : Int) : clampCode (-q) = -clampCode q := by
  unfold clampCode
  omega

/-- Every stored code fits a signed byte and is never `-128`. -/
theorem code_fits_int8 (b : Bool) (r : Int) : -128 < code b r ∧ code b r ≤ 127 := by
  cases b with
  | false => simp [code]
  | true =>
    have := clampCode_bounds r
    simp only [code, if_true]
    omega

/-- A zero scale stores a zero code. -/
theorem code_zero_scale (r : Int) : code false r = 0 := rfl

-- § scale layout

/-- The scale slot of element `d` in row `row` with `groups = head_dim / 64`. -/
def scaleIdx (groups row d : Nat) : Nat := row * groups + d / 64

/-- A dimension inside the row selects a group inside the row. -/
theorem group_lt (groups d : Nat) (hd : d < groups * 64) : d / 64 < groups := by
  omega

/-- Scale slots of row `row` lie in `[row*groups, (row+1)*groups)`. -/
theorem scaleIdx_in_row (groups row d : Nat) (hd : d < groups * 64) :
    row * groups ≤ scaleIdx groups row d ∧ scaleIdx groups row d < (row + 1) * groups := by
  have hg := group_lt groups d hd
  unfold scaleIdx
  rw [Nat.succ_mul]
  omega

/-- The four values a gather lane dequantizes share one scale. -/
theorem lane_one_group (q4 j : Nat) (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

/-- The 64 threads of an append block cover exactly one group. -/
theorem append_thread_group (g t : Nat) (ht : t < 64) : (g * 64 + t) / 64 = g := by
  omega

end Strata.KvQ8
