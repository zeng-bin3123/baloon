/-!
# MODULE: kv_q8_quant
INT8 KV-cache quantisation, 64-value groups with an fp16 scale (`src/kernels/cuda/kv_q8.cu`):
append quantises one cell per (kv head, group, K/V) block; gather dequantises to fp16 scratch
four values per thread.

## Source semantics
* Codes are `round(x / scale)` clamped to `[-127, 127]`; a zero scale gives code 0.
* Group index of dimension `d` is `d / 64`; scales are stored at `row * groups + g`.
* The gather thread index `i = id*(kv*per) + h*per + q4` decomposes by div/mod, and the
  destination index `(id*kv + h)*per + q4` equals `i`.
* The four values a thread dequantises share one scale because 64 is a multiple of 4.

## Form template identifiers
* `clamp127`   <- `q < -127 ? -127 : (q > 127 ? 127 : q)`
* `code`       <- `if (sf > 0) ... else 0`
* `groupOf`    <- `d / KV_Q8_GROUP`
* `scaleIdx`   <- `row * groups + g`
-/

namespace KvQ8Quant

-- § clamping

/-- Clamp a rounded quotient into the int8 code range `[-127, 127]`. -/
def clamp127 (x : Int) : Int := if x < -127 then -127 else if x > 127 then 127 else x

/-- Codes stay in `[-127, 127]` (so `-128` is never produced). -/
theorem clamp127_bounds (x : Int) : -127 ≤ clamp127 x ∧ clamp127 x ≤ 127 := by
  by_cases h1 : x < -127
  · unfold clamp127
    rw [if_pos h1]
    omega
  · by_cases h2 : x > 127
    · unfold clamp127
      rw [if_neg h1, if_pos h2]
      omega
    · unfold clamp127
      rw [if_neg h1, if_neg h2]
      omega

/-- In-range values are unchanged. -/
theorem clamp127_id (x : Int) (h1 : -127 ≤ x) (h2 : x ≤ 127) : clamp127 x = x := by
  unfold clamp127
  rw [if_neg (by omega), if_neg (by omega)]

/-- Clamping is idempotent. -/
theorem clamp127_idem (x : Int) : clamp127 (clamp127 x) = clamp127 x := by
  have h := clamp127_bounds x
  exact clamp127_id _ h.1 h.2

/-- Quantised code: a zero stored scale gives code zero, otherwise the clamped rounded quotient. -/
def code (scalePos : Bool) (r : Int) : Int := if scalePos then clamp127 r else 0

/-- Every code is in range. -/
theorem code_bounds (b : Bool) (r : Int) : -127 ≤ code b r ∧ code b r ≤ 127 := by
  cases b
  · unfold code
    rw [if_neg (by decide)]
    omega
  · unfold code
    rw [if_pos rfl]
    exact clamp127_bounds r

-- § group addressing

/-- Group of a dimension. -/
def groupOf (d : Nat) : Nat := d / 64

/-- With `head_dim = 64 * groups`, every dimension's group is in range. -/
theorem groupOf_lt (d headDim groups : Nat) (hd : headDim = 64 * groups) (h : d < headDim) :
    groupOf d < groups := by
  unfold groupOf
  omega

/-- Scale slot of (row, group). -/
def scaleIdx (row groups g : Nat) : Nat := row * groups + g

/-- A value index inside its row stays inside that row. -/
theorem value_index_in_row (row headDim groups g t : Nat) (hd : headDim = 64 * groups)
    (hg : g < groups) (ht : t < 64) : row * headDim + g * 64 + t < row * headDim + headDim := by
  omega

/-- The four values of one gather thread share a group, hence one scale. -/
theorem quad_shares_group (q4 j : Nat) (hj : j < 4) : (q4 * 4 + j) / 64 = (q4 * 4) / 64 := by
  omega

-- § gather decomposition

/-- Arithmetic core of the decomposition. -/
theorem decomp_aux (i kv per A R B C : Nat) (e1 : kv * per * A + R = i)
    (e2 : per * B + C = R) : (A * kv + B) * per + C = i := by
  have e3 : (A * kv + B) * per = kv * per * A + per * B := by
    rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm A (kv * per), Nat.mul_comm B per]
  omega

/-- The (id, head, quad) split reassembles to the flat thread index. -/
theorem gather_decomp (i kv per : Nat) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i :=
  decomp_aux i kv per _ _ _ _ (Nat.div_add_mod i (kv * per))
    (Nat.div_add_mod (i % (kv * per)) per)

/-- The destination index is the thread index itself: the write is a straight, coalesced copy. -/
theorem dst_eq_flat (id h q kv per : Nat) :
    (id * kv + h) * per + q = id * (kv * per) + h * per + q := by
  rw [Nat.add_mul, Nat.mul_assoc]

end KvQ8Quant
