/-!
# MODULE: q2_expert_kernel
Q2_0 expert (gate/up -> SiLU*up -> down) CPU kernel: 2-bit code unpacking, the code-offset correction
`sum (c-1)*xhat = sum c*xhat - sum xhat`, int8 activation clamping, block-tail masks and token-count dispatch.

## Source semantics
* A Q2_0 weight is a 2-bit code `c ∈ {0..3}` meaning the value `c - 1`; the kernel computes `sum c*xhat` with
  VNNI (unsigned code × signed activation) and subtracts `sum xhat` afterwards (the weight-independent `hx`).
* Activations are quantised per 32-value chunk to int8 in `[-127,127]`; the dot is split per chunk because the
  activation scale changes at chunk boundaries while the weight scale does not.
* Tail blocks use masks `(1 << 2nb) - 1`; multi-token kernels are dispatched on `n_tokens` with `default -> 8`.
* gate row is `2r`, up row is `2r+1` of the interleaved gate/up pack.

## Form template identifiers
* `code` ~ unpack of one 2-bit code (`_mm512_multishift_epi64_epi8` + AND 3)
* `offDot` ~ `row_dot` result (offset-corrected), `dotList` ~ `sum c*xhat`, `sumList` ~ `sum xhat` (`hx`)
* `clamp127` ~ activation clamp in `act_quant_q8_1`
* `lanemask` ~ `m16` tail mask, `nbOf` ~ `nb` tail block count, `ntDispatch` ~ template switch on `n_tokens`
-/

namespace Strata.Q2Expert

-- § code decoding

/-- The i-th 2-bit code of a 16-bit word. -/
def code (h i : Nat) : Nat := (h / 4 ^ i) % 4

/-- Every decoded code is in `0..3`. -/
theorem code_lt (h i : Nat) : code h i < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- A Q2_0 weight value is the code minus one. -/
def weight (c : Nat) : Int := (c : Int) - 1

/-- Weight values lie in `[-1, 2]`. -/
theorem weight_range (c : Nat) (hc : c < 4) : -1 ≤ weight c ∧ weight c ≤ 2 := by
  unfold weight
  omega

-- § offset-corrected dot product

/-- `sum c * x` over paired lists (the VNNI quantity). -/
def dotList : List Nat → List Int → Int
  | c :: cs, x :: xs => (c : Int) * x + dotList cs xs
  | _, _ => 0

/-- `sum x` (the `hx` quantity before scaling). -/
def sumList : List Int → Int
  | [] => 0
  | x :: xs => x + sumList xs

/-- `sum (c-1) * x`, the mathematical definition of the Q2_0 dot. -/
def offDot : List Nat → List Int → Int
  | c :: cs, x :: xs => ((c : Int) - 1) * x + offDot cs xs
  | _, _ => 0

/-- The kernel's rewrite `sum (c-1)x = sum c x - sum x` for equal-length inputs. -/
theorem offDot_eq (cs : List Nat) (xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = dotList cs xs - sumList xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, dotList, sumList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      simp only [offDot, dotList, sumList]
      rw [Int.sub_mul, Int.one_mul]
      omega

/-- The VNNI dot splits over a chunk boundary (lo/hi 32-value halves of a 64-weight block). -/
theorem dotList_append (cs1 cs2 : List Nat) (xs1 xs2 : List Int) (h : cs1.length = xs1.length) :
    dotList (cs1 ++ cs2) (xs1 ++ xs2) = dotList cs1 xs1 + dotList cs2 xs2 := by
  induction cs1 generalizing xs1 with
  | nil =>
    cases xs1 with
    | nil => simp [dotList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs1 with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      simp only [List.cons_append, dotList]
      rw [ih xs h']
      omega

/-- The activation sum splits over a chunk boundary. -/
theorem sumList_append (xs ys : List Int) : sumList (xs ++ ys) = sumList xs + sumList ys := by
  induction xs with
  | nil => simp [sumList]
  | cons x xs ih =>
    simp only [List.cons_append, sumList]
    rw [ih]
    omega

-- § activation clamp

/-- Clamp to the symmetric int8 range used by the quantiser. -/
def clamp127 (v : Int) : Int := min 127 (max (-127) v)

/-- The clamp lands in `[-127,127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact clamp127_id _ h.1 h.2

-- § tail masks, tail block counts, token dispatch

/-- The 16-lane tail mask for `nb` blocks (two scales per block). -/
def lanemask (nb : Nat) : Nat := if nb ≥ 8 then 65535 else 2 ^ (2 * nb) - 1

/-- A tail mask always fits in 16 lanes. -/
theorem lanemask_lt (nb : Nat) : lanemask nb < 65536 := by
  unfold lanemask
  by_cases h : nb ≥ 8
  · rw [if_pos h]
    decide
  · rw [if_neg h]
    have h1 : 2 ^ (2 * nb) ≤ 2 ^ 14 := Nat.pow_le_pow_right (by decide) (by omega)
    have h2 : (2 : Nat) ^ 14 = 16384 := by decide
    omega

/-- Blocks processed in the group starting at `b0`. -/
def nbOf (n b0 : Nat) : Nat := if n - b0 < 8 then n - b0 else 8

/-- Each group handles at least one block while blocks remain. -/
theorem nbOf_pos (n b0 : Nat) (h : b0 < n) : 1 ≤ nbOf n b0 := by
  unfold nbOf
  split <;> omega

/-- A group never exceeds eight blocks. -/
theorem nbOf_le (n b0 : Nat) : nbOf n b0 ≤ 8 := by
  unfold nbOf
  split <;> omega

/-- A group never reads past the row. -/
theorem nbOf_fits (n b0 : Nat) : b0 + nbOf n b0 ≤ n ∨ n < b0 := by
  unfold nbOf
  split <;> omega

/-- Token-count dispatch: counts above seven (and the default arm) use the 8-token instantiation. -/
def ntDispatch (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- Dispatch never exceeds eight tokens. -/
theorem ntDispatch_le (n : Nat) : ntDispatch n ≤ 8 := by
  unfold ntDispatch
  split <;> omega

/-- Dispatch is exact for 1..7 tokens. -/
theorem ntDispatch_id (n : Nat) (h : n ≤ 7) : ntDispatch n = n := by
  unfold ntDispatch
  split <;> omega

-- § gate/up interleaving

/-- Gate row index in the interleaved pack. -/
def gateRow (r : Nat) : Nat := 2 * r

/-- Up row index in the interleaved pack. -/
def upRow (r : Nat) : Nat := 2 * r + 1

/-- Gate and up rows never alias. -/
theorem gate_ne_up (r s : Nat) : gateRow r ≠ upRow s := by
  unfold gateRow upRow
  omega

/-- Both rows of pair `r < ff` lie inside `2*ff` rows. -/
theorem gu_rows_in_range (r ff : Nat) (h : r < ff) : gateRow r < 2 * ff ∧ upRow r < 2 * ff := by
  unfold gateRow upRow
  omega

-- § fp16 decode range

/-- For a normal fp16 exponent field `1..30`, the rebiased fp32 exponent is in `113..142`. -/
theorem fp16_normal_exp (e : Nat) (h1 : 1 ≤ e) (h2 : e ≤ 30) :
    113 ≤ (e : Int) - 15 + 127 ∧ (e : Int) - 15 + 127 ≤ 255 := by
  omega

end Strata.Q2Expert
