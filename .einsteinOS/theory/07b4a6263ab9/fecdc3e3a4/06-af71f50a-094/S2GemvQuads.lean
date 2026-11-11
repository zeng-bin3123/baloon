/-!
# MODULE: s2_gemv_quads
SOURCES: src/kernels/cuda/s2_gemv_quads.cu
The S2 GEMV with the code load amortised over a quad: one block per row, `threads_per_row` threads
striding over quads, each quad = one code byte + one 64-bit load of four FP16 activations, then a
shared-memory tree reduction.

## Source semantics
* Quad `q` covers elements `4q … 4q+3`: code byte `c[q]`, code `k` at bits `2k`, group `q >> 4`,
  activations at byte offset `8q` (one aligned `uint2`).
* Host: `n_in, n_out > 0` else no-op; ONLY `n_in % 4 = 0` is checked.  The scale pointer is
  `scales + o·(n_in/64)`, so a row of `n_in % 64 ≠ 0` reads a group index past its own scale run.
* Reduction: `partial[t] = (a0+a1)+(a2+a3)`, then `for (step = tpr/2; step > 0; step >>= 1)`
  `partial[t] += partial[t+step]` for `t < step`.  Exact for a power-of-two `tpr`; the host does NOT
  check that, and for other `tpr` some partials are never added.

## Form template identifiers
* `quadByte`, `quadGroup` ↔ `c[q]`, `s[q >> 4]`
* `stepAt`, `reduceLoop`, `blockResult` ↔ the shared-memory loop as written (any `tpr`)
* `halve`, `tree` ↔ the same loop specialised to `2^k` slots
-/

namespace Strata.S2Quads

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2

-- § quad addressing

/-- Element `4q + k` reads code byte `q`, bit shift `2k`, group `q >> 4`. -/
theorem quad_element (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 4 = q ∧ ((4 * q + k) % 4) * 2 = 2 * k ∧ (4 * q + k) / 64 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- The four codes of a quad, read as `(byte >> 2k) & 3`, are the base-4 digits of the byte. -/
theorem quad_codes (byte k : Nat) (hk : k < 4) :
    (byte >>> (2 * k)) &&& 3 = [byte % 4, byte / 4 % 4, byte / 16 % 4, byte / 64 % 4].getD k 0 := by
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.shiftRight_eq_div_pow, and3] <;> omega

/-- The `uint2` activation load of quad `q` starts at byte `8q` (four halves): aligned whenever `x` is. -/
theorem act_load_aligned (base q : Nat) (h : base % 8 = 0) : (base + 2 * (4 * q)) % 8 = 0 := by omega

/-- Each thread visits quads `t, t + tpr, …`; every quad is visited by thread `q % tpr`. -/
theorem quad_owner (q tpr : Nat) (h : 0 < tpr) : q % tpr < tpr ∧ q = q % tpr + (q / tpr) * tpr := by
  have := Nat.div_add_mod q tpr
  rw [Nat.mul_comm] at this
  exact ⟨Nat.mod_lt _ h, by omega⟩

-- § the n_in guard

/-- With `n_in % 64 = 0`, every quad's group is one of the row's `n_in/64` scales. -/
theorem group_in_row (n_in q : Nat) (h : n_in % 64 = 0) (hq : q < n_in / 4) : q >>> 4 < n_in / 64 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- **GUARD GAP**: the host checks only `n_in % 4`.  At `n_in = 68` it accepts, but the last quad
(`q = 16`) reads group 1 while the row has `68/64 = 1` scale — i.e. the first scale of the NEXT row
(or past the buffer on the last row).  S2 rows in the pack are multiples of 64, so this is latent; the
sibling `s2_gemv_q8` and `s2_gemv_fast` check `% 64`. -/
theorem n_in_guard_gap :
    68 % 4 = 0 ∧ 16 < 68 / 4 ∧ (16 >>> 4) = 1 ∧ ¬ (16 >>> 4 < 68 / 64) := by decide

-- § reduction exactness (power-of-two tpr)

/-- Pairwise sum of two lists. -/
theorem sum_zipWith_add : ∀ (a b : List Int), a.length = b.length →
    (List.zipWith (· + ·) a b).sum = a.sum + b.sum
  | [], [], _ => by simp
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    have ih := sum_zipWith_add xs ys (by simpa using h)
    simp only [List.zipWith_cons_cons, List.sum_cons, ih]
    omega

/-- One step on `2m` live slots. -/
def halve (l : List Int) (m : Nat) : List Int := List.zipWith (· + ·) (l.take m) (l.drop m)

/-- A step preserves the total. -/
theorem halve_sum (l : List Int) (m : Nat) (h : l.length = 2 * m) : (halve l m).sum = l.sum := by
  have hlen : (l.take m).length = (l.drop m).length := by
    simp [List.length_take, List.length_drop]; omega
  unfold halve
  rw [sum_zipWith_add _ _ hlen, ← List.sum_append, List.take_append_drop]

/-- A step halves the live slots. -/
theorem halve_length (l : List Int) (m : Nat) (h : l.length = 2 * m) : (halve l m).length = m := by
  unfold halve
  simp [List.length_zipWith, List.length_take, List.length_drop]
  omega

/-- All steps on `2^k` slots. -/
def tree : Nat → List Int → List Int
  | 0, l => l
  | k + 1, l => tree k (halve l (2 ^ k))

/-- **Exactness**: with `tpr = 2^k`, `partial[0]` ends as the sum of all thread partials. -/
theorem tree_eq_sum : ∀ (k : Nat) (l : List Int), l.length = 2 ^ k → tree k l = [l.sum]
  | 0, l, h => by
    match l, h with
    | [x], _ => simp [tree]
  | k + 1, l, h => by
    have h2 : l.length = 2 * 2 ^ k := by rw [h, Nat.pow_succ, Nat.mul_comm]
    have ih := tree_eq_sum k (halve l (2 ^ k)) (halve_length l _ h2)
    simp only [tree]
    rw [ih, halve_sum l _ h2]

-- § reduction hazard (any tpr)

/-- The loop body as written: `partial[t] += partial[t+step]` for `t < step`. -/
def stepAt (l : List Int) (off : Nat) : List Int :=
  l.mapIdx (fun t v => if t < off then v + l.getD (t + off) 0 else v)

/-- The loop `step = tpr/2, tpr/4, …, 1` (fuel-bounded). -/
def reduceLoop : Nat → List Int → Nat → List Int
  | 0, l, _ => l
  | fuel + 1, l, off => if off = 0 then l else reduceLoop fuel (stepAt l off) (off / 2)

/-- `partial[0]` after the loop. -/
def blockResult (l : List Int) (tpr : Nat) : Int := (reduceLoop 32 l (tpr / 2)).headD 0

/-- Sanity: on power-of-two blocks the literal loop returns the full sum. -/
theorem blockResult_pow2_examples :
    blockResult [1, 2, 3, 4, 5, 6, 7, 8] 8 = 36 ∧ blockResult (List.replicate 32 1) 32 = 32 := by decide

/-- **HAZARD**: `threads_per_row` is not validated here, and for a non-power-of-two the tree silently
drops partials — at `tpr = 3` one of three, at `tpr = 48` (the row count used elsewhere) some of 48.
`s_gemv_split` and `bf16_gemv_split` refuse such values; this launcher (like `s2_gemv_q8` and
`s2_gemv_fast`) relies on its callers. -/
theorem non_pow2_tpr_drops :
    blockResult [1, 1, 1] 3 = 2 ∧ blockResult (List.replicate 48 1) 48 < 48 := by decide

end Strata.S2Quads
