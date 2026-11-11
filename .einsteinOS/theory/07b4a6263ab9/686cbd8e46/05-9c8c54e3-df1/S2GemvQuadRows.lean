/-!
# MODULE: s2_gemv_quad_rows
SOURCES: src/kernels/cuda/s2_gemv_quads.cu
The S2 GEMV specialised for the 31.64 GiB of Q2_0 weights: one block per row, each thread consuming a QUAD
(one code byte = four codes, one 8-byte load of four fp16 activations, one scale), then a shared-memory tree.
This module proves the quad read is the plane's bit order, the shift-based group index, the quad's alignment,
and exposes two host-gate gaps: `n_in` is only checked mod 4, and `threads_per_row` is not checked at all.

## Source semantics
* Quad `q` covers elements `4q … 4q+3`: code byte `c[q]`, element `4q+k` at bits `2k`; group `q >> 4`
  (= `(4q) >> 6`); activations `x[4q … 4q+3]` as one `uint2` (needs an 8-byte-aligned `x`).
* Thread `tid` takes quads `q ≡ tid (mod tpr)`; four accumulators; then `partial[tid]` and a tree with
  steps `tpr/2, tpr/4, …, 1`, result in slot 0.
* Host: refuses `n_in % 4 ≠ 0` only.  Scales are laid out `n_in / 64` per row.

## Form template identifiers
`s2_gemv_quads_kernel` → quad model; `c[q]` → `quadByte`; `q >> 4` → `quadGroup`; host gate → `hostOk`;
tree loop → `halvingTree`; `threads_per_row` → `tpr`.
-/

namespace Strata.S2GemvQuadRows

-- § quad decoding

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- **Quad read = plane**: code `k` of quad byte `q` is plane element `4q + k`. -/
theorem quad_code (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    (byte q >>> (k * 2)) &&& 3 = planeCode byte (4 * q + k) := by
  unfold planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega]

/-- The kernel's literal shifts `0, 2, 4, 6` are `2k` for `k = 0 … 3`. -/
theorem quad_shifts : [0 * 2, 1 * 2, 2 * 2, 3 * 2] = [0, 2, 4, 6] := rfl

/-- `quadGroup q = q >> 4`. -/
def quadGroup (q : Nat) : Nat := q >>> 4

/-- **Group by shift**: `q >> 4 = (4q) / 64`, the group of every element of the quad. -/
theorem quadGroup_eq (q k : Nat) (hk : k < 4) : quadGroup q = (4 * q + k) / 64 := by
  unfold quadGroup; rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- The 8-byte activation load is aligned whenever the base is. -/
theorem uint2_aligned (xBase q : Nat) (h : xBase % 8 = 0) : (xBase + 2 * (4 * q)) % 8 = 0 := by omega

-- § host gate gap 1: n_in only mod 4

/-- `s2_gemv_quads`' host gate on `n_in`. -/
def hostOk (nIn : Nat) : Bool := 0 < nIn && nIn % 4 == 0

/-- With `n_in % 64 = 0` every quad's group index is inside the row's `n_in/64` scales. -/
theorem groups_in_row (nIn q : Nat) (h64 : nIn % 64 = 0) (hq : q < nIn / 4) : quadGroup q < nIn / 64 := by
  unfold quadGroup; rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- **Gap**: the gate accepts `n_in = 68`, yet quad 16 reads scale index 1 while the row has only
`68/64 = 1` scale — it reads the NEXT row's first scale (or past the buffer on the last row).  The S2 pack
always has `n_in % 64 = 0`, so this is latent; `s2_gemv_fast` and `s2_gemv_q8` do check `% 64`. -/
theorem gate_admits_out_of_row : hostOk 68 = true ∧ 16 < 68 / 4 ∧ 68 / 64 ≤ quadGroup 16 := by decide

-- § host gate gap 2: threads_per_row unchecked

/-- One tree level. -/
def treeStep (a : Nat → Int) (off : Nat) : Nat → Int :=
  fun t => if t < off then a t + a (t + off) else a t

/-- The loop `for (step = tpr/2; step > 0; step >>= 1)`, reading slot 0 (fuel bounds the step count). -/
def halvingGo (a : Nat → Int) : Nat → Nat → Int
  | 0, _ => a 0
  | fuel + 1, step => if step = 0 then a 0 else halvingGo (treeStep a step) fuel (step / 2)

/-- The kernel's reduction over `tpr` partials. -/
def halvingTree (a : Nat → Int) (tpr : Nat) : Int := halvingGo a 64 (tpr / 2)

/-- For powers of two the loop sums all partials (here: 1, 32, 128 and 256 unit partials). -/
theorem halving_pow2_examples :
    halvingTree (fun _ => 1) 1 = 1 ∧ halvingTree (fun _ => 1) 32 = 32 ∧
    halvingTree (fun _ => 1) 128 = 128 ∧ halvingTree (fun _ => 1) 256 = 256 := by decide

/-- **Gap**: the host never validates `tpr`.  For a non-power-of-two the halving loop DROPS partials:
with 3 threads the third partial is never added, with 96 threads 32 of them are lost. -/
theorem halving_drops_partials :
    halvingTree (fun _ => 1) 3 = 2 ∧ halvingTree (fun _ => 1) 96 = 64 := by decide

/-- The tree's reads stay inside `partial[0 … tpr-1]` for any `tpr`: `t < step ≤ tpr/2` gives
`t + step < tpr`.  The defect is lost terms, never an out-of-bounds read. -/
theorem halving_reads_in_bounds (t step tpr : Nat) (ht : t < step) (hs : step ≤ tpr / 2) : t + step < tpr := by
  omega

-- § quad ownership

/-- Thread `tid` owns quad `q` iff `q % tpr = tid`: every quad has exactly one owner for `tpr > 0`. -/
theorem quad_owner_unique (q tpr : Nat) (h : 0 < tpr) : q % tpr < tpr ∧ ∀ t, q % tpr = t → t = q % tpr := by
  exact ⟨Nat.mod_lt _ h, fun _ ht => ht.symm⟩

end Strata.S2GemvQuadRows
