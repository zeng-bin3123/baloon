/-!
# MODULE: s2_gemv_lut_staging
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
An S2 GEMV experiment with two levers — staging `x` in shared memory and a 256×4 constant-memory code table —
both measured negative or neutral and kept as a recorded negative result (NOT the fast path).  This module
proves the lookup table is exactly the plane decode with the −1 bias, the group-index shift, the shared-memory
layout (staged activations never overlap the reduction partials), and the host gates.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1` for every byte `b < 256`, `k < 4`; one `float4` per byte.
* Quad `q`: byte `c[q]`, scale `s[q >> 4]`, activations `x[4q … 4q+3]` (staged or direct).
* Shared memory, staged: halves `sx[0 … n_in)` in bytes `[0, 2·n_in)`, partials at float offset
  `MAX_SHARED_HALVES/2 = 2048`, i.e. byte 8192; size `8192 + 4·tpr`.  Unstaged: partials at 0, size `4·tpr`.
* Host: `n_in % 4` and `n_in % 64` required; staging requires `n_in ≤ 4096`; the table is uploaded once
  (process-global flag, not per device); `tpr` is not validated.
* Grid is `n_out` blocks, so `blockIdx < n_out` and the early return before the barrier is never taken.

## Form template identifiers
`c_codes` → `lut`; `MAX_SHARED_HALVES` → `maxHalves`; partial byte offset → `partialBase`;
`smem` → `smemBytes`; host gate → `hostOk`; `stage_x` → `stage`.
-/

namespace Strata.S2GemvLutStaging

-- § the code table

/-- `c_codes[b][k]` as filled by `ensure_lut`. -/
def lut (b k : Nat) : Int := (((b >>> (2 * k)) &&& 3 : Nat) : Int) - 1

/-- Plane code of element `e`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- Shift-and-mask is division-and-modulo. -/
theorem extract2 (x s : Nat) : (x >>> s) &&& 3 = x / 2 ^ s % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod (x >>> s) 2
  rw [Nat.shiftRight_eq_div_pow] at this ⊢
  simpa using this

/-- **The table is the biased plane decode**: `lut[c[q]][k] = code(4q + k) - 1`. -/
theorem lut_is_decode (byte : Nat → Nat) (q k : Nat) (hk : k < 4) :
    lut (byte q) k = ((planeCode byte (4 * q + k) : Nat) : Int) - 1 := by
  unfold lut planeCode
  rw [show (4 * q + k) / 4 = q by omega, show (4 * q + k) % 4 = k by omega, Nat.mul_comm 2 k]

/-- Table entries take only the values `-1, 0, 1, 2`. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  unfold lut
  have h : (b >>> (2 * k)) &&& 3 < 4 := by rw [extract2]; exact Nat.mod_lt _ (by decide)
  omega

/-- Spot values: byte `0b11100100` lists codes `0,1,2,3` → `-1,0,1,2`. -/
theorem lut_example : lut 0xE4 0 = -1 ∧ lut 0xE4 1 = 0 ∧ lut 0xE4 2 = 1 ∧ lut 0xE4 3 = 2 := by decide

/-- Group of quad `q`: `q >> 4 = 4q / 64`. -/
theorem quad_group (q k : Nat) (hk : k < 4) : q >>> 4 = (4 * q + k) / 64 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

-- § shared-memory layout

/-- `MAX_SHARED_HALVES`. -/
def maxHalves : Nat := 4096

/-- Byte offset of `partial[0]`: `MAX_SHARED_HALVES / 2` floats when staged, 0 otherwise. -/
def partialBase (stage : Bool) : Nat := if stage then (maxHalves / 2) * 4 else 0

/-- Dynamic shared bytes requested. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then maxHalves * 2 else 0) + tpr * 4

/-- **No overlap**: under the staging limit the staged halves end at or before the partials begin. -/
theorem staged_disjoint (nIn : Nat) (h : nIn ≤ maxHalves) : 2 * nIn ≤ partialBase true := by
  unfold partialBase maxHalves at *; simp; omega

/-- Every `partial[tid]`, `tid < tpr`, lies inside the requested allocation (both modes). -/
theorem partials_inside (stage : Bool) (tpr tid : Nat) (ht : tid < tpr) :
    partialBase stage + 4 * tid + 4 ≤ smemBytes stage tpr := by
  unfold partialBase smemBytes maxHalves
  cases stage <;> simp <;> omega

/-- The model width `n_embd = 2560` fits the staging buffer (5 KB of 8 KB). -/
theorem embd_fits : 2560 ≤ maxHalves ∧ 2 * 2560 = 5120 := by decide

-- § host gates

/-- `s2_gemv_fast`'s acceptance. -/
def hostOk (nIn nOut : Nat) (stage : Bool) : Bool :=
  0 < nIn && 0 < nOut && nIn % 4 == 0 && nIn % 64 == 0 && (!stage || nIn ≤ maxHalves)

/-- Accepted shapes keep every quad's group inside the row's `n_in / 64` scales. -/
theorem groups_in_row (nIn nOut q : Nat) (stage : Bool) (h : hostOk nIn nOut stage = true)
    (hq : q < nIn / 4) : q >>> 4 < nIn / 64 := by
  simp [hostOk] at h
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- Staging is refused above 4096 halves; unstaged has no such limit. -/
theorem staging_limit : hostOk 8192 1 true = false ∧ hostOk 8192 1 false = true := by decide

/-- One block per row: every launched block has `blockIdx < n_out`, so no thread returns before the barrier. -/
theorem no_early_return (nOut blk : Nat) (h : blk < nOut) : ¬ (nOut ≤ blk) := by omega

end Strata.S2GemvLutStaging
