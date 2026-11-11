/-!
# MODULE: s2_gemv_naive
SOURCES: src/kernels/cuda/s2_gemv.cu
The naive S2 GEMV, one thread per output row, FP16 activations: `y[o] = Σ_b Σ_{j<64} (code - 1)·s[o,b]·x[64b+j]`.
It is the reference the faster S2 kernels are checked against.  This module proves its per-row addressing is
the plane convention, its nested group loop is the flat row sum, and pins the host gate.

## Source semantics
* `nb = n_in / 64`; row `o`'s codes start at byte `o·nb·16`, scales at `o·nb`.
* Element `(b, j)`: byte `b·16 + (j >> 2)`, shift `(j & 3)·2`, mask 3, decoded `(code - 1)` in the integer
  domain, then `· d · x` (same order as the CPU reference).
* Host: `n_in ≤ 0 ∨ n_out ≤ 0` → no launch; `n_in % 64 ≠ 0` → exit.  Grid `⌈n_out/128⌉ × 128`, synchronous.

## Form template identifiers
`s2_gemv_kernel` → `rowSum`; `QK` → `64`; `nb` → `nb`; plane accessor → `planeCode`;
kernel accessor → `kernelCode`; host gate → `hostOk`.
-/

namespace Strata.S2GemvNaive

-- § addressing

/-- Global plane code of element `e`: byte `e/4`, bits `(e%4)·2`. -/
def planeCode (byte : Nat → Nat) (e : Nat) : Nat := (byte (e / 4) >>> ((e % 4) * 2)) &&& 3

/-- The kernel's read for row `o`, group `b`, element `j`. -/
def kernelCode (byte : Nat → Nat) (nb o b j : Nat) : Nat :=
  (byte (o * nb * 16 + b * 16 + (j >>> 2)) >>> ((j &&& 3) * 2)) &&& 3

/-- `j >> 2 = j / 4`. -/
theorem shr2 (j : Nat) : j >>> 2 = j / 4 := by rw [Nat.shiftRight_eq_div_pow]

/-- `j & 3 = j % 4`. -/
theorem and3 (j : Nat) : j &&& 3 = j % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod j 2; simpa using this

/-- **The kernel reads the plane**: row `o`, group `b`, element `j < 64` is global element
`o·n_in + 64·b + j` with `n_in = 64·nb`, in the row-major `[n_out × n_in]` plane. -/
theorem kernel_reads_plane (byte : Nat → Nat) (nb o b j : Nat) :
    kernelCode byte nb o b j = planeCode byte (o * (64 * nb) + 64 * b + j) := by
  unfold kernelCode planeCode
  rw [shr2, and3 j]
  have hp : o * (64 * nb) = 64 * (o * nb) := by rw [Nat.mul_left_comm]
  have hq : o * nb * 16 = 16 * (o * nb) := Nat.mul_comm _ _
  rw [hp, hq]
  generalize o * nb = P
  rw [show (64 * P + 64 * b + j) / 4 = 16 * P + b * 16 + j / 4 by omega,
    show (64 * P + 64 * b + j) % 4 = j % 4 by omega]

/-- Rows do not overlap: row `o`'s `16·nb` code bytes end where row `o+1`'s begin. -/
theorem rows_disjoint (nb o b j : Nat) (hb : b < nb) (hj : j < 64) :
    o * nb * 16 + b * 16 + j / 4 < (o + 1) * nb * 16 := by
  rw [Nat.add_mul, Nat.add_mul, Nat.one_mul]
  have : b * 16 + j / 4 < nb * 16 := by
    have : (b + 1) * 16 ≤ nb * 16 := Nat.mul_le_mul_right _ hb
    rw [Nat.add_mul] at this; omega
  omega

-- § the row sum

/-- `rsum f n = f 0 + … + f (n-1)`. -/
def rsum (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => rsum f n + f n

/-- Unfolding step. -/
theorem rsum_succ (f : Nat → Int) (n : Nat) : rsum f (n + 1) = rsum f n + f n := rfl

/-- `rsum` only looks at indices below its bound. -/
theorem rsum_congr {f g : Nat → Int} {n : Nat} (h : ∀ i, i < n → f i = g i) : rsum f n = rsum g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [rsum_succ, rsum_succ, ih (fun i hi => h i (by omega)), h n (by omega)]

/-- Splitting a range. -/
theorem rsum_split (f : Nat → Int) (m n : Nat) :
    rsum f (m + n) = rsum f m + rsum (fun i => f (m + i)) n := by
  induction n with
  | zero => exact (Int.add_zero _).symm
  | succ n ih => rw [← Nat.add_assoc, rsum_succ, ih, rsum_succ]; omega

/-- The decoded term for element `e` given the group's scale and the activation. -/
def term (code : Nat) (d x : Int) : Int := ((code : Int) - 1) * d * x

/-- The kernel's double loop for one row (exact arithmetic). -/
def rowSum (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) : Int :=
  rsum (fun b => rsum (fun j => term (code (b * 64 + j)) (scale b) (x (b * 64 + j))) 64) nb

/-- The flat reference: every element uses the scale of its group `e / 64`. -/
def flatSum (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) : Int :=
  rsum (fun e => term (code e) (scale (e / 64)) (x e)) (nb * 64)

/-- **The group loop is the flat sum.** -/
theorem rowSum_eq_flat (code : Nat → Nat) (scale : Nat → Int) (x : Nat → Int) (nb : Nat) :
    rowSum code scale x nb = flatSum code scale x nb := by
  unfold rowSum flatSum
  induction nb with
  | zero => rfl
  | succ m ih =>
    rw [Nat.succ_mul, rsum_split _ (m * 64) 64, rsum_succ, ih]
    congr 1
    exact rsum_congr (fun j hj => by rw [show (m * 64 + j) / 64 = m by omega])

-- § host gate

/-- `s2_gemv`'s acceptance: positive sizes and `n_in % 64 = 0` (otherwise exit, or no launch). -/
def hostOk (nIn nOut : Nat) : Bool := 0 < nIn && 0 < nOut && nIn % 64 == 0

/-- Under the gate `nb·64 = n_in`: no element of the row is skipped by the group loop. -/
theorem gate_exact (nIn nOut : Nat) (h : hostOk nIn nOut = true) : nIn / 64 * 64 = nIn := by
  simp [hostOk] at h; omega

/-- Grid `⌈n_out/128⌉` blocks of 128 threads covers every row. -/
theorem grid_covers (nOut : Nat) : nOut ≤ (nOut + 127) / 128 * 128 := by omega

/-- Model shapes: `n_embd = 2560` and the expert width 640 are both accepted; 2500 is refused. -/
theorem gate_examples : hostOk 2560 640 = true ∧ hostOk 640 2560 = true ∧ hostOk 2500 1 = false := by decide

end Strata.S2GemvNaive
