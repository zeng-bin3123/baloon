/-!
# MODULE: s2_gemv_fast_tree_reduce
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
Experimental S2 (2-bit) GEMV with a constant-memory code table and optional shared staging of x; measured
neutral/negative and kept as a negative result. One block per output row, a shared-memory halving tree reduction.

## Source semantics
* c_codes[b][k] = ((b >> 2k) & 3) - 1 for k in 0..3: four code values per byte, bias already applied.
* n_quads = n_in/4; the scale index of quad q is q >> 4 (one scale per 64 values).
* Staging x needs n_in <= 4096 halves; shared = 8 KiB (staged) + 4*threads_per_row bytes of partials.
* The reduction halves `step` from threads_per_row/2 to 1, adding partial[tid+step] into partial[tid].

## Form template identifiers
c_codes -> code_lut ; s2_gemv_fast_kernel -> gemv_row_block ; MAX_SHARED_HALVES -> stage_limit ;
partial -> reduce_buffer
-/

namespace Strata.S2GemvFast

/-- Sum of f over the first n indices. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Code value with the -1 bias applied. -/
def codeVal (b k : Nat) : Int := (((b >>> (2 * k)) &&& 3 : Nat) : Int) - 1

-- § code table

/-- Each LUT entry is in {-1, 0, 1, 2}. -/
theorem codeVal_range : ∀ b < 256, ∀ k < 4, -1 ≤ codeVal b k ∧ codeVal b k ≤ 2 := by
  decide +kernel

/-- The four 2-bit fields reconstruct the byte. -/
theorem byte_from_fields : ∀ b < 256,
    ((b >>> 0) &&& 3) + 4 * ((b >>> 2) &&& 3) + 16 * ((b >>> 4) &&& 3) + 64 * ((b >>> 6) &&& 3) = b := by
  decide +kernel

/-- Quad count times four is n_in. -/
theorem quads_cover (nIn : Nat) (h : nIn % 4 = 0) : nIn / 4 * 4 = nIn := by
  omega

/-- The scale index q >> 4 equals (4q)/64. -/
theorem scale_index (q : Nat) : q / 16 = (q * 4) / 64 := by
  omega

/-- A quad's four elements share its scale (elements 4q..4q+3 lie in one 64-group). -/
theorem quad_in_group (q k : Nat) (hk : k < 4) : (q * 4 + k) / 64 = (q * 4) / 64 := by
  omega

-- § shared memory

/-- The partial buffer starts 4096 B after the staged halves (2048 floats). -/
theorem partial_offset : 4096 / 2 * 4 = 4096 * 2 := by
  decide

/-- Shared size with staging for 1024 threads stays below 48 KiB. -/
theorem smem_fits : 4096 * 2 + 1024 * 4 ≤ 49152 := by
  decide

-- § tree reduction

/-- Splitting a prefix sum at n. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n + m) = sumTo f n + sumTo (fun j => f (j + n)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m + 1)) = sumTo f (n + m) + f (n + m) := rfl
    rw [e, ih, Nat.add_comm n m]
    simp only [sumTo]
    omega

/-- Pointwise sum commutes with the prefix sum. -/
theorem sumTo_add_fun (f g : Nat → Int) (s : Nat) :
    sumTo (fun i => f i + g i) s = sumTo f s + sumTo g s := by
  induction s with
  | zero => simp [sumTo]
  | succ s ih =>
    simp only [sumTo]
    rw [ih]
    omega

/-- One halving step: threads tid < s add partial[tid + s]; together they sum 2s entries. -/
theorem tree_step (f : Nat → Int) (s : Nat) :
    sumTo f (s + s) = sumTo (fun i => f i + f (i + s)) s := by
  rw [sumTo_add, sumTo_add_fun]

/-- The kernel's loop: step = 2^k, then 2^(k-1), ..., 1; the result is the value at tid 0. -/
def reduceK : Nat → (Nat → Int) → Int
  | 0, f => f 0
  | k + 1, f => reduceK k (fun i => f i + f (i + 2 ^ k))

/-- The halving-tree reduction computes the full sum of 2^k partials. -/
theorem reduceK_eq (k : Nat) : ∀ f : Nat → Int, reduceK k f = sumTo f (2 ^ k) := by
  induction k with
  | zero =>
    intro f
    simp [reduceK, sumTo]
  | succ k ih =>
    intro f
    have e : 2 ^ (k + 1) = 2 ^ k + 2 ^ k := by
      rw [Nat.pow_succ]
      omega
    rw [reduceK, ih, e, tree_step]

/-- Reads at partial[tid + step] stay inside the buffer when tid < step and 2*step <= threads. -/
theorem tree_read_in_bounds (tid step threads : Nat) (h : tid < step) (hs : 2 * step ≤ threads) :
    tid + step < threads := by
  omega

end Strata.S2GemvFast
