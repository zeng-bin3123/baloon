/-!
# MODULE: s2_gemv_lut_staged_experiment
Experimental S2 GEMV (block per output row) with an optional shared-memory staging of the FP16 activation and
a constant-memory byte→four-code table. Both levers measured neutral/negative; the kernel is kept as a
reference. This module records the table, indexing and shared-memory layout invariants.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1` for byte `b`, lane `k < 4`.
* One quad = one code byte = four consecutive elements; group = `q >> 4` (64-element groups).
* With staging, `x` occupies `MAX_SHARED_HALVES = 4096` halves (8 KB) and the partial-sum array starts at float
  offset `MAX_SHARED_HALVES / 2`.
* The halving tree over `threads_per_row` partials is deterministic.

## Form template identifiers
- `c_codes`                 → `byte_code_lut`
- `s2_gemv_fast_kernel`     → `block_per_row_lut_kernel`
- `STAGE_X`                 → `stage_activation_flag`
- `MAX_SHARED_HALVES`       → `shared_stage_capacity`
-/

namespace Strata.S2GemvFast

-- § reduction toolkit

/-- Prefix sum of the first `n` terms of `f`. -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumTo f n + f n

/-- Splitting a prefix sum at `a`. -/
theorem sumTo_add (f : Nat → Int) (a b : Nat) :
    sumTo f (a + b) = sumTo f a + sumTo (fun t => f (a + t)) b := by
  induction b with
  | zero => simp [sumTo]
  | succ b ih =>
    show sumTo f (a + b + 1) = sumTo f a + sumTo (fun t => f (a + t)) (b + 1)
    simp only [sumTo, ih]
    omega

/-- Prefix sums are additive in the summand. -/
theorem sumTo_add_fun (g h : Nat → Int) (n : Nat) :
    sumTo (fun t => g t + h t) n = sumTo g n + sumTo h n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih =>
    simp only [sumTo, ih]
    omega

/-- One halving step of the shared-memory tree preserves the total. -/
theorem sumTo_halve (f : Nat → Int) (s : Nat) :
    sumTo f (2 * s) = sumTo (fun t => f t + f (s + t)) s := by
  rw [Nat.two_mul, sumTo_add, sumTo_add_fun]

/-- The kernel's reduction loop `partial[t] += partial[t + step]`, steps `2^(k-1), ..., 1`, read at `t = 0`. -/
def redT : (Nat → Int) → Nat → Int
  | f, 0 => f 0
  | f, k + 1 => redT (fun t => f t + f (2 ^ k + t)) k

/-- The fixed tree reduction over `2^k` partials equals the plain sum of all partials. -/
theorem redT_eq (k : Nat) : ∀ f : Nat → Int, redT f k = sumTo f (2 ^ k) := by
  induction k with
  | zero => intro f; simp [redT, sumTo]
  | succ k ih =>
    intro f
    show redT (fun t => f t + f (2 ^ k + t)) k = _
    rw [ih, Nat.pow_succ, Nat.mul_comm, sumTo_halve]


-- § the lookup table

/-- Table entry for byte `b`, lane `k`. -/
def lutEntry (b k : Nat) : Int := (((b >>> (2 * k)) % 4 : Nat) : Int) - 1

/-- Every lookup entry is one of the ternary weights `-1, 0, 1, 2` (code 3 is the unused fourth level). -/
theorem lut_entry_range : ∀ b : Nat, b < 256 → ∀ k : Nat, k < 4 → -1 ≤ lutEntry b k ∧ lutEntry b k ≤ 2 := by
  intro b _ k _
  unfold lutEntry
  omega

/-- The four entries of a byte recombine into that byte (`Σ (entry+1) 4^k = b`): the table is lossless. -/
theorem lut_lossless : ∀ b : Nat, b < 256 →
    ((lutEntry b 0 + 1) + 4 * (lutEntry b 1 + 1) + 16 * (lutEntry b 2 + 1) + 64 * (lutEntry b 3 + 1)) = (b : Int) := by
  intro b hb
  simp only [lutEntry, Nat.reduceMul]
  omega

/-- Table size: 256 entries × 4 floats × 4 bytes = 4 KB, within constant memory. -/
theorem lut_bytes : 256 * 4 * 4 = 4096 := by decide

-- § indexing

/-- Quad `q` covers elements `4q .. 4q+3`, all in group `q >> 4`. -/
theorem quad_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q / 16 := by omega

/-- The group index as a shift equals the division. -/
theorem quad_group_shift (q : Nat) : q >>> 4 = q / 16 := Nat.shiftRight_eq_div_pow q 4

/-- A quad's four halves are 8 bytes at offset `8 q`, so the `uint2` load is 8-byte aligned. -/
theorem quad_load_aligned (q : Nat) : (q * 4 * 2) % 8 = 0 := by omega

/-- The code byte count of a row is `n_in / 4`; the scale count is `n_in / 64`; for `n_in = 64 m` they are
`16 m` and `m`. -/
theorem row_sizes (m : Nat) : (64 * m) / 4 = 16 * m ∧ (64 * m) / 64 = m := by omega

-- § shared memory layout

/-- Staging buffer: 4096 halves = 8192 bytes = 2048 floats, which is exactly the partial-array float offset. -/
theorem stage_offset_consistent : 4096 * 2 = 2048 * 4 ∧ 4096 / 2 = 2048 := by decide

/-- `n_embd = 2560` fits the staging limit. -/
theorem n_embd_fits_stage : 2560 ≤ 4096 := by decide

/-- Dynamic shared memory with staging: 8 KB for `x` plus `4 * tpr` for partials; for tpr = 256 it is 9 KB. -/
theorem stage_smem_bytes (tpr : Nat) : 4096 * 2 + tpr * 4 = 8192 + 4 * tpr := by omega

/-- Row ownership: block `o` reads codes at `o * (n_in / 4)`; distinct rows have disjoint code spans. -/
theorem row_span_disjoint (o o' stride k : Nat) (hlt : o < o') (hk : k < stride) :
    o * stride + k < o' * stride := by
  have h : (o + 1) * stride ≤ o' * stride := Nat.mul_le_mul_right stride hlt
  rw [Nat.add_mul] at h
  omega

/-- Thread-strided quad loop `q = tid, tid + tpr, ...` covers each quad exactly once. -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Pairwise partial combine `(a0 + a1) + (a2 + a3)` is the plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

end Strata.S2GemvFast
