/-!
# MODULE: s2_gemv_quad_amortised
S2 GEMV with the code load amortised over four elements: block per output row, thread `tid` takes quads
`q = tid, tid + tpr, ...`, loads one code byte, one scale and one 64-bit group of four FP16 activations, keeps
four accumulators and reduces through a shared-memory halving tree.

## Source semantics
* One code byte per quad; element `4q + k` is bits `[2k, 2k+2)`, decoded as `(code - 1) * d`.
* Group index of quad `q` is `q >> 4` because `(4q) >> 6 = q >> 4`.
* The four halves of a quad are 8 contiguous, 8-byte-aligned bytes.
* The reduction tree over `threads_per_row` partials is deterministic.

## Form template identifiers
- `s2_gemv_quads_kernel`  → `quad_amortised_kernel`
- `n_quads`               → `quad_count`
- `partial`               → `block_partials`
-/

namespace Strata.S2GemvQuads

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


/-- `(4q) >> 6 = q >> 4`: the group of a quad's first element. -/
theorem group_shift_identity (q : Nat) : (4 * q) >>> 6 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]
  omega

/-- All four elements of a quad share its group. -/
theorem quad_one_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q / 16 := by omega

/-- The code row has exactly one byte per quad: `n_in / 4` bytes; the row of output `o` starts at `o * (n_in / 4)`. -/
theorem code_row_start (o n_quads q : Nat) (hq : q < n_quads) : o * n_quads + q < (o + 1) * n_quads := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scale row start of output `o` is `o * (n_in / 64)`; quad `q` reads scale `q / 16` of its row, which exists. -/
theorem scale_in_row (m q : Nat) (hq : q < 16 * m) : q / 16 < m := by omega

/-- Four halves = 8 bytes at byte offset `8q`, aligned for `uint2`. -/
theorem half4_aligned (q : Nat) : (4 * q * 2) % 8 = 0 := by omega

/-- The four decoded fields of a code byte (bias -1 applied to the integer code) recompose the byte. -/
theorem fields_lossless : ∀ b : Nat, b < 256 →
    (((b % 4 : Nat) : Int) + 4 * (((b >>> 2) % 4 : Nat) : Int) + 16 * (((b >>> 4) % 4 : Nat) : Int)
      + 64 * (((b >>> 6) % 4 : Nat) : Int)) = (b : Int) := by
  intro b hb
  omega

/-- The `n_in % 4` precondition: with `n_in = 2560` there are 640 quads and 40 groups of 16 quads. -/
theorem n_embd_quads : 2560 / 4 = 640 ∧ 640 / 16 = 40 ∧ 2560 / 64 = 40 := by decide

/-- Thread-strided loop covers each quad exactly once (thread `q % tpr`, iteration `q / tpr`). -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Four-accumulator combine `(a0 + a1) + (a2 + a3)` is the plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

/-- The per-thread partials form a function of `tid`; the final value `partial[0]` of the tree over
`2^k` threads is the sum of all partials. -/
theorem block_reduction_total (k : Nat) (parts : Nat → Int) :
    redT parts k = sumTo parts (2 ^ k) := redT_eq k parts

end Strata.S2GemvQuads
