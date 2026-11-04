/-!
# MODULE: s2_gemv_q8_activation_contract
S2 (Q2_0) GEMV that consumes Q8_0-quantized activations as ggml does: block per output row, each thread takes
quads (one code byte = four elements), dequantizes the activation `xq * dx` in the loop, accumulates in four
partials and reduces through a shared-memory halving tree.

## Source semantics
* Activations are `n_in / 32` blocks of 34 bytes (`fp16 d`, `int8 qs[32]`).
* A quad `q` covers elements `4q .. 4q+3`, all inside Q8_0 block `q / 8` at offset `4 * (q % 8)`.
* The S2 group index is `q >> 4` (64-element groups).
* Required: `n_in % 32 = 0` and `n_in % 64 = 0`.

## Form template identifiers
- `s2_gemv_q8_kernel`   → `block_per_row_q8_act_kernel`
- `QK8_0`               → `act_block_elems`
- `ablk`, `off`         → `act_block_index`, `act_block_offset`
-/

namespace Strata.S2GemvQ8

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


-- § activation block addressing

/-- Quad `q` starts in activation block `(4 q) / 32 = q / 8`. -/
theorem quad_block_index (q : Nat) : (q * 4) / 32 = q / 8 := by omega

/-- Offset of quad `q` inside its block. -/
theorem quad_block_offset (q : Nat) : (q * 4) % 32 = 4 * (q % 8) := by omega

/-- All four elements of a quad lie inside the block: `off + 3 < 32`. -/
theorem quad_inside_block (q : Nat) : (q * 4) % 32 + 3 < 32 := by omega

/-- Block start in bytes is `34 * (q / 8)`; the fp16 header is read at that offset, data at `+ 2`. -/
theorem block_byte_offsets (q : Nat) : 34 * (q / 8) + 2 + (q * 4) % 32 + 3 < 34 * (q / 8 + 1) := by omega

/-- The S2 group of quad `q` and of each of its four elements coincide. -/
theorem quad_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- Per-element decode `(code - 1) * d` for the four 2-bit fields of a byte. -/
def field (byte k : Nat) : Int := (((byte >>> (2 * k)) % 4 : Nat) : Int) - 1

/-- The four decoded fields identify the byte. -/
theorem fields_lossless : ∀ b : Nat, b < 256 →
    ((field b 0 + 1) + 4 * (field b 1 + 1) + 16 * (field b 2 + 1) + 64 * (field b 3 + 1)) = (b : Int) := by
  intro b hb
  simp only [field, Nat.reduceMul]
  omega

-- § preconditions

/-- The two divisibility requirements reduce to `n_in % 64 = 0` (64 is a multiple of 32). -/
theorem requirements_reduce (n : Nat) (h : n % 64 = 0) : n % 32 = 0 ∧ n % 4 = 0 := by omega

/-- With `n_in = 2560` both requirements hold: 80 activation blocks, 640 quads per row. -/
theorem n_embd_requirements : 2560 % 32 = 0 ∧ 2560 % 64 = 0 ∧ 2560 / 32 = 80 ∧ 2560 / 4 = 640 := by decide

/-- A row's quads number `n_in / 4` and its scales `n_in / 64`; the last quad's group is the last scale. -/
theorem last_quad_group (m : Nat) (hm : 0 < m) : (64 * m / 4 - 1) / 16 = m - 1 := by omega

/-- Thread-strided quad loop covers each quad exactly once. -/
theorem quad_stride_cover (q tpr : Nat) : q = q % tpr + tpr * (q / tpr) := (Nat.mod_add_div q tpr).symm

/-- Four-partial combine equals plain sum over exact integers. -/
theorem quad_combine (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

/-- Dequantizing in the loop multiplies the weight `(c-1)*d` by `xq*dx`; the activation product commutes. -/
theorem dequant_commute (c d xq dx : Int) :
    ((c - 1) * d) * (xq * dx) = ((c - 1) * d) * (dx * xq) := by
  rw [Int.mul_comm xq dx]

end Strata.S2GemvQ8
