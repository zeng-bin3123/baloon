/-!
# MODULE: s_gemv_dequant_row_kernels
Naive and split S-family GEMV kernels: weights are dequantized on the fly from 2/4/8-bit codes
(optionally through the IQ4_NL codebook), multiplied with FP16, Q8_K or Q8_0 activations, and reduced per
output row (thread-per-row, block-per-row tree, or warp-per-row with a 16-way register accumulator).

## Source semantics
* A code of width `bits` is bits `[(i % per) * bits, +bits)` of byte `i / per`, `per = 8 / bits`.
* `w = decode(code) * d + b`; the bias is applied to the integer code, the offset to the weight before the
  activation multiply.
* The group index is `e >> group_shift` with `group_elems = 2^group_shift`.
* One lane-iteration of the warp kernel consumes 16 consecutive elements lying in one group and one
  activation block (Q8_K: 256, Q8_0: 32).
* The block-wide reduction is a fixed halving tree, hence deterministic (modelled over exact integers).

## Form template identifiers
- `s_gemv_kernel`            → `thread_per_row_kernel`
- `s_gemv_split_kernel`      → `block_per_row_tree_kernel`
- `s_gemv_q8_split_kernel`   → `warp_per_row_quant_act_kernel`
- `kIq4Nl`                   → `nonlinear_codebook`
- `group_shift`              → `log2_group`
- `QE` (=16)                 → `elems_per_lane_iter`
-/

namespace Strata.SGemv

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

/-- The explicit 16-accumulator regrouping at the end of the warp kernel is a plain sum. -/
theorem acc16_regroup (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15))) =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

-- § code unpacking and decode

/-- Code `i` of a row of bytes at width `bits` (mask `2^bits - 1` is the same as `% 2^bits`). -/
def codeAt (bytes : Nat → Nat) (bits i : Nat) : Nat :=
  (bytes (i / (8 / bits)) >>> ((i % (8 / bits)) * bits)) % 2 ^ bits

/-- Every unpacked code fits its width. -/
theorem codeAt_lt (bytes : Nat → Nat) (bits i : Nat) : codeAt bytes bits i < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- The four-code word is `4 * bits / 8` bytes for the three supported widths. -/
theorem quad_word_bytes :
    4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 ∧ 4 / (8 / 2) = 1 ∧ 4 / (8 / 4) = 2 ∧ 4 / (8 / 8) = 4 := by
  decide

/-- 2-bit packing: a byte is exactly the four fields in LSB-first order. -/
theorem repack_2bit : ∀ b : Nat, b < 256 →
    b = (b >>> 0) % 4 + 4 * ((b >>> 2) % 4) + 16 * ((b >>> 4) % 4) + 64 * ((b >>> 6) % 4) := by
  intro b hb
  omega

/-- 4-bit packing of a byte. -/
theorem repack_4bit (b : Nat) (hb : b < 256) : b = b % 16 + 16 * ((b >>> 4) % 16) := by
  omega

/-- 4-bit packing of the 16-bit four-code word used by S4 (`uint16_t` load). -/
theorem repack_4bit_word (v : Nat) (hv : v < 65536) :
    v = v % 16 + 16 * ((v >>> 4) % 16) + 256 * ((v >>> 8) % 16) + 4096 * ((v >>> 12) % 16) := by
  omega

/-- The IQ4_NL codebook. -/
def kIq4Nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Weight decode: codebook lookup or biased integer code. -/
def decode (iq4nl : Bool) (code : Nat) (bias : Int) : Int :=
  if iq4nl then kIq4Nl.getD (code % 16) 0 else (code : Int) + bias

/-- The codebook table has 16 entries, so the masked index is always in range. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- The codebook is strictly increasing (it is non-linear but monotone). -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by decide

/-- Non-codebook decode is the biased integer code. -/
theorem decode_affine (code : Nat) (bias : Int) : decode false code bias = (code : Int) + bias := by
  simp [decode]

/-- Codebook decode stays inside the table's value range. -/
theorem decode_iq4nl_range (code : Nat) (bias : Int) :
    -127 ≤ decode true code bias ∧ decode true code bias ≤ 113 := by
  have h : ∀ r : Nat, r < 16 → (-127 ≤ kIq4Nl.getD r 0 ∧ kIq4Nl.getD r 0 ≤ 113) := by decide
  simpa [decode] using h (code % 16) (Nat.mod_lt _ (by decide))

/-- Offset-before-multiply expands to the three exact terms (over exact integers the order is immaterial;
over floats it is not, which is why the kernels fix it). -/
theorem weight_then_activation (c bias d b x : Int) :
    ((c + bias) * d + b) * x = c * d * x + bias * d * x + b * x := by
  simp only [Int.add_mul]

-- § activation block geometry

/-- `block_q8_K` is `f32 d; i8 qs[256]; i16 bsums[16]` = 292 bytes, `d` 4-byte aligned. -/
theorem q8k_block_geometry : 4 + 256 + 2 * 16 = 292 ∧ 292 % 4 = 0 := by decide

/-- `block_q8_0` is `f16 d; i8 qs[32]` = 34 bytes, `d` 2-byte aligned. -/
theorem q8_0_block_geometry : 2 + 32 = 34 ∧ 34 % 2 = 0 := by decide

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K is structurally impossible. -/
theorem shexp_down_needs_q8_0 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

-- § warp kernel iteration structure

/-- Lane `l`, iteration `j` starts at element `16 * l + 512 * j`; a lane-iteration never straddles a Q8_K block. -/
theorem lane_iter_in_q8k_block (l j : Nat) (hl : l < 32) :
    (16 * l + 512 * j) % 256 + 16 ≤ 256 := by
  omega

/-- ... nor a Q8_0 block. -/
theorem lane_iter_in_q8_0_block (l j : Nat) (hl : l < 32) :
    (16 * l + 512 * j) % 32 + 16 ≤ 32 := by
  omega

/-- Every element `e` is owned by exactly one (lane, iteration, offset) triple. -/
theorem lane_iter_cover (e : Nat) :
    e = 16 * ((e / 16) % 32) + 512 * (e / 512) + e % 16 := by
  omega

/-- A multiple-of-16 row has no tail: every start position leaves a full 16-element step. -/
theorem no_tail_when_aligned (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : i < n) :
    i + 16 ≤ n := by
  omega

/-- Sixteen consecutive elements starting at a multiple of 16 share one group (group size `16 * m`). -/
theorem quad_same_group (i k m : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / (16 * m) = i / (16 * m) := by
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have h : (i + k) / 16 = i / 16 := by omega
  rw [h]

/-- The group index as a shift equals the division by `2^s`. -/
theorem group_shift_eq_div (e s : Nat) : e >>> s = e / 2 ^ s := Nat.shiftRight_eq_div_pow e s

/-- Row start offsets in the warp kernel: row `o` begins at `o * (n_in / per)`; distinct (block, warp) pairs
map to distinct rows. -/
theorem row_of_injective (wpb b w b' w' : Nat) (hw : w < wpb) (hw' : w' < wpb)
    (h : b * wpb + w = b' * wpb + w') : b = b' ∧ w = w' := by
  have hpos : 0 < wpb := by omega
  have h1 : (b * wpb + w) / wpb = b := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hw, Nat.zero_add]
  have h2 : (b' * wpb + w') / wpb = b' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hw', Nat.zero_add]
  have h3 : (b * wpb + w) % wpb = w := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hw
  have h4 : (b' * wpb + w') % wpb = w' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right]
    exact Nat.mod_eq_of_lt hw'
  constructor
  · calc b = (b * wpb + w) / wpb := h1.symm
      _ = (b' * wpb + w') / wpb := by rw [h]
      _ = b' := h2
  · calc w = (b * wpb + w) % wpb := h3.symm
      _ = (b' * wpb + w') % wpb := by rw [h]
      _ = w' := h4

/-- Thread `t` of a 32-lane-per-row block splits into warp and lane losslessly. -/
theorem warp_lane_split (t : Nat) : t = 32 * (t / 32) + t % 32 := by omega

end Strata.SGemv
