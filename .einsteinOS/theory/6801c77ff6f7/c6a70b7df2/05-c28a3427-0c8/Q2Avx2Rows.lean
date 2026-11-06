/-!
# MODULE: q2_avx2_rows
SOURCES: src/kernels/cpu/q2_avx2.cpp
Integer theory of the AVX2-only Q2_0 row kernel and activation quantizer: the 2-bit unpack is lossless and
value-ordered, the `maddubs` pair sums cannot saturate int16, the code-offset correction is exact, and the token
loop in groups of four covers every token once.

## Source semantics
* 16 code bytes hold 64 two-bit codes; value `v` is in byte `v/4` at bit `2*(v%4)`; values 0..31 come from
  bytes 0..7 and values 32..63 from bytes 8..15.
* `maddubs(codes, q)` adds adjacent products of an unsigned code (0..3) and a signed int8 (-127..127): each
  pair is within +-762, far from int16 saturation; `madd(.., ones)` then widens exactly.
* The block contribution is `d*(sx0*S0 + sx1*S1) - d*(hx0 + hx1)`; the integer part `S - sum` is exact.
* `act_quant_q8_1_avx2` clamps to [-127,127] and records the chunk sum (hence `hx = scale * sum`).

## Form template identifiers
code → unpacked_code, pairSum → maddubs_pair, clampQ → act_clamp, chunksK → token_groups,
blockOf → weight_block_of_value.
-/

namespace Strata.Kernels.Cpu.Q2Avx2Rows

-- § the 2-bit unpack

/-- The code of value `v` given its packed byte: `(byte / 4^(v%4)) % 4`, i.e. bits `2*(v%4)`. -/
def code (byte v : Nat) : Nat := (byte / 2 ^ (2 * (v % 4))) % 4

/-- Every unpacked code is in 0..3. -/
theorem code_lt_four (byte v : Nat) : code byte v < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- Unpacking is lossless: the four codes of a byte recombine to the byte. -/
theorem pack_unpack (b : Nat) (h : b < 256) :
    b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) = b := by
  omega

/-- The four codes of byte `b` are the four values `4i..4i+3`, in order. -/
theorem code_slots (b : Nat) :
    code b 0 = b % 4 ∧ code b 1 = b / 4 % 4 ∧ code b 2 = b / 16 % 4 ∧ code b 3 = b / 64 % 4 := by
  simp [code]

/-- Value `v` lives in byte `v / 4`. -/
def byteOf (v : Nat) : Nat := v / 4

/-- Values 0..31 come from bytes 0..7 (the `lo` vector). -/
theorem lo_bytes (v : Nat) (h : v < 32) : byteOf v < 8 := by
  unfold byteOf
  omega

/-- Values 32..63 come from bytes 8..15 (the `hi` vector). -/
theorem hi_bytes (v : Nat) (h1 : 32 ≤ v) (h2 : v < 64) : 8 ≤ byteOf v ∧ byteOf v < 16 := by
  unfold byteOf
  omega

/-- The 16 code bytes cover exactly the 64 values of one block. -/
theorem sixteen_bytes_sixtyfour_codes : 16 * 4 = 64 := by
  decide

-- § no int16 saturation in maddubs

/-- One `maddubs` lane: sum of two code*activation products. -/
def pairSum (c0 c1 q0 q1 : Int) : Int := c0 * q0 + c1 * q1

/-- A pair of products of codes (0..3) with int8 activations stays within +-762, so it never saturates int16. -/
theorem pairSum_bound (c0 c1 q0 q1 : Int)
    (h0 : 0 ≤ c0) (h1 : c0 ≤ 3) (h2 : 0 ≤ c1) (h3 : c1 ≤ 3)
    (hq0 : -127 ≤ q0) (hq1 : q0 ≤ 127) (hq2 : -127 ≤ q1) (hq3 : q1 ≤ 127) :
    -762 ≤ pairSum c0 c1 q0 q1 ∧ pairSum c0 c1 q0 q1 ≤ 762 := by
  unfold pairSum
  have a : c0 = 0 ∨ c0 = 1 ∨ c0 = 2 ∨ c0 = 3 := by omega
  have b : c1 = 0 ∨ c1 = 1 ∨ c1 = 2 ∨ c1 = 3 := by omega
  rcases a with rfl | rfl | rfl | rfl <;> rcases b with rfl | rfl | rfl | rfl <;> omega

/-- The bound leaves a wide margin below int16's range. -/
theorem pairSum_fits_int16 (c0 c1 q0 q1 : Int)
    (h0 : 0 ≤ c0) (h1 : c0 ≤ 3) (h2 : 0 ≤ c1) (h3 : c1 ≤ 3)
    (hq0 : -127 ≤ q0) (hq1 : q0 ≤ 127) (hq2 : -127 ≤ q1) (hq3 : q1 ≤ 127) :
    -32768 ≤ pairSum c0 c1 q0 q1 ∧ pairSum c0 c1 q0 q1 ≤ 32767 := by
  have := pairSum_bound c0 c1 q0 q1 h0 h1 h2 h3 hq0 hq1 hq2 hq3
  omega

-- § the exact code-offset correction

/-- Centered dot `(c-1)*x` over matching lists. -/
def offDot : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + offDot cs xs
  | _, _ => 0

/-- Raw-code dot over matching lists. -/
def codeDot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + codeDot cs xs
  | _, _ => 0

/-- The correction identity: centered dot = raw dot minus the activation sum. -/
theorem offDot_eq (cs xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = codeDot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, codeDot]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      have hm : (c - 1) * x = c * x - x := by
        rw [Int.sub_mul, Int.one_mul]
      simp only [offDot, codeDot, List.sum_cons]
      rw [hm, ih']
      omega

-- § the activation quantizer

/-- The int8 clamp. -/
def clampQ (v : Int) : Int := max (-127) (min 127 v)

/-- The clamp lands in [-127,127]. -/
theorem clampQ_range (v : Int) : -127 ≤ clampQ v ∧ clampQ v ≤ 127 := by
  unfold clampQ
  omega

/-- The clamp does not touch in-range values (so the truncation-after-±0.5 result is kept). -/
theorem clampQ_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clampQ v = v := by
  unfold clampQ
  omega

/-- Reduction of the four 8-lane int32 sums to one total: add halves, then two shuffles-and-adds (exact in int). -/
theorem total_tree (a b c d e f g h : Int) :
    (((a + e) + (c + g)) + ((b + f) + (d + h))) = a + b + c + d + e + f + g + h := by
  omega

-- § token groups of four

/-- Group sizes when walking `n` tokens in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- The groups cover exactly `n` tokens. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group has between 1 and `k` tokens, so the `rows<NT>` instance used always has NT in 1..4. -/
theorem chunksK_bounds (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, 0 < x ∧ x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Any token count is covered exactly by groups of four (fuel `n`). -/
theorem token_groups_cover (n : Nat) : (chunksK 4 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 4 by decide)
  omega

/-- The offset of group `g` into the token array is `4*g`, and the group's tokens lie below `4*g + 4`. -/
theorem group_offset (g t : Nat) (ht : t < 4) : 4 * g + t < 4 * (g + 1) := by
  omega

end Strata.Kernels.Cpu.Q2Avx2Rows
