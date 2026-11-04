/-!
# MODULE: cpu_q2_avx2_kernel
The AVX2-only Q2_0 row kernel and activation quantizer: the 2-bit unpack by byte/word interleave, the horizontal
reductions, the token chunking, and the round-half-away-from-zero quantizer rule.

## Source semantics
* `unpack64`: from 16 code bytes, value `4n + k` of the 64 outputs is `(byte_n >> 2k) & 3`, produced by
  masking four shifted copies and interleaving bytes then words; low vector = values 0..31 (bytes 0..7), high =
  32..63 (bytes 8..15).
* The 8-lane float/int reduction reduces to the sum of all lanes.
* `q2_0_gguf_rows_multi_avx2` handles `nt ≤ 4` directly and splits larger counts into chunks of ≤ 4 that sum to `nt`.
* The quantizer rule is `t + copysign(0.5, t)` truncated toward zero (half away from zero), then clamped to ±127.
  It differs from round-half-to-even exactly on ties.

## Form template identifiers
unpack64 → code_unpack_64, row_multi → multi_token_row_dot, rows → row_range_loop,
act_quant_q8_1_avx2 → activation_quantizer_avx2, h2f → half_to_float.
-/

namespace Strata.Cpu.Q2Avx2

-- § unpack index algebra

/-- 2-bit code `k` of a byte: `(b >> 2k) & 3`. -/
def code (b k : Nat) : Nat := (b / 4 ^ k) % 4

/-- Output position of byte `n`, code `k` after the two interleave stages. -/
def pos (n k : Nat) : Nat := 4 * n + k

/-- The position map is a bijection from (byte, code) pairs to value indices: decode recovers both. -/
theorem pos_decode (n k : Nat) (hk : k < 4) : pos n k / 4 = n ∧ pos n k % 4 = k := by
  unfold pos
  constructor <;> omega

/-- Every value index below `4m` is reached from a byte below `m`. -/
theorem pos_surjective (v m : Nat) (h : v < 4 * m) : v / 4 < m ∧ pos (v / 4) (v % 4) = v := by
  unfold pos
  constructor <;> omega

/-- The low vector holds values 0..31 (bytes 0..7); the high vector holds 32..63 (bytes 8..15). -/
theorem lo_hi_split (n k : Nat) (hk : k < 4) (hn : n < 8) : pos n k < 32 ∧ 32 ≤ pos (n + 8) k := by
  unfold pos
  constructor <;> omega

/-- Every code is in 0..3. -/
theorem code_lt_four (b k : Nat) : code b k < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- The four codes of a byte recombine into the byte (so no bits are lost by unpacking). -/
theorem codes_recombine (b : Nat) (h : b < 256) :
    code b 0 + 4 * code b 1 + 16 * code b 2 + 64 * code b 3 = b := by
  unfold code
  simp
  omega

-- § reductions

/-- The 8-lane integer reduction (128-bit fold, `shuffle 0x4E`, `shuffle 0xB1`) is the sum of all lanes. -/
theorem lane_reduction_sums (a0 a1 a2 a3 a4 a5 a6 a7 : Int) :
    ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)) = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 := by
  omega

/-- The correction term subtracted from the accumulated dot. -/
theorem corrected_dot (acc corr : Int) : acc - corr + corr = acc := by omega

-- § token chunking

/-- Tokens covered by repeatedly taking `min 4 remaining`. -/
def chunkTotal : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min 4 rem + chunkTotal f (rem - min 4 rem)

/-- With enough fuel the chunks cover exactly `rem` tokens. -/
theorem chunkTotal_eq : ∀ f rem, rem ≤ f → chunkTotal f rem = rem := by
  intro f
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    simp [chunkTotal]
  | succ f ih =>
    intro rem h
    by_cases hr : rem = 0
    · subst hr; simp [chunkTotal]
    · have := ih (rem - min 4 rem) (by omega)
      simp only [chunkTotal, hr, if_false]
      rw [this]
      omega

/-- The avx2 row entry covers all `nt` tokens. -/
theorem avx2_rows_cover (nt : Nat) : chunkTotal nt nt = nt := chunkTotal_eq nt nt (Nat.le_refl _)

-- § quantizer rule

/-- Round half away from zero of `n / d` (`d > 0`), using floor division on a non-negative numerator. -/
def rha (n d : Int) : Int :=
  if 0 ≤ n then (2 * n + d) / (2 * d) else -((-2 * n + d) / (2 * d))

/-- Round half to even of `n / d`, via floor quotient and non-negative remainder. -/
def rne (n d : Int) : Int :=
  let q := n / d
  let r := n % d
  if 2 * r < d then q else if 2 * r > d then q + 1 else if q % 2 = 0 then q else q + 1

/-- Clamp to the symmetric int8 range used by the quantizer. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- Ties go away from zero under the quantizer rule. -/
theorem rha_ties :
    rha 1 2 = 1 ∧ rha 3 2 = 2 ∧ rha 5 2 = 3 ∧ rha (-1) 2 = -1 ∧ rha (-3) 2 = -2 ∧ rha 0 2 = 0 := by
  decide

/-- Half-to-even disagrees with the quantizer rule on ties, which is why `_mm256_round_ps` is not used. -/
theorem rne_differs_on_ties : rne 1 2 ≠ rha 1 2 ∧ rne 5 2 ≠ rha 5 2 := by decide

/-- Away from ties both agree. -/
theorem rne_rha_agree_off_ties : rne 7 4 = rha 7 4 ∧ rne 1 4 = rha 1 4 ∧ rne (-7) 4 = rha (-7) 4 := by
  decide

/-- The clamp always lands in [-127, 127] and never touches in-range values. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- In-range values pass the clamp unchanged. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- If `0 ≤ n ≤ 127 d` (i.e. `|x| / s ≤ 127`), rounding gives at most 127, so the int cast is safe. -/
theorem rha_le_127 (n d : Int) (hd : 0 < d) (h0 : 0 ≤ n) (h : n ≤ 127 * d) : rha n d ≤ 127 := by
  unfold rha
  rw [if_pos h0]
  have : (2 * n + d) / (2 * d) < 128 := by
    rw [Int.ediv_lt_iff_lt_mul (by omega)]
    omega
  omega

end Strata.Cpu.Q2Avx2
