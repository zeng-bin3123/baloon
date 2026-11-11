/-!
# MODULE: q2_avx2_fallback
SOURCES: src/kernels/cpu/q2_avx2.cpp
The Q2_0 GGUF-row dot and the Q8_1 activation quantizer for CPUs without AVX-512, built from
SSE/AVX2 shifts, byte/word unpacks, `maddubs` and shuffles.

## Source semantics
* `unpack64` shifts 16-bit lanes by 0/2/4/6 and masks each byte with 3; despite the 16-bit shift
  each byte keeps its OWN codes (neighbouring bits land above bit 1 and are masked off).
* The unpack_lo/hi_epi8 then unpack_lo/hi_epi16 interleave puts value i of the block at byte i/4,
  slot i%4 — the same LSB-first order as the Q2_0 dequantizer and the AVX-512 kernel.
* `maddubs` of codes (<= 3) with int8 activations (|x| <= 127) never saturates (|pair| <= 762).
* Per block the result is d (s0 S0 + s1 S1) - d (hx0 + hx1) with hx = s * sum, equal to
  d (s0 (S0 - sum0) + s1 (S1 - sum1)): the code-offset correction of the AVX-512 kernel.
* The quantizer's integer horizontal sum (shuffle 0x4E then 0xB1) adds all four lanes.
* Rows at >4 tokens are processed in groups of at most 4 that cover every token.

## Form template identifiers
unpack64 -> unpackPos, row_multi -> blockResult, act_quant_q8_1_avx2 -> hsum4, shuffle imm -> shufLane,
q2_0_gguf_rows_multi_avx2 default -> groupsOf4
-/

namespace Strata.Q2Avx2

-- § unpack correctness

/-- Shifting a 16-bit word `lo + 256 hi` right by `2k` and masking each byte with 3 yields the codes
of `lo` and `hi` independently (no bit leaks across the byte boundary after the mask). -/
theorem shift16_bytes_independent (lo hi : Nat) (hlo : lo < 256) (hhi : hi < 256) :
    (lo + 256 * hi) / 1 % 256 % 4 = lo % 4 ∧ (lo + 256 * hi) / 1 / 256 % 4 = hi % 4 ∧
    (lo + 256 * hi) / 4 % 256 % 4 = lo / 4 % 4 ∧ (lo + 256 * hi) / 4 / 256 % 4 = hi / 4 % 4 ∧
    (lo + 256 * hi) / 16 % 256 % 4 = lo / 16 % 4 ∧ (lo + 256 * hi) / 16 / 256 % 4 = hi / 16 % 4 ∧
    (lo + 256 * hi) / 64 % 256 % 4 = lo / 64 % 4 ∧ (lo + 256 * hi) / 64 / 256 % 4 = hi / 64 % 4 := by
  omega

/-- `unpacklo_epi8(x, y)`: output byte `i` takes byte `i/2` of `x` (even `i`) or of `y` (odd `i`).
Returned as (which source, byte). -/
def unpack8 (i : Nat) : Nat × Nat := (i % 2, i / 2)

/-- `unpacklo_epi16(a, b)`: output byte `i` is byte `2 (i/4) + i%2` of `a` (word index even) or `b`. -/
def unpack16 (i : Nat) : Nat × Nat := (i / 2 % 2, 2 * (i / 4) + i % 2)

/-- Composition of the two interleaves: output byte `i` of the low 128 bits comes from code plane
`2 * src16 + src8` (= slot) and source byte `byte8` (= block byte). -/
def unpackPos (i : Nat) : Nat × Nat :=
  let (s16, j) := unpack16 i
  let (s8, b) := unpack8 j
  (2 * s16 + s8, b)

/-- Value `i` of the first 16 lands at code slot `i % 4` of byte `i / 4`; the `unpackhi` halves repeat
this with offsets 4, 8, 12 bytes, giving the dequantizer's order for all 64 values. -/
theorem unpackPos_order (i : Nat) (hi : i < 16) : unpackPos i = (i % 4, i / 4) := by
  simp only [unpackPos, unpack16, unpack8]
  refine Prod.ext ?_ ?_ <;> simp <;> omega

/-- The four 16-value groups (lo/hi of the epi16 unpack, from bytes 0..7 and 8..15) cover values
0..63 with block byte `i/4`: group `g`, local index `i` -> value `16 g + i`, byte `4 g + i / 4`. -/
theorem unpack_groups_cover (g i : Nat) (hg : g < 4) (hi : i < 16) :
    16 * g + i < 64 ∧ (16 * g + i) / 4 = 4 * g + i / 4 ∧ (16 * g + i) % 4 = i % 4 := by omega

-- § arithmetic

/-- `maddubs` pairs `c0 x0 + c1 x1` with codes in [0,3] and |x| <= 127 stay within ±762. -/
theorem maddubs_codes_bound (c0 c1 x0 x1 : Int) (hc : 0 ≤ c0 ∧ c0 ≤ 3 ∧ 0 ≤ c1 ∧ c1 ≤ 3)
    (hx : -127 ≤ x0 ∧ x0 ≤ 127 ∧ -127 ≤ x1 ∧ x1 ≤ 127) :
    -762 ≤ c0 * x0 + c1 * x1 ∧ c0 * x0 + c1 * x1 ≤ 762 := by
  obtain ⟨a0, a1, a2, a3⟩ := hc
  have t0 : c0 = 0 ∨ c0 = 1 ∨ c0 = 2 ∨ c0 = 3 := by omega
  have t1 : c1 = 0 ∨ c1 = 1 ∨ c1 = 2 ∨ c1 = 3 := by omega
  rcases t0 with rfl | rfl | rfl | rfl <;> rcases t1 with rfl | rfl | rfl | rfl <;> omega

/-- The per-block result the kernel accumulates. -/
def blockResult (d s0 s1 S0 S1 u0 u1 : Int) : Int := d * (s0 * S0) + d * (s1 * S1) - d * (s0 * u0 + s1 * u1)

/-- With `hx = s * sum`, the separate `corr` term equals subtracting the activation sum per chunk:
`d (s0 S0 + s1 S1) - d (hx0 + hx1) = d (s0 (S0 - u0) + s1 (S1 - u1))`. -/
theorem blockResult_correction (d s0 s1 S0 S1 u0 u1 : Int) :
    blockResult d s0 s1 S0 S1 u0 u1 = d * (s0 * (S0 - u0) + s1 * (S1 - u1)) := by
  unfold blockResult
  rw [Int.mul_sub s0, Int.mul_sub s1, ← Int.mul_add, ← Int.mul_sub]
  congr 1
  omega

-- § horizontal sum in the quantizer

/-- `_mm_shuffle_epi32` lane selector: lane `i` takes source lane `(imm >> 2i) & 3`. -/
def shufLane (imm i : Nat) : Nat := imm / 4 ^ i % 4

/-- 0x4E swaps the two 64-bit halves; 0xB1 swaps adjacent lanes. -/
theorem shuffle_constants :
    (List.range 4).map (shufLane 0x4E) = [2, 3, 0, 1] ∧ (List.range 4).map (shufLane 0xB1) = [1, 0, 3, 2] := by
  decide

/-- Lane 0 after `s += shuf(s, 0x4E); s += shuf(s, 0xB1)` holds the sum of all four lanes. -/
def hsum4 (a b c d : Int) : Int :=
  let s := [a + c, b + d, c + a, d + b]
  s[0]! + s[1]!

/-- The integer horizontal sum is exact. -/
theorem hsum4_exact (a b c d : Int) : hsum4 a b c d = a + b + c + d := by
  simp [hsum4]; omega

/-- A chunk sum of 32 clamped activations fits easily in int32 (|sum| <= 4064). -/
theorem chunk_sum_bound : 32 * 127 = 4064 ∧ 4064 < 2147483647 := by decide

-- § token grouping

/-- Group sizes of the default case: chunks of at most 4 tokens. -/
def groupsOf4 : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 4 n :: groupsOf4 fuel (n - min 4 n)

/-- The groups cover every token, and each has a width the switch handles (1..4). -/
theorem groupsOf4_cover (fuel n : Nat) (hf : n ≤ 4 * fuel) :
    (groupsOf4 fuel n).foldr (· + ·) 0 = n ∧ ∀ g ∈ groupsOf4 fuel n, 1 ≤ g ∧ g ≤ 4 := by
  induction fuel generalizing n with
  | zero => have : n = 0 := by omega
            subst this; simp [groupsOf4]
  | succ f ih =>
    cases n with
    | zero => simp [groupsOf4]
    | succ m =>
      obtain ⟨h1, h2⟩ := ih (m + 1 - min 4 (m + 1)) (by omega)
      simp only [groupsOf4, List.foldr, List.mem_cons]
      refine ⟨by omega, ?_⟩
      intro g hg
      rcases hg with rfl | hg
      · omega
      · exact h2 g hg

/-- GGUF Q2_0 rows stride 18 bytes per block: block `b` has its scale at `18 b` and codes at `18 b + 2`. -/
theorem gguf_block_stride (b nb : Nat) (hb : b < nb) : 18 * b + 2 + 16 ≤ 18 * nb := by omega

end Strata.Q2Avx2
