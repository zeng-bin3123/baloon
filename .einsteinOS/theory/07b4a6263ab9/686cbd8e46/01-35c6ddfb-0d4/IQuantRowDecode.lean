/-!
# MODULE: iquant_row_decode
SOURCES: src/kernels/cpu/iq_avx512.cpp
The i-quant expert rows (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22) in 512-bit lanes:
each 256-value block is decoded 64 values at a time into grid magnitudes, a 64-bit sign mask and a
vector of odd scales, then dotted with every token's Q8_K activations.

## Source semantics
* Block sizes 66/74/98/110/82 bytes; every field read for chunk j < 4 stays inside the block.
* Scales are `2 s + 1` for a 4-bit s: odd, in [1, 31].  Constants K: 1/8 for IQ2_*, 1/4 for IQ3_XXS,
  1 for IQ3_S.
* Packed u32 words hold four 7-bit sign indices and a 4-bit scale (bits 28..31), recomposing exactly.
* Grid indices are in range: IQ2_XXS < 256, IQ2_XS 9-bit < 512, IQ2_S 10-bit < 1024, IQ3_S 9-bit < 512,
  sign-table indices 7-bit < 128.
* `scales4`: int16 lane i takes scale i / 8; with (sa, sa, sb, sb) value v of the chunk uses sa for
  v < 32 and sb otherwise.
* No saturation: grid magnitudes <= 62 and |y| <= 128 keep `maddubs` pairs below 2^15, and the
  per-block int32 lane sums below 2^24 (exact in f32).
* More than 4 tokens in `dot_rows_nt` are processed in groups of at most 4 that cover all tokens.

## Form template identifiers
Fmt<TY>::decode -> chunkReads, scales4 -> laneScale, row_dot<TY,NT> -> rowDot, dot_rows_nt -> groupsOf4,
iq512_supported -> supported
-/

namespace Strata.IQ512

-- § formats

/-- `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Exactly the five i-quant types are handled; IQ1_M (29) stays on ggml-cpu. -/
theorem supported_set : (List.range 64).filter supported = [16, 17, 18, 21, 22] := by decide

/-- Block bytes as the sum of their fields. -/
theorem block_layouts :
    2 + 64 = 66 ∧ 2 + 64 + 8 = 74 ∧ 2 + 32 + 32 + 8 + 8 = 82 ∧ 2 + 64 + 32 = 98 ∧
    2 + 64 + 8 + 32 + 4 = 110 := by decide

/-- IQ2_XXS / IQ2_XS: codes `2 + 16 j .. +16`, IQ2_XS scales `66 + 2j, +1` stay inside 66/74 bytes. -/
theorem iq2_xxs_xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S: grid bytes `2+8j`, sign bytes `34+8j`, qh `66+2j`, scales `74+2j` inside 82 bytes. -/
theorem iq2_s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS: grid bytes `2+16j`, sign/scale words `66+8j .. +8` inside 98 bytes. -/
theorem iq3_xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: grid bytes `2+16j`, qh `66+2j`, signs `74+8j`, scale byte `106+j` inside 110 bytes. -/
theorem iq3_s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

-- § scales and packed words

/-- The scale formula `2 s + 1` of a 4-bit field. -/
def oddScale (s : Nat) : Nat := 2 * (s % 16) + 1

/-- Scales are odd and in `[1, 31]`. -/
theorem oddScale_range (s : Nat) : oddScale s % 2 = 1 ∧ 1 ≤ oddScale s ∧ oddScale s ≤ 31 := by
  unfold oddScale; omega

/-- A packed u32 holds four 7-bit sign indices (bits 0..27) and the 4-bit scale (bits 28..31). -/
theorem packed_word_recompose (a : Nat) (ha : a < 4294967296) :
    a = a % 128 + 128 * (a / 128 % 128) + 16384 * (a / 16384 % 128) + 2097152 * (a / 2097152 % 128) +
      268435456 * (a / 268435456) ∧ a / 268435456 < 16 := by omega

/-- Grid and sign-table indices stay inside their tables. -/
theorem grid_indices_in_range (b v q h k : Nat) (hk : k < 4) :
    b % 256 < 256 ∧ v % 512 < 512 ∧ v / 512 % 128 < 128 ∧
    q % 256 + 256 * (h / 4 ^ k % 4) < 1024 ∧ q % 256 + 256 * (h / 2 ^ k % 2) < 512 := by
  omega

/-- `scales4` lane map: int16 lane `i < 32` takes scale slot `i / 8`. -/
def laneScale (s0 s1 s2 s3 : Int) (i : Nat) : Int :=
  if i / 8 = 0 then s0 else if i / 8 = 1 then s1 else if i / 8 = 2 then s2 else s3

/-- A maddubs int16 lane `i` sums values `2i, 2i+1`; with `(sa, sa, sb, sb)` the first 32 values of the
chunk get `sa` and the last 32 get `sb`, matching the two u32 groups of the encoding. -/
theorem scales_follow_groups (sa sb : Int) (v : Nat) (hv : v < 64) :
    laneScale sa sa sb sb (v / 2) = if v < 32 then sa else sb := by
  unfold laneScale
  by_cases h : v < 32
  · simp only [h, if_true]
    have : v / 2 / 8 = 0 ∨ v / 2 / 8 = 1 := by omega
    rcases this with h0 | h1
    · simp [h0]
    · simp [h1]
  · simp only [h, if_false]
    have : v / 2 / 8 = 2 ∨ v / 2 / 8 = 3 := by omega
    rcases this with h2 | h3
    · simp [h2]
    · simp [h3]

-- § overflow freedom

/-- A `maddubs` int16 lane `g0 y0 + g1 y1` with grid magnitudes `<= 62` and `|y| <= 128` never
saturates (|.| <= 15872 < 32767). -/
theorem maddubs_no_saturation (g0 g1 y0 y1 : Int) (hg : 0 ≤ g0 ∧ g0 ≤ 62 ∧ 0 ≤ g1 ∧ g1 ≤ 62)
    (hy : -128 ≤ y0 ∧ y0 ≤ 128 ∧ -128 ≤ y1 ∧ y1 ≤ 128) :
    -32767 ≤ g0 * y0 + g1 * y1 ∧ g0 * y0 + g1 * y1 ≤ 32767 := by
  obtain ⟨a0, a1, a2, a3⟩ := hg
  obtain ⟨b0, b1, b2, b3⟩ := hy
  have p0 : -7936 ≤ g0 * y0 ∧ g0 * y0 ≤ 7936 := by
    constructor
    · have := Int.mul_le_mul_of_nonneg_left (show -128 ≤ y0 from b0) a0
      have := Int.mul_le_mul_of_nonneg_right a1 (show (0:Int) ≤ 128 by decide)
      omega
    · have := Int.mul_le_mul_of_nonneg_left b1 a0
      have := Int.mul_le_mul_of_nonneg_right a1 (show (0:Int) ≤ 128 by decide)
      omega
  have p1 : -7936 ≤ g1 * y1 ∧ g1 * y1 ≤ 7936 := by
    constructor
    · have := Int.mul_le_mul_of_nonneg_left (show -128 ≤ y1 from b2) a2
      have := Int.mul_le_mul_of_nonneg_right a3 (show (0:Int) ≤ 128 by decide)
      omega
    · have := Int.mul_le_mul_of_nonneg_left b3 a2
      have := Int.mul_le_mul_of_nonneg_right a3 (show (0:Int) ≤ 128 by decide)
      omega
  omega

/-- After `madd` with a scale `<= 31` (two int16 lanes) and accumulation over the 4 chunks of a block,
an int32 lane stays below 2^24, so the `cvtepi32_ps` before the float FMA is exact. -/
theorem block_lane_exact : 4 * (2 * (2 * 62 * 128) * 31) < 16777216 := by decide

-- § token grouping

/-- Group sizes of the default case of `dot_rows_nt`: chunks of at most 4. -/
def groupsOf4 : Nat → Nat → List Nat
  | 0, _ => []
  | _, 0 => []
  | fuel + 1, n => min 4 n :: groupsOf4 fuel (n - min 4 n)

/-- The groups cover all tokens and each is a template width 1..4. -/
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

/-- A 256-value Q8_K block is consumed as four 64-byte loads at `qs + 64 j`. -/
theorem q8k_chunk_offsets (j : Nat) (hj : j < 4) : 64 * j + 64 ≤ 256 := by omega

end Strata.IQ512
