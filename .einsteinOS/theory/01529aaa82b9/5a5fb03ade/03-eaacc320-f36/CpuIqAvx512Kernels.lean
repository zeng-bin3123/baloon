/-!
# MODULE: cpu_iq_avx512_kernels
Layout and dispatch contracts of the i-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) AVX-512 expert rows:
block byte sizes, every byte offset the decoders read, the sign-mask negation, the scale packing and the
token-batching loop.

## Source semantics
* Each 256-value block is decoded as four 64-value chunks `j = 0..3`; every offset read for chunk `j` must lie
  inside the block (`bytes` per format).
* Signs are applied as a masked subtract `0 - y` on lanes whose mask bit is set; applying it twice is identity.
* `scales4` packs four 16-bit scales into one qword; lane `l` of 32 int16 lanes reads field `l / 8`.
* Per-block scale factors are `2*s + 1` for a 4-bit `s`, hence odd and in 1..31.
* Token counts above 4 are processed in chunks of at most 4 whose sizes sum to the total.
* `iq512_supported` is true exactly for ggml types 16, 17, 18, 21, 22.

## Form template identifiers
Fmt<TY> → quant_format_decoder, bytes → block_bytes, K → format_constant, decode → block_chunk_decode,
row_dot → iq_row_dot, gu_rows → gate_up_rows, dot_rows → down_rows, iq512_supported → format_supported.
-/

namespace Strata.Cpu.IqAvx512

-- § block sizes

/-- Block bytes per format (type id ↦ bytes). -/
def blockBytes : Nat → Option Nat
  | 16 => some 66
  | 17 => some 74
  | 18 => some 98
  | 21 => some 110
  | 22 => some 82
  | _ => none

/-- Supported-type predicate, as in `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- A type has a block size exactly when it is supported. -/
theorem supported_iff_has_bytes (t : Nat) : supported t = true ↔ (blockBytes t).isSome = true := by
  constructor
  · intro h
    simp only [supported, Bool.or_eq_true, beq_iff_eq] at h
    rcases h with (((h | h) | h) | h) | h <;> subst h <;> rfl
  · intro h
    unfold blockBytes at h
    split at h <;> simp_all [supported]

/-- The byte sizes are the sums of the field sizes of each format. -/
theorem block_bytes_decomposition :
    66 = 2 + 64 ∧ 74 = 2 + 64 + 8 ∧ 82 = 2 + 64 + 8 + 8 ∧ 98 = 2 + 64 + 32 ∧
      110 = 2 + 64 + 8 + 32 + 4 := by decide

-- § every decoder read stays in bounds (chunk j < 4)

/-- IQ2_XXS reads `qs` words at `2 + 16 j .. 2 + 16 j + 15`. -/
theorem iq2xxs_reads_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS: grid words, then the scale bytes at `66 + 2j`, `66 + 2j + 1`. -/
theorem iq2xs_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S: grid bytes `2 + 8j`, signs as a u64 at `2 + 32 + 8j`, `qh` at `66 + 2j`, scales at `74 + 2j`. -/
theorem iq2s_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by omega

/-- IQ3_XXS: grid bytes at `2 + 16j`, then two u32 words at `2 + 64 + 8j`. -/
theorem iq3xxs_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S: grid bytes, `qh` pair at `66 + 2j`, signs u64 at `74 + 8j`, scale byte at `106 + j`. -/
theorem iq3s_reads_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by omega

/-- The four 64-value chunks tile the 256-value block. -/
theorem chunks_tile_block : 4 * 64 = 256 := by decide

-- § scales

/-- The decoded scale `2*s + 1` for a 4-bit `s` is odd, at least 1 and at most 31. -/
theorem scale_range (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- A byte splits into two 4-bit scales that recombine to the byte. -/
theorem byte_nibbles (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- The top nibble of a 32-bit word is `w >> 28`, which is below 16. -/
theorem top_nibble_lt (w : Nat) (h : w < 4294967296) : w / 268435456 < 16 := by omega

/-- Four 16-bit fields packed into a qword are recovered by dividing and reducing (field 0). -/
theorem pack4_field0 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) % 65536 = s0 := by omega

/-- Field 1. -/
theorem pack4_field1 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 65536 % 65536 = s1 := by omega

/-- Field 2. -/
theorem pack4_field2 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) (h2 : s2 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 4294967296 % 65536 = s2 := by omega

/-- Field 3. -/
theorem pack4_field3 (s0 s1 s2 s3 : Nat) (h0 : s0 < 65536) (h1 : s1 < 65536) (h2 : s2 < 65536)
    (h3 : s3 < 65536) :
    (s0 + 65536 * s1 + 4294967296 * s2 + 281474976710656 * s3) / 281474976710656 = s3 := by omega

/-- Every int16 lane `l < 32` of the maddubs result reads one of the four packed scales. -/
theorem scale_lane_field (l : Nat) (h : l < 32) : l / 8 < 4 := by omega

-- § masked sign subtract

/-- Apply the sign mask bit to an activation byte: `0 - y` when set, `y` otherwise. -/
def signed (bit : Bool) (y : Int) : Int := if bit then 0 - y else y

/-- Negating twice is the identity, so the mask is a pure sign flip. -/
theorem signed_involutive (bit : Bool) (y : Int) : signed bit (signed bit y) = y := by
  cases bit <;> simp [signed]

/-- Magnitudes are preserved by the sign mask. -/
theorem signed_abs (bit : Bool) (y : Int) : (signed bit y).natAbs = y.natAbs := by
  cases bit <;> simp [signed]

/-- A cleared mask bit leaves the activation untouched. -/
theorem signed_false (y : Int) : signed false y = y := by simp [signed]

-- § token-batching loop in dot_rows_nt

/-- Total tokens covered by repeatedly taking `min step remaining` (fuel-bounded loop). -/
def chunkTotal (step : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | f + 1, rem => if rem = 0 then 0 else min step rem + chunkTotal step f (rem - min step rem)

/-- With a positive step and enough fuel the chunk sizes sum to the token count. -/
theorem chunkTotal_eq (step : Nat) (hs : 0 < step) :
    ∀ f rem, rem ≤ f → chunkTotal step f rem = rem := by
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
    · have hm : 0 < min step rem := by omega
      have := ih (rem - min step rem) (by omega)
      simp only [chunkTotal, hr, if_false]
      rw [this]
      omega

/-- For `nt` tokens, the chunk-of-4 loop covers exactly `nt` tokens. -/
theorem nt_chunks_cover (nt : Nat) : chunkTotal 4 nt nt = nt :=
  chunkTotal_eq 4 (by decide) nt nt (Nat.le_refl _)

/-- Each chunk is at most 4 wide, as the templates `<TY,1..4>` require. -/
theorem chunk_le_four (rem : Nat) : min 4 rem ≤ 4 := by omega

end Strata.Cpu.IqAvx512
