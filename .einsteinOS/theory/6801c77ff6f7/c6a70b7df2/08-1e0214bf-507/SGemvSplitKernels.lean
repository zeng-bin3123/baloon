/-!
# MODULE: s_gemv_split_kernels
SOURCES: src/kernels/cuda/s_gemv.cu
S-family quantized GEMV (2/4/8-bit codes, per-group scale and optional offset, optional IQ4_NL codebook) with
FP16, Q8_K and Q8_0 activations; naive, block-per-row split, and warp-per-row 16-element-per-lane variants.

## Source semantics
* code i = (byte[i / PER_BYTE] >> ((i % PER_BYTE) * bits)) & (2^bits - 1), PER_BYTE = 8 / bits.
* Weight = decode(code) * scale + offset, applied to the weight before the activation multiply.
* Group sizes are powers of two (16, 32, 64); the group index is a shift by log2(group).
* A lane takes QE = 16 consecutive elements per iteration, stride 32*QE, so an iteration never straddles a
  Q8_K (256) or Q8_0 (32) activation block; the tail loop handles the remainder.
* Q8_K needs n_in % 256 = 0 and is refused for 2-bit; Q8_0 needs only n_in % 32 = 0 (ffn_down_shexp: 640).

## Form template identifiers
CODE_BITS -> code_width ; kIq4Nl -> nonlinear_codebook ; QE -> elems_per_lane ; group_shift -> group_log2 ;
s_gemv_q8_split_kernel -> warp_row_kernel ; s_gemv_split_kernel -> block_row_kernel
-/

namespace Strata.SGemv

-- § code packing

/-- PER_BYTE * bits = 8 for the three widths. -/
theorem per_byte_exact : ∀ b ∈ [2, 4, 8], (8 / b) * b = 8 := by
  decide

/-- Bytes of one four-code word: 1, 2 or 4. -/
theorem quad_word_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by
  decide

/-- Extracted code (div/mod form of shift/mask). -/
def codeAt (bits byte idx : Nat) : Nat := (byte / 2 ^ ((idx % (8 / bits)) * bits)) % 2 ^ bits

/-- A code is below 2^bits. -/
theorem codeAt_lt (bits byte idx : Nat) : codeAt bits byte idx < 2 ^ bits := by
  unfold codeAt
  exact Nat.mod_lt _ (Nat.two_pow_pos _)

/-- 2-bit: four consecutive elements 4q..4q+3 live in byte q at positions 0..3. -/
theorem quad_offsets_2bit (q k : Nat) (hk : k < 4) : (4 * q + k) / 4 = q ∧ (4 * q + k) % 4 = k := by
  omega

/-- 4-bit: four consecutive elements live in bytes 2q and 2q+1 (one uint16). -/
theorem quad_offsets_4bit (q k : Nat) (hk : k < 4) :
    (4 * q + k) / 2 = 2 * q + k / 2 ∧ (4 * q + k) % 2 = k % 2 := by
  omega

/-- 8-bit: element 4q+k is byte 4q+k (one uint32). -/
theorem quad_offsets_8bit (q k : Nat) : (4 * q + k) / 1 = 4 * q + k := by
  omega

/-- decode(code) = code + bias lies in [bias, bias + 2^bits). -/
def decodeAffine (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Affine decode range. -/
theorem decodeAffine_range (code bits : Nat) (bias : Int) (h : code < 2 ^ bits) :
    bias ≤ decodeAffine code bias ∧ decodeAffine code bias < (2 ^ bits : Nat) + bias := by
  unfold decodeAffine
  omega

-- § IQ4_NL codebook

/-- The non-linear codebook. -/
def kIq4Nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (indexed by a 4-bit code). -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by
  decide

/-- All entries fit int8. -/
theorem kIq4Nl_int8 : ∀ x ∈ kIq4Nl, -128 ≤ x ∧ x ≤ 127 := by
  decide

/-- A masked nibble indexes the 16-entry table. -/
theorem nibble_index_lt (code : Nat) : code % 16 < 16 := by
  omega

-- § group size and block alignment

/-- Group sizes 16, 32, 64 are powers of two, multiples of 16. -/
theorem group_sizes : ∀ s ∈ [4, 5, 6], (2 ^ s) &&& (2 ^ s - 1) = 0 ∧ 2 ^ s % 16 = 0 := by
  decide

/-- With QE = 16 and lane*16 + 512*t offsets, i % 16 = 0. -/
theorem iter_offset_aligned (lane t : Nat) : (lane * 16 + 512 * t) % 16 = 0 := by
  omega

/-- A 16-element lane iteration never straddles a Q8_K block or a Q8_0 block. -/
theorem no_straddle (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 ∧ i % 32 + 16 ≤ 32 := by
  omega

/-- The block-in-activation split is exact. -/
theorem block_split (i : Nat) : (i / 256) * 256 + i % 256 = i := by
  omega

/-- Q8_K requirement: n_in multiple of 256 gives whole blocks. -/
theorem q8k_whole (nIn : Nat) (h : nIn % 256 = 0) : nIn / 256 * 256 = nIn := by
  omega

/-- ffn_down_shexp (n_in = 640): fine for Q8_0, structurally impossible for Q8_K. -/
theorem shexp_down_q8_0_only : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § reduction regrouping

/-- Exact-arithmetic identity behind the 16-accumulator regrouping (floats differ; hence parity tolerances). -/
theorem sixteen_regroup (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 : Int) :
    (((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))) +
      (((a8 + a9) + (a10 + a11)) + ((a12 + a13) + (a14 + a15))) =
    a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 := by
  omega

/-- Four-accumulator regrouping of the block kernel. -/
theorem four_regroup (a b c d : Int) : (a + b) + (c + d) = a + b + c + d := by
  omega

-- § launch geometry

/-- Warp-per-row kernel: 8 warps per block cover all rows. -/
theorem warp_rows_cover (nOut : Nat) : nOut ≤ 8 * ((nOut + 7) / 8) := by
  omega

/-- Thread-per-row kernel: 128 threads per block cover all rows. -/
theorem thread_rows_cover (nOut : Nat) : nOut ≤ 128 * ((nOut + 127) / 128) := by
  omega

/-- threads_per_row powers of two up to 1024 pass the host check. -/
theorem tpr_power_of_two : ∀ k < 11, (2 ^ k) &&& (2 ^ k - 1) = 0 := by
  decide

/-- Codes consumed per row: bytes = n_in / PER_BYTE, and PER_BYTE divides 32 for every width. -/
theorem per_byte_divides_32 : ∀ b ∈ [2, 4, 8], 32 % (8 / b) = 0 := by
  decide

end Strata.SGemv
