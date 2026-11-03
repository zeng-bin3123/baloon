/-!
# MODULE: s_gemv_quad_decode
The S-family GEMV (`s_gemv.cu`): one thread / one block / one warp per output row, dequantising packed
2/4/8-bit codes on the fly and dotting against FP16, Q8_K or Q8_0 activations.  This module pins the index
arithmetic the optimised kernels rely on: packed-word bit order, group containment, lane partition,
alignment of the wide loads, tail-loop emptiness, activation-block strides and the IQ4_NL codebook.

## Source semantics
* Element `i` of a row has code `(c[i/PER_BYTE] >> ((i%PER_BYTE)*bits)) & mask`, `PER_BYTE = 8/bits`.
* The wide kernels read a little-endian word holding several consecutive codes; the bit order must match.
* `group_shift = log2 group_elems`, so `i >> shift = i / group_elems`.
* One lane-iteration of the Q8 split kernel (QE = 16) never straddles a Q8_K (256) or Q8_0 (32) block.
* The weight is decoded BEFORE the activation multiply (`w = code*d + b`, then `acc += w*x`).

## Form template identifiers
- `SForm.code_bits`  → `bits`
- `SForm.group_elems` → `g`
- `SForm.code_bias`  → `bias`
- `kIq4Nl`           → `kIq4Nl`
- `Q8K_BLOCK_BYTES / Q8_0_BLOCK_BYTES` → `q8kBytes / q80Bytes`
- lane index          → `lane`, vector length `n_in` → `n`
-/

namespace Strata.SGemv

-- § Packed word bit order

/-- S2: element `k` of a code byte is bits `[2k,2k+2)`; here the four fields of one byte, in element order. -/
theorem s2_fields (b : Nat) (hb : b < 256) :
    b = b % 4 + 4 * (b / 4 % 4) + 16 * (b / 16 % 4) + 64 * (b / 64 % 4) := by omega

/-- S4: a uint16 word `b0 + 256*b1` holds four nibbles; nibble 0 is the low nibble of the first byte. -/
theorem s4_word_k0 (b0 b1 : Nat) : (b0 + 256 * b1) % 16 = b0 % 16 := by omega

/-- S4 nibble 1 is the high nibble of the first byte. -/
theorem s4_word_k1 (b0 b1 : Nat) (h0 : b0 < 256) : (b0 + 256 * b1) / 16 % 16 = b0 / 16 := by omega

/-- S4 nibble 2 is the low nibble of the second byte. -/
theorem s4_word_k2 (b0 b1 : Nat) (h0 : b0 < 256) : (b0 + 256 * b1) / 256 % 16 = b1 % 16 := by omega

/-- S4 nibble 3 is the high nibble of the second byte. -/
theorem s4_word_k3 (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1) / 4096 % 16 = b1 / 16 := by omega

/-- S8: a uint32 word is four bytes, little endian; byte 0. -/
theorem s8_word_k0 (b0 b1 b2 b3 : Nat) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) % 256 = b0 % 256 := by omega

/-- S8 byte 1. -/
theorem s8_word_k1 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 256 % 256 = b1 % 256 := by omega

/-- S8 byte 2. -/
theorem s8_word_k2 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 65536 % 256 = b2 % 256 := by omega

/-- S8 byte 3. -/
theorem s8_word_k3 (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) :
    (b0 + 256 * b1 + 65536 * b2 + 16777216 * b3) / 16777216 % 256 = b3 % 256 := by omega

-- § Word widths and alignment

/-- Bytes per four-code word `QW = 4*bits/8`: 1, 2, 4 for S2, S4, S8 (not the octet width). -/
theorem word_bytes_values :
    4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Four words of one lane-iteration (QE = 16) cover exactly `16*bits/8` bytes. -/
theorem octet_bytes (bits : Nat) (h : bits = 2 ∨ bits = 4 ∨ bits = 8) :
    4 * (4 * bits / 8) = 16 * bits / 8 := by
  rcases h with rfl | rfl | rfl <;> decide

/-- The wide code loads are naturally aligned when `i` is a multiple of 16: S4 reads uint16 at byte `i/2`. -/
theorem s4_load_aligned (i : Nat) (h : i % 16 = 0) : (i / 2) % 2 = 0 := by omega

/-- S8 reads uint32 at byte `i`. -/
theorem s8_load_aligned (i : Nat) (h : i % 16 = 0) : i % 4 = 0 := by omega

/-- Row base alignment for the 16-wide kernel: a row of `n` elements (n multiple of 16) holds a whole number of words. -/
theorem row_base_aligned (n : Nat) (h : n % 16 = 0) :
    (n / 2) % 2 = 0 ∧ n % 4 = 0 ∧ (n / 4) % 1 = 0 := by omega

/-- The split kernel (QE = 4): `i` a multiple of 4 gives aligned uint16 (S4) and uint32 (S8) loads. -/
theorem quad_load_aligned (i : Nat) (h : i % 4 = 0) : (i / 2) % 2 = 0 ∧ i % 4 = 0 := by omega

/-- The paired-half load `x + i` is 8-byte aligned when `i % 4 = 0` (halves are 2 bytes). -/
theorem half4_aligned (i : Nat) (h : i % 4 = 0) : (2 * i) % 8 = 0 := by omega

-- § Group index as a shift

/-- `i >> s = i / 2^s`: the group index computed with the logarithm is the division it replaces. -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

/-- A quad (4 elements starting at a multiple of 4) lies in one group for every power-of-two group ≥ 4. -/
theorem quad_in_group (i : Nat) (h : i % 4 = 0) :
    (i + 3) / 4 = i / 4 ∧ (i + 3) / 8 = i / 8 ∧ (i + 3) / 16 = i / 16 ∧
    (i + 3) / 32 = i / 32 ∧ (i + 3) / 64 = i / 64 := by omega

/-- An octet/16-run starting at a multiple of 16 lies in one group for the group sizes 16, 32, 64. -/
theorem run16_in_group (i : Nat) (h : i % 16 = 0) :
    (i + 15) / 16 = i / 16 ∧ (i + 15) / 32 = i / 32 ∧ (i + 15) / 64 = i / 64 := by omega

/-- A multiple of 16 followed by 8 more elements stays in the same group for group sizes 16, 32, 64. -/
theorem octet_in_group (i : Nat) (h : i % 8 = 0) :
    (i + 7) / 8 = i / 8 ∧ (i + 7) / 16 = i / 16 ∧ (i + 7) / 32 = i / 32 ∧ (i + 7) / 64 = i / 64 := by omega

-- § Lane partition and tails

/-- Lane `lane` visits `i = lane*16 + 512*m`; every multiple of 16 is visited by exactly one `(lane, m)`. -/
theorem lane_cover (e : Nat) (h : e % 16 = 0) :
    e = ((e / 16) % 32) * 16 + 512 * (e / 512) := by omega

/-- The lane of a position is below 32. -/
theorem lane_lt (e : Nat) : (e / 16) % 32 < 32 := by omega

/-- Distinct `(lane, m)` give distinct positions. -/
theorem lane_inj (l1 m1 l2 m2 : Nat) (h1 : l1 < 32) (h2 : l2 < 32)
    (h : l1 * 16 + 512 * m1 = l2 * 16 + 512 * m2) : l1 = l2 ∧ m1 = m2 := by omega

/-- When `n % 16 = 0` the partial-quad tail loop of the 16-wide kernel is empty. -/
theorem tail16_empty (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : ¬ (i + 16 ≤ n)) : n ≤ i := by omega

/-- When `n % 4 = 0` the tail loop of the 4-wide kernel is empty. -/
theorem tail4_empty (i n : Nat) (hi : i % 4 = 0) (hn : n % 4 = 0) (h : ¬ (i + 4 ≤ n)) : n ≤ i := by omega

/-- Generic stride uniqueness: offsets below the period determine `(offset, iteration)`. -/
theorem stride_inj (a b P m1 m2 : Nat) (hP : 0 < P) (ha : a < P) (hb : b < P)
    (h : a + P * m1 = b + P * m2) : a = b ∧ m1 = m2 := by
  have h1 : (a + P * m1) / P = m1 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt ha, Nat.zero_add]
  have h2 : (b + P * m2) / P = m2 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt hb, Nat.zero_add]
  rw [h] at h1
  have hm : m1 = m2 := h1.symm.trans h2
  subst hm
  omega

/-- Thread `tid` of the split kernel visits `i = tid*4 + threads*4*m`; positions are unique per `(tid, m)`. -/
theorem split_stride_inj (tpr t1 m1 t2 m2 : Nat) (hp : 0 < tpr) (h1 : t1 < tpr) (h2 : t2 < tpr)
    (h : t1 * 4 + tpr * 4 * m1 = t2 * 4 + tpr * 4 * m2) : t1 = t2 ∧ m1 = m2 := by
  have := stride_inj (t1 * 4) (t2 * 4) (tpr * 4) m1 m2 (by omega) (by omega) (by omega) h
  omega

-- § Activation block strides

/-- A Q8_K block is `{f32 d; int8 qs[256]; int16 bsums[16]}`. -/
theorem q8k_block_bytes : 4 + 256 + 2 * 16 = 292 := by decide

/-- A Q8_0 block is `{fp16 d; int8 qs[32]}`. -/
theorem q80_block_bytes : 2 + 32 = 34 := by decide

/-- Every Q8_K block's `d` is 4-byte aligned when the base is. -/
theorem q8k_d_aligned (b : Nat) : (b * 292) % 4 = 0 := by omega

/-- Every Q8_0 block's `d` is 2-byte aligned (the stride is 34). -/
theorem q80_d_aligned (b : Nat) : (b * 34) % 2 = 0 := by omega

/-- A 16-run starting at a multiple of 16 stays inside one Q8_K block. -/
theorem run16_in_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A 16-run starting at a multiple of 16 stays inside one Q8_0 block. -/
theorem run16_in_q80 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Block index and in-block offset recompose the element index (Q8_K). -/
theorem q8k_split (i : Nat) : (i / 256) * 256 + i % 256 = i := by omega

/-- Block index and in-block offset recompose the element index (Q8_0). -/
theorem q80_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- `ffn_down_shexp` has `n_in = 640`: a multiple of 32 but not of 256, so Q8_K is structurally impossible. -/
theorem n640_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

-- § Weight decode order (integer domain bias, then scale, then offset)

/-- Integer-domain bias is applied to the code before the float conversion. -/
def biasedCode (code bias : Int) : Int := code + bias

/-- The S2 bias of -1 maps codes 0..3 to -1..2. -/
theorem s2_bias_range (c : Nat) (h : c < 4) : -1 ≤ biasedCode c (-1) ∧ biasedCode c (-1) ≤ 2 := by
  unfold biasedCode; omega

-- § The non-linear IQ4_NL codebook

/-- The 16-entry codebook, verbatim. -/
def kIq4Nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Sixteen entries, indexed by the low nibble. -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by decide

/-- Every low-nibble lookup is in range. -/
theorem iq4nl_index_lt (code : Nat) : code % 16 < kIq4Nl.length := by
  rw [kIq4Nl_length]; omega

/-- The codebook is strictly increasing (non-linear spacing, monotone). -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by decide

/-- Every entry fits in int8. -/
theorem kIq4Nl_int8 : ∀ x ∈ kIq4Nl, -128 ≤ x ∧ x ≤ 127 := by decide

/-- The codebook is not an affine map of the code: the first two gaps differ. -/
theorem kIq4Nl_nonlinear : (-104 : Int) - (-127) ≠ (-83 : Int) - (-104) := by decide

-- § Shape checks of the host wrappers

/-- A power of two is a multiple of 4 as soon as it is at least 4 (the quad-in-group guard). -/
theorem pow2_ge4_mod4 (s : Nat) (h : 2 ≤ s) : 2 ^ s % 4 = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, s = k + 2 := ⟨s - 2, by omega⟩
  have : 2 ^ (k + 2) = 2 ^ k * 4 := by
    rw [Nat.pow_succ, Nat.pow_succ]; omega
  omega

/-- Codes per row for each width. -/
theorem codes_per_row_values (n : Nat) (h : n % 4 = 0) :
    n / 4 * 4 = n ∧ n / 2 * 2 = n ∧ n / 1 = n := by omega

/-- The q8k split launcher uses `threads / 32` rows per block; the grid covers all rows. -/
theorem warp_grid_covers (n_out warps : Nat) (hw : 0 < warps) :
    n_out ≤ warps * ((n_out + warps - 1) / warps) := by
  have := Nat.div_add_mod (n_out + warps - 1) warps
  have := Nat.mod_lt (n_out + warps - 1) hw
  omega

end Strata.SGemv
