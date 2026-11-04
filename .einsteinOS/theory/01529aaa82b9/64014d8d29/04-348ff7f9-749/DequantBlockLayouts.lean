/-!
# MODULE: dequant_block_layouts
SOURCES: include/strata/artifact/gguf_reader.hpp, include/strata/artifact/dequant.hpp, src/kernels/cuda/iq_kernels.cu
Block geometry (elements, bytes) of every GGML type the engine reads, struct offsets, stride alignment,
row-byte arithmetic, and the value ranges produced by the scalar dequantizers.

## Source semantics
* `block_geometry` gives (elements per block, bytes per block); Q2_0 is 64 elements / 18 bytes.
* Block strides that are not multiples of 4 (18, 22, 34, 110, 210) force 2-byte loads in the kernels.
* A row of `n` elements occupies `n / elems * bytes` bytes; the weight total must not overflow size_t.
* Dequantized integer codes stay in fixed ranges (Q4_0: -8..7, Q5_0: -16..15, Q6_K: -32..31, ...).

## Form template identifiers
block_geometry -> block_shape_of_type ; kvalues_iq4nl -> nonlinear_codebook ; get_scale_min_k4 -> unpack_scale_min
-/

namespace Strata.DequantLayouts

-- § block geometry

/-- (elements per block, bytes per block), mirroring `block_geometry`. -/
def geom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 is 64 elements in 18 bytes: a 2-byte scale plus 64 two-bit codes. -/
theorem q2_0_geometry : geom 42 = some (64, 18) ∧ 2 + 64 * 2 / 8 = 18 := by decide

/-- Every type the native MMVQ dispatcher accepts has a known block geometry. -/
theorem supported_have_geometry :
    ∀ t ∈ [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29], (geom t).isSome = true := by
  decide

/-- Every block holds a divisor of 256 elements, so a 256-element super-block is a whole number of blocks. -/
theorem elems_divide_superblock :
    ∀ t ∈ [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 30, 42],
      (geom t).map (fun p => 256 % p.1) = some 0 := by
  decide

-- § struct layouts used by the native MMVQ kernels

/-- Q3_K: hmask 32, qs 64, scales 12, d 2 = 110 with offsets 32, 96, 108. -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by decide

/-- Q6_K: ql 128, qh 64, scales 16, d 2 = 210 with offsets 128, 192, 208. -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide

/-- Q5_K: dm 4, scales 12, qh 32, qs 128 = 176 with offsets 4, 16, 48. -/
theorem q5k_layout : 4 + 12 + 32 + 128 = 176 ∧ 4 + 12 = 16 ∧ 16 + 32 = 48 := by decide

/-- Q4_K: dm 4, scales 12, qs 128 = 144 with offsets 4, 16. -/
theorem q4k_layout : 4 + 12 + 128 = 144 ∧ 4 + 12 = 16 := by decide

/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128 = 136 with offsets 2, 4, 8. -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 = 4 ∧ 4 + 4 = 8 := by decide

/-- Q5_0: d 2, qh 4, qs 16 = 22 with offsets 2, 6. -/
theorem q50_layout : 2 + 4 + 16 = 22 ∧ 2 + 4 = 6 := by decide

/-- Block strides congruent to 2 modulo 4 need 16-bit loads; the others allow 32-bit loads. -/
theorem stride_alignment :
    ([18, 22, 34, 110, 210].all (fun b => b % 4 == 2)) = true ∧
    ([36, 136, 144, 176].all (fun b => b % 4 == 0)) = true := by
  decide

-- § row-byte arithmetic

/-- Bytes of a row of `n` elements for a block shape. -/
def rowBytes (e b n : Nat) : Nat := n / e * b

/-- A row of `k` whole blocks is `k * bytes`. -/
theorem rowBytes_whole (e b k : Nat) (he : 0 < e) : rowBytes e b (k * e) = k * b := by
  unfold rowBytes
  rw [Nat.mul_div_cancel _ he]

/-- If the row size is below `max / n_out` then the weight total does not overflow. -/
theorem weight_bytes_no_overflow (row nOut M : Nat) (hn : 0 < nOut) (h : row ≤ M / nOut) :
    row * nOut ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

-- § dequantized value ranges

/-- Q4_0: a nibble minus 8 is in [-8, 7]. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by omega

/-- Q5_0: nibble plus the fifth bit minus 16 is in [-16, 15]. -/
theorem q5_0_range (n b : Nat) (hn : n < 16) (hb : b ≤ 1) :
    -16 ≤ ((n : Int) + 16 * b) - 16 ∧ ((n : Int) + 16 * b) - 16 ≤ 15 := by omega

/-- Q6_K: low nibble plus two high bits minus 32 is in [-32, 31]. -/
theorem q6_k_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    -32 ≤ ((n : Int) + 16 * h) - 32 ∧ ((n : Int) + 16 * h) - 32 ≤ 31 := by omega

/-- Q3_K: two bits minus 0 or 4 is in [-4, 3]. -/
theorem q3_k_range (t : Nat) (ht : t < 4) (m : Nat) (hm : m = 0 ∨ m = 4) :
    -4 ≤ (t : Int) - m ∧ (t : Int) - m ≤ 3 := by omega

/-- Q2_0 symbols {-1, 0, +1, +2}. -/
theorem q2_0_symbols (c : Nat) (h : c < 4) : -1 ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by omega

/-- Q4_K / Q5_K scale for `j < 4` is a 6-bit value. -/
theorem scale_min_low (q : Nat) : q % 64 < 64 := Nat.mod_lt _ (by decide)

/-- Q4_K / Q5_K scale for `j >= 4`: 4 low bits plus two bits from a neighbour is below 64. -/
theorem scale_min_high (q q' : Nat) (hq' : q' < 256) : q % 16 + 16 * (q' / 64) < 64 := by omega

/-- The `m` half of the packed pair uses the high nibble, still below 64. -/
theorem scale_min_high_m (q q' : Nat) (hq : q < 256) (hq' : q' < 256) : q / 16 + 16 * (q' / 64) < 64 := by
  omega

/-- IQ4_XS: the 6-bit scale assembled from a nibble and two bits is below 64, so `ls - 32` is in [-32, 31]. -/
theorem iq4xs_scale (lo hi : Nat) :
    (lo % 16 + 16 * (hi % 4) < 64) ∧ -32 ≤ ((lo % 16 + 16 * (hi % 4) : Nat) : Int) - 32 ∧
      ((lo % 16 + 16 * (hi % 4) : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q5_K bit masks `u1 = 1,4,16,64` and `u2 = 2 u1` are disjoint bits that fit a byte. -/
theorem q5k_masks : ∀ u ∈ [1, 4, 16, 64], 2 * u < 256 ∧ (u &&& (2 * u)) = 0 := by decide

/-- The fifth bit adds at most 16 so a Q5_K code is at most 31. -/
theorem q5k_code_bound (n : Nat) (hn : n < 16) (b : Nat) (hb : b = 0 ∨ b = 16) : n + b ≤ 31 := by omega

end Strata.DequantLayouts
