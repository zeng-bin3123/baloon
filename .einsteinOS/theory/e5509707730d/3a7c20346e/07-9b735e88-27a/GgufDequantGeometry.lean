/-!
# MODULE: gguf_dequant_geometry
SOURCES: include/strata/artifact/dequant.hpp, include/strata/artifact/gguf_reader.hpp
Block geometry of the GGUF quantization types, in-bounds field access of the scalar dequantizers, the Q2_0
symbol mapping, fp16/bf16 conversion facts, and GGUF header parsing invariants.

## Source semantics
* Q2_0: 64 elements / 18 bytes, code c -> (c-1)*d, so values lie in {-d,0,d,2d}.
* Each K-quant/IQ layout sums to its declared block size and all indexed reads stay in the block.
* GGUF magic is "GGUF" little-endian, version 3 only; data starts at the next multiple of the alignment.
* A bounds-checked cursor read either fails or moves forward without passing the end.

## Form template identifiers
geom -> block_geometry, advance -> cursor_read, kv -> iq4nl_codebook
-/

namespace Strata.GgufDequant

-- § block geometry
/-- (elements, bytes) per block for the types in the table. -/
def geom : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 2 => some (32, 18)
  | 8 => some (32, 34)
  | 6 => some (32, 22)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 20 => some (32, 18)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Q2_0 is 64 elements in 18 bytes. -/
theorem geom_q2_0 : geom 42 = some (64, 18) := by decide
/-- Q2_0 bytes = 16 code bytes + 2 scale bytes. -/
theorem q2_0_bits : 18 * 8 = 64 * 2 + 16 := by decide
/-- Q4_0 / IQ4_NL: 4 bits per element plus a 2-byte scale. -/
theorem q4_0_bits : 18 * 8 = 32 * 4 + 16 := by decide
/-- Q8_0: 8 bits per element plus a 2-byte scale. -/
theorem q8_0_bits : 34 * 8 = 32 * 8 + 16 := by decide
/-- Q4_0 and IQ4_NL share a geometry. -/
theorem q4_0_iq4nl_same : geom 2 = geom 20 := by decide
/-- Q5_0: scale + 4 high-bit bytes + 16 nibble bytes. -/
theorem q5_0_layout : 2 + 4 + 16 = 22 := by decide
/-- Q6_K: ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6_K_layout : 128 + 64 + 16 + 2 = 210 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide
/-- Q4_K: d, dmin, 12 scale bytes, 128 quant bytes; qs at 16. -/
theorem q4_K_layout : 2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 = 16 := by decide
/-- Q5_K: d, dmin, 12 scales, 32 qh, 128 ql; qh at 16, ql at 48. -/
theorem q5_K_layout : 2 + 2 + 12 + 32 + 128 = 176 ∧ 16 + 32 = 48 := by decide
/-- Q3_K: hmask 32, qs 64, scales 12, d 2; scales at 96, d at 108. -/
theorem q3_K_layout : 32 + 64 + 12 + 2 = 110 ∧ 32 + 64 = 96 ∧ 96 + 12 = 108 := by decide
/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128; qs at 8. -/
theorem iq4_xs_layout : 2 + 2 + 4 + 128 = 136 ∧ 2 + 2 + 4 = 8 := by decide

-- § Q2_0 dequantization
/-- Code j of a block lives in byte j/4 (inside the 16 code bytes) at bit offset 2*(j%4). -/
theorem q2_0_code_position (j : Nat) (h : j < 64) :
    j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ (j % 4) * 2 % 2 = 0 := by omega

/-- The decoded value (c-1)*d is one of -d, 0, d, 2d. -/
theorem q2_0_value_set (c d : Int) (h0 : 0 ≤ c) (h3 : c ≤ 3) :
    (c - 1) * d = -d ∨ (c - 1) * d = 0 ∨ (c - 1) * d = d ∨ (c - 1) * d = 2 * d := by
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases h with rfl | rfl | rfl | rfl <;> omega

-- § K-quant index bounds
/-- Q6_K: reads at l+96, l+32 and sc[is+6] stay in bounds for l < 32. -/
theorem q6_K_in_bounds (l : Nat) (h : l < 32) :
    l + 96 < 128 ∧ l + 32 < 64 ∧ l < 32 ∧ l / 16 + 6 < 8 := by omega

/-- Q6_K: two 128-element halves consume 128 ql, 64 qh and 16 scales in total. -/
theorem q6_K_totals : 2 * 128 = 256 ∧ 2 * 64 = 128 ∧ 2 * 32 = 64 ∧ 2 * 8 = 16 := by decide

/-- A 6-bit value built from a nibble and two high bits is below 64, so subtracting 32 gives [-32,31]. -/
theorem six_bit_range (a h : Nat) (ha : a < 16) (hh : h < 4) :
    a + 16 * h < 64 ∧ -32 ≤ ((a + 16 * h : Nat) : Int) - 32 ∧ ((a + 16 * h : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q4_K/Q5_K scale-min unpack indices stay inside the 12 scale bytes for j < 8. -/
theorem scale_min_k4_in_bounds (j : Nat) (h : j < 8) :
    (j < 4 → j + 4 < 12) ∧ (4 ≤ j → j + 4 < 12 ∧ j - 4 < 12) := by omega

/-- Q4_K: four 64-element groups produce 256 outputs and consume 128 quant bytes and 8 scale pairs. -/
theorem q4_K_totals : 4 * (32 + 32) = 256 ∧ 4 * 32 = 128 ∧ 4 * 2 = 8 := by decide

/-- Q5_K high-bit masks 1,4,16,64 and 2,8,32,128 fit in a byte for all four groups. -/
theorem q5_K_masks : ∀ n : Nat, n < 4 → 2 ^ (2 * n) < 256 ∧ 2 * 2 ^ (2 * n) < 256 := by decide

/-- Q3_K: 2 halves x 4 shifts x 2 scale uses = 16 scales, and 2*4*2*16 = 256 outputs. -/
theorem q3_K_totals : 2 * 4 * 2 = 16 ∧ 2 * 4 * 2 * 16 = 256 := by decide

/-- Q3_K: reads at l+16 stay inside the 32-byte hmask and the 32-byte half of qs. -/
theorem q3_K_in_bounds (l : Nat) (h : l < 16) : l + 16 < 32 := by omega

/-- IQ4_XS: the 6-bit scale index ib/2 and bit shift 2*ib stay in range for ib < 8. -/
theorem iq4_xs_scale_bounds (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib + 2 ≤ 16 := by omega

/-- IQ4_NL codebook has 16 entries. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries (one per nibble). -/
theorem kv_length : kv.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kv_sorted : List.Pairwise (· < ·) kv := by decide

/-- Low and high nibbles of a byte are valid codebook indices. -/
theorem nibble_index_lt (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 := by omega

-- § fp16 / bf16
/-- Normal fp16 exponents 1..30 map to fp32 exponents 113..142, inside the normal fp32 range. -/
theorem fp16_normal_exp (e : Nat) (h1 : 1 ≤ e) (h2 : e ≤ 30) :
    1 ≤ e - 15 + 127 ∧ e - 15 + 127 ≤ 254 := by omega

/-- bf16 widening by a 16-bit shift is invertible. -/
theorem bf16_roundtrip (h : Nat) : (h * 65536) / 65536 = h := by omega

/-- bf16 widening stays below 2^32 for 16-bit inputs. -/
theorem bf16_fits (h : Nat) (hh : h < 65536) : h * 65536 < 4294967296 := by omega

-- § GGUF header
/-- The bytes "GGUF" read little-endian give 0x46554747. -/
theorem gguf_magic : 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 = 0x46554747 := by decide

/-- Rounding the header end up to the default alignment of 32 yields a multiple of 32 within 31 bytes. -/
theorem data_start_align32 (p : Nat) :
    (p + 31) / 32 * 32 % 32 = 0 ∧ p ≤ (p + 31) / 32 * 32 ∧ (p + 31) / 32 * 32 < p + 32 := by
  omega

/-- A bounds-checked cursor read. -/
def advance (pos size n : Nat) : Option Nat := if pos + n ≤ size then some (pos + n) else none

/-- A successful read moves forward and stays inside the file. -/
theorem advance_bounds (pos size n p' : Nat) (h : advance pos size n = some p') :
    pos ≤ p' ∧ p' ≤ size := by
  unfold advance at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A read that would pass the end fails. -/
theorem advance_fails (pos size n : Nat) (h : size < pos + n) : advance pos size n = none := by
  unfold advance
  rw [if_neg (by omega)]

/-- Only a sample of at most 64 array items is kept. -/
theorem array_sample_le (count : Nat) : min count 64 ≤ 64 := by omega

/-- The default guard constants: 10 used experts out of 512, 24 query heads over 2 KV heads. -/
theorem guard_defaults : 10 ≤ 512 ∧ 24 % 2 = 0 := by decide

end Strata.GgufDequant
