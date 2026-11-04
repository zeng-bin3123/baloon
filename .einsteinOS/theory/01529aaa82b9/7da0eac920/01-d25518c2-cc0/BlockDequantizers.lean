/-!
# MODULE: block_dequantizers
Scalar reference dequantizers for the GGUF block encodings (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K..Q6_K,
IQ4_XS) plus fp16/bf16 widening. The theory covers the index arithmetic, layout offsets and value ranges.

## Source semantics
* Q2_0 block = fp16 d + 16 bytes = 64 elements, 18 bytes; code {0,1,2,3} -> symbol {-1,0,+1,+2};
  4 codes per byte, least-significant first.
* Every K-quant offset/stride expression stays inside its super-block.
* A 6-bit scale assembled from a low nibble and two high bits is below 64.
* bf16 -> fp32 is exact (shift left 16); taking the high 16 bits back is a left inverse.

## Form template identifiers
* `dequantize_q2_0` -> `q2Value` / `q2Code`
* `get_scale_min_k4` -> `scale6_*`
* `kvalues_iq4nl` -> `kvaluesIq4nl`
* `dequantize_q6_K` -> `q6K_indices`
-/

namespace Strata.Dequant

-- § Q2_0

/-- Value of a Q2_0 code under scale `d` (scale modelled over `Int`). -/
def q2Value (code : Nat) (d : Int) : Int := ((code : Int) - 1) * d

/-- The 2-bit code of element `j` read out of byte `byte` (4 codes per byte, LSB first). -/
def q2Code (byte j : Nat) : Nat := (byte / 2 ^ ((j % 4) * 2)) % 4

/-- Every extracted Q2_0 code is a valid 2-bit code. -/
theorem q2Code_lt (byte j : Nat) : q2Code byte j < 4 := by
  unfold q2Code
  exact Nat.mod_lt _ (by decide)

/-- Q2_0 values are exactly one of `{-d, 0, d, 2d}`: the structural invariant the `--check` mode asserts. -/
theorem q2Value_mem (code : Nat) (d : Int) (h : code < 4) :
    q2Value code d = -d ∨ q2Value code d = 0 ∨ q2Value code d = d ∨ q2Value code d = 2 * d := by
  unfold q2Value
  have h4 : code = 0 ∨ code = 1 ∨ code = 2 ∨ code = 3 := by omega
  rcases h4 with rfl | rfl | rfl | rfl <;> omega

/-- Byte and bit indices of Q2_0 stay inside the 16-byte payload. -/
theorem q2_index_bounds (j : Nat) (h : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by
  omega

/-- Packing four codes into a byte and extracting them returns the same codes (bit order check). -/
theorem q2_pack_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    q2Code (c0 + 4 * c1 + 16 * c2 + 64 * c3) 0 = c0 ∧
    q2Code (c0 + 4 * c1 + 16 * c2 + 64 * c3) 1 = c1 ∧
    q2Code (c0 + 4 * c1 + 16 * c2 + 64 * c3) 2 = c2 ∧
    q2Code (c0 + 4 * c1 + 16 * c2 + 64 * c3) 3 = c3 := by
  simp only [q2Code, Nat.reduceMod, Nat.reduceMul, Nat.reducePow]
  omega

-- § fp16 / bf16

/-- The fp32 bit pattern of a bf16 value is the 16 bits shifted into the high half. -/
def bf16Bits (h : Nat) : Nat := h * 65536

/-- Taking the high 16 bits of the widened bf16 recovers it: the widening is exact. -/
theorem bf16_roundtrip (h : Nat) : bf16Bits h / 65536 = h := by
  unfold bf16Bits
  omega

/-- The widened bf16 fits in 32 bits. -/
theorem bf16_fits32 (h : Nat) (hh : h < 65536) : bf16Bits h < 4294967296 := by
  unfold bf16Bits
  omega

/-- A normal fp16 exponent (1..30) rebiases into the valid fp32 normal range. -/
theorem fp16_normal_exponent (e : Nat) (h1 : 1 ≤ e) (h2 : e ≤ 30) :
    0 < e - 15 + 127 ∧ e - 15 + 127 < 255 := by
  omega

/-- Subnormal renormalisation shifts at most 10 times and keeps the fp32 exponent positive. -/
theorem fp16_subnormal_exponent (s : Nat) (h : s ≤ 10) : 0 < 127 - 15 + 1 - s := by
  omega

-- § block geometry

/-- Q2_0 geometry: 2-byte scale plus 64/4 payload bytes is 18. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by
  decide

/-- Header plus payload sizes of the legacy 32-element blocks and the K-quant super-blocks. -/
theorem block_sizes :
    2 + 16 = 18 ∧ 2 + 32 = 34 ∧ 2 + 4 + 16 = 22 ∧ 128 + 64 + 16 + 2 = 210 ∧ 4 + 12 + 128 = 144 ∧
    4 + 12 + 32 + 128 = 176 ∧ 32 + 64 + 12 + 2 = 110 ∧ 2 + 2 + 4 + 128 = 136 := by
  decide

/-- Q6_K layout: ql, qh, scales, and the trailing fp16 `d` tile the 210 bytes exactly. -/
theorem q6K_layout : 128 + 64 = 192 ∧ 192 + 16 = 208 ∧ 208 + 2 = 210 := by
  decide

/-- Q6_K: every ql/qh/scale/output index used in a 128-element half stays in its array. -/
theorem q6K_indices (h l : Nat) (hh : h < 2) (hl : l < 32) :
    h * 64 + l + 32 < 128 ∧ h * 32 + l < 64 ∧ h * 8 + l / 16 + 6 < 16 ∧ h * 128 + l + 96 < 256 := by
  omega

/-- Q6_K: a 4-bit low part plus 2 high bits, minus 32, lies in [-32, 31]. -/
theorem q6K_value_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + hi * 16 : Nat) : Int) - 32 ∧ ((lo + hi * 16 : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q4_K / Q5_K: the scale-pair bytes `q[j]`, `q[j+4]`, `q[j-4]` all lie inside `scales[12]`. -/
theorem scale_min_indices (j : Nat) (h : j < 8) :
    (j < 4 → j + 4 < 12) ∧ (4 ≤ j → j + 4 < 12 ∧ j - 4 < 12) := by
  omega

/-- A 6-bit scale built from a low nibble and the top two bits of a byte is below 64. -/
theorem scale6_hi_bound (a b : Nat) (hb : b < 256) : a % 16 + (b / 64) * 16 < 64 := by
  omega

/-- A 6-bit scale built from a low nibble and a 2-bit field (IQ4_XS) is below 64. -/
theorem scale6_field_bound (a b : Nat) : a % 16 + (b % 4) * 16 < 64 := by
  omega

/-- Q4_K / Q5_K: group `g` of 64 touches scale indices `2g, 2g+1`, qs bytes below 128 and outputs below 256. -/
theorem q4K_group_bounds (g : Nat) (h : g < 4) :
    2 * g + 1 < 8 ∧ 32 * g + 31 < 128 ∧ 64 * g + 63 < 256 := by
  omega

/-- Q5_K: the high-bit masks are 1,4,16,64 (u1) and 2,8,32,128 (u2): a shift by 2 per group. -/
theorem q5K_masks :
    (List.range 4).map (fun g => 4 ^ g) = [1, 4, 16, 64] ∧
    (List.range 4).map (fun g => 2 * 4 ^ g) = [2, 8, 32, 128] := by
  decide

/-- Q3_K: the high-bit mask `m = 2^k` stays a byte for all 8 shifts used. -/
theorem q3K_mask_in_byte : ∀ k, k < 8 → 2 ^ k < 256 := by
  decide

/-- Q3_K: scale index (2 halves x 4 groups x 2) and the q/hmask offsets stay in range. -/
theorem q3K_indices (h j s l : Nat) (hh : h < 2) (hj : j < 4) (hs : s < 2) (hl : l < 16) :
    h * 8 + j * 2 + s < 16 ∧ h * 32 + s * 16 + l < 64 ∧ h * 128 + j * 32 + s * 16 + l < 256 := by
  omega

/-- Q3_K layout: hmask, qs, scales and the trailing fp16 `d` tile the 110 bytes. -/
theorem q3K_layout : 32 + 64 = 96 ∧ 96 + 12 = 108 ∧ 108 + 2 = 110 := by
  decide

/-- Q4_K / Q5_K layout offsets: scales at 4, qs/qh at 16, Q5_K ql at 48. -/
theorem k4_k5_layout : 4 + 12 = 16 ∧ 16 + 128 = 144 ∧ 16 + 32 = 48 ∧ 48 + 128 = 176 := by
  decide

-- § nibble layouts and codebook

/-- The IQ4_NL codebook, verbatim from ggml-common.h. -/
def kvaluesIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, one per nibble value. -/
theorem kvalues_length : kvaluesIq4nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kvalues_sorted : kvaluesIq4nl.Pairwise (· < ·) := by
  decide

/-- Every codebook entry fits an int8 and lies in [-127, 113]. -/
theorem kvalues_range : ∀ x ∈ kvaluesIq4nl, -127 ≤ x ∧ x ≤ 113 := by
  decide

/-- IQ4_XS: group `ib` of 32 reads qs bytes below 128 and writes outputs below 256. -/
theorem iq4xs_group_bounds (ib j : Nat) (hib : ib < 8) (hj : j < 16) :
    ib * 16 + j < 128 ∧ ib * 32 + j + 16 < 256 ∧ ib / 2 < 4 := by
  omega

/-- IQ4_XS: the assembled scale `ls` is below 64, so `ls - 32` lies in [-32, 31]. -/
theorem iq4xs_scale_range (a b : Nat) :
    -32 ≤ ((a % 16 + (b % 4) * 16 : Nat) : Int) - 32 ∧ ((a % 16 + (b % 4) * 16 : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q4_0: low nibble minus 8 lies in [-8, 7]. -/
theorem q4_0_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7 := by
  omega

/-- Q5_0: the fifth bit makes a 5-bit value below 32, so `x - 16` lies in [-16, 15]. -/
theorem q5_0_range (lo hb : Nat) (hl : lo < 16) (hh : hb < 2) :
    lo + hb * 16 < 32 ∧ -16 ≤ ((lo + hb * 16 : Nat) : Int) - 16 ∧ ((lo + hb * 16 : Nat) : Int) - 16 ≤ 15 := by
  omega

/-- Q5_0 / Q4_0 nibble placement: low nibbles fill 0..15 and high nibbles 16..31, covering all 32. -/
theorem nibble_cover (j : Nat) (h : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j ≠ j + 16 := by
  omega

end Strata.Dequant
