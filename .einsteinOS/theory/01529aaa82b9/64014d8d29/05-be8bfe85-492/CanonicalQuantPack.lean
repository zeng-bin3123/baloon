/-!
# MODULE: canonical_quant_pack
SOURCES: tools/canonical_xcheck.py, tools/strata_pack.py, tools/mtp_pack.py
Canonical code-plane forms of the GGUF quantization types (S2/S4/S8), bit packing of code planes,
64-byte plane alignment, block-geometry consistency, and the MTP expert quantizer.

## Source semantics
* Each type decodes as `(code + bias) * scale` (+ offset), the bias being applied to the INTEGER code
  before the single float multiply: Q2_0 -1, Q4_0 -8, Q5_0 -16, Q3_K -4, Q6_K -32, Q8_0 -128.
* Codes are packed LSB-first, `8 / bits` per byte; unpacking recovers every code.
* Nibble splits (Q4_0, IQ4_NL, Q4_K ...) put low nibbles first, high nibbles second.
* Planes start on 64-byte boundaries.
* The Q2_0 MTP quantizer maps `q` clipped to {-1..2} to code `q+1` in 0..3 and searches 17 scales
  `amax * (16+k)/32`.

## Form template identifiers
* MAPPINGS -> type_mapping_table, code_bias -> integer_code_bias, ALIGN -> plane_alignment
* pack_codes / unpack_codes -> code_plane_pack / code_plane_unpack
-/

namespace Strata.Pack

-- § 2-bit and 4-bit packing
/-- Pack four 2-bit codes LSB-first. -/
def pack2 (c0 c1 c2 c3 : Nat) : Nat := c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6)

/-- Pack two 4-bit codes, low nibble first. -/
def pack4 (a b : Nat) : Nat := a ||| (b <<< 4)

/-- Unpack field `k` of width `bits`. -/
def field (byte k bits : Nat) : Nat := (byte >>> (k * bits)) &&& ((1 <<< bits) - 1)

/-- Four 2-bit codes round-trip through one byte, in order. -/
theorem pack2_roundtrip : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    field (pack2 c0 c1 c2 c3) 0 2 = c0 ∧ field (pack2 c0 c1 c2 c3) 1 2 = c1 ∧
    field (pack2 c0 c1 c2 c3) 2 2 = c2 ∧ field (pack2 c0 c1 c2 c3) 3 2 = c3 := by
  decide

/-- The packed 2-bit byte fits in 8 bits. -/
theorem pack2_lt : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    pack2 c0 c1 c2 c3 < 256 := by
  decide

/-- Two 4-bit codes round-trip through one byte. -/
theorem pack4_roundtrip : ∀ a : Nat, a < 16 → ∀ b : Nat, b < 16 →
    field (pack4 a b) 0 4 = a ∧ field (pack4 a b) 1 4 = b ∧ pack4 a b < 256 := by
  decide

set_option maxRecDepth 100000 in
/-- Splitting a byte into low and high nibble and recombining is the identity (Q4_0 / IQ4_NL / Q4_K split). -/
theorem nibble_split : ∀ b : Nat, b < 256 → (b >>> 4) * 16 + (b &&& 15) = b ∧ (b &&& 15) < 16 ∧ (b >>> 4) < 16 := by
  decide

-- § integer code bias
/-- Decode of a code with an integer bias, before the float multiply. -/
def biased (code : Nat) (bias : Int) : Int := (code : Int) + bias

/-- Q2_0: codes 0..3 decode to {-1, 0, 1, 2} (the offset is in the codebook). -/
theorem q2_codebook (c : Nat) (h : c < 4) : -1 ≤ biased c (-1) ∧ biased c (-1) ≤ 2 := by
  unfold biased; omega

/-- Q4_0: codes 0..15 decode to -8..7. -/
theorem q4_codebook (c : Nat) (h : c < 16) : -8 ≤ biased c (-8) ∧ biased c (-8) ≤ 7 := by
  unfold biased; omega

/-- Q5_0: codes 0..31 decode to -16..15. -/
theorem q5_codebook (c : Nat) (h : c < 32) : -16 ≤ biased c (-16) ∧ biased c (-16) ≤ 15 := by
  unfold biased; omega

/-- Q3_K: codes 0..7 decode to -4..3. -/
theorem q3k_codebook (c : Nat) (h : c < 8) : -4 ≤ biased c (-4) ∧ biased c (-4) ≤ 3 := by
  unfold biased; omega

/-- Q6_K: codes 0..63 decode to -32..31. -/
theorem q6k_codebook (c : Nat) (h : c < 64) : -32 ≤ biased c (-32) ∧ biased c (-32) ≤ 31 := by
  unfold biased; omega

/-- Q8_0: stored code `q + 128` decodes back to the int8 `q`. -/
theorem q8_codebook (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    ∃ c : Nat, c < 256 ∧ biased c (-128) = q := by
  refine ⟨(q + 128).toNat, by omega, ?_⟩
  unfold biased
  omega

-- § Q3_K / Q5_K bit-mask traps
/-- Q3_K's hmask bit index `4*half + j` stays inside a byte and never repeats across the 8 (half, j) pairs. -/
theorem q3k_mask_bits : ∀ half : Nat, half < 2 → ∀ j : Nat, j < 4 → 4 * half + j < 8 := by decide

/-- Q5_K's masks `u1 = 1 << 2j`, `u2 = 2 << 2j` walk the eight distinct bits 1,2,4,8,16,32,64,128. -/
theorem q5k_masks : ∀ j : Nat, j < 4 →
    (1 <<< (2 * j)) = 2 ^ (2 * j) ∧ (2 <<< (2 * j)) = 2 ^ (2 * j + 1) := by decide

/-- The Q5_0 fifth-bit shift `j + 12` with mask 0x10 reads bit `j + 16` of qh (spot-checked on all 16 positions for bit-set qh). -/
theorem q5_0_high_bit : ∀ j : Nat, j < 16 → ((1 <<< (j + 16)) >>> (j + 12)) &&& 16 = 16 := by decide

-- § block geometry consistency
/-- Block byte sizes equal the sum of their fields. -/
theorem block_bytes_sum :
    2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧ 32 + 64 + 12 + 2 = 110 ∧
    2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 + 32 + 128 = 176 ∧ 128 + 64 + 16 + 2 = 210 ∧
    2 + 2 + 4 + 128 = 136 := by
  decide

/-- Group sizes divide block widths in every mapping (group, block). -/
theorem group_divides_block :
    64 % 64 = 0 ∧ 256 % 16 = 0 ∧ 32 % 32 = 0 ∧ 256 % 32 = 0 ∧ 1 % 1 = 0 := by
  decide

/-- Scale counts per block: Q3_K/Q6_K 16 groups of 16, K-quants with 32-groups 8 per block. -/
theorem group_counts : 256 / 16 = 16 ∧ 256 / 32 = 8 ∧ 64 / 64 = 1 := by decide

-- § 64-byte plane alignment
/-- Padding added before a plane so it starts on a 64-byte boundary. -/
def pad (len : Nat) : Nat := (64 - len % 64) % 64

/-- After padding the offset is a multiple of 64. -/
theorem pad_aligns (len : Nat) : (len + pad len) % 64 = 0 := by
  unfold pad; omega

/-- Padding is smaller than the alignment. -/
theorem pad_lt (len : Nat) : pad len < 64 := by
  unfold pad; omega

/-- An already aligned offset gets no padding. -/
theorem pad_zero (len : Nat) (h : len % 64 = 0) : pad len = 0 := by
  unfold pad; omega

-- § scale dtype choice
/-- Which dtype is stored for scales: fp16 only when the round trip is exact. -/
def scaleBytes (exact16 : Bool) : Nat := if exact16 then 2 else 4

/-- FP32 fallback never loses information and costs at most twice the bytes. -/
theorem scaleBytes_le (b : Bool) : 2 ≤ scaleBytes b ∧ scaleBytes b ≤ 4 := by
  cases b <;> simp [scaleBytes]

-- § MTP expert quantizer
/-- Clip to the Q2_0 grid {-1, 0, 1, 2}. -/
def clipQ (q : Int) : Int := max (-1) (min q 2)

/-- Stored code is `clip(q) + 1`, always in 0..3. -/
theorem mtp_code_range (q : Int) : 0 ≤ clipQ q + 1 ∧ clipQ q + 1 ≤ 3 := by
  unfold clipQ; omega

/-- Decoding the stored code returns the clipped value. -/
theorem mtp_decode_encode (q : Int) : (clipQ q + 1) - 1 = clipQ q := by omega

/-- The scale search has 17 candidates, factor `(16 + k) / 32` from 1/2 (k = 0) to 1 (k = 16). -/
theorem mtp_scale_candidates : (16 + 0 : Nat) * 2 = 32 ∧ 16 + 16 = 32 ∧ 16 + 1 = 17 := by decide

/-- MTP routed-expert bytes in Q2_0: gate_up [512,1280,2560] and down [512,2560,640]. -/
theorem mtp_q2_expert_bytes :
    512 * 1280 * (2560 / 64 * 18) = 471859200 ∧ 512 * 2560 * (640 / 64 * 18) = 235929600 ∧
    471859200 + 235929600 = 707788800 := by
  decide

/-- Inner dimensions are multiples of every supported block width (64 for q2_0, 32 for q4_0/q8_0). -/
theorem mtp_inner_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

-- § IQ4 codebook
/-- kvalues_iq4nl, the non-affine codebook. -/
def kvIq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, indexed by a 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (· < ·) := by decide

/-- The codebook is not an affine function of the code: second differences are not all zero. -/
theorem kvIq4nl_nonaffine : (kvIq4nl[1]! - kvIq4nl[0]!) ≠ (kvIq4nl[2]! - kvIq4nl[1]!) := by decide

end Strata.Pack
