-- Baloon.Pack: 2-bit and 4-bit packing, nibble splitting, integer code bias and
-- bit-mask index proofs for Q2_0/Q4_0/Q5_0/Q3_K/Q5_K/Q6_K/Q8_0.
-- Independent implementation proving the same mathematical facts as Ref.Pack.

namespace Baloon.Pack

def pack2 (c0 c1 c2 c3 : Nat) : Nat := (c0 ||| (c1 <<< 2) ||| (c2 <<< 4) ||| (c3 <<< 6))
def pack4 (a b : Nat) : Nat := (a ||| (b <<< 4))
def field (byte k bits : Nat) : Nat := ((byte >>> (k * bits)) &&& ((1 <<< bits) - 1))

theorem pack2_roundtrip : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    field (pack2 c0 c1 c2 c3) 0 2 = c0 ∧ field (pack2 c0 c1 c2 c3) 1 2 = c1 ∧
    field (pack2 c0 c1 c2 c3) 2 2 = c2 ∧ field (pack2 c0 c1 c2 c3) 3 2 = c3 := by { decide }

theorem pack2_lt : ∀ c0 : Nat, c0 < 4 → ∀ c1 : Nat, c1 < 4 → ∀ c2 : Nat, c2 < 4 → ∀ c3 : Nat, c3 < 4 →
    pack2 c0 c1 c2 c3 < 256 := by { decide }

theorem pack4_roundtrip : ∀ a : Nat, a < 16 → ∀ b : Nat, b < 16 →
    field (pack4 a b) 0 4 = a ∧ field (pack4 a b) 1 4 = b ∧ pack4 a b < 256 := by { decide }

set_option maxRecDepth 100000 in
theorem nibble_split : ∀ b : Nat, b < 256 →
    (b >>> 4) * 16 + (b &&& 15) = b ∧ (b &&& 15) < 16 ∧ (b >>> 4) < 16 := by { decide }

def biased (code : Nat) (bias : Int) : Int := ((code : Int) + bias)

theorem q2_codebook (c : Nat) (h : c < 4) : -1 ≤ biased c (-1) ∧ biased c (-1) ≤ 2 := by { unfold biased; omega }
theorem q4_codebook (c : Nat) (h : c < 16) : -8 ≤ biased c (-8) ∧ biased c (-8) ≤ 7 := by { unfold biased; omega }
theorem q5_codebook (c : Nat) (h : c < 32) : -16 ≤ biased c (-16) ∧ biased c (-16) ≤ 15 := by { unfold biased; omega }
theorem q3k_codebook (c : Nat) (h : c < 8) : -4 ≤ biased c (-4) ∧ biased c (-4) ≤ 3 := by { unfold biased; omega }
theorem q6k_codebook (c : Nat) (h : c < 64) : -32 ≤ biased c (-32) ∧ biased c (-32) ≤ 31 := by { unfold biased; omega }
theorem q8_codebook (q : Int) (h1 : -128 ≤ q) (h2 : q ≤ 127) :
    ∃ c : Nat, c < 256 ∧ biased c (-128) = q := by
  refine ⟨(q + 128).toNat, by omega, ?_⟩; unfold biased; omega

theorem q3k_mask_bits : ∀ half : Nat, half < 2 → ∀ j : Nat, j < 4 → 4 * half + j < 8 := by { decide }

/-- Q5_K masks u1 = 1 << 2j, u2 = 2 << 2j walk the eight distinct bits. -/
theorem q5k_masks : ∀ j : Nat, j < 4 →
    (1 <<< (2 * j)) = 2 ^ (2 * j) ∧ (2 <<< (2 * j)) = 2 ^ (2 * j + 1) := by { decide }

/-- Q5_0 fifth-bit shift j+12 with mask 0x10 reads bit j+16 of qh. -/
theorem q5_0_high_bit : ∀ j : Nat, j < 16 → ((1 <<< (j + 16)) >>> (j + 12)) &&& 16 = 16 := by { decide }

/-- Block byte sizes equal the sum of their fields. -/
theorem block_bytes_sum :
    2 + 16 = 18 ∧ 2 + 4 + 16 = 22 ∧ 2 + 32 = 34 ∧ 32 + 64 + 12 + 2 = 110 ∧
    2 + 2 + 12 + 128 = 144 ∧ 2 + 2 + 12 + 32 + 128 = 176 ∧ 128 + 64 + 16 + 2 = 210 ∧
    2 + 2 + 4 + 128 = 136 := by { decide }

/-- Group sizes divide block widths in every mapping. -/
theorem group_divides_block :
    64 % 64 = 0 ∧ 256 % 16 = 0 ∧ 32 % 32 = 0 ∧ 256 % 32 = 0 ∧ 1 % 1 = 0 := by { decide }

/-- Scale counts per block. -/
theorem group_counts : 256 / 16 = 16 ∧ 256 / 32 = 8 ∧ 64 / 64 = 1 := by { decide }

/-- Padding before a 64-byte-aligned plane. -/
def pad (len : Nat) : Nat := if len % 64 = 0 then 0 else 64 - len % 64

/-- After padding the offset is a multiple of 64. -/
theorem pad_aligns (len : Nat) : (len + pad len) % 64 = 0 := by
  unfold pad; split_ifs with h
  · omega
  · have := Nat.mod_lt len (show 0 < 64 from by norm_num); omega

/-- Padding is smaller than the alignment. -/
theorem pad_lt (len : Nat) : pad len < 64 := by
  unfold pad; split_ifs with h
  · norm_num
  · have := Nat.mod_lt len (show 0 < 64 from by norm_num); omega

/-- An already aligned offset gets no padding. -/
theorem pad_zero (len : Nat) (h : len % 64 = 0) : pad len = 0 := by unfold pad; simp [h]

/-- Scale dtype: fp16 (2 bytes) or fp32 (4 bytes). -/
def scaleBytes : Bool → Nat | true => 2 | false => 4

/-- FP32 fallback costs at most twice the bytes. -/
theorem scaleBytes_le (b : Bool) : 2 ≤ scaleBytes b ∧ scaleBytes b ≤ 4 := by
  cases b <;> simp [scaleBytes] <;> omega

-- ── MTP expert quantizer (Q2_0 on {-1,0,1,2}) ────────────────────────────────

/-- Clip to the Q2_0 grid {-1, 0, 1, 2}. -/
def clipQ (q : Int) : Int := if q < -1 then -1 else if q > 2 then 2 else q

/-- Stored code is clipQ(q) + 1, always in 0..3. -/
theorem mtp_code_range (q : Int) : 0 ≤ clipQ q + 1 ∧ clipQ q + 1 ≤ 3 := by
  simp only [clipQ]; split_ifs <;> omega

/-- Decoding the stored code returns the clipped value. -/
theorem mtp_decode_encode (q : Int) : (clipQ q + 1) - 1 = clipQ q := by
  simp only [clipQ]; split_ifs <;> omega

/-- Scale search: 17 candidates, factor (16+k)/32 from 1/2 (k=0) to 1 (k=16). -/
theorem mtp_scale_candidates : (16 + 0 : Nat) * 2 = 32 ∧ 16 + 16 = 32 ∧ 16 + 1 = 17 := by norm_num

/-- MTP routed-expert bytes in Q2_0: gate_up [512,1280,2560] and down [512,2560,640]. -/
theorem mtp_q2_expert_bytes :
    512 * 1280 * (2560 / 64 * 18) = 471859200 ∧ 512 * 2560 * (640 / 64 * 18) = 235929600 ∧
    471859200 + 235929600 = 707788800 := by norm_num

/-- Inner dimensions are multiples of every supported block width. -/
theorem mtp_inner_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by norm_num

-- ── IQ4 non-linear codebook ───────────────────────────────────────────────────

/-- kvalues_iq4nl: the non-affine 4-bit codebook. -/
def kvIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10] ++
  [1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by { decide }

/-- The codebook is strictly increasing. -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (· < ·) := by { decide }

/-- The codebook is non-affine: second differences are not all zero. -/
theorem kvIq4nl_nonaffine : (kvIq4nl[1]! - kvIq4nl[0]!) ≠ (kvIq4nl[2]! - kvIq4nl[1]!) := by { decide }

end Baloon.Pack
