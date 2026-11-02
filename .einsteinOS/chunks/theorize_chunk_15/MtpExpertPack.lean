/-!
# MODULE: mtp_expert_pack
Arithmetic of the MTP draft-block GGUF packer: the Q2_0 / Q4_0 / Q8_0 per-block quantizer ranges, byte
packing, and the byte accounting of the two fused routed-expert tensors.

## Source semantics
* Q2_0 quantizer: candidate scales `d = amax * f` with `f` from 0.5 to 1.0 in 17 steps; quantized level
  `q = clamp(rint(x/d), -1, 2)`; stored code is `q + 1 ∈ 0..3`, four per byte, LSB first.
* Q4_0 reference: code `min 15 (trunc(x/d + 8.5))`, low half in low nibbles, high half in high nibbles.
* Expert tensors `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]` are quantized along the
  last axis, whose length must be a multiple of the block size; stored last dim = `n/block*block_bytes`.
* Reported sizes: Q2_0 ≈ 0.71 GB, Q4_0 ≈ 1.42 GB, Q8_0 ≈ 2.67 GB.

## Form template identifiers
* `clampLevel`   ← `np.clip(np.rint(x * inv), -1, 2)`
* `q2Code`       ← `(... + 1).astype(np.uint8)`
* `packQ2`       ← `c[:,:,0] | c[:,:,1] << 2 | ...`
* `q4Code`       ← `np.minimum(15, np.trunc(x*inv + 8.5))`
* `storedLast`   ← `shape[-1] // block * block_bytes`
-/

namespace MtpPack

-- § Q2_0 quantizer ranges

/-- Clamp a rounded level to the Q2_0 grid `{-1, 0, 1, 2}`. -/
def clampLevel (q : Int) : Int := max (-1) (min 2 q)

/-- The clamped level lies in `-1..2`. -/
theorem clampLevel_range (q : Int) : -1 ≤ clampLevel q ∧ clampLevel q ≤ 2 := by
  unfold clampLevel; omega

/-- Clamping is the identity on grid levels. -/
theorem clampLevel_id (q : Int) (h1 : -1 ≤ q) (h2 : q ≤ 2) : clampLevel q = q := by
  unfold clampLevel; omega

/-- The stored Q2_0 code `clamp + 1` is in `0..3`. -/
theorem q2Code_range (q : Int) : 0 ≤ clampLevel q + 1 ∧ clampLevel q + 1 ≤ 3 := by
  have := clampLevel_range q
  omega

/-- Dequantizing a stored code recovers the clamped level: `(code - 1) = clamp`. -/
theorem q2_dequant_recovers_level (q : Int) : (clampLevel q + 1) - 1 = clampLevel q := by omega

/-- Candidate scale numerators `16 + i` over 32 span exactly 0.5 to 1.0 for `i = 0..16`. -/
theorem q2_candidate_span (i : Nat) (h : i ≤ 16) : 16 ≤ 16 + i ∧ 16 + i ≤ 32 := by omega

/-- There are 17 candidate scales. -/
theorem q2_candidate_count : (List.range 17).length = 17 := by decide

/-- Pack four 2-bit codes, LSB first. -/
def packQ2 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed Q2_0 byte fits in a byte. -/
theorem packQ2_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packQ2 c0 c1 c2 c3 < 256 := by unfold packQ2; omega

/-- The packed byte is injective in its codes. -/
theorem packQ2_injective (a0 a1 a2 a3 b0 b1 b2 b3 : Nat)
    (ha0 : a0 < 4) (ha1 : a1 < 4) (ha2 : a2 < 4) (ha3 : a3 < 4)
    (hb0 : b0 < 4) (hb1 : b1 < 4) (hb2 : b2 < 4) (hb3 : b3 < 4)
    (h : packQ2 a0 a1 a2 a3 = packQ2 b0 b1 b2 b3) : a0 = b0 ∧ a1 = b1 ∧ a2 = b2 ∧ a3 = b3 := by
  unfold packQ2 at h; omega

/-- A 64-element block yields 16 packed bytes plus a 2-byte scale = 18 bytes. -/
theorem q2_block_bytes : 64 / 4 = 16 ∧ 2 + 16 = 18 := by decide

-- § Q4_0 and Q8_0

/-- Q4_0 reference code: `min 15 n` where `n = trunc(x/d + 8.5)`. -/
def q4Code (n : Nat) : Nat := min 15 n

/-- The Q4_0 code is a 4-bit value. -/
theorem q4Code_lt (n : Nat) : q4Code n < 16 := by unfold q4Code; omega

/-- Q4_0 byte: first half in the low nibble, second half in the high nibble. -/
def packQ4 (lo hi : Nat) : Nat := lo + 16 * hi

/-- A packed Q4_0 byte fits in a byte. -/
theorem packQ4_lt (lo hi : Nat) (h1 : lo < 16) (h2 : hi < 16) : packQ4 lo hi < 256 := by
  unfold packQ4; omega

/-- Q4_0 nibbles are recovered from the packed byte. -/
theorem packQ4_roundtrip (lo hi : Nat) (h1 : lo < 16) (h2 : hi < 16) :
    packQ4 lo hi % 16 = lo ∧ packQ4 lo hi / 16 = hi := by
  unfold packQ4; omega

/-- Block byte sizes: Q4_0 is 2 + 16, Q8_0 is 2 + 32. -/
theorem q4_q8_block_bytes : 2 + 32 / 2 = 18 ∧ 2 + 32 = 34 := by decide

-- § expert tensor shapes and sizes

/-- Elements of the fused `gate_up_proj` tensor. -/
def gateUpElems : Nat := 512 * 1280 * 2560

/-- Elements of `down_proj`. -/
def downElems : Nat := 512 * 2560 * 640

/-- The fused gate/up output dimension is twice the down input dimension. -/
theorem gate_up_is_double : 1280 = 2 * 640 := by decide

/-- Last-axis lengths are multiples of both Q2_0/Q4_0/Q8_0 block sizes. -/
theorem inner_dims_aligned : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

/-- Stored last-axis byte length: `n / block * block_bytes`. -/
def storedLast (n block bytes : Nat) : Nat := n / block * bytes

/-- Q2_0 stored last dims for gate_up (2560) and down (640). -/
theorem q2_stored_last : storedLast 2560 64 18 = 720 ∧ storedLast 640 64 18 = 180 := by decide

/-- Q2_0 total expert bytes ≈ 0.71 GB. -/
theorem q2_total_bytes : (gateUpElems + downElems) / 64 * 18 = 707788800 := by decide

/-- Q4_0 total expert bytes ≈ 1.42 GB. -/
theorem q4_total_bytes : (gateUpElems + downElems) / 32 * 18 = 1415577600 := by decide

/-- Q8_0 total expert bytes ≈ 2.67 GB. -/
theorem q8_total_bytes : (gateUpElems + downElems) / 32 * 34 = 2673868800 := by decide

/-- Q4_0 is exactly twice Q2_0 in bytes; Q8_0 is 34/18 of Q4_0. -/
theorem q4_twice_q2 : 1415577600 = 2 * 707788800 := by decide

/-- Quantized tensors divide evenly into blocks for every format. -/
theorem experts_block_divisible :
    gateUpElems % 64 = 0 ∧ downElems % 64 = 0 ∧ gateUpElems % 32 = 0 ∧ downElems % 32 = 0 := by decide

/-- One-dimensional tensors are stored as F32 (4 bytes), dense matrices as BF16 (2 bytes): BF16 is half of F32. -/
theorem bf16_half_f32 (n : Nat) : n * 2 * 2 = n * 4 := by omega

end MtpPack
