/-!
# MODULE: mtp_pack_quantizers
Arithmetic of `tools/mtp_pack.py`: packing the MTP draft block's routed experts as Q2_0 / Q4_0 / Q8_0
with the fused layout `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]`.

## Source semantics
* Q2_0 scale search: 17 candidate scales `d = amax * f`, `f = 0.5 .. 1.0` in steps of 1/32; level grid
  {-1,0,1,2}; codes `q + 1` in 0..3, four per byte.
* Q4_0 reference: code `min 15 (trunc (x*inv + 8.5))`, low half and high half nibbles of 32 elements.
* Expert inner dims are multiples of the block size; GGUF raw shape last dim = `inner / block * block_bytes`.
* Reported sizes: q2_0 ~0.71 GB, q4_0 ~1.42 GB, q8_0 ~2.67 GB.

## Form template identifiers
* candNum k   <- `np.linspace(0.5, 1.0, 17)` as (16+k)/32
* expertBytes <- `shape[-1] // block * block_bytes` summed over the fused experts
-/

namespace Strata.MtpPack

-- § Q2_0 candidate scales

/-- Numerator of the k-th candidate factor `(16 + k) / 32`. -/
def candNum (k : Nat) : Nat := 16 + k

/-- There are 17 candidates k = 0..16. -/
theorem cand_count : (List.range 17).length = 17 := by decide

/-- Candidates span exactly [1/2, 1]. -/
theorem cand_endpoints : candNum 0 = 16 ∧ candNum 16 = 32 := by decide

/-- Candidates are strictly increasing. -/
theorem cand_strict (k : Nat) : candNum k < candNum (k + 1) := by
  unfold candNum
  omega

/-- At the smallest candidate scale, the +2 level reaches amax (2 * amax/2 = amax). -/
theorem cand_low_reaches_max : 2 * candNum 0 = 32 := by decide

/-- Every candidate lies in [16, 32] when k <= 16. -/
theorem cand_range (k : Nat) (h : k ≤ 16) : 16 ≤ candNum k ∧ candNum k ≤ 32 := by
  unfold candNum
  omega

/-- Clip to [-1,2] then add 1 gives a 2-bit code. -/
theorem q2_code_range (q : Int) : 0 ≤ max (-1) (min 2 q) + 1 ∧ max (-1) (min 2 q) + 1 < 4 := by
  omega

-- § Packing bytes

/-- Four 2-bit codes combine into a byte. -/
theorem pack_q2_byte_lt (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    a + 4 * b + 16 * c + 64 * d < 256 := by
  omega

/-- Q4_0: truncated index clamped at 15 is a nibble. -/
theorem q4_code_nibble (t : Nat) : min 15 t < 16 := by
  omega

/-- Low half and high half nibbles combine into a byte. -/
theorem pack_q4_byte_lt (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) : lo + 16 * hi < 256 := by
  omega

-- § Sizes

/-- Bytes of an expert tensor of `elems` elements in a block format. -/
def expertBytes (blockElems blockBytes elems : Nat) : Nat := elems / blockElems * blockBytes

/-- Element count of the fused gate_up tensor. -/
def guElems : Nat := 512 * 1280 * 2560

/-- Element count of the down tensor. -/
def dnElems : Nat := 512 * 2560 * 640

/-- Inner dims are whole blocks for both block sizes. -/
theorem inner_dims_divisible :
    2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by
  decide

/-- Raw last dims: Q2_0 gives 720 and 180 bytes per row. -/
theorem raw_last_dims_q2_0 : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by
  decide

/-- The fused gate_up row count is gate plus up. -/
theorem gate_up_rows : 2 * 640 = 1280 := by decide

/-- Q2_0 experts total 707,788,800 bytes (~0.71 GB). -/
theorem q2_0_total_bytes :
    expertBytes 64 18 guElems + expertBytes 64 18 dnElems = 707788800 := by
  decide

/-- Q4_0 experts total 1,415,577,600 bytes (~1.42 GB). -/
theorem q4_0_total_bytes :
    expertBytes 32 18 guElems + expertBytes 32 18 dnElems = 1415577600 := by
  decide

/-- Q8_0 experts total 2,673,868,800 bytes (~2.67 GB). -/
theorem q8_0_total_bytes :
    expertBytes 32 34 guElems + expertBytes 32 34 dnElems = 2673868800 := by
  decide

/-- Q2_0 is strictly smaller than Q4_0, which is strictly smaller than Q8_0. -/
theorem size_order :
    expertBytes 64 18 guElems + expertBytes 64 18 dnElems <
      expertBytes 32 18 guElems + expertBytes 32 18 dnElems ∧
    expertBytes 32 18 guElems + expertBytes 32 18 dnElems <
      expertBytes 32 34 guElems + expertBytes 32 34 dnElems := by
  decide

end Strata.MtpPack
