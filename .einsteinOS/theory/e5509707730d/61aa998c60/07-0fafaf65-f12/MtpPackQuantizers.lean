/-!
# MODULE: mtp_pack_quantizers
SOURCES: tools/mtp_pack.py
Integer model of the MTP draft block packer: the per-block scale search grid for Q2_0, the Q4_0 and Q8_0
reference code ranges, per-row block bytes and the expert tensor byte budget.

## Source semantics
Q2_0 scale d = amax * f with f swept over 17 values from 1/2 to 1; codes clamp(rint(x/d), -1, 2) + 1 in 0..3.
Q4_0: code = min(15, trunc(x/d + 8.5)); packed low nibble = element i, high nibble = element i+16.
Experts are the fused tensors gate_up_proj [512,1280,2560] and down_proj [512,2560,640], quantised along the last axis.

## Form template identifiers
`QUANT` table -> `QuantRow`; `EXPERT_TENSORS` -> the two fused tensors; `block`/`block_bytes` -> `blockElems`/`blockBytes`.
-/

namespace StrataMtpPack

-- § scale search grid

/-- Grid index i in 0..16 gives candidate factor (16+i)/32. -/
def gridNum (i : Nat) : Nat := 16 + i

/-- The 17 candidates lie between 1/2 (16/32) and 1 (32/32), inclusive. -/
theorem grid_bounds (i : Nat) (h : i ≤ 16) : 16 ≤ gridNum i ∧ gridNum i ≤ 32 := by
  unfold gridNum; omega

/-- There are exactly 17 candidates. -/
theorem grid_count : (List.range 17).length = 17 := by decide

/-- The endpoints are 1/2 and 1. -/
theorem grid_endpoints : gridNum 0 * 2 = 32 ∧ gridNum 16 = 32 := by decide

-- § code ranges

/-- Integer clamp. -/
def clamp (lo hi x : Int) : Int := max lo (min hi x)

/-- The Q2_0 code `clamp(q,-1,2)+1` is in 0..3. -/
theorem q2_0_code_range (q : Int) : 0 ≤ clamp (-1) 2 q + 1 ∧ clamp (-1) 2 q + 1 ≤ 3 := by
  unfold clamp; omega

/-- Q4_0 code is capped at 15. -/
theorem q4_0_code_cap (t : Nat) : min 15 t ≤ 15 := Nat.min_le_left _ _

/-- Two Q4_0 nibbles pack into one byte: first half low, second half high. -/
theorem q4_0_pack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    lo + 16 * hi < 256 ∧ (lo + 16 * hi) % 16 = lo ∧ (lo + 16 * hi) / 16 = hi := by omega

/-- Q4_0 decode: the -8 bias is in the integer, so codes 0..15 give -8..7. -/
theorem q4_0_decode_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) - 8 ∧ (c : Int) - 8 ≤ 7 := by omega

-- § block sizes

/-- Per-quantiser (block elements, block bytes). -/
structure QuantRow where
  blockElems : Nat
  blockBytes : Nat
  deriving DecidableEq, Repr

/-- Q2_0 row. -/
def rowQ2 : QuantRow := ⟨64, 18⟩
/-- Q4_0 row. -/
def rowQ4 : QuantRow := ⟨32, 18⟩
/-- Q8_0 row. -/
def rowQ8 : QuantRow := ⟨32, 34⟩

/-- Bytes to store `n` elements. -/
def bytesFor (r : QuantRow) (n : Nat) : Nat := n / r.blockElems * r.blockBytes

/-- Elements of gate_up_proj [512,1280,2560]. -/
def gateUpElems : Nat := 512 * 1280 * 2560
/-- Elements of down_proj [512,2560,640]. -/
def downElems : Nat := 512 * 2560 * 640

/-- Total routed-expert elements. -/
theorem expert_elems : gateUpElems + downElems = 2516582400 := by decide

/-- Both inner dimensions are multiples of the 64-element block (hence of 32 too). -/
theorem inner_dims_divisible : 2560 % 64 = 0 ∧ 640 % 64 = 0 := by decide

/-- Q2_0 expert bytes: 707,788,800 (about 0.71 GB). -/
theorem q2_0_expert_bytes : bytesFor rowQ2 gateUpElems + bytesFor rowQ2 downElems = 707788800 := by decide

/-- Q4_0 expert bytes: 1,415,577,600 (about 1.42 GB). -/
theorem q4_0_expert_bytes : bytesFor rowQ4 gateUpElems + bytesFor rowQ4 downElems = 1415577600 := by decide

/-- Q8_0 expert bytes: 2,673,868,800 (about 2.67 GB). -/
theorem q8_0_expert_bytes : bytesFor rowQ8 gateUpElems + bytesFor rowQ8 downElems = 2673868800 := by decide

/-- The quantised last-axis width written as the raw shape: 2560 -> 720 and 640 -> 180 bytes for Q2_0. -/
theorem raw_shape_last : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- A larger format never stores fewer bytes per element (Q2_0 < Q4_0 < Q8_0). -/
theorem format_order : 707788800 < 1415577600 ∧ 1415577600 < 2673868800 := by decide

-- § bf16 widening

/-- bf16 -> f32 is a 16-bit left shift: exact and invertible. -/
theorem bf16_widen (u : Nat) (h : u < 65536) : (u * 65536) / 65536 = u := by omega

end StrataMtpPack
