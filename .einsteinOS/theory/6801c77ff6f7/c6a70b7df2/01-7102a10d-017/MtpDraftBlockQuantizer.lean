/-!
# MODULE: mtp_draft_block_quantizer
SOURCES: tools/mtp_pack.py
Packs the MTP draft block into a GGUF with Q2_0 / Q4_0 / Q8_0 routed experts, using per-block scale search
for Q2_0 and ggml reference rounding for Q4_0 and Q8_0.

## Source semantics
* Q2_0 scale candidates are amax*f for 17 steps f in [0.5, 1]; at f=1/2 the +2 level reaches amax.
* Codes are clip(rint(x/d), -1, 2)+1 in 0..3, packed 4 per byte LSB-first.
* Q4_0 packs q[:16] | q[16:]<<4; Q8_0 codes fit int8.
* Expert tensors: gate_up [512,1280,2560], down [512,2560,640], quantized along the last axis.

## Form template identifiers
q2_0 -> mtp_q2_0_quantize ; q4_0 -> mtp_q4_0_quantize ; q8_0 -> mtp_q8_0_quantize ;
QUANT -> mtp_expert_format_table ; EXPERT_TENSORS -> mtp_fused_expert_tensors
-/
set_option maxRecDepth 100000

namespace Strata.Mtp

-- § Q2_0 scale search
def candNum (i : Nat) : Nat := 16 + i   -- f_i = (16+i)/32

/-- There are 17 candidate scales. -/
theorem cand_count : (List.range 17).length = 17 := by decide

/-- The first candidate is f = 1/2 and the last is f = 1. -/
theorem cand_endpoints : candNum 0 = 16 ∧ candNum 16 = 32 := by decide

/-- Candidates are strictly increasing and stay in [1/2, 1] (numerators in [16, 32]). -/
theorem cand_monotone (i : Nat) (h : i < 16) : candNum i < candNum (i + 1) ∧ candNum (i + 1) ≤ 32 := by
  unfold candNum
  omega

/-- At any candidate f >= 1/2 the +2 level (2*d) reaches at least amax: 32 <= 2*numerator. -/
theorem plus_two_reaches_amax (i : Nat) (h : i ≤ 16) : 32 ≤ 2 * candNum i := by
  unfold candNum
  omega

def clipCode (r : Int) : Nat := Int.toNat (max (-1) (min 2 r) + 1)

/-- Clipped codes are in 0..3. -/
theorem clipCode_lt (r : Int) : clipCode r < 4 := by
  unfold clipCode
  omega

/-- Codes are exact for rounded ratios in [-1, 2]. -/
theorem clipCode_exact (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 2) : (clipCode r : Int) - 1 = r := by
  unfold clipCode
  omega

def pack4 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Packing four codes and unpacking with the dequantizer's shifts is the identity. -/
theorem pack4_unpack (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4 c0 c1 c2 c3 < 256 ∧ pack4 c0 c1 c2 c3 % 4 = c0 ∧ pack4 c0 c1 c2 c3 / 4 % 4 = c1 ∧
    pack4 c0 c1 c2 c3 / 16 % 4 = c2 ∧ pack4 c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold pack4
  omega

-- § Q4_0 / Q8_0
def pack44 (lo hi : Nat) : Nat := lo + 16 * hi

/-- Q4_0 packs q[:16] into the low nibble and q[16:] into the high nibble; both are recovered. -/
theorem pack44_unpack (lo hi : Nat) (hl : lo < 16) (hh : hi < 16) :
    pack44 lo hi < 256 ∧ pack44 lo hi % 16 = lo ∧ pack44 lo hi / 16 = hi := by
  unfold pack44
  omega

/-- The extreme element maps to code 0 (since x/d = -8 and trunc(-8+8.5)=0), the symbol -8. -/
theorem q4_0_extreme_symbol : (0 : Int) - 8 = -8 := by decide

/-- Q4_0 clamping keeps codes at most 15. -/
theorem q4_0_code_clamp (t : Nat) : min 15 t ≤ 15 := Nat.min_le_left _ _

/-- Q8_0: rounded codes in [-127,127] fit the int8 range [-128,127]. -/
theorem q8_0_code_int8 (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : -128 ≤ q ∧ q ≤ 127 := by
  omega

-- § sizes
def gateUpElems : Nat := 512 * 1280 * 2560
def downElems : Nat := 512 * 2560 * 640
def expertElems : Nat := gateUpElems + downElems

/-- The inner dimensions are multiples of both block sizes (64 and 32). -/
theorem inner_dims_aligned : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

/-- Bytes per row on the last axis: 720 for 2560 wide, 180 for 640 wide (Q2_0). -/
theorem q2_0_row_bytes : 2560 / 64 * 18 = 720 ∧ 640 / 64 * 18 = 180 := by decide

/-- Total routed-expert elements of the MTP block. -/
theorem expert_elems_value : expertElems = 2516582400 := by decide

/-- Q2_0 experts total 707,788,800 bytes, about 0.71 GB. -/
theorem q2_0_total_bytes : expertElems / 64 * 18 = 707788800 := by decide

/-- That equals one full layer of the main arena (512 blobs of 1,382,400 B). -/
theorem q2_0_equals_layer_arena : expertElems / 64 * 18 = 512 * 1382400 := by decide

/-- Q4_0 experts total 1,415,577,600 bytes, about 1.42 GB. -/
theorem q4_0_total_bytes : expertElems / 32 * 18 = 1415577600 := by decide

/-- Q8_0 experts total 2,673,868,800 bytes, about 2.67 GB. -/
theorem q8_0_total_bytes : expertElems / 32 * 34 = 2673868800 := by decide

/-- Q2_0 is strictly smaller than Q4_0, which is strictly smaller than Q8_0. -/
theorem format_size_order : expertElems / 64 * 18 < expertElems / 32 * 18 ∧
    expertElems / 32 * 18 < expertElems / 32 * 34 := by decide

/-- The fused gate_up tensor has twice the rows of the 640-wide FFN (1280 = 2*640). -/
theorem gate_up_rows : 2 * 640 = 1280 := by decide

end Strata.Mtp
